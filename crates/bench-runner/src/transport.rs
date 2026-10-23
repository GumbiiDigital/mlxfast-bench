//! Line transport: the NDJSON-over-stdio seam between runner and engine.
//!
//! The [`Session`](crate::session::Session) is generic over [`LineTransport`] so the
//! same lifecycle/nonce/barrier logic drives either a real child process
//! ([`ChildStdioTransport`]) or the in-process [`MockEngine`](crate::mock::MockEngine)
//! the tests use (there is no live worker on this box).

use std::collections::{BTreeMap, VecDeque};
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, ChildStderr, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

/// H3 (cycle-3) — the outcome of a deadline-bounded read ([`LineTransport::read_line_deadline`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadOutcome {
    /// A line (without its trailing `'\n'`) arrived before the deadline.
    Line(String),
    /// The engine closed the stream (EOF).
    Eof,
    /// The deadline passed before a line arrived — the read was aborted (`RunTimeout` upstream).
    TimedOut,
}

/// Fixed public failure class and private-audit engine stderr from a failed child.
pub struct FailureDiagnostic {
    pub detail: String,
    pub engine_text: String,
}

/// One NDJSON line in each direction. Implementations own the framing (`'\n'`).
pub trait LineTransport {
    /// Write one NDJSON line. Implementations append the trailing `'\n'`;
    /// callers pass the bare serialized JSON with no newline.
    fn write_line(&mut self, line: &str) -> std::io::Result<()>;

    /// Read the next line, without its trailing `'\n'`. `Ok(None)` on EOF.
    fn read_line(&mut self) -> std::io::Result<Option<String>>;

    /// H3 (cycle-3) — read the next line, aborting with [`ReadOutcome::TimedOut`] if `deadline`
    /// passes before a line arrives. `deadline == None` means no bound (blocking). This is the seam
    /// benchd's RunTimeout safeguard drives: a hung engine that never responds yields `TimedOut`
    /// instead of wedging the harness forever.
    ///
    /// The DEFAULT impl delegates to the blocking [`read_line`](LineTransport::read_line) and cannot
    /// honor a deadline — transports whose read can genuinely block past the deadline (a real child
    /// process) MUST override this to observe the bound. [`ChildStdioTransport`] and the test
    /// [`MockEngine`](crate::mock::MockEngine) both override it.
    fn read_line_deadline(
        &mut self,
        _deadline: Option<std::time::Instant>,
    ) -> std::io::Result<ReadOutcome> {
        Ok(match self.read_line()? {
            Some(line) => ReadOutcome::Line(line),
            None => ReadOutcome::Eof,
        })
    }

    /// #134 — the POST-MORTEM for a stream-level failure: what the engine did on the way down.
    ///
    /// Called when a read ends the stream ([`ReadOutcome::Eof`]) without a response — the
    /// signature that blocked Proof A. Implementations that own a child process retain its exact
    /// wait status and redacted worker-stderr tail in the private audit, while returning only a
    /// fixed public class. `None` (the default) means "no child to autopsy" — the in-process
    /// [`MockEngine`](crate::mock::MockEngine) has nothing to add.
    ///
    /// The returned detail is a fixed benchd-written class. Engine text is private-audit only.
    fn failure_diagnostic(&mut self) -> Option<FailureDiagnostic> {
        None
    }
}

// --------------------------------------------------------------------------
// Engine child environment policy — STRICT ALLOWLIST, ported from Swift.
//
// Reference: `sanitizedRuntimeWorkerEnvironment(_:)`,
// mlxfast-challenge-dev/Sources/MLXFastHarness/QwenRuntimeWorker.swift:1376-1409.
// The engine subprocess runs submitted model code, which can read its whole
// environment. Building the child env FROM EMPTY (rather than a denylist over
// the inherited parent env) makes the child env byte-identical across the
// unscored gates pass and the scored timed pass by construction, closing a
// phase-oracle. Any variable whose value could differ between phases (every
// `MLXFAST_*` harness knob, `BENCH_*`, `GIT_CONFIG_*`, ...) is dropped.
// --------------------------------------------------------------------------

/// Exact env var names that pass through to the engine child. Byte-for-byte copy
/// of Swift `allowedExactKeys` (QwenRuntimeWorker.swift:1377-1391).
pub const ENGINE_ENV_ALLOWED_EXACT: &[&str] = &[
    "HF_HUB_OFFLINE",
    "HOME",
    "LANG",
    "LOGNAME",
    "PATH",
    "SHELL",
    "SSH_AUTH_SOCK",
    "TERM",
    "TMPDIR",
    "TRANSFORMERS_OFFLINE",
    "USER",
    // macOS per-user default text encoding consulted by CoreFoundation.
    "__CF_USER_TEXT_ENCODING",
];

/// Name prefixes that pass through to the engine child. The `DARKBLOOM_`/`DYLD_`/
/// `LC_`/`METAL_`/`MLX_`/`MTL_` set is the Swift `allowedPrefixes`
/// (QwenRuntimeWorker.swift:1392-1399). `DS4_`, `VLLM_` and `BENCH_WORKER_` are the
/// adapter additions. All three name ORGANIZER/BOX runtime config that tells a
/// spawned worker WHICH ALREADY-LOADED ENGINE TO ATTACH TO. None is
/// submission-controlled, and all are constant across a window, so they do not
/// reopen the phase-oracle the empty-start rule closes.
///
/// `DS4_` is the ds4 (DwarfStar) resident topology. `tools/serve-up.sh` boots ONE
/// `ds4-resident` per window and exports `DS4_RESIDENT_SOCKET`; every `cuda-engine`
/// benchd spawns reads that name and attaches, so the model loads ONCE PER WINDOW
/// instead of once per phase (David's binding "weights load once" rule). Without
/// this prefix, `spawn_command`'s `env_clear` drops the socket name and the worker
/// falls through to its own in-process or MOCK backend WITHOUT failing — and
/// official mode does not forward worker stderr, so that fall-through would not
/// appear in a ranked log. The rest of the family (`DS4_MODEL`, `DS4_MTP_PATH`,
/// `DS4_MTP_DRAFT_TOKENS`, `DS4_CTX_SIZE`, `DS4_EOS_IDS`, `DS4_ENGINE_IDENT`,
/// `DS4_RESIDENT_READY_FILE`, `DS4_RESIDENT_PHASE_TIMEOUT_S`, `DS4_LIB_DIR`) is the
/// same class of box config and rides the same prefix.
///
/// `VLLM_` STAYS. The vLLM-backed `cuda-engine` reads its serve endpoint from
/// `VLLM_URL` / `VLLM_SERIAL_URL` / `VLLM_MTP_URL` / `VLLM_MODEL`, and
/// `m5-machine-scripts` still drives that path: `converge/serve-up/serve-up-cuda.sh`
/// and `converge/calibration-driver/profile-cuda.sh` inject `VLLM_URL`/`VLLM_MODEL`
/// into this same env-cleared child so each calibration pass RECONNECTS to the
/// resident vLLM serve instead of reloading.
///
/// `BENCH_WORKER_` is the MLX bench-worker's RESIDENT topology. Box tooling sets
/// `BENCH_WORKER_RESIDENT_SOCKET` ONCE PER WINDOW, and every `bench-worker
/// runtime-worker` benchd spawns reads that name and attaches to the one
/// already-loaded resident instead of loading the checkpoint again (David's binding
/// "weights load once" rule, 2026-08-30). It is the MLX twin of `DS4_RESIDENT_SOCKET`
/// and carries the same argument: box config, constant across a window, never
/// submission-controlled. Note the prefix ends in `_`, so the harness name
/// `BENCH_GOLDEN_PATH` and an unrelated `BENCHWORKER_*` still drop.
///
/// Note `MLX_` does NOT match harness `MLXFAST_*` names — those stay excluded.
pub const ENGINE_ENV_ALLOWED_PREFIXES: &[&str] = &[
    "BENCH_WORKER_",
    "DARKBLOOM_",
    "DS4_",
    "DYLD_",
    "LC_",
    "METAL_",
    "MLX_",
    "MTL_",
    "VLLM_",
];

/// Forced into the child env so it can never recursively spawn another worker
/// (Swift QwenRuntimeWorker.swift:1407-1408). Its parent value (if any) is first
/// dropped by the allowlist, then this constant is written unconditionally.
pub const ENGINE_ENV_FORCED_KEY: &str = "MLXFAST_USE_RUNTIME_WORKER";
pub const ENGINE_ENV_FORCED_VALUE: &str = "0";

/// Build the sanitized engine child environment from a parent env map, matching
/// Swift `sanitizedRuntimeWorkerEnvironment`: start EMPTY, copy in only names in
/// [`ENGINE_ENV_ALLOWED_EXACT`] or beginning with an [`ENGINE_ENV_ALLOWED_PREFIXES`]
/// prefix, then force `MLXFAST_USE_RUNTIME_WORKER=0`. `BTreeMap` gives a
/// deterministic, testable result.
pub fn sanitized_engine_env<I, K, V>(parent: I) -> BTreeMap<String, String>
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<String>,
{
    let mut sanitized: BTreeMap<String, String> = BTreeMap::new();
    for (key, value) in parent {
        let key = key.into();
        let allowed = ENGINE_ENV_ALLOWED_EXACT.contains(&key.as_str())
            || ENGINE_ENV_ALLOWED_PREFIXES
                .iter()
                .any(|prefix| key.starts_with(prefix));
        if allowed {
            sanitized.insert(key, value.into());
        }
    }
    sanitized.insert(
        ENGINE_ENV_FORCED_KEY.to_string(),
        ENGINE_ENV_FORCED_VALUE.to_string(),
    );
    sanitized
}

/// Snapshot of the current process environment as `(String, String)` pairs,
/// lossily skipping any entry whose name or value is not valid UTF-8. Allowlist
/// names are ASCII, so a dropped non-UTF-8 name could never have passed anyway;
/// this mirrors Swift's `ProcessInfo.processInfo.environment` `[String: String]`
/// view, which likewise cannot represent non-UTF-8 entries.
fn current_process_env() -> Vec<(String, String)> {
    std::env::vars_os()
        .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
        .collect()
}

/// Monotonic-ish nanosecond discriminator for temp filenames (per-spawn uniqueness so
/// concurrent official spawns write distinct `.sb` profile files).
fn now_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

// --------------------------------------------------------------------------
// Engine stderr drain + redaction — ported from Swift `WorkerStderrDrain` /
// `redactedWorkerStderrLine` (QwenRuntimeWorker.swift:852-987).
// --------------------------------------------------------------------------

/// Prefix on every forwarded worker-stderr line (Swift
/// `WorkerStderrDrain.forwardedLinePrefix`, QwenRuntimeWorker.swift:868).
pub const WORKER_STDERR_FORWARD_PREFIX: &str = "mlxfast-worker: ";

/// Cap on the retained redacted stderr tail, in bytes (Swift
/// `WorkerStderrDrain.tailByteLimit`, QwenRuntimeWorker.swift:867). Public because it is the
/// documented bound on what a [`LineTransport::failure_diagnostic`] can carry into a sealed
/// record.
pub const WORKER_STDERR_TAIL_BYTE_LIMIT: usize = 64 * 1024;

/// #134 — how long a child that has already ended its stdout stream is allowed to finish writing
/// stderr and exit ON ITS OWN before benchd kills it.
///
/// Proof A's worker-stderr blackout was partly THIS: teardown killed the child the instant the
/// handshake read hit EOF, so a worker whose last words came after it closed stdout (a crash
/// handler, a deferred diagnostic, an exit path that flushes stderr last) lost them to the kill.
/// The grace is spent only on the failure path, and only until the child actually exits.
const WORKER_EXIT_GRACE: Duration = Duration::from_millis(2_000);

/// Poll interval while waiting out [`WORKER_EXIT_GRACE`].
const WORKER_EXIT_POLL: Duration = Duration::from_millis(20);

/// #134 — how long the stderr drain thread is given to consume what is left in the pipe and reach
/// EOF once the child is gone. Bounded (rather than an open-ended join) because the write end can
/// outlive the child itself; see [`WorkerStderrDrain::finish_bounded`].
const WORKER_STDERR_FLUSH_GRACE: Duration = Duration::from_millis(500);

/// Marker used when the worker produced no stderr at all — the Proof A shape, where the absence
/// itself is the finding and must not read as "we did not look".
const WORKER_STDERR_NONE: &str = "no worker stderr was emitted";

/// Separator between stderr lines flattened into a one-line diagnostic.
const WORKER_STDERR_JOIN: &str = " | ";

/// Per-line share of the flattened diagnostic. A single oversized line — a binary blob a worker
/// spewed while dying — must not be able to consume the whole tail budget and push every READABLE
/// line out of the record, which would reproduce #134's blackout by a different route. Lines
/// longer than this are CLIPPED (head + tail around a marker), never dropped.
const WORKER_STDERR_LINE_DIAGNOSTIC_LIMIT: usize = 2048;

/// Flatten a retained stderr tail into ONE line, bounded by [`WORKER_STDERR_TAIL_BYTE_LIMIT`].
///
/// Every line is scrubbed ([`crate::scrub::scrub_engine_text`]) BEFORE it is measured or emitted:
/// this string reaches a sealed artifact, so the budget must be spent on the bytes that actually
/// get sealed, not on the pre-scrub original. That also makes the accounting self-consistent —
/// the byte length measured here is exactly the byte length emitted.
///
/// Lines are taken from the END (a worker's last words are why we are here). An over-long line is
/// CLIPPED to [`WORKER_STDERR_LINE_DIAGNOSTIC_LIMIT`] rather than dropped, and a line that will
/// not fit the remaining budget is SKIPPED rather than ending the scan — an early `break` let one
/// oversized line discard every older readable line behind it. Whatever is lost is counted and
/// stated.
fn format_stderr_tail(tail: &[String]) -> String {
    if tail.is_empty() {
        return WORKER_STDERR_NONE.to_string();
    }
    let mut kept: Vec<String> = Vec::new();
    let mut total: usize = 0;
    let mut dropped: usize = 0;
    for line in tail.iter().rev() {
        let scrubbed = crate::scrub::scrub_engine_text(line);
        let piece = crate::scrub::clip_to_bytes(&scrubbed, WORKER_STDERR_LINE_DIAGNOSTIC_LIMIT);
        let separator = if kept.is_empty() {
            0
        } else {
            WORKER_STDERR_JOIN.len()
        };
        if piece.is_empty() || total + piece.len() + separator > WORKER_STDERR_TAIL_BYTE_LIMIT {
            // SKIP, do not stop: an older line may still fit.
            dropped += 1;
            continue;
        }
        total += piece.len() + separator;
        kept.push(piece);
    }
    kept.reverse();
    if dropped > 0 {
        // Pushed as an ELEMENT so the join cannot leave a dangling separator when nothing else fit.
        kept.insert(
            0,
            format!("[{dropped} older worker stderr line(s) dropped]"),
        );
    }
    format!("worker stderr tail: {}", kept.join(WORKER_STDERR_JOIN))
}

/// Per-line redaction for forwarded/retained worker stderr. Byte-for-byte port
/// of Swift `redactedWorkerStderrLine` (QwenRuntimeWorker.swift:980-987) and the
/// identical rule in `sanitizeWorkerDiagnostic` (:1314-1324): worker output comes
/// from submitted model code that has seen the (possibly private) golden, so any
/// line that looks like a token comparison — case-insensitively containing
/// `"expected"` or `"actual"` — collapses to a fixed marker so golden-adjacent
/// content (prompt text, expected/actual tokens, answer keys) can never leak.
pub fn redact_worker_stderr_line(line: &str) -> String {
    if contains_ignore_ascii_case(line, "expected") || contains_ignore_ascii_case(line, "actual") {
        "token-validation-failed".to_string()
    } else {
        line.to_string()
    }
}

/// Case-insensitive substring search that allocates nothing. It replaces a `to_lowercase()` copy
/// of EVERY drained stderr line — a whole-line allocation per line on the flood path this drain
/// exists to survive.
///
/// ASCII folding is exact for the two needles it serves (`expected`, `actual`): both are ASCII,
/// and no non-ASCII scalar lowercases to any of their letters, so a line that matched after
/// `to_lowercase()` matches here and no other line does.
fn contains_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    let (haystack, needle) = (haystack.as_bytes(), needle.as_bytes());
    haystack.len() >= needle.len()
        && haystack
            .windows(needle.len())
            .any(|window| window.eq_ignore_ascii_case(needle))
}

/// Owns the background thread draining an engine child's stderr pipe: each
/// completed line is redacted, forwarded to this process's stderr with the
/// `mlxfast-worker: ` prefix (so a chatty worker's undrained pipe can never
/// stall the run), and appended to a capped redacted tail for diagnostics.
struct WorkerStderrDrain {
    handle: Option<JoinHandle<()>>,
    tail: Arc<Mutex<StderrTail>>,
    private_recorded: bool,
}

/// The retained tail: the lines themselves plus the byte total they occupy under the tail budget
/// (each line plus its separator). Keeping the total alongside the lines is what makes retention
/// linear — the roll-off subtracts the line it drops instead of re-summing the whole tail — and
/// the deque lets it drop from the FRONT without shifting every remaining line.
#[derive(Default)]
struct StderrTail {
    lines: VecDeque<String>,
    bytes: usize,
}

impl StderrTail {
    /// The retained lines, oldest first — the shape `format_stderr_tail` and the callers expect.
    fn to_vec(&self) -> Vec<String> {
        self.lines.iter().cloned().collect()
    }
}

impl WorkerStderrDrain {
    /// `forward` controls whether each redacted line is ALSO echoed to this process's
    /// stderr (Swift `WorkerStderrDrain` emitter). Official runs set it FALSE (main.swift:1217
    /// `forwardsWorkerStderr && !officialRun`): worker output must not stream into CI logs, so
    /// it is redacted + retained in the diagnostic tail but never echoed. Local runs keep it
    /// true (the edit-loop convenience).
    fn start(stderr: ChildStderr, forward: bool) -> Self {
        let tail: Arc<Mutex<StderrTail>> = Arc::new(Mutex::new(StderrTail::default()));
        let tail_thread = Arc::clone(&tail);
        let handle = std::thread::Builder::new()
            .name("mlxfast.worker-stderr-drain".to_string())
            .spawn(move || drain_to_eof(stderr, &tail_thread, forward))
            .expect("spawn worker-stderr-drain thread");
        Self {
            handle: Some(handle),
            tail,
            private_recorded: false,
        }
    }

    /// Join the drain thread (blocks until the child closes stderr → EOF) and return the
    /// accumulated redacted tail lines. Test-only: production teardown must never block
    /// unboundedly on the stderr pipe, so it uses [`finish_bounded`](Self::finish_bounded)
    /// instead. Tests want the deterministic wait, not a grace that can expire under load.
    #[cfg(test)]
    fn join(&mut self) -> Vec<String> {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.tail_snapshot()
    }

    /// The retained tail as it stands right now, WITHOUT waiting for the drain thread.
    fn tail_snapshot(&self) -> Vec<String> {
        self.tail
            .lock()
            .expect("stderr tail lock poisoned")
            .to_vec()
    }

    /// #134 — give the drain thread `grace` to reach EOF and finish, then return the tail whether
    /// or not it did.
    ///
    /// An unbounded `JoinHandle::join` on the drain thread would block until the stderr WRITE END
    /// is closed by every holder — which is not guaranteed to be the child alone. A worker that
    /// leaves a grandchild holding the
    /// inherited pipe keeps it open after the worker itself is reaped, so an unbounded join on the
    /// teardown path would wedge benchd on exactly the runs it is meant to diagnose. When the
    /// grace expires the handle is DETACHED rather than joined: the thread is blocked in a pipe
    /// read, exits on its own when that pipe finally closes, and the tail it has already
    /// accumulated is safe to read behind the mutex meanwhile.
    fn finish_bounded(&mut self, grace: Duration) -> Vec<String> {
        if let Some(handle) = self.handle.take() {
            join_bounded(handle, grace);
        }
        let tail = self.tail_snapshot();
        if !self.private_recorded && !tail.is_empty() {
            self.private_recorded = true;
            let text = format_stderr_tail(&tail);
            if crate::error::record_private_engine_text("worker-stderr", &text).is_err() {
                eprintln!(
                    "benchd: private engine diagnostic audit write failed: class=worker-stderr"
                );
            }
        }
        tail
    }
}

/// Read the child stderr to EOF, redacting and forwarding each line. Fail-loud:
/// a read error is announced on our stderr and appended to the tail, never
/// silently dropped, before the drain stops.
/// Per-line content cap (Swift `WorkerStderrDrain` pendingLine limit,
/// QwenRuntimeWorker.swift:869): a single newline-less line longer than this is collapsed to a
/// fixed marker, so a malicious/buggy engine flooding stderr without a newline cannot grow the
/// drain buffer without bound (OOM). Matches Swift byte-for-byte.
const WORKER_STDERR_LINE_BYTE_LIMIT: usize = 65536;
const WORKER_STDERR_LINE_EXCEEDED: &str = "[worker stderr line exceeded 65536 bytes]";

enum LineRead {
    /// A `\n`-terminated (or final, at-EOF) line was read into `buf`.
    Line,
    /// The line exceeded the cap; `buf` holds nothing usable and the remainder up to the next
    /// `\n` (or EOF) was discarded. Emit the fixed marker.
    Exceeded,
    /// EOF with no bytes pending.
    Eof,
}

/// Read one line into `buf` up to and including a `\n`, but never accumulate more than `cap`
/// content bytes — mirrors Swift's capped `pendingLine`. On overflow, discard the rest of the
/// line so the next line starts clean.
fn read_line_capped<R: BufRead>(
    reader: &mut R,
    buf: &mut Vec<u8>,
    cap: usize,
) -> io::Result<LineRead> {
    read_line_capped_with_overflow(reader, buf, cap, true)
}

/// The stdout variant returns as soon as the cap is crossed because its caller ends the stream;
/// draining a hostile newline-less line first would let the worker keep that thread busy forever.
fn read_line_capped_fail_fast<R: BufRead>(
    reader: &mut R,
    buf: &mut Vec<u8>,
    cap: usize,
) -> io::Result<LineRead> {
    read_line_capped_with_overflow(reader, buf, cap, false)
}

fn read_line_capped_with_overflow<R: BufRead>(
    reader: &mut R,
    buf: &mut Vec<u8>,
    cap: usize,
    discard_overflow: bool,
) -> io::Result<LineRead> {
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(if buf.is_empty() {
                LineRead::Eof
            } else {
                LineRead::Line
            });
        }
        let room = cap.saturating_sub(buf.len());
        match available.iter().position(|&b| b == b'\n') {
            // Newline within the remaining room: take through it (inclusive) and we're done.
            Some(nl) if nl < room => {
                buf.extend_from_slice(&available[..=nl]);
                reader.consume(nl + 1);
                return Ok(LineRead::Line);
            }
            // Newline exists but only past the cap → overflow.
            Some(_) => {
                if discard_overflow {
                    reader.consume(room);
                    discard_to_newline(reader)?;
                }
                return Ok(LineRead::Exceeded);
            }
            // No newline in this chunk.
            None if available.len() <= room => {
                let n = available.len();
                buf.extend_from_slice(available);
                reader.consume(n);
                // keep reading (more chunks) for this same line
            }
            // No newline and this chunk alone reaches the cap → overflow.
            None => {
                if discard_overflow {
                    reader.consume(room);
                    discard_to_newline(reader)?;
                }
                return Ok(LineRead::Exceeded);
            }
        }
    }
}

/// Consume bytes up to and including the next `\n` (or EOF) without retaining them.
fn discard_to_newline<R: BufRead>(reader: &mut R) -> io::Result<()> {
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(());
        }
        if let Some(nl) = available.iter().position(|&b| b == b'\n') {
            reader.consume(nl + 1);
            return Ok(());
        }
        let n = available.len();
        reader.consume(n);
    }
}

/// #134 — join `handle` if it finishes within `grace`, otherwise DETACH it.
///
/// Both stdio reader threads block in a pipe read whose write end an orphaned grandchild can hold
/// open past the child's own death, so neither may be joined unboundedly on the teardown path.
/// A detached thread exits by itself once that pipe closes.
///
/// TRADE, accepted deliberately: each orphan-holding teardown leaks ONE thread for as long as the
/// grandchild lives (measured: +8 threads across 8 orphans, all reclaimed once the holders
/// exited — self-healing, and bounded by the number of legs a run spawns). The alternative is a
/// teardown that blocks forever on exactly the runs #134 exists to diagnose.
///
/// Second-order effect on the LOCAL path only: a detached drain that is still forwarding can
/// interleave `mlxfast-worker:` lines from a previous leg into the next leg's console output.
/// Nothing sealed is affected — each transport keeps its OWN tail, and official runs do not
/// forward at all — but console ordering across legs is no longer strictly sequential.
fn join_bounded(handle: JoinHandle<()>, grace: Duration) {
    let deadline = Instant::now() + grace;
    while !handle.is_finished() && Instant::now() < deadline {
        std::thread::sleep(WORKER_EXIT_POLL);
    }
    if handle.is_finished() {
        let _ = handle.join();
    }
}

fn drain_to_eof(stderr: ChildStderr, tail: &Arc<Mutex<StderrTail>>, forward: bool) {
    let mut reader = BufReader::new(stderr);
    let mut buf: Vec<u8> = Vec::new();
    loop {
        buf.clear();
        match read_line_capped(&mut reader, &mut buf, WORKER_STDERR_LINE_BYTE_LIMIT) {
            Ok(LineRead::Eof) => break, // EOF: child closed stderr.
            Ok(LineRead::Line) => {
                while matches!(buf.last(), Some(b'\n') | Some(b'\r')) {
                    buf.pop();
                }
                // Lossy UTF-8 mirrors Swift's `String(decoding:as:UTF8.self)`.
                let line = String::from_utf8_lossy(&buf);
                let redacted = redact_worker_stderr_line(&line);
                forward_and_retain(&redacted, tail, forward);
            }
            Ok(LineRead::Exceeded) => {
                // Over-long newline-less line collapsed (Swift pendingLine cap).
                forward_and_retain(WORKER_STDERR_LINE_EXCEEDED, tail, forward);
            }
            Err(e) => {
                // Fail-loud on drain error — never silently drop worker output.
                let msg = format!("[worker stderr drain error: {e}]");
                forward_and_retain(&msg, tail, forward);
                break;
            }
        }
    }
}

fn forward_and_retain(redacted: &str, tail: &Arc<Mutex<StderrTail>>, forward: bool) {
    // Forward to our stderr (Swift's default emitter fputs to stderr) ONLY when enabled.
    // Official sets `forward = false` (no-op emitter, main.swift:1217): the line is still
    // redacted + retained in the tail, but never echoed to this process's stderr, so
    // submitted code cannot stream hidden-prompt content into CI logs. The line is already
    // sealed, so even a benign-looking engine line cannot carry prompt text into the log.
    if forward {
        let mut err = std::io::stderr().lock();
        let sealed = crate::error::engine_text_seal(redacted);
        let _ = writeln!(
            err,
            "{WORKER_STDERR_FORWARD_PREFIX}class=worker-stderr {sealed}"
        );
        drop(err);
    }

    let mut tail = tail.lock().expect("stderr tail lock poisoned");
    // #134 — RETAIN a per-line-capped copy. The read cap counts RAW bytes, but what is stored
    // here is the post-`from_utf8_lossy` String, where every invalid byte became a 3-byte U+FFFD.
    // A ~30 KB binary blob therefore lands as ~90 KB — larger than the whole 64 KiB tail budget —
    // and the roll-off below would evict EVERY readable line behind it, reproducing #134's
    // blackout by a different route. Capping each retained line makes the two budgets consistent
    // and guarantees the tail always holds many lines rather than one blob. Forwarding above is
    // unaffected: a local operator still sees the full line.
    let kept = crate::scrub::clip_to_bytes(redacted, WORKER_STDERR_LINE_DIAGNOSTIC_LIMIT);
    tail.bytes += kept.len() + 1;
    tail.lines.push_back(kept);
    // Enforce the byte cap by dropping oldest lines (Swift keeps a byte-capped
    // rolling tail). The running total is maintained, not recomputed, and the drop is from the
    // front of a deque: retention is O(1) per line instead of O(tail) twice.
    while tail.bytes > WORKER_STDERR_TAIL_BYTE_LIMIT && tail.lines.len() > 1 {
        let dropped = tail.lines.pop_front().expect("tail has >1 line");
        tail.bytes -= dropped.len() + 1;
    }
}

/// The ranked protocol's largest response is the trusted top-1 cohort replay: 8 streams x 128
/// positions is under 128 KiB even with widest integer values. Keep more than 16x headroom for
/// envelope growth while still putting a hard bound on one worker-controlled allocation.
const WORKER_STDOUT_LINE_BYTE_LIMIT: usize = 2 * 1024 * 1024;
/// Backpressure keeps a fast worker from turning many valid lines into an unbounded benchd queue.
const WORKER_STDOUT_CHANNEL_CAPACITY: usize = 8;

fn open_inheritable_fds() -> io::Result<Vec<i32>> {
    #[cfg(unix)]
    {
        use std::ffi::CStr;

        struct OpenDir(*mut libc::DIR);

        impl Drop for OpenDir {
            fn drop(&mut self) {
                // A failed opendir leaves NULL; Apple's closedir dereferences it, so skip it.
                if !self.0.is_null() {
                    // SAFETY: this guard owns the successful opendir result.
                    unsafe { libc::closedir(self.0) };
                }
            }
        }

        // SAFETY: the path is a static NUL-terminated C string.
        let directory = OpenDir(unsafe { libc::opendir(c"/dev/fd".as_ptr()) });
        if directory.0.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: directory holds a valid DIR pointer.
        let listing_fd = unsafe { libc::dirfd(directory.0) };
        if listing_fd == -1 {
            return Err(io::Error::last_os_error());
        }
        let mut fds = Vec::new();
        loop {
            // SAFETY: directory remains open for this loop. A null pointer marks end of listing.
            let entry = unsafe { libc::readdir(directory.0) };
            if entry.is_null() {
                break;
            }
            // SAFETY: readdir returns a dirent with a NUL-terminated d_name.
            let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) };
            let Ok(name) = name.to_str() else {
                continue;
            };
            let Ok(fd) = name.parse::<i32>() else {
                continue;
            };
            if fd < 3 || fd == listing_fd {
                continue;
            }
            // SAFETY: F_GETFD only reads flags for this descriptor.
            let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
            if flags == -1 {
                let error = io::Error::last_os_error();
                if error.raw_os_error() == Some(libc::EBADF) {
                    continue;
                }
                return Err(error);
            }
            if flags & libc::FD_CLOEXEC == 0 {
                fds.push(fd);
            }
        }
        Ok(fds)
    }

    #[cfg(not(unix))]
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "official sandbox fd closing requires Unix",
    ))
}

fn close_inherited_fds_before_exec(command: &mut Command, fds: Vec<i32>) {
    #[cfg(unix)]
    // SAFETY: the parent built this fixed list before spawn. The closure only iterates it and calls
    // close, which is async-signal-safe. CLOEXEC descriptors are absent, including Rust's spawn
    // error pipe.
    unsafe {
        command.pre_exec(move || {
            for fd in &fds {
                libc::close(*fd);
            }
            Ok(())
        });
    }

    #[cfg(not(unix))]
    let _ = (command, fds);
}

/// Spawns an engine binary and speaks NDJSON over its stdio, matching the Swift
/// `RuntimeWorkerClient` invocation: `<exe> runtime-worker --weights <path> [extra...]`.
///
/// `<exe>` is the fork engine's benchd-facing worker product (`mlxfast-runtime-worker`, see
/// `benchd::measure_job::DEFAULT_MEASURE_WORKER_BIN`), and the argv leads with the
/// `runtime-worker` VERB — the unified generic-kind dispatch that routes `decode_begin`/
/// `decode_step` and `free_decode_begin`/`free_decode_run` by `spec.mode`. The verb string is
/// unchanged (benchd has always led with it); only the resolved binary was repointed to the fork.
///
/// The child environment is sanitized to a strict allowlist built from empty
/// ([`sanitized_engine_env`]) and the child's stderr is drained + redacted on a
/// background thread, mirroring the Swift harness.
///
/// H3 (cycle-3) — stdout is read on a background thread that delivers each line over a channel, so
/// [`read_line_deadline`](LineTransport::read_line_deadline) can bound the wait with `recv_timeout`
/// (the blocking pipe read cannot be interrupted portably otherwise). This is the watchdog seam
/// benchd's RunTimeout uses: a hung engine yields `TimedOut` instead of wedging the harness.
pub struct ChildStdioTransport {
    child: Child,
    stdin: ChildStdin,
    /// Lines from the stdout reader thread (already stripped of the trailing `'\n'`/`'\r'`). The
    /// sender is dropped on EOF or a read error, so a disconnected channel means end-of-stream.
    stdout_rx: Receiver<String>,
    stdout_reader: Option<JoinHandle<()>>,
    stderr_drain: WorkerStderrDrain,
}

/// H3 (cycle-3) — read child stdout to EOF on a background thread, sending each line (trailing
/// newline stripped) over `tx`. Exits on EOF or the first read error (dropping `tx` → the channel
/// disconnects, which the transport reads as end-of-stream, fail-closed). Also exits when the
/// receiver is gone (the transport was dropped).
fn stdout_reader_to_eof<R: io::Read>(stdout: R, tx: SyncSender<String>) {
    let mut reader = BufReader::new(stdout);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match read_line_capped_fail_fast(&mut reader, &mut buf, WORKER_STDOUT_LINE_BYTE_LIMIT) {
            Ok(LineRead::Eof) => break,
            Ok(LineRead::Line) => {
                while matches!(buf.last(), Some(b'\n') | Some(b'\r')) {
                    buf.pop();
                }
                let Ok(line) = String::from_utf8(buf.clone()) else {
                    break; // invalid UTF-8 is a read failure, as with BufRead::read_line
                };
                if tx.send(line).is_err() {
                    break; // receiver (transport) dropped
                }
            }
            Ok(LineRead::Exceeded) => break, // overlong protocol line: fail closed
            Err(_) => break, // read error → treat as end-of-stream (fail-closed upstream)
        }
    }
}

impl ChildStdioTransport {
    /// Assemble the argument vector the way the Swift client does, so tests can assert
    /// the command line without spawning anything.
    pub fn build_args(weights_path: &str, extra_args: &[String]) -> Vec<String> {
        let mut args = vec![
            "runtime-worker".to_string(),
            "--weights".to_string(),
            weights_path.to_string(),
        ];
        args.extend(extra_args.iter().cloned());
        args
    }

    /// Spawn `<executable> runtime-worker --weights <weights_path> [extra_args...]`,
    /// piping stdin/stdout/stderr. The child env is sanitized from the current
    /// process environment. Worker stderr is forwarded to this process (local default).
    pub fn spawn(
        executable: &str,
        weights_path: &str,
        extra_args: &[String],
    ) -> std::io::Result<Self> {
        Self::spawn_with_parent_env(executable, weights_path, extra_args, current_process_env())
    }

    /// B-2 official spawn: run the engine UNDER `sandbox-exec -f <profile>` (Seatbelt),
    /// with worker stderr forwarding forced OFF (still redacted + retained in the tail).
    /// The sandbox `plan` comes from [`crate::sandbox::resolve_official_sandbox`]; a
    /// [`crate::sandbox::SandboxProfile::Generated`] source is written to a temp `.sb` file
    /// first, an [`crate::sandbox::SandboxProfile::Override`] path is used verbatim. The child env is sanitized from
    /// the current process environment exactly as the unsandboxed path.
    pub fn spawn_official_sandboxed(
        plan: &crate::sandbox::OfficialSandboxPlan,
        weights_path: &str,
        extra_args: &[String],
        extra_env: &[(String, String)],
    ) -> std::io::Result<Self> {
        let profile_path = match &plan.profile {
            crate::sandbox::SandboxProfile::Override(p) => p.clone(),
            crate::sandbox::SandboxProfile::Generated(source) => {
                let path = std::env::temp_dir().join(format!(
                    "mlxfast-runtime-worker-{}-{}.sb",
                    std::process::id(),
                    // A per-spawn discriminator so concurrent official spawns don't collide.
                    now_nanos()
                ));
                std::fs::write(&path, source.as_bytes())?;
                path.to_string_lossy().to_string()
            }
        };
        let (program, args) = crate::sandbox::sandbox_exec_command(
            &profile_path,
            &plan.executable_path,
            weights_path,
            extra_args,
        );
        let inherited_fds = open_inheritable_fds()?;
        Self::spawn_command_with_overrides_and_inherited_fds(
            &program,
            &args,
            current_process_env(),
            extra_env,
            plan.forward_worker_stderr,
            Some(inherited_fds),
        )
    }

    /// Spawn with an explicit parent environment (the testable seam that proves
    /// the spawn path — not just [`sanitized_engine_env`] in isolation —
    /// `env_clear`s and applies the allowlist to the real child). Production
    /// callers use [`spawn`](Self::spawn), which supplies the current process env.
    pub fn spawn_with_parent_env<I, K, V>(
        executable: &str,
        weights_path: &str,
        extra_args: &[String],
        parent_env: I,
    ) -> std::io::Result<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self::spawn_with_parent_env_forwarding(
            executable,
            weights_path,
            extra_args,
            parent_env,
            true,
        )
    }

    /// Official-run spawn on a host WITHOUT Seatbelt: run the engine UNSANDBOXED
    /// (`<engine> runtime-worker --weights <W> [extra…]`) with worker-stderr forwarding forced
    /// OFF, sanitizing the child env from THIS process's environment exactly as the sandboxed
    /// official path does. This is the a8 ruling-(b) linux-official spawn: `/usr/bin/sandbox-exec`
    /// is a macOS facility that does not exist on the linux-aarch64 CUDA box, so official there
    /// runs the worker without the `sandbox-exec` wrapper — the ONLY difference from the sandboxed
    /// path is the absent wrapper (same env sanitization, same stderr suppression + redaction +
    /// retention). The seal records `sandbox: none (linux)` so the provenance is honest.
    pub fn spawn_unsandboxed_official(
        executable: &str,
        weights_path: &str,
        extra_args: &[String],
        extra_env: &[(String, String)],
    ) -> std::io::Result<Self> {
        let args = Self::build_args(weights_path, extra_args);
        Self::spawn_command_with_overrides(
            executable,
            &args,
            current_process_env(),
            extra_env,
            false,
        )
    }

    /// As [`spawn_with_parent_env`](Self::spawn_with_parent_env) but with an EXPLICIT
    /// stderr-forwarding policy.
    ///
    /// The unsandboxed production path always forwards (the edit-loop convenience); the OFFICIAL
    /// path does not ([`crate::sandbox::OfficialSandboxPlan::forward_worker_stderr`] is always
    /// false), and that path is what #134's erasure (ii) was really about — retention has to hold
    /// with forwarding OFF, because the retained tail is the only channel an official leg has.
    /// Reaching the official spawn needs a real `sandbox-exec` and a profile, so this is the seam
    /// that lets `forward = false` be exercised on its own.
    pub fn spawn_with_parent_env_forwarding<I, K, V>(
        executable: &str,
        weights_path: &str,
        extra_args: &[String],
        parent_env: I,
        forward_worker_stderr: bool,
    ) -> std::io::Result<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let args = Self::build_args(weights_path, extra_args);
        Self::spawn_command(executable, &args, parent_env, forward_worker_stderr)
    }

    /// The shared spawn core: run `program` with `args`, an env built FROM EMPTY + the
    /// allowlist ([`sanitized_engine_env`]) over `parent_env`, pipes on all three stdio, and
    /// a redacting stderr drain whose forwarding is governed by `forward_worker_stderr`.
    /// `program`/`args` are the FULL command line — for the unsandboxed path that is the
    /// engine + [`build_args`]; for the official path it is `sandbox-exec -f <profile>
    /// <engine> …` (see [`spawn_official_sandboxed`](Self::spawn_official_sandboxed)).
    fn spawn_command<I, K, V>(
        program: &str,
        args: &[String],
        parent_env: I,
        forward_worker_stderr: bool,
    ) -> std::io::Result<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self::spawn_command_with_overrides(program, args, parent_env, &[], forward_worker_stderr)
    }

    /// [`spawn_command`](Self::spawn_command) with PER-SPAWN environment OVERRIDES applied on top
    /// of the sanitized allowlist.
    ///
    /// WHY OVERRIDES EXIST. A paired ranked run measures two legs against two DIFFERENT resident
    /// engines, booted one at a time from two different trees. The name a worker connects by
    /// (`DS4_RESIDENT_SOCKET` / `BENCH_WORKER_RESIDENT_SOCKET`) is therefore a PER-LEG fact, and
    /// the parent process has no one value for it. The override map is how the caller states that
    /// fact for the ONE spawn it is making, without mutating this process's own environment.
    ///
    /// They are applied AFTER [`sanitized_engine_env`] and BEFORE the forced
    /// [`ENGINE_ENV_FORCED_KEY`], so an override cannot smuggle in a name the allowlist would have
    /// dropped for any other reason and cannot re-enable the recursive-worker guard: the forced key
    /// is re-applied last.
    fn spawn_command_with_overrides<I, K, V>(
        program: &str,
        args: &[String],
        parent_env: I,
        extra_env: &[(String, String)],
        forward_worker_stderr: bool,
    ) -> std::io::Result<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self::spawn_command_with_overrides_and_inherited_fds(
            program,
            args,
            parent_env,
            extra_env,
            forward_worker_stderr,
            None,
        )
    }

    fn spawn_command_with_overrides_and_inherited_fds<I, K, V>(
        program: &str,
        args: &[String],
        parent_env: I,
        extra_env: &[(String, String)],
        forward_worker_stderr: bool,
        inherited_fds: Option<Vec<i32>>,
    ) -> std::io::Result<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let mut sanitized = sanitized_engine_env(parent_env);
        for (key, value) in extra_env {
            sanitized.insert(key.clone(), value.clone());
        }
        sanitized.insert(
            ENGINE_ENV_FORCED_KEY.to_string(),
            ENGINE_ENV_FORCED_VALUE.to_string(),
        );
        let mut command = Command::new(program);
        command
            .args(args)
            // Build the child env FROM EMPTY, then apply only the allowlisted
            // names — the parent env is never inherited.
            .env_clear()
            .envs(&sanitized)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(fds) = inherited_fds {
            close_inherited_fds_before_exec(&mut command, fds);
        }
        let mut child = command.spawn()?;
        let stdin = child.stdin.take().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "child stdin was not captured",
            )
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "child stdout was not captured",
            )
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "child stderr was not captured",
            )
        })?;
        // H3 (cycle-3) — spawn the stdout reader thread delivering lines over a channel.
        let (tx, stdout_rx) = mpsc::sync_channel::<String>(WORKER_STDOUT_CHANNEL_CAPACITY);
        let stdout_reader = std::thread::Builder::new()
            .name("mlxfast.worker-stdout-reader".to_string())
            .spawn(move || stdout_reader_to_eof(stdout, tx))
            .expect("spawn worker-stdout-reader thread");
        Ok(Self {
            child,
            stdin,
            stdout_rx,
            stdout_reader: Some(stdout_reader),
            stderr_drain: WorkerStderrDrain::start(stderr, forward_worker_stderr),
        })
    }

    /// #134 — autopsy the engine child after its stream ended without a response.
    ///
    /// Gives the child [`WORKER_EXIT_GRACE`] to exit on its own (so a worker that writes its
    /// diagnosis AFTER closing stdout is not killed mid-sentence — the defect that made Proof A
    /// undiagnosable), kills it if it outlasts that, then collects the stderr drain and reports
    /// how the child ended together with the retained tail.
    ///
    /// Returns the benchd-written ending separately from the engine-written stderr tail.
    pub fn post_mortem(&mut self) -> FailureDiagnostic {
        let detail = self.reap_with_grace().describe();
        let tail = self.stderr_drain.finish_bounded(WORKER_STDERR_FLUSH_GRACE);
        FailureDiagnostic {
            detail,
            engine_text: format_stderr_tail(&tail),
        }
    }

    /// Wait out [`WORKER_EXIT_GRACE`] for a SELF-DIRECTED exit, then kill.
    fn reap_with_grace(&mut self) -> WorkerEnding {
        let deadline = Instant::now() + WORKER_EXIT_GRACE;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => return WorkerEnding::SelfDirected(status),
                Ok(None) => {}
                Err(e) => return WorkerEnding::WaitFailed(e.to_string()),
            }
            if Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(WORKER_EXIT_POLL);
        }
        // Still running after the grace. Killing it is OUR action, and must be attributed to us.
        let _ = self.child.kill();
        let _ = self.child.wait();
        WorkerEnding::KilledByBenchd
    }
}

/// #134 — HOW the engine child ended, kept as a type so the two cases cannot be described in the
/// same words.
///
/// The distinction is load-bearing, not cosmetic. The Proof A retry discriminates its leading
/// hypotheses on the phrase "killed by signal 9" (an OOM/kernel kill of the worker). If benchd's
/// own 2 s-grace `kill()` were also rendered through the wait status, a LIVE engine that merely
/// sent one undecodable line would seal "worker was killed by signal 9" — a false statement that
/// forges exactly the evidence the retry keys on.
enum WorkerEnding {
    /// The child exited on its own; the wait status is the engine's own verdict.
    SelfDirected(std::process::ExitStatus),
    /// The child was STILL RUNNING after the grace and benchd killed it. Its wait status would be
    /// SIGKILL — ours, not the engine's — so it is deliberately not reported.
    KilledByBenchd,
    /// `try_wait` itself failed; nothing can be claimed about the child.
    WaitFailed(String),
}

impl WorkerEnding {
    fn describe(&self) -> String {
        let exact = match self {
            WorkerEnding::SelfDirected(status) => describe_exit_status(status),
            WorkerEnding::KilledByBenchd => format!(
                "worker was STILL RUNNING {:?} after it ended its output stream and was killed by \
                 benchd (this is a harness kill, NOT a worker-side exit)",
                WORKER_EXIT_GRACE
            ),
            WorkerEnding::WaitFailed(e) => format!("worker wait failed: {e}"),
        };
        if crate::error::record_private_engine_value("worker-exit-status", &exact).is_err() {
            eprintln!(
                "benchd: private engine diagnostic audit write failed: class=worker-exit-status"
            );
        }
        "worker exited abnormally".to_string()
    }
}

/// Human description of a SELF-DIRECTED wait status. On unix a signal death (the shape an
/// OOM-kill / crash / sandbox abort takes) is reported as such — `ExitStatus::code()` is `None`
/// there, so without this the most diagnostic case would read as "no status code".
///
/// Only ever called for [`WorkerEnding::SelfDirected`]: see that type's doc for why a
/// benchd-initiated kill must never reach this function.
fn describe_exit_status(status: &std::process::ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return format!("worker was killed by signal {signal}");
        }
    }
    match status.code() {
        Some(code) => format!("worker exited with status {code}"),
        None => "worker exited with no status code".to_string(),
    }
}

impl LineTransport for ChildStdioTransport {
    fn write_line(&mut self, line: &str) -> std::io::Result<()> {
        // `stdin` is an UNBUFFERED `ChildStdin`, so every `write_all` is its own `write(2)` on
        // the pipe: framing the newline separately doubled the syscalls per request. Frame into
        // one buffer and write once. The bytes on the wire are unchanged — `line`, then `'\n'`.
        let mut framed = String::with_capacity(line.len() + 1);
        framed.push_str(line);
        framed.push('\n');
        self.stdin.write_all(framed.as_bytes())?;
        self.stdin.flush()
    }

    fn read_line(&mut self) -> std::io::Result<Option<String>> {
        // Blocking receive from the stdout reader thread; a disconnected channel is EOF.
        Ok(self.stdout_rx.recv().ok())
    }

    fn read_line_deadline(&mut self, deadline: Option<Instant>) -> std::io::Result<ReadOutcome> {
        match deadline {
            None => Ok(match self.stdout_rx.recv() {
                Ok(line) => ReadOutcome::Line(line),
                Err(_) => ReadOutcome::Eof, // sender dropped → EOF
            }),
            Some(deadline) => {
                let now = Instant::now();
                // Already past the deadline: bounded wait would be zero — report TimedOut.
                let timeout = deadline.saturating_duration_since(now);
                match self.stdout_rx.recv_timeout(timeout) {
                    Ok(line) => Ok(ReadOutcome::Line(line)),
                    Err(RecvTimeoutError::Timeout) => Ok(ReadOutcome::TimedOut),
                    Err(RecvTimeoutError::Disconnected) => Ok(ReadOutcome::Eof),
                }
            }
        }
    }

    /// #134 — a real child HAS a private autopsy: its wait status and redacted stderr tail.
    /// It is independent of stderr forwarding and never reaches the public error.
    fn failure_diagnostic(&mut self) -> Option<FailureDiagnostic> {
        Some(self.post_mortem())
    }
}

impl Drop for ChildStdioTransport {
    fn drop(&mut self) {
        // Best-effort teardown; the full shutdown/grace protocol is WS1-6.
        let _ = self.child.kill();
        let _ = self.child.wait();
        // Killing the child closes its stdout/stderr write ends → the reader/drain threads hit EOF
        // and exit; collect them so neither outlives the transport.
        if let Some(handle) = self.stdout_reader.take() {
            join_bounded(handle, WORKER_STDERR_FLUSH_GRACE);
        }
        // #134 — BOUNDED, not an open join: the write ends can outlive the child (an orphaned
        // grandchild still holding the inherited pipe), and teardown must not wedge on that.
        let _ = self.stderr_drain.finish_bounded(WORKER_STDERR_FLUSH_GRACE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_args_matches_swift_client() {
        let args = ChildStdioTransport::build_args("/weights/qwen", &[]);
        assert_eq!(args, vec!["runtime-worker", "--weights", "/weights/qwen"]);
    }

    #[test]
    fn build_args_appends_extra() {
        let extra = vec!["--verbose".to_string(), "--seed=7".to_string()];
        let args = ChildStdioTransport::build_args("/w", &extra);
        assert_eq!(
            args,
            vec!["runtime-worker", "--weights", "/w", "--verbose", "--seed=7"]
        );
    }

    /// Env-dump probe (function level): the sanitizer over a parent env holding
    /// allowed-exact, prefixed, `MLXFAST_*`, and random names yields an
    /// allowlist-only child env — `MLXFAST_*` absent, non-allowed absent, allowed
    /// + prefixed present, and `MLXFAST_USE_RUNTIME_WORKER` forced to "0".
    #[test]
    fn sanitized_engine_env_is_allowlist_only() {
        let parent = vec![
            // allowed exact
            ("HOME", "/Users/tester"),
            ("PATH", "/usr/bin:/bin"),
            ("__CF_USER_TEXT_ENCODING", "0x1F5:0x8000100:0x8000100"),
            ("HF_HUB_OFFLINE", "1"),
            // allowed prefixes
            ("MLX_DISABLE_COMPILE", "1"),
            ("DYLD_LIBRARY_PATH", "/opt/lib"),
            ("DARKBLOOM_COMPILED_DECODE", "1"),
            ("LC_ALL", "en_US.UTF-8"),
            ("METAL_DEVICE_WRAPPER_TYPE", "1"),
            ("MTL_HUD_ENABLED", "0"),
            // VLLM_ prefix — the cuda-engine adapter's serve endpoint config.
            ("VLLM_URL", "http://127.0.0.1:30017"),
            // DS4_ prefix — the ds4 resident socket the CUDA track's serve-up
            // exports so every spawned worker attaches to the ONE loaded engine.
            ("DS4_RESIDENT_SOCKET", "/tmp/ds4-resident.sock"),
            // MLXFAST_* harness namespace — must NEVER pass (MLX_ prefix must not
            // match these).
            ("MLXFAST_SCORE_PATH", "/tmp/score.json"),
            ("MLXFAST_CORRECTNESS_GOLDEN_PATH", "/tmp/golden.json"),
            ("MLXFAST_USE_RUNTIME_WORKER", "1"),
            // random / unrelated — dropped
            ("BENCH_GOLDEN_PATH", "/tmp/g"),
            ("GIT_CONFIG_GLOBAL", "/tmp/gc"),
            ("SECRET_TOKEN", "hunter2"),
        ];

        let child = sanitized_engine_env(parent);

        // allowed present
        assert_eq!(child.get("HOME").map(String::as_str), Some("/Users/tester"));
        assert_eq!(child.get("PATH").map(String::as_str), Some("/usr/bin:/bin"));
        assert!(child.contains_key("__CF_USER_TEXT_ENCODING"));
        assert!(child.contains_key("HF_HUB_OFFLINE"));
        // prefixed present
        assert!(child.contains_key("MLX_DISABLE_COMPILE"));
        assert!(child.contains_key("DYLD_LIBRARY_PATH"));
        assert!(child.contains_key("DARKBLOOM_COMPILED_DECODE"));
        assert!(child.contains_key("LC_ALL"));
        assert!(child.contains_key("METAL_DEVICE_WRAPPER_TYPE"));
        assert!(child.contains_key("MTL_HUD_ENABLED"));
        // VLLM_ prefix passes (adapter serve endpoint) while an unrelated var
        // (SECRET_TOKEN, below) still drops — the CUDA-adapter allowlist receipt.
        assert_eq!(
            child.get("VLLM_URL").map(String::as_str),
            Some("http://127.0.0.1:30017")
        );
        // DS4_ prefix passes: the resident socket must survive the env_clear or
        // the worker silently serves its own backend instead of attaching.
        assert_eq!(
            child.get("DS4_RESIDENT_SOCKET").map(String::as_str),
            Some("/tmp/ds4-resident.sock")
        );
        // MLXFAST_* never passes, except the forced control at "0"
        assert!(!child.contains_key("MLXFAST_SCORE_PATH"));
        assert!(!child.contains_key("MLXFAST_CORRECTNESS_GOLDEN_PATH"));
        assert_eq!(
            child.get("MLXFAST_USE_RUNTIME_WORKER").map(String::as_str),
            Some("0"),
            "recursion guard forced to 0, parent value discarded"
        );
        // random / unrelated dropped
        assert!(!child.contains_key("BENCH_GOLDEN_PATH"));
        assert!(!child.contains_key("GIT_CONFIG_GLOBAL"));
        assert!(!child.contains_key("SECRET_TOKEN"));

        // Exact allowlist membership: every key is allowed-exact, prefixed, or
        // the forced control — nothing else survived.
        for key in child.keys() {
            let ok = ENGINE_ENV_ALLOWED_EXACT.contains(&key.as_str())
                || ENGINE_ENV_ALLOWED_PREFIXES
                    .iter()
                    .any(|p| key.starts_with(p))
                || key == ENGINE_ENV_FORCED_KEY;
            assert!(ok, "unexpected key survived sanitization: {key}");
        }
    }

    /// `MLX_` (allowed prefix) must not swallow `MLXFAST_` (harness namespace):
    /// the boundary is exact — `MLXFAST_ANYTHING` is dropped while `MLX_ANYTHING`
    /// passes.
    #[test]
    fn mlx_prefix_does_not_match_mlxfast() {
        let child = sanitized_engine_env(vec![
            ("MLX_METAL_DEBUG", "1"),
            ("MLXFAST_NOTE", "leak-me"),
            ("MLXFAST_", "edge"),
        ]);
        assert!(child.contains_key("MLX_METAL_DEBUG"));
        assert!(!child.contains_key("MLXFAST_NOTE"));
        assert!(!child.contains_key("MLXFAST_"));
    }

    /// `BENCH_WORKER_` (allowed prefix) carries the MLX resident socket the box sets
    /// once per window, so a spawned worker attaches instead of reloading the
    /// checkpoint. The boundary is exact: the bare name `BENCH_WORKER` (no trailing
    /// underscore) and an unrelated `BENCHWORKER_X` both drop.
    #[test]
    fn bench_worker_prefix_passes_resident_socket_only() {
        let child = sanitized_engine_env(vec![
            (
                "BENCH_WORKER_RESIDENT_SOCKET",
                "/tmp/bench-worker-resident.sock",
            ),
            ("BENCH_WORKER", "drop-me"),
            ("BENCHWORKER_X", "drop-me"),
        ]);
        assert_eq!(
            child
                .get("BENCH_WORKER_RESIDENT_SOCKET")
                .map(String::as_str),
            Some("/tmp/bench-worker-resident.sock"),
            "resident socket must survive env_clear or the worker reloads the weights"
        );
        assert!(!child.contains_key("BENCH_WORKER"));
        assert!(!child.contains_key("BENCHWORKER_X"));
    }

    /// Redaction: a golden-adjacent stderr line (case-insensitively containing
    /// `expected` / `actual`) collapses to the fixed marker; benign lines pass
    /// through unchanged.
    #[test]
    fn redaction_collapses_golden_adjacent_lines() {
        assert_eq!(
            redact_worker_stderr_line("token[5] expected=1234 actual=5678"),
            "token-validation-failed"
        );
        assert_eq!(
            redact_worker_stderr_line("EXPECTED continuation: the quick brown fox"),
            "token-validation-failed"
        );
        assert_eq!(
            redact_worker_stderr_line("Actual answer key: B"),
            "token-validation-failed"
        );
        // Benign diagnostics are preserved verbatim.
        assert_eq!(
            redact_worker_stderr_line("loaded 1847 tensors"),
            "loaded 1847 tensors"
        );
        assert_eq!(redact_worker_stderr_line(""), "");
    }

    /// #134 — RETENTION is independent of FORWARDING. An official (sandboxed) spawn sets
    /// `forward = false` so worker output never streams into CI logs (`sandbox.rs`
    /// `OfficialSandboxPlan::forward_worker_stderr`), which is exactly why measure-job legs could
    /// never show a `mlxfast-worker:` line. The retained tail must still fill on that path — it is
    /// the ONLY channel by which an official leg's post-mortem can reach a sealed record.
    #[test]
    fn retained_tail_fills_even_when_forwarding_is_off() {
        let tail: Arc<Mutex<StderrTail>> = Arc::new(Mutex::new(StderrTail::default()));
        // `forward_and_retain` receives lines ALREADY redacted by `drain_to_eof`; this test is
        // about retention under `forward = false`, not about the redaction step.
        forward_and_retain(
            &redact_worker_stderr_line("weights mmap failed"),
            &tail,
            false,
        );
        forward_and_retain(
            &redact_worker_stderr_line("token expected=1 actual=2"),
            &tail,
            false,
        );
        let retained = tail.lock().unwrap().to_vec();
        assert_eq!(
            retained,
            vec![
                "weights mmap failed".to_string(),
                "token-validation-failed".to_string()
            ],
            "an unforwarded drain must still retain the (redacted) tail"
        );
    }

    /// EQUIVALENCE — the linear retention (deque + running byte total) and the allocation-free
    /// redaction must produce EXACTLY what the previous algorithms produced. The reference below
    /// is the old code, verbatim: `to_lowercase().contains(..)` for redaction, and a `Vec` that
    /// re-sums its bytes and `remove(0)`s the front. 10,000 lines is well past the tail budget,
    /// so this exercises the roll-off thousands of times, and the lines carry mixed case,
    /// non-ASCII text, and per-line-cap overflow.
    #[test]
    fn linear_retention_and_ascii_redaction_match_the_previous_algorithms() {
        fn reference_redact(line: &str) -> String {
            let lower = line.to_lowercase();
            if lower.contains("expected") || lower.contains("actual") {
                "token-validation-failed".to_string()
            } else {
                line.to_string()
            }
        }
        fn reference_retain(kept: &mut Vec<String>, redacted: &str) {
            kept.push(crate::scrub::clip_to_bytes(
                redacted,
                WORKER_STDERR_LINE_DIAGNOSTIC_LIMIT,
            ));
            let mut total: usize = kept.iter().map(|l| l.len() + 1).sum();
            while total > WORKER_STDERR_TAIL_BYTE_LIMIT && kept.len() > 1 {
                total -= kept.remove(0).len() + 1;
            }
        }

        let lines: Vec<String> = (0..10_000)
            .map(|i| match i % 5 {
                0 => format!("loaded {i} tensors"),
                1 => format!("token[{i}] EXPECTED=1 Actual=2"),
                2 => format!("Ω résumé line {i} — non-ASCII, no match"),
                3 => format!("{i} {}", "x".repeat(200)),
                // Past the per-line diagnostic cap, so clipping participates too.
                _ => format!(
                    "{i} {}",
                    "y".repeat(WORKER_STDERR_LINE_DIAGNOSTIC_LIMIT + 64)
                ),
            })
            .collect();

        let tail: Arc<Mutex<StderrTail>> = Arc::new(Mutex::new(StderrTail::default()));
        let mut reference: Vec<String> = Vec::new();
        for line in &lines {
            let redacted = redact_worker_stderr_line(line);
            assert_eq!(
                redacted,
                reference_redact(line),
                "redaction diverged: {line}"
            );
            forward_and_retain(&redacted, &tail, false);
            reference_retain(&mut reference, &redacted);
        }

        let retained = tail.lock().unwrap().to_vec();
        assert_eq!(
            retained, reference,
            "retained tail diverged from the previous algorithm"
        );
        // The retained bytes are the budget the roll-off enforces, not merely the same lines.
        let retained_bytes: usize = retained.iter().map(|l| l.len() + 1).sum();
        assert_eq!(
            tail.lock().unwrap().bytes,
            retained_bytes,
            "the running byte total drifted from the lines it counts"
        );
        assert!(
            retained_bytes <= WORKER_STDERR_TAIL_BYTE_LIMIT,
            "tail exceeded its byte budget: {retained_bytes}"
        );
        assert!(retained.len() > 1, "roll-off emptied the tail");
    }

    /// The flattened diagnostic is bounded by the tail budget even though joining adds a separator
    /// per line, and it keeps the LAST lines (a worker's last words) rather than the first.
    #[test]
    fn flattened_tail_is_bounded_and_keeps_the_newest_lines() {
        assert_eq!(format_stderr_tail(&[]), WORKER_STDERR_NONE);

        let small = vec!["first".to_string(), "last".to_string()];
        assert_eq!(
            format_stderr_tail(&small),
            "worker stderr tail: first | last"
        );

        // Every line is 100 bytes; 2000 of them is ~200 KiB, well past the 64 KiB budget.
        let mut flood: Vec<String> = (0..2000).map(|i| format!("{i:0>100}")).collect();
        flood.push("THE-LAST-WORDS".to_string());
        let formatted = format_stderr_tail(&flood);
        assert!(
            formatted.len() <= WORKER_STDERR_TAIL_BYTE_LIMIT + 128,
            "flattened tail exceeded its budget: {} bytes",
            formatted.len()
        );
        assert!(
            formatted.contains("THE-LAST-WORDS"),
            "the newest line was dropped"
        );
        assert!(
            formatted.contains("older worker stderr line(s) dropped"),
            "the loss was silent: {}",
            &formatted[..formatted.len().min(200)]
        );
        // The drop marker is an ELEMENT, so it can never leave a dangling separator.
        assert!(
            !formatted.ends_with(WORKER_STDERR_JOIN),
            "trailing separator with nothing after it"
        );
        assert!(
            !formatted.contains(&format!("{:0>100}", 0)),
            "the oldest line survived instead of being dropped"
        );
    }

    #[test]
    fn over_long_stderr_line_is_capped_not_unbounded() {
        use std::io::Cursor;
        // 200 KiB with no newline (a flood), then a newline, then a benign line.
        let mut data = vec![b'x'; 200 * 1024];
        data.push(b'\n');
        data.extend_from_slice(b"loaded 1847 tensors\n");
        let mut reader = BufReader::new(Cursor::new(data));
        let mut buf: Vec<u8> = Vec::new();
        // The over-long line overflows: `buf` never exceeds the cap (no OOM), marker emitted.
        let r = read_line_capped(&mut reader, &mut buf, WORKER_STDERR_LINE_BYTE_LIMIT).unwrap();
        assert!(matches!(r, LineRead::Exceeded));
        assert!(buf.len() <= WORKER_STDERR_LINE_BYTE_LIMIT);
        // The remainder up to the newline was discarded → the NEXT line starts clean.
        buf.clear();
        let r2 = read_line_capped(&mut reader, &mut buf, WORKER_STDERR_LINE_BYTE_LIMIT).unwrap();
        assert!(matches!(r2, LineRead::Line));
        while matches!(buf.last(), Some(b'\n') | Some(b'\r')) {
            buf.pop();
        }
        assert_eq!(String::from_utf8_lossy(&buf), "loaded 1847 tensors");
        // EOF after the last line.
        buf.clear();
        assert!(matches!(
            read_line_capped(&mut reader, &mut buf, WORKER_STDERR_LINE_BYTE_LIMIT).unwrap(),
            LineRead::Eof
        ));
    }

    #[test]
    fn stdout_reader_passes_normal_lines() {
        use std::io::Cursor;

        let (tx, rx) = mpsc::sync_channel(WORKER_STDOUT_CHANNEL_CAPACITY);
        stdout_reader_to_eof(Cursor::new(b"first\nsecond\r\n"), tx);

        assert_eq!(rx.recv().unwrap(), "first");
        assert_eq!(rx.recv().unwrap(), "second");
        assert!(rx.recv().is_err(), "EOF must disconnect the channel");
    }

    #[test]
    fn overlong_stdout_line_ends_stream_without_buffering_the_remainder() {
        use std::io::Cursor;

        let mut data = vec![b'x'; WORKER_STDOUT_LINE_BYTE_LIMIT + 1];
        data.extend_from_slice(b"\nthis-must-not-pass\n");
        let (tx, rx) = mpsc::sync_channel(WORKER_STDOUT_CHANNEL_CAPACITY);

        stdout_reader_to_eof(Cursor::new(data), tx);

        assert!(
            rx.recv().is_err(),
            "an overlong line must end the stream, not truncate or resume"
        );
    }

    #[cfg(unix)]
    #[test]
    fn official_child_closes_fd_above_lowered_soft_limit() {
        use std::fs::File;
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

        const CHILD_MODE: &str = "BENCH_RUNNER_HIGH_FD_TEST_CHILD";
        if std::env::var_os(CHILD_MODE).is_none() {
            let output = Command::new(std::env::current_exe().expect("current test executable"))
                .arg("official_child_closes_fd_above_lowered_soft_limit")
                .arg("--nocapture")
                .env(CHILD_MODE, "1")
                .output()
                .expect("spawn isolated fd test");
            assert!(
                output.status.success(),
                "isolated fd test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let source = File::open("/dev/null").expect("open probe fd");
        let raw = 200;
        // SAFETY: dup2 creates or replaces descriptor 200.
        assert_eq!(unsafe { libc::dup2(source.as_raw_fd(), raw) }, raw);
        // SAFETY: descriptor 200 is now owned by this test process.
        let inherited = unsafe { OwnedFd::from_raw_fd(raw) };
        let flags = unsafe { libc::fcntl(inherited.as_raw_fd(), libc::F_GETFD) };
        assert_eq!(flags & libc::FD_CLOEXEC, 0, "probe fd must be inheritable");

        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: these calls read and then lower this isolated process's descriptor soft limit.
        assert_eq!(
            unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) },
            0
        );
        assert!(
            limit.rlim_cur > 200,
            "test needs fd 200 below the initial limit"
        );
        limit.rlim_cur = 64;
        assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) }, 0);

        let mut child = Command::new("/bin/sh");
        child.args([
            "-c",
            &format!(
                "if [ -e /dev/fd/{} ]; then exit 9; fi",
                inherited.as_raw_fd()
            ),
        ]);
        let fds = open_inheritable_fds().expect("list inheritable fds");
        assert!(fds.contains(&raw), "fd {raw} was not listed");
        close_inherited_fds_before_exec(&mut child, fds);

        let status = child.status().expect("spawn fd probe");
        assert!(status.success(), "official child inherited fd {raw}");
    }

    #[cfg(unix)]
    #[test]
    fn official_child_exec_failure_reaches_parent() {
        let mut command = Command::new("/path/that/does/not/exist/bench-runner-worker");
        let fds = open_inheritable_fds().expect("list inheritable fds");
        close_inherited_fds_before_exec(&mut command, fds);

        let error = command.spawn().expect_err("exec must fail");
        assert_eq!(error.kind(), io::ErrorKind::NotFound, "{error}");
    }

    // ------------------------------------------------------------------
    // Integration tests exercising the real spawn/drain path via a tiny
    // shell "engine". Unix-only (they use `/bin/sh`); skipped elsewhere.
    // ------------------------------------------------------------------

    #[cfg(unix)]
    fn write_script(dir: &std::path::Path, name: &str, body: &str) -> String {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join(name);
        std::fs::write(&path, body).expect("write script");
        let mut perms = std::fs::metadata(&path).expect("stat script").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).expect("chmod script");
        path.to_string_lossy().to_string()
    }

    /// Env-dump probe (spawn path): a mock engine that dumps its received env to
    /// stdout proves `env_clear` + the allowlist reach the actual child — not
    /// just the pure function.
    #[cfg(unix)]
    #[test]
    fn spawn_path_sanitizes_child_env() {
        let dir = std::env::temp_dir().join(format!("b1-env-dump-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir tmp");
        // Ignores the injected `runtime-worker --weights ...` args and prints the
        // child's environment, one `KEY=VALUE` per line, to stdout.
        let engine = write_script(&dir, "dump_env.sh", "#!/bin/sh\nexec env\n");

        let parent = vec![
            ("HOME".to_string(), "/Users/tester".to_string()),
            ("PATH".to_string(), "/usr/bin:/bin".to_string()),
            ("MLX_DISABLE_COMPILE".to_string(), "1".to_string()),
            // Note: DYLD_* is deliberately NOT asserted below — macOS dyld
            // strips DYLD_* from child processes (SIP), so it never reaches the
            // child regardless of our allowlist. LC_ covers the prefix case.
            ("LC_ALL".to_string(), "en_US.UTF-8".to_string()),
            ("MLXFAST_SCORE_PATH".to_string(), "/tmp/leak".to_string()),
            ("MLXFAST_USE_RUNTIME_WORKER".to_string(), "1".to_string()),
            ("B1_RANDOM_UNRELATED".to_string(), "nope".to_string()),
        ];

        let mut transport =
            ChildStdioTransport::spawn_with_parent_env(&engine, "/weights", &[], parent)
                .expect("spawn dump engine");

        let mut lines = Vec::new();
        while let Some(line) = transport.read_line().expect("read child stdout") {
            lines.push(line);
        }
        let keys: std::collections::HashSet<&str> = lines
            .iter()
            .filter_map(|l| l.split_once('=').map(|(k, _)| k))
            .collect();

        // allowed / prefixed reached the child
        assert!(keys.contains("HOME"), "HOME missing: {lines:?}");
        assert!(keys.contains("PATH"), "PATH missing: {lines:?}");
        assert!(keys.contains("MLX_DISABLE_COMPILE"));
        assert!(keys.contains("LC_ALL"));
        // recursion guard forced to 0
        assert!(lines.iter().any(|l| l == "MLXFAST_USE_RUNTIME_WORKER=0"));
        // MLXFAST_* / random never reached the child
        assert!(!keys.contains("MLXFAST_SCORE_PATH"), "leaked: {lines:?}");
        assert!(!keys.contains("B1_RANDOM_UNRELATED"), "leaked: {lines:?}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Env-dump probe (spawn path, ds4 resident): the CUDA track's serve-up exports
    /// `DS4_RESIDENT_SOCKET` so every worker benchd spawns attaches to the ONE engine
    /// loaded for the window. `spawn_command` builds the child env FROM EMPTY, so that
    /// name reaches the worker only if the `DS4_` prefix is on the allowlist. This
    /// asserts on the REAL spawned child, because the failure it guards is silent:
    /// a worker that never sees the socket falls back to its own in-process/MOCK
    /// backend without erroring, and official mode does not forward worker stderr.
    ///
    /// It also pins the boundary — an unlisted name set in the same parent env must
    /// NOT reach the child, so a passing test cannot be explained by inheritance.
    #[cfg(unix)]
    #[test]
    fn spawn_path_passes_ds4_resident_socket_and_drops_unlisted() {
        let dir = std::env::temp_dir().join(format!("b1-ds4-env-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir tmp");
        let engine = write_script(&dir, "dump_env.sh", "#!/bin/sh\nexec env\n");

        let socket = "/tmp/ds4-resident-test.sock";
        let parent = vec![
            ("PATH".to_string(), "/usr/bin:/bin".to_string()),
            // The resident socket — what serve-up.sh exports for the window.
            ("DS4_RESIDENT_SOCKET".to_string(), socket.to_string()),
            // A second DS4_ name from the same box-config family.
            (
                "DS4_ENGINE_IDENT".to_string(),
                "ds4-resident-probe".to_string(),
            ),
            // NOT on the allowlist: a lookalike that shares no allowlisted prefix
            // (DS4 without the underscore) and an unrelated name.
            (
                "DS4RESIDENT_SOCKET".to_string(),
                "/tmp/lookalike".to_string(),
            ),
            ("B1_DS4_UNLISTED".to_string(), "nope".to_string()),
        ];

        let mut transport =
            ChildStdioTransport::spawn_with_parent_env(&engine, "/weights", &[], parent)
                .expect("spawn dump engine");

        let mut lines = Vec::new();
        while let Some(line) = transport.read_line().expect("read child stdout") {
            lines.push(line);
        }
        let keys: std::collections::HashSet<&str> = lines
            .iter()
            .filter_map(|l| l.split_once('=').map(|(k, _)| k))
            .collect();

        // The DS4_ family reached the child, VALUE INTACT — the worker can attach.
        assert!(
            lines
                .iter()
                .any(|l| l == &format!("DS4_RESIDENT_SOCKET={socket}")),
            "DS4_RESIDENT_SOCKET did not reach the worker; it would fall back to its \
             own backend silently: {lines:?}"
        );
        assert!(
            keys.contains("DS4_ENGINE_IDENT"),
            "DS4_ENGINE_IDENT missing: {lines:?}"
        );
        // The unlisted names did not — so the pass above is the allowlist, not
        // an inherited parent env.
        assert!(
            !keys.contains("DS4RESIDENT_SOCKET"),
            "an unlisted lookalike leaked: {lines:?}"
        );
        assert!(
            !keys.contains("B1_DS4_UNLISTED"),
            "an unlisted name leaked: {lines:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Drain path redaction: a golden-adjacent line the engine writes to stderr
    /// is collapsed in the drained tail; a benign line survives.
    #[cfg(unix)]
    #[test]
    fn spawn_path_drains_and_redacts_stderr() {
        let dir = std::env::temp_dir().join(format!("b1-stderr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir tmp");
        let engine = write_script(
            &dir,
            "stderr_engine.sh",
            "#!/bin/sh\nprintf 'token expected=1 actual=2\\n' 1>&2\nprintf 'loaded 1847 tensors\\n' 1>&2\n",
        );

        let mut transport = ChildStdioTransport::spawn_with_parent_env(
            &engine,
            "/weights",
            &[],
            Vec::<(String, String)>::new(),
        )
        .expect("spawn stderr engine");

        // Child exits after writing → stderr EOF → drain thread finishes.
        let tail = transport.stderr_drain.join();
        assert!(
            tail.iter().any(|l| l == "token-validation-failed"),
            "golden-adjacent line not redacted: {tail:?}"
        );
        assert!(
            !tail
                .iter()
                .any(|l| l.contains("expected") || l.contains("actual")),
            "raw golden-adjacent content leaked into tail: {tail:?}"
        );
        assert!(
            tail.iter().any(|l| l == "loaded 1847 tensors"),
            "benign line dropped: {tail:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// WIRE BYTES — the request path frames the newline into the same buffer as the line and
    /// issues ONE `write_all`. This captures what the child actually received on its stdin (a
    /// script copies exactly the expected byte count to a file, then acknowledges on stdout) and compares it
    /// to the framing the two-`write_all` version produced: `line`, then `'\n'`, per call, with
    /// nothing added, reordered, or dropped — including for an EMPTY line (a bare newline) and
    /// a line carrying a tab.
    #[cfg(unix)]
    #[test]
    fn request_path_writes_exactly_line_then_newline() {
        let dir = std::env::temp_dir().join(format!("b1-wire-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir tmp");
        let out = dir.join("wire.bin");
        let expected: &[u8] = b"a\n\nb\tc\n";
        let engine = write_script(
            &dir,
            "wire_engine.sh",
            &format!(
                "#!/bin/sh\nhead -c {} > '{}'\nprintf 'captured\\n'\n",
                expected.len(),
                out.display()
            ),
        );

        let mut transport = ChildStdioTransport::spawn_with_parent_env(
            &engine,
            "/weights",
            &[],
            Vec::<(String, String)>::new(),
        )
        .expect("spawn wire engine");

        for line in ["a", "", "b\tc"] {
            transport.write_line(line).expect("write line");
        }
        // The child acknowledges only after it has read all `expected.len()` bytes.
        assert_eq!(
            transport.read_line().expect("read ack"),
            Some("captured".to_string())
        );

        let captured = std::fs::read(&out).expect("read captured wire bytes");
        assert_eq!(
            captured, expected,
            "wire bytes changed: got {captured:?}, want {expected:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
