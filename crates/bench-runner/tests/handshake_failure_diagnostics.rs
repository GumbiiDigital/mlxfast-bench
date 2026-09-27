//! #134 — the worker-stderr blackout on the hello-handshake path, and its regression fence.
//!
//! Proof A died on every leg with a bare `engine closed the stream before returning a response`
//! and NOT ONE `mlxfast-worker:` line in any run record, which made the failure undiagnosable
//! from the artifacts alone. These tests drive the REAL spawn + `Session::connect` path against
//! scripted fake workers covering each way an engine can end its stream, and assert that benchd's
//! error now carries the engine's own last words (wait status + redacted stderr tail).
//!
//! Unix-only: the fake workers are `/bin/sh` scripts.

#![cfg(unix)]

use bench_runner::{ChildStdioTransport, RunnerError, Session};

fn write_script(dir: &std::path::Path, body: &str) -> String {
    use std::os::unix::fs::PermissionsExt;
    std::fs::create_dir_all(dir).expect("mkdir");
    let path = dir.join("fake_worker.sh");
    std::fs::write(&path, body).expect("write script");
    let mut perms = std::fs::metadata(&path).expect("stat").permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).expect("chmod");
    path.to_string_lossy().to_string()
}

/// Spawn the fake worker. Retries `ExecutableFileBusy`: a sibling test's fork can briefly hold the
/// write fd of a script this test just wrote (ETXTBSY), a parallel-test race, not a product fault.
fn spawn_fake_worker(engine: &str, forward: bool) -> ChildStdioTransport {
    let mut attempts = 0;
    loop {
        match ChildStdioTransport::spawn_with_parent_env_forwarding(
            engine,
            "/weights",
            &[],
            Vec::<(String, String)>::new(),
            forward,
        ) {
            Ok(t) => return t,
            Err(e) if e.kind() == std::io::ErrorKind::ExecutableFileBusy && attempts < 50 => {
                attempts += 1;
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(e) => panic!("spawn fake worker: {e}"),
        }
    }
}

/// Spawn `body` as the engine, attempt the hello handshake, and return the failure message.
/// Panics if the handshake unexpectedly SUCCEEDS — every worker here is supposed to die.
fn handshake_failure(tag: &str, body: &str) -> String {
    handshake_failure_forwarding(tag, body, true)
}

/// As [`handshake_failure`] but with an explicit stderr-forwarding policy, so the OFFICIAL
/// (`forward = false`) posture can be exercised without a real `sandbox-exec`.
fn handshake_failure_forwarding(tag: &str, body: &str, forward: bool) -> String {
    let dir = std::env::temp_dir().join(format!("bench134-{tag}-{}", std::process::id()));
    let engine = write_script(&dir, body);
    let transport = spawn_fake_worker(&engine, forward);
    let message = match handshake_error(&dir, transport, tag) {
        // The post-mortem is engine-controlled, so it rides beside benchd's own text; the log
        // form (what these tests read) is the two joined, exactly as the message used to read.
        RunnerError::ProtocolEngineText {
            detail,
            engine_text,
        } => format!("{detail} ({engine_text})"),
        other => panic!("{tag}: expected a ProtocolEngineText error, got {other}"),
    };
    message
}

fn handshake_error(
    dir: &std::path::Path,
    transport: ChildStdioTransport,
    tag: &str,
) -> RunnerError {
    let err = match Session::connect(transport) {
        Ok(_) => panic!("{tag}: expected the handshake to fail"),
        Err(e) => e,
    };
    std::fs::remove_dir_all(dir).ok();
    err
}

/// Spawn `body` and return the handshake error itself.
fn handshake_failure_error(tag: &str, body: &str) -> RunnerError {
    let dir = std::env::temp_dir().join(format!("bench134-{tag}-{}", std::process::id()));
    let engine = write_script(&dir, body);
    let transport = spawn_fake_worker(&engine, false);
    handshake_error(&dir, transport, tag)
}

/// Every non-empty substring of `marker` at least 4 bytes long that `text` contains — a sealed
/// text may share a letter or two with the marker by chance, never a run.
fn marker_runs_in(text: &str, marker: &str) -> Vec<String> {
    let mut hits = Vec::new();
    for i in 0..marker.len() {
        for j in (i + 4)..=marker.len() {
            if text.contains(&marker[i..j]) {
                hits.push(marker[i..j].to_string());
            }
        }
    }
    hits
}

const PROMPT_MARKER: &str = "HIDDENPROMPTqzx7Kv9w";

/// The SEALED form (`Display`) of a dying worker's error carries benchd's own signature and the
/// digest of the post-mortem, and not one run of the engine's text; the log form carries it all.
#[test]
fn sealed_display_of_a_dead_worker_carries_a_digest_not_its_stderr() {
    let err = handshake_failure_error(
        "seal-stderr",
        &format!("#!/bin/sh\nprintf 'prompt was {PROMPT_MARKER}\\n' 1>&2\nexit 3\n"),
    );
    let sealed = err.to_string();
    assert!(
        sealed.starts_with(
            "protocol violation: engine closed the stream before returning a response"
        ),
        "{sealed}"
    );
    assert!(sealed.contains("engine-text-sha256="), "{sealed}");
    assert!(sealed.contains("engine-text-bytes="), "{sealed}");
    assert_eq!(
        marker_runs_in(&sealed, PROMPT_MARKER),
        Vec::<String>::new(),
        "{sealed}"
    );
    assert!(
        err.diagnostic().contains(PROMPT_MARKER),
        "{}",
        err.diagnostic()
    );
}

/// A JSON-shaped line that fails to decode quotes nothing of itself: serde's own Display would
/// say `invalid type: string "<marker>"`, the sealed error names only the category and position.
#[test]
fn a_decode_error_does_not_quote_the_engine_line() {
    let err = handshake_failure_error(
        "seal-decode",
        &format!(
            "#!/bin/sh\n\
             printf '{{\"id\":0,\"ok\":true,\"nonce\":\"n1\",\"protocol_version\":\"{PROMPT_MARKER}\"}}\\n'\n\
             exit 0\n"
        ),
    );
    for text in [err.to_string(), err.diagnostic()] {
        assert!(
            text.contains("engine response line could not be decoded: data error at line 1"),
            "{text}"
        );
        assert_eq!(
            marker_runs_in(&text, PROMPT_MARKER),
            Vec::<String>::new(),
            "{text}"
        );
    }
}

/// The signature `measure_job` seals into `rejected_pairs[].reason` must stay recognisable: the
/// fix APPENDS the post-mortem, it does not reword the leading clause.
fn assert_keeps_signature(tag: &str, message: &str) {
    assert!(
        message.starts_with("engine closed the stream before returning a response"),
        "{tag}: the sealed signature changed: {message}"
    );
}

/// A worker that writes a diagnostic to stderr and exits before saying hello — the plain
/// "engine refused to start" shape. Its reason must reach benchd's error.
#[test]
fn dies_before_hello_surfaces_stderr_and_status() {
    let message = handshake_failure(
        "before-hello",
        "#!/bin/sh\nprintf 'fatal: could not open weights\\n' 1>&2\nexit 3\n",
    );
    assert_keeps_signature("before-hello", &message);
    assert!(
        message.contains("fatal: could not open weights"),
        "worker stderr missing from the error: {message}"
    );
    assert!(
        message.contains("worker exited with status 3"),
        "worker exit status missing from the error: {message}"
    );
}

/// A worker that emits a non-JSON preamble line and a TRUNCATED (newline-less) hello, then dies
/// mid-handshake. The partial line is never a response, so the read still ends in EOF — and the
/// stderr that explains the truncation must still arrive.
#[test]
fn dies_mid_hello_surfaces_stderr_and_status() {
    let message = handshake_failure(
        "mid-hello",
        "#!/bin/sh\n\
         printf 'loading weights...\\n'\n\
         printf '{\"id\":0,\"ok\":tr' \n\
         printf 'panic: head provenance digest failed\\n' 1>&2\n\
         exit 5\n",
    );
    assert!(
        message.starts_with("engine response line could not be decoded"),
        "mid-hello: unexpected signature: {message}"
    );
    assert!(
        message.contains("panic: head provenance digest failed"),
        "worker stderr missing from the error: {message}"
    );
    assert!(
        message.contains("worker exited with status 5"),
        "worker exit status missing from the error: {message}"
    );
}

/// A worker that floods stderr with far more than the 64 KiB retained tail, then dies. The
/// diagnostic must stay bounded, must keep the LAST (most diagnostic) lines, and must not carry
/// the oldest ones.
#[test]
fn writes_more_than_the_tail_limit_then_dies_stays_bounded() {
    // 4000 lines x ~40 bytes each is comfortably past the 64 KiB tail cap.
    let message = handshake_failure(
        "flood",
        "#!/bin/sh\n\
         i=0\n\
         while [ $i -lt 4000 ]; do printf 'flood line %s padding padding padding\\n' \"$i\" 1>&2; i=$((i+1)); done\n\
         printf 'LAST WORDS: metal allocation failed\\n' 1>&2\n\
         exit 7\n",
    );
    assert_keeps_signature("flood", &message);
    assert!(
        message.contains("LAST WORDS: metal allocation failed"),
        "the final (most diagnostic) stderr line was dropped: {message}"
    );
    assert!(
        message.contains("worker exited with status 7"),
        "worker exit status missing from the error: {message}"
    );
    assert!(
        !message.contains("flood line 0 "),
        "the tail kept the OLDEST lines instead of rolling: {message}"
    );
    // Bounded by the retained-tail cap plus the short status/base prose around it.
    assert!(
        message.len() <= bench_runner::WORKER_STDERR_TAIL_BYTE_LIMIT + 1024,
        "diagnostic exceeded the retained-tail bound: {} bytes",
        message.len()
    );
}

/// A worker that closes stdout but KEEPS RUNNING, then writes its last words and exits.
///
/// This is the defect that produced Proof A's total blackout: teardown killed the child the
/// instant the handshake read saw EOF, so stderr written after the stdout close was lost
/// entirely — it was not even forwarded live. The grace period must let it finish.
#[test]
fn closes_stdout_keeps_running_then_speaks_is_not_killed_first() {
    let message = handshake_failure(
        "late-stderr",
        "#!/bin/sh\nexec 1>&-\nsleep 0.2\nprintf 'late: metal device init failed\\n' 1>&2\nexit 4\n",
    );
    assert_keeps_signature("late-stderr", &message);
    assert!(
        message.contains("late: metal device init failed"),
        "stderr written AFTER the stdout close was lost to teardown: {message}"
    );
    assert!(
        message.contains("worker exited with status 4"),
        "worker exit status missing from the error: {message}"
    );
}

/// A worker that closes stdout and then hangs forever must NOT wedge benchd: the grace period is
/// bounded, after which the child is killed and reported as such.
#[test]
fn closes_stdout_and_hangs_is_killed_after_the_grace() {
    let message = handshake_failure(
        "hang",
        "#!/bin/sh\nexec 1>&-\nprintf 'still alive, stdout closed\\n' 1>&2\nsleep 5\n",
    );
    assert_keeps_signature("hang", &message);
    assert!(
        message.contains("STILL RUNNING") && message.contains("killed by benchd"),
        "a hung worker was not reported as killed by benchd after the grace: {message}"
    );
    assert!(
        !message.contains("killed by signal"),
        "a harness kill was rendered as a worker-side signal death: {message}"
    );
    assert!(
        message.contains("still alive, stdout closed"),
        "stderr written before the hang was lost: {message}"
    );
}

/// The Proof A shape itself: a worker that dies by SIGNAL having written NOTHING. The absence of
/// stderr is the finding, so it must be stated — and the signal is then the only diagnostic there
/// is, which the old error discarded entirely.
#[test]
fn silent_signal_death_reports_the_signal_and_says_stderr_was_empty() {
    let message = handshake_failure("signal", "#!/bin/sh\nkill -9 $$\n");
    assert_keeps_signature("signal", &message);
    assert!(
        message.contains("worker was killed by signal 9"),
        "signal death was not identified: {message}"
    );
    assert!(
        message.contains("no worker stderr was emitted"),
        "an empty stderr must be stated, not left ambiguous: {message}"
    );
}

/// Redaction still governs what reaches the diagnostic. The reason string is SEALED into
/// `results.json`, so a worker that prints golden-adjacent content on its way down must not
/// smuggle it into the run record through this new channel.
#[test]
fn golden_adjacent_stderr_is_redacted_before_it_can_be_sealed() {
    let message = handshake_failure(
        "redact",
        "#!/bin/sh\nprintf 'token[5] expected=1234 actual=5678\\n' 1>&2\nexit 1\n",
    );
    assert_keeps_signature("redact", &message);
    assert!(
        message.contains("token-validation-failed"),
        "the redaction marker is missing: {message}"
    );
    assert!(
        !message.contains("1234") && !message.contains("5678"),
        "golden-adjacent tokens leaked into a sealable reason: {message}"
    );
}

/// A2 (reviewer's defeat of the original bound test) — an oversized BINARY line must not evict
/// the readable lines behind it.
///
/// The old loop `break`(s) on the first line that does not fit, scanning newest-first, so a single
/// 30 KB blob as the FINAL line discarded every readable line before it — reproducing #134's
/// blackout by a different route. The bound test could not catch this: `len <= LIMIT + 128` passes
/// most easily when the result is EMPTY.
#[test]
fn an_oversized_binary_final_line_does_not_evict_the_readable_lines() {
    let message = handshake_failure(
        "binary-evict",
        "#!/bin/sh\n\
         printf 'CRITICAL: metal device init failed\\n' 1>&2\n\
         printf 'CRITICAL: second readable line\\n' 1>&2\n\
         # ~30 KB of 0xFF on one line, no newline until the end\n\
         i=0; while [ $i -lt 300 ]; do printf '\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377\\377' 1>&2; i=$((i+1)); done\n\
         printf '\\n' 1>&2\n\
         exit 6\n",
    );
    assert_keeps_signature("binary-evict", &message);
    assert!(
        message.contains("CRITICAL: metal device init failed"),
        "the oversized final line evicted the readable diagnosis: {message}"
    );
    assert!(
        message.contains("CRITICAL: second readable line"),
        "the oversized final line evicted a readable line: {message}"
    );
    assert!(
        message.contains("worker exited with status 6"),
        "worker exit status missing: {message}"
    );
}

/// F2 (negative control) — the pre-existing redaction is a KEYWORD filter, so a test that feeds it
/// a line containing its trigger words only proves the filter fires. This proves the property that
/// actually matters: secret-SHAPED stderr carrying none of those keywords cannot reach the
/// diagnostic. Before the scrubbing in `bench_runner::scrub`, every assertion below failed.
#[test]
fn secret_shaped_stderr_without_trigger_keywords_cannot_reach_the_diagnostic() {
    let message = handshake_failure(
        "secrets",
        "#!/bin/sh\n\
         printf 'open /Users/operator/pool-goldens/sample-001.json failed\\n' 1>&2\n\
         printf 'AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCY\\n' 1>&2\n\
         printf 'host=api.example.internal user=operator@ranked-box\\n' 1>&2\n\
         printf 'HOME=/Users/operator\\n' 1>&2\n\
         exit 9\n",
    );
    // Nothing secret-tier survives.
    for secret in [
        "/Users/operator/pool-goldens",
        "wJalrXUtnFEMIK7MDENGbPxRfiCY",
        "api.example.internal",
        "operator@ranked-box",
        "HOME=/Users/operator",
    ] {
        assert!(
            !message.contains(secret),
            "secret-tier content reached the sealable diagnostic: {secret:?} in {message}"
        );
    }
    // The diagnosis still survives: the failing filename and the exit status.
    assert!(
        message.contains("sample-001.json"),
        "scrubbing destroyed the diagnostic basename: {message}"
    );
    assert!(
        message.contains("worker exited with status 9"),
        "worker exit status missing: {message}"
    );
}

/// F4 — erasure (iii) was the OFFICIAL path, where `forward_worker_stderr` is false. Retention
/// must not depend on forwarding: with forwarding OFF nothing is echoed to this process, and the
/// retained tail is the only channel a sealed record has.
#[test]
fn tail_still_reaches_the_diagnostic_with_forwarding_off() {
    let message = handshake_failure_forwarding(
        "no-forward",
        "#!/bin/sh\nprintf 'official: weights mmap refused\\n' 1>&2\nexit 8\n",
        false,
    );
    assert_keeps_signature("no-forward", &message);
    assert!(
        message.contains("official: weights mmap refused"),
        "an unforwarded worker's tail did not reach the diagnostic: {message}"
    );
    assert!(
        message.contains("worker exited with status 8"),
        "worker exit status missing: {message}"
    );
}

/// B1 / F5 — a LIVE worker that merely sends one undecodable line must NOT be described as having
/// died. benchd kills it after the grace, and that kill is OURS: rendering it through the wait
/// status would seal "worker was killed by signal 9" for a healthy engine, forging exactly the
/// evidence the Proof A retry discriminates on.
#[test]
fn live_worker_with_one_bad_line_is_not_reported_as_a_signal_death() {
    // JSON-shaped (so it is taken as a response) but the wrong type for `protocol_version`;
    // the worker then stays alive on stdin.
    let message = handshake_failure(
        "live-bad-json",
        "#!/bin/sh\n\
         printf '{\"id\":0,\"ok\":true,\"nonce\":\"n1\",\"protocol_version\":\"not-a-number\"}\\n'\n\
         cat > /dev/null\n",
    );
    assert!(
        message.starts_with("engine response line could not be decoded"),
        "unexpected signature: {message}"
    );
    assert!(
        !message.contains("killed by signal"),
        "a LIVE worker was falsely sealed as a signal death: {message}"
    );
    assert!(
        message.contains("STILL RUNNING") && message.contains("killed by benchd"),
        "the harness kill was not attributed to benchd: {message}"
    );
}

/// A successful handshake is untouched by any of this: no post-mortem is taken and the child is
/// still live afterwards.
#[test]
fn healthy_worker_still_completes_the_handshake() {
    let dir = std::env::temp_dir().join(format!("bench134-ok-{}", std::process::id()));
    // A minimal conformant hello, then block on stdin so the session stays open.
    let engine = write_script(
        &dir,
        "#!/bin/sh\n\
         printf '{\"id\":0,\"ok\":true,\"nonce\":\"n1\",\"protocol_version\":1}\\n'\n\
         cat > /dev/null\n",
    );
    let transport = spawn_fake_worker(&engine, true);
    let (session, hello) = Session::connect(transport).expect("handshake should succeed");
    assert_eq!(hello.nonce, "n1");
    assert_eq!(hello.protocol_version, Some(1));
    assert!(!session.is_discarded());
    std::fs::remove_dir_all(&dir).ok();
}
