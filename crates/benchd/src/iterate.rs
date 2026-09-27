//! WS1-8 — `benchd iterate`: drive the engine end-to-end and assemble a sealed
//! `score.json`.
//!
//! Flow (mirrors Swift `QwenRuntime.localIterate` +
//! `runLocalIterateCheckedTimingWithWorker`): connect a [`Session`] → run the
//! correctness gate against the golden (bench-core `conformance` anchor + free-run
//! checks) → run the WS1-6 parent-side timing (prefill + decode) → compute the
//! estimated score with `score_default_weights` → assemble the [`ScorePayload`].
//!
//! Like Swift local-iterate, the pass/fail decision is: correctness passed AND the
//! timings + baselines + score are finite and positive. Acceptance bands and
//! speedup floors are reported in the metrics (`passed_*_speedup_floor`) but do not
//! gate the local estimated score (that is the official/ranked path's job).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use bench_core::conformance::{
    run_conformance, AnchorOutput, ConformanceReport, CorrectnessScope, EngineHandle, TopLogit,
};
use bench_core::constants::WindowShape;
use bench_core::golden::{BenchmarkGolden, GoldenFixture, Token};
use bench_core::score::{score_weighted, speedup, ScoringWeights, SpeedupFloors};
use bench_core::BenchError;
use bench_protocol::SpecConfig;
use bench_runner::{
    run_timed_benchmark_persistent_on_session, scrub_reason_for_seal, LineTransport, RunnerError,
    Session, TimingParams, TimingResult, VerifyMode,
};
use sha2::{Digest, Sha256};

use crate::score::{LocalPhases, PhaseStatus, ScoreMetrics, ScorePayload, ScorePerPrompt};

/// Swift `QwenRuntime.bandwidthSource` for the RAM-resident dense runtime.
const BANDWIDTH_SOURCE: &str = "ram_resident_model";
/// Swift local-iterate/local-submit `timingRepeats` (the CLI passes 1). benchd runs a
/// single prefill+decode timing pass, so the checked-timing repeat count is always 1.
const TIMING_REPEATS: i64 = 1;

/// F1 — the WORKSPACE HARNESS IDENTITY this run seals into `metrics.harness_hash`.
///
/// A newtype rather than a bare `String` because the invariant is the whole point: an instance
/// EXISTS only for a workspace whose nine harness roots were all present and hashed. There is no
/// `Default`, no empty constructor and no public field, so no code path can reintroduce the
/// pre-F1 `String::new()` stub by accident — sealing an empty or partial harness identity now
/// requires deleting this type, not forgetting a line.
///
/// **Why benchd computes this at all.** `metrics.harness_hash` gates publication: a downstream
/// consumer refuses a score whose harness identity is empty or malformed, so a benchd-authored
/// score must carry it. And it MUST be computed here rather than read off the engine wire — the
/// worker is participant-built, so a wire-reported hash would be attacker-controlled. Trusted
/// value, trusted-side computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessIdentity(String);

impl HarnessIdentity {
    /// Resolve the harness identity of the workspace rooted at `workspace_root`.
    ///
    /// The hash is computed over the roster roots that EXIST — a root absent on this engine surface
    /// is logged and skipped (the roster spans both the MLX and CUDA engines), so this never refuses
    /// on a missing root. `Err` only if the digest is somehow not well-formed; that re-check is
    /// belt-and-braces, making "a `HarnessIdentity` exists" mean "a 64-lowercase-hex digest exists",
    /// which is exactly the predicate the overlay will apply downstream.
    pub fn resolve(workspace_root: &Path) -> Result<HarnessIdentity, String> {
        let hash = bench_core::harness_hash::harness_hash(workspace_root);
        if !bench_core::harness_hash::is_well_formed_harness_hash(&hash) {
            return Err(format!(
                "harnessHash produced a malformed digest (expected 64 lowercase hex characters, \
                 got length {}); refusing to seal a harness identity that will not validate",
                hash.len()
            ));
        }
        Ok(HarnessIdentity(hash))
    }

    /// [`HarnessIdentity::resolve`] over the process's current working directory — the reference's
    /// own production resolution, where the roster roots are CWD-relative. This is the workspace an
    /// `iterate` run drives: the facade invokes benchd from the engine workspace root, and the
    /// integrity sidecar already treats the CWD as that root (see `relativize_for_seal`).
    pub fn resolve_from_current_dir() -> Result<HarnessIdentity, String> {
        // `getcwd()` is what Foundation prepends to a relative `URL(fileURLWithPath:)`, so
        // resolving the CWD explicitly and resolving it implicitly are the same computation —
        // verified equal against the reference. Delegating keeps ONE resolution path rather than a
        // second copy of the well-formedness contract.
        let cwd = std::env::current_dir().map_err(|e| {
            format!("harnessHash: cannot resolve the current working directory: {e}")
        })?;
        HarnessIdentity::resolve(&cwd)
    }

    /// The 64-lowercase-hex digest.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// TEST-ONLY construction from a literal digest, so a payload builder can be exercised without
    /// a harness workspace on disk. `#[cfg(test)]`, so no production path can reach it.
    ///
    /// It ASSERTS well-formedness rather than accepting any string: a test fixture must not be able
    /// to smuggle in the empty identity this change exists to eliminate, which keeps the
    /// "`iterate` seals a 64-hex non-empty hash" assertions honest.
    #[cfg(test)]
    pub(crate) fn for_test(hash: &str) -> HarnessIdentity {
        assert!(
            bench_core::harness_hash::is_well_formed_harness_hash(hash),
            "a test HarnessIdentity must still be a 64-lowercase-hex digest, got {hash:?}"
        );
        HarnessIdentity(hash.to_string())
    }

    /// The identity every payload-builder test passes. A RECOGNISABLE constant, so an assertion can
    /// prove a payload sealed THIS value rather than an incidental 64-hex string.
    #[cfg(test)]
    pub(crate) const TEST_HASH: &'static str =
        "7e57000000000000000000000000000000000000000000000000000000000000";

    /// [`HarnessIdentity::TEST_HASH`] as an identity. One definition for all three test modules.
    /// `&'static` so a test can hand it to a borrowing [`RunDigests`] without a local binding.
    #[cfg(test)]
    pub(crate) fn test_default() -> &'static HarnessIdentity {
        static IDENTITY: std::sync::LazyLock<HarnessIdentity> =
            std::sync::LazyLock::new(|| HarnessIdentity::for_test(HarnessIdentity::TEST_HASH));
        &IDENTITY
    }
}

/// The RUN IDENTITY every payload builder seals: the two digests benchd computes — which weights
/// it measured (`metrics.weights_hash` + its byte/file counts) and which harness produced the
/// result (`metrics.harness_hash`) — plus the resolved per-track MODEL identity the run was loaded
/// and judged under.
///
/// They travel together through every payload builder — the correctness-failure builders, the
/// preflight refusal, the official gating failures, the passing assembly — and neither is ever
/// used without the other, so they are one parameter rather than two. That is also what keeps F1
/// from pushing five payload builders past the argument-count limit: adding the harness identity
/// alongside the weights digest costs those signatures nothing instead of forcing a lint waiver.
///
/// `golden_hash` is deliberately NOT in here: it comes from the loaded golden fixture, is blanked
/// on the failure paths (#132(b) MIRROR-BLANK-STRICTLY), and has different provenance from these
/// two, which benchd computes itself and seals unconditionally.
///
/// `model` rides here for exactly the same reason the harness identity does. It is resolved ONCE
/// per run, from the run's track id, at the point the platform is resolved
/// (`bench_core::contract::model_identity`); every payload builder then seals the SAME identity,
/// and the deterministic `metrics.num_layers` field can never describe a different checkpoint than
/// the one the golden was loaded under. It is the per-track replacement for the compile-time
/// `NUM_HIDDEN_LAYERS` constant this module used to carry.
#[derive(Debug, Clone, Copy)]
pub struct RunDigests<'a> {
    pub weights: &'a DirDigest,
    pub harness: &'a HarnessIdentity,
    /// BORROWED, like the other two. [`bench_core::constants::TrackModelIdentity`] owns its
    /// `golden_model_type` as a `String` — the track fixture supplies that string — so the identity
    /// is no longer `Copy`. Carrying a reference keeps `RunDigests` itself `Copy` and keeps the
    /// "resolved once, sealed identically by every payload builder" property, which is the whole
    /// reason the identity rides here.
    pub model: &'a bench_core::constants::TrackModelIdentity,
}

/// WHAT ONE RUN IS SCORED AGAINST: the baseline pair its speedups divide by, and the speedup
/// FLOORS those speedups must clear (David ruling 2026-09-09: 0.95 decode AND 0.95 prefill,
/// enforced, configurable per project).
///
/// The two travel together because a payload must never disagree with itself: the floors sealed in
/// `metrics.decode_speedup_floor` / `metrics.prefill_speedup_floor` are the SAME value
/// [`apply_timing_metrics`] decides `passed_*_speedup_floor` with and
/// [`bench_core::score::evaluate_timed_run`] gates the official run with. One value in, one value
/// enforced, one value sealed.
///
/// The floors come from the `--contract` track fixture on EVERY `benchd iterate` run
/// (`contract::speedup_floors`, which refuses a fixture that declares none). STEP 3 (David
/// 2026-09-15) removed the local no-contract default: the per-track tables are gone, so a run
/// without a fixture cannot resolve a window or a model shape either and refuses before it gets
/// here. [`SpeedupFloors::DEFAULT`] survives only on the UNSCORED calibration path
/// ([`ScoringInputs::local`]), which seals no score for a floor to gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoringInputs {
    pub baseline_prefill_spt: f64,
    pub baseline_decode_spt: f64,
    pub floors: SpeedupFloors,
    /// THE RUN'S RESOLVED COMPOSITE WEIGHTS (STEP 1 of the constants→contract migration, David
    /// 2026-09-15): the `--contract` track fixture's pair when it declares one, else
    /// `ScoringWeights::DEFAULT` — the 0.75/0.25 constants. It rides HERE, beside `floors`, for the
    /// same reason: one value carries the weighting from the resolution to both the score a run
    /// publishes and the gate that judges it, so what a run seals is what it weighted.
    pub weights: ScoringWeights,
}

impl ScoringInputs {
    /// The pair as the payload builders write it: `(prefill, decode)`.
    pub fn baselines(&self) -> (f64, f64) {
        (self.baseline_prefill_spt, self.baseline_decode_spt)
    }

    /// A run that established NO DENOMINATOR — a refusal before any leg measured one — carrying
    /// the rules it was ARMED with. The pair is `0.0/0.0` because none is invented: the pair a
    /// paired run seals is the pair it measured. The floors and weights are still the track's, so
    /// a refusal states what it would have enforced.
    pub fn no_denominator(floors: SpeedupFloors, weights: ScoringWeights) -> ScoringInputs {
        ScoringInputs {
            baseline_prefill_spt: 0.0,
            baseline_decode_spt: 0.0,
            floors,
            weights,
        }
    }

    /// An UNSCORED run's inputs: the caller's pair against inert floors and weights. NEVER
    /// reachable from a scored run — every scored path resolves both from the track fixture and
    /// refuses absence. The calibration path writes a capture record and seals no score, so the
    /// values here reach no verdict.
    pub fn local(baseline_prefill_spt: f64, baseline_decode_spt: f64) -> ScoringInputs {
        ScoringInputs {
            baseline_prefill_spt,
            baseline_decode_spt,
            floors: SpeedupFloors::DEFAULT,
            weights: ScoringWeights::DEFAULT,
        }
    }
}

#[cfg(test)]
impl<'a> RunDigests<'a> {
    /// A `RunDigests` over `weights` carrying the shared test harness identity and the Qwen 3.8
    /// 125B-A6B (MLX) model identity — the track this module's fixtures are written to, resolved
    /// through the one accessor so a fixture cannot drift from the table.
    pub(crate) fn for_test(weights: &'a DirDigest) -> RunDigests<'a> {
        RunDigests {
            weights,
            harness: HarnessIdentity::test_default(),
            model: crate::testgolden::identity_125b(),
        }
    }
}

/// EVERYTHING ONE LOCAL-ITERATE FLOW RUNS UNDER, resolved once and threaded as ONE value.
///
/// These are not independent dials. They are the run's resolved shape (`mode`, `strict`,
/// `timed_only`, `window`), what it is scored against (`scoring`), what it seals (`golden`,
/// `digests`), and how the worker is driven (`spec`) — and every function in the
/// flow took the same list, positionally. Passing the group leaves the SEAMS (the session and
/// the two closures) as the only other arguments, which is what removed the
/// too-many-arguments lint this module used to silence on four functions (#191).
///
/// A caller that varies one field writes a struct update:
/// `FlowParams { timed_only: true, ..flow }`.
#[derive(Debug, Clone)]
pub struct FlowParams<'a> {
    pub golden: &'a GoldenFixture,
    /// What the run is scored against: the baseline pair its speedups divide by and the floors
    /// they must clear.
    pub scoring: ScoringInputs,
    pub mode: Mode,
    /// R3: `false` is Swift-exact for the local checked-timing modes (base `cases[]` only);
    /// `true` restores benchd's historical SUPERSET (base cases + anchors + free-run). Official
    /// mode always runs the full superset regardless.
    pub strict: bool,
    /// `true` SKIPS the teacher-forced correctness gate and runs ONLY the timed pass — the
    /// `--capture-baseline --capture-timed-only` path (a8/David ruling, capture doc §5), where
    /// the engine cannot change between passes of the organizer's own calibration. NEVER set on
    /// a scored or official run: official routes through `official::official_core_windowed`, and
    /// this flag is set only by the `--capture-baseline` branch, which writes no score. The timed
    /// pass itself — the free-run decode and its free-run seed oracle check — is unchanged on
    /// every pass; only the separate teacher-forced `cases[]` verification is skipped.
    ///
    /// `timed_only = false` REQUIRES a session (the gate runs on it) and the flow `expect`s one.
    pub timed_only: bool,
    pub digests: RunDigests<'a>,
    /// The resolved `--mtp-depth` / `--candidate-spec` request that rides the timed leg. `None`
    /// is the historical bare request.
    pub spec: Option<SpecConfig>,
    /// THE RUN'S RESOLVED WINDOW (STEP 1, David 2026-09-15): the timed decode depth, the checked
    /// step count and the warm-up count, resolved ONCE from the `--contract` fixture and threaded,
    /// so every phase of this flow measures and checks the SAME window.
    pub window: &'a WindowShape,
}

/// Which decode window + runtime label this run uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// 128-step checked decode (`BENCHMARK_DECODE_STEPS`); `runtime = "rust-local-iterate"`.
    LocalIterate,
    /// 1023-step checked decode; `runtime = "rust-local-submit"` (Swift
    /// `QwenRuntime.localIterate` with `decodeSteps = 1023`, main.swift:264-291). Reuses the
    /// local-iterate checked-timing machinery over a long continuous decode — of the golden's
    /// `benchmark` oracle when present, else `cases[0]` (see [`local_iterate_timing_params`]).
    LocalSubmit,
    /// 128-step decode; `runtime = "rust"`. (Stub-only in this wave; iterate uses it
    /// so the timing/score plumbing is mode-parametric.)
    Official,
}

impl Mode {
    /// The TIMED DECODE DEPTH this mode measures, out of the run's RESOLVED window
    /// ([`bench_core::constants::WindowShape`], STEP 1 of the constants→contract migration, David
    /// 2026-09-15). It used to read the module constants directly, which made every consumer its
    /// own resolution point; the window is now resolved ONCE per run and threaded, so a fixture
    /// that declares its own depth cannot reach some call sites and miss others.
    ///
    /// `LocalIterate` is `benchmark_decode_steps` — the reference's own
    /// `localIterateBenchmarkDecodeSteps = benchmarkDecodeSteps`, unchanged.
    pub const fn decode_steps(self, window: &WindowShape) -> usize {
        match self {
            Mode::LocalIterate => window.benchmark_decode_steps,
            Mode::LocalSubmit => window.local_submit_benchmark_decode_steps,
            Mode::Official => window.benchmark_decode_steps,
        }
    }

    /// How many `expected_tokens` the golden LOADER must require for this mode, mirroring the
    /// reference's own load call. Swift `QwenRuntime.localIterate` loads with
    /// `requiredSteps: options.benchmarkDecodeSteps + 1`
    /// (`mlxfast-qwen-38-27b-mtp-engine/Sources/MLXFastTrustedHarness/QwenRuntimeLocalIterate.swift@ebe3446:101-105`)
    /// — the `+ 1` is the SEED token: `expected_tokens[0]` is the prefill/decode-seed argmax and
    /// `expected_tokens[k + 1]` is decode step `k`'s output (`:776-777`, `:854-858`). Swift's
    /// official `benchmark` path takes the loader DEFAULT (`correctnessSteps`,
    /// `QwenRuntimeBenchmark.swift:88`), so Official does too.
    ///
    /// EXHAUSTIVE match (LANDMINE #60): a new `Mode` must state its loader arity explicitly.
    pub fn golden_required_steps(self, window: &WindowShape) -> usize {
        match self {
            Mode::LocalIterate | Mode::LocalSubmit => self.decode_steps(window) + 1,
            Mode::Official => window.correctness_steps,
        }
    }

    /// The `runtime` audit label. benchd is a Rust producer, so this is
    /// `rust-*` rather than the Swift `swift-*` — the one intentional value
    /// difference from the Swift score (field NAMES are identical).
    pub fn runtime(self) -> &'static str {
        match self {
            Mode::LocalIterate => "rust-local-iterate",
            Mode::LocalSubmit => "rust-local-submit",
            Mode::Official => "rust",
        }
    }

    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "local-iterate" => Some(Mode::LocalIterate),
            "local-submit" => Some(Mode::LocalSubmit),
            "official" => Some(Mode::Official),
            _ => None,
        }
    }

    /// The Swift `modeName` (`options.modeName`) — used in error strings that must match
    /// Swift byte-for-byte (e.g. the missing-baseline preflight error). Not the `runtime`
    /// audit label (which is `rust-*`).
    pub fn mode_name(self) -> &'static str {
        match self {
            Mode::LocalIterate => "local-iterate",
            Mode::LocalSubmit => "local-submit",
            Mode::Official => "official",
        }
    }

    /// Whether the local GPU cool gate runs by default for this mode (David 2026-08-17):
    /// local-iterate **OFF** (opt-in via `--cool-gate`; the A3 window showed gating degrades
    /// its single-shot measurement); local-submit **ON** (long continuous decode, thermal
    /// consistency matters). Official has no cool gate (that path never calls it). The facade
    /// always passes `--cool-gate` so it matches benchmark.sh regardless of the native default.
    ///
    /// EXHAUSTIVE match (LANDMINE #60): no wildcard `_`, so a new `Mode` variant must decide
    /// its cool-gate default explicitly rather than silently inheriting `false`.
    pub fn cool_gate_on_by_default(self) -> bool {
        match self {
            Mode::LocalIterate => false,
            Mode::LocalSubmit => true, // (P6 RULING: submit ON)
            Mode::Official => true, // (David 2026-09-06: the 40 C contract holds on the ranked path)
        }
    }

    /// Whether this mode brands a primary `cases[]` (teacher-forced) base-case failure
    /// Swift-style — i.e. it derives correctness from the fused checked-timing pass rather
    /// than a standalone conformance report, so a base-case mismatch is reported with the
    /// `modeName`-keyed case id, Swift checked-step numbering, and the
    /// `"<modeName> teacher-forced token mismatch"` error (§F1). True for BOTH
    /// `LocalIterate` and `LocalSubmit` (Swift `QwenRuntime.localIterate` drives both; the
    /// branding keys on `modeName`, so it is mode-DERIVED on the Swift side too — the Rust
    /// F1 arm must not hard-code `Mode::LocalIterate`). Official runs the standalone superset
    /// and keeps its own case/step report.
    pub fn is_local_checked_timing(self) -> bool {
        match self {
            Mode::LocalIterate | Mode::LocalSubmit => true,
            Mode::Official => false,
        }
    }
}

/// A directory digest of the transformed weights (Swift `DirectoryDigest`).
///
/// P2: a **byte-for-byte port** of Swift `directoryDigest`
/// (mlxfast-challenge-dev/Sources/MLXFastHarness/QwenRuntimePreflight.swift:172-239) so
/// `weights_hash` MATCHES the Swift score (moves to the DETERMINISTIC differ bucket). The
/// tree hash is, over files sorted by relative path (ignoring exact relative paths
/// `.benchmark-source.sha256` and `.gitkeep`): `relpath || 0x00 || SHA256(file bytes) ||
/// 0x00`, then SHA256 of that stream, hex-encoded. Each file's own SHA256 is streamed in
/// 8 MiB chunks (Swift `fileDigest`). `byte_count`/`file_count` are exact.
#[derive(Debug, Clone, PartialEq)]
pub struct DirDigest {
    pub sha256: String,
    pub byte_count: i64,
    pub file_count: i64,
}

impl DirDigest {
    /// The empty digest (sha256 of no files), for when no weights dir is provided.
    /// Test-only today: every production path digests a real weights dir.
    #[cfg(test)]
    pub fn empty() -> Self {
        DirDigest {
            sha256: crate::score::sha256_hex(&[]),
            byte_count: 0,
            file_count: 0,
        }
    }
}

/// [`dir_digest`] of the WEIGHTS tree, with a one-line note on stderr of what the seal cost.
///
/// The weights tree is the big one — about 105 GiB (MLX) or 135 GB (CUDA) — so an operator
/// watching a run needs to see its price, and a slow store shows up here first. The line goes
/// to stderr, which no seal covers, and the digest itself is untouched, so no measured or
/// scored number moves. The digest also keeps its place BEFORE the first engine spawn, which
/// is outside every timed window.
pub fn dir_digest_weights(root: &Path) -> io::Result<DirDigest> {
    let started = Instant::now();
    // The weights tree is IMMUTABLE for the life of a window and is hashed by two processes
    // minutes apart — `benchd weights-digest` for the window, then a scored `iterate` that must
    // hash it again for itself (`--weights-digest` is refused outside `--capture-baseline`, so a
    // scored seal never takes the value on trust). The sidecar carries the per-file digests
    // across that gap. Scoped to THIS call by the guard: a workspace walk later in the same
    // process is memo-only and never writes source-tree entries into it. The digest is assembled
    // from per-file digests exactly as before, so the value is byte-identical either way.
    let _weights_cache = crate::file_digest::weights_cache_session();
    let digest = dir_digest(root)?;
    eprintln!(
        "weights digest: {} bytes in {:.1} s, {} threads",
        digest.byte_count,
        started.elapsed().as_secs_f64(),
        crate::file_digest::workers_for(digest.file_count.max(0) as usize),
    );
    Ok(digest)
}

/// Compute a digest over the files under `root`, ignoring `.benchmark-source.sha256`
/// and `.gitkeep` (matching the Swift `ignoredRelativePaths`).
pub fn dir_digest(root: &Path) -> io::Result<DirDigest> {
    dir_digest_with_workers(root, None)
}

/// [`dir_digest`] with the hashing worker count pinned (tests pin it; production passes
/// `None` for `min(available_parallelism, DIR_DIGEST_MAX_WORKERS, file count)`).
fn dir_digest_with_workers(root: &Path, force_workers: Option<usize>) -> io::Result<DirDigest> {
    let mut files: Vec<(String, PathBuf)> = Vec::new();
    collect_files(root, root, &mut files)?;
    // Swift ignores by EXACT relative path (root-level), not by basename — a nested
    // `subdir/.gitkeep` is NOT ignored. Match that.
    files.retain(|(rel, _)| rel != ".benchmark-source.sha256" && rel != ".gitkeep");
    files.sort_by(|a, b| a.0.cmp(&b.0));

    // Every file's SHA256 depends on that file's bytes alone, and the fold below still feeds
    // the tree hasher strictly in sorted-relpath order, so hashing the files concurrently is
    // byte-identical to the serial loop — it only stops the ~78 GiB seal being bound to one
    // reader thread (~56 s of a scored run's wall time, before the warmup leg).
    //
    // The pool, the streamed 8 MiB F_NOCACHE read and the process memo are
    // `crate::file_digest` — the ONE per-file content digest in benchd. On a `measure-job`
    // the divergence gate has already hashed the SOURCE files of these same two workspaces,
    // so those files are served from the memo here and never re-opened; `.build`, `.git` and
    // `weights/`, which that gate holds out, are hashed here for the first time.
    let paths: Vec<&Path> = files.iter().map(|(_, path)| path.as_path()).collect();
    let hashed = crate::file_digest::sha256_files_parallel(&paths, force_workers)
        .map_err(io::Error::from)?;

    let mut tree = Sha256::new();
    let mut byte_count = 0i64;
    let mut file_count = 0i64;
    for (idx, file_sha, size) in hashed {
        // Per-file SHA256 (streamed, 8 MiB chunks — Swift `fileDigest`), then fold
        // `relpath || 0x00 || file_sha256(32 raw) || 0x00` into the tree hasher.
        byte_count += size as i64;
        file_count += 1;
        tree.update(files[idx].0.as_bytes());
        tree.update([0u8]);
        tree.update(file_sha);
        tree.update([0u8]);
    }
    // #58: the tree hash is STREAMED (multi-GB safetensors are never buffered whole), so it
    // cannot use the one-shot `sha256_hex` — but it renders through the same `hex_lower` the
    // one-shot helper uses, so both spell a digest identically.
    let sha256 = bench_core::hash::hex_lower(&tree.finalize());
    Ok(DirDigest {
        sha256,
        byte_count,
        file_count,
    })
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, out)?;
        } else {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push((rel, path));
        }
    }
    Ok(())
}

// --- Session -> EngineHandle adapter ---------------------------------------

/// Adapts a worker [`Session`] to the bench-core [`EngineHandle`] the conformance
/// gate needs: `correctness_begin`/`correctness_step` for teacher-forced primary + anchor
/// cases, and `correctness` for free runs. Also carries the per-sequence barrier: each
/// correctness sequence is its own sub-phase (see `drain_allocator`).
pub(crate) struct SessionEngine<'a, T: LineTransport> {
    pub(crate) session: &'a mut Session<T>,
    /// Whether `drain_allocator` has opened at least one sub-phase yet (so the first
    /// call opens without a preceding close).
    pub(crate) drained_once: bool,
}

fn to_bench_err(e: RunnerError) -> BenchError {
    BenchError::InvalidInput(format!("{e}"))
}

/// Convert a teacher-forced worker response (`correctness_begin`/`correctness_step`) into
/// the bench-core `AnchorOutput` (argmax token + top-k logits) the gates evaluate.
fn resp_to_anchor_output(resp: bench_protocol::WorkerResponse) -> Result<AnchorOutput, BenchError> {
    let token = resp.token.ok_or_else(|| {
        BenchError::InvalidInput("teacher-forced response missing token".to_string())
    })?;
    let top_logits = resp
        .top_logits
        .unwrap_or_default()
        .into_iter()
        .map(|l| TopLogit {
            token: l.token,
            logit: l.logit,
        })
        .collect();
    Ok(AnchorOutput { token, top_logits })
}

impl<T: LineTransport> EngineHandle for SessionEngine<'_, T> {
    fn anchor_forward(&mut self, context_tokens: &[Token]) -> Result<AnchorOutput, BenchError> {
        let resp = self
            .session
            .correctness_begin(context_tokens)
            .map_err(to_bench_err)?;
        resp_to_anchor_output(resp)
    }

    fn free_run(
        &mut self,
        prompt_tokens: &[Token],
        steps: usize,
    ) -> Result<Vec<Token>, BenchError> {
        let resp = self
            .session
            .correctness(prompt_tokens, steps as i64)
            .map_err(to_bench_err)?;
        Ok(resp.tokens.unwrap_or_default())
    }

    fn teacher_forced(
        &mut self,
        prompt_tokens: &[Token],
        forced_tokens: &[Token],
        steps: usize,
    ) -> Result<Vec<AnchorOutput>, BenchError> {
        // B3 / Swift compareTeacherForcedWithWorker: begin(prompt) -> prediction 0, then
        // feed the GOLDEN token each step -> prediction i+1, for `steps` predictions.
        let mut outputs = Vec::with_capacity(steps);
        let first = self
            .session
            .correctness_begin(prompt_tokens)
            .map_err(to_bench_err)?;
        outputs.push(resp_to_anchor_output(first)?);
        for &forced in forced_tokens.iter().take(steps.saturating_sub(1)) {
            let resp = self
                .session
                .correctness_step(forced)
                .map_err(to_bench_err)?;
            outputs.push(resp_to_anchor_output(resp)?);
        }
        Ok(outputs)
    }

    /// V1: route the per-sequence anti-memoization drain to a real worker. Each correctness
    /// sequence is its own barrier sub-phase: on the boundary BEFORE a new sequence, close
    /// the previous sub-phase — `phase_diagnostics` reaches the worker (a conformant engine
    /// drains its allocator at the sequence boundary) and the completed-work barrier is
    /// reconciled for the sequence just finished — then open the next. The first call only
    /// opens (nothing precedes it). The final sub-phase is closed by iterate_core after the
    /// gate, before prefill, so no correctness completed-work leaks into the timed phases.
    /// (Verifying the engine reports `cacheMemory == 0` needs a new response field and is a
    /// tracked follow-up; this makes the drain a real worker round-trip instead of a no-op.)
    fn drain_allocator(&mut self) -> Result<(), BenchError> {
        if self.drained_once {
            self.session.close_phase().map_err(to_bench_err)?;
        }
        self.drained_once = true;
        self.session.begin_phase();
        Ok(())
    }
}

// --- iterate core ----------------------------------------------------------

/// THE local-iterate flow: ONE worker per window (David 2026-08-30 load-once).
///
/// The correctness gate always runs on the shared `session` (when present), and the timing runs over
/// that SAME resident worker. A `--capture-timed-only` pass has no correctness session, so it opens
/// ONE worker via `spawn_timed` and runs both timed phases on it. The model therefore loads ONCE and
/// two full-model residencies are never live at once (on MLX the correctness session and a separate
/// timed worker were exactly the concurrent ~90 GiB residencies the box-4 190 GiB pressure was made
/// of). `spawn_timed` is NOT called on the gated pass.
pub fn iterate_flow_windowed<T, F, G>(
    session: Option<&mut Session<T>>,
    flow: FlowParams<'_>,
    spawn_timed: F,
    mut cool_gate: G,
) -> ScorePayload
where
    T: LineTransport,
    F: FnMut() -> bench_runner::Result<Session<T>>,
    G: FnMut(&str) -> bench_runner::Result<()>,
{
    let mode = flow.mode;
    let mut phases = LocalPhases::default();
    let timing_status = std::cell::Cell::new(PhaseStatus::NotRun);
    // Observe the existing gate without changing its threshold, retries, or timing.
    let observed_cool_gate = |phase: &str| {
        cool_gate(phase)?;
        timing_status.set(PhaseStatus::Failed);
        Ok(())
    };
    let mut payload = iterate_flow_with_phase_status(
        session,
        flow,
        spawn_timed,
        observed_cool_gate,
        &mut phases,
        &timing_status,
    );
    phases.timing = timing_status.get();
    // Local-only diagnostic: the official artifact carries no `local_phases` key.
    if mode.is_local_checked_timing() {
        payload.metrics.local_phases = Some(phases);
    }
    payload
}

fn iterate_flow_with_phase_status<T, F, G>(
    mut session: Option<&mut Session<T>>,
    flow: FlowParams<'_>,
    mut spawn_timed: F,
    mut cool_gate: G,
    phases: &mut LocalPhases,
    timing_status: &std::cell::Cell<PhaseStatus>,
) -> ScorePayload
where
    T: LineTransport,
    F: FnMut() -> bench_runner::Result<Session<T>>,
    G: FnMut(&str) -> bench_runner::Result<()>,
{
    let FlowParams {
        golden,
        scoring,
        mode,
        strict,
        timed_only,
        digests,
        spec,
        window,
    } = flow;
    // a8 capture-timed-only: passes 2-4/B skip the whole correctness gate below and drop to timing.
    if !timed_only {
        // a8 (finish #236): `timed_only == false` GUARANTEES a correctness session — `run_local_iterate`
        // now spawns one ONLY on this path (the gate runs on it), and every gated caller passes
        // `Some`. Bind it once here so the gate body below is byte-for-byte unchanged. `timed_only ==
        // true` never reaches this branch, so a `None` session is never unwrapped: the timed-only pass
        // that carries no correctness session (its spawn skipped) runs straight to the timing phase.
        // Reborrow the correctness session WITHOUT consuming the outer `session` Option, so on the
        // gated pass path the SAME session survives this gate block and is reused for the
        // timed phases below (load-once: one worker for the whole window). The reborrow's mutable borrow
        // of `session` ends at this block's close (its last use is inside the block), so the timing
        // section is free to move `session` afterwards.
        let sess = session
            .as_deref_mut()
            .expect("a flow with timed_only=false requires a correctness session for the gate");
        // 1. Correctness gate (primary teacher-forced cases + anchor + free-run) via bench-core
        // conformance. B1+V1: each correctness sequence's teacher-forced steps
        // (correctness_begin/step) are timed steps that bump the engine's completed_work, and
        // each sequence must be drained. `SessionEngine::drain_allocator` runs each sequence in
        // its own barrier sub-phase (close-previous-then-open-next via phase_diagnostics); the
        // LAST sub-phase is closed here, after the gate, before prefill — otherwise a
        // sequence's completed-work leaks into prefill's close_phase and trips the WS1-7 barrier
        // against a conformant engine. Swift's monolith runs correctness in-process with no
        // benchd↔engine split, so this bracketing is benchd-specific, not a Swift port.
        // R3: Swift-exact by default for the local checked-timing modes (local-iterate AND
        // local-submit: base cases ONLY); the superset is opt-in via `--strict`. Official always
        // runs the full superset.
        // Swift-exact by default for the local checked-timing modes (local-iterate AND
        // local-submit are both `QwenRuntime.localIterate`, which derives correctness solely from
        // the primary `cases.first` timing stream): evaluate base cases ONLY. `--strict` restores
        // benchd's superset (base cases + anchors + free-run). Official always runs the superset.
        let scope = if mode.is_local_checked_timing() && !strict {
            CorrectnessScope::BaseCasesOnly
        } else {
            CorrectnessScope::Full
        };
        phases.correctness = PhaseStatus::Failed;
        phases.correctness_checked_steps = None;
        let report = {
            let mut adapter = SessionEngine {
                session: &mut *sess,
                drained_once: false,
            };
            match run_conformance(
                &mut adapter,
                golden,
                bench_core::constants::CORRECTNESS_STEPS,
                scope,
                digests.model.vocab_size,
            ) {
                Ok(r) => r,
                Err(e) => {
                    return failed_payload(
                        mode,
                        FailureReport::message(format!("{e}"), false),
                        golden,
                        digests,
                        scoring,
                    )
                }
            }
        };
        phases.correctness = if report.passed {
            PhaseStatus::Passed
        } else {
            PhaseStatus::Failed
        };
        phases.correctness_checked_steps = Some(report.checked_steps());
        // #132(b) FINAL: the fail-path case count used to be computed here (the golden's TOTAL
        // correctness case count) and threaded into every blanked failure payload. Under
        // MIRROR-BLANK-STRICTLY it is zero on every one of those paths, so the binding is gone rather
        // than kept and ignored — the paths that DO report real counts derive their own from the
        // checked-timing pass, not from the golden's roster.

        // B1/V1 — ONE OWNER for the bracketing: close the LAST correctness sub-phase here,
        // reconciling the final sequence's timed steps and resetting the engine's completed_work
        // before prefill, on BOTH the pass exit AND the correctness-fail exit below (earlier
        // sub-phases were closed by drain_allocator between sequences; the conformance-Err exit
        // above already left a tainted, discarded session that cannot be closed). Do it before
        // branching on report.passed so a failed gate can never leak completed-work past here.
        if let Err(e) = sess.close_phase() {
            return failed_payload(
                mode,
                FailureReport::message(format!("{e}"), report.passed),
                golden,
                digests,
                scoring,
            );
        }

        if !report.passed {
            let failure = first_conformance_failure(&report);
            // §F1: local-iterate reports a base-case (`cases[]`) failure EXACTLY as Swift's fused
            // timing/correctness pass does — case = "local-iterate", checked-step numbering,
            // expected/actual populated, descriptive error. Anchor/free-run failures are the
            // benchd SUPERSET (Swift `--local-iterate` skips those gates), so they keep
            // benchd's own case/step report.
            let failure_report = match &failure {
                // Only WITHIN Swift's checked window (prefill/seed + `decode_steps` decode steps,
                // i.e. expected index ≤ decode_steps) does Swift see the token — beyond it Swift's
                // timing pass never reaches that index, so it PASSES and a base-case failure there
                // is the benchd SUPERSET (falls through to the arm below, keeping the case id).
                // MODE-DERIVED (§F1 / M-6): keys on `mode.is_local_checked_timing()` (true for BOTH
                // local-iterate AND local-submit) and brands with `mode.mode_name()` — Swift's
                // branding uses `modeName`, so submit's case id / error read "local-submit".
                Some(f)
                    if mode.is_local_checked_timing()
                        && f.is_base_case
                        && f.step
                            .is_some_and(|i| i as usize <= mode.decode_steps(window)) =>
                {
                    FailureReport::correctness(
                        format!("{} teacher-forced token mismatch", mode.mode_name()),
                        Some(mode.mode_name().to_string()),
                        f.step.map(|i| swift_local_iterate_checked_step(i as usize)),
                        f.expected,
                        f.actual,
                    )
                }
                // The benchd SUPERSET report: the error IS the failing case's name.
                Some(f) => FailureReport::correctness(
                    f.case.clone(),
                    Some(f.case.clone()),
                    f.step,
                    f.expected,
                    f.actual,
                ),
                None => {
                    FailureReport::correctness("correctness gate failed", None, None, None, None)
                }
            };
            // ITEM 1 (David's ruling — "Swift is the reference"): on a LOCAL-ITERATE correctness
            // FAILURE, RETAIN real timing/baselines/speedup-floor flags instead of blanking them
            // to 0/false. Swift's `--local-iterate` measures timing by TEACHER-FORCING the expected
            // tokens and timing wall-clock — correctness is a SEPARATE judgment — so a corrupted
            // cases[0] still yields real `decode_spt`, real baselines, and real floor flags. benchd
            // must match: run the TIME-ONLY (mismatch-tolerant) timing phase and emit a
            // failed-WITH-real-timing payload. Official/Track-B keep the blanked failed payload.
            if mode.is_local_checked_timing() {
                match local_iterate_timing_params(golden, mode, spec.clone(), window) {
                    // The `expected_tokens.len() <= decode_steps` early-fail genuinely cannot time
                    // (no full window to teacher-force), so it keeps the blanked failed payload.
                    Err(_too_short) => {
                        return failed_payload(mode, failure_report, golden, digests, scoring);
                    }
                    Ok(params) => {
                        // Retain-real-timing on a correctness FAILURE, over the SAME correctness
                        // worker (`sess`, healthy after the close_phase above), so the window never
                        // holds the correctness session and a second timed worker at once. This is a
                        // FAILED run (no score), so it feeds no ratio. TimeOnly tolerates the
                        // mismatch.
                        let timeonly_result = run_timed_benchmark_persistent_on_session(
                            sess,
                            &mut cool_gate,
                            &params,
                            VerifyMode::TimeOnly,
                        );
                        match timeonly_result {
                            Ok(timing) => {
                                timing_status.set(PhaseStatus::Passed);
                                return failed_with_real_timing_payload(
                                    mode,
                                    failure_report,
                                    golden,
                                    digests,
                                    &timing,
                                    scoring,
                                    window,
                                );
                            }
                            // The time-only timing pass itself failed (protocol / thermal abort /
                            // completed-work barrier) — those are real infra faults, not a token
                            // mismatch. Fall back to the blanked failed payload; the correctness
                            // failure is still reported.
                            Err(_timing_err) => {
                                return failed_payload(
                                    mode,
                                    failure_report,
                                    golden,
                                    digests,
                                    scoring,
                                );
                            }
                        }
                    }
                }
            }

            // #132/F-5 — UNREACHABLE in practice, and deliberately kept. Every arm of the
            // `mode.is_local_checked_timing()` block above returns, so this fall-through needs a mode
            // that is neither local-iterate nor local-submit — i.e. Official, which `execute_iterate`
            // routes to `official::official_core` and which `iterate_core` `unreachable!()`s on two
            // matches below. So it has NO test: there is no way to reach it without first defeating
            // that routing, and a test that defeated the routing would be pinning a fiction.
            //
            // It is covered structurally instead: it calls the same `failed_payload`, which applies
            // the #132(b) blank seal unconditionally, so it cannot diverge from the six tested sites
            // even if a future mode makes it live.
            return failed_payload(mode, failure_report, golden, digests, scoring);
        }
    } // end `if !timed_only` — capture-timed-only drops straight to the timing phase below.

    // 2. Timing (WS1-6). A8/DAVID RULING (author-with-what-you-replay), SCOPED SUPERSESSION of
    // RULING 1: when the golden carries a `benchmark` block, LOCAL-ITERATE/LOCAL-SUBMIT time the
    // benchmark ORACLE workload — built IDENTICALLY to `official::official_core` (prefill prompt +
    // decode seed + oracle all from `golden.benchmark`) — so the captured decode-seconds-per-token
    // (the official baseline DENOMINATOR) is measured over the SAME workload official scores. The
    // timed decode leg replays the incremental FREE-RUN path (PR #234), and the benchmark oracle is
    // now FREE-RUN authored (`record-correctness-golden --benchmark-free-run`), so replay and oracle
    // agree — closing the deterministic greedy fork between free-run and teacher-forced trajectories.
    // RULING 1 ("time cases[0], not the benchmark oracle") is PRESERVED for a benchmark-LESS golden
    // (participant goldens, `benchmark: None`): the fallback times cases[0] exactly as Swift
    // `--local-iterate` (prefill cases[0].prompt_tokens -> expected[0]; decode seeds the same prompt,
    // teacher-forces expected_tokens[1..=decode_steps]). The teacher-forced `cases[]` stay the
    // UNTIMED correctness gate's mechanism either way.
    let params = match mode {
        // local-submit reuses the local-iterate checked-timing machinery (Swift
        // `QwenRuntime.localIterate` with `decodeSteps = 1023`), so it times `cases[0]`
        // identically — only `decode_steps` differs (via `mode.decode_steps(&test_window())`).
        Mode::LocalIterate | Mode::LocalSubmit => {
            match local_iterate_timing_params(golden, mode, spec.clone(), window) {
                Ok(p) => p,
                // The `expected_tokens.len() <= decode_steps` early-fail (Swift guard): a genuine
                // "can't time" — correctness PASSED here, so `passed_correctness = true`.
                Err(e) => {
                    return failed_payload(
                        mode,
                        FailureReport::message(e, true),
                        golden,
                        digests,
                        scoring,
                    )
                }
            }
        }
        // Official is NOT a local checked-timing run: it uses the timed-FIRST, three-fresh-
        // sandboxed-worker flow in `crate::official` (B-2), never this one. The CLI routes
        // official there; this flow is local-iterate / local-submit only.
        Mode::Official => {
            unreachable!(
                "official mode must run through official::official_core_windowed, not the local \
                 iterate flow"
            )
        }
    };
    // §A — the engine lifecycle is a PER-MODE property. ONE worker per window (David 2026-08-30
    // load-once).
    let timing_result = match mode {
        // LOCAL-ITERATE / LOCAL-SUBMIT timing runs over the SAME resident worker as the correctness
        // gate — the gated pass reuses `session` (the correctness worker, healthy after its
        // close_phase), and a `--capture-timed-only` pass (no correctness session) opens ONE worker
        // via `spawn_timed` and runs both timed phases on it. The model loads ONCE and the
        // correctness session + a second timed worker never co-reside (the concurrent MLX
        // residencies box-4's 190 GiB was made of). The per-phase isolation is the worker's own
        // `beginMeasuredPhase` reset, which fires on `prefill` and `free_decode_begin`, so a
        // persistent worker gets the same reset a fresh one would; the ~1.6% warm-cache gap is
        // closed by that reset, and since the captured baseline and the scored candidate BOTH run
        // this mode the ratio stays unbiased.
        Mode::LocalIterate | Mode::LocalSubmit => match session {
            Some(resident) => run_timed_benchmark_persistent_on_session(
                resident,
                &mut cool_gate,
                &params,
                VerifyMode::Verify,
            ),
            None => match spawn_timed() {
                Ok(mut resident) => run_timed_benchmark_persistent_on_session(
                    &mut resident,
                    &mut cool_gate,
                    &params,
                    VerifyMode::Verify,
                ),
                Err(e) => Err(e),
            },
        },
        // Official never reaches this flow at all (see the params match above).
        Mode::Official => {
            unreachable!(
                "official mode must run through official::official_core_windowed, not the local \
                 iterate flow"
            )
        }
    };
    let timing = match timing_result {
        Ok(t) => {
            timing_status.set(PhaseStatus::Passed);
            t
        }
        Err(e) => {
            return failed_payload(
                mode,
                FailureReport::message(format!("{e}"), true),
                golden,
                digests,
                scoring,
            )
        }
    };

    // 3. Assemble the local estimated score.
    local_iterate_score(mode, &timing, scoring, golden, digests, window)
}

/// Run N capture passes over ONE persistent residency — the minimal a8 "Option-A capture
/// restructure" (David/a8 2026-08-31). ONE `iterate --capture-baseline --capture-passes <spec>`
/// invocation drives EVERY listed pass over the SAME resident worker, so the model loads ONCE for
/// the whole per-prompt window (8 loads per calibration window, one per prompt) instead of once per
/// pass (40).
///
/// SCORING-CRITICAL — THE MEASURED NUMBER PER PASS IS UNCHANGED. Every pass funnels through the
/// SAME [`iterate_flow_windowed`] a single-pass `--capture-baseline` invocation calls, with the
/// SAME `(golden, baselines, mode, strict, digests)`. The ONLY difference from N
/// separate invocations is that the passes share one residency instead of each spawning a fresh
/// worker that reloads the model. The per-pass timed measurement (`measure_prefill` +
/// `measure_free_run_decode`, the seconds-per-token formulas, the [`TimingResult`]) is therefore
/// byte-identical to a standalone pass.
///
/// PER-PHASE ISOLATION BETWEEN PASSES IS THE SAME #28 RESET. The worker's own `beginMeasuredPhase`
/// reset — re-attest + 6 GiB `cacheLimit` reset + `clearCache` — is REQUEST-triggered, fired at the
/// start of EVERY measured phase's opening verb (`prefill`, then `free_decode_begin`). It therefore
/// fires on pass K's prefill exactly as it fired on pass 1's, so pass K enters on the identical
/// starting state a freshly-spawned worker would — the accepted isolation substitute (a8/David
/// 2026-08-30, the same reset #239 relies on for the within-pass gate→timing and prefill→decode
/// boundaries). No warm state crosses a pass boundary beyond what that reset addresses.
///
/// GATE-ONCE-PER-WINDOW (a8): the FIRST pass runs the teacher-forced correctness gate
/// (`timed_only = false`); every later pass is `timed_only = true` (gate skipped), because the
/// engine cannot change between passes of the organizer's own calibration. This is exactly the
/// per-pass split the driver expressed across separate invocations (`[W]` warmup gated, A/B measured
/// with `--capture-timed-only`), now folded into one invocation. Every pass — the gated first one
/// included — still measures and returns its pair; whether a pass's timing is excluded (the warmup)
/// is an ANALYSIS-time decision on the emitted per-label records, not a benchd one.
///
/// Returns one [`ScorePayload`] per pass, in label order; the caller applies the capture
/// refuse-gates and merges each pass's pair into that pass's per-label record.
pub fn run_capture_passes_over_session<T, F, G>(
    session: &mut Session<T>,
    flow: FlowParams<'_>,
    pass_count: usize,
    mut spawn_timed: F,
    mut cool_gate: G,
) -> Vec<ScorePayload>
where
    T: LineTransport,
    F: FnMut() -> bench_runner::Result<Session<T>>,
    G: FnMut(&str) -> bench_runner::Result<()>,
{
    let mut payloads = Vec::with_capacity(pass_count);
    for _ in 0..pass_count {
        // GATELESS (a8/David 2026-08-31): EVERY pass is timed-only — the teacher-forced
        // correctness gate is dropped from calibration ENTIRELY (David's #36 ruling applies to
        // calibration universally, both platforms), so no pass (not even pass 0) runs it. The
        // correctness evidence is the free-run oracle token-match on every measured pass; the
        // goldens are session-authored+validated on both CUDA and MLX. `Some(&mut *session)`
        // reborrows the ONE resident worker for every pass, so every timed leg runs on that single
        // residency — the model never reloads. `&mut spawn_timed` / `&mut cool_gate` pass the
        // closures by mutable reference so the loop reuses them across passes (a `&mut FnMut` is
        // itself `FnMut`); `spawn_timed` is never CALLED on this path.
        let payload = iterate_flow_windowed(
            Some(&mut *session),
            FlowParams {
                timed_only: true,
                ..flow.clone()
            },
            &mut spawn_timed,
            &mut cool_gate,
        );
        payloads.push(payload);
    }
    payloads
}

/// Build the LOCAL-ITERATE / LOCAL-SUBMIT timing params (a8/David ruling, SCOPED supersession of
/// RULING 1):
///
/// * `golden.benchmark` PRESENT → time the benchmark ORACLE workload, built IDENTICALLY to
///   `official::official_core` ([`benchmark_timing_params`]). The captured decode-seconds-per-token
///   is the official baseline denominator, so it must be measured over the workload official scores;
///   the benchmark oracle is free-run authored, matching the free-run the timed leg replays.
/// * `golden.benchmark` ABSENT (participant golden) → time `cases[0]` exactly as Swift
///   `--local-iterate` ([`cases0_timing_params`]) — RULING 1 preserved.
///
/// Shared by the correctness-PASS timing path AND the correctness-FAILURE time-only path (ITEM 1):
/// the timed workload is identical; only whether tokens are VERIFIED differs.
///
/// SPEC (`--mtp-depth`): the resolved per-module speculative config rides on the TIMED free-run
/// decode window of BOTH shapes. `None` (the default) is byte-for-byte today's no-spec request; a
/// `Some` arms spec-never-ignored, so the engine's `effective_spec` echo must equal it or the leg is
/// discarded. The teacher-forced correctness gate carries NO spec — see `iterate_flow_windowed`.
fn local_iterate_timing_params(
    golden: &GoldenFixture,
    mode: Mode,
    spec: Option<SpecConfig>,
    window: &WindowShape,
) -> std::result::Result<TimingParams, String> {
    let decode_steps = mode.decode_steps(window);
    let params = match &golden.benchmark {
        Some(benchmark) => benchmark_timing_params(benchmark, decode_steps),
        None => cases0_timing_params(golden, decode_steps),
    };
    params.map(|p| p.with_spec(spec))
}

/// Time the golden's `benchmark` ORACLE — prefill prompt + decode seed + oracle all from
/// `golden.benchmark`, the SAME construction as `official::official_core` (official.rs:208-215).
/// Returns `Err(message)` when the oracle is shorter than the mode's decode window (a full window is
/// required to time) — a genuine "can't time" that stays a hard fail, mirroring the `cases[0]` guard.
fn benchmark_timing_params(
    benchmark: &BenchmarkGolden,
    decode_steps: usize,
) -> std::result::Result<TimingParams, String> {
    if benchmark.expected_decode_tokens.len() < decode_steps {
        return Err(format!(
            "benchmark oracle has {} decode tokens; timing this window needs at least {decode_steps}",
            benchmark.expected_decode_tokens.len()
        ));
    }
    Ok(benchmark_window_params(benchmark, decode_steps))
}

/// The golden `benchmark` ORACLE as a timed window of `decode_steps`: prefill prompt + expected
/// prefill token, decode seed + expected seed token, and the decode oracle.
///
/// UNGUARDED on purpose. Every caller already refuses a too-short oracle with its own message
/// (`benchmark_timing_params`, `official::official_timed_params` /
/// `official::official_warmup_params`, `measure_job::timing_params`), and the six positional
/// arguments — three `Vec<u32>`-shaped clones and three tokens — are exactly the kind of thing
/// that transposes silently when written out four times.
pub(crate) fn benchmark_window_params(
    benchmark: &BenchmarkGolden,
    decode_steps: usize,
) -> TimingParams {
    TimingParams::new(
        benchmark.prefill_prompt_tokens.clone(),
        benchmark.expected_prefill_token,
        benchmark.decode_seed_tokens.clone(),
        benchmark.expected_decode_seed_token,
        benchmark.expected_decode_tokens.clone(),
        decode_steps,
    )
}

/// Time `cases[0]`, EXACTLY as Swift `--local-iterate` (`runLocalIterateCheckedTimingWithWorker`):
/// prefill `cases[0].prompt_tokens` → `expected[0]`; decode SEEDS with the same prompt, then
/// teacher-forces expected_tokens[1..=decode_steps]. Returns `Err(message)` for the Swift guard
/// `expected_tokens.count > decode_steps` (a full window is required to time) — that case genuinely
/// cannot be timed and stays a hard fail. This is the RULING-1 fallback for a benchmark-LESS golden.
fn cases0_timing_params(
    golden: &GoldenFixture,
    decode_steps: usize,
) -> std::result::Result<TimingParams, String> {
    let case = &golden.cases[0]; // the loader guarantees a non-empty cases[]
    if case.expected_tokens.len() <= decode_steps {
        return Err(format!(
            "primary case {} has {} expected_tokens; local-iterate timing needs more than {decode_steps}",
            case.name,
            case.expected_tokens.len()
        ));
    }
    let seed_token = case.expected_tokens[0];
    let decode_tokens = case.expected_tokens[1..=decode_steps].to_vec();
    Ok(TimingParams::new(
        case.prompt_tokens.clone(),
        seed_token,
        case.prompt_tokens.clone(),
        seed_token,
        decode_tokens,
        decode_steps,
    ))
}

/// The shared timing→metrics body: writes the measured decode/prefill seconds-per-token, the
/// external Qwen baselines, the derived speedups, and the speedup-floor flags into `metrics`.
///
/// Factored out of `local_iterate_score` so BOTH the correctness-PASS score builder AND the
/// correctness-FAILURE builder (`failed_with_real_timing_payload`, ITEM 1) populate these
/// fields identically — the whole point of David's ruling is that a correctness failure must
/// carry the SAME real timing surface as a pass, not blanked zeros. Does NOT touch
/// `case_count`/`checked_steps` (those differ pass vs. fail) or `passed_correctness`/`error`.
pub(crate) fn apply_timing_metrics(
    metrics: &mut ScoreMetrics,
    timing: &TimingResult,
    scoring: ScoringInputs,
) {
    let (baseline_prefill_spt, baseline_decode_spt) = scoring.baselines();
    metrics.peak_ram_gb = finite_nonneg(timing.peak_ram_gb);
    metrics.bandwidth_gb_per_token = 0.0;
    metrics.bandwidth_source = BANDWIDTH_SOURCE.to_string();
    metrics.decode_seconds_per_token = finite_nonneg(timing.decode_seconds_per_token);
    metrics.prefill_seconds_per_token = finite_nonneg(timing.prefill_seconds_per_token);
    metrics.baseline_decode_seconds_per_token = finite_nonneg(baseline_decode_spt);
    metrics.baseline_prefill_seconds_per_token = finite_nonneg(baseline_prefill_spt);
    let decode_speedup = speedup(baseline_decode_spt, timing.decode_seconds_per_token);
    let prefill_speedup = speedup(baseline_prefill_spt, timing.prefill_seconds_per_token);
    metrics.decode_speedup = decode_speedup;
    metrics.prefill_speedup = prefill_speedup;
    // THE FLOORS THIS RUN WAS GIVEN, never a constant: `base_metrics` sealed the same
    // `scoring.floors` into `metrics.{decode,prefill}_speedup_floor`, so the flag and the floor it
    // was decided against can never disagree (David 2026-09-09).
    metrics.passed_decode_speedup_floor = decode_speedup >= scoring.floors.decode;
    metrics.passed_prefill_speedup_floor = prefill_speedup >= scoring.floors.prefill;
    let timed = timing.prefill_elapsed_seconds + timing.decode_elapsed_seconds;
    metrics.timed_benchmark_seconds = finite_nonneg(timed);
    metrics.benchmark_wall_seconds = finite_nonneg(timed);
}

/// A FAILED payload that RETAINS the real timing surface (ITEM 1 / David's ruling): a
/// correctness FAILURE on the local-iterate path still carries the measured decode/prefill
/// seconds-per-token, the real baselines, the derived speedups, and the real speedup-floor
/// flags (via `apply_timing_metrics`) — NOT blanked to 0/false as `failed_payload` does —
/// while still reporting `passed = false`, `passed_correctness = false`, and the correctness
/// `error` + `first_failing_*` fields. `score` stays `null` (a failed run has no score).
fn failed_with_real_timing_payload(
    mode: Mode,
    failure: FailureReport,
    golden: &GoldenFixture,
    digests: RunDigests<'_>,
    timing: &TimingResult,
    scoring: ScoringInputs,
    window: &WindowShape,
) -> ScorePayload {
    // Swift's LOCAL-ITERATE FAILURE report (QwenRuntimeLocalIterate.swift:636-644), NOT the
    // standalone-correctness counts: `caseCount = timingRepeats`, and
    // `checkedSteps = failureStep + 1` (else the full checked-timing count). benchd's
    // `first_failing_step` IS Swift's `failureStep`, so checked_steps = first_failing_step + 1.
    // Both are Det-gated, so this is what closes the local-iterate failure DET surface (#66/Fable
    // + item-1 red-team Finding 3): before this, benchd reported golden-total and diverged.
    //
    // `passed_correctness` is false by construction: this builder is only ever reached with a
    // `FailureReport::correctness`, which cannot claim otherwise.
    let audit = FailedAudit::Counts {
        case_count: TIMING_REPEATS,
        checked_steps: failure
            .step
            .map(|s| s + 1)
            .unwrap_or((mode.decode_steps(window) as i64 + 2) * TIMING_REPEATS),
    };
    failed_score(
        failed_metrics(mode, golden, digests, scoring, Some(timing)),
        failure,
        audit,
        None,
    )
}

/// The `metrics.error` [`local_iterate_score`] sets when the ESTIMATED SCORE could not be
/// formed — the timing pair, the baseline pair, or the estimate itself was not finite and
/// positive. Byte-unchanged from the inline literal it replaces.
///
/// It is a named constant because `--capture-baseline` has to tell this text apart from every
/// other `metrics.error`: the capture mode runs with inert `(0.0, 0.0)` baselines by
/// construction, so a HEALTHY capture pass always ends here, while a real TIMED-PHASE failure
/// returns earlier with its own cause. That makes this the one error text
/// [`crate::capture::refuse_unless_timed_phase_succeeded`] may admit.
pub const INVALID_LOCAL_SCORE_ERROR: &str =
    "local estimated score is invalid: timing metrics and external Qwen baselines must be finite and positive";

/// Port of Swift `localIterateScore`: publish the estimated
/// decode_speedup^0.75 * prefill_speedup^0.25 score. Returns a failed payload
/// (score `null`) if any timing/baseline/score value is non-finite or non-positive.
pub fn local_iterate_score(
    mode: Mode,
    timing: &TimingResult,
    scoring: ScoringInputs,
    golden: &GoldenFixture,
    digests: RunDigests<'_>,
    window: &WindowShape,
) -> ScorePayload {
    let (baseline_prefill_spt, baseline_decode_spt) = scoring.baselines();
    let decode_spt = timing.decode_seconds_per_token;
    let prefill_spt = timing.prefill_seconds_per_token;
    // THE RUN'S RESOLVED WEIGHTS, not the constants: a fixture that declares its own pair is
    // scored under it (STEP 1, David 2026-09-15), and a run with no fixture keeps the 0.75/0.25
    // default it always had.
    let est = score_weighted(
        decode_spt,
        prefill_spt,
        baseline_decode_spt,
        baseline_prefill_spt,
        scoring.weights,
    );

    let valid = decode_spt.is_finite()
        && decode_spt > 0.0
        && prefill_spt.is_finite()
        && prefill_spt > 0.0
        && baseline_decode_spt.is_finite()
        && baseline_decode_spt > 0.0
        && baseline_prefill_spt.is_finite()
        && baseline_prefill_spt > 0.0
        && est.is_finite()
        && est > 0.0;

    let mut metrics = base_metrics(mode, golden, digests, scoring);
    // Shared timing→metrics body (decode/prefill spt, baselines, speedups, floor flags) —
    // identical to the correctness-FAILURE builder, so a fail retains the same real surface.
    apply_timing_metrics(&mut metrics, timing, scoring);
    // The board record + speculative-decode seal for the ONE prompt this leg timed. Previously
    // sealed ONLY on the official path, which left every local-iterate/local-submit score with an
    // empty `per_prompt` and no statement of what speculated — the local modes measure a timed
    // prompt exactly as official does, so they record it identically.
    seal_timing_surface_facts(&mut metrics, golden, timing);
    // Report the CHECKED-TIMING counts, not the conformance case counts. Swift
    // `runLocalIterateCheckedTimingWithWorker` reports `caseCount = timingRepeats` and
    // `checkedSteps = (decodeSteps + 2) * timingRepeats` (the +2 is the measured prefill
    // token plus the decode seed forward). parity-diff.py lists both in the deterministic
    // must-match set, so anything else fails parity against a correct engine.
    metrics.case_count = TIMING_REPEATS;
    metrics.checked_steps = (mode.decode_steps(window) as i64 + 2) * TIMING_REPEATS;

    if !valid {
        metrics.passed_correctness = true;
        metrics.error = INVALID_LOCAL_SCORE_ERROR.to_string();
        return ScorePayload {
            score: None,
            passed: false,
            metrics,
        };
    }

    metrics.passed_correctness = true;
    ScorePayload {
        score: Some(est),
        passed: true,
        metrics,
    }
}

/// The `metrics.baseline_source` an UNSCORED local run seals: it says, in one string, that this
/// run had no denominator BY DESIGN and therefore has no score to argue with.
pub const BASELINE_SOURCE_LOCAL_UNSCORED: &str = "none (local mode: unscored)";

/// Turn a local run that had NO baseline into an UNSCORED one.
///
/// WHY THIS EXISTS. The Qwen 3.8 125B-A6B tracks store no baseline pair: a ranked run measures its
/// own on the box, against the organizer-staged reference tree. A participant iterating on a
/// laptop has no reference tree, and David requires the local benchmark to keep working. So the
/// local modes on those tracks run the CANDIDATE LEG ONLY: real timings, the real correctness gate,
/// and NO score — because a score with no denominator is not a score.
///
/// The run reaches here through `local_iterate_score` with `(0.0, 0.0)` baselines, which is
/// already the "no pair" shape: it seals the real timing surface and the real correctness verdict,
/// then reports `score = null` with [`INVALID_LOCAL_SCORE_ERROR`] because the estimate is not
/// finite and positive. That text is right for a run that EXPECTED a pair and got a bad one, and
/// wrong for a run that never had one. This converts exactly that case, and nothing else:
///
/// * `metrics.baseline_source` states the design (`"none (local mode: unscored)"`) and `score`
///   stays `null` — ALWAYS, so an unscored run can never be read as a scored one;
/// * the [`INVALID_LOCAL_SCORE_ERROR`] text is cleared and `passed` becomes `true` ONLY when the
///   correctness gate passed. A REAL failure — a correctness miss, an oracle mismatch, a protocol
///   fault — keeps its own error and its own `passed = false`, because none of those is about the
///   missing baseline.
///
/// The speedup and floor fields keep the values the zero pair produced. With
/// `baseline_* = 0`, `baseline_source = "none (local mode: unscored)"` and `score = null`, a
/// reader has one consistent story: there was no baseline, so there is no speedup and no score.
pub fn seal_local_unscored(payload: &mut ScorePayload) {
    payload.score = None;
    payload.metrics.baseline_source = Some(BASELINE_SOURCE_LOCAL_UNSCORED.to_string());
    if payload.metrics.passed_correctness && payload.metrics.error == INVALID_LOCAL_SCORE_ERROR {
        payload.metrics.error.clear();
        payload.passed = true;
    }
}

/// Why a run failed, in the shape the score's metrics report it.
///
/// #65: these five values used to be threaded through every failure path as a positional
/// `(fc, fs, exp, act, err)` tuple and then as five more parameters on `failed_payload`,
/// where nothing but argument order said which `Option<i64>` was the expected token and
/// which was the actual. Naming them removes that whole class of transposition.
///
/// `passed_correctness` rides along because it is part of the same story: whether the
/// correctness gate had already passed when this failure happened. The two constructors
/// encode the invariant every call site already obeyed — a failure carrying a case identity
/// IS a correctness failure, so it can never claim `passed_correctness = true`.
#[derive(Debug, Clone, Default)]
pub(crate) struct FailureReport {
    /// `metrics.error` — the human-readable refusal, byte-matched to Swift where it must be.
    pub error: String,
    /// Whether the correctness gate itself passed before this (later) failure.
    pub passed_correctness: bool,
    /// `metrics.first_failing_case`.
    pub case: Option<String>,
    /// `metrics.first_failing_step`, in Swift's checked-step numbering.
    pub step: Option<i64>,
    /// `metrics.expected_token` — what the golden said.
    pub expected_token: Option<i64>,
    /// `metrics.actual_token` — what the engine emitted.
    pub actual_token: Option<i64>,
}

impl FailureReport {
    /// A failure with a message only and no per-case identity: a protocol/transport fault, a
    /// preflight refusal, or a "cannot time this golden" guard. `passed_correctness` says
    /// whether the correctness gate had already passed when it happened.
    pub(crate) fn message(error: impl Into<String>, passed_correctness: bool) -> Self {
        FailureReport {
            error: error.into(),
            passed_correctness,
            ..Default::default()
        }
    }

    /// A CORRECTNESS failure: the failing case, the checked step, and the diverging token
    /// pair. `passed_correctness` is necessarily false — this IS the gate failing.
    pub(crate) fn correctness(
        error: impl Into<String>,
        case: Option<String>,
        step: Option<i64>,
        expected_token: Option<i64>,
        actual_token: Option<i64>,
    ) -> Self {
        FailureReport {
            error: error.into(),
            passed_correctness: false,
            case,
            step,
            expected_token,
            actual_token,
        }
    }
}

/// Swift's §F2 missing-paired-baselines refusal, minus the leading mode name.
///
/// Port of `QwenRuntime.requiredGoldenBenchmarkBaselines(_:context:)`, where the message is
/// built as three concatenated string literals and `context` is interpolated at the FRONT.
/// #62: this used to be an inline byte-literal in `execute_iterate` that only a
/// `starts_with` assertion guarded, so drift in the TAIL could ship. It is now single-sourced
/// here and pinned in full against a fixture captured from the Swift source
/// (`crates/benchd/tests/fixtures/swift-missing-paired-baselines.json`).
const MISSING_PAIRED_BASELINES_SUFFIX: &str =
    " requires external Qwen benchmark baselines for both prefill and decode; \
     Gemma baseline constants are not valid for Qwen";

/// The full Swift refusal for `mode` — `"<mode-name><suffix>"`, byte-for-byte what Swift
/// throws when a golden carries neither paired baseline and no override supplies them.
pub fn missing_paired_baselines_error(mode: Mode) -> String {
    format!("{}{MISSING_PAIRED_BASELINES_SUFFIX}", mode.mode_name())
}

/// The baseline pair a LOCAL checked-timing run (`local-iterate` / `local-submit`) scores
/// against: the compile-time official-runner constants, never the golden.
///
/// **#127 — RULED (David 2026-08-20, "MIRROR REFERENCE"):** the reference's `localIterate` reads
/// `MLXFastConstants.officialBaseline{Prefill,Decode}SecondsPerToken` directly
/// (`QwenRuntimeLocalIterate.swift@b26f76f:34,36`, used at `:291,317,382,386`) and never consults
/// the golden's `benchmark.baseline_*_seconds_per_token` — it does not even reach the
/// `resolvedBaseline*` accessor its own official path uses. benchd used to source the pair from
/// the golden, so a golden declaring any pair silently rescored benchd while leaving the reference
/// untouched (the §8 window split ~12× on score off a ~5% timing split, because the golden carried
/// the retired fork's Gemma-era constants). Under the ruling the golden's pair is INERT LEGACY
/// DATA on this leg: not required, not validated against these values, simply ignored.
///
/// Both local modes share this because the reference routes both through that one function
/// (`main.swift@b26f76f:315-340`); only the OFFICIAL path stays golden-authoritative.
pub fn local_mode_baselines(
    contract: &bench_core::contract::Contract,
    track_id: &str,
) -> Result<bench_core::constants::OfficialBaseline, String> {
    let baseline = bench_core::contract::official_baseline(contract, track_id)?.value;
    // B2 — the SCORING-TIME REGIME FENCE: a track must also declare WHAT it scores
    // (`docs/scored-regime-and-prefill-window.md`) before a local leg scores against this pair. It
    // runs after the baseline refusal so a track missing both facts refuses with the more specific
    // message first, and before the baseline is returned so no caller can score without it.
    // CONTRACT-FIRST (STEP 1, David 2026-09-15): the fixture's declared regime when it makes one,
    // the per-track table when it does not. The fence itself is unchanged — what moves is only
    // WHERE the regime comes from, and the run seals which source answered
    // (`metrics.contract_sources.scored_regime`).
    bench_core::contract::scored_regime(contract, track_id)?;
    Ok(baseline)
}

/// The reference's EARLY-REFUSE failure record: a refusal that happens BEFORE anything ran
/// (`score = null`, `passed = false`, `passed_correctness = false`) — e.g. §F2's missing
/// required baselines.
///
/// **#74 — RULED (David 2026-08-20, "MIRROR SWIFT EXACTLY"):** when the run refuses before it
/// runs, benchd seals the reference's record and nothing more. Swift derives all three
/// run-shaped fields from the `CorrectnessReport`, which on this path is `nil`, so they seal
/// EMPTY/ZERO — `goldenHash: correctness?.goldenHash ?? ""`, `checkedSteps:
/// correctness?.checkedSteps ?? 0`, `caseCount: correctness?.caseCount ?? 0`
/// (`QwenRuntimeBenchmark.swift@b26f76f:1161-1162,1176`, reached from
/// `QwenRuntimeLocalIterate.swift@b26f76f:40-74,197-198`). benchd used to seal the golden's
/// digest and the golden's TOTAL case count here, so a refusal that never ran anything still
/// read like a run that had (`golden_hash` present, `case_count`/`checked_steps` = 4 where the
/// reference sealed `""`/0/0) — the fail-POINT divergence #74 records.
///
/// **Who actually reaches this, and why the baseline pair is NOT zeroed.** Since the #127 ruling
/// the local legs take their baselines from the constants and never refuse for want of them, so the
/// only live caller is the OFFICIAL arm — and the official path is where the justification has to
/// come from. It holds there directly: this refusal fires exactly where the reference's official
/// path would have fallen back to the constants, because `resolvedBaseline*` is
/// `baseline* ?? officialBaseline*` (`Golden.swift@b26f76f:220-226`, consumed at
/// `QwenRuntimeBenchmark.swift@b26f76f:155-157,443-445`). benchd refuses instead of falling back
/// (deliberate ranked-path strictness — see `resolve_paired_baselines`), but the DENOMINATOR the
/// reference would have used at that same point is the constant pair, which is what the record
/// seals. The reference's own official refusal agrees independently: its
/// `baseline*SecondsPerToken` locals are initialised to `MLXFastConstants.officialBaseline*`
/// (`QwenRuntimeBenchmark.swift@b26f76f:33-34,350-351`) and any failure before the timed phase
/// seals them unchanged.
///
/// The LOCAL path reaches the same values by its own route — `failedScore`'s baseline parameters
/// DEFAULT to those constants (`QwenRuntimeBenchmark.swift@b26f76f:1130-1131`) and the local
/// refusal site passes no override (`QwenRuntimeLocalIterate.swift@b26f76f:49-68` names every
/// argument it forwards; the baselines are not among them). Both paths agree, which is why one
/// record shape serves both. The pre-#74 comment here asserted the opposite ("Swift emits
/// `baseline_* = 0`"), which was true of the RETIRED `mlxfast-challenge-dev` fork — it held
/// mutable `baseline*SecondsPerToken` locals initialised to `0.0` and forwarded them explicitly —
/// and is false of the reference this repo now mirrors.
///
/// NOTE that this therefore CHANGES THE OFFICIAL REFUSAL ARTIFACT too: an official run that
/// refuses for want of a baseline pair used to seal the golden's digest and case count and a zero
/// baseline pair, and now seals the reference's empty record with the constant pair. That is
/// #74's ruling applied where its only live caller is, not a widening of it — but "the official
/// path is untouched" is a statement about #127's baseline SOURCING, not about this artifact.
///
/// The whole record is pinned byte-for-byte against a constructed reference capture by
/// `tests::early_refuse_record_byte_matches_the_reference_capture`.
pub fn preflight_failed_payload(
    mode: Mode,
    golden: &GoldenFixture,
    digests: RunDigests<'_>,
    error: String,
    scoring: ScoringInputs,
) -> ScorePayload {
    // The blank seal needs no local override: `failed_payload` blanks `golden_hash`/counts on
    // EVERY local failure path (#132(b), FINAL), and `base_metrics` seals the caller's resolved
    // official baseline pair on every one (#132(a)) — the pair the caller obtained from
    // `bench_core::constants::official_baseline()`, which refuses by name while pending.
    let mut payload = failed_payload(
        mode,
        FailureReport::message(error, false),
        golden,
        digests,
        scoring,
    );
    if mode.is_local_checked_timing() {
        payload.metrics.local_phases = Some(LocalPhases::default());
    }
    payload
}

/// What a FAILED payload seals into the CORRECTNESS-DERIVED audit surface — `golden_hash`,
/// `case_count` and `checked_steps`, the three fields the reference derives from a
/// `CorrectnessReport` that may or may not exist at the point of failure.
///
/// The distinction is the reference's, not benchd's: `goldenHash` carries an INVARIANT — non-empty
/// means correctness completed — because it is only ever populated from a report that exists
/// (`QwenRuntimeBenchmark.swift@b26f76f:1161-1162,1176` read `?? ""` / `?? 0` off a nil report).
/// Sealing a real digest where correctness did NOT complete is the same field meaning something
/// weaker, which every downstream consumer silently inherits.
pub(crate) enum FailedAudit {
    /// The reference's `correctness == nil` defaults: `golden_hash = ""`, both counts 0. Every
    /// failure the reference reaches before its correctness report exists seals this.
    Blank,
    /// Correctness ran — or the reference would hold a report at the equivalent point — so the
    /// golden's digest stands and these are the counts to seal. They differ per path (the
    /// official paths seal the golden roster; the local checked-timing paths seal Swift's
    /// `timingRepeats` / `failureStep + 1`), which is why they are values and not a rule.
    Counts { case_count: i64, checked_steps: i64 },
}

/// The metrics a FAILED payload starts from: `base_metrics` for the run, plus — when `timing` is
/// `Some` — the real measured surface RETAINED on top of it (decode/prefill spt, the resolved
/// baselines, the derived speedups, the floor flags, the per-prompt board record and the spec
/// seal).
///
/// `Some` is David's ITEM-1 ruling: a failure on a leg that DID measure keeps its real numbers.
/// `None` leaves `base_metrics`' defaults, for a failure before or instead of the timed phase.
pub(crate) fn failed_metrics(
    mode: Mode,
    golden: &GoldenFixture,
    digests: RunDigests<'_>,
    scoring: ScoringInputs,
    timing: Option<&TimingResult>,
) -> ScoreMetrics {
    let mut metrics = base_metrics(mode, golden, digests, scoring);
    if let Some(timing) = timing {
        apply_timing_metrics(&mut metrics, timing, scoring);
        seal_timing_surface_facts(&mut metrics, golden, timing);
    }
    metrics
}

/// THE failed payload: `score = null`, `passed = false`, and the failure's audit surface.
///
/// One body for every failure sink in benchd. The ten builders that used to each write this out
/// differ in exactly three policy bits: whether the timed surface is retained (that one belongs to
/// [`failed_metrics`], which builds `metrics`), `audit` — blank-vs-retain for the
/// correctness-derived fields — and `commit`, `Some` on the OFFICIAL paths, which stamp
/// `metrics.commit`, and `None` on the local ones, which do not.
///
/// The SEAL BOUNDARY (#134) lives here and nowhere else: `failure.error` is usually a `RunnerError`
/// Display, scrubbed and capped at the one point where it stops being a log line and becomes a
/// persisted artifact. That Display already carries engine-controlled text (worker stderr tail,
/// the engine's `error` reasons) only as `engine-text-sha256=… engine-text-bytes=…`, so a
/// participant cannot read a hidden prompt back through a deliberate failure; the text itself goes
/// to benchd's stderr (`RunnerError::diagnostic`).
/// Official is the most exposed sink — its score.json travels and worker stderr is never forwarded
/// — so this scrub is the only thing between engine bytes and the artifact.
pub(crate) fn failed_score(
    mut metrics: ScoreMetrics,
    failure: FailureReport,
    audit: FailedAudit,
    commit: Option<&str>,
) -> ScorePayload {
    metrics.passed_correctness = failure.passed_correctness;
    if let Some(commit) = commit {
        metrics.commit = commit.to_string();
    }
    metrics.error = scrub_reason_for_seal(&failure.error);
    match audit {
        FailedAudit::Blank => {
            metrics.golden_hash = String::new();
            metrics.case_count = 0;
            metrics.checked_steps = 0;
        }
        FailedAudit::Counts {
            case_count,
            checked_steps,
        } => {
            metrics.case_count = case_count;
            metrics.checked_steps = checked_steps;
        }
    }
    metrics.first_failing_case = failure.case;
    metrics.first_failing_step = failure.step;
    metrics.expected_token = failure.expected_token;
    metrics.actual_token = failure.actual_token;
    ScorePayload {
        score: None,
        passed: false,
        metrics,
    }
}

/// A failed payload (`score = null`, `passed = false`) for a local failure on which the
/// REFERENCE's fused checked-timing pass did not complete — the blanked local failure surface.
///
/// The criterion is the REFERENCE's, not benchd's, and the difference is not cosmetic: at `:455`
/// and `:581` benchd's OWN conformance report exists (its gate ran separately and passed), so
/// "produced no correctness report" would be false of benchd at those two sites while the blank
/// is still correct. What decides is whether the reference — which derives correctness solely
/// from `runLocalIterateCheckedTiming*` — would have a `CorrectnessReport` at the equivalent
/// point. It does not: `correctnessReport` is assigned only at
/// `QwenRuntimeLocalIterate.swift@b26f76f:152`, after that pass returns.
///
/// **RULED (David 2026-08-20, interview, FINAL) — #132(b): MIRROR BLANK STRICTLY.** Every local
/// failure path that reaches here seals `golden_hash = ""` and zero counts, byte-matching the
/// reference. Zero DECLARED cells on this surface.
///
/// This REVERSES an earlier same-day ruling (keep-real-values + declare) once the field semantics
/// were laid out: the reference's `goldenHash` carries an INVARIANT — non-empty means correctness
/// completed — because it is only ever populated from a `CorrectnessReport` that exists
/// (`QwenRuntimeBenchmark.swift@b26f76f:1161-1162,1176` read `?? ""` / `?? 0` off a nil report).
/// Sealing benchd's real digest on a path where correctness did NOT complete is not extra
/// information, it is the same field meaning something weaker, and every downstream consumer that
/// trusts the invariant silently inherits the weakening. benchd's real data lives in LOGS.
/// A benchd-only provenance superset field (`loaded_golden_sha256` or similar) was explicitly
/// considered and REJECTED: failure records carry the reference's fields, and nothing else.
///
/// `case_count` is deliberately NOT a parameter. It was one, and every caller passed the golden's
/// roster into a record that describes no run; making the zero structural is what stops that
/// coming back. The paths that DO have a report build their payloads elsewhere
/// ([`failed_with_real_timing_payload`], [`local_iterate_score`]) and keep their real values —
/// the reference keeps its there too, because on that arm its `correctnessReport` IS set
/// (`QwenRuntimeLocalIterate.swift@b26f76f:152`, checked at `:164-174`).
fn failed_payload(
    mode: Mode,
    failure: FailureReport,
    golden: &GoldenFixture,
    digests: RunDigests<'_>,
    scoring: ScoringInputs,
) -> ScorePayload {
    // The blank seal. No caller may opt out — see the doc comment above.
    failed_score(
        failed_metrics(mode, golden, digests, scoring, None),
        failure,
        FailedAudit::Blank,
        None,
    )
}

/// Baseline metrics with the constant / default fields filled (Swift ScoreMetrics
/// defaults). Callers overwrite the fields their phase measured.
///
/// **The baseline pair is the CALLER's resolved pair, not zero (#132(a)).** `baselines` is
/// `(prefill, decode)` as the caller resolved it for the run's track through
/// `bench_core::constants::official_baseline`, which refuses by name for a track whose pair is
/// not captured — so no record here can carry another track's numbers. This is
/// the same fact #74 ruled on, applied where it actually reaches: the reference has NO failure
/// path that emits `baseline_* = 0`. Every local failure — the conformance `Err`, the
/// `close_phase` `Err`, the time-only pass failing after a correctness failure, the
/// window-too-short guard, the timed pass failing — returns through the ONE `failed()` closure at
/// `QwenRuntimeLocalIterate.swift@b26f76f:40-74`, which forwards error/correctness/digests and
/// names no baseline argument, so `failedScore`'s defaults stand
/// (`QwenRuntimeBenchmark.swift@b26f76f:1130-1131`). The official side agrees independently: its
/// `baseline*SecondsPerToken` locals are INITIALISED to those constants
/// (`QwenRuntimeBenchmark.swift@b26f76f:33-34,350-351`), so a failure before the timed phase seals
/// them unchanged.
///
/// Setting them HERE rather than at each exit is deliberate: #74's fix landed only on
/// `preflight_failed_payload`, which left five locally-reachable sites still zeroing the pair
/// (#132) — a per-site override is exactly the shape that let them drift apart. Callers whose
/// phase actually resolved a pair overwrite these via `apply_timing_metrics`; callers that never
/// got that far now inherit the reference's values instead of a zero the reference never emits.
///
/// **The OFFICIAL path does not take this default (#132/F-2).** There the reference overwrites its
/// baseline locals with `pairedBaseline ?? benchmarkGolden.resolvedBaseline*`
/// (`QwenRuntimeBenchmark.swift@b26f76f:442-445`) before the gates-only branch and before every
/// failure record, so those payloads carry the RESOLVED pair, not these constants. The three
/// official sites nothing else overwrites set it explicitly — see
/// `official::official_resolved_baselines`. This default is the LOCAL surface's answer.
///
/// NOTE this does NOT touch `golden_hash` / `case_count` / `checked_steps`, and must not.
/// **RULED (David 2026-08-20, interview — FINAL) — #132(b): MIRROR BLANK STRICTLY.** Every local
/// failure path where the REFERENCE's fused checked-timing pass did not complete seals
/// `golden_hash = ""` and zero counts, byte-matching it; zero DECLARED cells on that surface. This
/// reverses an earlier same-day ruling (keep-real-values + declare) — the deciding point is that
/// the reference's `goldenHash` carries an INVARIANT, *non-empty means correctness completed*,
/// because it is only ever populated from a `CorrectnessReport` that exists
/// (`QwenRuntimeBenchmark.swift@b26f76f:1161-1162,1176` read `?? ""` / `?? 0` off a nil report).
/// Sealing benchd's real digest where correctness did not complete makes that field mean something
/// weaker for every consumer that trusts it. benchd's real data lives in LOGS.
///
/// The blank is applied by [`failed_payload`], not here, and that placement is load-bearing:
/// `base_metrics` also feeds the #73 RETAIN-TIMING arm and the PASSING path, where the reference
/// DOES seal real values because its `correctnessReport` is set
/// (`QwenRuntimeLocalIterate.swift@b26f76f:152`, checked at `:164-174`). Blanking here would take
/// both with it. Pinned by `assert_ruled_blank_seal` (per-site) and by
/// `correctness_failure_retains_what_early_refuse_seals_empty` (the arm that must NOT blank).
///
/// **F1 — `harness_hash` is SEALED HERE, on every path.** This is the one funnel every `iterate`
/// payload passes through — local-iterate, local-submit, official, official gates-only, and every
/// failure builder — so sealing the resolved [`HarnessIdentity`] here covers all modes with one
/// assignment. It replaces a `String::new()` stub that made `metrics.harness_hash` empty on every
/// benchd-authored score, which the seam-3 overlay correctly refuses to publish. The identity is
/// resolved before the run starts (see `HarnessIdentity::resolve_from_current_dir` and its call
/// site in `execute_iterate`) over the roster roots that exist — a root absent on this engine
/// surface is logged and skipped — so by the time control reaches here a real digest exists and
/// cannot be `""`.
pub(crate) fn base_metrics(
    mode: Mode,
    golden: &GoldenFixture,
    digests: RunDigests<'_>,
    scoring: ScoringInputs,
) -> ScoreMetrics {
    let (baseline_prefill_spt, baseline_decode_spt) = scoring.baselines();
    // ONLY THE TWELVE REAL VALUES ARE NAMED. Every other field of `ScoreMetrics` was spelled out
    // here at its own `Default` — 80 lines of hand-maintained copy of the derive, which a field
    // added to the struct diverges from silently. `..Default::default()` is that same set of
    // values, stated once, and what this builder DECIDES is now the whole of what it writes.
    //
    // What the defaults mean, and why none of them is a decision this builder gets to make:
    //
    // * `contract_sha256` / `contract_sources` — THE MIGRATION PIN and its per-group SOURCES, both
    //   stamped by `execute_iterate` from its ONE contract load, so every payload arm carries the
    //   same digest. Absent here: this builder never reads the fixture.
    // * The board record (`per_prompt`), the speculative-decode seal (`effective_spec_*`, `spec_*`,
    //   `acceptance_lengths`) and the engine identity — EMPTY / ABSENT on every mode. They are
    //   written only by `seal_timing_surface_facts` after a run's timing is applied, and the engine
    //   identity only by `main` from the timed worker's hello. Each is omitted from the sealed JSON
    //   while unset, so a payload that never reaches those seals is byte-unchanged.
    // * The PAIRED-BASELINE seal (`baseline_*`, `candidate_leg_*`, `paired_legs`) — written by
    //   `official::seal_paired_baseline` on the ranked paired path only; every other path seals no
    //   key there.
    ScoreMetrics {
        // SEALED = ENFORCED (David 2026-09-09): the baselines and the floors this run resolved —
        // from the `--contract` track fixture on a scored run — are the ones every gate on this
        // payload used. Nothing here reads a constant.
        baseline_decode_seconds_per_token: baseline_decode_spt,
        baseline_prefill_seconds_per_token: baseline_prefill_spt,
        decode_speedup_floor: scoring.floors.decode,
        prefill_speedup_floor: scoring.floors.prefill,
        num_layers: digests.model.num_hidden_layers,
        golden_hash: golden.sha256.clone(),
        timestamp: iso8601_now(),
        harness_hash: digests.harness.as_str().to_string(),
        weights_hash: digests.weights.sha256.clone(),
        weights_byte_count: digests.weights.byte_count,
        weights_file_count: digests.weights.file_count,
        runtime: mode.runtime().to_string(),
        ..Default::default()
    }
}

/// Seal every ADDITIVE fact that belongs to the TIMED SURFACE of a single-leg run: the board-facing
/// `metrics.per_prompt` record, the speculative-decode seal (`effective_spec_*`, `spec_*`,
/// `acceptance_lengths`), and nothing else.
///
/// ONE HELPER, EVERY MODE. Official (`official::seal_official_per_prompt`) and the local
/// iterate/submit score builders call this, so the two families cannot drift about what a timed
/// prompt records. It is invoked from exactly the payloads that RETAIN the real timing surface — the
/// passing score and the failed-with-real-timing builders — never from a blanked/gates-only payload,
/// which measured no prompt and so seals nothing here.
///
/// WHAT IS SEALED, AND WHY EACH.
///
/// * `per_prompt` — one entry for the ONE prompt this leg timed, identified BY BYTES with the
///   golden's own sha256. Entries are never invented for prompts the run did not measure.
/// * `effective_spec_mode` / `effective_spec_depth` — WHAT SPECULATED. When the leg carried a spec,
///   this is the engine's own `effective_spec` ECHO, already validated equal to benchd's request
///   (spec-never-ignored). When it carried NONE, benchd requested the engine's default path, which
///   the protocol defines as serial — so the leg seals `serial` / depth `0` rather than leaving the
///   scored run with no statement of what ran at all.
/// * The COUNTERS (`spec_rounds`, `spec_drafted_total`, `spec_accepted_total`,
///   `spec_acceptance_rate`, `spec_verify_replay_disagreements`, `acceptance_lengths`) — only on a
///   leg that actually DRAFTED (`depth > 0`). On a serial leg they are structurally constant
///   (nothing drafted, one committed token per round, `acceptance_lengths` an all-ones vector of
///   length N), so sealing them would add N entries of pure noise to every serial score.
///   `spec_acceptance_rate` is further omitted when nothing was drafted: a zero denominator has no
///   rate, and `0.0` would read as "drafted plenty, accepted none".
///   `spec_verify_replay_disagreements` is further omitted when the ENGINE did not report the
///   counter at all — the wire field is optional, and `0` would claim a measurement the engine
///   never made.
///
/// NOTHING ENFORCED CHANGES. Every value here is AUDIT-ONLY: none feeds a seconds-per-token, a
/// speedup, a floor, a band or the score. `mtp_seconds_per_token_mean` is READ BACK from
/// `metrics.decode_seconds_per_token` (the enforced whole-window figure `apply_timing_metrics` just
/// wrote), so the two cannot drift and no second, decode-only quantity is introduced.
///
/// A result with NO free-run audit (the teacher-forced v1 decode path, which no scored run takes)
/// seals NOTHING.
pub(crate) fn seal_timing_surface_facts(
    metrics: &mut ScoreMetrics,
    golden: &GoldenFixture,
    timing: &TimingResult,
) {
    let Some(audit) = timing.free_run_audit.as_ref() else {
        return;
    };
    // The engine's echo when there is one; otherwise the serial default benchd asked for by
    // sending no spec at all.
    let (mode, depth) = match timing.effective_spec.as_ref() {
        Some(spec) => (spec.mode.clone(), spec_seal_depth(spec)),
        None => (bench_protocol::SPEC_MODE_SERIAL.to_string(), 0),
    };
    metrics.effective_spec_mode = Some(mode);
    metrics.effective_spec_depth = Some(depth);

    let drafted = depth > 0;
    if drafted {
        metrics.spec_rounds = Some(audit.rounds() as u64);
        metrics.spec_drafted_total = Some(audit.drafted_total());
        metrics.spec_accepted_total = Some(audit.accepted_total());
        metrics.spec_acceptance_rate = audit.acceptance_rate();
        // OPTIONAL on the wire, so OPTIONAL here: an engine that reported no disagreement counter
        // seals no key, and the sealed bytes of every pre-field engine's score are unchanged.
        metrics.spec_verify_replay_disagreements = audit.verify_replay_disagreements();
        // OPTIONAL on the wire, so OPTIONAL here: the verify path the engine reports (rectangular
        // window vs serial oracle) — the fact behind the decode number, never a scoring input.
        if let Some(v) = audit.verification() {
            metrics.spec_verification_mode = Some(v.mode.clone());
            metrics.spec_rectangular_verification_rounds = v.rectangular_rounds;
            metrics.spec_serial_verification_rounds = v.serial_rounds;
        }
        metrics.acceptance_lengths = audit.acceptance_lengths().to_vec();
    }

    metrics.per_prompt = vec![ScorePerPrompt {
        prompt_sha256: golden.sha256.clone(),
        effective_mean_draft_len: finite_nonneg(audit.effective_mean_draft_len()),
        mtp_seconds_per_token_mean: metrics.decode_seconds_per_token,
        spec_rounds: metrics.spec_rounds,
        spec_drafted_total: metrics.spec_drafted_total,
        spec_accepted_total: metrics.spec_accepted_total,
        // Mirrored from the run-level seal, which `main` fills from the TIMED worker's hello after
        // the payload is built; absent here until then.
        head_provenance_sha256: metrics.head_provenance_sha256.clone(),
    }];
}

/// The DEPTH a sealed `effective_spec` reports: `0` for `serial` (no drafter, so zero is its true
/// depth), else the module's own resolved depth — `mtp.depth`, or the `dflash` block's `depth`
/// lever. An `mtp` echo always carries a resolved depth (`spec_echo_honors_request` refuses one that
/// does not), so the `0` fallback is unreachable for a validated mtp echo and stays a fail-soft.
fn spec_seal_depth(spec: &bench_protocol::SpecConfig) -> i64 {
    if spec.mode == bench_protocol::SPEC_MODE_SERIAL {
        return 0;
    }
    spec.mtp
        .and_then(|m| m.depth)
        .map(i64::from)
        .or_else(|| spec.dflash_depth().map(i64::from))
        .unwrap_or(0)
}

pub(crate) fn finite_nonneg(v: f64) -> f64 {
    if v.is_finite() && v >= 0.0 {
        v
    } else {
        0.0
    }
}

/// First failing (case name, step) from a conformance report, in Swift's layered
/// evaluation order: primary teacher-forced cases first (first mismatch step), then
/// anchors (no step), then free-run (first mismatch step).
pub(crate) struct ConformanceFailure {
    pub(crate) case: String,
    pub(crate) step: Option<i64>,
    pub(crate) expected: Option<i64>,
    pub(crate) actual: Option<i64>,
    /// A primary/`cases[]` (teacher-forced) failure — the one local-iterate reports
    /// Swift-style (§F1). Anchor/free-run failures are the benchd superset.
    pub(crate) is_base_case: bool,
}

pub(crate) fn first_conformance_failure(report: &ConformanceReport) -> Option<ConformanceFailure> {
    if let Some(c) = report.base_cases.iter().find(|r| !r.passed) {
        return Some(ConformanceFailure {
            case: c.name.clone(),
            step: c.first_mismatch_step.map(|s| s as i64),
            expected: c.expected_token,
            actual: c.actual_token,
            is_base_case: true,
        });
    }
    if let Some(a) = report.anchors.iter().find(|r| !r.passed) {
        return Some(ConformanceFailure {
            case: a.name.clone(),
            step: None,
            expected: None,
            actual: None,
            is_base_case: false,
        });
    }
    if let Some(f) = report.free_run.iter().find(|r| !r.passed) {
        return Some(ConformanceFailure {
            case: f.name.clone(),
            step: f.first_mismatch_step.map(|s| s as i64),
            expected: None,
            actual: None,
            is_base_case: false,
        });
    }
    None
}

/// Map a benchd teacher-forced base-case mismatch index (the `expected_tokens` index) to
/// Swift's local-iterate checked-step numbering (`QwenRuntimeLocalIterate.swift:777-797`):
/// prefill produces `expected_tokens[0]` at step 0; decode step k produces
/// `expected_tokens[k+1]` at checked-step `k+2`. So index 0 → 0 (prefill), index i≥1 → i+1.
fn swift_local_iterate_checked_step(expected_index: usize) -> i64 {
    if expected_index == 0 {
        0
    } else {
        expected_index as i64 + 1
    }
}

// --- timestamp -------------------------------------------------------------

/// UTC ISO-8601 `yyyy-MM-ddTHH:mm:ssZ` for "now" (matches Swift `ISO8601DateFormatter`).
pub(crate) fn iso8601_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format_iso8601(secs)
}

fn format_iso8601(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let (y, mo, d) = civil_from_days(days);
    format!("{y:04}-{mo:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Howard Hinnant's `civil_from_days` (epoch 1970-01-01 == day 0).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m as u32, d)
}

/// THE WINDOW THE TEST-ONLY WRAPPERS AND THE UNIT TESTS MEASURE UNDER, as TEST DATA.
///
/// It is the window the 125B MLX track declares, and it is what every one of these tests read off
/// the module constants before the window became a resolved value. STEP 3 (David 2026-09-15)
/// deleted `WindowShape::table`, so the four values are stated here rather than looked up; a test
/// that wants a DIFFERENT window builds its own and says so.
#[cfg(test)]
pub(crate) fn test_window() -> bench_core::constants::WindowShape {
    bench_core::constants::WindowShape {
        correctness_steps: 64,
        benchmark_decode_steps: 128,
        local_submit_benchmark_decode_steps: 1023,
        official_prefill_warmup_runs: 1,
        official_prefill_timed_runs: 1,
        timed_divergence_replay: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bench_core::constants::{
        BENCHMARK_PREFILL_PROMPT_TOKENS, CORRECTNESS_PROMPT_TOKENS, REQUIRED_GOLDEN_MODEL_TYPE,
    };
    // The golden-arity test calls the loader DIRECTLY, at two different consumer arities over
    // ONE byte string — the contract it pins is precisely that those two answers differ, which
    // a builder that owns the arity cannot express.
    use bench_core::golden::load_golden_fixture;
    use bench_core::score::evaluate_timed_run;
    use bench_runner::mock::MockEngine;
    use serde_json::json;

    use crate::testgolden::{
        benchmark_oracle, benchmark_oracle_windowed, corrupt_beyond_budget, TestGolden,
        TEST_BASELINE,
    };

    /// THE [`FlowParams`] THESE TESTS RUN UNDER: the shared `TEST_BASELINE` pair, the in-tree
    /// window, no spec, and the correctness gate ON — the historical
    /// defaults the deleted `iterate_core` / `iterate_flow` wrappers used to supply positionally.
    /// A test that varies one of the others says so in a struct update:
    /// `FlowParams { timed_only: true, ..test_flow(…) }`.
    fn test_flow<'a>(
        golden: &'a GoldenFixture,
        mode: Mode,
        strict: bool,
        digests: RunDigests<'a>,
        window: &'a WindowShape,
    ) -> FlowParams<'a> {
        FlowParams {
            golden,
            scoring: ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
            mode,
            strict,
            timed_only: false,
            digests,
            spec: None,
            window,
        }
    }

    /// Option A (a8/David ruling): the local-iterate / local-submit timed leg verifies the free-run
    /// against `golden.benchmark`, NOT cases[0]. A conformant timed mock must therefore RETURN that
    /// benchmark block's oracle — `(expected_prefill_token, expected_decode_seed_token,
    /// expected_decode_tokens)` — so this reads them straight off the golden the test built. Deriving
    /// them (rather than restating literals) also makes the tests FAIL if the timed leg ever wrongly
    /// read cases[0] again: the benchmark tokens are deliberately distinct from cases[0]'s stream.
    fn bench_oracle(golden: &GoldenFixture) -> (i64, i64, Vec<i64>) {
        let b = golden
            .benchmark
            .as_ref()
            .expect("a benchmark-bearing golden times the benchmark oracle (Option A)");
        (
            b.expected_prefill_token,
            b.expected_decode_seed_token,
            b.expected_decode_tokens.clone(),
        )
    }

    /// THE ONE WINDOW WORKER these tests drive. The timed legs run on the SAME session as the
    /// correctness gate, so one mock answers BOTH: the teacher-forced `cases[]` stream the gate
    /// checks, and the benchmark oracle the free-run timed leg verifies against.
    fn window_engine(golden: &GoldenFixture, teacher_forced: Vec<i64>) -> MockEngine {
        let (bp, bs, bd) = bench_oracle(golden);
        MockEngine::new()
            .teacher_forced_tokens(teacher_forced)
            .free_run_capable()
            .oracle_tokens(bp, bs, bd)
    }

    /// [`window_engine`] for a gate that checks SEVERAL teacher-forced sequences.
    fn window_engine_sequences(golden: &GoldenFixture, sequences: Vec<Vec<i64>>) -> MockEngine {
        let (bp, bs, bd) = bench_oracle(golden);
        MockEngine::new()
            .teacher_forced_sequences(sequences)
            .free_run_capable()
            .oracle_tokens(bp, bs, bd)
    }

    /// #134 — the `score.*.json` `metrics.error` SINK (the sibling of measure-job's
    /// `rejected_pairs[].reason`). A timed-phase spawn/handshake failure seals a `RunnerError`
    /// Display here, and since #134 that carries the worker's own stderr — so the seal boundary
    /// must scrub it. Secret-SHAPED without any `expected`/`actual` trigger word, so the
    /// pre-existing keyword filter would pass every byte of it through.
    #[test]
    fn failed_payload_scrubs_engine_text_before_sealing_metrics_error() {
        let weights = DirDigest {
            sha256: "w".repeat(64),
            file_count: 1,
            byte_count: 1,
        };
        let golden = TestGolden::new().fixture();
        let failure = FailureReport::message(
            format!(
                "engine hello handshake failed: protocol violation: engine closed the stream \
                 before returning a response (worker exited with status 9; worker stderr tail: \
                 open /Users/operator/pool-goldens/sample-001.json failed | \
                 AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCY | \
                 host=api.example.internal | {})",
                "P".repeat(8192)
            ),
            false,
        );
        let payload = failed_payload(
            Mode::LocalIterate,
            failure,
            &golden,
            RunDigests::for_test(&weights),
            ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
        );
        let sealed = payload.metrics.error;

        for secret in [
            "/Users/operator/pool-goldens",
            "wJalrXUtnFEMIK7MDENGbPxRfiCY",
            "api.example.internal",
        ] {
            assert!(
                !sealed.contains(secret),
                "secret-tier content sealed into metrics.error: {secret:?}"
            );
        }
        assert!(
            sealed.len() <= bench_runner::SEALED_REASON_BYTE_LIMIT,
            "sealed metrics.error not capped: {} bytes",
            sealed.len()
        );
        assert!(
            sealed.starts_with("engine hello handshake failed"),
            "signature lost: {sealed}"
        );
        assert!(
            sealed.contains("sample-001.json"),
            "diagnosis lost: {sealed}"
        );
    }

    /// The local-iterate checked decode window, mode-derived rather than literal so a future
    /// window change moves every fixture with it (the `16` these tests used to hard-code was
    /// the retired Laguna fork's value, and is exactly how the window drift stayed invisible).
    fn li_steps() -> usize {
        Mode::LocalIterate.decode_steps(&test_window())
    }
    /// The `expected_tokens` arity the REFERENCE loader demands for that window: seed + one per
    /// decode step ([`Mode::golden_required_steps`]). Every local-iterate fixture below is
    /// built AND loaded at this arity via `TestGolden::steps`.
    fn li_expected() -> usize {
        li_steps() + 1
    }

    /// A no-op cool gate for tests (never reads the GPU). The real macmon gate lives in
    /// `crate::coolgate` and is exercised by its own unit tests.
    fn no_cool_gate(_phase: &str) -> bench_runner::Result<()> {
        Ok(())
    }

    /// The timed-worker spawner for a GATED flow. The window runs every phase on the correctness
    /// session, so this is never called; it refuses loudly if it ever is.
    fn unused_spawner() -> bench_runner::Result<Session<MockEngine>> {
        Err(bench_runner::RunnerError::Protocol(
            "a gated window must never open a second worker".to_string(),
        ))
    }

    /// The two-anchor `correctness_gates` block the phase-bracketing tests share: two
    /// single-token anchors over the same 8-token context, expecting 100 then 200.
    fn anchor_gates(first: &str, second: &str) -> serde_json::Value {
        json!({
            "anchors": [
                { "name": first, "context_tokens": vec![1i64; 8], "expected_token": 100, "accepted_tokens": [100] },
                { "name": second, "context_tokens": vec![1i64; 8], "expected_token": 200, "accepted_tokens": [200] }
            ]
        })
    }

    /// A minimal valid golden with a benchmark oracle and NO correctness gates
    /// (so the conformance gate passes vacuously against the mock).
    fn benchmark_golden() -> GoldenFixture {
        TestGolden::new()
            .case_name("case-a")
            .steps(li_expected())
            .fixture()
    }

    /// R3 helper: a golden whose primary case `p1` (expects `[2; li_expected()]`) is conformant, plus a
    /// `correctness_gates` block the mock engine CANNOT satisfy (anchor argmax ≠ 999 /
    /// free-run ≠ the golden's tokens). Under Swift-exact default the gate skips it; under
    /// `--strict` the superset evaluates and fails it.
    fn golden_with_bad_gates(gates: serde_json::Value) -> GoldenFixture {
        TestGolden::new()
            .steps(li_expected())
            .gates(gates)
            .fixture()
    }

    /// R3 helper: run local-iterate against `golden` with the given `strict` flag on a mock
    /// whose primary case + timed phases are conformant to `p1` (seed=2, window `[2; li_steps()]`).
    fn run_r3(golden: &GoldenFixture, strict: bool) -> ScorePayload {
        let (mut session, _) =
            Session::connect(window_engine(golden, vec![2i64; li_expected()])).unwrap();
        // Option A: the timed leg verifies the free-run against the golden's benchmark oracle.
        let (bp, bs, bd) = bench_oracle(golden);
        iterate_flow_windowed(
            Some(&mut session),
            test_flow(
                golden,
                Mode::LocalIterate,
                strict,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            move || {
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    bp,
                    bs,
                    bd.clone(),
                ))
                .map(|(s, _)| s)
            },
            no_cool_gate,
        )
    }

    #[test]
    fn r3_local_iterate_default_ignores_corrupted_anchor_but_strict_fails() {
        // R3 (Swift-exact by default): Swift `--local-iterate`
        // (QwenRuntimeLocalIterate.swift:106-108 / runLocalIterateCheckedTiming) judges
        // correctness SOLELY from the primary `cases.first` timing stream — it never
        // evaluates `correctness_gates.anchors`. A golden with a CORRUPTED anchor (engine
        // argmax ≠ 999, no accepted/rank tolerance) must therefore PASS by default, exactly
        // like a run with intact anchors, and FAIL only under the `--strict` superset.
        let corrupt = golden_with_bad_gates(json!({
            "anchors": [
                { "name": "bad-anchor", "context_tokens": vec![1i64; 8], "expected_token": 999, "accepted_tokens": [999] }
            ]
        }));

        // Default (NOT strict): the anchor gate is skipped → passes with a real score, and no
        // gate failure is recorded.
        let default_payload = run_r3(&corrupt, false);
        assert!(
            default_payload.passed,
            "Swift-exact default skips the anchor gate; error={}",
            default_payload.metrics.error
        );
        assert!(default_payload.metrics.passed_correctness);
        assert!(default_payload.score.is_some());
        assert_eq!(default_payload.metrics.first_failing_case, None);

        // A golden with NO anchor gate at all (same conformant primary case) produces the
        // SAME default result — proving the corrupt anchor is genuinely not evaluated.
        let no_gate_default = run_r3(&benchmark_golden(), false);
        assert!(no_gate_default.passed);
        assert_eq!(
            default_payload.passed, no_gate_default.passed,
            "default result is identical whether or not a (corrupt) anchor gate is present"
        );

        // --strict: the superset evaluates the anchor gate and FAILS on the corruption.
        let strict_payload = run_r3(&corrupt, true);
        assert!(
            !strict_payload.passed,
            "--strict must evaluate the anchor gate and fail the corruption"
        );
        assert!(!strict_payload.metrics.passed_correctness);
        assert_eq!(
            strict_payload.metrics.first_failing_case.as_deref(),
            Some("bad-anchor")
        );
    }

    #[test]
    fn r3_local_iterate_default_ignores_corrupted_free_run_but_strict_fails() {
        // R3: a CORRUPTED free-run gate (mock free-run returns [4000, 4001…], golden expects
        // [999, 998]) must PASS by default (Swift never evaluates free-run in local-iterate)
        // and FAIL only under `--strict`.
        let corrupt = golden_with_bad_gates(json!({
            "free_run": [
                { "name": "bad-free-run", "prompt_tokens": vec![1i64; CORRECTNESS_PROMPT_TOKENS], "expected_tokens": [999, 998] }
            ]
        }));

        let default_payload = run_r3(&corrupt, false);
        assert!(
            default_payload.passed,
            "Swift-exact default skips the free-run gate; error={}",
            default_payload.metrics.error
        );
        assert!(default_payload.metrics.passed_correctness);
        assert!(default_payload.score.is_some());
        assert_eq!(default_payload.metrics.first_failing_case, None);

        let strict_payload = run_r3(&corrupt, true);
        assert!(
            !strict_payload.passed,
            "--strict must evaluate the free-run gate and fail the corruption"
        );
        assert_eq!(
            strict_payload.metrics.first_failing_case.as_deref(),
            Some("bad-free-run")
        );
    }

    #[test]
    fn r3_local_iterate_default_still_fails_corrupted_primary_case() {
        // R3 must NOT weaken the base-case check: a corrupted primary `cases[0]` (golden
        // expects 999 at index 3, engine returns 2) still FAILS by default, reported
        // Swift-style. Only anchor/free-run/behavior corruptions now pass by default.
        let golden = TestGolden::new()
            .steps(li_expected())
            // David's blanket-10% ruling (2026-09-02): a base case REFUSES only when the mismatches
            // break `COHORT_TOKEN_TOLERANCE_PER_THOUSAND`, so the corruption must pass the budget;
            // the FIRST mismatch stays at index 3.
            .corrupt_expected_beyond_budget(3, 999)
            .fixture();
        let payload = run_r3(&golden, false);
        assert!(!payload.passed, "a corrupted primary case must always fail");
        assert!(!payload.metrics.passed_correctness);
        assert_eq!(
            payload.metrics.first_failing_case.as_deref(),
            Some("local-iterate")
        );
    }

    #[test]
    fn capture_timed_only_skips_the_gate_but_still_records_a_pair() {
        // a8/David ruling (capture doc §5): in `--capture-baseline` mode the engine cannot change
        // between passes of the organizer's OWN calibration, so the PLE-SSD-bound teacher-forced
        // correctness gate is run ONCE per prompt per window (the FIRST capture pass) and every
        // remaining pass is `--capture-timed-only` (`timed_only = true`), which SKIPS the gate.
        //
        // Model a golden whose PRIMARY cases[0] is corrupted (expects 999 at index 3) against a
        // shared-session engine that returns 2 there: the correctness gate WOULD FAIL. Baselines are
        // the INERT (0.0, 0.0) pair the capture path uses by construction, so a healthy timed pass
        // carries `passed_correctness = true` + the INVALID_LOCAL_SCORE_ERROR text — exactly the
        // payload `capture::refuse_unless_timed_phase_succeeded` admits.
        let golden = TestGolden::new()
            .steps(li_expected())
            // David's blanket-10% ruling (2026-09-02): a base case REFUSES only when the mismatches
            // break `COHORT_TOKEN_TOLERANCE_PER_THOUSAND`, so the corruption must pass the budget;
            // the FIRST mismatch stays at index 3.
            .corrupt_expected_beyond_budget(3, 999)
            .fixture();
        let (bp, bs, bd) = bench_oracle(&golden);
        // One run of the flow at the given `timed_only`. Shared session teacher-forces [2; …] (so
        // the corrupted cases[0] index-3 mismatches); the timed leg is free-run conformant.
        let run = |timed_only: bool| {
            let bd = bd.clone();
            // a8 (finish #236): a timed-only pass carries NO correctness session — its spawn is
            // skipped upstream (`run_local_iterate` passes `None`). The gated pass carries exactly
            // one. Mirror that here so the timed-only arm exercises the `None` path.
            let mut correctness = (!timed_only).then(|| {
                Session::connect(MockEngine::new().teacher_forced_tokens(vec![2i64; li_expected()]))
                    .unwrap()
                    .0
            });
            iterate_flow_windowed(
                correctness.as_mut(),
                // strict off — Swift-exact base-case gate
                FlowParams {
                    scoring: ScoringInputs::local(0.0, 0.0),
                    timed_only,
                    ..test_flow(
                        &golden,
                        Mode::LocalIterate,
                        false,
                        RunDigests::for_test(&DirDigest::empty()),
                        &test_window(),
                    )
                },
                move || {
                    Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                        bp,
                        bs,
                        bd.clone(),
                    ))
                    .map(|(s, _)| s)
                },
                no_cool_gate,
            )
        };

        // FIRST pass (`timed_only = false`, the gate runs): the corrupt primary case FAILS the gate.
        let gated = run(false);
        assert!(
            !gated.passed,
            "the first (gated) pass must run the correctness gate and fail the corrupt case"
        );
        assert!(!gated.metrics.passed_correctness);
        assert_eq!(
            gated.metrics.first_failing_case.as_deref(),
            Some("local-iterate")
        );

        // TIMED-ONLY pass (`timed_only = true`, the gate is skipped): the SAME corrupted golden now
        // yields a real timed pair — the gate never ran, so the corruption is invisible.
        let timed = run(true);
        assert!(
            timed.metrics.passed_correctness,
            "timed-only skips the gate, so no gate failure is recorded"
        );
        assert_eq!(
            timed.metrics.first_failing_case, None,
            "no correctness gate ran, so no failing case is reported"
        );
        assert_eq!(
            timed.metrics.error, INVALID_LOCAL_SCORE_ERROR,
            "a healthy inert-baseline capture pass carries exactly the invalid-local-score text"
        );
        assert!(
            timed.metrics.prefill_seconds_per_token > 0.0
                && timed.metrics.decode_seconds_per_token > 0.0,
            "the timed-only pass must still measure a finite positive prefill/decode pair"
        );

        // END TO END: the real capture gate ADMITS the timed-only payload, and the pair MERGES into
        // a record — this is what `--capture-baseline --capture-timed-only` appends for passes 2-4/B.
        crate::capture::refuse_unless_timed_phase_succeeded(&timed.metrics)
            .expect("the timed-only capture pass must pass the capture timed-phase gate");
        crate::capture::merge(
            None,
            crate::capture::CaptureIdentity {
                track_id: "qwen3.8-125b-a6b-mlx-v1".to_string(),
                mode: Mode::LocalIterate.mode_name().to_string(),
                decode_steps: Mode::LocalIterate.decode_steps(&test_window()) as i64,
                engine_sha256: "e".repeat(64),
                weights_sha256: "w".repeat(64),
                golden_sha256: golden.sha256.clone(),
                benchd_sha256: "b".repeat(64),
            },
            crate::capture::CaptureRun {
                prefill_seconds_per_token: timed.metrics.prefill_seconds_per_token,
                decode_seconds_per_token: timed.metrics.decode_seconds_per_token,
            },
        )
        .expect("the timed-only pair must be a valid capture run");
    }

    /// L12c — the SERIAL-LEG gate of the calibration, through the MOCK WORKER end to end.
    ///
    /// benchd puts no spec on a capture leg, but the REQUEST is only half of it: a resident serve
    /// started with speculation on drafts anyway, and the pair that leg measures would be pinned as
    /// the SERIAL denominator every scored speculative leg is divided by. The engine's own
    /// `effective_spec` ECHO is the only place that shows, and `seal_timing_surface_facts` writes it
    /// into `effective_spec_mode`/`effective_spec_depth` on every timed leg.
    ///
    /// POSITIVE control (the defect): a mock that echoes `{"mode":"mtp","depth":2}` on a seed
    /// forward that carried NO spec — the resident-arming case — seals `mtp`/2, and
    /// `capture::capture_run_from` refuses BY NAME. NEGATIVE control: the SAME flow with a
    /// conformant mock seals `serial`/0 and yields the pair, so the gate discriminates rather than
    /// always firing.
    ///
    /// It asserts through `capture_run_from` — the function that PRODUCES the recorded pair, and
    /// the exact call both `main.rs` recorders make — not through `refuse_spec_armed_engine`
    /// directly. A test on the bare refusal passes even when the gate is wired to nothing.
    #[test]
    fn a_capture_leg_whose_engine_speculated_refuses_by_name() {
        let golden = TestGolden::new().steps(li_expected()).fixture();
        let (bp, bs, bd) = bench_oracle(&golden);
        let run = |armed: bool| {
            let bd = bd.clone();
            iterate_flow_windowed(
                None,
                // timed-only: exactly what every calibration pass runs
                FlowParams {
                    scoring: ScoringInputs::local(0.0, 0.0),
                    timed_only: true,
                    ..test_flow(
                        &golden,
                        Mode::LocalIterate,
                        false,
                        RunDigests::for_test(&DirDigest::empty()),
                        &test_window(),
                    )
                },
                move || {
                    let engine =
                        MockEngine::new()
                            .free_run_capable()
                            .oracle_tokens(bp, bs, bd.clone());
                    // The engine speculating WITHOUT being asked: benchd sends no spec (the
                    // calibration never does), the engine echoes one anyway.
                    let engine = if armed {
                        engine.diverge_spec_echo(bench_protocol::SpecConfig::mtp(2))
                    } else {
                        engine
                    };
                    Session::connect(engine).map(|(s, _)| s)
                },
                no_cool_gate,
            )
        };

        // Both assertions go through `capture_run_from` — the GUARDED FORM, and the exact call the
        // two recorders in `main.rs` make. Asserting on `refuse_spec_armed_engine` directly would
        // pass even with the gate unwired from the capture path, which is the hole this test exists
        // to close.

        // NEGATIVE CONTROL — a conformant serial leg seals `serial`/0 and YIELDS a pair.
        let serial = run(false);
        assert_eq!(
            serial.metrics.effective_spec_mode.as_deref(),
            Some(bench_protocol::SPEC_MODE_SERIAL)
        );
        assert_eq!(serial.metrics.effective_spec_depth, Some(0));
        let pair = crate::capture::capture_run_from(&serial.metrics)
            .expect("a serial capture leg must yield a capture run");
        assert_eq!(
            pair.decode_seconds_per_token,
            serial.metrics.decode_seconds_per_token
        );

        // POSITIVE CONTROL — the armed engine's echo is sealed, and NO pair comes out.
        let armed = run(true);
        assert_eq!(armed.metrics.effective_spec_mode.as_deref(), Some("mtp"));
        assert_eq!(armed.metrics.effective_spec_depth, Some(2));
        let err = crate::capture::capture_run_from(&armed.metrics).unwrap_err();
        assert!(
            err.contains(crate::capture::CALIBRATION_SPEC_ARMED),
            "the refusal must name the sentinel: {err}"
        );
        assert!(err.contains("SERIAL denominator"), "{err}");
    }

    /// a8 Option-A capture restructure (David 2026-08-31) — the SCORING-CRITICAL invariant: N passes
    /// driven over ONE persistent residency measure BYTE-IDENTICALLY to N standalone single-pass
    /// invocations, and no pass reloads the model.
    ///
    /// PROOF OF LOAD-ONCE: `spawn_timed` bumps a counter and it must stay ZERO — every pass (the
    /// gated first one and every timed-only pass) runs over the ONE session handed in, so the
    /// timed-worker spawner is never called. That is the whole point: 5 passes, one model
    /// residency.
    ///
    /// PROOF OF BYTE-IDENTICAL MEASUREMENT: the exact TIMED WORKLOAD each pass feeds the engine — the
    /// prefill prompt, the decode seed, and the free-run token budget N — is captured via
    /// `record_timing` and asserted EQUAL to a standalone single-pass timed invocation's over the
    /// same golden. The measured quantity is `measure_prefill` / `measure_free_run_decode` over THAT
    /// workload (same `decode_steps`, same seconds-per-token formulas, same `iterate_flow_windowed`
    /// entry) — so identical inputs through identical code is a byte-identical measured number; only
    /// the wall-clock nanoseconds (never a byte-stable quantity) differ, exactly as they would
    /// between two standalone invocations.
    #[test]
    fn capture_passes_over_one_residency_measure_byte_identically_and_never_reload() {
        use bench_runner::mock::RecordedTiming;
        use std::cell::{Cell, RefCell};
        use std::rc::Rc;

        let golden = benchmark_golden();
        let (bp, bs, bd) = bench_oracle(&golden);

        // One conformant persistent engine: free-run capable (timed leg) + the benchmark oracle
        // (timed tokens) + teacher-forces the primary case's [2; …] stream (so the FIRST pass's gate
        // passes). It serves the correctness gate AND every timed leg over ONE session.
        let rec = Rc::new(RefCell::new(RecordedTiming::default()));
        let (mut session, _hello) = Session::connect(
            MockEngine::new()
                .free_run_capable()
                .oracle_tokens(bp, bs, bd.clone())
                .teacher_forced_tokens(vec![2i64; li_expected()])
                .record_timing(Rc::clone(&rec)),
        )
        .unwrap();

        // A timed-worker spawner that must NEVER be called — its firing would mean a pass reloaded
        // the model. It still returns a valid session type so the closure's
        // signature matches (`iterate_flow_windowed`'s `spawn_timed`).
        let spawns = Rc::new(Cell::new(0usize));
        let spawns_c = Rc::clone(&spawns);
        let bd_spawn = bd.clone();
        let spawn_timed = move || {
            spawns_c.set(spawns_c.get() + 1);
            Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                bp,
                bs,
                bd_spawn.clone(),
            ))
            .map(|(s, _)| s)
        };

        // W,A,A,B,B — five passes over the ONE residency.
        let payloads = run_capture_passes_over_session(
            &mut session,
            FlowParams {
                scoring: ScoringInputs::local(0.0, 0.0),
                ..test_flow(
                    &golden,
                    Mode::LocalIterate,
                    false,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                )
            },
            5,
            spawn_timed,
            no_cool_gate,
        );

        assert_eq!(payloads.len(), 5, "one payload per pass");
        assert_eq!(
            spawns.get(),
            0,
            "every pass must run over the ONE residency — no pass reloads"
        );
        // Every pass measured a healthy inert-baseline pair (the capture path's expected shape).
        for (i, p) in payloads.iter().enumerate() {
            assert!(
                p.metrics.passed_correctness,
                "pass {i} must pass (gate on pass 0, gate-skipped after)"
            );
            assert_eq!(
                p.metrics.error, INVALID_LOCAL_SCORE_ERROR,
                "pass {i} is a healthy inert-baseline capture pass"
            );
            assert!(
                p.metrics.prefill_seconds_per_token > 0.0
                    && p.metrics.decode_seconds_per_token > 0.0,
                "pass {i} must measure a finite positive pair"
            );
            assert!(
                crate::capture::refuse_unless_timed_phase_succeeded(&p.metrics).is_ok(),
                "the capture gate must admit pass {i}"
            );
        }
        // The multi-pass workload the engine actually received (final = the last pass, and every pass
        // feeds the same golden so every pass is identical). Snapshot it.
        let multi_workload = rec.borrow().clone();

        // REFERENCE: a standalone single-pass timed-only invocation over a FRESH engine, same golden.
        // Its recorded workload must equal the multi-pass one — same inputs → same measured quantity.
        let ref_rec = Rc::new(RefCell::new(RecordedTiming::default()));
        let (mut ref_session, _h) = Session::connect(
            MockEngine::new()
                .free_run_capable()
                .oracle_tokens(bp, bs, bd.clone())
                .record_timing(Rc::clone(&ref_rec)),
        )
        .unwrap();
        let bd_ref = bd.clone();
        let single = iterate_flow_windowed(
            Some(&mut ref_session),
            FlowParams {
                scoring: ScoringInputs::local(0.0, 0.0),
                // timed-only, exactly what a standalone A/B capture pass runs
                timed_only: true,
                ..test_flow(
                    &golden,
                    Mode::LocalIterate,
                    false,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                )
            },
            move || {
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    bp,
                    bs,
                    bd_ref.clone(),
                ))
                .map(|(s, _)| s)
            },
            no_cool_gate,
        );
        assert!(single.metrics.prefill_seconds_per_token > 0.0);
        let ref_workload = ref_rec.borrow().clone();

        assert_eq!(
            multi_workload, ref_workload,
            "a multi-pass pass must feed the engine the byte-identical timed workload a standalone \
             single-pass invocation does (prefill prompt, decode seed, free-run N) — so the measured \
             quantity is byte-identical; only wall-clock nanoseconds differ"
        );
        // And that workload is the benchmark oracle's (Option A): prompt + seed non-empty, N budget set.
        assert!(!multi_workload.prefill_prompt.is_empty());
        assert!(!multi_workload.decode_seed.is_empty());
        assert_eq!(
            multi_workload.free_decode_run_count,
            Some(Mode::LocalIterate.decode_steps(&test_window()) as u32),
            "the timed leg's free-run budget N is the mode's decode window, every pass"
        );
    }

    /// a8/David 2026-08-31 GATELESS: `--capture-passes` runs NO teacher-forced correctness gate on
    /// ANY pass — not even pass 0 (the TF gate is dropped from calibration entirely; the free-run
    /// oracle token-match per pass is the correctness evidence). Prove it positionally with a golden
    /// whose PRIMARY case is corrupt (index 3 expects 999) but whose BENCHMARK oracle is intact: with
    /// the gate gone, EVERY pass (including pass 0) SKIPS the corrupt primary case and yields a
    /// healthy timed pair, so the corruption is invisible to all of them — the TF gate never runs.
    #[test]
    fn capture_passes_runs_no_tf_gate_on_any_pass() {
        let golden = TestGolden::new()
            .steps(li_expected())
            .corrupt_expected_at(3, 999)
            .fixture();
        let (bp, bs, bd) = bench_oracle(&golden);

        let (mut session, _hello) = Session::connect(
            MockEngine::new()
                .free_run_capable()
                .oracle_tokens(bp, bs, bd.clone())
                // Teacher-forces [2; …], so the gate sees the corrupt index-3 (=999) mismatch.
                .teacher_forced_tokens(vec![2i64; li_expected()]),
        )
        .unwrap();

        let bd_spawn = bd.clone();
        let spawn_timed = move || {
            Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                bp,
                bs,
                bd_spawn.clone(),
            ))
            .map(|(s, _)| s)
        };

        let payloads = run_capture_passes_over_session(
            &mut session,
            FlowParams {
                scoring: ScoringInputs::local(0.0, 0.0),
                ..test_flow(
                    &golden,
                    Mode::LocalIterate,
                    false,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                )
            },
            3, // W, A, A — 3 passes, NONE gated (gateless)
            spawn_timed,
            no_cool_gate,
        );

        // GATELESS: NO pass runs the TF gate, so the corrupt primary case is invisible to EVERY pass
        // (including pass 0). All yield a healthy timed pair over the one residency and none report a
        // gate failure — proving the TF gate never runs under --capture-passes.
        for p in &payloads[..] {
            assert!(
                p.metrics.passed_correctness,
                "every pass skips the gate (gateless), so no gate failure is recorded"
            );
            assert_eq!(
                p.metrics.first_failing_case, None,
                "no gate ran on any pass, so no failing case is reported"
            );
            assert!(
                p.metrics.prefill_seconds_per_token > 0.0
                    && p.metrics.decode_seconds_per_token > 0.0
            );
        }
    }

    /// a8 (finish #236) — the SPAWN-COUNT the sub-win turns on: a `--capture-timed-only` pass
    /// constructs ZERO correctness sessions (its spawn is skipped), while a gated pass constructs
    /// exactly ONE. This mirrors `run_local_iterate`, which now spawns the shared correctness engine
    /// ONLY on the `!timed_only` arm and passes `None` otherwise; the counter here stands in for the
    /// `ChildStdioTransport` spawn that arm performs. Both arms drive the PRODUCTION `iterate_flow`,
    /// so the timed-only arm additionally proves the timed pass reaches a valid pair with no
    /// correctness session at all.
    #[test]
    fn timed_only_spawns_no_correctness_session_gated_spawns_one() {
        let golden = benchmark_golden();
        let (bp, bs, bd) = bench_oracle(&golden);

        let run = |timed_only: bool| -> (usize, ScorePayload) {
            let correctness_spawns = std::rc::Rc::new(std::cell::Cell::new(0usize));
            let bd = bd.clone();
            let timed = move || {
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    bp,
                    bs,
                    bd.clone(),
                ))
                .map(|(s, _)| s)
            };
            // The correctness session is constructed ONLY on the gated arm — exactly the branch
            // run_local_iterate takes. The counter is bumped at that one spawn site.
            let mut correctness = (!timed_only).then(|| {
                correctness_spawns.set(correctness_spawns.get() + 1);
                Session::connect(window_engine(&golden, vec![2i64; li_expected()]))
                    .unwrap()
                    .0
            });
            let payload = iterate_flow_windowed(
                correctness.as_mut(),
                FlowParams {
                    timed_only,
                    ..test_flow(
                        &golden,
                        Mode::LocalIterate,
                        false,
                        RunDigests::for_test(&DirDigest::empty()),
                        &test_window(),
                    )
                },
                timed,
                no_cool_gate,
            );
            (correctness_spawns.get(), payload)
        };

        // TIMED-ONLY: zero correctness sessions, yet a real finite-positive timed pair.
        let (timed_only_spawns, timed_payload) = run(true);
        assert_eq!(
            timed_only_spawns, 0,
            "a --capture-timed-only pass must spawn NO correctness session"
        );
        assert!(
            timed_payload.metrics.prefill_seconds_per_token > 0.0
                && timed_payload.metrics.decode_seconds_per_token > 0.0,
            "the timed-only pass still measures a finite positive pair with no correctness session"
        );

        // GATED: exactly one correctness session, and the run passes end to end (gate ran on it).
        let (gated_spawns, gated_payload) = run(false);
        assert_eq!(
            gated_spawns, 1,
            "a gated pass spawns exactly one correctness session"
        );
        assert!(
            gated_payload.passed,
            "the gated pass runs the gate on its session and passes; error={}",
            gated_payload.metrics.error
        );
    }

    /// a8 (finish #236) — the guard that makes the spawn-skip SAFE: `iterate_flow` with
    /// `timed_only = false` and NO session panics. This proves the gated path genuinely REQUIRES a
    /// correctness session (the gate runs on it), so the `None` shortcut is admissible ONLY for a
    /// timed-only pass — `run_local_iterate` must, and does, spawn one on the gated arm.
    #[test]
    #[should_panic(expected = "requires a correctness session")]
    fn gated_flow_without_a_session_panics() {
        let golden = benchmark_golden();
        let (bp, bs, bd) = bench_oracle(&golden);
        let _ = iterate_flow_windowed(
            None::<&mut Session<MockEngine>>,
            // NOT timed_only → the gate runs → the session is mandatory
            FlowParams {
                timed_only: false,
                ..test_flow(
                    &golden,
                    Mode::LocalIterate,
                    false,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                )
            },
            move || {
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    bp,
                    bs,
                    bd.clone(),
                ))
                .map(|(s, _)| s)
            },
            no_cool_gate,
        );
    }

    /// LOCAL-ITERATE NOW POPULATES `per_prompt`. It was sealed only on the official path, so every
    /// local score carried an empty array and the board had nothing to read from a local run — even
    /// though local-iterate times a prompt exactly as official does. The record carries the timed
    /// golden's own sha256, the audit's realized draft length, and the ENFORCED whole-window decode
    /// figure (never a second decode-only quantity).
    #[test]
    fn local_iterate_seals_the_per_prompt_record_for_the_prompt_it_timed() {
        let golden = benchmark_golden();
        let (mut session, _) = Session::connect(
            window_engine(&golden, vec![2i64; li_expected()])
                // 2 committed tokens per round over the whole window.
                .free_run_acceptance_lengths(vec![2; li_steps() / 2]),
        )
        .unwrap();
        let weights = DirDigest::empty();
        let payload = iterate_flow_windowed(
            Some(&mut session),
            test_flow(
                &golden,
                Mode::LocalIterate,
                false,
                RunDigests::for_test(&weights),
                &test_window(),
            ),
            unused_spawner,
            no_cool_gate,
        );
        assert!(payload.passed);
        assert_eq!(payload.metrics.per_prompt.len(), 1);
        let pp = &payload.metrics.per_prompt[0];
        assert_eq!(pp.prompt_sha256, golden.sha256);
        assert_eq!(pp.effective_mean_draft_len, 2.0);
        assert_eq!(
            pp.mtp_seconds_per_token_mean,
            payload.metrics.decode_seconds_per_token
        );
        // No spec was requested, so the leg is serial: the drafting counters stay absent on the
        // entry and on the run, and the entry keeps its historical three keys.
        assert_eq!(pp.spec_rounds, None);
        assert_eq!(pp.spec_drafted_total, None);
        assert_eq!(
            payload.metrics.effective_spec_mode.as_deref(),
            Some("serial")
        );
        assert_eq!(payload.metrics.effective_spec_depth, Some(0));
    }

    #[test]
    fn iterate_end_to_end_floors_pass_and_score_matches_core() {
        let golden = benchmark_golden();
        // Option A: the timed leg verifies each engine token against the golden's BENCHMARK oracle
        // (prefill 5, seed 6, decode [7; li_steps()] from the default benchmark_oracle), so the timed
        // mock must return exactly those. Correctness is separate: it runs on the shared session and
        // still verifies the teacher-forced [2; li_expected()] cases[0] stream.
        let (mut session, _) =
            Session::connect(window_engine(&golden, vec![2i64; li_expected()])).unwrap();
        let weights = DirDigest::empty();
        let baseline_prefill = TEST_BASELINE.prefill_seconds_per_token;
        let baseline_decode = TEST_BASELINE.decode_seconds_per_token;
        let (bp, bs, bd) = bench_oracle(&golden);

        let payload = iterate_flow_windowed(
            Some(&mut session),
            FlowParams {
                scoring: ScoringInputs::local(baseline_prefill, baseline_decode),
                ..test_flow(
                    &golden,
                    Mode::LocalIterate,
                    false,
                    RunDigests::for_test(&weights),
                    &test_window(),
                )
            },
            move || {
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    bp,
                    bs,
                    bd.clone(),
                ))
                .map(|(s, _)| s)
            },
            no_cool_gate,
        );

        assert!(payload.passed, "mock timing -> huge speedup should pass");
        let score = payload.score.expect("passing run carries a numeric score");
        assert!(score.is_finite() && score > 0.0);
        let m = &payload.metrics;
        // mock timing is tiny -> speedup huge -> both floors clear.
        assert!(m.passed_decode_speedup_floor);
        assert!(m.passed_prefill_speedup_floor);
        assert!(m.passed_correctness);
        assert_eq!(m.runtime, "rust-local-iterate");
        // LITERAL, not derived from the identity the payload was built with: `num_layers` is a
        // sealed DETERMINISTIC parity field (`parity.rs` ROSTER `Det` — benchd and the Swift
        // score.json must agree byte-for-byte), and this fixture runs the Qwen 3.8 125B-A6B row,
        // whose checkpoints declare 48 decoder layers. Comparing it against the table would pass
        // for ANY value the table happened to hold, which is not a test of the seal.
        assert_eq!(m.num_layers, 48);
        assert_eq!(m.golden_hash, golden.sha256);
        assert_eq!(m.bandwidth_source, BANDWIDTH_SOURCE);
        // Checked-timing counts must match Swift local-iterate for parity:
        // case_count = timingRepeats (1); checked_steps = (decode_steps + 2) * repeats.
        // NOT the conformance case count.
        assert_eq!(m.case_count, 1);
        assert_eq!(m.checked_steps, li_steps() as i64 + 2);

        // The score equals exactly what bench-core computes for (mock timing, baseline).
        let expected = score_weighted(
            m.decode_seconds_per_token,
            m.prefill_seconds_per_token,
            baseline_decode,
            baseline_prefill,
            ScoringWeights::DEFAULT,
        );
        assert_eq!(score, expected);

        // Sealed JSON: all metrics fields present + typed, and the sha256 sidecar
        // matches the exact bytes.
        let json = payload.to_sealed_json().unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let metrics_obj = value.get("metrics").unwrap().as_object().unwrap();
        // 56 SWIFT-PARITY keys + the ADDITIVE seals a timed local-iterate leg now records: the
        // board's `per_prompt` array, the run's `effective_spec_mode`/`effective_spec_depth`,
        // and the additive `local_phases` diagnostics.
        // This leg requests NO spec, so it seals `serial` / `0` and NONE of the drafting counters
        // (nothing drafted), and it runs against a mock with no head provenance, so no engine
        // identity key appears either.
        assert_eq!(
            metrics_obj.len(),
            60,
            "all ScoreMetrics fields must be present"
        );
        assert_eq!(
            metrics_obj["effective_spec_mode"],
            serde_json::json!("serial")
        );
        assert_eq!(metrics_obj["effective_spec_depth"], serde_json::json!(0));
        assert!(metrics_obj.contains_key("per_prompt"));
        for absent in [
            "spec_rounds",
            "spec_drafted_total",
            "spec_accepted_total",
            "spec_acceptance_rate",
            "spec_verify_replay_disagreements",
            "spec_verification_mode",
            "spec_rectangular_verification_rounds",
            "spec_serial_verification_rounds",
            "acceptance_lengths",
        ] {
            assert!(
                !metrics_obj.contains_key(absent),
                "a SERIAL leg drafts nothing, so {absent} must stay absent"
            );
        }
        for key in [
            "decode_seconds_per_token",
            "prefill_seconds_per_token",
            "decode_speedup",
            "prefill_speedup",
            "passed_decode_speedup_floor",
            "passed_prefill_speedup_floor",
            "baseline_decode_seconds_per_token",
            "baseline_prefill_seconds_per_token",
            "passed_correctness",
            "golden_hash",
            "runtime",
        ] {
            assert!(metrics_obj.contains_key(key), "missing metrics key {key}");
        }
        // nullable fields serialize as null, not omitted.
        assert!(metrics_obj.get("first_failing_layer").unwrap().is_null());

        let sidecar = crate::score::sha256_hex(json.as_bytes());
        assert_eq!(sidecar.len(), 64);
        // Recomputing over the same bytes is stable.
        assert_eq!(sidecar, crate::score::sha256_hex(json.as_bytes()));
    }

    #[test]
    fn slow_decode_fails_floor_and_band_but_score_still_published() {
        // Inject a slow decode (10x baseline) directly into the score assembly:
        // the decode speedup drops below the 0.95 floor and out of the acceptance
        // band, but local-iterate still publishes a positive estimated score.
        let golden = benchmark_golden();
        let weights = DirDigest::empty();
        let baseline_prefill = TEST_BASELINE.prefill_seconds_per_token;
        let baseline_decode = TEST_BASELINE.decode_seconds_per_token;
        let slow_decode = baseline_decode * 10.0;

        let timing = TimingResult {
            prefill_seconds_per_token: baseline_prefill,
            decode_seconds_per_token: slow_decode,
            decode_steps: li_steps(),
            prefill_prompt_tokens: BENCHMARK_PREFILL_PROMPT_TOKENS,
            prefill_elapsed_seconds: baseline_prefill * BENCHMARK_PREFILL_PROMPT_TOKENS as f64,
            decode_elapsed_seconds: slow_decode * 16.0,
            peak_ram_gb: 20.25,
            effective_spec: None,
            free_run_audit: None,
            deferred_divergence: None,
            phase_window: None,
        };

        let payload = local_iterate_score(
            Mode::LocalIterate,
            &timing,
            ScoringInputs::local(baseline_prefill, baseline_decode),
            &golden,
            RunDigests::for_test(&weights),
            &test_window(),
        );

        // decode speedup ~0.1 -> below the 0.95 floor.
        assert!(!payload.metrics.passed_decode_speedup_floor);
        assert!(payload.metrics.passed_prefill_speedup_floor);

        // The decode acceptance band fails (value far above +2% of reference).
        let eval = evaluate_timed_run(
            slow_decode,
            baseline_prefill,
            baseline_decode,
            baseline_prefill,
            TEST_BASELINE.bands,
            SpeedupFloors::DEFAULT,
            ScoringWeights::DEFAULT,
        );
        assert!(!eval.decode_band.passed);
        assert!(eval.prefill_band.passed);

        // local-iterate still publishes a finite positive score (bands/floors are
        // reported, not gated, on the local path).
        let score = payload.score.expect("finite positive score");
        assert!(score.is_finite() && score > 0.0);
    }

    #[test]
    fn invalid_timing_yields_null_score() {
        let golden = benchmark_golden();
        let weights = DirDigest::empty();
        let timing = TimingResult {
            prefill_seconds_per_token: 0.0, // non-positive -> invalid
            decode_seconds_per_token: 0.1,
            decode_steps: li_steps(),
            prefill_prompt_tokens: BENCHMARK_PREFILL_PROMPT_TOKENS,
            prefill_elapsed_seconds: 0.0,
            decode_elapsed_seconds: 1.6,
            peak_ram_gb: 0.0,
            effective_spec: None,
            free_run_audit: None,
            deferred_divergence: None,
            phase_window: None,
        };
        let payload = local_iterate_score(
            Mode::LocalIterate,
            &timing,
            ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
            &golden,
            RunDigests::for_test(&weights),
            &test_window(),
        );
        assert!(!payload.passed);
        assert!(payload.score.is_none());
        assert!(payload
            .metrics
            .error
            .contains("local estimated score is invalid"));
    }

    #[test]
    fn checked_steps_and_case_count_match_swift_checked_timing() {
        // (decode_steps + 2) * repeats, repeats = 1 for the LOCAL checked-timing modes:
        // decode_steps + 2 per mode (130 for local-iterate, 1025 for local-submit). case_count =
        // repeats = 1. (Official does NOT use this formula — it reports correctness case
        // counts via official::official_core, so it is not exercised here.)
        let golden = benchmark_golden();
        let weights = DirDigest::empty();
        let baseline_prefill = TEST_BASELINE.prefill_seconds_per_token;
        let baseline_decode = TEST_BASELINE.decode_seconds_per_token;
        for (mode, expected_checked) in [
            (
                Mode::LocalIterate,
                Mode::LocalIterate.decode_steps(&test_window()) as i64 + 2,
            ),
            (
                Mode::LocalSubmit,
                Mode::LocalSubmit.decode_steps(&test_window()) as i64 + 2,
            ),
        ] {
            let timing = TimingResult {
                prefill_seconds_per_token: baseline_prefill,
                decode_seconds_per_token: baseline_decode,
                decode_steps: mode.decode_steps(&test_window()),
                prefill_prompt_tokens: BENCHMARK_PREFILL_PROMPT_TOKENS,
                prefill_elapsed_seconds: baseline_prefill * BENCHMARK_PREFILL_PROMPT_TOKENS as f64,
                decode_elapsed_seconds: baseline_decode * mode.decode_steps(&test_window()) as f64,
                peak_ram_gb: 20.25,
                effective_spec: None,
                free_run_audit: None,
                deferred_divergence: None,
                phase_window: None,
            };
            let payload = local_iterate_score(
                mode,
                &timing,
                ScoringInputs::local(baseline_prefill, baseline_decode),
                &golden,
                RunDigests::for_test(&weights),
                &test_window(),
            );
            assert_eq!(payload.metrics.case_count, 1, "case_count for {mode:?}");
            assert_eq!(
                payload.metrics.checked_steps, expected_checked,
                "checked_steps for {mode:?}"
            );
        }
    }

    #[test]
    fn correctness_gate_failure_produces_failed_payload() {
        // A golden with a free-run gate whose expected tokens the mock cannot match
        // (mock free-run returns [4000, 4001, ...]).
        let golden = TestGolden::new()
            .without_model_type()
            .case_name("case-a")
            .steps(li_expected())
            .gates(json!({
                "free_run": [
                    { "name": "fr-1", "prompt_tokens": vec![1i64; CORRECTNESS_PROMPT_TOKENS], "expected_tokens": [999, 998] }
                ]
            }))
            .fixture_any_model_type();
        // Primary case-a expects [2; li_expected()]; make the engine conformant on it (B3) so the
        // FREE-RUN case fr-1 remains the first failure this test asserts.
        let (mut session, _) =
            Session::connect(window_engine(&golden, vec![2i64; li_expected()])).unwrap();
        // Correctness fails first, so spawn_timed is never invoked (a run that fails before
        // timing spawns no timed engines).
        let payload = iterate_flow_windowed(
            Some(&mut session),
            // R3: --strict exercises the free-run superset gate
            test_flow(
                &golden,
                Mode::LocalIterate,
                true,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            || Session::connect(MockEngine::new()).map(|(s, _)| s),
            no_cool_gate,
        );
        assert!(!payload.passed);
        assert!(payload.score.is_none());
        assert!(!payload.metrics.passed_correctness);
        assert_eq!(payload.metrics.first_failing_case.as_deref(), Some("fr-1"));
    }

    #[test]
    fn b1_anchor_gate_steps_do_not_leak_into_the_prefill_barrier() {
        // TWO anchors (each a timed correctness_begin) + the benchmark oracle, no primary
        // cases. B1: iterate_core must close the correctness phase (phase_diagnostics) so
        // the engine's completed_work (=2) is reconciled and reset BEFORE prefill. Without
        // the close_phase those 2 steps leak into prefill's barrier -> CompletedWorkMismatch
        // -> the run FAILS against a fully conformant engine. This asserts it passes.
        // At least one primary case is required (loader parity); it also runs teacher-
        // forced, so the correctness phase issues li_expected() (base) + 2 (anchor) timed steps.
        let golden = TestGolden::new()
            .case_name("c1")
            .steps(li_expected())
            .expected_fill(3)
            .gates(anchor_gates("anc-1", "anc-2"))
            .fixture();
        // Conformant engine: base case returns 3 for all li_expected() teacher-forced steps, then the
        // anchor argmax = 100 then 200 (teacher_forced oracle sets the top-1); timing oracle
        // matches the benchmark block. Per-sequence oracle (#55): one token list per
        // sequence (base case, anc-1, anc-2) — no hand-concatenation of a flat stream.
        let tf = vec![vec![3i64; li_expected()], vec![100], vec![200]];
        // Correctness on the shared session (teacher-forced cases[0] = [3; li_expected()] + anchors);
        // Option A times the golden's BENCHMARK oracle on FRESH engines per phase.
        let (mut session, _) = Session::connect(window_engine_sequences(&golden, tf)).unwrap();
        let (bp, bs, bd) = bench_oracle(&golden);
        let payload = iterate_flow_windowed(
            Some(&mut session),
            // R3: --strict so the anchor gate actually runs
            test_flow(
                &golden,
                Mode::LocalIterate,
                true,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            move || {
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    bp,
                    bs,
                    bd.clone(),
                ))
                .map(|(s, _)| s)
            },
            no_cool_gate,
        );
        assert!(
            payload.passed,
            "anchor gate steps must be reconciled by close_phase before prefill (B1); error={}",
            payload.metrics.error
        );
        assert!(payload.metrics.passed_correctness);
        // The score's case_count stays the checked-timing repeats (parity), not anchors.
        assert_eq!(payload.metrics.case_count, 1);
    }

    #[test]
    fn v1_drain_allocator_round_trips_phase_diagnostics_to_the_worker() {
        // V1: SessionEngine::drain_allocator must REACH a real worker, not be a no-op. With
        // a 3-sequence golden (one primary case + two anchors) the per-sequence barrier
        // closes a sub-phase at each boundary via phase_diagnostics — base->anchor1,
        // anchor1->anchor2, and the final close before prefill = 3 correctness round-trips
        // (the first sequence only opens) — plus the prefill + decode closes = 5 total. A
        // no-op drain would send ZERO phase_diagnostics during the correctness gate.
        let golden = TestGolden::new()
            .case_name("c1")
            .steps(li_expected())
            .expected_fill(3)
            .gates(anchor_gates("anc-1", "anc-2"))
            .fixture();
        let seen = std::rc::Rc::new(std::cell::Cell::new(0usize));
        // Per-sequence oracle (#55): one token list per sequence (base case, anc-1, anc-2).
        let tf = vec![vec![3i64; li_expected()], vec![100], vec![200]];
        // Correctness on the shared session (teacher-forced cases[0] = [3; li_expected()] + anchors);
        // Option A times the golden's BENCHMARK oracle on FRESH engines per phase. The SAME
        // phase-diagnostics counter is shared across the correctness session and both fresh timed
        // engines, so it observes every drain regardless of which process handled it.
        let (bp, bs, bd) = bench_oracle(&golden);
        let (mut session, _) = Session::connect(
            MockEngine::new()
                .teacher_forced_sequences(tf)
                .count_phase_diagnostics(seen.clone())
                .free_run_capable()
                .oracle_tokens(bp, bs, bd.clone()),
        )
        .unwrap();
        let seen_timed = seen.clone();
        let (bp, bs, bd) = bench_oracle(&golden);
        let payload = iterate_flow_windowed(
            Some(&mut session),
            // R3: --strict so the anchor sequences drain
            test_flow(
                &golden,
                Mode::LocalIterate,
                true,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            move || {
                Session::connect(
                    MockEngine::new()
                        .free_run_capable()
                        .oracle_tokens(bp, bs, bd.clone())
                        .count_phase_diagnostics(seen_timed.clone()),
                )
                .map(|(s, _)| s)
            },
            no_cool_gate,
        );
        assert!(
            payload.passed,
            "conformant multi-sequence run must pass; error={}",
            payload.metrics.error
        );
        assert_eq!(
            seen.get(),
            5,
            "3 correctness-sequence drains (shared session) + 1 prefill close + 1 decode \
             close, the timing closes now on FRESH per-phase engines, all reach a worker (V1)"
        );
    }

    #[test]
    fn fail_path_closes_final_subphase_and_reports_swift_failure_counts() {
        // 1 primary case + 2 anchors; the engine returns a WRONG token at primary step 0 so
        // correctness FAILS. 2.2: the final correctness sub-phase is still closed on the fail
        // exit -> 3 correctness phase_diagnostics (2 inter-sequence drains + 1 final close) on
        // THIS session; the retain-timing phase runs on FRESH separate sessions (item 1), so it
        // does not touch `seen`. Item-1 red-team Finding 3: the DET counts follow Swift's
        // LOCAL-ITERATE FAILURE report — caseCount = timingRepeats (1), checkedSteps =
        // failureStep + 1 — NOT the golden's total correctness cases.
        let golden = TestGolden::new()
            .steps(li_expected())
            .expected_fill(7)
            .gates(anchor_gates("a1", "a2"))
            .fixture();
        let seen = std::rc::Rc::new(std::cell::Cell::new(0usize));
        // Per-sequence oracle (#55): primary step 0 = 999 (WRONG; expected 7), rest 7; the
        // anchor sequences (100, 200) are conformant — one token list per sequence.
        let mut primary = vec![7i64; li_expected()];
        // David's blanket-10% ruling (2026-09-02): the base case now REFUSES only when the
        // mismatches break `COHORT_TOKEN_TOLERANCE_PER_THOUSAND`, so a correctness-FAILURE test
        // must corrupt past the budget. The FIRST mismatch stays at the index asserted below.
        corrupt_beyond_budget(&mut primary, 0, 999);
        let tf = vec![primary, vec![100], vec![200]];
        let (bp, bs, bd) = bench_oracle(&golden);
        let (mut session, _) = Session::connect(
            MockEngine::new()
                .teacher_forced_sequences(tf)
                .count_phase_diagnostics(seen.clone())
                .free_run_capable()
                .oracle_tokens(bp, bs, bd.clone()),
        )
        .unwrap();
        // Correctness fails at primary step 0; item 1 then runs the TIME-ONLY timing phase on
        // FRESH engines (Option A: oracle = the golden's benchmark block), so timing is retained.
        // Those fresh sessions have no `seen` counter, so only the correctness session's sub-phase
        // closes count toward `seen`.
        let (bp, bs, bd) = bench_oracle(&golden);
        let payload = iterate_flow_windowed(
            Some(&mut session),
            // R3: --strict so the anchor sub-phases run
            test_flow(
                &golden,
                Mode::LocalIterate,
                true,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            move || {
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    bp,
                    bs,
                    bd.clone(),
                ))
                .map(|(s, _)| s)
            },
            no_cool_gate,
        );
        assert!(!payload.passed);
        assert!(!payload.metrics.passed_correctness);
        // §F1: a local-iterate base-case failure is reported Swift-style. primary "p1" is
        // corrupted at index 0 (engine returns 999, golden expects 7) → case "local-iterate",
        // checked-step 0 (prefill), expected/actual populated, descriptive error.
        assert_eq!(
            payload.metrics.first_failing_case.as_deref(),
            Some("local-iterate")
        );
        assert_eq!(payload.metrics.first_failing_step, Some(0));
        assert_eq!(payload.metrics.expected_token, Some(7));
        assert_eq!(payload.metrics.actual_token, Some(999));
        assert_eq!(
            payload.metrics.error,
            "local-iterate teacher-forced token mismatch"
        );
        // Swift local-iterate FAILURE counts: caseCount = timingRepeats (1); checkedSteps =
        // failureStep + 1 = 0 + 1 = 1. (Before item-1 Finding 3, this was the golden total, 3.)
        assert_eq!(payload.metrics.case_count, TIMING_REPEATS);
        assert_eq!(
            payload.metrics.checked_steps, 1,
            "checkedSteps = first_failing_step(0) + 1"
        );
        // timing is RETAINED (baselines are the real constants, not blanked 0.0).
        assert!(payload.metrics.baseline_decode_seconds_per_token > 0.0);
        // 2.2: the final sub-phase was closed on the fail exit (2 inter-drains + 1 close), and the
        // retain-timing legs then close their own prefill and decode phases on the SAME worker.
        assert_eq!(
            seen.get(),
            5,
            "the final correctness sub-phase must be closed even on the fail path"
        );
    }

    #[test]
    fn local_iterate_correctness_failure_retains_real_timing() {
        // ITEM 1 (David's ruling — "Swift is the reference"): a local-iterate CORRECTNESS
        // FAILURE must RETAIN real timing/baselines/speedup-floor flags (Swift teacher-forces
        // the expected tokens and times anyway; correctness is judged separately) instead of
        // blanking them to 0/false. This mirrors the `primary` divergence: cases[0] carries a
        // flipped token so correctness fails, but the TIME-ONLY timing phase still runs on
        // FRESH engines and records real timing.
        //
        // The golden's cases[0] "p1" expects [2; li_expected()]; we corrupt index 3 (golden 999, engine
        // 2) so the teacher-forced correctness gate fails at a checked step, exactly as the
        // §F1 test does. The timed engines are conformant to cases[0]'s stream (seed=2, window
        // [2; 16]) — but under TIME-ONLY they'd be timed even if they diverged.
        // golden expects 999 at index 3; the conformant engine returns 2.
        let golden = TestGolden::new()
            .steps(li_expected())
            // David's blanket-10% ruling (2026-09-02): a base case REFUSES only when the mismatches
            // break `COHORT_TOKEN_TOLERANCE_PER_THOUSAND`, so the corruption must pass the budget;
            // the FIRST mismatch stays at index 3.
            .corrupt_expected_beyond_budget(3, 999)
            .fixture();
        let baseline_prefill = TEST_BASELINE.prefill_seconds_per_token;
        let baseline_decode = TEST_BASELINE.decode_seconds_per_token;
        // Correctness on the shared session (engine returns 2 everywhere → mismatch at index 3).
        // The timed phases spawn FRESH engines whose oracle is the golden's BENCHMARK block
        // (Option A); the TIME-ONLY retain-timing path tolerates any divergence and records real
        // timing regardless.
        let (mut session, _) =
            Session::connect(window_engine(&golden, vec![2i64; li_expected()])).unwrap();
        let spawned = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let spawned_c = spawned.clone();
        let (bp, bs, bd) = bench_oracle(&golden);
        let payload = iterate_flow_windowed(
            Some(&mut session),
            FlowParams {
                scoring: ScoringInputs::local(baseline_prefill, baseline_decode),
                ..test_flow(
                    &golden,
                    Mode::LocalIterate,
                    false,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                )
            },
            move || {
                spawned_c.set(spawned_c.get() + 1);
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    bp,
                    bs,
                    bd.clone(),
                ))
                .map(|(s, _)| s)
            },
            no_cool_gate,
        );

        // Correctness FAILED, so no score and the F1 fields are populated (unchanged behavior).
        assert!(!payload.passed);
        assert!(payload.score.is_none());
        let m = &payload.metrics;
        assert!(!m.passed_correctness);
        assert_eq!(m.first_failing_case.as_deref(), Some("local-iterate"));
        assert_eq!(
            m.first_failing_step,
            Some(4),
            "index 3 → Swift checked-step 4"
        );
        assert_eq!(m.expected_token, Some(999));
        assert_eq!(m.actual_token, Some(2));
        // Item-1 red-team Finding 3: the DET correctness counts follow Swift's LOCAL-ITERATE
        // FAILURE report — caseCount = timingRepeats, checkedSteps = failureStep + 1 — NOT the
        // standalone golden-total the blanked path used (which diverged from Swift).
        assert_eq!(
            m.case_count, TIMING_REPEATS,
            "Swift caseCount = timingRepeats on a local-iterate failure"
        );
        assert_eq!(
            m.checked_steps, 5,
            "Swift checkedSteps = first_failing_step(4) + 1"
        );

        // ITEM 1: the timing surface is REAL, not blanked — a blanked fail path measures nothing.
        // The time-only legs run on the window worker, so they open no second worker.
        assert!(m.decode_seconds_per_token > 0.0);
        assert!(m.prefill_seconds_per_token > 0.0);
        assert_eq!(
            spawned.get(),
            0,
            "the time-only legs run on the window worker, never a second one"
        );
        // Baselines are the real external Qwen constants (blanked path emits 0.0).
        assert!(m.baseline_decode_seconds_per_token > 0.0);
        assert!(m.baseline_prefill_seconds_per_token > 0.0);
        assert_eq!(m.baseline_decode_seconds_per_token, baseline_decode);
        assert_eq!(m.baseline_prefill_seconds_per_token, baseline_prefill);
        // Measured spt are finite (mock timing is tiny). speedup(baseline, 0.0) == 0, so a
        // NON-ZERO speedup + a TRUE floor together prove the spt is real and > 0 (not blanked).
        assert!(m.decode_seconds_per_token.is_finite());
        assert!(m.prefill_seconds_per_token.is_finite());
        assert!(m.decode_speedup > 0.0, "real decode speedup, not blanked 0");
        assert!(
            m.prefill_speedup > 0.0,
            "real prefill speedup, not blanked 0"
        );
        assert!(
            m.passed_decode_speedup_floor,
            "floor reflects the real (huge) mock speedup — proves decode_spt > 0"
        );
        assert!(
            m.passed_prefill_speedup_floor,
            "floor reflects the real (huge) mock speedup — proves prefill_spt > 0"
        );
    }

    #[test]
    fn local_iterate_primary_failure_reports_swift_checked_step() {
        // §F1 regression (deterministic; replaces the integration window). A primary
        // teacher-forced mismatch at expected_tokens[3] is reported EXACTLY as Swift
        // `--local-iterate` — case="local-iterate", checked-step 4 (prefill 0 / seed 1 /
        // decode k → k+2; index 3 = decode k=2 → 4), expected=<golden>, actual=<engine>,
        // error="local-iterate teacher-forced token mismatch". Target values captured from
        // the live failure-map window (Swift step 4, expected/actual populated).
        // golden expects 999 at index 3; the conformant engine returns 2.
        let golden = TestGolden::new()
            .steps(li_expected())
            // David's blanket-10% ruling (2026-09-02): a base case REFUSES only when the mismatches
            // break `COHORT_TOKEN_TOLERANCE_PER_THOUSAND`, so the corruption must pass the budget;
            // the FIRST mismatch stays at index 3.
            .corrupt_expected_beyond_budget(3, 999)
            .fixture();
        // Engine returns 2 for every teacher-forced step → matches golden at 0/1/2, mismatches
        // at index 3 (golden 999 vs engine 2). Correctness fails before timing, so the timed
        // spawner is never invoked.
        let (mut session, _) =
            Session::connect(window_engine(&golden, vec![2i64; li_expected()])).unwrap();
        let payload = iterate_flow_windowed(
            Some(&mut session),
            test_flow(
                &golden,
                Mode::LocalIterate,
                false,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            || Session::connect(MockEngine::new()).map(|(s, _)| s),
            no_cool_gate,
        );
        assert!(!payload.passed);
        let m = &payload.metrics;
        assert_eq!(m.first_failing_case.as_deref(), Some("local-iterate"));
        assert_eq!(
            m.first_failing_step,
            Some(4),
            "index 3 → Swift checked-step 4"
        );
        assert_eq!(m.expected_token, Some(999), "golden's expected token");
        assert_eq!(m.actual_token, Some(2), "engine's actual token");
        assert_eq!(m.error, "local-iterate teacher-forced token mismatch");
    }

    #[test]
    fn base_case_mismatches_are_always_inside_the_local_checked_window() {
        // §F1 window bound, restated at the REFERENCE window. The arm guard is
        // `f.step <= mode.decode_steps(&test_window())`: a base-case mismatch BEYOND the checked window is the
        // benchd SUPERSET and must not be branded "local-iterate" with an impossible step.
        //
        // At the reference window that guard can no longer fire for a BASE case: the conformance
        // gate only ever reports indices `< CORRECTNESS_STEPS` (64), and both local modes decode
        // at least that far (local-iterate 128, local-submit 1023). So every base-case index the
        // gate can produce is inside the window and takes the F1 branding arm. The guard stays as
        // a fail-closed bound on any future shorter window; this test pins WHY it is quiet.
        //
        // (It used to fire because benchd held the retired Laguna fork's 16-step window, where
        // index 40 was out of range. The superset arm is still exercised end-to-end by the
        // anchor/free-run `--strict` tests, which are `is_base_case = false`.)
        for mode in [Mode::LocalIterate, Mode::LocalSubmit] {
            assert!(
                mode.decode_steps(&test_window()) >= bench_core::constants::CORRECTNESS_STEPS,
                "{mode:?} decode window {} must cover every index the correctness gate reports \
                 (< {})",
                mode.decode_steps(&test_window()),
                bench_core::constants::CORRECTNESS_STEPS
            );
        }
        // The mapping itself is unchanged and still exact at the window edge.
        assert_eq!(swift_local_iterate_checked_step(0), 0, "prefill");
        assert_eq!(swift_local_iterate_checked_step(1), 2, "decode step 0");
        assert_eq!(
            swift_local_iterate_checked_step(li_steps()),
            li_steps() as i64 + 1,
            "last in-window decode step"
        );
    }

    #[test]
    fn local_iterate_golden_arity_matches_the_reference_loader() {
        // #109 window-4 E2 ROOT CAUSE. Swift `QwenRuntime.localIterate` loads the golden with
        // `requiredSteps: benchmarkDecodeSteps + 1` — `expected_tokens[0]` is the seed and
        // `expected_tokens[k + 1]` is decode step `k`'s output — so a golden carrying exactly
        // `decode_steps` tokens is one SHORT and the reference refuses it at LOAD time:
        //
        //   "primary-1.expected_tokens has 128 tokens; need at least 129"
        //
        // benchd loaded every mode at the flat `CORRECTNESS_STEPS` (64) and timed local-iterate
        // over a 16-step window, so it ACCEPTED that golden and reported a token mismatch from a
        // stream the reference never validated at all. Pin both halves of the contract.
        assert_eq!(
            Mode::LocalIterate.golden_required_steps(&test_window()),
            Mode::LocalIterate.decode_steps(&test_window()) + 1
        );
        assert_eq!(
            Mode::LocalSubmit.golden_required_steps(&test_window()),
            Mode::LocalSubmit.decode_steps(&test_window()) + 1
        );
        assert_eq!(
            Mode::Official.golden_required_steps(&test_window()),
            bench_core::constants::CORRECTNESS_STEPS,
            "official takes the reference loader DEFAULT"
        );

        // The window-4 golden's shape: a well-formed document whose primary case supplies
        // exactly `decode_steps` expected tokens. Built through the shared builder so the rest
        // of the document is the canonical shape and only the arity under test differs.
        let one_short = Mode::LocalIterate.decode_steps(&test_window());
        let bytes = TestGolden::new()
            .case_name("primary-1")
            .expected_tokens(vec![2i64; one_short])
            .bytes();

        // It still loads for the CORRECTNESS consumer (that one needs only 64) ...
        assert!(load_golden_fixture(
            &bytes,
            bench_core::constants::CORRECTNESS_STEPS,
            CORRECTNESS_PROMPT_TOKENS,
            crate::testgolden::identity_125b(),
            Some(REQUIRED_GOLDEN_MODEL_TYPE),
            None,
            None,
        )
        .is_ok());

        // ... and is refused, verbatim as the reference refuses it, for local-iterate.
        let err = load_golden_fixture(
            &bytes,
            Mode::LocalIterate.golden_required_steps(&test_window()),
            CORRECTNESS_PROMPT_TOKENS,
            crate::testgolden::identity_125b(),
            Some(REQUIRED_GOLDEN_MODEL_TYPE),
            None,
            None,
        )
        .expect_err("a decode_steps-long expected_tokens is one token short of the window");
        assert!(
            err.to_string().contains(&format!(
                "primary-1.expected_tokens has {one_short} tokens; need at least {}",
                Mode::LocalIterate.golden_required_steps(&test_window())
            )),
            "reference message shape, got: {err}"
        );
    }

    #[test]
    fn local_iterate_times_benchmark_when_present_else_cases0_stream() {
        // A8/DAVID RULING — SCOPED SUPERSESSION of RULING 1. The local-iterate timed leg now has
        // TWO branches, pinned here by making the benchmark oracle workload DISTINCT from cases[0]'s
        // (benchmark prompt = 22s; cases[0].prompt = 11s), so "which one did we time?" is observable:
        //
        //   * benchmark PRESENT → time the BENCHMARK oracle workload (prefill prompt + decode seed
        //     from golden.benchmark), so the captured decode-s/tok is measured over the workload
        //     official scores. SUPERSEDES RULING 1 for benchmark-carrying goldens.
        //   * benchmark ABSENT  → time cases[0]'s prompt/seed exactly as Swift --local-iterate.
        //     RULING 1, PRESERVED for participant goldens (benchmark: None).
        //
        // In BOTH branches the free-run timed leg issues NO teacher-forced decode_step calls (that
        // path is only the untimed correctness gate) and free_decode_runs the mode's window (li_steps()).
        let expected: Vec<i64> = (0..li_expected() as i64).map(|i| 3000 + i).collect();

        // --- Branch 1: benchmark PRESENT → time the benchmark oracle, NOT cases[0]. ---
        {
            let golden = TestGolden::new()
                .required_steps(li_expected())
                .prompt_fill(11)
                .expected_tokens(expected.clone())
                .benchmark(benchmark_oracle(22, 9))
                .fixture();
            let rec = std::rc::Rc::new(std::cell::RefCell::new(
                bench_runner::mock::RecordedTiming::default(),
            ));
            // ONE window worker: the correctness gate's teacher-forced cases[0] stream
            // (= expected[..]) AND the benchmark oracle the timed leg verifies, recording the
            // workload it was actually driven with.
            let (mut session, _) = Session::connect(
                window_engine(&golden, expected.clone()).record_timing(rec.clone()),
            )
            .unwrap();
            let payload = iterate_flow_windowed(
                Some(&mut session),
                test_flow(
                    &golden,
                    Mode::LocalIterate,
                    false,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                ),
                unused_spawner,
                no_cool_gate,
            );
            assert!(
                payload.passed,
                "conformant benchmark-oracle run must pass; error={}",
                payload.metrics.error
            );
            let r = rec.borrow();
            let benchmark = golden.benchmark.as_ref().unwrap();
            // The TIMED workload is the BENCHMARK oracle's (prefill prompt + decode seed), NOT cases[0]'s.
            assert_eq!(r.prefill_prompt, benchmark.prefill_prompt_tokens);
            assert_eq!(r.decode_seed, benchmark.decode_seed_tokens);
            assert_ne!(
                r.prefill_prompt, golden.cases[0].prompt_tokens,
                "benchmark present ⇒ must NOT time cases[0]'s prompt"
            );
            assert!(
                r.decode_step_inputs.is_empty(),
                "the free-run timed leg issues no teacher-forced decode_step calls"
            );
            assert_eq!(
                r.free_decode_run_count,
                Some(li_steps() as u32),
                "the free-run timed leg must free_decode_run the mode's window (li_steps())"
            );
        }

        // --- Branch 2: benchmark ABSENT (participant golden) → RULING 1 preserved, time cases[0]. ---
        {
            let golden = TestGolden::new()
                .required_steps(li_expected())
                .prompt_fill(11)
                .expected_tokens(expected.clone())
                .without_benchmark()
                .fixture();
            assert!(
                golden.benchmark.is_none(),
                "branch-2 golden is benchmark-less (the RULING-1 fallback)"
            );
            let rec = std::rc::Rc::new(std::cell::RefCell::new(
                bench_runner::mock::RecordedTiming::default(),
            ));
            // cases[0]'s stream: seed = expected[0], window = expected[1..=li_steps()].
            let seed = expected[0];
            let window = expected[1..=li_steps()].to_vec();
            let (mut session, _) = Session::connect(
                MockEngine::new()
                    .teacher_forced_tokens(expected.clone())
                    .free_run_capable()
                    .oracle_tokens(seed, seed, window.clone())
                    .record_timing(rec.clone()),
            )
            .unwrap();
            let payload = iterate_flow_windowed(
                Some(&mut session),
                test_flow(
                    &golden,
                    Mode::LocalIterate,
                    false,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                ),
                unused_spawner,
                no_cool_gate,
            );
            assert!(
                payload.passed,
                "conformant cases[0] run must pass; error={}",
                payload.metrics.error
            );
            let r = rec.borrow();
            // RULING 1 preserved: the TIMED workload is cases[0]'s prompt/seed.
            assert_eq!(r.prefill_prompt, golden.cases[0].prompt_tokens);
            assert_eq!(r.decode_seed, golden.cases[0].prompt_tokens);
            assert!(r.decode_step_inputs.is_empty());
            assert_eq!(r.free_decode_run_count, Some(li_steps() as u32));
        }
    }

    #[test]
    fn iso8601_known_epoch() {
        // 1970-01-01T00:00:00Z
        assert_eq!(format_iso8601(0), "1970-01-01T00:00:00Z");
        // 2026-08-15T00:00:00Z == 1_786_752_000 (verified against a civil calendar)
        assert_eq!(format_iso8601(1_786_752_000), "2026-08-15T00:00:00Z");
        // 2026-08-15T12:34:56Z
        assert_eq!(
            format_iso8601(1_786_752_000 + 45_296),
            "2026-08-15T12:34:56Z"
        );
    }

    #[test]
    fn dir_digest_counts_and_ignores() {
        let tmp = std::env::temp_dir().join(format!("benchd-digest-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join("a.bin"), b"hello").unwrap();
        fs::write(tmp.join(".gitkeep"), b"").unwrap();
        let d = dir_digest(&tmp).unwrap();
        assert_eq!(d.file_count, 1); // .gitkeep ignored
        assert_eq!(d.byte_count, 5);
        assert_eq!(d.sha256.len(), 64);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn dir_digest_matches_swift_tree_formula() {
        // P2: the tree hash is Swift `directoryDigest`'s exactly — over files sorted by
        // relative path (ignoring exact `.gitkeep` / `.benchmark-source.sha256`), fold
        // `relpath || 0x00 || sha256(file bytes) || 0x00`, then hex(sha256(stream)).
        let tmp = std::env::temp_dir().join(format!("benchd-digest-fmt-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("sub")).unwrap();
        fs::write(tmp.join("a.bin"), b"hello").unwrap();
        fs::write(tmp.join("sub/b.bin"), b"world").unwrap();
        fs::write(tmp.join(".gitkeep"), b"ignored").unwrap();
        fs::write(tmp.join(".benchmark-source.sha256"), b"ignored").unwrap();

        let d = dir_digest(&tmp).unwrap();
        assert_eq!(
            d.file_count, 2,
            ".gitkeep + .benchmark-source.sha256 ignored by exact rel"
        );
        assert_eq!(d.byte_count, 10);

        // Independently hand-compute the Swift tree hash.
        let sha = |b: &[u8]| {
            let mut h = Sha256::new();
            h.update(b);
            h.finalize()
        };
        let mut tree = Sha256::new();
        // Byte-lexicographic sort: "a.bin" < "sub/b.bin".
        for (rel, content) in [("a.bin", &b"hello"[..]), ("sub/b.bin", &b"world"[..])] {
            tree.update(rel.as_bytes());
            tree.update([0u8]);
            tree.update(sha(content));
            tree.update([0u8]);
        }
        let expected: String = tree.finalize().iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            d.sha256, expected,
            "tree hash must match the Swift directoryDigest formula"
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    /// Build a tree of 20 files with varied sizes — zero-length, sub-chunk, exactly the
    /// 8 MiB streaming chunk, and larger than it — plus a nested subdir, and return the
    /// `(relpath, bytes)` pairs sorted by relative path.
    fn build_varied_digest_tree(tmp: &Path) -> Vec<(String, Vec<u8>)> {
        const CHUNK: usize = 8 * 1024 * 1024;
        let sizes = [
            0,
            1,
            7,
            64,
            4096,
            65_536,
            CHUNK - 1,
            CHUNK,
            CHUNK + 1,
            CHUNK + 12_345,
            3,
            0,
            1024,
            999,
            131_072,
            17,
            2,
            8191,
            33,
            5,
        ];
        fs::create_dir_all(tmp.join("sub/deeper")).unwrap();
        let mut expected: Vec<(String, Vec<u8>)> = Vec::new();
        for (i, size) in sizes.iter().enumerate() {
            let rel = match i % 3 {
                0 => format!("f{i:02}.bin"),
                1 => format!("sub/f{i:02}.bin"),
                _ => format!("sub/deeper/f{i:02}.bin"),
            };
            let content: Vec<u8> = (0..*size).map(|b| ((b + i) % 251) as u8).collect();
            fs::write(tmp.join(&rel), &content).unwrap();
            expected.push((rel, content));
        }
        expected.sort_by(|a, b| a.0.cmp(&b.0));
        expected
    }

    /// The serial reference: fold `relpath || 0x00 || sha256(whole file) || 0x00` over the
    /// sorted pairs, one file at a time, exactly as the pre-parallel loop did.
    fn serial_reference_digest(files: &[(String, Vec<u8>)]) -> String {
        let mut tree = Sha256::new();
        for (rel, content) in files {
            let mut file = Sha256::new();
            file.update(content);
            tree.update(rel.as_bytes());
            tree.update([0u8]);
            tree.update(file.finalize());
            tree.update([0u8]);
        }
        tree.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn dir_digest_is_the_serial_fold_at_every_worker_count() {
        // The parallel per-file hash is digest-neutral: same per-file SHA256s, same sorted
        // fold order, at every worker count including one above the file count (the clamp).
        // One tree with zero-length and >8 MiB (multi-chunk) files serves every assertion.
        let tmp = std::env::temp_dir().join(format!("benchd-digest-fold-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let expected_files = build_varied_digest_tree(&tmp);
        let expected = serial_reference_digest(&expected_files);
        let expected_bytes = expected_files
            .iter()
            .map(|(_, c)| c.len() as i64)
            .sum::<i64>();

        for workers in [1usize, 2, 8, 64] {
            let d = dir_digest_with_workers(&tmp, Some(workers)).unwrap();
            assert_eq!(
                d.sha256, expected,
                "parallel hashing must reproduce the serial fold byte for byte at {workers} worker(s)"
            );
            assert_eq!(d.file_count, expected_files.len() as i64);
            assert_eq!(d.byte_count, expected_bytes);
        }
        // The default (available_parallelism-derived) count agrees too.
        let d = dir_digest(&tmp).unwrap();
        assert_eq!(d.sha256, expected);
        assert_eq!(d.file_count, expected_files.len() as i64);
        assert_eq!(d.byte_count, expected_bytes);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn dir_digest_matches_an_independently_computed_vector_across_the_stream_chunk() {
        // The sha2 crate is built with the `asm` feature, so on aarch64 the compressor is the
        // CPU's SHA-256 instructions. `weights_hash` is sealed into score.json and compared
        // across runs, so the value may not move with the backend. The reference here is
        // therefore NOT recomputed with sha2: it is a constant produced by a separate SHA-256
        // implementation (Python `hashlib`) over this exact tree.
        //
        // The tree crosses the 8 MiB streaming chunk on purpose (one file is 8 MiB + 12,345
        // bytes), because a wrong compressor is invisible in a single-block file.
        const CHUNK: usize = 8 * 1024 * 1024;
        const EXPECTED: &str = "63d9d596bc0521f767895750354fa77f3e8cef4a68cee61df80610eec837a521";

        let tmp = std::env::temp_dir().join(format!("benchd-digest-vec-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("sub")).unwrap();
        let big: Vec<u8> = (0..CHUNK + 12_345).map(|i| (i % 251) as u8).collect();
        fs::write(tmp.join("a.bin"), &big).unwrap();
        fs::write(tmp.join("c.bin"), b"").unwrap();
        fs::write(tmp.join("sub/b.bin"), b"world").unwrap();

        let d = dir_digest(&tmp).unwrap();
        assert_eq!(d.file_count, 3);
        assert_eq!(d.byte_count, (CHUNK + 12_345 + 5) as i64);
        assert_eq!(
            d.sha256, EXPECTED,
            "the weights digest must equal the independently computed vector"
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn correctness_failure_payload_seals_as_writable_failing_json() {
        // M-3: the OTHER failing class (a correctness mismatch, not preflight). iterate_core
        // returns a failed payload (score=None, passed=false) that must serialize to the sealed
        // score.json the common writers seal — proving the failing payload reaching
        // write_score/write_integrity_sidecar is well-formed. The writers do NOT branch on
        // failure class, so this + the preflight case in main.rs cover both.
        // corrupt cases[0] so the base-case gate fails
        let golden = TestGolden::new()
            .steps(li_expected())
            // David's blanket-10% ruling (2026-09-02): a base case REFUSES only when the mismatches
            // break `COHORT_TOKEN_TOLERANCE_PER_THOUSAND`, so the corruption must pass the budget;
            // the FIRST mismatch stays at index 3.
            .corrupt_expected_beyond_budget(3, 999)
            .fixture();
        let payload = run_r3(&golden, false);
        assert!(
            !payload.passed,
            "a corrupted primary case fails correctness"
        );
        assert!(payload.score.is_none(), "a failing run carries score=null");
        assert!(!payload.metrics.passed_correctness);

        // The failing payload seals to the byte-shaped JSON the writers persist as score.json.
        let json = payload
            .to_sealed_json()
            .expect("failing payload must serialize");
        assert!(json.contains("\"passed\": false"));
        assert!(json.contains("\"score\": null"));
        // The golden_hash field is still populated on failure (a real audit surface).
        assert_eq!(payload.metrics.golden_hash, golden.sha256);
    }

    /// #62: the §F2 missing-baseline refusal, asserted IN FULL against a fixture captured
    /// from the Swift source — for every mode, since Swift interpolates the mode name into
    /// the message. The old assertion was `starts_with("local-iterate requires external
    /// Qwen")`, which left the whole tail ("...both prefill and decode; Gemma baseline
    /// constants are not valid for Qwen") free to drift unnoticed, and compared benchd's
    /// output against a Rust literal restating it rather than against Swift.
    #[test]
    fn missing_paired_baselines_error_matches_swift_capture() {
        const CAPTURE: &str = include_str!("../tests/fixtures/swift-missing-paired-baselines.json");
        let capture: serde_json::Value = serde_json::from_str(CAPTURE).unwrap();
        let expected = &capture["messages"];

        for mode in [Mode::LocalIterate, Mode::LocalSubmit, Mode::Official] {
            let want = expected[mode.mode_name()]
                .as_str()
                .unwrap_or_else(|| panic!("Swift capture has no message for {}", mode.mode_name()));
            assert_eq!(
                missing_paired_baselines_error(mode),
                want,
                "benchd's missing-baseline refusal for {} diverged from the Swift capture \
                 ({}:{} @ {})",
                mode.mode_name(),
                capture["source_file"].as_str().unwrap(),
                capture["source_lines"].as_str().unwrap(),
                capture["source_commit"].as_str().unwrap(),
            );
        }

        // The capture must cover every mode that can reach the refusal — a mode added later
        // must not silently go unpinned.
        assert_eq!(
            expected.as_object().unwrap().len(),
            3,
            "Swift capture covers a different set of modes than benchd has"
        );
    }

    // --- M-6: local-submit -------------------------------------------------

    #[test]
    fn local_submit_mode_properties() {
        // The Mode plumbing: 1023-step decode window, `rust-local-submit` runtime label,
        // `local-submit` modeName, and the P6 cool-gate RULING (submit ON by default).
        assert_eq!(Mode::LocalSubmit.decode_steps(&test_window()), 1023);
        assert_eq!(Mode::LocalSubmit.runtime(), "rust-local-submit");
        assert_eq!(Mode::LocalSubmit.mode_name(), "local-submit");
        assert_eq!(Mode::parse("local-submit"), Some(Mode::LocalSubmit));
        assert!(
            Mode::LocalSubmit.cool_gate_on_by_default(),
            "P6 RULING: submit ON by default"
        );
        // Contrast: iterate OFF, official OFF; both local modes brand failures Swift-style.
        assert!(!Mode::LocalIterate.cool_gate_on_by_default());
        assert!(Mode::Official.cool_gate_on_by_default());
        assert!(Mode::LocalSubmit.is_local_checked_timing());
        assert!(Mode::LocalIterate.is_local_checked_timing());
        assert!(!Mode::Official.is_local_checked_timing());
    }

    /// A golden whose primary `cases[0]` carries enough expected_tokens to time a full
    /// 1023-step local-submit decode window (needs len > 1023), AND whose `benchmark` oracle is at
    /// least that long. Option A (a8/David) times the benchmark oracle for local-submit too, so the
    /// benchmark decode window must cover the 1023 steps — the default 128-step oracle would be one
    /// the timing guard refuses. The oracle decode fill is `7` (distinct from cases[0]'s `2`), so the
    /// conformant timed mock returns `[7; 1023]` (from `bench_oracle`), while the teacher-forced
    /// correctness gate still sees cases[0]'s `2`s.
    fn submit_golden(expected: Vec<i64>) -> GoldenFixture {
        // `required_steps` (not `steps`) — the caller supplies its own long window, but the
        // fixture is still validated against the local checked-decode arity.
        TestGolden::new()
            .required_steps(li_expected())
            .expected_tokens(expected)
            .benchmark(benchmark_oracle_windowed(1, 7, 1024))
            .fixture()
    }

    #[test]
    fn local_submit_scores_a_full_1023_step_window() {
        // M-6: local-submit reuses the local-iterate checked-timing machinery over a 1023-step
        // decode (Swift `QwenRuntime.localIterate` with decodeSteps=1023). Option A: it times the
        // golden's BENCHMARK oracle (submit_golden gives it a 1024-token oracle so the 1023 window
        // fits). A conformant run scores, labels `runtime=rust-local-submit`, and reports the Swift
        // checked-timing counts: case_count = timingRepeats = 1, checked_steps = (1023+2)*1.
        let golden = submit_golden(vec![2i64; 1025]); // cases[0] > 1023 for the correctness gate
        let (mut session, _) = Session::connect(window_engine(&golden, vec![2i64; 1025])).unwrap();
        let (bp, bs, bd) = bench_oracle(&golden);
        let payload = iterate_flow_windowed(
            Some(&mut session),
            test_flow(
                &golden,
                Mode::LocalSubmit,
                false,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            move || {
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    bp,
                    bs,
                    bd.clone(),
                ))
                .map(|(s, _)| s)
            },
            no_cool_gate,
        );
        assert!(
            payload.passed,
            "conformant local-submit run scores; error={}",
            payload.metrics.error
        );
        assert!(payload.score.is_some());
        assert_eq!(payload.metrics.runtime, "rust-local-submit");
        assert_eq!(payload.metrics.case_count, 1, "timingRepeats");
        assert_eq!(
            payload.metrics.checked_steps,
            (1023 + 2) * TIMING_REPEATS,
            "(decodeSteps + 2) * timingRepeats"
        );
    }

    #[test]
    fn local_submit_primary_failure_is_branded_local_submit() {
        // §F1 mode-derived (M-6): a corrupted primary cases[0] under local-submit is branded
        // with the modeName — case="local-submit", error="local-submit teacher-forced token
        // mismatch", Swift checked-step numbering — NOT hard-coded "local-iterate". A short
        // (64-token) case can't time a 1023-step window, so it takes the blanked failed
        // payload, but the mode-derived branding survives on that path too.
        let mut expected = vec![2i64; li_expected()];
        // David's blanket-10% ruling (2026-09-02): the base case now REFUSES only when the
        // mismatches break `COHORT_TOKEN_TOLERANCE_PER_THOUSAND`, so a correctness-FAILURE test
        // must corrupt past the budget. The FIRST mismatch stays at the index asserted below.
        // Golden expects 999 at index 3 and beyond; the conformant engine returns 2.
        corrupt_beyond_budget(&mut expected, 3, 999);
        let golden = submit_golden(expected);
        let (mut session, _) =
            Session::connect(window_engine(&golden, vec![2i64; li_expected()])).unwrap();
        let payload = iterate_flow_windowed(
            Some(&mut session),
            test_flow(
                &golden,
                Mode::LocalSubmit,
                false,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            || Session::connect(MockEngine::new()).map(|(s, _)| s),
            no_cool_gate,
        );
        assert!(!payload.passed);
        let m = &payload.metrics;
        assert_eq!(m.first_failing_case.as_deref(), Some("local-submit"));
        assert_eq!(
            m.first_failing_step,
            Some(4),
            "index 3 → Swift checked-step 4"
        );
        assert_eq!(m.expected_token, Some(999));
        assert_eq!(m.actual_token, Some(2));
        assert_eq!(m.error, "local-submit teacher-forced token mismatch");
    }

    #[test]
    fn local_submit_correctness_failure_retains_real_timing() {
        // M-6 coverage gap (red-team): local-submit INHERITS the local-iterate
        // correctness-failure retain-timing path (`mode.is_local_checked_timing()` is true for
        // both), but that path was only exercised under local-iterate. This is the submit twin
        // of `local_iterate_correctness_failure_retains_real_timing`: a CORRECTNESS FAILURE on a
        // golden with a FULL 1023-step window must RETAIN real timing/baselines/speedup-floor
        // flags (Swift teacher-forces + times regardless of correctness) — NOT blank them.
        //
        // cases[0] carries > 1023 expected_tokens (so a full window CAN be timed → retain-timing
        // path, not the too-short blank path) with the corrupt token at index 3, WITHIN the
        // 1023-step checked window (index ≤ 1023) so it brands "local-submit". The timed engines
        // are conformant to cases[0]'s stream (seed=2, window [2; 1023]); under TIME-ONLY they'd
        // time even if they diverged.
        // David's blanket-10% ruling (2026-09-02): the base case now REFUSES only when the
        // mismatches break `COHORT_TOKEN_TOLERANCE_PER_THOUSAND`, so a correctness-FAILURE test
        // must corrupt past the budget. The FIRST mismatch stays at the index asserted below.
        // Golden expects 999 at index 3 and beyond; the conformant engine returns 2.
        let mut expected = vec![2i64; 1025]; // > 1023 → full window available
        corrupt_beyond_budget(&mut expected, 3, 999);
        let golden = submit_golden(expected);
        let baseline_prefill = TEST_BASELINE.prefill_seconds_per_token;
        let baseline_decode = TEST_BASELINE.decode_seconds_per_token;
        // Correctness on the shared session (engine returns 2 everywhere → mismatch at index 3).
        // The timed phases spawn FRESH engines whose oracle is the golden's BENCHMARK block
        // (Option A); the TIME-ONLY retain-timing path tolerates any divergence.
        let (mut session, _) = Session::connect(window_engine(&golden, vec![2i64; 1025])).unwrap();
        let spawned = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let spawned_c = spawned.clone();
        let (bp, bs, bd) = bench_oracle(&golden);
        let payload = iterate_flow_windowed(
            Some(&mut session),
            FlowParams {
                scoring: ScoringInputs::local(baseline_prefill, baseline_decode),
                ..test_flow(
                    &golden,
                    Mode::LocalSubmit,
                    false,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                )
            },
            move || {
                spawned_c.set(spawned_c.get() + 1);
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    bp,
                    bs,
                    bd.clone(),
                ))
                .map(|(s, _)| s)
            },
            no_cool_gate,
        );

        // Correctness FAILED, so no score and the F1 fields are populated, branded "local-submit".
        assert!(!payload.passed);
        assert!(payload.score.is_none());
        let m = &payload.metrics;
        assert!(!m.passed_correctness);
        assert_eq!(m.first_failing_case.as_deref(), Some("local-submit"));
        assert_eq!(
            m.first_failing_step,
            Some(4),
            "index 3 → Swift checked-step 4"
        );
        assert_eq!(m.expected_token, Some(999));
        assert_eq!(m.actual_token, Some(2));
        assert_eq!(m.error, "local-submit teacher-forced token mismatch");
        // DET correctness counts follow Swift's LOCAL failure report: caseCount = timingRepeats,
        // checkedSteps = failureStep + 1 — NOT the blanked golden-total.
        assert_eq!(
            m.case_count, TIMING_REPEATS,
            "Swift caseCount = timingRepeats on a local-submit failure"
        );
        assert_eq!(
            m.checked_steps, 5,
            "Swift checkedSteps = first_failing_step(4) + 1"
        );

        // The timing surface is REAL, not blanked — a blanked fail path measures nothing. The
        // time-only legs run on the window worker, so they open no second worker.
        assert!(m.decode_seconds_per_token > 0.0);
        assert!(m.prefill_seconds_per_token > 0.0);
        assert_eq!(
            spawned.get(),
            0,
            "the time-only legs run on the window worker, never a second one"
        );
        // Baselines are the real external Qwen constants (blanked path emits 0.0).
        assert!(m.baseline_decode_seconds_per_token > 0.0);
        assert!(m.baseline_prefill_seconds_per_token > 0.0);
        assert_eq!(m.baseline_decode_seconds_per_token, baseline_decode);
        assert_eq!(m.baseline_prefill_seconds_per_token, baseline_prefill);
        // A NON-ZERO speedup + a TRUE floor together prove the spt is real and > 0 (not blanked).
        assert!(m.decode_seconds_per_token.is_finite());
        assert!(m.prefill_seconds_per_token.is_finite());
        assert!(m.decode_speedup > 0.0, "real decode speedup, not blanked 0");
        assert!(
            m.prefill_speedup > 0.0,
            "real prefill speedup, not blanked 0"
        );
        assert!(
            m.passed_decode_speedup_floor,
            "floor reflects the real (huge) mock speedup — proves decode_spt > 0"
        );
        assert!(
            m.passed_prefill_speedup_floor,
            "floor reflects the real (huge) mock speedup — proves prefill_spt > 0"
        );
    }

    // ---- #74 failure-surface: the TWO adjacent failure classes -------------------------
    //
    // RULED (David 2026-08-20, via structured interview) — per-class byte fidelity:
    //
    //   PRIMARY (a run HAPPENED and its correctness failed) → RETAIN the real measurements.
    //     Ruled 2026-08-17 and already implemented (retain-timing #73 + the R3 default flip);
    //     the reference retains because `--local-iterate` TEACHER-FORCES the expected tokens
    //     and times the pass regardless, judging correctness separately
    //     (`QwenRuntimeLocalIterate.swift@b26f76f:164-174` returns through `failed(...)` with
    //     the measured decode/prefill spt in hand).
    //
    //   EARLY REFUSE (nothing ran) → seal the reference's empty record.
    //     `QwenRuntimeLocalIterate.swift@b26f76f:197-198` catches into `failed(...)` with
    //     `correctnessReport` still nil, and `failedScore` reads all three run-shaped fields
    //     off that nil (`QwenRuntimeBenchmark.swift@b26f76f:1161-1162,1176`).
    //
    // The two must not bleed into each other, so they are pinned together, adjacent, below.

    /// The reference's EARLY-REFUSE record, pinned BYTE FOR BYTE against the constructed
    /// capture in `tests/fixtures/swift-early-refuse-failure-record.json`.
    ///
    /// PROVENANCE (review of #217): the fixture is HAND-CONSTRUCTED for this track — `num_layers`
    /// 48 and the sentinel baseline fields are NOT a capture of any Swift source (its
    /// `source_commit` says UNVERIFIED); it pins benchd's record SHAPE until a re-capture lands.
    ///
    /// The additive local phase diagnostic is asserted separately, then the comparison covers
    /// the WHOLE legacy sealed document, not a field sample: the fixture's
    /// `nondeterministic_or_runner_identity` list is the only escape hatch (wall-clock fields
    /// that have no fixed value on either side, plus the `commit`/`harness_hash`/`runtime`
    /// labels graded as environmental / runner-identity fields), the test substitutes exactly those
    /// from the actual payload, and everything else must match to the byte.
    #[test]
    fn early_refuse_record_matches_the_hand_constructed_shape() {
        const CAPTURE: &str =
            include_str!("../tests/fixtures/swift-early-refuse-failure-record.json");
        let capture: serde_json::Value = serde_json::from_str(CAPTURE).unwrap();
        let inputs = &capture["test_inputs"];

        let weights = DirDigest {
            sha256: inputs["weights_hash"].as_str().unwrap().to_string(),
            file_count: inputs["weights_file_count"].as_i64().unwrap(),
            byte_count: inputs["weights_byte_count"].as_i64().unwrap(),
        };
        // A golden that loaded fine and carries a full case roster: the point of the pin is
        // that NEITHER its digest NOR its case count reaches the record, because the refusal
        // happened before anything consumed them.
        let golden = TestGolden::new().fixture();
        assert!(
            !golden.sha256.is_empty() && golden.total_correctness_case_count() > 0,
            "precondition: the golden has both a digest and cases to (wrongly) leak"
        );
        let actual = preflight_failed_payload(
            Mode::LocalIterate,
            &golden,
            RunDigests::for_test(&weights),
            inputs["error"].as_str().unwrap().to_string(),
            ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
        );
        let mut actual: serde_json::Value =
            serde_json::from_str(&actual.to_sealed_json().unwrap()).unwrap();
        assert_eq!(
            actual["metrics"]
                .as_object_mut()
                .unwrap()
                .remove("local_phases"),
            Some(serde_json::json!({
                "correctness": "not_run", "correctness_checked_steps": 0, "timing": "not_run"
            })),
        );

        let mut expected = capture["record"].clone();
        let substituted: Vec<&str> = capture["nondeterministic_or_runner_identity"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        for key in &substituted {
            let live = actual["metrics"][key].clone();
            assert!(
                !live.is_null(),
                "{key} is declared substitutable but absent from the sealed payload"
            );
            expected["metrics"][*key] = live;
        }
        // The reference's record carries its officialBaseline* CONSTANTS in these two fields
        // (#74). A track that pins no pair has no number for the capture to hold, so the capture
        // holds a PLACEHOLDER STRING there and the test feeds the stand-in pair it passed — the
        // record SHAPE stays pinned byte-for-byte while no number exists to pin. Keying on "the
        // captured value is a string, not a number" means the placeholder is not a second name
        // this test has to be kept in step with.
        for (key, passed) in [
            (
                "baseline_prefill_seconds_per_token",
                TEST_BASELINE.prefill_seconds_per_token,
            ),
            (
                "baseline_decode_seconds_per_token",
                TEST_BASELINE.decode_seconds_per_token,
            ),
        ] {
            let is_placeholder = expected["metrics"][key].is_string();
            if is_placeholder {
                expected["metrics"][key] = serde_json::json!(passed);
            }
        }

        // Serialised the same way the writer does (sorted-key pretty), so this is a
        // byte comparison of the document, not a structural one.
        let want = serde_json::to_string_pretty(&expected).unwrap();
        let got = serde_json::to_string_pretty(&actual).unwrap();
        assert_eq!(
            got.as_bytes(),
            want.as_bytes(),
            "the early-refuse record diverged from the reference capture \
             ({} @ {}); substituted fields were {substituted:?}",
            capture["source_files"][0].as_str().unwrap(),
            capture["source_commit"].as_str().unwrap(),
        );

        // Spelled out, because these three ARE the #74 divergence and a future edit to the
        // capture must not quietly relax them.
        assert_eq!(actual["metrics"]["golden_hash"], serde_json::json!(""));
        assert_eq!(actual["metrics"]["case_count"], serde_json::json!(0));
        assert_eq!(actual["metrics"]["checked_steps"], serde_json::json!(0));
    }

    /// The other direction of the same contract, deliberately adjacent: a run that DID happen
    /// and failed correctness keeps its measurements. Guards the #74 fix against regressing
    /// the 2026-08-17 retain-timing ruling — the two classes seal opposite things, and a change
    /// that blanks this one is as wrong as one that populates the other.
    #[test]
    fn correctness_failure_retains_what_early_refuse_seals_empty() {
        let golden = TestGolden::new()
            .steps(li_expected())
            // David's blanket-10% ruling (2026-09-02): a base case REFUSES only when the mismatches
            // break `COHORT_TOKEN_TOLERANCE_PER_THOUSAND`, so the corruption must pass the budget;
            // the FIRST mismatch stays at index 3.
            .corrupt_expected_beyond_budget(3, 999)
            .fixture();
        let (mut session, _) =
            Session::connect(window_engine(&golden, vec![2i64; li_expected()])).unwrap();
        let (bp, bs, bd) = bench_oracle(&golden);
        let ran = iterate_flow_windowed(
            Some(&mut session),
            test_flow(
                &golden,
                Mode::LocalIterate,
                false,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            move || {
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    bp,
                    bs,
                    bd.clone(),
                ))
                .map(|(s, _)| s)
            },
            no_cool_gate,
        );
        let refused = preflight_failed_payload(
            Mode::LocalIterate,
            &golden,
            RunDigests::for_test(&DirDigest::empty()),
            missing_paired_baselines_error(Mode::LocalIterate),
            ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
        );

        // Both are failures with no score — the verdict is the same; only the SEAL differs.
        assert!(!ran.passed && ran.score.is_none());
        assert!(!refused.passed && refused.score.is_none());

        // RAN: the golden was hashed, the counts describe the checked pass, the timing and
        // baselines are real.
        assert_eq!(ran.metrics.golden_hash, golden.sha256);
        assert_eq!(ran.metrics.case_count, TIMING_REPEATS);
        assert_eq!(ran.metrics.checked_steps, 5);
        assert!(ran.metrics.decode_seconds_per_token > 0.0);
        assert_eq!(
            ran.metrics.baseline_decode_seconds_per_token,
            TEST_BASELINE.decode_seconds_per_token
        );

        // REFUSED: nothing ran, so nothing about a run is claimed.
        assert_eq!(refused.metrics.golden_hash, "");
        assert_eq!(refused.metrics.case_count, 0);
        assert_eq!(refused.metrics.checked_steps, 0);
        assert_eq!(refused.metrics.decode_seconds_per_token, 0.0);
        // …except the baseline pair, which the reference's `failedScore` defaults to the
        // official constants on BOTH classes (`QwenRuntimeBenchmark.swift@b26f76f:1130-1131`).
        assert_eq!(
            refused.metrics.baseline_decode_seconds_per_token,
            TEST_BASELINE.decode_seconds_per_token
        );
    }

    // ---- #127: the local baseline pair's SOURCE OF TRUTH ------------------------------

    /// The pair the local legs score against, pinned against a capture of the reference source
    /// rather than against a Rust literal restating it.
    ///
    /// This is the pin that would have caught the split the moment the reference forked: the
    /// capture also records the RETIRED fork's pair, and the test asserts benchd is not still
    /// carrying it.
    #[test]
    fn local_mode_baselines_match_the_reference_constants_capture() {
        const CAPTURE: &str =
            include_str!("../tests/fixtures/swift-official-baseline-constants.json");
        let capture: serde_json::Value = serde_json::from_str(CAPTURE).unwrap();
        let want = &capture["reference"];
        let stale = &capture["retired_fork"];

        // STEP 3 (David 2026-09-15): the pair comes from the track fixture and nowhere else, so
        // the test drives the resolver with the fixture this tree ships for its own track — the
        // same file `tests/fixtures/contract-full/qwen3.8-27b-mtp-v1.json` the oracle pins.
        let fixture = bench_core::contract::Contract::parse(include_bytes!(
            "../tests/fixtures/contract-full/qwen3.8-27b-mtp-v1.json"
        ))
        .unwrap();
        let resolved = local_mode_baselines(&fixture, bench_core::constants::TRACK_ID).unwrap();
        let (prefill, decode) = (
            resolved.prefill_seconds_per_token,
            resolved.decode_seconds_per_token,
        );
        assert_eq!(
            prefill,
            want["officialBaselinePrefillSecondsPerToken"]
                .as_f64()
                .unwrap(),
            "local prefill baseline diverged from {}:{} @ {}",
            capture["source_file"].as_str().unwrap(),
            capture["source_lines"].as_str().unwrap(),
            capture["source_commit"].as_str().unwrap(),
        );
        assert_eq!(
            decode,
            want["officialBaselineDecodeSecondsPerToken"]
                .as_f64()
                .unwrap(),
            "local decode baseline diverged from {}:{} @ {}",
            capture["source_file"].as_str().unwrap(),
            capture["source_lines"].as_str().unwrap(),
            capture["source_commit"].as_str().unwrap(),
        );
        // The specific wrong answer this issue is about.
        assert_ne!(
            prefill,
            stale["officialBaselinePrefillSecondsPerToken"]
                .as_f64()
                .unwrap(),
            "benchd is scoring against the RETIRED fork's prefill baseline"
        );
        assert_ne!(
            decode,
            stale["officialBaselineDecodeSecondsPerToken"]
                .as_f64()
                .unwrap(),
            "benchd is scoring against the RETIRED fork's decode baseline"
        );
    }

    /// A golden's DECLARED baseline pair is inert on the local legs: whatever it says, the run
    /// scores against the pair it was given, and the two are not required to agree.
    ///
    /// #127 INERTNESS (restored under the pair seam, review of #217): a golden's declared pair
    /// never reaches a local run's record — the run seals the pair it was given (the official
    /// constants once captured), not the golden's.
    #[test]
    fn a_goldens_declared_baseline_pair_does_not_reach_a_local_run() {
        // A pair the golden declares that is NOT the official constants (any such pair; the
        // retired fork's values are gone from the fixture mirror).
        let stale_prefill = 0.5_f64;
        let stale_decode = 0.25_f64;

        let golden = TestGolden::new()
            .steps(li_expected())
            .baselines(stale_prefill, stale_decode)
            .fixture();
        let declared = golden.benchmark.as_ref().expect("fixture has an oracle");
        assert_eq!(
            declared.baseline_prefill_seconds_per_token,
            Some(stale_prefill),
            "precondition: the golden really does declare the stale pair"
        );

        // What the local leg scores against — unchanged by the declaration above.
        // The seam: the pair a local run scores against is a PARAMETER of iterate_core — the
        // official constants once captured (`Platform::official_baseline`), a stand-in here.
        let prefill = TEST_BASELINE.prefill_seconds_per_token;
        let decode = TEST_BASELINE.decode_seconds_per_token;
        assert_ne!(prefill, stale_prefill);
        assert_ne!(decode, stale_decode);

        // And the run really does use it: a full local pass over this golden reports the
        // constants as its denominators, never the golden's declaration.
        let (mut session, _) =
            Session::connect(window_engine(&golden, vec![2i64; li_expected()])).unwrap();
        let (bp, bs, bd) = bench_oracle(&golden);
        let payload = iterate_flow_windowed(
            Some(&mut session),
            FlowParams {
                scoring: ScoringInputs::local(prefill, decode),
                ..test_flow(
                    &golden,
                    Mode::LocalIterate,
                    false,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                )
            },
            move || {
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    bp,
                    bs,
                    bd.clone(),
                ))
                .map(|(s, _)| s)
            },
            no_cool_gate,
        );
        assert!(payload.passed, "error={}", payload.metrics.error);
        assert_eq!(payload.metrics.baseline_prefill_seconds_per_token, prefill);
        assert_eq!(payload.metrics.baseline_decode_seconds_per_token, decode);
        assert_ne!(
            payload.metrics.baseline_decode_seconds_per_token, stale_decode,
            "the golden's declared decode baseline reached the sealed score"
        );
    }

    // ---- #132(a): the baseline pair on EVERY local failure path -----------------------
    //
    // #74's ruling said the reference's failure record carries the official-runner CONSTANTS,
    // never zeros. The fix landed only on `preflight_failed_payload`, leaving five locally
    // reachable sites still sealing `baseline_* = 0` — a value the reference emits nowhere,
    // because every one of these returns through the SAME `failed()` closure
    // (`QwenRuntimeLocalIterate.swift@b26f76f:40-74`), which names no baseline argument, so
    // `failedScore`'s defaults stand (`QwenRuntimeBenchmark.swift@b26f76f:1130-1131`).
    //
    // One assertion per site, keyed on that site's DISTINGUISHING signature rather than on
    // "some failure happened" — otherwise a test could pass by reaching a different exit than
    // the one it claims to cover.

    /// The two fields under test, plus the signature that says WHICH exit produced them.
    fn assert_reference_baselines(p: &ScorePayload, site: &str) {
        assert!(
            !p.passed,
            "{site}: precondition — this is a failing payload"
        );
        assert!(
            p.score.is_none(),
            "{site}: precondition — a failing run has no score"
        );
        assert_eq!(
            p.metrics.baseline_prefill_seconds_per_token, TEST_BASELINE.prefill_seconds_per_token,
            "{site}: sealed a prefill baseline the reference never emits"
        );
        assert_eq!(
            p.metrics.baseline_decode_seconds_per_token, TEST_BASELINE.decode_seconds_per_token,
            "{site}: sealed a decode baseline the reference never emits"
        );
    }

    /// The #132(b) RULED seal: a local failure record on which the REFERENCE's fused
    /// checked-timing pass did not complete describes no run.
    ///
    /// The criterion is the reference's, deliberately: at `:455` and `:581` benchd's own
    /// conformance report EXISTS (its gate is a separate phase and it ran), so a benchd-centric
    /// reading of "no report" would be false there while the blank is still right.
    ///
    /// **RULED (David 2026-08-20, interview, FINAL): MIRROR BLANK STRICTLY.** Every local failure
    /// class — all six below — seals `golden_hash = ""` and zero counts, byte-matching the
    /// reference. Zero DECLARED cells on this surface.
    ///
    /// This REVERSES an earlier same-day ruling on the same question (keep-real-values + declare
    /// on the three separated-phase classes), and the reason it reversed is worth keeping next to
    /// the assertions: the reference's `goldenHash` carries an INVARIANT — non-empty means
    /// correctness completed — because it is only ever populated from a `CorrectnessReport` that
    /// exists (`QwenRuntimeBenchmark.swift@b26f76f:1161-1162,1176` read `?? ""` / `?? 0` off a nil
    /// report). Sealing benchd's real digest where correctness did NOT complete is not extra
    /// information; it is the same field meaning something weaker, inherited by every consumer
    /// that trusts the invariant. benchd's real data lives in LOGS, never in sealed records, and a
    /// benchd-only provenance field was explicitly considered and rejected.
    ///
    /// Pinned per-site because "the run knows the digest, so seal it" is the intuitive direction
    /// and it is the ruled-AGAINST one — twice over now.
    fn assert_ruled_blank_seal(p: &ScorePayload, site: &str) {
        assert_eq!(
            p.metrics.golden_hash, "",
            "{site}: #132(b) RULED — a record for a run that produced no correctness report must \
             seal an EMPTY golden_hash; populating it weakens the reference's invariant"
        );
        assert_eq!(
            p.metrics.case_count, 0,
            "{site}: #132(b) RULED — no correctness report ⇒ no case count"
        );
        assert_eq!(
            p.metrics.checked_steps, 0,
            "{site}: #132(b) RULED — no correctness report ⇒ no checked-step count"
        );
    }

    /// A spawner that cannot produce a timed session — for the two sites whose timing phase
    /// fails outright rather than mismatching.
    fn failing_spawner() -> bench_runner::Result<Session<MockEngine>> {
        Err(bench_runner::RunnerError::Protocol(
            "timed engine spawn failed".to_string(),
        ))
    }

    #[test]
    fn site_431_conformance_error_seals_the_reference_baselines() {
        // The conformance gate itself errored (engine protocol fault), so no correctness
        // report exists: `case_count` is the literal 0 this exit passes.
        let golden = TestGolden::new().steps(li_expected()).fixture();
        let (mut session, _) = Session::connect(
            MockEngine::new()
                .teacher_forced_tokens(vec![2i64; li_expected()])
                .error_on("correctness_step", "engine exploded"),
        )
        .unwrap();
        let payload = iterate_flow_windowed(
            Some(&mut session),
            test_flow(
                &golden,
                Mode::LocalIterate,
                false,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            || {
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    2,
                    2,
                    vec![2i64; li_steps()],
                ))
                .map(|(s, _)| s)
            },
            no_cool_gate,
        );
        assert!(!payload.metrics.passed_correctness, "site :431 signature");
        assert_eq!(
            payload.metrics.local_phases,
            Some(LocalPhases {
                correctness: PhaseStatus::Failed,
                correctness_checked_steps: None,
                timing: PhaseStatus::NotRun,
            }),
        );
        assert!(
            // The engine's own error text is sealed as its digest, never verbatim.
            payload
                .metrics
                .error
                .contains(&bench_runner::seal_engine_text("engine exploded"))
                && !payload.metrics.error.contains("engine exploded"),
            "site :431 signature — reached a different exit: {:?}",
            payload.metrics.error
        );
        assert_reference_baselines(&payload, "iterate.rs:431 conformance Err");
        assert_ruled_blank_seal(&payload, "iterate.rs:431 conformance Err");
    }

    #[test]
    fn site_455_close_phase_error_seals_the_reference_baselines() {
        // The gate PASSED and then the phase close failed, so this exit is the one that
        // reports `passed_correctness = true` with the golden's total case count.
        let golden = TestGolden::new().steps(li_expected()).fixture();
        let (mut session, _) = Session::connect(
            MockEngine::new()
                .teacher_forced_tokens(vec![2i64; li_expected()])
                .cache_memory(4096),
        )
        .unwrap();
        let payload = iterate_flow_windowed(
            Some(&mut session),
            test_flow(
                &golden,
                Mode::LocalIterate,
                false,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            || {
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    2,
                    2,
                    vec![2i64; li_steps()],
                ))
                .map(|(s, _)| s)
            },
            no_cool_gate,
        );
        assert!(payload.metrics.passed_correctness, "site :455 signature");
        assert!(
            payload
                .metrics
                .error
                .contains("clear the MLX allocator cache"),
            "site :455 signature — reached a different exit: {:?}",
            payload.metrics.error
        );
        assert_reference_baselines(&payload, "iterate.rs:455 close_phase Err");
        assert_ruled_blank_seal(&payload, "iterate.rs:455 close_phase Err");
    }

    #[test]
    fn site_539_timeonly_failure_after_correctness_failure_seals_the_reference_baselines() {
        // Correctness FAILED, and the time-only retry FAULTED on the window worker — so the run
        // cannot retain real timing and falls back to the blanked payload. Signature: the F1
        // primary failure fields are populated AND the timing is absent.
        let golden = TestGolden::new()
            .steps(li_expected())
            // David's blanket-10% ruling (2026-09-02): a base case REFUSES only when the mismatches
            // break `COHORT_TOKEN_TOLERANCE_PER_THOUSAND`, so the corruption must pass the budget;
            // the FIRST mismatch stays at index 3.
            .corrupt_expected_beyond_budget(3, 999)
            .fixture();
        let (mut session, _) = Session::connect(
            window_engine(&golden, vec![2i64; li_expected()])
                .error_on("free_decode_run", "timed free-run failed"),
        )
        .unwrap();
        let payload = iterate_flow_windowed(
            Some(&mut session),
            test_flow(
                &golden,
                Mode::LocalIterate,
                false,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            failing_spawner,
            no_cool_gate,
        );
        assert!(!payload.metrics.passed_correctness, "site :539 signature");
        assert_eq!(
            payload.metrics.first_failing_case.as_deref(),
            Some("local-iterate"),
            "site :539 signature — the correctness failure is still reported"
        );
        assert_eq!(
            payload.metrics.decode_seconds_per_token, 0.0,
            "site :539 signature — no timing was retained"
        );
        assert!(
            payload
                .metrics
                .error
                .contains("teacher-forced token mismatch"),
            "site :539 signature — reached a different exit: {:?}",
            payload.metrics.error
        );
        assert_reference_baselines(&payload, "iterate.rs:539 time-only Err");
        assert_ruled_blank_seal(&payload, "iterate.rs:539 time-only Err");
    }

    /// #132/F-5 — the sibling site swept in with the five named ones: window-too-short on the
    /// CORRECTNESS-FAILED arm (`iterate.rs:515`), as opposed to `:581` which is the same guard on
    /// the correctness-PASSED arm. Distinguished from `:581` by `passed_correctness`, and from
    /// `:539` by never reaching the timing spawner at all.
    #[test]
    fn site_515_window_too_short_after_correctness_failure_seals_the_blank_record() {
        // A golden that is BOTH too short to time AND corrupt at an index inside its window:
        // correctness fails first, then the retain-timing attempt hits the guard.
        let mut expected = vec![2i64; li_steps()];
        // David's blanket-10% ruling (2026-09-02): the base case now REFUSES only when the
        // mismatches break `COHORT_TOKEN_TOLERANCE_PER_THOUSAND`, so a correctness-FAILURE test
        // must corrupt past the budget. The FIRST mismatch stays at the index asserted below.
        corrupt_beyond_budget(&mut expected, 3, 999);
        // Benchmark-LESS so the timed leg falls back to cases0_timing_params (the RULING-1 branch);
        // the too-short cases[0] window is what fires the guard. (A default benchmark block carries
        // exactly decode_steps tokens, so it would NOT be too short and this guard would not fire.)
        let golden = TestGolden::new()
            .required_steps(li_steps())
            .expected_tokens(expected)
            .without_benchmark()
            .fixture();
        assert_eq!(
            golden.cases[0].expected_tokens.len(),
            li_steps(),
            "precondition: at the guard boundary (len <= decode_steps), so timing is impossible"
        );
        let spawned = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let spawned_c = spawned.clone();
        // A benchmark-LESS golden: the guard fires before any timed leg, so the window worker
        // needs the correctness stream only.
        let (mut session, _) =
            Session::connect(MockEngine::new().teacher_forced_tokens(vec![2i64; li_expected()]))
                .unwrap();
        let payload = iterate_flow_windowed(
            Some(&mut session),
            test_flow(
                &golden,
                Mode::LocalIterate,
                false,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            move || {
                spawned_c.set(spawned_c.get() + 1);
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    2,
                    2,
                    vec![2i64; li_steps()],
                ))
                .map(|(s, _)| s)
            },
            no_cool_gate,
        );
        assert!(
            !payload.metrics.passed_correctness,
            "site :515 signature — correctness FAILED (this is what separates it from :581)"
        );
        assert!(
            payload
                .metrics
                .error
                .contains("teacher-forced token mismatch"),
            "site :515 signature — reached a different exit: {:?}",
            payload.metrics.error
        );
        assert_eq!(
            spawned.get(),
            0,
            "site :515 signature — the guard fires BEFORE any timed engine spawns \
             (that is what separates it from :539, where the spawn is attempted and fails)"
        );
        assert_reference_baselines(&payload, "iterate.rs:515 window too short after fail");
        assert_ruled_blank_seal(&payload, "iterate.rs:515 window too short after fail");
    }

    #[test]
    fn site_581_window_too_short_seals_the_reference_baselines() {
        // Correctness PASSED, but `expected_tokens.len() <= decode_steps` leaves no full
        // window to teacher-force, so the run cannot be timed at all. Signature:
        // `passed_correctness = true` with the Swift guard's message.
        // Benchmark-LESS so the timed leg falls back to cases0_timing_params (the RULING-1 branch),
        // where the too-short cases[0] window fires the Swift "timing needs more than" guard. A
        // default benchmark block carries exactly decode_steps tokens and would not be too short.
        let golden = TestGolden::new()
            .required_steps(li_steps())
            .expected_fill(2)
            .without_benchmark()
            .fixture();
        assert_eq!(
            golden.cases[0].expected_tokens.len(),
            li_steps(),
            "precondition: exactly at the guard boundary (len <= decode_steps)"
        );
        // A benchmark-LESS golden: the guard fires before any timed leg, so the window worker
        // needs the correctness stream only.
        let (mut session, _) =
            Session::connect(MockEngine::new().teacher_forced_tokens(vec![2i64; li_expected()]))
                .unwrap();
        let payload = iterate_flow_windowed(
            Some(&mut session),
            test_flow(
                &golden,
                Mode::LocalIterate,
                false,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            || {
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    2,
                    2,
                    vec![2i64; li_steps()],
                ))
                .map(|(s, _)| s)
            },
            no_cool_gate,
        );
        assert!(payload.metrics.passed_correctness, "site :581 signature");
        assert!(
            payload.metrics.error.contains("timing needs more than"),
            "site :581 signature — reached a different exit: {:?}",
            payload.metrics.error
        );
        assert_reference_baselines(&payload, "iterate.rs:581 window too short");
        assert_ruled_blank_seal(&payload, "iterate.rs:581 window too short");
    }

    #[test]
    fn local_phases_report_gate_pass_before_a_thermal_abort() {
        for abort_phase in ["prefill", "decode"] {
            let golden = TestGolden::new().steps(li_expected()).fixture();
            let (mut session, _) =
                Session::connect(window_engine(&golden, vec![2i64; li_expected()])).unwrap();
            let payload = iterate_flow_windowed(
                Some(&mut session),
                test_flow(
                    &golden,
                    Mode::LocalIterate,
                    false,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                ),
                || {
                    Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                        5,
                        6,
                        vec![7i64; li_steps()],
                    ))
                    .map(|(s, _)| s)
                },
                |phase| {
                    if phase == abort_phase {
                        Err(RunnerError::GateRejected {
                            phase: phase.to_string(),
                            reason: "GPU is hot and not cooling down".into(),
                        })
                    } else {
                        Ok(())
                    }
                },
            );
            assert!(!payload.passed);
            assert!(payload.score.is_none());
            assert!(
                payload
                    .metrics
                    .error
                    .contains("GPU is hot and not cooling down"),
                "{}",
                payload.metrics.error
            );
            assert!(
                payload.metrics.passed_correctness,
                "the untimed gate really passed"
            );
            assert_ruled_blank_seal(&payload, "thermal abort");
            let phases = payload.metrics.local_phases.as_ref().unwrap();
            assert_eq!(phases.correctness, PhaseStatus::Passed);
            assert!(phases.correctness_checked_steps.unwrap() > 0);
            assert_eq!(
                phases.timing,
                if abort_phase == "prefill" {
                    PhaseStatus::NotRun
                } else {
                    PhaseStatus::Failed
                }
            );
        }
    }

    #[test]
    fn local_phases_preserve_failed_correctness_on_phase_close_error() {
        let golden = TestGolden::new()
            .steps(li_expected())
            .corrupt_expected_beyond_budget(3, 999)
            .fixture();
        let (mut session, _) = Session::connect(
            MockEngine::new()
                .teacher_forced_tokens(vec![2i64; li_expected()])
                .cache_memory(4096),
        )
        .unwrap();
        let payload = iterate_flow_windowed(
            Some(&mut session),
            test_flow(
                &golden,
                Mode::LocalIterate,
                false,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            failing_spawner,
            no_cool_gate,
        );
        assert!(!payload.metrics.passed_correctness);
        assert!(payload
            .metrics
            .error
            .contains("clear the MLX allocator cache"));
        let phases = payload.metrics.local_phases.unwrap();
        assert_eq!(phases.correctness, PhaseStatus::Failed);
        assert!(phases.correctness_checked_steps.unwrap() > 0);
        assert_eq!(phases.timing, PhaseStatus::NotRun);
    }

    #[test]
    fn local_phases_mark_both_phases_passed_on_success() {
        let golden = TestGolden::new().steps(li_expected()).fixture();
        let (mut session, _) =
            Session::connect(window_engine(&golden, vec![2i64; li_expected()])).unwrap();
        let payload = iterate_flow_windowed(
            Some(&mut session),
            test_flow(
                &golden,
                Mode::LocalIterate,
                false,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            || {
                Session::connect(MockEngine::new().free_run_capable().oracle_tokens(
                    5,
                    6,
                    vec![7i64; li_steps()],
                ))
                .map(|(s, _)| s)
            },
            no_cool_gate,
        );
        assert!(payload.metrics.passed_correctness);
        let phases = payload.metrics.local_phases.unwrap();
        assert_eq!(phases.correctness, PhaseStatus::Passed);
        assert!(phases.correctness_checked_steps.unwrap() > 0);
        assert_eq!(phases.timing, PhaseStatus::Passed);
    }

    #[test]
    fn local_phases_timed_only_failure_reports_correctness_not_run() {
        // A timed-only (capture/calibration) pass runs no gate. The legacy `passed_correctness`
        // keeps its documented value (`capture.rs`: the capture recorder checks correctness
        // FIRST, so a timing failure must not be misreported as a gate failure); the additive
        // diagnostic is what says the gate never ran.
        let golden = TestGolden::new().steps(li_expected()).fixture();
        let payload = iterate_flow_windowed::<MockEngine, _, _>(
            None,
            FlowParams {
                timed_only: true,
                ..test_flow(
                    &golden,
                    Mode::LocalIterate,
                    false,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                )
            },
            failing_spawner,
            no_cool_gate,
        );
        assert!(payload.metrics.passed_correctness);
        assert_eq!(payload.metrics.local_phases, Some(LocalPhases::default()));
    }

    #[test]
    fn site_619_timed_benchmark_error_seals_the_reference_baselines() {
        // Correctness PASSED and the timed phase itself failed. Signature:
        // `passed_correctness = true` with the golden's total case count and no timing.
        let golden = TestGolden::new().steps(li_expected()).fixture();
        let (mut session, _) = Session::connect(
            window_engine(&golden, vec![2i64; li_expected()])
                .error_on("free_decode_run", "timed free-run failed"),
        )
        .unwrap();
        let payload = iterate_flow_windowed(
            Some(&mut session),
            test_flow(
                &golden,
                Mode::LocalIterate,
                false,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            failing_spawner,
            no_cool_gate,
        );
        assert!(payload.metrics.passed_correctness, "site :619 signature");
        assert_eq!(
            payload.metrics.decode_seconds_per_token, 0.0,
            "site :619 signature — the timed phase produced nothing"
        );
        assert!(
            // The engine's own error text is sealed as its digest, never verbatim.
            payload
                .metrics
                .error
                .contains(&bench_runner::seal_engine_text("timed free-run failed"))
                && !payload.metrics.error.contains("timed free-run failed"),
            "site :619 signature — reached a different exit: {:?}",
            payload.metrics.error
        );
        assert_reference_baselines(&payload, "iterate.rs:619 timed benchmark Err");
        assert_ruled_blank_seal(&payload, "iterate.rs:619 timed benchmark Err");
    }
    // -----------------------------------------------------------------------
    // F1 — the workspace harness identity
    // -----------------------------------------------------------------------

    /// F1 MUTATION PROOF (b) — EVERY MODE seals a 64-hex NON-EMPTY harness identity, and seals the
    /// one the run resolved.
    ///
    /// `base_metrics` is the single funnel every `iterate` payload passes through — official,
    /// official gates-only, local-iterate and local-submit alike — so this is the assertion that
    /// reds the moment `harness_hash: String::new()` comes back. It is stated per-mode rather than
    /// once because "all modes" was the ruling, and a mode-conditional stub would otherwise pass.
    #[test]
    fn base_metrics_seals_a_64_hex_harness_identity_on_every_mode() {
        let golden = benchmark_golden();
        let weights = DirDigest::empty();
        for mode in [Mode::LocalIterate, Mode::LocalSubmit, Mode::Official] {
            let metrics = base_metrics(
                mode,
                &golden,
                RunDigests::for_test(&weights),
                ScoringInputs::local(
                    TEST_BASELINE.prefill_seconds_per_token,
                    TEST_BASELINE.decode_seconds_per_token,
                ),
            );
            assert!(
                !metrics.harness_hash.is_empty(),
                "{}: harness_hash must never be sealed empty — the overlay refuses an empty \
                 harness identity, which is the whole reason F1 exists",
                mode.mode_name()
            );
            assert_eq!(
                metrics.harness_hash.len(),
                64,
                "{}: harness_hash must be a 64-character digest",
                mode.mode_name()
            );
            assert!(
                bench_core::harness_hash::is_well_formed_harness_hash(&metrics.harness_hash),
                "{}: harness_hash must be 64 LOWERCASE hex",
                mode.mode_name()
            );
            assert_eq!(
                metrics.harness_hash,
                HarnessIdentity::TEST_HASH,
                "{}: the sealed identity must be the one the run resolved, not an incidental value",
                mode.mode_name()
            );
        }
    }

    /// The FAILURE builders seal it too. A failed run still writes a score.json, and an artifact
    /// that cannot say which harness produced it is exactly what F1 removes — so the blank-seal
    /// rule (#132(b), which blanks `golden_hash` and the counts) must NOT extend to this field.
    #[test]
    fn failure_payloads_seal_the_harness_identity_too() {
        let golden = benchmark_golden();
        let weights = DirDigest::empty();
        let preflight = preflight_failed_payload(
            Mode::Official,
            &golden,
            RunDigests::for_test(&weights),
            "no baselines".to_string(),
            ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
        );
        assert_eq!(preflight.metrics.harness_hash, HarnessIdentity::TEST_HASH);
        // The blank-seal surface is untouched by F1 — proving this is the identity field only.
        assert!(preflight.metrics.golden_hash.is_empty());
        assert!(!preflight.passed);

        let failed = failed_payload(
            Mode::LocalIterate,
            FailureReport::message("boom".to_string(), false),
            &golden,
            RunDigests::for_test(&weights),
            ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
        );
        assert_eq!(failed.metrics.harness_hash, HarnessIdentity::TEST_HASH);
    }

    /// `HarnessIdentity::resolve` over a REAL synthetic workspace: a 64-hex identity that equals
    /// what `bench_core` computes for the same tree (no second algorithm in benchd).
    #[test]
    fn harness_identity_resolves_over_a_real_workspace() {
        let root = std::env::temp_dir().join(format!("benchd-f1-ws-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        for rel in [
            "Package.swift",
            "Sources/A.swift",
            "Tests/T.swift",
            "benchmark.json",
            "benchmark.sh",
            "setup.sh",
            "tools/t.sh",
            "README.md",
            "TASK.md",
        ] {
            let p = root.join(rel);
            fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
            fs::write(&p, rel).expect("write");
        }
        let identity = HarnessIdentity::resolve(&root).expect("a complete workspace resolves");
        assert!(bench_core::harness_hash::is_well_formed_harness_hash(
            identity.as_str()
        ));
        assert_eq!(
            identity.as_str(),
            bench_core::harness_hash::harness_hash(&root),
            "benchd must not carry a second copy of the algorithm"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// MISSING-ROOT SKIP (David's ruling) — a workspace missing roster roots does NOT refuse; it
    /// resolves a well-formed identity over the roots it has. The roster spans two engine surfaces
    /// and no single engine has every root, so a missing root is the norm. Even an empty directory
    /// resolves (to the empty-set digest), which the overlay's well-formedness gate still accepts.
    #[test]
    fn a_workspace_missing_roots_resolves_over_what_exists() {
        let root = std::env::temp_dir().join(format!("benchd-f1-nows-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("mkdir");
        let identity =
            HarnessIdentity::resolve(&root).expect("a missing root is skipped, never refused");
        assert!(
            bench_core::harness_hash::is_well_formed_harness_hash(identity.as_str()),
            "the identity over a sparse tree must still be 64-lowercase-hex: {}",
            identity.as_str()
        );
        let _ = fs::remove_dir_all(&root);
    }
}
