//! Std-only error type for the runner.
//!
//! No external error crates (`thiserror`/`anyhow`) — the offline build only has
//! `serde`, `serde_json`, and std. `RunnerError` is a hand-written enum implementing
//! `Display` + `std::error::Error`, with `From` conversions for the two foreign error
//! types that cross the transport boundary (`std::io::Error`, `serde_json::Error`).

use std::fmt;

/// Everything that can go wrong driving an engine session.
#[derive(Debug)]
pub enum RunnerError {
    /// Underlying transport I/O failure (pipe write, read, spawn).
    Io(std::io::Error),
    /// A line could not be (de)serialized as an Engine Protocol message.
    Json(serde_json::Error),
    /// Protocol-level violation that is not an engine-signalled error:
    /// malformed hello, unexpected EOF, response id mismatch, unparseable line at EOF, etc.
    Protocol(String),
    /// The response nonce did not equal the session nonce established at hello.
    NonceMismatch {
        expected: String,
        got: Option<String>,
    },
    /// The engine returned `ok:false`. `kind` is the request kind that failed;
    /// `message` is the engine's `error` string (or a placeholder if absent).
    Engine { kind: String, message: String },
    /// Phase-close barrier failure (architecture §3): the engine's reported
    /// `completed_work` did not equal the number of timed steps the runner issued.
    CompletedWorkMismatch { issued: i64, reported: Option<i64> },
    /// Allocator-drain failure (#54): the engine reported a non-zero `cache_memory`
    /// on `phase_diagnostics`, so the MLX free-buffer cache was not drained to zero at
    /// the phase boundary (Swift `resetRuntimeWorkerAllocatorForPhaseStart` fails closed
    /// unless `Memory.cacheMemory == 0`). `reported` is the non-zero byte count.
    AllocatorCacheNotDrained { reported: i64 },
    /// A timed-benchmark response token diverged from the golden benchmark oracle
    /// (Swift `BenchmarkTokenMismatchError`, thrown by `measureWorkerPrefillSecondsPerToken`
    /// / `measureWorkerDecode`). `label` names the phase (e.g. "benchmark decode token"),
    /// `step` is the 0-based step index, and `expected`/`actual` are the oracle vs engine
    /// tokens. A fast engine returning garbage on the timed path is rejected here rather
    /// than being credited with an inflated speedup.
    TokenMismatch {
        label: String,
        step: usize,
        expected: i64,
        actual: i64,
    },
    /// A request was attempted after a prior error discarded the session.
    SessionDiscarded,
    /// benchd tried to issue a capability-gated request (a v1.1 `free_decode_*` kind) to an
    /// engine that did not advertise the required `hello.capabilities` flag. An unadvertised
    /// capability is a hard protocol error, never a silent fallback (PROTOCOL-v1.1.md §2.1);
    /// the session is discarded fail-closed.
    CapabilityNotAdvertised { capability: String },
    /// A v1.1 free-run decode phase failed the §2.6 consistency TRIPLE or a §2.4 count
    /// invariant (`bench_core::free_run::verify_consistency`): a doctored acceptance histogram
    /// or a miscounted forward barrier. `detail` is the specific inconsistency. Fail-closed.
    FreeRunConsistency { detail: String },
    /// H3 (cycle-3) — a timed round-trip exceeded the benchd RunTimeout budget
    /// (PROTOCOL-v1.1 §2.2/§4: `N × band-ceiling × margin`). A hung / looping / stalling engine did
    /// not return within the wall-clock budget, so benchd aborts the read, discards the session
    /// (fail-closed), and the run fails — the transport has no watchdog otherwise, so this is what
    /// bounds a wedge inside the timed window. `phase` names the timed phase; `budget_seconds` is the
    /// armed budget. This never affects the score (a passing run finishes well inside the budget).
    RunTimeout { phase: String, budget_seconds: f64 },
    /// #108 (M2) — the §2.2 RunTimeout budget could NOT be computed for this leg (`N × band-ceiling
    /// × margin` was degenerate: a non-finite / non-positive ceiling or margin, or N == 0). The
    /// ceiling is CALIBRATION-DERIVED (`serial_mean × serial_band_high`), so this condition is
    /// reachable from a `BASELINE_CALIBRATION` file; benchd refuses to open a timed window it cannot
    /// bound rather than arming no deadline at all. `detail` is
    /// [`bench_core::score::run_timeout_budget`]'s diagnostic. The leg fails under its own reject
    /// class — never a silent unbounded run.
    RunTimeoutBudgetInvalid { detail: String },
    /// Spec-never-ignored (`docs/spec-config-design.md` §6): a `decode_begin` carried a `spec`
    /// but the engine's echoed `effective_spec` was ABSENT or DIVERGED from the request. benchd
    /// seals only the echo and never lets a leg silently run a different (or default) spec than the
    /// one requested, so a divergence discards the session fail-closed (mirrors the
    /// `CapabilityNotAdvertised` posture). `requested`/`effective` are the wire JSON for the audit.
    SpecEchoDivergence {
        requested: String,
        effective: Option<String>,
    },
    /// Medium (#105) — a `decode_begin` requested a NON-default speculative `mode` that the engine
    /// did NOT advertise as runnable on its `hello.spec_modes`. Enforced BEFORE THE TIMED SEED
    /// FORWARD is issued, so no forward work is ever charged to a mode the engine cannot run, and
    /// the session is discarded fail-closed (mirrors `CapabilityNotAdvertised`).
    ///
    /// Cycle-5 finding 6 — this was written "PRE-CLOCK", which is not true of the code: the wall
    /// clock starts at `measure_decode`'s `Instant::now()` (`timing.rs`) BEFORE `decode_begin_spec`
    /// is called, so the refusal happens with the clock already running. What it precedes is the
    /// timed seed forward — which is the property that matters (the discarded session is never
    /// scored) but is not the property "pre-clock" claims.
    ///
    /// `serial` (the default path) is always runnable and never gated. `advertised` is the engine's
    /// runnable-mode list.
    SpecModeNotRunnable {
        mode: String,
        advertised: Vec<String>,
    },
    /// v1.2 (COHORT) — BATCH-NEVER-IGNORED: a batched `free_decode_begin` carried an explicit
    /// `batch_size` but the engine's echoed `effective_batch_size` was ABSENT or DIVERGED from the
    /// request. The cohort width is a pinned identity, not a configuration hint (the same posture
    /// as `SpecEchoDivergence`): benchd never lets a leg silently run a narrower (or wider) cohort
    /// than the one requested, because that would seal a different physical arrangement of work
    /// under the requested B's series tag. The session is discarded fail-closed.
    BatchEchoDivergence {
        requested: u32,
        effective: Option<u32>,
    },
    /// v1.2 (COHORT) — the requested cohort width exceeds the `max_batch_size` the engine
    /// advertised on its hello. Refused PRE-GPU — before the cool gate and before the clock — so
    /// no timed window is ever opened against an engine that cannot serve the width. Deterministic
    /// for a given engine build (the second spawn reads the same hello), like
    /// `CapabilityNotAdvertised`.
    BatchWidthExceedsEngineMax { requested: u32, max_batch_size: u32 },
    /// A per-phase thermal / quiescence / clock-floor GATE rejected the leg (measure-job
    /// finding 5). A TYPED rejection so the measure-job classifier keys on the VARIANT, not
    /// on error prose (the old `is_gate_class` substring match on `RunnerError::Protocol` was
    /// brittle). `phase` names the timed phase ("prefill"/"decode"), `reason` is the human
    /// diagnostic. This is the one-gated-retry class; other errors fail closed.
    // UNVERIFIED(measure-job): the gate-class classification for the retry decision.
    GateRejected { phase: String, reason: String },
}

impl fmt::Display for RunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunnerError::Io(e) => write!(f, "transport io error: {e}"),
            RunnerError::Json(e) => write!(f, "protocol json error: {e}"),
            RunnerError::Protocol(msg) => write!(f, "protocol violation: {msg}"),
            RunnerError::NonceMismatch { expected, got } => match got {
                Some(got) => write!(
                    f,
                    "nonce mismatch: expected {expected:?}, got {got:?}"
                ),
                None => write!(f, "nonce mismatch: expected {expected:?}, got none"),
            },
            RunnerError::Engine { kind, message } => {
                write!(f, "engine reported failure on {kind:?}: {message}")
            }
            RunnerError::CompletedWorkMismatch { issued, reported } => match reported {
                Some(reported) => write!(
                    f,
                    "phase-close barrier: issued {issued} timed steps but engine reported {reported} completed"
                ),
                None => write!(
                    f,
                    "phase-close barrier: issued {issued} timed steps but engine reported no completed_work counter"
                ),
            },
            RunnerError::AllocatorCacheNotDrained { reported } => write!(
                f,
                "runtime worker failed to clear the MLX allocator cache at phase start (cache_memory={reported} bytes, expected 0)"
            ),
            RunnerError::TokenMismatch {
                label,
                step,
                expected: _,
                actual,
            } => write!(
                f,
                "{label} mismatch at step {step}: the engine returned token {actual}, which is not the token the golden expects at this step"
            ),
            RunnerError::SessionDiscarded => {
                write!(f, "session discarded by a prior error; no further requests permitted")
            }
            RunnerError::CapabilityNotAdvertised { capability } => write!(
                f,
                "engine did not advertise the {capability:?} capability; refusing the capability-gated request (fail-closed)"
            ),
            RunnerError::FreeRunConsistency { detail } => {
                write!(f, "free-run decode consistency failure: {detail}")
            }
            RunnerError::GateRejected { phase, reason } => {
                write!(f, "gate rejected ({phase}): {reason}")
            }
            RunnerError::SpecEchoDivergence {
                requested,
                effective,
            } => match effective {
                Some(effective) => write!(
                    f,
                    "spec echo divergence: requested {requested} but the engine echoed {effective} \
                     (spec-never-ignored, fail-closed — session discarded)"
                ),
                None => write!(
                    f,
                    "spec echo divergence: requested {requested} but the engine echoed no \
                     effective_spec (spec-never-ignored, fail-closed — session discarded)"
                ),
            },
            RunnerError::BatchEchoDivergence {
                requested,
                effective,
            } => match effective {
                Some(effective) => write!(
                    f,
                    "batch echo divergence: requested batch_size {requested} but the engine echoed \
                     effective_batch_size {effective} (batch-never-ignored, fail-closed — session \
                     discarded)"
                ),
                None => write!(
                    f,
                    "batch echo divergence: requested batch_size {requested} but the engine echoed \
                     no effective_batch_size (batch-never-ignored, fail-closed — session discarded)"
                ),
            },
            RunnerError::BatchWidthExceedsEngineMax {
                requested,
                max_batch_size,
            } => write!(
                f,
                "requested cohort batch_size {requested} exceeds the engine's advertised \
                 max_batch_size {max_batch_size}; refusing pre-GPU (fail-closed)"
            ),
            RunnerError::SpecModeNotRunnable { mode, advertised } => write!(
                f,
                "engine did not advertise speculative mode {mode:?} as runnable (hello.spec_modes = \
                 {advertised:?}); refusing the spec'd decode before any timed work (fail-closed — \
                 session discarded)"
            ),
            RunnerError::RunTimeout {
                phase,
                budget_seconds,
            } => write!(
                f,
                "run timeout ({phase}): the timed round-trip exceeded the {budget_seconds}s benchd \
                 budget (N × band-ceiling × margin); the engine did not respond in time — session \
                 discarded (fail-closed)"
            ),
            RunnerError::RunTimeoutBudgetInvalid { detail } => write!(
                f,
                "run timeout budget could not be armed: {detail}; benchd refuses to open a timed \
                 window with no §2.2 wall-clock bound (fail-closed)"
            ),
        }
    }
}

impl std::error::Error for RunnerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RunnerError::Io(e) => Some(e),
            RunnerError::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RunnerError {
    fn from(e: std::io::Error) -> Self {
        RunnerError::Io(e)
    }
}

impl From<serde_json::Error> for RunnerError {
    fn from(e: serde_json::Error) -> Self {
        RunnerError::Json(e)
    }
}

/// Convenience alias used throughout the runner.
pub type Result<T> = std::result::Result<T, RunnerError>;
