//! Engine lifecycle + parent-side wall-clock timing + phase sequencing.
//!
//! Spawns or connects to an engine, times protocol round trips (Instant), enforces the
//! phase-close barrier + completed-work counter, allocator-drain, and anti-memoization.
//! Reconstructs the /opt measure-job paired-baseline + thermal contract. See §3, §8.
//!
//! This wave (WS1-5 + WS1-7) lands the client seam: a [`LineTransport`] abstraction,
//! the [`Session`] lifecycle (hello handshake, nonce/id validation, typed requests,
//! fail-closed session discard), and the phase-close barrier
//! ([`Session::begin_phase`]/[`Session::close_phase`]) that verifies the engine's
//! reported completed-work counter equals the issued timed-step count. Client semantics
//! are ported from the Swift `RuntimeWorkerClient`.

pub mod error;
pub mod mock;
pub mod sandbox;
pub mod scrub;
pub mod session;
pub mod timing;
pub mod transport;
pub mod wire_crosscheck;

pub use error::{serde_error_summary, Result, RunnerError};
pub use sandbox::{
    build_seatbelt_profile, resolve_official_sandbox, sandbox_exec_command, seatbelt_escaped,
    validate_resident_socket, OfficialSandboxError, OfficialSandboxInputs, OfficialSandboxPlan,
    SandboxProfile, SANDBOX_EXEC_PATH,
};
pub use scrub::{
    scrub_engine_text, scrub_reason_for_seal, seal_byte_len, seal_engine_text,
    SEALED_REASON_BYTE_LIMIT,
};
pub use session::{Hello, Session};
pub use timing::{
    run_batched_free_run_decode_phase_fresh, run_decode_phase_fresh,
    run_free_run_decode_phase_fresh, run_free_run_decode_phase_fresh_time_only,
    run_free_run_timed_benchmark, run_prefill_phase_fresh, run_timed_benchmark,
    run_timed_benchmark_persistent_on_session, BatchedFreeRunPhaseTiming, CohortStreamParams,
    CohortTimingParams, FreeRunPhaseTiming, FreeRunTimingResult, PhaseTiming, TimingParams,
    TimingResult, VerifyMode,
};
pub use transport::{
    redact_worker_stderr_line, sanitized_engine_env, ChildStdioTransport, LineTransport,
    ENGINE_ENV_ALLOWED_EXACT, ENGINE_ENV_ALLOWED_PREFIXES, ENGINE_ENV_FORCED_KEY,
    ENGINE_ENV_FORCED_VALUE, WORKER_STDERR_FORWARD_PREFIX, WORKER_STDERR_TAIL_BYTE_LIMIT,
};
pub use wire_crosscheck::{
    captured_fixture_covers_cohort_wire, captured_fixture_covers_per_stream_timing,
    verify_captured_engine_wire, ENGINE_WIRE_V1_FIXTURE, ENGINE_WIRE_V1_SHA256,
};
