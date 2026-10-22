//! WS1-6 — parent-side wall-clock timing of the benchmark phases.
//!
//! A faithful port of the TIMING approach in the Swift
//! `QwenRuntimeBenchmark.measureWorkerPrefillSecondsPerToken` /
//! `measureWorkerDecode` (Sources/MLXFastHarness/QwenRuntimeBenchmark.swift): all
//! timing is measured in the trusted parent with `std::time::Instant` around
//! protocol round-trips; the worker's own reported per-step durations are
//! distrusted and never used as the score source.
//!
//! Each timed phase is wrapped in [`Session::begin_phase`]/[`Session::close_phase`]
//! so the WS1-7 completed-work barrier runs: the engine's reported `completed_work`
//! must equal the number of timed steps issued, or the run fails (§3).
//!
//! The entry point [`run_timed_benchmark`] is a pure function over a
//! `&mut Session<T>`, so it drives ANY transport — the in-process `MockEngine`
//! (tests) or a live `ChildStdioTransport`.

use std::time::Instant;

use bench_core::free_run::{
    timed_mode_batched_free_run, CohortFreeRunAudit, CohortFreeRunResponse, FreeRunAudit,
    FreeRunResponse, TIMED_MODE_FREE_RUN_V1_1,
};
use bench_core::prefill_window::PhaseWindow;
use bench_protocol::SpecConfig;

use crate::error::{
    seal_depth_clamp_reasons, seal_spec_mode, seal_verification_mode, Result, RunnerError,
};
use crate::session::Session;
use crate::transport::LineTransport;

/// Inputs to one timed benchmark: the prefill prompt, the decode seed, how many
/// checked decode steps to run (128 official, 16 local-iterate), AND the golden
/// benchmark oracle tokens each engine response is verified against.
///
/// The oracle fields (`expected_prefill_token`, `expected_decode_seed_token`,
/// `expected_decode_tokens`) come straight from the golden `BenchmarkGolden` block.
/// Threading them here makes it impossible to run the timed path without an oracle,
/// mirroring Swift `measureWorkerPrefillSecondsPerToken` / `measureWorkerDecode`, which
/// throw `BenchmarkTokenMismatchError` when an engine response diverges from the oracle.
#[derive(Debug, Clone)]
pub struct TimingParams {
    /// Prompt fed to the single timed `prefill` (should be
    /// `BENCHMARK_PREFILL_PROMPT_TOKENS` long for an official run).
    pub prefill_prompt_tokens: Vec<i64>,
    /// Optional prompt used only by the untimed prefill warm-up runs. `None` keeps the legacy
    /// behaviour: warm-up and timed runs use `prefill_prompt_tokens`.
    pub prefill_warmup_prompt_tokens: Option<Vec<i64>>,
    /// Golden oracle: the token the timed `prefill` must greedily produce
    /// (`BenchmarkGolden.expected_prefill_token`).
    ///
    /// #112 (L1) — OPTIONAL, because not every `--golden` document HAS a prefill oracle. A
    /// timed-prompt TAPE carries none (`seed_tokens` / `reference_seed_token` / `rows` describe
    /// the DECODE window only), and measure-job's legs time only that decode window. Those
    /// params are built by [`TimingParams::decode_only`], which leaves this `None` rather than
    /// borrowing some other field's token as a stand-in oracle: [`measure_prefill`] then REFUSES
    /// to run (`Protocol`) instead of timing a prompt against a fabricated expectation. The
    /// GoldenDocument path, which really does carry `expected_prefill_token`, keeps setting it
    /// via [`TimingParams::new`].
    pub expected_prefill_token: Option<i64>,
    /// Seed fed to `decode_begin` (should be `BENCHMARK_DECODE_SEED_TOKENS` long).
    pub decode_seed_tokens: Vec<i64>,
    /// Golden oracle: the token `decode_begin` (seed forward) must produce
    /// (`BenchmarkGolden.expected_decode_seed_token`).
    pub expected_decode_seed_token: i64,
    /// Golden oracle: the token each `decode_step` must produce, indexed by step
    /// (`BenchmarkGolden.expected_decode_tokens`). These are ALSO teacher-forced as the
    /// next step's input, exactly as Swift `measureWorkerDecode` feeds the oracle (not
    /// the engine's own return) forward.
    pub expected_decode_tokens: Vec<i64>,
    /// N — the number of decode tokens `decode_seconds_per_token` divides the decode window by.
    pub decode_steps: usize,
    /// Prefill warmup runs (Swift `benchmarkPrefillWarmupRuns`; 0 by default).
    pub prefill_warmup_runs: usize,
    /// Prefill timed runs (Swift `benchmarkPrefillTimedRuns`; 1 by default).
    pub prefill_timed_runs: usize,
    /// H3 (cycle-3) — the RunTimeout budget for the timed decode round-trips (PROTOCOL-v1.1 §2.2/§4).
    /// `Some(dur)` arms a wall-clock deadline (`now + dur`) around the timed decode window (the
    /// teacher-forced [`measure_decode`] and the free-run [`measure_free_run_decode`]); a hung engine
    /// that never returns raises [`RunnerError::RunTimeout`](crate::RunnerError::RunTimeout) and the
    /// session is discarded (fail-closed) instead of wedging the harness. `None` = no bound (the
    /// untimed default). Computed by the caller as `N × band-ceiling × margin`
    /// ([`bench_core::score::run_timeout_budget`]). It is a LIVENESS bound only — never a score input.
    pub run_timeout: Option<std::time::Duration>,
    /// The per-module speculative configuration (`docs/spec-config-design.md`) carried on the timed
    /// `decode_begin` / `free_decode_begin`. `None` (default) issues the legacy no-spec decode and
    /// does not check the echo; `Some(spec)` enforces SPEC-NEVER-IGNORED (the engine's echoed
    /// `effective_spec` must honor it, else the session is discarded fail-closed). The echoed spec is
    /// surfaced on the result ([`TimingResult::effective_spec`]) for benchd to seal.
    pub spec: Option<SpecConfig>,
}

/// Whether a timed phase VERIFIES each engine token against the golden oracle (and aborts
/// on a mismatch) or runs TIME-ONLY (teacher-forces the oracle tokens and measures
/// wall-clock, tolerating a token mismatch).
///
/// `Verify` is the scored path: an engine that returns a wrong-but-fast token is rejected
/// (`TokenMismatch`) before any speedup is credited. `TimeOnly` mirrors Swift's teacher-
/// forced timing, where TIMING and CORRECTNESS are independent measurements: the oracle
/// token is fed forward as the next input regardless of what the engine returned, and a
/// divergence does NOT abort the timing. It exists for the local-iterate correctness-FAILURE
/// path, where `cases[0]`'s tokens are already known-wrong but Swift still records real timing
/// (David's ruling: "Swift is the reference" — retain real timing on a correctness failure).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyMode {
    /// Compare every engine token to the oracle and abort (`TokenMismatch`) on divergence.
    Verify,
    /// Teacher-force the oracle tokens and measure wall-clock; do NOT abort on a mismatch.
    TimeOnly,
    /// Compare the prefill token to the oracle and abort on a mismatch, as `Verify` does. Do NOT
    /// abort on a different decode seed token or decode token: keep the tokens the engine emitted
    /// ([`TimingResult::emitted_tokens`]) so the caller can judge them. The paired path uses it for
    /// the candidate leg of a track that declares a timed token tolerance.
    PrefillOnly,
}

impl TimingParams {
    /// Construct params from the prompt/seed, golden oracle tokens, and decode window,
    /// using the Swift default prefill run counts (0 warmup, 1 timed).
    pub fn new(
        prefill_prompt_tokens: Vec<i64>,
        expected_prefill_token: i64,
        decode_seed_tokens: Vec<i64>,
        expected_decode_seed_token: i64,
        expected_decode_tokens: Vec<i64>,
        decode_steps: usize,
    ) -> Self {
        Self {
            prefill_prompt_tokens,
            prefill_warmup_prompt_tokens: None,
            expected_prefill_token: Some(expected_prefill_token),
            decode_seed_tokens,
            expected_decode_seed_token,
            expected_decode_tokens,
            decode_steps,
            prefill_warmup_runs: bench_core::constants::BENCHMARK_PREFILL_WARMUP_RUNS,
            prefill_timed_runs: bench_core::constants::BENCHMARK_PREFILL_TIMED_RUNS,
            run_timeout: None,
            spec: None,
        }
    }

    /// #112 (L1) — construct params for a DECODE-ONLY timed window, from a document that carries
    /// no prefill oracle at all (the timed-prompt TAPE).
    ///
    /// The prefill prompt is EMPTY and the prefill oracle is `None` — nothing is invented. Both
    /// halves fail loudly if a prefill phase is ever run with these params:
    /// [`run_timed_benchmark_persistent_on_session`] refuses an empty prefill prompt before any
    /// timed work, and [`measure_prefill`] refuses a missing oracle. Previously the tape path set
    /// `expected_prefill_token = reference_seed_token` — a DECODE-window oracle silently
    /// standing in for a prefill one, inert only because nothing on this path reads it.
    pub fn decode_only(
        decode_seed_tokens: Vec<i64>,
        expected_decode_seed_token: i64,
        expected_decode_tokens: Vec<i64>,
        decode_steps: usize,
    ) -> Self {
        Self {
            prefill_prompt_tokens: Vec::new(),
            prefill_warmup_prompt_tokens: None,
            expected_prefill_token: None,
            decode_seed_tokens,
            expected_decode_seed_token,
            expected_decode_tokens,
            decode_steps,
            prefill_warmup_runs: bench_core::constants::BENCHMARK_PREFILL_WARMUP_RUNS,
            prefill_timed_runs: bench_core::constants::BENCHMARK_PREFILL_TIMED_RUNS,
            run_timeout: None,
            spec: None,
        }
    }

    /// Builder — set the per-module speculative `spec` carried on the timed decode window
    /// (`docs/spec-config-design.md`). Existing call sites keep the no-spec default.
    pub fn with_spec(mut self, spec: Option<SpecConfig>) -> Self {
        self.spec = spec;
        self
    }

    /// Unmeasured prefill passes before the timed one(s), in the same session and phase.
    pub fn with_prefill_warmup_runs(mut self, runs: usize) -> Self {
        self.prefill_warmup_runs = runs;
        self
    }

    /// Use a different prompt for the untimed prefill warm-up runs.
    pub fn with_prefill_warmup_prompt_tokens(mut self, tokens: Option<Vec<i64>>) -> Self {
        self.prefill_warmup_prompt_tokens = tokens;
        self
    }

    /// Measured prefill passes after the warm-up ones, in the same session and phase. The phase's
    /// prefill time is their MEDIAN ([`median_seconds`]).
    pub fn with_prefill_timed_runs(mut self, runs: usize) -> Self {
        self.prefill_timed_runs = runs;
        self
    }

    /// H3 (cycle-3) — set the RunTimeout budget for the timed decode window (§2.2/§4). Builder form
    /// so existing call sites keep the untimed default; the measure-job caller arms it with
    /// `N × band-ceiling × margin`.
    pub fn with_run_timeout(mut self, run_timeout: Option<std::time::Duration>) -> Self {
        self.run_timeout = run_timeout;
        self
    }
}

/// The parent-measured timing of one benchmark, plus the raw elapsed times and
/// the worker-reported peak RAM (audit-only, from `phase_diagnostics`).
#[derive(Debug, Clone, PartialEq)]
pub struct TimingResult {
    /// `prefill_seconds_per_token` = median timed prefill elapsed / prompt token count.
    pub prefill_seconds_per_token: f64,
    /// `decode_seconds_per_token` = the DECODE WINDOW / `decode_steps`
    /// ([`bench_core::score::decode_window_seconds_per_token`]). The seed prefill is not in it.
    pub decode_seconds_per_token: f64,
    /// The `decode_step` count charged (what the seconds-per-token divides by).
    pub decode_steps: usize,
    /// Prompt token count the prefill seconds-per-token divides by.
    pub prefill_prompt_tokens: usize,
    /// Median of the timed prefill round-trip elapsed seconds (raw).
    pub prefill_elapsed_seconds: f64,
    /// The DECODE WINDOW, raw seconds: from the close of the seed prefill to the return of the
    /// decode run. The seed prefill is not in it.
    pub decode_elapsed_seconds: f64,
    /// benchd's own SPLIT of the free-run timed window at the `free_decode_begin` /
    /// `free_decode_run` boundary: the seed prefill and the decode window. The seed prefill is
    /// sealed REPORT-ONLY under its own seed-named key and never feeds a decode figure. `None` on
    /// the teacher-forced v1 decode path, which times no seed prefill.
    pub phase_window: Option<PhaseWindow>,
    /// Worker-reported peak RAM (GB), max over both phases' `phase_diagnostics`.
    /// Audit-only; never part of the score.
    pub peak_ram_gb: f64,
    /// The engine's echoed `effective_spec` from the timed `decode_begin` (`docs/spec-config-design.md`).
    /// `None` when the run carried no `spec` (legacy no-spec decode); `Some` carries the module-parsed
    /// spec the engine actually ran after the spec-never-ignored gate. Opaque blocks contain only
    /// request keys when benchd seals this per leg.
    pub effective_spec: Option<SpecConfig>,
    /// The timed decode window's WHOLE free-run [`FreeRunAudit`] — the per-round
    /// `acceptance_lengths` benchd collected plus the engine's self-reported drafted/accepted
    /// totals, already cross-checked by the §2.6 consistency triple.
    ///
    /// CARRIED, NOT NARROWED. The two free-run producers below used to keep only the audit's
    /// `effective_mean_draft_len` and drop the rest, so a single-leg caller (the official path) had
    /// no way to seal the acceptance facts. AUDIT-ONLY — nothing in here is, or may become, an input
    /// to any seconds-per-token, speedup, floor or band.
    ///
    /// `None` means NO free-run audit exists for this result — the teacher-forced v1 decode path
    /// ([`run_timed_benchmark`]) collects no acceptance histogram. It is NEVER a stand-in for a
    /// measured zero: a free-run leg that observed no drafting still carries a real histogram.
    pub free_run_audit: Option<FreeRunAudit>,
    /// The tokens the free-run decode window emitted: the seed token, then the N decode tokens.
    /// Empty on the teacher-forced v1 decode path.
    pub emitted_tokens: Vec<i64>,
}

impl TimingResult {
    /// The free-run audit's effective mean draft length — the mean number of committed tokens per
    /// verify round, computed by benchd from the per-round `acceptance_lengths` it collected (never
    /// an engine echo). `None` exactly when [`free_run_audit`](Self::free_run_audit) is absent.
    pub fn effective_mean_draft_len(&self) -> Option<f64> {
        self.free_run_audit
            .as_ref()
            .map(FreeRunAudit::effective_mean_draft_len)
    }
}

/// Refuse a prefill window whose prompt is empty, BEFORE any worker is spawned or any clock is
/// started. One copy for every entry point that opens a prefill window.
fn guard_prefill_params(params: &TimingParams) -> Result<()> {
    if params.prefill_prompt_tokens.is_empty() {
        return Err(RunnerError::Protocol(
            "benchmark prefill prompt must not be empty".to_string(),
        ));
    }
    Ok(())
}

/// Refuse a decode window whose seed, step count or oracle cannot support it, BEFORE any worker is
/// spawned or any clock is started. One copy for every entry point that opens a decode window.
///
/// Swift `measureWorkerDecode` guards `expectedTokens.count >= decodeSteps`; without enough oracle
/// tokens the per-step comparison (and teacher-forcing) is undefined.
fn guard_decode_params(params: &TimingParams) -> Result<()> {
    if params.decode_seed_tokens.is_empty() {
        return Err(RunnerError::Protocol(
            "benchmark decode seed must not be empty".to_string(),
        ));
    }
    if params.decode_steps == 0 {
        return Err(RunnerError::Protocol(
            "benchmark decode steps must be positive".to_string(),
        ));
    }
    if params.expected_decode_tokens.len() < params.decode_steps {
        return Err(RunnerError::Protocol(format!(
            "benchmark decode oracle has {} tokens; need at least {}",
            params.expected_decode_tokens.len(),
            params.decode_steps
        )));
    }
    Ok(())
}

/// Run the full timed benchmark (prefill phase then decode phase) against `session`.
///
/// Mirrors the Swift worker benchmark path. Returns a fully-populated
/// [`TimingResult`]. Any protocol/barrier failure propagates and discards the
/// session (fail-closed).
pub fn run_timed_benchmark<T: LineTransport>(
    session: &mut Session<T>,
    params: &TimingParams,
) -> Result<TimingResult> {
    guard_prefill_params(params)?;
    guard_decode_params(params)?;

    let mut peak_ram_gb = 0.0_f64;

    let (prefill_seconds_per_token, prefill_elapsed_seconds) =
        measure_prefill(session, params, VerifyMode::Verify, &mut peak_ram_gb)?;

    let (decode_seconds_per_token, decode_elapsed_seconds, effective_spec) =
        measure_decode(session, params, VerifyMode::Verify, &mut peak_ram_gb)?;

    Ok(TimingResult {
        prefill_seconds_per_token,
        decode_seconds_per_token,
        decode_steps: params.decode_steps,
        prefill_prompt_tokens: params.prefill_prompt_tokens.len(),
        prefill_elapsed_seconds,
        decode_elapsed_seconds,
        phase_window: None,
        peak_ram_gb,
        effective_spec,
        // The teacher-forced v1 decode path collects no acceptance histogram, so there is no
        // free-run audit to carry (absent, never a fabricated zero).
        free_run_audit: None,
        emitted_tokens: Vec::new(),
    })
}

/// The parent-measured timing of one v1.1 **oracle-verified free-run** benchmark
/// (PROTOCOL-v1.1.md), plus the AUDIT view of the MTP acceptance statistics.
///
/// Prefill is measured exactly as in v1 (`prefill_*` fields, RULED OQ8 — prefill untouched);
/// the decode leg is the free-run regime, so this is a NEW SERIES ([`Self::timed_mode`], §5) whose
/// `decode_seconds_per_token` MUST NEVER be compared to a v1 teacher-forced number. `audit`
/// carries the non-scored `audit_spec_*` metrics + the verbatim per-round `acceptance_lengths`.
#[derive(Debug, Clone, PartialEq)]
pub struct FreeRunTimingResult {
    /// `prefill_seconds_per_token` = median timed prefill elapsed / prompt token count (v1).
    pub prefill_seconds_per_token: f64,
    /// `decode_seconds_per_token` = the free-run DECODE WINDOW / N verified tokens
    /// ([`bench_core::score::decode_window_seconds_per_token`]).
    pub decode_seconds_per_token: f64,
    /// N — the number of externally verified committed tokens the decode window covered.
    pub verified_tokens: usize,
    /// Prompt token count the prefill seconds-per-token divides by.
    pub prefill_prompt_tokens: usize,
    /// Median of the timed prefill round-trip elapsed seconds (raw).
    pub prefill_elapsed_seconds: f64,
    /// The free-run DECODE WINDOW, raw seconds (`free_decode_run` only; the seed prefill is not in
    /// it).
    pub decode_elapsed_seconds: f64,
    /// benchd's own SPLIT of the free-run phase's clock at the `free_decode_begin` /
    /// `free_decode_run` boundary: the seed prefill and the decode window.
    pub phase_window: PhaseWindow,
    /// Worker-reported peak RAM (GB), max over both phases. Audit-only; never scored.
    pub peak_ram_gb: f64,
    /// The MTP acceptance AUDIT (§3): non-scored `audit_spec_*` metrics + raw acceptance_lengths.
    pub audit: FreeRunAudit,
    /// The scoring-series tag (§5): always [`TIMED_MODE_FREE_RUN_V1_1`], so downstream
    /// aggregation can never silently mix this number with the v1 teacher-forced series.
    pub timed_mode: &'static str,
    /// The engine's echoed `effective_spec` from `free_decode_begin` (`docs/spec-config-design.md`),
    /// accepted by the spec-never-ignored gate and projected onto the request's opaque keys. `None`
    /// when no `spec` was carried.
    pub effective_spec: Option<SpecConfig>,
}

/// Run the full v1.1 **oracle-verified free-run** timed benchmark: v1 prefill phase then the
/// free-run decode phase (PROTOCOL-v1.1.md §2). The engine MUST advertise the `free_run_decode`
/// capability on its hello, else benchd REFUSES the mode fail-closed (§2.1) before any work.
///
/// The decode phase drives `free_decode_begin` (seed forward, oracle-verified) then
/// `free_decode_run(N)`, splits its clock at that boundary and scores the decode window only,
/// exact-matches every committed
/// token against `expected_decode_tokens` (§2.7 hard fail on any divergence), and enforces the
/// §2.6 consistency triple at the phase-close barrier. Any protocol/oracle/barrier failure
/// propagates and discards the session (fail-closed).
pub fn run_free_run_timed_benchmark<T: LineTransport>(
    session: &mut Session<T>,
    params: &TimingParams,
) -> Result<FreeRunTimingResult> {
    guard_prefill_params(params)?;
    guard_decode_params(params)?;
    // §2.1: refuse the mode up front on an engine that did not advertise the capability, so no
    // prefill work is spent before the refusal (the session methods also refuse, belt-and-braces).
    if !session.supports_free_run_decode() {
        return Err(RunnerError::CapabilityNotAdvertised {
            capability: bench_protocol::CAPABILITY_FREE_RUN_DECODE.to_string(),
        });
    }

    let mut peak_ram_gb = 0.0_f64;

    let (prefill_seconds_per_token, prefill_elapsed_seconds) =
        measure_prefill(session, params, VerifyMode::Verify, &mut peak_ram_gb)?;

    let free_run = measure_free_run_decode(session, params, VerifyMode::Verify, &mut peak_ram_gb)?;

    Ok(FreeRunTimingResult {
        prefill_seconds_per_token,
        decode_seconds_per_token: free_run.decode_seconds_per_token(),
        verified_tokens: params.decode_steps,
        prefill_prompt_tokens: params.prefill_prompt_tokens.len(),
        prefill_elapsed_seconds,
        decode_elapsed_seconds: free_run.phase_window.decode_elapsed_seconds,
        phase_window: free_run.phase_window,
        peak_ram_gb,
        audit: free_run.audit,
        timed_mode: TIMED_MODE_FREE_RUN_V1_1,
        effective_spec: free_run.effective_spec,
    })
}

/// One timed phase's parent-measured result (seconds-per-token + raw elapsed + peak RAM). The
/// unit of PHASE-GRANULAR retry: a caller retries a single failing phase (prefill OR decode) with
/// a full precondition reset (fresh worker + cool gate) WITHOUT re-running the already-accepted
/// phase — mirroring the per-phase `run_phase` loop of the measure-job contract (§4).
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseTiming {
    pub seconds_per_token: f64,
    pub elapsed_seconds: f64,
    pub peak_ram_gb: f64,
    /// The engine's echoed, already-validated `effective_spec` (`docs/spec-config-design.md`) for a
    /// decode phase; `None` for prefill or a no-spec run. benchd seals this per leg.
    pub effective_spec: Option<SpecConfig>,
}

/// Run ONLY the prefill phase on a FRESH worker (`spawn` → `cool_gate("prefill")` →
/// `measure_prefill`, VERIFY mode). This is the retry unit for a phase-granular caller: a prefill
/// gate/measurement reject re-runs exactly this, never the decode phase. Mirrors the Swift
/// `prefillWorker` lifecycle (`QwenRuntimeLocalIterate.swift:704`, cool-gated at `:708`).
pub fn run_prefill_phase_fresh<T, F, G>(
    spawn: &mut F,
    cool_gate: &mut G,
    params: &TimingParams,
) -> Result<PhaseTiming>
where
    T: LineTransport,
    F: FnMut() -> Result<Session<T>>,
    G: FnMut(&str) -> Result<()>,
{
    guard_prefill_params(params)?;
    let mut session = spawn()?;
    cool_gate("prefill")?;
    let mut peak_ram_gb = 0.0_f64;
    let (seconds_per_token, elapsed_seconds) =
        measure_prefill(&mut session, params, VerifyMode::Verify, &mut peak_ram_gb)?;
    Ok(PhaseTiming {
        seconds_per_token,
        elapsed_seconds,
        peak_ram_gb,
        // Prefill is not a spec'd decode window; no echo to seal.
        effective_spec: None,
    })
}

/// Run ONLY the decode phase on a FRESH worker (`spawn` → `cool_gate("decode")` →
/// `measure_decode`, VERIFY mode). The retry unit for a phase-granular caller: a decode
/// gate/measurement reject re-runs exactly this, never the already-accepted prefill phase.
/// Mirrors the Swift `decodeWorker` lifecycle (`QwenRuntimeLocalIterate.swift:750`, cool-gated
/// at `:754`).
pub fn run_decode_phase_fresh<T, F, G>(
    spawn: &mut F,
    cool_gate: &mut G,
    params: &TimingParams,
) -> Result<PhaseTiming>
where
    T: LineTransport,
    F: FnMut() -> Result<Session<T>>,
    G: FnMut(&str) -> Result<()>,
{
    guard_decode_params(params)?;
    let mut session = spawn()?;
    cool_gate("decode")?;
    let mut peak_ram_gb = 0.0_f64;
    let (seconds_per_token, elapsed_seconds, effective_spec) =
        measure_decode(&mut session, params, VerifyMode::Verify, &mut peak_ram_gb)?;
    Ok(PhaseTiming {
        seconds_per_token,
        elapsed_seconds,
        peak_ram_gb,
        effective_spec,
    })
}

/// One v1.1 **free-run** decode phase's parent-measured result plus the §3 AUDIT view and the §5
/// series tag. The `seconds_per_token` here is the DECODE WINDOW over N
/// ([`bench_core::score::decode_window_seconds_per_token`]), measured by benchd's OWN parent clock,
/// and the ONLY scored number. The seed prefill is not in it.
#[derive(Debug, Clone, PartialEq)]
pub struct FreeRunPhaseTiming {
    /// Parent-measured `decode_seconds_per_token` = the DECODE WINDOW / N verified tokens.
    pub seconds_per_token: f64,
    /// benchd's own SPLIT of the timed window at the `free_decode_begin` / `free_decode_run`
    /// boundary: the seed prefill and the decode window. The split is measured on EVERY free-run
    /// leg; whether anything is ENFORCED on the seed half is the declared regime's decision
    /// ([`bench_core::prefill_window::certify_prefill_window`]).
    pub phase_window: PhaseWindow,
    /// Worker-reported peak RAM (GB) from the phase-close `phase_diagnostics`. Audit-only.
    pub peak_ram_gb: f64,
    /// The engine's echoed, already-validated (spec-never-ignored) `effective_spec` from
    /// `free_decode_begin`; `None` on a no-spec run.
    pub effective_spec: Option<SpecConfig>,
    /// The §3 AUDIT view (`audit_spec_*` metrics + the verbatim per-round `acceptance_lengths`),
    /// produced only after the §2.6 consistency triple passed at the phase-close barrier.
    pub audit: FreeRunAudit,
    /// The §5 series tag — always [`TIMED_MODE_FREE_RUN_V1_1`], so a caller cannot seal this
    /// number under the teacher-forced series.
    pub timed_mode: &'static str,
    /// The SEED PREFILL window (docs/single-stream-prefill-window.md): opens immediately before
    /// `free_decode_begin` and closes immediately after its `seed_token` is validated. It is not
    /// part of the decode figure.
    pub prefill_elapsed_seconds: f64,
    /// THE DECODE WINDOW: opens the instant the seed prefill closes (no untimed gap), closes on
    /// `free_decode_run`'s return. `seconds_per_token` divides this by N.
    pub decode_elapsed_seconds: f64,
    /// The prefill token count: the seed's `decode_seed_tokens.len()`.
    pub prefill_token_total: usize,
    /// The decode token count: `N` committed tokens (the divisor `seconds_per_token` uses).
    pub decode_token_total: usize,
}

/// Run ONLY the v1.1 **free-run** decode phase on a FRESH worker (`spawn` → capability check →
/// `cool_gate("decode")` → [`measure_free_run_decode`]). The free-run counterpart of
/// [`run_decode_phase_fresh`], and the entry point a SCORED caller (benchd's measure-job) drives:
/// one process, one cool gate, one timed window, the parent clock the only scored source.
///
/// §2.1 CAPABILITY REFUSAL BEFORE THE CLOCK: the engine's `hello` must advertise
/// `free_run_decode`. The check runs immediately after the spawn/handshake — BEFORE the cool gate
/// and BEFORE the timed window opens — so an engine that cannot free-run is refused fail-closed
/// ([`RunnerError::CapabilityNotAdvertised`]) without spending gate time or opening a clock that
/// would have to be discarded. An unadvertised capability is a hard protocol error, never a silent
/// fallback to the teacher-forced regime (that would silently swap the measured quantity class).
pub fn run_free_run_decode_phase_fresh<T, F, G>(
    spawn: &mut F,
    cool_gate: &mut G,
    params: &TimingParams,
) -> Result<FreeRunPhaseTiming>
where
    T: LineTransport,
    F: FnMut() -> Result<Session<T>>,
    G: FnMut(&str) -> Result<()>,
{
    run_free_run_decode_phase_fresh_impl(spawn, cool_gate, params, VerifyMode::Verify)
}

/// TIME-ONLY variant of [`run_free_run_decode_phase_fresh`]: the lifecycle (fresh worker, §2.1
/// capability refusal, cool gate, one timed window) is IDENTICAL and the returned
/// `seconds_per_token` is computed the SAME WAY (decode window ÷ committed tokens) — the ONLY
/// difference is that a §2.7 committed-token divergence is TOLERATED rather than fatal
/// ([`VerifyMode::TimeOnly`]).
///
/// CONFINEMENT — this is the ONLY entry point that reaches the single-stream free-run phase under
/// `TimeOnly`, and it exists SOLELY for `measure-noop`'s informational `noop_decode_speedup` rate:
/// the stock engine legitimately diverges from the teacher-forced tape when it free-runs, so a
/// RATE measurement must not abort on a value mismatch. It feeds NO scored/enforced value. Every
/// scored/participant caller (local-iterate, submit, official, any measure-job single-stream) goes
/// through [`run_free_run_decode_phase_fresh`] and keeps `VerifyMode::Verify` — the abort-on-
/// mismatch behavior is unchanged for them.
pub fn run_free_run_decode_phase_fresh_time_only<T, F, G>(
    spawn: &mut F,
    cool_gate: &mut G,
    params: &TimingParams,
) -> Result<FreeRunPhaseTiming>
where
    T: LineTransport,
    F: FnMut() -> Result<Session<T>>,
    G: FnMut(&str) -> Result<()>,
{
    run_free_run_decode_phase_fresh_impl(spawn, cool_gate, params, VerifyMode::TimeOnly)
}

/// Shared body of both free-run single-stream entry points. `verify` is threaded straight through
/// to [`measure_free_run_decode`], where it gates ONLY the §2.7 mismatch-abort; the lifecycle,
/// the guards, the §2.1 capability refusal, the cool gate and the timing computation are one and
/// the same for `Verify` and `TimeOnly`. Private, so `TimeOnly` cannot be reached except via the
/// clearly-named `run_free_run_decode_phase_fresh_time_only` above.
fn run_free_run_decode_phase_fresh_impl<T, F, G>(
    spawn: &mut F,
    cool_gate: &mut G,
    params: &TimingParams,
    verify: VerifyMode,
) -> Result<FreeRunPhaseTiming>
where
    T: LineTransport,
    F: FnMut() -> Result<Session<T>>,
    G: FnMut(&str) -> Result<()>,
{
    // Same fail-fast guards as the teacher-forced phase, checked BEFORE spawning any worker.
    guard_decode_params(params)?;
    let mut session = spawn()?;
    // §2.1 — refuse an engine that did not advertise `free_run_decode` BEFORE the clock (and before
    // the cool gate): no timed window is ever opened against an engine that cannot free-run.
    if !session.supports_free_run_decode() {
        return Err(RunnerError::CapabilityNotAdvertised {
            capability: bench_protocol::CAPABILITY_FREE_RUN_DECODE.to_string(),
        });
    }
    cool_gate("decode")?;
    let mut peak_ram_gb = 0.0_f64;
    let m = measure_free_run_decode(&mut session, params, verify, &mut peak_ram_gb)?;
    Ok(FreeRunPhaseTiming {
        seconds_per_token: m.decode_seconds_per_token(),
        phase_window: m.phase_window,
        peak_ram_gb,
        effective_spec: m.effective_spec,
        audit: m.audit,
        timed_mode: TIMED_MODE_FREE_RUN_V1_1,
        prefill_elapsed_seconds: m.phase_window.seed_prefill_elapsed_seconds,
        decode_elapsed_seconds: m.phase_window.decode_elapsed_seconds,
        prefill_token_total: params.decode_seed_tokens.len(),
        decode_token_total: params.decode_steps,
    })
}

/// One stream (cohort slot) of a v1.2 BATCHED free-run timed window: its decode seed and its
/// golden oracle. The cohort form of the decode-window third of [`TimingParams`] — per slot,
/// because every stream is oracle-checked independently even though the window is one clock.
#[derive(Debug, Clone)]
pub struct CohortStreamParams {
    /// Seed tokens fed to this slot in the batched `free_decode_begin` (`seed_tokens_by_stream`).
    pub decode_seed_tokens: Vec<i64>,
    /// Golden oracle: the token this slot's seed forward must produce
    /// (`seed_token_by_stream[slot]` is exact-matched against it).
    pub expected_decode_seed_token: i64,
    /// Golden oracle: this slot's static-tape continuation. UNDER (b) the cohort path NO LONGER
    /// exact-matches `tokens_by_stream[slot]` against this — token-correctness moved to the trusted
    /// per-stream tolerance gate (benchd `cohort_token_tolerance_gate`, judging against a LIVE
    /// reference argmax). This field is RETAINED only for the tape length/shape precondition
    /// (`expected_decode_tokens.len() >= decode_steps` in `validate`); its token VALUES are no longer
    /// the reference. (The single-stream path still exact-matches its own oracle.)
    pub expected_decode_tokens: Vec<i64>,
}

/// Inputs to one v1.2 BATCHED (cohort) free-run timed window: B streams in SLOT ORDER, one
/// identical per-stream token budget N, one spec for the whole cohort, one clock.
///
/// The cohort is CLOSED and RECTANGULAR by construction (batch-8 design brief D4): every slot
/// carries the same `decode_steps` budget, no refill, no EOS exit — because the engine commits
/// one COMMON width per round, any per-stream budget asymmetry would convert directly into
/// whole-cohort depth-zero rounds, suppressing the phenomenon being measured.
#[derive(Debug, Clone)]
pub struct CohortTimingParams {
    /// The cohort's streams, one per slot, in SLOT ORDER (slot order is pinned by the caller and
    /// sealed; the engine receives `seed_tokens_by_stream` in exactly this order).
    pub streams: Vec<CohortStreamParams>,
    /// The EXPLICIT cohort width B, carried on the wire on both batched verbs and enforced equal
    /// to `streams.len()` before any worker is spawned. Kept as its own field (not inferred at
    /// call sites) because B is a pinned identity: the engine must echo it back and benchd
    /// discards the leg on divergence.
    pub batch_size: u32,
    /// The per-stream committed-token budget N (identical for every slot; D4). The scored window
    /// covers `B * N` committed tokens.
    pub decode_steps: usize,
    /// The RunTimeout budget over the batched timed window (same liveness-only semantics as
    /// [`TimingParams::run_timeout`]).
    pub run_timeout: Option<std::time::Duration>,
    /// The ONE speculative spec for the whole cohort (`free_decode_begin.spec`, unchanged and
    /// singular in v1.2). Per-stream spec is deliberately not representable: the engine forbids
    /// mixed depths within a plan, and offering it here would only manufacture refused legs.
    pub spec: Option<SpecConfig>,
}

impl CohortTimingParams {
    /// Construct cohort params from the streams (in slot order) and the identical per-stream
    /// budget N. `batch_size` is set to `streams.len()` — the one place the width is derived;
    /// everywhere downstream it is an explicit, echoed, never-ignored identity.
    pub fn new(streams: Vec<CohortStreamParams>, decode_steps: usize) -> Self {
        let batch_size = streams.len() as u32;
        Self {
            streams,
            batch_size,
            decode_steps,
            run_timeout: None,
            spec: None,
        }
    }

    /// Builder — set the cohort-wide speculative `spec`.
    pub fn with_spec(mut self, spec: Option<SpecConfig>) -> Self {
        self.spec = spec;
        self
    }

    /// Builder — set the RunTimeout budget for the batched timed window.
    pub fn with_run_timeout(mut self, run_timeout: Option<std::time::Duration>) -> Self {
        self.run_timeout = run_timeout;
        self
    }

    /// The fail-fast guards shared by every batched entry point, checked BEFORE any worker is
    /// spawned: a non-empty cohort whose declared width matches its streams, a positive budget,
    /// and per slot a non-empty seed plus enough oracle tokens to check N committed tokens.
    fn validate(&self) -> Result<()> {
        if self.streams.is_empty() {
            return Err(RunnerError::Protocol(
                "batched free-run cohort must not be empty".to_string(),
            ));
        }
        if self.batch_size as usize != self.streams.len() {
            return Err(RunnerError::Protocol(format!(
                "batched free-run declared batch_size {} but carries {} streams",
                self.batch_size,
                self.streams.len()
            )));
        }
        if self.decode_steps == 0 {
            return Err(RunnerError::Protocol(
                "benchmark decode steps must be positive".to_string(),
            ));
        }
        for (slot, stream) in self.streams.iter().enumerate() {
            if stream.decode_seed_tokens.is_empty() {
                return Err(RunnerError::Protocol(format!(
                    "batched free-run stream {slot} decode seed must not be empty"
                )));
            }
            if stream.expected_decode_tokens.len() < self.decode_steps {
                return Err(RunnerError::Protocol(format!(
                    "batched free-run stream {slot} oracle has {} tokens; need at least {}",
                    stream.expected_decode_tokens.len(),
                    self.decode_steps
                )));
            }
        }
        Ok(())
    }
}

/// One v1.2 BATCHED free-run decode phase's parent-measured result — the cohort counterpart of
/// [`FreeRunPhaseTiming`].
///
/// The clock is split into two CONTIGUOUS windows: `prefill_elapsed_seconds` (brackets
/// `free_decode_begin`, the B-seed prefill: each stream's `decode_seed_tokens` forward, returning
/// `seed_token_by_stream`) and `decode_elapsed_seconds` (brackets `free_decode_run`, the free-run
/// decode of N tokens per stream). `decode_elapsed_seconds`'s clock opens the INSTANT
/// `prefill_elapsed_seconds`'s clock closes (see [`measure_batched_free_run_decode`]), so no
/// untimed gap lies between them. `seconds_per_token` is the DECODE WINDOW over `B * N`
/// ([`bench_core::score::decode_window_seconds_per_token`]); the seed prefill is not in it. The
/// two windows are the cohort composite's only numeric input
/// (`measure_job::shared_window_composite` sums them across the accepted pairs and takes one
/// ratio per component).
#[derive(Debug, Clone, PartialEq)]
pub struct BatchedFreeRunPhaseTiming {
    /// Parent-measured COHORT decode seconds-per-committed-token = the DECODE WINDOW over the full
    /// `B * N` ([`bench_core::score::decode_window_seconds_per_token`]) — never a union of
    /// per-stream windows, never `Σ(N − 1)`.
    pub seconds_per_token: f64,
    /// The SEED PREFILL window: parent clock opened immediately before
    /// `free_decode_begin_batched`, closed immediately after validating its response (the seed
    /// oracle checks are charged to this window — see [`measure_batched_free_run_decode`]).
    pub prefill_elapsed_seconds: f64,
    /// THE DECODE WINDOW: parent clock opened the instant the seed prefill closed (no untimed gap),
    /// closed on `free_decode_run_batched`'s return.
    pub decode_elapsed_seconds: f64,
    /// The seed PREFILL token total: the B streams' `decode_seed_tokens` lengths, summed ("the 8
    /// seeds' prompt tokens", David's ruling).
    pub prefill_token_total: usize,
    /// The DECODE token total: `B * N` committed tokens (the divisor `seconds_per_token` uses).
    pub decode_token_total: usize,
    /// Worker-reported peak RAM (GB) from the phase-close `phase_diagnostics`. Audit-only.
    pub peak_ram_gb: f64,
    /// The engine's echoed, already-validated (spec-never-ignored) `effective_spec`; `None` on a
    /// no-spec run.
    pub effective_spec: Option<SpecConfig>,
    /// The cohort AUDIT view (never scored), produced only after the cohort consistency
    /// quadruple passed at the phase-close barrier.
    pub audit: CohortFreeRunAudit,
    /// The per-batch series tag ([`timed_mode_batched_free_run`], D5): `batched_free_run_v1_2_b{B}`.
    /// A `String` (not a `&'static str`) because the tag carries the cohort width — which is what
    /// lets the EXISTING string-equality series fence refuse every cross-batch comparison with no
    /// new gate code.
    pub timed_mode: String,
    /// B — the cohort width this window ran (echoed and validated, sealed by the caller).
    pub batch_size: u32,
    /// REPORT-ONLY (per-stream arm-fill carry, gap G1) — the engine-reported per-slot monotonic
    /// nanoseconds from the batched `free_decode_begin` response
    /// ([`WorkerResponse::prefill_ns_by_stream`](bench_protocol::WorkerResponse::prefill_ns_by_stream)),
    /// carried VERBATIM (no sums, ratios, or seconds conversions). `None` when the response
    /// carried no vector. UNTRUSTED for scoring (engine-reported-time-untrusted / parent-clock
    /// doctrine): nothing enforced reads this — `seconds_per_token` above remains parent-clock
    /// only. Consumed by the per-stream attestation seal (PR-B) via
    /// `bench_core::per_stream_attestation`.
    pub prefill_ns_by_stream: Option<Vec<u64>>,
    /// REPORT-ONLY (gap G1) — the engine-reported per-slot monotonic nanoseconds from the batched
    /// `free_decode_run` response
    /// ([`WorkerResponse::decode_ns_by_stream`](bench_protocol::WorkerResponse::decode_ns_by_stream)),
    /// same verbatim-carry / untrusted-for-scoring posture as
    /// [`prefill_ns_by_stream`](Self::prefill_ns_by_stream).
    pub decode_ns_by_stream: Option<Vec<u64>>,
    /// REPORT-ONLY (gap G3) — the per-slot committed-token counts (K_slot), carried VERBATIM from
    /// the consistency-validated `tokens_len_by_stream` vector the cohort quadruple checked (one
    /// `len()` per slot of the response rectangle, in slot order) — NEVER reconstructed as
    /// `[N; B]` from the request parameters, which would substitute the request's shape for the
    /// response's and erase exactly the per-slot evidence the attestation's token-count floor
    /// (clause (e)) needs.
    pub tokens_len_by_stream: Vec<usize>,
    /// REPORT-ONLY — whether the engine's hello advertised the `per_stream_timing` capability.
    /// Lets the attestation seal (PR-B) distinguish "not advertised" (vectors legitimately
    /// absent) from "advertised but absent" (a wiring bug), without re-deriving it from the
    /// session after the fact.
    pub per_stream_timing_advertised: bool,
    /// (b) admission — the candidate's COMMITTED tokens (`B` inner arrays of `N`, SLOT ORDER),
    /// surfaced UNJUDGED. The runner no longer dies inline on a static-tape token mismatch (David's
    /// blanket-10% ruling); benchd's post-run TRUSTED-ORACLE tolerance gate replays THIS journal over
    /// the organizer's reference weights and applies the ≤10% per-stream bar. The rectangle shape (B x
    /// N) and the seed tokens are still oracle-checked inside the runner — only the per-token value
    /// comparison against the static tape moved to benchd.
    pub tokens_by_stream: Vec<Vec<i64>>,
}

/// Run ONLY the v1.2 BATCHED free-run decode phase on a FRESH worker — the cohort counterpart of
/// [`run_free_run_decode_phase_fresh`], modelled on it line for line: `spawn` → capability
/// refusal → width refusal → `cool_gate("decode")` → [`measure_batched_free_run_decode`]. One
/// process, one cool gate, one timed window, the parent clock the only scored source.
///
/// CAPABILITY REFUSAL BEFORE THE COOL GATE AND BEFORE THE CLOCK, mirroring v1.1 §2.1: the
/// engine's `hello` must advertise `batched_free_run_decode` — `free_run_decode` alone is a
/// single-stream engine and is refused fail-closed ([`RunnerError::CapabilityNotAdvertised`]),
/// never silently narrowed to B sequential streams (B independent processes with B independent KV
/// caches and no shared scheduler is not a batch). An advertised `hello.max_batch_size` narrower
/// than the requested B is refused at the same point, PRE-GPU
/// ([`RunnerError::BatchWidthExceedsEngineMax`]).
pub fn run_batched_free_run_decode_phase_fresh<T, F, G>(
    spawn: &mut F,
    cool_gate: &mut G,
    params: &CohortTimingParams,
) -> Result<BatchedFreeRunPhaseTiming>
where
    T: LineTransport,
    F: FnMut() -> Result<Session<T>>,
    G: FnMut(&str) -> Result<()>,
{
    // Fail-fast guards, checked BEFORE spawning any worker.
    params.validate()?;
    let mut session = spawn()?;
    // Refuse an engine that did not advertise `batched_free_run_decode` BEFORE the clock (and
    // before the cool gate): no timed window is ever opened against an engine that cannot run a
    // cohort, and the refusal can never be a silent fallback to the single-stream regime.
    if !session.supports_batched_free_run_decode() {
        return Err(RunnerError::CapabilityNotAdvertised {
            capability: bench_protocol::CAPABILITY_BATCHED_FREE_RUN_DECODE.to_string(),
        });
    }
    // Refuse an over-wide cohort PRE-GPU on the engine's own advertised ceiling (when present).
    if let Some(max) = session.max_batch_size() {
        if params.batch_size > max {
            return Err(RunnerError::BatchWidthExceedsEngineMax {
                requested: params.batch_size,
                max_batch_size: max,
            });
        }
    }
    cool_gate("decode")?;
    let mut peak_ram_gb = 0.0_f64;
    let m = measure_batched_free_run_decode(&mut session, params, &mut peak_ram_gb)?;
    Ok(BatchedFreeRunPhaseTiming {
        seconds_per_token: bench_core::score::decode_window_seconds_per_token(
            m.decode_elapsed_seconds,
            m.decode_token_total,
        ),
        prefill_elapsed_seconds: m.prefill_elapsed_seconds,
        decode_elapsed_seconds: m.decode_elapsed_seconds,
        prefill_token_total: m.prefill_token_total,
        decode_token_total: m.decode_token_total,
        peak_ram_gb,
        effective_spec: m.effective_spec,
        audit: m.audit,
        timed_mode: timed_mode_batched_free_run(params.batch_size),
        batch_size: params.batch_size,
        // REPORT-ONLY per-stream carry (gaps G1/G3): the engine-reported per-slot ns vectors, the
        // verbatim K_slot counts, and the advertisement flag ride along for the attestation seal.
        // `seconds_per_token` above reads none of these fields.
        prefill_ns_by_stream: m.prefill_ns_by_stream,
        decode_ns_by_stream: m.decode_ns_by_stream,
        tokens_len_by_stream: m.tokens_len_by_stream,
        per_stream_timing_advertised: m.per_stream_timing_advertised,
        // (b) admission — the committed rectangle, surfaced UNJUDGED for benchd's tolerance gate.
        tokens_by_stream: m.tokens_by_stream,
    })
}

/// THE timed benchmark, over a CALLER-HELD session: run the prefill phase then the free-run decode
/// phase on ONE resident worker the caller retains, so the SAME worker can go on to serve the
/// correctness phase — the model loads ONCE for the whole window (David 2026-08-30 load-once
/// directive; the box-4 acceptance proved a fresh worker per phase piled three concurrent ~90 GiB
/// residencies to ~190 GiB on a 128 GiB box). This is the ONLY timed-window lifecycle benchd has.
///
/// The per-phase ISOLATION is the worker's own `beginMeasuredPhase` reset — re-attest + 6 GiB
/// `cacheLimit` reset + `clearCache` — which it fires at the start of EVERY measured phase's opening
/// verb (`prefill`, then `free_decode_begin`). The reset is request-triggered, so a persistent worker
/// gets exactly the reset a freshly-spawned one would. The prefill phase closes its own barrier
/// inside [`measure_prefill`] (allocator drained to `cache_memory == 0`), so the decode phase enters
/// clean. `cool_gate` is the caller's per-phase GATE callback — benchd passes the quiescence gate
/// and the cool gate, in that order — and it is invoked once per phase (`"prefill"` before the
/// prefill timer, `"decode"` before the decode timer). `verify` gates the oracle-mismatch abort.
pub fn run_timed_benchmark_persistent_on_session<T, G>(
    session: &mut Session<T>,
    cool_gate: &mut G,
    params: &TimingParams,
    verify: VerifyMode,
) -> Result<TimingResult>
where
    T: LineTransport,
    G: FnMut(&str) -> Result<()>,
{
    // Fail-fast guards, checked BEFORE any timed work.
    guard_prefill_params(params)?;
    guard_decode_params(params)?;

    let mut peak_ram_gb = 0.0_f64;

    // PREFILL phase on the resident worker. `measure_prefill` brackets its own begin/close phase; the
    // worker's `beginMeasuredPhase` reset fires on the `prefill` verb, so this phase starts on a
    // clean allocator exactly as a fresh worker would.
    cool_gate("prefill")?;
    let (prefill_seconds_per_token, prefill_elapsed_seconds) =
        measure_prefill(session, params, verify, &mut peak_ram_gb)?;

    // DECODE phase on the SAME resident worker (load-once). §2.1 CAPABILITY REFUSAL first: the
    // worker's hello must advertise `free_run_decode`, else the timed decode is refused fail-closed
    // (never a silent fallback to the teacher-forced regime). The worker's `beginMeasuredPhase` reset
    // fires again on `free_decode_begin`, giving this phase the SAME cold-allocator precondition a
    // freshly-spawned decode worker would have — the accepted isolation substitute for a fresh
    // process (a8/David 2026-08-30). The prefill barrier already drained the allocator to
    // `cache_memory == 0` at its close, so the worker enters the decode phase clean.
    if !session.supports_free_run_decode() {
        return Err(RunnerError::CapabilityNotAdvertised {
            capability: bench_protocol::CAPABILITY_FREE_RUN_DECODE.to_string(),
        });
    }
    cool_gate("decode")?;
    let m = measure_free_run_decode(session, params, verify, &mut peak_ram_gb)?;

    Ok(TimingResult {
        prefill_seconds_per_token,
        decode_seconds_per_token: m.decode_seconds_per_token(),
        decode_steps: params.decode_steps,
        prefill_prompt_tokens: params.prefill_prompt_tokens.len(),
        prefill_elapsed_seconds,
        decode_elapsed_seconds: m.phase_window.decode_elapsed_seconds,
        phase_window: Some(m.phase_window),
        peak_ram_gb,
        effective_spec: m.effective_spec,
        // Carry the WHOLE audit out of the narrowing (it used to die with `m.audit` here).
        free_run_audit: Some(m.audit),
        emitted_tokens: m.emitted_tokens,
    })
}

/// Prefill phase: `warmup + timed` runs of one `prefill` round trip, averaging the
/// timed runs. Prefill is NOT a timed step (§3), so the completed-work barrier at
/// `close_phase` must see 0. Returns `(seconds_per_token, mean_timed_elapsed)`.
fn measure_prefill<T: LineTransport>(
    session: &mut Session<T>,
    params: &TimingParams,
    verify: VerifyMode,
    peak_ram_gb: &mut f64,
) -> Result<(f64, f64)> {
    let prompt_count = params.prefill_prompt_tokens.len();
    let total_runs = params.prefill_warmup_runs + params.prefill_timed_runs;
    if params.prefill_timed_runs == 0 {
        return Err(RunnerError::Protocol(
            "benchmark prefill needs at least one timed run".to_string(),
        ));
    }
    // #112 (L1) — a params object with NO prefill oracle (`TimingParams::decode_only`, built from
    // a timed-prompt tape) cannot time a prefill phase: there is nothing to hold the engine to.
    // Refuse loudly here rather than either fabricating an expectation or silently timing an
    // unchecked prefill. Checked for BOTH verify modes: TimeOnly does not compare the token, but a
    // caller reaching this phase with decode-only params is miswired either way.
    let expected_prefill_token = params.expected_prefill_token.ok_or_else(|| {
        RunnerError::Protocol(
            "benchmark prefill has no golden oracle: these TimingParams carry no \
             expected_prefill_token (a decode-only params object, e.g. one built from a \
             timed-prompt tape, cannot time a prefill phase)"
                .to_string(),
        )
    })?;

    session.begin_phase();

    let mut timed_elapsed: Vec<f64> = Vec::with_capacity(params.prefill_timed_runs);
    for run_index in 0..total_runs {
        // The worker holds submitted model code, so its reported prefill duration
        // is not trusted: the parent measures the full request/response wall time.
        let start = Instant::now();
        let is_warmup = run_index < params.prefill_warmup_runs;
        let prompt = if is_warmup {
            params
                .prefill_warmup_prompt_tokens
                .as_deref()
                .unwrap_or(&params.prefill_prompt_tokens)
        } else {
            &params.prefill_prompt_tokens
        };
        let resp = session.prefill(prompt)?;
        let elapsed = start.elapsed().as_secs_f64();
        let token = resp.token.ok_or_else(|| {
            RunnerError::Protocol("runtime worker prefill response missing token".to_string())
        })?;
        // Verify the engine's prefill token against the golden oracle (Swift
        // `comparePrefillToken` → `requireBenchmarkMatch`). An engine returning fast
        // garbage is rejected here instead of being credited with a prefill speedup.
        // TimeOnly tolerates the mismatch (correctness is judged separately upstream).
        let alternate_warmup = is_warmup && params.prefill_warmup_prompt_tokens.is_some();
        if !alternate_warmup && verify != VerifyMode::TimeOnly && token != expected_prefill_token {
            return Err(RunnerError::TokenMismatch {
                label: "benchmark prefill token".to_string(),
                step: run_index,
                expected: expected_prefill_token,
                actual: token,
            });
        }
        if run_index >= params.prefill_warmup_runs {
            timed_elapsed.push(elapsed);
        }
    }

    // Barrier: prefill issued no timed steps, so completed_work must be 0.
    let diag = session.close_phase()?;
    if let Some(ram) = diag.peak_ram_gb {
        if ram > *peak_ram_gb {
            *peak_ram_gb = ram;
        }
    }

    let median_elapsed = median_seconds(&timed_elapsed);
    let seconds_per_token =
        bench_core::score::prefill_seconds_per_token(median_elapsed, prompt_count);
    Ok((seconds_per_token, median_elapsed))
}

/// The MEDIAN of the timed prefill passes (the mean of the middle two for an even count). One pass
/// is that pass, so every track that times a single prefill is unchanged. A median, not a mean:
/// one slow pass (a clock that had not ramped yet) must not move the phase's number.
/// The arithmetic is [`bench_core::stats::even_n_median`], the one median implementation.
pub(crate) fn median_seconds(samples: &[f64]) -> f64 {
    assert!(
        !samples.is_empty(),
        "a median needs at least one timed pass"
    );
    bench_core::stats::even_n_median(samples)
}

/// Teacher-forced decode phase. Sequence: `decode_begin(seed)` (the seed prefill, outside the
/// decode clock) → start Instant → `decode_steps` × `decode_step(token)` TEACHER-FORCING the
/// golden oracle token as each next input → stop Instant. The clock covers the decode steps only:
/// decode is the decode window ([`bench_core::score::decode_window_seconds_per_token`]). The
/// barrier then verifies `completed_work == 1 + decode_steps`.
///
/// Two oracle checks mirror Swift `measureWorkerDecode` (`compareDecodeSeedToken` and the
/// per-step `expectedTokens[decodedStep]` comparison): the seed-forward token must equal
/// `expected_decode_seed_token`, and each step's token must equal
/// `expected_decode_tokens[step]`. Crucially the NEXT `decode_step` input is the ORACLE
/// token (seed for step 0, then `expected_decode_tokens[step-1]`), NOT the engine's own
/// return — so an engine that emits a wrong-but-fast token cannot both dodge the check
/// and steer its own cheaper continuation.
/// Returns `(seconds_per_token, elapsed, effective_spec)`. `effective_spec` is the engine's echoed,
/// already-validated (spec-never-ignored) spec from `decode_begin` — `None` when the run carried no
/// `spec`. The per-round `decode_step` reads are each bounded by the armed RunTimeout deadline, so a
/// stalled round aborts as `RunTimeout` rather than wedging (the per-round stall guardrail).
fn measure_decode<T: LineTransport>(
    session: &mut Session<T>,
    params: &TimingParams,
    verify: VerifyMode,
    peak_ram_gb: &mut f64,
) -> Result<(f64, f64, Option<SpecConfig>)> {
    session.begin_phase();

    // H3 (cycle-3) — arm the RunTimeout deadline over the decode round-trips (§2.2/§4): a hung
    // engine raises `RunTimeout` and the session is discarded, instead of wedging here forever. The
    // deadline is disarmed after the timed window so the untimed close-phase barrier is unbounded.
    if let Some(budget) = params.run_timeout {
        session.arm_run_deadline(Instant::now() + budget, budget.as_secs_f64());
    }
    // Spec-never-ignored: `decode_begin_spec` discards the session fail-closed if the engine's echoed
    // `effective_spec` diverges from `params.spec` (§6). The echoed spec is what benchd seals.
    let begin = session.decode_begin_spec(&params.decode_seed_tokens, params.spec.as_ref())?;
    let effective_spec = begin.effective_spec.clone();
    let seed_token = begin.seed_token.ok_or_else(|| {
        RunnerError::Protocol("runtime worker decode_begin response missing seed token".to_string())
    })?;
    if verify == VerifyMode::Verify && seed_token != params.expected_decode_seed_token {
        return Err(RunnerError::TokenMismatch {
            label: "benchmark decode seed token".to_string(),
            step: 0,
            expected: params.expected_decode_seed_token,
            actual: seed_token,
        });
    }
    // THE DECODE WINDOW opens here, after the seed prefill and its oracle check.
    let start = Instant::now();
    for decoded_step in 0..params.decode_steps {
        // Teacher-force the ORACLE token forward: seed for step 0, then the previous
        // expected decode token (Swift `inputToken = decodedStep == 0 ? expectedSeedToken
        // : expectedTokens[decodedStep - 1]`).
        let input_token = if decoded_step == 0 {
            params.expected_decode_seed_token
        } else {
            params.expected_decode_tokens[decoded_step - 1]
        };
        let resp = session.decode_step(input_token)?;
        let token = resp.token.ok_or_else(|| {
            RunnerError::Protocol("runtime worker decode_step response missing token".to_string())
        })?;
        let expected = params.expected_decode_tokens[decoded_step];
        if verify == VerifyMode::Verify && token != expected {
            return Err(RunnerError::TokenMismatch {
                label: "benchmark decode token".to_string(),
                step: decoded_step,
                expected,
                actual: token,
            });
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    // H3 (cycle-3) — timed window closed; the close-phase barrier below is UNTIMED.
    session.disarm_run_deadline();

    // Barrier is OUTSIDE the timed window (Swift measures elapsed before
    // phaseDiagnostics): 1 decode_begin + decode_steps decode_step = completed_work.
    let diag = session.close_phase()?;
    if let Some(ram) = diag.peak_ram_gb {
        if ram > *peak_ram_gb {
            *peak_ram_gb = ram;
        }
    }
    let effective_spec = effective_spec_for_audit(params.spec.as_ref(), effective_spec.as_ref());

    let seconds_per_token =
        bench_core::score::decode_window_seconds_per_token(elapsed, params.decode_steps);
    Ok((seconds_per_token, elapsed, effective_spec))
}

/// v1.1 free-run decode phase (PROTOCOL-v1.1.md §2.2). The seed-prefill clock starts BEFORE
/// `free_decode_begin`, which the driver oracle-checks against `expected_decode_seed_token`. The
/// DECODE WINDOW opens the instant that check is done. Then `free_decode_run(N)` free-runs the engine's own MTP loop
/// and returns all N committed tokens in one response; the clock stops on its return, and every
/// committed token is exact-matched against `expected_decode_tokens[i]` (§2.7 hard fail). The
/// phase-close barrier (OUTSIDE the timed window) then enforces the §2.6 consistency triple.
///
/// The engine free-runs its OWN committed tokens forward — unlike v1's `measure_decode`, no
/// oracle token is teacher-forced back during the run; benchd only verifies the committed
/// stream. Returns the split window, the audit and the emitted tokens.
///
/// `verify` gates the §2.7 per-token exact-match ABORT AND the seed-forward exact-match abort
/// (the two oracle checks — nothing else in the timed window or the barrier depends on it):
/// * [`VerifyMode::Verify`] (the scored/participant default) — the seed token, and any committed
///   token that diverges from the golden continuation, are HARD [`RunnerError::TokenMismatch`]s,
///   exactly as before.
/// * [`VerifyMode::TimeOnly`] — the noop-RATE path (`measure-noop` only): the token comparison is
///   still WALKED but a divergence does NOT abort, because the stock engine legitimately diverges
///   from the teacher-forced tape under free-run and a RATE measurement must tolerate it. This is
///   NOT teacher-forcing (the engine already free-ran `free_decode_run` autonomously; there is no
///   token to force back) — it purely skips benchd's mismatch-abort. The scored `elapsed` /
///   `seconds_per_token`, the §2.4 count invariant, the audit assembly and the §2.6 phase-close
///   barrier are IDENTICAL across both modes: the mode touches the abort branch and nothing else.
/// * [`VerifyMode::PrefillOnly`] — the same as `TimeOnly` here: no abort on a different seed token
///   or decode token. The caller judges the emitted tokens.
fn measure_free_run_decode<T: LineTransport>(
    session: &mut Session<T>,
    params: &TimingParams,
    verify: VerifyMode,
    peak_ram_gb: &mut f64,
) -> Result<FreeRunDecodeMeasurement> {
    let n_usize = params.decode_steps;
    let n = n_usize as u32;

    session.begin_phase();

    // PREFILL window: opens before `free_decode_begin` (the seed forward), closes after the seed
    // token is validated. The RunTimeout deadline is armed here and covers both windows.
    let prefill_start = Instant::now();
    // H3 (cycle-3) — arm the RunTimeout deadline over the free-run timed round-trips (§2.2/§4): a
    // hung/looping engine raises `RunTimeout` (session discarded) instead of stalling the harness
    // inside the timed window. Disarmed after the timed window (the barrier below is untimed).
    if let Some(budget) = params.run_timeout {
        session.arm_run_deadline(prefill_start + budget, budget.as_secs_f64());
    }
    // Spec-never-ignored on the free-run seed forward too (§6).
    let begin = session.free_decode_begin_spec(&params.decode_seed_tokens, params.spec.as_ref())?;
    let effective_spec = begin.effective_spec.clone();
    let seed_token = begin.seed_token.ok_or_else(|| {
        RunnerError::Protocol(
            "runtime worker free_decode_begin response missing seed token".to_string(),
        )
    })?;
    // `verify` gates the seed-forward abort exactly as the per-token abort below (and exactly as
    // the teacher-forced `measure_decode`'s seed check): `Verify` (scored/participant) aborts on a
    // divergent seed; `TimeOnly` (the local-iterate correctness-FAILURE retain-timing path and the
    // noop-RATE path) tolerates it and still records real wall-clock. Without this gate the
    // free-run TimeOnly path would blank the timing on a step-0 divergence where the teacher-forced
    // TimeOnly path retained it — a silent regression of the "retain real timing on a correctness
    // failure" ruling.
    if seed_token != params.expected_decode_seed_token && verify == VerifyMode::Verify {
        return Err(RunnerError::TokenMismatch {
            label: "benchmark free-run decode seed token".to_string(),
            step: 0,
            expected: params.expected_decode_seed_token,
            actual: seed_token,
        });
    }
    // The SPLIT: ONE reading of benchd's own clock serves as both the close of the seed prefill
    // and the open of the decode window, so the two are contiguous with no untimed gap between
    // them. It sits after the seed-oracle check, which is charged to the seed prefill. No new
    // message and no engine-reported duration: the boundary benchd splits on is the message
    // boundary it already drives.
    // DECODE WINDOW: opens the instant the seed prefill closed, closes on `free_decode_run`'s
    // return. Decode seconds per token divides this, and only this, by N.
    let decode_start = Instant::now();
    let prefill_elapsed = decode_start.duration_since(prefill_start).as_secs_f64();

    let run = session.free_decode_run(n)?;
    let decode_elapsed = decode_start.elapsed().as_secs_f64();
    // H3 (cycle-3) — timed window closed; the close-phase barrier / verification below is UNTIMED.
    session.disarm_run_deadline();

    let tokens = run.tokens.clone().ok_or_else(|| {
        RunnerError::Protocol("runtime worker free_decode_run response missing tokens".to_string())
    })?;
    // §2.4 count invariant, checked FIRST and typed as a CONSISTENCY fault (not a generic protocol
    // fault): a response carrying fewer/more than N committed token IDs is the same class of lie as
    // a doctored `acceptance_lengths` — an accounting mismatch benchd's counters falsify — so it
    // must be classified with the triple, not as infra. (Before this, a SHORT `tokens[]` tripped the
    // per-token loop's generic `Protocol` error while a LONG one reached `verify_consistency`'s
    // `TokenCount`, so the same invariant produced two different classes depending on the sign.)
    if tokens.len() != n_usize {
        return Err(RunnerError::FreeRunConsistency {
            detail: bench_core::free_run::FreeRunConsistencyError::TokenCount {
                expected: n_usize,
                got: tokens.len(),
            }
            .to_string(),
        });
    }
    // §2.2 / §2.7: exact-match every committed token against the golden continuation. A wrong
    // free-run token is a HARD failure (the same TokenMismatch class as v1 teacher-forced
    // decode), because under greedy the golden is the one correct continuation.
    //
    // `verify` gates ONLY the abort: under `VerifyMode::TimeOnly` (the `measure-noop` noop-RATE
    // path) the loop still WALKS every token, but a divergence does not abort — the stock engine
    // legitimately diverges from the teacher-forced tape under free-run, and a rate measurement
    // must tolerate that. Everything outside this `if` (the timed windows captured above, the
    // §2.4 count invariant, the audit assembly and the §2.6 barrier below) is unchanged and
    // identical across both modes.
    for (step, &expected) in params
        .expected_decode_tokens
        .iter()
        .take(n_usize)
        .enumerate()
    {
        let actual = tokens.get(step).copied().ok_or_else(|| {
            RunnerError::Protocol(format!(
                "free_decode_run returned {} committed tokens; need {n_usize}",
                tokens.len()
            ))
        })?;
        if actual != expected && verify == VerifyMode::Verify {
            return Err(RunnerError::TokenMismatch {
                label: "benchmark free-run decode token".to_string(),
                step,
                expected,
                actual,
            });
        }
    }

    // Assemble the AUDIT counters for the §2.6 triple; a missing counter is a protocol fault.
    let acceptance_lengths = run.acceptance_lengths.clone().ok_or_else(|| {
        RunnerError::Protocol("free_decode_run response missing acceptance_lengths".to_string())
    })?;
    let drafted_total = run.drafted_total.ok_or_else(|| {
        RunnerError::Protocol("free_decode_run response missing drafted_total".to_string())
    })?;
    let accepted_total = run.accepted_total.ok_or_else(|| {
        RunnerError::Protocol("free_decode_run response missing accepted_total".to_string())
    })?;
    let committed_total = run.committed_total.ok_or_else(|| {
        RunnerError::Protocol("free_decode_run response missing committed_total".to_string())
    })?;
    let fr = FreeRunResponse {
        tokens_len: tokens.len(),
        acceptance_lengths,
        drafted_total,
        accepted_total,
        committed_total,
        // OPTIONAL, so `None` is carried through as NOT REPORTED rather than treated as a protocol
        // fault: an engine built before the field existed still conforms.
        verify_replay_disagreements: run.verify_replay_disagreements,
        verification: bench_core::free_run::VerificationReport::from_wire(
            run.verification_mode
                .as_deref()
                .map(seal_verification_mode)
                .as_deref(),
            run.rectangular_verification_rounds,
            run.serial_verification_rounds,
        ),
    };

    // Phase-close barrier OUTSIDE the timed window: drain assertion + §2.6 consistency triple.
    let (audit, diag) = session.close_free_run_phase(&fr, n)?;
    if let Some(ram) = diag.peak_ram_gb {
        if ram > *peak_ram_gb {
            *peak_ram_gb = ram;
        }
    }
    let effective_spec = effective_spec_for_audit(params.spec.as_ref(), effective_spec.as_ref());

    Ok(FreeRunDecodeMeasurement {
        phase_window: PhaseWindow {
            seed_prefill_elapsed_seconds: prefill_elapsed,
            decode_elapsed_seconds: decode_elapsed,
            prefill_token_total: params.decode_seed_tokens.len(),
            decode_token_total: n_usize,
        },
        audit,
        effective_spec,
        emitted_tokens: std::iter::once(seed_token).chain(tokens).collect(),
    })
}

/// One v1.1 free-run decode phase as [`measure_free_run_decode`] measured it: the clock SPLIT at
/// the `free_decode_begin` / `free_decode_run` boundary into the seed prefill and the decode
/// window (docs/single-stream-prefill-window.md §2), and the engine echoes.
#[derive(Debug)]
struct FreeRunDecodeMeasurement {
    phase_window: PhaseWindow,
    audit: FreeRunAudit,
    effective_spec: Option<SpecConfig>,
    /// The seed token, then the N committed decode tokens.
    emitted_tokens: Vec<i64>,
}

fn effective_spec_for_audit(
    requested: Option<&SpecConfig>,
    effective: Option<&SpecConfig>,
) -> Option<SpecConfig> {
    match (requested, effective) {
        (Some(requested), Some(effective)) => Some(bench_protocol::project_spec_echo_for_audit(
            requested, effective,
        )),
        (None, Some(effective)) => {
            let mut sealed = effective.clone();
            sealed.mode = seal_spec_mode(&effective.mode);
            sealed.dflash = None;
            sealed.dspark = None;
            Some(sealed)
        }
        (_, None) => None,
    }
}

impl FreeRunDecodeMeasurement {
    /// The decode seconds per token: the decode window over N. The one definition.
    fn decode_seconds_per_token(&self) -> f64 {
        bench_core::score::decode_window_seconds_per_token(
            self.phase_window.decode_elapsed_seconds,
            self.phase_window.decode_token_total,
        )
    }
}

/// v1.2 BATCHED free-run decode phase — [`measure_free_run_decode`] generalized to the cohort.
///
/// The clock is split into TWO CONTIGUOUS windows:
///
/// 1. SEED PREFILL window — opens BEFORE the batched `free_decode_begin`, closes AFTER the
///    seed-token oracle checks below. Every instruction between `free_decode_begin` returning and
///    `free_decode_run` being issued is charged to ONE of the two windows, never to neither.
/// 2. DECODE WINDOW — opens THE INSTANT the seed prefill closes (the next statement after
///    `prefill_elapsed_seconds` is read is `Instant::now()` for the decode window — no
///    intervening work), closes on `free_decode_run`'s return. The caller divides this, and only
///    this, by `B * N` ([`bench_core::score::decode_window_seconds_per_token`]).
///
/// Every slot's seed token is exact-matched against that slot's golden (hard fail on any divergence).
/// (b) admission — the `B * N` committed tokens are NO LONGER exact-matched inline against the static
/// tape: under David's blanket-10% ruling the token-correctness decision moved to benchd's post-run
/// TRUSTED-ORACLE tolerance gate, so this function surfaces `tokens_by_stream` UNJUDGED (the rectangle
/// SHAPE is still enforced). The phase-close barrier (OUTSIDE both timed windows) then enforces the
/// cohort consistency quadruple.
///
/// The raw measurement [`measure_batched_free_run_decode`] returns, BEFORE the caller folds in
/// `peak_ram_gb` / `timed_mode` / `batch_size` and computes `seconds_per_token` from the decode
/// window to assemble the public [`BatchedFreeRunPhaseTiming`]. A named struct rather than a long
/// tuple (clippy::type_complexity) — the fields ARE (most of) [`BatchedFreeRunPhaseTiming`]'s
/// fields, so see that struct's doc for what each one means.
struct BatchedFreeRunDecodeMeasurement {
    prefill_elapsed_seconds: f64,
    decode_elapsed_seconds: f64,
    prefill_token_total: usize,
    decode_token_total: usize,
    audit: CohortFreeRunAudit,
    effective_spec: Option<SpecConfig>,
    /// REPORT-ONLY per-stream carry (gaps G1/G3) — see the public
    /// [`BatchedFreeRunPhaseTiming`] fields of the same names for the doctrine notes.
    prefill_ns_by_stream: Option<Vec<u64>>,
    decode_ns_by_stream: Option<Vec<u64>>,
    tokens_len_by_stream: Vec<usize>,
    per_stream_timing_advertised: bool,
    /// (b) admission — the candidate's COMMITTED tokens, `B` inner arrays of `N` in SLOT ORDER,
    /// surfaced UNJUDGED. Under (b) the runner no longer dies inline on a static-tape token mismatch;
    /// the token-correctness decision moved to benchd's post-run trusted-oracle tolerance gate, which
    /// needs this journal as the thing to replay + judge. The rectangle SHAPE (B streams, N each) is
    /// still enforced here fail-closed above — only the per-token value comparison moved out.
    tokens_by_stream: Vec<Vec<i64>>,
}

/// G3 (per-stream arm-fill carry) — the per-slot committed-token counts (K_slot), read from the
/// response rectangle AS-RECEIVED: one `len()` per slot, in slot order. Deliberately a function of
/// the RESPONSE alone — never of the request's `(B, N)` — so the carried vector is the same one
/// the cohort consistency quadruple validates, not a `[N; B]` reconstruction that would hold even
/// when the response's shape did not.
fn cohort_tokens_len_by_stream(tokens_by_stream: &[Vec<i64>]) -> Vec<usize> {
    tokens_by_stream.iter().map(|s| s.len()).collect()
}

fn measure_batched_free_run_decode<T: LineTransport>(
    session: &mut Session<T>,
    params: &CohortTimingParams,
    peak_ram_gb: &mut f64,
) -> Result<BatchedFreeRunDecodeMeasurement> {
    let n_usize = params.decode_steps;
    let n = n_usize as u32;
    let batch_size = params.batch_size;
    let b_usize = batch_size as usize;
    // COMPOSITE — the PREFILL token total: the B streams' seed (prompt) token counts, summed
    // ("the 8 seeds' prompt tokens"). Computed up front, independent of the clock.
    let prefill_token_total: usize = params
        .streams
        .iter()
        .map(|s| s.decode_seed_tokens.len())
        .sum();
    // Per-stream carry (G1) — record whether the hello advertised `per_stream_timing`. Read
    // OUTSIDE the timed windows (nothing here gates or refuses; REPORT-ONLY).
    let per_stream_timing_advertised = session.supports_per_stream_timing();

    session.begin_phase();

    let prefill_start = Instant::now();
    // Arm the RunTimeout deadline over the WHOLE batched round-trip (prefill window open through
    // decode window close) — unchanged budget arithmetic from the single-window phase, just now
    // spanning two contiguous sub-windows instead of one.
    if let Some(budget) = params.run_timeout {
        session.arm_run_deadline(prefill_start + budget, budget.as_secs_f64());
    }
    // Spec-never-ignored AND batch-never-ignored on the cohort seed forward: a divergent (or
    // missing) `effective_spec` / `effective_batch_size` echo discards the leg inside the call.
    let seeds: Vec<Vec<i64>> = params
        .streams
        .iter()
        .map(|s| s.decode_seed_tokens.clone())
        .collect();
    let begin = session.free_decode_begin_batched(&seeds, batch_size, params.spec.as_ref())?;
    let effective_spec = begin.effective_spec.clone();
    let seed_token_by_stream = begin.seed_token_by_stream.clone().ok_or_else(|| {
        RunnerError::Protocol(
            "runtime worker batched free_decode_begin response missing seed_token_by_stream"
                .to_string(),
        )
    })?;
    if seed_token_by_stream.len() != b_usize {
        return Err(RunnerError::Protocol(format!(
            "batched free_decode_begin returned {} seed tokens, expected B={batch_size}",
            seed_token_by_stream.len()
        )));
    }
    // Oracle-check EVERY slot's seed forward, in slot order. ANTI-CHEAT — this loop runs BEFORE
    // the prefill window closes, so its cost is charged to the prefill window rather than sitting
    // in an untimed gap ahead of the decode window's clock.
    for (slot, (&actual, stream)) in seed_token_by_stream
        .iter()
        .zip(params.streams.iter())
        .enumerate()
    {
        if actual != stream.expected_decode_seed_token {
            return Err(RunnerError::TokenMismatch {
                label: format!("benchmark batched free-run decode seed token (stream {slot})"),
                step: 0,
                expected: stream.expected_decode_seed_token,
                actual,
            });
        }
    }
    // PREFILL window closes here.
    let prefill_elapsed = prefill_start.elapsed().as_secs_f64();

    // DECODE window opens IMMEDIATELY — no untimed gap between the two windows (the anti-cheat
    // invariant: nothing runs between this line and the prefill close above).
    let decode_start = Instant::now();
    let run = session.free_decode_run_batched(n, batch_size)?;
    let decode_elapsed = decode_start.elapsed().as_secs_f64();
    // Timed windows closed; the close-phase barrier / verification below is UNTIMED.
    session.disarm_run_deadline();

    // Per-stream carry (G1) — lift the engine-reported per-slot ns vectors off the two responses
    // VERBATIM, outside the timed windows. `None` stays `None` (an engine that sent nothing sent
    // nothing); no shape/zero validation here — that is the attestation seal's job (PR-B), and a
    // report-only carry must not invent refusals.
    let prefill_ns_by_stream = begin.prefill_ns_by_stream.clone();
    let decode_ns_by_stream = run.decode_ns_by_stream.clone();

    let tokens_by_stream = run.tokens_by_stream.clone().ok_or_else(|| {
        RunnerError::Protocol(
            "runtime worker batched free_decode_run response missing tokens_by_stream".to_string(),
        )
    })?;
    // Rectangle-shape invariants, checked FIRST and typed as CONSISTENCY faults (not generic
    // protocol faults), mirroring the single-stream §2.4 posture: a response carrying the wrong
    // number of streams, or a stream with fewer/more than N committed tokens, is the same class
    // of lie as a doctored histogram — an accounting mismatch benchd's counters falsify.
    if tokens_by_stream.len() != b_usize {
        return Err(RunnerError::FreeRunConsistency {
            detail: bench_core::free_run::FreeRunConsistencyError::CohortWidth {
                batch_size,
                got: tokens_by_stream.len(),
            }
            .to_string(),
        });
    }
    for (slot, stream_tokens) in tokens_by_stream.iter().enumerate() {
        if stream_tokens.len() != n_usize {
            return Err(RunnerError::FreeRunConsistency {
                detail: bench_core::free_run::FreeRunConsistencyError::CohortStreamTokenCount {
                    slot,
                    expected: n_usize,
                    got: stream_tokens.len(),
                }
                .to_string(),
            });
        }
    }
    // (b) admission — the inline STATIC-TAPE exact-match on the committed tokens is REMOVED here.
    // Under the trusted-oracle tolerance gate (David's blanket-10% ruling, 2026-08-25) the cohort
    // token-correctness decision is no longer "every committed token equals the static tape row"
    // (die on the first divergence); it is "each stream diverges from the LIVE TRUSTED reference
    // argmax on ≤10% of its committed tokens", judged POST-RUN in benchd against an oracle that
    // replays THESE committed tokens over the organizer's reference weights. So the runner surfaces
    // `tokens_by_stream` UNJUDGED instead of dying on divergence. The seed check above stays EXACT
    // (seed is input-integrity, not decode output — tolerancing it would be a gaming hole), and the
    // rectangle SHAPE checks above (B streams, N tokens each) still fail closed — only the per-token
    // value comparison against the static tape moved out. The static tape is still loaded and pinned
    // by sha256 (the pool is pinned by bytes) and still feeds `params.validate()`'s length guard; it
    // is simply no longer the token-correctness oracle for the cohort path.

    // Assemble the AUDIT counters for the cohort quadruple; a missing counter is a protocol
    // fault. `depth_clamp_reasons` is the one tolerated absence: it is audit-only prose about WHY
    // depth was clamped, carries no cross-checkable accounting, and an engine with nothing to
    // report may omit the histogram entirely (treated as empty, sealed as such). Safe keys stay
    // verbatim. Other keys become stable engine-text seals.
    let acceptance_lengths = run.acceptance_lengths.clone().ok_or_else(|| {
        RunnerError::Protocol(
            "batched free_decode_run response missing acceptance_lengths".to_string(),
        )
    })?;
    let natural_accepted_by_stream = run.natural_accepted_by_stream.clone().ok_or_else(|| {
        RunnerError::Protocol(
            "batched free_decode_run response missing natural_accepted_by_stream".to_string(),
        )
    })?;
    let active_streams_by_round = run.active_streams_by_round.clone().ok_or_else(|| {
        RunnerError::Protocol(
            "batched free_decode_run response missing active_streams_by_round".to_string(),
        )
    })?;
    let rounds = run.rounds.ok_or_else(|| {
        RunnerError::Protocol("batched free_decode_run response missing rounds".to_string())
    })?;
    let drafted_total = run.drafted_total.ok_or_else(|| {
        RunnerError::Protocol("batched free_decode_run response missing drafted_total".to_string())
    })?;
    let accepted_total = run.accepted_total.ok_or_else(|| {
        RunnerError::Protocol("batched free_decode_run response missing accepted_total".to_string())
    })?;
    let committed_total = run.committed_total.ok_or_else(|| {
        RunnerError::Protocol(
            "batched free_decode_run response missing committed_total".to_string(),
        )
    })?;
    // G3 — build K_slot ONCE from the response rectangle AS-RECEIVED; the SAME vector feeds the
    // cohort consistency quadruple below AND rides out as the report-only carry, so the carried
    // copy is by construction the consistency-validated one (never a `[N; B]` reconstruction).
    let tokens_len_by_stream = cohort_tokens_len_by_stream(&tokens_by_stream);
    let cohort = CohortFreeRunResponse {
        batch_size,
        tokens_len_by_stream: tokens_len_by_stream.clone(),
        acceptance_lengths,
        natural_accepted_by_stream,
        active_streams_by_round,
        rounds,
        drafted_total,
        accepted_total,
        committed_total,
        depth_clamp_reasons: run
            .depth_clamp_reasons
            .as_ref()
            .map(seal_depth_clamp_reasons)
            .unwrap_or_default(),
    };

    // Phase-close barrier OUTSIDE the timed window: drain assertion + the cohort consistency
    // quadruple (completed_work == R + 1 — SCALAR, one forward per round regardless of B).
    let (audit, diag) = session.close_batched_free_run_phase(&cohort, n)?;
    if let Some(ram) = diag.peak_ram_gb {
        if ram > *peak_ram_gb {
            *peak_ram_gb = ram;
        }
    }
    let effective_spec = effective_spec_for_audit(params.spec.as_ref(), effective_spec.as_ref());

    // The caller (`run_batched_free_run_decode_phase_fresh`) divides the decode window by
    // `decode_token_total` = B * N.
    let decode_token_total = b_usize * n_usize;
    Ok(BatchedFreeRunDecodeMeasurement {
        prefill_elapsed_seconds: prefill_elapsed,
        decode_elapsed_seconds: decode_elapsed,
        prefill_token_total,
        decode_token_total,
        audit,
        effective_spec,
        prefill_ns_by_stream,
        decode_ns_by_stream,
        tokens_len_by_stream,
        per_stream_timing_advertised,
        // (b) admission — surface the candidate's committed rectangle UNJUDGED for benchd's trusted
        // oracle tolerance gate (its shape is already pinned to B x N above).
        tokens_by_stream,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn prefill_median_is_the_middle_pass_and_ignores_one_slow_pass() {
        use super::median_seconds;
        assert_eq!(median_seconds(&[0.5]), 0.5);
        assert_eq!(median_seconds(&[0.51, 0.50, 0.52]), 0.51);
        // One slow pass (5% over) does not move the median of five.
        assert_eq!(median_seconds(&[0.500, 0.502, 0.501, 0.525, 0.499]), 0.501);
        assert_eq!(median_seconds(&[0.4, 0.6]), 0.5);
    }

    use super::*;
    use crate::mock::MockEngine;
    use crate::transport::ReadOutcome;
    use std::cell::Cell;
    use std::rc::Rc;

    /// A transport wrapper that tracks how many wrapped transports are LIVE at once, so a test can
    /// prove that a timed window never holds two full-model workers at the same time. Construction bumps `live` (and `max` if it is a new high-water mark); `Drop`
    /// decrements `live` — modelling the reap a real `ChildStdioTransport::drop` performs. All reads
    /// delegate to the inner transport unchanged, so the timed measurement is untouched.
    struct LiveCounted<T: LineTransport> {
        inner: T,
        live: Rc<Cell<usize>>,
    }

    impl<T: LineTransport> LiveCounted<T> {
        fn new(inner: T, live: Rc<Cell<usize>>, max: Rc<Cell<usize>>) -> Self {
            let n = live.get() + 1;
            live.set(n);
            if n > max.get() {
                max.set(n);
            }
            Self { inner, live }
        }
    }

    impl<T: LineTransport> LineTransport for LiveCounted<T> {
        fn write_line(&mut self, line: &str) -> std::io::Result<()> {
            self.inner.write_line(line)
        }
        fn read_line(&mut self) -> std::io::Result<Option<String>> {
            self.inner.read_line()
        }
        fn read_line_deadline(
            &mut self,
            deadline: Option<std::time::Instant>,
        ) -> std::io::Result<ReadOutcome> {
            self.inner.read_line_deadline(deadline)
        }
        fn failure_diagnostic(&mut self) -> Option<crate::transport::FailureDiagnostic> {
            self.inner.failure_diagnostic()
        }
    }

    impl<T: LineTransport> Drop for LiveCounted<T> {
        fn drop(&mut self) {
            self.live.set(self.live.get() - 1);
        }
    }

    // Golden oracle the tests hold the engine to.
    const PREFILL_TOKEN: i64 = 5;
    const SEED_TOKEN: i64 = 6;

    /// Distinct oracle decode tokens so a wrong-step test can target exactly one.
    fn oracle_decode_tokens(decode_steps: usize) -> Vec<i64> {
        (0..decode_steps as i64).map(|i| 700 + i).collect()
    }

    /// Params carrying the oracle a `conformant_engine(decode_steps)` matches.
    fn params(decode_steps: usize) -> TimingParams {
        TimingParams::new(
            vec![1; 8],
            PREFILL_TOKEN,
            vec![2; 8],
            SEED_TOKEN,
            oracle_decode_tokens(decode_steps),
            decode_steps,
        )
    }

    /// A mock whose timed tokens exactly match `params(decode_steps)`'s oracle.
    fn conformant_engine(decode_steps: usize) -> MockEngine {
        MockEngine::new().oracle_tokens(
            PREFILL_TOKEN,
            SEED_TOKEN,
            oracle_decode_tokens(decode_steps),
        )
    }

    #[test]
    fn timed_benchmark_plumbing_official_steps() {
        let (mut session, _hello) = Session::connect(conformant_engine(128)).unwrap();
        let result = run_timed_benchmark(&mut session, &params(128)).unwrap();
        assert_eq!(result.decode_steps, 128);
        assert_eq!(result.prefill_prompt_tokens, 8);
        // Timing against the in-process mock is ~0; assert the numbers are finite
        // and non-negative (plumbing), not real wall-clock magnitudes.
        assert!(result.prefill_seconds_per_token.is_finite());
        assert!(result.prefill_seconds_per_token >= 0.0);
        assert!(result.decode_seconds_per_token.is_finite());
        assert!(result.decode_seconds_per_token >= 0.0);
        // Peak RAM comes from the mock's phase_diagnostics (20.25).
        assert_eq!(result.peak_ram_gb, 20.25);
        assert!(!session.is_discarded());
    }

    #[test]
    fn timed_benchmark_local_iterate_steps() {
        let (mut session, _hello) = Session::connect(conformant_engine(16)).unwrap();
        let result = run_timed_benchmark(&mut session, &params(16)).unwrap();
        assert_eq!(result.decode_steps, 16);
        assert!(result.decode_seconds_per_token.is_finite());
    }

    #[test]
    fn timed_benchmark_oracle_matches_passes() {
        // Explicit passing oracle case: every engine token equals the golden oracle,
        // so the run completes and the session survives.
        let (mut session, _hello) = Session::connect(conformant_engine(16)).unwrap();
        let result = run_timed_benchmark(&mut session, &params(16)).unwrap();
        assert_eq!(result.decode_steps, 16);
        assert!(!session.is_discarded());
    }

    #[test]
    fn timed_prefill_wrong_token_rejected() {
        // Engine returns a prefill token that differs from the oracle: rejected before
        // any speedup is credited.
        let engine =
            MockEngine::new().oracle_tokens(PREFILL_TOKEN + 1, SEED_TOKEN, oracle_decode_tokens(8));
        let (mut session, _hello) = Session::connect(engine).unwrap();
        let err = run_timed_benchmark(&mut session, &params(8)).unwrap_err();
        match err {
            RunnerError::TokenMismatch {
                label,
                expected,
                actual,
                ..
            } => {
                assert_eq!(label, "benchmark prefill token");
                assert_eq!(expected, PREFILL_TOKEN);
                assert_eq!(actual, PREFILL_TOKEN + 1);
            }
            other => panic!("expected TokenMismatch, got {other:?}"),
        }
    }

    #[test]
    fn only_an_alternate_warmup_prompt_skips_the_prefill_token_check() {
        let run = |alternate: bool| {
            let engine = MockEngine::new().oracle_tokens(
                PREFILL_TOKEN + 1,
                SEED_TOKEN,
                oracle_decode_tokens(8),
            );
            let (mut session, _) = Session::connect(engine).unwrap();
            let mut p = params(8).with_prefill_warmup_runs(1);
            if alternate {
                p = p.with_prefill_warmup_prompt_tokens(Some(vec![9; 8]));
            }
            let mut peak = 0.0;
            measure_prefill(&mut session, &p, VerifyMode::Verify, &mut peak).unwrap_err()
        };

        let old_prompt = run(false);
        assert!(matches!(
            old_prompt,
            RunnerError::TokenMismatch { step: 0, .. }
        ));
        let alternate_prompt = run(true);
        assert!(matches!(
            alternate_prompt,
            RunnerError::TokenMismatch { step: 1, .. }
        ));
    }

    #[test]
    fn timed_decode_seed_wrong_token_rejected() {
        // Engine's decode_begin seed token diverges from the oracle.
        let engine =
            MockEngine::new().oracle_tokens(PREFILL_TOKEN, SEED_TOKEN + 1, oracle_decode_tokens(8));
        let (mut session, _hello) = Session::connect(engine).unwrap();
        let err = run_timed_benchmark(&mut session, &params(8)).unwrap_err();
        match err {
            RunnerError::TokenMismatch {
                label,
                step,
                expected,
                actual,
            } => {
                assert_eq!(label, "benchmark decode seed token");
                assert_eq!(step, 0);
                assert_eq!(expected, SEED_TOKEN);
                assert_eq!(actual, SEED_TOKEN + 1);
            }
            other => panic!("expected TokenMismatch, got {other:?}"),
        }
    }

    #[test]
    fn timed_decode_wrong_token_rejected() {
        // Prefill + seed match, but the engine emits a wrong (fast-garbage) decode token
        // at step 3. The parent-side oracle check rejects it with a token-mismatch error
        // instead of crediting an inflated decode_speedup (the core of FINDING 1).
        let decode_steps = 8;
        let mut engine_tokens = oracle_decode_tokens(decode_steps);
        engine_tokens[3] = 999_999; // diverges from oracle token 703
        let engine = MockEngine::new().oracle_tokens(PREFILL_TOKEN, SEED_TOKEN, engine_tokens);
        let (mut session, _hello) = Session::connect(engine).unwrap();
        let err = run_timed_benchmark(&mut session, &params(decode_steps)).unwrap_err();
        match err {
            RunnerError::TokenMismatch {
                label,
                step,
                expected,
                actual,
            } => {
                assert_eq!(label, "benchmark decode token");
                assert_eq!(step, 3);
                assert_eq!(expected, 703);
                assert_eq!(actual, 999_999);
            }
            other => panic!("expected TokenMismatch, got {other:?}"),
        }
    }

    #[test]
    fn decode_phase_issues_begin_plus_steps() {
        // Drive the decode phase in isolation (the mock's completed_work_delta is a
        // global offset that would otherwise trip the prefill barrier first). Under-
        // reporting by 1 proves the phase issued exactly 1 decode_begin + N steps.
        let engine = conformant_engine(4).completed_work_delta(-1);
        let (mut session, _hello) = Session::connect(engine).unwrap();
        let mut peak = 0.0;
        let err =
            measure_decode(&mut session, &params(4), VerifyMode::Verify, &mut peak).unwrap_err();
        match err {
            RunnerError::CompletedWorkMismatch { issued, reported } => {
                assert_eq!(issued, 5); // 1 begin + 4 steps
                assert_eq!(reported, Some(4));
            }
            other => panic!("expected CompletedWorkMismatch, got {other:?}"),
        }
        assert!(session.is_discarded());
    }

    #[test]
    fn prefill_phase_barrier_sees_zero_timed_steps() {
        // Prefill issues no timed steps, so the barrier must see completed_work 0;
        // a mock that over-reports by 1 trips it (issued 0, reported 1).
        let engine = conformant_engine(4).completed_work_delta(1);
        let (mut session, _hello) = Session::connect(engine).unwrap();
        let mut peak = 0.0;
        let err =
            measure_prefill(&mut session, &params(4), VerifyMode::Verify, &mut peak).unwrap_err();
        match err {
            RunnerError::CompletedWorkMismatch { issued, reported } => {
                assert_eq!(issued, 0);
                assert_eq!(reported, Some(1));
            }
            other => panic!("expected prefill barrier failure, got {other:?}"),
        }
    }

    #[test]
    fn decode_only_params_carry_no_prefill_oracle_and_refuse_a_prefill_phase() {
        // #112 (L1) — a decode-only params object (the timed-prompt-tape shape) invents NOTHING:
        // the prefill prompt is empty AND the prefill oracle is absent.
        let p = TimingParams::decode_only(vec![2; 8], SEED_TOKEN, oracle_decode_tokens(4), 4);
        assert!(p.prefill_prompt_tokens.is_empty());
        assert_eq!(p.expected_prefill_token, None);

        // Timing a prefill phase with them FAILS LOUDLY rather than measuring an unchecked
        // prompt — in BOTH verify modes, since the fault is the miswiring, not the comparison.
        for verify in [VerifyMode::Verify, VerifyMode::TimeOnly] {
            let (mut session, _hello) = Session::connect(conformant_engine(4)).unwrap();
            let mut peak = 0.0;
            // A non-empty prompt with NO oracle: isolates measure_prefill's own guard from
            // the window runner's empty-prompt guard (each half refuses independently).
            let mut miswired = params(4);
            miswired.expected_prefill_token = None;
            let err = measure_prefill(&mut session, &miswired, verify, &mut peak).unwrap_err();
            assert!(
                matches!(&err, RunnerError::Protocol(m) if m.contains("no golden oracle")),
                "expected a Protocol refusal naming the missing oracle, got {err:?}"
            );
        }
    }

    #[test]
    fn time_only_tolerates_token_mismatch_and_still_measures() {
        // TIME-ONLY: an engine whose prefill, seed, AND decode tokens ALL diverge from the
        // oracle is NOT rejected — it is teacher-forced and timed anyway (Swift's teacher-
        // forced timing, correctness judged separately). Verify mode rejects the same engine.
        let decode_steps = 8;
        let mut engine_tokens = oracle_decode_tokens(decode_steps);
        for t in engine_tokens.iter_mut() {
            *t += 500_000; // every decode token wrong
        }
        let engine = MockEngine::new().oracle_tokens(
            PREFILL_TOKEN + 1, // wrong prefill
            SEED_TOKEN + 1,    // wrong seed
            engine_tokens,
        );
        let (mut session, _hello) = Session::connect(engine).unwrap();
        // Drive the phases directly on one session to exercise the tolerant comparisons.
        let mut peak = 0.0;
        let (p_spt, _p) = measure_prefill(
            &mut session,
            &params(decode_steps),
            VerifyMode::TimeOnly,
            &mut peak,
        )
        .unwrap();
        let (d_spt, _d, _es) = measure_decode(
            &mut session,
            &params(decode_steps),
            VerifyMode::TimeOnly,
            &mut peak,
        )
        .unwrap();
        assert!(p_spt.is_finite() && p_spt >= 0.0);
        assert!(d_spt.is_finite() && d_spt >= 0.0);
        assert!(!session.is_discarded());

        // The SAME divergent engine (identical all-wrong prefill+seed+decode tokens) is REJECTED
        // under Verify — proving TimeOnly relaxes ONLY the token comparison, not the machinery.
        let mut engine_tokens2 = oracle_decode_tokens(decode_steps);
        for t in engine_tokens2.iter_mut() {
            *t += 500_000;
        }
        let (mut session2, _h2) = Session::connect(MockEngine::new().oracle_tokens(
            PREFILL_TOKEN + 1,
            SEED_TOKEN + 1,
            engine_tokens2,
        ))
        .unwrap();
        let err = run_timed_benchmark(&mut session2, &params(decode_steps)).unwrap_err();
        assert!(matches!(err, RunnerError::TokenMismatch { .. }));
    }

    #[test]
    fn empty_prompt_rejected() {
        let (mut session, _hello) = Session::connect(conformant_engine(4)).unwrap();
        let p = TimingParams::new(
            vec![],
            PREFILL_TOKEN,
            vec![2; 8],
            SEED_TOKEN,
            oracle_decode_tokens(4),
            4,
        );
        assert!(run_timed_benchmark(&mut session, &p).is_err());
    }

    #[test]
    fn too_few_oracle_tokens_rejected() {
        // decode_steps exceeds the oracle length -> rejected up front (Swift guard).
        let (mut session, _hello) = Session::connect(conformant_engine(4)).unwrap();
        let p = TimingParams::new(
            vec![1; 8],
            PREFILL_TOKEN,
            vec![2; 8],
            SEED_TOKEN,
            oracle_decode_tokens(2), // only 2 oracle tokens
            4,                       // but 4 decode steps requested
        );
        let err = run_timed_benchmark(&mut session, &p).unwrap_err();
        assert!(matches!(err, RunnerError::Protocol(m) if m.contains("need at least 4")));
    }

    // ---- v1.1 oracle-verified free-run timed mode --------------------------

    /// A conformant v1.1 engine: advertises `free_run_decode` AND returns the golden oracle
    /// tokens for the free-run committed stream.
    fn free_run_engine(n: usize) -> MockEngine {
        conformant_engine(n).free_run_capable()
    }

    #[test]
    fn free_run_positive_control() {
        // Valid free-run round trip: committed stream == golden, §2.6 triple holds.
        let (mut session, hello) = Session::connect(free_run_engine(8)).unwrap();
        assert!(hello.supports_free_run_decode());
        let result = run_free_run_timed_benchmark(&mut session, &params(8)).unwrap();
        assert_eq!(result.timed_mode, "free_run_v1_1");
        assert_eq!(result.verified_tokens, 8);
        // Default mock acceptance is one token per round -> R = N = 8 rounds.
        assert_eq!(result.audit.rounds(), 8);
        assert_eq!(result.audit.acceptance_lengths(), &[1; 8]);
        assert_eq!(result.audit.verified_token_count(), 8);
        assert!(result.decode_seconds_per_token.is_finite());
        assert!(result.decode_seconds_per_token >= 0.0);
        assert_eq!(result.peak_ram_gb, 20.25);
        assert!(!session.is_discarded());
        // The AUDIT metrics are the flat, non-scored audit_spec_* family.
        let metrics = result.audit.to_metrics();
        assert!(metrics.iter().all(|(k, _)| k.starts_with("audit_spec_")));
    }

    #[test]
    fn persistent_on_session_runs_both_phases_on_one_worker_and_survives() {
        // Load-once (David 2026-08-30): prefill + free-run decode over ONE caller-held session, so
        // the same worker can go on to serve the window's correctness phase. Never a second
        // concurrent residency, and the session is left healthy and reusable.
        let live = Rc::new(Cell::new(0usize));
        let max = Rc::new(Cell::new(0usize));
        let (mut session, _hello) = Session::connect(LiveCounted::new(
            free_run_engine(8),
            Rc::clone(&live),
            Rc::clone(&max),
        ))
        .unwrap();
        let mut gate = |_: &str| Ok(());
        let r = run_timed_benchmark_persistent_on_session(
            &mut session,
            &mut gate,
            &params(8),
            VerifyMode::Verify,
        )
        .unwrap();
        assert_eq!(r.decode_steps, 8);
        assert_eq!(r.prefill_prompt_tokens, 8);
        assert!(r.prefill_seconds_per_token.is_finite() && r.prefill_seconds_per_token >= 0.0);
        assert!(r.decode_seconds_per_token.is_finite() && r.decode_seconds_per_token >= 0.0);
        assert_eq!(r.peak_ram_gb, 20.25);
        // ONE worker for both timed phases — never a second concurrent full-model residency.
        assert_eq!(max.get(), 1, "the persistent window opened a second worker");
        // The SAME worker is left healthy and drained (issued_steps reset), ready for correctness.
        assert!(!session.is_discarded());
        assert_eq!(session.issued_steps(), 0);
    }

    /// `PrefillOnly` runs the whole decode window past a different seed token and a different
    /// decode token and keeps what the engine emitted. It still stops at a different prefill
    /// token.
    #[test]
    fn prefill_only_keeps_the_emitted_tokens_and_still_checks_the_prefill_token() {
        let mut committed = oracle_decode_tokens(8);
        committed[3] += 1;
        let engine = MockEngine::new()
            .oracle_tokens(PREFILL_TOKEN, SEED_TOKEN + 1, committed.clone())
            .free_run_capable();
        let (mut session, _hello) = Session::connect(engine).unwrap();
        let mut gate = |_: &str| Ok(());
        let r = run_timed_benchmark_persistent_on_session(
            &mut session,
            &mut gate,
            &params(8),
            VerifyMode::PrefillOnly,
        )
        .expect("a different decode token does not stop a PrefillOnly leg");
        let mut expected = vec![SEED_TOKEN + 1];
        expected.extend(committed);
        assert_eq!(r.emitted_tokens, expected);

        let engine = MockEngine::new()
            .oracle_tokens(PREFILL_TOKEN + 1, SEED_TOKEN, oracle_decode_tokens(8))
            .free_run_capable();
        let (mut session, _hello) = Session::connect(engine).unwrap();
        let err = run_timed_benchmark_persistent_on_session(
            &mut session,
            &mut gate,
            &params(8),
            VerifyMode::PrefillOnly,
        )
        .unwrap_err();
        assert!(
            matches!(err, RunnerError::TokenMismatch { ref label, .. } if label == "benchmark prefill token"),
            "{err:?}"
        );
    }

    #[test]
    fn persistent_on_session_cools_both_phases_at_the_same_points_as_fresh() {
        // The cool gate is invoked once per phase, in order: prefill timer, then decode timer.
        let (mut session, _hello) = Session::connect(free_run_engine(8)).unwrap();
        let mut phases: Vec<String> = Vec::new();
        let mut gate = |p: &str| {
            phases.push(p.to_string());
            Ok(())
        };
        run_timed_benchmark_persistent_on_session(
            &mut session,
            &mut gate,
            &params(8),
            VerifyMode::Verify,
        )
        .unwrap();
        assert_eq!(phases, vec!["prefill".to_string(), "decode".to_string()]);
    }

    #[test]
    fn persistent_on_session_refuses_without_free_run_capability() {
        // §2.1 CAPABILITY REFUSAL: a v1-only engine is refused the free-run decode leg
        // fail-closed, before the decode timer.
        let (mut session, hello) = Session::connect(conformant_engine(8)).unwrap();
        assert!(!hello.supports_free_run_decode());
        let mut gate = |_: &str| Ok(());
        let err = run_timed_benchmark_persistent_on_session(
            &mut session,
            &mut gate,
            &params(8),
            VerifyMode::Verify,
        )
        .unwrap_err();
        match err {
            RunnerError::CapabilityNotAdvertised { capability } => {
                assert_eq!(capability, "free_run_decode");
            }
            other => panic!("expected CapabilityNotAdvertised, got {other:?}"),
        }
    }

    /// (d) SINGLE-STREAM HELLO (qwen3.8-125b-a6b tracks, `scored_batch_size: 1`): the engine no
    /// longer advertises `batched_free_run_decode` / `max_batch_size`, and the single-stream
    /// free-run series must not want them. Positive control here; the B=8 negative control is
    /// `batched_free_run_refused_without_capability_before_gate_and_clock` (refuses by name).
    #[test]
    fn free_run_single_stream_accepts_a_hello_without_the_batched_capability() {
        let (mut session, hello) = Session::connect(free_run_engine(8)).unwrap();
        assert!(hello.supports_free_run_decode());
        assert!(!hello.supports_batched_free_run_decode());
        assert_eq!(hello.max_batch_size, None);
        let result = run_free_run_timed_benchmark(&mut session, &params(8)).unwrap();
        assert_eq!(result.timed_mode, "free_run_v1_1");
        assert_eq!(result.verified_tokens, 8);
        assert!(!session.is_discarded());
    }

    /// SINGLE-STREAM PREFILL WINDOW (docs/single-stream-prefill-window.md §2): the phase result
    /// carries the seed prefill and the decode window, `seconds_per_token` divides the decode
    /// window only, and the token totals name the seed length and N.
    #[test]
    fn free_run_phase_splits_the_parent_clock_at_the_verb_boundary() {
        let mut spawn = || Session::connect(free_run_engine(8)).map(|(s, _)| s);
        let mut gate = |_: &str| Ok(());
        let t = run_free_run_decode_phase_fresh(&mut spawn, &mut gate, &params(8)).unwrap();
        assert!(t.prefill_elapsed_seconds.is_finite() && t.prefill_elapsed_seconds >= 0.0);
        assert!(t.decode_elapsed_seconds.is_finite() && t.decode_elapsed_seconds >= 0.0);
        assert_eq!(t.seconds_per_token, t.decode_elapsed_seconds / 8.0);
        assert_eq!(t.prefill_token_total, params(8).decode_seed_tokens.len());
        assert_eq!(t.decode_token_total, 8);
        assert_eq!(t.timed_mode, "free_run_v1_1");
    }

    #[test]
    fn free_run_batched_acceptance_positive_control() {
        // A batched multi-token acceptance histogram (R=3 rounds committing 3+3+2 = N=8) still
        // passes: sum == N and completed_work == R+1 = 4.
        let engine = free_run_engine(8).free_run_acceptance_lengths(vec![3, 3, 2]);
        let (mut session, _hello) = Session::connect(engine).unwrap();
        let result = run_free_run_timed_benchmark(&mut session, &params(8)).unwrap();
        assert_eq!(result.audit.rounds(), 3);
        assert_eq!(result.audit.effective_tokens_per_forward(), 8.0 / 3.0);
        assert!(!session.is_discarded());
    }

    #[test]
    fn free_run_refused_without_capability() {
        // §2.1: a v1-only engine (no capability) must be REFUSED the free-run mode, fail-closed,
        // before any work — an unadvertised capability is a hard protocol error.
        let (mut session, hello) = Session::connect(conformant_engine(8)).unwrap();
        assert!(!hello.supports_free_run_decode());
        let err = run_free_run_timed_benchmark(&mut session, &params(8)).unwrap_err();
        match err {
            RunnerError::CapabilityNotAdvertised { capability } => {
                assert_eq!(capability, "free_run_decode");
            }
            other => panic!("expected CapabilityNotAdvertised, got {other:?}"),
        }
    }

    #[test]
    fn free_run_seed_mismatch_hard_fail() {
        // A wrong seed forward is a hard TokenMismatch (§2.2 verify), same class as v1.
        let engine =
            MockEngine::new().oracle_tokens(PREFILL_TOKEN, SEED_TOKEN + 1, oracle_decode_tokens(8));
        let (mut session, _hello) = Session::connect(engine.free_run_capable()).unwrap();
        let err = run_free_run_timed_benchmark(&mut session, &params(8)).unwrap_err();
        match err {
            RunnerError::TokenMismatch { label, step, .. } => {
                assert_eq!(label, "benchmark free-run decode seed token");
                assert_eq!(step, 0);
            }
            other => panic!("expected TokenMismatch, got {other:?}"),
        }
    }

    #[test]
    fn free_run_wrong_committed_token_hard_fail() {
        // §2.7: a single wrong committed free-run token is a HARD failure at that index.
        let mut engine_tokens = oracle_decode_tokens(8);
        engine_tokens[5] = 999_999; // diverges from oracle token 705
        let engine = MockEngine::new().oracle_tokens(PREFILL_TOKEN, SEED_TOKEN, engine_tokens);
        let (mut session, _hello) = Session::connect(engine.free_run_capable()).unwrap();
        let err = run_free_run_timed_benchmark(&mut session, &params(8)).unwrap_err();
        match err {
            RunnerError::TokenMismatch {
                label,
                step,
                expected,
                actual,
            } => {
                assert_eq!(label, "benchmark free-run decode token");
                assert_eq!(step, 5);
                assert_eq!(expected, 705);
                assert_eq!(actual, 999_999);
            }
            other => panic!("expected TokenMismatch, got {other:?}"),
        }
    }

    #[test]
    fn free_run_mock_models_the_spec_true_begin_run_seam() {
        // #109 W3 finding 6 — the CONFORMANT mock's seam, stated as a fact rather than left
        // implicit. PROTOCOL-v1.1 §2.2: `free_decode_begin` returns `seed_token` (checked against
        // `expected_decode_seed_token`), and `free_decode_run(N)`'s `tokens[i]` is checked against
        // `expected_decode_tokens[i]` — the N tokens AFTER the seed. §2.1: the begin "establishes
        // the last-committed state"; the run commits N MORE.
        let p = params(8);
        let mut session = Session::connect(free_run_engine(8)).unwrap().0;
        let mut peak = 0.0;
        let audit = measure_free_run_decode(&mut session, &p, VerifyMode::Verify, &mut peak)
            .unwrap()
            .audit;
        // The seed is NOT one of the N verified tokens: it has its own oracle field, and the
        // window's first verified token is the one after it.
        assert_eq!(p.expected_decode_seed_token, SEED_TOKEN);
        assert_eq!(p.expected_decode_tokens[0], 700);
        assert_ne!(p.expected_decode_tokens[0], p.expected_decode_seed_token);
        assert!(!p
            .expected_decode_tokens
            .contains(&p.expected_decode_seed_token));
        // …and the §2.6 counters describe exactly that window: N tokens over R rounds, with the
        // seed forward as the separate `+1` in `completed_work == R + 1`.
        assert_eq!(audit.verified_token_count(), 8);
        assert_eq!(audit.rounds(), 8);
    }

    #[test]
    fn free_run_engine_that_reemits_the_seed_token_hard_fails_at_step_0() {
        // #109 W3 finding 6, the NEGATIVE control — window 3's engine, synthesized. Its
        // `free_decode_run` re-committed the token `free_decode_begin` had already returned and
        // PASSED its oracle check, so every following token landed one position late. On the box
        // this produced, verbatim:
        //
        //   candidate leg: benchmark free-run decode token mismatch at step 0: expected oracle
        //   token 11, engine returned 4625
        //
        // …where 4625 was the tape's own `reference_seed_token`. Under a one-token shift the same
        // stream was 16/16 token-exact against the reference on two independent tapes: the
        // speculation was right and the SEAM was wrong. benchd's oracle is the spec's, so this must
        // stay a hard TokenMismatch at step 0 naming the seed as the actual token.
        let oracle = oracle_decode_tokens(8);
        // [seed, oracle[0..7]] — N tokens, correct stream, shifted one position late.
        let mut shifted = vec![SEED_TOKEN];
        shifted.extend_from_slice(&oracle[..7]);
        let engine = MockEngine::new()
            .oracle_tokens(PREFILL_TOKEN, SEED_TOKEN, shifted)
            .free_run_capable();
        let (mut session, _hello) = Session::connect(engine).unwrap();
        let err = run_free_run_timed_benchmark(&mut session, &params(8)).unwrap_err();
        match err {
            RunnerError::TokenMismatch {
                label,
                step,
                expected,
                actual,
            } => {
                assert_eq!(label, "benchmark free-run decode token");
                assert_eq!(step, 0);
                assert_eq!(expected, oracle[0]);
                assert_eq!(
                    actual, SEED_TOKEN,
                    "the re-emitted seed is what shows up at step 0"
                );
            }
            other => panic!("expected TokenMismatch, got {other:?}"),
        }
    }

    #[test]
    fn free_run_doctored_acceptance_sum_fails_triple() {
        // §2.6 eq.2: acceptance_lengths that do not sum to N fail the triple, fail-closed —
        // even though every committed token is correct. Drive the decode phase directly (a
        // full run would trip the prefill barrier first only if prefill were perturbed; here
        // we isolate the free-run barrier).
        let engine = free_run_engine(8).free_run_acceptance_lengths(vec![1; 9]); // sums to 9
        let (mut session, _hello) = Session::connect(engine).unwrap();
        let mut peak = 0.0;
        let err = measure_free_run_decode(&mut session, &params(8), VerifyMode::Verify, &mut peak)
            .unwrap_err();
        match err {
            RunnerError::FreeRunConsistency { detail } => {
                assert!(
                    detail.contains("sum(acceptance_lengths)"),
                    "detail: {detail}"
                );
            }
            other => panic!("expected FreeRunConsistency, got {other:?}"),
        }
        assert!(session.is_discarded());
    }

    #[test]
    fn free_run_completed_work_not_r_plus_1_fails_triple() {
        // §2.6 eq.3: an under-reported completed_work counter (R rather than R+1) fails the
        // triple. Drive the decode phase directly so the prefill barrier isn't hit first.
        let engine = free_run_engine(8).completed_work_delta(-1);
        let (mut session, _hello) = Session::connect(engine).unwrap();
        let mut peak = 0.0;
        let err = measure_free_run_decode(&mut session, &params(8), VerifyMode::Verify, &mut peak)
            .unwrap_err();
        match err {
            RunnerError::FreeRunConsistency { detail } => {
                assert!(detail.contains("completed_work"), "detail: {detail}");
            }
            other => panic!("expected FreeRunConsistency, got {other:?}"),
        }
        assert!(session.is_discarded());
    }

    #[test]
    fn free_run_committed_total_not_n_fails() {
        // §2.4: committed_total must equal N; a self-reported 7 for N=8 fails, fail-closed.
        let engine = free_run_engine(8).free_run_totals(8, 0, 7);
        let (mut session, _hello) = Session::connect(engine).unwrap();
        let mut peak = 0.0;
        let err = measure_free_run_decode(&mut session, &params(8), VerifyMode::Verify, &mut peak)
            .unwrap_err();
        match err {
            RunnerError::FreeRunConsistency { detail } => {
                assert!(detail.contains("committed_total"), "detail: {detail}");
            }
            other => panic!("expected FreeRunConsistency, got {other:?}"),
        }
        assert!(session.is_discarded());
    }

    #[test]
    fn free_run_undrained_allocator_cache_fails_closed() {
        // The #54 drain assertion still applies at the free-run phase close (§2.6).
        let engine = free_run_engine(8).cache_memory(4096);
        let (mut session, _hello) = Session::connect(engine).unwrap();
        let mut peak = 0.0;
        let err = measure_free_run_decode(&mut session, &params(8), VerifyMode::Verify, &mut peak)
            .unwrap_err();
        assert!(matches!(
            err,
            RunnerError::AllocatorCacheNotDrained { reported: 4096 }
        ));
        assert!(session.is_discarded());
    }

    // ------------------------------------------------------------------
    // spec (docs/spec-config-design.md) — never-ignored echo enforcement
    // ------------------------------------------------------------------

    #[test]
    fn spec_echo_sealed_on_positive_control() {
        // A conformant engine echoes the requested spec verbatim: the run succeeds and the echoed
        // effective_spec is surfaced on the TimingResult for benchd to seal.
        use bench_protocol::SpecConfig;
        let (mut session, hello) = Session::connect(conformant_engine(8)).unwrap();
        assert!(hello.supports_spec_mode("mtp"));
        let p = params(8).with_spec(Some(SpecConfig::mtp(4)));
        let result = run_timed_benchmark(&mut session, &p).unwrap();
        assert_eq!(result.effective_spec, Some(SpecConfig::mtp(4)));
        assert!(!session.is_discarded());
    }

    #[test]
    fn no_spec_run_seals_no_echo() {
        // A no-spec run (legacy) carries no echo and never checks it.
        let (mut session, _hello) = Session::connect(conformant_engine(8)).unwrap();
        let result = run_timed_benchmark(&mut session, &params(8)).unwrap();
        assert_eq!(result.effective_spec, None);
    }

    #[test]
    fn spec_mode_not_advertised_rejects_before_the_timed_seed_forward_and_discards() {
        // Medium (#105) — a spec requesting a NON-default mode the engine did NOT advertise as
        // runnable (hello.spec_modes) is refused BEFORE THE TIMED SEED FORWARD and the session is
        // discarded fail-closed. Wires the previously-dead supports_spec_mode into the path.
        // Cycle-5 finding 6 — deliberately NOT named "pre-clock": `measure_decode` starts its wall
        // clock before `decode_begin_spec`, so the refusal is not before the clock; it is before any
        // forward work that clock would have measured.
        use bench_protocol::SpecConfig;
        // Engine advertises ONLY serial (mtp is a visible-but-stub mode, not runnable).
        let engine = conformant_engine(8).spec_modes(Some(vec!["serial".to_string()]));
        let (mut session, hello) = Session::connect(engine).unwrap();
        assert!(
            !hello.supports_spec_mode("mtp"),
            "mtp is not advertised as runnable"
        );
        let p = params(8).with_spec(Some(SpecConfig::mtp(4)));
        let err = run_timed_benchmark(&mut session, &p).unwrap_err();
        match err {
            RunnerError::SpecModeNotRunnable { mode, advertised } => {
                assert_eq!(mode, "mtp");
                assert_eq!(advertised, vec!["serial".to_string()]);
            }
            other => panic!("expected SpecModeNotRunnable, got {other:?}"),
        }
        assert!(session.is_discarded());
        // And serial (the default path) is ALWAYS runnable even against a serial-only engine.
        let engine2 = conformant_engine(8).spec_modes(Some(vec!["serial".to_string()]));
        let (mut session2, _h) = Session::connect(engine2).unwrap();
        let p2 = params(8).with_spec(Some(SpecConfig::serial()));
        assert!(
            run_timed_benchmark(&mut session2, &p2).is_ok(),
            "serial is never gated"
        );
    }

    #[test]
    fn spec_divergent_echo_rejects_and_discards() {
        // NEGATIVE CONTROL: the engine echoes a DIFFERENT spec than requested — spec-never-ignored
        // rejects the leg (SpecEchoDivergence) and discards the session fail-closed.
        use bench_protocol::SpecConfig;
        let engine = conformant_engine(8).diverge_spec_echo(SpecConfig::mtp(2));
        let (mut session, _hello) = Session::connect(engine).unwrap();
        let p = params(8).with_spec(Some(SpecConfig::mtp(4)));
        let err = run_timed_benchmark(&mut session, &p).unwrap_err();
        assert!(
            matches!(err, RunnerError::SpecEchoDivergence { .. }),
            "a divergent spec echo rejects; got {err:?}"
        );
        assert!(session.is_discarded());
    }

    #[test]
    fn spec_missing_echo_rejects_and_discards() {
        // NEGATIVE CONTROL: the engine ignored the spec (no effective_spec echo at all) — rejected.
        use bench_protocol::SpecConfig;
        let engine = conformant_engine(8).suppress_spec_echo();
        let (mut session, _hello) = Session::connect(engine).unwrap();
        let p = params(8).with_spec(Some(SpecConfig::mtp(4)));
        let err = run_timed_benchmark(&mut session, &p).unwrap_err();
        match err {
            RunnerError::SpecEchoDivergence { effective, .. } => assert_eq!(effective, None),
            other => panic!("expected SpecEchoDivergence, got {other:?}"),
        }
        assert!(session.is_discarded());
    }

    // ------------------------------------------------------------------
    // H3 (cycle-3) — RunTimeout: a hung engine must not wedge benchd
    // ------------------------------------------------------------------

    #[test]
    fn run_timeout_on_hung_engine_is_not_a_hang_and_discards_session() {
        // H3 CONTROL — a hung engine (never responds to the timed decode_begin) with a RunTimeout
        // deadline armed yields `RunTimeout` (NOT a hang) and DISCARDS the session (fail-closed).
        use std::time::Duration;
        let engine = conformant_engine(8).stall_on("decode_begin");
        let (mut session, _hello) = Session::connect(engine).unwrap();
        // Arm a short deadline, then issue the timed request directly (mirrors what measure_decode
        // does inside the timed window).
        session.arm_run_deadline(std::time::Instant::now() + Duration::from_millis(20), 0.02);
        let err = session.decode_begin(&[2i64; 8]).unwrap_err();
        assert!(
            matches!(err, RunnerError::RunTimeout { .. }),
            "a hung engine yields RunTimeout, not a hang; got {err:?}"
        );
        assert!(
            session.is_discarded(),
            "the session is discarded fail-closed after a RunTimeout"
        );
    }

    #[test]
    fn decode_phase_arms_run_timeout_from_params() {
        // H3 — the timed decode phase (measure_decode via run_decode_phase_fresh) ARMS the RunTimeout
        // from `params.run_timeout`: a hung engine aborts as RunTimeout instead of wedging the phase.
        use std::time::Duration;
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, _hello) =
                Session::connect(conformant_engine(8).stall_on("decode_begin"))?;
            Ok(session)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };
        let p = params(8).with_run_timeout(Some(Duration::from_millis(20)));
        let err = run_decode_phase_fresh(&mut spawn, &mut gate, &p).unwrap_err();
        assert!(
            matches!(err, RunnerError::RunTimeout { .. }),
            "the decode phase arms the RunTimeout from params; got {err:?}"
        );
    }

    #[test]
    fn free_run_phase_arms_run_timeout_on_hung_engine() {
        // H3 — the free-run timed phase (measure_free_run_decode) is bounded too: a hung engine that
        // never returns from free_decode_run aborts as RunTimeout, session discarded.
        use std::time::Duration;
        let engine = free_run_engine(8).stall_on("free_decode_run");
        let (mut session, _hello) = Session::connect(engine).unwrap();
        let p = params(8).with_run_timeout(Some(Duration::from_millis(20)));
        let mut peak = 0.0;
        let err =
            measure_free_run_decode(&mut session, &p, VerifyMode::Verify, &mut peak).unwrap_err();
        assert!(
            matches!(err, RunnerError::RunTimeout { .. }),
            "the free-run phase arms the RunTimeout; got {err:?}"
        );
        assert!(session.is_discarded());
    }

    #[test]
    fn generous_run_timeout_does_not_trip_a_healthy_run() {
        // H3 — a healthy (conformant, fast) run finishes WELL inside a generous budget: the deadline
        // is a liveness bound, never a false trip on a normal run.
        use std::time::Duration;
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, _hello) = Session::connect(conformant_engine(16))?;
            Ok(session)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };
        let p = params(16).with_run_timeout(Some(Duration::from_secs(3600)));
        let timing = run_decode_phase_fresh(&mut spawn, &mut gate, &p).unwrap();
        assert!(timing.seconds_per_token.is_finite() && timing.seconds_per_token >= 0.0);
    }

    // -----------------------------------------------------------------------
    // W3 — `run_free_run_decode_phase_fresh`: the SCORED free-run phase seam
    // -----------------------------------------------------------------------

    #[test]
    fn free_run_phase_fresh_positive_control_carries_audit_and_series_tag() {
        // The scored seam: one fresh worker, one cool gate, one timed free-run window. The parent
        // clock is the spt source, the §3 AUDIT rides along, and the §5 series tag is v1.1.
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, _hello) =
                Session::connect(free_run_engine(8).free_run_acceptance_lengths(vec![3, 3, 2]))?;
            Ok(session)
        };
        let mut gated: Vec<String> = Vec::new();
        let mut gate = |phase: &str| -> Result<()> {
            gated.push(phase.to_string());
            Ok(())
        };
        let timing = run_free_run_decode_phase_fresh(&mut spawn, &mut gate, &params(8)).unwrap();
        assert_eq!(timing.timed_mode, TIMED_MODE_FREE_RUN_V1_1);
        assert!(timing.seconds_per_token.is_finite() && timing.seconds_per_token >= 0.0);
        // N/R = 8/3: benchd DERIVES the acceptance stats from the per-round histogram it collected.
        assert_eq!(timing.audit.acceptance_lengths(), &[3, 3, 2]);
        assert_eq!(timing.audit.rounds(), 3);
        assert_eq!(timing.audit.verified_token_count(), 8);
        assert_eq!(
            gated,
            vec!["decode".to_string()],
            "exactly ONE cool gate, on the decode phase"
        );
    }

    /// DECODE MEANS THE DECODE WINDOW (David 2026-10-07), on the real clock: a seed prefill that
    /// takes 300 ms lands in the seed window and nowhere in the decode figure, on the fresh-phase
    /// path measure-job drives and on the resident path official, calibrate-baseline and iterate
    /// drive.
    #[test]
    fn slow_seed_prefill_is_never_charged_to_decode_seconds_per_token() {
        let seed_delay = std::time::Duration::from_millis(300);
        let slow_seed = || free_run_engine(8).sleep_on("free_decode_begin", seed_delay);

        let mut spawn = || Session::connect(slow_seed()).map(|(s, _)| s);
        let mut gate = |_: &str| Ok(());
        let fresh = run_free_run_decode_phase_fresh(&mut spawn, &mut gate, &params(8)).unwrap();
        assert!(fresh.phase_window.seed_prefill_elapsed_seconds >= 0.3);
        assert_eq!(
            fresh.seconds_per_token,
            fresh.phase_window.decode_elapsed_seconds / 8.0
        );
        assert!(
            fresh.seconds_per_token * 8.0 < 0.1,
            "decode {} s/tok carries the 300 ms seed prefill",
            fresh.seconds_per_token
        );

        let (mut session, _) = Session::connect(slow_seed()).unwrap();
        let resident = run_timed_benchmark_persistent_on_session(
            &mut session,
            &mut |_: &str| Ok(()),
            &params(8),
            VerifyMode::Verify,
        )
        .unwrap();
        let w = resident.phase_window.unwrap();
        assert!(w.seed_prefill_elapsed_seconds >= 0.3);
        assert_eq!(resident.decode_elapsed_seconds, w.decode_elapsed_seconds);
        assert_eq!(
            resident.decode_seconds_per_token,
            w.decode_elapsed_seconds / 8.0
        );
        assert!(
            resident.decode_seconds_per_token * 8.0 < 0.1,
            "decode {} s/tok carries the 300 ms seed prefill",
            resident.decode_seconds_per_token
        );
    }

    #[test]
    fn free_run_phase_split_measures_both_windows_and_decode_divides_only_the_decode_window() {
        // The clock SPLIT: benchd brackets the seed prefill with `free_decode_begin` and the
        // free-run decode with `free_decode_run`, on its own clock, at the message boundary it
        // already drives.
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, _hello) =
                Session::connect(free_run_engine(8).free_run_acceptance_lengths(vec![3, 3, 2]))?;
            Ok(session)
        };
        let mut gate = |_phase: &str| -> Result<()> { Ok(()) };
        let timing = run_free_run_decode_phase_fresh(&mut spawn, &mut gate, &params(8)).unwrap();
        let w = timing.phase_window;

        // Both halves are real, measurable intervals.
        assert!(w.seed_prefill_elapsed_seconds.is_finite() && w.seed_prefill_elapsed_seconds > 0.0);
        assert!(w.decode_elapsed_seconds.is_finite() && w.decode_elapsed_seconds > 0.0);
        assert_eq!(
            timing.seconds_per_token,
            w.decode_elapsed_seconds / 8.0,
            "decode seconds per token divides the decode window, never the seed prefill"
        );
        // The token totals are benchd's OWN configured counts: the seed length and N.
        assert_eq!(w.prefill_token_total, params(8).decode_seed_tokens.len()); // The token totals are benchd's OWN configured counts: the seed length and N.
        assert_eq!(w.prefill_token_total, params(8).decode_seed_tokens.len());
        assert_eq!(w.decode_token_total, 8);

        // …and the window this branch's own track produces is REPORT-ONLY: its declared regime
        // scores decode only, so certification is not armed and cannot refuse it.
        let regime = bench_core::contract::scored_regime(
            &bench_core::contract::Contract::parse(
                br#"{"track_id":"qwen3.8-27b-mtp-v1","scored_batch_size":1,
                     "prefill_gain_exponent":0.0,"decode_gain_exponent":1.0}"#,
            )
            .unwrap(),
            bench_core::constants::TRACK_ID,
        )
        .unwrap()
        .value;
        let certified = bench_core::prefill_window::certify_prefill_window(
            bench_core::constants::TRACK_ID,
            &regime,
            Some(w),
            None,
        )
        .expect("the existing track can never be refused by prefill certification");
        assert!(!certified.is_certified());
        assert_eq!(certified.window(), Some(w));
    }

    #[test]
    fn free_run_phase_fresh_refuses_uncapable_engine_before_the_gate_and_clock() {
        // §2.1 — an engine that does not advertise `free_run_decode` is refused BEFORE the cool gate
        // fires and before any timed window opens: no gate time is spent and no clock is discarded.
        let mut spawn = || -> Result<Session<MockEngine>> {
            // conformant_engine() does NOT call `.free_run_capable()`.
            let (session, _hello) = Session::connect(conformant_engine(8))?;
            Ok(session)
        };
        let mut gate_calls = 0usize;
        let mut gate = |_p: &str| -> Result<()> {
            gate_calls += 1;
            Ok(())
        };
        let err = run_free_run_decode_phase_fresh(&mut spawn, &mut gate, &params(8)).unwrap_err();
        match err {
            RunnerError::CapabilityNotAdvertised { capability } => {
                assert_eq!(capability, "free_run_decode");
            }
            other => panic!("expected CapabilityNotAdvertised, got {other:?}"),
        }
        assert_eq!(
            gate_calls, 0,
            "the refusal precedes the cool gate (and the clock)"
        );
    }

    #[test]
    fn free_run_phase_fresh_arms_run_timeout_from_params() {
        // §2.2 RunTimeout, armed through the SCORED seam: a hung engine aborts the phase instead of
        // wedging benchd. (The liveness bound is `N × band-ceiling × margin`, set by the caller.)
        use std::time::Duration;
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, _hello) =
                Session::connect(free_run_engine(8).stall_on("free_decode_run"))?;
            Ok(session)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };
        let p = params(8).with_run_timeout(Some(Duration::from_millis(20)));
        let err = run_free_run_decode_phase_fresh(&mut spawn, &mut gate, &p).unwrap_err();
        assert!(
            matches!(err, RunnerError::RunTimeout { .. }),
            "the free-run scored phase arms the RunTimeout from params; got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // NOOP-RATE TimeOnly — `run_free_run_decode_phase_fresh_time_only`: the CONFINED,
    // report-only, mismatch-tolerating single-stream free-run path (measure-noop only).
    // -----------------------------------------------------------------------

    /// A free-run engine that commits a stream diverging from `params(n)`'s oracle at exactly one
    /// step, while its SEED forward still matches — so the divergence is a §2.7 committed-token
    /// mismatch (not a seed/count/consistency fault), which is precisely what a noop RATE run must
    /// tolerate and a scored run must reject.
    fn free_run_engine_with_one_wrong_token(n: usize, wrong_step: usize) -> MockEngine {
        let mut committed = oracle_decode_tokens(n);
        committed[wrong_step] += 1; // one divergent committed token; seed unchanged
        MockEngine::new()
            .oracle_tokens(PREFILL_TOKEN, SEED_TOKEN, committed)
            .free_run_capable()
    }

    #[test]
    fn confinement_scored_verify_aborts_but_time_only_tolerates_same_mismatch() {
        // ★ CONFINEMENT (1) + tolerance (a): on the IDENTICAL engine + params, the SCORED
        // (Verify) single-stream free-run phase ABORTS with TokenMismatch — abort behavior is
        // preserved unchanged for the scored path — while the noop-RATE (TimeOnly) sibling
        // TOLERATES the divergence and still returns a timing. Only measure-noop opts into TimeOnly.
        let build = || free_run_engine_with_one_wrong_token(8, 3);
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };

        // SCORED path — Verify — must abort at the divergent step.
        let mut spawn_v = || -> Result<Session<MockEngine>> {
            let (session, _hello) = Session::connect(build())?;
            Ok(session)
        };
        let err = run_free_run_decode_phase_fresh(&mut spawn_v, &mut gate, &params(8)).unwrap_err();
        match err {
            RunnerError::TokenMismatch {
                step,
                expected,
                actual,
                ..
            } => {
                assert_eq!(step, 3, "aborts at exactly the divergent committed step");
                assert_eq!(expected, oracle_decode_tokens(8)[3]);
                assert_eq!(actual, oracle_decode_tokens(8)[3] + 1);
            }
            other => panic!("scored Verify path must reject a wrong token; got {other:?}"),
        }

        // NOOP-RATE path — TimeOnly — must tolerate the SAME mismatch and return a timing.
        let mut spawn_t = || -> Result<Session<MockEngine>> {
            let (session, _hello) = Session::connect(build())?;
            Ok(session)
        };
        let timing =
            run_free_run_decode_phase_fresh_time_only(&mut spawn_t, &mut gate, &params(8)).unwrap();
        assert!(
            timing.seconds_per_token.is_finite() && timing.seconds_per_token >= 0.0,
            "TimeOnly returns a parent-clock rate despite the token divergence"
        );
        // The full non-abort path still ran: §2.6 barrier + audit assembly are unchanged.
        assert_eq!(timing.timed_mode, TIMED_MODE_FREE_RUN_V1_1);
        assert_eq!(timing.audit.verified_token_count(), 8);
    }

    #[test]
    fn confinement_time_only_and_verify_share_the_timing_path_on_a_conformant_engine() {
        // ★ SAME TIMING (2): on a CONFORMANT engine (no mismatch to gate), the two modes traverse
        // the SAME path and produce the SAME derived results — the mode gates ONLY the abort
        // branch, so with nothing to abort the outputs coincide. Both compute seconds_per_token the
        // same way (parent wall-clock ÷ N committed tokens); the wall-clock magnitude itself is
        // ~0 against the in-process mock, so we assert the DERIVED, deterministic outputs are equal.
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };

        let mut spawn_v = || -> Result<Session<MockEngine>> {
            let (session, _hello) =
                Session::connect(free_run_engine(8).free_run_acceptance_lengths(vec![3, 3, 2]))?;
            Ok(session)
        };
        let v = run_free_run_decode_phase_fresh(&mut spawn_v, &mut gate, &params(8)).unwrap();

        let mut spawn_t = || -> Result<Session<MockEngine>> {
            let (session, _hello) =
                Session::connect(free_run_engine(8).free_run_acceptance_lengths(vec![3, 3, 2]))?;
            Ok(session)
        };
        let t =
            run_free_run_decode_phase_fresh_time_only(&mut spawn_t, &mut gate, &params(8)).unwrap();

        assert_eq!(v.timed_mode, t.timed_mode, "same series tag");
        assert_eq!(
            v.audit.acceptance_lengths(),
            t.audit.acceptance_lengths(),
            "same histogram: the audit assembly is identical across modes"
        );
        assert_eq!(v.audit.rounds(), t.audit.rounds());
        assert_eq!(
            v.audit.verified_token_count(),
            t.audit.verified_token_count(),
            "same N committed tokens — the seconds_per_token divisor is identical"
        );
        assert_eq!(v.peak_ram_gb, t.peak_ram_gb);
        assert!(v.seconds_per_token.is_finite() && t.seconds_per_token.is_finite());
    }

    // -----------------------------------------------------------------------
    // v1.2 COHORT — `run_batched_free_run_decode_phase_fresh`: the batched seam
    // -----------------------------------------------------------------------

    /// Slot-distinct cohort oracle: slot `s` seeds to `600 + s` and continues with
    /// `700 + s*1000 + i` — distinct across slots AND steps so a wrong-slot/wrong-step test can
    /// target exactly one cell of the B x N rectangle.
    fn cohort_oracle_slots(b: usize, n: usize) -> Vec<(i64, Vec<i64>)> {
        (0..b)
            .map(|s| {
                (
                    600 + s as i64,
                    (0..n as i64).map(|i| 700 + s as i64 * 1000 + i).collect(),
                )
            })
            .collect()
    }

    /// Cohort params matching `cohort_oracle_slots(b, n)`.
    fn cohort_params(b: usize, n: usize) -> CohortTimingParams {
        let streams = cohort_oracle_slots(b, n)
            .into_iter()
            .map(|(seed, tokens)| CohortStreamParams {
                decode_seed_tokens: vec![2; 8],
                expected_decode_seed_token: seed,
                expected_decode_tokens: tokens,
            })
            .collect();
        CohortTimingParams::new(streams, n)
    }

    /// A conformant batch-capable engine whose cohort tokens exactly match
    /// `cohort_params(b, n)`'s oracle, advertising `max_batch_size = 8`.
    fn cohort_engine(b: usize, n: usize) -> MockEngine {
        MockEngine::new()
            .batched_free_run_capable(8)
            .cohort_oracle(cohort_oracle_slots(b, n))
    }

    /// A no-op cool gate that records how often it fired (for the pre-gate refusal proofs).
    fn counting_gate(count: &mut usize) -> impl FnMut(&str) -> Result<()> + '_ {
        move |_p: &str| {
            *count += 1;
            Ok(())
        }
    }

    #[test]
    fn batched_free_run_phase_fresh_positive_control() {
        // The batched scored seam: one fresh worker, one cool gate, ONE timed window covering the
        // whole B=8 cohort. The parent clock over the full window, divided by B*N, is the spt; the
        // cohort AUDIT rides along; the series tag carries the batch width (D5).
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, hello) =
                Session::connect(cohort_engine(8, 4).free_run_acceptance_lengths(vec![3, 1]))?;
            assert!(hello.supports_batched_free_run_decode());
            assert_eq!(hello.max_batch_size, Some(8));
            Ok(session)
        };
        let mut gated: Vec<String> = Vec::new();
        let mut gate = |phase: &str| -> Result<()> {
            gated.push(phase.to_string());
            Ok(())
        };
        let timing =
            run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &cohort_params(8, 4))
                .unwrap();
        assert_eq!(timing.timed_mode, "batched_free_run_v1_2_b8");
        assert_eq!(timing.batch_size, 8);
        assert!(timing.seconds_per_token.is_finite() && timing.seconds_per_token >= 0.0);
        // spt is the DECODE WINDOW / (B * N) = 32. The seed prefill is not in it.
        assert_eq!(
            timing.seconds_per_token,
            timing.decode_elapsed_seconds / 32.0,
            "cohort spt must be the decode window over B * N"
        );
        assert_eq!(timing.decode_token_total, 32, "B * N = 8 * 4");
        assert!(
            timing.prefill_elapsed_seconds >= 0.0 && timing.prefill_elapsed_seconds.is_finite()
        );
        // 8 streams x 8 seed tokens each (`cohort_params`' fixed `decode_seed_tokens: vec![2; 8]`).
        assert_eq!(timing.prefill_token_total, 64, "8 streams x 8 seed tokens");
        assert_eq!(timing.audit.batch_size(), 8);
        assert_eq!(timing.audit.cohort_committed_total(), 32);
        assert_eq!(timing.audit.base().acceptance_lengths(), &[3, 1]);
        assert_eq!(timing.audit.rounds(), 2);
        assert_eq!(timing.peak_ram_gb, 20.25);
        assert_eq!(
            gated,
            vec!["decode".to_string()],
            "exactly ONE cool gate for the whole cohort window"
        );
    }

    #[test]
    fn batched_free_run_series_tag_carries_the_batch_width() {
        // D5 — the series tag is per batch point, so the existing string-equality series fence
        // refuses a cross-batch comparison with zero new gate code.
        for b in [1usize, 2, 8] {
            let mut spawn = || -> Result<Session<MockEngine>> {
                let (session, _hello) = Session::connect(cohort_engine(b, 4))?;
                Ok(session)
            };
            let mut gate = |_p: &str| -> Result<()> { Ok(()) };
            let timing = run_batched_free_run_decode_phase_fresh(
                &mut spawn,
                &mut gate,
                &cohort_params(b, 4),
            )
            .unwrap();
            assert_eq!(timing.timed_mode, format!("batched_free_run_v1_2_b{b}"));
        }
    }

    #[test]
    fn batched_free_run_phase_split_windows_are_contiguous_no_untimed_gap() {
        // ANTI-CHEAT (Gemma cohort scoring, David ruling 2026-08-23) — a STRUCTURAL proof that the
        // prefill/decode split introduces no untimed gap: wrap the WHOLE call in an outer clock
        // (started before the cool gate, stopped after the function returns) and prove the two
        // inner windows account for essentially all of it. If a future edit inserted untimed work
        // between the prefill window closing and the decode window opening (or anywhere inside
        // either window), the outer-minus-inner gap would grow well past this tolerance — against
        // a near-instant in-process mock, the honest gap is microseconds of call overhead.
        let mut spawn = || -> Result<Session<MockEngine>> {
            Session::connect(cohort_engine(8, 4).free_run_acceptance_lengths(vec![3, 1]))
                .map(|(s, _hello)| s)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };
        let outer_start = Instant::now();
        let timing =
            run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &cohort_params(8, 4))
                .unwrap();
        let outer_elapsed = outer_start.elapsed().as_secs_f64();

        let inner_total = timing.prefill_elapsed_seconds + timing.decode_elapsed_seconds;
        assert!(
            inner_total <= outer_elapsed,
            "the two inner windows ({inner_total}s) cannot exceed the outer wall clock \
             ({outer_elapsed}s) that contains spawn + both windows"
        );
        let untimed_gap = outer_elapsed - inner_total;
        assert!(
            untimed_gap < 0.25,
            "untimed gap between spawn/cool-gate overhead and the two charged windows was \
             {untimed_gap}s — far more than the sub-millisecond overhead an in-process mock call \
             should cost; this would indicate hidden untimed work outside the two windows"
        );
    }

    #[test]
    fn batched_free_run_prefill_token_total_sums_per_stream_seed_lengths() {
        // COMPOSITE — "the 8 seeds' prompt tokens" (David's ruling): the prefill token total is the
        // SUM of every stream's seed length, not a per-stream figure and not B*(one length).
        let mut params = cohort_params(8, 4);
        for (slot, stream) in params.streams.iter_mut().enumerate() {
            // Distinct per-slot lengths so a bug that reads only slot 0 (or multiplies by B) is
            // caught rather than passing by coincidence on a uniform fixture.
            stream.decode_seed_tokens = vec![9i64; slot + 1];
        }
        let expected_total: usize = (1..=8).sum(); // 1+2+...+8 = 36
        let mut spawn = || -> Result<Session<MockEngine>> {
            Session::connect(cohort_engine(8, 4)).map(|(s, _hello)| s)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };
        let timing = run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &params)
            .expect("conformant cohort engine");
        assert_eq!(timing.prefill_token_total, expected_total);
        assert_eq!(
            timing.decode_token_total, 32,
            "B * N = 8 * 4, unaffected by seed length"
        );
    }

    #[test]
    fn per_stream_ns_vectors_carried_verbatim_when_advertised_and_sent() {
        // G1 (per-stream arm-fill carry) — an engine that advertises `per_stream_timing` and
        // sends the two per-slot ns vectors gets them carried VERBATIM onto the phase timing:
        // slot-distinct values so an off-by-one/reorder/summarize bug cannot pass by coincidence.
        let prefill_ns: Vec<u64> = (0..8u64).map(|s| 1_000_001 + s * 13).collect();
        let decode_ns: Vec<u64> = (0..8u64).map(|s| 2_000_003 + s * 7).collect();
        let (p, d) = (prefill_ns.clone(), decode_ns.clone());
        let mut spawn = move || -> Result<Session<MockEngine>> {
            let (session, hello) = Session::connect(
                cohort_engine(8, 4).per_stream_timing_capable(p.clone(), d.clone()),
            )?;
            // The new accessor, both on the decoded hello and on the live session.
            assert!(hello.supports_per_stream_timing());
            assert!(session.supports_per_stream_timing());
            // Layered ON TOP of the batch capability, not replacing it.
            assert!(hello.supports_batched_free_run_decode());
            Ok(session)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };
        let timing =
            run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &cohort_params(8, 4))
                .unwrap();
        assert!(timing.per_stream_timing_advertised);
        assert_eq!(timing.prefill_ns_by_stream, Some(prefill_ns));
        assert_eq!(timing.decode_ns_by_stream, Some(decode_ns));
        // G3 — K_slot is read from the RESPONSE rectangle (here a conformant 8 x 4).
        assert_eq!(timing.tokens_len_by_stream, vec![4usize; 8]);
    }

    #[test]
    fn per_stream_fields_absent_when_capability_not_advertised() {
        // A plain batch-capable engine (no `per_stream_timing`, no vectors on the wire): the
        // carry reports exactly that — flag false, both vectors None. No refusal, no synthesis.
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, hello) = Session::connect(cohort_engine(8, 4))?;
            assert!(!hello.supports_per_stream_timing());
            assert!(!session.supports_per_stream_timing());
            Ok(session)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };
        let timing =
            run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &cohort_params(8, 4))
                .unwrap();
        assert!(!timing.per_stream_timing_advertised);
        assert_eq!(timing.prefill_ns_by_stream, None);
        assert_eq!(timing.decode_ns_by_stream, None);
        // K_slot is carried regardless — it comes from the committed rectangle the quadruple
        // validated, not from the per-stream capability.
        assert_eq!(timing.tokens_len_by_stream, vec![4usize; 8]);
    }

    #[test]
    fn tokens_len_by_stream_is_read_as_received_never_reconstructed_as_n_by_b() {
        // G3 — the carry is a function of the RESPONSE rectangle alone: a deliberately
        // NON-RECTANGULAR input must come back as-received ([3, 1, 5]), proving the vector is
        // per-slot `len()`s and not a `[N; B]` reconstruction from the request parameters (any
        // such reconstruction would return a uniform vector here and fail this assertion).
        let jagged = vec![vec![1i64, 2, 3], vec![4], vec![5, 6, 7, 8, 9]];
        assert_eq!(cohort_tokens_len_by_stream(&jagged), vec![3usize, 1, 5]);
        let empty_slot = vec![vec![], vec![7i64]];
        assert_eq!(cohort_tokens_len_by_stream(&empty_slot), vec![0usize, 1]);
    }

    #[test]
    fn per_stream_vectors_are_inert_cargo_enforced_metric_untouched() {
        // #182 doctrine (enforced-surface trace) — the carried engine-reported ns are ABSURD
        // (hours per slot): if they ever fed the enforced assembly, `seconds_per_token` would
        // explode. It must not — it stays the parent-clock decode window over B * N.
        let hour_ns = 3_600_000_000_000u64;
        let mut spawn = move || -> Result<Session<MockEngine>> {
            Session::connect(
                cohort_engine(8, 4)
                    .per_stream_timing_capable(vec![hour_ns; 8], vec![2 * hour_ns; 8]),
            )
            .map(|(s, _hello)| s)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };
        let timing =
            run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &cohort_params(8, 4))
                .unwrap();
        assert_eq!(
            timing.seconds_per_token,
            timing.decode_elapsed_seconds / 32.0,
            "spt must remain the parent-clock decode window over B * N"
        );
        // And the parent clock is what it is — an in-process mock run takes well under a minute,
        // while the engine CLAIMED 3 hours per slot. Engine-reported time never entered.
        assert!(
            timing.decode_elapsed_seconds + timing.prefill_elapsed_seconds < 60.0,
            "the parent windows ({} s, {} s) must be the parent clock, not the engine's claimed \
             hours",
            timing.prefill_elapsed_seconds,
            timing.decode_elapsed_seconds
        );
        assert_eq!(timing.decode_ns_by_stream, Some(vec![2 * hour_ns; 8]));
    }

    #[test]
    fn batched_free_run_refused_without_capability_before_gate_and_clock() {
        // The cohort form is gated by its OWN capability: a v1.1 engine advertising only
        // `free_run_decode` is refused BEFORE the cool gate fires and before any timed window
        // opens — never silently narrowed to the single-stream regime.
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, hello) = Session::connect(free_run_engine(4))?;
            assert!(hello.supports_free_run_decode());
            assert!(!hello.supports_batched_free_run_decode());
            Ok(session)
        };
        let mut gate_calls = 0usize;
        let err = {
            let mut gate = counting_gate(&mut gate_calls);
            run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &cohort_params(8, 4))
                .unwrap_err()
        };
        match err {
            RunnerError::CapabilityNotAdvertised { capability } => {
                assert_eq!(capability, "batched_free_run_decode");
            }
            other => panic!("expected CapabilityNotAdvertised, got {other:?}"),
        }
        assert_eq!(
            gate_calls, 0,
            "the refusal precedes the cool gate (and the clock)"
        );
    }

    #[test]
    fn batched_free_run_refuses_overwide_cohort_pre_gpu() {
        // An advertised `hello.max_batch_size` narrower than the requested B is refused at the
        // same pre-gate, pre-clock point as the capability check.
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, _hello) = Session::connect(
                MockEngine::new()
                    .batched_free_run_capable(4)
                    .cohort_oracle(cohort_oracle_slots(8, 4)),
            )?;
            Ok(session)
        };
        let mut gate_calls = 0usize;
        let err = {
            let mut gate = counting_gate(&mut gate_calls);
            run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &cohort_params(8, 4))
                .unwrap_err()
        };
        match err {
            RunnerError::BatchWidthExceedsEngineMax {
                requested,
                max_batch_size,
            } => {
                assert_eq!(requested, 8);
                assert_eq!(max_batch_size, 4);
            }
            other => panic!("expected BatchWidthExceedsEngineMax, got {other:?}"),
        }
        assert_eq!(
            gate_calls, 0,
            "the width refusal is pre-GPU: before the cool gate"
        );
    }

    #[test]
    fn batched_free_run_batch_echo_divergence_discards_leg() {
        // BATCH-NEVER-IGNORED: a divergent `effective_batch_size` echo discards the leg
        // fail-closed — a silently narrowed cohort can never be sealed as a B=8 measurement.
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, _hello) = Session::connect(cohort_engine(8, 4).diverge_batch_echo(4))?;
            Ok(session)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };
        let err =
            run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &cohort_params(8, 4))
                .unwrap_err();
        match err {
            RunnerError::BatchEchoDivergence {
                requested,
                effective,
            } => {
                assert_eq!(requested, 8);
                assert_eq!(effective, Some(4));
            }
            other => panic!("expected BatchEchoDivergence, got {other:?}"),
        }

        // A MISSING echo is the same divergence (the identity was ignored, not merely mis-echoed).
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, _hello) = Session::connect(cohort_engine(8, 4).suppress_batch_echo())?;
            Ok(session)
        };
        let err =
            run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &cohort_params(8, 4))
                .unwrap_err();
        match err {
            RunnerError::BatchEchoDivergence {
                requested,
                effective,
            } => {
                assert_eq!(requested, 8);
                assert_eq!(effective, None);
            }
            other => panic!("expected BatchEchoDivergence, got {other:?}"),
        }
    }

    #[test]
    fn batched_free_run_wrong_cohort_width_is_consistency_fault() {
        // An engine returning a 7-stream rectangle against a B=8 request is an ACCOUNTING lie of
        // the same class as a doctored histogram: a typed consistency fault, refused fail-closed.
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, _hello) = Session::connect(cohort_engine(8, 4).cohort_width_override(7))?;
            Ok(session)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };
        let err =
            run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &cohort_params(8, 4))
                .unwrap_err();
        match err {
            RunnerError::FreeRunConsistency { detail } => {
                assert!(
                    detail.contains("7 token streams") && detail.contains("B=8"),
                    "detail: {detail}"
                );
            }
            other => panic!("expected FreeRunConsistency, got {other:?}"),
        }
    }

    #[test]
    fn batched_free_run_surfaces_committed_tokens_unjudged_no_inline_die() {
        // (b) admission — the runner NO LONGER dies inline on a committed token that diverges from
        // the static tape. Under David's blanket-10% ruling the token-correctness decision moved to
        // benchd's post-run TRUSTED-ORACLE tolerance gate, so a slot that diverges from the tape is
        // surfaced UNJUDGED in `tokens_by_stream` rather than raising a `TokenMismatch`. (The SHAPE
        // and the SEED tokens are still oracle-checked in the runner — see the seed test below.)
        let mut slots = cohort_oracle_slots(8, 4);
        slots[5].1[2] = 999_999; // slot 5, step 2 diverges from the tape token 5702
        let divergent_slots = slots.clone();
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, _hello) = Session::connect(
                MockEngine::new()
                    .batched_free_run_capable(8)
                    .cohort_oracle(slots.clone()),
            )?;
            Ok(session)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };
        // The tape (`cohort_params`) still expects 5702 at slot 5 step 2, but the engine committed
        // 999_999 there. Pre-(b) this raised TokenMismatch; now the phase completes and surfaces the
        // committed rectangle verbatim for benchd to judge.
        let timing =
            run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &cohort_params(8, 4))
                .expect("(b): a tape-divergent committed token no longer dies inline");
        assert_eq!(timing.tokens_by_stream.len(), 8, "B streams surfaced");
        for (slot, stream) in timing.tokens_by_stream.iter().enumerate() {
            assert_eq!(stream.len(), 4, "N tokens per stream");
            // Surfaced verbatim — including the divergent slot-5 token benchd will judge.
            assert_eq!(stream, &divergent_slots[slot].1);
        }
        assert_eq!(
            timing.tokens_by_stream[5][2], 999_999,
            "divergence surfaced, not died on"
        );
    }

    #[test]
    fn batched_free_run_wrong_seed_token_in_one_slot_hard_fails() {
        // Every slot's seed forward is oracle-checked, in slot order.
        let mut slots = cohort_oracle_slots(8, 4);
        slots[3].0 = 999_999; // slot 3's seed diverges from oracle 603
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, _hello) = Session::connect(
                MockEngine::new()
                    .batched_free_run_capable(8)
                    .cohort_oracle(slots.clone()),
            )?;
            Ok(session)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };
        let err =
            run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &cohort_params(8, 4))
                .unwrap_err();
        match err {
            RunnerError::TokenMismatch {
                label,
                step,
                expected,
                ..
            } => {
                assert!(label.contains("stream 3"), "label: {label}");
                assert_eq!(step, 0);
                assert_eq!(expected, 603);
            }
            other => panic!("expected TokenMismatch, got {other:?}"),
        }
    }

    #[test]
    fn batched_completed_work_is_scalar_rounds_plus_one_never_scaled_by_b() {
        // NORMATIVE: a round is ONE engine forward regardless of B, so the batched phase's
        // `completed_work` barrier is the SCALAR R + 1 — an engine reporting B*R + 1 (counting
        // stream-rounds) fails the quadruple. With R = 2 at B = 8: scalar is 3; the stream-round
        // counter would be 17, off by (B-1)*R = 14.
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, _hello) = Session::connect(
                cohort_engine(8, 4)
                    .free_run_acceptance_lengths(vec![3, 1])
                    .completed_work_delta(14),
            )?;
            Ok(session)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };
        let err =
            run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &cohort_params(8, 4))
                .unwrap_err();
        match err {
            RunnerError::FreeRunConsistency { detail } => {
                assert!(
                    detail.contains("completed_work 17"),
                    "the barrier must reject the stream-round counter; detail: {detail}"
                );
            }
            other => panic!("expected FreeRunConsistency, got {other:?}"),
        }
    }

    #[test]
    fn batched_free_run_phase_arms_run_timeout_from_params() {
        // The batched window is RunTimeout-bounded exactly like the single-stream one: a hung
        // engine aborts the phase instead of wedging benchd inside the cohort window.
        use std::time::Duration;
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, _hello) =
                Session::connect(cohort_engine(8, 4).stall_on("free_decode_run"))?;
            Ok(session)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };
        let p = cohort_params(8, 4).with_run_timeout(Some(Duration::from_millis(20)));
        let err = run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &p).unwrap_err();
        assert!(
            matches!(err, RunnerError::RunTimeout { .. }),
            "the batched scored phase arms the RunTimeout from params; got {err:?}"
        );
    }

    #[test]
    fn batched_free_run_seals_safe_depth_clamp_reason_keys_verbatim() {
        // The clamp-reason histogram is what makes "did it actually speculate?" checkable: sealed
        // VERBATIM into the audit when reported, empty when the engine has nothing to report.
        let reasons: std::collections::BTreeMap<String, u32> = [
            ("automatic_rectangular_limit".to_string(), 2u32),
            ("tail_depth".to_string(), 3u32),
        ]
        .into_iter()
        .collect();
        let expected = reasons
            .iter()
            .map(|(key, count)| (key.clone(), u64::from(*count)))
            .collect();
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, _hello) =
                Session::connect(cohort_engine(8, 4).cohort_depth_clamp_reasons(reasons.clone()))?;
            Ok(session)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };
        let timing =
            run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &cohort_params(8, 4))
                .unwrap();
        assert_eq!(timing.audit.depth_clamp_reasons(), &expected);

        // Omitted histogram (nothing to report) ⇒ sealed empty, not refused: audit-only field.
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, _hello) = Session::connect(cohort_engine(8, 4))?;
            Ok(session)
        };
        let timing =
            run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &cohort_params(8, 4))
                .unwrap();
        assert!(timing.audit.depth_clamp_reasons().is_empty());
    }

    // -----------------------------------------------------------------------
    // MERGE GATE — B=1 through the v1.2 cohort verbs ≡ the v1.1 single-stream verbs
    // -----------------------------------------------------------------------

    #[test]
    fn merge_gate_b1_through_batched_verbs_matches_v1_1_behavior() {
        // D6 consequence, as a TEST not a claim: B = 1 through the new verbs must behave
        // identically to v1.1 — same oracle discipline, same consistency verdicts, same audit
        // statistics — so the batched path can never quietly become a second, looser regime.
        //
        // POSITIVE HALF: identical acceptance data yields an identical audit (the shared
        // `audit_spec_*` base) and an identical spt DENOMINATOR (B*N == N at B=1).
        let accept = vec![3, 3, 2];
        let (mut v11_session, _h) =
            Session::connect(free_run_engine(8).free_run_acceptance_lengths(accept.clone()))
                .unwrap();
        let mut peak = 0.0;
        let v11_audit =
            measure_free_run_decode(&mut v11_session, &params(8), VerifyMode::Verify, &mut peak)
                .unwrap()
                .audit;

        // The B=1 cohort runs the SAME oracle stream: seed 600, tokens 700..707.
        let mut b1 = cohort_params(1, 8);
        b1.streams[0].decode_seed_tokens = vec![2; 8];
        let mut spawn = || -> Result<Session<MockEngine>> {
            let (session, _hello) =
                Session::connect(cohort_engine(1, 8).free_run_acceptance_lengths(accept.clone()))?;
            Ok(session)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };
        let timing = run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &b1).unwrap();
        assert_eq!(
            timing.audit.base(),
            &v11_audit,
            "identical audit base at B=1"
        );
        assert_eq!(timing.audit.cohort_committed_total(), 8, "B*N == N at B=1");
        assert_eq!(
            timing.seconds_per_token,
            timing.decode_elapsed_seconds / 8.0,
            "B=1 cohort spt divides the decode window by the same N"
        );
        assert_eq!(timing.decode_token_total, 8, "B * N = 1 * 8");
        // The series tags DIFFER by design (b1 vs v1_1): behavior is identical, but the numbers
        // stay fence-separated — a b1 number can never band or rank against a v1.1 number.
        assert_eq!(timing.timed_mode, "batched_free_run_v1_2_b1");
        assert!(!bench_core::free_run::timed_modes_comparable(
            &timing.timed_mode,
            TIMED_MODE_FREE_RUN_V1_1
        ));

        // NEGATIVE HALF: the same doctored histogram (sum != N) is rejected in BOTH regimes with
        // the SAME consistency verdict.
        let doctored = vec![3u32, 3, 3];
        let (mut v11_bad, _h) =
            Session::connect(free_run_engine(8).free_run_acceptance_lengths(doctored.clone()))
                .unwrap();
        let v11_err =
            measure_free_run_decode(&mut v11_bad, &params(8), VerifyMode::Verify, &mut peak)
                .unwrap_err();
        let mut spawn_bad = || -> Result<Session<MockEngine>> {
            let (session, _hello) = Session::connect(
                cohort_engine(1, 8).free_run_acceptance_lengths(doctored.clone()),
            )?;
            Ok(session)
        };
        let b1_err =
            run_batched_free_run_decode_phase_fresh(&mut spawn_bad, &mut gate, &b1).unwrap_err();
        match (&v11_err, &b1_err) {
            (
                RunnerError::FreeRunConsistency { detail: v11_detail },
                RunnerError::FreeRunConsistency { detail: b1_detail },
            ) => {
                assert_eq!(
                    v11_detail, b1_detail,
                    "B=1 cohort and v1.1 must reject a doctored histogram identically"
                );
            }
            other => panic!("expected FreeRunConsistency from both regimes, got {other:?}"),
        }
    }

    #[test]
    fn batched_free_run_validates_cohort_shape_before_spawning() {
        // Fail-fast guards run BEFORE any worker is spawned: an empty cohort, a declared width
        // that disagrees with the streams, and a short per-slot oracle are all refused pre-spawn.
        let mut spawned = 0usize;
        let mut spawn = || -> Result<Session<MockEngine>> {
            spawned += 1;
            let (session, _hello) = Session::connect(cohort_engine(8, 4))?;
            Ok(session)
        };
        let mut gate = |_p: &str| -> Result<()> { Ok(()) };

        let empty = CohortTimingParams::new(Vec::new(), 4);
        let err =
            run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &empty).unwrap_err();
        assert!(matches!(&err, RunnerError::Protocol(m) if m.contains("must not be empty")));

        let mut mismatched = cohort_params(8, 4);
        mismatched.batch_size = 4; // declared width disagrees with the 8 streams
        let err = run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &mismatched)
            .unwrap_err();
        assert!(matches!(&err, RunnerError::Protocol(m) if m.contains("declared batch_size 4")));

        let mut short_oracle = cohort_params(8, 4);
        short_oracle.streams[6].expected_decode_tokens.truncate(2);
        let err = run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &short_oracle)
            .unwrap_err();
        assert!(
            matches!(&err, RunnerError::Protocol(m) if m.contains("stream 6") && m.contains("need at least 4"))
        );

        assert_eq!(spawned, 0, "shape guards precede every spawn");
    }

    #[test]
    fn free_run_short_token_array_is_a_consistency_fault_not_infra() {
        // §2.4 — a response carrying FEWER than N committed tokens is an ACCOUNTING lie (the same
        // class as a doctored histogram), so it raises `FreeRunConsistency`, not a generic protocol
        // error. Previously the sign of the mismatch decided the class; now both directions agree.
        let engine = free_run_engine(8).free_run_totals(8, 0, 8);
        let (mut session, _hello) = Session::connect(engine).unwrap();
        // Ask for 9 committed tokens while the oracle/engine only materializes 8.
        let mut p = params(9);
        p.expected_decode_tokens = oracle_decode_tokens(9);
        let mut peak = 0.0;
        let err =
            measure_free_run_decode(&mut session, &p, VerifyMode::Verify, &mut peak).unwrap_err();
        match err {
            RunnerError::FreeRunConsistency { detail } => {
                assert!(detail.contains("committed tokens"), "detail: {detail}");
            }
            other => panic!("expected FreeRunConsistency, got {other:?}"),
        }
    }
}
