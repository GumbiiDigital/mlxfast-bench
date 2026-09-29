//! B-2 — `benchd` OFFICIAL benchmark path, at parity with the Swift official run
//! (`QwenRuntimeBenchmark.benchmarkWithWorker`, main.swift official gating, benchmark.sh).
//!
//! The official path diverges from the local checked-timing path (`crate::iterate`) in
//! several load-bearing ways, all corrected from the original steer against the read-first
//! Swift map:
//!
//! 1. **Timed-runs FIRST, then gates** (cold-path / anti-memoization defense,
//!    Constants.swift:82-88): prefill → decode → floor/band/finite evaluation →
//!    correctness. The timed phases run before any correctness so the measured path is
//!    never warmed by the gates (QwenRuntimeBenchmark.swift:466-558 precede :560-581).
//! 2. **THREE fresh sandboxed workers** — a dedicated `RuntimeWorkerClient` per phase
//!    (prefill :469-484, decode :486-503, correctness :565-581), each closed/reaped before
//!    the next. No shared session, no warm caches across phases.
//! 3. **Full correctness set** — base cases + anchors + free_run (+ behavior/GPQA on the
//!    GPU path): `checkGates == true` ⇒ `caseCount = totalCorrectnessCaseCount`,
//!    `gates = correctnessGates` (QwenRuntimeCorrectness.swift:351/435). Contrast local's
//!    base-cases-only default.
//! 4. **128-step decode window + benchmark-ORACLE checks** (Constants.swift:70, seed 512):
//!    every timed token is verified against `golden.benchmark.expected_*`; a corrupted
//!    oracle FAILS official (`BenchmarkTokenMismatchError`) — the failure class the local
//!    path structurally cannot test.
//! 5. **Official gating** — 0.95 speedup floors, prefill band ±5% / decode +2%−5%, and a
//!    non-finite score fails, evaluated BEFORE correctness (Score.swift:50-126).
//! 6. **Sealing/integrity** — the sealed stdout payload is the coarsened score (2-sig-fig
//!    diagnostics, ranking fields untouched); the `metrics.commit` carries the resolved
//!    commit identifier (official-only).
//!
//! The SANDBOX fail-closed spawn + Seatbelt profile live in `bench_runner::sandbox`; this
//! module owns the phase orchestration + scoring. The REAL timed measurement and the
//! end-to-end both-sides run are B-3 on the GPU box — this module is macOS-buildable and
//! unit-tested against a stub `MockEngine` (no real engine, no GPU).

use bench_core::conformance::{run_conformance, ConformanceReport, CorrectnessScope};
use bench_core::constants::{AcceptanceBands, WindowShape};
use bench_core::golden::GoldenFixture;
use bench_core::score::{evaluate_timed_run, ScoringWeights, SpeedupFloors};
use bench_protocol::SpecConfig;
use bench_runner::{
    run_timed_benchmark_persistent_on_session, LineTransport, RunnerError, Session, TimingParams,
    TimingResult, VerifyMode,
};

use crate::iterate::{
    apply_timing_metrics, base_metrics, failed_metrics, failed_score, finite_nonneg,
    first_conformance_failure, FailedAudit, FailureReport, Mode, RunDigests, ScoringInputs,
    SessionEngine,
};
use crate::score::{ScoreMetrics, ScorePayload};

/// Seal the BOARD-facing `metrics.per_prompt` record for the ONE timed prompt this official run
/// measured (the challenge board's MTP column and per-prompt decode readout).
///
/// WHY THIS EXISTS. The free-run decode measurement computes
/// [`bench_core::free_run::FreeRunAudit::effective_mean_draft_len`] on every timed leg, but the two
/// free-run producers in `bench-runner/src/timing.rs` used to drop the whole audit when they
/// narrowed into [`TimingResult`], so the single-leg official path had nothing to seal and the board
/// showed a dash. The value now rides on `TimingResult::effective_mean_draft_len`.
///
/// ONE ENTRY, NEVER MORE. Official times the golden's benchmark oracle — ONE prompt — so exactly one
/// record is sealed; entries are never invented for pool prompts this run did not measure. The
/// identity is the golden's own sha256 (BIND BY BYTES), the same identity `metrics.golden_hash`
/// carries and the same rule the paired flow uses for its records.
///
/// NOTHING ENFORCED CHANGES. `mtp_seconds_per_token_mean` is READ BACK from
/// `metrics.decode_seconds_per_token` — the enforced whole-window figure `apply_timing_metrics` just
/// wrote — so the two cannot drift and no second, decode-only quantity is introduced (the RED-TEAM
/// REVERT notes in `bench-runner/src/timing.rs`). The array is additive: it feeds no score, floor or
/// band.
///
/// NO `head_provenance_sha256`. The engine's loaded-head digest arrives on the `hello`
/// (`bench_runner::Hello::head_provenance`), which the timed-worker spawn closures discard — a
/// [`Session`] does not retain it, and neither does [`TimingResult`]. The official path therefore has
/// no head digest in hand at this point, so the key is OMITTED rather than invented. The board reads
/// it optionally and simply flags no custom head.
///
/// WHERE IT IS SEALED, AND WHY THAT IS NOT THE #132(b) BLANK. The call sites are the three official
/// payloads that RETAIN the real measured timing surface: the passing score and the two
/// failed-with-timing builders. `per_prompt` belongs to that timing surface — both its numbers
/// describe the timed decode window that really ran — so it rides with the retained half. It is NOT
/// part of the correctness-derived surface `official_failed_timed_band` blanks to match Swift: that
/// blank exists because a non-empty `golden_hash` asserts "correctness completed", and
/// `prompt_sha256` asserts something else entirely, namely which prompt the clock measured. A
/// gates-only payload measured no prompt, so it seals nothing here.
///
/// A `None` audit (the teacher-forced v1 path, which official never takes) seals NOTHING.
fn seal_official_per_prompt(
    metrics: &mut ScoreMetrics,
    golden: &GoldenFixture,
    timing: &TimingResult,
) {
    crate::iterate::seal_timing_surface_facts(metrics, golden, timing);
}

/// Seal the ENGINE IDENTITY the TIMED worker announced on its `hello` — the worker whose leg is
/// scored.
///
/// WHY THIS EXISTS. Every benchd spawn site wrote `let (session, _hello) = Session::connect(...)`:
/// the handshake was validated (nonce, protocol version, capabilities) and then the engine's own
/// self-description was THROWN AWAY, so a sealed score said nothing about which engine build
/// produced it. The board's custom-head reader looks for `head_provenance_sha256`, and a CUDA run's
/// `backend` string carries the engine pin, overlay, nvcc and driver — all of it identity a scored
/// artifact should carry.
///
/// AUDIT-ONLY, ADDITIVE, FAIL-SOFT. Nothing here is scored, gated or compared: an engine that omits
/// a field simply seals no key for it (`skip_serializing_if`), so a pre-#106 engine's score is
/// byte-unchanged. The score is NOT refused on a divergence between the timed and correctness
/// workers — that would be a new refusal on the scored path — so the TIMED worker's hello is the one
/// sealed, by the rule that the sealed identity must be the identity that produced the sealed
/// number.
///
/// `head_provenance_sha256` is MIRRORED onto each `per_prompt` entry, which is where the board reads
/// it; the entries are already sealed by then, so this fills them in place.
///
/// THE RUNNER AND RESIDENT IDENTITIES ARE SEALED THE SAME WAY. The wire contract says `runner` and
/// `resident` are modeled "exactly like `head_provenance`" — so they are recorded here too, as
/// `runner_id` / `runner_model_type` / `runner_manifest_sha256` / `runner_build` and
/// `resident_pid` / `resident_load_epoch`. Each is absent from the sealed JSON when the hello
/// lacked it, never `0` and never `""`: an absent key is the honest statement "the worker said
/// nothing", which is a different claim from "the worker said zero". They are IDENTITY, recorded in
/// the score.json metrics and NEVER scored — nothing reads them back as an input to a number.
///
/// `pinned_head_sha256` is the head digest the TRUSTED reference leg reported on this run (paired
/// arm), or `None` when there is no reference leg; `head_provenance_sha256` seals only how the
/// engine's digest compares to it ([`crate::iterate::seal_head_digest`]).
pub fn seal_engine_identity(
    metrics: &mut ScoreMetrics,
    hello: &bench_runner::Hello,
    pinned_head_sha256: Option<&str>,
) {
    use crate::iterate::seal_hello_text as seal;
    // Free-text hello fields are engine-chosen: sealed as digests, logged in full.
    metrics.engine_backend = hello.backend.as_deref().map(|t| seal("backend", t));
    metrics.engine_device = hello.device.as_deref().map(|t| seal("device", t));
    metrics.engine_protocol_version = hello.protocol_version;
    use crate::iterate::{seal_digest_validity, seal_head_digest};
    metrics.head_provenance_sha256 = hello
        .head_provenance
        .as_ref()
        .map(|p| seal_head_digest(&p.sha256, pinned_head_sha256));
    metrics.runner_id = hello.runner.as_ref().map(|r| seal("runner.id", &r.id));
    metrics.runner_model_type = hello
        .runner
        .as_ref()
        .map(|r| seal("runner.model_type", &r.model_type));
    metrics.runner_manifest_sha256 = hello
        .runner
        .as_ref()
        .map(|r| seal_digest_validity("runner.manifest_sha256", &r.manifest_sha256));
    metrics.runner_build = hello
        .runner
        .as_ref()
        .map(|r| seal("runner.build", &r.build));
    // The resident pid/load_epoch are engine-chosen numbers: logged, never sealed (the keys stay
    // absent). The weights-load-once check reads them off the in-memory hello
    // ([`retain_timed_hello`]), never back out of the record.
    if let Some(r) = &hello.resident {
        eprintln!(
            "benchd: engine resident pid {} load_epoch {}",
            r.pid, r.load_epoch
        );
    }
    for entry in &mut metrics.per_prompt {
        entry
            .head_provenance_sha256
            .clone_from(&metrics.head_provenance_sha256);
    }
}

/// The named refusal raised when two phases of ONE timed window report DIFFERENT resident
/// identities ([`retain_timed_hello`]).
///
/// The resident's `load_epoch` is constant for the life of the process, so a changed `pid` or
/// `load_epoch` between the prefill worker's hello and the decode worker's hello means the window
/// did not run against one resident process: something reloaded the weights mid-window. That
/// violates weights-load-once, and it makes the sealed identity a lie — the score would name one
/// process while part of the window ran on another. It is refused rather than sealed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentIdentityChanged {
    /// The resident identity the FIRST phase of this window reported.
    pub first: Option<bench_protocol::ResidentIdentity>,
    /// The resident identity a LATER phase reported, which differs from `first`.
    pub then: Option<bench_protocol::ResidentIdentity>,
}

impl ResidentIdentityChanged {
    /// The refusal's stable NAME, in the style of the manifest gate's named failures.
    pub const NAME: &'static str = "resident_identity_changed_within_window";

    /// The refusal's stable name.
    pub fn name(&self) -> &'static str {
        Self::NAME
    }
}

impl std::fmt::Display for ResidentIdentityChanged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let show = |r: &Option<bench_protocol::ResidentIdentity>| match r {
            Some(r) => format!("pid {} load_epoch {}", r.pid, r.load_epoch),
            None => "no resident (the worker loaded its own weights)".to_string(),
        };
        write!(
            f,
            "{}: one timed window ran against two different resident identities — the first phase \
             reported {}, a later phase reported {}. The resident load_epoch is constant for the \
             life of the process, so this is a reload inside the window (weights-load-once).",
            self.name(),
            show(&self.first),
            show(&self.then)
        )
    }
}

/// Retain the TIMED worker's hello for sealing, and REFUSE a window whose phases disagree about the
/// resident process.
///
/// A timed leg can be assembled from several worker sessions — the official path spawns a fresh
/// worker per timed phase (prefill, then decode), each with its own hello. Every one of them must
/// report the SAME resident identity, because they are all supposed to be attached to the same
/// resident process for the whole window. This is the one place that sees both hellos, so it is
/// where the check belongs.
///
/// Only the RESIDENT identity is checked. The other identity fields are not: a worker legitimately
/// reports its own `pid` rather than the resident's, and the runner
/// identity is already gated by the conformance kit against a declared manifest.
///
/// `slot` holds the hello retained so far. On the first call it is filled. On a later call the new
/// hello's resident identity is compared with the retained one; equal (including both absent) keeps
/// the retained hello, and different is refused by name.
pub fn retain_timed_hello(
    slot: &mut Option<bench_runner::Hello>,
    hello: bench_runner::Hello,
) -> Result<(), ResidentIdentityChanged> {
    match slot {
        Some(first) if first.resident != hello.resident => Err(ResidentIdentityChanged {
            first: first.resident.clone(),
            then: hello.resident.clone(),
        }),
        Some(_) => Ok(()),
        None => {
            *slot = Some(hello);
            Ok(())
        }
    }
}

/// Whether an official run REQUIRES the runtime worker (Swift
/// `benchmarkRequiresRuntimeWorker`, QwenRuntimeBenchmark.swift:295-300): true iff the
/// golden declares any correctness BEHAVIOR case (hidden GPQA TTFT). Behavior TTFT is
/// measured in the trusted parent around sandboxed worker calls; the in-process path cannot
/// produce an equivalent trusted measurement, so its presence forces the worker path (and
/// benchd's official path fails closed without a sandboxed worker regardless).
///
/// Parity port + unit-tested; benchd's official run ALWAYS spawns sandboxed workers
/// (fail-closed), so this predicate is subsumed by that guarantee here and reserved for a
/// future worker-optional decision — hence `allow(dead_code)`.
#[allow(dead_code)]
pub fn benchmark_requires_runtime_worker(golden: &GoldenFixture) -> bool {
    golden
        .correctness_gates
        .as_ref()
        .is_some_and(|g| !g.behavior_cases().is_empty())
}

/// #61 — the OFFICIAL paired-baseline trusted override (Swift `PairedBaselineOverride`,
/// BenchmarkSupport.swift:35-80). The official timing machine measures the pinned reference
/// implementation immediately before the candidate and passes its seconds-per-token through
/// `MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN` /
/// `MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN`, so candidate speedups and floors are
/// priced against the same runner VM / hour / thermal state
/// (`docs/measure-job-contract.md@fe2da64`; evicted at cd5782e, resolve via git).
///
/// FAIL-CLOSED, byte-for-byte with Swift `fromEnvironment` (:50-79):
/// - both unset (after trim) ⇒ `Ok(None)` (no override; fall through to flags/golden);
/// - exactly one set ⇒ error "…must be provided together" (a half-set pair is an operator
///   wiring error and must stop the run, never silently degrade);
/// - a value that is not a FINITE POSITIVE double ⇒ error (mispricing the whole session).
///
/// Official-only: local modes never consult this (Swift strips both keys from the sandboxed
/// worker env, and only the benchmark/official paths read them). `prefill_raw`/`decode_raw`
/// are the raw env values (`None` if unset), passed in so the resolution is a pure,
/// testable function.
pub fn paired_baseline_from_env(
    prefill_raw: Option<&str>,
    decode_raw: Option<&str>,
) -> Result<Option<(f64, f64)>, String> {
    const PREFILL_KEY: &str = "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN";
    const DECODE_KEY: &str = "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN";
    let prefill = prefill_raw.unwrap_or("").trim();
    let decode = decode_raw.unwrap_or("").trim();
    if prefill.is_empty() && decode.is_empty() {
        return Ok(None);
    }
    if prefill.is_empty() || decode.is_empty() {
        return Err(format!(
            "{PREFILL_KEY} and {DECODE_KEY} must be provided together"
        ));
    }
    let prefill_v = parse_finite_positive(prefill).ok_or_else(|| {
        format!("{PREFILL_KEY} must be a finite positive seconds-per-token value")
    })?;
    let decode_v = parse_finite_positive(decode)
        .ok_or_else(|| format!("{DECODE_KEY} must be a finite positive seconds-per-token value"))?;
    Ok(Some((prefill_v, decode_v)))
}

/// THE reader of the `MLXFAST_PAIRED_BASELINE_*` pair from the process environment — one call site
/// shape for every path that consults it, so every path carries the SAME fail-closed posture.
///
/// Before #359 the two entry points disagreed: `main.rs` propagated the refusal with `?` while
/// [`official_resolved_baseline_pair`] did `.ok().flatten()`, which turned a half-set or
/// non-finite pair into "no override" and silently fell through to the golden's / contract's pair.
/// A mispriced denominator is exactly what [`paired_baseline_from_env`] exists to refuse, so the
/// silent posture is gone and this is the only way the pair is read.
pub fn paired_baseline_override() -> Result<Option<(f64, f64)>, String> {
    paired_baseline_from_env(
        std::env::var("MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN")
            .ok()
            .as_deref(),
        std::env::var("MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN")
            .ok()
            .as_deref(),
    )
}

/// A finite, strictly-positive `f64` from a string, else `None` (Swift `Double(raw)` +
/// `isFinite` + `> 0`).
fn parse_finite_positive(s: &str) -> Option<f64> {
    match s.parse::<f64>() {
        Ok(v) if v.is_finite() && v > 0.0 => Some(v),
        _ => None,
    }
}

/// Port of Swift `QwenRuntimePreflight.isCommitSHAHex` (:40-45): a lowercase-hex string of
/// 7..=40 chars (short-to-full commit SHA). NOTE the correction to the original "40-hex"
/// steer — Swift's `metrics.commit` predicate accepts 7-40 (only benchmark.sh's
/// `candidate.sha` recovery is strict-40); this mirrors the field's real source.
pub fn is_commit_sha_hex(s: &str) -> bool {
    let n = s.len();
    (7..=40).contains(&n)
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Port of Swift `QwenRuntimePreflight.commitIdentifier` (:26-45): a trimmed, valid-hex
/// `MLXFAST_COMMIT_SHA` wins; otherwise fall back to `git rev-parse --short HEAD`, or `""`.
/// `commit_sha_env` is the raw `MLXFAST_COMMIT_SHA` value (or `None` if unset), passed in so
/// the resolution is a pure, testable function; the git fallback is only consulted when the
/// env value is absent or malformed.
pub fn commit_identifier(commit_sha_env: Option<&str>) -> String {
    if let Some(v) = commit_sha_env {
        let trimmed = v.trim();
        if is_commit_sha_hex(trimmed) {
            return trimmed.to_string();
        }
    }
    crate::git_rev_parse_head(None, true).unwrap_or_default()
}

/// EVERYTHING ONE OFFICIAL RUN IS MEASURED AND SEALED UNDER, resolved once and threaded as ONE
/// value.
///
/// The golden it replays, what it is scored against, what it seals (`digests`, `commit`), and how
/// the worker is driven (`spec`, `window`) — every function on the official path took
/// the same list, positionally, which is what the too-many-arguments lint on three of them was
/// silenced for (#191). `bands` is deliberately NOT in here: the PAIRED path measures a
/// candidate window against its own live control leg and has no acceptance bands to state, so the
/// bands ride only on the entry that actually gates with them.
#[derive(Debug, Clone)]
pub struct OfficialParams<'a> {
    pub golden: &'a GoldenFixture,
    /// What the run is scored against: the baseline pair its speedups divide by and the floors
    /// they must clear.
    pub scoring: ScoringInputs,
    pub digests: RunDigests<'a>,
    pub commit: &'a str,
    /// The resolved `--mtp-depth` / `--candidate-spec` request that rides the timed leg. `None` is
    /// the historical bare request; a divergent engine echo discards the leg.
    pub spec: Option<SpecConfig>,
    pub window: &'a WindowShape,
}

/// The correctness-gate coordinates a FAILED official payload seals: which case diverged, at which
/// step, and how many teacher-forced steps the gate had checked when it did. They describe one
/// verdict, so they travel as one value.
struct CorrectnessFailure {
    first_failing_case: Option<String>,
    first_failing_step: Option<i64>,
    /// The REAL partial per-case checked-step sum through the failing gate (Swift
    /// `correctness?.checkedSteps`), or — for a pre-correctness timed fault that has no
    /// `ConformanceReport` to sum — the golden-derived placeholder.
    checked_steps: i64,
}

/// THE official flow: ONE model-holding worker for the WHOLE window (David 2026-08-30 load-once).
///
/// `spawn_timed` is called EXACTLY ONCE to open that worker; the timed prefill+decode phases run
/// over it via [`run_timed_benchmark_persistent_on_session`]; and the correctness phase runs over
/// the SAME session, so the model loads ONCE. The per-phase isolation is the worker's own
/// `beginMeasuredPhase` reset, which fires on `prefill`, `free_decode_begin`, and
/// `correctness_begin` alike. The gating/scoring order (timed-first → floors/bands → full
/// correctness) lives in [`finish_official`].
/// The TIMED leg's parameters: the golden's benchmark oracle, the resolved spec, and
/// the fixture's `official_prefill_warmup_runs` unmeasured prefill passes in the
/// timed session ahead of the one timed prefill (on MLX the timed session is a fresh attach, and
/// its first prefill is not a steady reading; on CUDA the resident engine is already warm and the
/// adapter refuses a second opener — see the constants).
pub fn official_timed_params(
    benchmark: &bench_core::golden::BenchmarkGolden,
    spec: Option<SpecConfig>,
    window: &WindowShape,
) -> TimingParams {
    // CONTRACT-FIRST (STEP 1, David 2026-09-15): the decode depth AND the warm-up count come from
    // the run's ONE resolved window. `platform` is gone from this signature on purpose — the
    // platform keys the TABLE's warm-up count, and that lookup now happens once, at the
    // resolution, rather than here.
    crate::iterate::benchmark_window_params(benchmark, Mode::Official.decode_steps(window))
        .with_spec(spec)
        .with_prefill_warmup_runs(window.official_prefill_warmup_runs)
        .with_prefill_timed_runs(window.official_prefill_timed_runs)
        // A CANDIDATE's timed divergence is deferred to the reference replay when the fixture
        // declares a replay policy (`bench_core::timed_replay`). The control leg overrides this.
        .with_divergence_deferred(window.timed_divergence_replay)
}

/// The MEASUREMENT-INTEGRITY WARMUP leg's parameters (coordinator ruling 2026-08-31): ONE
/// unmeasured, DISCARDED pass over the SAME benchmark prompt, decode seed and oracle the timed
/// legs use, at the SAME decode depth and with the SAME spec.
///
/// WHY IT EXISTS. On a FRESH resident serve the FIRST forward pays cold first-prefill JIT /
/// CUDA-graph-capture / flashinfer-JIT cost: a sealed CUDA null-control (serial-vs-serial, expected
/// score ~1.000) FLOOR-REFUSED at prefill_speedup 0.261 (532 vs 2041 tok/s) purely from that cold
/// first prefill. Both legs of a paired run are warmed identically, so the comparison is
/// warm-against-warm on both sides.
///
/// WHY IT CANNOT LEAK INTO THE MEASURED NUMBERS. It runs `VerifyMode::TimeOnly`, so a token
/// divergence does not abort it (catching an oracle mismatch stays the REAL timed legs' job), and
/// its `TimingResult` is DISCARDED on the spot — it never reaches any baseline, speedup, floor,
/// band or score. The depth matches the timed window (`Mode::Official.decode_steps(&test_window())`) because a
/// shorter warmup left the timed decode measurably colder. The SPEC matches the timed leg's,
/// because a speculating window JITs different shapes than a serial one.
///
/// `None` for a golden too short to warm, so the real timed leg surfaces the canonical
/// precondition error rather than a warmup-labelled one. That guard is no stricter than the timed
/// path's own.
fn official_warmup_params(
    benchmark: &bench_core::golden::BenchmarkGolden,
    spec: Option<SpecConfig>,
    window: &WindowShape,
) -> Option<TimingParams> {
    let warmup_decode_steps = Mode::Official.decode_steps(window);
    let runnable = benchmark.expected_decode_tokens.len() >= warmup_decode_steps
        && !benchmark.prefill_prompt_tokens.is_empty()
        && !benchmark.decode_seed_tokens.is_empty();
    runnable.then(|| {
        crate::iterate::benchmark_window_params(benchmark, warmup_decode_steps).with_spec(spec)
    })
}

/// WHICH stage of a timed window failed. The three classes carry different sealed error strings
/// and different failure payloads, so they stay distinct all the way out of [`run_timed_window`]
/// rather than being flattened into one message the caller has to re-parse.
enum TimedWindowFailure {
    /// The unmeasured warmup leg itself faulted.
    Warmup(RunnerError),
    /// The measured prefill/decode legs faulted — including the oracle-mismatch class.
    Timed(RunnerError),
}

/// ONE window's WARMUP leg and TIMED prefill+decode legs. This is the shared measurement body: the
/// ranked candidate leg ([`official_core_windowed`]), the SERIAL-CONTROL leg of a paired run and
/// every `calibrate-baseline` pass all run through it, so the three cannot drift apart.
///
/// ONE model-holding worker is opened, warmed and driven through both timed phases, and is RETURNED
/// still open so a caller that needs another phase over the same residency (official's correctness
/// gate) never loads the model twice. The caches are not isolated by a fresh process; they are
/// isolated by the worker's phase-start reset, which benchd asserts fail-closed at every phase close
/// (`cache_memory == 0`).
///
/// The unmeasured warmup leg is deliberately UNGATED — it is what heats the GPU — and the cool gate
/// then holds each TIMED phase to the per-phase contract. `verify` is how the timed legs check the
/// engine's tokens against the oracle.
fn run_timed_window<T, FT, G>(
    params: &TimingParams,
    warmup_params: Option<&TimingParams>,
    spawn_timed: &mut FT,
    cool_gate: &mut G,
    verify: VerifyMode,
) -> Result<(TimingResult, Session<T>), TimedWindowFailure>
where
    T: LineTransport,
    FT: FnMut() -> bench_runner::Result<Session<T>>,
    G: FnMut(&str) -> bench_runner::Result<()>,
{
    let mut no_cool_gate = |_phase: &str| -> bench_runner::Result<()> { Ok(()) };
    let mut session = spawn_timed().map_err(TimedWindowFailure::Timed)?;
    if let Some(wp) = warmup_params {
        // A warmup fault discards the session, so fail closed here rather than let the measured leg
        // report a bare SessionDiscarded.
        run_timed_benchmark_persistent_on_session(
            &mut session,
            &mut no_cool_gate,
            wp,
            VerifyMode::TimeOnly,
        )
        .map_err(TimedWindowFailure::Warmup)?;
    }
    let measured =
        run_timed_benchmark_persistent_on_session(&mut session, cool_gate, params, verify)
            .map_err(TimedWindowFailure::Timed)?;
    Ok((measured, session))
}

/// The CANDIDATE leg's timed window (warmup + prefill + decode, every token oracle-checked), or the
/// failed payload that ends the run. Shared by the single-leg window and the paired path, so an
/// oracle mismatch, a warmup fault and a protocol fault are classified ONCE. On the load-once
/// residency the measured session comes back still open for the correctness gate.
///
/// `verify` is [`VerifyMode::Verify`] everywhere but on the candidate leg of a paired run whose
/// track declares a timed token tolerance. There it is [`VerifyMode::PrefillOnly`]: the leg runs the
/// whole decode window and keeps the tokens it emitted, and the paired path judges them.
fn measure_candidate_window<T, FT, G>(
    run: OfficialParams<'_>,
    benchmark: &bench_core::golden::BenchmarkGolden,
    spawn_timed: &mut FT,
    cool_gate: &mut G,
    verify: VerifyMode,
) -> Result<(TimingResult, Session<T>), Box<ScorePayload>>
where
    T: LineTransport,
    FT: FnMut() -> bench_runner::Result<Session<T>>,
    G: FnMut(&str) -> bench_runner::Result<()>,
{
    let OfficialParams {
        golden,
        scoring,
        digests,
        commit,
        // The run's RESOLVED spec and window (STEP 1, David 2026-09-15). The window carries the
        // decode depth AND the unmeasured prefill-pass count; it REPLACES the `platform` this
        // function used to take — the platform was only ever here to key the warm-up count, and
        // that lookup now happens once, at the resolution.
        spec,
        window,
    } = run;
    let params = official_timed_params(benchmark, spec.clone(), window);

    // MEASUREMENT-INTEGRITY WARMUP + the TIMED legs, in the window shape this run measures
    // ([`run_timed_window`], which carries the whole rationale). The warmup leg is unmeasured and
    // discarded; the timed legs pass the caller's cool gate; the held session comes back so the
    // correctness phase below can reuse the ONE model residency.
    let warmup_params = official_warmup_params(benchmark, spec.clone(), window);
    let measured_result = run_timed_window(
        &params,
        warmup_params.as_ref(),
        spawn_timed,
        cool_gate,
        verify,
    );
    let (measured, held_session) = match measured_result {
        Ok(t) => t,
        Err(TimedWindowFailure::Warmup(e)) => {
            return Err(Box::new(official_failed(
                golden,
                digests,
                commit,
                format!("official warmup leg failed: {e}"),
                scoring,
            )));
        }
        Err(TimedWindowFailure::Timed(RunnerError::TokenMismatch { label, step, .. })) => {
            // The benchmark-ORACLE failure class the local path cannot test: a corrupted
            // oracle (or a fast-garbage engine) diverges and FAILS official. Byte-match Swift
            // `makeFailedScore` for a `BenchmarkTokenMismatchError`
            // (QwenRuntimeBenchmark.swift:668-676): `error = mismatch.description`,
            // `firstFailingCase = "benchmark"`, `firstFailingStep = mismatch.step`, and
            // `expectedToken`/`actualToken` are ALWAYS nil.
            //
            // Swift's PREFILL and decode-SEED comparisons go through `compareOne`
            // (Golden.swift:560-581) with `step: nil`, so `description` has NO " at step N"
            // suffix and `firstFailingStep = nil`. Only the decode-TOKEN class
            // (`compareDecodeTokens`, Golden.swift:533-551) carries a step. benchd's
            // `RunnerError::TokenMismatch.step` is non-optional, so distinguish by label.
            //
            // A8/DAVID RULING (timed decode = INCREMENTAL FREE-RUN): the timed decode leg now
            // drives `free_decode_begin` + `free_decode_run`, so its oracle-mismatch labels are
            // "benchmark free-run decode seed token" (step-less) and "benchmark free-run decode
            // token" (the stepped class). The PREFILL label is unchanged ("benchmark prefill
            // token", still teacher-forced). The sealed error strings therefore name the
            // free-run mechanism ("benchmark free-run decode token mismatch at step N", …) — an
            // intentional divergence from Swift's teacher-forced description, sanctioned by the
            // ruling: teacher-forced-per-step is retained ONLY for the untimed correctness gate.
            let is_decode_token_class = label == "benchmark free-run decode token";
            let (error, first_failing_step) = if is_decode_token_class {
                (
                    format!("{label} mismatch at step {step}"),
                    Some(step as i64),
                )
            } else {
                (format!("{label} mismatch"), None)
            };
            // The benchmark-ORACLE mismatch is a TIMED-phase failure BEFORE correctness runs
            // (official is timed-first). Like RULING-2's band/floor/finite path, Swift returns
            // via makeFailedScore(correctness: nil) — BLANKING the correctness audit fields
            // (golden_hash="", case_count=0, checked_steps=0) — but RETAINS the resolved
            // baselines (baselinePrefill/DecodeSecondsPerToken, set at :434-435 before the timed
            // phase). See official_failed_timed_oracle.
            return Err(Box::new(official_failed_timed_oracle(
                golden,
                digests,
                commit,
                error,
                first_failing_step,
                scoring,
            )));
        }
        Err(TimedWindowFailure::Timed(e)) => {
            // A non-oracle timed failure (protocol / completed-work barrier / spawn): fail
            // closed with the runner's message. No trustworthy timing to retain.
            return Err(Box::new(official_failed(
                golden,
                digests,
                commit,
                format!("{e}"),
                scoring,
            )));
        }
    };

    // Steps 2-4 (gating → correctness → assembly) operate on the measured `timing`; they are
    // factored into `finish_official` so they can be unit-tested with a SYNTHETIC in-band
    // TimingResult (a mock's ~0 wall-clock can never sit inside the acceptance band). The
    // correctness worker is the ONE resident session the timed phases ran on (load-once), so the
    // window never loads the model twice and never holds two residencies.
    Ok((measured, held_session))
}

pub fn official_core_windowed<T, FT, G>(
    run: OfficialParams<'_>,
    bands: AcceptanceBands,
    mut spawn_timed: FT,
    mut cool_gate: G,
) -> ScorePayload
where
    T: LineTransport,
    FT: FnMut() -> bench_runner::Result<Session<T>>,
    G: FnMut(&str) -> bench_runner::Result<()>,
{
    let OfficialParams {
        golden,
        scoring,
        digests,
        commit,
        spec,
        window,
    } = run;
    // Official times the golden's benchmark ORACLE workload (Swift's official benchmark),
    // NOT cases[0]. A benchmark-less golden fails preflight (Swift throws before spawning).
    let benchmark = match &golden.benchmark {
        Some(b) => b,
        None => {
            return official_failed(
                golden,
                digests,
                commit,
                "benchmark golden file must contain a benchmark oracle".to_string(),
                scoring,
            )
        }
    };
    // SPEC ON THE TIMED LEG (`--mtp-depth` / `--candidate-spec`). The official path used to build
    // its `TimingParams` with NO spec at all, so `free_decode_begin` went out bare and the engine
    // resolved its default — serial. On an engine where speculation is a SERVE-BOOT property that
    // was invisible; on a per-REQUEST engine it means a declared MTP leg was measured SERIAL while
    // being scored against the MTP oracle. The resolved spec now rides the timed window, and the
    // engine's `effective_spec` echo must equal it or the leg is discarded (spec-never-ignored,
    // enforced in `bench_runner`). `None` is byte-for-byte the historical bare request.
    let (measured, held_session) = match measure_candidate_window(
        OfficialParams {
            golden,
            scoring,
            digests,
            commit,
            spec,
            window,
        },
        benchmark,
        &mut spawn_timed,
        &mut cool_gate,
        VerifyMode::Verify,
    ) {
        Ok(measured) => measured,
        Err(payload) => return *payload,
    };
    // The single-leg path has no reference tree in reach, so a deferred divergence cannot be
    // replayed: it fails closed, named, exactly as the exact match would have.
    if measured.deferred_divergence.is_some() {
        return official_failed(
            golden,
            digests,
            commit,
            "the timed window diverged from the golden and the divergence was deferred to a \
             reference replay, but the single-leg official path has no reference to replay against"
                .to_string(),
            scoring,
        );
    }

    // Correctness on the SAME resident worker: the timed decode phase closed its barrier (allocator
    // drained), leaving the session healthy, and `correctness_begin` fires the worker's
    // `beginMeasuredPhase` reset again — the accepted per-phase isolation substitute. The
    // take-closure hands `finish_official` that one held session exactly once.
    let mut held = Some(held_session);
    finish_official(
        golden,
        scoring,
        bands,
        digests,
        commit,
        &measured,
        move || {
            held.take().ok_or_else(|| {
                RunnerError::Protocol(
                    "the window correctness phase was requested but the resident session was \
                     already consumed"
                        .to_string(),
                )
            })
        },
    )
}

/// The EXACT-MATCH name of the refusal "the serial-control leg did not complete".
pub const SERIAL_CONTROL_LEG_FAILED: &str = "SERIAL-CONTROL-LEG-FAILED";

/// LEG 1 of a paired ranked run: the SERIAL-CONTROL leg on the organizer-staged REFERENCE tree.
///
/// It is the same window the candidate leg runs — same prompt, same 128-token decode depth, same
/// per-platform prefill warm-up count, same unmeasured warmup leg, same per-phase cool gate — with
/// ONE difference: **no spec**. The control is serial by construction, because the score is a
/// speculative leg divided by a serial one.
///
/// `golden` is therefore the SERIAL golden: the tape a serial decode of that prompt produces. On a
/// track whose timed oracle at the candidate's depth is byte-identical to the serial tape it is
/// the candidate's own golden; on a track with per-depth oracle tapes it is the separate serial
/// live golden (`--control-golden`). Verifying a serial leg against a per-depth tape refuses at
/// the first step the two tapes disagree on.
///
/// It is also the WHOLE of what `benchd calibrate-baseline` measures: the calibrator calls this
/// function, once per pass, so a box's band and a box's ranked denominator can never be measured
/// two different ways. The session is opened and reaped inside this call — leg 1 holds no residency
/// while leg 2 runs, which is the sequential residency the ruling requires.
///
/// Every failure refuses BY NAME ([`SERIAL_CONTROL_LEG_FAILED`]): the reference tree is the
/// organizer's, so a fault here is never the candidate's fault and must not be reported as one.
pub fn run_serial_control_leg<T, FB, G>(
    golden: &GoldenFixture,
    window: &WindowShape,
    mut spawn_baseline: FB,
    mut cool_gate: G,
) -> Result<TimingResult, String>
where
    T: LineTransport,
    FB: FnMut() -> bench_runner::Result<Session<T>>,
    G: FnMut(&str) -> bench_runner::Result<()>,
{
    let benchmark = golden.benchmark.as_ref().ok_or_else(|| {
        format!(
            "{SERIAL_CONTROL_LEG_FAILED}: the golden carries no benchmark oracle, so there is no \
             prompt for the control leg to measure"
        )
    })?;
    // NO SPEC: the control leg is serial, and `None` is what puts nothing on the wire. NO
    // DEFERRAL: the reference must reproduce its own golden exactly, replay policy or not.
    let params = official_timed_params(benchmark, None, window).with_divergence_deferred(false);
    let warmup_params = official_warmup_params(benchmark, None, window);
    // EXACT in every mode: the reference tree must reproduce its own golden.
    let measured = run_timed_window(
        &params,
        warmup_params.as_ref(),
        &mut spawn_baseline,
        &mut cool_gate,
        VerifyMode::Verify,
    );
    // Reap leg 1's residency before returning: leg 2 loads next, and two model residencies must
    // never be live at once.
    measured
        .map(|(timing, session)| {
            drop(session);
            timing
        })
        .map_err(|e| match e {
            TimedWindowFailure::Warmup(e) => {
                format!("{SERIAL_CONTROL_LEG_FAILED}: reference warmup leg failed: {e}")
            }
            TimedWindowFailure::Timed(e) => format!("{SERIAL_CONTROL_LEG_FAILED}: {e}"),
        })
}

/// What a paired run seals about its baseline beyond the numbers themselves: which box, which
/// calibration bytes, which reference commit, and whether the band gate ran.
#[derive(Debug, Clone, Copy)]
pub struct PairedBaselineSeal<'a> {
    pub box_name: &'a str,
    pub calibration_sha256: &'a str,
    pub reference_commit: &'a str,
    /// `true` once the control leg passed this box's band. A run whose leg failed the band seals
    /// no score, so this is `true` wherever it is sealed.
    pub band_passed: bool,
    /// The control leg's measured `(prefill, decode)` seconds-per-token, or `None` when the leg
    /// never produced a timing (its own failure payload).
    pub leg: Option<(f64, f64)>,
}

/// Seal the paired-baseline facts onto a payload's metrics. `control_golden_sha256` is the digest
/// of the GOLDEN leg 1 verified its decode tokens against: the candidate golden's on a track whose
/// timed oracle at the declared depth IS the serial tape, and the serial live golden's on a track
/// that carries per-depth oracle tapes.
///
/// The CANDIDATE leg's numbers are READ BACK from the enforced fields
/// (`prefill_seconds_per_token` / `decode_seconds_per_token`) rather than passed in again, so the
/// named copies cannot drift from the numbers that were scored — the same discipline
/// `per_prompt.mtp_seconds_per_token_mean` follows. A payload with no candidate timing (a
/// preflight or control-leg refusal) seals no candidate keys: absent is "no leg ran", which is a
/// different claim from zero.
///
/// The historical `baseline_{prefill,decode}_seconds_per_token` fields are NOT written here. They
/// already carry the control leg's values, because the control leg's values are what the scoring
/// call was given — that is the point of the design, and the board keeps reading them.
pub fn seal_paired_baseline(
    metrics: &mut ScoreMetrics,
    seal: &PairedBaselineSeal<'_>,
    control_golden_sha256: &str,
) {
    metrics.baseline_source = Some(crate::baseline::BASELINE_SOURCE_SERIAL_CONTROL_LEG.to_string());
    metrics.baseline_box = Some(seal.box_name.to_string());
    metrics.baseline_calibration_sha256 = Some(seal.calibration_sha256.to_string());
    metrics.baseline_golden_sha256 = Some(control_golden_sha256.to_string());
    metrics.baseline_reference_commit = Some(seal.reference_commit.to_string());
    metrics.baseline_band_passed = Some(seal.band_passed);
    if let Some((prefill, decode)) = seal.leg {
        metrics.baseline_leg_prefill_seconds_per_token = Some(prefill);
        metrics.baseline_leg_decode_seconds_per_token = Some(decode);
    }
    metrics.candidate_leg_prefill_seconds_per_token =
        finite_positive(metrics.prefill_seconds_per_token);
    metrics.candidate_leg_decode_seconds_per_token =
        finite_positive(metrics.decode_seconds_per_token);
}

/// `Some(v)` for a finite, strictly-positive measurement; `None` for the zero placeholder a
/// payload carries when no leg ran.
fn finite_positive(v: f64) -> Option<f64> {
    (v.is_finite() && v > 0.0).then_some(v)
}

/// A paired REFUSAL: a failed payload carrying the paired seal, so a reader of a refused run
/// still learns which box, which calibration bytes and which reference commit it ran under. `leg`
/// is the control leg's measured pair when one was measured, and `None` when no leg completed —
/// absent is "no leg ran", never zero. `goldens` are the goldens of the pair that stopped.
fn paired_refusal(
    goldens: PairedGoldens<'_>,
    digests: RunDigests<'_>,
    commit: &str,
    error: String,
    seal: PairedBaselineSeal<'_>,
    leg: Option<(f64, f64)>,
    // The rules the run was ARMED with, over a pair of `0.0/0.0`: no denominator was established
    // and none is invented (`ScoringInputs::no_denominator`).
    scoring: ScoringInputs,
) -> ScorePayload {
    let mut payload = official_failed(goldens.candidate, digests, commit, error, scoring);
    let mut seal = seal;
    seal.band_passed = false;
    seal.leg = leg;
    seal_paired_baseline(&mut payload.metrics, &seal, &goldens.control.sha256);
    payload
}

/// THE TWO GOLDENS a paired run verifies against — one per leg, because the two legs decode two
/// different ways.
///
/// `candidate` is the golden the CANDIDATE leg (leg 2) verifies against, at the depth the
/// submission declares. `control` is the golden the SERIAL-CONTROL leg (leg 1) verifies against.
/// They are the SAME golden on a track whose timed oracle at that depth is byte-identical to the
/// serial tape, and DIFFERENT on a track that carries per-depth oracle tapes: leg 1 is serial by
/// construction, so it can only be verified against the serial tape. On the MLX engine the
/// per-depth tape diverges from the serial one at step 1, so one shared golden kills leg 1.
///
/// `prompt` is the prompt both goldens measure. It keys the calibration entry that gates leg 1.
#[derive(Clone, Copy)]
pub struct PairedGoldens<'a> {
    pub candidate: &'a GoldenFixture,
    pub control: &'a GoldenFixture,
    pub prompt: &'a str,
}

/// The ENGINE and WORKER lifecycles of a paired run's two legs, in one value so the paired entry
/// point states its inputs as two groups — WHO runs the legs, and WHAT window they run.
///
/// `open_*_leg` is a leg's per-platform ENGINE lifecycle: it returns a GUARD that is dropped the
/// moment its leg ends. It is given the index of the pair's goldens, so the caller knows which
/// golden the leg's workers measure. `spawn_*` opens that leg's WORKER, rooted at that leg's tree.
pub struct PairedLegs<LB, LC, FB, FT> {
    pub open_baseline_leg: LB,
    pub open_candidate_leg: LC,
    pub spawn_baseline: FB,
    pub spawn_timed: FT,
}

/// WHAT window a paired run measures: the acceptance bands, the candidate leg's requested spec, the
/// measured/checked window shape, and the per-phase cool gate both legs pass.
///
/// The PLATFORM used to ride here too, for one reason: it keyed the unmeasured prefill-pass count.
/// That count is now part of the RESOLVED window below (STEP 1 of the constants→contract
/// migration, David 2026-09-15), resolved once before the run, so the platform has nothing left to
/// decide here and is gone rather than carried unused.
pub struct PairedWindow<G> {
    pub bands: AcceptanceBands,
    pub spec: Option<SpecConfig>,
    /// The run's RESOLVED measurement window (STEP 1, David 2026-09-15): the timed decode depth,
    /// the checked-step count and the unmeasured prefill passes, from the pinned track fixture when
    /// it declares them and from the in-tree constants when it does not.
    pub window: WindowShape,
    pub cool_gate: G,
    /// THE GATE LOG this run records every gate POINT on (David 2026-09-17). The pair loop names
    /// the pair and the leg before each leg runs, so the points the `cool_gate` closure records
    /// carry their place in the run. The caller owns the log and seals it as `metrics.gates`
    /// after the run: the log outlives the refusal paths, so a run that stopped part-way still
    /// seals the gate points it did reach.
    pub gate_log: std::rc::Rc<crate::quiescegate::GateLog>,
    /// Pairs per scored run, from the pinned track fixture (`official_pairs`; David 2026-09-09
    /// ruled 2 on both platforms). Every pair is one serial-control leg then one candidate leg.
    /// Pair `k` (1-based) measures goldens `(k - 1) mod N` of the N the run was given.
    pub pairs: usize,
    /// The speedup floors this run enforces and seals, from the pinned track fixture
    /// (`decode_speedup_floor` / `prefill_speedup_floor`; David 2026-09-09 ruled 0.95 / 0.95,
    /// configurable per project). The fixture is the only source on the scored path.
    pub floors: SpeedupFloors,
    /// The composite weights this run scores and gates under, from the pinned track fixture
    /// (`score_decode_weight` / `score_prefill_weight`; STEP 1 of the constants→contract
    /// migration, David 2026-09-15), falling back to the 0.75/0.25 constants while no fixture
    /// declares them.
    pub weights: ScoringWeights,
    /// The fixture's TIMED-DIVERGENCE REPLAY policy (`bench_core::timed_replay`), or `None` for
    /// the exact-match timed window. When set, a candidate leg's deferred divergence is replayed
    /// through the REFERENCE tree's engine after the pairs, and the run fails by name if the
    /// reference would not have nearly chosen the committed tokens.
    pub replay: Option<bench_core::timed_replay::TimedReplayPolicy>,
    /// The timed token tolerance, in tokens per thousand, from the pinned track fixture
    /// (`timed_token_tolerance_per_thousand`). `None` keeps the exact rule: one different token
    /// fails the run. Refused together with `replay` (`contract::timed_token_policy`). See [`judge_candidate_tokens`].
    pub token_tolerance_per_thousand: Option<u32>,
    /// The near-tie limit of that tolerance, from the pinned track fixture
    /// (`timed_token_near_tie_relative_gap`). `None` counts every different token against the
    /// tolerance. With a limit, only near ties are tolerated. See [`judge_candidate_tokens`].
    pub near_tie_relative_gap: Option<f64>,
}

/// THE RANKED PAIRED PATH (David ruling 2026-09-08): two legs on one box in one job.
///
/// `goldens` holds one entry per `--golden`, in command-line order, and is never empty. Pair `k`
/// (1-based) measures entry `(k - 1) mod N`. The caller has already refused a pair count that is
/// not a multiple of N ([`crate::baseline::check_pairs_cover_goldens`]). In each pair:
///
/// 1. **Serial-control leg** on the REFERENCE tree (`spawn_baseline`), no speculation
///    ([`run_serial_control_leg`]), verified against the pair's `control` golden — the SERIAL tape.
/// 2. **Band check** of that leg against THIS BOX's calibration entry for the pair's prompt. The
///    band is a health gate on leg 1; no number in the file is ever a denominator. Outside the
///    band, the run dies by name and seals no score.
/// 3. **Candidate leg** on the submission tree (`spawn_timed`) at its declared depth, verified
///    against the pair's `candidate` golden, with the LIVE measurement from step 1 as its baseline
///    pair.
///
/// The run then scores ONE pair, the lower median over all pairs ([`scored_pair_index`]), and runs
/// the full correctness set on the scored pair's candidate golden. The score is therefore
/// `(ref_prefill/cand_prefill)^0.25 * (ref_decode/cand_decode)^0.75` of that pair, with the floors
/// and the band shape untouched.
///
/// RESIDENCY is sequential, and it brackets each leg on BOTH levels. `open_baseline_leg` /
/// `open_candidate_leg` are the platform's per-leg ENGINE lifecycle: on a platform whose worker
/// holds the model they do nothing, and on a platform whose worker is an adapter over a resident
/// engine they boot that leg's resident from that leg's own tree (see [`crate::legserve`]). Each
/// returns a GUARD that is dropped the moment its leg ends — before the next leg opens — so two
/// residents never hold GPU memory at once. Inside a leg, the worker itself is reaped the same
/// way: one model is loaded at a time, and each leg loads exactly once.
///
/// The paired seal rides on EVERY payload this returns, including the refusals, because "which box
/// and which calibration" is exactly what a reader of a refused run needs.
pub fn official_core_paired<T, L, LB, LC, FB, FT, G>(
    goldens: &[PairedGoldens<'_>],
    calibration: &crate::baseline::BaselineCalibration,
    seal: PairedBaselineSeal<'_>,
    digests: RunDigests<'_>,
    commit: &str,
    legs: PairedLegs<LB, LC, FB, FT>,
    window: PairedWindow<G>,
) -> ScorePayload
where
    T: LineTransport,
    LB: FnMut(usize) -> Result<L, String>,
    LC: FnMut(usize) -> Result<L, String>,
    FB: FnMut() -> bench_runner::Result<Session<T>>,
    FT: FnMut() -> bench_runner::Result<Session<T>>,
    G: FnMut(&str) -> bench_runner::Result<()>,
{
    // THE GATE LOG is sealed HERE, on ONE way out, so every payload the run can return carries it
    // — the scored one and every refusal alike. What a run gated on before it stopped is exactly
    // what a reader of a refused run needs (David 2026-09-17).
    let gate_log = std::rc::Rc::clone(&window.gate_log);
    let tolerance = window.token_tolerance_per_thousand;
    let near_tie_gap = window.near_tie_relative_gap;
    let mut payload =
        official_core_paired_inner(goldens, calibration, seal, digests, commit, legs, window);
    payload.metrics.gates = gate_log.records();
    // The token rule the run applied rides on every payload, so a reader of the score sees it.
    payload.metrics.timed_token_tolerance_per_thousand = tolerance;
    payload.metrics.timed_token_near_tie_relative_gap = near_tie_gap;
    payload
}

/// [`official_core_paired`]'s body, which returns from many places. The wrapper above seals the
/// gate log on all of them at once.
fn official_core_paired_inner<T, L, LB, LC, FB, FT, G>(
    goldens: &[PairedGoldens<'_>],
    calibration: &crate::baseline::BaselineCalibration,
    seal: PairedBaselineSeal<'_>,
    digests: RunDigests<'_>,
    commit: &str,
    legs: PairedLegs<LB, LC, FB, FT>,
    window: PairedWindow<G>,
) -> ScorePayload
where
    T: LineTransport,
    LB: FnMut(usize) -> Result<L, String>,
    LC: FnMut(usize) -> Result<L, String>,
    FB: FnMut() -> bench_runner::Result<Session<T>>,
    FT: FnMut() -> bench_runner::Result<Session<T>>,
    G: FnMut(&str) -> bench_runner::Result<()>,
{
    let PairedLegs {
        mut open_baseline_leg,
        mut open_candidate_leg,
        mut spawn_baseline,
        mut spawn_timed,
    } = legs;
    let PairedWindow {
        bands,
        spec,
        window: window_shape,
        mut cool_gate,
        gate_log,
        pairs,
        floors,
        weights,
        replay,
        token_tolerance_per_thousand,
        near_tie_relative_gap,
    } = window;
    // With a tolerance the candidate leg does not stop at a different decode token; the pairs are
    // judged after the last one ([`judge_candidate_tokens`]).
    let candidate_verify = match token_tolerance_per_thousand {
        Some(_) => VerifyMode::PrefillOnly,
        None => VerifyMode::Verify,
    };
    if pairs == 0 {
        return paired_refusal(
            goldens[0],
            digests,
            commit,
            "the paired official run was asked for 0 pairs; the track fixture must declare at least 1"
                .to_string(),
            seal,
            None,
            ScoringInputs::no_denominator(floors, weights),
        );
    }
    // Every candidate golden must carry its oracle before the first leg runs.
    let mut benchmarks = Vec::with_capacity(goldens.len());
    for pair_goldens in goldens {
        match &pair_goldens.candidate.benchmark {
            Some(b) => benchmarks.push(b),
            None => {
                return paired_refusal(
                    *pair_goldens,
                    digests,
                    commit,
                    "benchmark golden file must contain a benchmark oracle".to_string(),
                    seal,
                    None,
                    ScoringInputs::no_denominator(floors, weights),
                )
            }
        }
    }

    // PAIR LOOP. Every pair is the same two legs in the same order: the serial-control leg on the
    // reference tree, band-checked against the box calibration, then the candidate leg. Both legs
    // of a pair measure the pair's one prompt; the pairs cycle over the goldens. A leg's
    // ENGINE comes up before its worker and goes down before the next leg's comes up, so two
    // residents never hold GPU memory at once. The LAST pair's candidate session is kept open on
    // the load-once residency so the correctness gate runs over the same model residency.
    let mut records: Vec<PairedLegRecord> = Vec::with_capacity(pairs);
    let mut candidate_timings: Vec<TimingResult> = Vec::with_capacity(pairs);
    let mut held_session: Option<Session<T>> = None;
    let mut held_candidate_leg: Option<L> = None;
    for pair in 1..=pairs {
        let golden_index = (pair - 1) % goldens.len();
        let pair_goldens = goldens[golden_index];
        let golden = pair_goldens.candidate;
        let benchmark = benchmarks[golden_index];
        let baseline_leg = match open_baseline_leg(golden_index) {
            Ok(guard) => guard,
            Err(e) => {
                return with_measured_pairs(
                    paired_refusal(
                        pair_goldens,
                        digests,
                        commit,
                        e,
                        seal,
                        None,
                        ScoringInputs::no_denominator(floors, weights),
                    ),
                    records,
                )
            }
        };
        // NAME THE GATE POINTS this leg is about to make: pair N, the control leg. Every gate the
        // leg's timed phases run is recorded under that name (David 2026-09-17).
        gate_log.enter_leg(pair, crate::quiescegate::LEG_CONTROL);
        let control_result = run_serial_control_leg(
            pair_goldens.control,
            &window_shape,
            &mut spawn_baseline,
            &mut cool_gate,
        );
        drop(baseline_leg);
        let control = match control_result {
            Ok(t) => t,
            Err(e) => {
                return with_measured_pairs(
                    paired_refusal(
                        pair_goldens,
                        digests,
                        commit,
                        e,
                        seal,
                        None,
                        ScoringInputs::no_denominator(floors, weights),
                    ),
                    records,
                )
            }
        };
        let measured_leg = Some((
            control.prefill_seconds_per_token,
            control.decode_seconds_per_token,
        ));
        if let Err(e) = calibration.check_band(
            pair_goldens.prompt,
            control.prefill_seconds_per_token,
            control.decode_seconds_per_token,
        ) {
            let e = if pairs > 1 {
                format!("pair {pair} of {pairs}: {e}")
            } else {
                e
            };
            return with_measured_pairs(
                paired_refusal(
                    pair_goldens,
                    digests,
                    commit,
                    e,
                    seal,
                    measured_leg,
                    ScoringInputs::no_denominator(floors, weights),
                ),
                records,
            );
        }
        let candidate_leg = match open_candidate_leg(golden_index) {
            Ok(guard) => guard,
            Err(e) => {
                return with_measured_pairs(
                    paired_refusal(
                        pair_goldens,
                        digests,
                        commit,
                        e,
                        seal,
                        measured_leg,
                        ScoringInputs::no_denominator(floors, weights),
                    ),
                    records,
                )
            }
        };
        gate_log.enter_leg(pair, crate::quiescegate::LEG_CANDIDATE);
        let (timing, session) = match measure_candidate_window(
            OfficialParams {
                golden,
                // THE LIVE DENOMINATOR: this pair's own control leg, with the track's floors.
                scoring: ScoringInputs {
                    baseline_prefill_spt: control.prefill_seconds_per_token,
                    baseline_decode_spt: control.decode_seconds_per_token,
                    floors,
                    weights,
                },
                digests,
                commit,
                spec: spec.clone(),
                window: &window_shape,
            },
            benchmark,
            &mut spawn_timed,
            &mut cool_gate,
            candidate_verify,
        ) {
            Ok(measured) => measured,
            Err(payload) => {
                let mut payload = *payload;
                drop(candidate_leg);
                let mut seal = seal;
                seal.band_passed = true;
                seal.leg = measured_leg;
                seal_paired_baseline(&mut payload.metrics, &seal, &pair_goldens.control.sha256);
                payload.metrics.paired_legs = records;
                return payload;
            }
        };
        let (control_seed_window, control_decode_window) = window_per_token(&control);
        let (candidate_seed_window, candidate_decode_window) = window_per_token(&timing);
        // Tokens equal to the golden's oracle have no mismatch, and no replay runs for them. The
        // counts of any other pair stay unset until [`judge_candidate_tokens`] replays it.
        let exact = timing.emitted_tokens == oracle_tokens(benchmark, timing.decode_steps);
        // UNDER A REPLAY POLICY no per-pair count is sealed, exact or deferred: a 0 on one row and
        // a null on another would be a per-pair divergence bit. The replay's verdict is the run's.
        let exact_count = if replay.is_some() {
            None
        } else {
            exact.then_some(0)
        };
        records.push(PairedLegRecord {
            pair: pair as i64,
            prompt_sha256: golden.sha256.clone(),
            control_prefill_seconds_per_token: control.prefill_seconds_per_token,
            control_decode_seconds_per_token: control.decode_seconds_per_token,
            candidate_prefill_seconds_per_token: timing.prefill_seconds_per_token,
            candidate_decode_seconds_per_token: timing.decode_seconds_per_token,
            control_seed_prefill_window_seconds_per_token: control_seed_window,
            control_decode_window_seconds_per_token: control_decode_window,
            candidate_seed_prefill_window_seconds_per_token: candidate_seed_window,
            candidate_decode_window_seconds_per_token: candidate_decode_window,
            token_mismatch_count: exact_count,
            token_mismatch_first_step: None,
            token_mismatch_near_tie_count: near_tie_relative_gap.and(exact_count),
            token_mismatch_second_choice_count: exact_count,
            token_mismatch_second_choice_max_relative_gap: None,
        });
        candidate_timings.push(timing);
        if pair < pairs {
            drop(session);
            drop(candidate_leg);
        } else {
            held_session = Some(session);
            held_candidate_leg = Some(candidate_leg);
        }
    }

    // THE SCORED PAIR (David ruling 2026-09-17, the Laguna rule): pairs are NEVER averaged. Each
    // pair is scored on its own control leg, and the run scores ONE measured pair — the one whose
    // composite is the lower median over the pairs. Every other pair stays in `paired_legs` as
    // measured. With ONE pair this is that pair.
    // THE DEFERRED DIVERGENCES, per pair, taken before the timings are consumed. Each is UNVERIFIED
    // until the reference replay below judges it.
    let deferred: Vec<(usize, bench_runner::timing::DeferredDivergence)> = candidate_timings
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.deferred_divergence.clone().map(|d| (i + 1, d)))
        .collect();
    let scored = scored_pair_index(&records, weights);
    let scored_record = records[scored].clone();
    let scored_goldens = goldens[scored % goldens.len()];
    let candidate = &candidate_timings[scored];
    let control_prefill = scored_record.control_prefill_seconds_per_token;
    let control_decode = scored_record.control_decode_seconds_per_token;
    // The scored run's inputs: the scored pair's control leg as the denominator, the track
    // fixture's floors as the gate. One value, so the floors this run enforces are the floors it
    // seals.
    let paired_scoring = ScoringInputs {
        baseline_prefill_spt: control_prefill,
        baseline_decode_spt: control_decode,
        floors,
        weights,
    };

    let mut held = held_session;
    let mut payload = finish_official(
        scored_goldens.candidate,
        paired_scoring,
        bands,
        digests,
        commit,
        candidate,
        move || {
            held.take().ok_or_else(|| {
                RunnerError::Protocol(
                    "the window correctness phase was requested but the resident session was \
                     already consumed"
                        .to_string(),
                )
            })
        },
    );
    drop(held_candidate_leg);
    // THE REFERENCE REPLAY. The candidate's residency is gone (dropped above), so the reference
    // engine comes up alone, exactly as a control leg does. A run that already failed is not
    // replayed: nothing it could decide would change the verdict.
    if !deferred.is_empty() && payload.passed {
        let verdict = match replay {
            Some(policy) => replay_deferred_divergences(
                &mut open_baseline_leg,
                &mut spawn_baseline,
                &benchmarks,
                &deferred,
                &policy,
            ),
            None => Err((
                deferred[0].0,
                "a timed divergence was deferred without a replay policy".to_string(),
            )),
        };
        match verdict {
            Ok(summary) => eprintln!("benchd official: {summary}"),
            Err((pair, e)) => {
                // Named after the pair that FAILED (1-based): its golden is the one sealed, and
                // its control leg is the one that passed the band, as on the tolerance path.
                let failing_goldens = goldens[(pair - 1) % goldens.len()];
                let record = &records[pair - 1];
                let control_leg = (
                    record.control_prefill_seconds_per_token,
                    record.control_decode_seconds_per_token,
                );
                let mut failed = official_failed(
                    failing_goldens.candidate,
                    digests,
                    commit,
                    format!("TIMED-REPLAY-REJECTED: {e}"),
                    ScoringInputs::no_denominator(floors, weights),
                );
                let mut seal = seal;
                seal.band_passed = true;
                seal.leg = Some(control_leg);
                seal_paired_baseline(&mut failed.metrics, &seal, &failing_goldens.control.sha256);
                failed.metrics.paired_legs = records;
                return failed;
            }
        }
    }

    // THE TIMED TOKEN TOLERANCE (David 2026-09-28). The candidate's residency is gone, so the
    // reference engine can load. A pair over the limit, or a replay that did not complete, refuses
    // the whole run by name. The refusal replaces the payload above: the token verdict comes
    // first, as it did when one different token stopped the candidate leg.
    if let Some(limit) = token_tolerance_per_thousand {
        if let Err((failing, error)) = judge_candidate_tokens(
            limit,
            near_tie_relative_gap,
            &benchmarks,
            &candidate_timings,
            &mut records,
            &mut open_baseline_leg,
            &mut spawn_baseline,
        ) {
            let record = records[failing].clone();
            let failing_goldens = goldens[failing % goldens.len()];
            let control_leg = (
                record.control_prefill_seconds_per_token,
                record.control_decode_seconds_per_token,
            );
            let mut payload = official_failed_timed_oracle(
                failing_goldens.candidate,
                digests,
                commit,
                error,
                // Not the mismatch step: the candidate chose where its stream went (logged above).
                None,
                ScoringInputs {
                    baseline_prefill_spt: control_leg.0,
                    baseline_decode_spt: control_leg.1,
                    floors,
                    weights,
                },
            );
            let mut seal = seal;
            seal.band_passed = true;
            seal.leg = Some(control_leg);
            seal_paired_baseline(&mut payload.metrics, &seal, &failing_goldens.control.sha256);
            seal_token_counts(&mut payload.metrics, &record);
            payload.metrics.paired_legs = records;
            return payload;
        }
    }

    let mut seal = seal;
    seal.band_passed = true;
    seal.leg = Some((control_prefill, control_decode));
    seal_paired_baseline(&mut payload.metrics, &seal, &scored_goldens.control.sha256);
    seal_window_split(&mut payload.metrics, &scored_record);
    seal_token_counts(&mut payload.metrics, &records[scored]);
    // The payload builders sealed `per_prompt` for the scored pair's golden only, and only when
    // they kept the timing. Replace it with one record per distinct golden.
    if !payload.metrics.per_prompt.is_empty() {
        payload.metrics.per_prompt =
            paired_per_prompt(goldens, &records, &candidate_timings, weights);
    }
    payload.metrics.paired_legs = records;
    payload
}

/// REPLAY every deferred divergence through the REFERENCE tree's engine (`bench_core::timed_replay`).
///
/// One reference residency serves every pair: the leg guard boots it, one worker session attaches,
/// and each divergence is teacher-forced through `correctness_begin` / `correctness_step`, which
/// return the reference's raw top-8 logits at every position. A divergent PREFILL token is judged
/// at the prefill prompt's frontier. A divergent free-run STREAM is judged from its first divergent
/// position on (the positions before it matched the golden exactly), and its off-argmax count must
/// fit the budget over the whole window. Returns a one-line audit summary, or the first reason the
/// reference would not have nearly chosen what the candidate committed.
fn replay_deferred_divergences<T, L, LB, FB>(
    open_baseline_leg: &mut LB,
    spawn_baseline: &mut FB,
    benchmarks: &[&bench_core::golden::BenchmarkGolden],
    deferred: &[(usize, bench_runner::timing::DeferredDivergence)],
    policy: &bench_core::timed_replay::TimedReplayPolicy,
) -> Result<String, (usize, String)>
where
    T: LineTransport,
    LB: FnMut(usize) -> Result<L, String>,
    FB: FnMut() -> bench_runner::Result<Session<T>>,
{
    use bench_core::timed_replay::judge_position;
    fn top_of(resp: &bench_protocol::WorkerResponse) -> Result<Vec<(i64, f64)>, String> {
        resp.top_logits
            .as_ref()
            .map(|v| v.iter().map(|l| (l.token, l.logit)).collect())
            .ok_or_else(|| "the reference returned no top_logits".to_string())
    }
    // SEALED TEXT carries no candidate-chosen number: no committed token id and no position past
    // the first divergence (the candidate chose where its stream went). Both go to stderr.
    let log = |detail: String| eprintln!("benchd official: timed replay: {detail}");
    // Pair `k` measured golden `(k - 1) mod N`. ONE reference residency serves every pair: the
    // reference tree and weights are the same for every golden, and the index only names which
    // golden the leg's per-spawn sandbox plan denies.
    let benchmark_of = |pair: usize| benchmarks[(pair - 1) % benchmarks.len()];
    let first_pair = deferred.first().map_or(1, |(p, _)| *p);
    let guard = open_baseline_leg((first_pair - 1) % benchmarks.len()).map_err(|e| {
        (
            first_pair,
            format!("the reference engine for the replay did not come up: {e}"),
        )
    })?;
    let mut session = spawn_baseline().map_err(|e| {
        (
            first_pair,
            format!("the reference worker for the replay did not start: {e}"),
        )
    })?;
    let mut summary = Vec::new();
    for (pair, d) in deferred {
        let pair = *pair;
        let fail = |e: String| (pair, e);
        let benchmark = benchmark_of(pair);
        if let Some(token) = d.prefill_token {
            session.begin_phase();
            let resp = session
                .correctness_begin(&benchmark.prefill_prompt_tokens)
                .map_err(|e| fail(format!("pair {pair}: prefill replay: {e}")))?;
            let top = top_of(&resp).map_err(fail)?;
            let j = judge_position(&top, token, policy).map_err(|e| {
                log(format!("pair {pair}: prefill token {token}: {e}"));
                fail(format!("pair {pair}: timed prefill: {e}"))
            })?;
            if j.off_argmax {
                log(format!("pair {pair}: prefill token {token} is off-argmax"));
                return Err(fail(format!(
                    "pair {pair}: the timed prefill token is off-argmax for the reference"
                )));
            }
            session
                .close_phase()
                .map_err(|e| fail(format!("pair {pair}: prefill replay barrier: {e}")))?;
            summary.push(format!("pair {pair} prefill token {token} accepted"));
        }
        if let Some((first, stream)) = &d.stream {
            session.begin_phase();
            let mut resp = session
                .correctness_begin(&benchmark.decode_seed_tokens)
                .map_err(|e| fail(format!("pair {pair}: decode replay: {e}")))?;
            let mut off_argmax = 0usize;
            let mut worst_gap = 0.0f64;
            for (p, &token) in stream.iter().enumerate() {
                if p > 0 {
                    resp = session.correctness_step(stream[p - 1]).map_err(|e| {
                        log(format!("pair {pair}: decode replay failed at position {p}"));
                        fail(format!("pair {pair}: decode replay: {e}"))
                    })?;
                }
                if p < *first {
                    continue;
                }
                let top = top_of(&resp).map_err(fail)?;
                let j = judge_position(&top, token, policy).map_err(|e| {
                    log(format!(
                        "pair {pair}: timed decode position {p} (0 = the seed token), token \
                         {token}: {e}"
                    ));
                    fail(format!("pair {pair}: timed decode: {e}"))
                })?;
                if j.off_argmax {
                    off_argmax += 1;
                }
                worst_gap = worst_gap.max(j.gap.unwrap_or(0.0));
            }
            session
                .close_phase()
                .map_err(|e| fail(format!("pair {pair}: decode replay barrier: {e}")))?;
            let budget = policy.off_argmax_budget(stream.len());
            if off_argmax > budget {
                log(format!(
                    "pair {pair}: {off_argmax} off-argmax positions from position {first}"
                ));
                return Err(fail(format!(
                    "pair {pair}: more of the timed window's {} positions are off-argmax for the \
                     reference than the budget of {budget}",
                    stream.len()
                )));
            }
            summary.push(format!(
                "pair {pair} decode diverged at position {first}, replayed {} positions, \
                 {off_argmax} off-argmax (budget {budget}), worst in-top gap {worst_gap:.2}",
                stream.len() - first
            ));
        }
    }
    drop(session);
    drop(guard);
    Ok(format!("timed replay accepted: {}", summary.join("; ")))
}

/// One `per_prompt` record per distinct candidate golden, in command-line order. Each record
/// carries the drafting facts of one pair that measured its prompt. When a prompt has several
/// pairs, the record is the pair that the run's own rule picks among them: the lower median of
/// their composites ([`scored_pair_index`]). With one golden that pair is the scored pair, so the
/// record is the one the payload builders sealed.
fn paired_per_prompt(
    goldens: &[PairedGoldens<'_>],
    records: &[PairedLegRecord],
    timings: &[TimingResult],
    weights: ScoringWeights,
) -> Vec<crate::score::ScorePerPrompt> {
    let mut out = Vec::new();
    for (i, pair_goldens) in goldens.iter().enumerate() {
        let golden = pair_goldens.candidate;
        if goldens[..i]
            .iter()
            .any(|g| g.candidate.sha256 == golden.sha256)
        {
            continue;
        }
        let pairs: Vec<usize> = (0..records.len())
            .filter(|&p| records[p].prompt_sha256 == golden.sha256)
            .collect();
        let own: Vec<PairedLegRecord> = pairs.iter().map(|&p| records[p].clone()).collect();
        let timing = &timings[pairs[scored_pair_index(&own, weights)]];
        let mut metrics = ScoreMetrics {
            decode_seconds_per_token: crate::iterate::finite_nonneg(
                timing.decode_seconds_per_token,
            ),
            ..ScoreMetrics::default()
        };
        seal_official_per_prompt(&mut metrics, golden, timing);
        out.extend(metrics.per_prompt);
    }
    out
}

/// One pair's two legs, as measured, sealed for the audit trail (`metrics.paired_legs`).
pub use crate::score::PairedLegRecord;

/// One leg's free-run window SPLIT per token — `(seed prefill / seed length, decode / N)` — or
/// `(None, None)` when the leg timed a teacher-forced window that has no split. REPORT-ONLY.
fn window_per_token(timing: &TimingResult) -> (Option<f64>, Option<f64>) {
    let Some(w) = timing.phase_window else {
        return (None, None);
    };
    let per = |seconds: f64, tokens: usize| {
        (tokens > 0 && seconds.is_finite() && seconds > 0.0).then(|| seconds / tokens as f64)
    };
    (
        per(w.seed_prefill_elapsed_seconds, timing.prefill_prompt_tokens),
        per(w.decode_elapsed_seconds, timing.decode_steps),
    )
}

/// Seal the SCORED pair's window split onto the flat metrics
/// (`ScoreMetrics::*_leg_*_window_seconds_per_token`). These are the same row's figures as the
/// enforced ones, never a mean over pairs. A split is absent when that pair timed a teacher-forced
/// window.
fn seal_window_split(metrics: &mut ScoreMetrics, record: &PairedLegRecord) {
    metrics.baseline_leg_decode_window_seconds_per_token =
        record.control_decode_window_seconds_per_token;
    metrics.candidate_leg_decode_window_seconds_per_token =
        record.candidate_decode_window_seconds_per_token;
    metrics.baseline_leg_seed_prefill_window_seconds_per_token =
        record.control_seed_prefill_window_seconds_per_token;
    metrics.candidate_leg_seed_prefill_window_seconds_per_token =
        record.candidate_seed_prefill_window_seconds_per_token;
}

/// The index of the pair the run scores (David ruling 2026-09-17, the Laguna rule): the LOWER
/// MEDIAN of the per-pair composites — the order statistic at index `(n - 1) / 2` of the pairs
/// sorted by composite, which on an even count is the lower of the two central pairs, never their
/// mean. Each pair's composite is the track's weighted score of its candidate leg over its OWN
/// control leg. Ties keep measurement order, so equal pairs score the earlier one.
///
/// This is the same rule the measure job's `lower_median` applied to Laguna's per-pair ratios,
/// lifted to the whole pair so that every enforced figure the run seals comes from one measured
/// pair.
fn scored_pair_index(records: &[PairedLegRecord], weights: ScoringWeights) -> usize {
    let composite = |r: &PairedLegRecord| {
        bench_core::score::score_weighted(
            r.candidate_decode_seconds_per_token,
            r.candidate_prefill_seconds_per_token,
            r.control_decode_seconds_per_token,
            r.control_prefill_seconds_per_token,
            weights,
        )
    };
    let mut order: Vec<usize> = (0..records.len()).collect();
    order.sort_by(|&a, &b| composite(&records[a]).total_cmp(&composite(&records[b])));
    order[(records.len() - 1) / 2]
}

/// The EXACT-MATCH name of the refusal "a pair's candidate tokens differ from the reference
/// engine's choices on more than the track's timed token tolerance". The names do not contain the
/// word TOKEN: the seal scrubber reads `TOKEN…:` as a credential key and redacts the rest of the
/// line.
pub const TIMED_DIVERGENCE_OVER_TOLERANCE: &str = "TIMED-DIVERGENCE-OVER-TOLERANCE";

/// The EXACT-MATCH name of the refusal "a pair's candidate token differs from the reference
/// engine's choice and is not a near tie under the track's `timed_token_near_tie_relative_gap`".
pub const TIMED_DIVERGENCE_NOT_A_NEAR_TIE: &str = "TIMED-DIVERGENCE-NOT-A-NEAR-TIE";

/// The EXACT-MATCH name of the refusal "the reference engine's replay of the candidate's tokens did
/// not complete". Without the replay benchd cannot count the mismatches, so the run seals no score.
pub const TIMED_REPLAY_FAILED: &str = "TIMED-REPLAY-FAILED";

/// The golden's oracle for the positions a candidate leg emits: the seed token, then the first
/// `decode_steps` decode tokens.
fn oracle_tokens(benchmark: &bench_core::golden::BenchmarkGolden, decode_steps: usize) -> Vec<i64> {
    std::iter::once(benchmark.expected_decode_seed_token)
        .chain(
            benchmark
                .expected_decode_tokens
                .iter()
                .take(decode_steps)
                .copied(),
        )
        .collect()
}

/// The reference engine's choices at one replayed position: its argmax, its second choice, and the
/// relative gap between the two logits (`bench_core::near_tie::relative_gap`).
struct ReferenceChoice {
    first: i64,
    second: i64,
    second_gap: f64,
}

/// Read one teacher-forced response as a [`ReferenceChoice`]. The argmax is the response token;
/// the second choice is the highest logit of another token.
fn reference_choice(
    resp: &bench_protocol::WorkerResponse,
) -> bench_runner::Result<ReferenceChoice> {
    let first = resp.token.ok_or_else(|| {
        RunnerError::Protocol("the reference replay response carried no token".to_string())
    })?;
    let logits = resp.top_logits.as_deref().unwrap_or_default();
    let top = logits
        .iter()
        .map(|l| l.logit)
        .fold(f64::NEG_INFINITY, f64::max);
    let second = logits
        .iter()
        .filter(|l| l.token != first)
        .max_by(|a, b| a.logit.total_cmp(&b.logit))
        .ok_or_else(|| {
            RunnerError::Protocol(
                "the reference replay response carried no second choice".to_string(),
            )
        })?;
    Ok(ReferenceChoice {
        first,
        second: second.token,
        second_gap: bench_core::near_tie::relative_gap(top, second.logit),
    })
}

/// Replay one candidate leg's emitted tokens teacher-forced on the reference engine.
/// `correctness_begin(seed_prompt)` gives the reference choice for the seed position, and
/// `correctness_step(emitted[i])` gives the reference choice for position `i + 1`. The replay is
/// not timed.
fn replay_candidate_tokens<T: LineTransport>(
    session: &mut Session<T>,
    seed_prompt: &[i64],
    emitted: &[i64],
) -> bench_runner::Result<Vec<ReferenceChoice>> {
    session.begin_phase();
    let mut choices = Vec::with_capacity(emitted.len());
    let mut resp = session.correctness_begin(seed_prompt)?;
    for (i, &token) in emitted.iter().enumerate() {
        choices.push(reference_choice(&resp)?);
        if i + 1 < emitted.len() {
            resp = session.correctness_step(token)?;
        }
    }
    session.close_phase()?;
    Ok(choices)
}

/// THE TIMED TOKEN TOLERANCE of the paired path (David 2026-09-28): the blanket 10 % rule of
/// 2026-08-25 ([`bench_core::cohort_tolerance`]) applied to each pair's candidate leg.
///
/// A pair whose tokens equal the golden's oracle already carries 0 mismatches and is not replayed.
/// Every other pair is replayed on the REFERENCE tree: its engine boots the way the serial-control
/// leg boots it, ONE worker serves every replay, so the reference weights load once. At each
/// position the candidate's token is compared with the reference engine's choice for the
/// candidate's own prefix. The decision is `evaluate_cohort_token_tolerance`, with one pair per
/// stream: a pair passes when `mismatches * 1000 <= limit * tokens`.
///
/// NEAR TIES ONLY (David 2026-09-28: "10% for near ties only"). With a `near_tie_gap`, a mismatch
/// is tolerated only when the candidate's token is the reference engine's second choice and the
/// relative gap between its first and second choice is `<= near_tie_gap`. The first pair with any
/// other mismatch refuses the run, before the count rule. Near ties still count against `limit`.
///
/// The second-choice figures are REPORT-ONLY. They are written to the records and nothing reads
/// them to decide.
///
/// `Err` carries the index of the pair to name and the refusal text.
fn judge_candidate_tokens<T, L, LB, FB>(
    limit: u32,
    near_tie_gap: Option<f64>,
    benchmarks: &[&bench_core::golden::BenchmarkGolden],
    timings: &[TimingResult],
    records: &mut [PairedLegRecord],
    open_baseline_leg: &mut LB,
    spawn_baseline: &mut FB,
) -> Result<(), (usize, String)>
where
    T: LineTransport,
    LB: FnMut(usize) -> Result<L, String>,
    FB: FnMut() -> bench_runner::Result<Session<T>>,
{
    let replay_failed =
        |p: usize, e: String| (p, format!("{TIMED_REPLAY_FAILED}: pair {}: {e}", p + 1));
    let committed: Vec<Vec<i64>> = timings.iter().map(|t| t.emitted_tokens.clone()).collect();
    let mut reference = committed.clone();
    let pending: Vec<usize> = (0..records.len())
        .filter(|&p| records[p].token_mismatch_count.is_none())
        .collect();
    // The first pair with a mismatch that is not a near tie: its index, and the refusal text.
    let mut not_a_near_tie: Option<(usize, String)> = None;
    if let Some(&first) = pending.first() {
        let leg =
            open_baseline_leg(first % benchmarks.len()).map_err(|e| replay_failed(first, e))?;
        let mut session = spawn_baseline().map_err(|e| replay_failed(first, e.to_string()))?;
        for &p in &pending {
            let emitted = &committed[p];
            let choices = replay_candidate_tokens(
                &mut session,
                &benchmarks[p % benchmarks.len()].decode_seed_tokens,
                emitted,
            )
            .map_err(|e| replay_failed(p, e.to_string()))?;
            let second_choice_gaps: Vec<f64> = emitted
                .iter()
                .zip(&choices)
                .filter(|(&t, c)| t != c.first && t == c.second)
                .map(|(_, c)| c.second_gap)
                .collect();
            let record = &mut records[p];
            record.token_mismatch_first_step = emitted
                .iter()
                .zip(&choices)
                .position(|(&t, c)| t != c.first)
                .map(|i| i as i64);
            record.token_mismatch_second_choice_count = Some(second_choice_gaps.len() as i64);
            record.token_mismatch_second_choice_max_relative_gap =
                second_choice_gaps.into_iter().reduce(f64::max);
            if let Some(gap_limit) = near_tie_gap {
                let mut near_ties = 0i64;
                for (step, (&t, c)) in emitted.iter().zip(&choices).enumerate() {
                    if t == c.first {
                        continue;
                    }
                    // SEALED: no position (the candidate chose where its stream went) and the
                    // gap at 2 decimals. The exact values go to stderr.
                    let failed = if t != c.second {
                        "the token is not the reference engine's second choice".to_string()
                    } else if c.second_gap > gap_limit {
                        format!("the gap {:.2} is above the limit", c.second_gap)
                    } else {
                        near_ties += 1;
                        continue;
                    };
                    eprintln!(
                        "benchd official: pair {}: timed position {step} is not a near tie \
                         (reference gap {}, limit {gap_limit})",
                        p + 1,
                        c.second_gap
                    );
                    if not_a_near_tie.is_none() {
                        not_a_near_tie = Some((
                            p,
                            format!(
                                "{TIMED_DIVERGENCE_NOT_A_NEAR_TIE}: pair {} of {}: a timed token \
                                 differs from the reference engine's choice and is not a near tie \
                                 under the limit {gap_limit}: {failed}",
                                p + 1,
                                records.len(),
                            ),
                        ));
                    }
                }
                records[p].token_mismatch_near_tie_count = Some(near_ties);
            }
            reference[p] = choices.iter().map(|c| c.first).collect();
        }
        drop(session);
        drop(leg);
    }
    let verdict = bench_core::cohort_tolerance::evaluate_cohort_token_tolerance(
        &committed, &reference, limit,
    )
    .map_err(|e| replay_failed(0, e.to_string()))?;
    for count in &verdict.per_stream {
        records[count.slot].token_mismatch_count = Some(count.mismatches as i64);
    }
    // The exact counts are LOGGED; the sealed record carries them only bucketed
    // (`ScoreMetrics::with_coarsened_public_diagnostics`).
    for r in records.iter() {
        eprintln!(
            "benchd official: pair {}: timed mismatches {:?}, first at {:?}, near ties {:?}, \
             second choices {:?}, max second-choice gap {:?}",
            r.pair,
            r.token_mismatch_count,
            r.token_mismatch_first_step,
            r.token_mismatch_near_tie_count,
            r.token_mismatch_second_choice_count,
            r.token_mismatch_second_choice_max_relative_gap
        );
    }
    if let Some(refusal) = not_a_near_tie {
        return Err(refusal);
    }
    match verdict.first_failing {
        Some(f) => Err((
            f.slot,
            // SEALED without the mismatch count (the candidate's choice); the allowance is
            // benchd's own arithmetic on the fixture's limit and the window length.
            format!(
                "{TIMED_DIVERGENCE_OVER_TOLERANCE}: pair {} of {}: more of the candidate's {} timed \
                 tokens differ from the reference engine's choice for the same prefix than the {} \
                 the track fixture allows ({limit} per thousand)",
                f.slot + 1,
                records.len(),
                f.committed_len,
                limit as usize * f.committed_len / 1000,
            ),
        )),
        None => Ok(()),
    }
}

/// Seal one pair's token counts as the run-level figures.
fn seal_token_counts(metrics: &mut ScoreMetrics, record: &PairedLegRecord) {
    metrics.token_mismatch_count = record.token_mismatch_count;
    metrics.token_mismatch_first_step = record.token_mismatch_first_step;
    metrics.token_mismatch_near_tie_count = record.token_mismatch_near_tie_count;
    metrics.token_mismatch_second_choice_count = record.token_mismatch_second_choice_count;
    metrics.token_mismatch_second_choice_max_relative_gap =
        record.token_mismatch_second_choice_max_relative_gap;
}

/// A refusal that also seals the pairs already measured before it (`metrics.paired_legs`).
fn with_measured_pairs(mut payload: ScorePayload, records: Vec<PairedLegRecord>) -> ScorePayload {
    payload.metrics.paired_legs = records;
    payload
}

/// Steps 2-4 of the official flow, given the already-measured `timing`: official GATING
/// (non-finite → floors → bands), then FULL-scope correctness on a fresh worker, then the
/// passing-score assembly. Separated from the timed phase so both the gating (with synthetic
/// in-band timings) and the orchestration (with a mock timed phase) are unit-testable.
fn finish_official<T, FC>(
    golden: &GoldenFixture,
    scoring: ScoringInputs,
    bands: AcceptanceBands,
    digests: RunDigests<'_>,
    commit: &str,
    timing: &TimingResult,
    mut spawn_correctness: FC,
) -> ScorePayload
where
    T: LineTransport,
    FC: FnMut() -> bench_runner::Result<Session<T>>,
{
    let (baseline_prefill_spt, baseline_decode_spt) = scoring.baselines();
    // 2. Official GATING, evaluated BEFORE correctness (Swift Score.swift:50-126,
    //    QwenRuntimeBenchmark.swift:513-558): non-finite score → floors (0.95) → acceptance
    //    bands (prefill ±5%, decode +2%/−5%). The FIRST failure reason (in that priority)
    //    fails the run; real timing is retained.
    let eval = evaluate_timed_run(
        timing.decode_seconds_per_token,
        timing.prefill_seconds_per_token,
        baseline_decode_spt,
        baseline_prefill_spt,
        bands,
        // THE RUN'S OWN FLOORS (David 2026-09-09), from the `--contract` track fixture. The same
        // value seals `metrics.{decode,prefill}_speedup_floor` on every payload below, so the
        // artifact states the floor this gate enforced.
        scoring.floors,
        // …and THE RUN'S OWN COMPOSITE WEIGHTS (STEP 1, David 2026-09-15), resolved from the same
        // fixture. The gate and the published score now read ONE weighting.
        scoring.weights,
    );
    if let Some(reason) = eval.first_failure_reason() {
        // RULING 2 (trusted-core, official path): this is the TIMED-band failure — non-finite
        // score / speedup floor / acceptance band — which Swift evaluates BEFORE correctness
        // runs. `official_failed_timed_band` retains the real measured timing surface but
        // BLANKS the correctness-derived audit fields (golden_hash="", case_count=0,
        // checked_steps=0) to byte-match Swift's `correctness == nil` failed score. This is
        // DISTINCT from the correctness-failure path below, which is left unchanged.
        return official_failed_timed_band(golden, digests, commit, reason, timing, scoring);
    }

    // 3. CORRECTNESS on the THIRD fresh worker, FULL scope.
    //
    // ⚠️ MINOR-1 (B-2): `CorrectnessScope::Full` evaluates base cases + anchors + free_run,
    // but NOT the BEHAVIOR / GPQA-TTFT gates — bench-core conformance does not yet execute
    // them (the report carries no `behavior` vector; `benchmark_requires_runtime_worker`
    // DETECTS a behavior-carrying golden but nothing evaluates it). Behavior/GPQA execution is
    // deferred to B-3. UNTIL THEN, official is NOT a complete correctness gate for a
    // behavior-carrying golden: a corrupted behavior case would PASS. Do not treat a passing
    // official run over such a golden as full-correctness evidence.
    //
    // A spawn failure fails closed.
    let mut correctness_session = match spawn_correctness() {
        Ok(s) => s,
        Err(e) => {
            return official_failed_with_timing(
                golden,
                digests,
                commit,
                format!("correctness worker spawn failed: {e}"),
                timing,
                scoring,
            )
        }
    };
    let report = {
        let mut adapter = SessionEngine {
            session: &mut correctness_session,
            drained_once: false,
        };
        match run_conformance(
            &mut adapter,
            golden,
            bench_core::constants::CORRECTNESS_STEPS,
            CorrectnessScope::Full,
            digests.model.vocab_size,
        ) {
            Ok(r) => r,
            Err(e) => {
                return official_failed_with_timing(
                    golden,
                    digests,
                    commit,
                    format!("{e}"),
                    timing,
                    scoring,
                )
            }
        }
    };
    // Close the final correctness sub-phase (the per-sequence barrier owner, as in
    // iterate_core) so no completed-work leaks; a barrier failure fails the run.
    if let Err(e) = correctness_session.close_phase() {
        return official_failed_with_timing(
            golden,
            digests,
            commit,
            format!("{e}"),
            timing,
            scoring,
        );
    }

    if !report.passed {
        let (case, step, error) = official_correctness_failure(&report);
        return official_failed_with_timing_and_case(
            golden,
            digests,
            commit,
            error,
            timing,
            scoring,
            // The REAL partial per-case checked-step sum through the failing gate (Swift
            // `correctness?.checkedSteps`, :1143): anchor-fail and free-run-fail yield the
            // accumulated sum, NOT the placeholder case count.
            CorrectnessFailure {
                first_failing_case: case,
                first_failing_step: step,
                checked_steps: report.checked_steps(),
            },
        );
    }

    // 4. PASSING official score: the weighted-geometric-mean score (never coarsened),
    //    real timing surface, full correctness case counts, and the resolved commit.
    let mut metrics = base_metrics(Mode::Official, golden, digests, scoring);
    apply_timing_metrics(&mut metrics, timing, scoring);
    seal_official_per_prompt(&mut metrics, golden, timing);
    metrics.passed_correctness = true;
    metrics.commit = commit.to_string();
    // checkGates == true ⇒ caseCount = totalCorrectnessCaseCount (Swift
    // QwenRuntimeCorrectness.swift:351). checked_steps is now the REAL per-case checked-step
    // SUM (Swift `runLayeredCorrectness` accumulator, :192-306) — a passing official run sums
    // every evaluated base/anchor/free-run case, NOT the placeholder case count. (Behavior/GPQA
    // steps remain the documented B-3 gap; see ConformanceReport::checked_steps.)
    metrics.case_count = golden.total_correctness_case_count() as i64;
    metrics.checked_steps = report.checked_steps();
    ScorePayload {
        score: Some(eval.score),
        passed: true,
        metrics,
    }
}

/// GATES-ONLY official run (seam 1): `MLXFAST_BENCHMARK_SKIP_TIMED=1` skips the timed phases and
/// runs ONLY the correctness gates, sealing a `partial_result=true` gates-score — the seam-1
/// shape the paired overlay (`overlay-timing`) later completes with the measured timing. This is
/// benchd's parity implementation of the reference `mlxfast-swift` SKIP_TIMED path
/// (`main.swift@b26f76f:386,397` → `QwenRuntimeBenchmark.swift@b26f76f:457`; #132/F-7 corrected
/// this from `main.swift:321-322`, which is the local branch): a passing gates run is
/// `passed=true`, `partial_result=true`,
/// `passed_correctness=true`, `error==""`, `score=null`, with the timing fields left at their
/// zero placeholders (the overlay owns them). No timed phase runs, so no paired baselines are
/// needed here (correctness is oracle-only). Pure over the transport so tests drive it with an
/// in-process `MockEngine`.
/// The OFFICIAL path's baseline pair, in the reference's own resolution order (#132/F-2).
///
/// `pairedBaseline ?? benchmarkGolden.resolvedBaseline*`, where `resolvedBaseline*` is itself
/// `golden's declared pair ?? MLXFastConstants.officialBaseline*`
/// (`QwenRuntimeBenchmark.swift@b26f76f:441-445`, `Golden.swift@b26f76f:220-226`).
///
/// **Why this exists.** The reference OVERWRITES its baseline locals at `:442-445`, and it does so
/// BEFORE the `skipTimedBenchmark` branch at `:457` and before every `makeFailedScore` in
/// `benchmarkWithWorker` (`:533,:545,:560,:625,:638,:680,:690`). So on the official path the
/// initialiser values at `:350-351` are never what a sealed record carries — the resolved pair is.
/// #132(a) moved benchd's `base_metrics` default from `0.0` to the CONSTANTS, which is right for
/// the LOCAL surface (there the reference genuinely seals its constants) but leaked into the three
/// official payloads that no later step overwrites: the gates-only PASSING partial, the gates
/// failure, and `official_failed`. Those now resolve properly rather than inheriting either the
/// old `0.0` or the local surface's constants.
///
/// The env override is read here, mirroring the reference's inline
/// `try PairedBaselineOverride.fromEnvironment()` at `:441`. A half-set/invalid pair resolves to
/// `None` (falling through to the golden) rather than erroring: on the TIMED official path
/// `execute_iterate` has already hard-errored on exactly that condition before reaching this code,
/// and on the gates-only path benchd has never validated the env pair at all. That second gap is
/// PRE-EXISTING and untouched here — noted so it is not mistaken for something this helper
/// introduced.
pub fn official_resolved_baselines(
    contract: &bench_core::contract::Contract,
    golden: &GoldenFixture,
    track_id: &str,
) -> Result<(f64, f64), String> {
    let pair = official_resolved_baseline_pair(golden, track_id, contract)?;
    // B2 — the SCORING-TIME REGIME FENCE. A track must declare WHAT it scores — the batch size of
    // its scored point and the composite exponents (`docs/scored-regime-and-prefill-window.md`) —
    // before this resolution hands a denominator to a scored run.
    //
    // It sits OUTSIDE the resolution above, so EVERY source is fenced: the pair can arrive from
    // `MLXFAST_PAIRED_BASELINE_*` or from the golden without ever reaching the track table, and a
    // fence those two paths step around is not a fence. It sits AFTER it so a track missing both
    // facts still refuses with the more specific message — the baseline refusal names the two
    // sources it tried, which is what an operator needs first.
    // CONTRACT-FIRST (STEP 1, David 2026-09-15): the fixture's declared regime when it makes one,
    // the per-track table when it does not. Same fence, same position, resolved source.
    bench_core::contract::scored_regime(contract, track_id)?;
    Ok(pair)
}

/// The resolution itself: `pairedBaseline ?? golden's declared pair ?? the track's captured pair`,
/// in the reference's order. Split out of [`official_resolved_baselines`] so its every early return
/// passes through that function's regime fence.
fn official_resolved_baseline_pair(
    golden: &GoldenFixture,
    track_id: &str,
    contract: &bench_core::contract::Contract,
) -> Result<(f64, f64), String> {
    if let Some((prefill, decode)) = paired_baseline_override()? {
        return Ok((prefill, decode));
    }
    // Per-FIELD `?? officialBaseline*`, exactly as `resolvedBaseline*` is defined. The loader
    // already enforces the pair all-or-nothing, so the two spellings cannot diverge in practice —
    // this one just cannot drift from the reference's definition if that ever changes.
    //
    // The last source is the TRACK's captured pair, not a global. When the track has no captured
    // pair the resolution REFUSES BY NAME rather than falling back to some other track's numbers:
    // the message names the track, the pending sentinel, and the two sources already tried.
    let declared = golden.benchmark.as_ref();
    let golden_prefill = declared.and_then(|b| b.baseline_prefill_seconds_per_token);
    let golden_decode = declared.and_then(|b| b.baseline_decode_seconds_per_token);
    if let (Some(prefill), Some(decode)) = (golden_prefill, golden_decode) {
        return Ok((prefill, decode));
    }
    let track = bench_core::contract::official_baseline(contract, track_id)
        .map(|r| r.value)
        .map_err(|e| {
            format!(
                "no official baseline pair: \
             MLXFAST_PAIRED_BASELINE_{{PREFILL,DECODE}}_SECONDS_PER_TOKEN supplied none and the \
             golden declares no complete benchmark.baseline_{{prefill,decode}}_seconds_per_token \
             pair, and {e}"
            )
        })?;
    Ok((
        golden_prefill.unwrap_or(track.prefill_seconds_per_token),
        golden_decode.unwrap_or(track.decode_seconds_per_token),
    ))
}

/// The strict 40-char lowercase-hex shape of a RECORDED dispatch sha (the workflow-authored
/// `candidate.sha`). Distinct from the 7..=40 `is_commit_sha_hex` predicate that gates the
/// `metrics.commit` FIELD: a recorded dispatch sha is always a full commit id, so the record is
/// held to the strict shape the trusted shell validates before it writes the file.
fn is_dispatch_record_sha(s: &str) -> bool {
    s.len() == 40
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A proposed commit AGREES with the record iff it equals the record or is a hex PREFIX of it (the
/// engine's `commitIdentifier` may emit `git rev-parse --short HEAD`). The 40-hex record is the
/// authority either way; this is the disagreement predicate the seal binds on.
fn commit_agrees_with_record(record: &str, proposed: &str) -> bool {
    record == proposed || record.starts_with(proposed)
}
/// AUTHOR-AT-SEAL (DECIDE-3). The SOLE authority for the sealed `metrics.commit` is the sha the
/// in-repo dispatch script RECORDED from the CI/yukon dispatch context — the challenger
/// `candidate.sha` shape: a trusted-workflow-authored 40-hex commit id. benchd AUTHORS the sealed
/// commit FROM that record; a competitor-proposed commit — `MLXFAST_COMMIT_SHA`, which the engine's
/// `commitIdentifier` emits — is DEFENCE-IN-DEPTH ONLY: present-and-disagreeing is a die-class
/// refuse, never a silent override of the record. Participant git state is deliberately NOT an
/// input here: `git rev-parse` is unusable under the ranked sandbox (dubious-ownership under
/// `env -i`), which is precisely why the trusted dispatch context, not the checkout, is authority.
///
/// - `dispatch_record`: the RECORDED dispatched sha (trimmed `candidate.sha` contents), or `None`
///   when no dispatch context is present. `Some(_)` means a dispatch PROMISED a record, so a
///   malformed/empty value is a refuse — never a silent fall-through to the git identity.
/// - `proposed`: the raw, untrusted `MLXFAST_COMMIT_SHA` (or `None`).
/// - `scoring_mode`: `true` on a SCORING/ranked seal (the default measure-job mode; `false` only
///   under `--local-dev`). A scoring seal is FAIL-CLOSED: an absent dispatch record REFUSES rather
///   than falling back to the box git identity — the `git_rev_parse_head` fallback could
///   otherwise seal an EMPTY `metrics.commit` on a scoring run whose outer dispatch
///   never exported the context (the present-but-unwired threat). The git fallback survives ONLY in
///   dev/local mode, gated on the SAME `--local-dev` signal the official pair loop keys on
///   (`cfg.local_pair_budget`).
///
/// Contract:
/// - record present → it MUST be strict 40-char lowercase-hex (the workflow-authored shape); else
///   REFUSE. The sealed commit is AUTHORED from it. If `proposed` is present, valid-hex, and
///   neither equal to nor a hex prefix of the record (the engine may emit a short form of the SAME
///   commit), the seal is REFUSED — the proposal disagrees with what was dispatched.
/// - record absent + `scoring_mode` → REFUSE (die-class): a scoring seal must not fall back to git.
/// - record absent + dev/local → fallback to [`commit_identifier`] (no bind), behaviour unchanged.
pub fn author_sealed_commit(
    dispatch_record: Option<&str>,
    proposed: Option<&str>,
    scoring_mode: bool,
) -> Result<String, String> {
    let record = match dispatch_record {
        // A dispatch context was present: this record is the authority even if it is junk (a
        // dispatch that promised a record must not fall through to the git identity).
        Some(r) => r.trim(),
        // No dispatch context. A SCORING/ranked seal fails closed here — never the box git HEAD,
        // which can seal an empty commit on a scoring run whose dispatch never wired the context.
        None if scoring_mode => {
            return Err(
                "author-at-seal: a scoring/ranked seal requires the dispatched commit record \
                 (candidate.sha via MLXFAST_CANDIDATE_SHA_FILE); none was present — refusing to \
                 fall back to the box git identity on a scoring run (pass --local-dev for the \
                 unbound local resolution)"
                    .to_string(),
            );
        }
        // Dev/local only: the pre-existing, un-bound resolution.
        None => return Ok(commit_identifier(proposed)),
    };
    if !is_dispatch_record_sha(record) {
        return Err(format!(
            "author-at-seal: the dispatch record (candidate.sha) is {record:?}, not a \
             40-character lowercase-hex commit sha; refusing to seal a score against an \
             unidentified dispatch"
        ));
    }
    // Defence-in-depth cross-check. The record is already the authority; this only decides whether
    // a PRESENT proposal is a benign short form of the same commit or an actual DISAGREEMENT.
    if let Some(p) = proposed.map(str::trim).filter(|s| !s.is_empty()) {
        if is_commit_sha_hex(p) && !commit_agrees_with_record(record, p) {
            return Err(format!(
                "author-at-seal: the proposed commit (MLXFAST_COMMIT_SHA) {p:?} disagrees with the \
                 dispatched record {record:?}; refusing (the dispatch record is the sole authority \
                 for metrics.commit)"
            ));
        }
    }
    Ok(record.to_string())
}

pub fn official_gates_only<T, FC>(
    golden: &GoldenFixture,
    scoring: ScoringInputs,
    digests: RunDigests<'_>,
    commit: &str,
    mut spawn_correctness: FC,
) -> Result<ScorePayload, String>
where
    T: LineTransport,
    FC: FnMut() -> bench_runner::Result<Session<T>>,
{
    // #132/F-2: the BASELINE pair is not a zero placeholder and never was — the reference has
    // already overwritten it with `pairedBaseline ?? golden.resolvedBaseline*` before it reaches
    // the gates-only branch (`QwenRuntimeBenchmark.swift@b26f76f:442-445` then `:457`). The pair
    // is resolved ONCE by the CALLER, before anything spawns, and every record this function
    // seals carries that one pair. It is a parameter rather than an inline resolution because the
    // two official flows resolve it from different keys — the per-track table
    // ([`official_resolved_baselines`]) and, on the single-leg path, the run's PLATFORM — and a
    // second resolution here could disagree with the one the run was armed under.

    // CORRECTNESS on a fresh (sandboxed, in production) worker, FULL scope — identical to the
    // timed path's correctness step (official_core step 3), just with no preceding timed phase.
    // A spawn failure fails closed as a gates-only failed score.
    let mut correctness_session = match spawn_correctness() {
        Ok(s) => s,
        Err(e) => {
            return Ok(official_gates_failed(
                golden,
                scoring,
                digests,
                commit,
                GatesFailure::message(format!("correctness worker spawn failed: {e}")),
            ))
        }
    };
    let report = {
        let mut adapter = SessionEngine {
            session: &mut correctness_session,
            drained_once: false,
        };
        match run_conformance(
            &mut adapter,
            golden,
            bench_core::constants::CORRECTNESS_STEPS,
            CorrectnessScope::Full,
            digests.model.vocab_size,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(official_gates_failed(
                    golden,
                    scoring,
                    digests,
                    commit,
                    GatesFailure::message(format!("{e}")),
                ))
            }
        }
    };
    // Close the final correctness sub-phase (the per-sequence barrier owner) so no completed-work
    // leaks; a barrier failure fails the run.
    if let Err(e) = correctness_session.close_phase() {
        return Ok(official_gates_failed(
            golden,
            scoring,
            digests,
            commit,
            GatesFailure::message(format!("{e}")),
        ));
    }

    if !report.passed {
        let (case, step, error) = official_correctness_failure(&report);
        return Ok(official_gates_failed(
            golden,
            scoring,
            digests,
            commit,
            GatesFailure {
                error,
                first_failing_case: case,
                first_failing_step: step,
                checked_steps: report.checked_steps(),
            },
        ));
    }

    // PASSING gates-only score: partial_result=true (awaiting the timed overlay), null score,
    // full correctness case counts, resolved commit, zero timing placeholders.
    // #132/F-2: the BASELINE pair is not a zero placeholder and never was — the reference has
    // already overwritten it with `pairedBaseline ?? golden.resolvedBaseline*` before it reaches
    // the gates-only branch (`QwenRuntimeBenchmark.swift@b26f76f:442-445` then `:457`). `baselines`
    // is that resolved pair, resolved ONCE in `official_gates_only` before anything spawned, and
    // `base_metrics` seals it.
    let mut metrics = base_metrics(Mode::Official, golden, digests, scoring);
    metrics.passed_correctness = true;
    metrics.partial_result = true;
    metrics.commit = commit.to_string();
    metrics.case_count = golden.total_correctness_case_count() as i64;
    metrics.checked_steps = report.checked_steps();
    Ok(ScorePayload {
        score: None,
        passed: true,
        metrics,
    })
}

/// The correctness audit surface a gates-only FAILURE carries. One value rather than four
/// arguments: the four are written together at every site and read together by the seal, and a
/// step index or a checked-step count that travels apart from the error it belongs to is exactly
/// the kind of drift the seal cannot detect.
struct GatesFailure {
    error: String,
    first_failing_case: Option<String>,
    first_failing_step: Option<i64>,
    checked_steps: i64,
}

impl GatesFailure {
    /// A failure with no correctness report behind it (a spawn, conformance, or barrier fault):
    /// no case, no step, nothing checked.
    fn message(error: String) -> Self {
        Self {
            error,
            first_failing_case: None,
            first_failing_step: None,
            checked_steps: 0,
        }
    }
}

/// A FAILED gates-only payload (`score = null`, `passed = false`) that keeps `partial_result=true`
/// (it is still a gates-only shape, just fail-closed) with the correctness audit surface. No
/// timing is retained (the timed phase never ran).
fn official_gates_failed(
    golden: &GoldenFixture,
    scoring: ScoringInputs,
    digests: RunDigests<'_>,
    commit: &str,
    failure: GatesFailure,
) -> ScorePayload {
    // #132/F-2 — `baselines` is the pair `official_gates_only` resolved before it spawned
    // anything; the reference's failure records in `benchmarkWithWorker` are all reached AFTER
    // the `:442-445` overwrite, so they carry the resolved pair too.
    let mut payload = failed_score(
        failed_metrics(Mode::Official, golden, digests, scoring, None),
        FailureReport::correctness(
            failure.error,
            failure.first_failing_case,
            failure.first_failing_step,
            None,
            None,
        ),
        FailedAudit::Counts {
            case_count: golden.total_correctness_case_count() as i64,
            checked_steps: failure.checked_steps,
        },
        Some(commit),
    );
    // Still a gates-only shape, just fail-closed.
    payload.metrics.partial_result = true;
    payload
}

/// A FAILED official payload (`score = null`, `passed = false`) with NO retained timing
/// (used for oracle mismatch, benchmark-less golden, and pre-timing spawn faults).
/// `baselines` is the pair `official_core` already resolved (env ?? `--baseline-*` flags ??
/// golden) — #132/F-2. Threaded rather than re-derived, because `official_core`'s value is the
/// only one that has seen the flags; re-resolving from the golden here would silently drop them.
/// The reference does the same: its failure records read the locals it overwrote at
/// `QwenRuntimeBenchmark.swift@b26f76f:442-445`, not the initialisers at `:350-351`.
fn official_failed(
    golden: &GoldenFixture,
    digests: RunDigests<'_>,
    commit: &str,
    error: String,
    scoring: ScoringInputs,
) -> ScorePayload {
    let cases = golden.total_correctness_case_count() as i64;
    failed_score(
        failed_metrics(Mode::Official, golden, digests, scoring, None),
        FailureReport::message(error, false),
        FailedAudit::Counts {
            case_count: cases,
            checked_steps: cases,
        },
        Some(commit),
    )
}

/// A FAILED official payload that RETAINS the real timing surface (floor/band/finite failure
/// and post-timing correctness/barrier faults): the timed phases DID measure real numbers,
/// so the payload carries them (Swift's failed score keeps the measured decode/prefill spt +
/// speedups) while `score` stays `null` and `passed = false`.
fn official_failed_with_timing(
    golden: &GoldenFixture,
    digests: RunDigests<'_>,
    commit: &str,
    error: String,
    timing: &TimingResult,
    scoring: ScoringInputs,
) -> ScorePayload {
    official_failed_with_timing_and_case(
        golden,
        digests,
        commit,
        error,
        timing,
        scoring,
        // Pre-correctness/barrier timed faults have no ConformanceReport to sum; retain the
        // prior placeholder (golden case-count) for these UNLISTED fault paths — the listed
        // correctness-failure path below passes the real `report.checked_steps()`.
        CorrectnessFailure {
            first_failing_case: None,
            first_failing_step: None,
            checked_steps: golden.total_correctness_case_count() as i64,
        },
    )
}

/// RULING 2 (trusted-core, official path) — a FAILED official payload for a TIMED-BAND
/// failure (non-finite score / speedup floor / acceptance band), which Swift's official path
/// evaluates BEFORE correctness runs. Retains the real measured timing surface (like
/// [`official_failed_with_timing`]) but ALIGNS TO SWIFT by BLANKING the correctness-derived
/// audit fields: `golden_hash = ""`, `case_count = 0`, `checked_steps = 0`.
///
/// Why Swift blanks them here: `benchmarkWithWorker` is timed-FIRST, so the floor/band/finite
/// gates return via `makeFailedScore(correctness: correctnessReport, …)` while
/// `correctnessReport` is still `nil` — it is only assigned AFTER the correctness worker runs
/// (QwenRuntimeBenchmark.swift:520-558 precede :596). `failedScore` then defaults
/// `checkedSteps → correctness?.checkedSteps ?? 0` = 0, `caseCount → … ?? 0` = 0, and
/// `goldenHash → correctness?.goldenHash ?? ""` = "" (:1143/1144/1158). Official artifacts
/// feed downstream/organizer tooling that expects Swift's shape, so benchd (the PRODUCER)
/// aligns to Swift rather than zeroing in the differ (differ-side zeroing rejected).
///
/// SCOPE: only the timed-band-BEFORE-correctness path. The correctness-FAILURE path
/// ([`official_failed_with_timing_and_case`], where correctness DID run and Swift populates
/// these fields from the real report) is intentionally left unchanged.
fn official_failed_timed_band(
    golden: &GoldenFixture,
    digests: RunDigests<'_>,
    commit: &str,
    error: String,
    timing: &TimingResult,
    scoring: ScoringInputs,
) -> ScorePayload {
    // Align to Swift: correctness never ran at the timed band, so the correctness-derived
    // audit fields carry Swift's `correctness == nil` defaults.
    failed_score(
        failed_metrics(Mode::Official, golden, digests, scoring, Some(timing)),
        FailureReport::message(error, false),
        FailedAudit::Blank,
        Some(commit),
    )
}

/// A FAILED official payload for a benchmark-ORACLE token mismatch — a TIMED-phase failure
/// that Swift's timed-first official path hits BEFORE correctness runs, so it returns via
/// `makeFailedScore(correctness: correctnessReport /* still nil */)`
/// (QwenRuntimeBenchmark.swift:668-676 → failedScore :1143-1158).
///
/// Aligns to Swift on BOTH sides of the RULING-2 principle (producer matches Swift):
/// - BLANK the correctness-derived audit fields: `golden_hash = ""`, `case_count = 0`,
///   `checked_steps = 0` (`correctness?.… ?? default`).
/// - RETAIN the resolved baselines: `baseline_{prefill,decode}_seconds_per_token` carry the
///   golden/paired baseline values (resolved at :434-435, BEFORE the timed phase, and passed
///   straight into `failedScore`). The MEASURED decode/prefill spt stay 0 (the timed phase
///   never completed a trustworthy measurement), so `apply_timing_metrics` is NOT used here.
///
/// `first_failing_case` is always "benchmark"; `first_failing_step` is the decode-token step
/// (or None for the prefill/seed classes); expected/actual tokens are ALWAYS null (Swift
/// makeFailedScore).
fn official_failed_timed_oracle(
    golden: &GoldenFixture,
    digests: RunDigests<'_>,
    commit: &str,
    error: String,
    first_failing_step: Option<i64>,
    scoring: ScoringInputs,
) -> ScorePayload {
    let (baseline_prefill_spt, baseline_decode_spt) = scoring.baselines();
    // Correctness never ran (timed-first): blank the correctness audit surface to Swift's
    // `correctness == nil` defaults.
    let mut payload = failed_score(
        failed_metrics(Mode::Official, golden, digests, scoring, None),
        FailureReport::correctness(
            error,
            Some("benchmark".to_string()),
            first_failing_step,
            None,
            None,
        ),
        FailedAudit::Blank,
        Some(commit),
    );
    // Retain the resolved baselines (Swift keeps them; only the MEASURED spt stay 0).
    payload.metrics.baseline_prefill_seconds_per_token = finite_nonneg(baseline_prefill_spt);
    payload.metrics.baseline_decode_seconds_per_token = finite_nonneg(baseline_decode_spt);
    payload
}

/// As [`official_failed_with_timing`], plus the correctness `first_failing_*` fields for a
/// correctness-gate failure.
fn official_failed_with_timing_and_case(
    golden: &GoldenFixture,
    digests: RunDigests<'_>,
    commit: &str,
    error: String,
    timing: &TimingResult,
    scoring: ScoringInputs,
    correctness: CorrectnessFailure,
) -> ScorePayload {
    let CorrectnessFailure {
        first_failing_case,
        first_failing_step,
        checked_steps,
    } = correctness;
    failed_score(
        failed_metrics(Mode::Official, golden, digests, scoring, Some(timing)),
        // A floor/band failure means correctness never ran (Swift returns before it); a
        // correctness failure means it ran and failed. Either way passed_correctness = false.
        // OFFICIAL leaves expected/actual NULL (Swift `failedScore` reads only the explicit param
        // — nil on every official path — never `correctness?.expectedToken`;
        // QwenRuntimeBenchmark.swift:1155-1156). This is the LOCAL/OFFICIAL split: the local path
        // (iterate.rs) DOES populate them.
        FailureReport::correctness(error, first_failing_case, first_failing_step, None, None),
        // caseCount stays the golden TOTAL (Swift correctness-fail `caseCount =
        // totalCorrectnessCaseCount`); checked_steps is the caller-supplied real per-case sum
        // (partial through the failing gate) or, for report-less timed faults, the placeholder.
        FailedAudit::Counts {
            case_count: golden.total_correctness_case_count() as i64,
            checked_steps,
        },
        Some(commit),
    )
}

/// Extract the official correctness failure branding from a conformance report, using the
/// Swift per-gate message (QwenRuntimeCorrectness.swift): base → "teacher-forced token
/// mismatch", anchor → "anchor token mismatch", free-run → "free-run token mismatch". Runs
/// in the layered order Swift evaluates (base → anchors → free_run).
fn official_correctness_failure(
    report: &ConformanceReport,
) -> (Option<String>, Option<i64>, String) {
    // Reuse the shared first-failure walk (base → anchors → free_run) for case/step, then
    // attach the Swift per-gate message keyed on which vector produced it. Expected/actual
    // tokens are NOT returned: the OFFICIAL correctness-fail path leaves them null (see the
    // call site). first_failing_step follows Swift `correctness.firstFailingStep`.
    if let Some(f) = first_conformance_failure(report) {
        let is_anchor = report.anchors.iter().any(|a| a.name == f.case && !a.passed);
        let error = if f.is_base_case {
            "teacher-forced token mismatch".to_string()
        } else if is_anchor {
            "anchor token mismatch".to_string()
        } else {
            "free-run token mismatch".to_string()
        };
        // Swift `compareAnchorToken` reports firstFailingStep = 0 on an anchor fail
        // (QwenRuntimeCorrectnessCompare.swift:481) — NOT nil. `first_conformance_failure`
        // returns None for anchors (an anchor has no per-token step index), so override to 0
        // here to byte-match Swift's official anchor-fail shape. Base and free-run keep their
        // mismatch step (Swift base/free-run firstFailingStep = comparison step).
        let step = if is_anchor { Some(0) } else { f.step };
        (Some(f.case), step, error)
    } else {
        (None, None, "correctness gate failed".to_string())
    }
}

/// THE WINDOW THE TEST-ONLY WRAPPERS AND THE UNIT TESTS MEASURE UNDER — the 125B MLX track's own,
/// as TEST DATA. Shared with `crate::iterate::test_window`, which states the same four values for
/// the same reason: STEP 3 (David 2026-09-15) deleted `WindowShape::table`, so a test that needs a
/// window states one.
#[cfg(test)]
pub(crate) fn test_window() -> bench_core::constants::WindowShape {
    crate::iterate::test_window()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iterate::{DirDigest, HarnessIdentity};
    use crate::testgolden::track_baseline;

    use crate::testgolden::TEST_BASELINE;
    use bench_core::constants::{
        BENCHMARK_DECODE_SEED_TOKENS, BENCHMARK_DECODE_STEPS, BENCHMARK_PREFILL_PROMPT_TOKENS,
        CORRECTNESS_PROMPT_TOKENS,
    };
    use bench_core::golden::load_golden_fixture;
    use bench_runner::mock::MockEngine;
    use serde_json::json;
    use std::cell::Cell;

    /// THE [`OfficialParams`] THESE TESTS RUN UNDER: the shared `TEST_BASELINE` pair against the
    /// in-tree window, the stub commit, and no spec — the historical defaults the deleted
    /// `official_core` wrapper used to supply positionally. A test that varies one of the others
    /// says so in a struct update: `OfficialParams { spec: Some(spec), ..test_run(…) }`.
    fn test_run<'a>(
        golden: &'a GoldenFixture,
        digests: RunDigests<'a>,
        window: &'a WindowShape,
    ) -> OfficialParams<'a> {
        OfficialParams {
            golden,
            scoring: ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
            digests,
            commit: "deadbeef",
            spec: None,
            window,
        }
    }

    const PREFILL_TOKEN: i64 = 5;
    const SEED_TOKEN: i64 = 6;

    /// Distinct oracle decode tokens so a corrupted-oracle test can target one step.
    fn oracle_decode_tokens() -> Vec<i64> {
        (0..BENCHMARK_DECODE_STEPS as i64)
            .map(|i| 700 + i)
            .collect()
    }

    /// A golden whose benchmark oracle is (5, 6, [700..828)) and whose primary case is
    /// conformant to teacher-forced [2; 64]. `gates` is spliced into `correctness_gates`.
    fn official_golden(gates: Option<serde_json::Value>) -> GoldenFixture {
        official_golden_with_oracle(oracle_decode_tokens(), gates)
    }

    /// The same golden with a CHOSEN benchmark oracle tape, so a test can put two goldens that
    /// disagree on the decode tokens in front of the two legs of a paired run.
    fn official_golden_with_oracle(
        decode_tokens: Vec<i64>,
        gates: Option<serde_json::Value>,
    ) -> GoldenFixture {
        let mut doc = json!({
            "version": 1,
            "model_type": "qwen4_exp_text",
            "cases": [
                { "name": "case-a", "prompt_tokens": vec![1i64; CORRECTNESS_PROMPT_TOKENS], "expected_tokens": vec![2i64; 64] }
            ],
            "benchmark": {
                "prefill_prompt_tokens": vec![1i64; BENCHMARK_PREFILL_PROMPT_TOKENS],
                "expected_prefill_token": PREFILL_TOKEN,
                "decode_seed_tokens": vec![1i64; BENCHMARK_DECODE_SEED_TOKENS],
                "expected_decode_seed_token": SEED_TOKEN,
                "expected_decode_tokens": decode_tokens,
            }
        });
        if let Some(g) = gates {
            doc["correctness_gates"] = g;
        }
        let bytes = serde_json::to_vec(&doc).unwrap();
        load_golden_fixture(
            &bytes,
            64,
            CORRECTNESS_PROMPT_TOKENS,
            crate::testgolden::identity_125b(),
            Some("qwen4_exp_text"),
            None,
            None,
        )
        .unwrap()
    }

    /// #134 — the OFFICIAL `score.json` `metrics.error` SINK. This is the most exposed of the
    /// three seal boundaries: official worker stderr is never forwarded (so the retained tail is
    /// its only channel) and the artifact travels. Secret-SHAPED without any `expected`/`actual`
    /// trigger word, so the pre-existing keyword filter would pass every byte through.
    #[test]
    fn official_failed_scrubs_engine_text_before_sealing_metrics_error() {
        let payload = official_failed(
            &official_golden(None),
            RunDigests::for_test(&DirDigest::empty()),
            "commit",
            format!(
                "protocol violation: engine closed the stream before returning a response (worker \
                 exited with status 9; worker stderr tail: open \
                 /Users/operator/pool-goldens/sample-001.json failed | \
                 AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCY | host=api.example.internal | \
                 {})",
                "P".repeat(8192)
            ),
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
                "secret-tier content sealed into official metrics.error: {secret:?}"
            );
        }
        assert!(
            sealed.len() <= bench_runner::SEALED_REASON_BYTE_LIMIT,
            "sealed metrics.error not capped: {} bytes",
            sealed.len()
        );
        assert!(
            sealed.starts_with("protocol violation: engine closed the stream"),
            "signature lost: {sealed}"
        );
        assert!(
            sealed.contains("sample-001.json"),
            "diagnosis lost: {sealed}"
        );
    }

    const PROMPT_MARKER: &str = "HIDDENPROMPTqzx7Kv9w";

    /// What an engine-supplied identity string seals as.
    fn sealed_text(text: &str) -> String {
        format!(
            "sha256={} bytes={}",
            bench_core::hash::sha256_hex(text.as_bytes()),
            bench_runner::seal_byte_len(text.len())
        )
    }

    /// Every run of `marker` at least 4 bytes long that `text` contains.
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

    /// The hidden-prompt read-back channel: an engine that fails ON PURPOSE with prompt text in its
    /// `error` reason or in the stderr tail benchd retains when it dies. The SEALED official
    /// payload — every field, not just `metrics.error` — carries not one run of that text, and
    /// does carry its digest and byte count.
    #[test]
    fn official_sealed_record_carries_no_engine_text_only_its_digest() {
        let reason = format!("the prompt was {PROMPT_MARKER}");
        let tail = format!("worker exited with status 3; {reason}");
        let golden = official_golden(None);
        let run = |fail_on_spawn: usize, stderr: bool| {
            let spawn_n = Cell::new(0usize);
            official_core_windowed(
                test_run(
                    &golden,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                ),
                TEST_BASELINE.bands,
                || {
                    let n = spawn_n.get();
                    spawn_n.set(n + 1);
                    if n != fail_on_spawn {
                        Session::connect(conformant_engine()).map(|(s, _)| s)
                    } else if stderr {
                        // A worker that died before hello, with the prompt in its stderr tail.
                        Err(RunnerError::ProtocolEngineText {
                            detail: "engine closed the stream before returning a response"
                                .to_string(),
                            engine_text: tail.clone(),
                        })
                    } else {
                        Session::connect(conformant_engine().error_on("free_decode_run", &reason))
                            .map(|(s, _)| s)
                    }
                },
                |_phase: &str| Ok(()),
            )
        };
        for (label, payload, engine_text) in [
            ("engine error reason", run(0, false), &reason),
            ("worker stderr tail", run(0, true), &tail),
        ] {
            assert!(!payload.passed, "{label}");
            let sealed = serde_json::to_string(&payload).unwrap();
            assert_eq!(
                marker_runs_in(&sealed, PROMPT_MARKER),
                Vec::<String>::new(),
                "{label}: engine text sealed: {}",
                payload.metrics.error
            );
            assert!(
                payload
                    .metrics
                    .error
                    .contains(&bench_runner::seal_engine_text(engine_text)),
                "{label}: digest missing: {}",
                payload.metrics.error
            );
            assert!(
                payload.metrics.error.contains("engine-text-bytes="),
                "{label}: {}",
                payload.metrics.error
            );
        }
    }

    /// A PASSING sealed record carries no engine free text and no raw per-round vector: every
    /// hello string and the self-reported verify mode seal as digests, and `acceptance_lengths`
    /// seals as `{count, mean}`.
    #[test]
    fn a_passing_sealed_record_carries_no_hello_text_and_no_raw_acceptance_vector() {
        let golden = official_golden(None);
        let hist: Vec<u32> = (0..32).map(|i| if i % 2 == 0 { 1 } else { 3 }).collect();
        let rounds = hist.len();
        let n: u32 = hist.iter().sum();
        let mode = format!("rect-{PROMPT_MARKER}");
        let audit = bench_core::free_run::verify_consistency(
            &bench_core::free_run::FreeRunResponse {
                tokens_len: n as usize,
                acceptance_lengths: hist,
                drafted_total: 2 * rounds as u64,
                accepted_total: u64::from(n) - rounds as u64,
                committed_total: u64::from(n),
                verify_replay_disagreements: None,
                verification: Some(bench_core::free_run::VerificationReport {
                    mode: mode.clone(),
                    rectangular_rounds: Some(rounds as u64),
                    serial_rounds: Some(0),
                }),
            },
            n,
            rounds as i64 + 1,
        )
        .unwrap();
        let timing = TimingResult {
            effective_spec: Some(SpecConfig::mtp(2)),
            free_run_audit: Some(audit),
            ..in_band_timing()
        };
        let mut payload = finish_official(
            &golden,
            ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
            TEST_BASELINE.bands,
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            &timing,
            || Session::connect(conformant_engine()).map(|(s, _)| s),
        );
        assert!(payload.passed, "error={}", payload.metrics.error);
        let text = |field: &str| format!("{field}-{PROMPT_MARKER}");
        seal_engine_identity(
            &mut payload.metrics,
            &bench_runner::Hello {
                nonce: "n".to_string(),
                protocol_version: Some(1),
                backend: Some(text("backend")),
                device: Some(text("device")),
                capabilities: Vec::new(),
                spec_modes: Vec::new(),
                head_provenance: Some(bench_protocol::HeadProvenance {
                    sha256: text("head"),
                    bytes: 1,
                    file_count: 1,
                }),
                max_batch_size: None,
                runner: Some(bench_protocol::RunnerIdentity {
                    id: text("id"),
                    model_type: text("model"),
                    manifest_sha256: text("manifest"),
                    build: text("build"),
                }),
                resident: None,
            },
            None,
        );
        let json = payload.to_sealed_json().unwrap();
        assert_eq!(
            marker_runs_in(&json, PROMPT_MARKER),
            Vec::<String>::new(),
            "engine text sealed"
        );
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let m = &v["metrics"];
        for (key, raw) in [
            ("engine_backend", text("backend")),
            ("engine_device", text("device")),
            ("runner_id", text("id")),
            ("runner_model_type", text("model")),
            ("runner_build", text("build")),
            ("spec_verification_mode", mode.clone()),
        ] {
            assert_eq!(m[key], sealed_text(&raw), "{key}");
        }
        // Engine "digests" that are not 64 lowercase hex seal as `invalid`, mirror included.
        assert_eq!(m["head_provenance_sha256"], "invalid");
        assert_eq!(m["per_prompt"][0]["head_provenance_sha256"], "invalid");
        assert_eq!(m["runner_manifest_sha256"], "invalid");
        assert_eq!(
            m["acceptance_lengths"],
            serde_json::json!({"count": rounds, "mean": 2.0})
        );
        // No array anywhere in the record is the per-round vector (or any array that long).
        fn longest_array(v: &serde_json::Value) -> usize {
            match v {
                serde_json::Value::Array(a) => {
                    a.iter().map(longest_array).max().unwrap_or(0).max(a.len())
                }
                serde_json::Value::Object(o) => o.values().map(longest_array).max().unwrap_or(0),
                _ => 0,
            }
        }
        assert!(longest_array(&v) < rounds, "a per-round vector was sealed");
    }

    /// Engine digests never seal as themselves: `valid`/`invalid` for the manifest digest, and
    /// `match`/`mismatch`/`valid`/`invalid` for the head digest against the reference leg's.
    #[test]
    fn engine_digests_seal_as_a_verdict_never_the_value() {
        use crate::iterate::{seal_digest_validity, seal_head_digest};
        let good = "0123456789abcdef".repeat(4);
        let other = "fedcba9876543210".repeat(4);
        assert_eq!(seal_digest_validity("f", &good), "valid");
        assert_eq!(seal_head_digest(&good, Some(&good)), "match");
        assert_eq!(seal_head_digest(&good, Some(&other)), "mismatch");
        assert_eq!(seal_head_digest(&good, None), "valid");
        for bad in [
            good.to_uppercase(),
            good[..63].to_string(),
            format!("{good}0"),
            format!("{}{PROMPT_MARKER}", &good[..44]),
            String::new(),
        ] {
            assert_eq!(seal_digest_validity("f", &bad), "invalid", "{bad:?}");
            assert_eq!(seal_head_digest(&bad, Some(&bad)), "invalid", "{bad:?}");
        }
    }

    /// An engine that reports MARKER NUMBERS — a non-zero `cache_memory`, a precise peak RAM, a
    /// head digest other than the reference leg's — gets none of them into the sealed record.
    #[test]
    fn a_sealed_record_carries_no_engine_chosen_marker_numbers() {
        const CACHE_MARKER: i64 = 1_234_567;
        let golden = official_golden(None);
        let failed = official_core_windowed(
            test_run(
                &golden,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            TEST_BASELINE.bands,
            || Session::connect(conformant_engine().cache_memory(CACHE_MARKER)).map(|(s, _)| s),
            |_phase: &str| Ok(()),
        );
        assert!(!failed.passed);
        assert!(
            failed.metrics.error.contains("cache_memory non-zero"),
            "{}",
            failed.metrics.error
        );
        let json = failed.to_sealed_json().unwrap();
        assert!(!json.contains("1234567"), "{json}");

        let timing = TimingResult {
            peak_ram_gb: 13.371,
            ..in_band_timing()
        };
        let mut payload = finish_official(
            &golden,
            ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
            TEST_BASELINE.bands,
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            &timing,
            || Session::connect(conformant_engine()).map(|(s, _)| s),
        );
        assert!(payload.passed, "error={}", payload.metrics.error);
        let head = "0123456789abcdef".repeat(4);
        let manifest = "13579bdf".repeat(8);
        let pinned = "fedcba9876543210".repeat(4);
        seal_engine_identity(
            &mut payload.metrics,
            &bench_runner::Hello {
                nonce: "n".to_string(),
                protocol_version: Some(1),
                backend: None,
                device: None,
                capabilities: Vec::new(),
                spec_modes: Vec::new(),
                head_provenance: Some(bench_protocol::HeadProvenance {
                    sha256: head.clone(),
                    bytes: 1,
                    file_count: 1,
                }),
                max_batch_size: None,
                runner: Some(bench_protocol::RunnerIdentity {
                    id: "r".to_string(),
                    model_type: "m".to_string(),
                    manifest_sha256: manifest.clone(),
                    build: "b".to_string(),
                }),
                resident: None,
            },
            Some(&pinned),
        );
        let json = payload.to_sealed_json().unwrap();
        for marker in ["13.371", "13.37", &head, &manifest] {
            assert!(!json.contains(marker), "{marker} sealed: {json}");
        }
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let m = &v["metrics"];
        // 13.371 → 13.5 (nearest 0.5 GB) → 14 (the public 2-significant-figure coarsening).
        assert_eq!(m["peak_ram_gb"], 14.0);
        assert_eq!(m["head_provenance_sha256"], "mismatch");
        assert_eq!(m["per_prompt"][0]["head_provenance_sha256"], "mismatch");
        assert_eq!(m["runner_manifest_sha256"], "valid");
    }

    /// benchd's OWN refusals reach the sealed record byte-for-byte: they carry benchd's comparison
    /// against the golden (speedups, one token id, positions), never engine text.
    #[test]
    fn benchd_generated_refusals_seal_unchanged() {
        for own in [
            "performance floor failed: decode_speedup=0.900000 floor=0.950000 \
             prefill_speedup=1.000000 floor=0.950000"
                .to_string(),
            "TIMED-REPLAY-REJECTED: pair 1: timed decode: the committed choice is 2.09 logits \
             below the reference's top logit, above the declared 2 maximum"
                .to_string(),
            format!(
                "{}: serial-control leg outside this box's band: the decode leg measured 0.02 \
                 seconds per token, and box \"spark-1\" is calibrated at 0.01 with a ceiling of \
                 0.0103 (1.03 of the mean); the box is slower than when it was calibrated; \
                 refusing to seal a score",
                crate::baseline::SERIAL_CONTROL_LEG_OUTSIDE_BAND
            ),
        ] {
            let payload = official_failed(
                &official_golden(None),
                RunDigests::for_test(&DirDigest::empty()),
                "commit",
                own.clone(),
                ScoringInputs::local(
                    TEST_BASELINE.prefill_seconds_per_token,
                    TEST_BASELINE.decode_seconds_per_token,
                ),
            );
            assert_eq!(payload.metrics.error, own);
        }
    }

    /// A stub engine conformant on BOTH the timed oracle and the teacher-forced base case.
    fn conformant_engine() -> MockEngine {
        MockEngine::new()
            .teacher_forced_tokens(vec![2i64; 64])
            .free_run_capable()
            .oracle_tokens(PREFILL_TOKEN, SEED_TOKEN, oracle_decode_tokens())
    }

    /// Run the full official_core against `golden` with a `timed` engine factory and the official
    /// Qwen baselines. NOTE: a MockEngine's ~0 wall-clock cannot
    /// sit inside the acceptance band, so a conformant mock FAILS on the band — used to prove
    /// the timed-first orchestration + oracle passed and the band gate is wired. The passing
    /// SCORE / correctness-scope assertions go through `finish_official` with a synthetic
    /// in-band TimingResult (see below).
    fn run_official<FT>(golden: &GoldenFixture, timed: FT) -> ScorePayload
    where
        FT: Fn() -> MockEngine,
    {
        official_core_windowed(
            test_run(
                golden,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            TEST_BASELINE.bands,
            || Session::connect(timed()).map(|(s, _)| s),
            // The historical entry point has no cool gate; the ranked one passes the real gate.
            |_phase: &str| Ok(()),
        )
    }

    /// The free-run audit's effective mean draft length carried by [`in_band_timing`] — an
    /// arbitrary but DISTINCTIVE value, so a sealed `per_prompt` record proves the number travelled
    /// from the audit rather than being defaulted.
    const IN_BAND_MEAN_DRAFT_LEN: f64 = 4.25;

    /// A REAL [`FreeRunAudit`], built the only way one can be built — by passing the §2.6
    /// consistency triple — so a test's audit is never a shape the live path could not produce.
    /// `acceptance_lengths` sums to N and the round counter is R+1 by construction.
    fn audit_for_test(
        acceptance_lengths: Vec<u32>,
        drafted_total: u64,
        accepted_total: u64,
    ) -> bench_core::free_run::FreeRunAudit {
        let n: u32 = acceptance_lengths.iter().sum();
        let rounds = acceptance_lengths.len() as i64;
        bench_core::free_run::verify_consistency(
            &bench_core::free_run::FreeRunResponse {
                tokens_len: n as usize,
                acceptance_lengths,
                drafted_total,
                accepted_total,
                committed_total: n as u64,
                verify_replay_disagreements: None,
                verification: None,
            },
            n,
            rounds + 1,
        )
        .expect("synthetic free-run audit must satisfy the consistency triple")
    }

    /// A synthetic TimingResult sitting EXACTLY on the baselines (speedups 1.0, in-band,
    /// floors pass, score 1.0) — the only way to drive the passing gate deterministically.
    fn in_band_timing() -> TimingResult {
        TimingResult {
            prefill_seconds_per_token: TEST_BASELINE.prefill_seconds_per_token,
            decode_seconds_per_token: TEST_BASELINE.decode_seconds_per_token,
            decode_steps: BENCHMARK_DECODE_STEPS,
            prefill_prompt_tokens: BENCHMARK_PREFILL_PROMPT_TOKENS,
            prefill_elapsed_seconds: TEST_BASELINE.prefill_seconds_per_token * 512.0,
            decode_elapsed_seconds: TEST_BASELINE.decode_seconds_per_token
                * BENCHMARK_DECODE_STEPS as f64,
            peak_ram_gb: 20.25,
            effective_spec: None,
            // mean(4, 4, 4, 5) == 4.25 == IN_BAND_MEAN_DRAFT_LEN.
            free_run_audit: Some(audit_for_test(vec![4, 4, 4, 5], 0, 0)),
            deferred_divergence: None,
            phase_window: None,
            emitted_tokens: Vec::new(),
        }
    }

    /// Drive `finish_official` (gating → correctness → assembly) with an in-band timing and a
    /// correctness engine — deterministic, no wall-clock dependence.
    fn finish_with<FC>(golden: &GoldenFixture, correctness: FC) -> ScorePayload
    where
        FC: Fn() -> MockEngine,
    {
        finish_official(
            golden,
            ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
            TEST_BASELINE.bands,
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            &in_band_timing(),
            || Session::connect(correctness()).map(|(s, _)| s),
        )
    }

    /// An official golden that DECLARES a baseline pair — the fixture that can tell the
    /// reference's resolution chain apart from the constants fallback (#132/F-2).
    fn official_golden_with_baselines(prefill: f64, decode: f64) -> GoldenFixture {
        let mut doc = json!({
            "version": 1,
            "model_type": "qwen4_exp_text",
            "cases": [
                { "name": "case-a", "prompt_tokens": vec![1i64; CORRECTNESS_PROMPT_TOKENS], "expected_tokens": vec![2i64; 64] }
            ],
            "benchmark": {
                "prefill_prompt_tokens": vec![1i64; BENCHMARK_PREFILL_PROMPT_TOKENS],
                "expected_prefill_token": PREFILL_TOKEN,
                "decode_seed_tokens": vec![1i64; BENCHMARK_DECODE_SEED_TOKENS],
                "expected_decode_seed_token": SEED_TOKEN,
                "expected_decode_tokens": oracle_decode_tokens(),
            }
        });
        doc["benchmark"]["baseline_prefill_seconds_per_token"] = json!(prefill);
        doc["benchmark"]["baseline_decode_seconds_per_token"] = json!(decode);
        let bytes = serde_json::to_vec(&doc).unwrap();
        load_golden_fixture(
            &bytes,
            64,
            CORRECTNESS_PROMPT_TOKENS,
            crate::testgolden::identity_125b(),
            Some("qwen4_exp_text"),
            None,
            None,
        )
        .unwrap()
    }

    /// The `MLXFAST_PAIRED_BASELINE_*` pair is process-global, and since #359 EVERY baseline
    /// resolution reads it. Any test that resolves a baseline — or that asserts the pair is unset —
    /// takes this lock, so the env-setting case below cannot race one of them.
    static PAIRED_BASELINE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The two keys of the trusted override, in refusal order.
    const PAIRED_BASELINE_KEYS: [&str; 2] = [
        "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN",
        "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN",
    ];

    /// Run `f` with the two override keys set as given (`None` = removed), restoring the prior
    /// values afterwards. Holds [`PAIRED_BASELINE_ENV_LOCK`] for the duration.
    fn with_paired_baseline_env<T>(vars: [Option<&str>; 2], f: impl FnOnce() -> T) -> T {
        let _guard = PAIRED_BASELINE_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let prev: Vec<Option<std::ffi::OsString>> =
            PAIRED_BASELINE_KEYS.iter().map(std::env::var_os).collect();
        for (key, value) in PAIRED_BASELINE_KEYS.iter().zip(vars) {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
        let out = f();
        for (key, value) in PAIRED_BASELINE_KEYS.iter().zip(prev) {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
        out
    }

    /// #359 — A MALFORMED TRUSTED OVERRIDE REFUSES ON THE `official.rs` RESOLUTION PATH TOO.
    ///
    /// `MLXFAST_PAIRED_BASELINE_{PREFILL,DECODE}_SECONDS_PER_TOKEN` used to be read here through
    /// `.ok().flatten()`: a half-set or non-finite pair was SILENTLY discarded and the resolution
    /// fell through to the golden's declared pair, pricing the run against a denominator the
    /// operator did not ask for. It now propagates, with the same message `main.rs` has always
    /// printed.
    ///
    /// The golden here DECLARES a pair, so under the old posture this resolved to `(0.0004,
    /// 0.014)`. That is the fall-through this test forbids.
    #[test]
    fn a_half_set_override_refuses_on_the_official_resolution_path() {
        // The three refusals, spelled out whole — a `\` continuation inside a literal would
        // smuggle the source indentation into the expectation and hide a padding-space defect.
        const TOGETHER: &str = "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN and MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN must be provided together";
        const PREFILL_FINITE: &str = "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN must be a finite positive seconds-per-token value";
        const DECODE_FINITE: &str = "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN must be a finite positive seconds-per-token value";

        let golden = official_golden_with_baselines(0.0004, 0.014);
        for (vars, expect) in [
            ([Some("0.0004"), None], TOGETHER),
            ([None, Some("0.014")], TOGETHER),
            ([Some("0"), Some("0.014")], PREFILL_FINITE),
            ([Some("0.0004"), Some("nan")], DECODE_FINITE),
        ] {
            let err = with_paired_baseline_env(vars, || {
                official_resolved_baseline_pair(
                    &golden,
                    bench_core::constants::TRACK_ID,
                    &crate::testgolden::track_fixture(),
                )
            })
            .unwrap_err();
            assert_eq!(err, expect, "vars {vars:?}");
        }

        // POSITIVE CONTROL — a WELL-FORMED pair still wins over the golden's declared pair, and
        // an ABSENT pair still falls through to it. Only the malformed case changed.
        assert_eq!(
            with_paired_baseline_env([Some("0.0007"), Some("0.021")], || {
                official_resolved_baseline_pair(
                    &golden,
                    bench_core::constants::TRACK_ID,
                    &crate::testgolden::track_fixture(),
                )
            })
            .unwrap(),
            (0.0007, 0.021)
        );
        assert_eq!(
            with_paired_baseline_env([None, None], || {
                official_resolved_baseline_pair(
                    &golden,
                    bench_core::constants::TRACK_ID,
                    &crate::testgolden::track_fixture(),
                )
            })
            .unwrap(),
            (0.0004, 0.014)
        );
    }

    /// B2 — the SCORING-TIME REGIME FENCE, and the proof it CANNOT BE BYPASSED.
    ///
    /// The baseline pair has three sources and only the last one reaches the track table, so a
    /// regime fence placed inside the resolution would be stepped around by either of the first
    /// two. Here the golden DECLARES a pair: the resolution succeeds for any track id whatsoever,
    /// and the refusal has to come from the fence outside it.
    #[test]
    fn official_resolution_refuses_an_undeclared_regime_even_when_the_golden_supplies_the_pair() {
        const UNDECLARED: &str = "qwen3.9-27b-mlx-v1";
        let _env = PAIRED_BASELINE_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        // The golden supplies the pair, so the resolution itself never consults the track table.
        let golden = official_golden_with_baselines(0.0004, 0.014);
        assert_eq!(
            official_resolved_baseline_pair(
                &golden,
                UNDECLARED,
                &bench_core::contract::Contract::NONE_DECLARED
            )
            .unwrap(),
            (0.0004, 0.014),
            "precondition: the golden's declared pair resolves for ANY track — that is the bypass"
        );

        let err = official_resolved_baselines(
            &bench_core::contract::Contract::NONE_DECLARED,
            &golden,
            UNDECLARED,
        )
        .unwrap_err();
        assert!(err.contains(UNDECLARED), "must name the track: {err}");
        assert!(
            err.contains(bench_core::constants::SCORED_REGIME_PENDING),
            "must name the pending sentinel: {err}"
        );
        assert!(
            err.contains("scored_batch_size"),
            "must name what the fixture did not declare: {err}"
        );

        // POSITIVE CONTROL — the same golden under a fixture that DOES declare the regime
        // resolves, so the refusal is about the declaration and nothing else.
        assert_eq!(
            official_resolved_baselines(
                &crate::testgolden::track_fixture(),
                &golden,
                bench_core::constants::TRACK_ID
            )
            .unwrap(),
            (0.0004, 0.014)
        );
    }

    /// The OFFICIAL resolution REFUSES BY NAME for a track with no captured baseline, instead of
    /// falling through to some other track's numbers.
    ///
    /// The precedence is unchanged — trusted env pair, then the golden's declared pair, then the
    /// track's captured pair — so this fires only on the third source, with the first two silent.
    /// The message names all three: the track, the pending sentinel, and the sources tried.
    #[test]
    fn official_resolution_refuses_an_uncaptured_track_by_name() {
        const UNCAPTURED: &str = "qwen3.9-27b-mlx-v1";
        let _env = PAIRED_BASELINE_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        for key in PAIRED_BASELINE_KEYS {
            assert!(
                std::env::var(key).is_err(),
                "precondition: {key} must be unset for this resolution to reach the track table"
            );
        }
        // This golden declares no baseline pair, so the golden source is silent too.
        let golden = official_golden(None);
        assert!(golden
            .benchmark
            .as_ref()
            .unwrap()
            .baseline_prefill_seconds_per_token
            .is_none());

        let err = official_resolved_baselines(
            &bench_core::contract::Contract::NONE_DECLARED,
            &golden,
            UNCAPTURED,
        )
        .unwrap_err();
        assert!(err.contains(UNCAPTURED), "must name the track: {err}");
        assert!(
            err.contains(bench_core::constants::OFFICIAL_BASELINE_PENDING),
            "must name the pending sentinel: {err}"
        );
        // The two tried sources, as ONE contiguous operator-facing phrase. Asserted whole rather
        // than by fragments: a lost `\` line continuation in the format literal renders as runs of
        // padding spaces, which every `contains("MLXFAST_PAIRED_BASELINE")`-style fragment check
        // still passes. This is the assertion that catches it.
        assert!(
            err.starts_with(
                "no official baseline pair: \
                 MLXFAST_PAIRED_BASELINE_{PREFILL,DECODE}_SECONDS_PER_TOKEN supplied none and \
                 the golden declares no complete \
                 benchmark.baseline_{prefill,decode}_seconds_per_token pair, and "
            ),
            "exact refusal phrasing: {err}"
        );
        assert!(
            !err.contains("  "),
            "the message must carry no padding-space runs: {err}"
        );

        // The SAME golden under this tree's own track FIXTURE resolves, so the refusal is about
        // the declaration and nothing else.
        assert_eq!(
            official_resolved_baselines(
                &crate::testgolden::track_fixture(),
                &golden,
                bench_core::constants::TRACK_ID
            )
            .unwrap(),
            (
                track_baseline().prefill_seconds_per_token,
                track_baseline().decode_seconds_per_token
            )
        );
    }

    /// Drive `official_gates_only` (correctness-only, no timed phase) with a correctness engine.
    fn gates_only_with<FC>(golden: &GoldenFixture, correctness: FC) -> ScorePayload
    where
        FC: Fn() -> MockEngine,
    {
        // The caller (execute_iterate) resolves the pair: the golden's declared pair first, else
        // the official constants (pending ⇒ refused before this is reached).
        let (prefill, decode) = golden
            .benchmark
            .as_ref()
            .and_then(|b| {
                b.baseline_prefill_seconds_per_token
                    .zip(b.baseline_decode_seconds_per_token)
            })
            .unwrap_or((
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ));
        official_gates_only(
            golden,
            ScoringInputs::local(prefill, decode),
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            || Session::connect(correctness()).map(|(s, _)| s),
        )
        .expect("the branch's own track declares a baseline, so the gates path resolves one")
    }

    #[test]
    fn official_gates_only_conformant_is_partial_result_true_null_score() {
        // SKIP_TIMED gates-only (seam 1): a conformant correctness run seals passed=true,
        // partial_result=true, passed_correctness=true, error empty, NULL score, and the timing
        // placeholders stay zero (the paired overlay owns them).
        let golden = official_golden(None);
        let payload = gates_only_with(&golden, conformant_engine);
        assert!(payload.passed, "error={}", payload.metrics.error);
        assert!(
            payload.metrics.partial_result,
            "gates score must be partial"
        );
        assert!(payload.metrics.passed_correctness);
        assert!(payload.metrics.error.is_empty());
        assert!(payload.score.is_none(), "gates-only score must be null");
        assert_eq!(payload.metrics.commit, "deadbeef");
        assert_eq!(payload.metrics.runtime, "rust");
        // No timed phase ran: the MEASURED timing surface stays at its zero placeholders.
        assert_eq!(payload.metrics.decode_seconds_per_token, 0.0);
        assert_eq!(payload.metrics.prefill_seconds_per_token, 0.0);
        assert_eq!(payload.metrics.decode_speedup, 0.0);
        assert_eq!(payload.metrics.prefill_speedup, 0.0);
        // #132/F-2 — the BASELINE pair is NOT part of that zero surface, and its absence from
        // this enumeration is how a whole class of change stayed green: the reference resolves it
        // (`pairedBaseline ?? golden.resolvedBaseline*`) BEFORE the gates-only branch, so it is
        // never a placeholder. This golden declares no pair, so the resolution lands on the
        // constants; `official_gates_only_seals_the_goldens_declared_baselines` covers the arm
        // where the golden DOES declare one, which is what tells the chain apart.
        assert_eq!(
            payload.metrics.baseline_prefill_seconds_per_token,
            TEST_BASELINE.prefill_seconds_per_token
        );
        assert_eq!(
            payload.metrics.baseline_decode_seconds_per_token,
            TEST_BASELINE.decode_seconds_per_token
        );
        assert_ne!(payload.metrics.baseline_prefill_seconds_per_token, 0.0);
        assert_ne!(payload.metrics.baseline_decode_seconds_per_token, 0.0);
        // Correctness DID run: full case count + real checked-step sum.
        assert_eq!(
            payload.metrics.case_count,
            golden.total_correctness_case_count() as i64
        );
        assert_eq!(
            payload.metrics.checked_steps,
            bench_core::constants::CORRECTNESS_STEPS as i64
        );
    }

    /// #132/F-2 — the official path resolves its baseline pair the REFERENCE's way, on all three
    /// payloads that no later step overwrites.
    ///
    /// The reference overwrites its baseline locals with
    /// `pairedBaseline ?? benchmarkGolden.resolvedBaseline*`
    /// (`QwenRuntimeBenchmark.swift@b26f76f:442-445`) BEFORE the `skipTimedBenchmark` branch at
    /// `:457` and before every `makeFailedScore` in `benchmarkWithWorker`. So the golden's declared
    /// pair — not the constants, and not zero — is what these records carry. A golden declaring a
    /// DISTINCT pair is the only fixture that can tell those three answers apart.
    #[test]
    fn official_gates_only_seals_the_goldens_declared_baselines() {
        const DECLARED_PREFILL: f64 = 0.000123456;
        const DECLARED_DECODE: f64 = 0.00987654;
        // Distinct from BOTH wrong answers, so the assertions below are not satisfiable by
        // accident.
        assert_ne!(DECLARED_PREFILL, TEST_BASELINE.prefill_seconds_per_token);
        assert_ne!(DECLARED_DECODE, TEST_BASELINE.decode_seconds_per_token);

        let golden = official_golden_with_baselines(DECLARED_PREFILL, DECLARED_DECODE);

        // 1. the PASSING gates-only partial.
        let passing = gates_only_with(&golden, conformant_engine);
        assert!(passing.passed, "error={}", passing.metrics.error);
        assert!(passing.metrics.partial_result);
        assert_eq!(
            passing.metrics.baseline_prefill_seconds_per_token, DECLARED_PREFILL,
            "gates-only partial must seal the GOLDEN's pair, not the constants"
        );
        assert_eq!(
            passing.metrics.baseline_decode_seconds_per_token,
            DECLARED_DECODE
        );

        // 2. the gates-only FAILURE record.
        let failing = gates_only_with(&golden, || {
            MockEngine::new().error_on("correctness_step", "boom")
        });
        assert!(!failing.passed);
        assert!(failing.metrics.partial_result);
        assert_eq!(
            failing.metrics.baseline_prefill_seconds_per_token, DECLARED_PREFILL,
            "gates-only failure must seal the GOLDEN's pair too — the reference's failure records \
             are all reached after the :442-445 overwrite"
        );
        assert_eq!(
            failing.metrics.baseline_decode_seconds_per_token,
            DECLARED_DECODE
        );
    }

    /// An official golden with NO benchmark oracle — the fixture that reaches `official_core`'s
    /// oracle-less branch (`official.rs:192`) directly.
    fn official_golden_without_oracle() -> GoldenFixture {
        let doc = json!({
            "version": 1,
            "model_type": "qwen4_exp_text",
            "cases": [
                { "name": "case-a", "prompt_tokens": vec![1i64; CORRECTNESS_PROMPT_TOKENS], "expected_tokens": vec![2i64; 64] }
            ]
        });
        let bytes = serde_json::to_vec(&doc).unwrap();
        load_golden_fixture(
            &bytes,
            64,
            CORRECTNESS_PROMPT_TOKENS,
            crate::testgolden::identity_125b(),
            Some("qwen4_exp_text"),
            None,
            None,
        )
        .unwrap()
    }

    /// #132/F-8 — `official_failed` seals the pair `official_core` RESOLVED (env ?? `--baseline-*`
    /// flags ?? golden), on BOTH branches that reach it.
    ///
    /// **This replaces a test that never reached the code it claimed to cover.** The previous
    /// version drove `finish_official` with an oracle-BEARING golden, so the oracle-less branch at
    /// `:192` was never taken and the failing spawner routed to `official_failed_timed_oracle`
    /// (`:760`) — which already threaded the pair before this PR. It pinned pre-existing behavior:
    /// reverting `official_failed` to re-derive from the golden left the whole suite green.
    ///
    /// Each arm below carries a DISTINGUISHING assertion on the error string that only that branch
    /// can produce, so neither can silently start passing through some other exit.
    #[test]
    fn official_failed_seals_the_resolved_pair_on_both_branches_that_reach_it() {
        const RESOLVED_PREFILL: f64 = 0.000222222;
        const RESOLVED_DECODE: f64 = 0.00333333;
        const DECLARED_PREFILL: f64 = 0.000123456;
        const DECLARED_DECODE: f64 = 0.00987654;
        // All three candidate answers must be mutually distinct or the assertions prove nothing.
        assert_ne!(RESOLVED_PREFILL, DECLARED_PREFILL);
        assert_ne!(RESOLVED_PREFILL, TEST_BASELINE.prefill_seconds_per_token);
        assert_ne!(RESOLVED_DECODE, DECLARED_DECODE);
        assert_ne!(RESOLVED_DECODE, TEST_BASELINE.decode_seconds_per_token);

        // --- branch 1: official.rs:192, the ORACLE-LESS golden -------------------------------
        // Nothing spawns on this path; the golden declares no pair, so a re-derive would land on
        // the CONSTANTS and the resolved pair is what tells them apart.
        let no_oracle = official_golden_without_oracle();
        let a = official_core_windowed(
            OfficialParams {
                scoring: ScoringInputs::local(RESOLVED_PREFILL, RESOLVED_DECODE),
                ..test_run(
                    &no_oracle,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                )
            },
            TEST_BASELINE.bands,
            || Session::connect(conformant_engine()).map(|(s, _)| s),
            // The historical entry point has no cool gate; the ranked one passes the real gate.
            |_phase: &str| Ok(()),
        );
        assert!(!a.passed);
        assert_eq!(
            a.metrics.error, "benchmark golden file must contain a benchmark oracle",
            "branch :192 signature — reached a different exit"
        );
        assert_eq!(
            a.metrics.baseline_prefill_seconds_per_token, RESOLVED_PREFILL,
            ":192 must seal the RESOLVED pair; the constants here would mean official_failed \
             re-derived it and dropped the --baseline-* flags"
        );
        assert_eq!(a.metrics.baseline_decode_seconds_per_token, RESOLVED_DECODE);
        assert_ne!(
            a.metrics.baseline_prefill_seconds_per_token,
            TEST_BASELINE.prefill_seconds_per_token
        );

        // --- branch 2: official.rs:266, a TIMED failure on an oracle-bearing golden -----------
        // The golden DECLARES a pair here, so all three answers are distinguishable at once.
        let with_oracle = official_golden_with_baselines(DECLARED_PREFILL, DECLARED_DECODE);
        let b = official_core_windowed(
            OfficialParams {
                scoring: ScoringInputs::local(RESOLVED_PREFILL, RESOLVED_DECODE),
                ..test_run(
                    &with_oracle,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                )
            },
            TEST_BASELINE.bands,
            || -> bench_runner::Result<Session<MockEngine>> {
                Err(RunnerError::Protocol(
                    "timed worker spawn failed".to_string(),
                ))
            },
            // The historical entry point has no cool gate; the ranked one passes the real gate.
            |_phase: &str| Ok(()),
        );
        assert!(!b.passed);
        assert!(
            b.metrics.error.contains("timed worker spawn failed"),
            "branch :266 signature — reached a different exit: {:?}",
            b.metrics.error
        );
        assert_eq!(
            b.metrics.baseline_prefill_seconds_per_token, RESOLVED_PREFILL,
            ":266 must seal the RESOLVED pair, not the golden's declaration"
        );
        assert_eq!(b.metrics.baseline_decode_seconds_per_token, RESOLVED_DECODE);
        assert_ne!(
            b.metrics.baseline_prefill_seconds_per_token,
            DECLARED_PREFILL
        );
        assert_ne!(
            b.metrics.baseline_prefill_seconds_per_token,
            TEST_BASELINE.prefill_seconds_per_token
        );
    }

    #[test]
    fn official_gates_only_correctness_fail_is_failed_but_partial() {
        // A correctness mismatch in gates-only mode fails closed: passed=false,
        // passed_correctness=false, null score — but still the gates-only shape (partial_result
        // stays true; a failing gates-score never reaches the overlay).
        let golden = official_golden(None);
        let bad_correctness = || {
            MockEngine::new()
                .teacher_forced_tokens(vec![999i64; 64])
                .free_run_capable()
                .oracle_tokens(PREFILL_TOKEN, SEED_TOKEN, oracle_decode_tokens())
        };
        let payload = gates_only_with(&golden, bad_correctness);
        assert!(!payload.passed);
        assert!(!payload.metrics.passed_correctness);
        assert!(payload.metrics.partial_result);
        assert!(payload.score.is_none());
        assert!(!payload.metrics.error.is_empty());
    }

    /// A correctness engine that answers the [2;64] base case with `mismatches` non-accepted
    /// tokens, the FIRST at step 3, and is otherwise conformant (timed oracle included).
    fn base_case_engine_with_mismatches(mismatches: usize) -> MockEngine {
        let mut tokens = vec![2i64; 64];
        // Step 3 first, then from the END of the window, so step 3 stays the first mismatch.
        tokens[3] = 999;
        for (n, slot) in (0..64).rev().enumerate() {
            if n + 1 >= mismatches {
                break;
            }
            if slot != 3 {
                tokens[slot] = 999;
            }
        }
        MockEngine::new()
            .teacher_forced_tokens(tokens)
            .free_run_capable()
            .oracle_tokens(PREFILL_TOKEN, SEED_TOKEN, oracle_decode_tokens())
    }

    #[test]
    fn official_base_case_passes_inside_the_blanket_ten_percent_budget() {
        // David's ruling (2026-09-02) — "we can differ by 10%" — applied to the OFFICIAL
        // base-case teacher-forced gate: 6 non-accepted tokens over the 64-step window is
        // 6 * 1000 = 6000 <= COHORT_TOKEN_TOLERANCE_PER_THOUSAND (100) * 64 = 6400, so the
        // run PASSES correctness instead of refusing at the first mismatch.
        let golden = official_golden(None);
        let payload = finish_with(&golden, || base_case_engine_with_mismatches(6));
        assert!(payload.passed, "error={}", payload.metrics.error);
        assert!(payload.metrics.passed_correctness);
        assert!(payload.metrics.first_failing_case.is_none());
        assert!(payload.metrics.first_failing_step.is_none());
    }

    #[test]
    fn official_base_case_refuses_one_mismatch_over_the_budget() {
        // 7 * 1000 = 7000 > 6400 -> REFUSE, branded exactly as before, with the FIRST
        // failing step (3) preserved on the sealed payload.
        let golden = official_golden(None);
        let payload = finish_with(&golden, || base_case_engine_with_mismatches(7));
        assert!(!payload.passed);
        assert!(!payload.metrics.passed_correctness);
        assert!(payload.score.is_none());
        assert_eq!(payload.metrics.error, "teacher-forced token mismatch");
        assert_eq!(
            payload.metrics.first_failing_case.as_deref(),
            Some("case-a")
        );
        assert_eq!(payload.metrics.first_failing_step, Some(3));
        assert_eq!(
            payload.metrics.checked_steps, 4,
            "Swift FAIL shape: first mismatch step + 1"
        );
    }

    #[test]
    fn official_conformant_run_passes_with_score_and_commit() {
        // In-band timing (speedups 1.0) + conformant correctness ⇒ PASS, score ~1.0, commit
        // stamped, runtime = "rust".
        let golden = official_golden(None);
        let payload = finish_with(&golden, conformant_engine);
        assert!(payload.passed, "error={}", payload.metrics.error);
        assert!(payload.metrics.passed_correctness);
        let score = payload.score.expect("passing official run has a score");
        assert!((score - 1.0).abs() < 1e-9, "score={score}");
        assert_eq!(payload.metrics.commit, "deadbeef");
        assert_eq!(payload.metrics.runtime, "rust");
        // case_count = full correctness total (checkGates == true), not the timing repeats.
        assert_eq!(
            payload.metrics.case_count,
            golden.total_correctness_case_count() as i64
        );
        // checked_steps is now the REAL per-case sum (one base case, no gates → the full
        // teacher-forced window of 64), NOT the placeholder case count (1).
        assert_eq!(
            payload.metrics.checked_steps,
            bench_core::constants::CORRECTNESS_STEPS as i64
        );
        assert_ne!(
            payload.metrics.checked_steps,
            golden.total_correctness_case_count() as i64,
            "checked_steps must be the real step sum, not the placeholder case count"
        );
    }

    /// A golden with a base case + 2 anchors + 1 free-run case (4 declared correctness cases —
    /// the same count that used to be the checked_steps PLACEHOLDER). The conformant per-case
    /// checked-step SUM is 64 (base window) + 1 + 1 (anchors) + 5 (free-run prefix) = 71.
    fn full_gates_golden(free_run_expected: Vec<i64>) -> GoldenFixture {
        official_golden(Some(json!({
            "anchors": [
                { "name": "anc-1", "context_tokens": vec![1i64; 8], "expected_token": 7, "accepted_tokens": [7] },
                { "name": "anc-2", "context_tokens": vec![1i64; 8], "expected_token": 9, "accepted_tokens": [9] }
            ],
            "free_run": [
                { "name": "fr-1", "prompt_tokens": vec![1i64; CORRECTNESS_PROMPT_TOKENS], "expected_tokens": free_run_expected }
            ]
        })))
    }

    /// Conformant on base [2;64], anchors (argmax 7 then 9 via per-sequence teacher forcing),
    /// and the free-run stream (the mock's fixed `correctness` tokens 4000,4001,4002,…).
    fn full_gates_conformant_engine() -> MockEngine {
        MockEngine::new()
            .teacher_forced_sequences(vec![vec![2i64; 64], vec![7], vec![9]])
            .free_run_capable()
            .oracle_tokens(PREFILL_TOKEN, SEED_TOKEN, oracle_decode_tokens())
    }

    #[test]
    fn official_passing_checked_steps_is_real_per_case_sum() {
        // Leg-1 PASSING parity: a fully-conformant official run reports the REAL checked-step
        // SUM (64 + 1 + 1 + 5 = 71), NOT the placeholder total_correctness_case_count (4).
        let golden = full_gates_golden(vec![4000, 4001, 4002, 4003, 4004]);
        assert_eq!(
            golden.total_correctness_case_count(),
            4,
            "the old placeholder value"
        );
        let payload = finish_with(&golden, full_gates_conformant_engine);
        assert!(payload.passed, "error={}", payload.metrics.error);
        assert!(payload.metrics.passed_correctness);
        assert_eq!(
            payload.metrics.checked_steps, 71,
            "64 base + 1 + 1 anchors + 5 free-run"
        );
        assert_eq!(
            payload.metrics.case_count, 4,
            "caseCount stays the declared total"
        );
    }

    #[test]
    fn official_anchor_fail_partial_checked_steps_and_step_zero() {
        // Anchor-gate failure: base passes (64), anc-1 fails (+1) → partial sum 65. Swift also
        // reports first_failing_step = 0 (compareAnchorToken) and NULL expected/actual tokens.
        let golden = full_gates_golden(vec![4000, 4001, 4002, 4003, 4004]);
        // Corrupt anc-1: engine argmax 8 ∉ accepted {7}.
        let engine = || {
            MockEngine::new()
                .teacher_forced_sequences(vec![vec![2i64; 64], vec![8], vec![9]])
                .free_run_capable()
                .oracle_tokens(PREFILL_TOKEN, SEED_TOKEN, oracle_decode_tokens())
        };
        let payload = finish_with(&golden, engine);
        assert!(!payload.passed);
        assert_eq!(payload.metrics.error, "anchor token mismatch");
        assert_eq!(payload.metrics.first_failing_case.as_deref(), Some("anc-1"));
        assert_eq!(
            payload.metrics.checked_steps, 65,
            "64 base + 1 failing anchor"
        );
        assert_eq!(
            payload.metrics.first_failing_step,
            Some(0),
            "Swift anchor firstFailingStep=0"
        );
        assert_eq!(
            payload.metrics.expected_token, None,
            "official nulls expected_token"
        );
        assert_eq!(
            payload.metrics.actual_token, None,
            "official nulls actual_token"
        );
        // case_count + golden_hash stay populated (correctness DID run and fail).
        assert_eq!(payload.metrics.case_count, 4);
        assert_eq!(payload.metrics.golden_hash, golden.sha256);
    }

    #[test]
    fn official_free_run_fail_partial_checked_steps() {
        // Free-run-gate failure: base(64) + anc-1(1) + anc-2(1) + free-run fails at step 2 (+3)
        // → partial sum 69. Free-run stream is 4000,4001,4002,…; expected diverges at index 2.
        let golden = full_gates_golden(vec![4000, 4001, 999, 4003, 4004]);
        let payload = finish_with(&golden, full_gates_conformant_engine);
        assert!(!payload.passed);
        assert_eq!(payload.metrics.error, "free-run token mismatch");
        assert_eq!(payload.metrics.first_failing_case.as_deref(), Some("fr-1"));
        assert_eq!(
            payload.metrics.checked_steps, 69,
            "64 + 1 + 1 + 3 (free-run step 2 + 1)"
        );
        assert_eq!(
            payload.metrics.first_failing_step,
            Some(2),
            "free-run mismatch step"
        );
        assert_eq!(payload.metrics.expected_token, None);
        assert_eq!(payload.metrics.actual_token, None);
    }

    #[test]
    fn official_primary_correctness_fail_nulls_tokens() {
        // Primary (teacher-forced) correctness failure: Swift leaves expected_token/actual_token
        // NULL on the OFFICIAL path (unlike LOCAL). checked_steps = the partial base sum (fail
        // at step 0 → 1).
        let golden = official_golden(None);
        let engine = || {
            MockEngine::new()
                .teacher_forced_tokens(vec![3i64; 64]) // diverges from golden [2;64] at step 0
                .free_run_capable()
                .oracle_tokens(PREFILL_TOKEN, SEED_TOKEN, oracle_decode_tokens())
        };
        let payload = finish_with(&golden, engine);
        assert!(!payload.passed);
        assert_eq!(payload.metrics.error, "teacher-forced token mismatch");
        assert_eq!(
            payload.metrics.first_failing_case.as_deref(),
            Some("case-a")
        );
        assert_eq!(payload.metrics.first_failing_step, Some(0));
        assert_eq!(
            payload.metrics.expected_token, None,
            "official nulls expected_token"
        );
        assert_eq!(
            payload.metrics.actual_token, None,
            "official nulls actual_token"
        );
        assert_eq!(
            payload.metrics.checked_steps, 1,
            "fail at base step 0 → checkedSteps 1"
        );
    }

    #[test]
    fn official_timed_first_orchestration_reaches_band_gate() {
        // The full official_core with a CONFORMANT-ORACLE mock: the timed phases pass the
        // oracle (no TokenMismatch), and gating then FAILS on the acceptance band because a
        // mock's ~0 wall-clock is ~1000x faster than the baseline. This proves the timed-first
        // orchestration ran and the band gate is wired ahead of correctness.
        let golden = official_golden(None);
        let payload = run_official(&golden, conformant_engine);
        assert!(!payload.passed);
        assert!(
            payload.metrics.error.starts_with("acceptance band failed:")
                || payload
                    .metrics
                    .error
                    .starts_with("performance floor failed:"),
            "expected a band/floor failure (not an oracle mismatch), got: {}",
            payload.metrics.error
        );
    }

    #[test]
    fn official_timed_prefill_warms_once_in_its_own_session() {
        // David 2026-09-07: the timed session's first prefill is not a steady reading, so the
        // official timed leg runs ONE unmeasured prefill in that session before the timed one.
        let golden = official_golden(None);
        // The warm-up count is now a value the WINDOW carries, and the window comes from the
        // track fixture (STEP 3, David 2026-09-15) — so the two cases below differ by the window
        // handed in, not by a platform the function looks up.
        let mlx_window = WindowShape {
            official_prefill_warmup_runs: 1,
            ..crate::iterate::test_window()
        };
        let params = official_timed_params(golden.benchmark.as_ref().unwrap(), None, &mlx_window);
        assert_eq!(params.prefill_warmup_runs, 1);
        assert_eq!(params.prefill_timed_runs, 1);
        // CUDA: the resident engine is warm and the ds4 adapter refuses a second `prefill` opener
        // without a `phase_diagnostics` barrier, so its fixture declares no warm-up pass.
        let cuda_window = WindowShape {
            official_prefill_warmup_runs: 0,
            ..crate::iterate::test_window()
        };
        let cuda = official_timed_params(golden.benchmark.as_ref().unwrap(), None, &cuda_window);
        assert_eq!(cuda.prefill_warmup_runs, 0);
        assert_eq!(cuda.prefill_timed_runs, 1);
        // Nemotron: one warm-up pass, then the MEDIAN of five timed passes.
        let nemotron_window = WindowShape {
            official_prefill_warmup_runs: 1,
            official_prefill_timed_runs: 5,
            ..crate::iterate::test_window()
        };
        let nemotron =
            official_timed_params(golden.benchmark.as_ref().unwrap(), None, &nemotron_window);
        assert_eq!(nemotron.prefill_warmup_runs, 1);
        assert_eq!(nemotron.prefill_timed_runs, 5);
        // The local modes are untouched: their default is still the reference's zero.
        let local = TimingParams::new(vec![1], 1, vec![1], 1, vec![1, 2], 1);
        assert_eq!(
            local.prefill_warmup_runs,
            bench_core::constants::BENCHMARK_PREFILL_WARMUP_RUNS
        );
    }

    #[test]
    fn official_timed_phases_pass_through_the_cool_gate_in_order() {
        // David 2026-09-06: the 40 C per-phase contract applies to the single-leg path. The
        // TIMED prefill and decode each call the gate, in that order; the unmeasured warmup leg
        // does not (it is what heats the GPU). A gate refusal fails the run closed.
        use std::cell::RefCell;
        let golden = official_golden(None);
        let phases: RefCell<Vec<String>> = RefCell::new(Vec::new());
        let _ = official_core_windowed(
            test_run(
                &golden,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            TEST_BASELINE.bands,
            || Session::connect(conformant_engine()).map(|(s, _)| s),
            |phase: &str| {
                phases.borrow_mut().push(phase.to_string());
                Ok(())
            },
        );
        assert_eq!(phases.borrow().as_slice(), ["prefill", "decode"]);
        let refused = official_core_windowed(
            test_run(
                &golden,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            TEST_BASELINE.bands,
            || Session::connect(conformant_engine()).map(|(s, _)| s),
            |phase: &str| {
                Err(bench_runner::RunnerError::GateRejected {
                    phase: phase.to_string(),
                    reason: "GPU stayed above 40 C".to_string(),
                })
            },
        );
        assert!(!refused.passed);
        assert!(
            refused.metrics.error.contains("gate rejected (prefill)"),
            "{}",
            refused.metrics.error
        );
    }

    #[test]
    fn official_persistent_window_opens_one_worker_for_the_whole_timed_window() {
        // Load-once (David 2026-08-30): the window opens the model-holding worker EXACTLY ONCE for
        // the whole timed window (prefill + decode over one held session). The conformant mock's
        // ~0 wall-clock still fails the acceptance band ahead of correctness, proving the timed
        // phase ran and produced a real (out-of-band) timing.
        let golden = official_golden(None);
        let timed_spawns = Cell::new(0usize);
        let corr_spawns = Cell::new(0usize);
        let payload = official_core_windowed(
            test_run(
                &golden,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            TEST_BASELINE.bands,
            || {
                timed_spawns.set(timed_spawns.get() + 1);
                Session::connect(conformant_engine()).map(|(s, _)| s)
            },
            |_phase: &str| Ok(()),
        );
        assert!(!payload.passed);
        assert!(
            payload.metrics.error.starts_with("acceptance band failed:")
                || payload
                    .metrics
                    .error
                    .starts_with("performance floor failed:"),
            "expected a band/floor failure, got: {}",
            payload.metrics.error
        );
        assert_eq!(
            timed_spawns.get(),
            1,
            "the persistent window must open exactly ONE model-holding worker for the timed window"
        );
        // The band failed before correctness, so no correctness session was needed; and the
        // persistent path never uses the FRESH correctness spawner regardless — it reuses the
        // resident session.
        assert_eq!(corr_spawns.get(), 0);
    }

    #[test]
    fn official_warmup_leg_failure_fails_closed_before_timed_legs() {
        // The measurement-integrity warmup is WIRED and FAIL-CLOSED: a fault in the unmeasured
        // warmup leg fails the run with a warmup-labelled error and stops BEFORE any timed worker
        // spawns. The warmup worker (first spawn) faults its `free_decode_run`; because warmup runs
        // TimeOnly the fault must be a PROTOCOL error (an ok:false), not a token mismatch, to abort.
        let golden = official_golden(None);
        let spawn_n = Cell::new(0usize);
        let payload = official_core_windowed(
            test_run(
                &golden,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            TEST_BASELINE.bands,
            || {
                let n = spawn_n.get();
                spawn_n.set(n + 1);
                let engine = if n == 0 {
                    conformant_engine().error_on("free_decode_run", "warmup boom")
                } else {
                    conformant_engine()
                };
                Session::connect(engine).map(|(s, _)| s)
            },
            |_phase: &str| Ok(()),
        );
        assert!(!payload.passed);
        assert!(
            payload.metrics.error.contains("official warmup leg failed"),
            "warmup fault must surface a warmup-labelled failure, got: {}",
            payload.metrics.error
        );
        assert_eq!(
            spawn_n.get(),
            1,
            "a warmup fault must fail closed before any timed worker spawns"
        );
    }

    #[test]
    fn official_persistent_window_warmup_stays_on_the_one_resident_worker() {
        // The measurement-integrity warmup runs on the SAME held
        // session as the measured legs, so the model still loads ONCE: the warmup adds NO extra
        // spawn. The conformant mock's ~0 wall-clock still fails the acceptance band ahead of
        // correctness, proving the warmup did not corrupt the resident worker for the timed phases.
        let golden = official_golden(None);
        let timed_spawns = Cell::new(0usize);
        let payload = official_core_windowed(
            test_run(
                &golden,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            TEST_BASELINE.bands,
            || {
                timed_spawns.set(timed_spawns.get() + 1);
                Session::connect(conformant_engine()).map(|(s, _)| s)
            },
            |_phase: &str| Ok(()),
        );
        assert!(!payload.passed);
        assert!(
            payload.metrics.error.starts_with("acceptance band failed:")
                || payload
                    .metrics
                    .error
                    .starts_with("performance floor failed:"),
            "expected a band/floor failure (warmup ran, timed legs ran), got: {}",
            payload.metrics.error
        );
        assert_eq!(
            timed_spawns.get(),
            1,
            "the warmup shares the ONE resident worker — no extra spawn on the persistent path"
        );
    }

    /// The persistent official window closes FOUR phases over its ONE session, in this order:
    /// the warmup leg's prefill and decode, then the timed prefill and the timed decode. Each
    /// close is a cache boundary the parent asserts fail-closed, so a test names the boundary it
    /// proves by this ordinal.
    const WARMUP_PREFILL_CLOSE: usize = 1;
    const WARMUP_DECODE_CLOSE: usize = 2;
    const TIMED_PREFILL_CLOSE: usize = 3;
    const TIMED_DECODE_CLOSE: usize = 4;

    /// Run the persistent official window against an engine that reports `bytes` of undrained
    /// allocator cache on the `nth` phase close and a conformant `0` on every other one.
    fn official_with_undrained_cache_at(nth: usize, bytes: i64) -> ScorePayload {
        let golden = official_golden(None);
        official_core_windowed(
            test_run(
                &golden,
                RunDigests::for_test(&DirDigest::empty()),
                &test_window(),
            ),
            TEST_BASELINE.bands,
            || {
                Session::connect(conformant_engine().cache_memory_on_nth_phase_close(nth, bytes))
                    .map(|(s, _)| s)
            },
            |_phase: &str| Ok(()),
        )
    }

    #[test]
    fn official_persistent_window_refuses_an_undrained_cache_at_the_warmup_to_prefill_boundary() {
        // David 2026-09-15: the weights load once per window, so the caches must be PROVEN reset
        // between the legs that share the one worker. The warmup leg's LAST phase close is the
        // warmup-to-prefill boundary. An engine that carries 4 KiB of allocator cache across it
        // fails the run CLOSED, before the timed legs measure anything.
        let payload = official_with_undrained_cache_at(WARMUP_DECODE_CLOSE, 4096);
        assert!(!payload.passed);
        assert!(
            payload.metrics.error.contains("official warmup leg failed"),
            "the undrained boundary is inside the warmup leg, got: {}",
            payload.metrics.error
        );
        assert!(
            payload
                .metrics
                .error
                .contains("failed to clear the MLX allocator cache"),
            "the refusal must name the allocator drain, got: {}",
            payload.metrics.error
        );
        // The FIRST close of the same leg is asserted too, so the boundary test above is not the
        // only thing standing between a warmup phase and an undrained cache.
        let first = official_with_undrained_cache_at(WARMUP_PREFILL_CLOSE, 4096);
        assert!(!first.passed);
        assert!(
            first
                .metrics
                .error
                .contains("failed to clear the MLX allocator cache"),
            "{}",
            first.metrics.error
        );
    }

    #[test]
    fn official_persistent_window_refuses_an_undrained_cache_at_the_prefill_to_decode_boundary() {
        // The timed prefill's close is the prefill-to-decode boundary. On the persistent window
        // the decode phase runs on the SAME worker, so this close is the only thing that proves
        // the KV and allocator state of the prefill phase does not subsidize the decode phase.
        let payload = official_with_undrained_cache_at(TIMED_PREFILL_CLOSE, 4096);
        assert!(!payload.passed);
        assert!(
            payload
                .metrics
                .error
                .contains("failed to clear the MLX allocator cache"),
            "the refusal must name the allocator drain, got: {}",
            payload.metrics.error
        );
        assert!(
            !payload.metrics.error.contains("official warmup leg failed"),
            "this boundary is in the TIMED leg, not the warmup leg: {}",
            payload.metrics.error
        );
        // The window's LAST close is asserted on the same rule, so the measured decode phase
        // cannot end with an undrained allocator either.
        let last = official_with_undrained_cache_at(TIMED_DECODE_CLOSE, 4096);
        assert!(!last.passed);
        assert!(
            last.metrics
                .error
                .contains("failed to clear the MLX allocator cache"),
            "{}",
            last.metrics.error
        );
    }

    #[test]
    fn official_full_scope_evaluates_anchor_gate_unlike_local() {
        // GATE-SCOPE FLIP: official runs the FULL set. A golden with a CORRUPTED anchor
        // (engine argmax ≠ 999) that the LOCAL default would SKIP must FAIL official
        // correctness (reached here via in-band timing so the band does not mask it).
        let golden = official_golden(Some(json!({
            "anchors": [
                { "name": "bad-anchor", "context_tokens": vec![1i64; 8], "expected_token": 999, "accepted_tokens": [999] }
            ]
        })));
        let payload = finish_with(&golden, conformant_engine);
        assert!(
            !payload.passed,
            "official evaluates the full correctness set"
        );
        assert!(payload.score.is_none());
        assert!(!payload.metrics.passed_correctness);
        assert_eq!(payload.metrics.error, "anchor token mismatch");
        assert_eq!(
            payload.metrics.first_failing_case.as_deref(),
            Some("bad-anchor")
        );
        // Swift anchor firstFailingStep = 0 (compareAnchorToken), tokens null on the official path.
        assert_eq!(payload.metrics.first_failing_step, Some(0));
        assert_eq!(payload.metrics.expected_token, None);
        assert_eq!(payload.metrics.actual_token, None);
        // Partial checked-step sum: base window 64 + failing anchor 1 = 65.
        assert_eq!(payload.metrics.checked_steps, 65);
    }

    #[test]
    fn corrupted_benchmark_oracle_fails_official() {
        // THE failure class the local path cannot test: the golden's benchmark oracle says
        // decode step 3 → token 703, but the engine emits 999_999. Official's parent-side
        // oracle check FAILS the run (BenchmarkTokenMismatchError), branded case="benchmark".
        let golden = official_golden(None);
        let mut engine_tokens = oracle_decode_tokens();
        engine_tokens[3] = 999_999;
        let payload = run_official(&golden, move || {
            MockEngine::new()
                .teacher_forced_tokens(vec![2i64; 64])
                .free_run_capable()
                .oracle_tokens(PREFILL_TOKEN, SEED_TOKEN, engine_tokens.clone())
        });
        assert!(!payload.passed);
        assert!(payload.score.is_none());
        // A8 free-run timed leg: the decode-TOKEN mismatch class KEEPS the step suffix and its
        // label now names the free-run verb ("benchmark free-run decode token"). makeFailedScore
        // still nulls the tokens. The string diverges from Swift's teacher-forced description by
        // design — teacher-forced-per-step is retained only for the untimed correctness gate.
        assert_eq!(
            payload.metrics.error, "benchmark free-run decode token mismatch at step 3",
            "free-run decode-token class keeps the step suffix"
        );
        assert_eq!(
            payload.metrics.first_failing_case.as_deref(),
            Some("benchmark")
        );
        assert_eq!(payload.metrics.first_failing_step, Some(3));
        // MAJOR-1: Swift makeFailedScore sets expectedToken/actualToken = nil (ALWAYS).
        assert_eq!(payload.metrics.expected_token, None);
        assert_eq!(payload.metrics.actual_token, None);
        // ITEM B (this change): the oracle mismatch is a TIMED-phase failure BEFORE correctness,
        // so Swift returns via makeFailedScore(correctness: nil) — BLANK the correctness audit
        // fields but RETAIN the resolved baselines.
        assert_eq!(
            payload.metrics.golden_hash, "",
            "golden_hash blanked (correctness nil)"
        );
        assert_eq!(
            payload.metrics.case_count, 0,
            "case_count blanked (correctness nil)"
        );
        assert_eq!(
            payload.metrics.checked_steps, 0,
            "checked_steps blanked (correctness nil)"
        );
        assert_eq!(
            payload.metrics.baseline_decode_seconds_per_token,
            TEST_BASELINE.decode_seconds_per_token,
            "resolved decode baseline RETAINED"
        );
        assert_eq!(
            payload.metrics.baseline_prefill_seconds_per_token,
            TEST_BASELINE.prefill_seconds_per_token,
            "resolved prefill baseline RETAINED"
        );
        // The MEASURED spt stay 0 (the timed phase never completed a trustworthy measurement).
        assert_eq!(payload.metrics.decode_seconds_per_token, 0.0);
        assert_eq!(payload.metrics.prefill_seconds_per_token, 0.0);
    }

    #[test]
    fn corrupted_prefill_oracle_fails_official_no_step_suffix() {
        // The PREFILL oracle class: Swift compareOne carries step:nil, so the description has
        // NO " at step N" suffix and firstFailingStep = nil; tokens are nulled.
        let golden = official_golden(None);
        let payload = run_official(&golden, || {
            MockEngine::new()
                .teacher_forced_tokens(vec![2i64; 64])
                // Wrong PREFILL token (seed + decode conformant).
                .free_run_capable()
                .oracle_tokens(PREFILL_TOKEN + 1, SEED_TOKEN, oracle_decode_tokens())
        });
        assert!(!payload.passed);
        assert_eq!(
            payload.metrics.error, "benchmark prefill token mismatch",
            "prefill class has NO step suffix (Swift compareOne step:nil)"
        );
        assert_eq!(
            payload.metrics.first_failing_case.as_deref(),
            Some("benchmark")
        );
        assert_eq!(payload.metrics.first_failing_step, None);
        assert_eq!(payload.metrics.expected_token, None);
        assert_eq!(payload.metrics.actual_token, None);
    }

    #[test]
    fn corrupted_seed_oracle_fails_official_no_step_suffix() {
        // The decode-SEED oracle class: step-less, no suffix, nulled tokens. Under the a8 free-run
        // timed leg the seed forward is `free_decode_begin`, so the label names the free-run verb.
        let golden = official_golden(None);
        let payload = run_official(&golden, || {
            MockEngine::new()
                .teacher_forced_tokens(vec![2i64; 64])
                // Wrong SEED token (prefill + decode conformant).
                .free_run_capable()
                .oracle_tokens(PREFILL_TOKEN, SEED_TOKEN + 1, oracle_decode_tokens())
        });
        assert!(!payload.passed);
        assert_eq!(
            payload.metrics.error, "benchmark free-run decode seed token mismatch",
            "free-run seed class has NO step suffix"
        );
        assert_eq!(payload.metrics.first_failing_step, None);
        assert_eq!(payload.metrics.expected_token, None);
        assert_eq!(payload.metrics.actual_token, None);
    }

    #[test]
    fn benchmark_less_golden_fails_official() {
        let doc = json!({
            "version": 1, "model_type": "qwen4_exp_text",
            "cases": [{ "name": "p1", "prompt_tokens": vec![1i64; CORRECTNESS_PROMPT_TOKENS], "expected_tokens": vec![2i64; 64] }],
        });
        let bytes = serde_json::to_vec(&doc).unwrap();
        let golden = load_golden_fixture(
            &bytes,
            64,
            CORRECTNESS_PROMPT_TOKENS,
            crate::testgolden::identity_125b(),
            Some("qwen4_exp_text"),
            None,
            None,
        )
        .unwrap();
        let payload = run_official(&golden, conformant_engine);
        assert!(!payload.passed);
        assert_eq!(
            payload.metrics.error,
            "benchmark golden file must contain a benchmark oracle"
        );
    }

    #[test]
    fn decode_speedup_below_floor_fails_official() {
        // Candidate decode +10% slower than baseline ⇒ speedup ≈ 0.909 < 0.95 floor ⇒ the
        // decode acceptance band (+2% ceiling) trips FIRST in the priority order (finite →
        // floors → bands), failing official BEFORE correctness with the real timing retained.
        let golden = official_golden(None);
        let mut timing = in_band_timing();
        timing.decode_seconds_per_token = TEST_BASELINE.decode_seconds_per_token * 1.10;
        let payload = finish_official(
            &golden,
            ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
            TEST_BASELINE.bands,
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            &timing,
            || Session::connect(conformant_engine()).map(|(s, _)| s),
        );
        assert!(!payload.passed);
        assert!(payload.score.is_none());
        assert!(
            payload
                .metrics
                .error
                .starts_with("performance floor failed:")
                || payload.metrics.error.starts_with("acceptance band failed:"),
            "floor/band failure expected, got: {}",
            payload.metrics.error
        );
        // Real timing retained (not blanked): the measured decode spt is carried through.
        assert!(payload.metrics.decode_seconds_per_token > 0.0);
        assert!(!payload.metrics.passed_correctness, "correctness never ran");
    }

    /// FLOOR-ONLY GATING BANDS: tolerances wide enough that no acceptance band can fail the run,
    /// so a boundary run reads the SPEEDUP FLOOR and nothing else.
    const FLOOR_ONLY_BANDS: AcceptanceBands = AcceptanceBands {
        prefill_up_tolerance: 10.0,
        prefill_down_tolerance: 0.99,
        decode_up_tolerance: 10.0,
        decode_down_tolerance: 0.99,
        decode_down_enabled: false,
        prefill_down_enabled: false,
    };

    /// One boundary run through the official gate: the candidate's per-token times are POWERS OF
    /// TWO and the denominator is that time scaled by the wanted speedup, so `baseline / candidate`
    /// is EXACTLY the literal asked for (scaling by a power of two is exact in binary floating
    /// point). Only the floors can fail the run.
    fn floor_boundary_run(
        golden: &GoldenFixture,
        decode_speedup: f64,
        prefill_speedup: f64,
        floors: SpeedupFloors,
    ) -> ScorePayload {
        const CANDIDATE_DECODE_SPT: f64 = 0.125; // 2^-3
        const CANDIDATE_PREFILL_SPT: f64 = 0.0078125; // 2^-7
        let timing = TimingResult {
            prefill_seconds_per_token: CANDIDATE_PREFILL_SPT,
            decode_seconds_per_token: CANDIDATE_DECODE_SPT,
            decode_steps: BENCHMARK_DECODE_STEPS,
            prefill_prompt_tokens: BENCHMARK_PREFILL_PROMPT_TOKENS,
            prefill_elapsed_seconds: CANDIDATE_PREFILL_SPT * 512.0,
            decode_elapsed_seconds: CANDIDATE_DECODE_SPT * BENCHMARK_DECODE_STEPS as f64,
            peak_ram_gb: 20.25,
            effective_spec: None,
            free_run_audit: Some(audit_for_test(vec![4, 4, 4, 5], 0, 0)),
            deferred_divergence: None,
            phase_window: None,
            emitted_tokens: Vec::new(),
        };
        finish_official(
            golden,
            ScoringInputs {
                baseline_prefill_spt: CANDIDATE_PREFILL_SPT * prefill_speedup,
                baseline_decode_spt: CANDIDATE_DECODE_SPT * decode_speedup,
                floors,
                weights: ScoringWeights::DEFAULT,
            },
            FLOOR_ONLY_BANDS,
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            &timing,
            || Session::connect(conformant_engine()).map(|(s, _)| s),
        )
    }

    /// DAVID 2026-09-09 — THE FLOORS ARE THE TRACK'S, THEY ARE ENFORCED ON BOTH AXES, AND WHAT IS
    /// SEALED IS WHAT WAS ENFORCED.
    ///
    /// Decode and prefill are gated separately at the boundary (0.949 refused, exactly 0.95
    /// accepted), the refusal names the floor the run carried, and `metrics.*_speedup_floor` is
    /// that same value on the passing and the failing payload alike.
    ///
    /// REVERT-PROOF: put `SCORE_*_SPEEDUP_FLOOR` back into the gate or the seal and the 0.90 arm
    /// goes red; drop either axis and its own arm goes red.
    #[test]
    fn official_enforces_and_seals_the_contract_speedup_floors() {
        let golden = official_golden(None);

        // DECODE, one thousandth below the ruled floor: refused, and the message names 0.95.
        let below = floor_boundary_run(&golden, 0.949, 1.0, SpeedupFloors::DEFAULT);
        assert!(!below.passed);
        assert!(below.score.is_none());
        assert!(
            below
                .metrics
                .error
                .contains("decode_speedup=0.949000 floor=0.950000"),
            "{}",
            below.metrics.error
        );
        assert_eq!(below.metrics.decode_speedup_floor, 0.95);
        assert_eq!(below.metrics.prefill_speedup_floor, 0.95);
        assert!(!below.metrics.passed_decode_speedup_floor);
        assert!(below.metrics.passed_prefill_speedup_floor);

        // DECODE, exactly ON the floor: accepted (the gate is `>=`, not `>`).
        let at = floor_boundary_run(&golden, 0.95, 1.0, SpeedupFloors::DEFAULT);
        assert!(at.passed, "at the floor must pass: {}", at.metrics.error);
        assert!(at.score.is_some());
        assert!(at.metrics.passed_decode_speedup_floor);
        assert_eq!(at.metrics.decode_speedup_floor, 0.95);

        // PREFILL is a floor of its own: the same 0.949, on the other axis, refuses the run even
        // with decode exactly at parity.
        let prefill_below = floor_boundary_run(&golden, 1.0, 0.949, SpeedupFloors::DEFAULT);
        assert!(!prefill_below.passed);
        assert!(
            prefill_below
                .metrics
                .error
                .contains("prefill_speedup=0.949000 floor=0.950000"),
            "{}",
            prefill_below.metrics.error
        );
        assert!(!prefill_below.metrics.passed_prefill_speedup_floor);
        assert!(prefill_below.metrics.passed_decode_speedup_floor);
        assert!(floor_boundary_run(&golden, 1.0, 0.95, SpeedupFloors::DEFAULT).passed);

        // PER PROJECT: a track whose fixture declares 0.90 is enforced at 0.90 and SEALS 0.90 —
        // the constants are not consulted anywhere on this path.
        let looser = SpeedupFloors {
            decode: 0.90,
            prefill: 0.90,
        };
        let ninety = floor_boundary_run(&golden, 0.949, 0.949, looser);
        assert!(
            ninety.passed,
            "0.949 clears a 0.90 floor: {}",
            ninety.metrics.error
        );
        assert_eq!(ninety.metrics.decode_speedup_floor, 0.90);
        assert_eq!(ninety.metrics.prefill_speedup_floor, 0.90);
        assert!(ninety.metrics.passed_decode_speedup_floor);
        assert!(ninety.metrics.passed_prefill_speedup_floor);
        // And that same fixture still refuses below ITS floor, naming ITS floor.
        let under_ninety = floor_boundary_run(&golden, 0.899, 1.0, looser);
        assert!(!under_ninety.passed);
        assert!(
            under_ninety
                .metrics
                .error
                .contains("decode_speedup=0.899000 floor=0.900000"),
            "{}",
            under_ninety.metrics.error
        );
        assert_eq!(under_ninety.metrics.decode_speedup_floor, 0.90);
    }

    #[test]
    fn official_timed_band_failure_blanks_correctness_audit_fields_to_match_swift() {
        // RULING 2: an OFFICIAL run that fails at the TIMED band (before correctness runs) must
        // byte-match Swift's `correctness == nil` failed score — golden_hash="", case_count=0,
        // checked_steps=0 — because Swift's timed-first path returns via
        // makeFailedScore(correctness: nil) and downstream/organizer tooling expects that shape.
        // Drive a floor/band failure via +10% slower decode (correctness never runs).
        let golden = official_golden(None);
        // Guard: this golden really does carry a non-empty hash and non-zero case count that a
        // populate-path (or the old official_failed_with_timing) WOULD have surfaced — so the
        // blanking below is meaningful, not vacuously matching an already-empty golden.
        assert!(
            !golden.sha256.is_empty(),
            "test golden must have a real hash to blank"
        );
        assert!(
            golden.total_correctness_case_count() > 0,
            "test golden must have >0 correctness cases to blank"
        );

        let mut timing = in_band_timing();
        timing.decode_seconds_per_token = TEST_BASELINE.decode_seconds_per_token * 1.10;
        let payload = finish_official(
            &golden,
            ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
            TEST_BASELINE.bands,
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            &timing,
            || Session::connect(conformant_engine()).map(|(s, _)| s),
        );
        assert!(!payload.passed);
        assert!(payload.score.is_none());
        assert!(
            payload
                .metrics
                .error
                .starts_with("performance floor failed:")
                || payload.metrics.error.starts_with("acceptance band failed:"),
            "timed-band failure expected, got: {}",
            payload.metrics.error
        );
        // The RULING-2 alignment: byte-match Swift's blanked correctness-audit surface.
        assert_eq!(
            payload.metrics.golden_hash, "",
            "golden_hash blanked to match Swift"
        );
        assert_eq!(
            payload.metrics.case_count, 0,
            "case_count blanked to match Swift"
        );
        assert_eq!(
            payload.metrics.checked_steps, 0,
            "checked_steps blanked to match Swift"
        );
        // The REAL timing surface is still retained (only the correctness-audit fields blank).
        assert!(
            payload.metrics.decode_seconds_per_token > 0.0,
            "measured timing retained"
        );
        assert!(
            payload.metrics.prefill_seconds_per_token > 0.0,
            "measured timing retained"
        );
        assert!(
            !payload.metrics.passed_correctness,
            "correctness never ran at the timed band"
        );
    }

    #[test]
    fn official_correctness_failure_still_populates_audit_fields_unchanged() {
        // SCOPE GUARD for RULING 2: the correctness-FAILURE path (correctness DID run and
        // failed) is NOT the timed-band path and must remain UNCHANGED — golden_hash and the
        // case counts stay populated. Reuses the corrupted-anchor golden reached via in-band
        // timing so the band does not mask the correctness failure.
        let golden = official_golden(Some(json!({
            "anchors": [
                { "name": "bad-anchor", "context_tokens": vec![1i64; 8], "expected_token": 999, "accepted_tokens": [999] }
            ]
        })));
        let payload = finish_with(&golden, conformant_engine);
        assert!(!payload.passed);
        assert_eq!(payload.metrics.error, "anchor token mismatch");
        // Correctness-failure path is untouched: golden_hash + case counts stay populated.
        assert_eq!(
            payload.metrics.golden_hash, golden.sha256,
            "correctness-fail path unchanged"
        );
        assert_eq!(
            payload.metrics.case_count,
            golden.total_correctness_case_count() as i64,
            "correctness-fail path unchanged"
        );
        assert!(
            payload.metrics.checked_steps > 0,
            "correctness-fail path unchanged"
        );
    }

    #[test]
    fn behavior_gate_presence_requires_worker() {
        // benchmarkRequiresRuntimeWorker: a behavior case forces the worker path.
        let with_behavior = official_golden(Some(json!({
            "behavior": [
                { "name": "b1", "prompt_tokens": vec![1i64; 8], "accepted_token_sequences": [[1, 2]], "max_new_tokens": 4 }
            ]
        })));
        assert!(benchmark_requires_runtime_worker(&with_behavior));
        // No gates / no behavior ⇒ not required by this predicate.
        assert!(!benchmark_requires_runtime_worker(&official_golden(None)));
    }

    #[test]
    fn paired_baseline_env_fail_closed_semantics() {
        // Both unset ⇒ None (no override).
        assert_eq!(paired_baseline_from_env(None, None).unwrap(), None);
        assert_eq!(
            paired_baseline_from_env(Some("  "), Some("")).unwrap(),
            None
        );
        // Both set + finite positive ⇒ Some (trimmed).
        assert_eq!(
            paired_baseline_from_env(Some(" 0.01 "), Some("0.13")).unwrap(),
            Some((0.01, 0.13))
        );
        // Half-set (either side) ⇒ error "must be provided together".
        let e = paired_baseline_from_env(Some("0.01"), None).unwrap_err();
        assert!(e.contains("must be provided together"), "got: {e}");
        let e = paired_baseline_from_env(None, Some("0.13")).unwrap_err();
        assert!(e.contains("must be provided together"), "got: {e}");
        // Non-finite / non-positive / non-numeric ⇒ error, per-key.
        assert!(paired_baseline_from_env(Some("0"), Some("0.13")).is_err());
        assert!(paired_baseline_from_env(Some("-1"), Some("0.13")).is_err());
        assert!(paired_baseline_from_env(Some("inf"), Some("0.13")).is_err());
        assert!(paired_baseline_from_env(Some("nan"), Some("0.13")).is_err());
        assert!(paired_baseline_from_env(Some("0.01"), Some("cheese")).is_err());
    }

    #[test]
    fn commit_identifier_prefers_valid_env_sha() {
        // 7..=40 lowercase hex from MLXFAST_COMMIT_SHA wins (trimmed).
        assert_eq!(commit_identifier(Some("  a1b2c3d  ")), "a1b2c3d");
        assert_eq!(
            commit_identifier(Some("0123456789abcdef0123456789abcdef01234567")),
            "0123456789abcdef0123456789abcdef01234567"
        );
        // is_commit_sha_hex boundaries.
        assert!(is_commit_sha_hex("abcdef0")); // 7
        assert!(!is_commit_sha_hex("abcde")); // 6 too short
        assert!(!is_commit_sha_hex("A1B2C3D")); // uppercase rejected
        assert!(!is_commit_sha_hex("g1b2c3d")); // non-hex rejected
        assert!(!is_commit_sha_hex(
            "0123456789abcdef0123456789abcdef012345678"
        )); // 41 too long
    }

    // -----------------------------------------------------------------------
    // F1 — the SEAM-1 harness identity
    // -----------------------------------------------------------------------

    /// F1 MUTATION PROOF (b), at the seam that matters most: the OFFICIAL GATES-ONLY payload — the
    /// seam-1 artifact the overlay consumes — seals the resolved harness identity.
    ///
    /// Reds if `base_metrics` goes back to `harness_hash: String::new()`, or seals anything other
    /// than the identity the run resolved.
    #[test]
    fn official_gates_only_seals_the_resolved_harness_identity() {
        let golden = official_golden(None);
        let payload = gates_only_with(&golden, conformant_engine);
        assert_eq!(
            payload.metrics.harness_hash,
            HarnessIdentity::TEST_HASH,
            "the gates score must seal the harness identity the run resolved"
        );
        assert!(!payload.metrics.harness_hash.is_empty());
        assert!(bench_core::harness_hash::is_well_formed_harness_hash(
            &payload.metrics.harness_hash
        ));
    }

    /// Every OFFICIAL failure payload seals the identity too — a failed gates score is still an
    /// artifact, and a run that cannot say which harness produced it is the thing F1 removes.
    #[test]
    fn official_failure_payloads_seal_the_harness_identity() {
        let golden = official_golden(None);
        let failed = official_gates_failed(
            &golden,
            ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            GatesFailure {
                error: "boom".to_string(),
                first_failing_case: None,
                first_failing_step: None,
                checked_steps: 0,
            },
        );
        assert_eq!(failed.metrics.harness_hash, HarnessIdentity::TEST_HASH);
        assert!(!failed.passed);
    }

    // ---------------------------------------------------------------------------------------
    // BOARD `metrics.per_prompt` — the single-leg official path's MTP column.
    // ---------------------------------------------------------------------------------------

    /// The PASSING official payload seals exactly ONE `per_prompt` record — one timed prompt was
    /// measured, so one entry — carrying the golden's own sha256, the free-run audit's effective
    /// mean draft length, and the ENFORCED whole-window decode seconds-per-token.
    #[test]
    fn official_seals_one_per_prompt_record_for_the_timed_prompt() {
        let golden = official_golden(None);
        let payload = finish_with(&golden, conformant_engine);
        assert!(
            payload.passed,
            "precondition: this is the passing official path"
        );

        assert_eq!(
            payload.metrics.per_prompt.len(),
            1,
            "official times ONE prompt, so it seals ONE record — never an entry per pool prompt"
        );
        let pp = &payload.metrics.per_prompt[0];
        assert_eq!(
            pp.prompt_sha256, golden.sha256,
            "prompt identity is the golden's bytes (the same identity golden_hash carries)"
        );
        assert_eq!(pp.prompt_sha256, payload.metrics.golden_hash);
        assert_eq!(
            pp.effective_mean_draft_len, IN_BAND_MEAN_DRAFT_LEN,
            "the audit's own value, carried through the TimingResult narrowing"
        );
        assert_eq!(
            pp.mtp_seconds_per_token_mean, payload.metrics.decode_seconds_per_token,
            "the ENFORCED whole-window figure, never a second decode-only number"
        );
    }

    /// END TO END from the ENGINE: a mock that reports 32 rounds of 4 committed tokens must reach
    /// the sealed score as `effective_mean_draft_len = 4.0`. This is the whole chain the field was
    /// missing — `FreeRunAudit` -> `TimingResult` -> `ScoreMetrics` — driven through `official_core`
    /// rather than a synthetic timing. (A mock's ~0 wall clock cannot sit inside the acceptance
    /// band, so this lands on the band-failure payload, which retains the real timing surface.)
    #[test]
    fn official_per_prompt_carries_the_mocked_engines_measured_draft_length() {
        let golden = official_golden(None);
        let mtp_engine =
            || conformant_engine().free_run_acceptance_lengths(vec![4; BENCHMARK_DECODE_STEPS / 4]);
        let payload = run_official(&golden, mtp_engine);

        assert_eq!(payload.metrics.per_prompt.len(), 1);
        let pp = &payload.metrics.per_prompt[0];
        assert_eq!(pp.prompt_sha256, golden.sha256);
        assert_eq!(
            pp.effective_mean_draft_len, 4.0,
            "128 committed tokens over 32 rounds = 4.0 tokens per verify round"
        );
        assert_eq!(
            pp.mtp_seconds_per_token_mean,
            payload.metrics.decode_seconds_per_token
        );
    }

    /// Run the full official window WITH a requested spec, against a mock that reports the given
    /// acceptance histogram and drafted/accepted counters. A mock's ~0 wall clock cannot sit inside
    /// the acceptance band, so this lands on the band-failure payload — which RETAINS the real timed
    /// surface, exactly the payload the spec seal must reach.
    fn run_official_with_spec(
        golden: &GoldenFixture,
        spec: SpecConfig,
        acceptance_lengths: Vec<u32>,
        drafted: u64,
        accepted: u64,
        disagreements: Option<u64>,
    ) -> ScorePayload {
        let committed: u64 = acceptance_lengths.iter().map(|&x| u64::from(x)).sum();
        let timed = move || {
            let engine = conformant_engine()
                .free_run_acceptance_lengths(acceptance_lengths.clone())
                .free_run_totals(drafted, accepted, committed);
            match disagreements {
                Some(d) => engine.free_run_verify_replay_disagreements(d),
                // Not reported: the field never reaches the wire, which is the pre-field engine.
                None => engine,
            }
        };
        official_core_windowed(
            OfficialParams {
                spec: Some(spec),
                ..test_run(
                    golden,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                )
            },
            TEST_BASELINE.bands,
            || Session::connect(timed()).map(|(s, _)| s),
            |_phase: &str| Ok(()),
        )
    }

    /// A histogram of R rounds committing `per_round` tokens each that sums to exactly the official
    /// decode window, so the §2.6 triple holds for a synthetic depth.
    fn even_histogram(per_round: u32) -> Vec<u32> {
        assert_eq!(BENCHMARK_DECODE_STEPS % per_round as usize, 0);
        vec![per_round; BENCHMARK_DECODE_STEPS / per_round as usize]
    }

    /// THE DEFECT THIS CLOSES. The official path built its `TimingParams` with NO spec, so
    /// `free_decode_begin` went out bare and a per-request engine resolved SERIAL while the leg was
    /// scored against the MTP oracle. With `--mtp-depth` wired, the request carries the spec, the
    /// engine's echo is validated equal to it, and the SEALED score states the depth that ran —
    /// DIFFERENTLY for depth 1 and depth 2, so the seal tracks the request rather than a constant.
    #[test]
    fn official_seals_the_requested_mtp_depth_and_the_engines_acceptance_counters() {
        let golden = official_golden(None);

        // Depth 1: 64 rounds of 2 committed tokens; the drafter proposed 64 and 32 were accepted.
        let d1 =
            run_official_with_spec(&golden, SpecConfig::mtp(1), even_histogram(2), 64, 32, None);
        assert_eq!(
            d1.metrics.effective_spec_mode.as_deref(),
            Some("mtp"),
            "the engine's own effective_spec echo, validated equal to the request"
        );
        assert_eq!(d1.metrics.effective_spec_depth, Some(1));
        assert_eq!(d1.metrics.spec_rounds, Some(64));
        assert_eq!(d1.metrics.spec_drafted_total, Some(64));
        assert_eq!(d1.metrics.spec_accepted_total, Some(32));
        assert_eq!(d1.metrics.spec_acceptance_rate, Some(0.5));
        assert_eq!(
            d1.metrics.acceptance_lengths,
            Some(crate::score::AcceptanceLengthsSummary {
                count: 64,
                mean: 2.0
            }),
            "the per-round histogram is sealed as its summary, never verbatim"
        );

        // Depth 2: 32 rounds of 4, a different drafted/accepted pair. The DEPTH differs from the
        // depth-1 run, which is the whole point — a constant would pass one of these, not both.
        let d2 =
            run_official_with_spec(&golden, SpecConfig::mtp(2), even_histogram(4), 96, 72, None);
        assert_eq!(d2.metrics.effective_spec_depth, Some(2));
        assert_ne!(
            d1.metrics.effective_spec_depth,
            d2.metrics.effective_spec_depth
        );
        assert_eq!(d2.metrics.spec_rounds, Some(32));
        assert_eq!(d2.metrics.spec_drafted_total, Some(96));
        assert_eq!(d2.metrics.spec_accepted_total, Some(72));
        assert_eq!(d2.metrics.spec_acceptance_rate, Some(0.75));
        assert_eq!(d2.metrics.acceptance_lengths.map(|a| a.count), Some(32));

        // The per-prompt entry mirrors the same counters the board reads per prompt.
        let pp = &d2.metrics.per_prompt[0];
        assert_eq!(pp.spec_rounds, Some(32));
        assert_eq!(pp.spec_drafted_total, Some(96));
        assert_eq!(pp.spec_accepted_total, Some(72));
        assert_eq!(pp.effective_mean_draft_len, 4.0);
    }

    /// A drafting leg whose drafter proposed NOTHING seals no RATE: a zero denominator has no rate,
    /// and `0.0` would read as "drafted plenty, accepted none". The counters themselves still seal.
    #[test]
    fn official_omits_the_acceptance_rate_when_nothing_was_drafted() {
        let golden = official_golden(None);
        let payload =
            run_official_with_spec(&golden, SpecConfig::mtp(1), even_histogram(1), 0, 0, None);
        assert_eq!(payload.metrics.spec_drafted_total, Some(0));
        assert_eq!(payload.metrics.spec_acceptance_rate, None);
        let json = payload.to_sealed_json().unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["metrics"]
            .as_object()
            .unwrap()
            .get("spec_acceptance_rate")
            .is_none());
    }

    /// THE DEFAULT IS SERIAL, AND SAYS SO. With no spec flag the wire request is bare — today's
    /// behaviour byte-for-byte — and the seal records what that means: mode `serial`, depth `0`, and
    /// NONE of the drafting counters (a serial leg drafts nothing, and its all-ones histogram is
    /// structurally constant).
    #[test]
    fn official_serial_default_seals_mode_serial_depth_zero_and_no_counters() {
        let golden = official_golden(None);
        let payload = run_official(&golden, conformant_engine);
        assert_eq!(
            payload.metrics.effective_spec_mode.as_deref(),
            Some("serial")
        );
        assert_eq!(payload.metrics.effective_spec_depth, Some(0));
        assert_eq!(payload.metrics.spec_rounds, None);
        assert_eq!(payload.metrics.spec_drafted_total, None);
        assert_eq!(payload.metrics.spec_accepted_total, None);
        assert_eq!(payload.metrics.spec_acceptance_rate, None);
        assert!(payload.metrics.acceptance_lengths.is_none());
        // ...and none of them reach the sealed bytes.
        let json = payload.to_sealed_json().unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = v["metrics"].as_object().unwrap();
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
                !obj.contains_key(absent),
                "{absent} must stay absent on a serial leg"
            );
        }
    }

    /// SPEC-NEVER-IGNORED IS NOT WEAKENED BY THE WIRING. An engine that echoes a DIFFERENT depth
    /// than the one requested has its leg DISCARDED — the run fails with the divergence, and no
    /// depth is sealed from the engine's word.
    #[test]
    fn official_discards_a_leg_whose_engine_echoes_a_different_depth() {
        let golden = official_golden(None);
        let liar = || {
            conformant_engine()
                .free_run_acceptance_lengths(even_histogram(4))
                .diverge_spec_echo(SpecConfig::mtp(3))
        };
        let payload = official_core_windowed(
            OfficialParams {
                spec: Some(SpecConfig::mtp(2)),
                ..test_run(
                    &golden,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                )
            },
            TEST_BASELINE.bands,
            || Session::connect(liar()).map(|(s, _)| s),
            |_phase: &str| Ok(()),
        );
        assert!(!payload.passed);
        assert!(
            payload.metrics.error.contains("spec"),
            "expected a spec-echo divergence, got: {}",
            payload.metrics.error
        );
        assert_eq!(payload.metrics.effective_spec_depth, None);
        assert!(payload.metrics.per_prompt.is_empty());
    }

    /// THE VERIFY/REPLAY DISAGREEMENT COUNT IS SEALED WHEN THE ENGINE REPORTS IT.
    ///
    /// The depth-1 MTP cycle's rejecting rounds compare the two-row verify's argmax against the
    /// one-row replay's argmax; the tower is not batch-invariant, so the two can differ. The REPLAY
    /// stands — the divergence is never a refusal — and the engine COUNTS it. This seals that count
    /// on the timed leg beside the other spec counters, so a scored artifact states how often the
    /// two row shapes disagreed instead of leaving it invisible.
    #[test]
    fn official_seals_the_engines_verify_replay_disagreement_count() {
        let golden = official_golden(None);
        // 64 drafted, 32 accepted => 32 rejecting rounds; 30 of them disagreed.
        let payload = run_official_with_spec(
            &golden,
            SpecConfig::mtp(1),
            even_histogram(2),
            64,
            32,
            Some(30),
        );
        assert_eq!(payload.metrics.spec_verify_replay_disagreements, Some(30));
        let json = payload.to_sealed_json().unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            v["metrics"]["spec_verify_replay_disagreements"],
            serde_json::json!(30),
            "the sealed key is spec_verify_replay_disagreements"
        );
        // The counters it sits beside are untouched, and NOTHING ENFORCED moved.
        assert_eq!(payload.metrics.spec_drafted_total, Some(64));
        assert_eq!(payload.metrics.spec_accepted_total, Some(32));
        assert_eq!(payload.metrics.spec_acceptance_rate, Some(0.5));
    }

    /// THE VERIFY PATH IS SEALED WHEN THE ENGINE REPORTS IT. A rectangular window (one target
    /// forward over the 1+k candidates) and the serial oracle (one forward per candidate) produce
    /// the same tokens and very different decode numbers; the seal names which one ran so the
    /// number can be read. Absent on the wire ⇒ no key (a pre-field engine seals unchanged bytes).
    #[test]
    fn official_seals_the_engines_verification_path_when_reported() {
        let golden = official_golden(None);
        let hist = even_histogram(2);
        let committed: u64 = hist.iter().map(|&x| u64::from(x)).sum();
        let engine = move || {
            conformant_engine()
                .free_run_acceptance_lengths(hist.clone())
                .free_run_totals(64, 32, committed)
                .free_run_verification("rectangular", 64, 0)
        };
        let payload = official_core_windowed(
            OfficialParams {
                spec: Some(SpecConfig::mtp(1)),
                ..test_run(
                    &golden,
                    RunDigests::for_test(&DirDigest::empty()),
                    &test_window(),
                )
            },
            TEST_BASELINE.bands,
            || Session::connect(engine()).map(|(s, _)| s),
            |_phase: &str| Ok(()),
        );
        let rectangular = sealed_text("rectangular");
        assert_eq!(
            payload.metrics.spec_verification_mode.as_deref(),
            Some(rectangular.as_str()),
            "the engine's self-reported verify path seals as its digest"
        );
        assert_eq!(
            payload.metrics.spec_rectangular_verification_rounds,
            Some(64)
        );
        assert_eq!(payload.metrics.spec_serial_verification_rounds, Some(0));
        let v: serde_json::Value =
            serde_json::from_str(&payload.to_sealed_json().unwrap()).unwrap();
        assert_eq!(v["metrics"]["spec_verification_mode"], rectangular);
        assert_eq!(v["metrics"]["spec_rectangular_verification_rounds"], 64);
        assert_eq!(v["metrics"]["spec_serial_verification_rounds"], 0);
        let silent =
            run_official_with_spec(&golden, SpecConfig::mtp(1), even_histogram(2), 64, 32, None);
        assert_eq!(silent.metrics.spec_verification_mode, None);
        let v: serde_json::Value = serde_json::from_str(&silent.to_sealed_json().unwrap()).unwrap();
        assert!(v["metrics"]
            .as_object()
            .unwrap()
            .get("spec_verification_mode")
            .is_none());
    }

    /// ABSENT IS NOT ZERO. An engine that does not put the counter on the wire — every engine built
    /// before the field existed — seals NO key, so its score bytes are unchanged. `0` would claim a
    /// measurement that engine never made.
    #[test]
    fn official_omits_the_disagreement_count_when_the_engine_does_not_report_it() {
        let golden = official_golden(None);
        let payload =
            run_official_with_spec(&golden, SpecConfig::mtp(1), even_histogram(2), 64, 32, None);
        assert_eq!(payload.metrics.spec_verify_replay_disagreements, None);
        let json = payload.to_sealed_json().unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(
            v["metrics"]
                .as_object()
                .unwrap()
                .get("spec_verify_replay_disagreements")
                .is_none(),
            "a non-reporting engine must seal no disagreement key at all"
        );
    }

    /// AN IMPOSSIBLE COUNT IS A DEFECT, NOT A DATUM. Only a REJECTING round runs the one-row replay
    /// that can disagree, so the count is bounded by the rejected drafts
    /// (`drafted_total - accepted_total`). An engine claiming more describes rounds that did not
    /// happen: the leg is DISCARDED by name rather than sealed.
    #[test]
    fn official_discards_a_leg_reporting_more_disagreements_than_rejected_drafts() {
        let golden = official_golden(None);
        // 64 drafted, 32 accepted => 32 rejected. 33 is one more than can exist.
        let payload = run_official_with_spec(
            &golden,
            SpecConfig::mtp(1),
            even_histogram(2),
            64,
            32,
            Some(33),
        );
        assert!(!payload.passed);
        assert!(
            payload
                .metrics
                .error
                .contains("verify_replay_disagreements # > rejected drafts #"),
            "expected the named disagreement refusal, got: {}",
            payload.metrics.error
        );
        assert_eq!(payload.metrics.spec_verify_replay_disagreements, None);

        // The BOUNDARY still passes: exactly as many disagreements as rejected drafts is possible.
        let ok = run_official_with_spec(
            &golden,
            SpecConfig::mtp(1),
            even_histogram(2),
            64,
            32,
            Some(32),
        );
        assert_eq!(ok.metrics.spec_verify_replay_disagreements, Some(32));
    }

    /// The ENGINE IDENTITY seal takes the TIMED worker's hello verbatim and mirrors the head digest
    /// onto the per-prompt entry the board reads.
    #[test]
    fn engine_identity_seal_records_the_hello_and_mirrors_the_head_digest() {
        let mut metrics = ScoreMetrics {
            per_prompt: vec![crate::score::ScorePerPrompt::default()],
            ..Default::default()
        };
        let hello = bench_runner::Hello {
            nonce: "n".to_string(),
            protocol_version: Some(1),
            backend: Some("ds4-dfm-rs@abc123 overlay=def nvcc=12.8 driver=580".to_string()),
            device: Some("cuda sm_121".to_string()),
            capabilities: Vec::new(),
            spec_modes: Vec::new(),
            head_provenance: Some(bench_protocol::HeadProvenance {
                sha256: "ab".repeat(32),
                bytes: 7,
                file_count: 1,
            }),
            max_batch_size: None,
            runner: None,
            resident: None,
        };
        seal_engine_identity(&mut metrics, &hello, Some(&"ab".repeat(32)));
        assert_eq!(
            metrics.engine_backend,
            Some(sealed_text(
                "ds4-dfm-rs@abc123 overlay=def nvcc=12.8 driver=580"
            ))
        );
        assert_eq!(metrics.engine_device, Some(sealed_text("cuda sm_121")));
        assert_eq!(metrics.engine_protocol_version, Some(1));
        assert_eq!(metrics.head_provenance_sha256.as_deref(), Some("match"));
        assert_eq!(
            metrics.per_prompt[0].head_provenance_sha256, metrics.head_provenance_sha256,
            "the board reads the head digest off the per-prompt entry"
        );

        // An engine that announces none of it seals no key at all (a pre-#106 engine is unchanged).
        let mut bare = ScoreMetrics::default();
        seal_engine_identity(
            &mut bare,
            &bench_runner::Hello {
                nonce: "n".to_string(),
                protocol_version: None,
                backend: None,
                device: None,
                capabilities: Vec::new(),
                spec_modes: Vec::new(),
                head_provenance: None,
                max_batch_size: None,
                runner: None,
                resident: None,
            },
            None,
        );
        let json = ScorePayload {
            score: None,
            passed: false,
            metrics: bare,
        }
        .to_sealed_json()
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = v["metrics"].as_object().unwrap();
        // Every identity key is ABSENT, never sealed as 0 or "": an absent key says "the worker
        // said nothing", which is a different claim from "the worker said zero".
        for absent in [
            "engine_backend",
            "engine_device",
            "engine_protocol_version",
            "head_provenance_sha256",
            "runner_id",
            "runner_model_type",
            "runner_manifest_sha256",
            "runner_build",
            "resident_pid",
            "resident_load_epoch",
        ] {
            assert!(!obj.contains_key(absent), "{absent} must stay absent");
        }
    }

    /// The RUNNER and RESIDENT identities are sealed exactly like the head provenance — recorded in
    /// the score.json metrics, and read back out of the sealed bytes unchanged.
    #[test]
    fn runner_and_resident_identities_are_sealed_into_the_metrics() {
        let mut metrics = ScoreMetrics::default();
        seal_engine_identity(
            &mut metrics,
            &bench_runner::Hello {
                nonce: "n".to_string(),
                protocol_version: Some(1),
                backend: Some("mlx".to_string()),
                device: Some("m5".to_string()),
                capabilities: Vec::new(),
                spec_modes: Vec::new(),
                head_provenance: None,
                max_batch_size: None,
                runner: Some(bench_protocol::RunnerIdentity {
                    id: "layr/qwen4exp-125b-a6b".to_string(),
                    model_type: "qwen4_exp".to_string(),
                    manifest_sha256: "ab".repeat(32),
                    build: "c4089870".to_string(),
                }),
                resident: Some(bench_protocol::ResidentIdentity {
                    pid: 4242,
                    load_epoch: 1_756_944_000,
                }),
            },
            None,
        );
        assert_eq!(
            metrics.runner_id,
            Some(sealed_text("layr/qwen4exp-125b-a6b"))
        );
        assert_eq!(metrics.runner_model_type, Some(sealed_text("qwen4_exp")));
        assert_eq!(metrics.runner_manifest_sha256.as_deref(), Some("valid"));
        assert_eq!(metrics.runner_build, Some(sealed_text("c4089870")));
        // Engine-chosen resident numbers are logged, never sealed.
        assert_eq!(metrics.resident_pid, None);
        assert_eq!(metrics.resident_load_epoch, None);

        // ROUND TRIP through the sealed bytes: Yukon reads score.json as {score, metrics}, so the
        // keys must survive serialization under their sealed names and read back identically.
        let json = ScorePayload {
            score: Some(1.0),
            passed: true,
            metrics: metrics.clone(),
        }
        .to_sealed_json()
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = v["metrics"].as_object().unwrap();
        assert_eq!(obj["runner_id"], sealed_text("layr/qwen4exp-125b-a6b"));
        assert_eq!(obj["runner_model_type"], sealed_text("qwen4_exp"));
        assert_eq!(obj["runner_manifest_sha256"], "valid");
        assert_eq!(obj["runner_build"], sealed_text("c4089870"));
        assert!(!obj.contains_key("resident_pid"));
        assert!(!obj.contains_key("resident_load_epoch"));
        let back: ScorePayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.metrics, metrics);
    }

    /// WEIGHTS-LOAD-ONCE. A timed window assembled from several worker sessions must see ONE
    /// resident identity across every phase; a changed pid or load_epoch is a mid-window reload and
    /// is refused BY NAME rather than sealed.
    #[test]
    fn a_resident_identity_that_changes_mid_window_is_refused_by_name() {
        let hello_with = |pid: u32, load_epoch: u64| bench_runner::Hello {
            nonce: "n".to_string(),
            protocol_version: Some(1),
            backend: Some("mlx".to_string()),
            device: Some("m5".to_string()),
            capabilities: Vec::new(),
            spec_modes: Vec::new(),
            head_provenance: None,
            max_batch_size: None,
            runner: None,
            resident: Some(bench_protocol::ResidentIdentity { pid, load_epoch }),
        };

        // Same resident across both phases: retained, and the FIRST hello is the one kept.
        let mut slot = None;
        retain_timed_hello(&mut slot, hello_with(4242, 7)).unwrap();
        retain_timed_hello(&mut slot, hello_with(4242, 7)).unwrap();
        assert_eq!(slot.as_ref().unwrap().resident.as_ref().unwrap().pid, 4242);

        // A RELOAD inside the window (same pid, new load_epoch) is refused by name.
        let mut slot = None;
        retain_timed_hello(&mut slot, hello_with(4242, 7)).unwrap();
        let err = retain_timed_hello(&mut slot, hello_with(4242, 8))
            .expect_err("a changed load_epoch is a mid-window reload");
        assert_eq!(err.name(), "resident_identity_changed_within_window");
        assert!(err.to_string().contains("load_epoch 7"));
        assert!(err.to_string().contains("load_epoch 8"));

        // A DIFFERENT resident process is refused too.
        let mut slot = None;
        retain_timed_hello(&mut slot, hello_with(4242, 7)).unwrap();
        assert!(retain_timed_hello(&mut slot, hello_with(9001, 7)).is_err());

        // ...and so is a phase that reports no resident at all when the first one did: half a
        // window on a resident process and half on a self-loading worker is the same lie.
        let mut slot = None;
        retain_timed_hello(&mut slot, hello_with(4242, 7)).unwrap();
        let mut no_resident = hello_with(4242, 7);
        no_resident.resident = None;
        assert!(retain_timed_hello(&mut slot, no_resident).is_err());

        // A window with NO resident anywhere is untouched: a worker without one stays legal.
        let mut plain = hello_with(0, 0);
        plain.resident = None;
        let mut slot = None;
        retain_timed_hello(&mut slot, plain.clone()).unwrap();
        retain_timed_hello(&mut slot, plain).unwrap();
        assert!(slot.unwrap().resident.is_none());
    }

    /// A NON-DRAFTING free-run leg seals what the audit actually computed — one committed token per
    /// verify round — and a genuinely empty histogram seals 0.0 with the KEY STILL PRESENT. Neither
    /// is a placeholder: the board treats 0 as a real value, so benchd must never fabricate one.
    #[test]
    fn official_per_prompt_draft_length_is_never_fabricated() {
        let golden = official_golden(None);

        // The mock's conformant default is `vec![1; N]` — a leg that committed exactly one token
        // per round, i.e. no drafting gain. The honest seal is 1.0, not 0.0.
        let serial = run_official(&golden, conformant_engine);
        assert_eq!(serial.metrics.per_prompt[0].effective_mean_draft_len, 1.0);

        // A zero the audit really computed survives serialization as an explicit 0 (never dropped).
        let mut timing = in_band_timing();
        // An EMPTY histogram: zero rounds, zero committed tokens — mean 0.0, really computed.
        timing.free_run_audit = Some(audit_for_test(Vec::new(), 0, 0));
        let mut metrics = base_metrics(
            Mode::Official,
            &golden,
            RunDigests::for_test(&DirDigest::empty()),
            ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
        );
        apply_timing_metrics(
            &mut metrics,
            &timing,
            ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
        );
        seal_official_per_prompt(&mut metrics, &golden, &timing);
        let sealed = ScorePayload {
            score: Some(1.0),
            passed: true,
            metrics,
        }
        .to_sealed_json()
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&sealed).unwrap();
        assert_eq!(
            v["metrics"]["per_prompt"][0]["effective_mean_draft_len"],
            json!(0.0),
            "0 is a REAL measured value the board reads; it must be sealed, not omitted"
        );
    }

    /// The GATES-ONLY official payload measured no timed prompt, so it seals NO record and the key
    /// stays out of its JSON entirely.
    #[test]
    fn official_gates_only_seals_no_per_prompt() {
        let golden = official_golden(None);
        let payload = official_gates_failed(
            &golden,
            ScoringInputs::local(
                TEST_BASELINE.prefill_seconds_per_token,
                TEST_BASELINE.decode_seconds_per_token,
            ),
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            GatesFailure {
                error: "boom".to_string(),
                first_failing_case: None,
                first_failing_step: None,
                checked_steps: 0,
            },
        );
        assert!(payload.metrics.per_prompt.is_empty());
        let sealed = payload.to_sealed_json().unwrap();
        let v: serde_json::Value = serde_json::from_str(&sealed).unwrap();
        assert!(
            v["metrics"].get("per_prompt").is_none(),
            "an empty array is omitted, so a no-timing payload's bytes are unchanged"
        );
    }

    // ---- THE RANKED PAIRED PATH (David 2026-09-08) --------------------------------------

    /// A calibration whose band is deliberately WIDE. The mock engine's wall clock is ~0, so its
    /// measured seconds-per-token is a real but tiny positive number that no realistic band could
    /// contain; a wide band lets the orchestration under test run to its end instead of stopping
    /// at the health gate. The NARROW band is exercised on its own, against real numbers, in
    /// `baseline.rs` (`the_band_check_holds_on_both_axes_and_in_both_directions`).
    fn wide_calibration() -> crate::baseline::BaselineCalibration {
        crate::baseline::BaselineCalibration {
            version: crate::baseline::CALIBRATION_VERSION,
            track_id: "qwen3.8-125b-a6b-mlx-v1".to_string(),
            box_name: "m5-max-128gb-4-qwen38-125b-a6b-mlx".to_string(),
            reference_commit: "a".repeat(40),
            captured_at: "2026-09-08T00:00:00Z".to_string(),
            benchd_source_commit: "b".repeat(40),
            prompts: vec![wide_prompt_calibration("botany")],
            gates: Vec::new(),
        }
    }

    /// One prompt's WIDE band (see [`wide_calibration`]).
    fn wide_prompt_calibration(prompt: &str) -> crate::baseline::PromptCalibration {
        crate::baseline::PromptCalibration {
            prompt: prompt.to_string(),
            passes: 4,
            prefill_seconds_per_token_mean: 1e-6,
            decode_seconds_per_token_mean: 1e-6,
            prefill_cv: 0.0,
            decode_cv: 0.0,
            prefill_band_low: 1e-6,
            prefill_band_high: 1e6,
            decode_band_low: 1e-6,
            decode_band_high: 1e6,
        }
    }

    /// A calibration every real leg is SLOWER than: means of a nanosecond per token, so the
    /// control leg lands above the band's ceiling and the health gate refuses.
    fn narrow_prompt_calibration(prompt: &str) -> crate::baseline::PromptCalibration {
        crate::baseline::PromptCalibration {
            prefill_seconds_per_token_mean: 1e-9,
            decode_seconds_per_token_mean: 1e-9,
            prefill_band_low: crate::baseline::DEFAULT_PREFILL_BAND_LOW,
            prefill_band_high: crate::baseline::DEFAULT_PREFILL_BAND_HIGH,
            decode_band_low: crate::baseline::DEFAULT_DECODE_BAND_LOW,
            decode_band_high: crate::baseline::DEFAULT_DECODE_BAND_HIGH,
            ..wide_prompt_calibration(prompt)
        }
    }

    fn narrow_calibration() -> crate::baseline::BaselineCalibration {
        crate::baseline::BaselineCalibration {
            prompts: vec![narrow_prompt_calibration("botany")],
            ..wide_calibration()
        }
    }

    fn paired_seal_for_test<'a>(
        calibration: &'a crate::baseline::BaselineCalibration,
    ) -> PairedBaselineSeal<'a> {
        PairedBaselineSeal {
            box_name: &calibration.box_name,
            calibration_sha256: "c0ffee",
            reference_commit: &calibration.reference_commit,
            band_passed: false,
            leg: None,
        }
    }

    /// The paired WINDOW every paired test drives: the test bands, the load-once residency, no
    /// spec, MLX, and the caller's cool gate.
    fn paired_window_for_test<G>(cool_gate: G) -> PairedWindow<G>
    where
        G: FnMut(&str) -> bench_runner::Result<()>,
    {
        PairedWindow {
            bands: TEST_BASELINE.bands,
            spec: None,
            window: test_window(),
            cool_gate,
            gate_log: std::rc::Rc::new(crate::quiescegate::GateLog::new()),
            pairs: 1,
            // The tests drive the ruled floors; the per-project arms set their own.
            floors: SpeedupFloors::DEFAULT,
            weights: ScoringWeights::DEFAULT,
            replay: None,
            token_tolerance_per_thousand: None,
            near_tie_relative_gap: None,
        }
    }

    /// EVERY GATE POINT OF THE RUN IS SEALED SEPARATELY, IN RUN ORDER (David 2026-09-17).
    ///
    /// A two-pair run gates EIGHT times: two pairs x two legs (control, candidate) x two phases
    /// (prefill, decode), and each point runs BOTH gates. The seal names every one of them, so a
    /// reader can see that each timed window — not just the first — opened on an idle, cool box.
    /// The fake gate pair below records a different wait at each point, and the assertion is the
    /// whole array: place, phase, and both gates' readings.
    #[test]
    fn every_gate_point_of_a_two_pair_run_is_sealed_in_order() {
        use crate::quiescegate::{CoolReading, GateLog, QuiescenceReading};
        let golden = official_golden(None);
        let calibration = wide_calibration();
        let log = std::rc::Rc::new(GateLog::new());
        let log_for_gate = std::rc::Rc::clone(&log);
        // A FAKE PAIR OF GATES: each point waits one second longer than the last, so a dropped,
        // duplicated or reordered record shows up as a wrong number rather than a wrong count.
        let point = Cell::new(0u64);
        let gate = move |phase: &str| -> bench_runner::Result<()> {
            let n = point.get();
            point.set(n + 1);
            log_for_gate.record(
                phase,
                &crate::quiescegate::QuiesceRecord {
                    state: crate::quiescegate::QuiesceState::Waited,
                    waited_seconds: n,
                    load: Some(0.5),
                    gpu_util: Some(0.02),
                    skip_reason: None,
                },
                &crate::coolgate::CoolGateRecord {
                    state: crate::coolgate::GateState::Fired,
                    waited_seconds: 0,
                    gpu_temp_c: Some(38.5),
                    skip_reason: None,
                },
            );
            Ok(())
        };

        let payload = official_core_paired(
            &[PairedGoldens {
                candidate: &golden,
                control: &golden,
                prompt: "botany",
            }],
            &calibration,
            paired_seal_for_test(&calibration),
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            PairedLegs {
                open_baseline_leg: |_| Ok(()),
                open_candidate_leg: |_| Ok(()),
                spawn_baseline: || Session::connect(conformant_engine()).map(|(s, _)| s),
                spawn_timed: || Session::connect(conformant_engine()).map(|(s, _)| s),
            },
            PairedWindow {
                pairs: 2,
                gate_log: std::rc::Rc::clone(&log),
                ..paired_window_for_test(gate)
            },
        );

        let quiescence = |waited: u64| QuiescenceReading {
            state: "passed".to_string(),
            waited_seconds: waited,
            load: Some(0.5),
            gpu_util: Some(0.02),
            skip_reason: None,
        };
        let cool = || CoolReading {
            state: "passed".to_string(),
            waited_seconds: 0,
            gpu_temp_c: Some(38.5),
            skip_reason: None,
        };
        let expected: Vec<crate::quiescegate::GateRecord> = [
            (1, "control", "prefill"),
            (1, "control", "decode"),
            (1, "candidate", "prefill"),
            (1, "candidate", "decode"),
            (2, "control", "prefill"),
            (2, "control", "decode"),
            (2, "candidate", "prefill"),
            (2, "candidate", "decode"),
        ]
        .iter()
        .enumerate()
        .map(|(i, (pair, leg, phase))| crate::quiescegate::GateRecord {
            pair: Some(*pair),
            pass: None,
            leg: Some(leg.to_string()),
            phase: phase.to_string(),
            quiescence: quiescence(i as u64),
            cool: cool(),
        })
        .collect();
        assert_eq!(
            payload.metrics.gates, expected,
            "eight gate points, each naming its pair, leg and phase, with both gates' readings"
        );
    }

    /// TWO ROOTS, ONE BOX, IN ORDER. The paired run spawns leg 1's worker from the REFERENCE root
    /// and leg 2's from the CANDIDATE root, and the score's denominator is the number leg 1
    /// MEASURED — not any number in the calibration file.
    ///
    /// The mock's ~0 wall clock cannot sit inside the ACCEPTANCE band, so this lands on the
    /// band-failure payload; that payload retains the real timing surface and the paired seal,
    /// which is exactly the surface under test. The per-leg ENGINE lifecycle is asserted too: leg
    /// 1's guard is opened and dropped before leg 2's is opened.
    #[test]
    fn the_paired_run_measures_two_roots_in_order_and_scores_against_leg_one() {
        use std::rc::Rc;
        let golden = official_golden(None);
        let calibration = wide_calibration();
        let baseline_spawns = Cell::new(0usize);
        let candidate_spawns = Cell::new(0usize);
        // The lifecycle log: each entry is what happened, in the order it happened.
        let events: Rc<std::cell::RefCell<Vec<&'static str>>> =
            Rc::new(std::cell::RefCell::new(Vec::new()));

        /// A leg guard that records its own teardown, so the ordering claim is behavioural.
        struct LegGuard {
            events: Rc<std::cell::RefCell<Vec<&'static str>>>,
            label: &'static str,
        }
        impl Drop for LegGuard {
            fn drop(&mut self) {
                self.events.borrow_mut().push(self.label);
            }
        }

        let payload = official_core_paired(
            &[PairedGoldens {
                candidate: &golden,
                control: &golden,
                prompt: "botany",
            }],
            &calibration,
            paired_seal_for_test(&calibration),
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            PairedLegs {
                open_baseline_leg: |_| {
                    events.borrow_mut().push("baseline-engine-up");
                    Ok(LegGuard {
                        events: Rc::clone(&events),
                        label: "baseline-engine-down",
                    })
                },
                open_candidate_leg: |_| {
                    events.borrow_mut().push("candidate-engine-up");
                    Ok(LegGuard {
                        events: Rc::clone(&events),
                        label: "candidate-engine-down",
                    })
                },
                spawn_baseline: || {
                    baseline_spawns.set(baseline_spawns.get() + 1);
                    Session::connect(conformant_engine()).map(|(s, _)| s)
                },
                spawn_timed: || {
                    candidate_spawns.set(candidate_spawns.get() + 1);
                    Session::connect(conformant_engine()).map(|(s, _)| s)
                },
            },
            paired_window_for_test(|_phase: &str| Ok(())),
        );

        // ONE worker per leg on the load-once residency, from each root, and BOTH roots were used.
        assert_eq!(
            baseline_spawns.get(),
            1,
            "leg 1 opens exactly one worker, from the reference root"
        );
        assert_eq!(
            candidate_spawns.get(),
            1,
            "leg 2 opens exactly one worker, from the candidate root"
        );
        // SEQUENTIAL: leg 1's engine is torn down before leg 2's comes up.
        assert_eq!(
            events.borrow().as_slice(),
            [
                "baseline-engine-up",
                "baseline-engine-down",
                "candidate-engine-up",
                "candidate-engine-down",
            ]
        );

        // THE DENOMINATOR IS THE MEASURED LEG. It is a live measurement, so it is asserted by its
        // PROPERTIES rather than a literal: finite, positive, equal on both of the names that
        // carry it, and NOT the calibration file's mean (which no path may use as a denominator).
        let m = &payload.metrics;
        let leg_prefill = m.baseline_leg_prefill_seconds_per_token.unwrap();
        let leg_decode = m.baseline_leg_decode_seconds_per_token.unwrap();
        assert!(leg_prefill.is_finite() && leg_prefill > 0.0);
        assert!(leg_decode.is_finite() && leg_decode > 0.0);
        assert_eq!(m.baseline_prefill_seconds_per_token, leg_prefill);
        assert_eq!(m.baseline_decode_seconds_per_token, leg_decode);
        assert_ne!(
            leg_prefill, calibration.prompts[0].prefill_seconds_per_token_mean,
            "the calibration's mean must never be the denominator"
        );
        assert_eq!(m.baseline_source.as_deref(), Some("serial-control-leg"));
        assert_eq!(
            m.baseline_box.as_deref(),
            Some(calibration.box_name.as_str())
        );
        assert_eq!(m.baseline_calibration_sha256.as_deref(), Some("c0ffee"));
        assert_eq!(
            m.baseline_reference_commit.as_deref(),
            Some(calibration.reference_commit.as_str())
        );
        assert_eq!(m.baseline_band_passed, Some(true));
        // The CANDIDATE leg's numbers are read back from the enforced fields.
        assert_eq!(
            m.candidate_leg_prefill_seconds_per_token,
            Some(m.prefill_seconds_per_token)
        );
        assert_eq!(
            m.candidate_leg_decode_seconds_per_token,
            Some(m.decode_seconds_per_token)
        );
    }

    /// TWO PAIRS (David 2026-09-09): the same two legs, in the same order, TWICE — control, then
    /// candidate, engine up and down around each leg — and the enforced denominator / numerator
    /// are the per-role MEANS over the pairs, with both pairs sealed as measured.
    #[test]
    fn two_pairs_run_both_legs_twice_and_score_the_lower_median_pair() {
        use std::rc::Rc;
        let golden = official_golden(None);
        let calibration = wide_calibration();
        let baseline_spawns = Cell::new(0usize);
        let candidate_spawns = Cell::new(0usize);
        let events: Rc<std::cell::RefCell<Vec<&'static str>>> =
            Rc::new(std::cell::RefCell::new(Vec::new()));
        struct LegGuard {
            events: Rc<std::cell::RefCell<Vec<&'static str>>>,
            label: &'static str,
        }
        impl Drop for LegGuard {
            fn drop(&mut self) {
                self.events.borrow_mut().push(self.label);
            }
        }
        let mut window = paired_window_for_test(|_phase: &str| Ok(()));
        window.pairs = 2;

        let payload = official_core_paired(
            &[PairedGoldens {
                candidate: &golden,
                control: &golden,
                prompt: "botany",
            }],
            &calibration,
            paired_seal_for_test(&calibration),
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            PairedLegs {
                open_baseline_leg: |_| {
                    events.borrow_mut().push("baseline-engine-up");
                    Ok(LegGuard {
                        events: Rc::clone(&events),
                        label: "baseline-engine-down",
                    })
                },
                open_candidate_leg: |_| {
                    events.borrow_mut().push("candidate-engine-up");
                    Ok(LegGuard {
                        events: Rc::clone(&events),
                        label: "candidate-engine-down",
                    })
                },
                spawn_baseline: || {
                    baseline_spawns.set(baseline_spawns.get() + 1);
                    Session::connect(conformant_engine()).map(|(s, _)| s)
                },
                spawn_timed: || {
                    candidate_spawns.set(candidate_spawns.get() + 1);
                    Session::connect(conformant_engine()).map(|(s, _)| s)
                },
            },
            window,
        );

        assert_eq!(baseline_spawns.get(), 2, "one control worker per pair");
        assert_eq!(candidate_spawns.get(), 2, "one candidate worker per pair");
        assert_eq!(
            events.borrow().as_slice(),
            [
                "baseline-engine-up",
                "baseline-engine-down",
                "candidate-engine-up",
                "candidate-engine-down",
                "baseline-engine-up",
                "baseline-engine-down",
                "candidate-engine-up",
                "candidate-engine-down",
            ],
            "each pair is control then candidate, engines strictly sequential"
        );

        let m = &payload.metrics;
        assert_eq!(m.paired_legs.len(), 2, "both pairs are sealed as measured");
        assert_eq!(m.paired_legs[0].pair, 1);
        assert_eq!(m.paired_legs[1].pair, 2);
        for r in &m.paired_legs {
            assert!(
                r.control_prefill_seconds_per_token.is_finite()
                    && r.control_prefill_seconds_per_token > 0.0
            );
            assert!(
                r.control_decode_seconds_per_token.is_finite()
                    && r.control_decode_seconds_per_token > 0.0
            );
            assert!(
                r.candidate_prefill_seconds_per_token.is_finite()
                    && r.candidate_prefill_seconds_per_token > 0.0
            );
            assert!(
                r.candidate_decode_seconds_per_token.is_finite()
                    && r.candidate_decode_seconds_per_token > 0.0
            );
        }
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0);
        // PAIRS ARE NEVER AVERAGED: the enforced figures are ONE pair's, the lower median of the
        // per-pair composites — on two pairs, the lower composite.
        let composite = |r: &PairedLegRecord| {
            bench_core::score::score_weighted(
                r.candidate_decode_seconds_per_token,
                r.candidate_prefill_seconds_per_token,
                r.control_decode_seconds_per_token,
                r.control_prefill_seconds_per_token,
                ScoringWeights::DEFAULT,
            )
        };
        let scored = if composite(&m.paired_legs[1]) < composite(&m.paired_legs[0]) {
            &m.paired_legs[1]
        } else {
            &m.paired_legs[0]
        };
        // The ENFORCED denominator is the scored pair's control leg, on both names that carry it.
        let control_prefill = scored.control_prefill_seconds_per_token;
        let control_decode = scored.control_decode_seconds_per_token;
        assert!(close(
            m.baseline_leg_prefill_seconds_per_token.unwrap(),
            control_prefill
        ));
        assert!(close(
            m.baseline_leg_decode_seconds_per_token.unwrap(),
            control_decode
        ));
        assert!(close(m.baseline_prefill_seconds_per_token, control_prefill));
        assert!(close(m.baseline_decode_seconds_per_token, control_decode));
        // The ENFORCED numerator is the scored pair's candidate leg.
        assert!(close(
            m.prefill_seconds_per_token,
            scored.candidate_prefill_seconds_per_token
        ));
        assert!(close(
            m.decode_seconds_per_token,
            scored.candidate_decode_seconds_per_token
        ));
        assert_eq!(m.baseline_band_passed, Some(true));
        assert_eq!(m.baseline_source.as_deref(), Some("serial-control-leg"));
    }

    fn pair_with_composite(pair: i64, decode_gain: f64) -> PairedLegRecord {
        PairedLegRecord {
            pair,
            prompt_sha256: String::new(),
            control_prefill_seconds_per_token: 1.0,
            control_decode_seconds_per_token: 1.0,
            candidate_prefill_seconds_per_token: 1.0,
            candidate_decode_seconds_per_token: 1.0 / decode_gain,
            control_seed_prefill_window_seconds_per_token: None,
            control_decode_window_seconds_per_token: None,
            candidate_seed_prefill_window_seconds_per_token: None,
            candidate_decode_window_seconds_per_token: None,
            token_mismatch_count: Some(0),
            token_mismatch_first_step: None,
            token_mismatch_near_tie_count: None,
            token_mismatch_second_choice_count: Some(0),
            token_mismatch_second_choice_max_relative_gap: None,
        }
    }

    /// THE SCORED PAIR IS THE LOWER MEDIAN, NEVER A MEAN: on an odd count the middle pair, on an
    /// even count the lower of the two central pairs, on one pair that pair. The mean of these
    /// gains is a number no pair measured; the rule never returns it.
    #[test]
    fn the_scored_pair_is_the_lower_median_of_the_per_pair_composites() {
        let w = ScoringWeights::DEFAULT;
        let one = [pair_with_composite(1, 1.30)];
        assert_eq!(scored_pair_index(&one, w), 0);
        let three = [
            pair_with_composite(1, 1.30),
            pair_with_composite(2, 1.10),
            pair_with_composite(3, 1.20),
        ];
        assert_eq!(
            scored_pair_index(&three, w),
            2,
            "the middle composite is pair 3"
        );
        let four = [
            pair_with_composite(1, 1.40),
            pair_with_composite(2, 1.10),
            pair_with_composite(3, 1.30),
            pair_with_composite(4, 1.20),
        ];
        assert_eq!(
            scored_pair_index(&four, w),
            3,
            "the lower of the two central composites (1.20, 1.30) is pair 4"
        );
        let tied = [pair_with_composite(1, 1.20), pair_with_composite(2, 1.20)];
        assert_eq!(
            scored_pair_index(&tied, w),
            0,
            "ties keep measurement order"
        );
    }

    /// Two goldens of the same oracle that differ by their digest, and a calibration with a WIDE
    /// band for each of their prompts.
    fn two_goldens() -> (GoldenFixture, GoldenFixture) {
        let first = official_golden(None);
        let mut second = official_golden(None);
        second.sha256 = "b".repeat(64);
        (first, second)
    }

    /// THE PAIRS CYCLE OVER THE GOLDENS, in command-line order: with two goldens and four pairs,
    /// pairs 1 and 3 measure golden 1 and pairs 2 and 4 measure golden 2. Both legs of a pair are
    /// opened for the same golden. Each pair seals the golden it measured, `per_prompt` holds one
    /// record per golden in order, and the control golden sealed is the scored pair's.
    #[test]
    fn the_pairs_cycle_over_the_goldens_and_each_prompt_is_sealed_once() {
        let (first, second) = two_goldens();
        let calibration = crate::baseline::BaselineCalibration {
            prompts: vec![
                wide_prompt_calibration("botany"),
                wide_prompt_calibration("kelp"),
            ],
            ..wide_calibration()
        };
        let opened = std::cell::RefCell::new(Vec::new());
        let mut window = paired_window_for_test(|_phase: &str| Ok(()));
        window.pairs = 4;

        let payload = official_core_paired(
            &[
                PairedGoldens {
                    candidate: &first,
                    control: &first,
                    prompt: "botany",
                },
                PairedGoldens {
                    candidate: &second,
                    control: &second,
                    prompt: "kelp",
                },
            ],
            &calibration,
            paired_seal_for_test(&calibration),
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            PairedLegs {
                open_baseline_leg: |g| {
                    opened.borrow_mut().push(("control", g));
                    Ok(())
                },
                open_candidate_leg: |g| {
                    opened.borrow_mut().push(("candidate", g));
                    Ok(())
                },
                spawn_baseline: || Session::connect(conformant_engine()).map(|(s, _)| s),
                spawn_timed: || Session::connect(conformant_engine()).map(|(s, _)| s),
            },
            window,
        );

        assert_eq!(
            opened.borrow().as_slice(),
            [
                ("control", 0),
                ("candidate", 0),
                ("control", 1),
                ("candidate", 1),
                ("control", 0),
                ("candidate", 0),
                ("control", 1),
                ("candidate", 1),
            ]
        );
        let m = &payload.metrics;
        let measured: Vec<&str> = m
            .paired_legs
            .iter()
            .map(|r| r.prompt_sha256.as_str())
            .collect();
        assert_eq!(
            measured,
            [
                first.sha256.as_str(),
                second.sha256.as_str(),
                first.sha256.as_str(),
                second.sha256.as_str(),
            ]
        );
        // ONE RECORD PER GOLDEN, in order. Each is the lower median of its own two pairs.
        let per_prompt: Vec<&str> = m
            .per_prompt
            .iter()
            .map(|p| p.prompt_sha256.as_str())
            .collect();
        assert_eq!(per_prompt, [first.sha256.as_str(), second.sha256.as_str()]);
        for (record, own) in m.per_prompt.iter().zip([[0, 2], [1, 3]]) {
            let pairs = [m.paired_legs[own[0]].clone(), m.paired_legs[own[1]].clone()];
            let chosen = &pairs[scored_pair_index(&pairs, ScoringWeights::DEFAULT)];
            assert_eq!(
                record.mtp_seconds_per_token_mean,
                chosen.candidate_decode_seconds_per_token
            );
        }
        // The control golden sealed is the one the SCORED pair measured.
        let scored = &m.paired_legs[scored_pair_index(&m.paired_legs, ScoringWeights::DEFAULT)];
        assert_eq!(
            m.baseline_golden_sha256.as_deref(),
            Some(scored.prompt_sha256.as_str())
        );
    }

    /// EACH CONTROL LEG IS GATED BY ITS OWN PROMPT's BAND. Pair 1 measures a prompt with a wide
    /// band and passes; pair 2 measures a prompt with a narrow band and the run stops there, by
    /// name, naming that prompt. Pair 1 stays in the audit trail.
    #[test]
    fn each_control_leg_is_checked_against_the_band_of_its_own_prompt() {
        let (first, second) = two_goldens();
        let calibration = crate::baseline::BaselineCalibration {
            prompts: vec![
                wide_prompt_calibration("botany"),
                narrow_prompt_calibration("kelp"),
            ],
            ..wide_calibration()
        };
        let mut window = paired_window_for_test(|_phase: &str| Ok(()));
        window.pairs = 2;

        let payload = official_core_paired(
            &[
                PairedGoldens {
                    candidate: &first,
                    control: &first,
                    prompt: "botany",
                },
                PairedGoldens {
                    candidate: &second,
                    control: &second,
                    prompt: "kelp",
                },
            ],
            &calibration,
            paired_seal_for_test(&calibration),
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            PairedLegs {
                open_baseline_leg: |_| Ok(()),
                open_candidate_leg: |_| Ok(()),
                spawn_baseline: || Session::connect(conformant_engine()).map(|(s, _)| s),
                spawn_timed: || Session::connect(conformant_engine()).map(|(s, _)| s),
            },
            window,
        );

        assert!(payload.score.is_none());
        let error = &payload.metrics.error;
        assert!(
            error.contains(crate::baseline::SERIAL_CONTROL_LEG_OUTSIDE_BAND),
            "{error}"
        );
        assert!(
            error.contains("pair 2 of 2") && error.contains("kelp"),
            "{error}"
        );
        assert_eq!(payload.metrics.paired_legs.len(), 1);
        assert_eq!(payload.metrics.paired_legs[0].prompt_sha256, first.sha256);
        assert_eq!(
            payload.metrics.baseline_golden_sha256.as_deref(),
            Some(second.sha256.as_str()),
            "the refusal names the control golden of the pair that stopped"
        );
    }

    /// A FAULT IN PAIR 2 ends the run by name, seals no score, and keeps pair 1's measurement in
    /// the audit trail — nothing measured is thrown away, nothing unmeasured is invented.
    #[test]
    fn a_fault_in_the_second_pair_refuses_and_seals_the_first_pair() {
        let golden = official_golden(None);
        let calibration = wide_calibration();
        let candidate_spawns = Cell::new(0usize);
        let gates = Cell::new(0usize);
        let mut window = paired_window_for_test(|_phase: &str| {
            // Gate calls: pair 1 control (prefill, decode), pair 1 candidate (prefill, decode),
            // then pair 2's control prefill gate — which is refused.
            gates.set(gates.get() + 1);
            if gates.get() == 5 {
                Err(RunnerError::Protocol(
                    "GPU never cooled for pair 2".to_string(),
                ))
            } else {
                Ok(())
            }
        });
        window.pairs = 2;

        let payload = official_core_paired(
            &[PairedGoldens {
                candidate: &golden,
                control: &golden,
                prompt: "botany",
            }],
            &calibration,
            paired_seal_for_test(&calibration),
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            PairedLegs {
                open_baseline_leg: |_| Ok(()),
                open_candidate_leg: |_| Ok(()),
                spawn_baseline: || Session::connect(conformant_engine()).map(|(s, _)| s),
                spawn_timed: || {
                    candidate_spawns.set(candidate_spawns.get() + 1);
                    Session::connect(conformant_engine()).map(|(s, _)| s)
                },
            },
            window,
        );

        assert!(!payload.passed, "a faulted pair seals no score");
        assert!(payload.score.is_none());
        assert!(
            payload
                .metrics
                .error
                .contains("GPU never cooled for pair 2"),
            "the run dies by name: {}",
            payload.metrics.error
        );
        assert_eq!(
            candidate_spawns.get(),
            1,
            "pair 2's candidate leg never opens"
        );
        assert_eq!(
            payload.metrics.paired_legs.len(),
            1,
            "pair 1 stays in the audit trail"
        );
        assert_eq!(payload.metrics.paired_legs[0].pair, 1);
    }

    /// LEG 1 OUTSIDE THE BAND: the run dies BY NAME, seals no score, and leg 2 never opens — not
    /// its engine and not its worker. The seal still states which box and which calibration, and
    /// records the leg it measured.
    #[test]
    fn a_control_leg_outside_the_box_band_seals_no_score_and_never_opens_leg_two() {
        let golden = official_golden(None);
        let calibration = narrow_calibration();
        let candidate_spawns = Cell::new(0usize);
        let candidate_engine_ups = Cell::new(0usize);

        let payload = official_core_paired(
            &[PairedGoldens {
                candidate: &golden,
                control: &golden,
                prompt: "botany",
            }],
            &calibration,
            paired_seal_for_test(&calibration),
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            PairedLegs {
                open_baseline_leg: |_| Ok(()),
                open_candidate_leg: |_| {
                    candidate_engine_ups.set(candidate_engine_ups.get() + 1);
                    Ok(())
                },
                spawn_baseline: || Session::connect(conformant_engine()).map(|(s, _)| s),
                spawn_timed: || {
                    candidate_spawns.set(candidate_spawns.get() + 1);
                    Session::connect(conformant_engine()).map(|(s, _)| s)
                },
            },
            paired_window_for_test(|_phase: &str| Ok(())),
        );

        assert!(!payload.passed);
        assert!(payload.score.is_none(), "a refused run seals no score");
        assert!(
            payload
                .metrics
                .error
                .contains("serial-control leg outside this box's band"),
            "{}",
            payload.metrics.error
        );
        assert!(
            payload
                .metrics
                .error
                .contains(crate::baseline::SERIAL_CONTROL_LEG_OUTSIDE_BAND),
            "{}",
            payload.metrics.error
        );
        assert_eq!(
            candidate_engine_ups.get(),
            0,
            "leg 2's engine must not boot"
        );
        assert_eq!(candidate_spawns.get(), 0, "leg 2's worker must not spawn");
        assert_eq!(payload.metrics.baseline_band_passed, Some(false));
        assert_eq!(
            payload.metrics.baseline_source.as_deref(),
            Some("serial-control-leg")
        );
        assert!(payload
            .metrics
            .baseline_leg_prefill_seconds_per_token
            .is_some());
        // No candidate leg ran, so no candidate number is invented.
        assert_eq!(
            payload.metrics.candidate_leg_prefill_seconds_per_token,
            None
        );
        assert_eq!(payload.metrics.candidate_leg_decode_seconds_per_token, None);
        // And NO denominator was established: the enforced fields stay at their placeholders.
        assert_eq!(payload.metrics.baseline_prefill_seconds_per_token, 0.0);
        assert_eq!(payload.metrics.baseline_decode_seconds_per_token, 0.0);
    }

    /// A LEG-1 ENGINE that will not come up stops the run before anything is measured, and the
    /// refusal is the engine's own — never attributed to the candidate.
    #[test]
    fn a_control_leg_engine_that_cannot_boot_refuses_before_any_measurement() {
        let golden = official_golden(None);
        let calibration = wide_calibration();
        let baseline_spawns = Cell::new(0usize);
        let payload = official_core_paired(
            &[PairedGoldens {
                candidate: &golden,
                control: &golden,
                prompt: "botany",
            }],
            &calibration,
            paired_seal_for_test(&calibration),
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            PairedLegs {
                open_baseline_leg: |_| {
                    Err::<(), String>(
                        "LEG-SERVE-BOOT-FAILED: the reference resident died".to_string(),
                    )
                },
                open_candidate_leg: |_| Ok(()),
                spawn_baseline: || {
                    baseline_spawns.set(baseline_spawns.get() + 1);
                    Session::connect(conformant_engine()).map(|(s, _)| s)
                },
                spawn_timed: || Session::connect(conformant_engine()).map(|(s, _)| s),
            },
            paired_window_for_test(|_phase: &str| Ok(())),
        );
        assert!(!payload.passed);
        assert!(
            payload.metrics.error.contains("LEG-SERVE-BOOT-FAILED"),
            "{}",
            payload.metrics.error
        );
        assert_eq!(baseline_spawns.get(), 0, "no worker before its engine");
        assert_eq!(payload.metrics.baseline_band_passed, Some(false));
        assert_eq!(payload.metrics.baseline_leg_prefill_seconds_per_token, None);
    }

    /// The MLX shape: a per-depth oracle tape that agrees with the serial tape at step 0 and
    /// DIVERGES at step 1 — `botany.mtp1.golden.json` against `botany.golden.json`.
    fn per_depth_oracle_tokens() -> Vec<i64> {
        let mut tokens = oracle_decode_tokens();
        tokens[1] += 1_000;
        tokens
    }

    /// A stub engine conformant on the teacher-forced base case and on the oracle tape it is
    /// given, so the two legs of a paired run can be driven with two different tapes.
    fn engine_on_tape(decode_tokens: Vec<i64>) -> MockEngine {
        MockEngine::new()
            .teacher_forced_tokens(vec![2i64; 64])
            .free_run_capable()
            .oracle_tokens(PREFILL_TOKEN, SEED_TOKEN, decode_tokens)
    }

    /// EACH LEG VERIFIES AGAINST ITS OWN TAPE. On a track with per-depth oracle tapes the serial
    /// leg decodes the SERIAL tape and the candidate leg decodes the depth-N tape, and the two
    /// disagree from step 1 on. Given the serial golden as the CONTROL golden, leg 1 passes
    /// against the serial tape and leg 2 still verifies against the candidate golden's — so the
    /// run reaches the band gate, which is as far as a ~0-wall-clock mock can go.
    #[test]
    fn the_control_leg_verifies_against_the_control_golden_and_the_candidate_against_its_own() {
        let candidate_golden = official_golden_with_oracle(per_depth_oracle_tokens(), None);
        let control_golden = official_golden_with_oracle(oracle_decode_tokens(), None);
        let calibration = wide_calibration();
        let candidate_spawns = Cell::new(0usize);

        let payload = official_core_paired(
            &[PairedGoldens {
                candidate: &candidate_golden,
                control: &control_golden,
                prompt: "botany",
            }],
            &calibration,
            paired_seal_for_test(&calibration),
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            PairedLegs {
                open_baseline_leg: |_| Ok(()),
                open_candidate_leg: |_| Ok(()),
                // The REFERENCE tree runs serial, so its engine emits the SERIAL tape.
                spawn_baseline: || {
                    Session::connect(engine_on_tape(oracle_decode_tokens())).map(|(s, _)| s)
                },
                // The CANDIDATE runs at its declared depth, so its engine emits the depth-N tape.
                spawn_timed: || {
                    candidate_spawns.set(candidate_spawns.get() + 1);
                    Session::connect(engine_on_tape(per_depth_oracle_tokens())).map(|(s, _)| s)
                },
            },
            paired_window_for_test(|_phase: &str| Ok(())),
        );

        // LEG 1 PASSED: it was verified against the control golden's tape, not the candidate's.
        assert!(
            !payload.metrics.error.contains(SERIAL_CONTROL_LEG_FAILED),
            "{}",
            payload.metrics.error
        );
        assert_eq!(payload.metrics.baseline_band_passed, Some(true));
        let leg_decode = payload
            .metrics
            .baseline_leg_decode_seconds_per_token
            .expect("leg 1 measured");
        assert!(leg_decode.is_finite() && leg_decode > 0.0);
        // LEG 2 RAN, and it verified against the CANDIDATE golden's tape — a leg driven on the
        // depth-N tape and checked against the serial one would have failed the oracle here.
        assert_eq!(candidate_spawns.get(), 1);
        assert!(
            !payload
                .metrics
                .error
                .contains("benchmark free-run decode token mismatch"),
            "{}",
            payload.metrics.error
        );
        assert!(payload
            .metrics
            .candidate_leg_decode_seconds_per_token
            .is_some());
        // The seal names the golden leg 1 was verified against.
        assert_eq!(
            payload.metrics.baseline_golden_sha256.as_deref(),
            Some(control_golden.sha256.as_str())
        );
    }

    /// THE BUG THIS FLAG FIXES. The SAME two engines, with NO separate control golden: leg 1 is
    /// verified against the candidate's per-depth tape, which its serial decode cannot produce, so
    /// the run dies BY NAME at the first step the two tapes disagree on — step 1 — and leg 2 never
    /// runs. This is the MLX ranked failure `--control-golden` exists for.
    #[test]
    fn one_shared_golden_kills_the_serial_control_leg_on_a_per_depth_oracle_track() {
        let candidate_golden = official_golden_with_oracle(per_depth_oracle_tokens(), None);
        let calibration = wide_calibration();
        let candidate_spawns = Cell::new(0usize);

        let payload = official_core_paired(
            &[PairedGoldens {
                candidate: &candidate_golden,
                control: &candidate_golden,
                prompt: "botany",
            }],
            &calibration,
            paired_seal_for_test(&calibration),
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            PairedLegs {
                open_baseline_leg: |_| Ok(()),
                open_candidate_leg: |_| Ok(()),
                spawn_baseline: || {
                    Session::connect(engine_on_tape(oracle_decode_tokens())).map(|(s, _)| s)
                },
                spawn_timed: || {
                    candidate_spawns.set(candidate_spawns.get() + 1);
                    Session::connect(engine_on_tape(per_depth_oracle_tokens())).map(|(s, _)| s)
                },
            },
            paired_window_for_test(|_phase: &str| Ok(())),
        );

        assert!(!payload.passed);
        assert!(payload.score.is_none());
        assert!(
            payload.metrics.error.contains(SERIAL_CONTROL_LEG_FAILED),
            "{}",
            payload.metrics.error
        );
        assert!(
            payload
                .metrics
                .error
                .contains("benchmark free-run decode token mismatch at step 1"),
            "{}",
            payload.metrics.error
        );
        assert_eq!(candidate_spawns.get(), 0, "leg 2 never runs");
    }

    /// THE SEALED FIELDS OF A PASSING PAIRED RUN. The control leg is MEASURED through the mock
    /// engine from the reference root; the candidate leg is then driven at exactly that leg's
    /// seconds-per-token, which is the only deterministic way to put a mock inside the acceptance
    /// band (its ~0 wall clock is real, so a measured-against-measured ratio is not reproducible).
    /// What is under test is the SEAL and the SCORE of a passing paired run: speedups 1.0, score
    /// 1.0, both legs named, and the historical `baseline_*` fields carrying the leg-1 values so
    /// the board keeps reading.
    #[test]
    fn a_passing_paired_run_seals_both_legs_and_scores_the_live_ratio() {
        let golden = official_golden(None);
        let control = run_serial_control_leg(
            &golden,
            &test_window(),
            || Session::connect(conformant_engine()).map(|(s, _)| s),
            |_phase: &str| Ok(()),
        )
        .expect("the reference root's control leg must measure");
        assert!(control.prefill_seconds_per_token > 0.0);
        assert!(control.decode_seconds_per_token > 0.0);

        // The candidate leg, exactly on the control leg: speedups 1.0, in band, floors pass.
        let mut candidate = in_band_timing();
        candidate.prefill_seconds_per_token = control.prefill_seconds_per_token;
        candidate.decode_seconds_per_token = control.decode_seconds_per_token;
        let mut payload = finish_official(
            &golden,
            ScoringInputs::local(
                control.prefill_seconds_per_token,
                control.decode_seconds_per_token,
            ),
            TEST_BASELINE.bands,
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            &candidate,
            || Session::connect(conformant_engine()).map(|(s, _)| s),
        );
        let calibration = wide_calibration();
        let mut seal = paired_seal_for_test(&calibration);
        seal.band_passed = true;
        seal.leg = Some((
            control.prefill_seconds_per_token,
            control.decode_seconds_per_token,
        ));
        seal_paired_baseline(&mut payload.metrics, &seal, &golden.sha256);

        assert!(payload.passed, "{}", payload.metrics.error);
        assert_eq!(payload.score, Some(1.0));
        let m = &payload.metrics;
        assert_eq!(m.prefill_speedup, 1.0);
        assert_eq!(m.decode_speedup, 1.0);
        // The historical names carry the LEG-1 values, so the board reads them unchanged.
        assert_eq!(
            m.baseline_prefill_seconds_per_token,
            control.prefill_seconds_per_token
        );
        assert_eq!(
            m.baseline_decode_seconds_per_token,
            control.decode_seconds_per_token
        );
        assert_eq!(m.baseline_source.as_deref(), Some("serial-control-leg"));
        assert_eq!(
            m.baseline_box.as_deref(),
            Some(calibration.box_name.as_str())
        );
        assert_eq!(m.baseline_calibration_sha256.as_deref(), Some("c0ffee"));
        assert_eq!(
            m.baseline_reference_commit.as_deref(),
            Some(calibration.reference_commit.as_str())
        );
        assert_eq!(m.baseline_band_passed, Some(true));
        assert_eq!(
            m.baseline_leg_prefill_seconds_per_token,
            Some(control.prefill_seconds_per_token)
        );
        assert_eq!(
            m.baseline_leg_decode_seconds_per_token,
            Some(control.decode_seconds_per_token)
        );
        assert_eq!(
            m.candidate_leg_prefill_seconds_per_token,
            Some(candidate.prefill_seconds_per_token)
        );
        assert_eq!(
            m.candidate_leg_decode_seconds_per_token,
            Some(candidate.decode_seconds_per_token)
        );

        // EVERY paired key reaches the sealed JSON, and none of them is null.
        let sealed: serde_json::Value =
            serde_json::from_str(&payload.to_sealed_json().unwrap()).unwrap();
        for key in [
            "baseline_source",
            "baseline_box",
            "baseline_calibration_sha256",
            "baseline_reference_commit",
            "baseline_band_passed",
            "baseline_leg_prefill_seconds_per_token",
            "baseline_leg_decode_seconds_per_token",
            "candidate_leg_prefill_seconds_per_token",
            "candidate_leg_decode_seconds_per_token",
        ] {
            assert!(
                !sealed["metrics"][key].is_null(),
                "{key} must reach the sealed score"
            );
        }
        // …and a run that measured no control leg seals NONE of them: absent is not zero.
        let single_leg = finish_with(&golden, conformant_engine);
        let sealed: serde_json::Value =
            serde_json::from_str(&single_leg.to_sealed_json().unwrap()).unwrap();
        assert!(sealed["metrics"]["baseline_source"].is_null());
        assert!(sealed["metrics"]["baseline_leg_prefill_seconds_per_token"].is_null());
    }

    /// The CONTROL LEG IS SERIAL BY CONSTRUCTION: it never puts a spec on the wire, whatever the
    /// candidate declares. Proved through the engine's own `effective_spec` echo — a mock that
    /// would refuse an mtp request answers this leg, and the leg reports no spec.
    #[test]
    fn the_control_leg_requests_no_spec() {
        let golden = official_golden(None);
        let control = run_serial_control_leg(
            &golden,
            &test_window(),
            || Session::connect(conformant_engine().spec_modes(None)).map(|(s, _)| s),
            |_phase: &str| Ok(()),
        )
        .expect("a serial control leg must run against an engine that advertises no spec modes");
        assert!(
            control.effective_spec.is_none(),
            "the control leg carried a spec: {:?}",
            control.effective_spec
        );
    }

    /// The golden's oracle at every position a candidate leg emits: the seed token, then the
    /// decode tokens.
    fn emitted_oracle(decode: &[i64]) -> Vec<i64> {
        std::iter::once(SEED_TOKEN)
            .chain(decode.iter().copied())
            .collect()
    }

    /// A candidate engine that emits the oracle `decode` with the decode positions in `changed`
    /// moved. With `second_choice` the moved token is the reference engine's second choice (the
    /// mock's top logits are `t, t + 1, …`); without it the token is outside the top eight.
    fn candidate_with_changes(
        decode: &[i64],
        changed: &[usize],
        second_choice: bool,
    ) -> MockEngine {
        let mut emitted = decode.to_vec();
        for &i in changed {
            emitted[i] += if second_choice { 1 } else { 1_000 };
        }
        engine_on_tape(emitted)
    }

    /// The reference engine: exact on the control leg's tape, and on each replay it chooses the
    /// oracle at every position.
    fn reference_engine(decode: &[i64], replays: usize) -> MockEngine {
        engine_on_tape(decode.to_vec())
            .teacher_forced_sequences(vec![emitted_oracle(decode); replays])
    }

    /// One paired run of `pairs` pairs on `golden` under `tolerance`. `baseline_spawns` counts
    /// every reference worker: one per control leg, plus one when a replay runs.
    fn run_with_tolerance(
        golden: &GoldenFixture,
        tolerance: Option<u32>,
        near_tie_gap: Option<f64>,
        pairs: usize,
        candidate: impl Fn() -> MockEngine,
        baseline_spawns: &Cell<usize>,
    ) -> ScorePayload {
        let decode = golden
            .benchmark
            .as_ref()
            .unwrap()
            .expected_decode_tokens
            .clone();
        let calibration = wide_calibration();
        official_core_paired(
            &[PairedGoldens {
                candidate: golden,
                control: golden,
                prompt: "botany",
            }],
            &calibration,
            paired_seal_for_test(&calibration),
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            PairedLegs {
                open_baseline_leg: |_| Ok(()),
                open_candidate_leg: |_| Ok(()),
                spawn_baseline: || {
                    baseline_spawns.set(baseline_spawns.get() + 1);
                    Session::connect(reference_engine(&decode, pairs)).map(|(s, _)| s)
                },
                spawn_timed: || Session::connect(candidate()).map(|(s, _)| s),
            },
            PairedWindow {
                pairs,
                token_tolerance_per_thousand: tolerance,
                near_tie_relative_gap: near_tie_gap,
                ..paired_window_for_test(|_phase: &str| Ok(()))
            },
        )
    }

    /// UNDER OUR REPLAY POLICY a rejected replay is labelled with the pair that FAILED (its
    /// golden, its control leg, the band it passed), and no pair row seals a mismatch count, exact
    /// or deferred.
    #[test]
    fn a_rejected_replay_names_the_failing_pair_and_seals_no_per_pair_count() {
        let golden = official_golden(None);
        let decode = golden
            .benchmark
            .as_ref()
            .unwrap()
            .expected_decode_tokens
            .clone();
        let calibration = wide_calibration();
        let candidate_spawns = Cell::new(0usize);
        let payload = official_core_paired(
            &[PairedGoldens {
                candidate: &golden,
                control: &golden,
                prompt: "botany",
            }],
            &calibration,
            paired_seal_for_test(&calibration),
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            PairedLegs {
                open_baseline_leg: |_| Ok(()),
                open_candidate_leg: |_| Ok(()),
                spawn_baseline: || Session::connect(reference_engine(&decode, 1)).map(|(s, _)| s),
                // Pair 1 exact; pair 2 commits one token outside the reference's top eight.
                spawn_timed: || {
                    candidate_spawns.set(candidate_spawns.get() + 1);
                    let changed: &[usize] = if candidate_spawns.get() <= 1 {
                        &[]
                    } else {
                        &[7]
                    };
                    Session::connect(candidate_with_changes(
                        &oracle_decode_tokens(),
                        changed,
                        false,
                    ))
                    .map(|(s, _)| s)
                },
            },
            PairedWindow {
                pairs: 2,
                window: WindowShape {
                    timed_divergence_replay: true,
                    ..test_window()
                },
                replay: Some(bench_core::timed_replay::TimedReplayPolicy {
                    max_logit_gap: 2.0,
                    off_argmax_per_thousand: 0,
                }),
                // A mock's wall clock is noise: no floor may end the run before the replay.
                floors: SpeedupFloors {
                    decode: 0.0,
                    prefill: 0.0,
                },
                bands: AcceptanceBands {
                    prefill_up_tolerance: 1e9,
                    prefill_down_tolerance: 1e9,
                    decode_up_tolerance: 1e9,
                    decode_down_tolerance: 1e9,
                    decode_down_enabled: false,
                    prefill_down_enabled: false,
                },
                ..paired_window_for_test(|_phase: &str| Ok(()))
            },
        );
        let m = &payload.metrics;
        assert!(!payload.passed);
        assert!(
            m.error.starts_with("TIMED-REPLAY-REJECTED: pair 2:"),
            "{}",
            m.error
        );
        assert_eq!(m.golden_hash, golden.sha256);
        assert_eq!(m.baseline_band_passed, Some(true));
        assert_eq!(
            m.baseline_leg_decode_seconds_per_token,
            Some(m.paired_legs[1].control_decode_seconds_per_token)
        );
        let json = payload.to_sealed_json().unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        for row in v["metrics"]["paired_legs"].as_array().unwrap() {
            assert!(row["token_mismatch_count"].is_null(), "{json}");
            assert!(
                row["token_mismatch_second_choice_count"].is_null(),
                "{json}"
            );
        }
        assert!(v["metrics"].get("token_mismatch_count").is_none(), "{json}");
    }

    /// ABSENT TOLERANCE: one different decode token stops the candidate leg, exactly as before,
    /// and no replay runs.
    #[test]
    fn an_absent_tolerance_keeps_the_exact_rule() {
        let golden = official_golden(None);
        let spawns = Cell::new(0usize);
        let payload = run_with_tolerance(
            &golden,
            None,
            None,
            1,
            || candidate_with_changes(&oracle_decode_tokens(), &[5], false),
            &spawns,
        );
        assert!(payload.score.is_none());
        assert!(
            payload
                .metrics
                .error
                .contains("benchmark free-run decode token mismatch at step 5"),
            "{}",
            payload.metrics.error
        );
        assert_eq!(spawns.get(), 1, "the control leg only; no replay");
        assert_eq!(payload.metrics.timed_token_tolerance_per_thousand, None);
    }

    /// UNDER THE LIMIT: 12 of 129 tokens is at most 100 per thousand (12 * 1000 <= 100 * 129).
    /// The candidate leg runs to its end, one replay worker counts the mismatches, the run is not
    /// refused for its tokens, and the counts and the limit are sealed.
    #[test]
    fn mismatches_under_the_limit_pass_and_seal_the_counts() {
        let golden = official_golden(None);
        let spawns = Cell::new(0usize);
        let changed: Vec<usize> = (10..22).collect();
        let payload = run_with_tolerance(
            &golden,
            Some(100),
            None,
            1,
            || candidate_with_changes(&oracle_decode_tokens(), &changed, false),
            &spawns,
        );
        let m = &payload.metrics;
        assert!(
            !m.error.contains(TIMED_DIVERGENCE_OVER_TOLERANCE),
            "{}",
            m.error
        );
        assert!(!m.error.contains(TIMED_REPLAY_FAILED), "{}", m.error);
        assert!(!m.error.contains("token mismatch"), "{}", m.error);
        assert_eq!(spawns.get(), 2, "the control leg, then one replay worker");
        let record = &m.paired_legs[0];
        assert_eq!(record.token_mismatch_count, Some(12));
        // Decode position 10 is emitted position 11 (position 0 is the seed token).
        assert_eq!(record.token_mismatch_first_step, Some(11));
        assert_eq!(record.token_mismatch_second_choice_count, Some(0));
        assert_eq!(record.token_mismatch_second_choice_max_relative_gap, None);
        assert_eq!(m.token_mismatch_count, Some(12));
        assert_eq!(m.token_mismatch_first_step, Some(11));
        assert_eq!(m.timed_token_tolerance_per_thousand, Some(100));
    }

    /// OVER THE LIMIT: 13 of 129 tokens is over 100 per thousand. The run is refused by name,
    /// with the pair, the count and the limit, and seals no score.
    #[test]
    fn one_mismatch_over_the_limit_refuses_by_name() {
        let golden = official_golden(None);
        let spawns = Cell::new(0usize);
        let changed: Vec<usize> = (10..23).collect();
        let payload = run_with_tolerance(
            &golden,
            Some(100),
            None,
            1,
            || candidate_with_changes(&oracle_decode_tokens(), &changed, false),
            &spawns,
        );
        assert!(!payload.passed);
        assert!(payload.score.is_none());
        let error = &payload.metrics.error;
        assert!(error.contains(TIMED_DIVERGENCE_OVER_TOLERANCE), "{error}");
        assert!(error.contains("pair 1 of 1"), "{error}");
        assert!(error.contains("of the candidate's 129"), "{error}");
        assert!(
            error.contains("than the 12 the track fixture allows (100 per thousand)"),
            "{error}"
        );
        assert_eq!(
            payload.metrics.paired_legs[0].token_mismatch_count,
            Some(13)
        );
        assert_eq!(payload.metrics.token_mismatch_count, Some(13));
        // The candidate chose where its stream went: no step in the failure payload.
        assert_eq!(payload.metrics.first_failing_step, None);
        assert_eq!(payload.metrics.baseline_band_passed, Some(true));
        // THE SEALED RECORD carries no exact count (13), first step (11) or pair count text.
        let json = payload.to_sealed_json().unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let m = &v["metrics"];
        assert!(!m["error"].as_str().unwrap().contains("13"), "{json}");
        assert_eq!(
            m["token_mismatch_count"], 16,
            "bucketed to the next power of two"
        );
        assert!(m.get("token_mismatch_first_step").is_none(), "{json}");
        assert_eq!(m["paired_legs"][0]["token_mismatch_count"], 16);
        assert!(m["paired_legs"][0]["token_mismatch_first_step"].is_null());
        assert!(
            m.get("first_failing_step").is_none_or(|s| s.is_null()),
            "{json}"
        );
    }

    /// THE CONTROL LEG STAYS EXACT: even under the widest tolerance, one different token on the
    /// reference tree refuses the run by name before the candidate leg opens.
    #[test]
    fn the_control_leg_stays_exact_in_tolerant_mode() {
        let golden = official_golden(None);
        let calibration = wide_calibration();
        let candidate_spawns = Cell::new(0usize);
        let payload = official_core_paired(
            &[PairedGoldens {
                candidate: &golden,
                control: &golden,
                prompt: "botany",
            }],
            &calibration,
            paired_seal_for_test(&calibration),
            RunDigests::for_test(&DirDigest::empty()),
            "deadbeef",
            PairedLegs {
                open_baseline_leg: |_| Ok(()),
                open_candidate_leg: |_| Ok(()),
                spawn_baseline: || {
                    Session::connect(candidate_with_changes(&oracle_decode_tokens(), &[3], false))
                        .map(|(s, _)| s)
                },
                spawn_timed: || {
                    candidate_spawns.set(candidate_spawns.get() + 1);
                    Session::connect(conformant_engine()).map(|(s, _)| s)
                },
            },
            PairedWindow {
                token_tolerance_per_thousand: Some(1000),
                ..paired_window_for_test(|_phase: &str| Ok(()))
            },
        );
        let error = &payload.metrics.error;
        assert!(error.contains(SERIAL_CONTROL_LEG_FAILED), "{error}");
        assert!(
            error.contains("benchmark free-run decode token mismatch at step 3"),
            "{error}"
        );
        assert_eq!(candidate_spawns.get(), 0, "the candidate leg never opens");
    }

    /// EXACT CANDIDATES COST NOTHING: tokens equal to the oracle carry 0 mismatches, and no
    /// replay worker is spawned.
    #[test]
    fn an_exact_candidate_triggers_no_replay() {
        let golden = official_golden(None);
        let spawns = Cell::new(0usize);
        let payload = run_with_tolerance(&golden, Some(100), None, 2, conformant_engine, &spawns);
        assert_eq!(
            spawns.get(),
            2,
            "one worker per control leg and nothing more"
        );
        for record in &payload.metrics.paired_legs {
            assert_eq!(record.token_mismatch_count, Some(0));
            assert_eq!(record.token_mismatch_first_step, None);
            assert_eq!(record.token_mismatch_second_choice_count, Some(0));
        }
        assert_eq!(payload.metrics.token_mismatch_count, Some(0));
    }

    /// Every whole integer in a sealed document: each integer value, and each run of digits in a
    /// string that is not part of a decimal number.
    fn numbers_in(value: &serde_json::Value) -> Vec<i64> {
        match value {
            serde_json::Value::Number(n) => n.as_i64().into_iter().collect(),
            serde_json::Value::String(text) => text
                .split(|c: char| !c.is_ascii_digit() && c != '.')
                .filter(|word| !word.contains('.'))
                .filter_map(|word| word.parse().ok())
                .collect(),
            serde_json::Value::Array(items) => items.iter().flat_map(numbers_in).collect(),
            serde_json::Value::Object(map) => map.values().flat_map(numbers_in).collect(),
            _ => Vec::new(),
        }
    }

    /// NO TOKEN ID IS SEALED: not a hidden golden's decode token, not a token the candidate
    /// emitted, and not a first or second choice of the reference engine — on a passing replay
    /// and on a refusal alike.
    #[test]
    fn the_sealed_output_holds_no_token_id() {
        let decode: Vec<i64> = (0..BENCHMARK_DECODE_STEPS as i64)
            .map(|i| 247_000 + i)
            .collect();
        let golden = official_golden_with_oracle(decode.clone(), None);
        // Under the plain rule, and under a near-tie limit that refuses and one that passes.
        for changed in [(10..22).collect::<Vec<usize>>(), (10..23).collect()] {
            for second_choice in [false, true] {
                for near_tie_gap in [None, Some(0.09), Some(0.2)] {
                    let payload = run_with_tolerance(
                        &golden,
                        Some(100),
                        near_tie_gap,
                        1,
                        || candidate_with_changes(&decode, &changed, second_choice),
                        &Cell::new(0usize),
                    );
                    let sealed: serde_json::Value =
                        serde_json::from_str(&payload.to_sealed_json().unwrap()).unwrap();
                    let sealed_numbers = numbers_in(&sealed);
                    for token in &decode {
                        for id in [*token, token + 1, token + 1_000] {
                            assert!(
                                !sealed_numbers.contains(&id),
                                "token id {id} reached the sealed output"
                            );
                        }
                    }
                }
            }
        }
    }

    /// NEAR TIES ONLY: every mismatch is the reference's second choice inside the declared gap
    /// (the mock's gap is 0.1 at every position). The run is not refused for its tokens, and the
    /// near-tie count and the declared gap are sealed.
    #[test]
    fn near_ties_inside_the_gap_pass_and_seal_the_near_tie_count() {
        let golden = official_golden(None);
        let changed: Vec<usize> = (10..22).collect();
        let payload = run_with_tolerance(
            &golden,
            Some(100),
            Some(0.2),
            1,
            || candidate_with_changes(&oracle_decode_tokens(), &changed, true),
            &Cell::new(0usize),
        );
        let m = &payload.metrics;
        assert!(!m.error.contains("TIMED-DIVERGENCE"), "{}", m.error);
        assert_eq!(m.paired_legs[0].token_mismatch_count, Some(12));
        assert_eq!(m.paired_legs[0].token_mismatch_near_tie_count, Some(12));
        assert_eq!(m.token_mismatch_near_tie_count, Some(12));
        assert_eq!(m.timed_token_near_tie_relative_gap, Some(0.2));
    }

    /// INCLUSIVE: a gap exactly equal to the limit is a near tie.
    #[test]
    fn a_gap_equal_to_the_limit_is_a_near_tie() {
        let golden = official_golden(None);
        let payload = run_with_tolerance(
            &golden,
            Some(100),
            Some(0.1),
            1,
            || candidate_with_changes(&oracle_decode_tokens(), &[10], true),
            &Cell::new(0usize),
        );
        let m = &payload.metrics;
        assert!(!m.error.contains("TIMED-DIVERGENCE"), "{}", m.error);
        assert_eq!(m.paired_legs[0].token_mismatch_near_tie_count, Some(1));
    }

    /// A SECOND CHOICE ABOVE THE GAP is not a near tie. One such mismatch refuses the run by
    /// name, although one mismatch in 129 is inside the per-thousand limit.
    #[test]
    fn a_second_choice_above_the_gap_is_refused_as_not_a_near_tie() {
        let golden = official_golden(None);
        let payload = run_with_tolerance(
            &golden,
            Some(100),
            Some(0.09),
            1,
            || candidate_with_changes(&oracle_decode_tokens(), &[10], true),
            &Cell::new(0usize),
        );
        assert!(payload.score.is_none());
        let error = &payload.metrics.error;
        assert!(error.contains(TIMED_DIVERGENCE_NOT_A_NEAR_TIE), "{error}");
        assert!(error.contains("pair 1 of 1"), "{error}");
        // SEALED text: no position, the gap at 2 decimals.
        assert!(!error.contains("position"), "{error}");
        assert!(!error.contains("11"), "{error}");
        assert!(error.contains("under the limit 0.09"), "{error}");
        assert!(error.contains("the gap 0.10 is above the limit"), "{error}");
        assert_eq!(payload.metrics.paired_legs[0].token_mismatch_count, Some(1));
        assert_eq!(
            payload.metrics.paired_legs[0].token_mismatch_near_tie_count,
            Some(0)
        );
    }

    /// A TOKEN THAT IS NOT THE SECOND CHOICE is not a near tie, whatever the gap.
    #[test]
    fn a_token_that_is_not_the_second_choice_is_refused_as_not_a_near_tie() {
        let golden = official_golden(None);
        let payload = run_with_tolerance(
            &golden,
            Some(100),
            Some(0.2),
            1,
            || candidate_with_changes(&oracle_decode_tokens(), &[10], false),
            &Cell::new(0usize),
        );
        let error = &payload.metrics.error;
        assert!(error.contains(TIMED_DIVERGENCE_NOT_A_NEAR_TIE), "{error}");
        assert!(
            error.contains("the token is not the reference engine's second choice"),
            "{error}"
        );
    }

    /// NEAR TIES STILL COUNT: 13 near ties in 129 tokens is over 100 per thousand.
    #[test]
    fn near_ties_over_the_limit_are_refused_by_the_count_rule() {
        let golden = official_golden(None);
        let changed: Vec<usize> = (10..23).collect();
        let payload = run_with_tolerance(
            &golden,
            Some(100),
            Some(0.2),
            1,
            || candidate_with_changes(&oracle_decode_tokens(), &changed, true),
            &Cell::new(0usize),
        );
        let error = &payload.metrics.error;
        assert!(error.contains(TIMED_DIVERGENCE_OVER_TOLERANCE), "{error}");
        assert!(!error.contains(TIMED_DIVERGENCE_NOT_A_NEAR_TIE), "{error}");
        assert_eq!(payload.metrics.token_mismatch_near_tie_count, Some(13));
    }

    /// A TOLERANCE WITHOUT A GAP keeps the plain rule: a token outside the reference's top two
    /// is counted, not refused, and no near-tie figure is sealed.
    #[test]
    fn a_tolerance_without_a_gap_keeps_the_plain_rule() {
        let golden = official_golden(None);
        let payload = run_with_tolerance(
            &golden,
            Some(100),
            None,
            1,
            || candidate_with_changes(&oracle_decode_tokens(), &[10], false),
            &Cell::new(0usize),
        );
        let m = &payload.metrics;
        assert!(!m.error.contains("TIMED-DIVERGENCE"), "{}", m.error);
        assert_eq!(m.paired_legs[0].token_mismatch_count, Some(1));
        assert_eq!(m.paired_legs[0].token_mismatch_near_tie_count, None);
        assert_eq!(m.token_mismatch_near_tie_count, None);
        assert_eq!(m.timed_token_near_tie_relative_gap, None);
    }

    /// THE NEAR-TIE FIGURES DECIDE NOTHING. The same mismatch count gives the same decision
    /// whether each different token is the reference's second choice or far from its top eight;
    /// only the report-only figures change.
    #[test]
    fn the_near_tie_figures_change_no_decision() {
        let golden = official_golden(None);
        for (changed, refused) in [
            ((10..22).collect::<Vec<usize>>(), false),
            ((10..23).collect(), true),
        ] {
            let run = |second_choice: bool| {
                let spawns = Cell::new(0usize);
                run_with_tolerance(
                    &golden,
                    Some(100),
                    None,
                    1,
                    || candidate_with_changes(&oracle_decode_tokens(), &changed, second_choice),
                    &spawns,
                )
            };
            let far = run(false);
            let near = run(true);
            for payload in [&far, &near] {
                assert_eq!(
                    payload
                        .metrics
                        .error
                        .contains(TIMED_DIVERGENCE_OVER_TOLERANCE),
                    refused,
                    "{}",
                    payload.metrics.error
                );
                assert_eq!(
                    payload.metrics.token_mismatch_count,
                    Some(changed.len() as i64)
                );
            }
            assert_eq!(far.metrics.token_mismatch_second_choice_count, Some(0));
            assert_eq!(
                near.metrics.token_mismatch_second_choice_count,
                Some(changed.len() as i64)
            );
            assert_eq!(
                far.metrics.token_mismatch_second_choice_max_relative_gap,
                None
            );
            // The mock's top two logits are 10.0 and 9.0: (10 - 9) / 10.
            assert_eq!(
                near.metrics.token_mismatch_second_choice_max_relative_gap,
                Some(0.1)
            );
        }
    }

    /// A golden with no benchmark oracle has no prompt for a control leg, and the refusal names
    /// the leg rather than the candidate.
    #[test]
    fn a_control_leg_without_an_oracle_refuses_by_name() {
        let mut doc = serde_json::json!({
            "version": 1,
            "model_type": "qwen4_exp_text",
            "cases": [
                { "name": "case-a", "prompt_tokens": vec![1i64; CORRECTNESS_PROMPT_TOKENS], "expected_tokens": vec![2i64; 64] }
            ]
        });
        doc["cases"][0]["name"] = serde_json::json!("case-a");
        let bytes = serde_json::to_vec(&doc).unwrap();
        let golden = load_golden_fixture(
            &bytes,
            64,
            CORRECTNESS_PROMPT_TOKENS,
            crate::testgolden::identity_125b(),
            Some("qwen4_exp_text"),
            None,
            None,
        )
        .unwrap();
        let err = run_serial_control_leg(
            &golden,
            &test_window(),
            || Session::connect(conformant_engine()).map(|(s, _)| s),
            |_phase: &str| Ok(()),
        )
        .unwrap_err();
        assert!(err.contains(SERIAL_CONTROL_LEG_FAILED), "{err}");
        assert!(err.contains("benchmark oracle"), "{err}");
    }

    /// THE REFERENCE REPLAY judges a deferred divergence from the reference's own top-8 logits:
    /// the mock reference answers every teacher-forced step with top-1 = its sequence token (logit
    /// 10) and the next seven ids one logit apart, so `t` is the argmax, `t + 1` is off-argmax at a
    /// 1.0 gap and `t + 3` is beyond a 2.0 gap.
    mod timed_replay {
        use super::*;
        use bench_core::timed_replay::TimedReplayPolicy;
        use bench_runner::timing::DeferredDivergence;

        const POLICY: TimedReplayPolicy = TimedReplayPolicy {
            max_logit_gap: 2.0,
            off_argmax_per_thousand: 200,
        };

        fn reference() -> Vec<i64> {
            (0..20).map(|i| 100 + 10 * i).collect()
        }

        fn benchmark() -> bench_core::golden::BenchmarkGolden {
            let r = reference();
            bench_core::golden::BenchmarkGolden {
                prefill_prompt_tokens: vec![1, 2, 3],
                expected_prefill_token: r[0],
                decode_seed_tokens: vec![1, 2, 3],
                expected_decode_seed_token: r[0],
                expected_decode_tokens: r[1..].to_vec(),
                baseline_prefill_seconds_per_token: None,
                baseline_decode_seconds_per_token: None,
            }
        }

        fn replay(stream: Vec<i64>, first: usize) -> Result<String, String> {
            let mut open = |_golden: usize| -> Result<(), String> { Ok(()) };
            let mut spawn = || {
                Session::connect(
                    MockEngine::new()
                        .free_run_capable()
                        .teacher_forced_sequences(vec![reference()]),
                )
                .map(|(s, _)| s)
            };
            let benchmark = benchmark();
            replay_deferred_divergences(
                &mut open,
                &mut spawn,
                &[&benchmark],
                &[(
                    1,
                    DeferredDivergence {
                        prefill_token: None,
                        stream: Some((first, stream)),
                    },
                )],
                &POLICY,
            )
            .map_err(|(pair, e)| {
                assert_eq!(pair, 1, "the refusal names the pair that failed");
                e
            })
        }

        #[test]
        fn a_near_argmax_continuation_is_accepted() {
            let mut s = reference();
            s[3] += 1; // one off-argmax position, within the 2.0 gap and the budget of 4
            let summary = replay(s, 3).expect("accepted");
            assert!(summary.contains("1 off-argmax (budget 4)"), "{summary}");
        }

        #[test]
        fn a_token_beyond_the_gap_is_rejected_without_its_position() {
            let mut s = reference();
            s[5] += 3;
            let e = replay(s.clone(), 5).unwrap_err();
            // SEALED text: no position, no committed token id, the gap at 2 decimals.
            assert!(e.contains("timed decode: the committed choice"), "{e}");
            assert!(!e.contains("position"), "{e}");
            assert!(!e.contains(&s[5].to_string()), "{e}");
            assert!(e.contains("3.00 logits below"), "{e}");
            assert!(e.contains("above the declared 2 maximum"), "{e}");
        }

        #[test]
        fn too_many_off_argmax_positions_exceed_the_budget() {
            let mut s = reference();
            for p in [2, 4, 6, 8, 10] {
                s[p] += 1;
            }
            let e = replay(s, 2).unwrap_err();
            // SEALED text: not the candidate's off-argmax count.
            assert!(e.contains("of the timed window's 20 positions"), "{e}");
            assert!(!e.contains(" 5 "), "{e}");
            assert!(e.contains("budget of 4"), "{e}");
        }

        #[test]
        fn positions_before_the_divergence_are_not_judged() {
            let mut s = reference();
            s[1] += 5; // before `first`: matched the golden exactly, so the replay skips it
            s[7] += 1;
            assert!(replay(s, 7).is_ok());
        }
    }
}
