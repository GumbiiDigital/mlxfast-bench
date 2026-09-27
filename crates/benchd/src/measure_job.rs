//! A-1 — the MEASURE-JOB component of the Option-A paired ranked flow (seam 2).
//!
//! This REPLACES the old `--paired` monolith (`crate::paired`, removed): it does ONLY the
//! paired TIMING measurement and emits the scoring-agnostic superset `results.json`. It runs
//! NO correctness/gates (finding 12 dead-by-design — gates come from the seam-1 producer) and
//! authors NO `score.json` (that is seam 3 / the A-3 overlay). The published score / floor /
//! ceiling / merge live in A-3.
//!
//! The heart is the PAIR LOOP (design note §A-1, AMENDMENT 1): a run is not two legs but a
//! loop of alternating serial-control / candidate leg-PAIRS, each behind a per-phase gate with
//! one gated retry, run until `accepted_pair_count >= min_pairs` (targeting `target_pairs`).
//!
//! CLI contract (DRAFT-WF `qwen-mtp-ranked-benchmark.yml@064c0ff2:2088-2098`): the measure-job
//! takes two WORKSPACES (`--candidate`/`--baseline`, cloned on-box), a golden, a `--contract`
//! track fixture, and window/pair knobs, and emits `<out>/results.json`.
//!
//! Every contract-derived behavior NOT checked on a live ranked box carries
//! `// UNVERIFIED(measure-job)`; exact `--contract` field paths carry `// UNVERIFIED(B-4)`.

use std::collections::BTreeMap;
use std::path::Path;

use bench_core::free_run::FreeRunAudit;
use bench_core::golden::GoldenFixture;
use bench_core::tape::TimedPromptTape;
use bench_protocol::{
    SpecConfig, SPEC_MODE_DFLASH, SPEC_MODE_DSPARK, SPEC_MODE_MTP, SPEC_MODE_SERIAL,
};
use bench_runner::{scrub_reason_for_seal, CohortTimingParams, RunnerError, TimingParams};
use serde::{Deserialize, Serialize};

use crate::coolgate::GateState;
use crate::iterate::{finite_nonneg, DirDigest};
use crate::quiescegate::QuiesceRecord;

/// One gated retry per LEG: `MAX_ATTEMPTS = 2` with a FULL precondition reset between attempts (a
/// fresh worker per leg is spawned on every attempt, so the ONE cool gate + quiesce re-run before
/// each try). Finding R15 — the retry unit is now the LEG (one `mtp-timed` verb invocation, one
/// process, ONE gate), NOT the old prefill/decode two-phase pair. Contract:
/// `docs/measure-job-contract.md@7c6be14:279-291` (per-leg `run_phase` loop, W:1615-1640,1718).
/// Doc evicted at cd5782e (docs/EVICTED.md); the pin resolves via git.
/// (Repinned from the unresolvable `@c44e526b`; line correspondence re-verified.)
// UNVERIFIED(measure-job): the one-gated-retry-per-leg structure + full-reset-between-attempts.
const MAX_ATTEMPTS: usize = 2;

/// The maximum number of pair ATTEMPTS the loop will make before giving up, as a multiple of
/// `target_pairs`. Bounds a candidate that keeps rejecting so the loop terminates and fails
/// closed (die 5) rather than spinning. The wrapper's exact budget is un-mirrored.
// UNVERIFIED(measure-job): the total-pair-attempt budget multiple.
const PAIR_ATTEMPT_BUDGET_MULTIPLE: usize = 4;

/// R15 — the serial control runs the SAME timed verb (`mtp-timed`) at MTP DEPTH 0
/// (`QMTP_SERIAL_DEPTH=0`, W:304,1589); the candidate leg runs it at `--mtp-depth D`. Sealed as
/// `serial_control_depth`.
pub const SERIAL_CONTROL_DEPTH: usize = 0;

/// R12 — the sealed `mode` string the results.json carries, verbatim the live wrapper's seal
/// (`live-measure-qwen-mtp-job.sh` seal block). Distinct from the overlay's `scoring_mode`.
pub const MEASURE_JOB_MODE: &str = "qwen-native-mtp-paired-decode-only";

/// COHORT (batch-8 brief) — the sealed mode discriminator of a BATCHED cohort run, distinct from
/// [`MEASURE_JOB_MODE`] so no consumer can read a cohort seal as a per-prompt one. NAMING: pending
/// the orchestrator's naming-convention check.
pub const COHORT_MEASURE_JOB_MODE: &str = "batched-cohort-paired-decode-only";

/// #105 H-A — the SERIES TAG sealed on every leg of this measure-job. Model-2 (Option A) is the
/// TEACHER-FORCED paired regime: benchd feeds each expected token, so speculation can never gain
/// time here. Its numbers are a NEW SERIES ([`bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1`])
/// that MUST NEVER be compared to native-regime / v1.1 free-run numbers. #105 cycle-5 — that rule is
/// ENFORCED, not merely stamped: [`enforce_calibration_series_fence`] runs
/// `bench_core::free_run::timed_modes_comparable` over this constant and the calibration file's own
/// `timed_mode` on the BASELINE_CALIBRATION pre-read, so a cross-series band is die-6 before any
/// measuring. (Until that fence landed the tag had NO production reader — the comparability rule was
/// a claim, and a native-regime file demonstrably banded a Model-2 run to `Pass`.) The real mtp
/// SCORED path is v1.1 free-run, landing separately under its own tag.
pub const TIMED_MODE: &str = bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1;

/// #105 H-A / cycle-5 finding 3 — the TIMED REGIME both legs run, sealed top-level as
/// `timed_regime`. Both legs time a SERIAL teacher-forced decode window
/// ([`timed_decode_wire_spec`]), one invocation per leg (one process per leg), NOT a
/// separately-gated prefill+decode pair.
///
/// NAMED `timed_regime`, NOT `timed_verb` (cycle-5): `"tf-serial-timed"` is a REGIME label and
/// nothing invokes a verb by that name — the actual argv is `<engine> runtime-worker --weights W
/// --mtp-head H` (main.rs; #109 window-2 finding 3 removed the `--mtp-depth`/`--mtp-report` the verb
/// never accepted, see [`RUNTIME_WORKER_ACCEPTED_FLAGS`]). The prior `timed_verb` name asserted an
/// invocation that does not exist, the same class of false seal as the `"mtp-timed"` value H-A
/// removed (that value was false about the REGIME: this job is teacher-forced, so no leg runs an
/// mtp free-run).
///
/// The alternative the review offered — keep the field and seal the literal spawned argv as
/// `timed_invocation` — was NOT taken. The argv is per-leg and environment-dependent (resolved
/// engine path, per-leg head dir, per-attempt report path), so a single top-level string cannot be
/// literally true for both legs, and the parts of it that ARE run identity are already sealed
/// individually (`provenance.candidate_executable` / `baseline_executable`, `mtp_depth`,
/// `serial_control_depth`). (The per-attempt report path that made the argv attempt-dependent is
/// gone with `--mtp-report`; the argv is still per-leg via the head dir and the v1.1 spawn gate.)
/// A regime label is the thing the seal actually knows run-wide. No
/// consumer outside this crate reads the field (verified: no reference in any doc, script, fixture
/// or workflow), so the rename costs no compatibility.
pub const TIMED_REGIME: &str = "tf-serial-timed";

/// W3 — the timed REGIME label of a v1.1 **free-run** leg: one `free_decode_begin` +
/// `free_decode_run(N)` round trip inside one parent-clocked window (PROTOCOL-v1.1.md §2.2). Named
/// distinctly from [`TIMED_REGIME`] so the sealed per-leg regimes never claim the same measurement.
/// (Cycle-5 finding 3 applies here too: this is a REGIME label, not an invocation string.)
pub const FREE_RUN_TIMED_REGIME: &str = "free-run-v1_1-timed";

/// COHORT (batch-8 brief §4.5) — the timed REGIME label of a v1.2 BATCHED free-run leg: one batched
/// `free_decode_begin` + `free_decode_run(N, B)` round trip inside ONE parent-clocked window
/// covering the whole B-stream cohort. Named distinctly from [`FREE_RUN_TIMED_REGIME`] so the
/// sealed per-leg regimes never claim a single-stream and a cohort window are the same measurement.
pub const BATCHED_FREE_RUN_TIMED_REGIME: &str = "batched-free-run-v1_2-timed";

/// Batch-8 brief D0/D8 (RULED) — the ONE scored batch point: B = 8, the whole pinned pool run
/// CONCURRENTLY as one cohort. No sweep, no per-run batch-size choice: the width is a PINNED
/// IDENTITY (D9) declared by the track fixture (`scored_batch_size`), requested by benchd on the
/// wire, echoed by the engine (batch-never-ignored, enforced in the runner), and sealed into
/// results.json — never a plain CLI flag, which could silently differ between legs (the `.auto`-KV
/// failure class D9 names). NAMING: `scored_batch_size` is pending the orchestrator's
/// naming-convention check.
pub const SCORED_BATCH_SIZE_B8: u32 = 8;

/// The per-cohort accepted-pair target (`pairs_per_cohort`). RULED 4 by David 2026-08-26
/// ("you run it using 4 pairs instead of 2 of 8 batches") — 8 prompts × 4 pairs is the
/// challenger-grade sample mass the ruling buys, at the cost of the shorter scored window.
///
/// SUPERSESSION CHAIN (each entry supersedes the one above it):
/// 1. batch-8 brief D2 — default 4.
/// 2. David 2026-08-24 — RULED 2 ("do 2", choosing ~20-minute scored windows over 4-sample
///    medians from the presented trade table).
/// 3. David 2026-08-26 — RULED 4 (this constant), returning to the brief's sample count on
///    sample-mass grounds. The 8/24 ruling is SUPERSEDED, not reinterpreted.
///
/// MEDIAN SUPPORT: the published score stays the shared even-n rule
/// ([`MEDIAN_RULE_EVEN_N`], `bench_core::score::paired_decode_only_median`), and 4 keeps the
/// support EVEN, so the rule still means "mean of the two central order statistics" — at n = 4
/// that is the mean of the 2nd and 3rd sorted cohort ratios, a REAL median that discards the
/// extremes. (At the superseded n = 2 the same rule degenerated to the mean of both samples,
/// discarding nothing; the rule is unchanged, its support is strictly better.) `min_pairs`
/// still floors the accepted count.
/// NAMING: `pairs_per_cohort` is pending the orchestrator's naming-convention check.
pub const PAIRS_PER_COHORT_TARGET: usize = 4;

/// W3 — the top-level `timed_mode` SERIES DESCRIPTOR for a MIXED run: legs measured in two
/// different §5 series. It is deliberately NOT one of the two series tags — the run is not
/// homogeneous, and sealing it as either tag would let a downstream aggregator read a single series
/// where two were measured. The per-leg tags (`pairs[].{serial,candidate}_timed_mode`) and
/// [`TimedSeries`] carry the detail.
///
/// DESIGN NOTE — **RESOLVED per Fable ruling (same-series serial control).** The earlier free-run
/// shape crossed §5 on purpose: a teacher-forced depth-0 serial control divided into a v1.1
/// free-run candidate. The ruling closed it — when the candidate runs the free-run series, the
/// SERIAL CONTROL ALSO free-runs at depth 0 ([`serial_control_regime_for`]), so a scored free-run
/// run is HOMOGENEOUS `free_run_v1_1` with `legs_comparable` computing TRUE.
///
/// The rationale is the round-trip asymmetry, not just regime purity: a teacher-forced leg pays the
/// measured 27.2 ms protocol/spawn floor (M-5) once PER TOKEN — N times —
/// while a free-run leg pays it ONCE for the whole window. The M-5 cancellation argument ("both
/// sides of the ratio carry it") holds only when both sides pay it the same number of times, so the
/// crossed shape carried a systematic component that was round-trip structure rather than
/// speculation, flattering every candidate. §5 names this exact bug.
///
/// The descriptor is KEPT, and the fences that refuse it are kept, as DEFENSE IN DEPTH:
/// `measure-job`'s own paths can no longer produce a crossed run (both legs' regimes come from the
/// one [`serial_control_regime_for`] rule), but the overlay/parse fences still refuse a
/// hand-assembled or future-produced crossed file rather than trusting that invariant.
pub const MIXED_SERIES_DESCRIPTOR: &str = "mixed:teacher_forced_v1_serial+free_run_v1_1_candidate";

/// W3 — the free-run decode window N, RULED at `BENCHMARK_DECODE_STEPS` = 128 (PROTOCOL-v1.1.md
/// OQ3: "reuse `BENCHMARK_DECODE_STEPS`; parity with v1, no new constant or golden regen"). The
/// live wrapper's teacher-forced default window is [`DEFAULT_TOKENS`] (512); the free-run SERIES
/// pins its own window to the signed 128, and a caller that explicitly asks for a different
/// `--tokens` on the free-run path is REFUSED rather than silently re-windowed.
pub const FREE_RUN_DECODE_TOKENS: usize = bench_core::constants::BENCHMARK_DECODE_STEPS;

/// W3 — which TIMED REGIME a leg ran, i.e. which physical quantity its seconds-per-token measures
/// (PROTOCOL-v1.1.md §5). This is the machine-checked half of the comparability rule inside benchd:
/// every leg carries its regime, every seal carries the per-leg tags, and the overlay refuses to
/// aggregate a results.json whose legs disagree with the sealed descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegRegime {
    /// v1 teacher-forced: benchd feeds the golden token as each step's input and times N forced
    /// single-token forwards. Speculation is definitionally absent from the number.
    TeacherForcedV1,
    /// v1.1 oracle-verified free-run: the engine drives its own recurrence, benchd times the whole
    /// batched `free_decode_begin` + `free_decode_run(N)` round trip and exact-matches every
    /// committed token. Acceptance is folded into the wall clock — the number MTP can move.
    FreeRunV1_1,
    /// v1.2 BATCHED (cohort) free-run (batch-8 brief): the engine free-runs ALL pool prompts
    /// CONCURRENTLY through the cohort form of the same verbs, benchd times ONE window over the
    /// whole cohort and exact-matches every committed token in the B x N rectangle. The scored
    /// quantity is COHORT seconds-per-committed-token (`window_elapsed / (B * N)`, D1).
    ///
    /// Orchestrator naming ruling — the variant is GENERIC: the batch width is a pinned IDENTITY
    /// = DATA (D9), read from the fixture's `scored_batch_size`, so it is not dual-encoded into
    /// the type name where it could drift from the fixture value. The payload is a
    /// [`ScoredBatchPoint`], whose ONLY constructor is the exhaustive certify-match over KNOWN
    /// scored batch points — width and series tag are assigned TOGETHER there, so the
    /// `&'static str` tag plumbing and the zero-gate-code series fence (D5) survive, and an
    /// uncertified width is REFUSED at the data boundary, never formatted into a novel tag.
    BatchedFreeRunV1_2(ScoredBatchPoint),
}

/// COHORT (orchestrator naming ruling) — a CERTIFIED scored batch point: the batch width together
/// with the §5 series tag ruled for it, assignable ONLY through [`certify`](Self::certify)'s
/// exhaustive match. The pairing is what makes the generic [`LegRegime::BatchedFreeRunV1_2`]
/// fail-closed with no unreachable arms: a width without a certified series tag cannot construct
/// this type, so no code path ever has to invent (or format) a tag for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoredBatchPoint {
    batch_size: u32,
    timed_mode: &'static str,
}

impl ScoredBatchPoint {
    /// The ONE exhaustive match from data batch size to certified series tag. Currently the only
    /// ruled point is B = 8 (D0/D8: batch 8 with spec decoding, no sweep); a future scored width
    /// is added HERE — with its own ruling, tag and calibration — never by formatting a novel
    /// string at a call site.
    ///
    /// ANGLE-6 BINDING INTERLOCK (orchestrator ruling): certifying the B=8 point ADDITIONALLY
    /// requires the embedded captured engine-wire fixture to actually carry the v1.2 cohort
    /// crosscheck lines ([`bench_runner::captured_fixture_covers_cohort_wire`] — batched hello
    /// with `max_batch_size`, batched begin, batched run). "No scored window before the cohort
    /// crosscheck exists" is thereby a STRUCTURAL refusal at the data boundary, not a schedule:
    /// a benchd whose captured fixture is blind to the cohort legs cannot construct the batched
    /// regime at all.
    pub fn certify(batch_size: u32) -> Result<Self, String> {
        Self::certify_with_coverage(
            batch_size,
            bench_runner::captured_fixture_covers_cohort_wire(),
        )
    }

    /// The certify body with the cohort-coverage fact injected — the TESTABILITY SEAM for the
    /// refusal branch (so tests can prove the missing-coverage refusal without fabricating a
    /// fixture file), NOT a bypass: it is private to this module, and the PUBLIC
    /// [`certify`](Self::certify) always consults the REAL embedded fixture. The coverage gate
    /// is an ADDITIONAL fail-closed condition layered on the exhaustive width match, never a
    /// replacement for it — an uncertified width refuses regardless of coverage.
    fn certify_with_coverage(batch_size: u32, cohort_wire_covered: bool) -> Result<Self, String> {
        match batch_size {
            SCORED_BATCH_SIZE_B8 => {
                if !cohort_wire_covered {
                    return Err(format!(
                        "scored_batch_size {SCORED_BATCH_SIZE_B8} requires the COHORT CROSSCHECK \
                         to exist: the captured engine-wire fixture pinned by \
                         ENGINE_WIRE_V1_SHA256 does not carry the v1.2 cohort lines (a batched \
                         hello advertising batched_free_run_decode with max_batch_size, a \
                         batched free_decode_begin, and a batched free_decode_run), so benchd is \
                         structurally blind to the cohort legs — no B={SCORED_BATCH_SIZE_B8} \
                         scored window before the cohort crosscheck exists (angle-6 BINDING \
                         interlock); extend and repin the captured fixture in BOTH repos first"
                    ));
                }
                Ok(Self {
                    batch_size,
                    timed_mode: bench_core::free_run::TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8,
                })
            }
            other => Err(format!(
                "no certified series tag for scored_batch_size {other}: the ONE ruled batch point \
                 is B={SCORED_BATCH_SIZE_B8} (batch-8 brief D0/D8: batch 8 with spec decoding, no \
                 sweep) — a different scored width needs its own ruling, series tag and \
                 calibration, so it is refused rather than silently run"
            )),
        }
    }

    /// B — the certified cohort width (the fixture's `scored_batch_size`, post-certify).
    pub fn batch_size(self) -> u32 {
        self.batch_size
    }

    /// The §5 series tag certified for this width (e.g. `batched_free_run_v1_2_b8`).
    pub fn timed_mode(self) -> &'static str {
        self.timed_mode
    }
}

impl LegRegime {
    /// The §5 series tag this regime seals.
    pub fn timed_mode(self) -> &'static str {
        match self {
            LegRegime::TeacherForcedV1 => TIMED_MODE,
            LegRegime::FreeRunV1_1 => bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1,
            // The tag certified together with the width at `ScoredBatchPoint::certify` — the one
            // exhaustive data-level match; nothing is derived (or formatted) here.
            LegRegime::BatchedFreeRunV1_2(point) => point.timed_mode(),
        }
    }

    /// The timed REGIME label this leg seals (one invocation, one process, one gate per leg).
    pub fn timed_regime(self) -> &'static str {
        match self {
            LegRegime::TeacherForcedV1 => TIMED_REGIME,
            LegRegime::FreeRunV1_1 => FREE_RUN_TIMED_REGIME,
            LegRegime::BatchedFreeRunV1_2(_) => BATCHED_FREE_RUN_TIMED_REGIME,
        }
    }

    /// Whether this regime free-runs (drives its own recurrence) rather than being teacher-forced.
    /// TRUE for the batched cohort regime too: its legs are spawned with the same v1.1 gate
    /// ([`leg_spawn_args`]), request a wire spec, and echo `effective_spec` — the cohort form adds
    /// width, not a different spawn surface.
    pub fn is_free_run(self) -> bool {
        matches!(
            self,
            LegRegime::FreeRunV1_1 | LegRegime::BatchedFreeRunV1_2(_)
        )
    }

    /// The CERTIFIED scored batch point of a batched regime (`None` off the batched regime) — the
    /// one place downstream code reads the cohort width and its series tag from, so neither can
    /// diverge from the certify-match that admitted the fixture value.
    pub fn scored_batch_point(self) -> Option<ScoredBatchPoint> {
        match self {
            LegRegime::BatchedFreeRunV1_2(point) => Some(point),
            LegRegime::TeacherForcedV1 | LegRegime::FreeRunV1_1 => None,
        }
    }
}

/// W3 — the PRODUCTION rule that picks the candidate leg's regime from its DECLARED spec: a
/// speculating candidate (`mode = mtp`, and `dflash` when that module lands) is scored in the v1.1
/// FREE-RUN regime, because teacher forcing structurally cannot execute speculation (PROTOCOL-v1.1
/// §1.1 — every input is dictated by the harness, so acceptance is invisible to the clock). A
/// `serial` candidate has nothing to free-run and stays teacher-forced.
///
/// This REPLACES the Model-2 downgrade for speculating candidates: previously a declared-mtp
/// candidate had its timed wire spec forced to `serial` ([`timed_decode_wire_spec`]) and was scored
/// as a teacher-forced leg — an honest number, but of the wrong quantity class for MTP. Under
/// David's Option-A ruling the free-run regime IS the scored mtp path.
pub fn candidate_regime_for_spec(spec: &SpecConfig) -> LegRegime {
    match spec.mode.as_str() {
        SPEC_MODE_SERIAL => LegRegime::TeacherForcedV1,
        // mtp today; dflash/dspark join here as their modules land (all are speculating regimes,
        // and every speculating regime needs a free-run window to show its acceptance).
        _ => LegRegime::FreeRunV1_1,
    }
}

/// **Fable ruling — the SAME-SERIES SERIAL CONTROL.** The serial control leg runs THE SAME TIMED
/// REGIME as the candidate. When the candidate runs the v1.1 free-run series, the control also
/// free-runs — at DEPTH 0, with the serial wire spec ([`timed_decode_wire_spec`]), so it drives the
/// engine's existing non-speculating `[1]*N` path through one `free_decode_begin` +
/// `free_decode_run(N)` round trip.
///
/// This is the §5 rule applied to the denominator: `decode_speedup = baseline_spt / candidate_spt`
/// must divide two numbers of the SAME measured quantity. Crossing it is a scoring bug, not a
/// conservative choice, and the concrete bias is structural: a teacher-forced leg pays the measured
/// 27.2 ms protocol floor (M-5) N times, a free-run leg once, so the ratio
/// would carry a round-trip component that has nothing to do with speculation. See the DESIGN NOTE
/// on [`MIXED_SERIES_DESCRIPTOR`].
///
/// Consequence for the seal: a scored free-run run is HOMOGENEOUS `free_run_v1_1`, and
/// `legs_comparable` COMPUTES true. This is the ONLY rule that sets a leg's regime, so measure-job
/// can no longer produce a crossed run at all.
pub fn serial_control_regime_for(candidate_regime: LegRegime) -> LegRegime {
    candidate_regime
}

/// W3 (fence reconciliation) — THE series a run measures in, as one function. Because
/// [`serial_control_regime_for`] makes both legs share a regime, a run has exactly one §5 series
/// tag, and every fence keys on this same value:
///
/// - the CALIBRATION pre-read ([`enforce_calibration_series_fence`], die-6 before any measuring) —
///   the band divides this run's pooled SERIAL mean, so the calibration must have been measured in
///   the series this run's serial control ran;
/// - the BOOTSTRAP author — a `--calibration-bootstrap` run stamps the file with the series it
///   measured, so the file passes its own fence next time and dies against any other series;
/// - the SEAL (`Results::timed_mode`) and the A-3 overlay's §5 fence, which recomputes the same
///   verdict from the per-leg tags.
///
/// One value, one decision function (`bench_core::free_run::timed_modes_comparable`), everywhere.
/// The previous hardcoded [`TIMED_MODE`] at the calibration seam was correct only while every run
/// was teacher-forced: a free-run run would have been banded against teacher-forced calibration —
/// the §5 cross-series bug, one level up from the one the overlay fence catches.
pub fn run_timed_mode(candidate_regime: LegRegime) -> &'static str {
    serial_control_regime_for(candidate_regime).timed_mode()
}

/// #105 H-C — the HONEST `spec_source` decorations. The prior candidate default sealed
/// `"contract-default"`, which is FALSE: no contract speculative-block parsing exists, so the seal
/// claimed a source that isn't there. The candidate default spec is built from the `--mtp-depth`
/// convenience flag (or its default), so its honest source is [`SPEC_SOURCE_MTP_DEPTH_FLAG`]; a
/// `--candidate-spec`/`--baseline-spec` JSON override is [`SPEC_SOURCE_CLI_OVERRIDE`]; the baseline
/// default `{"mode":"serial"}` is [`SPEC_SOURCE_SERIAL_DEFAULT`]. If real contract-surface parsing
/// ever lands, it gets its own source string — the decoration always names where the spec ACTUALLY
/// came from.
pub const SPEC_SOURCE_MTP_DEPTH_FLAG: &str = "mtp-depth-flag";
/// David ruling 2026-08-27 — the honest source when NO `--mtp-depth` was given: the candidate spec
/// is the ENGINE-DECIDES `{"mode":"mtp","mtp":{}}` ([`SpecConfig::mtp_engine_default`]). Depth is
/// the participant's variable (their drafter code sets it), so benchd names no depth at all; the
/// engine's module resolves its own and reports it in the `effective_spec` echo. This RETIRES the
/// old `"mtp-depth-default"` provenance (benchd injecting `DEFAULT_MTP_DEPTH = 2` as a request the
/// engine was then forced to echo — the 2026-08-27 depth experiment sealed that parroted 2 while an
/// in-engine pin drafted at 1). Old artifacts still carry the retired string; nothing writes it.
pub const SPEC_SOURCE_MTP_ENGINE_DEFAULT: &str = "mtp-engine-default";
/// #105 H-C — the honest source for a `--candidate-spec`/`--baseline-spec` JSON override.
pub const SPEC_SOURCE_CLI_OVERRIDE: &str = "cli-override";
/// #105 H-C — the honest source for the baseline default `{"mode":"serial"}`.
pub const SPEC_SOURCE_SERIAL_DEFAULT: &str = "serial-default";

/// R15 — the sealed `prefill_component` (W:1931). The seed prefill is INSIDE the single timed
/// decode window; there is NO separately-scored prefill phase, so the component is `"none"`.
pub const PREFILL_COMPONENT_NONE: &str = "none";

// R13's `DEFAULT_MTP_DEPTH = 2` (the injected default when `--mtp-depth` was omitted) is RETIRED —
// David ruling 2026-08-27: an omitted flag builds the engine-decides spec instead
// ([`SPEC_SOURCE_MTP_ENGINE_DEFAULT`]); benchd injects no depth of its own.

/// R13 — the default `--tokens` decode window (live wrapper default 512, W:539; was 128).
pub const DEFAULT_TOKENS: usize = 512;

/// David ruling (cycle-3) — the DEFENSIVE `--mtp-depth` CAP, mirroring the engine's
/// `MLXFAST_MAX_DRAFT_DEPTH` bound (32). Defense-in-depth on the operator CLI: the engine is the
/// real trust boundary, but benchd rejects an absurd depth before spawning. SUBMISSION-PROOF: on the
/// OFFICIAL/scored path this readonly constant is the cap and the env override is IGNORED; the
/// `MLXFAST_MAX_DRAFT_DEPTH` override is honored ONLY in local-dev (see [`resolve_max_draft_depth_cap`]).
pub const DEFAULT_MAX_DRAFT_DEPTH_CAP: usize = 32;

/// David ruling (cycle-3) — the env var an OPERATOR may set to raise/lower the `--mtp-depth` cap in
/// LOCAL-DEV ONLY. On the official path it is IGNORED (submission-proof), exactly like the engine.
pub const MAX_DRAFT_DEPTH_ENV: &str = "MLXFAST_MAX_DRAFT_DEPTH";

// ---------------------------------------------------------------------------
// R16 — the sealed aggregate CONSTANTS (exact live names/values, W:1852-1941)
// ---------------------------------------------------------------------------

/// R16 — `aggregate.aggregation`: the per-side pooling rule the `*_mean`/`mtp_decode_speedup`
/// figures use (a ratio of the two pooled means), verbatim the live seal.
pub const AGGREGATION_RATIO_OF_MEANS: &str = "ratio_of_means";

/// R16 — `aggregate.score_anchor`: the serial control is the unit reference (serial = 1.0), so a
/// speedup > 1 means the candidate is faster.
pub const SCORE_ANCHOR_SERIAL_ONE: &str = "serial = 1.0";

/// R16 — `aggregate.scoring_aggregation`: the PUBLISHED score is the median of the per-prompt raw
/// serial-relative speedups (`raw_decode_speedup_median`), verbatim.
pub const SCORING_AGGREGATION_MEDIAN_OF_PER_PROMPT: &str =
    "median_of_per_prompt_raw_serial_relative_speedup";

/// R16 — `aggregate.median_rule`: the PUBLISHED median uses the even-n mean-of-two-central-order-
/// statistics rule (the same rule the A-3 overlay recomputes, R18). NAME-TRAP: this is the rule for
/// `raw_decode_speedup_median`, NOT the per-pair LOWER-median diagnostic `mtp_decode_speedup_median`.
pub const MEDIAN_RULE_EVEN_N: &str = "even_n_mean_of_two_central_order_statistics";

/// R16 — `aggregate.mtp_max_draft_depth`: the sealed maximum MTP draft depth (8), verbatim the live
/// wrapper's published constant (Y:2535 passes 8).
pub const MTP_MAX_DRAFT_DEPTH: usize = 8;

/// Seal the ARM-SPECIFIC draft-depth lever(s) for an aggregate, from the candidate leg's
/// engine-ECHOED effective spec — the single decision that keeps `mtp_max_draft_depth` and
/// `dflash_effective_depth` mutually exclusive so a run never carries the wrong arm's depth field.
///
/// * A DFlash run — candidate leg echoed `mode == "dflash"` — seals `dflash_effective_depth` from
///   the echo's OWN `dflash.depth` ([`SpecConfig::dflash_depth`], structurally decoded out of the
///   opaque module block) and carries NO `mtp_max_draft_depth`: the MTP cap does not govern that arm.
/// * Every other run (the MTP arm and its serial control / teacher-forced runs, which echo `mtp` or
///   `serial`, and the batched MTP cohort path) seals the fixed [`MTP_MAX_DRAFT_DEPTH`] cap and no
///   dflash field — unchanged from before this seal existed.
///
/// Returns `(mtp_max_draft_depth, dflash_effective_depth)`. When no candidate echo is available (an
/// empty / all-die-5 pool) it defaults to the historical MTP cap, so the seal shape for a run that
/// measured nothing is unchanged. A dflash echo whose block omits a numeric `depth` yields
/// `(None, None)` — honest, never a fabricated depth and never the MTP constant standing in.
fn arm_draft_depth_seals(
    candidate_effective_spec: Option<&SpecConfig>,
) -> (Option<usize>, Option<usize>) {
    match candidate_effective_spec {
        Some(spec) if spec.mode == SPEC_MODE_DFLASH => {
            (None, spec.dflash_depth().map(|d| d as usize))
        }
        _ => (Some(MTP_MAX_DRAFT_DEPTH), None),
    }
}

/// R16 — the `teacher_forced_v1` `aggregate.decode_speedup_floor`: the LOOSE SANITY floor
/// (`MIN_ACCEPTED_SPEEDUP` = 0.50). NAME-TRAP: this is NOT the ranked 0.90 performance floor (which
/// lives in the A-3 overlay / yml, R20); it is the loose per-accepted-pair sanity value, sealed
/// under its exact live name.
///
/// #117 — the value is now SCOPED to the teacher-forced series. It mirrors the 3.6-epoch live
/// wrapper (W:2204/2256), which only ever measured that series, so the parity claim binds there and
/// nowhere else; #109 comment 5350423826 forbids carrying either series' justification into the
/// other. The `free_run_v1_1` series seals [`FREE_RUN_DECODE_SPEEDUP_FLOOR`] instead — see
/// [`decode_speedup_floor_verdict`].
pub const DECODE_SPEEDUP_FLOOR: f64 = 0.50;

/// #117 — the RULED `free_run_v1_1` `aggregate.decode_speedup_floor`. David, #109 comment
/// 5353123259 (2026-08-20): "floor stays 0.90, no sub-floor bootstrap governance built — the stock
/// free-run median landing below 0.90 'shouldn't happen; ignore the case.'"
///
/// This is an ALIAS, not a second copy of the number: the ranked A-3 overlay floors the same
/// quantity with the same constant ([`bench_core::score::score_paired_decode_only`]), so benchd has
/// exactly ONE definition of the 0.90 floor and the measure-job seal cannot drift from the score
/// the overlay computes over it.
pub use bench_core::constants::QWEN_MTP_DECODE_SPEEDUP_FLOOR as FREE_RUN_DECODE_SPEEDUP_FLOOR;

/// R16 — `aggregate.published_speedup_ceiling`: the wrapper-published plausibility ceiling (5.0,
/// W:491). The ENFORCING ceiling lives in the yml (R20); this seals the wrapper's published value.
// Ceiling 5.0 RULED (David, 2026-08-19) — the wrapper's published plausibility ceiling (W:491,
// 8/17 op-decision). The 3.6-epoch yml still shows 3.0, but 5.0 is the ruling; not churned.
pub const PUBLISHED_SPEEDUP_CEILING: f64 = 5.0;

// ---------------------------------------------------------------------------
// COMPOSITE COHORT SCORING (Gemma track, David ruling 2026-08-23, verbatim)
// ---------------------------------------------------------------------------
//
// "we want to score gemma's benchmark on prefill gains ^ .25 * decode ^ .75. The numbers used
// will be the aggregate across all 8 (summed)" / "so when we get a baseline from the 8 baseline
// prompts, we add their numbers together" / "same with the 8 golden prompts".
//
// Interpretation: RATIO OF SUMS, both legs, per component. Per leg, per component, the 8 streams'
// numbers are aggregated to ONE cohort figure ("added together"); component gain =
// baseline_aggregate / candidate_aggregate (serial-anchored, matching the existing convention: a
// faster candidate scores > 1); composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75 — the
// classic challenge overlay form (`decode_speedup ^ 0.75 * prefill_speedup ^ 0.25`), applied here
// to the two COHORT-level gains rather than two single-number legs.
//
// SHARED-WINDOW RULING (David, superseding the 2026-08-23 per-stream-sums reading) — the
// composite's per-component aggregate is benchd's OWN PARENT-CLOCKED SHARED WINDOW, summed over
// the accepted pairs. ZERO engine-reported input feeds the score, so the composite has NO
// attestation surface to defend at all. [`shared_window_composite`] is the whole scoring path.
//
// WHY (the security argument, recorded here because it is the reason the design is what it is):
//
//  1. An earlier revision read "add their numbers together" as PER-STREAM SUMS. That
//     implementation was BLOCKED in audit: its inputs are the ENGINE-REPORTED per-slot ns vectors
//     (`prefill_ns_by_stream` / `decode_ns_by_stream`), and their attestation
//     (`bench_core::per_stream_attestation`) FLAGS but never REFUSES — a candidate reporting
//     `decode_ns_by_stream = [1; 8]` scores a decode gain around 2e10 with nothing in the
//     pipeline stopping it. benchd's standing parent-clock doctrine (`bench_runner::timing`)
//     distrusts engine self-timed durations for anything SCORED; a published score is precisely
//     where that doctrine has to bite.
//
//  2. The same audit PROVED the two readings EQUIVALENT on the geometry this track runs. The
//     cohort is RECTANGULAR LOCKSTEP — B streams, one fixed budget N, no EOS, no refill — already
//     enforced by the cohort consistency QUADRUPLE (`bench_core::free_run::verify_cohort_
//     consistency`) and the D4 fixed-budget rule. Writing ᾱ for a leg's mean per-stream
//     ATTRIBUTION factor (a stream's reported time / that leg's parent window), the two
//     aggregates relate as `G_sum = G_window × ᾱˢ / ᾱᶜ`; with honest attribution (α ≡ 1) they are
//     IDENTICAL. The α-ratio is not measurement — it is attacker-controlled ATTRIBUTION, i.e.
//     pure gaming surface bolted onto a quantity the parent clock already knows exactly.
//
// So SHARED-WINDOW is not an approximation of the per-stream reading: on this geometry it IS that
// reading, minus the attack. David ruled the now-redundant per-stream SCORING machinery removed;
// #188/#189's per-stream carry and attestation SEAL stay merged untouched as REPORT-ONLY
// diagnostics (box-calibration evidence, never a scored input).
//
// AGGREGATION, per component — RATIO OF SUMS across the ACCEPTED PAIRS, serial numerator:
//
//     prefill_gain = Σ_pairs serial_prefill_window_seconds / Σ_pairs candidate_prefill_window_seconds
//     decode_gain  = Σ_pairs serial_decode_window_seconds  / Σ_pairs candidate_decode_window_seconds
//     composite    = prefill_gain ^ exp_prefill * decode_gain ^ exp_decode
//
// Ratio-of-sums (NOT mean-of-per-pair-ratios, NOT a median of them) is the direct translation of
// "we add their numbers together", and is exactly the shape the per-stream-sum reading has at
// α ≡ 1 — the equivalence above is stated per component and survives the pair-level sum because
// both legs of every pair time the same rectangular cohort.

/// Orchestrator ruling (2026-08-23) — the PREFILL component's RULED exponent (David: "prefill
/// gains ^ .25 * decode ^ .75"). FIXTURE-PINNED IDENTITY, exactly like [`SCORED_BATCH_SIZE_B8`]:
/// this constant is NOT a fallback a missing/wrong fixture declaration defaults to — it is one arm
/// of the exhaustive match [`ScoredExponents::certify`] checks a fixture's DECLARED
/// `scored_exponents` against. A batched-regime run whose fixture omits the field, or declares any
/// other pair, refuses (fail-loud) rather than silently applying this value.
pub const PREFILL_GAIN_EXPONENT: f64 = 0.25;
/// The DECODE component's RULED exponent (David: 0.75). See [`PREFILL_GAIN_EXPONENT`] — same
/// fixture-pinned-identity posture, same certify-match arm, never a fallback.
pub const DECODE_GAIN_EXPONENT: f64 = 0.75;

/// The pinned `(prefill, decode)` exponent PAIR the composite score is raised to, sealed alongside
/// `composite_score` so a reader never has to cross-reference the code constants
/// ([`PREFILL_GAIN_EXPONENT`] / [`DECODE_GAIN_EXPONENT`]) to know what was actually applied. This
/// is ALSO the CERTIFIED type [`ScoredExponents::certify`] returns — "the certified values are
/// what the seal reports" (orchestrator ruling): one type serves as both the certify-match's
/// output and the seal's field type, so there is no second copy that could drift from what was
/// actually certified.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ScoredExponents {
    pub prefill_gain_exponent: f64,
    pub decode_gain_exponent: f64,
}

/// The ONE ruled pair — both exponents are RULED constants (David, 2026-08-23), never per-run
/// data. This is the match's ANSWER, not an unconditional value: nothing seals it directly anymore
/// (see [`ScoredExponents::certify`]) — a run reaches this value only by the fixture's
/// `scored_exponents` declaring EXACTLY this pair.
const SCORED_EXPONENTS: ScoredExponents = ScoredExponents {
    prefill_gain_exponent: PREFILL_GAIN_EXPONENT,
    decode_gain_exponent: DECODE_GAIN_EXPONENT,
};

impl ScoredExponents {
    /// Certify a fixture's DECLARED exponent pair against the ONE ruled pair — the exhaustive
    /// match, mirroring [`ScoredBatchPoint::certify`] exactly:
    ///
    /// * `None` (the field ABSENT) — REFUSED. This function is only ever called on the batched
    ///   cohort regime (single-stream configs never call it — composite scoring is cohort-only),
    ///   so "absent" here means a batched run whose fixture never pinned the exponent identity.
    ///   The pair is read from the ONE migrated schema (`bench_core::contract::Contract`'s flat
    ///   `prefill_gain_exponent` / `decode_gain_exponent`), so the two numbers have ONE spelling in
    ///   a fixture rather than one here and one on the scored single-leg path.
    ///   Fail-loud: no silent default to [`PREFILL_GAIN_EXPONENT`] / [`DECODE_GAIN_EXPONENT`].
    /// * `Some(declared)` NOT bit-identical to the ruled pair — REFUSED, naming both the declared
    ///   and ruled values. Compared via `to_bits()` (an EXACT match, not an epsilon tolerance):
    ///   0.25 and 0.75 are exactly representable in `f64` (both dyadic rationals), so a fixture's
    ///   JSON literal and the Rust constant are the identical value or they are not — there is no
    ///   "close enough" for a pinned identity.
    /// * `Some(declared)` bit-identical to the ruled pair — the ONLY accepted case, returning the
    ///   pinned [`SCORED_EXPONENTS`] (never a value re-formatted from `declared`, exactly as
    ///   [`ScoredBatchPoint::certify`] returns its own pinned data on a match, never the input
    ///   echoed back).
    pub fn certify(declared: Option<(f64, f64)>) -> Result<Self, String> {
        let Some((prefill, decode)) = declared else {
            return Err(format!(
                "the batched cohort regime requires the fixture to declare prefill_gain_exponent \
                 and decode_gain_exponent (beside scored_batch_size): the composite score's \
                 exponent pair is a PINNED IDENTITY, not a code default — absent, this refuses \
                 rather than silently applying {{prefill_gain_exponent: {PREFILL_GAIN_EXPONENT}, \
                 decode_gain_exponent: {DECODE_GAIN_EXPONENT}}}"
            ));
        };
        let ruled = SCORED_EXPONENTS;
        let matches = prefill.to_bits() == ruled.prefill_gain_exponent.to_bits()
            && decode.to_bits() == ruled.decode_gain_exponent.to_bits();
        if matches {
            Ok(ruled)
        } else {
            Err(format!(
                "no certified exponent pair for the declared {{prefill_gain_exponent: {prefill}, \
                 decode_gain_exponent: {decode}}}: the ONE ruled pair (David, 2026-08-23) is \
                 {{prefill_gain_exponent: {PREFILL_GAIN_EXPONENT}, decode_gain_exponent: \
                 {DECODE_GAIN_EXPONENT}}} — a different pair needs its own ruling, refused rather \
                 than silently run"
            ))
        }
    }
}

/// #117 — the sealed `aggregate.{decode_speedup_floor, decode_speedup_floor_met}` PAIR, resolved by
/// the §5 series the run measured in. ONE function so the floor that is SEALED and the verdict
/// computed AGAINST it can never drift apart (the drift #117 reports is exactly that: window 4's
/// free-run legs sealed `0.5` while the ruled gate was `0.90`).
///
/// * `free_run_v1_1` — [`FREE_RUN_DECODE_SPEEDUP_FLOOR`], the RULED 0.90. The subject is the
///   PUBLISHED median (`aggregate.raw_decode_speedup_median`), which is the quantity the ruling, the
///   sealed `scoring_aggregation`, and the ranked A-3 overlay all floor. A median below 0.90 now
///   seals `decode_speedup_floor_met: false` instead of passing against 0.50.
/// * `teacher_forced_v1` — UNCHANGED: the loose [`DECODE_SPEEDUP_FLOOR`] sanity value against the
///   POOLED ratio-of-means (the live wrapper's `num_lt "${speedup}" "${MIN_ACCEPTED_SPEEDUP}"`,
///   W:2204/2256). The 0.90 ruling is a free-run ruling and is NOT inherited here.
///
/// SEAL-ONLY, and this is load-bearing: the verdict is written to `results.json` and is NOT wired to
/// `candidate_accepted` and NOT wired to any exit code — a sub-floor run still exits 0 from
/// `measure-job` with `decode_speedup_floor_met: false` sealed. The FAIL-CLOSED enforcement lives
/// one seam later, in the A-3 overlay ([`bench_core::score::score_paired_decode_only`] →
/// `PairedDecodeFailure::Floor` → `score: null`, `passed: false`, nonzero exit). What this function
/// buys is that the seal now NAMES the gate the overlay will apply, instead of naming 0.50 while the
/// overlay applied 0.90. (The `..._fails_closed` test name refers to that overlay outcome, not to a
/// refusal inside measure-job.)
pub fn decode_speedup_floor_verdict(
    regime: LegRegime,
    pooled_ratio_of_means: f64,
    published_median: f64,
) -> (f64, bool) {
    match regime {
        LegRegime::TeacherForcedV1 => (
            DECODE_SPEEDUP_FLOOR,
            pooled_ratio_of_means >= DECODE_SPEEDUP_FLOOR,
        ),
        // Batch-8 brief D1 (RULED) — the existing floor/ceiling/median machinery is reused
        // UNCHANGED for the cohort series: the scored quantity keeps the ratio shape (serial
        // cohort spt / candidate cohort spt), so the free-run 0.90 floor applies to the published
        // median exactly as on the single-stream free-run series. ZERO new scoring constants.
        LegRegime::FreeRunV1_1 | LegRegime::BatchedFreeRunV1_2(_) => (
            FREE_RUN_DECODE_SPEEDUP_FLOOR,
            published_median >= FREE_RUN_DECODE_SPEEDUP_FLOOR,
        ),
    }
}

/// R16 — the honest `evaluation_target.target_id` default when the `--prompt/--prompt-sha256/
/// --target-id` trio is ABSENT (the run evaluates the default golden pool, not an explicit prompt).
pub const DEFAULT_EVALUATION_TARGET_ID: &str = "default";

/// R16 — the sealed `evaluation_target{target_id, explicit_prompt, prompt_sha256}` (W:1865). Built
/// from the R13 `--prompt/--prompt-sha256/--target-id` trio: when the trio is supplied it names the
/// explicit pinned prompt; when ABSENT it seals the honest [`DEFAULT_EVALUATION_TARGET_ID`] with no
/// fabricated prompt sha (`explicit_prompt = false`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvaluationTarget {
    /// The declared `--target-id`, or the honest `"default"` marker when the trio is absent.
    pub target_id: String,
    /// Whether an EXPLICIT pinned `--prompt` (carrying its `--prompt-sha256`) was supplied.
    pub explicit_prompt: bool,
    /// The pinned `--prompt-sha256` when an explicit prompt was supplied; OMITTED otherwise (a
    /// default-pool run has no single pinned prompt — never fabricated).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_sha256: Option<String>,
}

/// R16 (medium cycle-3) — the sealed `candidate`/`baseline` block: the leg's workspace path and its
/// accept `verdict` (W:1889-1890). The live wrapper only seals results.json on the ACCEPT path so it
/// hardcodes `verdict: "ACCEPT"`; our superset also seals on a die-5 path, so the candidate verdict
/// is HONEST (`"REJECT"` when the candidate did not clear die-5).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WorkspaceVerdict {
    pub workspace: String,
    pub verdict: &'static str,
}

/// R16 — ONE observed on-box telemetry sample the cool gate recorded for a leg (the peak GPU temp
/// and the steady loaded clock). Threaded through [`LegInvocation`] from the gate; folded run-wide
/// into the sealed [`Telemetry`]. A real run supplies these from the sampled telemetry stream.
// UNVERIFIED(measure-job): the on-box per-block sampled telemetry stream (GPU temp / steady freq) is
// an engine/gate-protocol addition the current path does not emit; this is the benchd-side fold +
// seal of those samples. When no sample is available the top-level `telemetry` is OMITTED, never
// fabricated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TelemetrySample {
    pub gpu_temp_c: f64,
    pub steady_freq_mhz: f64,
}

/// R16 — the sealed run-wide `telemetry{max_gpu_temp, min_steady_freq_mhz}` (W:1852-1941): the
/// OBSERVED maximum GPU temperature and minimum steady loaded clock across every accepted leg's
/// samples. Medium (cycle-3) — the telemetry OBJECT is ALWAYS sealed (matching the live shape, which
/// always emits `telemetry: {...}`); a field with NO observed sample is an HONEST `null`, never a
/// fabricated number and never silently dropped.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Telemetry {
    pub max_gpu_temp: Option<f64>,
    pub min_steady_freq_mhz: Option<f64>,
}

/// R16 — folds observed [`TelemetrySample`]s into the run-wide [`Telemetry`] (max temp / min steady
/// freq). Non-finite samples are IGNORED (never fabricated). `into_telemetry` always yields a
/// [`Telemetry`] object; a quantity with no finite sample stays `None` (sealed as an honest `null`).
#[derive(Debug, Default)]
struct TelemetryAccumulator {
    max_gpu_temp: Option<f64>,
    min_steady_freq_mhz: Option<f64>,
}

impl TelemetryAccumulator {
    fn observe(&mut self, s: &TelemetrySample) {
        if s.gpu_temp_c.is_finite() {
            self.max_gpu_temp = Some(
                self.max_gpu_temp
                    .map_or(s.gpu_temp_c, |m| m.max(s.gpu_temp_c)),
            );
        }
        if s.steady_freq_mhz.is_finite() {
            self.min_steady_freq_mhz = Some(
                self.min_steady_freq_mhz
                    .map_or(s.steady_freq_mhz, |m| m.min(s.steady_freq_mhz)),
            );
        }
    }

    fn into_telemetry(self) -> Telemetry {
        Telemetry {
            max_gpu_temp: self.max_gpu_temp,
            min_steady_freq_mhz: self.min_steady_freq_mhz,
        }
    }
}

// ---------------------------------------------------------------------------
// R13 — CLI-surface flag validation (parse-RECOGNIZE + validate; pure + unit-tested)
// ---------------------------------------------------------------------------

/// R13 — `--exactness-probe {none|once|per-prompt|per-pair}` (default `once`, W:560). The MODE is
/// PARSED + VALIDATED + STORED here; the untimed `mtp-verify` hard gate that consumes it is R15
/// (run_exactness_probe W:1546-1583) — DEFERRED, this component only recognises the mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExactnessProbe {
    None,
    /// Probe the FIRST prompt only (the default).
    #[default]
    Once,
    PerPrompt,
    PerPair,
}

impl ExactnessProbe {
    /// Parse the `--exactness-probe` value; a usage error (die-9 style) names the allowed set.
    pub fn parse(s: &str) -> Result<ExactnessProbe, String> {
        match s {
            "none" => Ok(ExactnessProbe::None),
            "once" => Ok(ExactnessProbe::Once),
            "per-prompt" => Ok(ExactnessProbe::PerPrompt),
            "per-pair" => Ok(ExactnessProbe::PerPair),
            other => Err(format!(
                "--exactness-probe must be one of none|once|per-prompt|per-pair, got {other:?}"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ExactnessProbe::None => "none",
            ExactnessProbe::Once => "once",
            ExactnessProbe::PerPrompt => "per-prompt",
            ExactnessProbe::PerPair => "per-pair",
        }
    }
}

/// David ruling (cycle-3) — resolve the effective `--mtp-depth` CAP, SUBMISSION-PROOF like the
/// engine. The OFFICIAL/scored path (`local_dev == false`) ALWAYS uses the readonly
/// [`DEFAULT_MAX_DRAFT_DEPTH_CAP`] (32) and IGNORES `MLXFAST_MAX_DRAFT_DEPTH` — an env override can
/// never widen a scored submission. In LOCAL-DEV a valid numeric override replaces the cap; a
/// missing/blank/non-numeric value falls back to the constant (fail-closed to 32, never uncapped).
pub fn resolve_max_draft_depth_cap(local_dev: bool, env_override: Option<&str>) -> usize {
    if !local_dev {
        // OFFICIAL: env is not even consulted — the cap is the fixed constant.
        return DEFAULT_MAX_DRAFT_DEPTH_CAP;
    }
    match env_override.map(str::trim).filter(|s| !s.is_empty()) {
        Some(v) => v.parse::<usize>().unwrap_or(DEFAULT_MAX_DRAFT_DEPTH_CAP),
        None => DEFAULT_MAX_DRAFT_DEPTH_CAP,
    }
}

/// The DEFAULT track allowed-modes list (`docs/spec-config-design.md` §4.5): the modes a submission
/// may declare on a track whose `--contract` fixture declares no list of its own. `serial` (the
/// pinned baseline denominator) and `mtp`.
///
/// **David ruling (2026-08-26) — the contract's own `allowed_modes` OVERRIDES this when present**
/// ([`resolve_allowed_modes`]). Until that ruling this doc comment already CLAIMED the override
/// existed ("The contract's own allowed-modes list overrides this when present") while no such
/// field was modelled and this constant had exactly one hardcoded consumer — so `dflash` was
/// refused on EVERY track, in EVERY regime, with no fixture able to say otherwise. The claim is now
/// true.
///
/// This constant stays the ABSENT-case answer rather than being widened, and that is the whole
/// point of making the fence contract-driven: widening the global default would have enabled
/// `dflash` on the qwen3.8 and laguna fixtures too, which never declared it and whose calibration,
/// floors and leaderboard were never measured for it. A fixture opts IN; silence keeps the old
/// answer.
pub const DEFAULT_ALLOWED_MODES: [&str; 2] = [SPEC_MODE_SERIAL, SPEC_MODE_MTP];

/// The modes a track fixture may DECLARE in its `allowed_modes` list — the vocabulary
/// [`resolve_allowed_modes`] certifies against, so a typo or a mode with no benchd support is
/// refused AT THE FIXTURE rather than surviving to become an opaque engine-side reject.
///
/// `dspark` is deliberately ABSENT: `docs/spec-config-design.md` §4 records it as reserved pending
/// cudafast#26, so no fixture may declare it yet. It is refused BY NAME (a distinct diagnosis from
/// an unknown string) in [`resolve_allowed_modes`].
pub const DECLARABLE_MODES: [&str; 3] = [SPEC_MODE_SERIAL, SPEC_MODE_MTP, SPEC_MODE_DFLASH];

/// **David ruling (2026-08-26)** — resolve the track's EFFECTIVE allowed-modes list from the
/// `--contract` fixture's optional `allowed_modes`, FAIL-CLOSED on a list benchd cannot honour.
///
/// * ABSENT ⇒ [`DEFAULT_ALLOWED_MODES`]. Every track that never declared a list keeps EXACTLY the
///   behaviour it had before the ruling — this is the other-track protection, and it is why the
///   ruling is implemented as a per-fixture opt-in rather than as a wider global default.
/// * PRESENT ⇒ the declared list, after certification:
///   * EMPTY refuses — a track that declares a list must say what is in it; an empty list would
///     refuse its own serial baseline and read as "no modes", which no fixture can have meant.
///   * an entry outside [`DECLARABLE_MODES`] refuses, naming the entry. `dspark` gets its own
///     message (reserved, module not landed) so a fixture author is not told their spelling is
///     wrong when their timing is.
///   * a DUPLICATE entry refuses. A list is a set; a repeat is a fixture edit that went wrong, and
///     silently de-duplicating it would hide that.
///   * a list WITHOUT `serial` refuses. The baseline leg is pinned serial (`validate_baseline_is_serial`,
///     #105 H-B) and is validated against this same list, so a list omitting `serial` cannot
///     describe a runnable track — it would die later, at the baseline leg, saying something less
///     useful.
///
/// Pure and total over its input: no filesystem, no GPU, no contract file — the whole truth table
/// is unit-testable.
pub fn resolve_allowed_modes(declared: Option<&[String]>) -> Result<Vec<String>, String> {
    let Some(declared) = declared else {
        return Ok(DEFAULT_ALLOWED_MODES
            .iter()
            .map(|m| m.to_string())
            .collect());
    };
    if declared.is_empty() {
        return Err(
            "--contract declares an EMPTY allowed_modes list: a track that declares the field must \
             say which modes it admits (an empty list admits nothing, not even the pinned serial \
             baseline) — remove the field to accept the default, or list the modes"
                .to_string(),
        );
    }
    let mut resolved: Vec<String> = Vec::with_capacity(declared.len());
    for mode in declared {
        if mode == SPEC_MODE_DSPARK {
            return Err(format!(
                "--contract declares allowed_modes entry {mode:?}, which is RESERVED: no dspark \
                 module has landed in either engine (docs/spec-config-design.md §4), so benchd \
                 cannot admit a submission declaring it. Declarable modes are \
                 {DECLARABLE_MODES:?}"
            ));
        }
        if !DECLARABLE_MODES.contains(&mode.as_str()) {
            return Err(format!(
                "--contract declares allowed_modes entry {mode:?}, which is not a mode benchd \
                 knows: the declarable set is {DECLARABLE_MODES:?}. A track fixture may not invent \
                 a mode — refusing before any timed work"
            ));
        }
        if resolved.iter().any(|m| m == mode) {
            return Err(format!(
                "--contract declares allowed_modes entry {mode:?} more than once: the list is a \
                 SET, and a repeated entry is a fixture edit that went wrong — refusing rather \
                 than silently de-duplicating it"
            ));
        }
        resolved.push(mode.clone());
    }
    if !resolved.iter().any(|m| m == SPEC_MODE_SERIAL) {
        return Err(format!(
            "--contract declares allowed_modes {resolved:?}, which does not include \
             {SPEC_MODE_SERIAL:?}: the baseline leg is PINNED serial (#105 H-B, the serial \
             denominator is not CLI-steerable) and is validated against this same list, so a list \
             without {SPEC_MODE_SERIAL:?} cannot describe a runnable track"
        ));
    }
    Ok(resolved)
}

/// **David ruling (2026-08-26)** — THE track mode fence, contract-driven: resolve the track's
/// allowed-modes list ([`resolve_allowed_modes`]) and validate BOTH declared leg specs against it.
///
/// One function so there is one place the list is resolved and one place both legs are checked —
/// the shape the pre-ruling code had at the CLI boundary, moved to where the contract is readable.
/// Returns the RESOLVED list so the caller can seal/report which vocabulary this run was admitted
/// under, rather than re-deriving it.
///
/// Called PRE-GPU, from `execute_measure_job`, immediately after `Contract::parse` — see the call
/// site for why the check had to MOVE from CLI-parse time (the contract is not read there, which is
/// exactly why the override never existed).
pub fn enforce_track_allowed_modes(
    candidate_spec: &SpecConfig,
    baseline_spec: &SpecConfig,
    declared: Option<&[String]>,
) -> Result<Vec<String>, String> {
    let allowed = resolve_allowed_modes(declared)?;
    let source = if declared.is_some() {
        "the --contract fixture's allowed_modes"
    } else {
        "benchd's DEFAULT_ALLOWED_MODES (the --contract fixture declares no allowed_modes)"
    };
    let refs: Vec<&str> = allowed.iter().map(String::as_str).collect();
    for (leg, spec) in [("candidate", candidate_spec), ("baseline", baseline_spec)] {
        validate_spec_mode_allowed(spec, &refs)
            .map_err(|e| format!("{leg} leg: {e} — the list came from {source}"))?;
    }
    Ok(allowed)
}

/// **David ruling (2026-08-26)** — is `mode` runnable in the BATCHED COHORT regime?
///
/// The ONE exhaustive statement of a fact about the ENGINE, kept here because benchd is the FINAL
/// validator and a refusal it can make PRE-GPU costs no gated box time, where the engine's own
/// refusal costs a whole spawn per leg per pair:
///
/// * `serial` / `mtp` — cohort-capable. The batch-8 cohort driver runs them.
/// * `dflash` — **SINGLE-STREAM ONLY**. The engine's cohort driver refuses `dflash` BY NAME
///   (`Sources/MLXFastHarness/Gemma4RuntimeCohortDriver.swift`, "this engine runs single-stream
///   only"), and the gemma4 `benchmark.json` describes the DFlash arm as single-stream in the
///   participant-facing text. This is a property of the mode, not a benchd preference.
/// * anything else — FAIL-CLOSED. An unknown mode is not assumed cohort-capable.
///
/// Consumed by [`effective_candidate_regime`]: a single-stream-only mode is NOT upgraded to the
/// fixture's declared cohort width, it keeps its spec-derived single-stream regime, and the regime
/// it actually ran is SEALED in `results.timed_mode`. Nothing is silent: the overlay's §5 series
/// fence already refuses to mix a `free_run_v1_1` file with a `batched_free_run_v1_2_b8` one, so
/// the two regimes stay separate values by construction rather than by convention.
pub fn mode_is_cohort_capable(mode: &str) -> bool {
    match mode {
        SPEC_MODE_SERIAL | SPEC_MODE_MTP => true,
        SPEC_MODE_DFLASH => false,
        _ => false,
    }
}

/// Spec re-home of the anti-DDoS 32 cap (`docs/spec-config-design.md` step 4): the cap is a
/// bounds-check on the MODULE's `mtp.depth` field, NOT a top-level benchmarker flag. A non-mtp spec
/// (serial/dflash/dspark) has no `mtp.depth`, so the cap does not apply. On the official/scored path
/// `cap` is the readonly constant 32 (env ignored, submission-proof); `--local-dev` may raise it.
pub fn validate_spec_capped(spec: &SpecConfig, cap: usize) -> Result<(), String> {
    // An engine-decides mtp block (`depth` absent — David ruling 2026-08-27) has no requested depth
    // to bound; the engine's own envelope clamp governs, and the echo reports what it resolved.
    if let Some(depth) = spec.mtp.as_ref().and_then(|m| m.depth) {
        if depth as usize > cap {
            return Err(format!(
                "spec mtp.depth {depth} exceeds the maximum draft depth cap {cap} (defensive bound \
                 mirroring the engine's {MAX_DRAFT_DEPTH_ENV}); on the official/scored path the cap \
                 is the readonly constant {DEFAULT_MAX_DRAFT_DEPTH_CAP} and the env override is \
                 ignored"
            ));
        }
    }
    Ok(())
}

/// Depth-0-via-serial-mode reachability (`docs/spec-config-design.md` step 4): candidate validation
/// keys on the spec's MODE being in the track's allowed-modes list — NOT on a depth-int floor. The
/// serial baseline is `{"mode":"serial"}` (no depth field), so depth-0 stops being a candidate depth
/// and the old ">= 2" straggler dissolves. A mode outside the allowed list REJECTS before any GPU
/// work. This is the CLI/track ALLOWED-LIST gate only; the ENGINE-runnable-mode gate (against the
/// hello's `spec_modes`) is enforced at the runner seam ([`bench_runner`]) before the timed seed
/// forward, not here.
///
/// Also fail-closed on internal shape via [`validate_spec_module_coherent`]: exactly the ONE module
/// block matching the mode may be present (CROSS-MODULE keys reject), and an `mtp` mode must carry a
/// depth `>= 1` (an `mtp(0)` candidate is the serial control, not a candidate — it rejects).
pub fn validate_spec_mode_allowed(spec: &SpecConfig, allowed: &[&str]) -> Result<(), String> {
    if !allowed.contains(&spec.mode.as_str()) {
        return Err(format!(
            "spec mode {:?} is not in the track's allowed-modes list {allowed:?} (a submission \
             declaring a mode outside the allowed list is rejected before any timed work)",
            spec.mode
        ));
    }
    validate_spec_module_coherent(spec)
}

/// Medium (#105) — the per-module COHERENCE gate. Exactly the ONE module block matching the spec's
/// mode may be present; ANY other block is a CROSS-MODULE key and REJECTS (e.g. `{"mode":"serial",
/// "mtp":{…}}` or `{"mode":"mtp","mtp":{…},"dflash":{…}}`). Fail-closed shape rules:
/// - `serial`: NO module block (serial has no drafter/depth);
/// - `mtp`: the `mtp` block present with `depth >= 1` (an `mtp(0)` is the serial control, not a
///   candidate — it rejects), and no `dflash`/`dspark` block;
/// - `dflash`/`dspark`: their own block present, and no other module block.
///
/// This runs at the CLI boundary (both legs) so a cross-module or degenerate spec dies pre-GPU. The
/// wire envelope stays CLOSED (`deny_unknown_fields`) for UNKNOWN keys; this gate adds the semantic
/// mode↔block coherence the closed envelope alone cannot express (both blocks are optional there).
pub fn validate_spec_module_coherent(spec: &SpecConfig) -> Result<(), String> {
    // The set of module blocks that must be ABSENT for this mode (a present one is a cross-module key).
    let (require_mtp, forbid): (bool, &[(&str, bool)]) = match spec.mode.as_str() {
        SPEC_MODE_SERIAL => (
            false,
            &[
                ("mtp", spec.mtp.is_some()),
                ("dflash", spec.dflash.is_some()),
                ("dspark", spec.dspark.is_some()),
            ],
        ),
        SPEC_MODE_MTP => (
            true,
            &[
                ("dflash", spec.dflash.is_some()),
                ("dspark", spec.dspark.is_some()),
            ],
        ),
        "dflash" => (
            false,
            &[
                ("mtp", spec.mtp.is_some()),
                ("dspark", spec.dspark.is_some()),
            ],
        ),
        "dspark" => (
            false,
            &[
                ("mtp", spec.mtp.is_some()),
                ("dflash", spec.dflash.is_some()),
            ],
        ),
        _ => (false, &[]),
    };
    for (name, present) in forbid.iter().copied() {
        if present {
            return Err(format!(
                "spec mode {:?} must not carry a {name:?} module block (cross-module key — exactly \
                 the ONE block matching the mode may be present)",
                spec.mode
            ));
        }
    }
    if require_mtp {
        match &spec.mtp {
            None => {
                return Err(
                    "spec mode \"mtp\" must carry its mtp module block (e.g. {\"mode\":\"mtp\",\
                     \"mtp\":{\"depth\":2}})"
                        .to_string(),
                );
            }
            Some(mtp) if mtp.depth == Some(0) => {
                return Err(
                    "spec mode \"mtp\" with mtp.depth 0 is not a candidate: depth 0 is the serial \
                     control ({\"mode\":\"serial\"}), so an mtp(0) candidate is rejected"
                        .to_string(),
                );
            }
            // A present block with an ABSENT depth is the engine-decides form (David ruling
            // 2026-08-27): the module resolves its own depth and the echo reports it.
            Some(_) => {}
        }
    }
    Ok(())
}

/// Parse a `--candidate-spec`/`--baseline-spec` JSON override into a [`SpecConfig`], FAIL-CLOSED on
/// malformed JSON or an unknown key (the spec envelope is closed). Recorded with
/// `spec_source: "cli-override"` in provenance.
pub fn parse_spec_override(json: &str) -> Result<SpecConfig, String> {
    serde_json::from_str(json).map_err(|e| format!("spec JSON parse failed: {e}"))
}

/// #105 (Engine-can't-speculate-on-TF) — the SERIAL spec: what a leg runs when nothing speculates.
/// Model-2 (Option A) is TEACHER-FORCED — benchd feeds each expected token, so an mtp drafter can
/// never gain time in that window and TF legs score `mode=serial` only. The candidate's DECLARED spec
/// (which may be mtp) is preserved separately as `results.candidate_spec` provenance; it is the spec
/// the candidate WOULD run in the v1.1 free-run series, never what a teacher-forced window measures.
///
/// This is the spec the FREE-RUN serial control requests on the wire. A TF leg requests NOTHING —
/// see [`requested_wire_spec`] and the coordinator ruling recorded there.
pub fn timed_decode_wire_spec() -> SpecConfig {
    SpecConfig::serial()
}

/// **Coordinator ruling (#109, leg B) — TF legs send NO spec on the wire.** The spec a leg REQUESTS
/// on its timed window, by regime; the wire-surface counterpart of [`leg_spawn_args`], and keyed off
/// the same fact, because the two surfaces are one decision:
///
/// * **teacher-forced ⇒ `None`.** A TF leg is spawned GATE-OFF (no `--speculative-protocol v1.1`),
///   and a gate-off worker speaks strict v1: it REJECTS any `spec` on the wire at the session's
///   spec guard, and its teacher-forced request kinds run serial forwards unconditionally (both
///   proven live in window 1, legs A1a/A1b). Requesting a spec on a gate-off leg is therefore
///   self-contradictory — benchd would be asking for an echo the worker is gated out of producing,
///   and the runner's spec-never-ignored check would discard every TF session for the absence.
///   The gate-off spawn IS the proof of serial semantics; nothing needs to be asked for.
/// * **v1.1 free-run ⇒ `Some(wire_spec)`.** The gate is ON, the echo is available and REQUIRED, and
///   spec-never-ignored is enforced exactly as before — the candidate carries its declared
///   speculating spec, the control the depth-0 serial one.
pub fn requested_wire_spec(wire_spec: &SpecConfig, regime: LegRegime) -> Option<SpecConfig> {
    regime.is_free_run().then(|| wire_spec.clone())
}

/// **Coordinator ruling (#109, leg B)** — the PROVENANCE label for a leg's sealed `effective_spec`,
/// naming where the sealed value came from, since the two regimes now source it differently.
///
/// A FREE-RUN leg's effective spec is the engine's WIRE echo, captured and validated never-ignored.
pub const EFFECTIVE_SPEC_SOURCE_WIRE_ECHO: &str = "wire-effective-spec-echo";

/// **Coordinator ruling (#109, leg B)** — a TEACHER-FORCED leg's effective spec is DERIVED FROM THE
/// SPAWN SURFACE: the leg was spawned without the v1.1 gate, so the worker speaks strict v1 and its
/// teacher-forced window is serial unconditionally. There is no echo to seal and none is expected;
/// the serial regime is a fact about how the process was started, which benchd controls and records.
pub const EFFECTIVE_SPEC_SOURCE_GATE_OFF_V1_SPAWN: &str = "gate-off-v1-spawn";

/// #109 window-2 finding 3 — the COMPLETE option surface the engine's generic `runtime-worker` verb
/// accepts (`Sources/MLXFastRuntimeWorkerCLI/main.swift`, `requireOnly(values:)`; usage text mirrored
/// in the window artifact `engine-worker-option-surface.txt`). The verb exits 1 on the FIRST unknown
/// option, BEFORE the hello — so any flag benchd emits outside this set kills every leg pre-GPU with
/// "engine closed the stream before returning a response".
///
/// `--weights` is prepended by the transport ([`bench_runner::ChildStdioTransport::build_args`]);
/// benchd's per-leg extra args may only draw from the rest. This constant is the machine-checked
/// fence: [`leg_spawn_args`]'s output is asserted ⊆ this set in both regimes.
///
/// `--resource NAME=PATH` (runner contract §8.1, repeatable) is the RESOURCE PASSTHROUGH
/// the `--engine-resource` CLI flag builds — see [`crate::engine_resource`].
pub const RUNTIME_WORKER_ACCEPTED_FLAGS: [&str; 5] = [
    "--weights",
    "--mtp-head",
    DFLASH_HEAD_FLAG,
    SPECULATIVE_PROTOCOL_FLAG,
    crate::engine_resource::RESOURCE_FLAG,
];

/// **David ruling (2026-08-26) — the PER-LEG DFlash head channel.**
///
/// Before this lane the engine's DFlash drafter loader resolved a BARE RELATIVE `./dflash-head`
/// against the WORKER PROCESS's current directory, with no argv flag and no env override
/// (`Sources/MLXFastHarness/Gemma4A4BAssistantHead.swift`,
/// `gemma4DFlashHeadDirectoryName = "dflash-head"` →
/// `loadGemma4DFlashHeadIfStaged` → `URL(fileURLWithPath: directoryName)`). benchd spawns BOTH legs
/// with no `current_dir` (`bench_runner::ChildStdioTransport::spawn_command`), so both legs inherit
/// benchd's own CWD — and therefore both would have loaded THE SAME `./dflash-head`, whichever
/// workspace benchd happened to be run from. The baseline leg would have been charged the
/// CANDIDATE's drafter, or vice versa, with nothing in the artifact able to show it.
///
/// The fix mirrors the mechanism the MTP head has used in production since R14/R15 rather than
/// inventing one: an explicit PER-LEG argv path, resolved parent-side, existence-checked parent-side,
/// and fenced by [`RUNTIME_WORKER_ACCEPTED_FLAGS`]. The rejected alternative was spawning each leg
/// with `current_dir` set to its own workspace; that is a WIDER change than it looks, because
/// `--weights`, `--mtp-head` and (on the unsandboxed spawn paths) the engine path are all passed to
/// the child VERBATIM and would silently re-resolve against the new child CWD if any of them were
/// relative. An argv channel moves exactly one path and leaves every other resolution where it was.
///
/// The engine side of this flag lands with it: `runtimeWorkerAcceptedOptionFlags` gains
/// `--dflash-head` and the loader takes the explicit path, fail-closed, exactly as the MTP head's
/// `resolveGemma4AssistantHeadStaging(explicitDirectoryPath:defaultDirectoryName:)` already does.
/// The two fences are a matched pair by construction: the engine's is the authority, and this
/// constant is benchd's machine-checked mirror of it.
pub const DFLASH_HEAD_FLAG: &str = "--dflash-head";

/// #109 window-2 finding 3 — build the per-leg spawn args for the generic `runtime-worker` verb.
/// `--mtp-head <dir>` on every leg (both legs load a head; residency charges the denominator), plus
/// `--dflash-head <dir>` when the run has a DFlash head to stage.
///
/// The `--mtp-depth <D>` this used to emit was REMOVED. It belongs to a DIFFERENT binary's verb
/// (`mlxfast-swift mtp-timed`), and the `runtime-worker` the measure-job actually spawns rejects it
/// unconditionally. #105 cycle-5 finding 4 tied the argv depth channel to the wire spec so the two
/// could not disagree; removing the argv channel outright is the cleaner end state of that finding —
/// the `decode_begin` / `free_decode_begin` `spec` is now the SINGLE channel through which a depth
/// reaches the engine, and the runner's spec-never-ignored echo check is its only guard.
///
/// `dflash_head_dir` is `Option` and the flag is OMITTED when it is `None`, deliberately: an
/// MTP-only track (qwen3.8, and gemma4's own mtp arm on a box with no drafter staged) must keep
/// spawning EXACTLY the argv it spawned before this lane, so the change cannot perturb a run that
/// has nothing to do with dflash. Absent flag ⇒ the engine's CWD default, i.e. the pre-ruling
/// behaviour, unchanged.
pub fn timed_leg_base_args(head_dir: &str, dflash_head_dir: Option<&str>) -> Vec<String> {
    let mut args = vec!["--mtp-head".to_string(), head_dir.to_string()];
    if let Some(dir) = dflash_head_dir {
        args.push(DFLASH_HEAD_FLAG.to_string());
        args.push(dir.to_string());
    }
    args
}

/// The engine gates every v1.1 wire field (`spec_modes`, `capabilities`, `head_provenance`,
/// `effective_spec`, `mlx_*`, `top_logit_margin`) behind this spawn flag — a worker spawned
/// WITHOUT it speaks strict v1 and rejects any `spec` on the wire. Free-run legs therefore
/// MUST carry it (both legs of a free-run pair: the depth-0 serial control speaks the same
/// v1.1 session shape), and teacher-forced legs MUST NOT (their gate-off spawn IS the
/// standing v1-compat proof — a TF leg that grew this flag would silently stop proving it).
pub const SPECULATIVE_PROTOCOL_FLAG: &str = "--speculative-protocol";
pub const SPECULATIVE_PROTOCOL_V1_1: &str = "v1.1";

pub fn leg_spawn_args(
    head_dir: &str,
    dflash_head_dir: Option<&str>,
    regime: LegRegime,
) -> Vec<String> {
    let mut args = timed_leg_base_args(head_dir, dflash_head_dir);
    args.extend(speculative_protocol_args(regime));
    args
}

/// THE ONE PLACE the v1.1 spawn gate is decided. Both spawn-argv builders — this crate's
/// [`leg_spawn_args`] (the measure-job legs) and `main.rs`'s `free_run_spawn_args` (the
/// iterate/official spawns) — take their flag from here, so the gate cannot be applied in one
/// builder and forgotten in the other.
pub fn speculative_protocol_args(regime: LegRegime) -> Vec<String> {
    if regime.is_free_run() {
        vec![
            SPECULATIVE_PROTOCOL_FLAG.to_string(),
            SPECULATIVE_PROTOCOL_V1_1.to_string(),
        ]
    } else {
        Vec::new()
    }
}

/// **David ruling (2026-08-26)** — build BOTH legs' spawn argv in ONE place: `(serial_control,
/// candidate)`.
///
/// [`leg_spawn_args`] can only build ONE leg and cannot know which head belongs to it, so the
/// SELECTION — pinned head to the serial control, BYO head to the candidate, for each of the two
/// head families — used to live as four hand-written field accesses at the `main.rs` call site.
/// That is exactly the shape of the bug this lane exists to close: a leg reading the OTHER leg's
/// head is a silent wrong measurement, not a crash, and a call site that hand-picks four fields is
/// where such a swap hides. Making the selection a function makes it TESTABLE — and
/// `main.rs`'s spawn wiring is explicitly `UNVERIFIED(measure-job)`, so anything left up there is
/// covered by nothing.
///
/// The serial control's REGIME is derived here too ([`serial_control_regime_for`], the Fable
/// same-series rule), so the one function that decides which head each leg gets also decides which
/// series each leg runs — the two facts that must agree between the legs, decided together.
///
/// CROSS-MODE SCOPING — the DFlash argv channel belongs to a `dflash` CANDIDATE and to nothing
/// else. `dflash_heads` is resolved from the environment (`QMTP_DFLASH_HEAD_DIR` /
/// `QMTP_CANDIDATE_DFLASH_HEAD_DIR`) MODE-INDEPENDENTLY, so a box that still has the export in its
/// shell from an earlier dflash run would otherwise hand `--dflash-head` to BOTH legs of an `mtp`
/// (or `serial`) pair. On any engine build that predates the flag, the `runtime-worker` verb exits
/// 1 on the first unknown option BEFORE the hello, which reaches benchd as the opaque *"engine
/// closed the stream before returning a response"* — one indistinguishable infra reject per leg
/// per pair, on a run that has nothing to do with dflash. `RUNTIME_WORKER_ACCEPTED_FLAGS` cannot
/// catch that: it mirrors the flag surface of the engine benchd was BUILT against, not the one on
/// the box. Scoping the emission to the declared candidate mode is what keeps a stale export
/// inert, and it is the same scope the env's own die-8 requirement uses
/// ([`enforce_dflash_head_present`]) — one mode fence, applied to both halves of the channel.
///
/// A `dflash` candidate is UNCHANGED: both legs still receive their own per-leg dir (the serial
/// control the PINNED drafter, the candidate its BYO one), because the control leg loads the
/// drafter too and its residency charges the denominator.
pub fn paired_leg_spawn_args(
    heads: &HeadDirs,
    dflash_heads: Option<&HeadDirs>,
    candidate_regime: LegRegime,
    candidate_mode: &str,
) -> (Vec<String>, Vec<String>) {
    let dflash_heads = dflash_heads.filter(|_| candidate_mode == SPEC_MODE_DFLASH);
    let serial = leg_spawn_args(
        &heads.head_dir,
        dflash_heads.map(|d| d.head_dir.as_str()),
        serial_control_regime_for(candidate_regime),
    );
    let candidate = leg_spawn_args(
        &heads.candidate_head_dir,
        dflash_heads.map(|d| d.candidate_head_dir.as_str()),
        candidate_regime,
    );
    (serial, candidate)
}

/// **David ruling (2026-08-26)** — a `dflash` candidate REQUIRES the per-leg DFlash head env
/// (`QMTP_DFLASH_HEAD_DIR`), die-8, PRE-GPU. The mirror of the standing `QMTP_HEAD_DIR` refusal for
/// the MTP head, and for the same reason: without it the run would fall back to the engine's CWD
/// default and BOTH legs would silently load whichever `./dflash-head` benchd's own working
/// directory happens to contain — which is precisely the per-leg confusion this lane exists to
/// close. Failing to configure the head must not degrade into measuring the wrong one.
///
/// Non-dflash candidates are UNAFFECTED: the env stays optional there (an mtp run on a box that
/// also has a drafter staged may pass it, and passing it costs only the drafter's residency on both
/// legs — symmetric, so it cannot bias the ratio).
///
/// Pure over its two inputs so the truth table needs no box.
pub fn enforce_dflash_head_present(
    candidate_mode: &str,
    dflash_head_dirs: Option<&HeadDirs>,
) -> Result<(), String> {
    if candidate_mode != SPEC_MODE_DFLASH || dflash_head_dirs.is_some() {
        return Ok(());
    }
    Err(format!(
        "the candidate spec declares mode {SPEC_MODE_DFLASH:?} but QMTP_DFLASH_HEAD_DIR is unset: \
         the pinned DFlash drafter is required for a dflash measure run (the serial control leg \
         loads it too, so residency charges the denominator), and without it BOTH legs would fall \
         back to the engine's CWD-relative ./dflash-head default and load the SAME drafter \
         directory regardless of which workspace they were spawned for — die 8. Stage the heads \
         with tools/stage-ranked-heads.sh, which exports QMTP_DFLASH_HEAD_DIR and \
         QMTP_CANDIDATE_DFLASH_HEAD_DIR"
    ))
}

/// #109 window-2 finding 3 — the PRODUCTION fence over the spawn surface: every flag-shaped token a
/// leg would be spawned with must be one the generic `runtime-worker` verb accepts
/// ([`RUNTIME_WORKER_ACCEPTED_FLAGS`]). The engine's own rejection of an unknown option is a bare
/// exit-1 before the hello, which reaches benchd as the opaque *"engine closed the stream before
/// returning a response"* — a whole window was spent isolating that message back to two flag names.
/// Checked BEFORE any worker is spawned so a future flag addition dies naming itself, at run start,
/// instead of one indistinguishable infra reject per leg per pair.
pub fn validate_spawn_argv(extra_args: &[String]) -> Result<(), String> {
    for flag in extra_args.iter().filter(|a| a.starts_with("--")) {
        if !RUNTIME_WORKER_ACCEPTED_FLAGS.contains(&flag.as_str()) {
            return Err(format!(
                "spawn argv carries {flag}, which the engine's generic `runtime-worker` verb does \
                 not accept (it takes exactly {RUNTIME_WORKER_ACCEPTED_FLAGS:?} and exits 1 on the \
                 first unknown option, BEFORE the hello — every leg would die pre-GPU as a protocol \
                 violation). #109 window-2 finding 3"
            ));
        }
    }
    Ok(())
}

/// #109 window-2 finding 3 — every FLAG-shaped token in a built spawn argv (a leading `--`), so the
/// argv-surface test observes the REAL argv rather than re-deriving the answer it is checking.
/// Test-only: production has no reason to introspect its own argv.
#[cfg(test)]
fn flags_in_args(args: &[String]) -> Vec<&str> {
    args.iter()
        .map(String::as_str)
        .filter(|a| a.starts_with("--"))
        .collect()
}

/// #105 (Engine-can't-speculate-on-TF) — the SEAL GUARD: refuse to record an mtp regime that did not
/// run. TF legs score `mode=serial` only, and this is the machine-checked half of that rule.
///
/// **Coordinator ruling (#109, leg B) — the guard INVERTED, and got stronger.** It used to take a
/// PRESENT echo and refuse a non-serial mode. Under the ruling a TF leg requests no spec
/// ([`requested_wire_spec`]) and its worker is spawned gate-off, so:
///
/// * an ABSENT echo is the EXPECTED state — the serial regime is sealed from the SPAWN SURFACE
///   ([`EFFECTIVE_SPEC_SOURCE_GATE_OFF_V1_SPAWN`]), which benchd controls, rather than from an engine
///   self-description it has to trust;
/// * a PRESENT echo of ANY mode is now the ANOMALY and is REFUSED fail-closed. A gate-off worker is
///   gated out of emitting `effective_spec` at all, so an echo appearing on a TF leg means the leg
///   was not the gate-off v1 process benchd believes it spawned — a wrong binary, a wrong spawn, or a
///   forged line. That refusal is the new tamper check, and it is strictly harder to satisfy than the
///   old mode comparison: a forger cannot pass by echoing `serial`, because ANY echo rejects.
pub fn tf_regime_is_serial(wire_effective_spec: Option<&SpecConfig>) -> Result<(), String> {
    if let Some(echo) = wire_effective_spec {
        return Err(format!(
            "teacher-forced leg carries an engine-echoed effective_spec (mode {:?}), but a TF leg is \
             spawned WITHOUT {SPECULATIVE_PROTOCOL_FLAG} {SPECULATIVE_PROTOCOL_V1_1} and requests no \
             spec, so a conformant gate-off worker cannot have produced one — refusing the leg \
             fail-closed rather than sealing a regime this process could not have reported. \
             {TF_DOWNGRADE_NOTE}",
            echo.mode
        ));
    }
    Ok(())
}

/// **#109 W3 finding 5** — the HEAD-IDENTITY counterpart of [`tf_regime_is_serial`], on the same
/// gate-off spawn surface and by the same logic.
///
/// The engine gates `head_provenance` behind [`SPECULATIVE_PROTOCOL_FLAG`] exactly as it gates
/// `effective_spec` ([`leg_spawn_args`] and the comment on the flag itself). A TF leg is spawned
/// GATE-OFF, so its hello CANNOT carry the object — window 3's pre-window sanity re-proved it
/// twenty minutes before leg B ran (gate-off hello: `head_provenance` ABSENT; gate-on: present).
/// So:
///
/// * an ABSENT `head_provenance` is the EXPECTED state on a TF leg, and requiring one there is
///   unsatisfiable BY CONSTRUCTION — that requirement is what blocked leg B for a whole window;
/// * a PRESENT `head_provenance` on a TF leg is the ANOMALY and is REFUSED fail-closed, for the
///   identical reason [`tf_regime_is_serial`] refuses a present echo: the leg was not the gate-off
///   v1 process benchd believes it spawned.
///
/// Head identity is a fact of the regime that HAS a drafting head. A TF leg proven serial by
/// [`tf_regime_is_serial`] has no drafting head to identify, and seals none.
pub fn tf_hello_carries_no_head_provenance(
    wire_head_provenance: Option<&bench_protocol::HeadProvenance>,
) -> Result<(), String> {
    if let Some(hp) = wire_head_provenance {
        return Err(format!(
            "teacher-forced leg carries a hello head_provenance (sha256 {:?}, {} bytes), but a TF \
             leg is spawned WITHOUT {SPECULATIVE_PROTOCOL_FLAG} {SPECULATIVE_PROTOCOL_V1_1} and the \
             engine gates head_provenance behind that flag, so a conformant gate-off worker cannot \
             have produced one — refusing the leg fail-closed rather than sealing a head identity \
             this process could not have reported. {TF_DOWNGRADE_NOTE}",
            hp.sha256, hp.bytes
        ));
    }
    Ok(())
}

/// #105 cycle-5 finding 5 — the POINTER a reader needs when they find an mtp candidate scored as
/// serial. The downgrade itself is sound design (teacher forcing cannot let a drafter speculate);
/// leaving it UNDOCUMENTED was the finding — a run declaring `{"mode":"mtp"}` came back sealed
/// `mode=serial` with nothing on the artifact saying why, or where mtp scoring actually lives.
/// Emitted in the [`RejectClass::NonSerialTfRegime`] error and sealed as `tf_downgrade_note`.
pub const TF_DOWNGRADE_NOTE: &str =
    "mtp scoring requires v1.1 free-run; TF leg downgraded to serial";

/// #105 cycle-5 finding 5 — the sealed `tf_downgrade_note`, emitted EXACTLY when the candidate's
/// DECLARED regime differs from the regime the timed window actually ran
/// ([`timed_decode_wire_spec`]) — i.e. when a downgrade really happened. A run whose candidate
/// declared serial was not downgraded and seals no note (the seal states a fact about this run, it
/// is not boilerplate).
pub fn tf_downgrade_note(declared_candidate_spec: &SpecConfig) -> Option<&'static str> {
    (declared_candidate_spec.mode != timed_decode_wire_spec().mode).then_some(TF_DOWNGRADE_NOTE)
}

/// W3 — the FREE-RUN counterpart of [`tf_regime_is_serial`]: the seal guard for a v1.1 CANDIDATE
/// leg. A free-run candidate leg exists to let the engine's speculation show up on the clock, so its
/// echoed effective regime MUST be a speculating one; a `serial` echo means the engine ran the
/// free-run window WITHOUT its drafter, which would seal a v1.1 number that is really a serial
/// free-run and silently flatter (or deflate) the candidate. Refused fail-closed — benchd seals the
/// regime the engine echoed, and refuses to file a non-speculative echo under the speculating series.
///
/// **Fable ruling (same-series serial control) — SCOPE.** This rule is now CANDIDATE-LEG ONLY. The
/// serial control also free-runs ([`serial_control_regime_for`]), and it free-runs at DEPTH 0, so it
/// echoes `serial` BY DESIGN; its mirror guard is
/// [`free_run_serial_control_is_non_speculating`]. Applying this predicate to the control leg would
/// refuse the very shape the ruling requires.
pub fn free_run_regime_is_speculative(effective_spec: &SpecConfig) -> Result<(), String> {
    if effective_spec.mode == SPEC_MODE_SERIAL {
        return Err(format!(
            "free-run CANDIDATE leg echoed effective_spec mode {:?}: the v1.1 free-run window is the \
             SCORED speculating regime, so a serial echo means the drafter did not run — refusing to \
             seal a non-speculative measurement under the free-run series",
            effective_spec.mode
        ));
    }
    Ok(())
}

/// **Fable ruling (same-series serial control)** — the seal guard for the free-run SERIAL CONTROL
/// leg, the exact mirror of [`free_run_regime_is_speculative`]. The control is the ratio's
/// DENOMINATOR (serial = 1.0) measured in the free-run series: same verb, same N, same clock as the
/// candidate, but with NO drafter. So its echoed effective regime MUST be `serial` — a speculating
/// echo means the "control" drafted, which would deflate the published speedup by inflating the
/// denominator's speed, exactly the CLI-steerable-denominator failure [`validate_baseline_is_serial`]
/// exists to prevent, arriving through the engine echo instead of the CLI.
pub fn free_run_serial_control_is_non_speculating(
    effective_spec: &SpecConfig,
) -> Result<(), String> {
    if effective_spec.mode != SPEC_MODE_SERIAL {
        return Err(format!(
            "free-run SERIAL CONTROL leg echoed effective_spec mode {:?} (non-serial): the control is \
             the free-run series' DENOMINATOR and must run at depth 0 with no drafter — a speculating \
             control would seal a denominator that is not the serial anchor",
            effective_spec.mode
        ));
    }
    Ok(())
}

/// **Fable ruling (same-series serial control)** — the ACCEPTANCE-HISTOGRAM assertion on the
/// free-run serial control leg. A depth-0 free-run window commits exactly one token per verify round
/// (the engine's existing non-speculating path), so its `acceptance_lengths` MUST be `[1] * N`.
///
/// This is the second, independent channel on the same fact the effective-spec echo carries: the
/// echo is what the engine SAYS it ran, the histogram is what it DEMONSTRABLY did — and unlike the
/// echo, the histogram is already cross-checked by the §2.6 triple (it sums to N and its length is
/// pinned by `completed_work`), so it cannot be doctored to hide a drafting control. Any round
/// committing more than one token means the control speculated; the leg fails.
pub fn free_run_serial_control_histogram_is_unit(
    acceptance_lengths: &[u32],
    n: usize,
) -> Result<(), String> {
    if acceptance_lengths.len() != n {
        return Err(format!(
            "free-run SERIAL CONTROL leg committed {} verify rounds over N={n} tokens: a depth-0 \
             control commits exactly one token per round, so the expected histogram is [1]*{n}",
            acceptance_lengths.len()
        ));
    }
    if let Some((i, len)) = acceptance_lengths
        .iter()
        .enumerate()
        .find(|(_, &len)| len != 1)
    {
        return Err(format!(
            "free-run SERIAL CONTROL leg acceptance_lengths[{i}] = {len} (expected 1): a round that \
             committed more than one token means the control SPECULATED — the denominator is not the \
             serial anchor and the leg fails"
        ));
    }
    Ok(())
}

/// #105 H-B — the BASELINE spec is PINNED to `mode == "serial"`. The serial control is the ratio's
/// denominator (serial = 1.0); allowing a `--baseline-spec` to steer it to a non-serial regime would
/// let a caller pick a slower/faster baseline and inflate the published speedup. A non-serial
/// baseline is a HARD ERROR before any GPU work — the serial denominator must not be CLI-steerable.
/// (`--candidate-spec` stays free; only the baseline is pinned.)
pub fn validate_baseline_is_serial(spec: &SpecConfig) -> Result<(), String> {
    if spec.mode != SPEC_MODE_SERIAL {
        return Err(format!(
            "--baseline-spec must be {{\"mode\":\"serial\"}}: the baseline is the serial DENOMINATOR \
             (serial = 1.0) and must not be CLI-steerable off serial (got mode {:?}) — a non-serial \
             baseline could swap the denominator and inflate the speedup",
            spec.mode
        ));
    }
    Ok(())
}

/// R13 — `--prompt-sha256` must be exactly 64 LOWERCASE hex characters.
pub fn validate_prompt_sha256(s: &str) -> Result<(), String> {
    let ok = s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    if ok {
        Ok(())
    } else {
        Err(format!(
            "--prompt-sha256 must be 64 lowercase hex characters, got {s:?}"
        ))
    }
}

/// R13 — `--target-id` must match `[A-Za-z0-9._-]+` (non-empty).
pub fn validate_target_id(s: &str) -> Result<(), String> {
    let ok = !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    if ok {
        Ok(())
    } else {
        Err(format!(
            "--target-id must match [A-Za-z0-9._-]+ (non-empty), got {s:?}"
        ))
    }
}

/// One pool prompt's timed workload, in either document shape `--golden` accepts.
///
/// THE WINDOW-20260819 FINDING. `measure-job` used to model exactly one `--golden` shape —
/// [`GoldenFixture`] — while ALSO requiring (R4) that the same bytes hash to exactly one
/// `timed_prompt_pool` pin. The live pool pins TEACHER-FORCING TAPES, not GoldenDocuments, so
/// the two conditions could not both hold: a real GoldenDocument loaded and then died-8 as
/// "no pinned per-prompt no-op reference", and a pinned pool object died-8 at load with
/// "unknown field `emitted_tokens`". Every invocation died pre-GPU. The fix is this enum: the
/// TAPE is a first-class golden input (`bench_core::tape`, schema derived from the reference
/// Swift decoder + the 8 live pinned objects), and the legacy GoldenDocument path is retained
/// for the fixtures and offline harnesses that already use it. Neither loader was loosened.
#[derive(Debug, Clone, PartialEq)]
pub enum TimedPrompt {
    /// The LIVE pool document: a teacher-forcing tape (`{seed_tokens, reference_seed_token,
    /// rows, …}`) — the shape every `timed_prompt_pool[].sha256` actually pins.
    Tape(TimedPromptTape),
    /// LEGACY: a `GoldenDocument` carrying a `benchmark` oracle block.
    Golden(GoldenFixture),
}

/// Kind label for a [`TimedPrompt::Tape`] — used in diagnostics and the preflight line.
pub const PROMPT_KIND_TAPE: &str = "timed-prompt-tape";
/// Kind label for a [`TimedPrompt::Golden`].
pub const PROMPT_KIND_GOLDEN: &str = "golden-document";

impl TimedPrompt {
    /// The prompt IDENTITY: sha256 of the exact `--golden` bytes (BIND BY BYTES). This is what
    /// R4 matches against `timed_prompt_pool[].sha256` and what `per_prompt` seals.
    pub fn sha256(&self) -> &str {
        match self {
            TimedPrompt::Tape(t) => &t.sha256,
            TimedPrompt::Golden(g) => &g.sha256,
        }
    }

    /// #112 (L3) — the BYTE COUNT of those same `--golden` bytes: the other half of the
    /// canonical-golden identity (`sha256` + `bytes`), matched against a pool entry's optional
    /// `bytes` by [`validate_goldens_pinned`].
    pub fn byte_len(&self) -> u64 {
        match self {
            TimedPrompt::Tape(t) => t.byte_len,
            TimedPrompt::Golden(g) => g.byte_len,
        }
    }

    /// Which document shape this prompt was loaded from (for honest diagnostics).
    pub fn kind(&self) -> &'static str {
        match self {
            TimedPrompt::Tape(_) => PROMPT_KIND_TAPE,
            TimedPrompt::Golden(_) => PROMPT_KIND_GOLDEN,
        }
    }
}

/// R13 — `--golden` is REPEATABLE (a `Vec`); a DUPLICATE DIGEST is FATAL (die-8 style, pre-GPU):
/// the same golden bytes (identical sha256) passed twice is a hard error, never silently deduped
/// (live wrapper validate_golden_set W:699). The per-golden LOOP is R7; this is only the dup guard.
pub fn check_golden_digests(digests: &[String]) -> Result<(), String> {
    let mut seen = std::collections::HashSet::new();
    for d in digests {
        if !seen.insert(d.as_str()) {
            return Err(format!(
                "duplicate --golden digest {d}: the same golden bytes were passed twice \
                 (dup-digest is fatal — die 8)"
            ));
        }
    }
    Ok(())
}

/// Medium (cycle-3) — every `--golden` must be PINNED: its sha256 resolves to EXACTLY ONE
/// `timed_prompt_pool` entry carrying a POSITIVE numeric `noop_decode_speedup`, else die-8 BEFORE
/// any GPU work. Mirrors the live wrapper's `noop_reference_for_golden` (W:663-679): zero matches,
/// more than one match (AMBIGUOUS), a non-number, or a non-positive value all REJECT. Real official
/// inputs are always pinned — an unpinned golden would otherwise measure to a results.json the
/// ranked jq rejects (a missing/non-positive per-prompt noop) only AFTER 45 minutes of gated box
/// time; catch it in a second.
///
/// #112 (L3) — the resolved entry's `bytes`, when it declares one, is now CHECKED against the
/// `--golden` file's own byte count: a canonical golden is pinned by `sha256` AND `bytes`
/// together, and the byte half was previously parsed past. Mismatch is die-8. An entry with no
/// `bytes` key keeps the sha-only pin, so offline fixtures that never carried it are unaffected.
///
/// `prompts` is the loaded `--golden` set (already dup-guarded by [`check_golden_digests`] on
/// their digests). The check is unchanged by the tape work — same exactly-one rule on the sha of
/// the RAW BYTES — but the ZERO-MATCH diagnostic is now KIND-AWARE: when the unpinned input is a
/// legacy `GoldenDocument`, the message says so and names the shape the live pool actually pins,
/// because "golden X has no pinned reference" alone sent the 20260819 window hunting for a wrong
/// fixture when the real cause was that no GoldenDocument can EVER match a tape pin.
pub fn validate_goldens_pinned(
    prompts: &[TimedPrompt],
    pool: &[PromptPoolEntry],
) -> Result<(), String> {
    for prompt in prompts {
        let sha = prompt.sha256();
        let matches: Vec<&PromptPoolEntry> = pool
            .iter()
            .filter(|e| e.sha256.eq_ignore_ascii_case(sha))
            .collect();
        match matches.as_slice() {
            [] => {
                let shape_hint = match prompt {
                    TimedPrompt::Golden(_) => format!(
                        " — this input loaded as a LEGACY {PROMPT_KIND_GOLDEN} (keys version/\
                         model_type/cases/correctness_gates/benchmark), while the live \
                         timed_prompt_pool pins {PROMPT_KIND_TAPE} objects (keys seed_tokens/\
                         reference_seed_token/rows/reference_self_consistent/emitted_tokens); a \
                         GoldenDocument's bytes can therefore never match a pool pin. Pass the \
                         PINNED POOL OBJECT itself as --golden"
                    ),
                    TimedPrompt::Tape(_) => String::new(),
                };
                return Err(format!(
                    "golden sha256 {sha} has no pinned per-prompt no-op reference in the contract's \
                     timed_prompt_pool — die 8 (real official inputs are always pinned; the pool \
                     objects and their noop_decode_speedup values must be pinned together){shape_hint}"
                ));
            }
            [entry] => {
                // #112 (L3) — the pin is `sha256` + `bytes`, so CHECK the byte half when the
                // entry declares it (the live pool entries all do). It was parsed past before,
                // leaving one half of a two-part identity unverified. Absent ⇒ the sha-only pin
                // stands, unchanged.
                if let Some(pinned_bytes) = entry.bytes {
                    let actual = prompt.byte_len();
                    if actual != pinned_bytes {
                        return Err(format!(
                            "golden sha256 {sha} resolves to a timed_prompt_pool entry pinning \
                             {pinned_bytes} bytes, but the --golden file is {actual} bytes — \
                             die 8 (a canonical golden is pinned by sha256 AND bytes; the two \
                             halves must agree)"
                        ));
                    }
                }
                match entry.noop_decode_speedup {
                    Some(n) if n.is_finite() && n > 0.0 => {}
                    _ => {
                        return Err(format!(
                            "golden sha256 {sha} resolves to a timed_prompt_pool entry whose \
                             noop_decode_speedup is missing/non-numeric/non-positive — die 8 (a \
                             pinned no-op reference must be a positive number)"
                        ));
                    }
                }
            }
            _ => {
                return Err(format!(
                    "golden sha256 {sha} matches {} timed_prompt_pool entries (AMBIGUOUS) — die 8 \
                     (each golden must resolve to EXACTLY ONE pinned pool entry)",
                    matches.len()
                ));
            }
        }
    }
    Ok(())
}

/// The anti-lottery ≥N-DISTINCT COVERAGE gate (die-8, pre-GPU) — benchd is the FINAL validator.
///
/// The published ranked score is the MEDIAN over the pool of each prompt's RAW serial-relative
/// ratio-of-means, serial anchored at 1.0, no normalization
/// (`docs/measure-job-contract.md@fe2da64`, evicted — "Published score = median over the 8
/// prompts …", W:2207-2210; the all-8-median aggregation of
/// `docs/parity-completion-gate.md` §3). That median is only well-defined over the FULL DISTINCT
/// pinned pool: a run whose TIMED coverage is a SUBSET, carries a DUPLICATE (fewer distinct than
/// the pool), or a SUBSTITUTION (a timed prompt whose sha256 matches no pin) would publish a median
/// over a hand-picked support — the lottery this gate forbids. Per the track fixture's
/// `timed_prompt_pool_note`, the ≥N-distinct requirement is benchd's to enforce here; only the
/// `r2_path` distinctness/download is organizer-side.
///
/// The predicate is SET-EQUALITY of the timed prompts' sha256 with the pool's pinned sha256, plus a
/// no-duplicate check on the timed side. It runs at the SAME pre-GPU point, on the SAME
/// `&golden_fixtures` slice, and maps onto the SAME exit-8 path as [`validate_goldens_pinned`] /
/// [`check_golden_digests`]. It is deliberately STRONGER than [`validate_goldens_pinned`] — which
/// accepts a SUBSET, since each supplied golden pins individually — and re-asserts the duplicate /
/// substitution refusals as ONE coverage invariant, so the anti-lottery property holds even if an
/// upstream guard is later weakened. `N` is NOT hard-coded: it is the fixture's distinct
/// `timed_prompt_pool` cardinality (8 on the live 3.8 track), resolved from the `--contract` pin
/// authority — never a compiled-in constant.
///
/// Secret-tier: operates ONLY on sha256 pins (the fixture's committed PINS-ONLY identity) and the
/// loaded golden bytes' own sha256 — it carries no pool name, key, path, or content.
pub fn validate_timed_pool_coverage(
    prompts: &[TimedPrompt],
    pool: &[PromptPoolEntry],
) -> Result<(), String> {
    use std::collections::BTreeSet;

    // The DISTINCT pinned support. The pins ARE the fixture's authority for "the 8 distinct"; a pin
    // authority carrying a duplicate sha has no well-defined distinct pool (the same AMBIGUOUS
    // defect [`validate_goldens_pinned`] rejects per-golden, asserted here on the pool itself).
    let mut pinned: BTreeSet<String> = BTreeSet::new();
    for e in pool {
        if !pinned.insert(e.sha256.to_ascii_lowercase()) {
            return Err(format!(
                "timed_prompt_pool pins a DUPLICATE sha256 ({}): the pinned pool has no \
                 well-defined distinct support — die 8 (the pins define the distinct pool)",
                e.sha256.to_ascii_lowercase()
            ));
        }
    }
    if pinned.is_empty() {
        return Err(
            "timed_prompt_pool is empty: no pinned pool to cover — die 8 (a scoring run must time \
             the fixture-pinned pool)"
                .to_string(),
        );
    }

    // The DISTINCT timed support, refusing a DUPLICATE up front (fewer distinct than timed): a run
    // that times the same prompt twice covers fewer than the pool's distinct pins.
    let mut timed: BTreeSet<String> = BTreeSet::new();
    for p in prompts {
        let sha = p.sha256().to_ascii_lowercase();
        if !timed.insert(sha.clone()) {
            return Err(format!(
                "timed coverage repeats sha256 {sha}: a scoring run must time each pinned pool \
                 prompt EXACTLY ONCE ({} distinct timed < pool {} — a DUPLICATE, fewer than 8 \
                 distinct) — die 8",
                timed.len(),
                pinned.len()
            ));
        }
    }

    // EXACT coverage: the distinct timed support must EQUAL the distinct pinned support. This one
    // predicate refuses a SUBSET (a pin the run never timed), a SUBSTITUTION (a timed sha matching
    // no pin), and any wrong count at once.
    if timed != pinned {
        let missing: Vec<&String> = pinned.difference(&timed).collect();
        let extra: Vec<&String> = timed.difference(&pinned).collect();
        return Err(format!(
            "timed coverage is not EXACTLY the {} distinct fixture-pinned pool prompts (timed {} \
             distinct; {} pinned prompt(s) NOT timed [SUBSET]: {missing:?}; {} timed prompt(s) \
             match NO pin [SUBSTITUTION]: {extra:?}) — die 8 (a scoring run must time exactly the \
             pinned pool: no subset, no duplicate, no substitution — the published score is the \
             median over all {})",
            pinned.len(),
            timed.len(),
            missing.len(),
            extra.len(),
            pinned.len()
        ));
    }
    Ok(())
}

/// COHORT (batch-8 brief D2) — one sealed cohort-member record: which pinned pool prompt occupies
/// which slot of the batched window. The member list is the cohort half of the golden identity
/// discipline: a cohort golden is pinned per `(backend, batch_size, cohort composition, slot
/// order)` (D7), and this is the sealed statement of that composition and order. `bytes` is the
/// prompt's actual byte count, already verified equal to the pool pin.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CohortMember {
    /// This member's SLOT in the batched window (0-based). Slot order = POOL ORDER (D2, ruled):
    /// slot `i` runs pool entry `i`, so the composition is pinned by the fixture, never chosen at
    /// run time.
    pub slot_index: usize,
    /// The sha256 of the slot's `--golden` bytes (bind by bytes), equal to the pool pin.
    pub prompt_sha256: String,
    /// The slot's golden byte count, equal to the pool entry's pinned `bytes`.
    pub bytes: u64,
}

/// COHORT — the DERIVED cohort identity: sha256 over the ordered member pins
/// (`{slot}:{sha256}:{bytes}` lines). NOT the hash of any prompt's bytes — it names the
/// (composition, slot order) pair the window ran, so per-pair records can carry one identity for
/// the one measurement unit. Deterministic from the sealed member list, so any consumer can
/// recompute it.
pub fn cohort_sha256(members: &[CohortMember]) -> String {
    let mut lines = String::new();
    for m in members {
        lines.push_str(&format!(
            "{}:{}:{}\n",
            m.slot_index, m.prompt_sha256, m.bytes
        ));
    }
    bench_core::hash::sha256_hex(lines.as_bytes())
}

/// COHORT (batch-8 brief §4.5) — the cohort-membership gate (die-8, pre-GPU): the batched window's
/// composition is EXACTLY the fixture-pinned pool, in POOL ORDER, one slot per pin, each slot
/// pinned by `sha256` AND `bytes` TOGETHER. The cohort generalization of [`validate_goldens_pinned`]
/// + [`validate_timed_pool_coverage`], strictly stronger than both on the batched path:
///
/// - the cohort size must equal the DECLARED `scored_batch_size` AND the pool's cardinality —
///   the cohort IS the whole pool (D2: all 8 distinct prompts, concurrently, one window);
/// - SLOT ORDER = POOL ORDER, checked slot by slot (a permuted cohort is a different pinned
///   identity, refused — the permutation tripwire is a separate report-only concept, D7);
/// - every pool entry must pin `bytes` (the sha-only legacy pin is NOT accepted for a cohort:
///   an unpinned byte half would leave one half of the member identity unverified), and the
///   loaded golden's byte count must equal it;
/// - duplicates are structurally excluded by pool-order equality over a duplicate-free pool.
///
/// Returns the SEALED member list `per_cohort[].members` is built from.
pub fn validate_cohort_membership(
    prompts: &[TimedPrompt],
    pool: &[PromptPoolEntry],
    batch_size: u32,
) -> Result<Vec<CohortMember>, String> {
    // The pool must be a duplicate-free pin authority (same defect class as the coverage gate).
    let mut seen = std::collections::BTreeSet::new();
    for e in pool {
        if !seen.insert(e.sha256.to_ascii_lowercase()) {
            return Err(format!(
                "timed_prompt_pool pins a DUPLICATE sha256 ({}): the pinned pool has no \
                 well-defined cohort composition — die 8",
                e.sha256.to_ascii_lowercase()
            ));
        }
    }
    if pool.len() != batch_size as usize {
        return Err(format!(
            "timed_prompt_pool pins {} prompts but the declared scored_batch_size is {batch_size}: \
             the cohort IS the whole pool (D2 — all pool prompts run concurrently, one per slot), \
             so the two cardinalities must be equal — die 8",
            pool.len()
        ));
    }
    if prompts.len() != batch_size as usize {
        return Err(format!(
            "cohort carries {} goldens but the declared scored_batch_size is {batch_size}: a \
             batched window times EXACTLY one slot per declared stream — die 8",
            prompts.len()
        ));
    }
    let mut members = Vec::with_capacity(prompts.len());
    for (slot_index, (prompt, entry)) in prompts.iter().zip(pool.iter()).enumerate() {
        let sha = prompt.sha256().to_ascii_lowercase();
        if !entry.sha256.eq_ignore_ascii_case(&sha) {
            return Err(format!(
                "cohort slot {slot_index} carries golden sha256 {sha} but pool entry {slot_index} \
                 pins {} — SLOT ORDER IS POOL ORDER (D2, ruled): the cohort composition and order \
                 are pinned by the fixture, never chosen per run — die 8",
                entry.sha256.to_ascii_lowercase()
            ));
        }
        let Some(pinned_bytes) = entry.bytes else {
            return Err(format!(
                "pool entry {slot_index} (sha256 {sha}) pins no `bytes`: a cohort member is pinned \
                 by sha256 AND bytes together, and a sha-only pin leaves half the identity \
                 unverified — die 8 (add the byte pin to the fixture entry)",
            ));
        };
        let actual = prompt.byte_len();
        if actual != pinned_bytes {
            return Err(format!(
                "cohort slot {slot_index} golden (sha256 {sha}) is {actual} bytes but the pool \
                 entry pins {pinned_bytes} — die 8 (the two halves of the pin must agree)"
            ));
        }
        members.push(CohortMember {
            slot_index,
            prompt_sha256: sha,
            bytes: pinned_bytes,
        });
    }
    Ok(members)
}

/// COHORT (batch-8 brief D9) — resolve the regime the candidate leg will ACTUALLY run from the
/// spec-derived regime and the track fixture's declared `scored_batch_size`. The batch width is a
/// PINNED IDENTITY, so the fixture — never a CLI flag — is what selects the batched regime:
///
/// - no declared width ⇒ the spec-derived regime stands (single-stream, unchanged);
/// - the RULED B = 8 ⇒ the batched cohort regime, REQUIRING a speculating candidate: a serial
///   candidate at batch 8 makes both legs identical and the ratio 1.0 by construction (D0
///   alternative (c), rejected as a scored axis);
/// - any other width ⇒ refused. There is ONE ruled batch point and no sweep (D0/D8); a second
///   scored width would need its own regime variant, series tag, and calibration — by ruling, not
///   by a wider parameter.
///
/// **David ruling (2026-08-26) — the upgrade is now MODE-AWARE.** A candidate whose mode is not
/// cohort-capable ([`mode_is_cohort_capable`]) keeps its spec-derived SINGLE-STREAM regime even on
/// a fixture that pins a width, and `dflash` is the mode that makes this real: the engine's cohort
/// driver refuses `dflash` BY NAME, so upgrading a dflash candidate to B=8 could only ever produce
/// a wire refusal one spawn later — after the gated box time was already spent. The gemma4 fixture
/// pins `scored_batch_size: 8` for its MTP arm, and before this rule that pin made the whole track
/// structurally closed to dflash even once the mode fence admitted it.
///
/// This is NOT silent. The regime is SEALED (`results.timed_mode`), the caller reports the
/// divergence on stderr, and the overlay's §5 series fence already refuses to aggregate or compare
/// a `free_run_v1_1` file with a `batched_free_run_v1_2_b8` one — so the single-stream dflash arm
/// and the cohort mtp arm are SEPARATE values by construction, never two numbers pooled into one.
/// That separation is the mechanized form of the laguna dflash track's ruled
/// `"leaderboard": "separate … namespace, never mixed with the serial track"`.
///
/// No second flag is introduced to say whether the pinned width was applied: the caller answers
/// that by asking the RETURNED regime ([`LegRegime::scored_batch_point`]), which is also the value
/// every downstream decision already keys on. A parallel boolean could drift from the regime; the
/// regime cannot drift from itself.
pub fn effective_candidate_regime(
    candidate_mode: &str,
    spec_regime: LegRegime,
    scored_batch_size: Option<u32>,
) -> Result<LegRegime, String> {
    match scored_batch_size {
        // ABSENT declares nothing; STEP 3 of the constants→contract migration makes absence a
        // refusal at the resolver, not here.
        None => Ok(spec_regime),
        // 1 IS THE SINGLE-STREAM POINT (David 2026-09-15: batching is part of the configuration).
        // A fixture that states the width it measures at states `1` for the single-stream path —
        // the same declaration the batched path makes, with a different number — so `1` selects
        // the spec-derived single-stream regime rather than being read as an uncertified cohort
        // width. Only a width GREATER than 1 asks for a cohort.
        Some(1) => Ok(spec_regime),
        Some(width) => {
            // Orchestrator naming ruling — the width is DATA, certified through the ONE
            // exhaustive match ([`ScoredBatchPoint::certify`]): an uncertified width refuses
            // there, never reaching a regime. Certified FIRST, before the mode exemption below, so
            // a fixture that pins a nonsense width is refused for that reason on EVERY mode — a
            // single-stream-only candidate must not become the way an uncertified width goes
            // unnoticed.
            let point =
                ScoredBatchPoint::certify(width).map_err(|e| format!("--contract declares {e}"))?;
            if !spec_regime.is_free_run() {
                return Err(format!(
                    "--contract declares scored_batch_size {} but the candidate spec is serial: a \
                     serial candidate at batch {} makes both cohort legs identical and the ratio \
                     1.0 by construction (batch-8 brief D0 alt (c)) — there is no track to \
                     measure; declare a speculating candidate spec",
                    point.batch_size(),
                    point.batch_size(),
                ));
            }
            // David ruling (2026-08-26) — a SINGLE-STREAM-ONLY mode is not upgraded. It keeps the
            // free-run single-stream regime its spec derived, which is the only regime it can
            // physically run.
            if !mode_is_cohort_capable(candidate_mode) {
                return Ok(spec_regime);
            }
            Ok(LegRegime::BatchedFreeRunV1_2(point))
        }
    }
}

/// #142 — the CAPTURED engine-wire crosscheck, run AT MEASURE TIME (previously it existed only as
/// a `cargo test` and never ran under `benchd measure-job`). benchd re-verifies its embedded
/// captured engine-wire reference against the mirror-integrity reference sha256 AND re-parses every
/// line under its own CLOSED `WorkerResponse` (`deny_unknown_fields`). A disagreement — the
/// captured bytes no longer hash to the reference, or a line no longer parses under benchd's schema
/// — is a die-8 pre-GPU prereq failure, exactly like the other integrity prereqs.
///
/// Why this is not redundant with [`validate_goldens_pinned`]: that check binds the `--golden`
/// bytes to the CONTRACT's OWN `timed_prompt_pool` sha (the mirror pointer field is `serde`-ignored,
/// so the contract's self-declared pin is the only thing consulted). This crosscheck binds the run
/// to an INDEPENDENT captured reference both repos hold — the one the offline crosscheck test
/// pins — so a run cannot pass on a contract that merely agrees with itself.
///
/// Thin by design: the crosscheck BODY is [`bench_runner::verify_captured_engine_wire`] (shared with
/// the offline crosscheck test); this wrapper only re-labels its diagnostic as a die-8 prereq so the
/// caller maps it straight onto the exit-8 path.
pub fn crosscheck_captured_engine_wire(
    captured: &[u8],
    reference_sha256: &str,
) -> Result<(), String> {
    bench_runner::verify_captured_engine_wire(captured, reference_sha256)
        .map(|_| ())
        .map_err(|e| {
            format!("{e} — die 8 (the captured engine-wire crosscheck must hold at measure time)")
        })
}

// ---------------------------------------------------------------------------
// Pair ordering
// ---------------------------------------------------------------------------

/// Within-pair leg order. QMTP ALTERNATES so serial-first / mtp-first stays balanced across the
/// run's ACCEPTED pairs and ordering advantages neither side. The alternation is keyed on the
/// ACCEPTED-PAIR index (the count of pairs accepted so far), NOT the raw attempt index — so
/// rejected attempts in between can never skew the accepted pairs toward one order: even accepted
/// pair → `mtp-first`, odd → `serial-first`.
// UNVERIFIED(measure-job): the alternation rule is cited from
// `docs/measure-job-contract.md@7c6be14:132` (evicted at cd5782e; resolves via git — repinned
// from the unresolvable `@c44e526b`); the wrapper internals are un-mirrored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairOrder {
    MtpFirst,
    SerialFirst,
}

impl PairOrder {
    /// The alternation rule keyed on the ACCEPTED-PAIR index (accepted pair 0 → mtp-first, 1 →
    /// serial-first, …), so interleaved rejected attempts never skew the accepted pairs' balance.
    pub fn for_accepted_index(accepted_pair_index: usize) -> Self {
        if accepted_pair_index.is_multiple_of(2) {
            PairOrder::MtpFirst
        } else {
            PairOrder::SerialFirst
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            PairOrder::MtpFirst => "mtp-first",
            PairOrder::SerialFirst => "serial-first",
        }
    }
}

// ---------------------------------------------------------------------------
// Rejection classes (finding 5 — matched on the TYPED variant, not error prose)
// ---------------------------------------------------------------------------

/// Why a leg (and therefore its pair) was rejected — RESTRICTED to what THIS component can
/// actually produce (`docs/measure-job-contract.md@7c6be14:268-273`, evicted — resolves via
/// git; repinned from the unresolvable `@c44e526b`): token-mismatch(parity),
/// implausible s-per-tok, row-accounting, the thermal-gate stall/ceiling timeout, and a
/// spawn/protocol `Infra` catch-all.
///
/// Finding R19 (live wrapper W:1718-1734, 2005-2032) — the pair loop gives ONE gated retry
/// (`MAX_ATTEMPTS=2`), and ANY pair still failing after it FOLDS INTO die-5 (candidate rejected,
/// exit 5). NO class gets a distinct mid-pair exit: the thermal timeout does NOT die exit-2 — it is
/// a retryable pair failure that folds into die-5 like the rest.
///
/// #108 (L1) — the retry is class-agnostic EXCEPT for [`RejectClass::is_deterministic`] classes,
/// which are TERMINAL: their second attempt is guaranteed to reach the identical verdict, so the
/// retry only spends a spawn + cool gate before the same die-5. The DESTINATION is unchanged (still
/// die-5, still this class recorded) — only the wasted attempt is skipped. Apart from that one
/// predicate this enum remains a pure PROVENANCE label; it drives no hard-die-vs-pair-failure
/// decision (those methods are gone with finding R8).
///
/// The contract taxonomy ALSO names `throttle` (a steady loaded sample below the leg's clock
/// floor), `insufficient-telemetry` (`< MIN_LOADED_SAMPLES`), and the `4×-p50` single-block
/// stall. NONE are emittable here — they need a per-block sampled telemetry stream this component
/// does not own — so they are deliberately NOT modelled as (dead) variants; the enum names only
/// what the code can produce, and those classes return when that telemetry stream is real.
// UNVERIFIED(measure-job): the not-yet-emittable contract classes — throttle, insufficient-telemetry,
// 4x-p50-stall — are deferred until a per-block sampled telemetry stream exists (no block stream here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectClass {
    /// `all_tokens_matched != true` — the parity failure. Finding R19 — retryable ONCE like every
    /// other class (the live wrapper retries parity too); a persistent parity fail folds into die-5.
    TokenMismatchParity,
    /// An implausible seconds-per-token (non-finite / ≤ 0 / below the abs floor). Retryable once.
    ImplausibleSpt,
    /// A row-accounting mismatch (the completed-work barrier, allocator-drain fault). Retryable once.
    RowAccounting,
    /// The thermal cool-gate stall/ceiling timeout (the gate exhausted its 900 s budget). Finding
    /// R19 — a RETRYABLE pair failure that FOLDS INTO die-5, NOT a distinct exit-2 hard die.
    GateThermal,
    /// A spawn/protocol infra fault (a worker that never came up). Finding R19 — also a retryable
    /// pair failure folding into die-5; no longer a hard die with a distinct nonzero infra exit.
    Infra,
    /// #105 (Engine-can't-speculate-on-TF) — a TEACHER-FORCED leg carried a wire `effective_spec`
    /// echo. TF legs score `mode=serial` only, and under the coordinator's leg-B ruling they are
    /// spawned gate-off and request no spec, so a conformant worker emits NO echo: any echo means the
    /// leg was not the gate-off v1 process benchd spawned, and the seal refuses it fail-closed
    /// ([`tf_regime_is_serial`]). The label is kept (the finding it names is the same one — an mtp
    /// regime must never be sealed on a window that could not have run it); only the trigger widened
    /// from "non-serial echo" to "any echo". #109 W3 finding 5 widened it once more, to the OTHER
    /// v1.1-gated hello field on the same spawn surface: a TF leg whose hello carries
    /// `head_provenance` ([`tf_hello_carries_no_head_provenance`]) is the identical tamper case and
    /// files here. Folds into die-5 on persistence like the rest.
    NonSerialTfRegime,
    /// W3 — the v1.1 free-run candidate leg echoed a `serial` effective regime (the drafter did not
    /// run), so the measurement cannot be sealed under the free-run series
    /// ([`free_run_regime_is_speculative`]). Folds into die-5 on persistence like the rest.
    FreeRunRegimeNotSpeculative,
    /// Fable ruling (same-series serial control) — the MIRROR of
    /// [`RejectClass::FreeRunRegimeNotSpeculative`], on the other leg: the free-run SERIAL CONTROL
    /// speculated. Either it echoed a non-serial effective regime
    /// ([`free_run_serial_control_is_non_speculating`]) or it committed a round of more than one
    /// token ([`free_run_serial_control_histogram_is_unit`]). Its OWN label rather than reusing the
    /// candidate's: a drafting DENOMINATOR deflates the published speedup, the opposite direction of
    /// error from a non-drafting numerator, and an operator reading the reject record needs to know
    /// which side moved. Folds into die-5 on persistence like the rest.
    FreeRunSerialControlSpeculated,
    /// W3 — the v1.1 free-run leg breached the §2.6 consistency TRIPLE / §2.4 count invariants
    /// (doctored `acceptance_lengths`, under-reported `completed_work`, a token count != N). Its own
    /// provenance label rather than the `row-accounting` catch-all: this is the ACCOUNTING class
    /// that makes a doctored audit histogram falsifiable, and an operator reading a reject record
    /// needs to see WHICH accounting barrier fired.
    FreeRunConsistency,
    /// W3 — the §2.2 RunTimeout liveness bound tripped (`N × band-ceiling × margin`): the engine did
    /// not return N committed tokens inside its budget. The session is already discarded fail-closed
    /// at the session seam; here it is a retryable pair failure that folds into die-5, with its own
    /// label so a hung/looping engine is not filed under the `infra` catch-all.
    RunTimeout,
    /// #108 (M2) — the §2.2 RunTimeout budget could not be ARMED for this leg (`N × band-ceiling ×
    /// margin` was degenerate). Its own label rather than the `infra` catch-all, and distinct from
    /// [`RejectClass::RunTimeout`]: that one means the deadline FIRED, this one means there was no
    /// computable deadline to arm. The ceiling is calibration-derived (`serial_mean ×
    /// serial_band_high`), so an operator reading this reject record needs to be pointed at the
    /// calibration file, not at the engine.
    RunTimeoutBudgetInvalid,
    /// #108 (L1) — the engine did not ADVERTISE the v1.1 `free_decode` capability its
    /// `hello.capabilities` must carry before benchd will drive a free-run leg (PROTOCOL-v1.1 §2.1;
    /// [`RunnerError::CapabilityNotAdvertised`]). Its own label rather than the `infra` catch-all:
    /// an operator reading this reject record needs to see that the ENGINE BUILD cannot run this
    /// series at all — nothing about the box, the weights, or the thermal state is wrong.
    ///
    /// TERMINAL ([`RejectClass::is_deterministic`]): the hello handshake is a pure property of the
    /// binary under test, so the second attempt reads the identical capability list. Retrying spends
    /// a full spawn + cool gate to reach the same refusal.
    FreeRunCapabilityMissing,
    /// B2 — the leg's PREFILL WINDOW did not certify under the track's DECLARED regime
    /// (`bench_core::prefill_window::certify_prefill_window`): the window was not observed, was not
    /// measurable, or two measurements of it disagreed beyond the pinned tolerance. Its own label
    /// rather than the `row-accounting` catch-all, because an operator reading this reject record
    /// needs to see that the SCORED PREFILL HALF is the part that failed — not the decode window and
    /// not the box.
    ///
    /// ARMED BY THE REGIME ONLY. A track whose declared `prefill_gain_exponent` is `0.0` scores no
    /// prefill, certification is report-only there, and this class is UNREACHABLE — which is the
    /// state of `qwen3.8-27b-mtp-v1`, the track this branch serves
    /// (`prefill_certification_is_report_only_on_this_branchs_track`).
    PrefillWindowUncertified,
    /// (b) admission — a cohort stream exceeded the per-stream token tolerance: it diverged from the
    /// TRUSTED oracle's reference argmax on MORE than
    /// [`bench_core::constants::COHORT_TOKEN_TOLERANCE_PER_THOUSAND`] of every 1000 of its committed
    /// tokens (David's blanket-10% ruling). PER-STREAM: any single stream over the bar rejects the
    /// WHOLE run. Its own provenance label (never the `token-mismatch-parity` catch-all): this is a
    /// TOLERANCE verdict against a live trusted reference, not the old exact-match parity die, and an
    /// operator reading the reject record needs to see which bar the candidate failed.
    ///
    /// TERMINAL ([`RejectClass::is_deterministic`]): the candidate's committed tokens and the trusted
    /// reference argmax over the organizer's pinned weights are input-determined, so a re-measure
    /// re-derives the same tolerance verdict. NON-RETRYABLE — it folds into the pair rejection so
    /// `candidate_accepted` (min-pairs floor) goes false and the overlay fail-closed holds.
    CohortTokenTolerance,
    /// (b) admission — N2 INTEGRITY: the trusted oracle's echoed committed token did NOT equal the
    /// candidate's own committed journal at some stream × position, so the oracle replayed a DIFFERENT
    /// journal than the candidate emitted. A distinct HARD integrity error, never a tolerance
    /// decision: the tolerance verdict is only meaningful once the replay is proven to have used the
    /// candidate's real tokens. Its own label so an operator sees an integrity breach, not a mere
    /// tolerance miss.
    ///
    /// TERMINAL ([`RejectClass::is_deterministic`]): the echo mismatch is a fixed property of the
    /// candidate journal and the trusted replay, so a retry re-derives it. Folds into the pair
    /// rejection like the rest.
    CohortReplayIntegrity,
}

impl RejectClass {
    pub fn as_str(self) -> &'static str {
        match self {
            RejectClass::TokenMismatchParity => "token-mismatch-parity",
            RejectClass::ImplausibleSpt => "implausible-s-per-tok",
            RejectClass::RowAccounting => "row-accounting",
            RejectClass::GateThermal => "gate-thermal",
            RejectClass::Infra => "infra",
            RejectClass::NonSerialTfRegime => "non-serial-tf-regime",
            RejectClass::FreeRunRegimeNotSpeculative => "free-run-regime-not-speculative",
            RejectClass::FreeRunSerialControlSpeculated => "free-run-serial-control-speculated",
            RejectClass::FreeRunConsistency => "free-run-consistency",
            RejectClass::RunTimeout => "run-timeout",
            RejectClass::RunTimeoutBudgetInvalid => "run-timeout-budget-invalid",
            RejectClass::FreeRunCapabilityMissing => "free-run-capability-missing",
            RejectClass::PrefillWindowUncertified => "prefill-window-uncertified",
            RejectClass::CohortTokenTolerance => "cohort-token-tolerance",
            RejectClass::CohortReplayIntegrity => "cohort-replay-integrity",
        }
    }

    /// #108 (L1) — whether this class is a DETERMINISTIC condition, i.e. one whose second attempt is
    /// guaranteed to reach the identical verdict, so the R19 class-agnostic retry is EXEMPTED and the
    /// reject is TERMINAL.
    ///
    /// R19's retry exists for the transient population: a spawn that lost a race, a thermal gate that
    /// timed out under a passing load, a stalled round trip. The reset it performs (fresh worker,
    /// fresh cool gate) is what makes those retryable. A deterministic condition has no such
    /// population: the retry re-derives the same answer from the same unchanged input, and the only
    /// thing the extra attempt buys is a second spawn and a second cool gate before the same die-5 —
    /// which also DELAYS the honest diagnostic behind a gate wait, on a run that was already lost.
    ///
    /// The design intends deterministic rejects to be terminal; this is the mechanism. Only classes
    /// that are provably input-determined belong here — when in doubt, a class stays retryable,
    /// because a wrongly-terminal class turns a flake into a failed run.
    pub fn is_deterministic(self) -> bool {
        match self {
            // The hello capability list is a property of the engine BINARY. A second spawn of the
            // same binary advertises the same capabilities.
            RejectClass::FreeRunCapabilityMissing => true,
            // B2 — the scored regime is a lookup over a compile-time table. The second attempt
            // reads the same entry, and the preflight fence already refused this run.
            // (b) admission — both cohort-oracle classes are input-determined: the candidate's
            // committed tokens and the trusted reference argmax over the organizer's pinned weights
            // re-derive the same verdict on a retry. TERMINAL, so a lost run is not delayed behind a
            // second spawn + cool gate before the same die-5.
            RejectClass::CohortTokenTolerance | RejectClass::CohortReplayIntegrity => true,
            // Everything else is (or may be) transient, or is a per-attempt measurement outcome, and
            // keeps R19's one gated retry.
            //
            // `RunTimeoutBudgetInvalid` is also input-determined (the budget is computed once per
            // run), so it is a CANDIDATE for this list — but it is left retryable here deliberately:
            // it is unreachable through a parse-accepted calibration (#108 M2 bounds the band), so
            // exempting it would buy nothing measurable, and the conservative default for a class
            // nobody has observed firing is to keep the retry.
            RejectClass::TokenMismatchParity
            | RejectClass::ImplausibleSpt
            | RejectClass::RowAccounting
            | RejectClass::GateThermal
            | RejectClass::Infra
            | RejectClass::NonSerialTfRegime
            | RejectClass::FreeRunRegimeNotSpeculative
            | RejectClass::FreeRunSerialControlSpeculated
            | RejectClass::FreeRunConsistency
            | RejectClass::RunTimeout
            | RejectClass::RunTimeoutBudgetInvalid
            // B2 — a window that failed certification is a per-attempt MEASUREMENT outcome (the
            // clock split is re-measured on the retry's fresh worker), so it keeps R19's one
            // gated retry like every other measurement class.
            | RejectClass::PrefillWindowUncertified => false,
        }
    }
}

/// Classify a runner error into a reject class, matching on the TYPED variant (finding 5), so
/// a rename of an error message can never re-route the recorded class. The old `is_gate_class`
/// substring match on `RunnerError::Protocol` prose is gone. Finding R19 — this labels provenance
/// only; every class is retried once and folds into die-5 on persistence.
/// #134 — SEAL BOUNDARY for `results.rejected_pairs[].reason`.
///
/// The reason is engine-controlled: since #134 a transport-level `RunnerError` carries the
/// worker's own stderr tail, and this string is persisted into `results.json`. Scrub secrets
/// (absolute paths, `user@host`, credential-shaped `KEY=VALUE`, control bytes) and cap the length
/// before it becomes an artifact. Named, rather than inlined at the call site, so the property is
/// unit-testable without driving a whole measure run.
fn sealed_reject_reason(leg: &str, e: &RunnerError) -> String {
    scrub_reason_for_seal(&format!("{leg} leg: {e}"))
}

fn classify(e: &RunnerError) -> RejectClass {
    match e {
        // A `GateRejected` is the cool gate's stall/ceiling ABORT (its only producer here: the gate
        // has already exhausted its 900 s cooling budget). Finding R19 — that thermal timeout is a
        // RETRYABLE pair failure that folds into die-5, not a distinct exit-2 hard die.
        RunnerError::GateRejected { .. } => RejectClass::GateThermal,
        RunnerError::CompletedWorkMismatch { .. }
        | RunnerError::AllocatorCacheNotDrained { .. } => RejectClass::RowAccounting,
        RunnerError::TokenMismatch { .. } => RejectClass::TokenMismatchParity,
        // H3 (cycle-3) / W3 — a RunTimeout (hung/looping engine) is a fail-closed liveness reject:
        // the session is already discarded at the transport/session seam; here it is a RETRYABLE
        // pair failure (one gated retry, R19 class-agnostic) that, on persistence, folds into die-5
        // — the pair fails, benchd is NOT wedged. It carries its OWN provenance label (W3) so a
        // stalling engine is distinguishable from a spawn fault in the reject record.
        RunnerError::RunTimeout { .. } => RejectClass::RunTimeout,
        // #108 (M2) — a §2.2 budget that could not be ARMED (degenerate `N × ceiling × margin`),
        // distinct from the deadline having FIRED: the fault is in the calibration-derived ceiling,
        // not in the engine, and the reject record must say so.
        RunnerError::RunTimeoutBudgetInvalid { .. } => RejectClass::RunTimeoutBudgetInvalid,
        // #108 (L1) — the engine build does not advertise the v1.1 free-run capability (§2.1). Its
        // own class (the operator must be pointed at the ENGINE, not at the box), and DETERMINISTIC:
        // `RejectClass::is_deterministic` exempts it from the R19 retry, since the second spawn of
        // the same binary reads the same `hello.capabilities`.
        RunnerError::CapabilityNotAdvertised { .. } => RejectClass::FreeRunCapabilityMissing,
        // W3 — the §2.6 consistency TRIPLE / §2.4 count invariants of the v1.1 free-run window.
        RunnerError::FreeRunConsistency { .. } => RejectClass::FreeRunConsistency,
        _ => RejectClass::Infra,
    }
}

/// A construction context for a REJECT that makes invalid states unrepresentable (finding 7):
/// the class, the human reason, and which leg produced it are carried together rather than as
/// loose positional args. A rejected pair CONTRIBUTES NOTHING — it is never fabricated into a
/// per-pair timing record, and no `raw_median` is invented for it.
#[derive(Debug, Clone)]
pub struct RejectCtx {
    pub class: RejectClass,
    pub leg: &'static str,
    pub reason: String,
}

// ---------------------------------------------------------------------------
// --contract (track fixture) — AMENDMENT 2 schema
// ---------------------------------------------------------------------------

/// One entry of the `--contract` track fixture's `timed_prompt_pool[]`.
///
/// RE-EXPORTED, not redeclared. The measure-job used to carry a four-field `Contract` of its own
/// (`track_name`, `timed_prompt_pool`, `official_scoring_enabled`, `allowed_modes`) plus this
/// entry type, beside the migrated `bench_core::contract::Contract`, and `execute_measure_job`
/// parsed the SAME bytes into both — two typed schemas for one file, each free to drift. Those
/// four fields now live on the ONE schema, so a fixture is read once and described once.
pub use bench_core::contract::PromptPoolEntry;

/// The thermal thresholds a run uses. Finding R21 — the live thermal thresholds are READONLY
/// WRAPPER CONSTANTS (`live-measure-qwen-mtp-job.sh` W:422-429: `readonly GATE_TEMP=40`, `readonly
/// COOL_TIMEOUT=900`, `MIN_FREQ_SERIAL/MTP=1100`, `STALL_P50_FACTOR=4`) — "FIXED, NOT OVERRIDABLE"
/// by the contract or env; the live track fixture carries NO threshold fields. So the cool gate
/// ALWAYS enforces the fixed 40 °C constant and the contract `calibration` is NOT consulted (the
/// R5 contract-override path is reverted). The constants are still recorded in `results.json` for
/// provenance, each stamped with an HONEST wrapper source (`wrapper-constant-40` / …) so a consumer
/// can never mistake a fixed constant for a contract/env value.
///
/// FREE FUNCTION, and deliberately so: it took `&self` on the retired shadow contract and never
/// read a field of it, which is exactly the shape that made "the fixture can move the gate" look
/// plausible. It cannot — there is no fixture in the signature.
// UNVERIFIED(measure-job): the readonly wrapper constants are cited from W:422-429.
///
/// R14 — `loaded_util` is the ONE thermal field that IS env-driven (`GPU_LOADED_UTIL`, default
/// 0.70, W:403); the caller resolves it via [`resolve_loaded_util`] and passes it in with its
/// honest source. `cool_gate_c` / `clock_floor_mhz` stay FIXED wrapper constants (R21).
pub fn thermal_thresholds(loaded_util: f64, loaded_util_source: &'static str) -> ThermalThresholds {
    ThermalThresholds {
        cool_gate_c: 40.0,
        clock_floor_mhz: 1100.0,
        loaded_util,
        cool_gate_c_source: "wrapper-constant-40",
        clock_floor_mhz_source: "wrapper-constant-1100",
        loaded_util_source,
    }
}

/// R12 — resolve the SEALED CONSTANT `track_id` (the workflow-declared track id): env
/// `MLXFAST_QWEN_MTP_TRACK_ID` when set, else the `--contract` track fixture's own `track_id`.
/// The constant≡contract≡env rule: when BOTH the env AND the contract carry a track id they MUST
/// be EQUAL — a mismatch is a HARD ERROR (never silently pick one side). FAIL-CLOSED when neither
/// is available (the sealed `track_id` is a required constant, never fabricated from `--tag`).
///
/// All THREE legs are now checked. The resolved id must also equal the track this tree serves
/// (`bench_core::constants::TRACK_ID`) — see `bench_core::constants::enforce_declared_track` for
/// why an agreeing env/contract pair naming a FOREIGN track is still a defect: the per-track
/// baseline constants would be this track's while the seal says that one, and no later
/// cross-check can tell.
pub fn resolve_track_id(
    env_track_id: Option<&str>,
    scored: &bench_core::contract::Contract,
) -> Result<String, String> {
    let contract_track_id = scored.track_id.as_deref();
    let env = env_track_id.map(str::trim).filter(|s| !s.is_empty());
    let contract = contract_track_id.map(str::trim).filter(|s| !s.is_empty());
    let resolved = match (env, contract) {
        (Some(e), Some(c)) if e != c => {
            return Err(format!(
                "track_id mismatch: env MLXFAST_QWEN_MTP_TRACK_ID ({e:?}) != --contract track_id \
                 ({c:?}); the workflow-declared track id must be ONE value (constant≡contract≡env)"
            ))
        }
        (Some(e), _) => e.to_string(),
        (None, Some(c)) => c.to_string(),
        (None, None) => {
            return Err(
                "no track_id: set env MLXFAST_QWEN_MTP_TRACK_ID or provide a --contract track \
                 fixture carrying its own track_id (the sealed track_id is a required constant, \
                 not --tag)"
                    .to_string(),
            )
        }
    };
    // PAIRED-FLOW ENTRY FENCE — first, and by TRACK. A track whose sole scored path is the
    // single-leg `iterate --mode official` (David 2026-08-30) refuses measure-job BY NAME. The
    // paired flow is retired FOR THOSE TRACKS, not deleted: this file, its subcommands and its
    // tests all still serve the tracks that score through them. Checked before the tree fence
    // below so the operator reads "this track scores through the single leg", not the generic
    // "this tree serves another track".
    bench_core::contract::enforce_paired_flow_available(scored, &resolved)?;
    bench_core::constants::enforce_declared_track(&resolved)?;
    // B2 — and the track must DECLARE WHAT IT SCORES: the batch size of its scored point and the
    // composite exponents (`docs/scored-regime-and-prefill-window.md`). Resolved here, at the one
    // place a run's track id is settled, so the refusal lands in preflight rather than after a GPU
    // window.
    //
    // The fence above already pins `resolved` to `TRACK_ID`, and `TRACK_ID`'s regime is declared, so
    // ON THIS BRANCH this line is satisfied by construction. It is not decoration: the case it stops
    // is a NEW release branch that sets its own `TRACK_ID` and forgets the regime table — which is
    // precisely the shape of the defect the per-track baseline table was cut to fix, on the other
    // half of "what does this track score".
    bench_core::contract::scored_regime(scored, &resolved)?;
    Ok(resolved)
}

/// R12 — resolve the OPTIONAL `track_name`: env `MLXFAST_QWEN_MTP_TRACK_NAME` when set, else the
/// `--contract` fixture's `track_name` if present, else `None` (OMITTED from the seal rather than
/// fabricated).
pub fn resolve_track_name(
    env_track_name: Option<&str>,
    contract_track_name: Option<&str>,
) -> Option<String> {
    env_track_name
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or_else(|| contract_track_name.map(str::trim).filter(|s| !s.is_empty()))
        .map(str::to_string)
}

/// The thermal thresholds actually used, with per-field provenance (findings 8 / R21). Recorded in
/// `results.json`; finding R21 — these are FIXED readonly wrapper constants (never a contract/env
/// value), each stamped with an honest `wrapper-*` source so the seal cannot claim a false origin.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ThermalThresholds {
    pub cool_gate_c: f64,
    pub clock_floor_mhz: f64,
    pub loaded_util: f64,
    pub cool_gate_c_source: &'static str,
    pub clock_floor_mhz_source: &'static str,
    pub loaded_util_source: &'static str,
}

// ---------------------------------------------------------------------------
// finding 2 — worker-executable resolution
// ---------------------------------------------------------------------------

/// Resolve one leg's runtime-worker executable (finding 2). The explicit workspace/engine
/// WINS; a `MLXFAST_RUNTIME_WORKER_EXECUTABLE` override that CONFLICTS with it is a HARD ERROR
/// (never silently shadowed — a mismatched override would price the wrong binary). When the
/// workspace declares no engine, the override is used; when neither is present, it is an error.
/// Both legs' resolved executables are logged in `results.json`.
// UNVERIFIED(measure-job): the workspace→engine convention is a live-box concern.
pub fn resolve_worker_executable(
    workspace_engine: Option<&str>,
    override_env: Option<&str>,
) -> Result<String, String> {
    let ws = workspace_engine.map(str::trim).filter(|s| !s.is_empty());
    let ov = override_env.map(str::trim).filter(|s| !s.is_empty());
    match (ws, ov) {
        (Some(w), Some(o)) if w != o => Err(format!(
            "MLXFAST_RUNTIME_WORKER_EXECUTABLE ({o:?}) conflicts with the resolved workspace \
             engine ({w:?}); the explicit workspace engine wins — unset the override or make \
             them agree"
        )),
        (Some(w), _) => Ok(w.to_string()),
        (None, Some(o)) => Ok(o.to_string()),
        (None, None) => Err(
            "no runtime-worker executable: the workspace declares none and \
                 MLXFAST_RUNTIME_WORKER_EXECUTABLE is unset"
                .to_string(),
        ),
    }
}

/// The default engine binary inside a leg's workspace `.build/release/` — the binary that exposes
/// the `runtime-worker` subcommand. The fork engine's benchd-facing worker is the
/// `mlxfast-runtime-worker` product (its `runtime-worker` verb is the unified generic-kind
/// dispatch — `decode_begin`/`decode_step` + `free_decode_begin`/`free_decode_run`, routed by
/// `spec.mode`), spawned as `mlxfast-runtime-worker runtime-worker --weights <DIR>`; override the
/// binary name via `MLXFAST_MEASURE_WORKER_BIN` if a workspace names it differently.
pub const DEFAULT_MEASURE_WORKER_BIN: &str = "mlxfast-runtime-worker";

/// (b) admission — the env var the ORGANIZER points at their TRUSTED oracle worker binary (a build
/// of the organizer's UNMODIFIED, non-editable engine tree). It is the SOLE source of the trusted
/// oracle bin — [`resolve_trusted_oracle_worker_bin`] reads ONLY this, and FAILS CLOSED when it is
/// unset. There is deliberately NO default path and NO fallback to the candidate/baseline resolver:
/// falling back to the candidate build would let a candidate's editable forward code produce the
/// reference argmax, collapsing the entire anti-gaming guarantee (the oracle would judge the degraded
/// model against ITSELF).
pub const TRUSTED_ORACLE_WORKER_BIN_ENV: &str = "MLXFAST_TRUSTED_ORACLE_WORKER_BIN";

/// (b) admission — resolve the TRUSTED oracle worker binary from
/// [`TRUSTED_ORACLE_WORKER_BIN_ENV`], FAIL-CLOSED if unset or empty.
///
/// SECURITY (oracle weights + build provenance): this resolver is a SEPARATE function from the
/// candidate/baseline [`resolve_worker_executable`] / [`resolve_workspace_engine`] path and shares NO
/// fallback branch with it. It is STRUCTURALLY IMPOSSIBLE to reach [`DEFAULT_MEASURE_WORKER_BIN`] (the
/// candidate product name) or any candidate-workspace `.build/release` engine from here — the only
/// value this function can return is whatever the organizer put in
/// `MLXFAST_TRUSTED_ORACLE_WORKER_BIN`. So the oracle's forward AND weight-load code are the
/// organizer's, never the participant's editable code (N1 build half). Reachable ONLY from the oracle
/// spawn path; the candidate/baseline legs never call it.
///
/// The organizer MUST point the env at a binary built from the pinned / non-editable organizer tree.
pub fn resolve_trusted_oracle_worker_bin() -> Result<String, String> {
    match std::env::var(TRUSTED_ORACLE_WORKER_BIN_ENV) {
        Ok(v) if !v.trim().is_empty() => Ok(v.trim().to_string()),
        _ => Err(format!(
            "the trusted oracle worker binary is not configured: set {TRUSTED_ORACLE_WORKER_BIN_ENV} \
             to a build of the organizer's UNMODIFIED engine tree. The (b) reference argmax MUST come \
             from a TRUSTED build — benchd FAILS CLOSED here and NEVER falls back to the candidate \
             worker ({DEFAULT_MEASURE_WORKER_BIN}), because a candidate-built oracle could poison the \
             reference argmax (anti-gaming collapse)"
        )),
    }
}

/// Whether `path` is a regular file with an execute bit (unix) / exists as a file (elsewhere).
fn is_executable_file(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        match std::fs::metadata(path) {
            Ok(m) => m.is_file() && (m.permissions().mode() & 0o111 != 0),
            Err(_) => false,
        }
    }
    #[cfg(not(unix))]
    {
        path.is_file()
    }
}

/// Resolve one leg's runtime-worker executable FROM its WORKSPACE directory. `--candidate` /
/// `--baseline` are WORKSPACES, not binaries: the leg's engine is `<ws>/.build/release/<bin>`
/// (`bin` defaults to [`DEFAULT_MEASURE_WORKER_BIN`], overridable via `MLXFAST_MEASURE_WORKER_BIN`).
/// FAIL-CLOSED if that path does not exist or is not an executable file, naming the expected path.
/// The workspace-resolved path is then reconciled with an explicit `MLXFAST_RUNTIME_WORKER_EXECUTABLE`
/// override via [`resolve_worker_executable`] (finding-2 semantics: the workspace engine WINS; a
/// CONFLICTING override is a hard error). The proven spawn is then `<engine> runtime-worker
/// --weights <DIR>` where `<DIR>` is the SEPARATE `--weights` argument, not the workspace.
// UNVERIFIED(measure-job): the workspace→`.build/release/<bin>` layout is a live-box convention —
// first exercised on-box; the proven official path spawned `mlxfast-runtime-worker runtime-worker
// --weights <DIR>`.
pub fn resolve_workspace_engine(
    workspace: &str,
    worker_bin: &str,
    override_env: Option<&str>,
) -> Result<String, String> {
    let ws = workspace.trim();
    if ws.is_empty() {
        return Err(
            "empty workspace directory (--candidate/--baseline are WORKSPACES)".to_string(),
        );
    }
    let bin = worker_bin.trim();
    let bin = if bin.is_empty() {
        DEFAULT_MEASURE_WORKER_BIN
    } else {
        bin
    };
    let engine = Path::new(ws).join(".build").join("release").join(bin);
    let engine_str = engine.to_string_lossy().to_string();
    if !is_executable_file(&engine) {
        return Err(format!(
            "workspace engine not found or not executable: {engine_str} (expected \
             <workspace>/.build/release/{bin}; set MLXFAST_MEASURE_WORKER_BIN to override the \
             binary name)"
        ));
    }
    // The workspace engine is now a concrete path; reconcile against any explicit override.
    resolve_worker_executable(Some(&engine_str), override_env)
}

/// The Metal shader library the worker loads at first-GPU-use. Metal resolves `mlx.metallib` from
/// BESIDE the worker executable (a pinned M5 release ships the pair together, `docs/architecture.md`
/// §pin), so its identity is FIXED and its location is "next to the resolved binary".
pub const MLX_METALLIB_SIBLING: &str = "mlx.metallib";

/// PRE-GPU adjacency guard (#42 box-leg): verify the RESOLVED worker's sibling `mlx.metallib` exists
/// next to it, fail-closed. `engine` is a path already produced by [`resolve_workspace_engine`] —
/// this NEVER re-resolves or alters resolution; it only asserts the adjacency Metal will silently
/// rely on. A worker missing its sibling metallib does NOT fail at spawn: it dies LATE, at the first
/// `MLXArray` inside the GPU window, after gated box time is already spent. Checking it at preflight
/// turns that late in-window death into a clear pre-GPU refusal that names the missing file.
///
/// The measure-job worker on this fork is always the MLX/Metal `mlxfast-runtime-worker`, so every
/// measure run is a Metal run; the check binds unconditionally here. It is a pure filesystem
/// existence check (no GPU, no spawn), so it runs identically off-box and on the `--preflight-only`
/// path.
pub fn verify_worker_metallib_sibling(engine: &str) -> Result<(), String> {
    let sibling = Path::new(engine).with_file_name(MLX_METALLIB_SIBLING);
    if !sibling.is_file() {
        return Err(format!(
            "resolved worker {engine} is missing its sibling `{MLX_METALLIB_SIBLING}` ({}): Metal \
             loads the shader library from beside the worker binary, so without it the run dies at \
             the first MLXArray inside the GPU window. A pinned release ships the binary and \
             `{MLX_METALLIB_SIBLING}` together — stage the metallib next to the worker (die 8, \
             pre-GPU)",
            sibling.display(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// The measure-job configuration resolved at the CLI boundary (findings 4/8 parse here).
#[derive(Debug, Clone)]
pub struct MeasureJobConfig {
    /// R12 — the SEALED CONSTANT `track_id` (the workflow-declared track id), resolved from env
    /// `MLXFAST_QWEN_MTP_TRACK_ID` or the `--contract` fixture (constant≡contract≡env). NOT `--tag`.
    pub track_id: String,
    /// R12 — the optional `track_name`, sealed when available (env/contract), omitted otherwise.
    pub track_name: Option<String>,
    /// R12 — the per-run identity `--tag`, sealed SEPARATELY from the constant `track_id`.
    pub tag: String,
    /// R16 (medium cycle-3) — the run timestamp sealed top-level (`date -u`, W:1866). Set by the CLI
    /// to `iso8601_now()`; a FIXED value in tests keeps `build_results` deterministic.
    pub run_timestamp: String,
    /// `--tokens` → the depth-0 decode window both legs time.
    pub tokens: usize,
    /// R13 — `--mtp-depth` → the candidate MTP depth (serial control is the depth-0 constant).
    /// Recorded, not scored. Since the spec re-home (`docs/spec-config-design.md`), depth is a MODULE
    /// field on [`candidate_spec`](Self::candidate_spec); this mirror stays for the sealed
    /// `mtp_depth`/provenance and is derived from the candidate spec's `mtp.depth`. `None` = the
    /// engine-decides spec (David ruling 2026-08-27): no depth was requested, none is sealed — the
    /// operating value is the engine's echoed `effective_spec`, per leg.
    pub mtp_depth: Option<usize>,
    /// The candidate leg's declared per-module speculative spec (`docs/spec-config-design.md`): from
    /// the submission/contract, or a `--candidate-spec` override. Plumbed onto the timed decode
    /// window (the runner enforces the never-ignored echo) and sealed as the declared spec.
    pub candidate_spec: SpecConfig,
    /// The baseline leg's declared spec — defaults to `{"mode":"serial"}`; a `--baseline-spec`
    /// override is recorded with `spec_source: "cli-override"`.
    pub baseline_spec: SpecConfig,
    /// Honest provenance of where `candidate_spec` came from (`"mtp-depth-flag"` /
    /// `"mtp-depth-default"` / `"cli-override"`), sealed so a consumer can see the source. The flag
    /// and default cases are distinguished (cycle-5 finding 5): only one of them names a flag the
    /// operator actually passed. #105 H-C — never the FALSE
    /// `"contract-default"` (no contract speculative-block parsing exists).
    pub candidate_spec_source: String,
    /// Honest provenance of where `baseline_spec` came from (`"serial-default"` / `"cli-override"`).
    pub baseline_spec_source: String,
    /// `--min-pairs` → the run fails closed (die 5) below this many accepted pairs.
    pub min_pairs: usize,
    /// `--target-pairs` → the loop stops accepting once it reaches this many.
    pub target_pairs: usize,
    /// The parsed contract's per-prompt no-op refs (carried into the superset).
    pub prompt_pool: Vec<PromptPoolEntry>,
    /// The thermal thresholds used + provenance (finding 8).
    pub thermal: ThermalThresholds,
    /// The candidate/baseline resolved executables (finding 2), logged in `results.json`.
    pub candidate_executable: String,
    pub baseline_executable: String,
    /// R14 — the RESOLVED baseline calibration used for the serial-band check (mean/band/source +
    /// decode_tokens), or `None` when `BASELINE_CALIBRATION` was not provided. Recorded in provenance;
    /// the die-6 ENFORCEMENT runs in `execute_measure_job` against the pooled serial mean.
    pub calibration: Option<ResolvedCalibration>,
    /// R14 — `BASELINE_BAND_ENFORCE` (default 1): when true, a MISSING calibration fails closed (die-6).
    pub band_enforce: bool,
    /// R13 — `--calibration-bootstrap`: skip the band check and mark the run for authoring.
    pub calibration_bootstrap: bool,
    /// R13/R14 — the declared `--target-id` (recognised input), recorded for provenance / calibration keying.
    pub target_id: Option<String>,
    /// R13/R16 — the declared `--prompt-sha256` (the pinned prompt of the R13 trio), sealed into
    /// `evaluation_target`; `None` when the trio is absent (default-pool run).
    pub prompt_sha256: Option<String>,
    /// R13 — the declared `--exactness-probe` mode, recorded for provenance (gate is R15).
    pub exactness_probe: ExactnessProbe,
    /// H6/H3 (cycle-3) — LOCAL-DEV mode (`--local-dev`, default false = OFFICIAL/ranked). In the
    /// RANKED/official path a pair that fails after its one gated retry is an IMMEDIATE die-5 (W:2005-
    /// 2032) — no budget loop trying more pairs. The `PAIR_ATTEMPT_BUDGET_MULTIPLE` budget loop that
    /// keeps attempting fresh pairs up to `target_pairs × 4` is allowed ONLY here (local-dev), never
    /// on the official path.
    pub local_pair_budget: bool,
    /// W3 — the TIMED REGIME the CANDIDATE leg runs. Production derives this from the declared
    /// candidate spec via [`candidate_regime_for_spec`] (mtp/dflash ⇒ v1.1 free-run, serial ⇒
    /// teacher-forced), so the CLI has exactly one rule and no separate flag to drift from it; it is
    /// a config field rather than a re-derivation so the pure core is drivable in BOTH regimes from
    /// a test without also having to forge a spec. It is the ONLY regime input: the SERIAL control
    /// leg's regime is DERIVED from it by [`serial_control_regime_for`] (the Fable same-series
    /// ruling), so the two legs cannot be configured into different series.
    ///
    /// [`run_measure_job`] cross-checks it against `candidate_spec` fail-closed, so an incoherent
    /// pair (free-run regime declared for a serial candidate) never measures.
    pub candidate_regime: LegRegime,
    /// Orchestrator ruling (2026-08-23) — the CERTIFIED composite-score exponent pair
    /// ([`ScoredExponents::certify`]'s output), resolved by the caller BEFORE this config is built
    /// (mirroring how `candidate_regime` already carries its own certified [`ScoredBatchPoint`]).
    /// `Some` on every batched-cohort config (required — [`build_cohort_results`] refuses to seal
    /// without it); `None` on every single-stream config (never consulted there — composite
    /// scoring is cohort-only).
    pub scored_exponents: Option<ScoredExponents>,
    /// THE TRACK'S RESOLVED SCORED REGIME — the batch width and the two composite exponents, read
    /// out of the `--contract` track fixture (STEP 3 of the constants→contract migration, David
    /// 2026-09-15: the per-track table is gone and an absent declaration refuses).
    ///
    /// Resolved ONCE, where the track id is settled, and carried here so every leg of the run
    /// certifies its prefill window against the SAME regime. It used to be looked up per leg from
    /// a compile-time table; there is no table to look it up in any more, and a per-leg
    /// re-resolution could not have disagreed with itself anyway — carrying it says so.
    pub scored_regime: bench_core::constants::ScoredRegime,
}

// ---------------------------------------------------------------------------
// R14 — BASELINE_CALIBRATION (JSON file) + serial-band enforcement (die-6)
// ---------------------------------------------------------------------------

fn default_band_low() -> f64 {
    0.95
}
fn default_band_high() -> f64 {
    1.05
}

/// #108 (M2) — the HARD CAP on `serial_band_high`. The band multiplier is not only the die-6 drift
/// verdict: `band_ceiling = serial_mean × band_high` is also the §2.2 RunTimeout arithmetic
/// ([`bench_core::score::run_timeout_budget`], `N × ceiling × margin`), so an unbounded `band_high`
/// is an unbounded liveness deadline. A `1e9` band_high does not "loosen the band" — it disarms the
/// only wall-clock bound benchd has inside the timed window, and it does so through a config file
/// rather than through code. 100 is far outside any honest drift band (a ±5% band is 1.05) while
/// still bounding the deadline at `N × serial_mean × 100 × margin`.
pub const SERIAL_BAND_HIGH_CAP: f64 = 100.0;

/// #108 (M2) — validate one `(low, high)` band pair against the bounds every band must satisfy,
/// wherever it was declared (top-level defaults or a per-target override — the SAME bounds; an
/// override is not a way around them). `where_` names the declaring site for the diagnostic.
///
/// The bounds encode what a band IS: a multiplier around a measured serial mean. `low` must be
/// finite and in the open interval `0 < low < 1`; `high` finite with `1 < high <=`
/// [`SERIAL_BAND_HIGH_CAP`]. So the band actually brackets the reference from both sides — a `low`
/// at or above 1, or a `high` at or below 1, does not bracket anything. A `high` of `0.0` or `-1` is
/// likewise not a "wide" band — it makes the
/// die-6 predicate `ratio <= high` unsatisfiable AND drives the §2.2 ceiling non-positive, which
/// (before this) silently DISARMED the RunTimeout. Refused at PARSE so a hostile or fat-fingered
/// file never reaches either consumer.
fn validate_band_bounds(where_: &str, low: f64, high: f64) -> Result<(), String> {
    if !(low.is_finite() && low > 0.0 && low < 1.0) {
        return Err(format!(
            "BASELINE_CALIBRATION {where_} serial_band_low ({low}) is outside (0.0, 1.0): the band \
             low is a multiplier BELOW the calibrated serial mean, so it must be finite, positive \
             and less than 1 — die 6 (re-author with --calibration-bootstrap)"
        ));
    }
    if !(high.is_finite() && high > 1.0 && high <= SERIAL_BAND_HIGH_CAP) {
        return Err(format!(
            "BASELINE_CALIBRATION {where_} serial_band_high ({high}) is outside (1.0, \
             {SERIAL_BAND_HIGH_CAP}]: the band high is a multiplier ABOVE the calibrated serial \
             mean, and it ALSO sets the PROTOCOL-v1.1 §2.2 RunTimeout ceiling (serial_mean × \
             band_high). A non-positive/absurd value does not widen the band — it makes the drift \
             check unsatisfiable and disarms the only wall-clock bound on the timed window — die 6 \
             (re-author with --calibration-bootstrap)"
        ));
    }
    Ok(())
}

/// H6/H2 (cycle-3) — parse `BASELINE_BAND_ENFORCE` fail-closed, mirroring the wrapper's
/// `BASELINE_BAND_ENFORCE="${BASELINE_BAND_ENFORCE:-1}"` (W:336): UNSET and EMPTY-STRING BOTH mean
/// ENFORCED — an empty env value must NEVER read as "disabled" (a missing band must never read as
/// "in band", the wrapper's stated invariant W:1416-1419). ONLY an explicit `"0"` disables the band;
/// any other value enforces.
pub fn band_enforce_from_env(raw: Option<&str>) -> bool {
    match raw {
        Some(v) => v.trim() != "0",
        None => true,
    }
}

/// R14 — the `BASELINE_CALIBRATION` JSON file (env), REPLACING the dead scalar
/// `MLXFAST_QWEN_MTP_SERIAL_CALIBRATION_SPT`. Top-level defaults apply when a target omits its band;
/// each per-target entry carries its own serial mean and a REQUIRED `decode_tokens` (no inherit).
/// FAIL-CLOSED on malformed JSON (die-6). Schema: live wrapper W:196-239.
///
/// #105 cycle-5 (HIGH — the series fence) — the file additionally REQUIRES its own `timed_mode`
/// series tag and `track_id`, both cross-checked against the run BEFORE any banding by
/// [`enforce_calibration_series_fence`]. Before this, a native-regime calibration could band a
/// Model-2 result to a `Pass`: the series tag was not modeled at all and the file's own `track_id`
/// was parsed nowhere, so nothing tied the calibrated quantity to the quantity being measured.
/// `docs/model2-calibration.md` §1 makes the segregation a HARD RULE ("Model-2 numbers are NEVER
/// compared to native-regime numbers ... A Model-2 run is gated ONLY against Model-2 calibration")
/// and §5 requires the calibration legs to run in the SAME series as the scored candidate; both are
/// now machine-enforced rather than conventional. Both fields are REQUIRED (serde has no `default`),
/// so a legacy/native file that declares neither fails the parse fail-closed → die-6.
#[derive(Debug, Clone, Deserialize)]
pub struct BaselineCalibration {
    /// #105 cycle-5 (HIGH) — the REQUIRED series tag naming the regime this file's means were
    /// MEASURED under ([`bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1`] /
    /// [`bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1`], or any other series label — an unknown
    /// label is simply not comparable to this run's and dies). A band is a per-series artifact: the
    /// seed prefill and the decode window are charged differently in each regime, so a mean measured
    /// under one clock/regime says nothing about a run under another.
    pub timed_mode: String,
    /// #105 cycle-5 (HIGH) — the REQUIRED track id the file was authored FOR, cross-checked against
    /// the run's resolved `track_id` ([`resolve_track_id`]). The wrapper already authors this
    /// (`{track_id, targets{...}}`, W:1468-1528) — it was parsed nowhere, so a calibration authored
    /// for another track could band this one.
    pub track_id: String,
    /// H6/H2 (cycle-3) — the top-level mean is OPTIONAL, not required. The live wrapper's
    /// `write_calibration_bootstrap` (W:1468-1528) authors `{track_id, targets{...}}` with NO
    /// top-level `serial_decode_seconds_per_token_mean` (it is only ever a per-target fallback);
    /// requiring it here rejected EVERY wrapper-authored file with a die-6. A run keying on
    /// `--target-id` resolves the per-target mean; a top-level resolve (no `--target-id`) still
    /// fails closed when this is absent (see `resolve`).
    #[serde(default)]
    pub serial_decode_seconds_per_token_mean: Option<f64>,
    #[serde(default = "default_band_low")]
    pub serial_band_low: f64,
    #[serde(default = "default_band_high")]
    pub serial_band_high: f64,
    /// Top-level decode_tokens (optional at the top; REQUIRED per-target with no inherit).
    #[serde(default)]
    pub decode_tokens: Option<usize>,
    #[serde(default)]
    pub targets: std::collections::HashMap<String, TargetCalibration>,
}

/// R14 — one per-target calibration entry. `serial_band_low/high` INHERIT the top-level defaults
/// when omitted; `decode_tokens` is REQUIRED and NEVER inherited (a target without it is a parse error).
#[derive(Debug, Clone, Deserialize)]
pub struct TargetCalibration {
    pub serial_decode_seconds_per_token_mean: f64,
    #[serde(default)]
    pub serial_band_low: Option<f64>,
    #[serde(default)]
    pub serial_band_high: Option<f64>,
    /// REQUIRED, no inherit (serde has no `default`, so a missing field fails the parse).
    pub decode_tokens: usize,
}

/// R14 — the calibration RESOLVED for a specific target (or the top-level default): the mean the
/// pooled serial mean is divided by, the band it must land in, the (optional) pinned decode_tokens,
/// and an honest `source` so a consumer can see which entry drove the check. Recorded in provenance.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResolvedCalibration {
    pub serial_mean: f64,
    pub band_low: f64,
    pub band_high: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_tokens: Option<usize>,
    /// #105 cycle-5 — the calibration file's own series tag, carried through the resolve so
    /// provenance seals WHICH series the band that gated this run was measured under. The
    /// comparability decision itself is made earlier by [`enforce_calibration_series_fence`]; this is
    /// the audit trail of what it checked.
    pub timed_mode: String,
    /// #105 cycle-5 — the calibration file's own track id, likewise sealed for audit.
    pub track_id: String,
    pub source: String,
}

impl BaselineCalibration {
    /// Parse the calibration file bytes, FAIL-CLOSED on malformed JSON (die-6 upstream).
    ///
    /// #105 cycle-5 — the REQUIRED series/track identity is checked HERE, by key, so an untagged
    /// file gets an honest diagnostic instead of serde's bare `missing field` prose. An UNTAGGED
    /// file is refused rather than defaulted: a band whose series is unknown cannot be shown
    /// comparable to this run's, and "unknown" must never read as "same" (the same fail-closed
    /// posture `band_enforce_from_env` takes — a missing band must never read as in band). Every
    /// pre-fence (native-regime, wrapper-authored) calibration file lands here, which is the
    /// intended outcome: `docs/model2-calibration.md` §3a — "a native-regime `serial_mean` would
    /// drift-fail every honest Model-2 run ... and must not be installed."
    pub fn parse(bytes: &[u8]) -> Result<BaselineCalibration, String> {
        let value: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| format!("BASELINE_CALIBRATION parse failed: {e}"))?;
        for key in ["timed_mode", "track_id"] {
            let present = value
                .get(key)
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .is_some_and(|s| !s.is_empty());
            if !present {
                return Err(format!(
                    "BASELINE_CALIBRATION carries no non-empty `{key}` — a calibration file must \
                     declare the SERIES its means were measured in (`timed_mode`) and the TRACK it \
                     was authored for (`track_id`) so they can be cross-checked against the run \
                     before banding; an untagged band must NEVER be assumed comparable — die 6 \
                     (re-author with --calibration-bootstrap, which stamps both)"
                ));
            }
        }
        let parsed: BaselineCalibration = serde_json::from_value(value)
            .map_err(|e| format!("BASELINE_CALIBRATION parse failed: {e}"))?;
        // #108 (M2) — the BAND BOUNDS, checked at PARSE (before any consumer sees the file): the
        // top-level defaults AND every per-target override, against the SAME bounds. An override
        // inherits the top-level value when omitted, so both sites must be bounded or the bound is
        // only advisory. See [`validate_band_bounds`] for why an out-of-range `band_high` is a
        // liveness hole, not merely a loose band.
        validate_band_bounds("top-level", parsed.serial_band_low, parsed.serial_band_high)?;
        for (tid, t) in &parsed.targets {
            validate_band_bounds(
                &format!("targets[{tid}]"),
                t.serial_band_low.unwrap_or(parsed.serial_band_low),
                t.serial_band_high.unwrap_or(parsed.serial_band_high),
            )?;
        }
        Ok(parsed)
    }

    /// Resolve the calibration for `target_id`: a matching `targets[<tid>]` entry (band inheriting
    /// the top-level defaults, decode_tokens from the target) when a target is declared, else the
    /// TOP-LEVEL calibration. The `source` names which was used.
    ///
    /// FAIL-CLOSED: when a `--target-id` IS declared but the file carries no matching `targets[<tid>]`
    /// entry (or its serial mean is not a finite positive value), this is a MISWIRED rotation, not a
    /// tolerable fallback to the top-level baseline — an `Err` (die-6 upstream), mirroring the live
    /// wrapper's `require_target_calibration` (W:1367-1379). A target must NEVER be validated against
    /// another entry's baseline.
    pub fn resolve(&self, target_id: Option<&str>) -> Result<ResolvedCalibration, String> {
        if let Some(tid) = target_id {
            match self.targets.get(tid) {
                Some(t)
                    if t.serial_decode_seconds_per_token_mean.is_finite()
                        && t.serial_decode_seconds_per_token_mean > 0.0 =>
                {
                    return Ok(ResolvedCalibration {
                        serial_mean: t.serial_decode_seconds_per_token_mean,
                        band_low: t.serial_band_low.unwrap_or(self.serial_band_low),
                        band_high: t.serial_band_high.unwrap_or(self.serial_band_high),
                        decode_tokens: Some(t.decode_tokens),
                        timed_mode: self.timed_mode.clone(),
                        track_id: self.track_id.clone(),
                        source: format!("baseline-calibration:target:{tid}"),
                    });
                }
                Some(_) => {
                    return Err(format!(
                        "BASELINE_CALIBRATION has a targets[{tid}] entry with no finite positive \
                         serial_decode_seconds_per_token_mean — die 6 (a target must not be banded \
                         against a missing/invalid mean; re-author with --calibration-bootstrap)"
                    ));
                }
                None => {
                    return Err(format!(
                        "BASELINE_CALIBRATION has no entry for target {tid} (a declared target must \
                         NOT fall back to the top-level baseline — that is a miswired rotation) — \
                         die 6; install the per-target calibration or author it with \
                         --calibration-bootstrap"
                    ));
                }
            }
        }
        // TOP-LEVEL resolve (no `--target-id`): the top-level mean is a FALLBACK the wrapper only
        // sometimes authors. H6/H2 — fail closed (die-6) when it is absent or not a finite positive
        // value, rather than banding against a fabricated 0 (a wrapper-authored target-only file has
        // no top-level mean; the caller must pass `--target-id` to select a per-target entry).
        match self.serial_decode_seconds_per_token_mean {
            Some(m) if m.is_finite() && m > 0.0 => Ok(ResolvedCalibration {
                serial_mean: m,
                band_low: self.serial_band_low,
                band_high: self.serial_band_high,
                decode_tokens: self.decode_tokens,
                timed_mode: self.timed_mode.clone(),
                track_id: self.track_id.clone(),
                source: "baseline-calibration:top".to_string(),
            }),
            _ => Err(
                "BASELINE_CALIBRATION carries no top-level serial_decode_seconds_per_token_mean and \
                 no --target-id was given to select a per-target entry (the wrapper-authored file \
                 keys the serial denominator per target) — die 6; pass --target-id or author a \
                 top-level mean"
                    .to_string(),
            ),
        }
    }
}

/// #105 cycle-5 (HIGH — the series fence) — the PRE-BANDING gate: refuse to band this run against a
/// calibration measured in a DIFFERENT series, or authored for a DIFFERENT track. Returns `Err`
/// (die-6 upstream, BEFORE any measurement and therefore before any banding) on either mismatch.
///
/// The series decision is delegated to [`bench_core::free_run::timed_modes_comparable`] — this is
/// that predicate's PRODUCTION caller, so the "machine-checked" comparability rule the free-run
/// module documents is now actually machine-checked on the path that gates a run, not a write-only
/// tag. Two timed numbers are comparable ONLY if they carry the same series tag; a calibration mean
/// and the pooled serial mean it divides are exactly such a pair of timed numbers.
///
/// The attack this closes (executed in review): a `BASELINE_CALIBRATION` self-declaring a
/// native-regime series with a frontier-era serial mean banded a Model-2 (benchd-clock) result to a
/// `Pass` under `BASELINE_BAND_ENFORCE=1`. Nothing in the band check compared the two regimes,
/// because the band check only ever saw a bare `f64` and a window. `docs/model2-calibration.md` §1
/// states the rule ("A Model-2 run is gated ONLY against Model-2 calibration") and names
/// `timed_modes_comparable` as its enforcement point; this function is that enforcement point.
///
/// The track cross-check is the same class of miswiring one level up: [`BaselineCalibration::resolve`]
/// already refuses to band a declared `--target-id` against another ENTRY's baseline, but the FILE
/// as a whole could still belong to another track entirely. A file authored for `qwen3.6-…` must not
/// gate a `qwen3.8-…` run.
pub fn enforce_calibration_series_fence(
    cal: &BaselineCalibration,
    run_timed_mode: &str,
    run_track_id: &str,
) -> Result<(), String> {
    let file_mode = cal.timed_mode.trim();
    if !bench_core::free_run::timed_modes_comparable(file_mode, run_timed_mode) {
        return Err(format!(
            "BASELINE_CALIBRATION was measured in series {file_mode:?} but this run measures series \
             {run_timed_mode:?} — the two are NOT comparable (bench_core::free_run::\
             timed_modes_comparable), so this calibration must NOT band this run — die 6; a run is \
             gated ONLY against calibration authored in its own series (docs/model2-calibration.md \
             §1/§5). Re-author with --calibration-bootstrap under this series."
        ));
    }
    let file_track = cal.track_id.trim();
    if file_track != run_track_id {
        return Err(format!(
            "BASELINE_CALIBRATION was authored for track {file_track:?} but this run resolved track \
             {run_track_id:?} — a track must NEVER be banded against another track's baseline \
             (miswired rotation) — die 6"
        ));
    }
    Ok(())
}

/// R14 — the serial-band verdict CATEGORY. `Pass` = the pooled serial mean is in band at the right
/// window; `WarnOutOfBand` = out of band (or an unrecorded/invalid calibration mean) but
/// `BASELINE_BAND_ENFORCE=0` DOWNGRADED it to a warning (does NOT die); `Die6` = fail-closed → exit 6
/// (drift under enforcement, a window that is absent/mismatched, or an invalid measured mean).
/// Mirrors the live wrapper `serial_band_check` (W:1416-1466): window + measured-mean checks are
/// HARD regardless of `BASELINE_BAND_ENFORCE`; the ratio and a missing calibration mean are the only
/// checks the enforce flag can downgrade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SerialBandVerdict {
    Pass,
    WarnOutOfBand,
    Die6,
}

/// R14 — the SEALED serial-band outcome (results.json provenance): the pooled serial mean actually
/// measured, the calibration mean/band/window it was checked against, the ratio, and an HONEST
/// pass/fail verdict + reason. Computed by [`evaluate_serial_band`]; NOT fabricated. `passed` is the
/// band health (in band at the right window) and is FALSE for a warn-only out-of-band run even
/// though the process did not die.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SerialBandOutcome {
    pub pooled_serial_mean: f64,
    pub calibration_mean: f64,
    pub band_low: f64,
    pub band_high: f64,
    /// `pooled_serial_mean / calibration_mean`; omitted when the window/means made it uncomputable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ratio: Option<f64>,
    /// The calibration's pinned window (no inherit); omitted when the calibration carried none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_decode_tokens: Option<usize>,
    pub run_decode_tokens: usize,
    pub window_ok: bool,
    pub in_band: bool,
    pub enforced: bool,
    pub verdict: SerialBandVerdict,
    pub passed: bool,
    pub source: String,
    pub detail: String,
}

/// R14 — evaluate the serial band AFTER measuring: the POOLED serial mean
/// (`aggregate.baseline_serial_seconds_per_token_mean`) divided by the calibration mean must lie in
/// `[band_low, band_high]`, and the calibration `decode_tokens` must be PRESENT (no inherit) and
/// EQUAL `--tokens`. Pure + unit-tested; the returned [`SerialBandOutcome`] is sealed into
/// provenance and drives the die-6 verdict via [`enforce_serial_band`].
///
/// Fail-closed layering (matches the wrapper): a missing/mismatched window and a non-finite measured
/// mean are ALWAYS `Die6`; an out-of-band ratio or a non-finite calibration mean are `Die6` under
/// `band_enforce` and `WarnOutOfBand` otherwise.
pub fn evaluate_serial_band(
    pooled_serial_mean: f64,
    tokens: usize,
    cal: &ResolvedCalibration,
    band_enforce: bool,
) -> SerialBandOutcome {
    let make = |verdict: SerialBandVerdict,
                in_band: bool,
                window_ok: bool,
                ratio: Option<f64>,
                detail: String| SerialBandOutcome {
        pooled_serial_mean,
        calibration_mean: cal.serial_mean,
        band_low: cal.band_low,
        band_high: cal.band_high,
        ratio,
        calibration_decode_tokens: cal.decode_tokens,
        run_decode_tokens: tokens,
        window_ok,
        in_band,
        enforced: band_enforce,
        verdict,
        passed: matches!(verdict, SerialBandVerdict::Pass),
        source: cal.source.clone(),
        detail,
    };

    // Window (HARD, never downgraded): decode_tokens must be present (no inherit) AND equal the run
    // window. The seed prefill is charged inside the decode window, so seconds/token is NOT
    // comparable across token counts — a band at the wrong window is meaningless (W check_calibration_window).
    match cal.decode_tokens {
        None => {
            return make(
                SerialBandVerdict::Die6,
                false,
                false,
                None,
                format!(
                    "calibration [{}] records no decode_tokens (it does NOT inherit from the top \
                     level — the window is a property of the measurement that produced this mean), \
                     so the band cannot be bound to this run's {tokens}-token window — die 6; \
                     re-author with --calibration-bootstrap --tokens {tokens}",
                    cal.source
                ),
            );
        }
        Some(dt) if dt != tokens => {
            return make(
                SerialBandVerdict::Die6,
                false,
                false,
                None,
                format!(
                    "calibration band was authored at {dt} decode tokens but this run measured \
                     {tokens}; seconds/token is not comparable across token counts [{}] — die 6",
                    cal.source
                ),
            );
        }
        Some(_) => {}
    }

    // Measured pooled mean invalid → die-6 regardless of enforce (the measurement, not the band).
    if !(pooled_serial_mean.is_finite() && pooled_serial_mean > 0.0) {
        return make(
            SerialBandVerdict::Die6,
            false,
            true,
            None,
            format!(
                "measured pooled serial mean is not a finite positive value ({pooled_serial_mean}) — die 6"
            ),
        );
    }

    // Calibration mean invalid / unrecorded → die-6 under enforce, warn-only otherwise.
    if !(cal.serial_mean.is_finite() && cal.serial_mean > 0.0) {
        let verdict = if band_enforce {
            SerialBandVerdict::Die6
        } else {
            SerialBandVerdict::WarnOutOfBand
        };
        return make(
            verdict,
            false,
            true,
            None,
            format!(
                "calibration [{}] carries no finite positive serial mean ({}) — {}",
                cal.source,
                cal.serial_mean,
                if band_enforce {
                    "die 6"
                } else {
                    "warn only (BASELINE_BAND_ENFORCE=0)"
                }
            ),
        );
    }

    let ratio = pooled_serial_mean / cal.serial_mean;
    let in_band = ratio >= cal.band_low && ratio <= cal.band_high;
    if in_band {
        make(
            SerialBandVerdict::Pass,
            true,
            true,
            Some(ratio),
            format!(
                "serial mean {pooled_serial_mean} / calibration mean {} = {ratio:.6} in band \
                 [{}, {}] [{}]",
                cal.serial_mean, cal.band_low, cal.band_high, cal.source
            ),
        )
    } else {
        let verdict = if band_enforce {
            SerialBandVerdict::Die6
        } else {
            SerialBandVerdict::WarnOutOfBand
        };
        make(
            verdict,
            false,
            true,
            Some(ratio),
            format!(
                "serial calibration drift: pooled serial mean {pooled_serial_mean} / calibration \
                 mean {} = {ratio:.6}, outside band [{}, {}] [{}] — {}",
                cal.serial_mean,
                cal.band_low,
                cal.band_high,
                cal.source,
                if band_enforce {
                    "die 6"
                } else {
                    "warn only (BASELINE_BAND_ENFORCE=0)"
                }
            ),
        )
    }
}

/// R14 — enforce the serial band: die-6 (an `Err` carrying the reason) iff [`evaluate_serial_band`]
/// returns a [`SerialBandVerdict::Die6`]. A `WarnOutOfBand` (under `BASELINE_BAND_ENFORCE=0`) and a
/// `Pass` both return `Ok`. The sealed [`SerialBandOutcome`] in provenance is computed from the SAME
/// pure function with the SAME inputs, so the seal and the verdict never diverge.
pub fn enforce_serial_band(
    pooled_serial_mean: f64,
    tokens: usize,
    cal: &ResolvedCalibration,
    band_enforce: bool,
) -> Result<(), String> {
    let outcome = evaluate_serial_band(pooled_serial_mean, tokens, cal, band_enforce);
    match outcome.verdict {
        SerialBandVerdict::Die6 => Err(outcome.detail),
        SerialBandVerdict::Pass | SerialBandVerdict::WarnOutOfBand => Ok(()),
    }
}

/// R13/R14 — a bootstrap run AUTHORS the band only after a FULLY-ACCEPTED, parity-true run. A new
/// track starts uncalibrated, so `--calibration-bootstrap` is an authoring path, not merely "skip the
/// check": but authoring off a rejected or parity-false run would seal a meaningless mean. Mirrors the
/// live wrapper gate (W:2239 — write only after acceptance + the parity re-audit).
pub fn should_author_bootstrap(candidate_accepted: bool, parity_all_ok: bool) -> bool {
    candidate_accepted && parity_all_ok
}

/// R13/R14 — build the merged `baseline-calibration.json` bytes for `--calibration-bootstrap`: author
/// `targets[<target_id>]` from a fully-accepted parity-true run (its pooled serial mean at the run's
/// window), PRESERVING every other target already in `existing`. Mirrors the live wrapper's
/// `write_calibration_bootstrap` (W): a fresh file also gets a self-consistent top level (this mean +
/// default band + this window); an existing top level is left intact. FAIL-CLOSED on malformed
/// `existing` (never clobber other targets by discarding an unreadable file). The per-target entry
/// records its own `decode_tokens` (no inherit) + depth so the later band check can bind the window.
///
/// Offline: pure JSON assembly; the caller writes the returned bytes atomically. The on-box session
/// accumulation (sessions[], pair-weighted republish, provisional-until-N) is not modeled here.
// UNVERIFIED(measure-job): the on-box multi-session accumulation / provisional-until-CALIBRATION_MIN_SESSIONS
// behaviour is a live-wrapper detail; offline we author a single provisional entry.
pub struct BootstrapAuthorInput<'a> {
    /// The `--target-id` the entry is keyed under (bootstrap requires one).
    pub target_id: &'a str,
    /// #105 cycle-5 — the SERIES this run measured in, authored as the file's required `timed_mode`
    /// so the authored band round-trips through [`enforce_calibration_series_fence`] instead of
    /// failing its own fence on the next run. Merging into an existing file authored in a DIFFERENT
    /// series is refused (a file holds ONE series' bands).
    pub timed_mode: &'a str,
    /// #105 cycle-5 — the run's resolved track id, authored as the file's required `track_id`.
    /// Merging into a file authored for a different track is refused.
    pub track_id: &'a str,
    /// The pooled serial mean the run measured (the banded quantity).
    pub pooled_serial_mean: f64,
    /// The run's decode window; recorded as the entry's `decode_tokens` (no inherit).
    pub tokens: usize,
    /// The run's declared MTP depth knob; `None` (engine-decides) authors NO `mtp_depth` key.
    pub mtp_depth: Option<usize>,
    pub serial_control_depth: usize,
    pub pairs_total: usize,
}

pub fn build_bootstrap_calibration(
    existing: Option<&[u8]>,
    input: &BootstrapAuthorInput<'_>,
) -> Result<String, String> {
    use serde_json::{json, Map, Value};

    let BootstrapAuthorInput {
        target_id,
        timed_mode,
        track_id,
        pooled_serial_mean,
        tokens,
        mtp_depth,
        serial_control_depth,
        pairs_total,
    } = *input;

    if !(pooled_serial_mean.is_finite() && pooled_serial_mean > 0.0) {
        return Err(format!(
            "refusing to author a calibration from a non-finite/non-positive serial mean ({pooled_serial_mean})"
        ));
    }

    let mut root: Map<String, Value> = match existing {
        Some(bytes) if !bytes.iter().all(u8::is_ascii_whitespace) => {
            match serde_json::from_slice(bytes) {
                Ok(Value::Object(m)) => m,
                Ok(_) => {
                    return Err(
                        "existing BASELINE_CALIBRATION is not a JSON object; refusing to \
                            clobber it during bootstrap authoring"
                            .to_string(),
                    )
                }
                Err(e) => {
                    return Err(format!(
                        "existing BASELINE_CALIBRATION is malformed ({e}); refusing to clobber it \
                     during bootstrap authoring (fix or remove it first)"
                    ))
                }
            }
        }
        _ => Map::new(),
    };

    // #105 cycle-5 — the file's SERIES + TRACK are file-wide identity, not per-target: refuse to
    // merge a band measured in this series/track into a file that declares another one. Silently
    // preserving the existing tag would author a target whose mean was measured under a series the
    // file does not claim — exactly the mislabeling the fence exists to catch, manufactured by the
    // authoring path itself. (A fresh/tag-less file simply adopts this run's identity below.)
    for (key, run_value) in [("timed_mode", timed_mode), ("track_id", track_id)] {
        if let Some(existing_value) = root.get(key).and_then(Value::as_str) {
            if existing_value.trim() != run_value {
                return Err(format!(
                    "existing BASELINE_CALIBRATION declares {key} {:?} but this run authored under \
                     {run_value:?}; refusing to merge a band across {key}s (a calibration file holds \
                     one series/track — author the new one at a separate path)",
                    existing_value.trim()
                ));
            }
        }
    }
    // Delta re-review HIGH: adopting this run's identity is legal ONLY for a genuinely fresh
    // file. A non-empty file with NO series tag is pre-fence legacy — its existing target means
    // were measured under an undeclared (native) regime, and stamping this run's tag onto them
    // launders every preserved entry into the new series. Refuse; the operator authors the new
    // series at a fresh path (or deletes the legacy file explicitly).
    let has_substance = root.keys().any(|k| k != "timed_mode" && k != "track_id");
    if has_substance && (root.get("timed_mode").is_none() || root.get("track_id").is_none()) {
        return Err(
            "existing BASELINE_CALIBRATION is non-empty but carries no series identity \
             (timed_mode/track_id) — a pre-fence legacy file; refusing to adopt it into this \
             run's series (its existing entries were measured under an undeclared regime). \
             Author the new-series calibration at a fresh path, or remove the legacy file \
             explicitly first."
                .to_string(),
        );
    }
    root.insert("timed_mode".to_string(), json!(timed_mode));
    root.insert("track_id".to_string(), json!(track_id));

    // A fresh file gets a self-consistent top level; an existing one keeps its own.
    root.entry("serial_decode_seconds_per_token_mean")
        .or_insert_with(|| json!(pooled_serial_mean));
    root.entry("serial_band_low")
        .or_insert_with(|| json!(default_band_low()));
    root.entry("serial_band_high")
        .or_insert_with(|| json!(default_band_high()));
    root.entry("decode_tokens").or_insert_with(|| json!(tokens));

    let targets = root
        .entry("targets")
        .or_insert_with(|| Value::Object(Map::new()));
    let targets = targets
        .as_object_mut()
        .ok_or_else(|| "existing `targets` is not a JSON object".to_string())?;
    let mut entry = json!({
        "serial_decode_seconds_per_token_mean": pooled_serial_mean,
        "serial_band_low": default_band_low(),
        "serial_band_high": default_band_high(),
        "decode_tokens": tokens,
        "serial_control_depth": serial_control_depth,
        "provisional": true,
        "pairs_total": pairs_total,
    });
    // Absent-not-null: an engine-decides run (David ruling 2026-08-27) declared no depth knob, so
    // the calibration entry carries no `mtp_depth` key rather than a null or an invented value.
    if let Some(depth) = mtp_depth {
        entry["mtp_depth"] = json!(depth);
    }
    targets.insert(target_id.to_string(), entry);

    serde_json::to_string_pretty(&Value::Object(root))
        .map_err(|e| format!("calibration serialization failed: {e}"))
}

/// R13/R14 — write bootstrap-authored calibration bytes ATOMICALLY (temp sibling + rename), so a
/// crash never leaves a half-written calibration (mirrors the wrapper's temp+rename). The temp lives
/// in the destination directory so the rename is same-filesystem.
pub fn write_bootstrap_calibration(path: &std::path::Path, json: &str) -> Result<(), String> {
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
    let tmp = match dir {
        Some(d) => d.join(format!(
            ".{}.tmp.{}",
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("baseline-calibration.json"),
            std::process::id()
        )),
        None => std::path::PathBuf::from(format!(
            ".baseline-calibration.json.tmp.{}",
            std::process::id()
        )),
    };
    std::fs::write(&tmp, format!("{json}\n"))
        .map_err(|e| format!("could not write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("could not atomically install {}: {e}", path.display())
    })
}

/// R14 — resolve `GPU_LOADED_UTIL` (env, default 0.70): the util threshold the telemetry
/// "loaded/steady" definition uses. UNLIKE `GATE_TEMP`/`COOL_TIMEOUT` (fixed wrapper constants,
/// R21), this IS env-driven (live wrapper W:403). Returns the value + an honest source. FAIL-CLOSED
/// on a present-but-invalid value (non-finite / outside (0, 1]).
pub fn resolve_loaded_util(raw: Option<&str>) -> Result<(f64, &'static str), String> {
    match raw.map(str::trim).filter(|s| !s.is_empty()) {
        None => Ok((0.70, "env-GPU_LOADED_UTIL-default-0.70")),
        Some(s) => match s.parse::<f64>() {
            Ok(v) if v.is_finite() && v > 0.0 && v <= 1.0 => Ok((v, "env-GPU_LOADED_UTIL")),
            _ => Err(format!(
                "GPU_LOADED_UTIL must be a finite utilisation in (0, 1], got {s:?}"
            )),
        },
    }
}

/// R14 — the per-leg head directories. The SERIAL leg always uses the PINNED head; the candidate
/// leg uses its own (BYO), which DEFAULTS to the pinned head when unset. The actual head-into-verb
/// wiring is R15; this component resolves + existence-checks only.
///
/// David ruling (2026-08-26) — the SAME type now carries BOTH head families, resolved from their
/// own env pairs by two calls to the one [`resolve_head_dirs`]: the native-MTP head
/// (`QMTP_HEAD_DIR` / `QMTP_CANDIDATE_HEAD_DIR`) and the DFlash drafter
/// (`QMTP_DFLASH_HEAD_DIR` / `QMTP_CANDIDATE_DFLASH_HEAD_DIR`). One type, because the two families
/// have IDENTICAL semantics — pinned leg, BYO leg, BYO defaults to pinned — and a second struct
/// would have been the same three lines with a different name and a second place for the defaulting
/// rule to drift.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HeadDirs {
    pub head_dir: String,
    pub candidate_head_dir: String,
}

/// R14 — resolve one head family's dirs from its env values (PURE defaulting; the filesystem
/// existence check is the caller's). `None` when the PINNED env is unset — the head wiring is
/// deferred to R15, so the pinned head is only REQUIRED once the timed verb consumes it (and for
/// the DFlash family, only when the candidate declares mode `dflash` —
/// [`enforce_dflash_head_present`]).
pub fn resolve_head_dirs(head: Option<&str>, candidate: Option<&str>) -> Option<HeadDirs> {
    let head = head.map(str::trim).filter(|s| !s.is_empty())?;
    let candidate = candidate
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(head);
    Some(HeadDirs {
        head_dir: head.to_string(),
        candidate_head_dir: candidate.to_string(),
    })
}

// ---------------------------------------------------------------------------
// One timed leg's outcome: benchd's parent clock + the engine's WIRE echoes
// ---------------------------------------------------------------------------

/// R15 — one timed leg invocation's outcome: benchd's OWN parent-side wall clock (the scored
/// number), the engine's WIRE echoes (`effective_spec`, `head_provenance`, and the free-run §3
/// audit), plus the ONE cool-gate state the leg recorded (the single gate before the single timed
/// invocation). This is the seam the pure core drives: a test supplies mock echoes; the real path
/// (main.rs) spawns one `runtime-worker` process per leg with `--mtp-head` (+ the free-run v1.1
/// spawn gate) and reads every echo off the wire.
///
/// #109 window-2 finding 3 — the parsed `MtpTimedReport` that used to sit here is RETIRED with the
/// `--mtp-report` flag that produced it. Under the generic `runtime-worker` verb that file never
/// exists (and the flag killed the spawn), so every fact it carried now has exactly one source, the
/// wire: the head sha from the `hello`'s [`bench_protocol::HeadProvenance`], the effective spec from
/// the `decode_begin`/`free_decode_begin` echo, and the draft statistics from benchd's OWN histogram
/// math over the free-run audit.
pub struct LegInvocation {
    /// H1 (cycle-3) — benchd's OWN parent-side wall clock ÷ its configured token total for this leg.
    /// This is the ONLY scored seconds-per-token, and since #109 window-2 finding 3 retired the
    /// worker report file it is also the ONLY parent-clock number that exists at all — there is no
    /// second, worker-authored claim left to demote. Set by the caller (main.rs) from
    /// `run_decode_phase_fresh` / `run_free_run_decode_phase_fresh`'s own measured spt.
    pub benchd_seconds_per_token: f64,
    pub gate_state: GateState,
    /// The QUIESCENCE gate that ran immediately BEFORE this leg's cool gate: its state, its wait,
    /// and the load / GPU utilization it passed on (David 2026-09-17). Sealed beside the cool
    /// gate's state so a run's artifact shows the window opened on an IDLE box, not only a cold
    /// one.
    pub quiesce: QuiesceRecord,
    /// R16 — the ONE telemetry sample the leg's cool gate observed (peak GPU temp + steady loaded
    /// clock), folded run-wide into the sealed `telemetry`. `None` when no sample was available (the
    /// on-box telemetry stream is deferred — the top-level `telemetry` is then OMITTED honestly).
    pub telemetry: Option<TelemetrySample>,
    /// The WIRE engine-echoed `effective_spec` (`docs/spec-config-design.md`) benchd's runner
    /// captured from the timed `decode_begin`, ALREADY validated EQUAL to the requested spec
    /// (spec-never-ignored — a divergence discarded the session upstream, so a present leg's echo
    /// always matched). `None` for a no-spec (legacy) run. Sealed per leg as the effective spec.
    pub wire_effective_spec: Option<SpecConfig>,
    /// #109 window-2 finding 3 — the WIRE `head_provenance` the engine echoed on its `hello`
    /// (`bench_runner::Hello::head_provenance`): the sha256 of the head bytes it ACTUALLY loaded,
    /// plus size/shard count. This replaces the retired report file's `head_provenance_sha256` as the
    /// candidate leg's head-identity source — the field the engine's usage text calls part of the
    /// v1.1 speculative surface (`--speculative-protocol v1.1` "opts the hello into … spec_modes /
    /// capabilities / head_provenance + effective_spec echoes"), proven live in window 2's
    /// `FREERUN_ACCEPTED` isolation. `None` when the engine omitted it. AUDIT/provenance only, never
    /// scored; a candidate leg without it FAILS CLOSED in [`validate_leg_report`].
    pub wire_head_provenance: Option<bench_protocol::HeadProvenance>,
    /// W3 — the TIMED REGIME this leg actually ran, set by the caller from which runner entry point
    /// it drove (`run_decode_phase_fresh` ⇒ teacher-forced, `run_free_run_decode_phase_fresh` ⇒
    /// v1.1 free-run). Sealed per leg so no downstream consumer has to infer the measured quantity.
    pub regime: LegRegime,
    /// W3 — the §3 AUDIT of a v1.1 free-run leg (the verbatim per-round `acceptance_lengths` plus
    /// the derived `audit_spec_*` family), produced by the runner ONLY after the §2.6 consistency
    /// TRIPLE passed at the phase-close barrier. REQUIRED on a free-run leg (a missing audit fails
    /// the leg closed — benchd never fabricates it) and `None` on a teacher-forced leg, which has no
    /// acceptance to report. AUDIT ONLY: nothing here is ever an input to the score.
    pub free_run_audit: Option<FreeRunAudit>,
    /// B2 — benchd's own SPLIT of this leg's timed window at the `free_decode_begin` /
    /// `free_decode_run` boundary (`bench_runner::FreeRunPhaseTiming::phase_window`,
    /// `docs/scored-regime-and-prefill-window.md`). `Some` on a free-run leg, `None` on a
    /// teacher-forced leg, which drives neither verb and so opens no prefill window.
    ///
    /// It is benchd's clock throughout — no engine reports a duration on this protocol. What is
    /// DONE with it is the track's declared regime's decision: report-only when the track scores no
    /// prefill, a certified input when it does.
    pub phase_window: Option<bench_core::prefill_window::PhaseWindow>,
    /// COHORT (batch-8 brief §4.5) — the cohort AUDIT of a v1.2 BATCHED free-run leg, produced by
    /// the runner ONLY after the cohort consistency QUADRUPLE passed at the phase-close barrier.
    /// REQUIRED on a batched leg (a missing audit fails the leg closed) and `None` on every other
    /// regime. Its per-round common-width base is the SAME [`FreeRunAudit`] shape the single-stream
    /// regime produces; the cohort vectors (per-stream natural walks, active streams, depth-clamp
    /// reasons) ride along as sealed diagnostics. AUDIT ONLY: never an input to the score.
    pub cohort_audit: Option<bench_core::free_run::CohortFreeRunAudit>,
    /// COMPOSITE (Gemma cohort scoring) — this leg's PREFILL/DECODE phase-split window, from
    /// [`bench_runner::BatchedFreeRunPhaseTiming`]'s phase fields. SCORED INPUT under the
    /// SHARED-WINDOW ruling: these two parent-clocked elapsed times are what
    /// [`shared_window_composite`] sums into the composite's two gains. They are still NOT the
    /// ENFORCED `seconds_per_token`, which stays the whole-window figure computed by
    /// `bench_runner` itself (see that struct's doc for the red-team revert that pinned it there);
    /// the composite is a SECOND published quantity over the same parent clock, never a
    /// re-derivation of the enforced one. REQUIRED on the v1.2 batched cohort regime (a missing
    /// value fails the leg closed in [`validate_leg_report`], the same posture as `cohort_audit`)
    /// and `None` on every other regime — the v1.1 single-stream path has no second window to seal
    /// (its seed forward stays folded inside the one timed window, `prefill_component: "none"`,
    /// untouched by this ruling).
    pub cohort_phase_windows: Option<CohortPhaseWindows>,
    /// REPORT-ONLY (per-stream arm-fill lane, gap G2) — the per-stream timing CARRY the batched
    /// runner lifted off the wire ([`bench_runner::BatchedFreeRunPhaseTiming`]'s per-stream
    /// fields, PR-A), threaded VERBATIM so the attestation seal can run per accepted leg. `Some`
    /// exactly when the leg ran the v1.2 batched verbs (the only wire surface that carries the
    /// vectors); `None` everywhere else. DELIBERATELY UNVALIDATED in [`validate_leg_report`]:
    /// this is report-only cargo — its structural defects seal as a named `attestation_refused`
    /// reason on the pair record (never a leg rejection), so its presence rule is not enforced
    /// the way `cohort_audit`'s is. UNTRUSTED for scoring (parent-clock doctrine): nothing
    /// enforced reads it.
    pub per_stream_timing: Option<PerStreamTimingCarry>,
    /// (b) admission — the candidate cohort leg's COMMITTED tokens (`B` inner arrays of `N`, SLOT
    /// ORDER), surfaced UNJUDGED from the runner's [`bench_runner::BatchedFreeRunPhaseTiming`]. The
    /// runner no longer dies inline on a static-tape token mismatch; benchd's post-run TRUSTED-ORACLE
    /// tolerance gate replays THIS journal over the organizer's reference weights and applies the
    /// ≤10% per-stream bar. REQUIRED on the batched candidate leg (the tolerance gate has nothing to
    /// judge without it) and `None` on every other regime / the serial control leg (only the
    /// candidate journal is judged). Set by the caller (main.rs) from the batched runner result.
    pub cohort_committed_tokens_by_stream: Option<Vec<Vec<i64>>>,
}

/// COMPOSITE (Gemma cohort scoring) — one leg's PREFILL/DECODE window split: the two contiguous
/// elapsed times ([`bench_runner::BatchedFreeRunPhaseTiming`]'s anti-cheat invariant — no untimed
/// gap between them) and the token totals each divides.
///
/// PARENT CLOCK, and SCORED under the SHARED-WINDOW ruling: both figures are benchd's own
/// `Instant::now()` brackets around the batched verbs, and they are the composite's ONLY numeric
/// input ([`shared_window_composite`]). The ENFORCED `seconds_per_token` is a separate quantity —
/// the WHOLE window (`prefill_elapsed_seconds + decode_elapsed_seconds`), computed by
/// `bench_runner` itself and never reconstructed from these split-out halves — and this ruling
/// does not touch it. Carried on [`LegInvocation`] / [`LegMeasurement`] /
/// [`PairCohortPhaseWindows`] verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CohortPhaseWindows {
    /// Parent-clock elapsed seconds over `free_decode_begin` (the B-seed prefill), oracle checks
    /// included (the anti-cheat invariant: no untimed gap before the decode window opens).
    pub prefill_elapsed_seconds: f64,
    /// Parent-clock elapsed seconds over `free_decode_run` (the N-token free-run decode).
    pub decode_elapsed_seconds: f64,
    /// The B streams' seed (prompt) token counts, summed — "the 8 seeds' prompt tokens".
    pub prefill_token_total: usize,
    /// `B * N` committed decode tokens.
    pub decode_token_total: usize,
}

impl From<&bench_runner::BatchedFreeRunPhaseTiming> for CohortPhaseWindows {
    fn from(t: &bench_runner::BatchedFreeRunPhaseTiming) -> Self {
        Self {
            prefill_elapsed_seconds: t.prefill_elapsed_seconds,
            decode_elapsed_seconds: t.decode_elapsed_seconds,
            prefill_token_total: t.prefill_token_total,
            decode_token_total: t.decode_token_total,
        }
    }
}

/// REPORT-ONLY (per-stream arm-fill lane, gap G2) — the per-stream timing evidence one batched
/// leg carried, lifted VERBATIM from [`bench_runner::BatchedFreeRunPhaseTiming`]'s PR-A fields
/// (see those fields' docs for the G1/G3 verbatim-carry doctrine). This is the attestation
/// seal's INPUT bundle, not a validated claim: no shape/zero checks happen on the way here —
/// `bench_core::per_stream_attestation::attest_leg` is the one place that judges these values,
/// and its clause (a)/(b) refusals seal as `attestation_refused` on the pair record rather than
/// rejecting anything.
#[derive(Debug, Clone)]
pub struct PerStreamTimingCarry {
    /// Engine-reported per-slot monotonic ns for the cohort-prefill phase (`None` = the response
    /// carried no vector).
    pub prefill_ns_by_stream: Option<Vec<u64>>,
    /// Engine-reported per-slot monotonic ns for the decode phase (same absence convention).
    pub decode_ns_by_stream: Option<Vec<u64>>,
    /// K_slot per slot, VERBATIM from the consistency-validated response rectangle (G3 — never a
    /// `[N; B]` reconstruction).
    pub tokens_len_by_stream: Vec<usize>,
    /// Whether the engine's hello advertised the `per_stream_timing` capability — lets the seal
    /// distinguish "not advertised" (attestation honestly absent) from "advertised but absent"
    /// (a wiring bug, sealed as a refusal).
    pub advertised: bool,
}

impl From<&bench_runner::BatchedFreeRunPhaseTiming> for PerStreamTimingCarry {
    fn from(t: &bench_runner::BatchedFreeRunPhaseTiming) -> Self {
        Self {
            prefill_ns_by_stream: t.prefill_ns_by_stream.clone(),
            decode_ns_by_stream: t.decode_ns_by_stream.clone(),
            tokens_len_by_stream: t.tokens_len_by_stream.clone(),
            advertised: t.per_stream_timing_advertised,
        }
    }
}

// ---------------------------------------------------------------------------
// Pair-loop core
// ---------------------------------------------------------------------------

/// R15 — one accepted leg's VALIDATED measurement: the scored seconds-per-token (benchd's own
/// parent clock), the ONE cool-gate state, the engine-echoed effective spec (sealed as fact), and
/// the head the engine loaded (the candidate/MTP leg's wire `head_provenance`).
#[derive(Debug, Clone)]
struct LegMeasurement {
    /// H1 (cycle-3) — the ONLY scored number: benchd's OWN parent-side wall clock
    /// ([`LegInvocation::benchd_seconds_per_token`]), NOT any worker-authored value.
    seconds_per_token: f64,
    gate_state: GateState,
    /// The leg's quiescence-gate record (set by the caller from the `LegInvocation`, like
    /// `gate_state`).
    quiesce: QuiesceRecord,
    /// #109 window-2 finding 3 — the head the engine loaded, from the WIRE `hello`
    /// (`head_provenance.sha256`); `None` on a leg whose hello omitted the object (the v1-only
    /// surface). Required on the candidate leg.
    head_provenance_sha256: Option<String>,
    /// The effective mean draft length. Free-run legs only, and COMPUTED BY BENCHD from the
    /// acceptance histogram it collected (`None` on a teacher-forced leg, which cannot draft).
    effective_mean_draft_len: Option<f64>,
    /// The non-drafting round count — same source and same regime restriction as above.
    non_drafting_round_count: Option<usize>,
    /// R16 — the ONE telemetry sample the leg's cool gate observed (set by the caller from the
    /// `LegInvocation`, like `gate_state`).
    telemetry: Option<TelemetrySample>,
    /// R16 (medium cycle-3) — the number of attempts this leg took (1, or 2 after its one gated
    /// retry). Set by [`run_leg_with_retry`] on success; sealed as the pair's `*_attempts`.
    attempts: usize,
    /// The effective spec this leg ran. On a FREE-RUN leg it is the WIRE `effective_spec`
    /// (`docs/spec-config-design.md`) the runner captured + validated never-ignored; on a TF leg it is
    /// the serial spec DERIVED from the gate-off spawn surface (coordinator ruling #109 leg B).
    effective_spec: SpecConfig,
    /// Which of those two the value above came from
    /// ([`EFFECTIVE_SPEC_SOURCE_WIRE_ECHO`] / [`EFFECTIVE_SPEC_SOURCE_GATE_OFF_V1_SPAWN`]), sealed per
    /// leg so a reader never has to guess whether a sealed regime was measured or derived.
    effective_spec_source: &'static str,
    /// W3 — the TIMED REGIME this leg ran (sealed per leg as `*_timed_mode`).
    regime: LegRegime,
    /// W3 — the §3 free-run AUDIT (`None` on a teacher-forced leg). Validated present on a free-run
    /// leg, with `verified_token_count == N`.
    free_run_audit: Option<FreeRunAudit>,
    /// B2 — the leg's certified (or report-only) prefill/decode window split.
    phase_window: Option<bench_core::prefill_window::PhaseWindow>,
    /// B2 — whether that window was actually held to the certification checks, i.e. whether the
    /// track's declared regime scores prefill. Sealed per pair so a reader never has to infer
    /// whether the number in front of them was enforced or merely recorded.
    phase_window_certified: bool,
    /// B2 — whether a SECOND measurement of the same window corroborated benchd's clock
    /// (`bench_core::prefill_window::PREFILL_WINDOW_CROSS_CHECK_*`). A separate fact from
    /// `phase_window_certified`: today it is always `not-observed`, because no message on this
    /// protocol carries a duration, so an armed pass is self-consistent and NOT corroborated.
    phase_window_cross_check: &'static str,
    /// COHORT — the cohort AUDIT (`None` off the batched regime). Validated present on a batched
    /// leg, with the base covering exactly the per-stream N.
    cohort_audit: Option<bench_core::free_run::CohortFreeRunAudit>,
    /// COMPOSITE (Gemma cohort scoring) — this leg's phase-split window (`None` off the batched
    /// regime). Validated present on a batched leg (mirrors `cohort_audit`'s presence rule).
    cohort_phase_windows: Option<CohortPhaseWindows>,
    /// REPORT-ONLY (gap G2) — the per-stream timing carry, PASSED THROUGH from
    /// [`LegInvocation::per_stream_timing`] with no validation (see that field's doc for why the
    /// report-only posture forbids a presence/shape gate here). Consumed by
    /// [`per_stream_attestation_seal`] on the accepted path only.
    per_stream_timing: Option<PerStreamTimingCarry>,
    /// (b) admission — the CANDIDATE cohort leg's committed tokens (SLOT ORDER), the journal the
    /// trusted-oracle tolerance gate replays + judges. `None` on the serial control leg and off the
    /// batched regime. Carried on the measurement so [`run_pair`]'s post-run gate reads the candidate
    /// journal without re-plumbing the runner result.
    cohort_committed_tokens_by_stream: Option<Vec<Vec<i64>>>,
}

/// B2 — one accepted pair's two clock SPLITS, one per leg. Every number is benchd's own parent
/// clock and benchd's own configured token counts; no engine reports a duration on this protocol.
///
/// The gains a prefill-scoring regime computes are ratios over these:
/// `prefill_gain = sum(serial_prefill_window_seconds) / sum(candidate_prefill_window_seconds)`,
/// and likewise for decode. Whether they are computed at all is the declared regime's decision —
/// see `certified`.
#[derive(Debug, Clone, Serialize)]
pub struct PairPhaseWindows {
    /// The serial-control leg's seed-prefill window, in seconds.
    pub serial_prefill_window_seconds: f64,
    /// The candidate leg's seed-prefill window, in seconds.
    pub candidate_prefill_window_seconds: f64,
    /// The serial-control leg's free-run decode window, in seconds.
    pub serial_decode_window_seconds: f64,
    /// The candidate leg's free-run decode window, in seconds.
    pub candidate_decode_window_seconds: f64,
    /// Tokens the prefill windows covered (the seed length; the same on both legs).
    pub prefill_token_total: usize,
    /// Tokens the decode windows covered (N; the same on both legs).
    pub decode_token_total: usize,
    /// Whether the track's declared regime SCORES prefill, so these windows were held to
    /// `bench_core::prefill_window::certify_prefill_window`'s checks. `false` means report-only:
    /// benchd measured the split and enforced nothing on it, because nothing scores it.
    pub certified: bool,
    /// The SECOND fact, sealed as a NAME rather than inferred from a boolean:
    /// `"not-observed"` when no independent measurement of these windows existed to compare
    /// benchd's clock against, `"agreed"` when one existed and agreed inside the tolerance. (A
    /// disagreement never seals — it refuses.)
    ///
    /// `certified: true, cross_check: "not-observed"` is the honest description of an armed pass
    /// today: the halves account for the whole window benchd measured, and nothing else confirmed
    /// it. That is SELF-CONSISTENT, not corroborated, and the record says so instead of letting a
    /// single `certified` flag imply more than was checked.
    pub cross_check: &'static str,
}

/// One accepted pair's per-pair record (design note `results.pairs[]`). `parity_ok` is true by
/// construction — a token mismatch rejects the pair before it can be accepted.
#[derive(Debug, Clone, Serialize)]
pub struct PairRecord {
    pub parity_ok: bool,
    /// R15 — the serial-control leg's scored seconds-per-token = its report's
    /// `parent_measured_seconds_per_token` (the parent's wall-clock ÷ token total; worker
    /// self-timing is never scored).
    pub serial_seconds_per_token: f64,
    /// R15 — the candidate (MTP) leg's scored `parent_measured_seconds_per_token`.
    pub mtp_seconds_per_token: f64,
    pub order: String,
    /// The raw serial-relative ratio for this pair (serial / mtp), a diagnostic. NOT floored
    /// or ceilinged here (scoring is A-3). Finite by construction (implausible pairs rejected).
    pub raw_ratio: f64,
    /// R16 (medium cycle-3) — the pair's raw ratio under the LIVE per-pair field name `speedup`
    /// (W:2045-2046, `[$pairs[].speedup]`). Same value as `raw_ratio`, sealed under the wrapper's name.
    pub speedup: f64,
    /// R16 (medium cycle-3) — this pair's position in the pool prompt loop + the golden's sha256
    /// (live per-pair `prompt_index`/`prompt_sha256`, W:2047-2048). Stamped in `build_results` where
    /// the prompt is known.
    pub prompt_index: usize,
    pub prompt_sha256: String,
    /// R16 (medium cycle-3) — the attempts each leg took (1, or 2 after its one gated retry). Live
    /// per-pair `serial_attempts`/`mtp_attempts` (W:2046). Real counts from `run_leg_with_retry`.
    pub serial_attempts: usize,
    pub mtp_attempts: usize,
    /// R16 (medium cycle-3) — the live per-pair `serial_first_block_seconds`/`mtp_first_block_seconds`
    /// (W:2047). This is an on-box sub-interval the sampled decode journal produces; the current
    /// offline path has no such sample, so these are OMITTED (honest null), never fabricated.
    // UNVERIFIED(measure-job): the on-box first-block sub-interval is not emitted by the current worker.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub serial_first_block_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp_first_block_seconds: Option<f64>,
    /// finding 1: the recorded thermal-gate state for each leg (fired / waited /
    /// skipped-no-reader), so the run carries provenance of whether thermal enforcement happened.
    /// R15 — ONE cool gate per leg now (a single `mtp-timed` invocation), so this is that one
    /// gate's state directly, no prefill/decode fold.
    pub serial_gate_state: String,
    pub candidate_gate_state: String,
    /// The QUIESCENCE-gate state of each leg (fired / waited / skipped-no-reader), with the wait
    /// and the readings it passed on: the 1-minute load average and the GPU utilization fraction.
    /// Same three spellings and the same per-leg shape as the cool gate's state above, because the
    /// two gates guard the same window — quiescence first, then cool (David 2026-09-17). The
    /// readings are OMITTED on a skip, where nothing was observed.
    pub serial_quiesce_state: String,
    pub candidate_quiesce_state: String,
    pub serial_quiesce_waited_seconds: u64,
    pub candidate_quiesce_waited_seconds: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub serial_quiesce_load: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_quiesce_load: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub serial_quiesce_gpu_util: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_quiesce_gpu_util: Option<f64>,
    /// The effective spec each leg actually ran. Sealed as fact, NEVER the declared value — the
    /// candidate's DECLARED mtp spec is `results.candidate_spec` provenance. On a free-run pair this
    /// is the engine's WIRE `effective_spec` ([`SpecConfig`]) the runner captured + validated
    /// never-ignored; on a teacher-forced pair both legs are `{"mode":"serial"}` derived from the
    /// gate-off spawn surface. Which of the two is stated per leg by `*_effective_spec_source` —
    /// read them together, never the spec alone.
    pub serial_effective_spec: SpecConfig,
    pub candidate_effective_spec: SpecConfig,
    /// **Coordinator ruling (#109, leg B)** — the PROVENANCE of the two specs above, per leg:
    /// [`EFFECTIVE_SPEC_SOURCE_WIRE_ECHO`] (the engine said so, and benchd validated the echo against
    /// the request) or [`EFFECTIVE_SPEC_SOURCE_GATE_OFF_V1_SPAWN`] (benchd spawned the worker without
    /// the v1.1 gate, so its window is serial by construction and NO echo exists — nor may one, see
    /// [`tf_regime_is_serial`]). Sealed because a `{"mode":"serial"}` that was MEASURED and one that
    /// was DERIVED are different evidentiary claims, and the record should not blur them.
    pub serial_effective_spec_source: &'static str,
    pub candidate_effective_spec_source: &'static str,
    /// R15 — the candidate (MTP) leg's `head_provenance.sha256` (the declared BYO head the engine
    /// loaded), sealed per pair. #109 window-2 finding 3 — sourced from the engine's `hello` on the
    /// WIRE; present by construction (a candidate leg whose hello omits it rejects the pair).
    pub head_provenance_sha256: String,
    /// The candidate leg's effective mean draft length, COMPUTED BY BENCHD from the free-run
    /// acceptance histogram. `0.0` on a teacher-forced pair, where teacher forcing feeds every token
    /// and no drafting can occur. Sourced into per_prompt + the aggregate.
    pub effective_mean_draft_len: f64,
    /// The candidate leg's non-drafting round count — same source and same TF value as above.
    pub non_drafting_round_count: usize,
    /// W3 — the §5 SERIES TAG each leg's number belongs to. Sealed PER LEG (not only run-wide) so
    /// the record is self-describing: any consumer re-deriving a ratio from `serial_seconds_per_token
    /// / mtp_seconds_per_token` can check from THIS record that the two numbers measure the same
    /// physical quantity, without trusting the run-wide descriptor above it.
    ///
    /// #108 (L3) — the pre-ruling wording here said a scored free-run run "pairs a teacher-forced
    /// serial control with a v1.1 free-run candidate", i.e. that the two tags on a record NORMALLY
    /// differ. The Fable same-series ruling ([`serial_control_regime_for`]) closed exactly that
    /// crossing: the control runs the candidate's regime, so on every run measure-job produces the
    /// two tags are EQUAL. They are still sealed per leg — as the evidence for that claim rather
    /// than as a warning about a crossing — and #108 (M1) makes them the SOURCE the run-wide
    /// descriptor is derived from, so a leg driven in the wrong series refuses the seal instead of
    /// being described by it.
    pub serial_timed_mode: &'static str,
    pub candidate_timed_mode: &'static str,
    /// W3 (§3, RULED OQ4) — the CANDIDATE leg's verbatim per-round `acceptance_lengths[]`, persisted
    /// as collected (not just the aggregates) so cross-run analysis and any re-check of the §2.6
    /// triple have the full histogram. OMITTED on a teacher-forced pair — never fabricated.
    /// AUDIT ONLY: never an input to the score.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audit_spec_acceptance_lengths: Option<Vec<u32>>,
    /// B2 — benchd's own SPLIT of each leg's timed window at the `free_decode_begin` /
    /// `free_decode_run` boundary (`docs/scored-regime-and-prefill-window.md`). OMITTED on a
    /// teacher-forced pair, whose legs drive neither verb and open no prefill window.
    ///
    /// REPORT-ONLY on this branch's track: `qwen3.8-27b-mtp-v1` declares
    /// `prefill_gain_exponent: 0.0`, so `certified` is `false` here and no scored value reads these
    /// numbers. They are recorded because benchd measured them, not because anything depends on them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_windows: Option<PairPhaseWindows>,
    /// W3 (§3, RULED OQ4) — the derived `audit_spec_*` family, FLATTENED onto this record so the keys
    /// carry the flat prefix the spec ruled (no nested `metrics.audit.mtp` object). Empty (no keys
    /// emitted at all) on a teacher-forced pair. AUDIT ONLY: nothing here is scored, and
    /// `audit_spec_acceptance_rate` / `audit_spec_drafted_total` remain engine-self-reported.
    #[serde(flatten)]
    pub audit_spec: BTreeMap<String, f64>,
    /// COHORT (batch-8 brief D6) — the candidate leg's per-stream PRE-`min` natural accept walks
    /// (B x R), sealed VERBATIM so straggler throttling is a visible fact rather than a silent
    /// fold into the common committed width. OMITTED on every non-batched pair. AUDIT ONLY —
    /// these are correlated readings from one window, never per-stream samples.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audit_cohort_natural_accepted_by_stream: Option<Vec<Vec<u32>>>,
    /// COHORT (D4) — streams still generating at each round (length R, non-increasing under the
    /// closed cohort), sealing the tail. OMITTED off the batched regime. AUDIT ONLY.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audit_cohort_active_streams_by_round: Option<Vec<u32>>,
    /// COHORT (D6) — the engine's depth-clamp reason histogram over the window, sealed VERBATIM:
    /// the evidence for whether the window ACTUALLY SPECULATED (a cohort clamped to depth zero is
    /// a legitimate outcome the record must be able to state). OMITTED off the batched regime.
    /// AUDIT ONLY.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audit_cohort_depth_clamp_reasons: Option<BTreeMap<String, u32>>,
    /// COMPOSITE (Gemma cohort scoring) — this pair's PREFILL/DECODE window split (both legs),
    /// RAW parent-clocked seconds. OMITTED on every non-batched pair (the v1.1 single-stream path
    /// is untouched).
    ///
    /// SCORED INPUT (SHARED-WINDOW ruling): these four numbers, summed across the accepted pairs,
    /// ARE the composite's two gains ([`crate::measure_job::shared_window_composite`]). The record
    /// still carries RAW WINDOWS ONLY — no per-pair ratio is computed or sealed here, because the
    /// ruled aggregate is a ratio of SUMS across pairs, which no single pair can state. Sealing
    /// the raw per-pair windows is what makes `per_cohort[].composite` INDEPENDENTLY RECOMPUTABLE
    /// from the artifact alone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cohort_phase_windows: Option<PairCohortPhaseWindows>,
    /// REPORT-ONLY (per-stream arm-fill lane, gap G2) — the SERIAL leg's sealed per-stream
    /// attestation: the verbatim engine-reported evidence plus either the clause (c)-(f)
    /// [`bench_core::per_stream_attestation::PerStreamAttestation`] verdict or a named
    /// `attestation_refused` reason (clause (a)/(b) structural failures — NOT a run failure, NOT
    /// a pair rejection; nothing scores on this data). OMITTED when the leg carried no
    /// per-stream channel at all (non-batched regime, or capability not advertised and no
    /// vectors on the wire). ADDITIVE diagnostic: every pre-existing field of this record is
    /// byte-identical with or without it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub serial_per_stream_attestation: Option<PerStreamAttestationSeal>,
    /// REPORT-ONLY (gap G2) — the CANDIDATE leg's sealed per-stream attestation; same posture as
    /// [`serial_per_stream_attestation`](Self::serial_per_stream_attestation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_per_stream_attestation: Option<PerStreamAttestationSeal>,
    /// REPORT-ONLY (gap G2) — the pair's
    /// [`bench_core::per_stream_attestation::composite_diagnostic`] over the two legs' sealed
    /// SUM aggregates, raised to the CERTIFIED exponent pair the contract declared
    /// ([`ScoredExponents::certify`]'s output, threaded through the cohort pair loop — never the
    /// raw code constants). Present only when BOTH legs sealed an Ok verdict AND the certified
    /// pair was available; UNSCORED and DELIBERATELY SO — `PerCohort::composite` is computed by
    /// [`shared_window_composite`] from the PARENT clock alone and never reads this field or any
    /// other engine-reported number. This diagnostic is the per-stream reading of the same
    /// quantity, kept for box calibration; the two agreeing is evidence, the two disagreeing is a
    /// calibration finding, and NEITHER moves the score.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_stream_composite_diagnostic:
        Option<bench_core::per_stream_attestation::PerStreamCompositeDiagnostic>,
    /// REPORT-ONLY (NEAR-TIE STATS SEAL) — this pair's [`CohortNearTieSeal`]: the near-tie
    /// characterization of the trusted oracle's replay report, sealed as the SIBLING of the ≤10%
    /// tolerance outcome that the same gate reached over the same report. OMITTED off the batched
    /// cohort regime (the single-stream path runs no trusted-oracle gate at all, so there is no
    /// replay report to characterize). ADDITIVE: every pre-existing field of this record is
    /// byte-identical with or without it, and NOTHING in `candidate_accepted` or any
    /// [`RejectClass`] reads it — see [`CohortNearTieSeal`]'s structural report-only note.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cohort_near_tie_seal: Option<CohortNearTieSeal>,
}

/// COMPOSITE (Gemma cohort scoring) — one accepted pair's PREFILL/DECODE window split across BOTH
/// legs: the SHARED parent clock over the whole concurrent B-stream cohort, raw elapsed seconds +
/// token totals. NO ratio and NO gain is computed here, deliberately: the ruled aggregate is a
/// ratio of SUMS across the accepted pairs ([`crate::measure_job::shared_window_composite`]), and
/// a per-pair ratio is a DIFFERENT statistic that no consumer should be able to mistake for the
/// score. This struct's four seconds fields are the composite's raw material and the artifact's
/// recompute trail.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PairCohortPhaseWindows {
    pub serial_prefill_window_seconds: f64,
    pub candidate_prefill_window_seconds: f64,
    pub serial_decode_window_seconds: f64,
    pub candidate_decode_window_seconds: f64,
    /// The B streams' seed (prompt) token counts, summed. Sealed per pair for self-containment;
    /// constant across every accepted pair of one cohort run (same cohort, same window shape).
    pub prefill_token_total: usize,
    /// `B * N` committed decode tokens. Same constancy note as `prefill_token_total`.
    pub decode_token_total: usize,
}

impl PairCohortPhaseWindows {
    /// Assemble the raw window record from the two legs' phase-split windows.
    /// `finite_nonneg`-clamped (never a negative or non-finite value sealed), mirroring how
    /// `PairRecord::{serial,mtp}_seconds_per_token` are sealed elsewhere in this file. The clamp
    /// maps any degenerate input to `0.0`, which [`shared_window_composite`]'s positivity guard
    /// then REFUSES rather than scores — the clamp hides nothing from the composite.
    fn compute(serial: CohortPhaseWindows, candidate: CohortPhaseWindows) -> Self {
        Self {
            serial_prefill_window_seconds: finite_nonneg(serial.prefill_elapsed_seconds),
            candidate_prefill_window_seconds: finite_nonneg(candidate.prefill_elapsed_seconds),
            serial_decode_window_seconds: finite_nonneg(serial.decode_elapsed_seconds),
            candidate_decode_window_seconds: finite_nonneg(candidate.decode_elapsed_seconds),
            // Both legs time the SAME cohort/window shape, so their token totals agree by
            // construction; either side names the same fact.
            prefill_token_total: serial.prefill_token_total,
            decode_token_total: serial.decode_token_total,
        }
    }
}

/// REPORT-ONLY (per-stream arm-fill lane, gap G2) — one accepted leg's sealed per-stream
/// attestation: the verbatim inputs (so a review can recompute every clause without any other
/// record) plus EXACTLY ONE of `verdict` / `attestation_refused`.
///
/// The inputs sealed here are the same ones [`per_stream_attestation_seal`] fed
/// `bench_core::per_stream_attestation::attest_leg`: the raw wire vectors (G1, verbatim), K_slot
/// (G3, verbatim), R from the leg's cohort audit, and the leg's parent-clock phase windows (the
/// same two values the pair's `cohort_phase_windows` carries — duplicated here so ONE object is
/// recompute-sufficient). The box calibration pass (spec step 3) consumes `verdict`'s
/// `raw_ratio`s to derive ε/δ/δ′; nothing scored reads any of this.
#[derive(Debug, Clone, Serialize)]
pub struct PerStreamAttestationSeal {
    /// Whether the engine's hello advertised `per_stream_timing` (always true on a sealed
    /// `verdict`; a refusal can carry either value — `false` names the vectors-without-
    /// advertisement wiring anomaly).
    pub advertised: bool,
    /// The engine-reported per-slot prefill ns, VERBATIM (`None` = absent from the response).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_ns_by_stream: Option<Vec<u64>>,
    /// The engine-reported per-slot decode ns, VERBATIM (`None` = absent from the response).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_ns_by_stream: Option<Vec<u64>>,
    /// K_slot per slot, VERBATIM from the consistency-validated response rectangle (G3).
    pub tokens_len_by_stream: Vec<usize>,
    /// R — the benchd-observed round count (`CohortFreeRunAudit::rounds()`), the divisor of
    /// clause (e)'s `step_time`.
    pub rounds: usize,
    /// The leg's parent-clock PREFILL window (clause (c)/(d)'s prefill denominator).
    pub prefill_window_seconds: f64,
    /// The leg's parent-clock DECODE window (clause (c)/(d)/(e)'s decode denominator).
    pub decode_window_seconds: f64,
    /// The clause (c)-(f) verdict, when the inputs were structurally attestable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verdict: Option<bench_core::per_stream_attestation::PerStreamAttestation>,
    /// The NAMED clause (a)/(b) structural refusal
    /// ([`bench_core::per_stream_attestation::PerStreamAttestationError`]'s display form —
    /// missing vector when advertised, length mismatch, zero duration, degenerate window/rounds)
    /// when no verdict could be computed. Sealed on the pair record as evidence; NEVER a run
    /// failure and NEVER a pair rejection — no existing check tightens or loosens on it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attestation_refused: Option<String>,
}

/// REPORT-ONLY (gap G2) — build one accepted leg's [`PerStreamAttestationSeal`], or `None` when
/// the leg has no per-stream channel to attest:
///
/// - `None` when the leg carried no [`PerStreamTimingCarry`] (non-batched regime), or no cohort
///   audit / phase windows (off the batched regime nothing supplies R or the windows — on the
///   validated batched path all three are `Some` by construction);
/// - `None` when the capability was NOT advertised and no vector arrived anyway — the honest
///   "engine has no instrumentation" case: attestation absent, run unaffected;
/// - otherwise `Some`, with `attest_leg`'s Ok verdict or its named clause (a)/(b) refusal.
///   (Vectors present WITHOUT the advertisement still attest — `attest_leg` refuses that as
///   clause (a), sealing the wiring anomaly by name instead of dropping the evidence.)
fn per_stream_attestation_seal(m: &LegMeasurement) -> Option<PerStreamAttestationSeal> {
    let carry = m.per_stream_timing.as_ref()?;
    let windows = m.cohort_phase_windows.as_ref()?;
    let audit = m.cohort_audit.as_ref()?;
    if !carry.advertised
        && carry.prefill_ns_by_stream.is_none()
        && carry.decode_ns_by_stream.is_none()
    {
        return None;
    }
    let outcome = bench_core::per_stream_attestation::attest_leg(
        bench_core::per_stream_attestation::PerStreamAttestationInputs {
            capability_advertised: carry.advertised,
            // B from the leg's OWN validated cohort audit (echoed-and-validated width, D9-pinned
            // to the certified batch point upstream in `validate_leg_report`).
            batch_size: audit.batch_size(),
            prefill_ns_by_stream: carry.prefill_ns_by_stream.as_deref(),
            decode_ns_by_stream: carry.decode_ns_by_stream.as_deref(),
            tokens_len_by_stream: &carry.tokens_len_by_stream,
            prefill_window_seconds: windows.prefill_elapsed_seconds,
            decode_window_seconds: windows.decode_elapsed_seconds,
            rounds: audit.rounds(),
        },
    );
    let (verdict, attestation_refused) = match outcome {
        Ok(v) => (Some(v), None),
        // Clause (a)/(b) — sealed BY NAME as evidence; the pair proceeds untouched (report-only:
        // nothing scores on this yet, so nothing may tighten or loosen on it).
        Err(e) => (None, Some(e.to_string())),
    };
    Some(PerStreamAttestationSeal {
        advertised: carry.advertised,
        prefill_ns_by_stream: carry.prefill_ns_by_stream.clone(),
        decode_ns_by_stream: carry.decode_ns_by_stream.clone(),
        tokens_len_by_stream: carry.tokens_len_by_stream.clone(),
        rounds: audit.rounds(),
        prefill_window_seconds: windows.prefill_elapsed_seconds,
        decode_window_seconds: windows.decode_elapsed_seconds,
        verdict,
        attestation_refused,
    })
}

/// REPORT-ONLY (NEAR-TIE STATS SEAL, David's ruling "hold 10% + seal near-tie stats", 2026-08-25)
/// — the per-pair measurement of the batched-oracle CROSS-STREAM ROUNDING CHANNEL that the audit
/// rated "real but low-exploitability, bounded by the raw-mismatch budget — bound ARGUED not
/// MEASURED". This seal is the MEASUREMENT that discharges that conditional.
///
/// One seal per ACCEPTED pair, because one pair IS one candidate cohort run and therefore ONE
/// trusted-oracle `cohort_reference_replay` report: the numbers describe THAT replay, and
/// averaging them across pairs would state a quantity no single replay produced. (A rejected pair
/// seals no [`PairRecord`] at all, so it carries no near-tie stats either — its rejection reason
/// already names the failing slot and its counts.)
///
/// The seal carries the oracle report's own provenance (`logit_provenance` / `logit_topk` /
/// `rel_envelope`, VERBATIM) plus EXACTLY ONE of `stats` / `near_tie_refused`, mirroring
/// [`PerStreamAttestationSeal`]'s posture: an engine that does not emit the AUDIT-ONLY gap fields
/// gets a NAMED refusal, never a fabricated zero.
///
/// ★ REPORT-ONLY, STRUCTURALLY. [`cohort_near_tie_seal`] is TOTAL — its return type has no error
/// variant, so no value computed here can reach [`RejectCtx`], [`RejectClass`], or
/// `candidate_accepted`. It is called at the very END of
/// [`cohort_token_tolerance_gate`], AFTER the N2 integrity check and the ≤10% tolerance verdict
/// are both final. The near-tie definition, its gap index, and the engine citation live in
/// [`bench_core::near_tie`].
#[derive(Debug, Clone, Serialize)]
pub struct CohortNearTieSeal {
    /// The oracle's logit provenance tag (e.g. `"post_softcap"`), VERBATIM. Provenance for the
    /// gaps: a relative gap only means what it says against the logit surface it was taken on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_provenance: Option<String>,
    /// The ranked readout depth K the oracle ran at, VERBATIM. K < 2 ⇒ no top-2 gap exists and the
    /// seal refuses by name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_topk: Option<u32>,
    /// The report-level `rel_envelope`, VERBATIM — the envelope the near-tie predicate was
    /// evaluated at. Restated here even when `stats` is present (where it also appears) so a
    /// REFUSED seal still shows what the oracle did or did not declare.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rel_envelope: Option<f64>,
    /// The measurement, when every position carried the gap fields it needs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<bench_core::near_tie::NearTieStats>,
    /// The NAMED reason no measurement could be stated (the oracle omitted `rel_envelope`, or a
    /// position omitted `ranked_relative_gaps` / `committed_relative_gap`, or emitted a top-K too
    /// shallow to carry a top-2 gap — the OLD-ENGINE case). Sealed as evidence; NEVER a run
    /// failure, NEVER a pair rejection, and the ≤10% tolerance gate is entirely unaffected — it
    /// works on `sequential_argmax` alone and needs none of these fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub near_tie_refused: Option<String>,
}

/// REPORT-ONLY — build one accepted pair's [`CohortNearTieSeal`] from the trusted oracle's report.
///
/// TOTAL BY CONSTRUCTION: there is no `Result` here, deliberately. Every failure mode — an oracle
/// that omits `rel_envelope`, an engine that does not emit `ranked_relative_gaps` or
/// `committed_relative_gap`, a top-K under 2, a structural shape [`bench_core::near_tie`] refuses
/// — becomes a NAMED `near_tie_refused` string on a seal that is still emitted. That is what makes
/// the report-only guarantee STRUCTURAL rather than a promise: this function has no way to reject
/// a pair even if it wanted to.
///
/// Slot order is the report's own (the caller already PINNED `stream.slot == i` as an integrity
/// precondition before this runs), so `stats.per_stream[i]` describes cohort member `i`.
fn cohort_near_tie_seal(report: &bench_protocol::CohortReferenceReplayReport) -> CohortNearTieSeal {
    let refuse = |reason: String| CohortNearTieSeal {
        logit_provenance: report.logit_provenance.clone(),
        logit_topk: report.logit_topk,
        rel_envelope: report.rel_envelope,
        stats: None,
        near_tie_refused: Some(reason),
    };
    // The envelope is REPORT-level: without it there is no near-tie predicate to evaluate.
    let Some(rel_envelope) = report.rel_envelope else {
        return refuse(
            "the trusted oracle's replay report omitted `rel_envelope`, so the near-tie predicate \
             (`ranked_relative_gaps[1] <= rel_envelope`) has no threshold — no near-tie statistic \
             is stated (the ≤10% tolerance gate is unaffected: it reads `sequential_argmax` only)"
                .to_string(),
        );
    };
    // Lift each position's OPTIONAL audit fields into the core's required readout, refusing BY
    // NAME at the first position that cannot supply one. A partial seal is never assembled: a
    // near-tie count over a subset of positions would understate the flippable set while looking
    // like a full measurement.
    let mut positions_by_stream: Vec<Vec<bench_core::near_tie::PositionGaps>> =
        Vec::with_capacity(report.streams.len());
    for stream in &report.streams {
        let mut positions = Vec::with_capacity(stream.positions.len());
        for (index, p) in stream.positions.iter().enumerate() {
            let Some(gaps) = p.ranked_relative_gaps.as_ref() else {
                return refuse(format!(
                    "the trusted oracle emitted no `ranked_relative_gaps` at slot {} position \
                     {index} (an engine without the ranked audit readout) — no near-tie statistic \
                     is stated rather than a fabricated one; the ≤10% tolerance gate is \
                     unaffected (it reads `sequential_argmax` only)",
                    stream.slot
                ));
            };
            let Some(&top2_relative_gap) = gaps.get(bench_core::near_tie::NEAR_TIE_GAP_INDEX)
            else {
                return refuse(format!(
                    "the trusted oracle's `ranked_relative_gaps` at slot {} position {index} has \
                     {} entries, with no index {} (the top-1→top-2 gap) — the ranked readout is \
                     too shallow to characterize a near-tie; none is stated",
                    stream.slot,
                    gaps.len(),
                    bench_core::near_tie::NEAR_TIE_GAP_INDEX
                ));
            };
            let Some(committed_relative_gap) = p.committed_relative_gap else {
                return refuse(format!(
                    "the trusted oracle emitted no `committed_relative_gap` at slot {} position \
                     {index} — the mismatch-depth statistics have no input; no near-tie statistic \
                     is stated",
                    stream.slot
                ));
            };
            positions.push(bench_core::near_tie::PositionGaps {
                committed_token: p.committed_token,
                sequential_argmax: p.sequential_argmax,
                top2_relative_gap,
                committed_relative_gap,
            });
        }
        positions_by_stream.push(positions);
    }
    match bench_core::near_tie::near_tie_stats(
        &positions_by_stream,
        rel_envelope,
        bench_core::constants::COHORT_TOKEN_TOLERANCE_PER_THOUSAND,
    ) {
        Ok(stats) => CohortNearTieSeal {
            logit_provenance: report.logit_provenance.clone(),
            logit_topk: report.logit_topk,
            rel_envelope: report.rel_envelope,
            stats: Some(stats),
            near_tie_refused: None,
        },
        // A shape the near-tie core refuses is one the TOLERANCE gate already refused upstream
        // (empty stream / no streams), so this arm is unreachable on the accepted path — but it is
        // an Err fold, NOT a `?`: the seal degrades to a named refusal, never to a rejection.
        Err(e) => refuse(format!("{e}")),
    }
}

/// One rejected pair, recorded for provenance (contributes NOTHING to the aggregate).
#[derive(Debug, Clone, Serialize)]
pub struct RejectRecord {
    pub order: String,
    pub class: String,
    pub leg: String,
    pub reason: String,
}

/// The full outcome of a measure-job run.
pub struct MeasureJobOutcome {
    pub results: Results,
    /// `accepted_pair_count >= min_pairs`. False ⇒ die 5 (candidate rejected), exit 5. Finding R19
    /// — this is the ONLY rejection verdict: EVERY reject class (thermal timeout included) folds
    /// into die-5 after its one gated retry; there is no distinct mid-pair infra/thermal exit.
    pub candidate_accepted: bool,
}

/// Build the depth-0 timing params for one pool prompt (the same workload both legs time),
/// with the decode window set by `--tokens`.
///
/// BOTH document shapes land on the SAME [`TimingParams`] decode fields, because both describe
/// the same thing — a seed prompt, the token its forward must produce, and the reference chain
/// each `decode_step` must produce:
///
/// | `TimingParams` field        | tape (live pool)               | legacy GoldenDocument            |
/// |-----------------------------|--------------------------------|----------------------------------|
/// | `decode_seed_tokens`        | `seed_tokens`                  | `benchmark.decode_seed_tokens`   |
/// | `expected_decode_seed_token`| `reference_seed_token`         | `benchmark.expected_decode_seed_token` |
/// | `expected_decode_tokens`    | `rows[i].sequential_argmax`    | `benchmark.expected_decode_tokens` |
///
/// The tape mapping is the live wrapper's own semantics: the reference driver opens the timed
/// window with `beginMTPDecode(seedTokens: golden.seedTokens)`, hard-fails when the seed
/// forward's token ≠ `referenceSeedToken` (`seed_token_mismatch`), and checks emitted index `i`
/// against `i == 0 ? referenceSeedToken : rows[i - 1].sequentialArgmax`. benchd's
/// `measure_decode` does exactly that with these three fields — seed forward oracle-checked
/// against `expected_decode_seed_token`, then step `i` teacher-forced from (and checked against)
/// `expected_decode_tokens`. Same fields, same order, same oracle.
///
/// PREFILL: `measure-job` legs run ONLY the timed decode window (one `mtp-timed` verb: the seed
/// prefill is INSIDE the decode clock, `prefill_component: "none"`), so the prefill fields are
/// never read on this path. A tape carries no prefill oracle and none is INVENTED — #112 (L1)
/// made that literally true: the tape branch builds [`TimingParams::decode_only`], which leaves
/// the prefill prompt EMPTY **and** `expected_prefill_token` `None`. (It previously set that
/// oracle from `tape.reference_seed_token` — a DECODE-window token standing in for a prefill
/// one; inert, because nothing on this path reads it, but not "no oracle invented".) A future
/// prefill phase on this path now fails loudly at BOTH guards — the window runner refuses
/// the empty prompt, `measure_prefill` refuses the missing oracle — rather than timing a
/// fabricated prompt against a borrowed expectation.
///
/// Errors when the prompt cannot oracle the requested window (a legacy golden with no benchmark
/// block, or either shape with fewer reference tokens than `--tokens`).
fn timing_params(prompt: &TimedPrompt, tokens: usize) -> Result<TimingParams, String> {
    match prompt {
        TimedPrompt::Tape(tape) => {
            // The tape oracles `rows.len()` post-seed tokens; benchd charges `--tokens`
            // `decode_step` calls and reads one row each. (The Swift driver counts the seed
            // argmax as emitted[0] and so wants N+1 rows for an N-token window; benchd's N
            // does not include the seed forward, so N rows is its honest requirement.)
            if tape.row_count() < tokens {
                return Err(format!(
                    "timed-prompt tape carries {} reference rows; --tokens {tokens} needs at least \
                     that many (rows[i] is the reference token emitted at index i+1)",
                    tape.row_count()
                ));
            }
            // #112 (L1) — DECODE-ONLY params: the tape has no prefill oracle, so none is set.
            // (This used to pass `reference_seed_token` as `expected_prefill_token` — a DECODE
            // oracle standing in for a prefill one, inert only because measure-job never runs a
            // prefill phase. `decode_only` leaves it `None` and `measure_prefill` refuses it.)
            Ok(TimingParams::decode_only(
                tape.seed_tokens.clone(),
                tape.reference_seed_token,
                tape.row_argmax_chain(),
                tokens,
            ))
        }
        TimedPrompt::Golden(golden) => {
            let b = golden.benchmark.as_ref().ok_or_else(|| {
                "benchmark golden file must contain a benchmark oracle".to_string()
            })?;
            if b.expected_decode_tokens.len() < tokens {
                return Err(format!(
                    "benchmark decode oracle has {} tokens; --tokens {tokens} needs at least that many",
                    b.expected_decode_tokens.len()
                ));
            }
            Ok(crate::iterate::benchmark_window_params(b, tokens))
        }
    }
}

/// COHORT (batch-8 brief §4.5) — build the ONE batched-window params object from the WHOLE cohort,
/// in slot order. Each slot's stream is derived by the SAME [`timing_params`] mapping the
/// single-stream path uses (tape and legacy GoldenDocument shapes alike), so the per-slot oracle
/// fields cannot drift from the one documented mapping; the per-slot objects are then folded into
/// a [`CohortTimingParams`] carrying the identical per-stream budget N (D4 — fixed, identical,
/// no refill, no EOS) and the EXPLICIT cohort width.
pub fn cohort_timing_params(
    prompts: &[TimedPrompt],
    tokens: usize,
) -> Result<CohortTimingParams, String> {
    let mut streams = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let p = timing_params(prompt, tokens)?;
        streams.push(bench_runner::CohortStreamParams {
            decode_seed_tokens: p.decode_seed_tokens,
            expected_decode_seed_token: p.expected_decode_seed_token,
            expected_decode_tokens: p.expected_decode_tokens,
        });
    }
    Ok(CohortTimingParams::new(streams, tokens))
}

/// The engine's WEIGHTLESS remedy that attaches a `.benchmark` oracle to a golden that lacks one
/// (engine `main.swift`). Named in the pre-gates refusal below so the operator is handed the fix,
/// not left to infer it from a generic "cannot oracle this window" message.
pub const ATTACH_BENCHMARK_ORACLE_REMEDY: &str = "attach-benchmark-oracle";

/// PRE-GATES guard (2b box-leg): a golden routed to the ranked/timed (gates) phase as a legacy
/// `GoldenDocument` MUST carry a `.benchmark` oracle — a `benchmark`/`official` window is TIMED
/// against that oracle. Goldens generated for correctness / local-iterate legitimately lack it
/// (correctness is oracle-optional), so this binds ONLY the `TimedPrompt::Golden` shape that reaches
/// the timed window; a `TimedPrompt::Tape` carries its reference rows directly and is exempt.
///
/// Without this, a benchmark-less golden still dies pre-GPU — but at the generic per-prompt
/// [`validate_prompt_windows`] refusal, which frames a MISSING ORACLE as a token-count problem
/// ("cannot oracle this run's N-token decode window"). Refuse EARLY and CLEARLY here instead, naming
/// the engine's weightless [`ATTACH_BENCHMARK_ORACLE_REMEDY`] so the operator attaches the oracle
/// rather than chasing a phantom window-length shortfall. Same die-8 pre-GPU point, actionable
/// message. Call BEFORE `validate_prompt_windows` so this specific diagnostic wins.
pub fn validate_gates_goldens_carry_oracle(prompts: &[TimedPrompt]) -> Result<(), String> {
    for prompt in prompts {
        if let TimedPrompt::Golden(g) = prompt {
            if g.benchmark.is_none() {
                return Err(format!(
                    "golden sha256 {} ({}) is routed to the ranked gates phase but carries no \
                     `.benchmark` oracle: a benchmark/official run is TIMED against that oracle. A \
                     golden generated for correctness/local-iterate does not carry one — attach it \
                     with the engine's weightless `{ATTACH_BENCHMARK_ORACLE_REMEDY}` remedy before \
                     feeding this golden to the gates phase (die 8, pre-GPU)",
                    g.sha256,
                    prompt.kind(),
                ));
            }
        }
    }
    Ok(())
}

/// #112 (M1) — prove EVERY loaded `--golden` can oracle THIS RUN'S decode window, as a PRE-GPU
/// check.
///
/// The rows-vs-window rule lived only inside [`timing_params`], which the pair loop calls per
/// prompt — i.e. AFTER `--preflight-only` has already returned 0. So an operator could prove a
/// tape pool "satisfiable" offline and then have the real run die on the first prompt because the
/// tape was shorter than the window. Running the SAME function over every prompt here closes that:
/// the check is by construction the one the pair loop applies (never a re-implementation that can
/// drift from it), and it is cheap and pure.
///
/// `tokens` is the RULED window and must be the value the legs will actually run:
/// * v1.1 FREE-RUN — N is FIXED at [`FREE_RUN_DECODE_TOKENS`]; [`validate_candidate_regime_coherent`]
///   has already refused any other `--tokens`, so `cfg.tokens` IS the ruled N here;
/// * TEACHER-FORCED — the window is `--tokens` as given.
///
/// Call this AFTER `validate_candidate_regime_coherent` (so the free-run window is already pinned)
/// and BEFORE the `--preflight-only` return.
pub fn validate_prompt_windows(prompts: &[TimedPrompt], tokens: usize) -> Result<(), String> {
    for prompt in prompts {
        timing_params(prompt, tokens).map_err(|e| {
            format!(
                "golden sha256 {} ({}) cannot oracle this run's {tokens}-token decode window: \
                 {e} — die 8 (pre-GPU: every --golden must carry enough reference tokens for the \
                 window its legs will time, or the run dies on the first prompt after the gate)",
                prompt.sha256(),
                prompt.kind(),
            )
        })?;
    }
    Ok(())
}

/// B2 — one leg's PREFILL-WINDOW certification, split out of [`validate_leg_report`] so both of its
/// failure modes are reachable from a test with a `track_id` argument, rather than only through the
/// one track this binary is built for.
///
/// TWO DISTINCT CLASSES, because they are two different faults:
///
/// * the WINDOW does not certify — [`RejectClass::PrefillWindowUncertified`], retryable, because it
///   is a per-attempt MEASUREMENT outcome: the retry's fresh worker re-measures the clock split.
///
/// ARMED BY THE REGIME ONLY: with `prefill_gain_exponent == 0.0` the certification is
/// `Ok(ReportOnly(..))` for EVERY input, so neither class can fire on a decode-only track.
fn certify_leg_prefill_window(
    leg: &'static str,
    track_id: &str,
    regime: &bench_core::constants::ScoredRegime,
    inv: &LegInvocation,
) -> Result<bench_core::prefill_window::PrefillWindowCertification, RejectCtx> {
    bench_core::prefill_window::certify_prefill_window(
        track_id,
        regime,
        inv.phase_window,
        // No message on this protocol carries an engine-reported duration, so there is no second
        // measurement of this window to cross-check. `None` is the honest input; a fabricated one
        // would certify benchd's clock against itself. The certification seals
        // `cross_check = not-observed` for exactly this reason.
        None,
    )
    .map_err(|reason| RejectCtx {
        class: RejectClass::PrefillWindowUncertified,
        leg,
        reason: format!("{leg} leg: {reason}"),
    })
}

/// R15 / H1 (cycle-3) — validate ONE leg's measurement, extracting the sealable facts or a typed
/// reject. The SCORED seconds-per-token is `benchd_seconds_per_token` — benchd's OWN parent-side
/// wall clock — NOT any worker/report-authored value. FAIL-CLOSED, never fabricating a missing value:
/// - `benchd_seconds_per_token` (the parent clock) must be finite and strictly positive (else
///   [`RejectClass::ImplausibleSpt`]);
/// - on a FREE-RUN leg the engine-echoed `effective_spec` must be PRESENT (else a reject — benchd
///   seals only the engine echo and NEVER a fabricated one); on a TF leg it must be ABSENT (see the
///   regime split below);
/// - on a FREE-RUN leg the candidate (MTP) leg's `hello` must carry a non-empty
///   `head_provenance.sha256` (the head the engine loaded), sealed per prompt; the free-run serial
///   control loads the pinned head, so its head sha is optional. #109 W3 finding 5 — a TEACHER-FORCED
///   leg neither requires nor accepts the field: it is spawned gate-off, the engine gates
///   `head_provenance` behind that flag, and a present object rejects the leg
///   ([`tf_hello_carries_no_head_provenance`]). A TF series seals no head identity.
///
/// The echoed depth is sealed VERBATIM — benchd does NOT reject a leg whose echoed depth differs
/// from the requested one (the whole point of the echo is to record what the engine ACTUALLY ran,
/// not what was requested). #109 window-2 finding 3 — the request side of that depth is now the wire
/// `spec` ALONE; the `--mtp-depth` argv channel is gone.
///
/// W3 — the checks are REGIME-AWARE (`inv.regime`):
/// - a TEACHER-FORCED leg must carry NO effective-spec echo ([`tf_regime_is_serial`], inverted under
///   the coordinator's leg-B ruling); its serial regime is sealed from the gate-off spawn surface;
/// - a v1.1 FREE-RUN leg must echo a SPECULATING regime ([`free_run_regime_is_speculative`]) and
///   MUST carry the §3 AUDIT the runner produced after the §2.6 triple passed, with
///   `verified_token_count == N`. Its draft statistics are then COMPUTED BY BENCHD from the
///   per-round `acceptance_lengths` it collected, NOT taken from an engine echo, so no engine claim
///   can launder a disagreement into the seal.
fn validate_leg_report(
    leg: &'static str,
    is_candidate: bool,
    n: usize,
    scored_regime: &bench_core::constants::ScoredRegime,
    inv: &LegInvocation,
) -> Result<LegMeasurement, RejectCtx> {
    let regime = inv.regime;
    let wire_effective_spec = inv.wire_effective_spec.clone();
    let free_run_audit = inv.free_run_audit.clone();
    // H1 (cycle-3) — the SCORED spt is benchd's OWN parent clock; the report's claim is audit only.
    let spt = inv.benchd_seconds_per_token;
    if !(spt.is_finite() && spt > 0.0) {
        return Err(RejectCtx {
            class: RejectClass::ImplausibleSpt,
            leg,
            reason: format!("{leg} leg benchd-measured seconds_per_token is implausible ({spt})"),
        });
    }
    // The leg's EFFECTIVE SPEC and where it came from — REGIME-SPLIT under the coordinator's leg-B
    // ruling, because the two regimes are spawned onto two different wire surfaces.
    let (effective_spec, effective_spec_source) = match regime {
        // TEACHER-FORCED — spawned GATE-OFF, requests no spec ([`requested_wire_spec`]), and a
        // gate-off worker is gated out of emitting `effective_spec`. So the absence of an echo is the
        // EXPECTED state and the serial regime is DERIVED FROM THE SPAWN SURFACE, a fact benchd owns
        // rather than one it has to take the engine's word for. `tf_regime_is_serial` is the tamper
        // check on the other side: any echo AT ALL rejects the leg, because this process could not
        // have produced one. (The candidate's DECLARED spec lives in provenance, never sealed here.)
        LegRegime::TeacherForcedV1 => {
            if let Err(reason) = tf_regime_is_serial(inv.wire_effective_spec.as_ref()) {
                return Err(RejectCtx {
                    class: RejectClass::NonSerialTfRegime,
                    leg,
                    reason: format!("{leg} leg: {reason}"),
                });
            }
            (
                timed_decode_wire_spec(),
                EFFECTIVE_SPEC_SOURCE_GATE_OFF_V1_SPAWN,
            )
        }
        // v1.1 FREE-RUN — the gate is ON, a spec was requested, and the echo is REQUIRED. FAIL-CLOSED
        // (R1's core rule): benchd seals ONLY the engine-echoed effective spec here, and a leg whose
        // echo is ABSENT is rejected — the value is NEVER fabricated. (The runner's never-ignored
        // check already discarded the session on a DIVERGENT echo, so a leg that reaches this seal
        // echoed a spec EQUAL to the request; here we additionally require it to be PRESENT.)
        //
        // W3 / Fable ruling — then the per-leg regime guard, which differs BY LEG because the two
        // legs run the same series for OPPOSITE reasons:
        //   * CANDIDATE — must echo a SPECULATING regime; a serial echo means the drafter did not run
        //     and the number is not a free-run-series candidate measurement.
        //   * SERIAL CONTROL — must echo `serial`; it free-runs at depth 0 precisely so the
        //     denominator is the same measured quantity with no speculation, and a non-serial echo
        //     means the control drafted.
        // Neither direction is a downgrade path: both refuse the leg fail-closed.
        //
        // COHORT — the batched regime shares this arm verbatim: its legs are spawned with the same
        // v1.1 gate, request the same wire spec (ONE spec for the whole cohort, D6), and the same
        // per-leg direction split applies (candidate must speculate, control must be serial).
        LegRegime::FreeRunV1_1 | LegRegime::BatchedFreeRunV1_2(_) => {
            let Some(echo) = wire_effective_spec else {
                return Err(RejectCtx {
                    class: RejectClass::Infra,
                    leg,
                    reason: format!(
                        "{leg} free-run leg carries no engine-echoed effective_spec (fail-closed; \
                         the v1.1 gate is ON and a spec was requested, so the echo is required and \
                         benchd never fabricates it)"
                    ),
                });
            };
            let checked = if is_candidate {
                free_run_regime_is_speculative(&echo)
                    .map_err(|r| (RejectClass::FreeRunRegimeNotSpeculative, r))
            } else {
                free_run_serial_control_is_non_speculating(&echo)
                    .map_err(|r| (RejectClass::FreeRunSerialControlSpeculated, r))
            };
            if let Err((class, reason)) = checked {
                return Err(RejectCtx {
                    class,
                    leg,
                    reason: format!("{leg} leg: {reason}"),
                });
            }
            (echo, EFFECTIVE_SPEC_SOURCE_WIRE_ECHO)
        }
    };
    // COHORT — a batched leg's audit channel is the COHORT audit, and the two channels are
    // mutually exclusive BY REGIME: a cohort audit off the batched regime (or a single-stream
    // audit on it, refused below) is a fabricated claim about a window that was never run in that
    // shape. On the batched regime the audit is REQUIRED, its base must cover exactly the
    // per-stream N benchd requested, and its width must be the ONE ruled batch point.
    let cohort_audit = match (regime, inv.cohort_audit.clone()) {
        (LegRegime::BatchedFreeRunV1_2(_), None) => {
            return Err(RejectCtx {
                class: RejectClass::FreeRunConsistency,
                leg,
                reason: format!(
                    "{leg} leg ran the v1.2 batched free-run regime but carries no cohort AUDIT \
                     (fail-closed; the cohort consistency quadruple must have passed and benchd \
                     never fabricates the acceptance histogram)"
                ),
            });
        }
        (LegRegime::BatchedFreeRunV1_2(point), Some(audit)) => {
            if audit.base().verified_token_count() != n {
                return Err(RejectCtx {
                    class: RejectClass::FreeRunConsistency,
                    leg,
                    reason: format!(
                        "{leg} leg cohort AUDIT covers {} verified tokens per stream, but the \
                         timed window requested N={n}: the acceptance histogram does not describe \
                         the clocked window",
                        audit.base().verified_token_count()
                    ),
                });
            }
            // D9 — the cohort width is a PINNED IDENTITY: the audit's echoed-and-validated width
            // must be the CERTIFIED batch point this regime carries (fixture data, certified by
            // `ScoredBatchPoint::certify`). A different width under this series tag would be a
            // tagged number measured at some other B.
            if audit.batch_size() != point.batch_size() {
                return Err(RejectCtx {
                    class: RejectClass::FreeRunConsistency,
                    leg,
                    reason: format!(
                        "{leg} leg cohort AUDIT carries batch_size {} but the {:?} series seals \
                         exactly B={}: a number measured at another width must never be sealed \
                         under this series tag",
                        audit.batch_size(),
                        point.timed_mode(),
                        point.batch_size(),
                    ),
                });
            }
            // D3 — the BATCHED serial control's STRUCTURAL assertion: the cohort commits ONE
            // COMMON width per round (§2.4), so the depth-0 invariant — common width EXACTLY 1
            // every round, and R == N — is literally the same unit-histogram check as the
            // single-stream control's. Reusing the one function is the §2.4 payoff: zero new
            // assertion code, one definition of "the control did not speculate".
            if !is_candidate {
                if let Err(reason) =
                    free_run_serial_control_histogram_is_unit(audit.base().acceptance_lengths(), n)
                {
                    return Err(RejectCtx {
                        class: RejectClass::FreeRunSerialControlSpeculated,
                        leg,
                        reason: format!("{leg} leg (batched cohort): {reason}"),
                    });
                }
            }
            Some(audit)
        }
        (LegRegime::TeacherForcedV1 | LegRegime::FreeRunV1_1, Some(_)) => {
            return Err(RejectCtx {
                class: RejectClass::FreeRunConsistency,
                leg,
                reason: format!(
                    "{leg} leg ran a non-batched regime but carries a COHORT audit: a cohort \
                     histogram can only come from a batched window — refusing the fabricated claim"
                ),
            });
        }
        (LegRegime::TeacherForcedV1 | LegRegime::FreeRunV1_1, None) => None,
    };
    // W3 — a free-run leg MUST carry the §3 AUDIT the runner produced after the §2.6 triple passed,
    // covering exactly the N committed tokens benchd requested. A missing audit is fail-closed
    // (benchd never fabricates one); an audit whose verified-token count is not N means the sealed
    // acceptance histogram describes a different window than the one that was clocked.
    let free_run_audit = match (regime, free_run_audit) {
        (LegRegime::FreeRunV1_1, None) => {
            return Err(RejectCtx {
                class: RejectClass::FreeRunConsistency,
                leg,
                reason: format!(
                    "{leg} leg ran the v1.1 free-run regime but carries no free-run AUDIT \
                     (fail-closed; the §2.6 triple must have passed and benchd never fabricates the \
                     acceptance histogram)"
                ),
            });
        }
        (LegRegime::FreeRunV1_1, Some(audit)) => {
            if audit.verified_token_count() != n {
                return Err(RejectCtx {
                    class: RejectClass::FreeRunConsistency,
                    leg,
                    reason: format!(
                        "{leg} leg free-run AUDIT covers {} verified tokens, but the timed window \
                         requested N={n}: the acceptance histogram does not describe the clocked window",
                        audit.verified_token_count()
                    ),
                });
            }
            // Fable ruling (same-series serial control) — the CONTROL's histogram must be `[1]*N`.
            // Checked from the audit the §2.6 triple already validated, so this reads the engine's
            // demonstrated behaviour rather than its self-description (the echo checked above).
            if !is_candidate {
                if let Err(reason) =
                    free_run_serial_control_histogram_is_unit(audit.acceptance_lengths(), n)
                {
                    return Err(RejectCtx {
                        class: RejectClass::FreeRunSerialControlSpeculated,
                        leg,
                        reason: format!("{leg} leg: {reason}"),
                    });
                }
            }
            Some(audit)
        }
        // A teacher-forced leg has no acceptance to report; an audit attached to one would be a
        // fabricated free-run claim on a window that never free-ran. A BATCHED leg's acceptance
        // lives exclusively in its cohort audit (validated above) — a single-stream audit riding
        // alongside it would be a second, unvalidated claim about the same window.
        (LegRegime::TeacherForcedV1 | LegRegime::BatchedFreeRunV1_2(_), Some(_)) => {
            return Err(RejectCtx {
                class: RejectClass::FreeRunConsistency,
                leg,
                reason: format!(
                    "{leg} leg carries a single-stream free-run AUDIT its regime cannot have \
                     produced (teacher forcing feeds every token; a batched window's acceptance \
                     lives in the cohort audit) — refusing the fabricated claim"
                ),
            });
        }
        (LegRegime::TeacherForcedV1 | LegRegime::BatchedFreeRunV1_2(_), None) => None,
    };
    // COMPOSITE (Gemma cohort scoring) — the phase-split window, REQUIRED on the batched regime
    // (mirrors `cohort_audit`'s presence rule exactly): a missing value fails the leg closed
    // (benchd never fabricates a window it did not clock), and a value present off the batched
    // regime is a fabricated claim about a second window the single-stream/teacher-forced verbs
    // never open.
    let cohort_phase_windows = match (regime, inv.cohort_phase_windows) {
        (LegRegime::BatchedFreeRunV1_2(_), None) => {
            return Err(RejectCtx {
                class: RejectClass::FreeRunConsistency,
                leg,
                reason: format!(
                    "{leg} leg ran the v1.2 batched free-run regime but carries no phase-split \
                     window (fail-closed; the prefill/decode clock split must have run and \
                     benchd never fabricates the elapsed times)"
                ),
            });
        }
        (LegRegime::BatchedFreeRunV1_2(_), Some(w)) => Some(w),
        (LegRegime::TeacherForcedV1 | LegRegime::FreeRunV1_1, Some(_)) => {
            return Err(RejectCtx {
                class: RejectClass::FreeRunConsistency,
                leg,
                reason: format!(
                    "{leg} leg carries a COHORT PHASE-SPLIT window its regime cannot have \
                     produced (only the v1.2 batched free-run verbs clock a separate prefill \
                     window) — refusing the fabricated claim"
                ),
            });
        }
        (LegRegime::TeacherForcedV1 | LegRegime::FreeRunV1_1, None) => None,
    };
    // (b) admission — the CANDIDATE cohort leg's committed tokens, the journal the trusted-oracle
    // tolerance gate replays + judges. REQUIRED on the batched CANDIDATE leg (the gate has nothing to
    // judge without it); the serial CONTROL leg is never token-judged (only the candidate is scored
    // for correctness), so its journal is dropped here; and a value present off the batched regime is
    // a fabricated claim about a rectangle the single-stream/TF verbs never committed.
    let cohort_committed_tokens_by_stream = match (regime, is_candidate) {
        (LegRegime::BatchedFreeRunV1_2(_), true) => {
            let tokens = inv
                .cohort_committed_tokens_by_stream
                .clone()
                .ok_or_else(|| RejectCtx {
                    class: RejectClass::CohortReplayIntegrity,
                    leg,
                    reason: format!(
                        "{leg} candidate leg ran the v1.2 batched cohort but surfaced no committed \
                         tokens (fail-closed; the trusted-oracle tolerance gate has no journal to \
                         replay and benchd never fabricates one)"
                    ),
                })?;
            Some(tokens)
        }
        // The serial control leg is not token-judged; drop its journal even on the batched regime.
        (LegRegime::BatchedFreeRunV1_2(_), false) => None,
        // Off the batched regime, a committed-rectangle is a fabricated claim.
        (LegRegime::TeacherForcedV1 | LegRegime::FreeRunV1_1, _) => {
            if inv.cohort_committed_tokens_by_stream.is_some() {
                return Err(RejectCtx {
                    class: RejectClass::CohortReplayIntegrity,
                    leg,
                    reason: format!(
                        "{leg} leg carries a COHORT committed-token rectangle its regime cannot have \
                         produced (only the v1.2 batched free-run verbs commit a B x N rectangle) — \
                         refusing the fabricated claim"
                    ),
                });
            }
            None
        }
    };
    // #109 window-2 finding 3 — the head identity comes off the WIRE `hello`
    // (`head_provenance.sha256`), the engine's echo of the head bytes it actually loaded. The
    // retired report file was the only other channel that ever carried it.
    //
    // #109 W3 finding 5 — and it is REGIME-SCOPED, for the same reason the effective-spec check
    // above is: the engine gates `head_provenance` behind the v1.1 spawn flag, and a TF leg is
    // spawned gate-off. Requiring the field on a TF candidate leg demanded a value that leg's own
    // spawn surface forbids — unsatisfiable by construction, and no engine change could have fixed
    // it. The requirement belongs to the regime that HAS a head channel.
    let head = inv
        .wire_head_provenance
        .as_ref()
        .map(|hp| hp.sha256.trim())
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    match regime {
        // FREE-RUN (single-stream or batched cohort) — the gate is ON, so the head channel exists.
        // The CANDIDATE leg REQUIRES it (fail-closed, unchanged): it is the identity of the BYO
        // head the engine actually loaded, and these series are the ones that score drafting. The
        // serial control loads the pinned head and its sha is optional.
        LegRegime::FreeRunV1_1 | LegRegime::BatchedFreeRunV1_2(_) => {
            if is_candidate && head.is_none() {
                return Err(RejectCtx {
                    class: RejectClass::Infra,
                    leg,
                    reason: format!(
                        "{leg} (MTP) leg hello carries no head_provenance.sha256 (fail-closed; the \
                         candidate leg must report the head the engine loaded)"
                    ),
                });
            }
        }
        // TEACHER-FORCED — spawned gate-off, so a conformant hello carries NO head_provenance and
        // the leg NEITHER REQUIRES NOR ACCEPTS one. A present object is the tamper case, refused by
        // the same logic as a present effective-spec echo ([`tf_hello_carries_no_head_provenance`]).
        // A TF leg therefore seals no head identity at all — `head` is `None` here by construction.
        LegRegime::TeacherForcedV1 => {
            if let Err(reason) =
                tf_hello_carries_no_head_provenance(inv.wire_head_provenance.as_ref())
            {
                return Err(RejectCtx {
                    class: RejectClass::NonSerialTfRegime,
                    leg,
                    reason: format!("{leg} leg: {reason}"),
                });
            }
        }
    }
    // The per-round draft statistics, COMPUTED BY BENCHD from the `acceptance_lengths` histogram it
    // collected and triple-checked (`docs/spec-config-design.md` step 3 — "NOT an engine echo"), so
    // a lying engine cannot move them. A teacher-forced leg has no histogram and no drafting to
    // describe: teacher forcing feeds every token, so a TF leg's draft statistics are `None`.
    //
    // #109 window-2 finding 3 (RETIRED) — R16 used to REQUIRE the TF candidate leg to echo
    // `effective_mean_draft_len` / `non_drafting_round_count` through the report file. That
    // requirement is retired with the file: the generic `runtime-worker` verb writes no report, and
    // the numbers it demanded describe drafting that `tf_regime_is_serial` already proves cannot have
    // happened on a TF leg. The drafting statistics now live exclusively in the free-run series.
    // COHORT — the batched leg's draft statistics come from the cohort audit's per-round
    // common-width BASE, which is the SAME [`FreeRunAudit`] derivation the single-stream regime
    // uses: one definition of the `audit_spec_*` family across both regimes, still computed by
    // benchd from the histogram it collected, never an engine echo.
    let stats_base: Option<&FreeRunAudit> = free_run_audit
        .as_ref()
        .or_else(|| cohort_audit.as_ref().map(|c| c.base()));
    let (effective_mean_draft_len, non_drafting_round_count) = match stats_base {
        Some(audit) => (
            Some(audit.effective_mean_draft_len()),
            Some(audit.non_drafting_round_count()),
        ),
        None => (None, None),
    };
    // B2 — PREFILL-WINDOW CERTIFICATION, against the track this tree serves.
    //
    // The track is `bench_core::constants::TRACK_ID` and not a threaded parameter for the same
    // reason `official.rs` resolves its baselines against it: this tree serves exactly one track,
    // and `resolve_track_id` has already refused any run that declared another one
    // (`enforce_declared_track`) before a single leg was measured. There is no second source here to
    // cross-check against, so inventing one would be inventing a fact.
    let certification =
        certify_leg_prefill_window(leg, bench_core::constants::TRACK_ID, scored_regime, inv)?;

    Ok(LegMeasurement {
        seconds_per_token: spt,
        gate_state: GateState::SkippedNoReader, // overwritten by the caller with the real state
        quiesce: QuiesceRecord::default(),      // overwritten by the caller with the real record
        head_provenance_sha256: head,
        effective_mean_draft_len,
        non_drafting_round_count,
        telemetry: None, // overwritten by the caller with the leg's observed sample
        attempts: 1,     // overwritten by run_leg_with_retry with the real attempt count
        effective_spec,
        effective_spec_source,
        regime,
        free_run_audit,
        phase_window: certification.window(),
        phase_window_certified: certification.is_certified(),
        phase_window_cross_check: certification.cross_check(),
        cohort_audit,
        cohort_phase_windows,
        // REPORT-ONLY (gap G2) — passed through UNVALIDATED, deliberately: this function's
        // verdicts decide leg acceptance, and no per-stream evidence defect may tighten (or
        // loosen) that decision. Structural defects are judged by `attest_leg` on the ACCEPTED
        // path and seal as `attestation_refused`, never as a rejection here.
        per_stream_timing: inv.per_stream_timing.clone(),
        cohort_committed_tokens_by_stream,
    })
}

/// W3 — the fail-closed coherence check between the DECLARED candidate spec and the regime the
/// candidate leg will actually run, plus the free-run series' RULED window:
///
/// 1. a FREE-RUN candidate regime requires a SPECULATING declared spec — free-running a serial
///    candidate measures nothing MTP can move, and would seal a v1.1 number for a serial engine;
/// 2. a free-run run's decode window must be the RULED N ([`FREE_RUN_DECODE_TOKENS`] = 128,
///    PROTOCOL-v1.1 OQ3). The teacher-forced default is 512 ([`DEFAULT_TOKENS`], the live wrapper's
///    window); a caller who lands on the free-run path with some other window is REFUSED here
///    rather than silently re-windowed, because N is the divisor of the scored seconds-per-token.
///
/// The reverse pairing (a teacher-forced regime with a declared mtp spec) stays LEGAL: that is the
/// Model-2 shape, where the candidate's declared spec is provenance and the timed wire is downgraded
/// to serial ([`timed_decode_wire_spec`]).
pub fn validate_candidate_regime_coherent(cfg: &MeasureJobConfig) -> Result<(), String> {
    if !cfg.candidate_regime.is_free_run() {
        return Ok(());
    }
    if cfg.candidate_spec.mode == SPEC_MODE_SERIAL {
        return Err(format!(
            "candidate regime is the v1.1 free-run series but the declared candidate spec is \
             {:?}: a serial candidate has no speculation to free-run, so the free-run window would \
             measure (and seal) a serial number under the speculating series",
            cfg.candidate_spec.mode
        ));
    }
    if cfg.tokens != FREE_RUN_DECODE_TOKENS {
        return Err(format!(
            "free-run decode window is --tokens {} but PROTOCOL-v1.1 RULES N = \
             {FREE_RUN_DECODE_TOKENS} (BENCHMARK_DECODE_STEPS): N divides the scored \
             seconds-per-token, so benchd refuses to re-window the free-run series silently",
            cfg.tokens
        ));
    }
    Ok(())
}

/// R15 — run ONE timed leg: ONE `mtp-timed` verb invocation, ONE process, ONE cool gate before the
/// single timed invocation, with ONE gated retry (`MAX_ATTEMPTS = 2`, full precondition reset — a
/// fresh worker + cool gate — between attempts; the contract's per-leg `run_phase` loop, W:1615-
/// 1640,1718). This REPLACES the old prefill/decode two-phase-per-leg structure: the seed prefill is
/// INSIDE the single decode window (`prefill_component: "none"`), so there is exactly one gated
/// timed invocation per leg. Finding R19 — the retry is CLASS-AGNOSTIC: EVERY rejection class
/// (parity, implausible, row-accounting, thermal-gate timeout, missing-echo, spawn/protocol infra)
/// re-attempts exactly once; a class that still fails after its one retry rejects the leg (folding
/// into die-5 upstream). #108 (L1) — with ONE exemption: a [`RejectClass::is_deterministic`] class
/// is TERMINAL and skips the retry, because the reset the retry performs cannot change an
/// input-determined verdict. The leg still rejects, with the same class, into the same die-5.
/// The `measure` closure encapsulates the spawn + ONE cool gate + timed verb +
/// report parse, returning the report and the gate state it recorded (tests supply a MOCK report).
// UNVERIFIED(measure-job): the on-box spawn + cool-gate + `mtp-timed` verb + report-parse wiring is
// contract-derived, not checked on a live box.
/// WHAT THE TWO LEGS OF A PAIR MEASURE: the decode depth of the timed window, and the scored
/// regime that window's prefill half is certified against.
///
/// ONE value because they describe ONE window: the depth says how long it is, the regime says
/// whether anything about it is enforced. The regime is RESOLVED ONCE per run, from the track
/// fixture, and carried on `MeasureJobConfig`, so every leg of every pair certifies against the
/// same declaration.
#[derive(Clone, Copy)]
struct PairWindow<'a> {
    steps: usize,
    scored_regime: &'a bench_core::constants::ScoredRegime,
}

fn run_leg_with_retry<FM, P>(
    leg: &'static str,
    is_candidate: bool,
    measure: &mut FM,
    params: &P,
    window: PairWindow<'_>,
) -> Result<LegMeasurement, RejectCtx>
where
    FM: FnMut(&P) -> bench_runner::Result<LegInvocation>,
{
    let mut last = RejectCtx {
        class: RejectClass::Infra,
        leg,
        reason: format!("{leg} leg did not run"),
    };
    for _attempt in 0..MAX_ATTEMPTS {
        match measure(params) {
            Ok(inv) => match validate_leg_report(
                leg,
                is_candidate,
                window.steps,
                window.scored_regime,
                &inv,
            ) {
                Ok(mut m) => {
                    m.gate_state = inv.gate_state;
                    m.quiesce = inv.quiesce;
                    m.telemetry = inv.telemetry;
                    // R16 — the real attempt count (1-based): this leg succeeded on attempt `_attempt`.
                    m.attempts = _attempt + 1;
                    return Ok(m);
                }
                Err(ctx) => last = ctx,
            },
            Err(e) => {
                // Finding R19 — CLASS-AGNOSTIC single gated retry: a spawn/gate/verb failure of any
                // class re-attempts exactly once, then folds into die-5 via `Err(last)` below.
                last = RejectCtx {
                    class: classify(&e),
                    leg,
                    reason: sealed_reject_reason(leg, &e),
                };
            }
        }
        // #108 (L1) — the ONE exemption to R19's class-agnostic retry: a DETERMINISTIC reject
        // ([`RejectClass::is_deterministic`]) is TERMINAL. The retry's value is the precondition
        // reset (fresh worker + fresh cool gate) it performs between attempts, and a condition
        // determined by unchanged input is untouched by that reset — the second attempt spends a
        // spawn and a cool-gate wait to reach the identical verdict, delaying the honest diagnostic.
        // Break out with `last` intact; the leg still rejects and still folds into die-5 exactly as
        // before, with the same class recorded — only the wasted attempt is gone.
        if last.class.is_deterministic() {
            break;
        }
    }
    Err(last)
}

/// The result of attempting ONE pair. Finding R19 — a `Rejected` carries ONLY the sealed record
/// (its class is a provenance label): the loop no longer branches on class, because there is no
/// hard-die vs per-pair distinction anymore — every reject just fails the pair and folds into die-5.
enum PairAttempt {
    // `PairRecord` is much larger than `RejectRecord`; box it so the enum stays small
    // (clippy::large_enum_variant) without an `#[allow]`.
    Accepted(Box<PairRecord>),
    Rejected(RejectRecord),
}

/// The pair loop's two REPORT-ONLY side channels — neither participates in the accept/reject
/// decision, so they ride together in one struct and keep [`run_pair`]'s scored/security args
/// (the two measure closures, the `token_gate`, `params`, `n`, `order`) as the direct arguments.
/// Both parent lanes (#188/#189 per-stream carry+seal, #190 token-tolerance gate) each added one
/// direct arg to `run_pair`; bundling these two keeps the signature at the house arity ceiling
/// (the too-many-arguments lint) without an `#[allow]` and without touching any security arg.
struct PairSideChannels<'a> {
    /// R16 — the run-wide telemetry accumulator both accepted legs fold their observed sample into.
    telemetry: &'a mut TelemetryAccumulator,
    /// REPORT-ONLY (gap G2) — the CERTIFIED exponent pair from the contract
    /// ([`ScoredExponents::certify`]'s output; `MeasureJobConfig::scored_exponents`, threaded by the
    /// cohort pair loop — the single-stream loop passes `None`, its legs carry no per-stream
    /// channel). Consumed ONLY by the per-pair `per_stream_composite_diagnostic` seal; nothing here
    /// changes which pairs accept.
    per_stream_exponents: Option<ScoredExponents>,
}

/// Run one pair: the serial-control (depth-0, pinned head) and candidate (depth D, declared head)
/// legs in the alternation `order`, EACH one `mtp-timed` verb invocation with ONE cool gate + one
/// gated retry (R15). If EITHER leg rejects, the pair is rejected (contributes nothing). Otherwise
/// the pair is accepted, sealing each leg's `parent_measured_seconds_per_token`, the engine-echoed
/// effective spec, and the candidate's head provenance — plus, on the batched cohort regime, both
/// legs' phase-split windows and the per-pair component gains derived from them (COMPOSITE, Gemma
/// cohort scoring; `None` off the batched regime).
///
/// The two REPORT-ONLY side channels (the telemetry accumulator and the certified per-stream
/// exponent pair) ride in [`PairSideChannels`]; neither changes which pairs accept.
fn run_pair<FS, FC, FG, P>(
    order: PairOrder,
    measure_serial: &mut FS,
    measure_candidate: &mut FC,
    token_gate: &mut FG,
    params: &P,
    window: PairWindow<'_>,
    channels: PairSideChannels<'_>,
) -> PairAttempt
where
    FS: FnMut(&P) -> bench_runner::Result<LegInvocation>,
    FC: FnMut(&P) -> bench_runner::Result<LegInvocation>,
    // (b) admission — the POST-RUN token-correctness gate over the CANDIDATE leg's measurement. On
    // the cohort path this is the trusted-oracle ≤10% per-stream tolerance gate (+ N2 integrity); on
    // the single-stream path it is a no-op (`Ok(())`), because that regime's token correctness is
    // still enforced inline in the runner. Called AFTER both legs succeed so a pair is rejected only
    // for a correctness failure, never fabricated into an accepted record.
    // The gate's Ok side carries the pair's REPORT-ONLY [`CohortNearTieSeal`] (`None` on the
    // single-stream no-op, and on any cohort report the seal builder had nothing to read). It is a
    // VALUE the gate hands back, never an input to the gate's verdict.
    FG: FnMut(&LegMeasurement) -> Result<Option<CohortNearTieSeal>, RejectCtx>,
{
    // The two REPORT-ONLY side channels, unpacked so the body below reads them by their own names;
    // neither is consulted for the accept/reject decision (see [`PairSideChannels`]).
    let PairSideChannels {
        telemetry,
        per_stream_exponents,
    } = channels;
    // Run the two legs in the alternation order (ordering advantages neither side). The
    // measurement is independent of order; the order is recorded for the audit.
    let (serial, candidate) = match order {
        PairOrder::MtpFirst => {
            let candidate =
                match run_leg_with_retry("candidate", true, measure_candidate, params, window) {
                    Ok(m) => m,
                    Err(ctx) => return PairAttempt::Rejected(reject_record(order, ctx)),
                };
            let serial = match run_leg_with_retry("serial", false, measure_serial, params, window) {
                Ok(m) => m,
                Err(ctx) => return PairAttempt::Rejected(reject_record(order, ctx)),
            };
            (serial, candidate)
        }
        PairOrder::SerialFirst => {
            let serial = match run_leg_with_retry("serial", false, measure_serial, params, window) {
                Ok(m) => m,
                Err(ctx) => return PairAttempt::Rejected(reject_record(order, ctx)),
            };
            let candidate =
                match run_leg_with_retry("candidate", true, measure_candidate, params, window) {
                    Ok(m) => m,
                    Err(ctx) => return PairAttempt::Rejected(reject_record(order, ctx)),
                };
            (serial, candidate)
        }
    };

    // (b) admission — the POST-RUN token-correctness gate on the CANDIDATE leg, AFTER the candidate
    // cohort run produced its committed journal. On the cohort path it spawns the TRUSTED oracle over
    // the organizer's reference weights, verifies the oracle replayed the candidate's REAL journal
    // (N2), and applies the ≤10% per-stream tolerance bar; any failure rejects the WHOLE pair (which,
    // in an official run, is an immediate die-5). On the single-stream path it is a no-op. Only the
    // candidate journal is judged — the serial control is not token-scored.
    //
    // The gate's Ok value is the REPORT-ONLY near-tie seal over the same oracle report the gate
    // just judged; it is sealed on the accepted record below and read by nothing else.
    let cohort_near_tie_seal = match token_gate(&candidate) {
        Ok(seal) => seal,
        Err(ctx) => return PairAttempt::Rejected(reject_record(order, ctx)),
    };

    // R15 — the ONLY scored number per leg is the report's `parent_measured_seconds_per_token`.
    let serial_spt = serial.seconds_per_token;
    let mtp_spt = candidate.seconds_per_token;
    // serial / mtp: a faster candidate ⇒ ratio > 1. Finite by construction (both plausible).
    let raw_ratio = if mtp_spt > 0.0 {
        serial_spt / mtp_spt
    } else {
        0.0
    };
    // The candidate leg's head provenance. Present by construction on a FREE-RUN candidate
    // (validate_leg_report requires it there); EMPTY on a teacher-forced pair, whose gate-off legs
    // cannot report a head at all (#109 W3 finding 5) — the empty string is filtered back out to an
    // omitted per-prompt field, never sealed as a blank identity.
    let head_provenance_sha256 = candidate.head_provenance_sha256.clone().unwrap_or_default();
    // R16 — the pair is ACCEPTED, so fold BOTH legs' observed telemetry samples into the run-wide
    // accumulator (rejected pairs contribute nothing — this runs only on the accepted path).
    if let Some(s) = serial.telemetry.as_ref() {
        telemetry.observe(s);
    }
    if let Some(s) = candidate.telemetry.as_ref() {
        telemetry.observe(s);
    }
    // The candidate leg's draft statistics come from benchd's own free-run histogram; a
    // teacher-forced candidate has none, and its honest value is 0 (teacher forcing feeds every
    // token, so no round could draft).
    let effective_mean_draft_len = candidate.effective_mean_draft_len.unwrap_or(0.0);
    let non_drafting_round_count = candidate.non_drafting_round_count.unwrap_or(0);
    // B2 — the pair's two clock splits. Sealed only when BOTH legs opened a free-run window: a
    // half-populated block would invite a gain computed against a leg that never measured one.
    let phase_windows = match (serial.phase_window, candidate.phase_window) {
        (Some(s), Some(c)) => Some(PairPhaseWindows {
            serial_prefill_window_seconds: s.seed_prefill_elapsed_seconds,
            candidate_prefill_window_seconds: c.seed_prefill_elapsed_seconds,
            serial_decode_window_seconds: s.decode_elapsed_seconds,
            candidate_decode_window_seconds: c.decode_elapsed_seconds,
            prefill_token_total: c.prefill_token_total,
            decode_token_total: c.decode_token_total,
            certified: serial.phase_window_certified && candidate.phase_window_certified,
            // Corroborated only if BOTH legs were: one leg's second source says nothing about the
            // other's, and the gain divides the two.
            cross_check: if serial.phase_window_cross_check
                == bench_core::prefill_window::PREFILL_WINDOW_CROSS_CHECK_AGREED
                && candidate.phase_window_cross_check
                    == bench_core::prefill_window::PREFILL_WINDOW_CROSS_CHECK_AGREED
            {
                bench_core::prefill_window::PREFILL_WINDOW_CROSS_CHECK_AGREED
            } else {
                bench_core::prefill_window::PREFILL_WINDOW_CROSS_CHECK_NOT_OBSERVED
            },
        }),
        _ => None,
    };
    // REPORT-ONLY (gap G2) — the per-stream attestation, per leg of this ACCEPTED pair (rejected
    // pairs seal no timing at all, so there is nothing to attest on that path). Runs AFTER both
    // legs' verdicts are final: no outcome here can reach the acceptance decision above.
    let serial_per_stream_attestation = per_stream_attestation_seal(&serial);
    let candidate_per_stream_attestation = per_stream_attestation_seal(&candidate);
    // The pair's composite DIAGNOSTIC over the two sealed SUM aggregates, at the CERTIFIED
    // exponent pair — only when both legs attested Ok (a refused/absent leg has no sums to
    // pair). UNSCORED: `PerCohort::composite` is the parent-clocked `shared_window_composite`,
    // which never reads this or any other engine-reported number.
    let per_stream_composite_diagnostic = match (
        serial_per_stream_attestation
            .as_ref()
            .and_then(|s| s.verdict.as_ref()),
        candidate_per_stream_attestation
            .as_ref()
            .and_then(|c| c.verdict.as_ref()),
        per_stream_exponents,
    ) {
        (Some(sv), Some(cv), Some(exponents)) => {
            Some(bench_core::per_stream_attestation::composite_diagnostic(
                sv,
                cv,
                exponents.prefill_gain_exponent,
                exponents.decode_gain_exponent,
            ))
        }
        _ => None,
    };
    PairAttempt::Accepted(Box::new(PairRecord {
        parity_ok: true,
        serial_seconds_per_token: finite_nonneg(serial_spt),
        mtp_seconds_per_token: finite_nonneg(mtp_spt),
        order: order.as_str().to_string(),
        raw_ratio,
        // R16 — the same ratio under the live per-pair name `speedup`.
        speedup: raw_ratio,
        // R16 — stamped in build_results (prompt loop knows the index + golden sha).
        prompt_index: 0,
        prompt_sha256: String::new(),
        // R16 — the real per-leg attempt counts (1, or 2 after the one gated retry).
        serial_attempts: serial.attempts,
        mtp_attempts: candidate.attempts,
        // R16 — the on-box first-block sub-interval is not emitted offline (honest omit).
        serial_first_block_seconds: None,
        mtp_first_block_seconds: None,
        serial_gate_state: serial.gate_state.as_str().to_string(),
        candidate_gate_state: candidate.gate_state.as_str().to_string(),
        serial_quiesce_state: serial.quiesce.state.as_str().to_string(),
        candidate_quiesce_state: candidate.quiesce.state.as_str().to_string(),
        serial_quiesce_waited_seconds: serial.quiesce.waited_seconds,
        candidate_quiesce_waited_seconds: candidate.quiesce.waited_seconds,
        serial_quiesce_load: serial.quiesce.load,
        candidate_quiesce_load: candidate.quiesce.load,
        serial_quiesce_gpu_util: serial.quiesce.gpu_util,
        candidate_quiesce_gpu_util: candidate.quiesce.gpu_util,
        // The effective spec each leg ran, with its provenance. On a free-run job the CONTROL echoes
        // serial (depth 0) and the CANDIDATE echoes its speculating spec — the two legs' guards
        // enforce exactly that. On a teacher-forced job both legs are serial from the gate-off spawn
        // surface, with no echo involved (coordinator ruling #109 leg B).
        serial_effective_spec: serial.effective_spec.clone(),
        candidate_effective_spec: candidate.effective_spec.clone(),
        serial_effective_spec_source: serial.effective_spec_source,
        candidate_effective_spec_source: candidate.effective_spec_source,
        head_provenance_sha256,
        effective_mean_draft_len,
        non_drafting_round_count,
        // W3 — the per-leg series tags + the candidate's free-run AUDIT (§3). Both are OMITTED /
        // empty on a teacher-forced candidate, so a TF pair record is byte-unchanged apart from the
        // two tags.
        serial_timed_mode: serial.regime.timed_mode(),
        candidate_timed_mode: candidate.regime.timed_mode(),
        phase_windows,
        // COHORT — the acceptance histogram / audit metrics come from whichever audit channel the
        // candidate's regime produced: the single-stream §3 audit, or the cohort audit's per-round
        // common-width base (the SAME `audit_spec_*` derivation) plus the flat `audit_cohort_*`
        // family. Both channels are AUDIT ONLY.
        audit_spec_acceptance_lengths: candidate
            .free_run_audit
            .as_ref()
            .map(|a| a.acceptance_lengths().to_vec())
            .or_else(|| {
                candidate
                    .cohort_audit
                    .as_ref()
                    .map(|c| c.base().acceptance_lengths().to_vec())
            }),
        audit_spec: candidate
            .free_run_audit
            .as_ref()
            .map(|a| a.to_metrics().into_iter().collect())
            .or_else(|| {
                candidate
                    .cohort_audit
                    .as_ref()
                    .map(|c| c.to_metrics().into_iter().collect())
            })
            .unwrap_or_default(),
        // COHORT — the candidate's per-stream / per-round cohort vectors, sealed VERBATIM as
        // diagnostics (never scored; never aggregated as samples — inside one window the streams
        // are correlated readings). OMITTED on every non-batched pair.
        audit_cohort_natural_accepted_by_stream: candidate
            .cohort_audit
            .as_ref()
            .map(|c| c.natural_accepted_by_stream().to_vec()),
        audit_cohort_active_streams_by_round: candidate
            .cohort_audit
            .as_ref()
            .map(|c| c.active_streams_by_round().to_vec()),
        audit_cohort_depth_clamp_reasons: candidate
            .cohort_audit
            .as_ref()
            .map(|c| c.depth_clamp_reasons().clone()),
        // COMPOSITE (Gemma cohort scoring) — both legs carry `Some` together (validated per leg
        // in `validate_leg_report`, which requires the window on the batched regime and forbids
        // it elsewhere) or `None` together on a non-batched pair; `zip` reflects exactly that.
        cohort_phase_windows: serial
            .cohort_phase_windows
            .zip(candidate.cohort_phase_windows)
            .map(|(s, c)| PairCohortPhaseWindows::compute(s, c)),
        // REPORT-ONLY (gap G2) — the per-leg attestation seals + the pair's composite
        // diagnostic, computed above. All three are ADDITIVE (`skip_serializing_if` keeps every
        // pre-existing record byte-identical when absent) and none feeds a scored field.
        serial_per_stream_attestation,
        candidate_per_stream_attestation,
        per_stream_composite_diagnostic,
        cohort_near_tie_seal,
    }))
}

fn reject_record(order: PairOrder, ctx: RejectCtx) -> RejectRecord {
    RejectRecord {
        order: order.as_str().to_string(),
        class: ctx.class.as_str().to_string(),
        leg: ctx.leg.to_string(),
        reason: ctx.reason,
    }
}

/// (b) admission — the TRUSTED-ORACLE token-correctness gate over ONE candidate cohort leg (David's
/// blanket-10% ruling, 2026-08-25). Replaces the runner's removed inline exact-match die.
///
/// Steps, fail-closed:
/// 1. read the candidate's OWN committed journal (`cohort_committed_tokens_by_stream`);
/// 2. spawn the TRUSTED oracle over the ORGANIZER reference weights (`oracle`) and replay THAT journal
///    teacher-forced from the organizer-fixed `replay_seeds_by_stream`;
/// 3. N2 — verify the oracle's echoed committed tokens EQUAL the candidate journal (byte/id) before
///    any counting; a divergence is a HARD integrity error ([`RejectClass::CohortReplayIntegrity`]);
/// 4. apply the ≤10% PER-STREAM tolerance bar against the reference argmax; any stream over →
///    [`RejectClass::CohortTokenTolerance`], rejecting the WHOLE pair;
/// 5. REPORT-ONLY — build the [`CohortNearTieSeal`] over the SAME report and hand it back as the
///    Ok value. Step 5 runs only after steps 1-4 are final and cannot fail (the builder is total),
///    so it can neither change nor delay a verdict; see [`CohortNearTieSeal`].
///
/// ★ ORACLE WEIGHTS PROVENANCE (anti-gaming linchpin): this function passes `oracle` ONLY tokens
/// (`replay_seeds_by_stream` from the organizer cohort params, and the candidate's committed journal
/// as the thing to JUDGE). It NEVER passes a weights path — the oracle's `--weights` is fixed at SPAWN
/// time by the closure built in main.rs from the SAME organizer-controlled source as the trusted
/// baseline leg, never from anything candidate-controlled. So even a candidate that loaded rogue
/// weights in its OWN cohort run only makes its journal diverge MORE from the organizer-weights
/// reference (→ rejected), never helps it pass. The candidate controls neither the trusted build's
/// code (fail-closed [`resolve_trusted_oracle_worker_bin`]) nor the oracle's weights.
fn cohort_token_tolerance_gate<FO>(
    candidate: &LegMeasurement,
    replay_seeds_by_stream: &[Vec<i64>],
    oracle: &mut FO,
) -> Result<Option<CohortNearTieSeal>, RejectCtx>
where
    FO: FnMut(
        &[Vec<i64>],
        &[Vec<i64>],
    ) -> bench_runner::Result<bench_protocol::CohortReferenceReplayReport>,
{
    // 1. The candidate's committed journal — validated present on the batched candidate leg.
    let committed = candidate
        .cohort_committed_tokens_by_stream
        .as_ref()
        .ok_or_else(|| RejectCtx {
            class: RejectClass::CohortReplayIntegrity,
            leg: "candidate",
            reason: "candidate cohort leg surfaced no committed tokens for the trusted-oracle \
                     tolerance gate (fail-closed)"
                .to_string(),
        })?;

    // 2. Spawn the TRUSTED oracle over the ORGANIZER reference weights and replay THIS journal. The
    // only inputs are TOKENS: the organizer-fixed replay seeds and the candidate journal to JUDGE —
    // never a weights path (see the provenance note above).
    let report = oracle(replay_seeds_by_stream, committed).map_err(|e| RejectCtx {
        class: RejectClass::CohortReplayIntegrity,
        leg: "candidate",
        reason: format!("trusted cohort_reference_replay oracle failed: {e}"),
    })?;

    // Re-shape the report into slot-ordered echoed-committed + reference-argmax rectangles, PINNING
    // SLOT ORDER (stream i must report slot i) — a slot-order violation is an integrity fault, not a
    // silently reordered comparison.
    let b = committed.len();
    if report.streams.len() != b {
        return Err(RejectCtx {
            class: RejectClass::CohortReplayIntegrity,
            leg: "candidate",
            reason: format!(
                "trusted oracle returned {} streams but the candidate committed {b} (shape \
                 mismatch — the replay did not describe the candidate journal)",
                report.streams.len()
            ),
        });
    }
    let mut echoed_committed: Vec<Vec<i64>> = Vec::with_capacity(b);
    let mut reference_argmax: Vec<Vec<i64>> = Vec::with_capacity(b);
    for (i, stream) in report.streams.iter().enumerate() {
        if stream.slot != i as i64 {
            return Err(RejectCtx {
                class: RejectClass::CohortReplayIntegrity,
                leg: "candidate",
                reason: format!(
                    "trusted oracle stream index {i} reported slot {} (SLOT ORDER violated — the \
                     reference cannot be aligned to the candidate journal)",
                    stream.slot
                ),
            });
        }
        echoed_committed.push(stream.positions.iter().map(|p| p.committed_token).collect());
        reference_argmax.push(
            stream
                .positions
                .iter()
                .map(|p| p.sequential_argmax)
                .collect(),
        );
    }

    // 3. N2 — verify the oracle replayed the candidate's REAL journal BEFORE counting mismatches.
    bench_core::cohort_tolerance::verify_replay_echo_matches_committed(
        committed,
        &echoed_committed,
    )
    .map_err(|e| RejectCtx {
        class: RejectClass::CohortReplayIntegrity,
        leg: "candidate",
        reason: format!("N2 integrity: {e}"),
    })?;

    // 4. The ≤10% PER-STREAM tolerance decision against the trusted reference argmax.
    let verdict = bench_core::cohort_tolerance::evaluate_cohort_token_tolerance(
        committed,
        &reference_argmax,
        bench_core::constants::COHORT_TOKEN_TOLERANCE_PER_THOUSAND,
    )
    .map_err(|e| RejectCtx {
        // A structural shape mismatch between the oracle output and the journal is an INTEGRITY fault
        // (the replay did not describe the journal), never a tolerance miss.
        class: RejectClass::CohortReplayIntegrity,
        leg: "candidate",
        reason: format!("cohort tolerance structural fault: {e}"),
    })?;
    if let Some(failing) = verdict.first_failing {
        return Err(RejectCtx {
            class: RejectClass::CohortTokenTolerance,
            leg: "candidate",
            reason: format!(
                "cohort stream {} diverged from the trusted reference argmax on {} of {} committed \
                 tokens (over the {}/1000 per-stream tolerance) — the whole run is rejected",
                failing.slot, failing.mismatches, failing.committed_len, verdict.tolerance_per_thousand
            ),
        });
    }

    // 5. REPORT-ONLY — the NEAR-TIE STATS SEAL over the SAME report the verdict above was reached
    // on. Reached ONLY once every verdict is final (N2 passed, no stream over the bar), and
    // returned as a VALUE: `cohort_near_tie_seal` is total, so no outcome of this line can become
    // a `RejectCtx`. Deliberately placed last so the ordering matches the guarantee.
    Ok(Some(cohort_near_tie_seal(&report)))
}

/// One golden's completed pair-loop outcome: the golden it measured (for the BY-BYTES per_prompt
/// binding) plus that prompt's accepted + rejected pairs. Finding R7 — the run measures EVERY
/// golden in the pool, one `PromptRun` (⇒ one `per_prompt` record) each.
struct PromptRun<'a> {
    prompt: &'a TimedPrompt,
    accepted: Vec<PairRecord>,
    rejected: Vec<RejectRecord>,
}

/// COHORT — the ONE cohort's completed pair-loop outcome: the sealed member list it measured plus
/// its accepted + rejected pairs. The batched counterpart of [`PromptRun`], carried as one value
/// so [`build_cohort_results`] takes the run it seals rather than loose parallel vectors.
struct CohortRun {
    members: Vec<CohortMember>,
    accepted: Vec<PairRecord>,
    rejected: Vec<RejectRecord>,
}

/// Run the full measure-job pair loop OVER THE GOLDEN POOL and assemble the superset
/// `results.json`. Pure over the leg seam so tests drive it with MOCK reports.
///
/// R15 — `measure_serial` runs the serial-control leg (ONE `mtp-timed` verb at depth 0, PINNED
/// head), `measure_candidate` the candidate leg (ONE `mtp-timed` verb at `--mtp-depth D`, DECLARED
/// head). Each closure encapsulates the spawn + ONE cool gate + timed verb + report parse for one
/// leg (one process per leg), returning the parsed [`LegInvocation`] (report + recorded gate
/// state), or a typed [`RunnerError`] on abort. The core owns the ONE gated retry per leg
/// ([`run_leg_with_retry`]) and seals ONLY the report's `parent_measured_seconds_per_token` (worker
/// self-timing is never scored), the engine-echoed `effective_spec`, and the candidate head
/// provenance.
///
/// Finding R7 — the run iterates EVERY golden in `goldens`: for each golden it runs its own pair
/// loop (alternating serial/candidate pairs until that prompt has `target_pairs` accepted, or the
/// per-prompt attempt budget is spent), producing ONE `per_prompt` record BOUND BY BYTES to that
/// golden's sha256. The measure closures are golden-agnostic — each golden's timed workload flows
/// through its own [`TimingParams`] (built from that golden's benchmark oracle) passed to the
/// closure. Argv order carries no scoring meaning.
///
/// FAILS CLOSED to `candidate_accepted = false` (die 5) if ANY prompt accepts `< min_pairs`
/// (the per-prompt floor) OR the run-total floor `accepted_pair_count >= min_pairs * pool_size`
/// is unmet. Sealed results carry ONLY accepted pairs, pooled across all prompts
/// (`accepted_pair_count == pairs.len() == sum of per-prompt accepted`), and
/// `prompt_count == pool_size == per_prompt.len()`.
// UNVERIFIED(measure-job): the per-golden pool iteration + BY-BYTES per_prompt binding is
// contract-derived (validate_golden_set / measure_prompt W:687-712,2112), not checked on a live box.
pub fn run_measure_job<FS, FC>(
    goldens: &[TimedPrompt],
    weights: &DirDigest,
    commit: &str,
    cfg: &MeasureJobConfig,
    mut measure_serial: FS,
    mut measure_candidate: FC,
) -> Result<MeasureJobOutcome, String>
where
    FS: FnMut(&TimingParams) -> bench_runner::Result<LegInvocation>,
    FC: FnMut(&TimingParams) -> bench_runner::Result<LegInvocation>,
{
    if goldens.is_empty() {
        return Err("measure-job requires at least one --golden (the pool is empty)".to_string());
    }
    // W3 — REGIME COHERENCE, checked before any measuring: the declared candidate spec and the
    // regime the candidate leg will run must agree, and the free-run series must use its RULED
    // window. Both are fail-closed rather than silently reconciled.
    validate_candidate_regime_coherent(cfg)?;

    let budget = cfg.target_pairs.max(1) * PAIR_ATTEMPT_BUDGET_MULTIPLE;
    let mut prompt_runs: Vec<PromptRun> = Vec::with_capacity(goldens.len());
    // R16 — fold every ACCEPTED leg's observed telemetry sample run-wide (max temp / min steady
    // freq); sealed as the top-level `telemetry` (OMITTED when no sample was ever observed).
    let mut telemetry = TelemetryAccumulator::default();

    // H6/H3 (cycle-3) — once the OFFICIAL path hits its immediate die-5 (a pair failing after its
    // one gated retry), we stop MEASURING further prompts but STILL emit an honest EMPTY per_prompt
    // record for every remaining pool prompt, so `prompt_count == pool_size == per_prompt.len()`
    // holds and every pool prompt is listed on the die-5 path (R16 per_prompt-always-emitted).
    let mut official_die5 = false;

    // (b) admission — the single-stream regime has NO trusted-oracle tolerance gate: its token
    // correctness is still enforced INLINE in the runner (teacher-forced exact match), so the pair
    // gate is a no-op here. (Only the batched cohort path replaces the inline die with the ≤10%
    // per-stream trusted-oracle gate — see [`run_cohort_measure_job`].) With no oracle report
    // there is nothing to characterize, so the REPORT-ONLY near-tie seal is `None` on every
    // single-stream pair — the record omits the field entirely, byte-unchanged.
    let mut token_gate =
        |_: &LegMeasurement| -> Result<Option<CohortNearTieSeal>, RejectCtx> { Ok(None) };

    // Finding R7 — iterate the WHOLE pool: one pair loop per golden, one per_prompt record each.
    for prompt in goldens {
        if official_die5 {
            // A prior prompt already failed a pair (official immediate die-5). Do NOT measure this
            // prompt; list it honestly with zero accepted pairs so the die-5 seal is complete.
            prompt_runs.push(PromptRun {
                prompt,
                accepted: Vec::new(),
                rejected: Vec::new(),
            });
            continue;
        }

        let params = timing_params(prompt, cfg.tokens)?;

        let mut accepted: Vec<PairRecord> = Vec::new();
        let mut rejects: Vec<RejectRecord> = Vec::new();

        // `attempts` bounds THIS prompt's loop (a rejecting candidate must terminate + fail
        // closed); the leg ORDER is keyed on the ACCEPTED-PAIR index (`accepted.len()`) WITHIN
        // this prompt, NOT the attempt count — so rejected attempts in between never skew the
        // prompt's accepted pairs toward one order (the alternation keys on accepted pairs).
        let mut attempts = 0usize;
        while accepted.len() < cfg.target_pairs {
            // LOCAL-DEV — the budget loop bounds this prompt's ATTEMPTS to `target_pairs ×
            // PAIR_ATTEMPT_BUDGET_MULTIPLE`; once spent, fall through to the per-prompt die-5 floor.
            // (OFFICIAL has no budget loop — it stops on the first failed pair, below.)
            if cfg.local_pair_budget && attempts >= budget {
                break;
            }
            let order = PairOrder::for_accepted_index(accepted.len());
            match run_pair(
                order,
                &mut measure_serial,
                &mut measure_candidate,
                &mut token_gate,
                &params,
                PairWindow {
                    steps: params.decode_steps,
                    scored_regime: &cfg.scored_regime,
                },
                PairSideChannels {
                    telemetry: &mut telemetry,
                    // Single-stream legs carry no per-stream channel; no exponent pair is certified
                    // (or needed) on this path.
                    per_stream_exponents: None,
                },
            ) {
                PairAttempt::Accepted(rec) => accepted.push(*rec),
                // A pair FAILED after its one gated retry (R19 class-agnostic; the retry lives in
                // `run_leg_with_retry`). The two paths diverge here:
                PairAttempt::Rejected(record) => {
                    rejects.push(record);
                    if !cfg.local_pair_budget {
                        // OFFICIAL/ranked (W:2005-2032) — a pair failing after its retry is an
                        // IMMEDIATE die-5. No budget loop trying more pairs: stop this prompt and
                        // the whole run right here.
                        official_die5 = true;
                        break;
                    }
                    // LOCAL-DEV — record the reject and keep attempting (bounded by the budget
                    // guard at the loop top).
                }
            }
            attempts += 1;
        }

        prompt_runs.push(PromptRun {
            prompt,
            accepted,
            rejected: rejects,
        });
    }

    // #108 (M1) — `build_results` REFUSES the seal (Err) when the series the pairs were MEASURED in
    // disagrees with the run's own regime rule; it never seals a MIXED descriptor from this path.
    let results = build_results(
        weights,
        commit,
        cfg,
        prompt_runs,
        SERIAL_CONTROL_DEPTH,
        telemetry.into_telemetry(),
    )?;
    // Invariant: accepted_pair_count == pairs.len() (pooled across every prompt).
    debug_assert_eq!(results.accepted_pair_count, results.pairs.len());
    // Invariant: prompt_count == pool_size == per_prompt.len().
    debug_assert_eq!(results.prompt_count, goldens.len());
    debug_assert_eq!(results.per_prompt.len(), goldens.len());
    Ok(MeasureJobOutcome {
        candidate_accepted: results.candidate_accepted,
        results,
    })
}

/// THE THREE SEAMS the cohort measure-job is driven through.
///
/// `measure_serial` and `measure_candidate` time one leg each. `oracle` is the (b)-admission
/// TRUSTED-ORACLE seam: given `(replay_seeds_by_stream, committed_by_stream)` it spawns the
/// organizer's TRUSTED worker over the ORGANIZER's reference weights and returns the per-stream
/// reference-argmax report. The orchestrator owns the GATE DECISION (N2 + the <=10% per-stream
/// tolerance) but NOT the spawn or the weights: the real oracle closure (built in main.rs) fixes
/// the trusted build (fail-closed resolver) and the organizer weights, and tests inject a mock.
/// Keeping the spawn behind this seam is what keeps the security-critical build+weights provenance
/// in main.rs while the orchestrator stays pure and unit-testable.
///
/// The three are DIFFERENT generic closure types, so this is a plain field grouping that relocates
/// no behaviour. What it buys is naming the seams as one thing, apart from the run's orchestration
/// inputs — which is what removed the too-many-arguments silencer this orchestrator carried
/// (#191).
pub struct CohortSeams<FS, FC, FO> {
    pub measure_serial: FS,
    pub measure_candidate: FC,
    pub oracle: FO,
}

/// COHORT (batch-8 brief §4.5) — run the BATCHED measure-job: ONE cohort (the whole pinned pool,
/// concurrently, slot order = pool order), ONE pair loop, `pairs_per_cohort` accepted pairs. The
/// pair MACHINERY is [`run_pair`] / [`run_leg_with_retry`] UNCHANGED — same alternation keyed on
/// the accepted-pair index, same one gated retry per leg, same official immediate-die-5 vs
/// local-dev budget-loop split, same fresh-process + one-cool-gate lifecycle inside the measure
/// closures — only the params type is the cohort one, so each leg times ONE window over all B
/// streams and its scored number is COHORT seconds-per-committed-token (D1).
///
/// `members` is the SEALED member list [`validate_cohort_membership`] produced (die-8, pre-GPU) —
/// threaded through rather than re-derived so the seal states exactly what was validated.
///
/// FAILS CLOSED to `candidate_accepted = false` (die 5) when the cohort accepts fewer than
/// `min_pairs` pairs — the per-cohort floor, the D2 translation of the per-prompt floor.
/// The three measure/oracle SEAMS ride in one [`CohortSeams`] (#191): a run's ORCHESTRATION INPUTS
/// (pool, membership, weights, commit, cfg) and the seams it is DRIVEN through are two different
/// kinds of thing, and separating them is what lets this function state its inputs without
/// silencing the too-many-arguments lint.
pub fn run_cohort_measure_job<FS, FC, FO>(
    goldens: &[TimedPrompt],
    members: Vec<CohortMember>,
    weights: &DirDigest,
    commit: &str,
    cfg: &MeasureJobConfig,
    seams: CohortSeams<FS, FC, FO>,
) -> Result<MeasureJobOutcome, String>
where
    FS: FnMut(&CohortTimingParams) -> bench_runner::Result<LegInvocation>,
    FC: FnMut(&CohortTimingParams) -> bench_runner::Result<LegInvocation>,
    // (b) admission — the TRUSTED-ORACLE seam. Given `(replay_seeds_by_stream, committed_by_stream)`
    // it spawns the organizer's TRUSTED worker over the ORGANIZER's reference weights and returns the
    // per-stream reference-argmax report. This function owns the GATE DECISION (N2 + the ≤10%
    // per-stream tolerance) but NOT the spawn/weights: the real oracle closure (built in main.rs)
    // fixes the trusted build (fail-closed resolver) and the organizer weights, and tests inject a
    // mock. Keeping the spawn behind this seam is what keeps the security-critical build+weights
    // provenance in main.rs while this core stays pure and unit-testable.
    FO: FnMut(
        &[Vec<i64>],
        &[Vec<i64>],
    ) -> bench_runner::Result<bench_protocol::CohortReferenceReplayReport>,
{
    let CohortSeams {
        mut measure_serial,
        mut measure_candidate,
        mut oracle,
    } = seams;
    let Some(point) = cfg.candidate_regime.scored_batch_point() else {
        return Err(format!(
            "run_cohort_measure_job requires the batched cohort regime; cfg declares {:?} — the \
             single-stream path is run_measure_job",
            cfg.candidate_regime
        ));
    };
    // Same coherence gate as the single-stream path (the batched regime is a free-run regime, so
    // the speculating-candidate and ruled-window rules bind identically).
    validate_candidate_regime_coherent(cfg)?;
    // D2 (RULED) — an OFFICIAL cohort run accepts exactly [`PAIRS_PER_COHORT_TARGET`] pairs: the
    // published even-n median is defined over that ruled sample count, so an official run at some
    // other target would publish a median over a different support. Local-dev may explore other
    // targets; the floor (`min_pairs <= target_pairs`) is already validated at parse.
    if !cfg.local_pair_budget && cfg.target_pairs != PAIRS_PER_COHORT_TARGET {
        return Err(format!(
            "official batched cohort run declares target_pairs {} but the RULED pairs_per_cohort \
             target is {PAIRS_PER_COHORT_TARGET} (David ruling 2026-08-26, superseding the \
             2026-08-24 ruling of 2: {PAIRS_PER_COHORT_TARGET} accepted pairs per scored window) \
             — refused; --local-dev may explore other targets",
            cfg.target_pairs
        ));
    }
    // D2 (RULED) — THE FLOOR IS PART OF THE RULING, not a separate dial. `target_pairs` alone does
    // not make an official run publish over the ruled support: the pair loop stops accepting at the
    // TARGET but only FAILS below `min_pairs`, so an official run declaring
    // `min_pairs < PAIRS_PER_COHORT_TARGET` would accept a short cohort — say 2 of the ruled 4 —
    // and publish a median over half the support the ruling bought, with nothing downstream saying
    // so. The parse-time check is only `min_pairs <= target_pairs`, which such a run satisfies, so
    // the target refusal above does NOT cover this and the two are genuinely independent gates.
    //
    // Enforced at this same PRE-GPU seam, and OFFICIAL-ONLY: `--local-dev` still explores any
    // floor below its target (that is the whole point of the dev path, and the retry-budget logic
    // above depends on it), so this must never widen into a blanket min == target rule.
    if !cfg.local_pair_budget && cfg.min_pairs != PAIRS_PER_COHORT_TARGET {
        return Err(format!(
            "official batched cohort run declares min_pairs {} but the RULED pairs_per_cohort \
             floor is {PAIRS_PER_COHORT_TARGET} (David ruling 2026-08-26: the floor equals the \
             target, so a short cohort cannot publish a median over a narrower support than the \
             ruling defines) — refused; --local-dev may explore other floors",
            cfg.min_pairs
        ));
    }
    // Belt-and-braces on the membership the caller validated pre-GPU: the seal below states these
    // facts, so re-assert them at the seam rather than trusting the call order.
    if members.len() != point.batch_size() as usize || goldens.len() != members.len() {
        return Err(format!(
            "cohort shape incoherent at the measure seam: {} members / {} goldens, expected \
             B={} of each (validate_cohort_membership must run first)",
            members.len(),
            goldens.len(),
            point.batch_size(),
        ));
    }
    for (member, golden) in members.iter().zip(goldens.iter()) {
        if !member.prompt_sha256.eq_ignore_ascii_case(golden.sha256()) {
            return Err(format!(
                "cohort slot {} member pin {} does not match the golden at that slot ({}) — slot \
                 order is pool order and the sealed member list must describe the run",
                member.slot_index,
                member.prompt_sha256,
                golden.sha256()
            ));
        }
    }

    // ONE params object for the whole cohort window (identical per-stream budget N, D4).
    let params = cohort_timing_params(goldens, cfg.tokens)?;

    // (b) admission — the ORACLE REPLAY SEEDS: each stream's decode seed tokens, in SLOT ORDER, taken
    // from the SAME organizer-controlled cohort params both legs run (never from anything
    // candidate-controlled). The oracle replays the candidate's committed tokens teacher-forced from
    // these seeds.
    let replay_seeds_by_stream: Vec<Vec<i64>> = params
        .streams
        .iter()
        .map(|s| s.decode_seed_tokens.clone())
        .collect();
    // (b) admission — the trusted-oracle TOKEN-CORRECTNESS GATE applied to the candidate leg after
    // each cohort run. It (1) reads the candidate's own committed journal, (2) spawns the trusted
    // oracle over the organizer reference weights via `oracle`, (3) N2-verifies the oracle replayed
    // THAT journal, then (4) applies the ≤10% per-stream tolerance bar; a failure rejects the pair.
    let mut token_gate =
        |candidate: &LegMeasurement| -> Result<Option<CohortNearTieSeal>, RejectCtx> {
            cohort_token_tolerance_gate(candidate, &replay_seeds_by_stream, &mut oracle)
        };

    let budget = cfg.target_pairs.max(1) * PAIR_ATTEMPT_BUDGET_MULTIPLE;
    let mut telemetry = TelemetryAccumulator::default();
    let mut accepted: Vec<PairRecord> = Vec::new();
    let mut rejects: Vec<RejectRecord> = Vec::new();
    let mut attempts = 0usize;
    while accepted.len() < cfg.target_pairs {
        // LOCAL-DEV — the budget loop bounds the cohort's attempts; OFFICIAL stops on the first
        // failed pair (immediate die-5), exactly as on the single-stream path.
        if cfg.local_pair_budget && attempts >= budget {
            break;
        }
        let order = PairOrder::for_accepted_index(accepted.len());
        match run_pair(
            order,
            &mut measure_serial,
            &mut measure_candidate,
            &mut token_gate,
            &params,
            PairWindow {
                steps: cfg.tokens,
                scored_regime: &cfg.scored_regime,
            },
            PairSideChannels {
                telemetry: &mut telemetry,
                // REPORT-ONLY (gap G2) — the CERTIFIED exponent pair the contract declared
                // (`ScoredExponents::certify`'s output; `build_cohort_results` refuses a batched
                // seal without one), for the per-pair composite DIAGNOSTIC only.
                per_stream_exponents: cfg.scored_exponents,
            },
        ) {
            PairAttempt::Accepted(rec) => accepted.push(*rec),
            PairAttempt::Rejected(record) => {
                rejects.push(record);
                if !cfg.local_pair_budget {
                    break;
                }
            }
        }
        attempts += 1;
    }

    let results = build_cohort_results(
        weights,
        commit,
        cfg,
        CohortRun {
            members,
            accepted,
            rejected: rejects,
        },
        SERIAL_CONTROL_DEPTH,
        telemetry.into_telemetry(),
    )?;
    debug_assert_eq!(results.accepted_pair_count, results.pairs.len());
    Ok(MeasureJobOutcome {
        candidate_accepted: results.candidate_accepted,
        results,
    })
}

// ---------------------------------------------------------------------------
// results.json superset (serves BOTH regimes — B-4 diff decides)
// ---------------------------------------------------------------------------

/// The scoring-agnostic `results.json` (seam 2 out). Satisfies DRAFT-WF validation
/// `@2145-2153` (parity_all_ok, accepted_pair_count, pairs length, aggregate means > 0) AND
/// carries the MEDIAN-regime fields (per-prompt raw ratios + medians) so the overlay/organizer
/// (A-3 / seam 3) picks the regime. Does NOT compute the final published score.
#[derive(Debug, Clone, Serialize)]
pub struct Results {
    /// R12 — the SEALED CONSTANT track id (workflow-declared), NOT `--tag`.
    pub track_id: String,
    /// R12 — the optional human track name; omitted from the seal when unavailable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub track_name: Option<String>,
    /// R12 — the per-run identity (`--tag`), sealed SEPARATELY from the constant `track_id`.
    pub tag: String,
    /// R16 (medium cycle-3) — the sealed run timestamp (`date -u +%Y-%m-%dT%H:%M:%SZ`, W:1866,1885).
    /// Threaded in from the caller (a fixed value in tests) so `build_results` stays deterministic.
    pub timestamp: String,
    /// R16 — the sealed `evaluation_target{target_id, explicit_prompt, prompt_sha256}` (the R13
    /// trio, or the honest default-pool marker when the trio is absent).
    pub evaluation_target: EvaluationTarget,
    /// R12 — the sealed mode discriminator ([`MEASURE_JOB_MODE`]).
    pub mode: &'static str,
    /// #105 H-A / W3 — the top-level SERIES DESCRIPTOR. For a HOMOGENEOUS run it is that run's
    /// single §5 series tag ([`TIMED_MODE`] = teacher-forced v1 for the Model-2 TF shape,
    /// `free_run_v1_1` for a free-run run). The per-leg truth is
    /// `pairs[].{serial,candidate}_timed_mode` and [`Results::timed_series`], and #108 (M1) makes
    /// those tags the SOURCE this descriptor is derived from rather than a restatement of cfg.
    ///
    /// measure-job NEVER seals [`MIXED_SERIES_DESCRIPTOR`] here: under the Fable ruling both legs
    /// share one regime, so an observed crossing is a defect and refuses the seal outright. The
    /// descriptor exists for the parse/overlay layer, which must be able to NAME a crossed file it
    /// did not produce — deliberately NOT either tag, so no downstream aggregation, leaderboard, or
    /// regression gate can read one series where two were measured.
    ///
    /// #105 cycle-5 — SCOPE OF THE ENFORCEMENT, precisely: benchd machine-checks this tag on the
    /// BASELINE_CALIBRATION pre-read ([`enforce_calibration_series_fence`], die-6 before banding)
    /// and again in the A-3 overlay's §5 series fence. This field is the seal a DOWNSTREAM consumer
    /// (the board) must key on to make the same check on its own comparisons — benchd cannot
    /// enforce a comparison it does not run.
    pub timed_mode: &'static str,
    /// W3 — the sealed SERIES DESCRIPTOR block: which regime each leg ran, whether the run is
    /// homogeneous, and whether the two legs' numbers are §5-comparable
    /// (`bench_core::free_run::timed_modes_comparable`, the machine-checked rule). `legs_comparable`
    /// is always COMPUTED from the two per-leg tags, never asserted.
    pub timed_series: TimedSeries,
    /// R15/R16 — the ONE timed REGIME both legs ran ([`TIMED_REGIME`] = `"tf-serial-timed"`; #105 H-A
    /// removed the false `"mtp-timed"` value, cycle-5 finding 3 the false `timed_verb` NAME — see
    /// [`TIMED_REGIME`] for why this is a regime label and not an invocation string). Medium
    /// (cycle-3) HONESTY — sealed ONLY when a timed measurement actually completed (>= 1 accepted
    /// pair whose echo validated); OMITTED on a die-5 path where NO timed measurement was taken, so
    /// the seal never asserts a regime ran when a none path was taken. W3 — ALSO omitted when the
    /// two legs ran DIFFERENT regimes: a single value would then assert one regime for a two-regime
    /// run. The per-leg regimes are in [`Results::timed_series`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timed_regime: Option<&'static str>,
    /// #105 cycle-5 finding 5 — the v1.1 POINTER, sealed when the candidate's DECLARED regime was
    /// downgraded to serial for the timed window ([`tf_downgrade_note`]). Teacher forcing feeds
    /// every token, so an mtp candidate cannot speculate here and is measured — honestly — as
    /// serial; this states that on the artifact and names where mtp scoring actually lives, so a
    /// reader who finds `candidate_spec.mode == "mtp"` next to a serial effective regime is not
    /// left to infer a bug. Omitted when no downgrade happened.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tf_downgrade_note: Option<&'static str>,
    /// R16 (medium cycle-3) — the candidate workspace + its accept verdict (`{workspace, verdict}`,
    /// W:1889). `verdict` is honest: `"ACCEPT"` iff the candidate cleared die-5, else `"REJECT"`.
    pub candidate: WorkspaceVerdict,
    /// R16 (medium cycle-3) — the baseline (pinned serial) workspace + its verdict (W:1890). The
    /// baseline workspace is a die-8 prereq (present + usable by measure time), so `"ACCEPT"`; the
    /// serial DENOMINATOR band drift is a SEPARATE die-6 sealed in `provenance.serial_band_outcome`.
    pub baseline: WorkspaceVerdict,
    /// R16 (medium cycle-3) — the top-level `decode_tokens` (= `--tokens`, W:1877), sealed alongside
    /// `mtp_depth`/`serial_control_depth`.
    pub decode_tokens: usize,
    /// R15 — the seed prefill is INSIDE the single timed decode window; there is NO separately-
    /// scored prefill phase ([`PREFILL_COMPONENT_NONE`] = `"none"`).
    pub prefill_component: &'static str,
    /// Only accepted pairs are sealed, all parity-ok by construction ⇒ always true.
    pub parity_all_ok: bool,
    pub accepted_pair_count: usize,
    /// finding R2 — the die-5 verdict SEALED as a fact: `accepted_pair_count >= min_pairs`. A
    /// run that die-5'd carries `candidate_accepted=false` so the overlay (seam 3) can fail-closed
    /// on it instead of scoring a rejected candidate green.
    pub candidate_accepted: bool,
    /// finding R2 — the `--min-pairs` threshold this run required, sealed so the overlay can
    /// re-check `accepted_pair_count >= min_pairs` against the fact rather than trusting a flag.
    pub min_pairs: usize,
    /// R16 — the live `min_pairs_per_prompt` seal (= `--min-pairs`, the PER-PROMPT floor). Same value
    /// as `min_pairs`, sealed under the exact live name (W:1852-1941).
    pub min_pairs_per_prompt: usize,
    /// R16 — the live `pairs_per_prompt` seal (= `--target-pairs`, the per-prompt accept target).
    pub pairs_per_prompt: usize,
    pub prompt_count: usize,
    /// R15 — the serial control's depth (0); the serial leg runs the same `mtp-timed` verb at
    /// `--mtp-depth 0`.
    pub serial_control_depth: usize,
    /// R15 — the candidate leg's REQUESTED `--mtp-depth D`, sealed only when one was actually
    /// requested. ABSENT (David ruling 2026-08-27) on an engine-decides run: this field is the
    /// REQUEST, never the operating depth — `require_spec_echo` forces the engine to echo the
    /// request verbatim, so it structurally cannot show an in-engine value (the 2026-08-27 depth
    /// experiment sealed a parroted 2 while the pin drafted at 1). Consumers judge depth by the
    /// echoed `effective_spec` and the realized `effective_mean_draft_len`, never this.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp_depth: Option<usize>,
    /// The candidate leg's DECLARED per-module speculative spec (`docs/spec-config-design.md`) and
    /// its honest source (`mtp-depth-flag` / `mtp-engine-default` / `cli-override`). The
    /// engine-echoed effective spec each
    /// leg actually ran is sealed per pair as `pairs[].serial_effective_spec` /
    /// `pairs[].candidate_effective_spec` (cycle-5 finding 6 — the old text named
    /// `*_wire_effective_spec`, a field name that does not exist on the record).
    pub candidate_spec: SpecConfig,
    pub candidate_spec_source: String,
    /// The baseline leg's DECLARED spec (defaults to `{"mode":"serial"}`) and its source.
    pub baseline_spec: SpecConfig,
    pub baseline_spec_source: String,
    pub pairs: Vec<PairRecord>,
    pub aggregate: Aggregate,
    /// One record per pool prompt on the single-stream path. EMPTY on a batched cohort run, whose
    /// unit of measurement is the COHORT: the pool prompts are all timed — concurrently, in one
    /// window — and their identities are sealed in `per_cohort[].members` instead. The
    /// `prompt_count == pool_size` seal still holds on both paths (it counts distinct pool
    /// prompts timed, not per_prompt records).
    pub per_prompt: Vec<PerPrompt>,
    /// COHORT (batch-8 brief D2) — the per-cohort seal (member list + cohort aggregates). Present
    /// EXACTLY on a batched cohort run; omitted on every single-stream run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_cohort: Option<Vec<PerCohort>>,
    /// COHORT (D9) — the PINNED cohort width this run scored, echoed-and-validated on every leg.
    /// Present exactly on a batched run. NAMING: pending the orchestrator's naming-convention check.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scored_batch_size: Option<u32>,
    /// COHORT (D2) — the per-cohort accepted-pair target (= `--target-pairs`; the unit the
    /// single-stream `pairs_per_prompt` seal plays on this path). Present exactly on a batched
    /// run. NAMING: pending the orchestrator's naming-convention check.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pairs_per_cohort: Option<usize>,
    /// COHORT — the per-cohort floor (= `--min-pairs`), the die-5 threshold the cohort's accepted
    /// pairs are held to. Present exactly on a batched run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_pairs_per_cohort: Option<usize>,
    /// R16 — the sealed run-wide `telemetry{max_gpu_temp, min_steady_freq_mhz}`. Medium (cycle-3) —
    /// ALWAYS emitted (matching the live shape); a quantity with no observed sample is an honest
    /// `null` inside the object, never a silently-dropped key and never a fabricated number.
    pub telemetry: Telemetry,
    /// Provenance (findings 1/2/8): the gate state summary, resolved leg executables, thermal
    /// thresholds + source, and the serial calibration reference threaded in fail-closed.
    pub provenance: Provenance,
    /// Rejected pairs, recorded (contributes nothing). Honest — no fabricated timing.
    pub rejected_pairs: Vec<RejectRecord>,
    pub commit: String,
    pub weights_hash: String,
}

/// W3 — the sealed SERIES DESCRIPTOR (PROTOCOL-v1.1.md §5, machine-checked). A `results.json`
/// carries an all-teacher-forced run OR an all-free-run run; this block states WHICH, per leg, so
/// the overlay/scoring path can refuse to aggregate a file whose legs disagree with it — and so a
/// reader never has to infer the measured quantity from a depth field.
///
/// `legs_comparable` is computed by `bench_core::free_run::timed_modes_comparable`, never asserted.
/// Under the Fable ruling it computes TRUE on every run measure-job produces (both legs share one
/// regime, [`serial_control_regime_for`]); it is still COMPUTED, and the overlay still refuses a
/// file whose sealed verdict disagrees with a recomputation, so a crossed file from any other
/// source is caught rather than assumed impossible. See the DESIGN NOTE on
/// [`MIXED_SERIES_DESCRIPTOR`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TimedSeries {
    /// The §5 series tag the SERIAL CONTROL leg's numbers belong to. Fable ruling: the SAME series
    /// as the candidate — the depth-0 control runs whichever regime the candidate runs, so the
    /// ratio's denominator and numerator are the same measured quantity.
    pub serial_leg_timed_mode: &'static str,
    /// The §5 series tag the CANDIDATE leg's numbers belong to (teacher-forced v1, or free-run v1.1).
    pub candidate_leg_timed_mode: &'static str,
    /// The timed REGIME label each leg ran (one invocation per leg).
    pub serial_leg_timed_regime: &'static str,
    pub candidate_leg_timed_regime: &'static str,
    /// True iff both legs ran the SAME regime (an all-teacher-forced or all-free-run run). #108
    /// (M1) — computed from the tags the ACCEPTED PAIRS carry, not from the run's config.
    pub homogeneous: bool,
    /// The §5 comparability verdict between the two legs' series tags, computed by the shared
    /// `timed_modes_comparable` rule over those same OBSERVED tags. FALSE on the mixed shape —
    /// which measure-job refuses to seal at all (see [`observed_series_tags`]).
    pub legs_comparable: bool,
}

/// Both regimes' aggregate fields. `*_mean` are the ratio-of-means-regime per-side means;
/// the `*_median`/`min` are the per-pair diagnostics; `raw_decode_speedup_median` is the
/// MEDIAN-regime figure (median of per-prompt raw ratio-of-means) the A-3 overlay targets.
#[derive(Debug, Clone, Serialize)]
pub struct Aggregate {
    pub baseline_serial_seconds_per_token_mean: f64,
    pub candidate_mtp_seconds_per_token_mean: f64,
    /// R16 — the POOLED raw ratio-of-means (pooled serial mean / pooled mtp mean). A sanity figure,
    /// NOT the published score.
    pub mtp_decode_speedup: f64,
    /// R16 NAME-TRAP — the per-PAIR LOWER-median (`sort | .[(len-1)/2 | floor]`) of the per-pair raw
    /// ratios. A diagnostic, DISTINCT from the published even-n `raw_decode_speedup_median`.
    pub mtp_decode_speedup_median: f64,
    pub mtp_decode_speedup_min: f64,
    /// R16 — the per-side pooling rule ([`AGGREGATION_RATIO_OF_MEANS`]).
    pub aggregation: &'static str,
    /// R16 — THE PUBLISHED SCORE: the even-n median of the per-prompt raw ratios (R7). The A-3
    /// overlay recomputes + compares this (R18).
    pub raw_decode_speedup_median: f64,
    /// R16 — the scoring anchor ([`SCORE_ANCHOR_SERIAL_ONE`], serial = 1.0).
    pub score_anchor: &'static str,
    /// R16 — the published scoring aggregation name ([`SCORING_AGGREGATION_MEDIAN_OF_PER_PROMPT`]).
    pub scoring_aggregation: &'static str,
    /// R16 — the PUBLISHED median rule ([`MEDIAN_RULE_EVEN_N`]); NAME-TRAP: for
    /// `raw_decode_speedup_median`, NOT the per-pair lower-median.
    pub median_rule: &'static str,
    /// R16 — the per-prompt raw ratio-of-means, in pool order (the vector the published median is
    /// computed over).
    pub raw_ratios: Vec<f64>,
    /// R16 — RETIRED diagnostic (informational): the even-n median of the per-prompt normalized
    /// ratios. NOT scored.
    pub normalized_decode_speedup_median_informational: f64,
    /// R16 — RETIRED diagnostic (informational): the per-prompt normalized ratios (raw / no-op ref),
    /// for prompts that carry a no-op reference. NOT scored.
    pub normalized_ratios_informational: Vec<f64>,
    /// R16 — the per-prompt effective mean draft length benchd COMPUTES from the per-round
    /// `acceptance_lengths` it collected (`free_run::effective_mean_draft_len`, the trusted-measured
    /// realized value) — NOT an engine echo. In pool order.
    pub effective_mean_draft_len_by_prompt: Vec<f64>,
    /// R16 — the total non-drafting round count benchd COMPUTES from the per-round
    /// `acceptance_lengths` (`free_run::non_drafting_round_count`, trusted-measured — NOT an engine
    /// echo), summed across prompts.
    pub non_drafting_round_count_total: usize,
    /// R16 — the sealed maximum MTP draft depth ([`MTP_MAX_DRAFT_DEPTH`] = 8). Sealed on the MTP
    /// arm and its serial control / teacher-forced runs (the arms whose candidate leg echoes `mtp`
    /// or `serial`) — the runs this cap governs. OMITTED on a DFlash run: the MTP cap is meaningless
    /// there, and stamping it would misdescribe the run; a dflash run carries `dflash_effective_depth`
    /// instead. Sealed by [`arm_draft_depth_seals`], which gates exactly one of the two depth fields on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp_max_draft_depth: Option<usize>,
    /// The DFlash arm's engine-ECHOED effective draft depth, structurally decoded out of the
    /// candidate leg's `effective_spec.dflash.depth` ([`SpecConfig::dflash_depth`]) — the dflash
    /// counterpart of `mtp_max_draft_depth`, NEVER a reuse of the MTP constant. Sealed EXACTLY on a
    /// DFlash run (candidate leg echoed `mode == "dflash"`); OMITTED on the MTP/serial arms, whose
    /// depth cap is `mtp_max_draft_depth`. Sealed by [`arm_draft_depth_seals`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dflash_effective_depth: Option<usize>,
    /// R16 — the per-prompt candidate head provenance sha256, in pool order.
    pub head_provenance_sha256_by_prompt: Vec<String>,
    /// R16 — the prefill component ([`PREFILL_COMPONENT_NONE`] = "none"; seed prefill inside decode).
    pub prefill_component: &'static str,
    /// R16 / #117 — the floor this run's §5 SERIES declares, from
    /// [`decode_speedup_floor_verdict`]: on `free_run_v1_1` the RULED
    /// [`FREE_RUN_DECODE_SPEEDUP_FLOOR`] (0.90, #109 comment 5353123259); on `teacher_forced_v1` the
    /// R16 NAME-TRAP LOOSE SANITY floor ([`DECODE_SPEEDUP_FLOOR`] = 0.50 = MIN_ACCEPTED_SPEEDUP),
    /// which is NOT a performance floor. Read it together with the series tag — the number alone
    /// does not say which gate ran.
    pub decode_speedup_floor: f64,
    /// R16 / #117 — whether the run cleared `decode_speedup_floor`, against the subject that floor
    /// governs: the PUBLISHED median on `free_run_v1_1`, the POOLED ratio-of-means
    /// (`mtp_decode_speedup`) on `teacher_forced_v1`. Never the per-pair minimum
    /// (`mtp_decode_speedup_min`) — that was the old, 4x-stricter, wrong semantic.
    pub decode_speedup_floor_met: bool,
    /// R16 — the wrapper-published plausibility ceiling ([`PUBLISHED_SPEEDUP_CEILING`] = 5.0).
    pub published_speedup_ceiling: f64,
}

/// One pool prompt's aggregate (MEDIAN regime): per-side means, the raw ratio-of-means, its
/// accepted-pair count, and the pool's no-op reference speedup (informational, NOT a divisor).
#[derive(Debug, Clone, Serialize)]
pub struct PerPrompt {
    /// finding R7 — this prompt's POSITION in the `--golden` pool (0-based). Argv order carries
    /// no scoring meaning (the binding is BY BYTES via `prompt_sha256`); this index is provenance
    /// only, so a consumer can correlate the per_prompt record with the golden it measured.
    pub prompt_index: usize,
    /// finding R4 — the prompt identity is the sha256 of the ACTUAL `--golden` bytes
    /// (`GoldenFixture::sha256`), NEVER an unverified copy of `timed_prompt_pool[0].sha256`.
    pub prompt_sha256: String,
    /// finding R4 — where `prompt_sha256` came from: `"contract-pool-match"` when the golden's
    /// real sha matched a `timed_prompt_pool` entry (and that entry's no-op ref is carried), or
    /// `"golden-bytes"` when it matched no pool entry (the golden's own sha is emitted, no pool
    /// no-op ref invented). So a consumer can never mistake a pool copy for the run's prompt.
    pub prompt_sha256_source: &'static str,
    pub parity_ok: bool,
    pub accepted_pair_count: usize,
    pub serial_seconds_per_token_mean: f64,
    pub mtp_seconds_per_token_mean: f64,
    pub raw_ratio_of_means: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noop_reference_decode_speedup: Option<f64>,
    /// R16 — RETIRED diagnostic (informational): the normalized ratio = `raw_ratio_of_means` /
    /// `noop_reference_decode_speedup`. Retained but NOT scored; OMITTED when this prompt carries no
    /// no-op reference (a `golden-bytes` miss) — never fabricated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalized_ratio: Option<f64>,
    /// R16 — the candidate leg's engine-echoed effective mean draft length (from this prompt's FIRST
    /// accepted pair). OMITTED when the prompt accepted no pair (a die-5 prompt) — never fabricated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_mean_draft_len: Option<f64>,
    /// R16 — the candidate leg's engine-echoed non-drafting round count (from the first accepted
    /// pair). OMITTED when the prompt accepted no pair.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub non_drafting_round_count: Option<usize>,
    /// R15 — the candidate (MTP) leg's `head_provenance.sha256` (the head the engine loaded),
    /// sealed from this prompt's first accepted pair (fills the R7 placeholder). Omitted when the
    /// prompt accepted no pairs (a die-5 prompt) — never fabricated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_provenance_sha256: Option<String>,
    /// Medium (#105) — the SINGLE engine-ECHOED effective spec (the wire [`SpecConfig`]) the candidate
    /// leg actually ran for this prompt, sealed from the first accepted pair (never the declared
    /// value). On this teacher-forced job it is `{"mode":"serial"}`. Omitted when the prompt accepted
    /// no pairs. The head the engine loaded is sealed separately as `head_provenance_sha256`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_spec: Option<SpecConfig>,
}

/// COHORT (batch-8 brief D2) — one cohort's aggregate: the sealed MEMBER LIST (slot order = pool
/// order), the per-side COHORT seconds-per-committed-token means, and the ratio-of-means
/// diagnostic. The batched counterpart of [`PerPrompt`] — but the MEDIAN's samples are the
/// accepted PAIRS' cohort ratios, not per-cohort records: one batched run has ONE cohort (the
/// whole pool) measured `pairs_per_cohort` times, so this record is the identity + diagnostic
/// seal, and `aggregate.raw_ratios` carries the per-pair samples the published even-n median is
/// computed over.
#[derive(Debug, Clone, Serialize)]
pub struct PerCohort {
    /// This cohort's index (0 — one scored cohort per run at the ruled B=8; a vector so a future
    /// multi-cohort ruling extends rather than reshapes the seal).
    pub cohort_index: usize,
    /// The DERIVED cohort identity ([`cohort_sha256`]): sha256 over the ordered member pins,
    /// recomputable by any consumer from `members`. Also stamped on every accepted pair as its
    /// `prompt_sha256` (the pair's measurement unit IS the cohort).
    pub cohort_sha256: String,
    /// The sealed member list: which pinned pool prompt occupies which slot (D2 — the whole pool,
    /// pool order, one prompt per slot; every member pinned by sha256 AND bytes).
    pub members: Vec<CohortMember>,
    /// The cohort width B this record describes (the ruled [`SCORED_BATCH_SIZE_B8`]).
    pub batch_size: u32,
    pub parity_ok: bool,
    pub accepted_pair_count: usize,
    /// Mean over accepted pairs of the serial-control leg's COHORT seconds-per-committed-token
    /// (`window_elapsed / (B * N)`, D1).
    pub serial_seconds_per_token_mean: f64,
    /// Mean over accepted pairs of the candidate leg's cohort seconds-per-committed-token.
    pub candidate_seconds_per_token_mean: f64,
    /// serial mean / candidate mean — the ratio-of-means DIAGNOSTIC (the published score is the
    /// even-n median of the per-PAIR ratios, `aggregate.raw_decode_speedup_median`).
    pub raw_ratio_of_means: f64,
    /// The candidate leg's benchd-computed draft stats + identities from the FIRST accepted pair
    /// (omitted on a die-5 cohort with no accepted pair — never fabricated).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_mean_draft_len: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub non_drafting_round_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_provenance_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_spec: Option<SpecConfig>,
    // -----------------------------------------------------------------------
    // COMPOSITE COHORT SCORING (Gemma track) — the SHARED-WINDOW composite (`composite` below,
    // computed by `shared_window_composite` from the parent clock alone) plus the window/token
    // DIAGNOSTICS the same windows also summarize.
    // -----------------------------------------------------------------------
    /// DIAGNOSTIC — the B streams' seed (prompt) token counts, summed ("the 8 seeds' prompt
    /// tokens"). Constant across accepted pairs (same cohort, same window shape every pair);
    /// sealed once here. NOT a scoring input: the composite divides WINDOWS by WINDOWS, so token
    /// counts cancel out of it entirely.
    pub prefill_token_total: usize,
    /// DIAGNOSTIC — `B * N` committed decode tokens. Same constancy note as `prefill_token_total`.
    pub decode_token_total: usize,
    /// DIAGNOSTIC — mean over accepted pairs of the serial-control leg's PREFILL window elapsed
    /// seconds (mirrors `serial_seconds_per_token_mean` for the decode side, which IS the
    /// ENFORCED whole-window figure — this is its prefill-only sub-window, not itself enforced).
    ///
    /// A MEAN, and therefore NOT the composite's input: the composite's `prefill_gain` is a ratio
    /// of SUMS over the same per-pair windows (`pairs[].cohort_phase_windows`). With every pair
    /// weighted equally the two agree numerically, but the sealed score is computed from the sums,
    /// never from these means — stated so a reader recomputing the score uses the right array.
    pub serial_prefill_window_seconds_mean: f64,
    /// DIAGNOSTIC — mean over accepted pairs of the candidate leg's PREFILL window elapsed
    /// seconds.
    pub candidate_prefill_window_seconds_mean: f64,
    /// DIAGNOSTIC — mean over accepted pairs of the serial-control leg's DECODE window elapsed
    /// seconds (NOT `serial_seconds_per_token_mean`'s ENFORCED whole window — this is the decode
    /// sub-window alone).
    pub serial_decode_window_seconds_mean: f64,
    /// DIAGNOSTIC — mean over accepted pairs of the candidate leg's DECODE window elapsed seconds.
    pub candidate_decode_window_seconds_mean: f64,
    /// The exponent pair `composite.composite_score` was raised to — the fixture-CERTIFIED
    /// identity ([`ScoredExponents::certify`]'s output), never the code constants read directly.
    /// Sealed unconditionally: certification is REQUIRED on every batched run (an uncertified
    /// config refuses the whole cohort path in [`build_cohort_results`]), so this fact is known
    /// even on a run whose `composite` is absent for some other reason.
    pub composite_scored_exponents: ScoredExponents,
    /// PUBLISHED COMPOSITE SCORE — `prefill_gain ^ 0.25 * decode_gain ^ 0.75` over the
    /// SHARED-WINDOW gains ([`shared_window_composite`]): benchd's own parent-clocked phase
    /// windows, summed across the accepted pairs, serial over candidate. NO engine-reported
    /// number reaches this value.
    ///
    /// `None` on exactly ONE path since the composite became the published score (David
    /// 2026-08-27, ATOMIC): a die-5 cohort, which accepted zero pairs and has therefore already
    /// refused on its own terms — `composite_absent_reason` says so. Every OTHER way the composite
    /// can fail to compute (a missing/degenerate window on an ACCEPTED pair, a degenerate gain)
    /// now fails the whole run in [`build_cohort_results`] instead of reaching this field. Never
    /// fabricated, never silently defaulted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub composite: Option<CompositeCohortScore>,
    /// Sealed WHENEVER `composite` is `None` — the FAIL-LOUD reason a reader would otherwise have
    /// to guess at. EXACTLY ONE of the two is present: an absent composite always carries a
    /// reason, and a present composite carries none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub composite_absent_reason: Option<String>,
    /// REPORT-ONLY (per-stream arm-fill lane, gap G9) — the SLOT-ORDER PROVENANCE of every
    /// per-stream attestation this run sealed: which cohort member each verdict slot attests.
    /// Present exactly when >= 1 accepted pair sealed a per-stream attestation; see
    /// [`PerStreamSlotOrderSeal`] for the rule and the checks behind it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_stream_attestation_slot_order: Option<PerStreamSlotOrderSeal>,
}

/// REPORT-ONLY (gap G9) — the sealed statement binding attestation SLOT ORDER to the cohort
/// MEMBER LIST, so a review can check "verdict slot `i` is member `i`" from the artifact alone.
///
/// [`PER_STREAM_SLOT_ORDER_RULE`] holds by construction, and `build_cohort_results` RE-ASSERTS
/// the two structural halves before sealing this object (refusing the seal on a violation, the
/// same posture as its phase-window invariant): (1) `members[i].slot_index == i` (slot order =
/// pool order, the die-8 membership gate's D2 rule); (2) every sealed verdict's `batch_size`
/// equals the member count (each per-slot verdict vector's length equals `batch_size` inside
/// `attest_leg` already). The remaining link — response slot `i` IS request slot `i` — is the
/// runner's per-slot oracle exact-match (each slot's committed stream must equal THAT slot's
/// golden continuation, so a permuted response cannot pass), and PR-A's verbatim carry preserves
/// that response order into the vectors the verdicts index.
#[derive(Debug, Clone, Serialize)]
pub struct PerStreamSlotOrderSeal {
    /// [`PER_STREAM_SLOT_ORDER_RULE`] — the machine-readable name of the binding.
    pub rule: &'static str,
    /// `members[i].prompt_sha256`, in slot order: the prompt identity attestation slot `i`
    /// attests, restated here so checking a verdict slot against its member needs no join.
    pub slot_prompt_sha256: Vec<String>,
}

/// REPORT-ONLY (gap G9) — the slot-order rule's sealed name: attestation verdict slot `i`
/// (every per-slot vector in [`PerStreamAttestationSeal`]) describes cohort member `i`
/// (`per_cohort[].members[i]`, slot order = pool order).
pub const PER_STREAM_SLOT_ORDER_RULE: &str = "attestation-slot-i-is-cohort-member-slot-i";

/// COMPOSITE (Gemma cohort scoring) — the published composite and its two component gains, as
/// computed by [`shared_window_composite`]. Every number here derives from benchd's own
/// parent-clocked phase windows; see [`PerCohort::composite`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CompositeCohortScore {
    /// `Σ_pairs serial_prefill_window_seconds / Σ_pairs candidate_prefill_window_seconds` over the
    /// ACCEPTED pairs — serial-anchored, so a faster candidate scores > 1. Ratio of SUMS, the
    /// direct reading of "add their numbers together"; identical to the per-stream-summed
    /// aggregate under honest attribution (α ≡ 1) on this rectangular lockstep cohort — see this
    /// file's SHARED-WINDOW header block for the equivalence and why the parent clock is the side
    /// of it that gets scored.
    pub prefill_gain: f64,
    /// The DECODE-side gain under the SAME ratio-of-sums rule, over
    /// `{serial,candidate}_decode_window_seconds`.
    pub decode_gain: f64,
    /// `prefill_gain ^ scored_exponents.prefill_gain_exponent * decode_gain ^
    /// scored_exponents.decode_gain_exponent` (David's ruling: prefill^0.25 * decode^0.75) —
    /// `scored_exponents` lives one level up, on [`PerCohort::composite_scored_exponents`].
    pub composite_score: f64,
    /// The floor `composite_score` is checked against, from the SAME regime-scoped decision
    /// ([`decode_speedup_floor_verdict`]) that resolves `aggregate.decode_speedup_floor` — ZERO
    /// new scoring constants, applying the existing free-run 0.90 floor to the published
    /// composite (the historical overlay precedent for a `^0.25 * ^0.75` figure).
    pub composite_speedup_floor: f64,
    /// SEAL-ONLY IN THIS ARTIFACT, exactly like `aggregate.decode_speedup_floor_met` (NOT wired
    /// to `candidate_accepted` or to this binary's exit code — see that field's doc for why). It
    /// is NOT unenforced: the FAIL-CLOSED enforcement seam is the overlay, and since David's
    /// 2026-08-27 ruling that seam gates on THIS composite (`crate::overlay`'s
    /// `validate_cohort_composite` requires it, re-derives it from the gains sealed beside it and
    /// refuses a disagreement as tamper, and the floor is applied to the composite it publishes) —
    /// not on `aggregate.raw_decode_speedup_median`, which is a diagnostic on the batched regime.
    pub composite_speedup_floor_met: bool,
}

/// SHARED-WINDOW COMPOSITE (David's ruling; the audit that produced it is recorded in this file's
/// SHARED-WINDOW header block) — the cohort's PUBLISHED composite score, computed from benchd's
/// OWN parent-clocked phase windows and from nothing else.
///
/// INPUT PROVENANCE, exhaustively. `windows` is the ACCEPTED pairs' [`PairCohortPhaseWindows`],
/// each of whose four seconds fields is an `Instant::now()` bracket taken on benchd's side of the
/// pipe (`bench_runner::timing::measure_batched_free_run_decode`). Not one engine-reported number
/// is read here. The engine-reported per-stream vectors (`prefill_ns_by_stream` /
/// `decode_ns_by_stream`, #188/#189) are REPORT-ONLY and are deliberately absent from this
/// function's SIGNATURE — "the score cannot read them" is therefore a property of the types, not
/// a promise about the body, and the anti-regression test
/// `composite_ignores_engine_reported_per_stream_ns_entirely` pins it end to end.
///
/// MATH — ratio of SUMS across the accepted pairs, per component, SERIAL-ANCHORED (a faster
/// candidate scores > 1):
///
/// ```text
/// prefill_gain = Σ serial_prefill_window_seconds / Σ candidate_prefill_window_seconds
/// decode_gain  = Σ serial_decode_window_seconds  / Σ candidate_decode_window_seconds
/// composite    = prefill_gain ^ exp_prefill * decode_gain ^ exp_decode
/// ```
///
/// The exponents are the CERTIFIED pair ([`ScoredExponents::certify`]'s output, threaded from the
/// contract through `MeasureJobConfig::scored_exponents`), never [`PREFILL_GAIN_EXPONENT`] /
/// [`DECODE_GAIN_EXPONENT`] read directly: an uncertified batched config never reaches this
/// function, because [`build_cohort_results`] refuses first.
///
/// This mirrors the per-stream-SUM semantics with α ≡ 1 (see the header block): on the
/// rectangular lockstep cohort the two aggregates are the same quantity, and the parent-clocked
/// side of the identity is the one with no attacker-controlled term in it.
///
/// REFUSALS — each returns `Err(reason)`. NEVER a fabricated number. What [`build_cohort_results`]
/// then does with the reason depends on whether the cohort measured anything (David's 2026-08-27
/// ATOMIC rule; see the match there): a die-5 cohort seals `composite: None` +
/// `composite_absent_reason`, because that run has already refused on its own terms, while a
/// cohort with accepted pairs and no computable composite FAILS THE WHOLE RUN rather than sealing
/// a scoreless artifact a consumer could still rank on under the superseded decode-only regime.
///
/// * ZERO accepted pairs (a die-5 cohort) — there is no measurement to divide.
/// * A NON-POSITIVE or NON-FINITE window on any accepted pair. This should be unreachable:
///   [`validate_leg_report`] requires a phase-split window on every batched leg FAIL-CLOSED, and
///   [`PairCohortPhaseWindows::compute`] clamps a non-finite/negative value to `0.0`. Guarded
///   anyway — defence in depth on the divisor of a PUBLISHED SCORE is cheap, and the alternative
///   to guarding is dividing by zero and sealing an infinity.
/// * A NON-FINITE or NON-POSITIVE gain or composite (overflowed sums, or any ratio that is not a
///   positive real).
fn shared_window_composite(
    windows: &[PairCohortPhaseWindows],
    scored_exponents: ScoredExponents,
    candidate_regime: LegRegime,
) -> Result<CompositeCohortScore, String> {
    if windows.is_empty() {
        return Err(
            "no accepted pair carries a parent-clocked phase window, so the cohort has no \
             measured prefill/decode time to take a ratio of (a die-5 cohort accepts zero pairs) \
             — refusing to seal a composite rather than fabricating one"
                .to_string(),
        );
    }
    let prefill_gain = window_sum_gain("prefill", windows, |w| {
        (
            w.serial_prefill_window_seconds,
            w.candidate_prefill_window_seconds,
        )
    })?;
    let decode_gain = window_sum_gain("decode", windows, |w| {
        (
            w.serial_decode_window_seconds,
            w.candidate_decode_window_seconds,
        )
    })?;
    // The classic challenge overlay form at the CERTIFIED exponents: prefill^0.25 * decode^0.75.
    let composite_score = prefill_gain.powf(scored_exponents.prefill_gain_exponent)
        * decode_gain.powf(scored_exponents.decode_gain_exponent);
    if !composite_score.is_finite() || composite_score <= 0.0 {
        return Err(format!(
            "the composite score (prefill_gain {prefill_gain} ^ {} * decode_gain {decode_gain} ^ \
             {}) is {composite_score}, not a positive finite value — refusing to seal a \
             degenerate score",
            scored_exponents.prefill_gain_exponent, scored_exponents.decode_gain_exponent,
        ));
    }
    // The EXISTING regime-scoped floor decision that resolves `aggregate.decode_speedup_floor` —
    // ZERO new scoring constants. On the batched regime that function reads its `published_median`
    // argument, so `composite_score` is passed for both rather than duplicating the branch here.
    // SEAL-ONLY, exactly like `aggregate.decode_speedup_floor_met`: not wired to
    // `candidate_accepted` or to any exit code (see `CompositeCohortScore::composite_speedup_
    // floor_met`).
    let (composite_speedup_floor, composite_speedup_floor_met) =
        decode_speedup_floor_verdict(candidate_regime, composite_score, composite_score);
    Ok(CompositeCohortScore {
        prefill_gain,
        decode_gain,
        composite_score,
        composite_speedup_floor,
        composite_speedup_floor_met,
    })
}

/// One component's SHARED-WINDOW gain: `Σ_pairs serial / Σ_pairs candidate` over the accepted
/// pairs' parent-clocked windows, with `pick` naming which pair of window fields the component
/// reads (prefill or decode).
///
/// RATIO OF SUMS, not a mean or median of per-pair ratios: the sums are formed FIRST, across every
/// accepted pair, and divided once. Every addend is checked positive and finite BEFORE it enters a
/// sum, so a degenerate window can neither silently contribute `0.0` to a denominator nor poison a
/// sum to `NaN` — it names itself and refuses. `component` and the pair INDEX are quoted in the
/// refusal so the reason identifies which number was unusable.
fn window_sum_gain(
    component: &str,
    windows: &[PairCohortPhaseWindows],
    pick: fn(&PairCohortPhaseWindows) -> (f64, f64),
) -> Result<f64, String> {
    let mut serial_sum = 0.0f64;
    let mut candidate_sum = 0.0f64;
    for (i, w) in windows.iter().enumerate() {
        let (serial, candidate) = pick(w);
        for (leg, seconds) in [("serial", serial), ("candidate", candidate)] {
            if !seconds.is_finite() || seconds <= 0.0 {
                return Err(format!(
                    "accepted pair {i}'s {leg} {component} window is {seconds} seconds — the \
                     composite is a ratio of parent-clocked windows and every accepted pair must \
                     contribute a POSITIVE FINITE one (validate_leg_report requires a phase-split \
                     window on every batched leg fail-closed, so this is defence in depth on a \
                     scored value); refusing to seal a composite over an unusable window"
                ));
            }
        }
        serial_sum += serial;
        candidate_sum += candidate;
    }
    // Both sums are strictly positive by the loop above; they can still reach `inf` by overflow,
    // and `inf / inf` is `NaN` — so the ratio itself is checked rather than assumed.
    let gain = serial_sum / candidate_sum;
    if !gain.is_finite() || gain <= 0.0 {
        return Err(format!(
            "the {component} gain (Σ serial {serial_sum} s / Σ candidate {candidate_sum} s over \
             {} accepted pairs) is {gain}, not a positive finite ratio — refusing to seal a \
             degenerate gain",
            windows.len()
        ));
    }
    Ok(gain)
}

// Per-stream timing instrumentation (spec steps 1-2) — the DIAGNOSTIC pairing of two legs'
// per-stream attestation verdicts into the shape `CompositeCohortScore` would eventually carry
// lives in `bench_core::per_stream_attestation::composite_diagnostic`, not here: this crate is
// bin-only (no lib target), so a pure function with no production caller would be flagged dead
// code, and the math itself has no benchd-specific concerns (it takes the exponent pair as
// plain `f64`s rather than depending on this crate's `ScoredExponents`) — its natural home is
// the same library crate every other pure scoring/consistency rule (`bench_core::score::score`,
// `bench_core::free_run::verify_cohort_consistency`) already lives in. Call it as
// `bench_core::per_stream_attestation::composite_diagnostic(&serial_verdict, &candidate_verdict,
// exponents.prefill_gain_exponent, exponents.decode_gain_exponent)` once a caller has both legs'
// `bench_core::per_stream_attestation::PerStreamAttestation` verdicts (`attest_leg`) in hand.
//
// UNATTESTED / UNSCORED, same posture as `attest_leg`'s own callers, and now PERMANENTLY so under
// the SHARED-WINDOW ruling: `PerCohort::composite` is computed by `shared_window_composite` from
// the parent clock, and no engine-reported per-stream number feeds it. That diagnostic's value is
// CALIBRATION EVIDENCE (does the engine's self-timing agree with benchd's clock?) — a disagreement
// is a finding to investigate, never a movement in the score.

/// Run provenance recorded alongside the timings.
#[derive(Debug, Clone, Serialize)]
pub struct Provenance {
    pub candidate_executable: String,
    pub baseline_executable: String,
    pub thermal: ThermalThresholds,
    /// R14 — the resolved baseline calibration the serial-band check used (mean/band/source +
    /// decode_tokens), or omitted when no `BASELINE_CALIBRATION` was provided. NOT fabricated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_calibration: Option<ResolvedCalibration>,
    /// R14 — whether the serial band was ENFORCED (`BASELINE_BAND_ENFORCE`, default true) and whether
    /// `--calibration-bootstrap` SKIPPED the check + marked the run for authoring.
    pub calibration_band_enforce: bool,
    pub calibration_bootstrap: bool,
    /// R14/R103 — the SEALED serial-band verdict (pooled serial mean vs the calibration mean/band/
    /// window: ratio + honest pass/fail + reason). Present ONLY when a calibration was resolved, the
    /// candidate was accepted (a valid pooled serial mean), and the run was not a bootstrap-author
    /// pass; omitted otherwise. Computed by the SAME pure `evaluate_serial_band` the die-6 verdict
    /// uses, so the seal never disagrees with the exit code. NOT fabricated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub serial_band_outcome: Option<SerialBandOutcome>,
    /// R13 — the declared `--target-id` (recognised input; keys the calibration), omitted when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_id: Option<String>,
    /// R13 — the declared `--exactness-probe` mode (the mtp-verify gate that consumes it is R15).
    pub exactness_probe: &'static str,
}

/// R16 NAME-TRAP — the per-pair LOWER-median: the order statistic at index `(len-1)/2` (floor),
/// i.e. the live wrapper's `sort | .[(len-1)/2 | floor]` rule (W). For an EVEN count this is the
/// LOWER of the two central order statistics (NOT their mean) — deliberately DISTINCT from the
/// published even-n `raw_decode_speedup_median` (which averages the two central). Uses `total_cmp`
/// with a finite guard BEFORE aggregation (finding 6); empty / non-finite-containing slice → `0.0`.
fn lower_median(values: &[f64]) -> f64 {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    sorted[(sorted.len() - 1) / 2]
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Minimum over the finite members (finding 6 guard); `0.0` for an empty / all-non-finite set.
fn min_finite(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(None, |acc: Option<f64>, v| {
            Some(acc.map_or(v, |m| m.min(v)))
        })
        .unwrap_or(0.0)
}

/// Assemble the `results.json` from the pool's per-golden pair loops (finding R7). ONE `per_prompt`
/// record per golden (BOUND BY BYTES to that golden's sha256, with its own no-op ref); `pairs[]`
/// pools EVERY accepted pair across all prompts; the aggregate means pool across all pairs; the
/// PUBLISHED `raw_decode_speedup_median` is the even-n median of the per-prompt raw ratio-of-means
/// (reusing [`bench_core::score::paired_decode_only_median`]). The die-5 verdict (finding R2/R7) is
/// the PER-PROMPT floor (EVERY prompt accepts `>= min_pairs`) AND the run-total floor
/// (`accepted >= min_pairs * pool_size`), computed here so it is sealed coherently with the pairs.
/// #108 (M1) — the OBSERVED §5 series tags of a run: the `(serial, candidate)` pair of per-leg tags
/// the ACCEPTED PAIRS actually carry, cross-checked against the tags this run's regime rule implies.
///
/// The seal must state what was MEASURED, not restate what was CONFIGURED. Every accepted pair
/// stamps `serial_timed_mode` / `candidate_timed_mode` from the regime its leg invocation reported
/// ([`LegInvocation::regime`]), so those tags are the only observation of the series a number
/// belongs to. This function turns them into the run-wide pair, refusing three ways:
///
/// 1. pairs that DISAGREE WITH EACH OTHER on a leg's tag (one `pairs[]` pooling two series — the
///    exact shape the overlay's §5 fence refuses in a file, refused here at the producer);
/// 2. an observed tag that DISAGREES with the cfg-derived expectation (`cfg.candidate_regime` and
///    the [`serial_control_regime_for`] control it implies) — the write side and the measured side
///    describing different runs;
/// 3. observed legs that CROSS series (serial != candidate) — see below.
///
/// All three are a HARD ERROR naming BOTH the observed and the expected tag, never a silently
/// sealed [`MIXED_SERIES_DESCRIPTOR`]. The reason is that measure-job cannot legitimately produce a
/// crossed run at all: [`serial_control_regime_for`] is the ONE rule that sets a leg's regime and it
/// gives both legs the same one. So a crossing observed HERE is a defect in this component (a leg
/// driven with the wrong regime), and sealing it as an honest MIXED two-series measurement would
/// publish that defect as a result. The MIXED descriptor stays alive for the PARSE/OVERLAY layer,
/// which validates files it did not produce and must still be able to name the crossed shape.
///
/// With NO accepted pair (a die-5 run) there is nothing to observe, so the cfg-derived tags are
/// sealed as the run's declared series — an honest statement of what the run was configured to
/// measure, and the only one available.
fn observed_series_tags(
    all_pairs: &[PairRecord],
    serial_regime: LegRegime,
    candidate_regime: LegRegime,
) -> Result<(&'static str, &'static str), String> {
    let expected_serial = serial_regime.timed_mode();
    let expected_candidate = candidate_regime.timed_mode();
    let Some(first) = all_pairs.first() else {
        // Nothing measured (die-5): the declared series is all there is to seal.
        return Ok((expected_serial, expected_candidate));
    };
    let (serial_tag, candidate_tag) = (first.serial_timed_mode, first.candidate_timed_mode);
    // (1) every accepted pair must agree with the first on BOTH legs.
    for (i, p) in all_pairs.iter().enumerate() {
        for (leg, observed, run_wide) in [
            ("serial", p.serial_timed_mode, serial_tag),
            ("candidate", p.candidate_timed_mode, candidate_tag),
        ] {
            if observed != run_wide {
                return Err(format!(
                    "measure-job REFUSES to seal: pairs[{i}].{leg}_timed_mode ({observed:?}) \
                     disagrees with pairs[0].{leg}_timed_mode ({run_wide:?}) — one pairs[] pooling \
                     two §5 series has no single series to describe, and its median would be taken \
                     over two physical quantities"
                ));
            }
        }
    }
    // (2) the observed tags must be the ones this run's regime rule implies.
    for (leg, observed, expected) in [
        ("serial", serial_tag, expected_serial),
        ("candidate", candidate_tag, expected_candidate),
    ] {
        if observed != expected {
            return Err(format!(
                "measure-job REFUSES to seal: the {leg} leg MEASURED in series {observed:?} but the \
                 run's regime rule expects {expected:?} (candidate regime {candidate_regime:?} ⇒ \
                 serial control {serial_regime:?}) — the seal must state the measured series, and a \
                 leg driven in a series the run did not declare is a defect, not a result"
            ));
        }
    }
    // (3) observed legs that cross series. Unreachable while (2) holds and
    // `serial_control_regime_for` is the identity, but checked on the OBSERVED values so the refusal
    // does not depend on that invariant holding.
    if serial_tag != candidate_tag {
        return Err(format!(
            "measure-job REFUSES to seal a CROSS-SERIES run: serial leg MEASURED {serial_tag:?}, \
             candidate leg MEASURED {candidate_tag:?} (expected {expected_serial:?} / \
             {expected_candidate:?}). PROTOCOL-v1.1 §5 — every pair's raw_ratio would divide two \
             DIFFERENT measured quantities. measure-job's own paths cannot produce this shape (one \
             rule sets both legs' regimes), so it is a defect here, NOT an honest MIXED measurement \
             to seal: `{MIXED_SERIES_DESCRIPTOR}` exists for the parse/overlay layer to name crossed \
             files it did not produce."
        ));
    }
    Ok((serial_tag, candidate_tag))
}

/// The R16 sealed EVALUATION TARGET — the R13 trio, or the honest default-pool marker. Identical
/// on the single-stream and the batched cohort path, so it is stated once.
fn evaluation_target(cfg: &MeasureJobConfig) -> EvaluationTarget {
    EvaluationTarget {
        target_id: cfg
            .target_id
            .clone()
            .unwrap_or_else(|| DEFAULT_EVALUATION_TARGET_ID.to_string()),
        explicit_prompt: cfg.prompt_sha256.is_some(),
        prompt_sha256: cfg.prompt_sha256.clone(),
    }
}

/// R14/R103 (single-stream) and D5 (batched) — the SERIAL-BAND verdict sealed into provenance,
/// computed by the SAME pure [`evaluate_serial_band`] the die-6 exit uses.
///
/// Only meaningful with a resolved calibration, an accepted candidate (so `pooled_serial_mean` is
/// a valid measurement), and a non-bootstrap run — bootstrap AUTHORS the band, it does not check
/// it, and the `calibration_bootstrap` flag already conveys that. Omitted otherwise, never
/// fabricated.
fn serial_band_outcome(
    cfg: &MeasureJobConfig,
    pooled_serial_mean: f64,
    candidate_accepted: bool,
) -> Option<SerialBandOutcome> {
    match &cfg.calibration {
        Some(cal) if candidate_accepted && !cfg.calibration_bootstrap => Some(
            evaluate_serial_band(pooled_serial_mean, cfg.tokens, cal, cfg.band_enforce),
        ),
        _ => None,
    }
}

/// The run's PROVENANCE block. Every field is a straight copy off the config except the band
/// outcome, which the caller computed with [`serial_band_outcome`].
fn provenance(
    cfg: &MeasureJobConfig,
    serial_band_outcome: Option<SerialBandOutcome>,
) -> Provenance {
    Provenance {
        candidate_executable: cfg.candidate_executable.clone(),
        baseline_executable: cfg.baseline_executable.clone(),
        thermal: cfg.thermal.clone(),
        baseline_calibration: cfg.calibration.clone(),
        calibration_band_enforce: cfg.band_enforce,
        calibration_bootstrap: cfg.calibration_bootstrap,
        serial_band_outcome,
        target_id: cfg.target_id.clone(),
        exactness_probe: cfg.exactness_probe.as_str(),
    }
}

/// The candidate/baseline WORKSPACE VERDICTS. The candidate verdict is HONEST — ACCEPT iff it
/// cleared die-5; the baseline workspace is a die-8 prereq present by measure time, so it is
/// always ACCEPT (its serial-band die-6 is sealed separately in `provenance.serial_band_outcome`).
fn workspace_verdicts(
    cfg: &MeasureJobConfig,
    candidate_accepted: bool,
) -> (WorkspaceVerdict, WorkspaceVerdict) {
    (
        WorkspaceVerdict {
            workspace: cfg.candidate_executable.clone(),
            verdict: if candidate_accepted {
                "ACCEPT"
            } else {
                "REJECT"
            },
        },
        WorkspaceVerdict {
            workspace: cfg.baseline_executable.clone(),
            verdict: "ACCEPT",
        },
    )
}

/// W3 / Fable ruling — the SERIES DESCRIPTOR trio (`timed_series`, the top-level `timed_mode`, and
/// the honest `timed_regime`), derived from what the run OBSERVED.
///
/// #108 (M1) — the write side does not RESTATE its own cfg input. `homogeneous` /
/// `legs_comparable` are computed from the per-leg tags the ACCEPTED PAIRS actually carry
/// ([`observed_series_tags`]), and the cfg-derived expectation is a CROSS-CHECK, not the source.
/// Any disagreement REFUSES the seal (`Err`, die-6-class hard error naming both sides) rather than
/// sealing a MIXED descriptor: measure-job's own paths cannot legitimately produce a crossed run
/// (the ONE [`serial_control_regime_for`] rule gives both legs the same regime), so an observed
/// crossing is a BUG in this component, and a bug must not be published as an honest two-series
/// measurement. [`MIXED_SERIES_DESCRIPTOR`] survives for the parse/overlay layer, which validates
/// hand-assembled / foreign files it did not produce.
///
/// The MIXED branch of `timed_mode` is therefore UNREACHABLE by construction; it is kept as a
/// total match so a future regime addition cannot silently seal one tag for two series.
///
/// R16 / #105 H-A — `timed_regime` HONESTY: the TRUTHFUL regime label ("tf-serial-timed", never
/// the false "mtp-timed") is sealed ONLY when a timed measurement actually completed (>= 1
/// accepted pair whose echo validated). A die-5 path where NO pair was ever accepted took a none
/// path — the regime never produced a scored measurement — so it OMITS the label rather than
/// asserting it ran. W3 — also omitted when the two legs ran DIFFERENT regimes, where a single
/// value would assert one regime for a two-regime run.
fn observed_timed_series(
    accepted: &[PairRecord],
    candidate_regime: LegRegime,
) -> Result<(TimedSeries, &'static str, Option<&'static str>), String> {
    let serial_regime = serial_control_regime_for(candidate_regime);
    let (serial_tag, candidate_tag) =
        observed_series_tags(accepted, serial_regime, candidate_regime)?;
    let timed_series = TimedSeries {
        serial_leg_timed_mode: serial_tag,
        candidate_leg_timed_mode: candidate_tag,
        serial_leg_timed_regime: serial_regime.timed_regime(),
        candidate_leg_timed_regime: candidate_regime.timed_regime(),
        // OBSERVED, not asserted: both are computed from the tags above, and the tags came from
        // the pair records (or, with no accepted pair to observe, from the run's one regime rule).
        homogeneous: serial_tag == candidate_tag,
        legs_comparable: bench_core::free_run::timed_modes_comparable(serial_tag, candidate_tag),
    };
    let timed_mode = if timed_series.homogeneous {
        candidate_tag
    } else {
        MIXED_SERIES_DESCRIPTOR
    };
    let timed_regime = if accepted.is_empty() || !timed_series.homogeneous {
        None
    } else {
        Some(serial_regime.timed_regime())
    };
    Ok((timed_series, timed_mode, timed_regime))
}

/// The ONE `Results` seal both build paths share. Every field `build_results` and
/// `build_cohort_results` copy IDENTICALLY is stated here, once; each site spreads this base with
/// `..` and names only the fields where the two paths genuinely differ.
///
/// The base IS the single-stream seal: the shape fields carry their single-stream values (`mode`,
/// the tf-downgrade note, an empty `per_cohort` block), so the batched site's literal reads as its
/// DELTA from the single-stream seal rather than as a second full copy. `serial_control_depth`
/// carries [`SERIAL_CONTROL_DEPTH`] and both sites restate the depth their caller passed.
///
/// R16 — the batched path's `min_pairs_per_prompt` / `pairs_per_prompt` are the shape-stable
/// mirrors of its honest per-cohort fields (same values under the per-prompt names), which is why
/// the two paths seal them identically and they live here.
///
/// #108 (M1) — REFUSES the seal (`Err`) when the series the pairs were MEASURED in disagrees with
/// the run's own regime rule; neither path ever seals a MIXED descriptor.
///
/// R14/R103 (single-stream) and D5 (batched) — the serial-band verdict sealed into provenance is
/// computed here from `aggregate.baseline_serial_seconds_per_token_mean`, so the mean the band
/// divides is the SAME value the seal publishes and the two cannot drift apart. On the batched
/// path that mean is the pooled COHORT serial seconds-per-committed-token; the calibration file's
/// own `timed_mode` was already fenced against the run's b8 series tag on the pre-read, so it was
/// measured at the same B by construction, and `decode_tokens` stays the PER-STREAM window N.
fn shared_results_base(
    weights: &DirDigest,
    commit: &str,
    cfg: &MeasureJobConfig,
    telemetry: Telemetry,
    pairs: Vec<PairRecord>,
    candidate_accepted: bool,
    aggregate: Aggregate,
) -> Result<Results, String> {
    let (timed_series, timed_mode, timed_regime) =
        observed_timed_series(&pairs, cfg.candidate_regime)?;
    let (candidate, baseline) = workspace_verdicts(cfg, candidate_accepted);
    let serial_band_outcome = serial_band_outcome(
        cfg,
        aggregate.baseline_serial_seconds_per_token_mean,
        candidate_accepted,
    );
    Ok(Results {
        track_id: cfg.track_id.clone(),
        track_name: cfg.track_name.clone(),
        tag: cfg.tag.clone(),
        timestamp: cfg.run_timestamp.clone(),
        // R16 — the sealed evaluation target (the R13 trio, or the honest default-pool marker).
        evaluation_target: evaluation_target(cfg),
        mode: MEASURE_JOB_MODE,
        timed_mode,
        timed_series,
        timed_regime,
        // #105 cycle-5 finding 5 — sealed only when the DECLARED candidate regime was actually
        // downgraded for the timed window (a declared-serial candidate seals no note). W3 — and only
        // on a TEACHER-FORCED candidate leg: a free-run candidate runs its declared mtp spec on the
        // wire, so nothing was downgraded and the note would be false.
        tf_downgrade_note: (!cfg.candidate_regime.is_free_run())
            .then(|| tf_downgrade_note(&cfg.candidate_spec))
            .flatten(),
        candidate,
        baseline,
        decode_tokens: cfg.tokens,
        prefill_component: PREFILL_COMPONENT_NONE,
        parity_all_ok: true,
        accepted_pair_count: pairs.len(),
        candidate_accepted,
        min_pairs: cfg.min_pairs,
        min_pairs_per_prompt: cfg.min_pairs,
        pairs_per_prompt: cfg.target_pairs,
        // The pool size the caller counted, and the depth it ran the serial control at: both sites
        // state their own.
        prompt_count: 0,
        serial_control_depth: SERIAL_CONTROL_DEPTH,
        mtp_depth: cfg.mtp_depth,
        candidate_spec: cfg.candidate_spec.clone(),
        candidate_spec_source: cfg.candidate_spec_source.clone(),
        baseline_spec: cfg.baseline_spec.clone(),
        baseline_spec_source: cfg.baseline_spec_source.clone(),
        pairs,
        aggregate,
        per_prompt: Vec::new(),
        // COHORT — the cohort seal fields belong to the batched path (build_cohort_results);
        // a single-stream run omits them entirely.
        per_cohort: None,
        scored_batch_size: None,
        pairs_per_cohort: None,
        min_pairs_per_cohort: None,
        telemetry,
        provenance: provenance(cfg, serial_band_outcome),
        // Rejected pairs contribute nothing but are recorded: both sites state their own.
        rejected_pairs: Vec::new(),
        commit: commit.to_string(),
        weights_hash: weights.sha256.clone(),
    })
}

fn build_results(
    weights: &DirDigest,
    commit: &str,
    cfg: &MeasureJobConfig,
    prompt_runs: Vec<PromptRun>,
    serial_control_depth: usize,
    telemetry: Telemetry,
) -> Result<Results, String> {
    let pool_size = prompt_runs.len();
    let mut per_prompt: Vec<PerPrompt> = Vec::with_capacity(pool_size);
    let mut all_pairs: Vec<PairRecord> = Vec::new();
    let mut all_rejects: Vec<RejectRecord> = Vec::new();
    // Finding R7 — the PER-PROMPT die-5 floor: EVERY prompt must accept `>= min_pairs`. A single
    // prompt below its floor fails the whole run closed (die-5), independent of the run total.
    let mut every_prompt_met_floor = true;

    for (prompt_index, mut run) in prompt_runs.into_iter().enumerate() {
        let serial_spts: Vec<f64> = run
            .accepted
            .iter()
            .map(|p| p.serial_seconds_per_token)
            .collect();
        let mtp_spts: Vec<f64> = run
            .accepted
            .iter()
            .map(|p| p.mtp_seconds_per_token)
            .collect();
        let serial_mean = mean(&serial_spts);
        let mtp_mean = mean(&mtp_spts);
        let raw_ratio_of_means = if mtp_mean > 0.0 {
            serial_mean / mtp_mean
        } else {
            0.0
        };

        // finding R4/R7 — the prompt IDENTITY is the sha256 of THIS golden's bytes (BIND BY BYTES),
        // never a cross-prompt or `timed_prompt_pool[0]` copy. Match the golden's real sha against
        // the contract pool: a hit sources THAT entry's no-op ref (labeled `contract-pool-match`);
        // a miss emits the golden's own sha labeled `golden-bytes` with NO fabricated pool no-op ref.
        let golden_sha256 = run.prompt.sha256().to_string();
        let (prompt_sha256_source, noop_ref) = match cfg
            .prompt_pool
            .iter()
            .find(|e| e.sha256.eq_ignore_ascii_case(&golden_sha256))
        {
            Some(e) => ("contract-pool-match", e.noop_decode_speedup),
            None => ("golden-bytes", None),
        };

        if run.accepted.len() < cfg.min_pairs {
            every_prompt_met_floor = false;
        }

        // R15 — seal the candidate (MTP) leg's head provenance + the engine-echoed effective spec
        // from this prompt's FIRST accepted pair (they are the candidate head/depth the engine
        // actually ran, consistent across the prompt's pairs). A prompt that accepted NO pair (a
        // die-5 prompt) has no report, so these are OMITTED — never fabricated.
        let head_provenance_sha256 = run
            .accepted
            .first()
            .map(|p| p.head_provenance_sha256.clone())
            .filter(|s| !s.trim().is_empty());
        let effective_spec = run
            .accepted
            .first()
            .map(|p| p.candidate_effective_spec.clone());
        // R16 — the candidate leg's engine-echoed draft stats from the FIRST accepted pair (OMITTED
        // for a die-5 prompt with no accepted pair — never fabricated).
        let effective_mean_draft_len = run.accepted.first().map(|p| p.effective_mean_draft_len);
        let non_drafting_round_count = run.accepted.first().map(|p| p.non_drafting_round_count);
        // R16 — the RETIRED informational normalized ratio = raw ratio-of-means / no-op reference.
        // Only when a positive no-op reference exists AND the raw ratio is finite; else OMITTED.
        let normalized_ratio = noop_ref.and_then(|nr| {
            if nr > 0.0 && raw_ratio_of_means.is_finite() {
                Some(raw_ratio_of_means / nr)
            } else {
                None
            }
        });

        // One per_prompt record PER golden (pool_size entries always — even a prompt that fell
        // below its floor is listed, with its real accepted count, so the die-5 seal is honest and
        // `prompt_count == pool_size == per_prompt.len()` holds unconditionally).
        per_prompt.push(PerPrompt {
            prompt_index,
            prompt_sha256: golden_sha256.clone(),
            prompt_sha256_source,
            parity_ok: true,
            accepted_pair_count: run.accepted.len(),
            serial_seconds_per_token_mean: serial_mean,
            mtp_seconds_per_token_mean: mtp_mean,
            raw_ratio_of_means,
            noop_reference_decode_speedup: noop_ref,
            normalized_ratio,
            effective_mean_draft_len,
            non_drafting_round_count,
            head_provenance_sha256,
            effective_spec,
        });

        // R16 (medium cycle-3) — stamp each accepted pair with THIS prompt's index + sha256 (the
        // live per-pair `prompt_index`/`prompt_sha256`, bound BY BYTES to the golden it measured).
        for p in run.accepted.iter_mut() {
            p.prompt_index = prompt_index;
            p.prompt_sha256 = golden_sha256.clone();
        }

        all_pairs.append(&mut run.accepted);
        all_rejects.append(&mut run.rejected);
    }

    // Finding R7 — the PUBLISHED score is a REAL median over the pool: the even-n median (mean of
    // the two central order statistics for even n) of the per-prompt raw ratio-of-means. Reuse the
    // shared `bench_core::score` median (the same rule the A-3 overlay recomputes, R18) rather than
    // duplicating it. Median-of-one for a single-golden pool is degenerate but valid.
    let per_prompt_ratios: Vec<f64> = per_prompt.iter().map(|p| p.raw_ratio_of_means).collect();
    let raw_decode_speedup_median =
        bench_core::score::paired_decode_only_median(&per_prompt_ratios);

    // The run-total floor still holds alongside the per-prompt floor (finding R7).
    let candidate_accepted =
        every_prompt_met_floor && all_pairs.len() >= cfg.min_pairs.saturating_mul(pool_size);

    // Aggregate POOLED across every accepted pair of every prompt (finding R7).
    let serial_spts: Vec<f64> = all_pairs
        .iter()
        .map(|p| p.serial_seconds_per_token)
        .collect();
    let mtp_spts: Vec<f64> = all_pairs.iter().map(|p| p.mtp_seconds_per_token).collect();
    let per_pair_ratios: Vec<f64> = all_pairs.iter().map(|p| p.raw_ratio).collect();
    let pooled_serial_mean = mean(&serial_spts);
    let pooled_mtp_mean = mean(&mtp_spts);
    // R16 — the POOLED raw ratio-of-means (sanity, NOT the score).
    let mtp_decode_speedup = if pooled_mtp_mean > 0.0 {
        pooled_serial_mean / pooled_mtp_mean
    } else {
        0.0
    };
    let mtp_decode_speedup_min = min_finite(&per_pair_ratios);
    // R16 (medium cycle-3) — `decode_speedup_floor_met` is a POOLED/published semantic, NOT a per-
    // pair min. The live wrapper checks the POOLED raw `mtp_decode_speedup` against the sanity floor
    // (W:2204/2256, `num_lt "${speedup}" "${MIN_ACCEPTED_SPEEDUP}"` where `speedup` is the pooled
    // serial_mean/mtp_mean), then seals `decode_speedup_floor_met: $floorok`. We previously used the
    // per-pair minimum ratio (4x stricter, wrong semantic).
    //
    // #117 — the floor and the verdict now come from ONE regime-scoped decision
    // ([`decode_speedup_floor_verdict`]). Teacher-forced keeps the wrapper's loose 0.50 on the
    // pooled ratio exactly as above; `free_run_v1_1` seals David's RULED 0.90 (#109 comment
    // 5353123259) against the PUBLISHED median, so a sub-floor free-run median fails closed instead
    // of passing at 0.50 (the window-4 drift this issue reports).
    let (decode_speedup_floor, decode_speedup_floor_met) = decode_speedup_floor_verdict(
        cfg.candidate_regime,
        mtp_decode_speedup,
        raw_decode_speedup_median,
    );

    // R16 — per-prompt-sourced aggregate vectors (in pool order), exact live names.
    let raw_ratios: Vec<f64> = per_prompt.iter().map(|p| p.raw_ratio_of_means).collect();
    let normalized_ratios_informational: Vec<f64> = per_prompt
        .iter()
        .filter_map(|p| p.normalized_ratio)
        .collect();
    // The retired informational median (even-n over the available normalized ratios; 0.0 if none).
    let normalized_decode_speedup_median_informational =
        if normalized_ratios_informational.is_empty() {
            0.0
        } else {
            bench_core::score::paired_decode_only_median(&normalized_ratios_informational)
        };
    let effective_mean_draft_len_by_prompt: Vec<f64> = per_prompt
        .iter()
        .filter_map(|p| p.effective_mean_draft_len)
        .collect();
    let non_drafting_round_count_total: usize = per_prompt
        .iter()
        .filter_map(|p| p.non_drafting_round_count)
        .sum();
    let head_provenance_sha256_by_prompt: Vec<String> = per_prompt
        .iter()
        .filter_map(|p| p.head_provenance_sha256.clone())
        .collect();

    // Gate the ARM-SPECIFIC depth field on from the candidate leg's engine-ECHOED effective spec
    // (the first prompt that accepted a pair — the echo is one arm across the run). A dflash run
    // seals `dflash_effective_depth` and drops the MTP cap; every other run keeps the MTP cap.
    let candidate_effective_spec = per_prompt.iter().find_map(|p| p.effective_spec.clone());
    let (mtp_max_draft_depth, dflash_effective_depth) =
        arm_draft_depth_seals(candidate_effective_spec.as_ref());

    let aggregate = Aggregate {
        baseline_serial_seconds_per_token_mean: pooled_serial_mean,
        candidate_mtp_seconds_per_token_mean: pooled_mtp_mean,
        mtp_decode_speedup,
        // R16 NAME-TRAP — the per-pair LOWER-median (distinct from the published even-n median).
        mtp_decode_speedup_median: lower_median(&per_pair_ratios),
        mtp_decode_speedup_min,
        aggregation: AGGREGATION_RATIO_OF_MEANS,
        raw_decode_speedup_median,
        score_anchor: SCORE_ANCHOR_SERIAL_ONE,
        scoring_aggregation: SCORING_AGGREGATION_MEDIAN_OF_PER_PROMPT,
        median_rule: MEDIAN_RULE_EVEN_N,
        raw_ratios,
        normalized_decode_speedup_median_informational,
        normalized_ratios_informational,
        effective_mean_draft_len_by_prompt,
        non_drafting_round_count_total,
        mtp_max_draft_depth,
        dflash_effective_depth,
        head_provenance_sha256_by_prompt,
        prefill_component: PREFILL_COMPONENT_NONE,
        decode_speedup_floor,
        decode_speedup_floor_met,
        published_speedup_ceiling: PUBLISHED_SPEEDUP_CEILING,
    };

    // Everything both seal sites copy identically comes from the shared base; this literal names
    // only what the single-stream path seals differently.
    Ok(Results {
        prompt_count: pool_size,
        serial_control_depth,
        per_prompt,
        rejected_pairs: all_rejects,
        ..shared_results_base(
            weights,
            commit,
            cfg,
            telemetry,
            all_pairs,
            candidate_accepted,
            aggregate,
        )?
    })
}

/// COHORT (batch-8 brief §4.5) — assemble the `results.json` of a BATCHED cohort run: ONE cohort
/// (the sealed member list), `pairs[]` = its accepted pairs, and THE PUBLISHED
/// `raw_decode_speedup_median` = the even-n median of the per-PAIR cohort ratios (D2: the cohort
/// is the measurement unit and the accepted pairs are the median's >= `min_pairs` samples —
/// reusing [`bench_core::score::paired_decode_only_median`], the SAME rule as the per-prompt
/// path). The die-5 verdict is the per-COHORT floor: `accepted >= min_pairs`.
///
/// Deliberate seal shape:
/// - every accepted pair is stamped with the DERIVED cohort identity ([`cohort_sha256`]) as its
///   `prompt_sha256` and `prompt_index = 0` (its cohort's index) — the pair's measurement unit is
///   the cohort, and a blank identity is never sealed;
/// - `per_prompt` is EMPTY and `per_cohort` carries the member list; `prompt_count` still equals
///   the pool size (every pinned prompt was timed — concurrently);
/// - the normalized-ratio informational family is EMPTY: the pinned per-prompt
///   `noop_decode_speedup` references are SINGLE-STREAM quantities, and dividing a cohort ratio
///   by one would cross series — omitted honestly, never fabricated;
/// - `pairs_per_prompt` / `min_pairs_per_prompt` keep their values (= the per-cohort target/floor)
///   for shape stability, with `pairs_per_cohort` / `min_pairs_per_cohort` as the honest names.
fn build_cohort_results(
    weights: &DirDigest,
    commit: &str,
    cfg: &MeasureJobConfig,
    run: CohortRun,
    serial_control_depth: usize,
    telemetry: Telemetry,
) -> Result<Results, String> {
    let Some(point) = cfg.candidate_regime.scored_batch_point() else {
        return Err(
            "build_cohort_results requires the batched cohort regime (the certified batch point \
             is the seal's width source)"
                .to_string(),
        );
    };
    // Orchestrator ruling (2026-08-23) — belt-and-braces, mirroring the batch-point guard just
    // above: `MeasureJobConfig::scored_exponents` must already carry the CERTIFIED exponent pair
    // (the caller certifies it against the fixture's `scored_exponents` before this config is
    // built — see `ScoredExponents::certify`); a batched config reaching here without one is a
    // wiring defect upstream, refused rather than silently falling back to the code constants.
    let Some(scored_exponents) = cfg.scored_exponents else {
        return Err(
            "build_cohort_results requires a CERTIFIED scored_exponents on the batched cohort \
             regime (the composite score's pinned exponent identity) — the caller must certify \
             the fixture's declaration via ScoredExponents::certify before constructing this \
             config; refusing rather than silently defaulting to the code constants"
                .to_string(),
        );
    };
    let CohortRun {
        members,
        mut accepted,
        rejected: rejects,
    } = run;
    let cohort_id = cohort_sha256(&members);
    // Stamp every accepted pair with the cohort identity (the analogue of the per-prompt stamp).
    for p in accepted.iter_mut() {
        p.prompt_index = 0;
        p.prompt_sha256 = cohort_id.clone();
    }

    let serial_spts: Vec<f64> = accepted
        .iter()
        .map(|p| p.serial_seconds_per_token)
        .collect();
    let candidate_spts: Vec<f64> = accepted.iter().map(|p| p.mtp_seconds_per_token).collect();
    let per_pair_ratios: Vec<f64> = accepted.iter().map(|p| p.raw_ratio).collect();
    let pooled_serial_mean = mean(&serial_spts);
    let pooled_candidate_mean = mean(&candidate_spts);
    let raw_ratio_of_means = if pooled_candidate_mean > 0.0 {
        pooled_serial_mean / pooled_candidate_mean
    } else {
        0.0
    };

    // D2 — THE PUBLISHED SCORE: the even-n median over the accepted pairs' cohort ratios (the
    // same shared median rule as the per-prompt path; `pairs_per_cohort = 4` (RULED 2026-08-26,
    // superseding the 2026-08-24 ruling of 2) keeps the sample count EVEN, so the
    // two-central-order-statistics rule matters — at n = 4 it is the mean of the 2nd and 3rd
    // sorted ratios, so the fastest and slowest cohort windows do not enter the published score).
    let raw_decode_speedup_median = bench_core::score::paired_decode_only_median(&per_pair_ratios);

    // The per-COHORT floor (the D2 translation of the per-prompt floor; one cohort, so the
    // run-total floor is the same predicate).
    let candidate_accepted = accepted.len() >= cfg.min_pairs;

    // #117 — the same regime-scoped floor decision as the single-stream path; the batched regime
    // reuses the free-run 0.90 against the published median (D1: machinery unchanged).
    let (decode_speedup_floor, decode_speedup_floor_met) = decode_speedup_floor_verdict(
        cfg.candidate_regime,
        raw_ratio_of_means,
        raw_decode_speedup_median,
    );

    // -------------------------------------------------------------------------------------------
    // COMPOSITE COHORT SCORING (Gemma track) — the SHARED-WINDOW composite: benchd's OWN
    // parent-clocked prefill/decode windows, summed across the accepted pairs, ratio'd
    // serial-over-candidate, raised to the CERTIFIED exponent pair. Zero engine-reported input.
    // See this file's SHARED-WINDOW header block for the ruling and the blocked per-stream design
    // it replaced, and `shared_window_composite` for the math and the refusal set.
    // -------------------------------------------------------------------------------------------
    // `validate_leg_report` requires `cohort_phase_windows` on every batched leg (fail-closed, the
    // same posture as `cohort_audit`), so every leg of every ACCEPTED pair on this (batched-only)
    // path carries one; `run_pair` combines the two legs' windows into `PairRecord.cohort_phase_
    // windows`. A pair accepted with no window sealed would mean that invariant broke somewhere
    // upstream — refused here (the WHOLE run, fail-closed) rather than silently scoring the
    // remaining pairs over a cohort whose window record is incomplete.
    let phase_windows: Vec<PairCohortPhaseWindows> = accepted
        .iter()
        .map(|p| {
            p.cohort_phase_windows.ok_or_else(|| {
                "batched cohort pair accepted with no phase-split window sealed on either leg — \
                 validate_leg_report should have refused this leg fail-closed before acceptance; \
                 refusing to seal partial window diagnostics rather than silently omitting them"
                    .to_string()
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    // THE SCORED PATH — the composite, over `phase_windows` (the parent clock) and the CERTIFIED
    // exponent pair. EXACTLY ONE of `composite` / `composite_absent_reason` is `Some`: the one
    // refusal that still seals (die-5, below) seals the named reason and no number, never a
    // fabricated score and never a silent omission.
    // Note what is NOT in scope here: `accepted`'s per-stream attestation seals are never
    // consulted, so no engine-reported duration can reach this value.
    //
    // ATOMIC PROMOTION (David's ruling 2026-08-27, which made this composite the PUBLISHED score
    // of a batched cohort run). The two refusal cases are NOT alike, so they are not treated
    // alike:
    //
    // * ZERO accepted pairs — the run has ALREADY refused on its own terms (die-5:
    //   `candidate_accepted` is false below, and there is no score for any regime to publish). The
    //   honest artifact naming the cause is the whole value of that path, so it still seals.
    // * ONE OR MORE accepted pairs and no computable composite — a run that measured something
    //   and cannot be scored. Sealing it would leave exactly the half-state a report-only ->
    //   scored promotion must not leave: `candidate_accepted` true, `aggregate.raw_decode_speedup_
    //   median` present and finite, and NO published score — an artifact a consumer could still
    //   rank on under the SUPERSEDED decode-only regime. The overlay refuses such a seal at
    //   publish time (`validate_cohort_composite`), but the artifact itself would still exist and
    //   still read as an accepted run; the run refuses HERE instead, by name, so the half-state is
    //   never written.
    let (composite, composite_absent_reason) =
        match shared_window_composite(&phase_windows, scored_exponents, cfg.candidate_regime) {
            Ok(score) => (Some(score), None),
            Err(reason) if phase_windows.is_empty() => (None, Some(reason)),
            Err(reason) => {
                return Err(format!(
                    "the batched cohort accepted {} pair(s) but no composite could be computed \
                     from their parent-clocked windows, and the composite IS this regime's \
                     published score — refusing the run rather than sealing a result that \
                     carries a decode-only median and no score: {reason}",
                    phase_windows.len()
                ))
            }
        };

    // REPORT-ONLY (gap G9) — the per-stream attestation SLOT-ORDER provenance, sealed when any
    // accepted pair sealed an attestation. Both structural halves of the rule are RE-ASSERTED
    // here before sealing (refused like the phase-window invariant above — an internal
    // contradiction must never be papered over by a confident-looking seal):
    // members in slot order, and every sealed verdict sized to exactly that member count.
    let any_attestation_sealed = accepted.iter().any(|p| {
        p.serial_per_stream_attestation.is_some() || p.candidate_per_stream_attestation.is_some()
    });
    let per_stream_attestation_slot_order = if any_attestation_sealed {
        for (i, m) in members.iter().enumerate() {
            if m.slot_index != i {
                return Err(format!(
                    "per-stream slot-order seal (G9): members[{i}].slot_index is {} — the \
                     membership gate seals slot order = pool order, so an out-of-place member \
                     means the list this seal would describe is not the one that ran; refusing \
                     the seal rather than binding verdict slots to a permuted list",
                    m.slot_index
                ));
            }
        }
        for p in accepted.iter() {
            for (leg, seal) in [
                ("serial", &p.serial_per_stream_attestation),
                ("candidate", &p.candidate_per_stream_attestation),
            ] {
                if let Some(v) = seal.as_ref().and_then(|s| s.verdict.as_ref()) {
                    if v.batch_size as usize != members.len() {
                        return Err(format!(
                            "per-stream slot-order seal (G9): a {leg} attestation verdict covers \
                             B={} slots but the cohort seals {} members — verdict slot i cannot \
                             be bound to member i; refusing the seal",
                            v.batch_size,
                            members.len()
                        ));
                    }
                }
            }
        }
        Some(PerStreamSlotOrderSeal {
            rule: PER_STREAM_SLOT_ORDER_RULE,
            slot_prompt_sha256: members.iter().map(|m| m.prompt_sha256.clone()).collect(),
        })
    } else {
        None
    };

    let prefill_token_total = phase_windows.first().map_or(0, |w| w.prefill_token_total);
    let decode_token_total = phase_windows.first().map_or(0, |w| w.decode_token_total);
    let serial_prefill_window_seconds_mean = mean(
        &phase_windows
            .iter()
            .map(|w| w.serial_prefill_window_seconds)
            .collect::<Vec<f64>>(),
    );
    let candidate_prefill_window_seconds_mean = mean(
        &phase_windows
            .iter()
            .map(|w| w.candidate_prefill_window_seconds)
            .collect::<Vec<f64>>(),
    );
    let serial_decode_window_seconds_mean = mean(
        &phase_windows
            .iter()
            .map(|w| w.serial_decode_window_seconds)
            .collect::<Vec<f64>>(),
    );
    let candidate_decode_window_seconds_mean = mean(
        &phase_windows
            .iter()
            .map(|w| w.candidate_decode_window_seconds)
            .collect::<Vec<f64>>(),
    );

    // First-accepted-pair identities/diagnostics (omitted on a die-5 cohort — never fabricated).
    let head_provenance_sha256 = accepted
        .first()
        .map(|p| p.head_provenance_sha256.clone())
        .filter(|s| !s.trim().is_empty());
    let effective_spec = accepted.first().map(|p| p.candidate_effective_spec.clone());
    let effective_mean_draft_len = accepted.first().map(|p| p.effective_mean_draft_len);
    let non_drafting_round_count = accepted.first().map(|p| p.non_drafting_round_count);

    // Same arm-specific depth gate as the single-stream path (computed before `effective_spec`
    // moves into `per_cohort`). The batch cohort driver is MTP/serial-only — dflash is single-
    // stream by mode (`mode_is_cohort_capable`) — so this keeps the MTP cap here; the shared helper
    // still drives it so the two paths can never diverge on the rule.
    let (mtp_max_draft_depth, dflash_effective_depth) =
        arm_draft_depth_seals(effective_spec.as_ref());

    let per_cohort = PerCohort {
        cohort_index: 0,
        cohort_sha256: cohort_id,
        batch_size: point.batch_size(),
        parity_ok: true,
        accepted_pair_count: accepted.len(),
        serial_seconds_per_token_mean: pooled_serial_mean,
        candidate_seconds_per_token_mean: pooled_candidate_mean,
        raw_ratio_of_means,
        effective_mean_draft_len,
        non_drafting_round_count,
        head_provenance_sha256: head_provenance_sha256.clone(),
        effective_spec,
        members,
        prefill_token_total,
        decode_token_total,
        serial_prefill_window_seconds_mean,
        candidate_prefill_window_seconds_mean,
        serial_decode_window_seconds_mean,
        candidate_decode_window_seconds_mean,
        // The CERTIFIED value threaded in above — "the certified values are what the seal
        // reports" (orchestrator ruling) — never the raw code constant read directly. Sealed
        // unconditionally: certification is required on every batched run regardless of whether
        // `composite` below is populated.
        composite_scored_exponents: scored_exponents,
        // The SHARED-WINDOW composite (or the named reason there isn't one) — see above.
        composite,
        composite_absent_reason,
        // REPORT-ONLY (gap G9) — computed above, before `members` moved into this record.
        per_stream_attestation_slot_order,
    };
    let prompt_count = per_cohort.members.len();

    let aggregate = Aggregate {
        baseline_serial_seconds_per_token_mean: pooled_serial_mean,
        candidate_mtp_seconds_per_token_mean: pooled_candidate_mean,
        mtp_decode_speedup: raw_ratio_of_means,
        mtp_decode_speedup_median: lower_median(&per_pair_ratios),
        mtp_decode_speedup_min: min_finite(&per_pair_ratios),
        aggregation: AGGREGATION_RATIO_OF_MEANS,
        raw_decode_speedup_median,
        score_anchor: SCORE_ANCHOR_SERIAL_ONE,
        scoring_aggregation: SCORING_AGGREGATION_MEDIAN_OF_PER_PROMPT,
        median_rule: MEDIAN_RULE_EVEN_N,
        // D2 — the samples the published median is computed over: the per-PAIR cohort ratios.
        raw_ratios: per_pair_ratios,
        // Single-stream no-op references do not divide cohort ratios (cross-series): empty, honest.
        normalized_decode_speedup_median_informational: 0.0,
        normalized_ratios_informational: Vec::new(),
        effective_mean_draft_len_by_prompt: effective_mean_draft_len.into_iter().collect(),
        non_drafting_round_count_total: non_drafting_round_count.unwrap_or(0),
        mtp_max_draft_depth,
        dflash_effective_depth,
        head_provenance_sha256_by_prompt: head_provenance_sha256.into_iter().collect(),
        prefill_component: PREFILL_COMPONENT_NONE,
        decode_speedup_floor,
        decode_speedup_floor_met,
        published_speedup_ceiling: PUBLISHED_SPEEDUP_CEILING,
    };

    // Everything both seal sites copy identically comes from the shared base; this literal names
    // only what the BATCHED path seals differently.
    Ok(Results {
        mode: COHORT_MEASURE_JOB_MODE,
        // The batched regime free-runs the declared speculating spec on the wire — nothing is
        // downgraded, so the note is never sealed here (same rule as the free-run path).
        tf_downgrade_note: None,
        prompt_count,
        serial_control_depth,
        per_cohort: Some(vec![per_cohort]),
        scored_batch_size: Some(point.batch_size()),
        pairs_per_cohort: Some(cfg.target_pairs),
        min_pairs_per_cohort: Some(cfg.min_pairs),
        rejected_pairs: rejects,
        ..shared_results_base(
            weights,
            commit,
            cfg,
            telemetry,
            accepted,
            candidate_accepted,
            aggregate,
        )?
    })
}

impl Results {
    /// Serialize to pretty, sorted-key JSON bytes (matches the `score.json` sealing shape).
    pub fn to_sealed_json(&self) -> Result<String, serde_json::Error> {
        let value = serde_json::to_value(self)?;
        serde_json::to_string_pretty(&value)
    }
}

#[cfg(test)]
mod seal_boundary_tests {
    use super::*;

    /// #134 — the `rejected_pairs[].reason` SINK. A transport failure now carries the worker's own
    /// stderr into this string, so the seal boundary must scrub it. Nothing secret-tier may reach
    /// `results.json`, and the reason must stay bounded regardless of how much the engine printed.
    #[test]
    fn sealed_reject_reason_scrubs_and_caps_engine_text() {
        let e = RunnerError::Protocol(format!(
            "engine closed the stream before returning a response (worker exited with status 9; \
             worker stderr tail: open /Users/operator/pool-goldens/sample-001.json failed | \
             AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCY | host=api.example.internal | {})",
            "P".repeat(8192)
        ));
        let sealed = sealed_reject_reason("candidate", &e);

        for secret in [
            "/Users/operator/pool-goldens",
            "wJalrXUtnFEMIK7MDENGbPxRfiCY",
            "api.example.internal",
        ] {
            assert!(
                !sealed.contains(secret),
                "secret-tier content sealed into rejected_pairs[].reason: {secret:?}"
            );
        }
        assert!(
            sealed.len() <= bench_runner::SEALED_REASON_BYTE_LIMIT,
            "sealed reason not capped: {} bytes",
            sealed.len()
        );
        // The leg prefix and the signature the classifier/readers key on survive.
        assert!(sealed.starts_with("candidate leg: "), "{sealed}");
        assert!(
            sealed.contains("engine closed the stream before returning a response"),
            "{sealed}"
        );
        // And the diagnosis is still there.
        assert!(sealed.contains("sample-001.json"), "{sealed}");
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn spawn_gate_free_run_legs_carry_the_flag_tf_legs_do_not() {
        // Window-prep gap: the engine's v1.1 fields are spawn-gated. BOTH free-run legs carry
        // --speculative-protocol v1.1 (the depth-0 control speaks the same session shape); TF
        // legs stay gate-off — that spawn IS the v1-compat proof.
        let mtp = SpecConfig::mtp(2);
        let serial = SpecConfig::serial();
        let cand_regime = candidate_regime_for_spec(&mtp);
        assert!(cand_regime.is_free_run());
        let ctl_regime = serial_control_regime_for(cand_regime);
        for regime in [cand_regime, ctl_regime] {
            let args = leg_spawn_args("/h", None, regime);
            let i = args.iter().position(|a| a == SPECULATIVE_PROTOCOL_FLAG);
            let i = i.expect("free-run leg argv must carry the spawn gate");
            assert_eq!(
                args.get(i + 1).map(String::as_str),
                Some(SPECULATIVE_PROTOCOL_V1_1)
            );
        }
        // TF pair: serial candidate regime, serial control — neither carries the flag.
        let tf_cand = candidate_regime_for_spec(&serial);
        assert!(!tf_cand.is_free_run());
        for regime in [tf_cand, serial_control_regime_for(tf_cand)] {
            let args = leg_spawn_args("/h", None, regime);
            assert!(
                !args.iter().any(|a| a == SPECULATIVE_PROTOCOL_FLAG),
                "a TF leg must spawn gate-off (v1-compat proof): {args:?}"
            );
        }
    }

    /// #109 window-2 finding 3 — THE ARGV-SURFACE FENCE. The engine's generic `runtime-worker` verb
    /// accepts exactly `{--weights, --mtp-head, --speculative-protocol}` and exits 1 on the FIRST
    /// unknown option, BEFORE the hello; benchd was spawning it with `--mtp-depth` and `--mtp-report`
    /// as well, so every leg of every pair died pre-GPU as *"protocol violation: engine closed the
    /// stream before returning a response"* (window-2 report, issue #109: legs B/C1 `accepted_pairs=0`,
    /// C2 die-6, D1/D2 never measured).
    ///
    /// The window's OWN failure cannot be reproduced here — it needs the real engine binary on a real
    /// box, and no offline fixture can stand in for "this Swift CLI rejects this flag". So the
    /// regression control is the fence itself: the argv benchd BUILDS, in both regimes, asserted
    /// against the accepted-set constant that mirrors the verb's `requireOnly(values:)` list, plus the
    /// two retired flag names by name so re-adding either fails here rather than on the box.
    #[test]
    fn w2f3_spawn_argv_is_a_subset_of_the_accepted_runtime_worker_surface() {
        let mtp = SpecConfig::mtp(2);
        let serial = SpecConfig::serial();
        let free_cand = candidate_regime_for_spec(&mtp);
        let tf_cand = candidate_regime_for_spec(&serial);
        let regimes = [
            ("free-run candidate", free_cand),
            ("free-run control", serial_control_regime_for(free_cand)),
            ("TF candidate", tf_cand),
            ("TF control", serial_control_regime_for(tf_cand)),
        ];
        for (leg, regime) in regimes {
            let args = leg_spawn_args("/heads/h", None, regime);
            // The transport prepends `runtime-worker --weights W`, so the full spawned argv is
            // exactly `--weights` plus whatever this builds.
            let full: Vec<String> = ["--weights".to_string(), "/weights/qwen".to_string()]
                .into_iter()
                .chain(args.iter().cloned())
                .collect();
            for flag in flags_in_args(&full) {
                assert!(
                    RUNTIME_WORKER_ACCEPTED_FLAGS.contains(&flag),
                    "{leg}: spawn argv carries {flag}, outside the verb's accepted surface \
                     {RUNTIME_WORKER_ACCEPTED_FLAGS:?} — the engine would exit 1 before the hello \
                     (argv: {full:?})"
                );
            }
            // The production fence agrees with this test, on the same argv.
            validate_spawn_argv(&args).unwrap_or_else(|e| panic!("{leg}: {e}"));
            // Named retirements: `--mtp-depth` belongs to a DIFFERENT binary's verb
            // (`mlxfast-swift mtp-timed`) and `--mtp-report` to no verb at all.
            for retired in ["--mtp-depth", "--mtp-report"] {
                assert!(
                    !full.iter().any(|a| a == retired),
                    "{leg}: {retired} is retired from the spawn argv (window-2 finding 3)"
                );
            }
            // `--mtp-head` is on EVERY leg: both legs load a head, so residency charges the
            // denominator too.
            assert_eq!(
                args.iter()
                    .position(|a| a == "--mtp-head")
                    .and_then(|i| args.get(i + 1)),
                Some(&"/heads/h".to_string()),
                "{leg}: every leg loads its own head"
            );
        }
        // The production fence is not vacuous: it REFUSES the exact argv the window caught.
        let e = validate_spawn_argv(&[
            "--mtp-head".to_string(),
            "/h".to_string(),
            "--mtp-depth".to_string(),
            "0".to_string(),
        ])
        .expect_err("the window's own argv must be refused");
        assert!(
            e.contains("--mtp-depth"),
            "the refusal names the offending flag: {e}"
        );
    }

    use super::*;
    use bench_core::constants::{
        BENCHMARK_DECODE_SEED_TOKENS, BENCHMARK_DECODE_STEPS, BENCHMARK_PREFILL_PROMPT_TOKENS,
        CORRECTNESS_PROMPT_TOKENS,
    };
    // THE ONE track-fixture schema. The measure-job used to declare a four-field `Contract` of its
    // own; these tests now read the same fixture through the same struct production does.
    use bench_core::contract::Contract;
    use bench_core::golden::load_golden_fixture;
    use serde_json::json;
    use std::cell::Cell;

    const PREFILL_TOKEN: i64 = 5;
    const SEED_TOKEN: i64 = 6;

    // R15 — the mock reports' scored seconds-per-token. The candidate is faster, so the raw ratio
    // (serial / mtp) is > 1. These are the `parent_measured_seconds_per_token` the core seals.
    const SERIAL_SPT: f64 = 0.040;
    const CANDIDATE_SPT: f64 = 0.020;
    // R15 — per-side head shas: the serial control loads the PINNED head, the candidate the DECLARED
    // BYO head. Distinct so a test can prove `head_provenance_sha256` comes from the CANDIDATE leg.
    const SERIAL_HEAD_SHA: &str =
        "5e51a10000000000000000000000000000000000000000000000000000000000";
    const CANDIDATE_HEAD_SHA: &str =
        "ca7d1de00000000000000000000000000000000000000000000000000000000";
    // W3 — the draft depth the v1.1 free-run candidate declares (and echoes on the wire).
    const FREE_RUN_DEPTH: u32 = 4;
    // #109 window-2 finding 3 — the R16 mock report echoes (EFFECTIVE_MEAN_DRAFT_LEN /
    // NON_DRAFTING_ROUND_COUNT) are retired with the `--mtp-report` file that carried them. Draft
    // statistics now have exactly one source, benchd's free-run histogram, so a fixture value for
    // "what the engine claimed" has nothing left to stand for.
    // R16 — per-leg observed telemetry samples: max temp across legs = 39.25, min steady freq = 1155.
    const SERIAL_GPU_TEMP_C: f64 = 38.5;
    const SERIAL_STEADY_FREQ_MHZ: f64 = 1180.0;
    const CANDIDATE_GPU_TEMP_C: f64 = 39.25;
    const CANDIDATE_STEADY_FREQ_MHZ: f64 = 1155.0;

    fn serial_telemetry() -> Option<TelemetrySample> {
        Some(TelemetrySample {
            gpu_temp_c: SERIAL_GPU_TEMP_C,
            steady_freq_mhz: SERIAL_STEADY_FREQ_MHZ,
        })
    }

    fn candidate_telemetry() -> Option<TelemetrySample> {
        Some(TelemetrySample {
            gpu_temp_c: CANDIDATE_GPU_TEMP_C,
            steady_freq_mhz: CANDIDATE_STEADY_FREQ_MHZ,
        })
    }

    fn oracle_decode_tokens() -> Vec<i64> {
        (0..BENCHMARK_DECODE_STEPS as i64)
            .map(|i| 700 + i)
            .collect()
    }

    /// A golden with a given case NAME (varies the bytes ⇒ a distinct sha256) and a given benchmark
    /// decode oracle. Same-oracle-but-different-name goldens are DISTINCT pool members (distinct
    /// sha). The golden's oracle only has to satisfy `timing_params` (enough decode tokens); the
    /// R15 mock reports are what the core scores.
    fn measure_golden_with(name: &str, oracle: Vec<i64>) -> TimedPrompt {
        let doc = json!({
            "version": 1,
            "model_type": "gemma4_text",
            "cases": [
                { "name": name, "prompt_tokens": vec![1i64; CORRECTNESS_PROMPT_TOKENS], "expected_tokens": vec![2i64; 64] }
            ],
            "benchmark": {
                "prefill_prompt_tokens": vec![1i64; BENCHMARK_PREFILL_PROMPT_TOKENS],
                "expected_prefill_token": PREFILL_TOKEN,
                "decode_seed_tokens": vec![1i64; BENCHMARK_DECODE_SEED_TOKENS],
                "expected_decode_seed_token": SEED_TOKEN,
                "expected_decode_tokens": oracle,
            }
        });
        let bytes = serde_json::to_vec(&doc).unwrap();
        TimedPrompt::Golden(
            load_golden_fixture(
                &bytes,
                64,
                CORRECTNESS_PROMPT_TOKENS,
                // The fixture declares `gemma4_text`, so it loads under the GEMMA row's identity
                // — the per-track table is what makes a non-125B golden loadable on this tree.
                &bench_core::constants::TrackModelIdentity::new("gemma4_text", 262_144, 64, 1_024),
                Some("gemma4_text"),
                None,
                None,
            )
            .unwrap(),
        )
    }

    fn measure_golden() -> TimedPrompt {
        measure_golden_with("case-a", oracle_decode_tokens())
    }

    /// A SECOND, distinct golden (different case name ⇒ different bytes ⇒ different sha256), for
    /// multi-prompt pool tests (the per_prompt-per-golden binding + the die-5-lists-every-prompt path).
    fn measure_golden_b() -> TimedPrompt {
        measure_golden_with("case-b", oracle_decode_tokens())
    }

    /// A SYNTHESIZED timed-prompt TAPE, in the shape the live `timed_prompt_pool` pins: schema
    /// derived from the reference Swift decoder (`QwenMTPReferenceGolden`) and cross-checked
    /// against the 8 live pinned objects; CONTENT INVENTED (no organizer bytes in this repo).
    /// `seed_marker` varies the bytes ⇒ a distinct sha256, so two tapes are distinct pool members.
    fn measure_tape_with(seed_marker: i64, chain: Vec<i64>) -> TimedPrompt {
        let rows: Vec<serde_json::Value> = chain
            .iter()
            .map(|t| {
                json!({
                    "sequential_argmax": t,
                    "top1_logit": 19.5,
                    "top2_logits": [19.5, 18.375],
                    "top2_tokens": [t, 321],
                })
            })
            .collect();
        let doc = json!({
            "emitted_tokens": chain,
            "reference_seed_token": SEED_TOKEN,
            "reference_self_consistent": true,
            "rows": rows,
            "seed_tokens": vec![seed_marker; BENCHMARK_DECODE_SEED_TOKENS],
        });
        let bytes = serde_json::to_vec(&doc).unwrap();
        TimedPrompt::Tape(
            bench_core::tape::load_timed_prompt_tape(
                &bytes,
                crate::testgolden::identity_125b(),
                None,
            )
            .unwrap(),
        )
    }

    /// The default synthesized tape: a full `BENCHMARK_DECODE_STEPS` reference chain.
    fn measure_tape() -> TimedPrompt {
        measure_tape_with(1, oracle_decode_tokens())
    }

    /// The byte length [`prompt_with_sha`] gives its synthetic prompts. Arbitrary but FIXED, so a
    /// pool entry declaring `bytes` can either agree with it or (deliberately) not.
    const SYNTHETIC_PROMPT_BYTES: u64 = 4_096;

    /// A prompt of a chosen SHAPE carrying a chosen sha256 (and a chosen byte count), for the
    /// pin-check tests.
    fn prompt_with_sha_bytes(kind: &str, sha: &str, byte_len: u64) -> TimedPrompt {
        match kind {
            PROMPT_KIND_TAPE => TimedPrompt::Tape(bench_core::tape::TimedPromptTape {
                seed_tokens: vec![1],
                reference_seed_token: 2,
                rows: vec![bench_core::tape::TimedPromptTapeRow {
                    sequential_argmax: 3,
                    top2_tokens: None,
                    top2_logits: None,
                    top1_logit: None,
                }],
                reference_self_consistent: Some(true),
                emitted_tokens: None,
                sha256: sha.to_string(),
                byte_len,
            }),
            _ => TimedPrompt::Golden(GoldenFixture {
                model_type: Some("gemma4_text".to_string()),
                model_provenance: None,
                cases: Vec::new(),
                correctness_gates: None,
                benchmark: None,
                sha256: sha.to_string(),
                byte_len,
            }),
        }
    }

    fn prompt_with_sha(kind: &str, sha: &str) -> TimedPrompt {
        prompt_with_sha_bytes(kind, sha, SYNTHETIC_PROMPT_BYTES)
    }

    fn tape_sha(sha: &str) -> TimedPrompt {
        prompt_with_sha(PROMPT_KIND_TAPE, sha)
    }

    /// #109 window-2 finding 3 — the WIRE `head_provenance` a conformant engine echoes on its
    /// `hello`: the head-identity channel that replaced the retired report file. `bytes`/`file_count`
    /// are audit-only companions of the sha benchd actually seals.
    fn head_prov(sha: &str) -> Option<bench_protocol::HeadProvenance> {
        Some(bench_protocol::HeadProvenance {
            sha256: sha.to_string(),
            bytes: 849_405_215,
            file_count: 7,
        })
    }

    /// One conformant leg's WIRE facts. #109 window-2 finding 3 — this REPLACED the `report()`
    /// helper that built a mock `--mtp-report` file; under the generic `runtime-worker` verb no such
    /// file exists.
    ///
    /// #109 W3 finding 5 — it used to carry a `head` too, and [`inv_wire`] stamped it onto EVERY
    /// teacher-forced leg. That was a fake no engine can produce: a TF leg is spawned gate-off and
    /// the engine gates `head_provenance` behind the v1.1 flag, so a real gate-off hello has no head
    /// at all. The fake modelling an impossible engine is precisely why the unsatisfiable
    /// requirement passed its own tests for two windows. benchd's OWN clock reading is now
    /// everything a conformant TF leg carries besides its (absent) spec echo.
    struct LegEcho {
        spt: f64,
    }

    fn echo(spt: f64) -> LegEcho {
        LegEcho { spt }
    }

    /// Wrap a leg's wire facts as a CONFORMANT TEACHER-FORCED `LegInvocation`. Coordinator ruling
    /// (#109, leg B) — a TF leg is spawned gate-off and requests no spec, so the conformant shape
    /// carries NO effective-spec echo; its serial regime is sealed from the spawn surface. #109 W3
    /// finding 5 — and NO `head_provenance`, which the same gate-off spawn forbids. Use [`inv_wire`]
    /// to inject an echo, or set `wire_head_provenance` directly, to drive either TF anomaly.
    fn inv(
        echo: LegEcho,
        gate_state: GateState,
        telemetry: Option<TelemetrySample>,
    ) -> LegInvocation {
        inv_wire(echo, None, gate_state, telemetry)
    }

    /// Like [`inv`] but with an EXPLICIT wire `effective_spec` echo, so a test can drive the
    /// tamper case (an echo on a gate-off TF leg) or, on a free-run leg, an absent/divergent one.
    fn inv_wire(
        echo: LegEcho,
        wire_effective_spec: Option<SpecConfig>,
        gate_state: GateState,
        telemetry: Option<TelemetrySample>,
    ) -> LegInvocation {
        LegInvocation {
            benchd_seconds_per_token: echo.spt,
            gate_state,
            quiesce: QuiesceRecord::default(),
            telemetry,
            wire_effective_spec,
            // #109 W3 finding 5 — the gate-off surface: no head_provenance, ever, on a TF leg.
            wire_head_provenance: None,
            // The teacher-forced (Model-2) shape by default; [`inv_free_run`] builds the v1.1 leg.
            regime: LegRegime::TeacherForcedV1,
            free_run_audit: None,
            // B2 — a TF leg drives neither free-run verb, so it opens no prefill window.
            phase_window: None,
            cohort_audit: None,
            cohort_phase_windows: None,
            per_stream_timing: None,
            cohort_committed_tokens_by_stream: None,
        }
    }

    /// W3 — a v1.1 FREE-RUN candidate leg: the SPECULATING wire echo the free-run window requires,
    /// plus the §3 AUDIT the runner produces after the §2.6 triple passes. `acceptance_lengths` is
    /// the per-round histogram, and `n` the verified-token count the window covered.
    fn inv_free_run(
        spt: f64,
        acceptance_lengths: Vec<u32>,
        n: usize,
    ) -> bench_runner::Result<LegInvocation> {
        let audit = free_run_audit(&acceptance_lengths, n);
        Ok(LegInvocation {
            benchd_seconds_per_token: spt,
            wire_head_provenance: head_prov(CANDIDATE_HEAD_SHA),
            gate_state: GateState::Fired,
            quiesce: QuiesceRecord::default(),
            telemetry: candidate_telemetry(),
            wire_effective_spec: Some(SpecConfig::mtp(FREE_RUN_DEPTH)),
            regime: LegRegime::FreeRunV1_1,
            free_run_audit: Some(audit),
            phase_window: Some(free_run_phase_window()),
            cohort_audit: None,
            cohort_phase_windows: None,
            per_stream_timing: None,
            cohort_committed_tokens_by_stream: None,
        })
    }

    /// The DFlash arm's draft-depth lever a test drives (an arbitrary non-default value distinct
    /// from the MTP cap of 8, so a test that finds it in the seal can only have read the ECHO —
    /// never a stray MTP constant).
    const DFLASH_DEPTH: u32 = 5;

    /// A v1.1 FREE-RUN candidate leg for the DFLASH arm — the exact shape of [`inv_free_run`] but
    /// echoing a `dflash` effective_spec carrying `dflash.depth` (single-stream: dflash is
    /// cohort-incapable by mode). Drives the dflash depth seal the same way the mtp free-run leg
    /// drives the mtp path.
    fn inv_free_run_dflash(
        spt: f64,
        depth: u32,
        acceptance_lengths: Vec<u32>,
        n: usize,
    ) -> bench_runner::Result<LegInvocation> {
        let audit = free_run_audit(&acceptance_lengths, n);
        Ok(LegInvocation {
            benchd_seconds_per_token: spt,
            wire_head_provenance: head_prov(CANDIDATE_HEAD_SHA),
            gate_state: GateState::Fired,
            quiesce: QuiesceRecord::default(),
            telemetry: candidate_telemetry(),
            wire_effective_spec: Some(SpecConfig::dflash(depth)),
            regime: LegRegime::FreeRunV1_1,
            free_run_audit: Some(audit),
            phase_window: Some(free_run_phase_window()),
            cohort_audit: None,
            cohort_phase_windows: None,
            per_stream_timing: None,
            cohort_committed_tokens_by_stream: None,
        })
    }

    /// B2 — the clock SPLIT a conformant free-run leg produces: a seed-prefill half, a decode half,
    /// and the whole window measured once. Shaped like a real reading (the halves account for the
    /// whole), so a closure-seam leg presents what `run_free_run_decode_phase_fresh` presents.
    fn free_run_phase_window() -> bench_core::prefill_window::PhaseWindow {
        bench_core::prefill_window::PhaseWindow {
            seed_prefill_elapsed_seconds: 0.25,
            decode_elapsed_seconds: 4.75,
            whole_window_elapsed_seconds: 5.0,
            prefill_token_total: BENCHMARK_DECODE_SEED_TOKENS,
            decode_token_total: BENCHMARK_DECODE_STEPS,
        }
    }

    /// Fable ruling — the closure-seam fake for the SAME-SERIES SERIAL CONTROL leg: the free-run
    /// regime, the depth-0 SERIAL wire echo, and the `[1]*N` histogram a non-speculating free-run
    /// window commits. The engine-driven counterpart is `free_run_serial_leg`; this one is for the
    /// negative controls whose subject is the CANDIDATE leg, so the control stays cheap and
    /// conformant.
    fn ok_free_run_serial() -> bench_runner::Result<LegInvocation> {
        let n = BENCHMARK_DECODE_STEPS;
        Ok(LegInvocation {
            benchd_seconds_per_token: SERIAL_SPT,
            wire_head_provenance: head_prov(SERIAL_HEAD_SHA),
            gate_state: GateState::Fired,
            quiesce: QuiesceRecord::default(),
            telemetry: serial_telemetry(),
            wire_effective_spec: Some(SpecConfig::serial()),
            regime: LegRegime::FreeRunV1_1,
            free_run_audit: Some(free_run_audit(&vec![1u32; n], n)),
            phase_window: Some(free_run_phase_window()),
            cohort_audit: None,
            cohort_phase_windows: None,
            per_stream_timing: None,
            cohort_committed_tokens_by_stream: None,
        })
    }

    /// W3 — build a [`FreeRunAudit`] the way the RUNNER does: through
    /// `bench_core::free_run::verify_consistency`, so a test can never fabricate an audit that the
    /// §2.6 triple would have rejected. `completed_work` is set to the conformant `R + 1`.
    fn free_run_audit(acceptance_lengths: &[u32], n: usize) -> FreeRunAudit {
        let rounds = acceptance_lengths.len() as i64;
        let sum: u64 = acceptance_lengths.iter().map(|&x| x as u64).sum();
        bench_core::free_run::verify_consistency(
            &bench_core::free_run::FreeRunResponse {
                tokens_len: n,
                acceptance_lengths: acceptance_lengths.to_vec(),
                drafted_total: sum + rounds as u64,
                accepted_total: sum.saturating_sub(rounds as u64),
                committed_total: n as u64,
                verify_replay_disagreements: None,
                verification: None,
            },
            n as u32,
            rounds + 1,
        )
        .expect("the fixture histogram must satisfy the §2.6 triple")
    }

    /// A conformant serial-control leg (serial regime, pinned head), `Fired` gate, with a telemetry sample.
    fn ok_serial() -> bench_runner::Result<LegInvocation> {
        Ok(inv(echo(SERIAL_SPT), GateState::Fired, serial_telemetry()))
    }

    /// A conformant candidate leg. #105 (Engine-can't-speculate-on-TF) — under Option A the candidate
    /// leg's TIMED window is SERIAL teacher-forced decode (depth 0), so it echoes the SERIAL effective
    /// regime, NOT its declared mtp depth. It still loads + reports its own workspace head
    /// (`CANDIDATE_HEAD_SHA`). `Fired` gate, with telemetry.
    fn ok_candidate() -> bench_runner::Result<LegInvocation> {
        Ok(inv(
            echo(CANDIDATE_SPT),
            GateState::Fired,
            candidate_telemetry(),
        ))
    }

    /// #105 (Engine-can't-speculate-on-TF) NEGATIVE CONTROL — a TEACHER-FORCED candidate leg that
    /// carries a wire `effective_spec` echo. Coordinator ruling (#109, leg B): a gate-off worker is
    /// gated out of emitting one, so ANY echo here is the tamper signal and [`tf_regime_is_serial`]
    /// refuses the leg; it folds into die-5. `spec` selects what the forger claims — an mtp echo
    /// (the original attack) or a plausible-looking `serial` one (which no longer buys a pass).
    fn candidate_tf_echo(spec: SpecConfig) -> bench_runner::Result<LegInvocation> {
        Ok(inv_wire(
            echo(CANDIDATE_SPT),
            Some(spec),
            GateState::Fired,
            candidate_telemetry(),
        ))
    }

    /// A retryable reject injected via the measure closure (the tests' leg-error seam). Finding R19
    /// — EVERY class is retryable (one gated retry per leg, then fold into die-5). A GateRejected
    /// behaves the SAME way.
    fn retryable_reject() -> RunnerError {
        RunnerError::AllocatorCacheNotDrained { reported: 4096 }
    }

    /// The regime this tree's track declares — the SINGLE-STREAM decode-only point. Test data
    /// since STEP 3 (David 2026-09-15) deleted `SCORED_REGIMES_BY_TRACK`; the production value is
    /// the track fixture's declaration, and the round-trip oracle pins the two to the same numbers.
    const TEST_REGIME: bench_core::constants::ScoredRegime = bench_core::constants::ScoredRegime {
        scored_batch_size: 1,
        prefill_gain_exponent: 0.0,
        decode_gain_exponent: 1.0,
    };

    fn test_cfg(min_pairs: usize, target_pairs: usize) -> MeasureJobConfig {
        MeasureJobConfig {
            scored_regime: TEST_REGIME,
            track_id: "qwen3.8-27b-mtp-v1".to_string(),
            track_name: None,
            tag: "qwen-mtp-mjob-test".to_string(),
            run_timestamp: "2026-08-19T00:00:00Z".to_string(),
            tokens: BENCHMARK_DECODE_STEPS,
            mtp_depth: Some(2),
            candidate_spec: SpecConfig::mtp(2),
            baseline_spec: SpecConfig::serial(),
            candidate_spec_source: SPEC_SOURCE_MTP_DEPTH_FLAG.to_string(),
            baseline_spec_source: "serial-default".to_string(),
            min_pairs,
            target_pairs,
            prompt_pool: vec![PromptPoolEntry {
                sha256: "poolsha".to_string(),
                // A fabricated sha with no file behind it: sha-only pin, no `bytes` half.
                bytes: None,
                noop_decode_speedup: Some(0.994),
            }],
            thermal: thermal_thresholds(0.70, "env-GPU_LOADED_UTIL-default-0.70"),
            candidate_executable: "cand-ws".to_string(),
            baseline_executable: "base-ws".to_string(),
            calibration: None,
            band_enforce: true,
            calibration_bootstrap: false,
            target_id: None,
            prompt_sha256: None,
            exactness_probe: ExactnessProbe::Once,
            // Default the tests to OFFICIAL (immediate die-5), matching the production default.
            // Budget-loop tests flip this to true explicitly.
            local_pair_budget: false,
            // W3 — the LEGACY Model-2 shape by default (both legs teacher-forced), so the existing
            // suite keeps testing exactly what it tested. [`free_run_cfg`] builds the v1.1 shape.
            candidate_regime: LegRegime::TeacherForcedV1,
            // Composite scoring is cohort-only; the single-stream default never consults this.
            // [`cohort_cfg`] overrides it to the certified ruled pair.
            scored_exponents: None,
        }
    }

    /// W3 — the config a SCORED v1.1 free-run run uses: a speculating candidate spec, the RULED
    /// window `N = FREE_RUN_DECODE_TOKENS`, and the free-run candidate regime. (The coherence guard
    /// in `run_measure_job` requires all three to agree.)
    fn free_run_cfg(min_pairs: usize, target_pairs: usize) -> MeasureJobConfig {
        MeasureJobConfig {
            tokens: FREE_RUN_DECODE_TOKENS,
            mtp_depth: Some(FREE_RUN_DEPTH as usize),
            candidate_spec: SpecConfig::mtp(FREE_RUN_DEPTH),
            candidate_regime: LegRegime::FreeRunV1_1,
            ..test_cfg(min_pairs, target_pairs)
        }
    }

    /// R15 — a conformant run over a single golden: serial + candidate legs both return well-formed
    /// mock reports. #105 — both legs echo the SERIAL effective regime (teacher-forced), regardless of
    /// the candidate's declared `cfg.mtp_depth`.
    fn identity_run(cfg: &MeasureJobConfig) -> MeasureJobOutcome {
        run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            cfg,
            |_p| ok_serial(),
            move |_p| ok_candidate(),
        )
        .unwrap()
    }

    #[test]
    fn spec_seal_records_declared_specs_sources_and_wire_echoes() {
        // docs/spec-config-design.md steps 4/5 — the run seals the DECLARED candidate/baseline specs
        // + their sources at the top level, and the engine-echoed WIRE effective_spec per leg per pair.
        let cfg = test_cfg(1, 1);
        let out = identity_run(&cfg);
        assert!(out.candidate_accepted);
        // Top-level declared specs + honest sources.
        assert_eq!(out.results.candidate_spec, SpecConfig::mtp(2));
        assert_eq!(out.results.baseline_spec, SpecConfig::serial());
        assert_eq!(out.results.candidate_spec_source, "mtp-depth-flag");
        assert_eq!(out.results.baseline_spec_source, "serial-default");
        // #105 — the SINGLE per-pair engine-echoed effective spec: BOTH legs → serial (teacher-forced;
        // the candidate DECLARES mtp(2) as provenance above, but the timed decode runs the serial
        // effective regime).
        let pair = &out.results.pairs[0];
        assert_eq!(pair.serial_effective_spec, SpecConfig::serial());
        assert_eq!(pair.candidate_effective_spec, SpecConfig::serial());
        // The sealed JSON carries them.
        let v: serde_json::Value =
            serde_json::from_str(&out.results.to_sealed_json().unwrap()).unwrap();
        assert_eq!(v["candidate_spec"]["mode"], json!("mtp"));
        assert_eq!(v["candidate_spec"]["mtp"]["depth"], json!(2));
        assert_eq!(v["baseline_spec"]["mode"], json!("serial"));
        assert_eq!(v["candidate_spec_source"], json!("mtp-depth-flag"));
    }

    #[test]
    fn pair_loop_reaches_min_pairs_and_seals_only_accepted() {
        let cfg = test_cfg(3, 4);
        let out = identity_run(&cfg);
        assert!(out.candidate_accepted, "identity run accepts the candidate");
        // Loop stops once target_pairs accepted.
        assert_eq!(out.results.accepted_pair_count, 4);
        // Invariant: accepted_pair_count == pairs length.
        assert_eq!(out.results.pairs.len(), 4);
        assert!(out.results.parity_all_ok);
        // Alternation: pair 0 mtp-first, pair 1 serial-first, ...
        assert_eq!(out.results.pairs[0].order, "mtp-first");
        assert_eq!(out.results.pairs[1].order, "serial-first");
    }

    #[test]
    fn results_superset_satisfies_draftwf_validation_and_carries_median_regime() {
        let cfg = test_cfg(3, 4);
        let out = identity_run(&cfg);
        let v = serde_json::to_value(&out.results).unwrap();
        // DRAFT-WF @2145-2153 predicates.
        assert_eq!(v["track_id"], json!("qwen3.8-27b-mtp-v1"));
        assert_eq!(v["parity_all_ok"], json!(true));
        assert!(v["accepted_pair_count"].as_u64().unwrap() >= cfg.min_pairs as u64);
        assert_eq!(
            v["pairs"].as_array().unwrap().len(),
            v["accepted_pair_count"].as_u64().unwrap() as usize
        );
        assert!(
            v["aggregate"]["baseline_serial_seconds_per_token_mean"]
                .as_f64()
                .unwrap()
                > 0.0
        );
        assert!(
            v["aggregate"]["candidate_mtp_seconds_per_token_mean"]
                .as_f64()
                .unwrap()
                > 0.0
        );
        // Per-pair fields present.
        assert!(v["pairs"][0]["serial_seconds_per_token"].as_f64().unwrap() > 0.0);
        assert!(v["pairs"][0]["mtp_seconds_per_token"].as_f64().unwrap() > 0.0);
        assert_eq!(v["pairs"][0]["parity_ok"], json!(true));
        // MEDIAN-regime fields.
        assert!(v["aggregate"].get("mtp_decode_speedup_median").is_some());
        assert!(v["aggregate"].get("raw_decode_speedup_median").is_some());
        let pp = &v["per_prompt"][0];
        assert!(pp["raw_ratio_of_means"].as_f64().unwrap() > 0.0);
        assert_eq!(pp["accepted_pair_count"].as_u64().unwrap(), 4);
        // finding R4 — the prompt identity is the golden's REAL sha (not the pool's "poolsha"); the
        // test pool sha does not match the golden, so the source is labeled golden-bytes and NO pool
        // no-op ref is fabricated.
        assert_eq!(pp["prompt_sha256"], json!(measure_golden().sha256()));
        assert_ne!(pp["prompt_sha256"], json!("poolsha"));
        assert_eq!(pp["prompt_sha256_source"], json!("golden-bytes"));
        assert!(
            pp.get("noop_reference_decode_speedup").is_none()
                || pp["noop_reference_decode_speedup"].is_null()
        );
    }

    #[test]
    fn r7_three_golden_pool_yields_three_per_prompt_and_pooled_median() {
        // R7 — a 3-golden pool (DISTINCT bytes via distinct case names, SAME benchmark oracle so a
        // single conformant engine matches all three) yields THREE per_prompt entries, one bound
        // BY BYTES to each golden's own sha256. `prompt_count == pool_size == per_prompt.len() == 3`;
        // `accepted_pair_count == pairs.len() == sum of per-prompt accepted`; the published
        // `raw_decode_speedup_median` is the even-n median over the 3 per-prompt raw ratios.
        let goldens = vec![
            measure_golden_with("case-a", oracle_decode_tokens()),
            measure_golden_with("case-b", oracle_decode_tokens()),
            measure_golden_with("case-c", oracle_decode_tokens()),
        ];
        let cfg = test_cfg(2, 3);
        let out = run_measure_job(
            &goldens,
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            |_p| ok_candidate(),
        )
        .unwrap();
        assert!(
            out.candidate_accepted,
            "every prompt meets its min_pairs floor"
        );
        assert_eq!(out.results.per_prompt.len(), 3, "one per_prompt per golden");
        assert_eq!(out.results.prompt_count, 3, "prompt_count == pool_size");
        // prompt_index is 0,1,2 in pool order; the identities are the goldens' own shas.
        for (i, pp) in out.results.per_prompt.iter().enumerate() {
            assert_eq!(pp.prompt_index, i, "per_prompt {i} carries its pool index");
            assert_eq!(
                pp.prompt_sha256,
                goldens[i].sha256(),
                "bound BY BYTES to its golden"
            );
            assert_eq!(
                pp.accepted_pair_count, 3,
                "each prompt reaches target_pairs"
            );
        }
        // pairs[] pools EVERY accepted pair across all prompts; the count invariants hold.
        let sum: usize = out
            .results
            .per_prompt
            .iter()
            .map(|p| p.accepted_pair_count)
            .sum();
        assert_eq!(
            out.results.accepted_pair_count, sum,
            "accepted == sum of per-prompt"
        );
        assert_eq!(
            out.results.accepted_pair_count,
            out.results.pairs.len(),
            "accepted == pairs.len()"
        );
        assert_eq!(
            out.results.accepted_pair_count, 9,
            "3 prompts * 3 target pairs"
        );
        // The PUBLISHED median is a real even-n median over the per-prompt raw ratios (reusing the
        // shared bench_core rule the A-3 overlay recomputes, R18) — recompute + compare exactly.
        let ratios: Vec<f64> = out
            .results
            .per_prompt
            .iter()
            .map(|p| p.raw_ratio_of_means)
            .collect();
        let expected = bench_core::score::paired_decode_only_median(&ratios);
        assert_eq!(
            out.results.aggregate.raw_decode_speedup_median, expected,
            "published median == even-n median over the per-prompt raw ratios"
        );
    }

    #[test]
    fn r7_prompt_below_min_pairs_dies5() {
        // R7 — the PER-PROMPT die-5 floor: if ANY prompt accepts `< min_pairs`, the whole run fails
        // closed (die-5) even when other prompts pass. golden A's candidate leg reports a good pair
        // (accepts); golden B's candidate leg ALWAYS parity-fails (token-mismatch), so golden B
        // accepts 0 < min_pairs=1 ⇒ die-5. The candidate closure distinguishes the goldens by their
        // decode oracle (golden B's starts at 900). per_prompt still lists BOTH prompts honestly.
        let divergent: Vec<i64> = (0..BENCHMARK_DECODE_STEPS as i64)
            .map(|i| 900 + i)
            .collect();
        let goldens = vec![
            measure_golden_with("case-a", oracle_decode_tokens()),
            measure_golden_with("case-b", divergent),
        ];
        let cfg = test_cfg(1, 1);
        let out = run_measure_job(
            &goldens,
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            |p: &TimingParams| -> bench_runner::Result<LegInvocation> {
                // Golden B (divergent oracle, first decode token 900) ⇒ the candidate leg reports a
                // token-mismatch (parity) reject every attempt; golden A reports a good pair.
                if p.expected_decode_tokens.first() == Some(&900) {
                    Err(RunnerError::TokenMismatch {
                        label: "benchmark decode token".to_string(),
                        step: 3,
                        expected: 903,
                        actual: 999_999,
                        expected_from_engine: false,
                    })
                } else {
                    ok_candidate()
                }
            },
        )
        .unwrap();
        assert!(
            !out.candidate_accepted,
            "a prompt below min_pairs must die 5"
        );
        assert_eq!(
            out.results.per_prompt.len(),
            2,
            "per_prompt lists BOTH prompts even on die-5"
        );
        assert_eq!(out.results.prompt_count, 2);
        assert_eq!(
            out.results.per_prompt[0].accepted_pair_count, 1,
            "golden A met its floor"
        );
        assert_eq!(
            out.results.per_prompt[1].accepted_pair_count, 0,
            "golden B fell below its floor"
        );
        // The pooled pairs invariant still holds (only golden A's accepted pair is sealed).
        assert_eq!(out.results.accepted_pair_count, out.results.pairs.len());
        assert_eq!(out.results.accepted_pair_count, 1);
        assert!(
            out.results
                .rejected_pairs
                .iter()
                .any(|r| r.class == "token-mismatch-parity"),
            "golden B's parity failures are sealed for provenance"
        );
    }

    #[test]
    fn r7_single_golden_pool_median_of_one() {
        // R7 — a single-golden pool still works: one per_prompt, prompt_count 1, and the published
        // median is the degenerate-but-valid median-of-one (that prompt's raw ratio-of-means).
        let cfg = test_cfg(2, 3);
        let out = identity_run(&cfg);
        assert!(out.candidate_accepted);
        assert_eq!(out.results.per_prompt.len(), 1);
        assert_eq!(out.results.prompt_count, 1);
        assert_eq!(out.results.per_prompt[0].prompt_index, 0);
        let only = out.results.per_prompt[0].raw_ratio_of_means;
        assert_eq!(
            out.results.aggregate.raw_decode_speedup_median, only,
            "median-of-one is that prompt's raw ratio"
        );
        assert_eq!(out.results.accepted_pair_count, out.results.pairs.len());
    }

    #[test]
    fn r7_per_prompt_sha256_distinct_and_each_carries_own_noop() {
        // R7 — every per_prompt sha256 is DISTINCT (the pool is distinct-by-bytes), and each prompt
        // carries ITS OWN no-op reference from the contract-pool entry matching that prompt's sha
        // (never a cross-prompt copy). The contract pool lists all three shas with distinct noops.
        let goldens = vec![
            measure_golden_with("case-a", oracle_decode_tokens()),
            measure_golden_with("case-b", oracle_decode_tokens()),
            measure_golden_with("case-c", oracle_decode_tokens()),
        ];
        let mut cfg = test_cfg(2, 3);
        cfg.prompt_pool = vec![
            PromptPoolEntry {
                sha256: goldens[0].sha256().to_string(),
                bytes: Some(goldens[0].byte_len()),
                noop_decode_speedup: Some(0.991),
            },
            PromptPoolEntry {
                sha256: goldens[1].sha256().to_string(),
                bytes: Some(goldens[1].byte_len()),
                noop_decode_speedup: Some(0.992),
            },
            PromptPoolEntry {
                sha256: goldens[2].sha256().to_string(),
                bytes: Some(goldens[2].byte_len()),
                noop_decode_speedup: Some(0.993),
            },
        ];
        let out = run_measure_job(
            &goldens,
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            |_p| ok_candidate(),
        )
        .unwrap();
        // All three shas distinct (R17 predicate 9: unique == pool_size).
        let mut shas: Vec<&str> = out
            .results
            .per_prompt
            .iter()
            .map(|p| p.prompt_sha256.as_str())
            .collect();
        shas.sort_unstable();
        shas.dedup();
        assert_eq!(shas.len(), 3, "all per_prompt sha256 are distinct");
        // Each prompt carries its OWN matched no-op ref, in bytes-binding order.
        for (i, want) in [0.991, 0.992, 0.993].into_iter().enumerate() {
            let pp = &out.results.per_prompt[i];
            assert_eq!(pp.prompt_sha256_source, "contract-pool-match");
            assert!(
                (pp.noop_reference_decode_speedup.unwrap() - want).abs() < 1e-12,
                "per_prompt {i} carries its own pool no-op ref {want}, got {:?}",
                pp.noop_reference_decode_speedup
            );
        }
    }

    #[test]
    fn r4_prompt_sha256_is_golden_bytes_not_pool_copy() {
        // finding R4 — a golden whose sha is NOT in the contract pool must seal the golden's OWN sha
        // labeled `golden-bytes`, never pool[0]. The prior code copied pool[0].sha256 unverified.
        let cfg = test_cfg(1, 1); // pool = [{sha256:"poolsha", ...}], which is not the golden's sha
        let out = identity_run(&cfg);
        let pp = &out.results.per_prompt[0];
        assert_eq!(
            pp.prompt_sha256,
            measure_golden().sha256(),
            "identity = the real golden sha"
        );
        assert_ne!(
            pp.prompt_sha256, "poolsha",
            "never the unverified pool copy"
        );
        assert_eq!(pp.prompt_sha256_source, "golden-bytes");
        assert!(
            pp.noop_reference_decode_speedup.is_none(),
            "no pool no-op ref on a miss"
        );
    }

    #[test]
    fn r4_prompt_sha256_contract_pool_match_carries_noop_ref() {
        // finding R4 — when the golden's REAL sha IS a pool entry, the identity is labeled
        // `contract-pool-match` and the no-op ref comes from THAT matched entry.
        let mut cfg = test_cfg(1, 1);
        cfg.prompt_pool = vec![PromptPoolEntry {
            sha256: measure_golden().sha256().to_string(), // the golden's actual sha, so it matches
            bytes: Some(measure_golden().byte_len()),
            noop_decode_speedup: Some(0.9942),
        }];
        let out = identity_run(&cfg);
        let pp = &out.results.per_prompt[0];
        assert_eq!(pp.prompt_sha256, measure_golden().sha256());
        assert_eq!(pp.prompt_sha256_source, "contract-pool-match");
        assert!((pp.noop_reference_decode_speedup.unwrap() - 0.9942).abs() < 1e-12);
    }

    #[test]
    fn rejected_pair_does_not_count_and_loop_continues() {
        // A RETRYABLE reject on the candidate leg's first two attempts (pair 0 is mtp-first, so the
        // candidate leg runs first), then passes forever: pair 0 is rejected (contributes nothing),
        // the loop continues and accepts the rest.
        let calls = Cell::new(0usize);
        let measure_candidate = |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
            let n = calls.get();
            calls.set(n + 1);
            if n < 2 {
                Err(retryable_reject())
            } else {
                ok_candidate()
            }
        };
        // H6/H3 — reject-then-continue is the LOCAL-DEV budget loop; official mode would die-5
        // immediately on the failed pair (covered by h6_official_pair_failure_is_immediate_die5).
        let mut cfg = test_cfg(3, 4);
        cfg.local_pair_budget = true;
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            measure_candidate,
        )
        .unwrap();
        assert!(out.candidate_accepted);
        assert_eq!(
            out.results.accepted_pair_count, 4,
            "still reaches target after a reject"
        );
        assert_eq!(out.results.pairs.len(), 4);
        assert_eq!(
            out.results.rejected_pairs.len(),
            1,
            "the rejected pair is recorded"
        );
        assert_eq!(out.results.rejected_pairs[0].class, "row-accounting");
    }

    #[test]
    fn accepted_below_min_pairs_fails_closed_die5() {
        // A reject that ALWAYS fires on the candidate leg ⇒ every pair rejects after its one gated
        // retry ⇒ the loop exhausts its attempt budget with 0 accepted < min_pairs ⇒ die 5. Finding
        // R19 — every class folds into die-5 the same way; the loop keeps going and fails closed.
        let cfg = test_cfg(3, 4);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            |_p: &TimingParams| -> bench_runner::Result<LegInvocation> { Err(retryable_reject()) },
        )
        .unwrap();
        assert!(!out.candidate_accepted, "accepted < min_pairs must die 5");
        assert_eq!(out.results.accepted_pair_count, 0);
        assert!(
            out.results.pairs.is_empty(),
            "sealed results carry only accepted pairs"
        );
        assert!(!out.results.rejected_pairs.is_empty());
    }

    #[test]
    fn h6_official_pair_failure_is_immediate_die5_no_budget_loop() {
        // H6/H3 (cycle-3) — the RANKED/official path (default, local_pair_budget=false): a pair that
        // fails after its one gated retry is an IMMEDIATE die-5 (W:2005-2032). The candidate rejects
        // pair 0's BOTH attempts then would pass — but official mode does NOT try more pairs, so the
        // run stops at 0 accepted (unlike the local-dev budget loop, which recovers). Every pool
        // prompt is still listed honestly (per_prompt full-length, R16).
        let calls = Cell::new(0usize);
        let measure_candidate = |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
            let n = calls.get();
            calls.set(n + 1);
            if n < 2 {
                Err(retryable_reject())
            } else {
                ok_candidate()
            }
        };
        // TWO goldens: the first prompt's pair fails → immediate die-5. The SECOND prompt must still
        // appear in per_prompt (with zero accepted pairs), never silently dropped.
        let cfg = test_cfg(3, 4); // official (local_pair_budget=false)
        let out = run_measure_job(
            &[measure_golden(), measure_golden_b()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            measure_candidate,
        )
        .unwrap();
        assert!(
            !out.candidate_accepted,
            "official: a failed pair is an immediate die-5"
        );
        assert_eq!(
            out.results.accepted_pair_count, 0,
            "no budget loop trying more pairs"
        );
        // Exactly ONE pair attempt (both legs) was made before the immediate die-5; the candidate
        // was NOT re-attempted on a fresh pair (local-dev would have: >2 calls).
        assert_eq!(
            out.results.rejected_pairs.len(),
            1,
            "one rejected pair, then stop"
        );
        // per_prompt lists BOTH pool prompts honestly (die-5 path still emits every prompt).
        assert_eq!(
            out.results.per_prompt.len(),
            2,
            "every pool prompt listed on the die-5 path"
        );
        assert_eq!(out.results.prompt_count, 2);
        assert!(out
            .results
            .per_prompt
            .iter()
            .all(|p| p.accepted_pair_count == 0));
        // R16 timed_regime HONESTY — no timed measurement completed (0 accepted pairs), so the
        // regime is OMITTED, never asserting a timed regime ran on a none path.
        assert_eq!(
            out.results.timed_regime, None,
            "die-5 with no accepted pair omits timed_regime"
        );
        let v = serde_json::to_value(&out.results).unwrap();
        assert!(
            v.get("timed_regime").is_none(),
            "timed_regime key absent on the none path"
        );
    }

    #[test]
    fn gate_reject_on_first_attempt_then_success_is_accepted_retryable_r19() {
        // finding R19 — a thermal-gate reject (RunnerError::GateRejected, surfaced by the candidate
        // leg's ONE cool gate) is a RETRYABLE leg failure, NOT a hard die: a reject on ONLY the
        // first leg invocation is retried and the pair is ACCEPTED on the retry.
        let calls = Cell::new(0usize);
        let measure_candidate = |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
            let n = calls.get();
            calls.set(n + 1);
            if n == 0 {
                Err(RunnerError::GateRejected {
                    phase: "mtp-timed".to_string(),
                    reason: "GPU did not reach 40C within 900s".to_string(),
                })
            } else {
                ok_candidate()
            }
        };
        let cfg = test_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            measure_candidate,
        )
        .unwrap();
        assert!(
            out.candidate_accepted,
            "a thermal reject is retried and the pair accepts on retry"
        );
        assert_eq!(out.results.accepted_pair_count, 1);
        assert!(
            calls.get() >= 2,
            "the thermal reject was RETRIED (not aborted on the first call)"
        );
    }

    #[test]
    fn gate_reject_persisting_after_retry_folds_into_die5_not_exit2_r19() {
        // finding R19 — a thermal-gate reject that PERSISTS after its one gated retry does NOT
        // produce a distinct exit-2 hard die: it folds into die-5 (candidate rejected). The reject
        // fires on EVERY gate call, so every pair fails after its retry and the loop fails closed to
        // die-5 with 0 accepted; the reject is sealed with the honest "gate-thermal" class. The
        // verdict is `candidate_accepted=false` (→ exit 5), never a mid-pair exit-2.
        let calls = Cell::new(0usize);
        let measure_candidate = |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
            calls.set(calls.get() + 1);
            Err(RunnerError::GateRejected {
                phase: "mtp-timed".to_string(),
                reason: "GPU did not reach 40C within 900s".to_string(),
            })
        };
        let cfg = test_cfg(3, 4);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            measure_candidate,
        )
        .unwrap();
        assert!(
            !out.candidate_accepted,
            "a persistent thermal reject folds into die-5 (candidate rejected)"
        );
        assert_eq!(out.results.accepted_pair_count, 0);
        assert!(
            calls.get() >= 2,
            "the thermal reject was RETRIED once per pair before folding into die-5"
        );
        assert!(
            !out.results.rejected_pairs.is_empty(),
            "the thermal reject is sealed for provenance"
        );
        assert_eq!(
            out.results.rejected_pairs[0].class, "gate-thermal",
            "a thermal timeout records the honest gate-thermal class, not 'infra'"
        );
    }

    #[test]
    fn leg_level_retry_recovers_after_one_retryable_reject() {
        // R15/R19 — one gated retry PER LEG: the candidate leg's first invocation rejects, its
        // second passes ⇒ the leg (and pair) still accepts. `measure` is called twice for the leg
        // (attempt + gated retry), proving the retry unit is the whole leg (one `mtp-timed` verb).
        let calls = Cell::new(0usize);
        let measure_candidate = |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
            let n = calls.get();
            calls.set(n + 1);
            if n == 0 {
                Err(retryable_reject())
            } else {
                ok_candidate()
            }
        };
        let cfg = test_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            measure_candidate,
        )
        .unwrap();
        assert!(out.candidate_accepted, "leg recovers after one gated retry");
        assert_eq!(out.results.accepted_pair_count, 1);
        assert_eq!(
            calls.get(),
            2,
            "the leg was invoked twice: first reject + one gated retry"
        );
    }

    /// David 2026-09-17 — the QUIESCENCE gate seals beside the cool gate, per leg: its state, the
    /// seconds it waited, and the load / GPU utilization it passed on. A reader of the artifact can
    /// see that the window opened on an IDLE box, not only on a cold one.
    #[test]
    fn leg_quiescence_gate_recorded_directly_per_leg() {
        use crate::quiescegate::QuiesceState;
        let cfg = test_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| {
                let mut leg = inv(echo(SERIAL_SPT), GateState::Fired, serial_telemetry());
                leg.quiesce = QuiesceRecord {
                    state: QuiesceState::Waited,
                    waited_seconds: 45,
                    load: Some(1.25),
                    gpu_util: Some(0.03),
                    skip_reason: None,
                };
                Ok(leg)
            },
            |_p| {
                let mut leg = inv(echo(CANDIDATE_SPT), GateState::Fired, candidate_telemetry());
                leg.quiesce = QuiesceRecord {
                    state: QuiesceState::Fired,
                    waited_seconds: 0,
                    load: Some(0.50),
                    gpu_util: Some(0.01),
                    skip_reason: None,
                };
                Ok(leg)
            },
        )
        .unwrap();
        assert_eq!(out.results.accepted_pair_count, 1);
        let pair = &out.results.pairs[0];
        assert_eq!(pair.serial_quiesce_state, "waited");
        assert_eq!(pair.serial_quiesce_waited_seconds, 45);
        assert_eq!(pair.serial_quiesce_load, Some(1.25));
        assert_eq!(pair.serial_quiesce_gpu_util, Some(0.03));
        assert_eq!(pair.candidate_quiesce_state, "fired");
        assert_eq!(pair.candidate_quiesce_waited_seconds, 0);
        assert_eq!(pair.candidate_quiesce_load, Some(0.50));
        assert_eq!(pair.candidate_quiesce_gpu_util, Some(0.01));
        // A skipped gate seals the skip and OMITS the readings it never took.
        let sealed = serde_json::to_value(pair).expect("the pair record serializes");
        assert_eq!(sealed["serial_quiesce_state"], "waited");
        assert_eq!(sealed["candidate_quiesce_gpu_util"], 0.01);
        let skipped = serde_json::to_value(PairRecord {
            serial_quiesce_state: QuiesceState::SkippedNoReader.as_str().to_string(),
            serial_quiesce_load: None,
            serial_quiesce_gpu_util: None,
            ..pair.clone()
        })
        .expect("the pair record serializes");
        assert_eq!(skipped["serial_quiesce_state"], "skipped-no-reader");
        assert!(skipped.get("serial_quiesce_load").is_none());
    }

    #[test]
    fn leg_gate_state_recorded_directly_per_leg() {
        // R15 — ONE cool gate per leg (a single `mtp-timed` invocation), so the recorded per-leg
        // gate state is that ONE gate's state DIRECTLY, with no prefill/decode fold. A `Waited`
        // serial leg and a `Fired` candidate leg are sealed as recorded, per leg.
        let cfg = test_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| Ok(inv(echo(SERIAL_SPT), GateState::Waited, serial_telemetry())),
            |_p| {
                Ok(inv(
                    echo(CANDIDATE_SPT),
                    GateState::Fired,
                    candidate_telemetry(),
                ))
            },
        )
        .unwrap();
        assert_eq!(out.results.accepted_pair_count, 1);
        assert_eq!(
            out.results.pairs[0].serial_gate_state, "waited",
            "serial leg's ONE gate state is recorded directly"
        );
        assert_eq!(
            out.results.pairs[0].candidate_gate_state, "fired",
            "candidate leg's ONE gate state is recorded directly"
        );
    }

    #[test]
    fn leg_reject_retries_the_whole_leg_once() {
        // R15 — the retry unit is the WHOLE LEG (one `mtp-timed` verb), NOT a sub-phase. A reject on
        // the candidate leg's first invocation retries the ENTIRE leg exactly once. For the (mtp-
        // first) pair: the candidate leg is invoked twice (reject + gated retry), the serial leg
        // once ⇒ 2 candidate calls + 1 serial call. There is no separate prefill/decode phase to
        // re-run independently.
        let serial_calls = Cell::new(0usize);
        let candidate_calls = Cell::new(0usize);
        let measure_serial = |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
            serial_calls.set(serial_calls.get() + 1);
            ok_serial()
        };
        let measure_candidate = |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
            let n = candidate_calls.get();
            candidate_calls.set(n + 1);
            if n == 0 {
                Err(retryable_reject())
            } else {
                ok_candidate()
            }
        };
        let cfg = test_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            measure_serial,
            measure_candidate,
        )
        .unwrap();
        assert!(out.candidate_accepted, "the leg retry recovers the pair");
        assert_eq!(out.results.accepted_pair_count, 1);
        assert_eq!(
            candidate_calls.get(),
            2,
            "the candidate leg is invoked twice: first reject + one gated retry (whole-leg retry)"
        );
        assert_eq!(
            serial_calls.get(),
            1,
            "the serial leg runs once — a candidate-leg reject never re-runs the serial leg"
        );
    }

    #[test]
    fn accepted_pairs_alternate_order_despite_interleaved_rejects() {
        // F3: the leg ORDER keys on the ACCEPTED-PAIR index, not the raw attempt index. The
        // candidate leg rejects the first pair attempt (both leg invocations), so accepted pair 0 is
        // produced by the SECOND attempt — yet it must still be `mtp-first`, and the accepted pairs
        // must stay balanced/alternating regardless of the rejected attempt.
        let calls = Cell::new(0usize);
        let measure_candidate = |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
            let n = calls.get();
            calls.set(n + 1);
            if n < 2 {
                Err(retryable_reject())
            } else {
                ok_candidate()
            }
        };
        // H6/H3 — interleaved-reject-then-continue is the LOCAL-DEV budget loop.
        let mut cfg = test_cfg(3, 4);
        cfg.local_pair_budget = true;
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            measure_candidate,
        )
        .unwrap();
        assert!(out.candidate_accepted);
        assert_eq!(out.results.accepted_pair_count, 4);
        assert_eq!(
            out.results.rejected_pairs.len(),
            1,
            "one interleaved reject"
        );
        // The load-bearing F3 assertion: accepted pair 0 is mtp-first even though it was the 2nd
        // attempt (the rejected attempt did NOT consume the mtp-first slot).
        assert_eq!(out.results.pairs[0].order, "mtp-first");
        // Accepted pairs alternate + are balanced (2 mtp-first, 2 serial-first), not all-one-order.
        for (i, p) in out.results.pairs.iter().enumerate() {
            let want = if i % 2 == 0 {
                "mtp-first"
            } else {
                "serial-first"
            };
            assert_eq!(p.order, want, "accepted pair {i} order");
        }
        let mtp = out
            .results
            .pairs
            .iter()
            .filter(|p| p.order == "mtp-first")
            .count();
        let serial = out
            .results
            .pairs
            .iter()
            .filter(|p| p.order == "serial-first")
            .count();
        assert_eq!(
            (mtp, serial),
            (2, 2),
            "accepted order balance under rejects"
        );
    }

    #[test]
    fn token_mismatch_parity_retries_once_then_folds_into_die5_r19() {
        // finding R19 — a parity/token-mismatch is RETRYABLE once too, exactly like every other
        // class. A candidate leg whose `mtp-timed` report ALWAYS carries a token-mismatch ⇒ the leg
        // rejects after its ONE gated retry ⇒ the pair (and, with min_pairs unmet, the run) folds
        // into die-5. We count candidate leg invocations to PROVE the retry ran: 2 (attempt + gated
        // retry). A no-retry parity would have invoked it once.
        let candidate_calls = Cell::new(0usize);
        let measure_candidate = |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
            candidate_calls.set(candidate_calls.get() + 1);
            Err(RunnerError::TokenMismatch {
                label: "benchmark decode token".to_string(),
                step: 3,
                expected: 703,
                actual: 999_999,
                expected_from_engine: false,
            })
        };
        let cfg = test_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            measure_candidate,
        )
        .unwrap();
        assert!(
            !out.candidate_accepted,
            "a persistent parity failure folds into die-5"
        );
        assert_eq!(out.results.accepted_pair_count, 0);
        assert!(out
            .results
            .rejected_pairs
            .iter()
            .any(|r| r.class == "token-mismatch-parity"));
        // The load-bearing R19 assertion: the parity failure was RETRIED — each pair attempt invokes
        // the candidate leg TWICE (attempt + one gated retry), so the total is an even multiple of 2
        // (>= 2). A no-retry parity would invoke it once per attempt (odd/×1).
        let n = candidate_calls.get();
        assert!(
            n >= 2 && n.is_multiple_of(2),
            "parity retried once per leg (even, >=2), got {n}"
        );
    }

    // ------------------------------------------------------------------
    // R15 — one `mtp-timed` verb per leg, per-side heads, engine-echoed effective_spec
    // ------------------------------------------------------------------

    #[test]
    fn one_timed_regime_per_leg_both_legs_serial_tf_regime() {
        // R15 — exactly ONE timed invocation per leg (one process per leg): for the single (mtp-first)
        // pair, the serial leg is invoked ONCE and the candidate leg ONCE. #105 — under Option A BOTH
        // legs time a SERIAL teacher-forced decode window (effective depth 0); the candidate's DECLARED
        // mtp depth D (=8) is sealed only as the top-level `mtp_depth` provenance, never as the
        // effective regime.
        let serial_calls = Cell::new(0usize);
        let candidate_calls = Cell::new(0usize);
        let mut cfg = test_cfg(1, 1);
        cfg.mtp_depth = Some(8); // DECLARED D (candidate_spec provenance)
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p: &TimingParams| {
                serial_calls.set(serial_calls.get() + 1);
                ok_serial()
            },
            |_p: &TimingParams| {
                candidate_calls.set(candidate_calls.get() + 1);
                ok_candidate()
            },
        )
        .unwrap();
        assert!(out.candidate_accepted);
        assert_eq!(
            serial_calls.get(),
            1,
            "ONE timed invocation for the serial leg"
        );
        assert_eq!(
            candidate_calls.get(),
            1,
            "ONE timed invocation for the candidate leg"
        );
        // The ONLY scored number is the report's parent_measured_seconds_per_token.
        let pair = &out.results.pairs[0];
        assert_eq!(pair.serial_seconds_per_token, SERIAL_SPT);
        assert_eq!(pair.mtp_seconds_per_token, CANDIDATE_SPT);
        // #105 — BOTH legs seal the SINGLE SERIAL effective regime; the mtp regime never ran.
        assert_eq!(
            pair.serial_effective_spec,
            SpecConfig::serial(),
            "serial control is serial"
        );
        assert_eq!(
            pair.candidate_effective_spec,
            SpecConfig::serial(),
            "candidate TF regime is serial"
        );
        // #105 H-A — the series tag + TRUTHFUL verb (never the false "mtp-timed"); the declared depth
        // is still sealed as provenance.
        assert_eq!(out.results.timed_mode, "teacher_forced_v1");
        assert_eq!(out.results.timed_regime, Some("tf-serial-timed"));
        assert_eq!(out.results.serial_control_depth, 0);
        assert_eq!(
            out.results.mtp_depth,
            Some(8),
            "declared depth D sealed as provenance"
        );
    }

    #[test]
    fn tf_seal_refuses_a_candidate_leg_that_carries_any_effective_spec_echo() {
        // #105 (Engine-can't-speculate-on-TF) + the coordinator's leg-B ruling — a TF leg is spawned
        // gate-off and requests no spec, so a conformant worker cannot emit `effective_spec` at all.
        // ANY echo on a TF leg is therefore the tamper signal: the leg rejects (after its gated
        // retry) and the run folds into die-5. The seal never records a regime this process could
        // not have reported.
        //
        // BOTH shapes are checked, and the second is the point of the inversion: the original attack
        // (an mtp echo, depth 5) still rejects, and a forger who echoes a plausible `serial` — which
        // the OLD mode-comparison guard would have waved through — now rejects too.
        for (label, spec) in [
            ("mtp echo (the original attack)", SpecConfig::mtp(5)),
            ("serial echo (passed the old guard)", SpecConfig::serial()),
        ] {
            let out = run_measure_job(
                &[measure_golden()],
                &DirDigest::empty(),
                "deadbeef",
                &test_cfg(1, 1),
                |_p: &TimingParams| ok_serial(),
                move |_p: &TimingParams| candidate_tf_echo(spec.clone()),
            )
            .unwrap();
            assert!(
                !out.candidate_accepted,
                "{label}: a TF echo rejects → die-5"
            );
            assert!(out.results.pairs.is_empty(), "{label}: no pair accepted");
            assert!(
                out.results
                    .rejected_pairs
                    .iter()
                    .any(|r| r.class == "non-serial-tf-regime"),
                "{label}: the reject class names the TF regime refusal"
            );
        }
    }

    #[test]
    fn legb_tf_pair_with_no_echo_accepts_and_seals_the_gate_off_spawn_source() {
        // COORDINATOR RULING (#109, leg B) — the positive half. A TF pair whose legs carry NO
        // effective-spec echo (the shape a gate-off v1 worker actually produces, which before this
        // ruling was rejected as "carries no engine-echoed effective_spec" and blocked leg B) now
        // ACCEPTS, and seals its serial regime with the provenance that it was DERIVED FROM THE
        // SPAWN SURFACE rather than measured off the wire.
        let out = identity_run(&test_cfg(1, 1));
        assert!(
            out.candidate_accepted,
            "a gate-off TF pair with no echo is the CONFORMANT shape"
        );
        let pair = &out.results.pairs[0];
        assert_eq!(pair.serial_effective_spec, SpecConfig::serial());
        assert_eq!(pair.candidate_effective_spec, SpecConfig::serial());
        assert_eq!(
            pair.serial_effective_spec_source,
            EFFECTIVE_SPEC_SOURCE_GATE_OFF_V1_SPAWN
        );
        assert_eq!(
            pair.candidate_effective_spec_source,
            EFFECTIVE_SPEC_SOURCE_GATE_OFF_V1_SPAWN
        );
        // Sealed in the artifact under that name, so a reader can tell a DERIVED serial from a
        // MEASURED one without reading benchd's source.
        let v = serde_json::to_value(&out.results).unwrap();
        assert_eq!(
            v["pairs"][0]["candidate_effective_spec_source"],
            json!("gate-off-v1-spawn")
        );
        assert_eq!(
            v["pairs"][0]["serial_effective_spec_source"],
            json!("gate-off-v1-spawn")
        );
    }

    #[test]
    fn legb_tf_legs_request_no_spec_free_run_legs_still_do() {
        // COORDINATOR RULING (#109, leg B) — the WIRE surface, keyed off the same regime fact as the
        // SPAWN surface, because they are one decision: gate-off spawn ⇒ no spec requested; gate-on
        // spawn ⇒ spec requested. Asserted together here so the two can never drift apart.
        let mtp = SpecConfig::mtp(2);
        let serial = SpecConfig::serial();
        let free_cand = candidate_regime_for_spec(&mtp);
        let tf_cand = candidate_regime_for_spec(&serial);
        for (leg, regime) in [
            ("TF candidate", tf_cand),
            ("TF control", serial_control_regime_for(tf_cand)),
        ] {
            assert_eq!(
                requested_wire_spec(&serial, regime),
                None,
                "{leg}: a gate-off leg requests NO spec"
            );
            assert!(
                !leg_spawn_args("/h", None, regime)
                    .iter()
                    .any(|a| a == SPECULATIVE_PROTOCOL_FLAG),
                "{leg}: ...and that is exactly the leg spawned without the v1.1 gate"
            );
        }
        for (leg, regime, spec) in [
            ("free-run candidate", free_cand, &mtp),
            (
                "free-run control",
                serial_control_regime_for(free_cand),
                &serial,
            ),
        ] {
            assert_eq!(
                requested_wire_spec(spec, regime).as_ref(),
                Some(spec),
                "{leg}: a gate-on leg still requests its spec verbatim"
            );
            assert!(
                leg_spawn_args("/h", None, regime)
                    .iter()
                    .any(|a| a == SPECULATIVE_PROTOCOL_FLAG),
                "{leg}: ...and that is the leg spawned WITH the v1.1 gate"
            );
        }
    }

    #[test]
    fn r15_the_sealed_spt_is_benchds_own_parent_clock() {
        // R15 / H1 — the ONLY scored number is benchd's OWN parent-side wall clock
        // (`LegInvocation::benchd_seconds_per_token`). #109 window-2 finding 3 — this used to also
        // assert that the report's `worker_self_seconds_per_token` (mocked at 3× the parent value)
        // never leaked into the seal; with the report file retired there is no worker-authored timing
        // ANYWHERE in the path, so the property is now true by construction rather than by check.
        let out = identity_run(&test_cfg(1, 1));
        let pair = &out.results.pairs[0];
        assert_eq!(
            pair.serial_seconds_per_token, SERIAL_SPT,
            "serial spt = benchd's clock"
        );
        assert_eq!(
            pair.mtp_seconds_per_token, CANDIDATE_SPT,
            "mtp spt = benchd's clock"
        );
        // The aggregate pooled means are the same benchd-clock values.
        assert_eq!(
            out.results.aggregate.baseline_serial_seconds_per_token_mean,
            SERIAL_SPT
        );
        assert_eq!(
            out.results.aggregate.candidate_mtp_seconds_per_token_mean,
            CANDIDATE_SPT
        );
    }

    // #109 window-2 finding 3 (RETIRED TEST) — `h1_forged_report_timing_does_not_move_the_score_only_
    // _the_audit_echo` drove a candidate whose `--mtp-report` file FORGED a 1e-9 s/tok parent claim
    // and asserted the forgery reached only the `*_report_echo_seconds_per_token` audit fields. Both
    // the attack and the fields it inspected are gone with the report file: the generic
    // `runtime-worker` verb writes no report, so a worker has no channel through which to claim a
    // parent-clock number at all. H1's property is unchanged and stronger — benchd's own clock was
    // already the only scored source, and is now the only such number in existence. The surviving
    // half of H1's evidence is `h1_implausible_benchd_clock_rejects_the_leg` below (an implausible
    // benchd clock fails closed) plus `r15_the_sealed_spt_is_benchds_own_parent_clock` above.

    #[test]
    fn h1_implausible_benchd_clock_rejects_the_leg() {
        // H1 (cycle-3) — the SCORED source is benchd's clock, so an implausible benchd measurement
        // (non-finite / non-positive) rejects the leg, even with every wire echo present and well
        // formed.
        let cfg = test_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
                Ok(LegInvocation {
                    benchd_seconds_per_token: 0.0, // benchd's own clock is implausible → reject
                    // #109 W3 finding 5 — the conformant gate-off TF hello carries NO head.
                    wire_head_provenance: None,
                    gate_state: GateState::Fired,
                    quiesce: QuiesceRecord::default(),
                    telemetry: candidate_telemetry(),
                    // The conformant gate-off TF shape (no echo) — so the ONLY thing wrong with this
                    // leg is the clock.
                    wire_effective_spec: None,
                    // Legacy Model-2 shape: the teacher-forced regime, so no free-run AUDIT and
                    // no free-run window.
                    regime: LegRegime::TeacherForcedV1,
                    free_run_audit: None,
                    phase_window: None,
                    cohort_audit: None,
                    cohort_phase_windows: None,
                    per_stream_timing: None,
                    cohort_committed_tokens_by_stream: None,
                })
            },
        )
        .unwrap();
        assert!(
            !out.candidate_accepted,
            "an implausible benchd clock fails closed even with every echo present"
        );
        assert_eq!(out.results.accepted_pair_count, 0);
    }

    #[test]
    fn r15_per_side_heads_seal_candidate_head_provenance() {
        // R15 — per-side heads: the serial leg loads the PINNED head, the candidate the DECLARED
        // BYO head (distinct shas). The sealed `head_provenance_sha256` comes from the CANDIDATE
        // (MTP) leg. Medium (#105) — the head is now sealed SEPARATELY (not via a second effective-spec
        // echo), so the single effective_spec channel carries the regime while head provenance flows
        // through `head_provenance_sha256`.
        //
        // #109 W3 finding 5 — this is a FREE-RUN test now. Head identity only exists on the regime
        // whose legs are spawned gate-on; the teacher-forced counterpart is
        // `w3f5_teacher_forced_pair_seals_no_head_provenance`.
        let cfg = free_run_cfg(1, 1);
        let n = BENCHMARK_DECODE_STEPS;
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_free_run_serial(),
            move |_p: &TimingParams| inv_free_run(CANDIDATE_SPT, vec![4; n / 4], n),
        )
        .unwrap();
        assert!(out.candidate_accepted);
        let pair = &out.results.pairs[0];
        assert_eq!(
            pair.head_provenance_sha256, CANDIDATE_HEAD_SHA,
            "head provenance = candidate (declared) head"
        );
        assert_ne!(
            pair.head_provenance_sha256, SERIAL_HEAD_SHA,
            "NOT the serial pinned head"
        );
        // The free-run series' two echoes: speculating candidate, depth-0 serial control.
        assert_eq!(pair.serial_effective_spec, SpecConfig::serial());
        assert_eq!(
            pair.candidate_effective_spec,
            SpecConfig::mtp(FREE_RUN_DEPTH)
        );
        // Sealed per prompt (fills the R7 placeholder).
        assert_eq!(
            out.results.per_prompt[0].head_provenance_sha256.as_deref(),
            Some(CANDIDATE_HEAD_SHA),
            "head_provenance_sha256 sealed per prompt from the candidate leg's hello"
        );
        // The MTP arm's depth cap is sealed; the dflash-specific field is OMITTED (mirror of the
        // dflash direction in `dflash_run_seals_dflash_depth_and_drops_the_mtp_field`).
        assert_eq!(
            out.results.aggregate.mtp_max_draft_depth,
            Some(MTP_MAX_DRAFT_DEPTH)
        );
        assert_eq!(out.results.aggregate.dflash_effective_depth, None);
    }

    #[test]
    fn dflash_run_seals_dflash_depth_and_drops_the_mtp_field() {
        // The dflash counterpart of `r15_per_side_heads_seal_candidate_head_provenance`: a
        // single-stream free-run whose candidate leg echoes a DFLASH effective_spec carrying
        // `dflash.depth`. The aggregate must (a) seal `dflash_effective_depth` = that ECHOED depth,
        // structurally decoded out of the opaque `dflash` block (never the MTP constant), and
        // (b) OMIT `mtp_max_draft_depth` entirely — that MTP-specific cap is meaningless on this arm.
        let cfg = MeasureJobConfig {
            candidate_spec: SpecConfig::dflash(DFLASH_DEPTH),
            ..free_run_cfg(1, 1)
        };
        let n = BENCHMARK_DECODE_STEPS;
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_free_run_serial(),
            move |_p: &TimingParams| {
                inv_free_run_dflash(CANDIDATE_SPT, DFLASH_DEPTH, vec![4; n / 4], n)
            },
        )
        .unwrap();
        assert!(out.candidate_accepted, "the dflash candidate clears die-5");

        // The candidate leg's echoed effective spec is the dflash spec carrying the depth lever.
        assert_eq!(
            out.results.pairs[0].candidate_effective_spec,
            SpecConfig::dflash(DFLASH_DEPTH),
            "candidate leg echoed the dflash effective_spec"
        );

        // Typed seal: dflash depth sealed from the ECHO, the MTP cap absent.
        let agg = &out.results.aggregate;
        assert_eq!(
            agg.dflash_effective_depth,
            Some(DFLASH_DEPTH as usize),
            "dflash_effective_depth = the engine-echoed dflash.depth"
        );
        assert_eq!(
            agg.mtp_max_draft_depth, None,
            "the MTP-specific cap is NOT stamped on a dflash run"
        );
        assert_ne!(
            agg.dflash_effective_depth,
            Some(MTP_MAX_DRAFT_DEPTH),
            "NAME-TRAP: the dflash depth is the ECHO (5), never the MTP constant (8)"
        );

        // Serialized shape the website reads: the dflash field present, the mtp field OMITTED (not
        // a fabricated null), the two arms' depth fields mutually exclusive.
        let v = serde_json::to_value(&out.results).unwrap();
        let agg_json = &v["aggregate"];
        assert_eq!(agg_json["dflash_effective_depth"], json!(DFLASH_DEPTH));
        assert!(
            agg_json.get("mtp_max_draft_depth").is_none(),
            "mtp_max_draft_depth is OMITTED from a dflash run's serialized aggregate"
        );
    }

    #[test]
    fn w3f5_teacher_forced_pair_seals_no_head_provenance() {
        // #109 W3 finding 5, DIRECTION 1 (the unblock) — a conformant TEACHER-FORCED pair carries no
        // `head_provenance` on either leg, because both were spawned gate-off and the engine gates
        // the field behind `--speculative-protocol v1.1`. Window 3's leg B died here on all 8 prompts
        // with `accepted_pairs=0`; the requirement was unsatisfiable by construction.
        //
        // The regression is named by window 3's VERBATIM diagnostic:
        //   "candidate (MTP) leg hello carries no head_provenance.sha256 (fail-closed; the candidate
        //    leg must report the head the engine loaded)"
        // — which a TF run must never produce again.
        let out = identity_run(&test_cfg(1, 1));
        assert!(
            out.candidate_accepted,
            "a gate-off TF pair with no head_provenance is CONFORMANT"
        );
        assert_eq!(out.results.accepted_pair_count, 1);
        for rej in &out.results.rejected_pairs {
            assert!(
                !rej.reason.contains("carries no head_provenance.sha256"),
                "window-3 finding 5 verbatim diagnostic came back on a TF leg: {}",
                rej.reason
            );
        }
        // A TF series seals NO head identity — never a blank one, never a fabricated one.
        assert_eq!(out.results.pairs[0].head_provenance_sha256, "");
        assert!(out.results.per_prompt[0].head_provenance_sha256.is_none());
        assert!(out
            .results
            .aggregate
            .head_provenance_sha256_by_prompt
            .is_empty());
    }

    #[test]
    fn w3f5_teacher_forced_leg_hello_carrying_head_provenance_is_refused() {
        // #109 W3 finding 5, DIRECTION 2 (the anomaly) — TF legs neither REQUIRE nor ACCEPT the
        // field. A gate-off worker is gated out of emitting `head_provenance`, so a present object
        // means the leg was not the process benchd spawned. Refused fail-closed under the same class
        // as a present effective-spec echo — the identical tamper case on the identical surface.
        for leg_under_test in ["candidate", "serial"] {
            let cfg = test_cfg(1, 1);
            let out = run_measure_job(
                &[measure_golden()],
                &DirDigest::empty(),
                "deadbeef",
                &cfg,
                move |_p| {
                    let mut i = ok_serial()?;
                    if leg_under_test == "serial" {
                        i.wire_head_provenance = head_prov(SERIAL_HEAD_SHA);
                    }
                    Ok(i)
                },
                move |_p: &TimingParams| {
                    let mut i = ok_candidate()?;
                    if leg_under_test == "candidate" {
                        i.wire_head_provenance = head_prov(CANDIDATE_HEAD_SHA);
                    }
                    Ok(i)
                },
            )
            .unwrap();
            assert!(
                !out.candidate_accepted,
                "a gate-off TF {leg_under_test} leg that reported a head must fail closed"
            );
            assert_eq!(out.results.accepted_pair_count, 0);
            let rej = &out.results.rejected_pairs[0];
            assert_eq!(rej.class, "non-serial-tf-regime");
            assert!(
                rej.reason.contains("carries a hello head_provenance"),
                "reject names the anomaly: {}",
                rej.reason
            );
            assert!(rej.reason.contains(SPECULATIVE_PROTOCOL_FLAG));
        }
    }

    #[test]
    fn w3f5_the_predicate_is_the_gate_off_surface_both_directions() {
        // The unit-level statement of both directions, independent of the run loop.
        assert!(tf_hello_carries_no_head_provenance(None).is_ok());
        let hp = head_prov(CANDIDATE_HEAD_SHA).unwrap();
        let err = tf_hello_carries_no_head_provenance(Some(&hp)).unwrap_err();
        assert!(err.contains(CANDIDATE_HEAD_SHA));
        assert!(err.contains("gate-off worker cannot have produced one"));
        // The two v1.1-gated hello fields ride the SAME spawn gate, so the two predicates agree
        // about what a gate-off leg may carry: nothing.
        assert!(tf_regime_is_serial(None).is_ok());
        assert!(tf_regime_is_serial(Some(&SpecConfig::serial())).is_err());
    }

    #[test]
    fn r15_effective_spec_sealed_from_engine_echo_not_declared_depth() {
        // R15/R1 — benchd seals ONLY the ENGINE-ECHOED effective_spec, NEVER the declared value.
        // #105 — the config DECLARES `--mtp-depth 2` (mtp), but under teacher forcing the engine echoes
        // the SERIAL effective regime. The sealed per-prompt + per-pair effective spec must carry the
        // ECHO (serial), NOT the declared mtp depth (2).
        let cfg = test_cfg(1, 1); // cfg.mtp_depth == 2 (declared)
        let out = identity_run(&cfg);
        assert!(out.candidate_accepted);
        // The declared depth is still sealed as the top-level knob (what we asked for / provenance)...
        assert_eq!(
            out.results.mtp_depth,
            Some(2),
            "top-level mtp_depth is the declared knob"
        );
        // ...but the effective_spec echo (what the engine ACTUALLY ran) is SERIAL, sealed as fact.
        assert_eq!(
            out.results.per_prompt[0].effective_spec.as_ref().unwrap(),
            &SpecConfig::serial(),
            "per_prompt effective_spec seals the ENGINE ECHO (serial), not the declared mtp depth (2)"
        );
        assert_eq!(
            out.results.pairs[0].candidate_effective_spec,
            SpecConfig::serial()
        );
    }

    #[test]
    fn r15_free_run_missing_effective_spec_echo_fails_closed() {
        // Medium (#105) — a leg WITHOUT the engine-echoed WIRE effective_spec FAILS CLOSED (benchd
        // never fabricates the echo): the candidate leg rejects every attempt ⇒ die-5, 0 accepted
        // pairs.
        //
        // Coordinator ruling (#109, leg B) — this is now a FREE-RUN test. The rule it guards was
        // never about legs in general; it is about legs that ASKED for a spec. A free-run leg is
        // spawned gate-on and requests one, so a missing echo means the engine ignored the request —
        // fail closed. A TF leg asks for nothing and correctly echoes nothing; requiring an echo
        // there is what blocked leg B, and is covered from the other side by
        // `legb_tf_pair_with_no_echo_accepts_and_seals_the_gate_off_spawn_source`.
        let cfg = free_run_cfg(1, 1);
        let n = BENCHMARK_DECODE_STEPS;
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_free_run_serial(),
            move |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
                let mut inv = inv_free_run(CANDIDATE_SPT, vec![4; n / 4], n)?;
                inv.wire_effective_spec = None; // engine ignored the requested spec — fail closed
                Ok(inv)
            },
        )
        .unwrap();
        assert!(
            !out.candidate_accepted,
            "a missing effective_spec echo fails the run closed (die-5)"
        );
        assert_eq!(out.results.accepted_pair_count, 0);
        assert!(
            !out.results.rejected_pairs.is_empty(),
            "the fail-closed reject is sealed for provenance"
        );
        // No effective_spec fabricated on the (0-accepted) per_prompt record.
        assert!(out.results.per_prompt[0].effective_spec.is_none());
        assert!(out.results.per_prompt[0].head_provenance_sha256.is_none());
    }

    #[test]
    fn r15_candidate_missing_wire_head_provenance_fails_closed() {
        // R15 — the FREE-RUN candidate (MTP) leg MUST report the head it loaded; a leg without it
        // fails closed (die-5), never fabricated. #109 window-2 finding 3 — the channel is the WIRE
        // `hello` (`head_provenance.sha256`), not the retired `--mtp-report` file, so this drives an
        // engine whose hello omitted the object. #109 W3 finding 5 — the requirement is SCOPED to
        // this regime, and here it is UNCHANGED: the gate is on, so the field is available and
        // required.
        let cfg = free_run_cfg(1, 1);
        let n = BENCHMARK_DECODE_STEPS;
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_free_run_serial(),
            move |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
                let mut inv = inv_free_run(CANDIDATE_SPT, vec![4; n / 4], n)?;
                inv.wire_head_provenance = None;
                Ok(inv)
            },
        )
        .unwrap();
        assert!(
            !out.candidate_accepted,
            "a candidate hello without head_provenance fails closed"
        );
        assert_eq!(out.results.accepted_pair_count, 0);
        assert!(
            out.results.rejected_pairs[0]
                .reason
                .contains("carries no head_provenance.sha256"),
            "the fail-closed diagnostic is unchanged on the regime that owns the field: {}",
            out.results.rejected_pairs[0].reason
        );
    }

    #[test]
    fn w2f3_blank_wire_head_sha_is_not_a_head_provenance() {
        // #109 window-2 finding 3 — an engine that echoes the head_provenance OBJECT but with an
        // empty/whitespace sha has told benchd nothing; the candidate leg must fail closed exactly as
        // if the object were absent, rather than sealing a blank head identity. #109 W3 finding 5 —
        // driven on the FREE-RUN regime, the one that has a head channel at all.
        let cfg = free_run_cfg(1, 1);
        let n = BENCHMARK_DECODE_STEPS;
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_free_run_serial(),
            move |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
                let mut inv = inv_free_run(CANDIDATE_SPT, vec![4; n / 4], n)?;
                inv.wire_head_provenance = head_prov("   ");
                Ok(inv)
            },
        )
        .unwrap();
        assert!(
            !out.candidate_accepted,
            "a blank head sha is not a head provenance"
        );
        assert_eq!(out.results.accepted_pair_count, 0);
    }

    #[test]
    fn r15_seals_timed_regime_prefill_component_none_and_control_depth() {
        // R15 point 6 — the top-level seal carries the TRUTHFUL timed_regime "tf-serial-timed" (#105 H-A,
        // never the false "mtp-timed"), the series tag "teacher_forced_v1", prefill_component "none"
        // (seed prefill INSIDE the decode window, no separate scored prefill phase), and
        // serial_control_depth 0.
        let out = identity_run(&test_cfg(1, 1));
        let v = serde_json::to_value(&out.results).unwrap();
        assert_eq!(v["timed_mode"], json!("teacher_forced_v1"));
        assert_eq!(v["timed_regime"], json!("tf-serial-timed"));
        // #105 cycle-5 finding 3 — the RETIRED key: `timed_verb` named an invocation nothing runs
        // (the real argv is `<engine> runtime-worker …`). It must not come back by copy-paste.
        assert!(
            v.get("timed_verb").is_none(),
            "the retired `timed_verb` key must not be sealed — the value is a REGIME label"
        );
        assert_eq!(v["prefill_component"], json!("none"));
        assert_eq!(v["serial_control_depth"], json!(0));
        assert_eq!(v["mtp_depth"], json!(2));
        // Per-pair records carry no separate prefill seconds-per-token (the field is gone).
        assert!(v["pairs"][0].get("prefill_seconds_per_token").is_none());
    }

    // #109 window-2 finding 3 (RETIRED TEST) — `r15_mtp_timed_report_parse_fails_closed_on_malformed`
    // covered `MtpTimedReport::parse`, the fail-closed reader for the `--mtp-report` file. Both the
    // reader and the flag that produced the file are gone. The equivalent fail-closed posture on the
    // channel that REPLACED it is the protocol layer's own: `bench_protocol::HeadProvenance` is
    // `deny_unknown_fields` with all three members required, and a malformed hello line is a hard
    // `RunnerError::Protocol` in `Session::connect` — see
    // `bench_protocol::tests::response_roundtrip_hello_with_head_provenance`.

    #[test]
    fn contract_parse_reads_pool_but_thermal_is_fixed_wrapper_constants_r21() {
        // finding R21 — the live thermal thresholds are READONLY WRAPPER CONSTANTS (GATE_TEMP=40,
        // MIN_FREQ=1100), FIXED and NOT overridable: a contract that carries a `calibration` block
        // with a different cool_gate_c is IGNORED (the fixture is supposed to carry no threshold
        // fields at all). The pool no-op refs still parse; only the thermal override is reverted.
        let contract_bytes = serde_json::to_vec(&json!({
            "official_scoring_enabled": true,
            "timed_prompt_pool": [
                { "sha256": "aaaa", "bytes": 10, "noop_decode_speedup": 0.9940390645 }
            ],
            // A stray calibration block MUST NOT move the gate — thresholds are wrapper constants.
            "calibration": { "cool_gate_c": 42.0, "clock_floor_mhz": 1500.0, "loaded_util": 0.9 }
        }))
        .unwrap();
        let contract = Contract::parse(&contract_bytes).unwrap();
        assert_eq!(contract.timed_prompt_pool.len(), 1);
        assert_eq!(contract.timed_prompt_pool[0].sha256, "aaaa");
        assert!(
            (contract.timed_prompt_pool[0].noop_decode_speedup.unwrap() - 0.9940390645).abs()
                < 1e-12
        );
        // The contract's 42.0 is IGNORED — the gate enforces the fixed 40, stamped honestly. It
        // cannot do otherwise: `thermal_thresholds` takes no fixture at all.
        let th = thermal_thresholds(0.70, "env-GPU_LOADED_UTIL-default-0.70");
        assert_eq!(
            th.cool_gate_c, 40.0,
            "the fixed wrapper constant, never the contract's 42"
        );
        assert_eq!(th.cool_gate_c_source, "wrapper-constant-40");
        assert_eq!(th.clock_floor_mhz, 1100.0);
        assert_eq!(th.clock_floor_mhz_source, "wrapper-constant-1100");
        // R14 — loaded_util is env-driven (GPU_LOADED_UTIL, default 0.70), never the contract's 0.9.
        assert_eq!(th.loaded_util, 0.70);
        assert_eq!(th.loaded_util_source, "env-GPU_LOADED_UTIL-default-0.70");
    }

    /// #108 (M2) — a calibration file with the given band overrides, otherwise well-formed.
    /// `target_band` is the per-target `(low, high)` override (`None` ⇒ inherit the top-level).
    fn calibration_bytes(
        top_low: serde_json::Value,
        top_high: serde_json::Value,
        target_band: Option<(serde_json::Value, serde_json::Value)>,
    ) -> Vec<u8> {
        let mut target = json!({
            "serial_decode_seconds_per_token_mean": 0.038,
            "decode_tokens": 512
        });
        if let Some((low, high)) = target_band {
            target["serial_band_low"] = low;
            target["serial_band_high"] = high;
        }
        serde_json::to_vec(&json!({
            "serial_decode_seconds_per_token_mean": 0.038,
            "decode_tokens": 512,
            "timed_mode": TIMED_MODE,
            "track_id": "qwen3.8-27b-mtp-v1",
            "serial_band_low": top_low,
            "serial_band_high": top_high,
            "targets": { "qwen3.8-27b-mtp-v1": target }
        }))
        .unwrap()
    }

    #[test]
    fn m2_out_of_range_band_bounds_are_refused_at_parse() {
        // #108 (M2) — the band bounds are checked at PARSE, before any consumer sees the file.
        // `band_high` is not just the die-6 drift verdict: it is the §2.2 RunTimeout ceiling
        // (serial_mean × band_high), so 0.0 / -1 do not "widen the band" — they used to make the
        // computed ceiling non-positive and DISARM the deadline; 1e9 makes it unbounded.
        for (high, what) in [
            (json!(0.0), "zero"),
            (json!(-1.0), "negative"),
            (json!(1.0e9), "absurd"),
            (json!(1.0), "not above the mean"),
        ] {
            let err =
                BaselineCalibration::parse(&calibration_bytes(json!(0.95), high.clone(), None))
                    .expect_err(&format!("a {what} band_high must be refused at parse"));
            assert!(err.contains("serial_band_high"), "{what}: {err}");
            assert!(err.contains("die 6"), "{what}: names the die — {err}");
            assert!(
                err.contains("RunTimeout") && err.contains("wall-clock"),
                "{what}: the diagnostic must say WHY (the deadline), not just 'out of range' — {err}"
            );
        }
        // NaN cannot be written as JSON, so it arrives as a non-finite the parse must still refuse:
        // serde rejects the literal, which is itself fail-closed. Prove the bound directly too.
        assert!(
            serde_json::from_str::<serde_json::Value>(r#"{"serial_band_high": NaN}"#).is_err(),
            "a NaN literal is not JSON — the file fails the parse before the bounds are read"
        );
        let nan = validate_band_bounds("top-level", 0.95, f64::NAN).unwrap_err();
        assert!(nan.contains("serial_band_high"), "{nan}");
        let inf = validate_band_bounds("top-level", 0.95, f64::INFINITY).unwrap_err();
        assert!(inf.contains("serial_band_high"), "{inf}");

        // band_low: finite, in (0, 1).
        for (low, what) in [
            (json!(0.0), "zero"),
            (json!(-0.5), "negative"),
            (json!(1.0), "not below the mean"),
            (json!(2.0), "above the mean"),
        ] {
            let err = BaselineCalibration::parse(&calibration_bytes(low, json!(1.05), None))
                .expect_err(&format!("a {what} band_low must be refused at parse"));
            assert!(err.contains("serial_band_low"), "{what}: {err}");
            assert!(err.contains("die 6"), "{what}: {err}");
        }

        // A PER-TARGET override is bounded by the SAME rule — an override is not a way around it.
        let err = BaselineCalibration::parse(&calibration_bytes(
            json!(0.95),
            json!(1.05),
            Some((json!(0.95), json!(1.0e9))),
        ))
        .expect_err("a per-target band_high override must be refused at parse too");
        assert!(
            err.contains("targets[qwen3.8-27b-mtp-v1]"),
            "the site is named: {err}"
        );
        assert!(err.contains("serial_band_high"), "{err}");

        // The honest file still parses, and the ceiling of the range is accepted exactly.
        assert!(
            BaselineCalibration::parse(&calibration_bytes(json!(0.95), json!(1.05), None)).is_ok()
        );
        let capped = BaselineCalibration::parse(&calibration_bytes(
            json!(0.95),
            json!(SERIAL_BAND_HIGH_CAP),
            None,
        ))
        .expect("the cap itself is IN range (inclusive)");
        assert_eq!(capped.serial_band_high, SERIAL_BAND_HIGH_CAP);
    }

    #[test]
    fn m2_a_hostile_in_range_calibration_cannot_unbound_the_run_timeout() {
        // #108 (M2) — the property the bounds BUY. Whatever a parse-accepted file declares, the §2.2
        // budget it can produce is bounded by `N × serial_mean × SERIAL_BAND_HIGH_CAP × margin` — the
        // file cannot stretch the window past that, and (the other direction) cannot collapse it to
        // "no deadline at all", because a degenerate product is now an Err rather than a None.
        let n = FREE_RUN_DECODE_TOKENS;
        let margin = bench_core::constants::RUN_TIMEOUT_MARGIN;
        // The most hostile band a parse will accept.
        let cal = BaselineCalibration::parse(&calibration_bytes(
            json!(f64::MIN_POSITIVE),
            json!(SERIAL_BAND_HIGH_CAP),
            None,
        ))
        .expect("the extremes of the ALLOWED range still parse")
        .resolve(Some("qwen3.8-27b-mtp-v1"))
        .unwrap();
        let budget =
            bench_core::score::run_timeout_budget(n, cal.serial_mean * cal.band_high, margin)
                .expect("an in-range band always yields an armable deadline");
        let worst_case = n as f64 * cal.serial_mean * SERIAL_BAND_HIGH_CAP * margin;
        assert!(
            budget.as_secs_f64() <= worst_case + 1e-9,
            "budget {}s exceeds the N × mean × {SERIAL_BAND_HIGH_CAP} × margin bound ({worst_case}s)",
            budget.as_secs_f64()
        );
        // Sanity: the bound is a real bound, not vacuous — the pre-M2 1e9 band would have blown it.
        assert!(
            n as f64 * cal.serial_mean * 1.0e9 * margin > worst_case,
            "the refused 1e9 band_high would have produced a strictly larger window"
        );
        // And the disarm direction: a non-positive ceiling is an ERROR, never a silent None.
        assert!(bench_core::score::run_timeout_budget(n, 0.0, margin).is_err());
    }

    #[test]
    fn r14_baseline_calibration_parse_defaults_and_target_inherit() {
        // R14 — top-level band defaults 0.95/1.05; a target inherits the top band but carries its
        // OWN (required, no-inherit) decode_tokens; the source names which entry resolved.
        let bytes = serde_json::to_vec(&json!({
            "serial_decode_seconds_per_token_mean": 0.037994794617,
            "decode_tokens": 512,
            // #105 cycle-5 — the REQUIRED series/track identity every calibration file now carries.
            "timed_mode": TIMED_MODE,
            "track_id": "qwen3.8-27b-mtp-v1",
            "targets": {
                "qwen3.8-27b-mtp-v1": {
                    "serial_decode_seconds_per_token_mean": 0.038,
                    "decode_tokens": 512
                }
            }
        }))
        .unwrap();
        let cal = BaselineCalibration::parse(&bytes).unwrap();
        assert_eq!(cal.serial_band_low, 0.95, "top band low default");
        assert_eq!(cal.serial_band_high, 1.05, "top band high default");

        // No target-id → the top-level calibration.
        let top = cal.resolve(None).unwrap();
        assert_eq!(top.source, "baseline-calibration:top");
        assert!((top.serial_mean - 0.037994794617).abs() < 1e-12);
        assert_eq!(top.decode_tokens, Some(512));
        assert_eq!((top.band_low, top.band_high), (0.95, 1.05));

        // A matched target → its mean/decode_tokens, band inherited from the top.
        let t = cal.resolve(Some("qwen3.8-27b-mtp-v1")).unwrap();
        assert_eq!(t.source, "baseline-calibration:target:qwen3.8-27b-mtp-v1");
        assert!((t.serial_mean - 0.038).abs() < 1e-12);
        assert_eq!(t.decode_tokens, Some(512));
        assert_eq!(
            (t.band_low, t.band_high),
            (0.95, 1.05),
            "target inherits the top band"
        );

        // FAIL-CLOSED: a declared target with no matching entry is die-6 (NOT a top-level fallback),
        // mirroring the wrapper's require_target_calibration.
        let miswired = cal.resolve(Some("no-such-target")).unwrap_err();
        assert!(
            miswired.contains("no entry for target no-such-target"),
            "{miswired}"
        );
        assert!(miswired.contains("die 6"), "{miswired}");
    }

    #[test]
    fn h6_wrapper_authored_calibration_parses_and_resolves() {
        // H6/H2 (cycle-3) — the loader MUST accept the shape the LIVE wrapper's
        // `write_calibration_bootstrap` AUTHORS (W:1468-1528, verified against the read-only mirror
        // 4e38853e): top level carries ONLY `track_id` + `targets{}` (NO top-level
        // `serial_decode_seconds_per_token_mean`), and each target entry carries the full authored
        // field set (sessions/session_count/pairs_total/cv/authored_*/provisional). Requiring the
        // top-level mean, or `deny_unknown_fields`, would die-6 on every wrapper-authored file.
        // This fixture is the wrapper's EXACT authored shape, NOT our own output.
        //
        // #105 cycle-5 — the wrapper's shape is authored under the NATIVE regime and carries NO
        // `timed_mode`, so as-is it is now REFUSED (asserted below as the untagged negative control):
        // an untagged band must never be assumed comparable. The structural tolerances this test
        // exists to lock — no required top-level mean, no `deny_unknown_fields` — are re-asserted
        // against the SAME fixture once it declares this run's series.
        let wrapper_authored = json!({
            "track_id": "qwen3.6-27b-mtp-v1",
            "targets": {
                "lowsim-prose-qwen-v1": {
                    "serial_decode_seconds_per_token_mean": 0.037994794617,
                    "serial_band_low": 0.95,
                    "serial_band_high": 1.05,
                    "decode_tokens": 512,
                    "mtp_depth": 2,
                    "serial_control_depth": 0,
                    "sessions": [
                        { "tag": "qwen-mtp-mjob-20260817T101500Z",
                          "serial_mean": 0.0380, "pairs": 3, "at": "2026-08-17T10:15:00Z" }
                    ],
                    "session_count": 1,
                    "pairs_total": 3,
                    "serial_cross_session_cv_pct": null,
                    "authored_by": "measure-qwen-mtp-job.sh --calibration-bootstrap",
                    "authored_at": "2026-08-17T10:15:00Z",
                    "updated_at": "2026-08-17T10:15:00Z",
                    "authored_tag": "qwen-mtp-mjob-20260817T101500Z",
                    "provisional": true
                }
            }
        });
        // #105 cycle-5 NEGATIVE CONTROL — the wrapper's own shape carries no `timed_mode`, so it is
        // refused fail-closed with the series diagnostic (never a bare serde `missing field`).
        let untagged = BaselineCalibration::parse(&serde_json::to_vec(&wrapper_authored).unwrap())
            .expect_err("an untagged (native-regime) wrapper file must NOT band a tagged run");
        assert!(untagged.contains("`timed_mode`"), "{untagged}");
        assert!(untagged.contains("die 6"), "{untagged}");

        // The SAME fixture, now declaring this run's series: the structural tolerances hold.
        let mut wrapper_authored = wrapper_authored;
        wrapper_authored["timed_mode"] = json!(TIMED_MODE);
        let bytes = serde_json::to_vec(&wrapper_authored).unwrap();
        // Parses despite NO top-level mean and the extra per-target authoring fields.
        let cal = BaselineCalibration::parse(&bytes).expect("wrapper-authored file must parse");
        assert!(
            cal.serial_decode_seconds_per_token_mean.is_none(),
            "no top-level mean authored"
        );
        // Resolving BY --target-id yields the per-target denominator + window + inherited band.
        let r = cal
            .resolve(Some("lowsim-prose-qwen-v1"))
            .expect("wrapper-authored target must resolve");
        assert!((r.serial_mean - 0.037994794617).abs() < 1e-12);
        assert_eq!(r.decode_tokens, Some(512));
        assert_eq!((r.band_low, r.band_high), (0.95, 1.05));
        assert_eq!(r.source, "baseline-calibration:target:lowsim-prose-qwen-v1");
        // A TOP-LEVEL resolve (no --target-id) on a target-only file fails CLOSED (die-6), never
        // bands against a fabricated 0.
        let e = cal.resolve(None).unwrap_err();
        assert!(
            e.contains("no top-level serial_decode_seconds_per_token_mean"),
            "{e}"
        );
        assert!(e.contains("die 6"), "{e}");
    }

    #[test]
    fn h105_c5_tf_downgrade_is_documented_in_the_error_and_the_seal() {
        // #105 cycle-5 finding 5 — the downgrade is acceptable DESIGN; undocumented was the finding.
        // The v1.1 pointer must appear in BOTH places a reader can land: the rejection error and the
        // sealed artifact.
        let e = tf_regime_is_serial(Some(&bench_protocol::SpecConfig::mtp(2)))
            .expect_err("an echo on a gate-off TF leg must reject");
        assert!(
            e.contains(TF_DOWNGRADE_NOTE),
            "the error must carry the v1.1 pointer: {e}"
        );
        assert!(e.contains("v1.1 free-run"), "{e}");
        // Coordinator ruling (#109, leg B) — and the conformant gate-off shape (NO echo) is accepted
        // by the same guard, so the pointer above is emitted on the anomaly only.
        tf_regime_is_serial(None).expect("a gate-off TF leg carries no echo, and that is correct");

        // Sealed: an mtp-declaring candidate carries the note (a downgrade DID happen)...
        let mut cfg = test_cfg(1, 1);
        cfg.candidate_spec = bench_protocol::SpecConfig::mtp(2);
        let v = serde_json::to_value(&identity_run(&cfg).results).unwrap();
        assert_eq!(
            v["candidate_spec"]["mode"],
            json!("mtp"),
            "precondition: declared mtp"
        );
        assert_eq!(
            v["pairs"][0]["candidate_effective_spec"]["mode"],
            json!("serial")
        );
        assert_eq!(v["tf_downgrade_note"], json!(TF_DOWNGRADE_NOTE));

        // ...and a candidate that declared serial was never downgraded, so it seals NO note.
        let mut serial_cfg = test_cfg(1, 1);
        serial_cfg.candidate_spec = bench_protocol::SpecConfig::serial();
        let sv = serde_json::to_value(&identity_run(&serial_cfg).results).unwrap();
        assert!(
            sv.get("tf_downgrade_note").is_none(),
            "the note states a fact about THIS run, not boilerplate"
        );
    }

    #[test]
    fn h105_c5_the_wire_spec_is_the_only_depth_channel() {
        // #105 cycle-5 finding 4 — there used to be TWO channels through which a depth reached the
        // engine (the spawn argv's `--mtp-depth` and the `decode_begin` spec), set from different
        // sources; that finding tied the argv one to the wire one so they could not disagree.
        //
        // #109 window-2 finding 3 closes it the cleaner way: the argv channel is GONE. It was never a
        // channel at all — the spawned `runtime-worker` verb REJECTS `--mtp-depth` (it belongs to
        // `mlxfast-swift mtp-timed`, a different binary), so the depth benchd wrote there could never
        // have been read by anything. The wire spec is now the SINGLE source by construction, and
        // divergence is unconstructible because there is nothing left to diverge from.
        //
        // Both legs, exactly as main.rs builds them: the argv is inspected as BUILT, so this observes
        // the real spawn surface rather than re-deriving the answer.
        for (leg, head) in [("serial", "/heads/pinned"), ("candidate", "/heads/byo")] {
            let args = timed_leg_base_args(head, None);
            assert_eq!(
                flags_in_args(&args),
                vec!["--mtp-head"],
                "{leg} leg argv carries the head and NOTHING else — no depth flag in any form"
            );
            assert_eq!(args[1], head, "{leg} leg loads its own head");
        }

        // The depth channel that DOES exist: the wire spec each leg requests, echoed back as
        // `effective_spec` and guarded never-ignored by the runner. Under teacher forcing it is
        // serial on both legs (`timed_decode_wire_spec`, depth = SERIAL_CONTROL_DEPTH by definition);
        // a free-run candidate carries its declared mtp depth verbatim.
        assert_eq!(timed_decode_wire_spec().mode, SPEC_MODE_SERIAL);
        assert!(
            timed_decode_wire_spec().mtp.is_none(),
            "the TF wire spec requests no depth at all"
        );
        let mtp_wire = bench_protocol::SpecConfig::mtp(4);
        assert_eq!(
            mtp_wire.mtp.as_ref().and_then(|m| m.depth),
            Some(4),
            "free-run depth rides the spec"
        );

        // A DEPTH-CARRYING wire spec still produces a depth-free argv: the two are no longer linked
        // in either direction, so no future depth can leak back onto the command line.
        assert_eq!(
            flags_in_args(&timed_leg_base_args("/heads/byo", None)),
            vec!["--mtp-head"]
        );
        for regime in [LegRegime::TeacherForcedV1, LegRegime::FreeRunV1_1] {
            assert!(
                !leg_spawn_args("/heads/byo", None, regime)
                    .iter()
                    .any(|a| a == "--mtp-depth"),
                "{regime:?}: no depth flag reaches the argv"
            );
        }
    }

    #[test]
    fn h105_c5_series_fence_die6s_the_native_regime_file_that_banded_a_model2_run() {
        // #105 cycle-5 (HIGH) — the review's EXACT attack input, as a test: a BASELINE_CALIBRATION
        // self-declaring the NATIVE regime series with a frontier-era serial mean, pointed at a
        // Model-2 run under band_enforce=true. Before the fence this banded to `Pass`.
        //
        // The two assertions below are the whole finding: the BAND still says Pass (the band check
        // only ever sees an f64 + a window — it cannot tell the regimes apart, and that is not its
        // job), and the FENCE is what refuses the file. If the fence were ever removed, the first
        // assertion is exactly what would ship.
        let attack = serde_json::to_vec(&json!({
            "timed_mode": "native_mtp_v1",
            "track_id": "qwen3.8-27b-mtp-v1",
            "serial_decode_seconds_per_token_mean": 0.038,
            "decode_tokens": 512,
            "targets": { "t1": { "serial_decode_seconds_per_token_mean": 0.038, "decode_tokens": 512 } }
        }))
        .unwrap();
        let cal = BaselineCalibration::parse(&attack).expect("the attack file is well-formed JSON");

        // (a) BANDING ALONE PASSES IT — the pre-fence behaviour, pinned so the finding stays legible.
        let resolved = cal.resolve(Some("t1")).unwrap();
        let banded = evaluate_serial_band(0.038, 512, &resolved, true);
        assert_eq!(
            banded.verdict,
            SerialBandVerdict::Pass,
            "the band check cannot distinguish series — this is precisely why the fence must run first"
        );

        // (b) THE FENCE REFUSES IT — die-6, naming both series, BEFORE any of that runs.
        let e = enforce_calibration_series_fence(&cal, TIMED_MODE, "qwen3.8-27b-mtp-v1")
            .expect_err("a native-regime calibration must never band a Model-2 run");
        assert!(
            e.contains("native_mtp_v1"),
            "the file's series must be named: {e}"
        );
        assert!(
            e.contains(TIMED_MODE),
            "the run's series must be named: {e}"
        );
        assert!(e.contains("NOT comparable"), "{e}");
        assert!(e.contains("die 6"), "{e}");
    }

    #[test]
    fn h105_c5_series_fence_track_mismatch_is_die6_and_matching_series_passes() {
        // #105 cycle-5 — the fence's other leg (the file's OWN track_id vs the RESOLVED track) and
        // the positive control (same series + same track → Ok, and the band then runs normally).
        let bytes = serde_json::to_vec(&json!({
            "timed_mode": TIMED_MODE,
            "track_id": "qwen3.6-27b-mtp-v1",
            "serial_decode_seconds_per_token_mean": 0.038,
            "decode_tokens": 512,
        }))
        .unwrap();
        let cal = BaselineCalibration::parse(&bytes).unwrap();

        // Same series, WRONG track → die-6 (a track is never banded against another track's file).
        let e = enforce_calibration_series_fence(&cal, TIMED_MODE, "qwen3.8-27b-mtp-v1")
            .expect_err("a calibration authored for another track must not band this one");
        assert!(
            e.contains("qwen3.6-27b-mtp-v1") && e.contains("qwen3.8-27b-mtp-v1"),
            "{e}"
        );
        assert!(e.contains("die 6"), "{e}");

        // MATCHING series + track → Pass through the fence, and the band then evaluates normally.
        assert!(enforce_calibration_series_fence(&cal, TIMED_MODE, "qwen3.6-27b-mtp-v1").is_ok());
        let resolved = cal.resolve(None).unwrap();
        assert_eq!(
            resolved.timed_mode, TIMED_MODE,
            "the checked series is sealed for audit"
        );
        assert_eq!(resolved.track_id, "qwen3.6-27b-mtp-v1");
        assert_eq!(
            evaluate_serial_band(0.038, 512, &resolved, true).verdict,
            SerialBandVerdict::Pass
        );

        // The fence is bench_core's predicate, not a local re-implementation: the v1.1 free-run
        // series is likewise refused for this teacher-forced run, and accepted for its own.
        let v11 = serde_json::to_vec(&json!({
            "timed_mode": bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1,
            "track_id": "qwen3.6-27b-mtp-v1",
            "serial_decode_seconds_per_token_mean": 0.038,
            "decode_tokens": 512,
        }))
        .unwrap();
        let v11 = BaselineCalibration::parse(&v11).unwrap();
        assert!(enforce_calibration_series_fence(&v11, TIMED_MODE, "qwen3.6-27b-mtp-v1").is_err());
        assert!(enforce_calibration_series_fence(
            &v11,
            bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1,
            "qwen3.6-27b-mtp-v1"
        )
        .is_ok());
    }

    #[test]
    fn w3_calibration_and_overlay_fences_key_on_one_run_series() {
        // W3 (fence reconciliation) — ONE series story. The run's series is not a constant: it is
        // [`run_timed_mode`], derived from the candidate regime via the same-series serial-control
        // rule, and it is what the CALIBRATION fence checks (die-6, pre-measure), what the bootstrap
        // authors, and what the seal + overlay §5 fence recompute.
        assert_eq!(run_timed_mode(LegRegime::TeacherForcedV1), TIMED_MODE);
        assert_eq!(
            run_timed_mode(LegRegime::FreeRunV1_1),
            bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1
        );

        // THE CROSS TEST: a TEACHER-FORCED calibration file against a FREE-RUN-series run → die-6.
        // This is the §5 bug one level up from the overlay fence — a v1.1 pooled serial mean banded
        // against a v1 teacher-forced band, which the previously hardcoded TF tag would have waved
        // through because the run always claimed to be teacher-forced.
        let tf_file = serde_json::to_vec(&json!({
            "timed_mode": TIMED_MODE,
            "track_id": "qwen3.8-27b-mtp-v1",
            "serial_decode_seconds_per_token_mean": 0.038,
            "decode_tokens": 128,
        }))
        .unwrap();
        let tf_file = BaselineCalibration::parse(&tf_file).unwrap();
        let e = enforce_calibration_series_fence(
            &tf_file,
            run_timed_mode(LegRegime::FreeRunV1_1),
            "qwen3.8-27b-mtp-v1",
        )
        .expect_err("a teacher-forced band must never gate a free-run-series run");
        assert!(
            e.contains(TIMED_MODE),
            "the file's series must be named: {e}"
        );
        assert!(
            e.contains("free_run_v1_1"),
            "the run's series must be named: {e}"
        );
        assert!(e.contains("die 6"), "{e}");
        // And the same file DOES band a teacher-forced run — the fence is per-series, not a ban.
        assert!(enforce_calibration_series_fence(
            &tf_file,
            run_timed_mode(LegRegime::TeacherForcedV1),
            "qwen3.8-27b-mtp-v1"
        )
        .is_ok());
    }

    #[test]
    fn h105_c5_bootstrap_authors_the_series_and_track_it_measured_under() {
        // #105 cycle-5 — the authoring path round-trips its own fence: a bootstrap-authored file
        // carries the run's series + track, so the NEXT same-series run bands against it and a
        // cross-series run dies. And merging across series/track is refused outright.
        let input = BootstrapAuthorInput {
            target_id: "t1",
            timed_mode: TIMED_MODE,
            track_id: "qwen3.8-27b-mtp-v1",
            pooled_serial_mean: 0.038,
            tokens: 512,
            mtp_depth: Some(2),
            serial_control_depth: 0,
            pairs_total: 6,
        };
        let authored = build_bootstrap_calibration(None, &input).unwrap();
        let parsed = BaselineCalibration::parse(authored.as_bytes())
            .expect("an authored file must satisfy the schema it will be read back under");
        assert!(
            enforce_calibration_series_fence(&parsed, TIMED_MODE, "qwen3.8-27b-mtp-v1").is_ok()
        );
        assert!(
            enforce_calibration_series_fence(&parsed, "native_mtp_v1", "qwen3.8-27b-mtp-v1")
                .is_err(),
            "the authored band is comparable ONLY within its own series"
        );

        // Merging THIS run's band into a file authored under another series is refused (authoring
        // must not manufacture the mislabeling the fence exists to catch).
        let foreign = json!({
            "timed_mode": "native_mtp_v1",
            "track_id": "qwen3.8-27b-mtp-v1",
            "targets": { "other": { "serial_decode_seconds_per_token_mean": 0.9, "decode_tokens": 512 } }
        });
        let e = build_bootstrap_calibration(
            Some(serde_json::to_vec(&foreign).unwrap().as_slice()),
            &input,
        )
        .expect_err("cross-series merge must be refused");
        assert!(e.contains("timed_mode"), "{e}");
    }

    #[test]
    fn r103_resolve_target_with_nonpositive_mean_fails_closed() {
        // R103 — a target ENTRY present but with a zero/invalid serial mean is die-6, never a
        // silent fallback (require_target_calibration bands only a positive mean, W:1372-1374).
        let bytes = serde_json::to_vec(&json!({
            "serial_decode_seconds_per_token_mean": 0.038,
            "decode_tokens": 512,
            "timed_mode": TIMED_MODE,
            "track_id": "qwen3.8-27b-mtp-v1",
            "targets": { "t": { "serial_decode_seconds_per_token_mean": 0.0, "decode_tokens": 512 } }
        }))
        .unwrap();
        let cal = BaselineCalibration::parse(&bytes).unwrap();
        let e = cal.resolve(Some("t")).unwrap_err();
        assert!(e.contains("no finite positive"), "{e}");
        assert!(e.contains("die 6"), "{e}");
    }

    #[test]
    fn h6_band_enforce_empty_string_means_enforced() {
        // H6/H2 (cycle-3) — the wrapper's `${BASELINE_BAND_ENFORCE:-1}` fail-closed default: UNSET
        // and EMPTY both ENFORCE; ONLY an explicit "0" disables; anything else enforces.
        assert!(band_enforce_from_env(None), "unset → ENFORCED");
        assert!(
            band_enforce_from_env(Some("")),
            "empty string → ENFORCED (fail-closed)"
        );
        assert!(band_enforce_from_env(Some("   ")), "whitespace → ENFORCED");
        assert!(band_enforce_from_env(Some("1")), "1 → ENFORCED");
        assert!(
            band_enforce_from_env(Some("true")),
            "any other value → ENFORCED"
        );
        assert!(!band_enforce_from_env(Some("0")), "explicit 0 → disabled");
        assert!(!band_enforce_from_env(Some(" 0 ")), "trimmed 0 → disabled");
    }

    #[test]
    fn r14_baseline_calibration_fails_closed_on_malformed_and_missing_target_decode_tokens() {
        // R14 — malformed JSON fails closed; a target WITHOUT decode_tokens is a parse error
        // (decode_tokens is REQUIRED, never inherited).
        assert!(BaselineCalibration::parse(b"not json").is_err());
        let no_dt = serde_json::to_vec(&json!({
            "serial_decode_seconds_per_token_mean": 0.038,
            "targets": { "t": { "serial_decode_seconds_per_token_mean": 0.038 } }
        }))
        .unwrap();
        assert!(
            BaselineCalibration::parse(&no_dt).is_err(),
            "a target missing decode_tokens must fail the parse (no inherit)"
        );
    }

    #[test]
    fn r14_enforce_serial_band_die6_paths() {
        // R14 — the pooled serial mean / calibration mean must land in the band, and pinned
        // decode_tokens must equal --tokens; otherwise die-6.
        let cal = ResolvedCalibration {
            serial_mean: 0.038,
            band_low: 0.95,
            band_high: 1.05,
            decode_tokens: Some(512),
            timed_mode: TIMED_MODE.to_string(),
            track_id: "qwen3.8-27b-mtp-v1".to_string(),
            source: "baseline-calibration:top".to_string(),
        };
        // In-band ratio (0.038/0.038 = 1.0) with matching tokens → OK.
        assert!(enforce_serial_band(0.038, 512, &cal, true).is_ok());
        // Ratio just inside the band edges is OK; outside is die-6.
        assert!(enforce_serial_band(0.038 * 0.96, 512, &cal, true).is_ok());
        assert!(
            enforce_serial_band(0.038 * 0.90, 512, &cal, true).is_err(),
            "below band → die 6"
        );
        assert!(
            enforce_serial_band(0.038 * 1.10, 512, &cal, true).is_err(),
            "above band → die 6"
        );
        // decode_tokens mismatch → die-6 even when the ratio is perfect.
        let e = enforce_serial_band(0.038, 128, &cal, true).unwrap_err();
        assert!(
            e.contains("decode tokens") || e.contains("decode_tokens"),
            "tokens mismatch is die-6: {e}"
        );
        // A non-finite / non-positive pooled mean → die-6.
        assert!(enforce_serial_band(0.0, 512, &cal, true).is_err());
        assert!(enforce_serial_band(f64::NAN, 512, &cal, true).is_err());
    }

    #[test]
    fn r103_window_absent_is_die6_even_when_ratio_is_perfect() {
        // R103 — the calibration decode_tokens does NOT inherit; a resolved calibration with NO
        // window (None) cannot bind the band and is die-6 (check_calibration_window, absent case),
        // regardless of BASELINE_BAND_ENFORCE.
        let cal = ResolvedCalibration {
            serial_mean: 0.038,
            band_low: 0.95,
            band_high: 1.05,
            decode_tokens: None,
            timed_mode: TIMED_MODE.to_string(),
            track_id: "qwen3.8-27b-mtp-v1".to_string(),
            source: "baseline-calibration:top".to_string(),
        };
        for enforce in [true, false] {
            let o = evaluate_serial_band(0.038, 512, &cal, enforce);
            assert_eq!(
                o.verdict,
                SerialBandVerdict::Die6,
                "window absent → die-6 (enforce={enforce})"
            );
            assert!(!o.window_ok);
            assert!(o.detail.contains("no decode_tokens"), "{}", o.detail);
            assert!(enforce_serial_band(0.038, 512, &cal, enforce).is_err());
        }
    }

    #[test]
    fn r103_band_enforce_zero_warns_but_does_not_die() {
        // R103 — BASELINE_BAND_ENFORCE=0 downgrades an out-of-band ratio to a WARNING (no die), but
        // the honest outcome still records passed=false / in_band=false. The window + measured-mean
        // checks stay HARD.
        let cal = ResolvedCalibration {
            serial_mean: 0.038,
            band_low: 0.95,
            band_high: 1.05,
            decode_tokens: Some(512),
            timed_mode: TIMED_MODE.to_string(),
            track_id: "qwen3.8-27b-mtp-v1".to_string(),
            source: "baseline-calibration:top".to_string(),
        };
        // Drift, enforce=1 → die-6.
        let hard = evaluate_serial_band(0.038 * 1.20, 512, &cal, true);
        assert_eq!(hard.verdict, SerialBandVerdict::Die6);
        assert!(!hard.passed && !hard.in_band);
        // Same drift, enforce=0 → warn only (does not die) but still honestly not-in-band.
        let warn = evaluate_serial_band(0.038 * 1.20, 512, &cal, false);
        assert_eq!(warn.verdict, SerialBandVerdict::WarnOutOfBand);
        assert!(
            !warn.passed && !warn.in_band,
            "warn is still an honest failure"
        );
        assert!(
            enforce_serial_band(0.038 * 1.20, 512, &cal, false).is_ok(),
            "warn does not die"
        );
        // Window mismatch stays HARD even under enforce=0.
        assert!(
            enforce_serial_band(0.038, 128, &cal, false).is_err(),
            "window mismatch is die-6 under enforce=0"
        );
    }

    #[test]
    fn r14_resolve_loaded_util_env_driven_default_070() {
        // R14 — GPU_LOADED_UTIL is env-driven (default 0.70); a present value in (0,1] wins; an
        // invalid value fails closed.
        assert_eq!(
            resolve_loaded_util(None).unwrap(),
            (0.70, "env-GPU_LOADED_UTIL-default-0.70")
        );
        assert_eq!(resolve_loaded_util(Some("  ")).unwrap().0, 0.70);
        assert_eq!(
            resolve_loaded_util(Some("0.85")).unwrap(),
            (0.85, "env-GPU_LOADED_UTIL")
        );
        assert_eq!(resolve_loaded_util(Some("1")).unwrap().0, 1.0);
        assert!(resolve_loaded_util(Some("0")).is_err());
        assert!(resolve_loaded_util(Some("1.5")).is_err());
        assert!(resolve_loaded_util(Some("nan")).is_err());
        assert!(resolve_loaded_util(Some("cheese")).is_err());
    }

    #[test]
    fn r14_resolve_head_dirs_candidate_defaults_to_pinned() {
        // R14 — QMTP_CANDIDATE_HEAD_DIR defaults to QMTP_HEAD_DIR when unset; None when the pinned
        // head is unset (head wiring deferred to R15).
        assert_eq!(resolve_head_dirs(None, None), None);
        assert_eq!(
            resolve_head_dirs(None, Some("/cand")),
            None,
            "no pinned head → None"
        );
        let both_default = resolve_head_dirs(Some("/pinned"), None).unwrap();
        assert_eq!(both_default.head_dir, "/pinned");
        assert_eq!(
            both_default.candidate_head_dir, "/pinned",
            "candidate defaults to pinned"
        );
        let byo = resolve_head_dirs(Some("/pinned"), Some("/byo")).unwrap();
        assert_eq!(
            byo.candidate_head_dir, "/byo",
            "explicit BYO candidate head wins"
        );
    }

    #[test]
    fn r14_provenance_records_calibration_and_declared_inputs() {
        // R14 — the resolved calibration + band/bootstrap flags + declared target-id/exactness are
        // recorded in results.json provenance (not fabricated).
        let mut cfg = test_cfg(1, 1);
        cfg.target_id = Some("qwen3.8-27b-mtp-v1".to_string());
        cfg.exactness_probe = ExactnessProbe::PerPrompt;
        cfg.calibration = Some(ResolvedCalibration {
            serial_mean: 0.038,
            band_low: 0.95,
            band_high: 1.05,
            decode_tokens: Some(BENCHMARK_DECODE_STEPS),
            timed_mode: TIMED_MODE.to_string(),
            track_id: "qwen3.8-27b-mtp-v1".to_string(),
            source: "baseline-calibration:target:qwen3.8-27b-mtp-v1".to_string(),
        });
        let out = identity_run(&cfg);
        let v = serde_json::to_value(&out.results).unwrap();
        let p = &v["provenance"];
        assert_eq!(p["target_id"], json!("qwen3.8-27b-mtp-v1"));
        assert_eq!(p["exactness_probe"], json!("per-prompt"));
        assert_eq!(p["calibration_band_enforce"], json!(true));
        assert_eq!(p["calibration_bootstrap"], json!(false));
        assert!((p["baseline_calibration"]["serial_mean"].as_f64().unwrap() - 0.038).abs() < 1e-12);
        assert_eq!(
            p["baseline_calibration"]["source"],
            json!("baseline-calibration:target:qwen3.8-27b-mtp-v1")
        );
        // GPU_LOADED_UTIL env-driven source is recorded on the thermal block.
        assert_eq!(
            p["thermal"]["loaded_util_source"],
            json!("env-GPU_LOADED_UTIL-default-0.70")
        );
    }

    #[test]
    fn r103_bootstrap_authoring_new_file_and_merge_preserves_other_targets() {
        // R103 — build_bootstrap_calibration authors the per-target entry (own decode_tokens, band
        // defaults, provisional) and, when merging, preserves every other target already present.
        let input = BootstrapAuthorInput {
            target_id: "qwen3.8-27b-mtp-v1",
            timed_mode: TIMED_MODE,
            track_id: "qwen3.8-27b-mtp-v1",

            pooled_serial_mean: 0.038,
            tokens: 512,
            mtp_depth: Some(2),
            serial_control_depth: 0,
            pairs_total: 6,
        };
        // Fresh file: self-consistent top level + the one target.
        let fresh = build_bootstrap_calibration(None, &input).unwrap();
        let v: serde_json::Value = serde_json::from_str(&fresh).unwrap();
        assert!(
            (v["serial_decode_seconds_per_token_mean"].as_f64().unwrap() - 0.038).abs() < 1e-12
        );
        assert_eq!(v["decode_tokens"], json!(512));
        let t = &v["targets"]["qwen3.8-27b-mtp-v1"];
        assert!(
            (t["serial_decode_seconds_per_token_mean"].as_f64().unwrap() - 0.038).abs() < 1e-12
        );
        assert_eq!(
            t["decode_tokens"],
            json!(512),
            "per-target window recorded (no inherit)"
        );
        assert_eq!(t["provisional"], json!(true));
        assert_eq!(t["mtp_depth"], json!(2));
        // The authored file round-trips through the parser + resolves to the authored target.
        let parsed = BaselineCalibration::parse(fresh.as_bytes()).unwrap();
        assert!(
            (parsed
                .resolve(Some("qwen3.8-27b-mtp-v1"))
                .unwrap()
                .serial_mean
                - 0.038)
                .abs()
                < 1e-12
        );

        // Merge into an existing SAME-SERIES file carrying a DIFFERENT target → both survive.
        // (The file must declare its series/track: adopt-on-absent is refused below.)
        let existing = serde_json::to_vec(&json!({
            "timed_mode": TIMED_MODE,
            "track_id": "qwen3.8-27b-mtp-v1",
            "serial_decode_seconds_per_token_mean": 0.040,
            "decode_tokens": 256,
            "targets": { "other-target-v1": { "serial_decode_seconds_per_token_mean": 0.040, "decode_tokens": 256 } }
        }))
        .unwrap();
        let merged = build_bootstrap_calibration(Some(&existing), &input).unwrap();
        let mv: serde_json::Value = serde_json::from_str(&merged).unwrap();
        assert!(
            mv["targets"]["other-target-v1"].is_object(),
            "existing target preserved"
        );
        assert!(
            mv["targets"]["qwen3.8-27b-mtp-v1"].is_object(),
            "new target authored"
        );
        assert_eq!(
            mv["decode_tokens"],
            json!(256),
            "existing top level left intact"
        );

        // Delta re-review HIGH (laundering): a NON-EMPTY existing file with NO series identity is
        // pre-fence legacy — bootstrap must REFUSE to adopt it into this run's series, because
        // stamping the tag would launder its native-regime target means past the fence. This is
        // the review's executed scenario verbatim: legacy native mean 0.038 under another target,
        // then a bootstrap for a new target on the same track.
        let legacy_untagged = serde_json::to_vec(&json!({
            "serial_decode_seconds_per_token_mean": 0.038,
            "decode_tokens": 512,
            "targets": { "native_target": { "serial_decode_seconds_per_token_mean": 0.038, "decode_tokens": 512 } }
        }))
        .unwrap();
        let laundered = build_bootstrap_calibration(Some(&legacy_untagged), &input);
        let msg = laundered.expect_err("untagged non-empty legacy file must be refused");
        assert!(
            msg.contains("no series identity") && msg.contains("fresh path"),
            "diagnostic names the legacy condition and the remedy: {msg}"
        );
        // Half-tagged is equally legacy: track_id alone (the wrapper-authored shape) refuses too.
        let half_tagged = serde_json::to_vec(&json!({
            "track_id": "qwen3.8-27b-mtp-v1",
            "targets": { "native_target": { "serial_decode_seconds_per_token_mean": 0.038, "decode_tokens": 512 } }
        }))
        .unwrap();
        assert!(
            build_bootstrap_calibration(Some(&half_tagged), &input).is_err(),
            "track_id-only (wrapper-authored) legacy file must also refuse adoption"
        );

        // Malformed existing → FAIL CLOSED (never clobber other targets).
        assert!(build_bootstrap_calibration(Some(b"{ not json"), &input).is_err());
        // A non-finite mean is refused.
        let bad = BootstrapAuthorInput {
            pooled_serial_mean: f64::NAN,
            ..input_copy(&input)
        };
        assert!(build_bootstrap_calibration(None, &bad).is_err());
    }

    // Local helper: BootstrapAuthorInput is not Clone (holds a &str); rebuild it for the NAN case.
    fn input_copy<'a>(i: &BootstrapAuthorInput<'a>) -> BootstrapAuthorInput<'a> {
        BootstrapAuthorInput {
            target_id: i.target_id,
            timed_mode: TIMED_MODE,
            track_id: "qwen3.8-27b-mtp-v1",

            pooled_serial_mean: i.pooled_serial_mean,
            tokens: i.tokens,
            mtp_depth: i.mtp_depth,
            serial_control_depth: i.serial_control_depth,
            pairs_total: i.pairs_total,
        }
    }

    #[test]
    fn r103_bootstrap_write_is_atomic_roundtrip() {
        // R103 — the atomic writer installs the bytes at the destination (temp + rename), parseable back.
        let dir = std::env::temp_dir().join(format!("bench103-boot-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("baseline-calibration.json");
        let json = build_bootstrap_calibration(
            None,
            &BootstrapAuthorInput {
                target_id: "t1",
                timed_mode: TIMED_MODE,
                track_id: "qwen3.8-27b-mtp-v1",

                pooled_serial_mean: 0.037,
                tokens: 512,
                mtp_depth: Some(2),
                serial_control_depth: 0,
                pairs_total: 3,
            },
        )
        .unwrap();
        write_bootstrap_calibration(&path, &json).unwrap();
        let back = std::fs::read(&path).unwrap();
        assert!(
            (BaselineCalibration::parse(&back)
                .unwrap()
                .resolve(Some("t1"))
                .unwrap()
                .serial_mean
                - 0.037)
                .abs()
                < 1e-12
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn r103_should_author_bootstrap_requires_accept_and_parity() {
        assert!(should_author_bootstrap(true, true));
        assert!(
            !should_author_bootstrap(false, true),
            "rejected run authors nothing"
        );
        assert!(
            !should_author_bootstrap(true, false),
            "parity-false run authors nothing"
        );
    }

    #[test]
    fn r103_seals_serial_band_outcome_pass_and_drift_negative_control() {
        // R103 — the calibration verdict (mean/band/ratio/pass-fail/source) is SEALED into
        // provenance.serial_band_outcome, computed by the same evaluate_serial_band the exit uses.
        // The identity run's pooled serial mean is SERIAL_SPT (0.040).
        let make_cfg = |cal_mean: f64| {
            let mut cfg = test_cfg(1, 1);
            cfg.target_id = Some("qwen3.8-27b-mtp-v1".to_string());
            cfg.calibration = Some(ResolvedCalibration {
                serial_mean: cal_mean,
                band_low: 0.95,
                band_high: 1.05,
                decode_tokens: Some(BENCHMARK_DECODE_STEPS),
                timed_mode: TIMED_MODE.to_string(),
                track_id: "qwen3.8-27b-mtp-v1".to_string(),
                source: "baseline-calibration:target:qwen3.8-27b-mtp-v1".to_string(),
            });
            cfg
        };

        // In band: cal mean == measured 0.040 → ratio 1.0, passed=true.
        let out = identity_run(&make_cfg(SERIAL_SPT));
        let v = serde_json::to_value(&out.results).unwrap();
        let o = &v["provenance"]["serial_band_outcome"];
        assert_eq!(o["verdict"], json!("pass"));
        assert_eq!(o["passed"], json!(true));
        assert_eq!(o["in_band"], json!(true));
        assert_eq!(o["window_ok"], json!(true));
        assert!(
            (o["ratio"].as_f64().unwrap() - 1.0).abs() < 1e-9,
            "ratio ~1.0"
        );
        assert!((o["pooled_serial_mean"].as_f64().unwrap() - SERIAL_SPT).abs() < 1e-12);
        assert_eq!(
            o["source"],
            json!("baseline-calibration:target:qwen3.8-27b-mtp-v1")
        );

        // NEGATIVE CONTROL — drift: cal mean 0.030 → ratio 1.333, outside band; die-6 verdict sealed
        // as an HONEST failure (the results are still sealed; main.rs turns Die6 into exit 6).
        let drift = identity_run(&make_cfg(0.030));
        let dv = serde_json::to_value(&drift.results).unwrap();
        let d = &dv["provenance"]["serial_band_outcome"];
        assert_eq!(d["verdict"], json!("die6"));
        assert_eq!(d["passed"], json!(false));
        assert_eq!(d["in_band"], json!(false));
        assert!(d["ratio"].as_f64().unwrap() > 1.05, "ratio above band");
        // The verdict the EXIT uses matches the sealed outcome.
        assert!(enforce_serial_band(
            SERIAL_SPT,
            BENCHMARK_DECODE_STEPS,
            &make_cfg(0.030).calibration.unwrap(),
            true
        )
        .is_err());
    }

    #[test]
    fn r103_bootstrap_run_skips_the_band_and_seals_no_outcome() {
        // R103 negative control — a --calibration-bootstrap run AUTHORS the band; it does not check
        // it. Even with a calibration that WOULD drift, no serial_band_outcome is sealed and the
        // bootstrap flag is recorded.
        let mut cfg = test_cfg(1, 1);
        cfg.target_id = Some("qwen3.8-27b-mtp-v1".to_string());
        cfg.calibration_bootstrap = true;
        cfg.calibration = Some(ResolvedCalibration {
            serial_mean: 0.030, // would drift if checked
            band_low: 0.95,
            band_high: 1.05,
            decode_tokens: Some(BENCHMARK_DECODE_STEPS),
            timed_mode: TIMED_MODE.to_string(),
            track_id: "qwen3.8-27b-mtp-v1".to_string(),
            source: "baseline-calibration:target:qwen3.8-27b-mtp-v1".to_string(),
        });
        let out = identity_run(&cfg);
        let v = serde_json::to_value(&out.results).unwrap();
        assert_eq!(v["provenance"]["calibration_bootstrap"], json!(true));
        assert!(
            v["provenance"]["serial_band_outcome"].is_null(),
            "bootstrap seals no band verdict"
        );
    }

    #[test]
    fn worker_executable_override_wins_or_conflicts() {
        // finding 2: workspace engine wins; a conflicting override is a hard error; override
        // fills in when the workspace declares none; neither ⇒ error.
        assert_eq!(
            resolve_worker_executable(Some("/ws/engine"), None).unwrap(),
            "/ws/engine"
        );
        assert_eq!(
            resolve_worker_executable(Some("/ws/engine"), Some("/ws/engine")).unwrap(),
            "/ws/engine"
        );
        assert!(resolve_worker_executable(Some("/ws/engine"), Some("/other")).is_err());
        assert_eq!(
            resolve_worker_executable(None, Some("/override")).unwrap(),
            "/override"
        );
        assert!(resolve_worker_executable(None, None).is_err());
    }

    /// Make `<ws>/.build/release/<bin>` a real executable file inside a fresh temp dir; return the
    /// workspace path and the resolved engine path.
    fn make_ws_with_engine(bin: &str) -> (std::path::PathBuf, String) {
        let ws = std::env::temp_dir().join(format!(
            "measure-ws-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let rel = ws.join(".build").join("release");
        std::fs::create_dir_all(&rel).unwrap();
        let engine = rel.join(bin);
        std::fs::write(&engine, b"#!/bin/sh\nexit 0\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&engine, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let engine_str = engine.to_string_lossy().to_string();
        (ws, engine_str)
    }

    #[test]
    fn metallib_sibling_guard_fails_closed_when_absent_and_passes_when_present() {
        // #42 box-leg — the pre-GPU adjacency guard. A resolved worker with no sibling
        // `mlx.metallib` is refused (naming the file); staging the sibling makes it pass.
        let (ws, engine) = make_ws_with_engine(DEFAULT_MEASURE_WORKER_BIN);
        let err = verify_worker_metallib_sibling(&engine).unwrap_err();
        assert!(err.contains(MLX_METALLIB_SIBLING), "err={err}");
        assert!(err.contains("die 8"), "err={err}");
        // Stage the sibling next to the resolved binary → the guard passes.
        let sibling = Path::new(&engine).with_file_name(MLX_METALLIB_SIBLING);
        std::fs::write(&sibling, b"").unwrap();
        assert!(verify_worker_metallib_sibling(&engine).is_ok());
        std::fs::remove_dir_all(&ws).ok();
    }

    #[test]
    fn workspace_engine_resolves_build_release_binary() {
        // The WORKSPACE contract: `<ws>/.build/release/<bin>` is resolved (not the ws path
        // verbatim), the default bin is `mlxfast-runtime-worker`, and a present executable resolves.
        let (ws, engine) = make_ws_with_engine(DEFAULT_MEASURE_WORKER_BIN);
        let resolved =
            resolve_workspace_engine(&ws.to_string_lossy(), DEFAULT_MEASURE_WORKER_BIN, None)
                .unwrap();
        assert_eq!(resolved, engine);
        // An overridable binary name resolves the alternate file inside the same workspace.
        let (ws2, engine2) = make_ws_with_engine("alt-worker");
        assert_eq!(
            resolve_workspace_engine(&ws2.to_string_lossy(), "alt-worker", None).unwrap(),
            engine2
        );
        std::fs::remove_dir_all(&ws).ok();
        std::fs::remove_dir_all(&ws2).ok();
    }

    #[test]
    fn workspace_engine_fails_closed_when_absent() {
        // Fail-closed: a workspace with no `.build/release/<bin>` is a hard error naming the path.
        let ws = std::env::temp_dir().join(format!("measure-ws-absent-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        let err = resolve_workspace_engine(&ws.to_string_lossy(), DEFAULT_MEASURE_WORKER_BIN, None)
            .unwrap_err();
        assert!(
            err.contains(".build/release/mlxfast-runtime-worker"),
            "err={err}"
        );
        assert!(err.contains("not found or not executable"), "err={err}");
        std::fs::remove_dir_all(&ws).ok();
    }

    #[test]
    fn workspace_engine_override_conflict_is_hard_error() {
        // finding-2 override semantics survive workspace resolution: a CONFLICTING explicit
        // override (≠ the resolved workspace engine) is a hard error; an AGREEING one is fine.
        let (ws, engine) = make_ws_with_engine(DEFAULT_MEASURE_WORKER_BIN);
        assert!(resolve_workspace_engine(
            &ws.to_string_lossy(),
            DEFAULT_MEASURE_WORKER_BIN,
            Some("/some/other/engine"),
        )
        .is_err());
        assert_eq!(
            resolve_workspace_engine(
                &ws.to_string_lossy(),
                DEFAULT_MEASURE_WORKER_BIN,
                Some(&engine)
            )
            .unwrap(),
            engine
        );
        std::fs::remove_dir_all(&ws).ok();
    }

    #[test]
    fn lower_median_is_floor_order_statistic_and_guards_before_aggregating() {
        // R16 NAME-TRAP — the per-pair diagnostic uses the LOWER-median (order statistic at
        // (len-1)/2 floor), NOT the even-n mean-of-two-central. For [1,2,3,4] that is index 1 = 2.0
        // (the LOWER central), DISTINCT from the published even-n median 2.5. Odd n = the middle.
        assert_eq!(
            lower_median(&[1.0, 3.0, 2.0]),
            2.0,
            "odd n → middle order statistic"
        );
        assert_eq!(
            lower_median(&[1.0, 2.0, 3.0, 4.0]),
            2.0,
            "even n → LOWER central (index (4-1)/2=1), NOT the even-n mean 2.5"
        );
        assert_ne!(
            lower_median(&[1.0, 2.0, 3.0, 4.0]),
            bench_core::score::paired_decode_only_median(&[1.0, 2.0, 3.0, 4.0]),
            "the per-pair lower-median is DISTINCT from the published even-n median"
        );
        // finding 6: a finite guard BEFORE aggregation (no NaN panic).
        assert_eq!(lower_median(&[]), 0.0);
        assert_eq!(
            lower_median(&[1.0, f64::NAN, 2.0]),
            0.0,
            "a NaN member yields 0, never a panic"
        );
    }

    #[test]
    fn results_sealed_json_is_sorted_and_pretty() {
        let out = identity_run(&test_cfg(3, 4));
        let json = out.results.to_sealed_json().unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["track_id"], json!("qwen3.8-27b-mtp-v1"));
        // Provenance carries both legs' resolved executables + thermal source (findings 1/2/8).
        assert_eq!(v["provenance"]["candidate_executable"], json!("cand-ws"));
        assert_eq!(v["provenance"]["baseline_executable"], json!("base-ws"));
        assert_eq!(
            v["provenance"]["thermal"]["cool_gate_c_source"],
            json!("wrapper-constant-40")
        );
    }

    /// A SCORED CONTRACT for `resolve_track_id`'s tests: it declares the track (or none), plus the
    /// regime and the two flow flags the resolver's own fences read. STEP 3 (David 2026-09-15)
    /// moved those three from per-track tables onto the fixture, so a test that drives the
    /// resolver has to hand it one.
    fn scored_for(
        track: Option<&str>,
        paired_flow_retired: bool,
    ) -> bench_core::contract::Contract {
        bench_core::contract::Contract {
            track_id: track.map(str::to_string),
            scored_batch_size: Some(1),
            prefill_gain_exponent: Some(0.0),
            decode_gain_exponent: Some(1.0),
            scores_against_live_control_leg: Some(paired_flow_retired),
            paired_flow_retired: Some(paired_flow_retired),
            ..bench_core::contract::Contract::NONE_DECLARED
        }
    }

    #[test]
    fn resolve_track_id_env_contract_and_cross_check() {
        // R12 — env wins when set; contract used when env unset; env==contract → that one value;
        // env≠contract hard-errors (constant≡contract≡env); neither present fails closed;
        // whitespace/empty is treated as absent.
        const TRACK: &str = bench_core::constants::TRACK_ID;
        assert_eq!(
            resolve_track_id(Some(TRACK), &scored_for(None, false)).unwrap(),
            TRACK
        );
        assert_eq!(
            resolve_track_id(None, &scored_for(Some(TRACK), false)).unwrap(),
            TRACK
        );
        assert_eq!(
            resolve_track_id(Some(TRACK), &scored_for(Some(TRACK), false)).unwrap(),
            TRACK
        );
        // env≠contract → HARD ERROR (the "one value" rule).
        let err = resolve_track_id(Some(TRACK), &scored_for(Some("qwen3.6-27b-mtp-v1"), false))
            .unwrap_err();
        assert!(
            err.contains("track_id mismatch"),
            "mismatch must be named: {err}"
        );
        // neither → fail closed.
        assert!(resolve_track_id(None, &scored_for(None, false)).is_err());
        // whitespace/empty is absent.
        assert!(resolve_track_id(Some("  "), &scored_for(None, false)).is_err());
        assert_eq!(
            resolve_track_id(Some("  "), &scored_for(Some(TRACK), false)).unwrap(),
            TRACK
        );
    }

    /// The THIRD leg: env and contract can AGREE with each other and still name a track this tree
    /// does not serve. That is the mis-attribution the two-leg rule never caught — the run would
    /// seal a foreign `track_id` over this track's per-track baseline pair — so the resolver
    /// refuses by name, naming BOTH the tree's track and the declared one.
    /// CALL-SITE PROOF — the paired flow is closed to the SINGLE-LEG tracks at the one place a
    /// measure-job run settles its track id, not merely in the constants helper.
    ///
    /// The refusal is the SPECIFIC one (it names the single-leg path the track does score
    /// through), which is what proves the fence runs BEFORE the generic tree fence rather than
    /// being shadowed by it. NEGATIVE CONTROL: the paired tracks still resolve, so this is not a
    /// fence that refuses everything.
    #[test]
    fn resolve_track_id_refuses_the_single_leg_tracks_by_name() {
        for track in ["qwen3.8-125b-a6b-mlx-v1", "qwen3.8-125b-a6b-cuda-v1"] {
            let err = resolve_track_id(Some(track), &scored_for(Some(track), true)).unwrap_err();
            assert!(
                err.contains(bench_core::constants::PAIRED_FLOW_RETIRED_FOR_TRACK),
                "{track}: must name the sentinel, not the generic fence: {err}"
            );
            assert!(err.contains(track), "{track}: must name the track: {err}");
            assert!(
                err.contains("iterate --mode official"),
                "{track}: must name the path it DOES score through: {err}"
            );
        }
        // NEGATIVE CONTROL: this tree's own paired track still resolves through measure-job.
        assert_eq!(
            resolve_track_id(
                Some(bench_core::constants::TRACK_ID),
                &scored_for(Some(bench_core::constants::TRACK_ID), false)
            )
            .unwrap(),
            bench_core::constants::TRACK_ID
        );
    }

    #[test]
    fn resolve_track_id_refuses_a_foreign_track_naming_both() {
        const FOREIGN: &str = "gemma4-26b-a4b-mlx-v1";
        assert_ne!(FOREIGN, bench_core::constants::TRACK_ID);
        // Agreeing env+contract, agreeing with each other, both foreign.
        for (env, contract) in [
            (Some(FOREIGN), Some(FOREIGN)),
            (Some(FOREIGN), None),
            (None, Some(FOREIGN)),
        ] {
            let err = resolve_track_id(env, &scored_for(contract, false)).unwrap_err();
            assert!(err.contains("track_id fence"), "{err}");
            assert!(err.contains(FOREIGN), "must name the declared track: {err}");
            assert!(
                err.contains(bench_core::constants::TRACK_ID),
                "must name the tree's track: {err}"
            );
        }
    }

    #[test]
    fn resolve_track_name_env_then_contract_else_none() {
        // R12 — env wins, else contract, else None (omitted, never fabricated).
        assert_eq!(
            resolve_track_name(Some("env-name"), Some("c-name")).as_deref(),
            Some("env-name")
        );
        assert_eq!(
            resolve_track_name(None, Some("c-name")).as_deref(),
            Some("c-name")
        );
        assert_eq!(resolve_track_name(None, None), None);
        assert_eq!(resolve_track_name(Some("  "), None), None);
    }

    /// R12 — `track_name` (the LABEL) and `track_id` (the sealed CONSTANT `resolve_track_id`
    /// reads) both parse off the ONE track-fixture schema, so a fixture states each of them once.
    #[test]
    fn contract_parses_track_name_and_track_id_off_the_one_schema() {
        let bytes = serde_json::to_vec(&json!({
            "track_id": "qwen3.6-27b-mtp-v1",
            "track_name": "mlxfast-challenge-dev-qwen-mtp",
            "timed_prompt_pool": []
        }))
        .unwrap();
        let c = Contract::parse(&bytes).unwrap();
        assert_eq!(
            c.track_name.as_deref(),
            Some("mlxfast-challenge-dev-qwen-mtp")
        );
        assert_eq!(c.track_id.as_deref(), Some("qwen3.6-27b-mtp-v1"));
        // A fixture carrying neither parses to None for both.
        let bare = Contract::parse(b"{}").unwrap();
        assert_eq!(bare.track_name, None);
        assert_eq!(bare.track_id, None);
    }

    #[test]
    fn seal_track_id_tag_and_mode_are_distinct() {
        // R12 — the seal carries the CONSTANT track_id and the per-run tag as DISTINCT top-level
        // fields, plus the optional track_name and the live mode string.
        let mut cfg = test_cfg(3, 4);
        cfg.track_id = "qwen3.6-27b-mtp-v1".to_string();
        cfg.tag = "qwen-mtp-mjob-20260818".to_string();
        cfg.track_name = Some("mlxfast-challenge-dev-qwen-mtp".to_string());
        let out = identity_run(&cfg);
        let v = serde_json::to_value(&out.results).unwrap();
        assert_eq!(v["track_id"], json!("qwen3.6-27b-mtp-v1"));
        assert_eq!(v["tag"], json!("qwen-mtp-mjob-20260818"));
        assert_ne!(
            v["track_id"], v["tag"],
            "track_id (constant) and tag (per-run) are distinct"
        );
        assert_eq!(v["track_name"], json!("mlxfast-challenge-dev-qwen-mtp"));
        assert_eq!(v["mode"], json!("qwen-native-mtp-paired-decode-only"));
    }

    #[test]
    fn medium_unpinned_golden_is_die8_pre_gpu() {
        // Medium (cycle-3) — a golden whose sha256 does not resolve to EXACTLY ONE positive
        // noop_decode_speedup in the pool is die-8 BEFORE GPU (wrapper noop_reference_for_golden).
        let pool = vec![
            PromptPoolEntry {
                sha256: "aa".to_string(),
                bytes: None,
                noop_decode_speedup: Some(0.99),
            },
            PromptPoolEntry {
                sha256: "bb".to_string(),
                bytes: None,
                noop_decode_speedup: Some(0.98),
            },
        ];
        // All pinned → OK.
        assert!(validate_goldens_pinned(&[tape_sha("aa"), tape_sha("bb")], &pool).is_ok());
        // Case-insensitive sha match (build_results uses the same rule).
        assert!(validate_goldens_pinned(&[tape_sha("AA")], &pool).is_ok());

        // Absent from the pool → die-8.
        let e = validate_goldens_pinned(&[tape_sha("cc")], &pool).unwrap_err();
        assert!(e.contains("no pinned per-prompt no-op reference"), "{e}");
        assert!(e.contains("die 8"), "{e}");

        // Present but noop omitted / non-positive → die-8.
        let no_noop = vec![PromptPoolEntry {
            sha256: "aa".to_string(),
            bytes: None,
            noop_decode_speedup: None,
        }];
        assert!(validate_goldens_pinned(&[tape_sha("aa")], &no_noop)
            .unwrap_err()
            .contains("die 8"));
        let zero = vec![PromptPoolEntry {
            sha256: "aa".to_string(),
            bytes: None,
            noop_decode_speedup: Some(0.0),
        }];
        assert!(validate_goldens_pinned(&[tape_sha("aa")], &zero)
            .unwrap_err()
            .contains("non-positive"));

        // Ambiguous (two pool entries for one sha) → die-8.
        let dup = vec![
            PromptPoolEntry {
                sha256: "aa".to_string(),
                bytes: None,
                noop_decode_speedup: Some(0.99),
            },
            PromptPoolEntry {
                sha256: "aa".to_string(),
                bytes: None,
                noop_decode_speedup: Some(0.97),
            },
        ];
        assert!(validate_goldens_pinned(&[tape_sha("aa")], &dup)
            .unwrap_err()
            .contains("AMBIGUOUS"));
    }

    /// The anti-lottery ≥N-distinct COVERAGE gate. A pool of N distinct pins (N=8, the live-track
    /// cardinality) whose sha are `sha(0)..sha(7)`, each with a positive no-op.
    fn coverage_pool_8() -> Vec<PromptPoolEntry> {
        (0..8)
            .map(|i| PromptPoolEntry {
                sha256: format!("{:064x}", i + 1),
                bytes: None,
                noop_decode_speedup: Some(1.0),
            })
            .collect()
    }

    /// The N tapes whose sha256 are EXACTLY the N distinct pinned shas of [`coverage_pool_8`].
    fn coverage_tapes_8() -> Vec<TimedPrompt> {
        (0..8)
            .map(|i| tape_sha(&format!("{:064x}", i + 1)))
            .collect()
    }

    #[test]
    fn timed_coverage_must_be_exactly_the_distinct_pinned_pool_die8() {
        let pool = coverage_pool_8();

        // BASELINE (non-vacuous): the correct 8-distinct-pinned run PASSES. This is the mutation
        // guard — a gate that returned Err unconditionally would fail HERE.
        validate_timed_pool_coverage(&coverage_tapes_8(), &pool)
            .expect("the exactly-8-distinct-pinned run must PASS the coverage gate");

        // SUBSET — a 7-prompt run (one pinned prompt never timed) → die-8.
        let subset: Vec<TimedPrompt> = coverage_tapes_8().into_iter().take(7).collect();
        let e =
            validate_timed_pool_coverage(&subset, &pool).expect_err("a 7-prompt SUBSET must die-8");
        assert!(e.contains("die 8"), "subset must be die-8: {e}");
        assert!(
            e.contains("not EXACTLY") && e.contains("SUBSET"),
            "subset diagnostic must name the exact-coverage breach: {e}"
        );

        // DUPLICATE — 8 timed prompts but only 7 DISTINCT (prompt 0 repeated in place of 7) → die-8.
        let mut dup = coverage_tapes_8();
        dup[7] = tape_sha(&format!("{:064x}", 1)); // == sha(0), so 7 distinct across 8 timed
        let e = validate_timed_pool_coverage(&dup, &pool)
            .expect_err("8 timed with <8 distinct (a DUPLICATE) must die-8");
        assert!(e.contains("die 8"), "duplicate must be die-8: {e}");
        assert!(
            e.contains("repeats") || e.contains("DUPLICATE"),
            "duplicate diagnostic must name the repeat: {e}"
        );

        // SUBSTITUTION — 8 distinct timed, but one sha256 matches NO pin → die-8.
        let mut sub = coverage_tapes_8();
        sub[3] = tape_sha(&format!("{:064x}", 0xdead_beefu64)); // not in the pinned set
        let e = validate_timed_pool_coverage(&sub, &pool)
            .expect_err("a timed prompt matching no pin (SUBSTITUTION) must die-8");
        assert!(e.contains("die 8"), "substitution must be die-8: {e}");
        assert!(
            e.contains("SUBSTITUTION") && e.contains("match NO pin"),
            "substitution diagnostic must name the unpinned timed prompt: {e}"
        );

        // A pool that itself pins a duplicate sha has no well-defined distinct support → die-8.
        let mut bad_pool = coverage_pool_8();
        bad_pool[7].sha256 = bad_pool[0].sha256.clone();
        let e = validate_timed_pool_coverage(&coverage_tapes_8(), &bad_pool)
            .expect_err("a pool pinning a duplicate sha must die-8");
        assert!(
            e.contains("die 8") && e.contains("DUPLICATE sha256"),
            "duplicate-pin pool must die-8: {e}"
        );
    }

    #[test]
    fn captured_engine_wire_crosscheck_die8_at_measure_time() {
        // #142 — the measure-job gate (`crosscheck_captured_engine_wire`) is the function
        // `execute_measure_job` calls pre-GPU; this is the LIVE measure-job path, NOT the offline
        // cargo-test crosscheck. MATCH: benchd's embedded captured reference verifies against its
        // pinned mirror-integrity sha and parses under WorkerResponse → Ok (no die).
        crosscheck_captured_engine_wire(
            bench_runner::ENGINE_WIRE_V1_FIXTURE.as_bytes(),
            bench_runner::ENGINE_WIRE_V1_SHA256,
        )
        .expect(
            "embedded captured reference matches the mirror-integrity reference at measure time",
        );

        // MISMATCH: change a byte inside a string value (the nonce) so the captured sha no longer
        // equals the reference, while every line STAYS valid JSON and still parses under
        // `WorkerResponse` — so only the sha gate can reject it (a real byte disagreement, not a
        // trivial pass-regardless default, and not a parse error standing in for the sha check).
        // This is the mutation the test kills: neuter the sha gate and these bytes parse cleanly, so
        // `crosscheck_captured_engine_wire` returns Ok, the `expect_err` panics, and the test fails.
        let tampered =
            bench_runner::ENGINE_WIRE_V1_FIXTURE.replace("session-nonce", "session-xonce");
        let err = crosscheck_captured_engine_wire(
            tampered.as_bytes(),
            bench_runner::ENGINE_WIRE_V1_SHA256,
        )
        .expect_err("captured-bytes disagreement with the mirror-integrity reference must die");
        assert!(
            err.contains("die 8"),
            "captured-wire mismatch must be die-8: {err}"
        );
        assert!(
            err.contains("mirror-integrity reference"),
            "the die-8 diagnostic must name the reference it disagreed with: {err}"
        );
    }

    #[test]
    fn pool_entry_bytes_are_enforced_when_present_and_optional_when_absent() {
        // #112 (L3) — a canonical golden is pinned by sha256 AND bytes. The byte half was parsed
        // past and never checked; now it is enforced whenever the entry declares it.
        let prompt = prompt_with_sha_bytes(PROMPT_KIND_TAPE, "aa", SYNTHETIC_PROMPT_BYTES);

        // BOTH halves agree → accepted.
        let matching = vec![PromptPoolEntry {
            sha256: "aa".to_string(),
            bytes: Some(SYNTHETIC_PROMPT_BYTES),
            noop_decode_speedup: Some(0.99),
        }];
        assert!(validate_goldens_pinned(std::slice::from_ref(&prompt), &matching).is_ok());

        // Same sha, DIFFERENT declared byte count → die-8, naming both numbers. (Unreachable in
        // practice without a sha collision, which is the point: the pin is checked as a whole.)
        let mismatched = vec![PromptPoolEntry {
            sha256: "aa".to_string(),
            bytes: Some(SYNTHETIC_PROMPT_BYTES + 1),
            noop_decode_speedup: Some(0.99),
        }];
        let e = validate_goldens_pinned(std::slice::from_ref(&prompt), &mismatched).unwrap_err();
        assert!(e.contains("die 8"), "{e}");
        assert!(
            e.contains(&format!("{} bytes", SYNTHETIC_PROMPT_BYTES + 1)),
            "{e}"
        );
        assert!(
            e.contains(&format!("{} bytes", SYNTHETIC_PROMPT_BYTES)),
            "{e}"
        );
        assert!(e.contains("sha256 AND bytes"), "{e}");

        // NO declared byte count → the sha-only pin stands, exactly as before (offline fixtures
        // that never carried `bytes` keep working).
        let sha_only = vec![PromptPoolEntry {
            sha256: "aa".to_string(),
            bytes: None,
            noop_decode_speedup: Some(0.99),
        }];
        assert!(validate_goldens_pinned(&[prompt], &sha_only).is_ok());

        // The byte check runs on the LEGACY shape too, not only on tapes.
        let golden = prompt_with_sha_bytes(PROMPT_KIND_GOLDEN, "aa", SYNTHETIC_PROMPT_BYTES + 9);
        let e = validate_goldens_pinned(&[golden], &matching).unwrap_err();
        assert!(e.contains("die 8"), "{e}");
    }

    #[test]
    fn gates_golden_missing_the_benchmark_oracle_is_refused_naming_the_remedy() {
        // 2b box-leg — a legacy GoldenDocument routed to the gates phase MUST carry the `.benchmark`
        // oracle. A benchmark-less Golden is refused EARLY, and the message names the engine's
        // weightless attach-benchmark-oracle remedy (RED if `validate_gates_goldens_carry_oracle`
        // is reverted: the benchmark-less golden would then fall through to the generic per-prompt
        // window refusal that frames a missing oracle as a token-count shortfall).
        let no_oracle = prompt_with_sha_bytes(PROMPT_KIND_GOLDEN, "aa", SYNTHETIC_PROMPT_BYTES);
        assert!(matches!(&no_oracle, TimedPrompt::Golden(g) if g.benchmark.is_none()));
        let e = validate_gates_goldens_carry_oracle(std::slice::from_ref(&no_oracle)).unwrap_err();
        assert!(e.contains(ATTACH_BENCHMARK_ORACLE_REMEDY), "{e}");
        assert!(e.contains("`.benchmark` oracle"), "{e}");
        assert!(e.contains("die 8"), "{e}");

        // A Golden that DOES carry the oracle is accepted.
        assert!(validate_gates_goldens_carry_oracle(&[measure_golden()]).is_ok());

        // A TAPE carries its reference rows directly and is EXEMPT — it is never asked for an oracle.
        assert!(validate_gates_goldens_carry_oracle(&[measure_tape()]).is_ok());
    }

    #[test]
    fn window_20260819_die8_repro_golden_document_vs_a_tape_pinned_pool() {
        // THE BLOCKING FINDING, as a regression test. The live `timed_prompt_pool` pins TAPES. Before
        // this change `--golden` modelled ONLY GoldenDocument, so both directions died pre-GPU and
        // no input could satisfy the contract. Both directions are asserted here.
        let tape = measure_tape();
        let pool = vec![PromptPoolEntry {
            sha256: tape.sha256().to_string(),
            bytes: Some(tape.byte_len()),
            noop_decode_speedup: Some(0.9206),
        }];

        // (1) The PINNED POOL OBJECT (a tape) now LOADS as a golden and PINS. This is the direction
        // that used to die-8 at load with "unknown field `emitted_tokens`".
        assert_eq!(tape.kind(), PROMPT_KIND_TAPE);
        assert!(
            validate_goldens_pinned(&[tape], &pool).is_ok(),
            "the pinned pool tape must satisfy the R4 pin check"
        );

        // (2) A real GoldenDocument against the same tape-pinned pool STILL dies-8 — correctly, since
        // its bytes are not what the pool pins — but the diagnostic is now HONEST about the cause
        // instead of sending the reader hunting for a wrong fixture.
        let golden = measure_golden();
        assert_eq!(golden.kind(), PROMPT_KIND_GOLDEN);
        let e = validate_goldens_pinned(&[golden], &pool).unwrap_err();
        assert!(e.contains("no pinned per-prompt no-op reference"), "{e}");
        assert!(e.contains("die 8"), "{e}");
        assert!(
            e.contains(PROMPT_KIND_GOLDEN),
            "names the shape it got: {e}"
        );
        assert!(
            e.contains(PROMPT_KIND_TAPE),
            "names the shape the pool pins: {e}"
        );
        assert!(
            e.contains("can therefore never match a pool pin"),
            "states WHY the two can never meet: {e}"
        );

        // R4 itself is UNCHANGED by the tape work: exactly-one, positive, fail-closed — including
        // for tapes (a tape absent from the pool is die-8 exactly as a golden is).
        let unpinned = measure_tape_with(9, oracle_decode_tokens());
        let e = validate_goldens_pinned(&[unpinned], &pool).unwrap_err();
        assert!(e.contains("no pinned per-prompt no-op reference"), "{e}");
    }

    #[test]
    fn tape_timing_params_carry_the_live_wrapper_leg_semantics() {
        // The legs must consume the SAME fields the live wrapper's `mtp-timed` verb does:
        // seed_tokens = the timed seed prefill's prompt; reference_seed_token = the token that
        // prefill must produce; rows[i].sequential_argmax = the token emitted at index i+1.
        let chain = oracle_decode_tokens();
        let tape = measure_tape_with(7, chain.clone());
        let params = timing_params(&tape, 8).unwrap();
        assert_eq!(
            params.decode_seed_tokens,
            vec![7i64; BENCHMARK_DECODE_SEED_TOKENS]
        );
        assert_eq!(params.expected_decode_seed_token, SEED_TOKEN);
        assert_eq!(params.expected_decode_tokens, chain);
        assert_eq!(params.decode_steps, 8);
        // No prefill oracle exists on a tape and none is invented (measure-job legs never open a
        // prefill phase — the seed prefill is inside the decode clock). #112 (L1): BOTH prefill
        // fields are unset, so the claim holds for the oracle too — it used to be silently set
        // from `reference_seed_token`.
        assert!(params.prefill_prompt_tokens.is_empty());
        assert_eq!(params.expected_prefill_token, None);

        // A window longer than the tape's reference chain fails CLOSED, pre-GPU.
        let short = measure_tape_with(7, chain[..4].to_vec());
        let e = timing_params(&short, 8).unwrap_err();
        assert!(e.contains("4 reference rows"), "{e}");
        assert!(e.contains("--tokens 8"), "{e}");

        // The legacy GoldenDocument mapping is untouched.
        let golden = measure_golden();
        let gp = timing_params(&golden, 8).unwrap();
        assert_eq!(gp.expected_decode_seed_token, SEED_TOKEN);
        assert_eq!(gp.expected_decode_tokens, oracle_decode_tokens());
        assert_eq!(
            gp.prefill_prompt_tokens.len(),
            BENCHMARK_PREFILL_PROMPT_TOKENS
        );
    }

    #[test]
    fn tape_pool_runs_the_whole_pair_loop_and_binds_per_prompt_by_tape_bytes() {
        // End to end over the core: a POOL OF TAPES (the live shape) drives the same pair loop, and
        // each per_prompt record is bound BY BYTES to its own tape's sha — resolving its pinned
        // no-op reference from the contract pool (`contract-pool-match`), which is the whole point
        // of making the tape the golden input.
        let tapes = vec![
            measure_tape_with(1, oracle_decode_tokens()),
            measure_tape_with(2, oracle_decode_tokens()),
        ];
        let mut cfg = test_cfg(2, 2);
        cfg.prompt_pool = vec![
            PromptPoolEntry {
                sha256: tapes[0].sha256().to_string(),
                bytes: Some(tapes[0].byte_len()),
                noop_decode_speedup: Some(0.9206),
            },
            PromptPoolEntry {
                sha256: tapes[1].sha256().to_string(),
                bytes: Some(tapes[1].byte_len()),
                noop_decode_speedup: Some(0.797),
            },
        ];
        let out = run_measure_job(
            &tapes,
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            move |_p| ok_candidate(),
        )
        .unwrap();
        assert!(out.candidate_accepted);
        assert_eq!(out.results.prompt_count, 2);
        assert_eq!(out.results.per_prompt.len(), 2);
        assert_eq!(out.results.accepted_pair_count, out.results.pairs.len());
        for (i, pp) in out.results.per_prompt.iter().enumerate() {
            assert_eq!(
                pp.prompt_sha256,
                tapes[i].sha256(),
                "bound BY BYTES to its tape"
            );
            assert_eq!(pp.prompt_sha256_source, "contract-pool-match");
        }
        assert_eq!(
            out.results.per_prompt[0].noop_reference_decode_speedup,
            Some(0.9206)
        );
    }

    #[test]
    fn spec_mode_allowed_is_the_candidate_gate_depth0_via_serial() {
        // Depth-0-via-serial-mode (docs/spec-config-design.md step 4): candidate validation keys on
        // the MODE being in the track's allowed list — NOT a depth-int floor. Serial is a valid mode
        // with NO depth field, so depth-0 stops being a candidate depth and the ">= 2" straggler
        // dissolves.
        let allowed = DEFAULT_ALLOWED_MODES;
        assert!(validate_spec_mode_allowed(&SpecConfig::serial(), &allowed).is_ok());
        assert!(validate_spec_mode_allowed(&SpecConfig::mtp(2), &allowed).is_ok());
        // depth 1 (the old "diagnostic") is a perfectly valid mtp spec now — the mode is what gates.
        assert!(validate_spec_mode_allowed(&SpecConfig::mtp(1), &allowed).is_ok());
        // A mode outside the allowed list rejects before any timed work.
        let err = validate_spec_mode_allowed(
            &SpecConfig {
                mode: "dflash".to_string(),
                mtp: None,
                dflash: Some(serde_json::json!({})),
                dspark: None,
            },
            &allowed,
        )
        .unwrap_err();
        assert!(err.contains("allowed-modes"), "{err}");
        // Internal shape: mtp mode must carry its block; serial must not.
        assert!(validate_spec_mode_allowed(
            &SpecConfig {
                mode: "mtp".to_string(),
                mtp: None,
                dflash: None,
                dspark: None
            },
            &allowed
        )
        .is_err());
        assert!(validate_spec_mode_allowed(
            &SpecConfig {
                mode: "serial".to_string(),
                mtp: Some(bench_protocol::MtpSpec { depth: Some(2) }),
                dflash: None,
                dspark: None,
            },
            &allowed
        )
        .is_err());
    }

    #[test]
    fn medium_spec_module_coherence_cross_module_and_mtp0_reject() {
        // Medium (#105) — CROSS-MODULE keys reject: exactly the ONE block matching the mode may be
        // present. mtp(0) is the serial control, not a candidate — it rejects.
        assert!(validate_spec_module_coherent(&SpecConfig::serial()).is_ok());
        assert!(validate_spec_module_coherent(&SpecConfig::mtp(2)).is_ok());
        // mtp mode carrying a stray dflash block (cross-module) → reject.
        let cross = SpecConfig {
            mode: "mtp".to_string(),
            mtp: Some(bench_protocol::MtpSpec { depth: Some(2) }),
            dflash: Some(serde_json::json!({})),
            dspark: None,
        };
        let e = validate_spec_module_coherent(&cross).unwrap_err();
        assert!(e.contains("cross-module key"), "{e}");
        // serial mode carrying a dflash block (cross-module) → reject.
        let serial_cross = SpecConfig {
            mode: "serial".to_string(),
            mtp: None,
            dflash: Some(serde_json::json!({})),
            dspark: None,
        };
        assert!(validate_spec_module_coherent(&serial_cross)
            .unwrap_err()
            .contains("cross-module key"));
        // mtp(0) candidate → reject (depth 0 is the serial control).
        let mtp0 = validate_spec_module_coherent(&SpecConfig::mtp(0)).unwrap_err();
        assert!(mtp0.contains("mtp.depth 0"), "{mtp0}");
        // mtp(1) is a valid candidate depth (>= 1).
        assert!(validate_spec_module_coherent(&SpecConfig::mtp(1)).is_ok());
    }

    #[test]
    fn h_b_baseline_spec_pinned_to_serial() {
        // #105 H-B — the serial DENOMINATOR is not CLI-steerable: a serial baseline passes, any
        // non-serial baseline is a hard error (pre-GPU), so a caller can't swap the denominator.
        assert!(validate_baseline_is_serial(&SpecConfig::serial()).is_ok());
        let e = validate_baseline_is_serial(&SpecConfig::mtp(2)).unwrap_err();
        assert!(e.contains("must be {\"mode\":\"serial\"}"), "{e}");
        assert!(validate_baseline_is_serial(&SpecConfig {
            mode: "dflash".to_string(),
            mtp: None,
            dflash: Some(serde_json::json!({})),
            dspark: None,
        })
        .is_err());
    }

    #[test]
    fn david_spec_depth_cap_is_submission_proof() {
        // David ruling — the 32 cap re-homed onto the module's mtp.depth (docs/spec-config-design.md
        // step 4). OFFICIAL uses the readonly 32 constant and IGNORES the env; local-dev honors it.
        assert_eq!(
            resolve_max_draft_depth_cap(false, None),
            32,
            "official default cap"
        );
        assert_eq!(
            resolve_max_draft_depth_cap(false, Some("999")),
            32,
            "official IGNORES MLXFAST_MAX_DRAFT_DEPTH (submission-proof)"
        );
        assert_eq!(
            resolve_max_draft_depth_cap(true, None),
            32,
            "local default cap when env unset"
        );
        assert_eq!(
            resolve_max_draft_depth_cap(true, Some("999")),
            999,
            "local honors the override"
        );
        assert_eq!(
            resolve_max_draft_depth_cap(true, Some("")),
            32,
            "local blank → constant"
        );
        assert_eq!(
            resolve_max_draft_depth_cap(true, Some("garbage")),
            32,
            "local non-numeric → constant"
        );

        // OFFICIAL: mtp.depth 33 rejected against the readonly 32; MLXFAST_MAX_DRAFT_DEPTH=999 ignored.
        let official_cap = resolve_max_draft_depth_cap(false, Some("999"));
        assert!(validate_spec_capped(&SpecConfig::mtp(33), official_cap).is_err());
        assert!(
            validate_spec_capped(&SpecConfig::mtp(32), official_cap).is_ok(),
            "cap value accepted"
        );
        let e = validate_spec_capped(&SpecConfig::mtp(33), official_cap).unwrap_err();
        assert!(e.contains("exceeds the maximum draft depth cap 32"), "{e}");
        // The cap applies ONLY to mtp.depth — a serial spec has no module depth, so it is never capped.
        assert!(validate_spec_capped(&SpecConfig::serial(), official_cap).is_ok());

        // LOCAL: with MLXFAST_MAX_DRAFT_DEPTH=999, mtp.depth 33 is honored.
        let local_cap = resolve_max_draft_depth_cap(true, Some("999"));
        assert!(validate_spec_capped(&SpecConfig::mtp(33), local_cap).is_ok());
    }

    #[test]
    fn r13_exactness_probe_parse_and_default() {
        // R13 — the four modes parse; default is `once`; an unknown value is a usage error.
        assert_eq!(ExactnessProbe::parse("none").unwrap(), ExactnessProbe::None);
        assert_eq!(ExactnessProbe::parse("once").unwrap(), ExactnessProbe::Once);
        assert_eq!(
            ExactnessProbe::parse("per-prompt").unwrap(),
            ExactnessProbe::PerPrompt
        );
        assert_eq!(
            ExactnessProbe::parse("per-pair").unwrap(),
            ExactnessProbe::PerPair
        );
        assert!(ExactnessProbe::parse("sometimes").is_err());
        assert_eq!(ExactnessProbe::default(), ExactnessProbe::Once);
        assert_eq!(ExactnessProbe::Once.as_str(), "once");
        assert_eq!(ExactnessProbe::PerPair.as_str(), "per-pair");
    }

    #[test]
    fn r13_validate_prompt_sha256_64_lowercase_hex() {
        // R13 — exactly 64 lowercase hex chars.
        assert!(validate_prompt_sha256(&"a".repeat(64)).is_ok());
        assert!(validate_prompt_sha256(&"0123456789abcdef".repeat(4)).is_ok());
        assert!(
            validate_prompt_sha256(&"A".repeat(64)).is_err(),
            "uppercase rejected"
        );
        assert!(
            validate_prompt_sha256(&"a".repeat(63)).is_err(),
            "too short"
        );
        assert!(validate_prompt_sha256(&"a".repeat(65)).is_err(), "too long");
        assert!(
            validate_prompt_sha256(&"g".repeat(64)).is_err(),
            "non-hex rejected"
        );
    }

    #[test]
    fn r13_validate_target_id_charset() {
        // R13 — [A-Za-z0-9._-]+ (non-empty).
        assert!(validate_target_id("qwen3.8-27b_mtp-v1").is_ok());
        assert!(validate_target_id("A._-9").is_ok());
        assert!(validate_target_id("").is_err(), "empty rejected");
        assert!(validate_target_id("has space").is_err());
        assert!(validate_target_id("slash/here").is_err());
    }

    #[test]
    fn r13_check_golden_digests_dup_is_fatal() {
        // R13 — duplicate sha256 among --goldens is a die-8 hard error; distinct is fine.
        assert!(check_golden_digests(&["aa".to_string(), "bb".to_string()]).is_ok());
        assert!(check_golden_digests(&[]).is_ok());
        assert!(check_golden_digests(&["aa".to_string()]).is_ok());
        let err = check_golden_digests(&["aa".to_string(), "bb".to_string(), "aa".to_string()])
            .unwrap_err();
        assert!(err.contains("duplicate --golden digest"), "err={err}");
        assert!(err.contains("die 8"), "dup-digest is die-8 style: {err}");
    }

    #[test]
    fn r16_seals_full_top_level_aggregate_and_per_prompt_schema() {
        // R16 — a pool run seals the COMPLETE live results.json shape with the EXACT field names.
        // A 3-golden pool so the medians/vectors are non-degenerate.
        let goldens = vec![
            measure_golden_with("case-a", oracle_decode_tokens()),
            measure_golden_with("case-b", oracle_decode_tokens()),
            measure_golden_with("case-c", oracle_decode_tokens()),
        ];
        let mut cfg = test_cfg(2, 3);
        cfg.prompt_pool = vec![
            PromptPoolEntry {
                sha256: goldens[0].sha256().to_string(),
                bytes: Some(goldens[0].byte_len()),
                noop_decode_speedup: Some(0.99),
            },
            PromptPoolEntry {
                sha256: goldens[1].sha256().to_string(),
                bytes: Some(goldens[1].byte_len()),
                noop_decode_speedup: Some(0.99),
            },
            PromptPoolEntry {
                sha256: goldens[2].sha256().to_string(),
                bytes: Some(goldens[2].byte_len()),
                noop_decode_speedup: Some(0.99),
            },
        ];
        let out = run_measure_job(
            &goldens,
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            |_p| ok_candidate(),
        )
        .unwrap();
        let v = serde_json::to_value(&out.results).unwrap();

        // --- top-level new seals ---
        assert_eq!(
            v["pairs_per_prompt"],
            json!(3),
            "pairs_per_prompt == target_pairs"
        );
        assert_eq!(
            v["min_pairs_per_prompt"],
            json!(2),
            "min_pairs_per_prompt == min_pairs"
        );
        assert_eq!(
            v["min_pairs"],
            json!(2),
            "R2 min_pairs kept alongside the live-named field"
        );
        // evaluation_target defaults honestly when the R13 trio is absent.
        let et = &v["evaluation_target"];
        assert_eq!(
            et["target_id"],
            json!("default"),
            "no --target-id → honest default marker"
        );
        assert_eq!(
            et["explicit_prompt"],
            json!(false),
            "no explicit prompt supplied"
        );
        assert!(
            et.get("prompt_sha256").is_none(),
            "no fabricated prompt sha on a default-pool run"
        );

        // --- aggregate: three DISTINCT medians / floors (NAME-TRAPS) ---
        let agg = &v["aggregate"];
        // The PUBLISHED score is the even-n median over the per-prompt raw ratios.
        let ratios: Vec<f64> = out
            .results
            .per_prompt
            .iter()
            .map(|p| p.raw_ratio_of_means)
            .collect();
        let published = bench_core::score::paired_decode_only_median(&ratios);
        assert_eq!(
            agg["raw_decode_speedup_median"].as_f64().unwrap(),
            published,
            "PUBLISHED even-n median"
        );
        // The per-PAIR LOWER-median is a DISTINCT diagnostic field with a DISTINCT rule.
        let per_pair: Vec<f64> = out.results.pairs.iter().map(|p| p.raw_ratio).collect();
        assert_eq!(
            agg["mtp_decode_speedup_median"].as_f64().unwrap(),
            lower_median(&per_pair),
            "per-pair LOWER-median"
        );
        // The SANITY floor is 0.50 (MIN_ACCEPTED_SPEEDUP), NOT the ranked 0.90. #117 — `test_cfg` is
        // the TEACHER-FORCED regime, which is the one this value is scoped to; the free-run seal is
        // covered by `issue117_free_run_seals_the_ruled_090_floor`.
        assert_eq!(
            agg["decode_speedup_floor"].as_f64().unwrap(),
            0.50,
            "loose SANITY floor 0.50, not 0.90"
        );
        assert_ne!(
            agg["decode_speedup_floor"].as_f64().unwrap(),
            0.90,
            "NAME-TRAP: not the ranked 0.90 floor"
        );
        assert_eq!(
            agg["decode_speedup_floor_met"],
            json!(true),
            "candidate (2x) clears the sanity floor"
        );
        // pooled sanity speedup (NOT the score).
        assert_eq!(
            agg["mtp_decode_speedup"].as_f64().unwrap(),
            SERIAL_SPT / CANDIDATE_SPT
        );
        assert!(agg["mtp_decode_speedup_min"].as_f64().unwrap() > 0.0);
        // verbatim aggregation strings.
        assert_eq!(agg["aggregation"], json!("ratio_of_means"));
        assert_eq!(agg["score_anchor"], json!("serial = 1.0"));
        assert_eq!(
            agg["scoring_aggregation"],
            json!("median_of_per_prompt_raw_serial_relative_speedup")
        );
        assert_eq!(
            agg["median_rule"],
            json!("even_n_mean_of_two_central_order_statistics")
        );
        assert_eq!(agg["prefill_component"], json!("none"));
        assert_eq!(
            agg["mtp_max_draft_depth"],
            json!(8),
            "sealed max draft depth 8"
        );
        // This is an MTP-arm run (serial teacher-forced echo), so the DFlash-specific depth field is
        // OMITTED entirely — it belongs only to a dflash run
        // (`dflash_run_seals_dflash_depth_and_drops_the_mtp_field`).
        assert!(
            agg.get("dflash_effective_depth").is_none(),
            "no dflash depth field on an MTP-arm run"
        );
        assert_eq!(agg["published_speedup_ceiling"].as_f64().unwrap(), 5.0);
        // per-prompt-sourced vectors (pool order, length pool_size).
        assert_eq!(agg["raw_ratios"].as_array().unwrap().len(), 3);
        assert_eq!(
            agg["effective_mean_draft_len_by_prompt"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        // #109 window-2 finding 3 — this is a TEACHER-FORCED run, so its draft statistics are 0 by
        // regime (benchd's histogram math is free-run-only, and the TF report echo that used to
        // supply a non-zero value here is retired with the `--mtp-report` file). The vector shape is
        // unchanged; only the honest value is.
        assert_eq!(
            agg["effective_mean_draft_len_by_prompt"][0]
                .as_f64()
                .unwrap(),
            0.0
        );
        // #109 W3 finding 5 — and for the same regime reason, a TEACHER-FORCED run seals NO head
        // identity: both legs were spawned gate-off, and the engine gates `head_provenance` behind
        // the v1.1 flag. The vector is present and EMPTY (omitted per prompt, never blank-filled);
        // its populated counterpart is the free-run series
        // (`r15_per_side_heads_seal_candidate_head_provenance`).
        assert_eq!(
            agg["head_provenance_sha256_by_prompt"]
                .as_array()
                .unwrap()
                .len(),
            0
        );
        assert_eq!(agg["non_drafting_round_count_total"], json!(0));
        // retired informational normalized diagnostics.
        assert_eq!(
            agg["normalized_ratios_informational"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert!(agg
            .get("normalized_decode_speedup_median_informational")
            .is_some());
        let want_norm = (SERIAL_SPT / CANDIDATE_SPT) / 0.99;
        assert!(
            (agg["normalized_ratios_informational"][0].as_f64().unwrap() - want_norm).abs() < 1e-9
        );

        // --- per_prompt new seals ---
        let pp = &v["per_prompt"][0];
        assert!(
            (pp["normalized_ratio"].as_f64().unwrap() - want_norm).abs() < 1e-9,
            "informational normalized ratio"
        );
        // TF run ⇒ zero drafting, sealed as a fact about the regime (see above).
        assert_eq!(pp["effective_mean_draft_len"].as_f64().unwrap(), 0.0);
        assert_eq!(pp["non_drafting_round_count"], json!(0));

        // --- telemetry sealed from the observed samples ---
        let tel = &v["telemetry"];
        assert_eq!(
            tel["max_gpu_temp"].as_f64().unwrap(),
            CANDIDATE_GPU_TEMP_C,
            "max temp across legs"
        );
        assert_eq!(
            tel["min_steady_freq_mhz"].as_f64().unwrap(),
            CANDIDATE_STEADY_FREQ_MHZ,
            "min steady freq across legs"
        );
    }

    #[test]
    fn r16_evaluation_target_reflects_the_r13_trio_when_supplied() {
        // R16 — when the R13 --prompt/--prompt-sha256/--target-id trio is supplied, evaluation_target
        // reflects it (explicit_prompt true, the pinned sha + target-id sealed).
        let mut cfg = test_cfg(1, 1);
        cfg.target_id = Some("qwen3.8-27b-mtp-v1".to_string());
        cfg.prompt_sha256 = Some("a".repeat(64));
        let out = identity_run(&cfg);
        let et = &serde_json::to_value(&out.results).unwrap()["evaluation_target"];
        assert_eq!(et["target_id"], json!("qwen3.8-27b-mtp-v1"));
        assert_eq!(et["explicit_prompt"], json!(true));
        assert_eq!(et["prompt_sha256"], json!("a".repeat(64)));
    }

    #[test]
    fn r16_telemetry_always_sealed_null_when_no_sample_observed() {
        // R16 (medium cycle-3) — with NO telemetry sample on either leg, the top-level `telemetry`
        // OBJECT is STILL sealed (matching the live shape), with honest `null` fields — never a
        // silently-dropped key and never a fabricated number. The run still accepts.
        let cfg = test_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| Ok(inv(echo(SERIAL_SPT), GateState::Fired, None)),
            |_p| Ok(inv(echo(CANDIDATE_SPT), GateState::Fired, None)),
        )
        .unwrap();
        assert!(out.candidate_accepted);
        let v = serde_json::to_value(&out.results).unwrap();
        let tel = v
            .get("telemetry")
            .expect("telemetry object ALWAYS sealed (live shape)");
        assert!(
            tel.is_object(),
            "telemetry is the {{max_gpu_temp, min_steady_freq_mhz}} object"
        );
        assert!(
            tel.get("max_gpu_temp").unwrap().is_null(),
            "no sample → honest null, not fabricated"
        );
        assert!(
            tel.get("min_steady_freq_mhz").unwrap().is_null(),
            "no sample → honest null"
        );
    }

    #[test]
    fn medium_r16_field_gaps_sealed_against_live_shape() {
        // Medium (cycle-3) — the previously-missing R16 keys are now sealed, cross-checked against
        // the live seal (W:1864-1936): top-level timestamp / candidate{} / baseline{} / decode_tokens
        // / telemetry-always / timed_regime; per-pair speedup / *_attempts / prompt_index / prompt_sha256.
        let out = identity_run(&test_cfg(1, 1));
        let v = serde_json::to_value(&out.results).unwrap();

        // Top-level additions.
        assert_eq!(
            v["timestamp"], "2026-08-19T00:00:00Z",
            "sealed run timestamp (date -u)"
        );
        assert_eq!(
            v["decode_tokens"], out.results.decode_tokens,
            "top-level decode_tokens"
        );
        assert_eq!(
            v["timed_regime"], "tf-serial-timed",
            "TRUTHFUL timed_regime present when a timed measurement ran"
        );
        assert_eq!(
            v["timed_mode"], "teacher_forced_v1",
            "the sealed series tag (#105 H-A)"
        );
        assert_eq!(
            v["candidate"]["verdict"], "ACCEPT",
            "candidate block verdict"
        );
        assert!(
            v["candidate"]["workspace"].is_string(),
            "candidate block workspace"
        );
        assert_eq!(v["baseline"]["verdict"], "ACCEPT", "baseline block verdict");
        assert!(
            v["baseline"]["workspace"].is_string(),
            "baseline block workspace"
        );
        assert!(
            v.get("telemetry").is_some(),
            "telemetry object always sealed"
        );

        // Per-pair live field NAMES.
        let p0 = &v["pairs"][0];
        assert!(p0.get("speedup").is_some(), "pairs[].speedup (live name)");
        assert_eq!(p0["speedup"], p0["raw_ratio"], "speedup == raw_ratio value");
        assert_eq!(
            p0["serial_attempts"], 1,
            "pairs[].serial_attempts (real count)"
        );
        assert_eq!(p0["mtp_attempts"], 1, "pairs[].mtp_attempts (real count)");
        assert_eq!(p0["prompt_index"], 0, "pairs[].prompt_index");
        assert_eq!(
            p0["prompt_sha256"], out.results.per_prompt[0].prompt_sha256,
            "pairs[].prompt_sha256"
        );
    }

    #[test]
    fn medium_decode_speedup_floor_met_is_pooled_not_per_pair_min() {
        // Medium (cycle-3) — `decode_speedup_floor_met` is the POOLED raw ratio >= 0.50 (W:2204/2256),
        // NOT the per-pair minimum. Construct a run whose per-pair MIN ratio is BELOW the floor but
        // whose POOLED ratio is ABOVE it: the old per-pair-min rule would have sealed false; the
        // pooled rule seals true.
        let serial_spt = 1.0_f64; // constant serial denominator
        let candidate_calls = Cell::new(0usize);
        let measure_candidate = move |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
            let n = candidate_calls.get();
            candidate_calls.set(n + 1);
            // pair 0: mtp slow (spt 3.0 ⇒ ratio 1/3 < 0.50); pair 1: mtp fast (spt 0.4 ⇒ ratio 2.5).
            let spt = if n == 0 { 3.0 } else { 0.4 };
            Ok(inv(echo(spt), GateState::Fired, None))
        };
        let cfg = test_cfg(1, 2); // one prompt, two pairs
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            move |_p| Ok(inv(echo(serial_spt), GateState::Fired, None)),
            measure_candidate,
        )
        .unwrap();
        assert!(out.candidate_accepted);
        let agg = &out.results.aggregate;
        // Per-pair MIN ratio is below the 0.50 sanity floor ...
        assert!(
            agg.mtp_decode_speedup_min < DECODE_SPEEDUP_FLOOR,
            "min per-pair ratio below floor"
        );
        // ... but the POOLED ratio is above it ...
        assert!(
            agg.mtp_decode_speedup >= DECODE_SPEEDUP_FLOOR,
            "pooled ratio above floor"
        );
        // ... so floor_met is TRUE (pooled semantic), not FALSE (per-pair-min semantic).
        assert!(
            agg.decode_speedup_floor_met,
            "floor_met uses the POOLED ratio, not the per-pair min"
        );
    }

    #[test]
    fn issue117_free_run_seals_the_ruled_090_floor() {
        // #117 — the RULED floor is ENCODED in the seal. David, #109 comment 5353123259: "floor
        // stays 0.90, no sub-floor bootstrap governance built". Window 4's free-run legs sealed
        // `aggregate.decode_speedup_floor = 0.5` (the wrapper's teacher-forced sanity value) while
        // the gate everyone quoted was 0.90; a free-run run now seals the floor it is actually
        // judged by, and its verdict is computed against THAT number.
        let n = BENCHMARK_DECODE_STEPS;
        let cfg = free_run_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_free_run_serial(),
            move |_p: &TimingParams| inv_free_run(CANDIDATE_SPT, vec![4; n / 4], n),
        )
        .unwrap();
        assert!(out.candidate_accepted);
        assert_eq!(
            out.results.timed_mode, "free_run_v1_1",
            "the ruling is scoped to this series"
        );
        let v = serde_json::to_value(&out.results).unwrap();
        let agg = &v["aggregate"];
        assert_eq!(
            agg["decode_speedup_floor"].as_f64().unwrap(),
            0.90,
            "the SEALED free-run floor is the ruled 0.90, not the 0.5 window 4 carried"
        );
        assert_eq!(
            agg["decode_speedup_floor"].as_f64().unwrap(),
            FREE_RUN_DECODE_SPEEDUP_FLOOR,
            "and it is the ONE constant the ranked overlay floors with, not a second copy"
        );
        // The 2x candidate clears it; the verdict is sealed alongside the floor it used.
        assert_eq!(agg["decode_speedup_floor_met"], json!(true));
    }

    #[test]
    fn issue117_free_run_median_below_the_ruled_floor_fails_closed() {
        // #117 — the boundary. A free-run median just BELOW 0.90 must seal `floor_met: false`; the
        // same run passed under the old 0.50 sanity value, which is precisely the drift the issue
        // reports. Serial 0.040 s/tok against a candidate at 0.040 / 0.8999 s/tok ⇒ a single-prompt
        // median of 0.8999 — below the floor, comfortably above 0.50.
        let n = BENCHMARK_DECODE_STEPS;
        let below = 0.8999_f64;
        let cfg = free_run_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_free_run_serial(),
            move |_p: &TimingParams| inv_free_run(SERIAL_SPT / below, vec![4; n / 4], n),
        )
        .unwrap();
        let agg = &out.results.aggregate;
        assert!(
            agg.raw_decode_speedup_median < FREE_RUN_DECODE_SPEEDUP_FLOOR,
            "the median is below the ruled floor: {}",
            agg.raw_decode_speedup_median
        );
        assert!(
            agg.raw_decode_speedup_median > DECODE_SPEEDUP_FLOOR,
            "and above the old 0.50 sanity value, so only the RULED floor can catch it"
        );
        assert_eq!(agg.decode_speedup_floor, FREE_RUN_DECODE_SPEEDUP_FLOOR);
        assert!(
            !agg.decode_speedup_floor_met,
            "a sub-0.90 free-run median FAILS CLOSED instead of passing at 0.50"
        );
    }

    #[test]
    fn issue117_free_run_median_at_or_above_the_ruled_floor_passes() {
        // #117 — the other side of the boundary. EXACTLY at the floor is INSIDE it: the gate is
        // `>= floor`, the same sense as `bench_core::score::score_paired_decode_only`'s `< floor`
        // refusal, so the two surfaces agree on the boundary run rather than splitting it. Pinned on
        // the pure decision (a measured ratio cannot be steered onto 0.90 bit-exactly), then driven
        // end-to-end just above it.
        let floor = FREE_RUN_DECODE_SPEEDUP_FLOOR;
        assert!(
            decode_speedup_floor_verdict(LegRegime::FreeRunV1_1, floor, floor).1,
            "a median ON the floor MEETS it"
        );
        // The TRUE next-representable-below, via the bit pattern. NOT `floor - f64::EPSILON`:
        // `f64::EPSILON` is the ulp at 1.0, and 0.90 sits in the next binade DOWN, so its ulp is
        // HALF that — subtracting EPSILON would step 2 ulps and stop testing the boundary.
        let one_ulp_below = f64::from_bits(floor.to_bits() - 1);
        assert!(
            !decode_speedup_floor_verdict(LegRegime::FreeRunV1_1, floor, one_ulp_below).1,
            "one ulp below the floor does not"
        );
        let n = BENCHMARK_DECODE_STEPS;
        let cfg = free_run_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_free_run_serial(),
            move |_p: &TimingParams| inv_free_run(SERIAL_SPT / 0.905, vec![4; n / 4], n),
        )
        .unwrap();
        let agg = &out.results.aggregate;
        assert!(
            agg.raw_decode_speedup_median >= floor,
            "median {} clears the floor",
            agg.raw_decode_speedup_median
        );
        assert!(
            agg.decode_speedup_floor_met,
            "a median above the ruled floor still passes"
        );
    }

    #[test]
    fn issue117_the_ruled_floor_is_scoped_to_the_free_run_series() {
        // #117 — SCOPE. #109 comment 5350423826: the 0.90 floor's justification is series-specific
        // and §5 forbids inheriting it across series. The teacher-forced seal therefore keeps the
        // live wrapper's loose `MIN_ACCEPTED_SPEEDUP` on the POOLED ratio, unchanged, and only the
        // free-run series moves to the ruled 0.90 on the PUBLISHED median. Driven on the pure
        // decision function so BOTH regimes and BOTH subjects are pinned in one place.
        let (tf_floor, tf_met) =
            decode_speedup_floor_verdict(LegRegime::TeacherForcedV1, 0.60, 0.60);
        assert_eq!(
            tf_floor, DECODE_SPEEDUP_FLOOR,
            "TF keeps the 0.50 wrapper sanity floor"
        );
        assert!(
            tf_met,
            "0.60 pooled clears 0.50: the ruled 0.90 is NOT inherited into TF"
        );
        let (fr_floor, fr_met) = decode_speedup_floor_verdict(LegRegime::FreeRunV1_1, 0.60, 0.60);
        assert_eq!(
            fr_floor, FREE_RUN_DECODE_SPEEDUP_FLOOR,
            "free-run seals the ruled 0.90"
        );
        assert!(
            !fr_met,
            "the SAME numbers fail closed under the ruled floor"
        );
        // The SUBJECT differs per series, and each verdict reads only its own: a TF run is judged on
        // the pooled ratio-of-means, a free-run run on the published median.
        assert!(
            decode_speedup_floor_verdict(LegRegime::TeacherForcedV1, 0.60, 0.10).1,
            "TF ignores the median: its subject is the pooled ratio (W:2204/2256)"
        );
        assert!(
            decode_speedup_floor_verdict(LegRegime::FreeRunV1_1, 0.10, 1.50).1,
            "free-run ignores the pooled ratio: its subject is the published median"
        );
    }

    #[test]
    fn w2f3_teacher_forced_pairs_seal_zero_drafting_not_an_engine_echo() {
        // #109 window-2 finding 3 — R16's `r16_candidate_report_missing_draft_echo_fails_closed`
        // required the TF candidate leg to ECHO `effective_mean_draft_len` /
        // `non_drafting_round_count` through the `--mtp-report` file, and rejected the pair when it
        // did not. That requirement is RETIRED with the file: the generic `runtime-worker` verb writes
        // no report, so no TF leg could ever satisfy it, and the numbers it demanded describe drafting
        // that `tf_regime_is_serial` already proves cannot have happened under teacher forcing.
        //
        // What replaces it: a TF pair seals ZERO drafting as a fact about the regime, and benchd's own
        // histogram math is the only draft-statistics source anywhere (free-run only — covered by
        // `free_run_positive_control_seals_series_and_audit`, whose closing assertions read the
        // computed `effective_mean_draft_len`/`non_drafting_round_count` off benchd's own histogram).
        let out = identity_run(&test_cfg(1, 1));
        assert!(
            out.candidate_accepted,
            "a TF pair accepts without any draft echo"
        );
        assert_eq!(
            out.results.pairs[0].effective_mean_draft_len, 0.0,
            "teacher forcing feeds every token: no round can draft"
        );
        assert_eq!(out.results.pairs[0].non_drafting_round_count, 0);
        assert_eq!(
            out.results.per_prompt[0].effective_mean_draft_len,
            Some(0.0)
        );
        assert_eq!(out.results.per_prompt[0].non_drafting_round_count, Some(0));
    }

    #[test]
    fn seal_omits_track_name_when_absent() {
        // R12 — track_name is optional: OMITTED from the seal (not null) when unavailable.
        let out = identity_run(&test_cfg(1, 1)); // test_cfg sets track_name = None
        let v = serde_json::to_value(&out.results).unwrap();
        assert!(
            v.get("track_name").is_none(),
            "absent track_name is omitted, not null"
        );
        // The mode is still sealed to the live value.
        assert_eq!(v["mode"], json!("qwen-native-mtp-paired-decode-only"));
    }

    // -----------------------------------------------------------------------
    // W3 — the SCORED v1.1 free-run leg, driven END-TO-END from a real MockEngine
    //
    // These are INTEGRATION controls, not closure-seam fakes: each doctored response
    // travels mock engine -> runner (oracle exact-match / §2.6 triple / §2.2 RunTimeout)
    // -> measure-job classification -> die-5, so the whole chain is under test.
    //
    // Fable ruling (same-series serial control): BOTH legs are engine-driven free-run legs.
    // The candidate free-runs its declared mtp spec; the SERIAL CONTROL free-runs the serial
    // spec at depth 0 and commits `[1]*N` (the engine's non-speculating path). Same verb,
    // same N, same params — so the ratio divides one measured quantity.
    // -----------------------------------------------------------------------

    use bench_runner::mock::MockEngine;
    use bench_runner::Session;

    /// A conformant v1.1 engine for the free-run window: advertises `free_run_decode`, returns the
    /// golden seed + the golden continuation as its committed stream, and reports a histogram of
    /// `rounds` rounds committing `per_round` tokens each (sum == N, completed_work == R + 1).
    fn free_run_mock(oracle: &[i64], per_round: u32, rounds: usize) -> MockEngine {
        MockEngine::new()
            .free_run_capable()
            .oracle_tokens(PREFILL_TOKEN, SEED_TOKEN, oracle.to_vec())
            .free_run_acceptance_lengths(vec![per_round; rounds])
    }

    /// W3 — the ENGINE-LEVEL leg seam, mirroring `main.rs`'s free-run branch: spawn a fresh mock,
    /// run the SCORED free-run phase (benchd's own parent clock, the §2.6 triple at the barrier),
    /// and wrap the outcome as the `LegInvocation` the pair loop consumes. `build` is called per
    /// attempt so the one gated retry gets a genuinely fresh engine.
    fn free_run_leg_with_spec<B>(
        build: B,
        leg_spec: SpecConfig,
        params: &TimingParams,
    ) -> bench_runner::Result<LegInvocation>
    where
        B: Fn() -> MockEngine,
    {
        // #109 window-2 finding 3 — capture the hello's `head_provenance` exactly as main.rs now
        // does: the wire IS the head-identity channel, so this seam proves the plumbing end to end
        // (mock hello → `Hello::head_provenance` → `LegInvocation` → the sealed pair record) rather
        // than hand-supplying a head the engine never echoed.
        let wire_head_provenance = std::cell::RefCell::new(None);
        let mut spawn = || -> bench_runner::Result<Session<MockEngine>> {
            let (session, hello) = Session::connect(build())?;
            *wire_head_provenance.borrow_mut() = hello.head_provenance.clone();
            Ok(session)
        };
        let mut gate = |_phase: &str| -> bench_runner::Result<()> { Ok(()) };
        // Mirror main.rs: each leg carries ITS OWN wire spec on the window (the candidate's declared
        // speculating spec; the control's serial spec), so the runner's spec-never-ignored check has
        // an echo to validate and benchd has one to seal.
        let params = params.clone().with_spec(Some(leg_spec));
        let timing = bench_runner::run_free_run_decode_phase_fresh(&mut spawn, &mut gate, &params)?;
        Ok(LegInvocation {
            // H1 — benchd's OWN parent clock is the only scored number, free-run included.
            benchd_seconds_per_token: timing.seconds_per_token,
            // The head the mock's hello actually echoed. A free-run leg needs no draft echoes at all
            // — benchd computes those from the histogram it collected.
            wire_head_provenance: wire_head_provenance.into_inner(),
            gate_state: GateState::Fired,
            quiesce: QuiesceRecord::default(),
            telemetry: candidate_telemetry(),
            wire_effective_spec: timing.effective_spec,
            regime: LegRegime::FreeRunV1_1,
            free_run_audit: Some(timing.audit),
            // B2 — the SPLIT the runner measured on its own clock, threaded exactly as main.rs
            // threads it: this seam proves the plumbing end to end rather than hand-supplying a
            // window the runner never produced.
            phase_window: Some(timing.phase_window),
            cohort_audit: None,
            cohort_phase_windows: None,
            per_stream_timing: None,
            cohort_committed_tokens_by_stream: None,
        })
    }

    /// The CANDIDATE free-run leg: the declared speculating spec on the wire.
    fn free_run_leg<B>(build: B, params: &TimingParams) -> bench_runner::Result<LegInvocation>
    where
        B: Fn() -> MockEngine,
    {
        free_run_leg_with_spec(build, SpecConfig::mtp(FREE_RUN_DEPTH), params)
    }

    /// Fable ruling — the SAME-SERIES SERIAL CONTROL leg: the same free-run verb and the same N as
    /// the candidate, but the SERIAL wire spec at depth 0, so the engine takes its non-speculating
    /// path and commits exactly one token per verify round (`[1]*N`).
    fn free_run_serial_leg(params: &TimingParams) -> bench_runner::Result<LegInvocation> {
        let oracle = oracle_decode_tokens();
        let n = BENCHMARK_DECODE_STEPS;
        let mut inv = free_run_leg_with_spec(
            move || free_run_mock(&oracle, 1, n),
            timed_decode_wire_spec(),
            params,
        )?;
        // The control loads the PINNED head, like main.rs's serial plan. #109 window-2 finding 3 —
        // the head now arrives on the hello, and `MockEngine` echoes one fixed head for every spawn,
        // so this seam overrides the captured sha to keep the two legs distinguishable. (The head the
        // engine ECHOES is still the only thing benchd seals — the closure seam's
        // `r15_per_side_heads_seal_candidate_head_provenance` is what proves the candidate's head, not
        // the control's, reaches the record.)
        inv.wire_head_provenance = head_prov(SERIAL_HEAD_SHA);
        inv.telemetry = serial_telemetry();
        Ok(inv)
    }

    /// Run one free-run measure-job over a single golden: the candidate leg driven by `build`, the
    /// serial control by the ruling's depth-0 free-run leg.
    fn free_run_run<B>(cfg: &MeasureJobConfig, build: B) -> MeasureJobOutcome
    where
        B: Fn() -> MockEngine + Copy,
    {
        run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            cfg,
            free_run_serial_leg,
            move |p: &TimingParams| free_run_leg(build, p),
        )
        .unwrap()
    }

    #[test]
    fn free_run_positive_control_seals_series_and_audit() {
        // The scored path end-to-end: a conformant v1.1 engine free-runs the golden continuation,
        // the pair is accepted, and the seal describes BOTH regimes honestly.
        let oracle = oracle_decode_tokens();
        let n = BENCHMARK_DECODE_STEPS;
        let cfg = free_run_cfg(1, 1);
        // 32 rounds committing 4 tokens each = 128 = N.
        let out = free_run_run(&cfg, || free_run_mock(&oracle, 4, n / 4));
        assert!(
            out.candidate_accepted,
            "a conformant free-run candidate is accepted"
        );
        assert_eq!(out.results.accepted_pair_count, 1);

        // --- the SERIES DESCRIPTOR (item 3, under the Fable same-series ruling) ---
        assert_eq!(
            out.results.timed_mode, "free_run_v1_1",
            "both legs ran the free-run series, so the run seals that ONE tag"
        );
        let series = &out.results.timed_series;
        assert_eq!(series.serial_leg_timed_mode, "free_run_v1_1");
        assert_eq!(series.candidate_leg_timed_mode, "free_run_v1_1");
        assert_eq!(series.serial_leg_timed_regime, FREE_RUN_TIMED_REGIME);
        assert_eq!(series.candidate_leg_timed_regime, FREE_RUN_TIMED_REGIME);
        assert!(
            series.homogeneous,
            "the ruling makes a scored free-run run homogeneous"
        );
        assert!(
            series.legs_comparable,
            "§5: same series on both legs, so the ratio divides one measured quantity"
        );
        assert_eq!(
            out.results.timed_regime,
            Some(FREE_RUN_TIMED_REGIME),
            "one regime ran on both legs, so the top-level label is truthful"
        );
        assert!(
            out.results.tf_downgrade_note.is_none(),
            "nothing was downgraded: the free-run candidate ran its declared mtp spec on the wire"
        );
        // Per-leg tags on the pair record.
        let pair = &out.results.pairs[0];
        assert_eq!(pair.serial_timed_mode, "free_run_v1_1");
        assert_eq!(pair.candidate_timed_mode, "free_run_v1_1");
        assert_eq!(
            pair.serial_effective_spec.mode.as_str(),
            SPEC_MODE_SERIAL,
            "the control free-ran at depth 0: its echoed effective regime is serial"
        );

        // COORDINATOR RULING (#109, leg B) — the free-run side is UNCHANGED by the ruling: the gate
        // is on, a spec was requested, the echo was required and enforced, and BOTH legs' effective
        // specs are sealed as MEASURED off the wire. The ruling's `gate-off-v1-spawn` provenance is
        // the TF path's alone and must never appear on a free-run record.
        assert_eq!(
            pair.serial_effective_spec_source,
            EFFECTIVE_SPEC_SOURCE_WIRE_ECHO
        );
        assert_eq!(
            pair.candidate_effective_spec_source,
            EFFECTIVE_SPEC_SOURCE_WIRE_ECHO
        );
        assert_ne!(
            pair.candidate_effective_spec_source,
            EFFECTIVE_SPEC_SOURCE_GATE_OFF_V1_SPAWN
        );

        // --- the §3 AUDIT, persisted per leg ---
        let lengths = pair
            .audit_spec_acceptance_lengths
            .as_ref()
            .expect("the free-run candidate persists its per-round histogram verbatim");
        assert_eq!(lengths.len(), n / 4, "R = 32 rounds");
        assert_eq!(
            lengths.iter().map(|&x| x as usize).sum::<usize>(),
            n,
            "sum == N"
        );
        assert_eq!(
            pair.audit_spec.get("audit_spec_verified_token_count"),
            Some(&(n as f64))
        );
        assert_eq!(
            pair.audit_spec.get("audit_spec_rounds"),
            Some(&((n / 4) as f64))
        );
        assert_eq!(
            pair.audit_spec.get("audit_spec_mean_acceptance_length"),
            Some(&4.0)
        );
        assert_eq!(
            pair.audit_spec
                .get("audit_spec_effective_tokens_per_forward"),
            Some(&4.0)
        );

        // The sealed JSON carries the audit family FLAT (RULED OQ4 — no nested object).
        let v: serde_json::Value =
            serde_json::from_str(&out.results.to_sealed_json().unwrap()).unwrap();
        let p0 = &v["pairs"][0];
        assert_eq!(p0["audit_spec_rounds"], json!((n / 4) as f64));
        assert_eq!(
            p0["audit_spec_acceptance_lengths"]
                .as_array()
                .unwrap()
                .len(),
            n / 4
        );
        assert!(
            p0.get("audit").is_none(),
            "flat prefix, never a nested `audit` object"
        );
        assert_eq!(v["timed_mode"], json!("free_run_v1_1"));
        assert_eq!(v["timed_series"]["legs_comparable"], json!(true));
        assert_eq!(v["timed_series"]["homogeneous"], json!(true));

        // benchd COMPUTES the draft stats from its own histogram (4 committed per round, no round
        // fell back to a bare base-model token) — NOT from the engine's echo, which is absent here.
        let pp = &out.results.per_prompt[0];
        assert_eq!(pp.effective_mean_draft_len, Some(4.0));
        assert_eq!(pp.non_drafting_round_count, Some(0));
    }

    /// B2 — PREFILL-WINDOW CERTIFICATION is REPORT-ONLY on the track this branch serves, end to
    /// end through the ENGINE seam (mock engine → runner clock split → leg → pair record).
    ///
    /// `qwen3.8-27b-mtp-v1` declares `prefill_gain_exponent: 0.0`, so:
    ///
    /// 1. the window is MEASURED on every free-run leg and sealed per pair;
    /// 2. it is sealed `certified: false` — nothing is enforced on it, and a reader is told so;
    /// 3. NO new refusal fires: no pair is rejected, and none carries the
    ///    `prefill-window-uncertified` class.
    #[test]
    fn prefill_certification_is_report_only_on_this_branchs_track() {
        let oracle = oracle_decode_tokens();
        let n = BENCHMARK_DECODE_STEPS;
        let cfg = free_run_cfg(1, 1);
        let out = free_run_run(&cfg, || free_run_mock(&oracle, 4, n / 4));
        assert!(out.candidate_accepted);
        assert_eq!(out.results.accepted_pair_count, 1);

        // The declaration this behaviour follows from.
        let regime = TEST_REGIME;
        assert!(
            !regime.prefill_is_scored(),
            "precondition: this branch's track scores decode only"
        );

        // (1) the window is there, and both halves are real intervals from benchd's own clock.
        let w = out.results.pairs[0]
            .phase_windows
            .as_ref()
            .expect("a free-run pair seals the clock split benchd measured");
        for v in [
            w.serial_prefill_window_seconds,
            w.candidate_prefill_window_seconds,
            w.serial_decode_window_seconds,
            w.candidate_decode_window_seconds,
        ] {
            assert!(
                v.is_finite() && v > 0.0,
                "window half {v} must be measurable"
            );
        }
        assert_eq!(w.prefill_token_total, BENCHMARK_DECODE_SEED_TOKENS);
        assert_eq!(w.decode_token_total, n);

        // (2) REPORT-ONLY, stated on the record rather than left to be inferred — and the SECOND
        // fact stated separately: nothing corroborated benchd's clock, because nothing could.
        assert!(
            !w.certified,
            "a decode-only track must seal the window as report-only"
        );
        assert_eq!(
            w.cross_check,
            bench_core::prefill_window::PREFILL_WINDOW_CROSS_CHECK_NOT_OBSERVED,
            "no second measurement of the window exists on this protocol"
        );

        // (3) no new refusal fired.
        assert!(
            out.results.rejected_pairs.is_empty(),
            "certification must reject nothing on this track: {:?}",
            out.results.rejected_pairs
        );
        assert!(
            !out.results
                .rejected_pairs
                .iter()
                .any(|r| r.class == RejectClass::PrefillWindowUncertified.as_str()),
            "the prefill-window class must be unreachable on a decode-only track"
        );
    }

    /// B2 — the leg-level certification has ONE failure class now, and that is the point.
    ///
    /// STEP 3 (David 2026-09-15) removed the other. The regime used to be looked up PER LEG out of
    /// a compile-time table, so a leg could fail on "this track declares no regime" — a fault about
    /// the run's inputs, filed under a leg's measurement. The regime is now resolved ONCE, from the
    /// track fixture, where the track id is settled, and carried on `MeasureJobConfig`; a leg is
    /// handed the resolved value and cannot fail to resolve it. `RejectClass::ScoredRegimeUndeclared`
    /// went with the lookup: an unreachable reject class is a class that describes nothing.
    ///
    /// What is left is the WINDOW fault, which is a real per-attempt MEASUREMENT outcome and stays
    /// retryable.
    #[test]
    fn leg_certification_carries_only_the_window_fault() {
        let inv = inv_free_run(
            CANDIDATE_SPT,
            vec![4u32; BENCHMARK_DECODE_STEPS / 4],
            BENCHMARK_DECODE_STEPS,
        )
        .unwrap();
        // This branch's own track, under the regime the run resolved, certifies REPORT-ONLY.
        let regime = bench_core::constants::ScoredRegime {
            scored_batch_size: 1,
            prefill_gain_exponent: 0.0,
            decode_gain_exponent: 1.0,
        };
        let ok =
            certify_leg_prefill_window("candidate", bench_core::constants::TRACK_ID, &regime, &inv)
                .expect("the resolved regime is report-only, so certification cannot refuse");
        assert!(!ok.is_certified());
        assert_eq!(
            ok.cross_check(),
            bench_core::prefill_window::PREFILL_WINDOW_CROSS_CHECK_NOT_OBSERVED
        );
        // The WINDOW class is the one that survives, and it stays retryable.
        assert!(!RejectClass::PrefillWindowUncertified.is_deterministic());
    }

    /// B2 — a TEACHER-FORCED pair drives neither free-run verb, so it opens no prefill window and
    /// seals none. The block is OMITTED, never a fabricated or half-populated one.
    #[test]
    fn a_teacher_forced_pair_seals_no_phase_window() {
        let cfg = test_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            |_p| ok_candidate(),
        )
        .unwrap();
        assert_eq!(out.results.accepted_pair_count, 1);
        assert!(
            out.results.pairs[0].phase_windows.is_none(),
            "a teacher-forced pair has no free-run window to seal"
        );
        let json = serde_json::to_value(&out.results.pairs[0]).unwrap();
        assert!(
            json.get("phase_windows").is_none(),
            "the key must be OMITTED, not sealed as null"
        );
    }

    #[test]
    fn free_run_negative_control_doctored_acceptance_sum_kills_the_leg() {
        // §2.6 triple eq. 2 — a histogram that does not sum to N. The engine's committed stream is
        // still the golden (external verification PASSES), so ONLY the triple catches this.
        let oracle = oracle_decode_tokens();
        let n = BENCHMARK_DECODE_STEPS;
        let cfg = free_run_cfg(1, 1);
        // 33 rounds x 4 = 132 != 128.
        let out = free_run_run(&cfg, || free_run_mock(&oracle, 4, n / 4 + 1));
        assert!(
            !out.candidate_accepted,
            "a doctored acceptance histogram fails the run closed"
        );
        assert_eq!(out.results.accepted_pair_count, 0);
        let rej = &out.results.rejected_pairs[0];
        assert_eq!(
            rej.class, "free-run-consistency",
            "classified as the free-run accounting barrier"
        );
        assert_eq!(rej.leg, "candidate");
        assert!(
            rej.reason.contains("acceptance_lengths"),
            "reason names the doctored field: {}",
            rej.reason
        );
    }

    #[test]
    fn free_run_negative_control_under_reported_completed_work_kills_the_leg() {
        // §2.6 triple eq. 3 — `completed_work != R + 1`: the engine under-reports its verify-round
        // forwards, which is how "defer the work, claim the tokens" would look.
        let oracle = oracle_decode_tokens();
        let n = BENCHMARK_DECODE_STEPS;
        let cfg = free_run_cfg(1, 1);
        let out = free_run_run(&cfg, || {
            free_run_mock(&oracle, 4, n / 4).completed_work_delta(-1)
        });
        assert!(!out.candidate_accepted);
        let rej = &out.results.rejected_pairs[0];
        assert_eq!(rej.class, "free-run-consistency");
        assert!(
            rej.reason.contains("completed_work"),
            "reason names the forward counter: {}",
            rej.reason
        );
    }

    #[test]
    fn free_run_negative_control_token_count_mismatch_kills_the_leg() {
        // §2.4 — the response carries FEWER than N committed tokens (the engine materialized 127 of
        // the 128 it was asked for). This is an accounting lie, not an infra fault.
        let oracle = oracle_decode_tokens();
        let n = BENCHMARK_DECODE_STEPS;
        let short: Vec<i64> = oracle[..n - 1].to_vec();
        let cfg = free_run_cfg(1, 1);
        let out = free_run_run(&cfg, || free_run_mock(&short, 4, n / 4));
        assert!(!out.candidate_accepted);
        let rej = &out.results.rejected_pairs[0];
        assert_eq!(rej.class, "free-run-consistency");
        assert!(
            rej.reason.contains("committed tokens"),
            "reason names the count invariant: {}",
            rej.reason
        );
    }

    #[test]
    fn free_run_negative_control_wrong_committed_token_is_a_parity_kill() {
        // §2.7 — a single wrong free-run token is a HARD failure (the same class as a teacher-forced
        // decode mismatch), because under greedy the golden is the one correct continuation.
        let mut perturbed = oracle_decode_tokens();
        perturbed[64] += 1;
        let n = BENCHMARK_DECODE_STEPS;
        let cfg = free_run_cfg(1, 1);
        let out = free_run_run(&cfg, || free_run_mock(&perturbed, 4, n / 4));
        assert!(
            !out.candidate_accepted,
            "a wrong committed token fails the run closed"
        );
        assert_eq!(out.results.rejected_pairs[0].class, "token-mismatch-parity");
    }

    #[test]
    fn free_run_negative_control_run_timeout_kills_the_leg_with_die5_semantics() {
        // §2.2 — the RunTimeout liveness bound (N x band-ceiling x margin, armed by the caller). A
        // hung engine does NOT wedge benchd: the leg fails, the session is discarded, and after the
        // one gated retry the pair folds into die-5.
        let oracle = oracle_decode_tokens();
        let n = BENCHMARK_DECODE_STEPS;
        let cfg = free_run_cfg(1, 1);
        let budget = std::time::Duration::from_millis(20);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_free_run_serial(),
            move |p: &TimingParams| {
                let armed = p.clone().with_run_timeout(Some(budget));
                free_run_leg(
                    || free_run_mock(&oracle, 4, n / 4).stall_on("free_decode_run"),
                    &armed,
                )
            },
        )
        .unwrap();
        assert!(
            !out.candidate_accepted,
            "a timed-out leg is a leg failure (die-5 semantics)"
        );
        assert_eq!(out.results.accepted_pair_count, 0);
        assert_eq!(
            out.results.rejected_pairs[0].class, "run-timeout",
            "the stall is classified as the liveness bound, not the infra catch-all"
        );
    }

    #[test]
    fn free_run_negative_control_uncapable_engine_is_refused() {
        // §2.1 — an engine that does not advertise `free_run_decode` is REFUSED (the runner does it
        // before the gate and the clock; here we assert the run consequence: no accepted pair).
        let oracle = oracle_decode_tokens();
        let cfg = free_run_cfg(1, 1);
        let out = free_run_run(&cfg, || {
            // NO `.free_run_capable()` — a v1-only engine.
            MockEngine::new().oracle_tokens(PREFILL_TOKEN, SEED_TOKEN, oracle.to_vec())
        });
        assert!(
            !out.candidate_accepted,
            "an engine that cannot free-run is refused, not downgraded"
        );
        assert!(
            out.results.rejected_pairs[0]
                .reason
                .contains("free_run_decode"),
            "the refusal names the missing capability: {}",
            out.results.rejected_pairs[0].reason
        );
    }

    #[test]
    fn free_run_leg_refuses_a_serial_effective_regime_echo() {
        // W3 seal guard — an engine that runs the free-run WINDOW but echoes `serial` did not draft;
        // sealing its number under the speculating series would misattribute a serial measurement.
        let cfg = free_run_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_free_run_serial(),
            |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
                let mut inv = inv_free_run(CANDIDATE_SPT, vec![4; 32], BENCHMARK_DECODE_STEPS)?;
                inv.wire_effective_spec = Some(SpecConfig::serial());
                Ok(inv)
            },
        )
        .unwrap();
        assert!(!out.candidate_accepted);
        assert_eq!(
            out.results.rejected_pairs[0].class,
            "free-run-regime-not-speculative"
        );
    }

    #[test]
    fn free_run_serial_control_that_speculates_is_a_leg_failure() {
        // Fable ruling — the MIRROR guard, on the denominator. The control free-runs at depth 0; an
        // engine that echoes a SPECULATING effective regime on that leg drafted the control, which
        // would inflate the denominator's speed and DEFLATE the published speedup. Refused with its
        // own class, so a drafting control is never mistaken for a non-drafting candidate.
        let cfg = free_run_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
                let mut inv = ok_free_run_serial()?;
                inv.wire_effective_spec = Some(SpecConfig::mtp(FREE_RUN_DEPTH));
                Ok(inv)
            },
            |_p: &TimingParams| inv_free_run(CANDIDATE_SPT, vec![4; 32], BENCHMARK_DECODE_STEPS),
        )
        .unwrap();
        assert!(
            !out.candidate_accepted,
            "a speculating control fails the run closed"
        );
        let rej = &out.results.rejected_pairs[0];
        assert_eq!(rej.class, "free-run-serial-control-speculated");
        assert_eq!(rej.leg, "serial");
        assert!(
            rej.reason.contains("DENOMINATOR"),
            "reason names the harm: {}",
            rej.reason
        );
    }

    #[test]
    fn free_run_serial_control_histogram_must_be_all_ones() {
        // Fable ruling — the SECOND, independent channel on the same fact. Even with a conformant
        // `serial` echo, a control whose per-round histogram shows a round committing MORE THAN ONE
        // token demonstrably speculated. The histogram is already pinned by the §2.6 triple (it sums
        // to N, its length is cross-checked by `completed_work`), so it cannot be doctored to hide
        // this the way a self-reported echo could.
        let cfg = free_run_cfg(1, 1);
        let n = BENCHMARK_DECODE_STEPS;
        // A fully §2.6-conformant histogram — N rounds, summing to N, `completed_work == R+1` — that
        // is nonetheless NOT `[1]*N`: one round committed 2 and another committed 0. The triple
        // cannot see this; only the unit-histogram rule can.
        let mut drafting = vec![0u32, 2];
        drafting.extend(std::iter::repeat_n(1u32, n - 2));
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            move |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
                let mut inv = ok_free_run_serial()?;
                // The echo still says `serial` — only the histogram betrays the drafting.
                inv.free_run_audit = Some(free_run_audit(&drafting, n));
                Ok(inv)
            },
            |_p: &TimingParams| inv_free_run(CANDIDATE_SPT, vec![4; 32], BENCHMARK_DECODE_STEPS),
        )
        .unwrap();
        assert!(!out.candidate_accepted);
        let rej = &out.results.rejected_pairs[0];
        assert_eq!(rej.class, "free-run-serial-control-speculated");
        assert_eq!(rej.leg, "serial");
        assert!(
            rej.reason.contains("acceptance_lengths[") && rej.reason.contains("SPECULATED"),
            "reason names the offending round: {}",
            rej.reason
        );
        // The pure predicate, directly: `[1]*N` passes; a short round count (the shape a genuinely
        // drafting control produces — fewer rounds than tokens) does not.
        assert!(free_run_serial_control_histogram_is_unit(&vec![1u32; n], n).is_ok());
        let e = free_run_serial_control_histogram_is_unit(&vec![4u32; n / 4], n).unwrap_err();
        assert!(e.contains("32 verify rounds"), "{e}");
    }

    #[test]
    fn free_run_serial_control_accepts_the_depth_zero_echo_the_candidate_leg_refuses() {
        // Fable ruling — the RELAXATION is leg-scoped, both directions. The SAME `serial` echo that
        // is a leg failure on the candidate is REQUIRED on the control; the same `mtp` echo that is
        // required on the candidate is a leg failure on the control. One predicate per leg, no
        // shared "accept either" weakening that would let a non-drafting candidate through.
        let serial_echo = SpecConfig::serial();
        let mtp_echo = SpecConfig::mtp(FREE_RUN_DEPTH);
        assert!(free_run_serial_control_is_non_speculating(&serial_echo).is_ok());
        assert!(free_run_regime_is_speculative(&serial_echo).is_err());
        assert!(free_run_regime_is_speculative(&mtp_echo).is_ok());
        assert!(free_run_serial_control_is_non_speculating(&mtp_echo).is_err());
    }

    #[test]
    fn free_run_both_legs_share_one_regime_window_and_timeout() {
        // Fable ruling — the structural half: the control's regime is DERIVED from the candidate's
        // (never set independently), and the pair loop hands BOTH legs the very same `TimingParams`,
        // so N and the armed §2.2 RunTimeout are identical by construction rather than by
        // convention. That identity is what makes the 27.2 ms M-5 round-trip floor cancel in the
        // ratio: both sides now pay it exactly once.
        assert_eq!(
            serial_control_regime_for(LegRegime::FreeRunV1_1),
            LegRegime::FreeRunV1_1
        );
        assert_eq!(
            serial_control_regime_for(LegRegime::TeacherForcedV1),
            LegRegime::TeacherForcedV1
        );
        /// What each leg saw of the shared window: (N, the armed RunTimeout).
        type LegWindow = (usize, Option<std::time::Duration>);
        let seen: std::rc::Rc<std::cell::RefCell<Vec<LegWindow>>> = Default::default();
        let cfg = free_run_cfg(1, 1);
        let (s, c) = (seen.clone(), seen.clone());
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            move |p: &TimingParams| {
                s.borrow_mut().push((p.decode_steps, p.run_timeout));
                ok_free_run_serial()
            },
            move |p: &TimingParams| {
                c.borrow_mut().push((p.decode_steps, p.run_timeout));
                inv_free_run(CANDIDATE_SPT, vec![4; 32], BENCHMARK_DECODE_STEPS)
            },
        )
        .unwrap();
        assert!(out.candidate_accepted);
        let seen = seen.borrow();
        assert_eq!(seen.len(), 2, "one params observation per leg");
        assert_eq!(
            seen[0], seen[1],
            "both legs get the SAME N and the SAME RunTimeout"
        );
        assert_eq!(
            seen[0].0, FREE_RUN_DECODE_TOKENS,
            "the RULED N=128 on both legs"
        );
        // Both legs are PARENT-CLOCKED: the scored spt sealed per leg is the `benchd_seconds_per_token`
        // the closure returned, never the report's own claim (which differs here on purpose).
        let pair = &out.results.pairs[0];
        assert_eq!(pair.serial_seconds_per_token, SERIAL_SPT);
        assert_eq!(pair.mtp_seconds_per_token, CANDIDATE_SPT);
    }

    #[test]
    fn free_run_leg_without_audit_fails_closed() {
        // A free-run leg with no §3 AUDIT is refused: benchd seals only the histogram the runner
        // produced after the triple passed, and NEVER fabricates one.
        let cfg = free_run_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_free_run_serial(),
            |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
                let mut inv = inv_free_run(CANDIDATE_SPT, vec![4; 32], BENCHMARK_DECODE_STEPS)?;
                inv.free_run_audit = None;
                Ok(inv)
            },
        )
        .unwrap();
        assert!(!out.candidate_accepted);
        assert_eq!(out.results.rejected_pairs[0].class, "free-run-consistency");
    }

    #[test]
    fn free_run_audit_covering_a_different_window_fails_closed() {
        // The AUDIT must describe the window that was CLOCKED: an audit over a different N means the
        // sealed acceptance histogram and the scored seconds-per-token disagree about the divisor.
        let cfg = free_run_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_free_run_serial(),
            |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
                // A self-consistent audit — but over N=64, not the run's N=128.
                inv_free_run(CANDIDATE_SPT, vec![4; 16], 64)
            },
        )
        .unwrap();
        assert!(!out.candidate_accepted);
        let rej = &out.results.rejected_pairs[0];
        assert_eq!(rej.class, "free-run-consistency");
        assert!(
            rej.reason.contains("verified tokens"),
            "reason: {}",
            rej.reason
        );
    }

    #[test]
    fn teacher_forced_leg_carrying_a_free_run_audit_is_refused() {
        // The inverse fabrication: a teacher-forced leg cannot have earned acceptance (benchd fed
        // every token), so an audit attached to one is a fabricated free-run claim.
        let cfg = test_cfg(1, 1);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
                let mut inv = inv(echo(CANDIDATE_SPT), GateState::Fired, candidate_telemetry());
                inv.free_run_audit = Some(free_run_audit(&[4; 32], BENCHMARK_DECODE_STEPS));
                Ok(inv)
            },
        )
        .unwrap();
        assert!(!out.candidate_accepted);
        assert_eq!(out.results.rejected_pairs[0].class, "free-run-consistency");
    }

    #[test]
    fn all_teacher_forced_run_still_seals_a_homogeneous_series() {
        // The OTHER legal shape: an all-TF run seals the single tag, `homogeneous`, comparable legs,
        // one timed_regime, and NO audit keys — the Model-2 seal is unchanged apart from the tags.
        let out = identity_run(&test_cfg(1, 1));
        assert_eq!(out.results.timed_mode, "teacher_forced_v1");
        assert!(out.results.timed_series.homogeneous);
        assert!(out.results.timed_series.legs_comparable);
        assert_eq!(out.results.timed_regime, Some(TIMED_REGIME));
        let pair = &out.results.pairs[0];
        assert_eq!(pair.serial_timed_mode, "teacher_forced_v1");
        assert_eq!(pair.candidate_timed_mode, "teacher_forced_v1");
        assert!(pair.audit_spec_acceptance_lengths.is_none());
        assert!(
            pair.audit_spec.is_empty(),
            "no audit_spec_* keys on a teacher-forced pair"
        );
    }

    #[test]
    fn l1_unadvertised_free_run_capability_is_its_own_class_and_is_not_retried() {
        // #108 (L1) — an engine that does not advertise the v1.1 free-run capability (§2.1) is a
        // DETERMINISTIC condition: the hello handshake is a property of the binary, so the retry's
        // reset (fresh worker + fresh cool gate) cannot change the answer. It gets its own reject
        // class (the operator must be pointed at the ENGINE BUILD, not the box) and is TERMINAL.
        let candidate_calls = Cell::new(0usize);
        let measure_candidate = |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
            candidate_calls.set(candidate_calls.get() + 1);
            Err(bench_runner::RunnerError::CapabilityNotAdvertised {
                capability: "free_run_decode".to_string(),
            })
        };
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &free_run_cfg(1, 1),
            |_p| ok_serial(),
            measure_candidate,
        )
        .expect("a rejected leg still seals a die-5 results.json");
        assert!(!out.candidate_accepted, "the run fails closed");
        assert_eq!(out.results.accepted_pair_count, 0);
        // Its OWN class, not the `infra` catch-all.
        let rej = &out.results.rejected_pairs[0];
        assert_eq!(rej.class, "free-run-capability-missing");
        assert_eq!(rej.leg, "candidate");
        assert!(
            rej.reason.contains("free_run_decode"),
            "the capability is named: {}",
            rej.reason
        );
        // NOT retried: exactly ONE invocation, where a retryable class would have taken two.
        assert_eq!(
            candidate_calls.get(),
            1,
            "a deterministic reject is terminal — no second spawn + cool gate for the same verdict"
        );
        assert!(RejectClass::FreeRunCapabilityMissing.is_deterministic());

        // CONTROL — the exemption is narrow: a retryable class still takes its one gated retry, so
        // the difference above is the exemption and not a broken retry loop.
        let retry_calls = Cell::new(0usize);
        let out = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &test_cfg(1, 1),
            |_p| ok_serial(),
            |_p: &TimingParams| -> bench_runner::Result<LegInvocation> {
                retry_calls.set(retry_calls.get() + 1);
                Err(retryable_reject())
            },
        )
        .unwrap();
        assert!(!out.candidate_accepted);
        assert_eq!(
            retry_calls.get(),
            2,
            "a retryable class keeps R19's one gated retry"
        );
        assert!(!classify(&retryable_reject()).is_deterministic());
    }

    #[test]
    fn observed_tf_serial_leg_under_a_free_run_run_refuses_the_seal() {
        // #108 (M1) — the reviewer's exact drive. The cfg declares the free-run series (so
        // `serial_control_regime_for` says BOTH legs free-run), but the SERIAL leg is actually driven
        // TEACHER-FORCED. Every leg-level guard passes (a TF leg echoing `serial` is a legal TF leg,
        // and the free-run candidate is conformant), so the pair is ACCEPTED and the crossing is
        // visible ONLY in the per-leg tags the pair carries.
        //
        // Before M1 the write side restated cfg and sealed `homogeneous: true` /
        // `serial_leg_timed_mode: "free_run_v1_1"` over a leg that was teacher-forced — a §5 lie the
        // overlay's fence could not catch, because the file was internally consistent. Now the seal
        // is REFUSED, naming both the observed and the expected tag.
        let oracle = oracle_decode_tokens();
        let n = BENCHMARK_DECODE_STEPS;
        let cfg = free_run_cfg(1, 1);
        let err = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            // The TEACHER-FORCED serial leg — the wrong series for this run.
            |_p| ok_serial(),
            move |p: &TimingParams| free_run_leg(|| free_run_mock(&oracle, 4, n / 4), p),
        )
        .err()
        .expect("a leg measured in a series the run did not declare must not be sealed");
        assert!(err.contains("REFUSES to seal"), "{err}");
        assert!(
            err.contains("serial leg"),
            "the failing leg is named: {err}"
        );
        // BOTH sides named: what was measured, and what the regime rule expected.
        assert!(
            err.contains("\"teacher_forced_v1\""),
            "observed tag named: {err}"
        );
        assert!(
            err.contains("\"free_run_v1_1\""),
            "expected tag named: {err}"
        );
        // And it is NOT laundered into the MIXED descriptor: measure-job never seals that shape.
        assert!(
            !err.contains(MIXED_SERIES_DESCRIPTOR),
            "the refusal is a defect report, not a MIXED seal: {err}"
        );
    }

    #[test]
    fn observed_series_derivation_leaves_both_honest_shapes_unchanged() {
        // #108 (M1) — the other half: deriving from the OBSERVED tags must not perturb either legal
        // shape. Both still seal their single tag, `homogeneous`, comparable legs, and one regime.
        let fr = free_run_run(&free_run_cfg(1, 1), || {
            free_run_mock(&oracle_decode_tokens(), 4, BENCHMARK_DECODE_STEPS / 4)
        });
        assert!(fr.candidate_accepted);
        assert_eq!(fr.results.timed_mode, "free_run_v1_1");
        assert!(fr.results.timed_series.homogeneous);
        assert!(fr.results.timed_series.legs_comparable);
        assert_eq!(fr.results.timed_regime, Some(FREE_RUN_TIMED_REGIME));

        let tf = identity_run(&test_cfg(1, 1));
        assert!(tf.candidate_accepted);
        assert_eq!(tf.results.timed_mode, "teacher_forced_v1");
        assert!(tf.results.timed_series.homogeneous);
        assert!(tf.results.timed_series.legs_comparable);
        assert_eq!(tf.results.timed_regime, Some(TIMED_REGIME));

        // A die-5 run has NO pair to observe: it honestly seals its DECLARED series (and omits the
        // top-level regime label, which would assert a measurement that never completed).
        let die5 = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &test_cfg(1, 1),
            |_p| ok_serial(),
            |_p: &TimingParams| Err(bench_runner::RunnerError::Protocol("boom".to_string())),
        )
        .expect("a die-5 run still seals a results.json");
        assert!(!die5.candidate_accepted);
        assert_eq!(die5.results.accepted_pair_count, 0);
        assert_eq!(die5.results.timed_mode, "teacher_forced_v1");
        assert!(die5.results.timed_series.homogeneous);
        assert_eq!(die5.results.timed_regime, None);
    }

    /// NEGATIVE CONTROL for the per-prompt window guard ([`validate_prompt_windows`]).
    ///
    /// The guard's only other coverage asserts its message is ABSENT (the gates-oracle diagnostic
    /// wins over it in `measure_job_tape_golden.rs`), which stays green if the guard is deleted
    /// outright. Nothing proved it FIRES. These controls do: a prompt one reference token short of
    /// the window is refused BY NAME on BOTH shapes, and a prompt that exactly covers the window is
    /// accepted — so a widened or removed guard fails here instead of passing preflight and dying
    /// on the first prompt of a gated box run.
    #[test]
    fn prompt_window_guard_refuses_a_short_prompt_by_name_on_both_shapes() {
        let n = BENCHMARK_DECODE_STEPS;
        let full = oracle_decode_tokens();
        let short: Vec<i64> = full[..n - 1].to_vec();

        // POSITIVE control: a prompt that EXACTLY covers the window is accepted, on both shapes —
        // so the negative controls below cannot pass by refusing everything.
        let tape = measure_tape_with(1, full.clone());
        let golden = measure_golden_with("exact", full.clone());
        validate_prompt_windows(&[tape, golden], n)
            .expect("a prompt covering the window exactly must be accepted");

        // NEGATIVE control, per SHAPE, each at the window that shape can actually fall short of:
        //   * a TAPE oracles `rows.len()` post-seed tokens and carries no loader-side minimum, so it
        //     falls short one row below the free-run window N;
        //   * a GOLDEN cannot be built short of [`BENCHMARK_DECODE_STEPS`] — `load_golden_fixture`
        //     already refuses that at load — so its shortfall is the TEACHER-FORCED window
        //     ([`DEFAULT_TOKENS`] = 512) against the 128-token oracle it legitimately carries. That
        //     is the production shape this guard catches pre-GPU.
        // The refusal names the prompt's sha256, its SHAPE, and the window it could not oracle —
        // the three facts an operator needs to find the offending file.
        for (short_prompt, window) in [
            (measure_tape_with(2, short.clone()), n),
            (
                measure_golden_with("tf-window", full.clone()),
                DEFAULT_TOKENS,
            ),
        ] {
            let n = window;
            let sha = short_prompt.sha256().to_string();
            let kind = short_prompt.kind();
            let err = validate_prompt_windows(&[short_prompt], n)
                .expect_err("a prompt short of the window must be refused");
            assert!(
                err.contains(&sha),
                "the refusal must name the sha256: {err}"
            );
            assert!(err.contains(kind), "the refusal must name the shape: {err}");
            assert!(
                err.contains(&format!("{n}-token decode window")),
                "the refusal must name the window: {err}"
            );
            assert!(err.contains("die 8"), "the refusal must be die-8: {err}");
        }

        // A GOOD prompt must not mask a bad one: the guard is over the WHOLE set, so a short prompt
        // behind a covering one is still refused, and the refusal names the SHORT one.
        let good = measure_tape_with(3, full);
        let bad = measure_tape_with(4, short);
        let bad_sha = bad.sha256().to_string();
        let err = validate_prompt_windows(&[good, bad], n)
            .expect_err("a short prompt anywhere in the set must be refused");
        assert!(
            err.contains(&bad_sha),
            "the refusal must name the SHORT prompt, not the covering one: {err}"
        );
    }

    #[test]
    fn free_run_regime_coherence_is_fail_closed() {
        // The regime, the declared spec and the window must agree BEFORE any measuring.
        // (1) free-run regime + serial candidate spec: nothing to free-run.
        let mut cfg = free_run_cfg(1, 1);
        cfg.candidate_spec = SpecConfig::serial();
        let err = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            |_p: &TimingParams| ok_candidate(),
        )
        .err()
        .expect("an incoherent regime/spec pair must not measure");
        assert!(err.contains("no speculation to free-run"), "{err}");

        // (2) free-run regime at a window other than the RULED N.
        let mut cfg = free_run_cfg(1, 1);
        cfg.tokens = 64;
        let err = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            |_p| ok_serial(),
            |_p: &TimingParams| ok_candidate(),
        )
        .err()
        .expect("a non-RULED free-run window must not measure");
        assert!(err.contains("RULES N = 128"), "{err}");

        // (3) the legacy Model-2 pairing (teacher-forced regime + declared mtp spec) stays LEGAL.
        assert!(validate_candidate_regime_coherent(&test_cfg(1, 1)).is_ok());
    }

    #[test]
    fn candidate_regime_derivation_is_the_single_production_rule() {
        // The rule main.rs uses: a speculating declared spec selects the free-run series.
        assert_eq!(
            candidate_regime_for_spec(&SpecConfig::serial()),
            LegRegime::TeacherForcedV1
        );
        assert_eq!(
            candidate_regime_for_spec(&SpecConfig::mtp(2)),
            LegRegime::FreeRunV1_1
        );
        // dflash (and any future speculating module) lands on the free-run side too.
        let dflash = SpecConfig {
            mode: bench_protocol::SPEC_MODE_DFLASH.to_string(),
            mtp: None,
            dflash: Some(json!({})),
            dspark: None,
        };
        assert_eq!(candidate_regime_for_spec(&dflash), LegRegime::FreeRunV1_1);
    }

    // -----------------------------------------------------------------------
    // COHORT (batch-8 brief §4.5) — the batched cohort measure-job
    // -----------------------------------------------------------------------

    const COHORT_B: usize = SCORED_BATCH_SIZE_B8 as usize;

    /// COMPOSITE (Gemma cohort scoring) — the conformant PREFILL windows the cohort test fixtures
    /// use, distinct from the DECODE-side `SERIAL_SPT`/`CANDIDATE_SPT` (2x) so a test can tell the
    /// two components apart: prefill_gain = 0.400 / 0.100 = 4.0, decode_gain = 2.0.
    const SERIAL_PREFILL_ELAPSED: f64 = 0.400;
    const CANDIDATE_PREFILL_ELAPSED: f64 = 0.100;

    /// Build a conformant [`CohortPhaseWindows`] for a leg whose DECODE-side scored `spt` is
    /// `spt` (so `decode_elapsed_seconds = spt * B * N`, keeping `benchd_seconds_per_token` and
    /// the phase-split window mutually consistent, exactly as the real runner's
    /// `BatchedFreeRunPhaseTiming` guarantees), with an explicit `prefill_elapsed`.
    fn cohort_phase_windows_for(spt: f64, prefill_elapsed: f64, n: usize) -> CohortPhaseWindows {
        let decode_token_total = COHORT_B * n;
        CohortPhaseWindows {
            prefill_elapsed_seconds: prefill_elapsed,
            // An arbitrary but FIXED per-stream seed length (8) for these mock legs — this
            // fixture exercises the aggregation math (sums / medians / gains), not the real
            // derivation from `BENCHMARK_DECODE_SEED_TOKENS` (which `cohort_timing_params`
            // performs against real goldens).
            prefill_token_total: COHORT_B * 8,
            decode_elapsed_seconds: spt * decode_token_total as f64,
            decode_token_total,
        }
    }

    /// The batched cohort regime at the one certified batch point, built through the same
    /// certify-match production uses.
    fn b8_regime() -> LegRegime {
        LegRegime::BatchedFreeRunV1_2(ScoredBatchPoint::certify(SCORED_BATCH_SIZE_B8).unwrap())
    }

    /// The B distinct pool tapes of the cohort, in POOL ORDER (distinct seed markers ⇒ distinct
    /// sha256 per slot).
    fn cohort_goldens() -> Vec<TimedPrompt> {
        (0..COHORT_B)
            .map(|s| measure_tape_with(100 + s as i64, oracle_decode_tokens()))
            .collect()
    }

    /// The contract pool pinning exactly those tapes, in the same order, sha256 + bytes together.
    fn cohort_pool(goldens: &[TimedPrompt]) -> Vec<PromptPoolEntry> {
        goldens
            .iter()
            .map(|g| PromptPoolEntry {
                sha256: g.sha256().to_string(),
                bytes: Some(g.byte_len()),
                noop_decode_speedup: Some(0.99),
            })
            .collect()
    }

    /// The config a SCORED batched cohort run uses: speculating candidate spec, the RULED window,
    /// the batched regime, and the RULED per-cohort target.
    fn cohort_cfg(min_pairs: usize, target_pairs: usize) -> MeasureJobConfig {
        MeasureJobConfig {
            tokens: FREE_RUN_DECODE_TOKENS,
            mtp_depth: Some(FREE_RUN_DEPTH as usize),
            candidate_spec: SpecConfig::mtp(FREE_RUN_DEPTH),
            candidate_regime: b8_regime(),
            // The CERTIFIED ruled pair — a cohort config carries it exactly as the real caller
            // (main.rs) would after `ScoredExponents::certify` succeeds.
            scored_exponents: Some(SCORED_EXPONENTS),
            ..test_cfg(min_pairs, target_pairs)
        }
    }

    /// Build a [`CohortFreeRunAudit`] the way the RUNNER does — through
    /// `verify_cohort_consistency` — so a test can never fabricate an audit the cohort quadruple
    /// would have rejected. Conformant shape: every row's natural walk equals the common width,
    /// all `b` streams active every round, `completed_work = R + 1` (scalar).
    fn cohort_audit_b(
        b: u32,
        acceptance_lengths: &[u32],
        n: usize,
    ) -> bench_core::free_run::CohortFreeRunAudit {
        let rounds = acceptance_lengths.len();
        let resp = bench_core::free_run::CohortFreeRunResponse {
            batch_size: b,
            tokens_len_by_stream: vec![n; b as usize],
            acceptance_lengths: acceptance_lengths.to_vec(),
            natural_accepted_by_stream: vec![acceptance_lengths.to_vec(); b as usize],
            active_streams_by_round: vec![b; rounds],
            rounds: rounds as u32,
            drafted_total: b as u64 * n as u64,
            accepted_total: b as u64 * (n as u64 - rounds as u64),
            committed_total: b as u64 * n as u64,
            depth_clamp_reasons: [("tail_depth".to_string(), 2u32)].into_iter().collect(),
        };
        bench_core::free_run::verify_cohort_consistency(&resp, n as u32, rounds as i64 + 1)
            .expect("test cohort audit must satisfy the quadruple")
    }

    fn cohort_audit(
        acceptance_lengths: &[u32],
        n: usize,
    ) -> bench_core::free_run::CohortFreeRunAudit {
        cohort_audit_b(SCORED_BATCH_SIZE_B8, acceptance_lengths, n)
    }

    /// (b) admission — a fixed B x N committed rectangle with DISTINCT per-slot token spaces, so a
    /// slot-order or "reads only slot 0" bug in the trusted-oracle gate is visible. Slot `s` commits
    /// `70000 + s*1000 + i` at position `i`.
    fn cohort_committed_rect(n: usize) -> Vec<Vec<i64>> {
        (0..SCORED_BATCH_SIZE_B8 as usize)
            .map(|s| {
                (0..n as i64)
                    .map(|i| 70_000 + s as i64 * 1000 + i)
                    .collect()
            })
            .collect()
    }

    /// (b) admission — a MOCK trusted oracle that PASSES: it echoes the candidate's committed tokens
    /// back verbatim as `committed_token` AND reports each `sequential_argmax` EQUAL to the committed
    /// token, so N2 passes and every stream has 0 mismatches (a clean ≤10% pass). Slot order is the
    /// input order.
    fn mock_oracle_pass(
        _replay_seeds_by_stream: &[Vec<i64>],
        committed_by_stream: &[Vec<i64>],
    ) -> bench_runner::Result<bench_protocol::CohortReferenceReplayReport> {
        let streams = committed_by_stream
            .iter()
            .enumerate()
            .map(
                |(slot, tokens)| bench_protocol::CohortReferenceReplayStream {
                    slot: slot as i64,
                    positions: tokens
                        .iter()
                        .map(|&t| bench_protocol::CohortReferenceReplayPosition {
                            committed_token: t,
                            sequential_argmax: t,
                            ..Default::default()
                        })
                        .collect(),
                },
            )
            .collect();
        Ok(bench_protocol::CohortReferenceReplayReport {
            streams,
            ..Default::default()
        })
    }

    /// A conformant batched CANDIDATE leg: speculating wire echo, candidate head, cohort audit
    /// with a multi-token common-width histogram (it drafted), and the committed journal surfaced for
    /// the (b) trusted-oracle tolerance gate.
    fn inv_cohort_candidate(spt: f64) -> bench_runner::Result<LegInvocation> {
        let n = FREE_RUN_DECODE_TOKENS;
        Ok(LegInvocation {
            benchd_seconds_per_token: spt,
            wire_head_provenance: head_prov(CANDIDATE_HEAD_SHA),
            gate_state: GateState::Fired,
            quiesce: QuiesceRecord::default(),
            telemetry: candidate_telemetry(),
            wire_effective_spec: Some(SpecConfig::mtp(FREE_RUN_DEPTH)),
            regime: b8_regime(),
            free_run_audit: None,
            // B2 — a BATCHED cohort leg carries its split on `cohort_phase_windows`; the
            // single-stream `phase_window` channel is a v1.1 free-run fact and is None here.
            phase_window: None,
            cohort_audit: Some(cohort_audit(&vec![4u32; n / 4], n)),
            cohort_phase_windows: Some(cohort_phase_windows_for(spt, CANDIDATE_PREFILL_ELAPSED, n)),
            per_stream_timing: None,
            cohort_committed_tokens_by_stream: Some(cohort_committed_rect(n)),
        })
    }

    /// A conformant batched SERIAL CONTROL leg: serial wire echo, pinned head, and the `[1]*N`
    /// common-width histogram a non-speculating cohort commits.
    fn ok_cohort_serial() -> bench_runner::Result<LegInvocation> {
        let n = FREE_RUN_DECODE_TOKENS;
        Ok(LegInvocation {
            benchd_seconds_per_token: SERIAL_SPT,
            wire_head_provenance: head_prov(SERIAL_HEAD_SHA),
            gate_state: GateState::Fired,
            quiesce: QuiesceRecord::default(),
            telemetry: serial_telemetry(),
            wire_effective_spec: Some(SpecConfig::serial()),
            regime: b8_regime(),
            free_run_audit: None,
            // B2 — a BATCHED cohort leg carries its split on `cohort_phase_windows`; the
            // single-stream `phase_window` channel is a v1.1 free-run fact and is None here.
            phase_window: None,
            cohort_audit: Some(cohort_audit(&vec![1u32; n], n)),
            cohort_phase_windows: Some(cohort_phase_windows_for(
                SERIAL_SPT,
                SERIAL_PREFILL_ELAPSED,
                n,
            )),
            per_stream_timing: None,
            // (b) admission — the serial CONTROL is not token-judged; `validate_leg_report` drops
            // this on the control leg, but a real cohort leg surfaces it, so mirror that here.
            cohort_committed_tokens_by_stream: Some(cohort_committed_rect(n)),
        })
    }

    /// A conformant batched cohort run over the whole pinned pool.
    fn cohort_identity_run(cfg: &MeasureJobConfig) -> Result<MeasureJobOutcome, String> {
        let goldens = cohort_goldens();
        let members =
            validate_cohort_membership(&goldens, &cohort_pool(&goldens), SCORED_BATCH_SIZE_B8)
                .expect("conformant cohort membership");
        run_cohort_measure_job(
            &goldens,
            members,
            &DirDigest::empty(),
            "deadbeef",
            cfg,
            CohortSeams {
                measure_serial: |_p: &CohortTimingParams| ok_cohort_serial(),
                measure_candidate: |_p: &CohortTimingParams| inv_cohort_candidate(CANDIDATE_SPT),
                oracle: mock_oracle_pass,
            },
        )
    }

    /// A conformant cohort run driven with an ARBITRARY trusted-oracle mock, so a test can vary ONLY
    /// the oracle (holding both legs conformant) and observe the (b) gate's verdict.
    fn cohort_run_with_oracle<FO>(oracle: FO) -> Result<MeasureJobOutcome, String>
    where
        FO: FnMut(
            &[Vec<i64>],
            &[Vec<i64>],
        ) -> bench_runner::Result<bench_protocol::CohortReferenceReplayReport>,
    {
        let goldens = cohort_goldens();
        let members =
            validate_cohort_membership(&goldens, &cohort_pool(&goldens), SCORED_BATCH_SIZE_B8)
                .expect("conformant cohort membership");
        run_cohort_measure_job(
            &goldens,
            members,
            &DirDigest::empty(),
            "deadbeef",
            &cohort_cfg(PAIRS_PER_COHORT_TARGET, PAIRS_PER_COHORT_TARGET),
            CohortSeams {
                measure_serial: |_p: &CohortTimingParams| ok_cohort_serial(),
                measure_candidate: |_p: &CohortTimingParams| inv_cohort_candidate(CANDIDATE_SPT),
                oracle,
            },
        )
    }

    /// (b) admission — an oracle whose reference argmax diverges from the candidate's committed token
    /// on the FIRST `mismatches` positions of `bad_slot` ONLY. N2 always passes (committed_token is
    /// echoed truthfully); only the tolerance count is driven. Distinct per-slot so a "reads slot 0"
    /// bug is visible.
    fn mock_oracle_diverge_slot(
        bad_slot: usize,
        mismatches: usize,
    ) -> impl FnMut(
        &[Vec<i64>],
        &[Vec<i64>],
    ) -> bench_runner::Result<bench_protocol::CohortReferenceReplayReport> {
        move |_seeds: &[Vec<i64>], committed: &[Vec<i64>]| {
            let streams = committed
                .iter()
                .enumerate()
                .map(
                    |(slot, tokens)| bench_protocol::CohortReferenceReplayStream {
                        slot: slot as i64,
                        positions: tokens
                            .iter()
                            .enumerate()
                            .map(|(pos, &t)| {
                                // N2 echo is ALWAYS truthful; only the reference argmax diverges.
                                let argmax = if slot == bad_slot && pos < mismatches {
                                    t + 7
                                } else {
                                    t
                                };
                                bench_protocol::CohortReferenceReplayPosition {
                                    committed_token: t,
                                    sequential_argmax: argmax,
                                    ..Default::default()
                                }
                            })
                            .collect(),
                    },
                )
                .collect();
            Ok(bench_protocol::CohortReferenceReplayReport {
                streams,
                ..Default::default()
            })
        }
    }

    #[test]
    fn cohort_tolerance_gate_accepts_just_under_threshold() {
        // N = 128, so 10% = 12.8; 12 mismatches in one slot is 9.375% — UNDER threshold, ACCEPTS.
        let n = FREE_RUN_DECODE_TOKENS;
        assert_eq!(n, 128, "this boundary test assumes N=128");
        let outcome = cohort_run_with_oracle(mock_oracle_diverge_slot(3, 12)).unwrap();
        assert!(
            outcome.candidate_accepted,
            "12/128 (9.375%) is under the 10% per-stream bar — must accept"
        );
    }

    #[test]
    fn cohort_tolerance_gate_rejects_one_stream_over_threshold_naming_the_class() {
        // 13/128 = 10.16% in slot 3 ONLY — OVER the 10% per-stream bar; the WHOLE run is rejected
        // (die-5), proving a single stream over threshold rejects the cohort.
        let outcome = cohort_run_with_oracle(mock_oracle_diverge_slot(3, 13)).unwrap();
        assert!(
            !outcome.candidate_accepted,
            "13/128 (10.16%) in one stream must reject the whole run"
        );
        let rejects = &outcome.results.rejected_pairs;
        assert!(
            !rejects.is_empty(),
            "a tolerance failure must record a reject"
        );
        assert_eq!(
            rejects[0].class, "cohort-token-tolerance",
            "the tolerance reject carries its own provenance class"
        );
        assert!(
            rejects[0].reason.contains("stream 3"),
            "the reject names the failing slot: {}",
            rejects[0].reason
        );
    }

    #[test]
    fn cohort_tolerance_gate_is_per_stream_not_cohort_average() {
        // Slot 0 at 104/128 (81%) mismatch, every other slot perfect. A per-COHORT-AVERAGE gate
        // would see 104 / (8*128) ≈ 10.2% and be near the line; more importantly the per-STREAM gate
        // rejects on slot 0 unconditionally. (The pure-function test proves the averaging mutation
        // directly; this proves the wired gate rejects at the run level.)
        let outcome = cohort_run_with_oracle(mock_oracle_diverge_slot(0, 104)).unwrap();
        assert!(!outcome.candidate_accepted);
        assert_eq!(
            outcome.results.rejected_pairs[0].class,
            "cohort-token-tolerance"
        );
        assert!(outcome.results.rejected_pairs[0]
            .reason
            .contains("stream 0"));
    }

    #[test]
    fn cohort_replay_integrity_n2_echo_mismatch_is_distinct_class() {
        // N2 — the oracle echoes a WRONG committed_token at slot 2 position 5 (it replayed a
        // different journal). This is a HARD INTEGRITY error under a DISTINCT class, NOT a tolerance
        // decision (even though only ONE token differs, which the tolerance bar alone would pass).
        let oracle = move |_seeds: &[Vec<i64>], committed: &[Vec<i64>]| {
            let streams = committed
                .iter()
                .enumerate()
                .map(
                    |(slot, tokens)| bench_protocol::CohortReferenceReplayStream {
                        slot: slot as i64,
                        positions: tokens
                            .iter()
                            .enumerate()
                            .map(|(pos, &t)| {
                                // The oracle echoes a DIFFERENT committed token than the candidate at 2,5.
                                let echoed = if slot == 2 && pos == 5 { t + 1 } else { t };
                                bench_protocol::CohortReferenceReplayPosition {
                                    committed_token: echoed,
                                    sequential_argmax: t,
                                    ..Default::default()
                                }
                            })
                            .collect(),
                    },
                )
                .collect();
            Ok(bench_protocol::CohortReferenceReplayReport {
                streams,
                ..Default::default()
            })
        };
        let outcome = cohort_run_with_oracle(oracle).unwrap();
        assert!(!outcome.candidate_accepted, "an N2 breach fails closed");
        assert_eq!(
            outcome.results.rejected_pairs[0].class, "cohort-replay-integrity",
            "N2 is a DISTINCT integrity class, never folded into a tolerance reject"
        );
        assert!(outcome.results.rejected_pairs[0].reason.contains("N2"));
    }

    #[test]
    fn cohort_oracle_transport_failure_is_integrity_reject_not_a_pass() {
        // If the trusted oracle cannot be reached at all, the run FAILS CLOSED (integrity reject) —
        // never silently accepted (which would skip the correctness bar entirely).
        let oracle = |_seeds: &[Vec<i64>], _committed: &[Vec<i64>]| {
            Err(RunnerError::Protocol(
                "trusted oracle unreachable".to_string(),
            ))
        };
        let outcome = cohort_run_with_oracle(oracle).unwrap();
        assert!(
            !outcome.candidate_accepted,
            "an unreachable oracle fails closed, never passes"
        );
        assert_eq!(
            outcome.results.rejected_pairs[0].class,
            "cohort-replay-integrity"
        );
    }

    // -----------------------------------------------------------------------
    // NEAR-TIE STATS SEAL (report-only) — David's "hold 10% + seal near-tie stats" ruling.
    // -----------------------------------------------------------------------

    /// The envelope the RICH mock oracle declares; every near-tie assertion below is relative to it.
    const NEAR_TIE_MOCK_ENVELOPE: f64 = 0.05;
    /// A top-2 relative gap INSIDE `NEAR_TIE_MOCK_ENVELOPE` — a flippable near-tie.
    const NEAR_TIE_GAP: f64 = 0.01;
    /// A top-2 relative gap far OUTSIDE the envelope — a confident reference position.
    const CONFIDENT_GAP: f64 = 0.9;

    /// What the RICH mock oracle emits, so each absent-field test names ONE omission.
    #[derive(Clone, Copy, PartialEq)]
    enum GapEmission {
        /// The modern engine: `rel_envelope` + a K=4 ranked readout + `committed_relative_gap`.
        Full,
        /// The OLD engine: the two REQUIRED fields only, every audit field absent
        /// (`..Default::default()`, exactly like [`mock_oracle_pass`]).
        None,
        /// Gaps present, but the report omits the report-level `rel_envelope`.
        NoEnvelope,
        /// A ranked readout at K=1 — no index 1, so no top-1→top-2 gap exists.
        ShallowTopK,
        /// Ranked gaps present, but no `committed_relative_gap` on any position.
        NoCommittedGap,
    }

    /// A MOCK trusted oracle with the AUDIT gap fields, driving the near-tie seal on KNOWN values.
    ///
    /// Shape (per stream of 128 committed positions):
    ///  * MISMATCH: the first `mismatches` positions of `bad_slot` report a divergent
    ///    `sequential_argmax` (`+7`); every other position matches. N2 is always truthful.
    ///  * NEAR-TIE: every position with `index % 4 == 0` gets top-2 gap [`NEAR_TIE_GAP`] (inside
    ///    the envelope), all others [`CONFIDENT_GAP`]. So 32 of 128 positions per stream are
    ///    near-ties, INDEPENDENT of whether they mismatched — the seal's overlap arithmetic has to
    ///    intersect two DIFFERENT sets, not read one twice.
    ///  * `committed_relative_gap`: `(index + 1) / 64` on a MISMATCH (exact binary fractions, so
    ///    min/median compare exactly), `0.0` on a match (committed IS the reference argmax).
    ///  * `ranked_relative_gaps` is `[0.0, top2, top2 + 0.1, top2 + 0.2]` — index 0 is identically
    ///    0.0 (the engine's own formula) and index 2 is the top-1→top-3 gap, so reading either
    ///    instead of index 1 yields a DIFFERENT near-tie count.
    fn mock_oracle_with_gaps(
        bad_slot: usize,
        mismatches: usize,
        emission: GapEmission,
    ) -> impl FnMut(
        &[Vec<i64>],
        &[Vec<i64>],
    ) -> bench_runner::Result<bench_protocol::CohortReferenceReplayReport> {
        move |_seeds: &[Vec<i64>], committed: &[Vec<i64>]| {
            let streams = committed
                .iter()
                .enumerate()
                .map(
                    |(slot, tokens)| bench_protocol::CohortReferenceReplayStream {
                        slot: slot as i64,
                        positions: tokens
                            .iter()
                            .enumerate()
                            .map(|(index, &t)| {
                                let mismatched = slot == bad_slot && index < mismatches;
                                let top2 = if index % 4 == 0 {
                                    NEAR_TIE_GAP
                                } else {
                                    CONFIDENT_GAP
                                };
                                let ranked = match emission {
                                    GapEmission::None => None,
                                    GapEmission::ShallowTopK => Some(vec![0.0]),
                                    _ => Some(vec![0.0, top2, top2 + 0.1, top2 + 0.2]),
                                };
                                let committed_gap = match emission {
                                    GapEmission::None | GapEmission::NoCommittedGap => None,
                                    _ if mismatched => Some((index as f64 + 1.0) / 64.0),
                                    _ => Some(0.0),
                                };
                                bench_protocol::CohortReferenceReplayPosition {
                                    // N2 echo is ALWAYS truthful.
                                    committed_token: t,
                                    sequential_argmax: if mismatched { t + 7 } else { t },
                                    ranked_relative_gaps: ranked,
                                    committed_relative_gap: committed_gap,
                                    ..Default::default()
                                }
                            })
                            .collect(),
                    },
                )
                .collect();
            let (provenance, topk, envelope) = match emission {
                GapEmission::None => (None, None, None),
                GapEmission::NoEnvelope => (Some("post_softcap".to_string()), Some(4), None),
                GapEmission::ShallowTopK => (
                    Some("post_softcap".to_string()),
                    Some(1),
                    Some(NEAR_TIE_MOCK_ENVELOPE),
                ),
                _ => (
                    Some("post_softcap".to_string()),
                    Some(4),
                    Some(NEAR_TIE_MOCK_ENVELOPE),
                ),
            };
            Ok(bench_protocol::CohortReferenceReplayReport {
                logit_provenance: provenance,
                logit_topk: topk,
                rel_envelope: envelope,
                streams,
                ..Default::default()
            })
        }
    }

    /// The seal on the run's FIRST accepted pair (the near-tie tests all seal identically on every
    /// accepted pair — one pair is one oracle report, and the mock is deterministic).
    fn first_pair_near_tie_seal(outcome: &MeasureJobOutcome) -> &CohortNearTieSeal {
        outcome.results.pairs[0]
            .cohort_near_tie_seal
            .as_ref()
            .expect("a batched pair seals its near-tie characterization")
    }

    #[test]
    fn near_tie_seal_states_exact_counts_overlap_and_headroom() {
        // (a) EXACT-VALUE seal test. Slot 3 mismatches on its first 12 of 128 positions (9.375% —
        // the honest-drama case, UNDER the 10% bar, so the run ACCEPTS and a seal exists).
        //
        // Near-ties are every 4th position: 32 per stream, 256 across the B=8 cohort. Of slot 3's
        // 12 mismatched positions (indices 0..11), exactly indices 0, 4, 8 are near-ties → the
        // OVERLAP is 3 and the GENUINE-divergence remainder is 9. Reading gap index 0 instead of 1
        // would report 128 near-ties per stream (index 0 is identically 0.0); reading index 2
        // would report 0 (0.11 > 0.05); a strict `<` at the envelope, or intersecting the wrong
        // way, all move these numbers.
        let n = FREE_RUN_DECODE_TOKENS;
        assert_eq!(n, 128, "this exact-value test assumes N=128");
        let outcome = cohort_run_with_oracle(mock_oracle_with_gaps(3, 12, GapEmission::Full))
            .expect("a conformant cohort run");
        assert!(
            outcome.candidate_accepted,
            "12/128 = 9.375% is under the 10% bar — the run accepts and seals"
        );
        let seal = first_pair_near_tie_seal(&outcome);
        assert_eq!(seal.logit_provenance.as_deref(), Some("post_softcap"));
        assert_eq!(seal.logit_topk, Some(4));
        assert_eq!(seal.rel_envelope, Some(NEAR_TIE_MOCK_ENVELOPE));
        assert_eq!(seal.near_tie_refused, None, "a full report refuses nothing");
        let stats = seal.stats.as_ref().expect("a full report seals stats");

        // The DEFINITION, sealed so no consumer has to infer it.
        assert_eq!(stats.rel_envelope, NEAR_TIE_MOCK_ENVELOPE);
        assert_eq!(
            stats.near_tie_gap_index, 1,
            "index 1 of ranked_relative_gaps is the top-1→top-2 gap"
        );
        assert_eq!(
            stats.near_tie_predicate,
            "ranked_relative_gaps[1] <= rel_envelope"
        );

        // The DIVERGENT stream, exactly.
        assert_eq!(stats.per_stream.len(), SCORED_BATCH_SIZE_B8 as usize);
        let bad = &stats.per_stream[3];
        assert_eq!(bad.slot, 3);
        assert_eq!(bad.committed, 128);
        assert_eq!(bad.mismatches, 12);
        assert_eq!(bad.mismatch_per_thousand, 93.75, "12/128 * 1000");
        assert_eq!(
            bad.near_tie_positions, 32,
            "every 4th of 128 positions — a REFERENCE property, matched or not"
        );
        assert_eq!(
            bad.near_tie_mismatches, 3,
            "the OVERLAP: mismatched indices 0, 4, 8 are the near-ties among 0..11"
        );
        assert_eq!(
            bad.non_near_tie_mismatches, 9,
            "GENUINE divergence — the reference was confident and the candidate diverged anyway"
        );
        // committed_relative_gap over the 12 MISMATCHED positions: (i+1)/64 for i in 0..11, i.e.
        // 1/64 .. 12/64. min = 1/64; even-n median = mean of the 6th and 7th = (6/64 + 7/64)/2.
        assert_eq!(bad.min_committed_relative_gap_on_mismatch, Some(1.0 / 64.0));
        assert_eq!(
            bad.median_committed_relative_gap_on_mismatch,
            Some(13.0 / 128.0)
        );

        // Every OTHER stream: clean, but still carrying its own near-tie census, and with the two
        // mismatch-depth statistics ABSENT rather than fabricated as 0.0.
        for (slot, s) in stats.per_stream.iter().enumerate() {
            assert_eq!(s.slot, slot);
            assert_eq!(s.committed, 128);
            if slot == 3 {
                continue;
            }
            assert_eq!(s.mismatches, 0, "slot {slot} is clean");
            assert_eq!(s.mismatch_per_thousand, 0.0);
            assert_eq!(s.near_tie_positions, 32, "slot {slot} still has near-ties");
            assert_eq!(s.near_tie_mismatches, 0);
            assert_eq!(s.non_near_tie_mismatches, 0);
            assert_eq!(s.min_committed_relative_gap_on_mismatch, None);
            assert_eq!(s.median_committed_relative_gap_on_mismatch, None);
        }

        // COHORT totals + the HEADROOM stat.
        assert_eq!(stats.committed_total, 8 * 128);
        assert_eq!(stats.mismatches_total, 12);
        assert_eq!(stats.near_tie_positions_total, 8 * 32);
        assert_eq!(stats.near_tie_mismatches_total, 3);
        assert_eq!(stats.non_near_tie_mismatches_total, 9);
        assert_eq!(
            stats.budget_per_thousand,
            bench_core::constants::COHORT_TOKEN_TOLERANCE_PER_THOUSAND
        );
        assert_eq!(
            stats.max_stream_mismatch_per_thousand, 93.75,
            "the WORST stream, not the cohort average"
        );
        assert_eq!(
            stats.headroom_per_thousand, 6.25,
            "93.75 against the 100/1000 budget — the honest-submission creep warning"
        );
        // The cohort AVERAGE would read 12/1024*1000 ≈ 11.7 and hide the creep entirely; the
        // headroom stat is per-stream because the GATE is per-stream.
        let average = 12.0 * 1000.0 / 1024.0;
        assert!(
            stats.max_stream_mismatch_per_thousand > 8.0 * average - 1.0,
            "sanity: the max is the per-stream reading, not the average ({average})"
        );

        // WHERE IT LANDS in results.json: `pairs[i].cohort_near_tie_seal`, a SIBLING of the same
        // pair's other report-only seals — values only, no verdict field anywhere in it.
        let json = serde_json::to_value(&outcome.results).unwrap();
        let sealed = &json["pairs"][0]["cohort_near_tie_seal"];
        assert_eq!(sealed["rel_envelope"], json!(NEAR_TIE_MOCK_ENVELOPE));
        assert_eq!(sealed["stats"]["near_tie_gap_index"], json!(1));
        assert_eq!(
            sealed["stats"]["per_stream"][3]["near_tie_mismatches"],
            json!(3)
        );
        assert_eq!(
            sealed["stats"]["per_stream"][3]["non_near_tie_mismatches"],
            json!(9)
        );
        assert_eq!(
            sealed["stats"]["max_stream_mismatch_per_thousand"],
            json!(93.75)
        );
        assert!(
            sealed["near_tie_refused"].is_null(),
            "a computed seal carries no refusal"
        );
    }

    /// Recursively delete every `cohort_near_tie_seal` key from a serialized artifact, so the
    /// REMAINDER can be compared byte-for-byte against a run that never computed one.
    fn strip_near_tie_seal(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                map.remove("cohort_near_tie_seal");
                for v in map.values_mut() {
                    strip_near_tie_seal(v);
                }
            }
            serde_json::Value::Array(items) => {
                for v in items.iter_mut() {
                    strip_near_tie_seal(v);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn near_tie_seal_is_a_behavioral_no_op_across_accept_and_every_reject_class() {
        // (b) REPORT-ONLY PIN — the no-op PROOF. For each scenario, run the SAME cohort twice: once
        // with an oracle emitting the gap fields (⇒ the seal computes real stats) and once with the
        // identical oracle emitting NONE of them (⇒ no seal computation happens at all). Then
        // assert the two artifacts are IDENTICAL once the additive seal is stripped — same
        // candidate_accepted, same accepted pairs, same reject classes and reasons, same every
        // scored number. Nothing the seal computes can reach an outcome.
        //
        // The scenarios span the ACCEPT path and BOTH reject classes the gate can produce, so a
        // "the seal only matters on rejects" or "only on accepts" bug has nowhere to hide.
        let scenarios: [(&str, usize, usize); 4] = [
            ("clean accept", 0, 0),
            ("9.375% accept (under the bar)", 3, 12),
            ("10.16% tolerance reject", 3, 13),
            ("81% tolerance reject", 0, 104),
        ];
        for (name, bad_slot, mismatches) in scenarios {
            let with_seal = cohort_run_with_oracle(mock_oracle_with_gaps(
                bad_slot,
                mismatches,
                GapEmission::Full,
            ))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            let without_seal = cohort_run_with_oracle(mock_oracle_with_gaps(
                bad_slot,
                mismatches,
                GapEmission::None,
            ))
            .unwrap_or_else(|e| panic!("{name}: {e}"));

            assert_eq!(
                with_seal.candidate_accepted, without_seal.candidate_accepted,
                "{name}: candidate_accepted must not depend on the near-tie seal"
            );
            let mut a = serde_json::to_value(&with_seal.results).unwrap();
            let mut b = serde_json::to_value(&without_seal.results).unwrap();
            if with_seal.candidate_accepted {
                // The seal really WAS computed on the gap-emitting side, and really was NOT on the
                // gap-free side — otherwise the comparison below would pass vacuously by
                // comparing two identical seal-free artifacts.
                assert!(
                    first_pair_near_tie_seal(&with_seal).stats.is_some(),
                    "{name}: the gap-emitting run must actually seal stats for this pin to mean \
                     anything"
                );
                assert!(
                    without_seal.results.pairs[0]
                        .cohort_near_tie_seal
                        .as_ref()
                        .expect("an old-engine run still seals its named refusal")
                        .stats
                        .is_none(),
                    "{name}: the gap-free run computes no stats"
                );
                assert_ne!(
                    a, b,
                    "{name}: before the strip the seal IS the observable difference"
                );
            } else {
                // A rejected run accepts no pair, so it seals no PairRecord and therefore no
                // near-tie stats at all. The two artifacts must ALREADY be identical — the
                // strongest form of the no-op claim on the reject path.
                assert!(
                    with_seal.results.pairs.is_empty(),
                    "{name}: a die-5 run accepts no pair"
                );
                assert_eq!(
                    a, b,
                    "{name}: on the reject path the two artifacts are identical BEFORE any strip \
                     — the reject class and reason cannot depend on the seal"
                );
            }
            strip_near_tie_seal(&mut a);
            strip_near_tie_seal(&mut b);
            assert_eq!(
                a, b,
                "{name}: with the additive seal removed the two artifacts must be IDENTICAL — no \
                 accept/reject outcome, reject class, reason, or scored number may differ"
            );
        }
    }

    #[test]
    fn near_tie_seal_refuses_by_name_on_an_engine_without_the_gap_fields_and_the_gate_still_gates()
    {
        // (c) ABSENT OPTIONAL FIELDS. Each emission below omits ONE thing; the seal must name
        // exactly what it could not read, seal NO stats, and leave the ≤10% tolerance gate working
        // on `sequential_argmax` alone.
        let cases: [(GapEmission, &str); 4] = [
            // The OLD ENGINE: no audit fields at all.
            (GapEmission::None, "ranked_relative_gaps"),
            (GapEmission::NoEnvelope, "rel_envelope"),
            (GapEmission::ShallowTopK, "index 1"),
            (GapEmission::NoCommittedGap, "committed_relative_gap"),
        ];
        for (emission, needle) in cases {
            // An UNDER-threshold run so a pair accepts and a seal exists to inspect.
            let accepted = cohort_run_with_oracle(mock_oracle_with_gaps(3, 12, emission)).unwrap();
            assert!(
                accepted.candidate_accepted,
                "{needle}: the tolerance gate needs only committed_token + sequential_argmax — an \
                 engine without the audit gap fields must still be judged normally"
            );
            let seal = first_pair_near_tie_seal(&accepted);
            assert!(
                seal.stats.is_none(),
                "{needle}: no near-tie statistic may be FABRICATED when the input is absent"
            );
            let reason = seal
                .near_tie_refused
                .as_ref()
                .unwrap_or_else(|| panic!("{needle}: an absent input must be refused BY NAME"));
            assert!(
                reason.contains(needle),
                "{needle}: the refusal must name the missing field, got: {reason}"
            );
            // ABSENT, not null-or-zero: the `stats` key is omitted from the artifact entirely.
            let json = serde_json::to_value(&accepted.results).unwrap();
            let sealed = &json["pairs"][0]["cohort_near_tie_seal"];
            assert!(
                sealed.get("stats").is_none(),
                "{needle}: a refused seal omits `stats` rather than emitting a null/zero shape"
            );
            assert!(sealed["near_tie_refused"].is_string());

            // And the GATE still REJECTS over the bar on the very same emission — the seal's
            // absence neither loosens nor tightens the ≤10% decision.
            let rejected = cohort_run_with_oracle(mock_oracle_with_gaps(3, 13, emission)).unwrap();
            assert!(
                !rejected.candidate_accepted,
                "{needle}: 13/128 = 10.16% must still reject without any gap field"
            );
            assert_eq!(
                rejected.results.rejected_pairs[0].class, "cohort-token-tolerance",
                "{needle}: the reject class is unchanged by the seal"
            );
        }
    }

    #[test]
    fn near_tie_seal_is_absent_on_the_single_stream_regime() {
        // The single-stream path runs NO trusted-oracle replay, so there is no report to
        // characterize: the field is OMITTED, never an empty or fabricated object.
        let outcome = identity_run(&test_cfg(2, 2));
        assert!(outcome.candidate_accepted);
        let json = serde_json::to_value(&outcome.results).unwrap();
        for pair in json["pairs"].as_array().expect("pairs") {
            assert!(
                pair.get("cohort_near_tie_seal").is_none(),
                "a single-stream pair record must be byte-unchanged by this lane"
            );
        }
    }

    #[test]
    fn trusted_oracle_resolver_fails_closed_and_never_falls_back_to_candidate_bin() {
        // ★ ANTI-GAMING LINCHPIN (proven, not just documented): with the trusted-oracle env UNSET,
        // the resolver returns an ERROR and NEVER yields the candidate worker bin. A silent fallback
        // to the candidate build would let candidate-editable forward code produce the reference
        // argmax — total anti-gaming collapse.
        let saved = std::env::var(TRUSTED_ORACLE_WORKER_BIN_ENV).ok();

        // UNSET → hard error whose message names the missing configuration; the returned value is
        // NEVER the candidate bin (there is no Ok path at all when unset).
        std::env::remove_var(TRUSTED_ORACLE_WORKER_BIN_ENV);
        let err = resolve_trusted_oracle_worker_bin()
            .expect_err("an unset trusted-oracle env must FAIL CLOSED, never resolve");
        assert!(
            err.contains(TRUSTED_ORACLE_WORKER_BIN_ENV),
            "the error must name the missing env var: {err}"
        );
        assert!(
            !err.contains(&format!("Ok({DEFAULT_MEASURE_WORKER_BIN}")),
            "sanity: no accidental Ok"
        );

        // EMPTY → same fail-closed (a blank env is treated as unset, never as a bin path).
        std::env::set_var(TRUSTED_ORACLE_WORKER_BIN_ENV, "   ");
        assert!(
            resolve_trusted_oracle_worker_bin().is_err(),
            "a blank trusted-oracle env must also FAIL CLOSED"
        );

        // SET to a distinct trusted path → resolves to EXACTLY that, and NEVER the candidate bin.
        // (Structural: the resolver reads only this env; it shares no branch with the candidate
        // resolver, so DEFAULT_MEASURE_WORKER_BIN is unreachable from here.)
        let trusted = "/organizer/trusted/mlxfast-runtime-worker-TRUSTED";
        std::env::set_var(TRUSTED_ORACLE_WORKER_BIN_ENV, trusted);
        let resolved = resolve_trusted_oracle_worker_bin().expect("a set env resolves");
        assert_eq!(
            resolved, trusted,
            "resolves to exactly the organizer-provided bin"
        );
        assert_ne!(
            resolved, DEFAULT_MEASURE_WORKER_BIN,
            "the oracle bin is NEVER the candidate/baseline default bin"
        );

        // Restore prior env so subsequent tests are unaffected.
        match saved {
            Some(v) => std::env::set_var(TRUSTED_ORACLE_WORKER_BIN_ENV, v),
            None => std::env::remove_var(TRUSTED_ORACLE_WORKER_BIN_ENV),
        }
    }

    #[test]
    fn cohort_membership_seals_the_member_list_in_pool_order() {
        let goldens = cohort_goldens();
        let pool = cohort_pool(&goldens);
        let members = validate_cohort_membership(&goldens, &pool, SCORED_BATCH_SIZE_B8).unwrap();
        assert_eq!(members.len(), COHORT_B);
        for (i, m) in members.iter().enumerate() {
            assert_eq!(m.slot_index, i, "slot order is pool order");
            assert_eq!(m.prompt_sha256, goldens[i].sha256().to_ascii_lowercase());
            assert_eq!(m.bytes, goldens[i].byte_len());
        }
        // The derived cohort identity is deterministic and recomputable.
        assert_eq!(cohort_sha256(&members), cohort_sha256(&members));
    }

    #[test]
    fn cohort_membership_negative_controls() {
        let goldens = cohort_goldens();
        let pool = cohort_pool(&goldens);

        // A PERMUTED cohort is a different pinned identity: refused (slot order IS pool order).
        let mut permuted = goldens.clone();
        permuted.swap(2, 5);
        let err = validate_cohort_membership(&permuted, &pool, SCORED_BATCH_SIZE_B8).unwrap_err();
        assert!(err.contains("SLOT ORDER IS POOL ORDER"), "{err}");

        // A sha-only pool pin (no `bytes`) is not a cohort pin: half the identity is unverified.
        let mut sha_only = pool.clone();
        sha_only[3].bytes = None;
        let err =
            validate_cohort_membership(&goldens, &sha_only, SCORED_BATCH_SIZE_B8).unwrap_err();
        assert!(err.contains("pins no `bytes`"), "{err}");

        // A byte-half mismatch refuses (the two halves must agree).
        let mut wrong_bytes = pool.clone();
        wrong_bytes[0].bytes = Some(1);
        let err =
            validate_cohort_membership(&goldens, &wrong_bytes, SCORED_BATCH_SIZE_B8).unwrap_err();
        assert!(err.contains("must agree"), "{err}");

        // A subset cohort (7 goldens against B=8) refuses.
        let err =
            validate_cohort_membership(&goldens[..7], &pool, SCORED_BATCH_SIZE_B8).unwrap_err();
        assert!(err.contains("carries 7 goldens"), "{err}");

        // A pool whose cardinality is not the declared width refuses (the cohort IS the pool).
        let err =
            validate_cohort_membership(&goldens, &pool[..7], SCORED_BATCH_SIZE_B8).unwrap_err();
        assert!(err.contains("pins 7 prompts"), "{err}");

        // A duplicate pool pin has no well-defined composition.
        let mut dup = pool.clone();
        dup[1].sha256 = dup[0].sha256.clone();
        let err = validate_cohort_membership(&goldens, &dup, SCORED_BATCH_SIZE_B8).unwrap_err();
        assert!(err.contains("DUPLICATE"), "{err}");
    }

    #[test]
    fn effective_candidate_regime_is_fixture_driven_and_ruled() {
        // No declared width ⇒ the spec-derived regime stands.
        assert_eq!(
            effective_candidate_regime(SPEC_MODE_MTP, LegRegime::FreeRunV1_1, None).unwrap(),
            LegRegime::FreeRunV1_1
        );
        // The ruled B=8 upgrades a speculating candidate to the batched cohort regime.
        assert_eq!(
            effective_candidate_regime(SPEC_MODE_MTP, LegRegime::FreeRunV1_1, Some(8)).unwrap(),
            b8_regime()
        );
        // A serial candidate at batch 8 is D0 alt (c): ratio 1.0 by construction — refused.
        let err = effective_candidate_regime(SPEC_MODE_SERIAL, LegRegime::TeacherForcedV1, Some(8))
            .unwrap_err();
        assert!(err.contains("1.0 by construction"), "{err}");
        // Any other width is not the ruled batch point — refused, never silently run.
        let err =
            effective_candidate_regime(SPEC_MODE_MTP, LegRegime::FreeRunV1_1, Some(4)).unwrap_err();
        assert!(err.contains("ONE ruled batch point"), "{err}");
    }

    #[test]
    fn certify_b8_is_interlocked_on_the_captured_cohort_crosscheck() {
        // ANGLE-6 BINDING INTERLOCK — the refusal branch, proven through the module-private
        // seam (`certify_with_coverage`) so no fake fixture file is needed: with the cohort
        // coverage fact FALSE, the ruled B=8 point itself refuses, and the refusal names the
        // missing-cohort-crosscheck condition (a structural refusal, not a schedule).
        let err = ScoredBatchPoint::certify_with_coverage(SCORED_BATCH_SIZE_B8, false)
            .expect_err("B=8 without the cohort crosscheck must refuse");
        assert!(err.contains("cohort crosscheck exists"), "{err}");
        assert!(err.contains("angle-6"), "{err}");
        assert!(
            err.contains("BOTH repos"),
            "the refusal must say how to fix it (extend + repin both repos): {err}"
        );

        // With coverage TRUE the certify-match stands unchanged.
        let point = ScoredBatchPoint::certify_with_coverage(SCORED_BATCH_SIZE_B8, true).unwrap();
        assert_eq!(point.batch_size(), SCORED_BATCH_SIZE_B8);
        assert_eq!(point.timed_mode(), "batched_free_run_v1_2_b8");

        // The coverage gate is ADDITIONAL, not a replacement: an uncertified width refuses on
        // the exhaustive match regardless of the coverage fact — in BOTH positions.
        for covered in [false, true] {
            let err = ScoredBatchPoint::certify_with_coverage(4, covered).unwrap_err();
            assert!(err.contains("ONE ruled batch point"), "{err}");
        }

        // And the PUBLIC path — which always consults the REAL embedded fixture — passes today,
        // because the 2026-08-23 joint repin extended the capture with the cohort lines
        // (`bench_runner::captured_fixture_covers_cohort_wire` is the consulted fact).
        assert!(bench_runner::captured_fixture_covers_cohort_wire());
        assert_eq!(
            ScoredBatchPoint::certify(SCORED_BATCH_SIZE_B8).unwrap(),
            point
        );
    }

    #[test]
    fn cohort_positive_control_seals_the_cohort_shape() {
        // min 4, RULED target 4 (2026-08-26, superseding the 2026-08-24 ruling of 2): a conformant
        // run accepts exactly pairs_per_cohort = 4 pairs of the ONE cohort and seals the cohort
        // shape. min == target mirrors the engine wrapper, which passes --min-pairs 4
        // --target-pairs 4.
        let outcome = cohort_identity_run(&cohort_cfg(
            PAIRS_PER_COHORT_TARGET,
            PAIRS_PER_COHORT_TARGET,
        ))
        .unwrap();
        assert!(outcome.candidate_accepted);
        let r = &outcome.results;
        assert_eq!(r.mode, COHORT_MEASURE_JOB_MODE);
        // D5 — the series tag carries the batch width; both legs homogeneous in it.
        assert_eq!(r.timed_mode, "batched_free_run_v1_2_b8");
        assert_eq!(
            r.timed_series.serial_leg_timed_mode,
            "batched_free_run_v1_2_b8"
        );
        assert_eq!(
            r.timed_series.candidate_leg_timed_mode,
            "batched_free_run_v1_2_b8"
        );
        assert!(r.timed_series.homogeneous && r.timed_series.legs_comparable);
        assert_eq!(r.timed_regime, Some("batched-free-run-v1_2-timed"));
        // The cohort seal: one cohort, 8 members in pool order, width pinned.
        assert_eq!(r.scored_batch_size, Some(8));
        assert_eq!(r.pairs_per_cohort, Some(PAIRS_PER_COHORT_TARGET));
        assert_eq!(r.min_pairs_per_cohort, Some(PAIRS_PER_COHORT_TARGET));
        assert!(
            r.per_prompt.is_empty(),
            "per_prompt is replaced by per_cohort"
        );
        let cohorts = r.per_cohort.as_ref().expect("per_cohort sealed");
        assert_eq!(cohorts.len(), 1);
        let c = &cohorts[0];
        assert_eq!(c.batch_size, 8);
        assert_eq!(c.members.len(), COHORT_B);
        assert_eq!(c.accepted_pair_count, PAIRS_PER_COHORT_TARGET);
        assert_eq!(
            c.cohort_sha256,
            cohort_sha256(&c.members),
            "recomputable identity"
        );
        // prompt_count still counts the distinct pool prompts timed (all 8, concurrently).
        assert_eq!(r.prompt_count, COHORT_B);
        // Every pair is stamped with the COHORT identity, never a blank.
        assert_eq!(r.pairs.len(), PAIRS_PER_COHORT_TARGET);
        for p in &r.pairs {
            assert_eq!(p.prompt_sha256, c.cohort_sha256);
            assert_eq!(p.prompt_index, 0);
            // The cohort audit diagnostics ride on the pair, sealed verbatim.
            assert!(p.audit_cohort_active_streams_by_round.is_some());
            assert!(p.audit_cohort_natural_accepted_by_stream.is_some());
            assert_eq!(
                p.audit_cohort_depth_clamp_reasons.as_ref().unwrap()["tail_depth"],
                2
            );
            assert!(p.audit_spec.contains_key("audit_cohort_batch_size"));
        }
        // D2 — the published median is the even-n median over the accepted PAIRS' cohort ratios.
        assert_eq!(r.aggregate.raw_ratios.len(), PAIRS_PER_COHORT_TARGET);
        let expected_ratio = SERIAL_SPT / CANDIDATE_SPT;
        assert!((r.aggregate.raw_decode_speedup_median - expected_ratio).abs() < 1e-12);
        // D1 — the free-run floor/ceiling machinery applies unchanged (0.90 / 5.0).
        assert_eq!(
            r.aggregate.decode_speedup_floor,
            FREE_RUN_DECODE_SPEEDUP_FLOOR
        );
        assert!(r.aggregate.decode_speedup_floor_met);
        assert_eq!(
            r.aggregate.published_speedup_ceiling,
            PUBLISHED_SPEEDUP_CEILING
        );
        // The alternation still keys on the accepted-pair index, across ALL 4 ruled pairs.
        let expected_orders = ["mtp-first", "serial-first", "mtp-first", "serial-first"];
        assert_eq!(r.pairs.len(), expected_orders.len());
        for (pair, expected) in r.pairs.iter().zip(expected_orders) {
            assert_eq!(pair.order, expected);
        }
    }

    #[test]
    fn cohort_official_run_requires_the_ruled_pairs_target() {
        // D2 — an OFFICIAL cohort run at a non-ruled target refuses before measuring; local-dev
        // may explore.
        let err = cohort_identity_run(&cohort_cfg(2, 3))
            .err()
            .expect("a non-ruled official pairs target must refuse");
        assert!(err.contains("RULED pairs_per_cohort"), "{err}");
        let mut local = cohort_cfg(2, 3);
        local.local_pair_budget = true;
        assert!(cohort_identity_run(&local).is_ok());
    }

    #[test]
    fn cohort_official_run_refuses_the_superseded_pairs_target_of_two() {
        // MUTATION PROOF for the 2026-08-26 ruling: the conformance gate did not merely move, it
        // RETARGETED. 2 was the RULED value under the superseded 2026-08-24 ruling and would have
        // passed this gate before the change; an official run declaring it must now REFUSE, and
        // the diagnostic must name the NEW target so an operator running the old wrapper is told
        // what to change. Local-dev may still explore 2.
        // Compile-time: this proof is vacuous unless the ruled target has moved off the superseded
        // 2, so a revert should fail the BUILD rather than leave a green test asserting nothing.
        const { assert!(PAIRS_PER_COHORT_TARGET != 2) };
        let err = cohort_identity_run(&cohort_cfg(2, 2))
            .err()
            .expect("the superseded target of 2 must refuse on an official run");
        assert!(err.contains("RULED pairs_per_cohort"), "{err}");
        assert!(err.contains("target_pairs 2"), "{err}");
        assert!(
            err.contains(&format!("target is {PAIRS_PER_COHORT_TARGET}")),
            "the refusal must name the ruled target: {err}"
        );
        let mut local = cohort_cfg(2, 2);
        local.local_pair_budget = true;
        assert!(cohort_identity_run(&local).is_ok());
    }

    #[test]
    fn cohort_official_run_requires_the_ruled_pairs_floor_not_just_the_target() {
        // THE FLOOR IS PART OF THE RULING. An official run at the ruled TARGET but a slack FLOOR
        // (min 2 / target 4) satisfies the parse-time `min_pairs <= target_pairs` check and passes
        // the target refusal above, yet would accept a 2-pair cohort and publish a median over half
        // the ruled support. It must refuse, by name, at the same pre-GPU seam.
        // Compile-time, so a future ruling that moved the target back to 2 would fail to BUILD
        // rather than leave this control silently vacuous (a floor of 2 must be genuinely below
        // the ruled target for the refusal below to be about the floor at all).
        const { assert!(PAIRS_PER_COHORT_TARGET > 2) };
        let err = cohort_identity_run(&cohort_cfg(2, PAIRS_PER_COHORT_TARGET))
            .err()
            .expect("an official run with a floor below the ruled target must refuse");
        assert!(err.contains("RULED pairs_per_cohort"), "{err}");
        assert!(err.contains("min_pairs 2"), "{err}");
        assert!(
            err.contains(&format!("floor is {PAIRS_PER_COHORT_TARGET}")),
            "the refusal must name the ruled floor: {err}"
        );

        // NON-VACUITY — the SAME run at the ruled floor accepts, so the refusal is about the floor
        // and not about some unrelated defect in this fixture.
        assert!(
            cohort_identity_run(&cohort_cfg(
                PAIRS_PER_COHORT_TARGET,
                PAIRS_PER_COHORT_TARGET
            ))
            .is_ok(),
            "min == target == the ruled count must still run"
        );

        // LOCAL-DEV IS NOT OVER-RESTRICTED — the dev path still explores a floor below its target.
        let mut local = cohort_cfg(2, PAIRS_PER_COHORT_TARGET);
        local.local_pair_budget = true;
        assert!(
            cohort_identity_run(&local).is_ok(),
            "--local-dev must keep min < target available"
        );
    }

    #[test]
    fn cohort_serial_control_structural_assertion_common_width_unit_and_r_equals_n() {
        // D3 — the batched serial control must demonstrate depth 0 STRUCTURALLY: common committed
        // width exactly 1 every round and R == N. A control whose cohort committed width 2 (R =
        // N/2) speculated; the pair rejects and the official run dies 5.
        let goldens = cohort_goldens();
        let members =
            validate_cohort_membership(&goldens, &cohort_pool(&goldens), SCORED_BATCH_SIZE_B8)
                .unwrap();
        let n = FREE_RUN_DECODE_TOKENS;
        let outcome = run_cohort_measure_job(
            &goldens,
            members,
            &DirDigest::empty(),
            "deadbeef",
            &cohort_cfg(PAIRS_PER_COHORT_TARGET, PAIRS_PER_COHORT_TARGET),
            CohortSeams {
                measure_serial: |_p: &CohortTimingParams| {
                    let mut inv = ok_cohort_serial()?;
                    inv.cohort_audit = Some(cohort_audit(&vec![2u32; n / 2], n));
                    Ok(inv)
                },
                measure_candidate: |_p: &CohortTimingParams| inv_cohort_candidate(CANDIDATE_SPT),
                oracle: mock_oracle_pass,
            },
        )
        .unwrap();
        assert!(!outcome.candidate_accepted, "official immediate die-5");
        let reject = &outcome.results.rejected_pairs[0];
        assert_eq!(reject.class, "free-run-serial-control-speculated");
        assert_eq!(reject.leg, "serial");
        assert!(
            reject.reason.contains("one token per round"),
            "the unit-histogram assertion names the fact: {}",
            reject.reason
        );
    }

    #[test]
    fn cohort_candidate_without_cohort_audit_rejects() {
        let goldens = cohort_goldens();
        let members =
            validate_cohort_membership(&goldens, &cohort_pool(&goldens), SCORED_BATCH_SIZE_B8)
                .unwrap();
        let outcome = run_cohort_measure_job(
            &goldens,
            members,
            &DirDigest::empty(),
            "deadbeef",
            &cohort_cfg(PAIRS_PER_COHORT_TARGET, PAIRS_PER_COHORT_TARGET),
            CohortSeams {
                measure_serial: |_p: &CohortTimingParams| ok_cohort_serial(),
                measure_candidate: |_p: &CohortTimingParams| {
                    let mut inv = inv_cohort_candidate(CANDIDATE_SPT)?;
                    inv.cohort_audit = None;
                    Ok(inv)
                },
                oracle: mock_oracle_pass,
            },
        )
        .unwrap();
        assert!(!outcome.candidate_accepted);
        let reject = &outcome.results.rejected_pairs[0];
        assert_eq!(reject.class, "free-run-consistency");
        assert!(
            reject.reason.contains("no cohort AUDIT"),
            "{}",
            reject.reason
        );
    }

    #[test]
    fn cohort_audit_at_the_wrong_width_rejects() {
        // D9 — a b8-series leg whose validated audit carries another width must never seal: the
        // series tag names B, and a number measured at B=4 under the b8 tag is a category error.
        let goldens = cohort_goldens();
        let members =
            validate_cohort_membership(&goldens, &cohort_pool(&goldens), SCORED_BATCH_SIZE_B8)
                .unwrap();
        let n = FREE_RUN_DECODE_TOKENS;
        let outcome = run_cohort_measure_job(
            &goldens,
            members,
            &DirDigest::empty(),
            "deadbeef",
            &cohort_cfg(PAIRS_PER_COHORT_TARGET, PAIRS_PER_COHORT_TARGET),
            CohortSeams {
                measure_serial: |_p: &CohortTimingParams| ok_cohort_serial(),
                measure_candidate: |_p: &CohortTimingParams| {
                    let mut inv = inv_cohort_candidate(CANDIDATE_SPT)?;
                    inv.cohort_audit = Some(cohort_audit_b(4, &vec![4u32; n / 4], n));
                    Ok(inv)
                },
                oracle: mock_oracle_pass,
            },
        )
        .unwrap();
        assert!(!outcome.candidate_accepted);
        let reject = &outcome.results.rejected_pairs[0];
        assert_eq!(reject.class, "free-run-consistency");
        assert!(reject.reason.contains("batch_size 4"), "{}", reject.reason);
    }

    #[test]
    fn single_stream_leg_carrying_a_cohort_audit_rejects() {
        // The two audit channels are regime-exclusive: a cohort audit on a single-stream free-run
        // leg is a fabricated claim about a window that never ran batched.
        let n = FREE_RUN_DECODE_TOKENS;
        let outcome = run_measure_job(
            &[measure_tape()],
            &DirDigest::empty(),
            "deadbeef",
            &free_run_cfg(1, 1),
            |_p| ok_free_run_serial(),
            |_p: &TimingParams| {
                let mut inv = inv_free_run(CANDIDATE_SPT, vec![4; n / 4], n)?;
                inv.cohort_audit = Some(cohort_audit(&vec![4u32; n / 4], n));
                Ok(inv)
            },
        )
        .unwrap();
        assert!(!outcome.candidate_accepted);
        let reject = &outcome.results.rejected_pairs[0];
        assert_eq!(reject.class, "free-run-consistency");
        assert!(reject.reason.contains("COHORT audit"), "{}", reject.reason);
    }

    #[test]
    fn cohort_serial_band_applies_under_the_b8_series_tag() {
        // D5 — the SAME serial-band machinery gates the cohort serial spt, fenced to the b8
        // series: a calibration authored under the single-stream free-run series is NOT
        // comparable and dies 6 at the pre-read fence; a b8-tagged one bands normally.
        let b8_tag = bench_core::free_run::TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8;
        let cal_file = BaselineCalibration {
            timed_mode: b8_tag.to_string(),
            track_id: "qwen3.8-27b-mtp-v1".to_string(),
            serial_decode_seconds_per_token_mean: Some(SERIAL_SPT),
            serial_band_low: 0.95,
            serial_band_high: 1.05,
            decode_tokens: Some(FREE_RUN_DECODE_TOKENS),
            targets: Default::default(),
        };
        // The fence: this run's series is the b8 tag (run_timed_mode over the batched regime).
        assert_eq!(run_timed_mode(b8_regime()), b8_tag);
        assert!(enforce_calibration_series_fence(&cal_file, b8_tag, "qwen3.8-27b-mtp-v1").is_ok());
        // A single-stream free-run calibration must NEVER band a b8 run (and vice versa).
        let err = enforce_calibration_series_fence(
            &cal_file,
            bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1,
            "qwen3.8-27b-mtp-v1",
        )
        .unwrap_err();
        assert!(err.contains("NOT comparable"), "{err}");

        // The band itself, applied to the pooled COHORT serial mean through the run: in band.
        let resolved = ResolvedCalibration {
            serial_mean: SERIAL_SPT,
            band_low: 0.95,
            band_high: 1.05,
            decode_tokens: Some(FREE_RUN_DECODE_TOKENS),
            timed_mode: b8_tag.to_string(),
            track_id: "qwen3.8-27b-mtp-v1".to_string(),
            source: "targets[test]".to_string(),
        };
        let mut cfg = cohort_cfg(PAIRS_PER_COHORT_TARGET, PAIRS_PER_COHORT_TARGET);
        cfg.calibration = Some(resolved.clone());
        let outcome = cohort_identity_run(&cfg).unwrap();
        let band = outcome
            .results
            .provenance
            .serial_band_outcome
            .as_ref()
            .expect("band sealed");
        assert_eq!(band.verdict, SerialBandVerdict::Pass);
        assert!(band.passed && band.window_ok && band.in_band);

        // Out of band: the cohort serial mean drifted 3x from the calibrated mean — die-6 verdict.
        let mut drifted = cohort_cfg(PAIRS_PER_COHORT_TARGET, PAIRS_PER_COHORT_TARGET);
        drifted.calibration = Some(ResolvedCalibration {
            serial_mean: SERIAL_SPT / 3.0,
            ..resolved
        });
        let outcome = cohort_identity_run(&drifted).unwrap();
        let band = outcome
            .results
            .provenance
            .serial_band_outcome
            .as_ref()
            .expect("band sealed");
        assert_eq!(band.verdict, SerialBandVerdict::Die6);
        assert!(!band.passed);
    }

    #[test]
    fn cohort_run_refuses_a_non_batched_config() {
        let goldens = cohort_goldens();
        let members =
            validate_cohort_membership(&goldens, &cohort_pool(&goldens), SCORED_BATCH_SIZE_B8)
                .unwrap();
        let err = run_cohort_measure_job(
            &goldens,
            members,
            &DirDigest::empty(),
            "deadbeef",
            &free_run_cfg(2, PAIRS_PER_COHORT_TARGET),
            CohortSeams {
                measure_serial: |_p: &CohortTimingParams| ok_cohort_serial(),
                measure_candidate: |_p: &CohortTimingParams| inv_cohort_candidate(CANDIDATE_SPT),
                oracle: mock_oracle_pass,
            },
        )
        .err()
        .expect("a non-batched config must refuse the cohort path");
        assert!(err.contains("requires the batched cohort regime"), "{err}");
    }

    #[test]
    fn cohort_die5_floor_is_per_cohort() {
        // A cohort accepting fewer than min_pairs fails closed (die 5) with the honest seal: the
        // cohort record lists its real accepted count and the per-cohort floor is stated.
        let goldens = cohort_goldens();
        let members =
            validate_cohort_membership(&goldens, &cohort_pool(&goldens), SCORED_BATCH_SIZE_B8)
                .unwrap();
        let mut first = true;
        let outcome = run_cohort_measure_job(
            &goldens,
            members,
            &DirDigest::empty(),
            "deadbeef",
            &cohort_cfg(PAIRS_PER_COHORT_TARGET, PAIRS_PER_COHORT_TARGET),
            CohortSeams {
                measure_serial: |_p: &CohortTimingParams| ok_cohort_serial(),
                measure_candidate: move |_p: &CohortTimingParams| {
                    if first {
                        first = false;
                        inv_cohort_candidate(CANDIDATE_SPT)
                    } else {
                        // The second pair's candidate leg dies (infra): official immediate die-5.
                        Err(RunnerError::Protocol("engine died".to_string()))
                    }
                },
                oracle: mock_oracle_pass,
            },
        )
        .unwrap();
        assert!(!outcome.candidate_accepted, "1 accepted < min 2 — die 5");
        let r = &outcome.results;
        assert_eq!(r.accepted_pair_count, 1);
        let c = &r.per_cohort.as_ref().unwrap()[0];
        assert_eq!(c.accepted_pair_count, 1);
        assert_eq!(r.candidate.verdict, "REJECT");
    }

    // -----------------------------------------------------------------------
    // COMPOSITE COHORT SCORING (Gemma track) — the SHARED-WINDOW composite: the ratio of
    // parent-clocked window SUMS across accepted pairs, raised to the CERTIFIED exponents. These
    // tests pin the exact arithmetic, the fail-loud refusals, and — the lane's REQUIRED
    // anti-regression proof — that NO engine-reported per-stream number can move the score.
    // -----------------------------------------------------------------------

    /// A batched leg with FULLY EXPLICIT windows: the scored `spt` (which fixes the decode window
    /// at `spt * B * N`) and the prefill window, chosen per call so a two-pair cohort can carry
    /// DIFFERENT windows on each pair. Otherwise identical to [`ok_cohort_serial`] /
    /// [`inv_cohort_candidate`].
    fn inv_cohort_leg(
        candidate: bool,
        spt: f64,
        prefill_elapsed: f64,
        per_stream_timing: Option<PerStreamTimingCarry>,
    ) -> bench_runner::Result<LegInvocation> {
        let n = FREE_RUN_DECODE_TOKENS;
        Ok(LegInvocation {
            benchd_seconds_per_token: spt,
            wire_head_provenance: head_prov(if candidate {
                CANDIDATE_HEAD_SHA
            } else {
                SERIAL_HEAD_SHA
            }),
            gate_state: GateState::Fired,
            quiesce: QuiesceRecord::default(),
            telemetry: if candidate {
                candidate_telemetry()
            } else {
                serial_telemetry()
            },
            wire_effective_spec: Some(if candidate {
                SpecConfig::mtp(FREE_RUN_DEPTH)
            } else {
                SpecConfig::serial()
            }),
            regime: b8_regime(),
            free_run_audit: None,
            // B2 — a BATCHED cohort leg carries its split on `cohort_phase_windows`; the
            // single-stream `phase_window` channel is a v1.1 free-run fact and is None here.
            phase_window: None,
            cohort_audit: Some(if candidate {
                cohort_audit(&vec![4u32; n / 4], n)
            } else {
                cohort_audit(&vec![1u32; n], n)
            }),
            cohort_phase_windows: Some(cohort_phase_windows_for(spt, prefill_elapsed, n)),
            per_stream_timing,
            cohort_committed_tokens_by_stream: Some(cohort_committed_rect(n)),
        })
    }

    /// One pair's window plan: `(serial_spt, candidate_spt, serial_prefill, candidate_prefill)`.
    type PairWindows = (f64, f64, f64, f64);

    /// Drive a conformant cohort run whose pairs carry the GIVEN per-pair windows, optionally with
    /// an engine-reported per-stream carry attached to each leg. Asserts every planned pair was
    /// consumed exactly once, so a retry or a short run can never silently shift the plan.
    fn cohort_run_with_windows(
        plan: &[PairWindows],
        serial_carry: Option<PerStreamTimingCarry>,
        candidate_carry: Option<PerStreamTimingCarry>,
    ) -> MeasureJobOutcome {
        try_cohort_run_with_windows(plan, serial_carry, candidate_carry)
            .expect("the cohort run must seal")
    }

    /// [`cohort_run_with_windows`] without the success expectation, so a NEGATIVE control can
    /// observe the run's REFUSAL. The per-leg call-count assertions are deliberately not made
    /// here: a refused run is allowed to stop early.
    fn try_cohort_run_with_windows(
        plan: &[PairWindows],
        serial_carry: Option<PerStreamTimingCarry>,
        candidate_carry: Option<PerStreamTimingCarry>,
    ) -> Result<MeasureJobOutcome, String> {
        let goldens = cohort_goldens();
        let members =
            validate_cohort_membership(&goldens, &cohort_pool(&goldens), SCORED_BATCH_SIZE_B8)
                .expect("conformant cohort membership");
        let serial_calls = std::cell::Cell::new(0usize);
        let candidate_calls = std::cell::Cell::new(0usize);
        let outcome = run_cohort_measure_job(
            &goldens,
            members,
            &DirDigest::empty(),
            "deadbeef",
            &cohort_cfg(PAIRS_PER_COHORT_TARGET, PAIRS_PER_COHORT_TARGET),
            CohortSeams {
                measure_serial: |_p: &CohortTimingParams| {
                    let i = serial_calls.get();
                    serial_calls.set(i + 1);
                    let (spt, _, prefill, _) = plan[i];
                    inv_cohort_leg(false, spt, prefill, serial_carry.clone())
                },
                measure_candidate: |_p: &CohortTimingParams| {
                    let i = candidate_calls.get();
                    candidate_calls.set(i + 1);
                    let (_, spt, _, prefill) = plan[i];
                    inv_cohort_leg(true, spt, prefill, candidate_carry.clone())
                },
                oracle: mock_oracle_pass,
            },
        )?;
        assert_eq!(
            serial_calls.get(),
            plan.len(),
            "one serial leg per planned pair"
        );
        assert_eq!(
            candidate_calls.get(),
            plan.len(),
            "one candidate leg per planned pair"
        );
        Ok(outcome)
    }

    /// One plan entry per RULED accepted pair ([`PAIRS_PER_COHORT_TARGET`], 4 since David's
    /// 2026-08-26 ruling), each with DIFFERENT windows on every component, chosen so that the
    /// RATIO OF SUMS is numerically DISTINCT from the mean (and the median) of the per-pair
    /// ratios — the whole point of the fixture, since an implementation that averaged per-pair
    /// gains would otherwise pass an exact-value test unnoticed.
    ///
    /// * prefill per-pair ratios 4.0, 1.5, 6.0, 0.5 (mean 3.0, even-n median 2.75), while the
    ///   ratio of sums is 1.500/0.800 = 1.875.
    /// * decode per-pair ratios 2.0, 3.0, 4.0, 1.0 (mean 2.5, even-n median 2.5), while the
    ///   ratio of sums is 0.029/0.0135 = 2.148… (the `B * N` factor cancels out of the ratio).
    ///
    /// The array LENGTH is tied to the ruled target on purpose: a future pairs ruling that moves
    /// the constant without re-authoring this fixture fails to COMPILE rather than driving the
    /// mock plan off its end at runtime.
    const UNEVEN_PAIR_PLAN: [PairWindows; PAIRS_PER_COHORT_TARGET] = [
        (0.004, 0.002, 0.400, 0.100),
        (0.009, 0.003, 0.300, 0.200),
        (0.010, 0.0025, 0.600, 0.100),
        (0.006, 0.006, 0.200, 0.400),
    ];

    /// The gains [`UNEVEN_PAIR_PLAN`] must produce, accumulated in the SAME order and association
    /// the implementation uses (pair 0 through pair 3), so the comparison is exact rather than
    /// epsilon-dependent on how the literals were folded.
    fn uneven_plan_gains() -> (f64, f64) {
        let bn = (COHORT_B * FREE_RUN_DECODE_TOKENS) as f64;
        let prefill = (0.400f64 + 0.300f64 + 0.600f64 + 0.200f64)
            / (0.100f64 + 0.200f64 + 0.100f64 + 0.400f64);
        let decode = (0.004f64 * bn + 0.009f64 * bn + 0.010f64 * bn + 0.006f64 * bn)
            / (0.002f64 * bn + 0.003f64 * bn + 0.0025f64 * bn + 0.006f64 * bn);
        (prefill, decode)
    }

    #[test]
    fn composite_is_the_ratio_of_window_sums_across_accepted_pairs() {
        // EXACT VALUE. Both gains are Σ serial / Σ candidate over the accepted pairs' PARENT-clocked
        // windows, serial-anchored; the composite is those two raised to the CERTIFIED exponents.
        let outcome = cohort_run_with_windows(&UNEVEN_PAIR_PLAN, None, None);
        assert!(outcome.candidate_accepted);
        let c = &outcome.results.per_cohort.as_ref().unwrap()[0];
        assert_eq!(c.accepted_pair_count, PAIRS_PER_COHORT_TARGET);
        let composite = c
            .composite
            .expect("a conformant cohort must seal a composite");
        assert!(
            c.composite_absent_reason.is_none(),
            "present composite carries no reason"
        );

        let (prefill_gain, decode_gain) = uneven_plan_gains();
        assert_eq!(composite.prefill_gain, prefill_gain);
        assert_eq!(composite.decode_gain, decode_gain);
        let expected =
            prefill_gain.powf(PREFILL_GAIN_EXPONENT) * decode_gain.powf(DECODE_GAIN_EXPONENT);
        assert_eq!(composite.composite_score, expected);

        // NON-VACUITY 1 — ratio-of-SUMS, not mean-of-per-pair-ratios. Both alternatives are
        // computed here and must MISS the sealed value by a wide margin.
        let mean_of_prefill_ratios = (4.0f64 + 1.5f64 + 6.0f64 + 0.5f64) / 4.0;
        let mean_of_decode_ratios = (2.0f64 + 3.0f64 + 4.0f64 + 1.0f64) / 4.0;
        assert!(
            (composite.prefill_gain - mean_of_prefill_ratios).abs() > 1.0,
            "prefill_gain {} must be the ratio of sums ({prefill_gain}), not the mean of per-pair \
             ratios ({mean_of_prefill_ratios})",
            composite.prefill_gain
        );
        assert!(
            (composite.decode_gain - mean_of_decode_ratios).abs() > 0.3,
            "decode_gain {} must be the ratio of sums ({decode_gain}), not the mean of per-pair \
             ratios ({mean_of_decode_ratios})",
            composite.decode_gain
        );

        // NON-VACUITY 2 — the EXPONENT SPLIT is load-bearing: the SWAPPED pair (0.75 prefill /
        // 0.25 decode) produces a different number, so a fixture that certified the wrong way
        // round could not coincidentally match. The sealed pair is the ruled one.
        let swapped =
            prefill_gain.powf(DECODE_GAIN_EXPONENT) * decode_gain.powf(PREFILL_GAIN_EXPONENT);
        assert!(
            (composite.composite_score - swapped).abs() > 1e-3,
            "composite {} must depend on WHICH exponent goes with which gain (swapped: {swapped})",
            composite.composite_score
        );
        assert_eq!(c.composite_scored_exponents, SCORED_EXPONENTS);
        assert_eq!(c.composite_scored_exponents.prefill_gain_exponent, 0.25);
        assert_eq!(c.composite_scored_exponents.decode_gain_exponent, 0.75);

        // The serial anchor: a faster candidate scores > 1 on both components and on the composite.
        assert!(composite.prefill_gain > 1.0 && composite.decode_gain > 1.0);
        assert!(composite.composite_score > 1.0);
        // The floor comes from the EXISTING regime-scoped decision, not a new constant.
        assert_eq!(
            composite.composite_speedup_floor,
            FREE_RUN_DECODE_SPEEDUP_FLOOR
        );
        assert!(composite.composite_speedup_floor_met);
    }

    /// ★ THE ANTI-REGRESSION PIN. Absurd ENGINE-REPORTED per-stream ns — the candidate claiming
    /// 1 ns per stream, the serial claiming ~1000 hours per stream — which under the BLOCKED
    /// per-stream-sum design would have scored a decode gain around 2.9e24. The composite must be
    /// BIT-IDENTICAL to the same cohort run without any per-stream carry at all.
    fn absurd_carry(ns: u64) -> PerStreamTimingCarry {
        PerStreamTimingCarry {
            prefill_ns_by_stream: Some(vec![ns; COHORT_B]),
            decode_ns_by_stream: Some(vec![ns; COHORT_B]),
            tokens_len_by_stream: vec![FREE_RUN_DECODE_TOKENS; COHORT_B],
            advertised: true,
        }
    }

    #[test]
    fn composite_ignores_engine_reported_per_stream_ns_entirely() {
        // Identical parent-clocked windows on both runs; the ONLY difference is the engine-reported
        // per-stream evidence riding along. If ANY engine-reported number reached the scoring path,
        // these two composites could not be equal.
        let clean = cohort_run_with_windows(&UNEVEN_PAIR_PLAN, None, None);
        let doctored = cohort_run_with_windows(
            &UNEVEN_PAIR_PLAN,
            // SERIAL claims ~1000 hours per stream, CANDIDATE claims 1 ns per stream: the
            // maximally self-serving lie in the direction that inflates a per-stream-sum gain.
            Some(absurd_carry(3_600_000_000_000_000)),
            Some(absurd_carry(1)),
        );

        let clean_c = &clean.results.per_cohort.as_ref().unwrap()[0];
        let doctored_c = &doctored.results.per_cohort.as_ref().unwrap()[0];
        let clean_score = clean_c.composite.expect("clean run seals a composite");
        let doctored_score = doctored_c
            .composite
            .expect("doctored run seals a composite");

        assert_eq!(
            doctored_score, clean_score,
            "the composite must be untouched by engine-reported per-stream ns"
        );
        assert_eq!(doctored_score.prefill_gain, uneven_plan_gains().0);
        assert_eq!(doctored_score.decode_gain, uneven_plan_gains().1);
        // The lie was ABSURD in the direction that would have paid: prove the counterfactual is
        // enormous, so "unchanged" is a real result and not a coincidence of small numbers.
        let per_stream_sum_gain =
            (3_600_000_000_000_000u64 * COHORT_B as u64) as f64 / (COHORT_B as f64);
        assert!(
            per_stream_sum_gain > 1e14,
            "the doctored evidence must be wildly self-serving to make this test meaningful"
        );

        // NON-VACUITY — the doctored vectors really did travel the pipeline: the REPORT-ONLY
        // attestation seal recorded them on the accepted pairs (that is #188/#189's job, untouched
        // here). So the score ignoring them is a choice the scoring path makes, not an artefact of
        // the evidence never arriving.
        let sealed_attestations = doctored
            .results
            .pairs
            .iter()
            .filter(|p| p.serial_per_stream_attestation.is_some())
            .count();
        assert!(
            sealed_attestations > 0,
            "the doctored per-stream evidence must reach the sealed record (report-only)"
        );
        assert!(
            clean
                .results
                .pairs
                .iter()
                .all(|p| p.serial_per_stream_attestation.is_none()),
            "the control run carries no per-stream evidence at all"
        );
    }

    #[test]
    fn composite_absent_with_a_named_reason_when_no_pair_accepts() {
        // FAIL-LOUD, zero accepted pairs (die-5): no windows, so nothing to divide. `composite` is
        // None with a reason that says so — never a fabricated number, and the run still seals.
        let goldens = cohort_goldens();
        let members =
            validate_cohort_membership(&goldens, &cohort_pool(&goldens), SCORED_BATCH_SIZE_B8)
                .unwrap();
        let outcome = run_cohort_measure_job(
            &goldens,
            members,
            &DirDigest::empty(),
            "deadbeef",
            &cohort_cfg(PAIRS_PER_COHORT_TARGET, PAIRS_PER_COHORT_TARGET),
            CohortSeams {
                measure_serial: |_p: &CohortTimingParams| ok_cohort_serial(),
                measure_candidate: |_p: &CohortTimingParams| {
                    Err(RunnerError::Protocol(
                        "engine died on the first pair".to_string(),
                    ))
                },
                oracle: mock_oracle_pass,
            },
        )
        .unwrap();
        assert!(!outcome.candidate_accepted);
        let c = &outcome.results.per_cohort.as_ref().unwrap()[0];
        assert_eq!(c.accepted_pair_count, 0);
        assert!(c.composite.is_none());
        let reason = c
            .composite_absent_reason
            .as_ref()
            .expect("an absent composite always carries a reason");
        assert!(
            reason.contains("no accepted pair") && reason.contains("die-5"),
            "the reason must name the zero-accepted-pairs cause: {reason}"
        );
        // The certified exponent identity is still sealed — it is a fact about the contract, not
        // about whether this particular run could be scored.
        assert_eq!(c.composite_scored_exponents, SCORED_EXPONENTS);
    }

    #[test]
    fn composite_degenerate_window_refuses_the_whole_batched_run() {
        // NEGATIVE CONTROL for the promotion — the ATOMIC rule (David 2026-08-27). A pair ACCEPTS
        // (nothing upstream rejects a zero-length phase window) but its candidate PREFILL window
        // is 0.0 seconds, so the composite cannot be computed. Now that the composite IS the
        // published score, a batched run that accepted pairs and cannot produce one has NO score:
        // it refuses by name. The half-state it must never reach is a sealed artifact that says
        // `candidate_accepted` and carries a decode-only median a consumer could rank on.
        let mut plan = UNEVEN_PAIR_PLAN;
        plan[0].3 = 0.0;
        let Err(err) = try_cohort_run_with_windows(&plan, None, None) else {
            panic!("a batched run that cannot be scored must refuse, not seal");
        };
        assert!(
            err.contains("candidate prefill window") && err.contains("accepted pair 0"),
            "the refusal must name the offending leg, component and pair: {err}"
        );
        assert!(
            err.contains("published score"),
            "the refusal must say WHY an unscoreable batched run is fatal: {err}"
        );
        assert!(
            err.contains("accepted 4 pair(s)"),
            "the refusal must say how many pairs the cohort had measured: {err}"
        );

        // NON-VACUITY — the SAME plan with a conformant window seals a scored artifact. The
        // refusal is the zeroed window, not the fixture.
        let ok = cohort_run_with_windows(&UNEVEN_PAIR_PLAN, None, None);
        assert!(ok.results.per_cohort.as_ref().unwrap()[0]
            .composite
            .is_some());
    }

    #[test]
    fn composite_sub_floor_run_still_seals_a_scored_artifact() {
        // SCOPE FENCE for the atomic rule: it fires on an UNSCOREABLE run, never on a BAD score.
        // A composite far below the 0.90 floor is a legitimate measurement — it seals, with
        // `composite_speedup_floor_met` false, and the overlay is the seam that refuses to publish
        // a score below the floor.
        let quarter_speed: PairWindows = (
            SERIAL_SPT,
            SERIAL_SPT * 4.0,
            SERIAL_PREFILL_ELAPSED,
            SERIAL_PREFILL_ELAPSED * 4.0,
        );
        let slow = [quarter_speed; PAIRS_PER_COHORT_TARGET];
        let outcome = cohort_run_with_windows(&slow, None, None);
        let composite = outcome.results.per_cohort.as_ref().unwrap()[0]
            .composite
            .expect("a measurable-but-slow cohort still seals its composite");
        assert!((composite.composite_score - 0.25).abs() < 1e-12);
        assert!(!composite.composite_speedup_floor_met);
    }

    #[test]
    fn composite_window_diagnostics_are_sealed_raw_per_pair() {
        // The per-pair RAW windows are sealed verbatim (they are the composite's recompute trail),
        // and the per-cohort MEANS remain what they always were. `serial_prefill_window_seconds_
        // mean` equals the fixture's SERIAL_PREFILL_ELAPSED constant directly (a MEAN over
        // identical per-pair windows), and NOTHING on `PairCohortPhaseWindows` computes a ratio.
        let outcome = cohort_identity_run(&cohort_cfg(
            PAIRS_PER_COHORT_TARGET,
            PAIRS_PER_COHORT_TARGET,
        ))
        .unwrap();
        assert!(outcome.candidate_accepted);
        let c = &outcome.results.per_cohort.as_ref().unwrap()[0];

        assert!(
            (c.serial_prefill_window_seconds_mean - SERIAL_PREFILL_ELAPSED).abs() < 1e-9,
            "serial_prefill_window_seconds_mean {} != {SERIAL_PREFILL_ELAPSED}",
            c.serial_prefill_window_seconds_mean
        );
        assert!(
            (c.candidate_prefill_window_seconds_mean - CANDIDATE_PREFILL_ELAPSED).abs() < 1e-9,
            "candidate_prefill_window_seconds_mean {} != {CANDIDATE_PREFILL_ELAPSED}",
            c.candidate_prefill_window_seconds_mean
        );
        assert_eq!(c.prefill_token_total, COHORT_B * 8);
        assert_eq!(c.decode_token_total, COHORT_B * FREE_RUN_DECODE_TOKENS);

        for p in &outcome.results.pairs {
            let w = p
                .cohort_phase_windows
                .expect("batched pair carries phase windows");
            assert!((w.serial_prefill_window_seconds - SERIAL_PREFILL_ELAPSED).abs() < 1e-9);
            assert!((w.candidate_prefill_window_seconds - CANDIDATE_PREFILL_ELAPSED).abs() < 1e-9);
            assert_eq!(w.prefill_token_total, c.prefill_token_total);
            assert_eq!(w.decode_token_total, c.decode_token_total);
        }
    }

    #[test]
    fn composite_on_the_identity_cohort_is_the_closed_form_two_to_the_five_quarters() {
        // The standard fixture cohort (every pair identical): prefill_gain 0.400/0.100 = 4,
        // decode_gain SERIAL_SPT/CANDIDATE_SPT = 2, so the composite is 4^0.25 * 2^0.75 = 2^1.25 —
        // a closed form that pins the exponent application independently of the uneven fixture.
        let outcome = cohort_identity_run(&cohort_cfg(
            PAIRS_PER_COHORT_TARGET,
            PAIRS_PER_COHORT_TARGET,
        ))
        .unwrap();
        let c = &outcome.results.per_cohort.as_ref().unwrap()[0];
        let composite = c.composite.expect("a conformant cohort seals a composite");
        assert!(
            (composite.prefill_gain - 4.0).abs() < 1e-12,
            "{composite:?}"
        );
        assert!((composite.decode_gain - 2.0).abs() < 1e-12, "{composite:?}");
        assert!(
            (composite.composite_score - 2.0f64.powf(1.25)).abs() < 1e-12,
            "composite {} != 2^1.25",
            composite.composite_score
        );
        assert!(c.composite_absent_reason.is_none());
        assert_eq!(c.composite_scored_exponents, SCORED_EXPONENTS);
    }

    #[test]
    fn composite_serializes_with_its_gains_and_no_absent_reason() {
        // The JSON shape on a scored run: `composite` is PRESENT with both gains and the score,
        // `composite_absent_reason` is OMITTED (skip_serializing_if), `composite_scored_exponents`
        // is PRESENT — the seal is visible in the actual artifact bytes, not just the struct.
        let outcome = cohort_identity_run(&cohort_cfg(
            PAIRS_PER_COHORT_TARGET,
            PAIRS_PER_COHORT_TARGET,
        ))
        .unwrap();
        let json = outcome.results.to_sealed_json().unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let cohort = &value["per_cohort"][0];
        assert!(
            cohort.get("composite_absent_reason").is_none(),
            "a scored run seals no absent-reason, not even null"
        );
        let composite = &cohort["composite"];
        assert!((composite["composite_score"].as_f64().unwrap() - 2.0f64.powf(1.25)).abs() < 1e-12);
        assert!((composite["prefill_gain"].as_f64().unwrap() - 4.0).abs() < 1e-12);
        assert!((composite["decode_gain"].as_f64().unwrap() - 2.0).abs() < 1e-12);
        assert!(composite["composite_speedup_floor"].is_number());
        assert!(composite["composite_speedup_floor_met"].is_boolean());
        assert!(cohort["composite_scored_exponents"].is_object());

        // The per-pair windows the score is recomputable from are in the same artifact.
        let pair_windows = &value["pairs"][0]["cohort_phase_windows"];
        assert!(pair_windows["serial_prefill_window_seconds"].is_number());
        assert!(pair_windows["candidate_decode_window_seconds"].is_number());
    }

    #[test]
    fn composite_die5_empty_cohort_seals_without_panicking() {
        // The FIRST pair attempt fails: official immediate die-5 with ZERO accepted pairs. The
        // window collection must not panic on the empty case (no phase windows to collect) — it
        // seals an honest die-5 result exactly as the pre-existing decode-only path already did on
        // this same empty-accepted shape, with no fabricated token totals.
        let goldens = cohort_goldens();
        let members =
            validate_cohort_membership(&goldens, &cohort_pool(&goldens), SCORED_BATCH_SIZE_B8)
                .unwrap();
        let outcome = run_cohort_measure_job(
            &goldens,
            members,
            &DirDigest::empty(),
            "deadbeef",
            &cohort_cfg(PAIRS_PER_COHORT_TARGET, PAIRS_PER_COHORT_TARGET),
            CohortSeams {
                measure_serial: |_p: &CohortTimingParams| ok_cohort_serial(),
                measure_candidate: |_p: &CohortTimingParams| {
                    Err(RunnerError::Protocol(
                        "engine died on the very first pair".to_string(),
                    ))
                },
                oracle: mock_oracle_pass,
            },
        )
        .unwrap();
        assert!(!outcome.candidate_accepted);
        let c = &outcome.results.per_cohort.as_ref().unwrap()[0];
        assert_eq!(c.accepted_pair_count, 0);
        assert_eq!(c.prefill_token_total, 0);
        assert_eq!(c.decode_token_total, 0);
        // Scored-value posture on the empty case is asserted by
        // `composite_absent_with_a_named_reason_when_no_pair_accepts`.
        assert!(c.composite.is_none());
        assert!(c.composite_absent_reason.is_some());
    }

    #[test]
    fn composite_fields_absent_from_the_single_stream_v1_1_path() {
        // B=1 equivalence / "the v1.1 single-stream path is UNTOUCHED": a non-batched run has NO
        // `per_cohort` at all, so none of the composite/diagnostic fields (which live exclusively
        // inside `PerCohort`) can appear on it — and the regime this run measures in is still
        // selected purely from the DECLARED spec (`candidate_regime_for_spec`), unaffected by any
        // of the composite work ("fixture-only regime selection unchanged"). `test_cfg`'s default
        // `scored_exponents: None` is never consulted here (single-stream never certifies it) —
        // the run below succeeds precisely BECAUSE `run_measure_job`/`build_results` never read
        // that field at all.
        let outcome = run_measure_job(
            &[measure_golden()],
            &DirDigest::empty(),
            "deadbeef",
            &test_cfg(1, 1),
            |_p| ok_serial(),
            |_p| {
                Ok(inv(
                    echo(CANDIDATE_SPT),
                    GateState::Fired,
                    candidate_telemetry(),
                ))
            },
        )
        .unwrap();
        assert!(outcome.results.per_cohort.is_none());
        assert!(outcome.results.scored_batch_size.is_none());
        assert_eq!(
            candidate_regime_for_spec(&SpecConfig::mtp(2)),
            LegRegime::FreeRunV1_1
        );
        assert_eq!(
            candidate_regime_for_spec(&SpecConfig::serial()),
            LegRegime::TeacherForcedV1
        );
    }

    // -----------------------------------------------------------------------
    // PER-STREAM ATTESTATION SEAL (per-stream arm-fill lane PR-B, gaps G2/G9) — REPORT-ONLY,
    // driven END-TO-END through the LIVE measure-job path: a real `MockEngine` (PR-A's
    // `per_stream_timing_capable` builder) -> `run_batched_free_run_decode_phase_fresh` (the
    // verbatim carry) -> `LegInvocation`/`validate_leg_report` pass-through -> `run_pair`'s
    // `attest_leg` seal -> `build_cohort_results` -> the sealed `results.json` bytes. These are
    // the lane's NON-VACUITY controls: doctored wire evidence must become a visible FLAGGED
    // verdict in the sealed artifact, while nothing scored moves in either direction.
    // -----------------------------------------------------------------------

    /// A conformant batch-capable engine for the FULL cohort window (`cohort_goldens()`'s B=8
    /// pool at the ruled N=128): every slot seeds to the shared `SEED_TOKEN` and continues with
    /// the shared oracle chain, committing `per_round` tokens over `rounds` rounds.
    fn cohort_seal_mock(per_round: u32, rounds: usize) -> MockEngine {
        let slots: Vec<(i64, Vec<i64>)> = (0..COHORT_B)
            .map(|_| (SEED_TOKEN, oracle_decode_tokens()))
            .collect();
        MockEngine::new()
            .batched_free_run_capable(8)
            .cohort_oracle(slots)
            .free_run_acceptance_lengths(vec![per_round; rounds])
    }

    /// The ENGINE-LEVEL batched leg seam, mirroring `main.rs`'s `measure_cohort_leg` line for
    /// line: spawn a fresh mock, run the batched phase (benchd's parent clock, the cohort
    /// quadruple at the barrier), and lift the phase windows AND the PR-A per-stream carry onto
    /// the `LegInvocation` exactly as production does — so the seal under test consumes wire
    /// evidence that travelled the real chain, not hand-built structs.
    fn cohort_engine_leg<B>(
        build: B,
        leg_spec: SpecConfig,
        params: &CohortTimingParams,
    ) -> bench_runner::Result<LegInvocation>
    where
        B: Fn() -> MockEngine,
    {
        let regime = b8_regime();
        let wire_head_provenance = std::cell::RefCell::new(None);
        let mut spawn = || -> bench_runner::Result<Session<MockEngine>> {
            let (session, hello) = Session::connect(build())?;
            *wire_head_provenance.borrow_mut() = hello.head_provenance.clone();
            Ok(session)
        };
        let mut gate = |_phase: &str| -> bench_runner::Result<()> { Ok(()) };
        let params = params
            .clone()
            .with_spec(requested_wire_spec(&leg_spec, regime));
        let t =
            bench_runner::run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &params)?;
        let cohort_phase_windows = CohortPhaseWindows::from(&t);
        let per_stream_timing = PerStreamTimingCarry::from(&t);
        Ok(LegInvocation {
            benchd_seconds_per_token: t.seconds_per_token,
            gate_state: GateState::Fired,
            quiesce: QuiesceRecord::default(),
            telemetry: None,
            wire_effective_spec: t.effective_spec,
            wire_head_provenance: wire_head_provenance.into_inner(),
            regime,
            free_run_audit: None,
            // B2 — a BATCHED cohort leg carries its split on `cohort_phase_windows`; the
            // single-stream `phase_window` channel is a v1.1 free-run fact and is None here.
            phase_window: None,
            cohort_audit: Some(t.audit),
            cohort_phase_windows: Some(cohort_phase_windows),
            per_stream_timing: Some(per_stream_timing),
            // (b) admission — surface the engine's committed rectangle UNJUDGED, exactly as the
            // real cohort leg (main.rs) does; the trusted-oracle gate judges it downstream.
            cohort_committed_tokens_by_stream: Some(t.tokens_by_stream),
        })
    }

    /// Run one engine-driven cohort measure-job: the serial control commits `[1]*N` under the
    /// serial wire spec, the candidate `[4]*(N/4)` under its declared mtp spec — both through
    /// [`cohort_engine_leg`].
    fn cohort_engine_run<S, C>(serial_engine: S, candidate_engine: C) -> MeasureJobOutcome
    where
        S: Fn() -> MockEngine,
        C: Fn() -> MockEngine,
    {
        let goldens = cohort_goldens();
        let members =
            validate_cohort_membership(&goldens, &cohort_pool(&goldens), SCORED_BATCH_SIZE_B8)
                .expect("conformant cohort membership");
        run_cohort_measure_job(
            &goldens,
            members,
            &DirDigest::empty(),
            "deadbeef",
            &cohort_cfg(PAIRS_PER_COHORT_TARGET, PAIRS_PER_COHORT_TARGET),
            CohortSeams {
                measure_serial: |p: &CohortTimingParams| {
                    cohort_engine_leg(&serial_engine, timed_decode_wire_spec(), p)
                },
                measure_candidate: |p: &CohortTimingParams| {
                    cohort_engine_leg(&candidate_engine, SpecConfig::mtp(FREE_RUN_DEPTH), p)
                },
                oracle: mock_oracle_pass,
            },
        )
        .expect("the engine-driven cohort run must seal")
    }

    /// Per-slot ns vectors small enough to sit INSIDE any real parent window (clause (c) clean by
    /// construction), slot-distinct so a reorder/summarize bug cannot pass by coincidence.
    fn modest_ns(base: u64) -> Vec<u64> {
        (0..COHORT_B as u64).map(|s| base + s).collect()
    }

    #[test]
    fn per_stream_doctored_decode_vector_flags_in_the_sealed_results() {
        // THE lane's REQUIRED non-vacuity proof, live-path: one slot's engine-reported decode
        // interval claims ~1000 HOURS — structurally impossible inside the parent-clocked decode
        // window (clause (c) bounding) — and the doctored evidence must surface as a FLAGGED
        // verdict in the sealed results.json, while the run itself is UNTOUCHED (report-only:
        // the pair still accepts, the published median still comes from the parent clocks).
        let mut doctored = modest_ns(2_000);
        doctored[3] = 3_600_000_000_000_000; // ~1000 hours of claimed monotonic ns
        let d = doctored.clone();
        let out = cohort_engine_run(
            move || {
                cohort_seal_mock(1, FREE_RUN_DECODE_TOKENS)
                    .per_stream_timing_capable(modest_ns(1_000), modest_ns(2_000))
            },
            move || {
                cohort_seal_mock(4, FREE_RUN_DECODE_TOKENS / 4)
                    .per_stream_timing_capable(modest_ns(1_000), d.clone())
            },
        );
        // REPORT-ONLY — the flag rejects NOTHING: the pair accepted, the run accepted, and the
        // published median is parent-clock data untouched by the absurd engine claim.
        assert!(
            out.candidate_accepted,
            "a flagged attestation verdict must not reject the candidate (nothing scores on it)"
        );
        assert_eq!(out.results.accepted_pair_count, PAIRS_PER_COHORT_TARGET);
        assert!(out.results.aggregate.raw_decode_speedup_median.is_finite());
        assert!(out.results.aggregate.raw_decode_speedup_median > 0.0);

        // The struct-level verdict: slot 3 flags clause (c) bounding, the honest slots do not.
        let att = out.results.pairs[0]
            .candidate_per_stream_attestation
            .as_ref()
            .expect("the capable candidate leg seals an attestation");
        assert!(att.advertised);
        assert!(att.attestation_refused.is_none());
        let verdict = att.verdict.as_ref().expect("structurally attestable");
        assert_eq!(verdict.batch_size, 8);
        assert!(
            verdict.decode_bounding[3].flagged_at_zero_tolerance,
            "the doctored slot MUST flag clause (c) at zero slack: {:?}",
            verdict.decode_bounding[3]
        );
        assert!(verdict.decode_bounding[3].raw_ratio > 1.0);
        assert!(
            !verdict.decode_bounding[0].flagged_at_zero_tolerance,
            "an honest slot must not flag bounding: {:?}",
            verdict.decode_bounding[0]
        );

        // And the SAME flag is visible in the sealed results.json BYTES — the artifact the box
        // calibration pass (and any reviewer) actually reads.
        let v: serde_json::Value =
            serde_json::from_str(&out.results.to_sealed_json().unwrap()).unwrap();
        let sealed = &v["pairs"][0]["candidate_per_stream_attestation"];
        assert_eq!(sealed["advertised"], json!(true));
        assert_eq!(
            sealed["verdict"]["decode_bounding"][3]["flagged_at_zero_tolerance"],
            json!(true),
            "the doctored slot's flag must be visible in the sealed artifact"
        );
        assert_eq!(
            sealed["verdict"]["decode_bounding"][0]["flagged_at_zero_tolerance"],
            json!(false)
        );
        assert_eq!(
            sealed["verdict"]["tolerance_state"],
            json!("unpinned-tolerances")
        );
        // The composite DIAGNOSTIC still seals (a flagged clause is evidence, not a refusal).
        assert!(v["pairs"][0]["per_stream_composite_diagnostic"].is_object());
        // And the SCORED composite is the SHARED-WINDOW one: recomputed here straight from the
        // sealed per-pair PARENT-clocked windows, it matches the sealed score exactly — so the
        // slot-3 lie that flagged above moved nothing, on a LIVE engine-driven run.
        let pairs = v["pairs"].as_array().unwrap();
        let window_sum = |key: &str| -> f64 {
            pairs
                .iter()
                .map(|p| p["cohort_phase_windows"][key].as_f64().unwrap())
                .sum()
        };
        let decode_gain = window_sum("serial_decode_window_seconds")
            / window_sum("candidate_decode_window_seconds");
        let prefill_gain = window_sum("serial_prefill_window_seconds")
            / window_sum("candidate_prefill_window_seconds");
        let composite = &v["per_cohort"][0]["composite"];
        assert!(v["per_cohort"][0].get("composite_absent_reason").is_none());
        assert!((composite["decode_gain"].as_f64().unwrap() - decode_gain).abs() < 1e-12);
        assert!((composite["prefill_gain"].as_f64().unwrap() - prefill_gain).abs() < 1e-12);
        assert!(
            (composite["composite_score"].as_f64().unwrap()
                - prefill_gain.powf(PREFILL_GAIN_EXPONENT)
                    * decode_gain.powf(DECODE_GAIN_EXPONENT))
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn per_stream_happy_path_seals_clean_verdicts_composite_and_slot_order() {
        // The conformant capable engine: verdicts seal with NO refusal and NO bounding flag,
        // the raw evidence (vectors, K_slot, R, windows) is sealed verbatim for recomputation,
        // the pair's composite diagnostic seals at the CERTIFIED exponent pair, and the G9
        // slot-order provenance binds verdict slot i to cohort member i.
        let out = cohort_engine_run(
            || {
                cohort_seal_mock(1, FREE_RUN_DECODE_TOKENS)
                    .per_stream_timing_capable(modest_ns(1_000), modest_ns(2_000))
            },
            || {
                cohort_seal_mock(4, FREE_RUN_DECODE_TOKENS / 4)
                    .per_stream_timing_capable(modest_ns(1_000), modest_ns(2_000))
            },
        );
        assert!(out.candidate_accepted);
        for p in &out.results.pairs {
            for (leg, att, rounds) in [
                (
                    "serial",
                    &p.serial_per_stream_attestation,
                    FREE_RUN_DECODE_TOKENS,
                ),
                (
                    "candidate",
                    &p.candidate_per_stream_attestation,
                    FREE_RUN_DECODE_TOKENS / 4,
                ),
            ] {
                let att = att.as_ref().expect("capable leg seals an attestation");
                assert!(att.advertised, "{leg}");
                assert!(att.attestation_refused.is_none(), "{leg}");
                // The verbatim evidence, recompute-sufficient in ONE object: vectors (G1),
                // K_slot (G3, the response rectangle's per-slot len — N per slot here), R, and
                // the leg's parent windows.
                assert_eq!(
                    att.prefill_ns_by_stream.as_deref(),
                    Some(&modest_ns(1_000)[..]),
                    "{leg}"
                );
                assert_eq!(
                    att.decode_ns_by_stream.as_deref(),
                    Some(&modest_ns(2_000)[..]),
                    "{leg}"
                );
                assert_eq!(
                    att.tokens_len_by_stream,
                    vec![FREE_RUN_DECODE_TOKENS; COHORT_B],
                    "{leg}"
                );
                assert_eq!(att.rounds, rounds, "{leg}");
                assert!(att.prefill_window_seconds > 0.0 && att.decode_window_seconds > 0.0);
                let verdict = att.verdict.as_ref().expect("clean inputs attest");
                // Clause (c) clean on BOTH phases by construction (ns << any real window).
                for cv in verdict
                    .prefill_bounding
                    .iter()
                    .chain(verdict.decode_bounding.iter())
                {
                    assert!(!cv.flagged_at_zero_tolerance, "{leg}: {cv:?}");
                }
                // The sealed SUM diagnostics are the exact sums of the verbatim vectors.
                assert_eq!(verdict.prefill_sum_ns, modest_ns(1_000).iter().sum::<u64>());
                assert_eq!(verdict.decode_sum_ns, modest_ns(2_000).iter().sum::<u64>());
                assert_eq!(verdict.tolerance_state, "unpinned-tolerances");
            }
            // Identical per-stream sums on both legs ⇒ both gains exactly 1.0, and the composite
            // (at the certified 0.25/0.75) is exactly 1.0 — proving the diagnostic consumed the
            // sealed sums, not some other aggregate.
            let comp = p
                .per_stream_composite_diagnostic
                .as_ref()
                .expect("both legs attested Ok, so the pair composite seals");
            assert!((comp.prefill_gain - 1.0).abs() < 1e-12);
            assert!((comp.decode_gain - 1.0).abs() < 1e-12);
            assert!((comp.composite_score - 1.0).abs() < 1e-12);
            assert_eq!(comp.tolerance_state, "unpinned-tolerances");
        }
        // G9 — the slot-order provenance: sealed once per cohort, restating each verdict slot's
        // member identity in slot order (= pool order), review-checkable from the artifact alone.
        let c = &out.results.per_cohort.as_ref().unwrap()[0];
        let slot_order = c
            .per_stream_attestation_slot_order
            .as_ref()
            .expect("attestations sealed, so the slot-order provenance seals");
        assert_eq!(slot_order.rule, PER_STREAM_SLOT_ORDER_RULE);
        let goldens = cohort_goldens();
        assert_eq!(slot_order.slot_prompt_sha256.len(), COHORT_B);
        for (i, sha) in slot_order.slot_prompt_sha256.iter().enumerate() {
            assert_eq!(
                *sha,
                goldens[i].sha256().to_ascii_lowercase(),
                "verdict slot {i} must be bound to cohort member {i} (pool order)"
            );
            assert_eq!(*sha, c.members[i].prompt_sha256, "restated, not re-derived");
        }
        // And in the sealed bytes.
        let v: serde_json::Value =
            serde_json::from_str(&out.results.to_sealed_json().unwrap()).unwrap();
        assert_eq!(
            v["per_cohort"][0]["per_stream_attestation_slot_order"]["rule"],
            json!(PER_STREAM_SLOT_ORDER_RULE)
        );
        assert_eq!(
            v["per_cohort"][0]["per_stream_attestation_slot_order"]["slot_prompt_sha256"]
                .as_array()
                .unwrap()
                .len(),
            COHORT_B
        );
    }

    #[test]
    fn per_stream_capability_absent_seals_no_attestation_and_run_is_unaffected() {
        // An engine WITHOUT the capability (and no vectors on the wire): the attestation is
        // ABSENT — not refused, not defaulted — and the run is byte-for-byte the pre-lane shape
        // (no new keys anywhere).
        let out = cohort_engine_run(
            || cohort_seal_mock(1, FREE_RUN_DECODE_TOKENS),
            || cohort_seal_mock(4, FREE_RUN_DECODE_TOKENS / 4),
        );
        assert!(out.candidate_accepted, "the run is unaffected");
        assert_eq!(out.results.accepted_pair_count, PAIRS_PER_COHORT_TARGET);
        for p in &out.results.pairs {
            assert!(p.serial_per_stream_attestation.is_none());
            assert!(p.candidate_per_stream_attestation.is_none());
            assert!(p.per_stream_composite_diagnostic.is_none());
        }
        let v: serde_json::Value =
            serde_json::from_str(&out.results.to_sealed_json().unwrap()).unwrap();
        for key in [
            "serial_per_stream_attestation",
            "candidate_per_stream_attestation",
            "per_stream_composite_diagnostic",
        ] {
            assert!(
                v["pairs"][0].get(key).is_none(),
                "{key} must be OMITTED (additive serde), not null"
            );
        }
        assert!(
            v["per_cohort"][0]
                .get("per_stream_attestation_slot_order")
                .is_none(),
            "no attestation sealed ⇒ no slot-order seal"
        );
    }

    #[test]
    fn per_stream_structural_length_mismatch_seals_a_named_refusal_not_a_rejection() {
        // Clause (b) STRUCTURAL failure, live-path: the capable candidate engine reports a
        // 7-entry decode vector against B=8. The seal names the refusal on the pair record —
        // and NOTHING else moves: the pair still accepts, the run still accepts (report-only:
        // no existing check tightens or loosens).
        let short_decode: Vec<u64> = (0..7u64).map(|s| 2_000 + s).collect();
        let sd = short_decode.clone();
        let out = cohort_engine_run(
            || {
                cohort_seal_mock(1, FREE_RUN_DECODE_TOKENS)
                    .per_stream_timing_capable(modest_ns(1_000), modest_ns(2_000))
            },
            move || {
                cohort_seal_mock(4, FREE_RUN_DECODE_TOKENS / 4)
                    .per_stream_timing_capable(modest_ns(1_000), sd.clone())
            },
        );
        assert!(
            out.candidate_accepted,
            "a structural attestation refusal is NOT a run failure"
        );
        assert_eq!(
            out.results.accepted_pair_count, PAIRS_PER_COHORT_TARGET,
            "a structural attestation refusal is NOT a pair rejection"
        );
        assert!(out.results.rejected_pairs.is_empty());
        let att = out.results.pairs[0]
            .candidate_per_stream_attestation
            .as_ref()
            .expect("the defective evidence is sealed, not dropped");
        assert!(att.verdict.is_none(), "no verdict on refused inputs");
        let reason = att
            .attestation_refused
            .as_ref()
            .expect("the refusal is NAMED");
        assert!(
            reason.contains("expected B=8") && reason.contains("decode"),
            "the reason names the phase and the width: {reason}"
        );
        // The defective vector itself is sealed verbatim as evidence.
        assert_eq!(att.decode_ns_by_stream.as_deref(), Some(&short_decode[..]));
        // The SERIAL leg's clean attestation still seals; the pair composite cannot (one side
        // has no verdict) and is honestly omitted.
        assert!(out.results.pairs[0]
            .serial_per_stream_attestation
            .as_ref()
            .and_then(|s| s.verdict.as_ref())
            .is_some());
        assert!(out.results.pairs[0]
            .per_stream_composite_diagnostic
            .is_none());
        // Visible by name in the sealed artifact.
        let v: serde_json::Value =
            serde_json::from_str(&out.results.to_sealed_json().unwrap()).unwrap();
        assert!(
            v["pairs"][0]["candidate_per_stream_attestation"]["attestation_refused"].is_string()
        );
        assert!(v["pairs"][0]["candidate_per_stream_attestation"]
            .get("verdict")
            .is_none());
    }

    // -----------------------------------------------------------------------
    // COMPOSITE EXPONENTS AS A FIXTURE-PINNED IDENTITY (orchestrator ruling, 2026-08-23)
    // -----------------------------------------------------------------------

    #[test]
    fn scored_exponents_certify_accepts_the_one_ruled_pair() {
        // certify-positive: the fixture declares EXACTLY the ruled pair, bit-identical.
        let certified = ScoredExponents::certify(Some((0.25, 0.75))).unwrap();
        assert_eq!(certified, SCORED_EXPONENTS);
        assert_eq!(certified.prefill_gain_exponent, PREFILL_GAIN_EXPONENT);
        assert_eq!(certified.decode_gain_exponent, DECODE_GAIN_EXPONENT);
    }

    #[test]
    fn scored_exponents_certify_refuses_a_wrong_pair() {
        // Wholly wrong, never a formatted/derived acceptance: neither component matches.
        let err = ScoredExponents::certify(Some((0.5, 0.5))).unwrap_err();
        assert!(err.contains("no certified exponent pair"), "{err}");
        assert!(err.contains("0.5"), "names the declared value: {err}");
        assert!(
            err.contains("0.25") && err.contains("0.75"),
            "names the ONE ruled pair too: {err}"
        );

        // A SWAPPED pair (the two ruled numbers, wrong components) is not "close enough" either —
        // the identity is the (prefill, decode) ASSIGNMENT, not just the multiset of values.
        assert!(ScoredExponents::certify(Some((0.75, 0.25))).is_err());

        // A NEAR miss (one component right, one off by a hair) also refuses — exact bit identity,
        // never an epsilon tolerance.
        assert!(ScoredExponents::certify(Some((0.25, 0.750_000_1))).is_err());
    }

    #[test]
    fn scored_exponents_certify_refuses_absence() {
        // absent field ⇒ fail-loud, no silent default to the code constants.
        let err = ScoredExponents::certify(None).unwrap_err();
        assert!(
            err.contains("requires the fixture to declare prefill_gain_exponent"),
            "{err}"
        );
    }

    #[test]
    fn composite_batched_regime_without_certified_exponents_refuses_at_the_seal() {
        // Belt-and-braces: a BATCHED config whose `scored_exponents` is `None` (the real caller,
        // main.rs, always certifies it alongside `candidate_regime` before this config is built —
        // a `None` here is a wiring defect) is refused at `build_cohort_results`, never silently
        // falling back to the code constants. This is the "absence under the batched regime"
        // refusal, exercised at the seam this crate can drive without main.rs's CLI plumbing.
        let mut cfg = cohort_cfg(PAIRS_PER_COHORT_TARGET, PAIRS_PER_COHORT_TARGET);
        cfg.scored_exponents = None;
        let goldens = cohort_goldens();
        let members =
            validate_cohort_membership(&goldens, &cohort_pool(&goldens), SCORED_BATCH_SIZE_B8)
                .unwrap();
        let err = run_cohort_measure_job(
            &goldens,
            members,
            &DirDigest::empty(),
            "deadbeef",
            &cfg,
            CohortSeams {
                measure_serial: |_p: &CohortTimingParams| ok_cohort_serial(),
                measure_candidate: |_p: &CohortTimingParams| inv_cohort_candidate(CANDIDATE_SPT),
                oracle: mock_oracle_pass,
            },
        )
        .err()
        .expect("a batched config with no certified scored_exponents must refuse");
        assert!(
            err.contains("requires a CERTIFIED scored_exponents"),
            "{err}"
        );
    }

    #[test]
    fn the_scored_regime_round_trips_through_the_one_migrated_schema() {
        // THE FIXTURE SHAPE, after the unification: the width and the exponent pair are FLAT
        // siblings on `bench_core::contract::Contract`, which is the ONE schema that declares
        // them. The paired flow reads them there, exactly as the scored single-leg path does, so
        // a fixture states each number once.
        let json = serde_json::json!({
            "track_id": "gemma4-26b-a4b-mlx-v1",
            "scored_batch_size": 8,
            "prefill_gain_exponent": 0.25,
            "decode_gain_exponent": 0.75
        });
        let scored =
            bench_core::contract::Contract::parse(&serde_json::to_vec(&json).unwrap()).unwrap();
        assert_eq!(scored.scored_batch_size, Some(8));
        assert_eq!(scored.prefill_gain_exponent, Some(0.25));
        assert_eq!(scored.decode_gain_exponent, Some(0.75));
        assert_eq!(
            ScoredExponents::certify(
                scored
                    .prefill_gain_exponent
                    .zip(scored.decode_gain_exponent)
            )
            .unwrap(),
            SCORED_EXPONENTS
        );
        // A fixture that omits them parses fine — the refusal is at CERTIFY time
        // (and only on the batched regime), never at parse time.
        let minimal = bench_core::contract::Contract::parse(br#"{"track_id":"t"}"#).unwrap();
        assert_eq!(minimal.scored_batch_size, None);
        assert_eq!(minimal.prefill_gain_exponent, None);
    }

    // -----------------------------------------------------------------------
    // David ruling (2026-08-26) — DFlash as a first-class SCORED mode
    // -----------------------------------------------------------------------

    /// The contract parses `allowed_modes` as a TRI-STATE, and the three states stay DISTINCT.
    ///
    /// ABSENT must survive as `None` rather than collapsing into an empty `Vec`: absence means
    /// "this fixture has no opinion, use the default" (the state EVERY other track's fixture is in)
    /// while an empty list is a fixture that declared the field and listed nothing — a refusal.
    /// A `#[serde(default)] Vec<String>` would have made those two indistinguishable.
    #[test]
    fn contract_parses_allowed_modes_as_a_tri_state() {
        let declared =
            Contract::parse(br#"{"track_id":"t","allowed_modes":["serial","mtp","dflash"]}"#)
                .unwrap();
        assert_eq!(
            declared.allowed_modes.as_deref(),
            Some(
                [
                    "serial".to_string(),
                    "mtp".to_string(),
                    "dflash".to_string()
                ]
                .as_slice()
            )
        );

        // An EMPTY declared list is not absence — it parses as `Some([])` and is refused by
        // `resolve_allowed_modes`, not silently defaulted here.
        let empty = Contract::parse(br#"{"track_id":"t","allowed_modes":[]}"#).unwrap();
        assert_eq!(empty.allowed_modes.as_deref(), Some([].as_slice()));

        // ABSENT stays ABSENT — the shape every OTHER track's fixture is in today.
        let silent = Contract::parse(br#"{"track_id":"t"}"#).unwrap();
        assert_eq!(silent.allowed_modes, None);
    }

    /// [`resolve_allowed_modes`] — the whole truth table.
    ///
    /// REVERT-PROOF: delete the contract branch (always return the default) and the dflash-admitting
    /// case goes red; drop the certification loop and the four fixture-malformed cases go red; widen
    /// [`DEFAULT_ALLOWED_MODES`] to include `dflash` instead of reading the contract and the
    /// OTHER-TRACK PROTECTION case goes red.
    #[test]
    fn resolve_allowed_modes_truth_table() {
        // ABSENT ⇒ the default, unchanged. THE other-track protection: the qwen3.8 and laguna
        // fixtures declare no list, so this is the answer they keep getting, and it does NOT
        // contain dflash.
        let default = resolve_allowed_modes(None).unwrap();
        assert_eq!(default, vec!["serial".to_string(), "mtp".to_string()]);
        assert!(
            !default.iter().any(|m| m == SPEC_MODE_DFLASH),
            "widening the global default would enable dflash on every track: {default:?}"
        );

        // DECLARED ⇒ the declared list, in declaration order.
        let declared = vec![
            SPEC_MODE_SERIAL.to_string(),
            SPEC_MODE_MTP.to_string(),
            SPEC_MODE_DFLASH.to_string(),
        ];
        assert_eq!(resolve_allowed_modes(Some(&declared)).unwrap(), declared);

        // A narrower declared list is honoured too — the field is not "add to the default".
        let serial_only = vec![SPEC_MODE_SERIAL.to_string()];
        assert_eq!(
            resolve_allowed_modes(Some(&serial_only)).unwrap(),
            serial_only
        );

        // EMPTY refuses.
        let empty: Vec<String> = vec![];
        let e = resolve_allowed_modes(Some(&empty)).unwrap_err();
        assert!(e.contains("EMPTY allowed_modes"), "{e}");

        // An UNKNOWN entry refuses, naming the entry.
        let typo = vec![SPEC_MODE_SERIAL.to_string(), "dflsah".to_string()];
        let e = resolve_allowed_modes(Some(&typo)).unwrap_err();
        assert!(
            e.contains("dflsah") && e.contains("not a mode benchd knows"),
            "{e}"
        );

        // `dspark` refuses BY NAME — reserved, not misspelled, so the message must differ from the
        // unknown-string one or a fixture author is told the wrong thing.
        let reserved = vec![SPEC_MODE_SERIAL.to_string(), SPEC_MODE_DSPARK.to_string()];
        let e = resolve_allowed_modes(Some(&reserved)).unwrap_err();
        assert!(e.contains("RESERVED") && e.contains("dspark"), "{e}");

        // A DUPLICATE refuses rather than being silently de-duplicated.
        let dup = vec![
            SPEC_MODE_SERIAL.to_string(),
            SPEC_MODE_MTP.to_string(),
            SPEC_MODE_MTP.to_string(),
        ];
        let e = resolve_allowed_modes(Some(&dup)).unwrap_err();
        assert!(e.contains("more than once"), "{e}");

        // A list WITHOUT serial refuses: the baseline leg is pinned serial and is checked against
        // this same list.
        let no_serial = vec![SPEC_MODE_MTP.to_string(), SPEC_MODE_DFLASH.to_string()];
        let e = resolve_allowed_modes(Some(&no_serial)).unwrap_err();
        assert!(e.contains("does not include \"serial\""), "{e}");
    }

    /// [`enforce_track_allowed_modes`] — the gate itself, over both legs.
    ///
    /// The two load-bearing cases are the pair: `dflash` is ADMITTED when the fixture declares it,
    /// and REFUSED when the fixture is silent. That pair IS the ruling — "enable dflash" without
    /// "and only where a fixture said so" would have been a global widening.
    #[test]
    fn track_allowed_modes_gate_admits_dflash_only_when_the_contract_declares_it() {
        let dflash = SpecConfig {
            mode: SPEC_MODE_DFLASH.to_string(),
            mtp: None,
            dflash: Some(serde_json::json!({})),
            dspark: None,
        };
        let serial = SpecConfig::serial();
        let declared = vec![
            SPEC_MODE_SERIAL.to_string(),
            SPEC_MODE_MTP.to_string(),
            SPEC_MODE_DFLASH.to_string(),
        ];

        // ADMITTED — the gemma4 shape after this lane. Returns the resolved vocabulary.
        let admitted = enforce_track_allowed_modes(&dflash, &serial, Some(&declared)).unwrap();
        assert_eq!(admitted, declared);

        // REFUSED when the fixture declares nothing — OTHER-TRACK PROTECTION, and the state the
        // qwen3.8/laguna fixtures are in. The refusal names the leg, the mode, and the source of
        // the list, so an operator can tell "this track never declared it" from "this track
        // declared a list that excludes it".
        let e = enforce_track_allowed_modes(&dflash, &serial, None).unwrap_err();
        assert!(e.contains("candidate leg"), "{e}");
        assert!(e.contains("dflash"), "{e}");
        assert!(e.contains("DEFAULT_ALLOWED_MODES"), "{e}");

        // REFUSED when the fixture declares a list that excludes it — a DIFFERENT source clause.
        let mtp_only = vec![SPEC_MODE_SERIAL.to_string(), SPEC_MODE_MTP.to_string()];
        let e = enforce_track_allowed_modes(&dflash, &serial, Some(&mtp_only)).unwrap_err();
        assert!(e.contains("allowed_modes"), "{e}");

        // mtp on a default track is UNCHANGED by this lane — the regression control.
        let mtp = SpecConfig::mtp(2);
        assert!(enforce_track_allowed_modes(&mtp, &serial, None).is_ok());

        // A malformed contract list refuses BEFORE either leg is judged, so a fixture error is
        // never reported as a submission error.
        let bad = vec!["nonsense".to_string()];
        let e = enforce_track_allowed_modes(&mtp, &serial, Some(&bad)).unwrap_err();
        assert!(
            !e.contains("candidate leg") && e.contains("nonsense"),
            "a fixture error must not be reported as a leg error: {e}"
        );
    }

    /// [`mode_is_cohort_capable`] and the MODE-AWARE cohort upgrade.
    ///
    /// The gemma4 fixture pins `scored_batch_size: 8`. Before this ruling that pin alone kept the
    /// track structurally closed to dflash — admitting the mode would only have moved the refusal
    /// from benchd to the engine's cohort driver, one spawn of gated box time later.
    ///
    /// REVERT-PROOF: delete the `mode_is_cohort_capable` guard and the dflash case flips to the b8
    /// regime, going red; make it accept every mode and the same case goes red.
    #[test]
    fn dflash_is_single_stream_only_and_is_not_upgraded_to_the_cohort_regime() {
        assert!(mode_is_cohort_capable(SPEC_MODE_SERIAL));
        assert!(mode_is_cohort_capable(SPEC_MODE_MTP));
        assert!(
            !mode_is_cohort_capable(SPEC_MODE_DFLASH),
            "the engine's cohort driver refuses dflash BY NAME"
        );
        // FAIL-CLOSED on anything else: an unknown mode is never assumed cohort-capable.
        assert!(!mode_is_cohort_capable(SPEC_MODE_DSPARK));
        assert!(!mode_is_cohort_capable("whatever"));

        // mtp at the pinned width still upgrades — the mtp arm is untouched.
        assert_eq!(
            effective_candidate_regime(SPEC_MODE_MTP, LegRegime::FreeRunV1_1, Some(8)).unwrap(),
            b8_regime()
        );
        // dflash at the SAME pinned width keeps the single-stream free-run regime.
        assert_eq!(
            effective_candidate_regime(SPEC_MODE_DFLASH, LegRegime::FreeRunV1_1, Some(8)).unwrap(),
            LegRegime::FreeRunV1_1
        );
        // …and the width is still CERTIFIED first, on every mode: a single-stream-only candidate
        // must not become the way an uncertified fixture width goes unnoticed.
        let e = effective_candidate_regime(SPEC_MODE_DFLASH, LegRegime::FreeRunV1_1, Some(4))
            .unwrap_err();
        assert!(e.contains("ONE ruled batch point"), "{e}");
    }

    /// PER-LEG DFLASH HEADS — the argv each leg is actually spawned with.
    ///
    /// This is the machine-checked form of the finding: the engine resolves a bare relative
    /// `./dflash-head` against the WORKER's CWD, benchd's spawn sets no `current_dir`, so both legs
    /// inherited benchd's CWD and would have loaded ONE directory. With the flag, each leg
    /// carries ITS OWN path, and the test proves it by giving the two legs DIFFERENT directories
    /// and reading the two argvs back.
    ///
    /// REVERT-PROOF: pass the same dir to both legs (the pre-lane behaviour) and the inequality
    /// assertions go red; drop the flag from `timed_leg_base_args` and every `--dflash-head`
    /// assertion goes red.
    #[test]
    fn dflash_head_is_resolved_per_leg() {
        let heads = resolve_head_dirs(Some("/pinned/dflash"), Some("/byo/dflash")).unwrap();
        assert_eq!(heads.head_dir, "/pinned/dflash");
        assert_eq!(heads.candidate_head_dir, "/byo/dflash");

        let serial_args = leg_spawn_args("/mtp", Some(&heads.head_dir), LegRegime::FreeRunV1_1);
        let candidate_args = leg_spawn_args(
            "/mtp",
            Some(&heads.candidate_head_dir),
            LegRegime::FreeRunV1_1,
        );

        // Each leg carries the flag ONCE, with ITS OWN value.
        for (label, args, expected) in [
            ("serial", &serial_args, "/pinned/dflash"),
            ("candidate", &candidate_args, "/byo/dflash"),
        ] {
            let at = args
                .iter()
                .position(|a| a == DFLASH_HEAD_FLAG)
                .unwrap_or_else(|| panic!("{label} leg carries no {DFLASH_HEAD_FLAG}: {args:?}"));
            assert_eq!(args[at + 1], expected, "{label} leg argv: {args:?}");
            assert_eq!(
                args.iter().filter(|a| *a == DFLASH_HEAD_FLAG).count(),
                1,
                "{label} leg argv: {args:?}"
            );
        }
        // THE property: the two legs do not resolve to the same drafter directory.
        assert_ne!(
            serial_args, candidate_args,
            "both legs would load the same drafter — the bug this flag closes"
        );

        // …and the PRODUCTION SELECTION agrees. `paired_leg_spawn_args` is what `execute_measure_job`
        // actually calls, so this is the assertion that covers the WIRING and not just the builder:
        // the serial control must receive the PINNED drafter and the candidate its OWN, for both
        // head families at once. A swap of any one of the four is a silent wrong measurement — the
        // candidate's drafter resident on the DENOMINATOR leg — and each swap fails here.
        let mtp_heads = resolve_head_dirs(Some("/pinned/mtp"), Some("/byo/mtp")).unwrap();
        let (serial_wired, candidate_wired) = paired_leg_spawn_args(
            &mtp_heads,
            Some(&heads),
            LegRegime::FreeRunV1_1,
            SPEC_MODE_DFLASH,
        );
        let value_after = |args: &[String], flag: &str| -> String {
            let at = args.iter().position(|a| a == flag).expect("flag present");
            args[at + 1].clone()
        };
        assert_eq!(value_after(&serial_wired, "--mtp-head"), "/pinned/mtp");
        assert_eq!(
            value_after(&serial_wired, DFLASH_HEAD_FLAG),
            "/pinned/dflash"
        );
        assert_eq!(value_after(&candidate_wired, "--mtp-head"), "/byo/mtp");
        assert_eq!(
            value_after(&candidate_wired, DFLASH_HEAD_FLAG),
            "/byo/dflash"
        );
        // NO CROSS-LEG RESIDENCY: the candidate's BYO paths must appear on NEITHER of the serial
        // control's channels. The serial leg is the scored DENOMINATOR; a candidate-supplied head
        // resident there contaminates it.
        assert!(
            !serial_wired.iter().any(|a| a.starts_with("/byo/")),
            "candidate-side head leaked onto the denominator leg: {serial_wired:?}"
        );
        // The serial control runs the SAME series as the candidate (Fable same-series rule), so its
        // v1.1 spawn gate must be present exactly when the candidate's is.
        assert_eq!(
            serial_wired.contains(&SPECULATIVE_PROTOCOL_FLAG.to_string()),
            candidate_wired.contains(&SPECULATIVE_PROTOCOL_FLAG.to_string())
        );
        // No drafter staged ⇒ neither leg carries the flag (the MTP-only no-perturbation control,
        // through the production selection this time).
        let (s_none, c_none) =
            paired_leg_spawn_args(&mtp_heads, None, LegRegime::FreeRunV1_1, SPEC_MODE_DFLASH);
        for args in [&s_none, &c_none] {
            assert!(!args.iter().any(|a| a == DFLASH_HEAD_FLAG), "{args:?}");
        }

        // BYO defaulting is preserved (one env set ⇒ both legs the pinned dir), exactly as the MTP
        // head family behaves.
        let collapsed = resolve_head_dirs(Some("/pinned/dflash"), None).unwrap();
        assert_eq!(collapsed.head_dir, collapsed.candidate_head_dir);

        // ABSENT ⇒ the flag is OMITTED, so an MTP-only track spawns EXACTLY the argv it spawned
        // before this lane. This is the no-perturbation control.
        let mtp_only = leg_spawn_args("/mtp", None, LegRegime::FreeRunV1_1);
        assert!(
            !mtp_only.iter().any(|a| a == DFLASH_HEAD_FLAG),
            "{mtp_only:?}"
        );
        assert_eq!(
            mtp_only,
            vec![
                "--mtp-head".to_string(),
                "/mtp".to_string(),
                SPECULATIVE_PROTOCOL_FLAG.to_string(),
                SPECULATIVE_PROTOCOL_V1_1.to_string(),
            ]
        );

        // The engine's option surface must ADMIT the flag, or every leg dies pre-GPU as an opaque
        // "engine closed the stream" — the #109 window-2 failure this fence exists to prevent.
        assert!(RUNTIME_WORKER_ACCEPTED_FLAGS.contains(&DFLASH_HEAD_FLAG));
        for args in [&serial_args, &candidate_args, &mtp_only] {
            validate_spawn_argv(args).unwrap();
        }
    }

    /// The DFlash head is REQUIRED for a dflash candidate, and only for one.
    ///
    /// Without it both legs fall back to the engine's CWD default and load the same drafter — the
    /// exact silent-wrong-measurement the per-leg channel exists to prevent, so an unset env must
    /// be a refusal and never a degradation.
    #[test]
    fn dflash_head_env_is_required_for_a_dflash_candidate() {
        let staged = resolve_head_dirs(Some("/pinned/dflash"), None);
        assert!(staged.is_some());

        // dflash + staged ⇒ ok. dflash + unstaged ⇒ refusal naming the env and the fallback.
        assert!(enforce_dflash_head_present(SPEC_MODE_DFLASH, staged.as_ref()).is_ok());
        let e = enforce_dflash_head_present(SPEC_MODE_DFLASH, None).unwrap_err();
        assert!(e.contains("QMTP_DFLASH_HEAD_DIR"), "{e}");
        assert!(e.contains("./dflash-head"), "{e}");

        // NEGATIVE CONTROL — the requirement is scoped to the dflash mode. An mtp or serial run on
        // a box with no drafter staged is UNAFFECTED; a gate that refused those would have broken
        // every existing track instead of enabling one mode.
        for mode in [SPEC_MODE_SERIAL, SPEC_MODE_MTP] {
            assert!(
                enforce_dflash_head_present(mode, None).is_ok(),
                "mode {mode} must not require a DFlash drafter"
            );
            assert!(enforce_dflash_head_present(mode, staged.as_ref()).is_ok());
        }
    }

    /// The `--dflash-head` EMISSION is scoped to a `dflash` candidate, not to whether the env
    /// happens to be exported.
    ///
    /// The env behind `dflash_head_dirs` is resolved MODE-INDEPENDENTLY in `execute_measure_job`,
    /// so before this scoping an operator shell still carrying `QMTP_DFLASH_HEAD_DIR` from an
    /// earlier dflash run put `--dflash-head` on BOTH legs of every subsequent pair — including an
    /// `mtp` pair. Against any engine build that predates the flag, the `runtime-worker` verb exits
    /// 1 on the first unknown option BEFORE the hello, so each leg of each pair died as the opaque
    /// "engine closed the stream before returning a response". `validate_spawn_argv` cannot catch
    /// it: it mirrors the flag surface benchd was BUILT against, not the engine on the box.
    ///
    /// REVERT-PROOF: drop the `candidate_mode` filter from `paired_leg_spawn_args` (emit whenever
    /// the dirs are `Some`) and the mtp/serial assertions below go red; scope it the wrong way
    /// (emit only for non-dflash) and the dflash assertions go red.
    #[test]
    fn dflash_spawn_flag_is_scoped_to_a_dflash_candidate() {
        let mtp_heads = resolve_head_dirs(Some("/pinned/mtp"), Some("/byo/mtp")).unwrap();
        let dflash_heads = resolve_head_dirs(Some("/pinned/dflash"), Some("/byo/dflash")).unwrap();

        // THE FINDING — env staged (dirs are `Some`), candidate mode is NOT dflash ⇒ NEITHER leg
        // carries the flag, and the argv is byte-for-byte the argv the same run spawned before the
        // DFlash channel existed.
        for mode in [SPEC_MODE_MTP, SPEC_MODE_SERIAL] {
            let (serial, candidate) = paired_leg_spawn_args(
                &mtp_heads,
                Some(&dflash_heads),
                LegRegime::FreeRunV1_1,
                mode,
            );
            for (leg, args) in [("serial", &serial), ("candidate", &candidate)] {
                assert!(
                    !args.iter().any(|a| a == DFLASH_HEAD_FLAG),
                    "{mode} candidate: {leg} leg carries {DFLASH_HEAD_FLAG} — a stale \
                     QMTP_DFLASH_HEAD_DIR export would kill this pair pre-hello on a pre-flag \
                     engine: {args:?}"
                );
                assert!(
                    !args.iter().any(|a| a.contains("dflash")),
                    "{mode} candidate: {leg} leg argv mentions a drafter it never loads: {args:?}"
                );
            }
            // NO PERTURBATION: exactly the pre-lane argv for both legs.
            assert_eq!(
                serial,
                leg_spawn_args(
                    "/pinned/mtp",
                    None,
                    serial_control_regime_for(LegRegime::FreeRunV1_1)
                ),
                "{mode} candidate: serial leg argv perturbed"
            );
            assert_eq!(
                candidate,
                leg_spawn_args("/byo/mtp", None, LegRegime::FreeRunV1_1),
                "{mode} candidate: candidate leg argv perturbed"
            );
        }

        // POSITIVE CONTROL — the same staged env on a `dflash` candidate still gives BOTH legs
        // their OWN per-leg drafter dir (the control leg loads it too; its residency charges the
        // denominator). Scoping the emission must not have defanged the channel it scopes.
        let (serial, candidate) = paired_leg_spawn_args(
            &mtp_heads,
            Some(&dflash_heads),
            LegRegime::FreeRunV1_1,
            SPEC_MODE_DFLASH,
        );
        let value_after = |args: &[String], flag: &str| -> String {
            let at = args
                .iter()
                .position(|a| a == flag)
                .unwrap_or_else(|| panic!("{flag} absent: {args:?}"));
            args[at + 1].clone()
        };
        assert_eq!(value_after(&serial, DFLASH_HEAD_FLAG), "/pinned/dflash");
        assert_eq!(value_after(&candidate, DFLASH_HEAD_FLAG), "/byo/dflash");
    }
}
