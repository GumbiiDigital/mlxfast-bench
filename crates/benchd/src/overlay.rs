//! A-3 — the OVERLAY component of the Option-A paired ranked flow (seam 3, LOCAL).
//!
//! `benchd overlay-timing` is benchd's LOCAL merge of the seam-1 producer's sealed `gates-score.json`
//! (seam 1, `partial_result=true`) with the measure-job's `results.json` (seam 2) into a sealed
//! ranked `score.json`. On the RANKED path the organizer's trusted shell owns seam 3 and authors
//! the published `score.json` (design note OPEN-2 CLOSED); this subcommand is the LOCAL-only
//! estimate AND the verifiable seam-3 PARITY reference — it mirrors the seam-3 semantics so a
//! local run matches what the organizer would seal. It does NOT wire a ranked `score.json` path
//! into production.
//!
//! Aggregation is the 3.8 MEDIAN regime (design note OPEN-1 CLOSED): the ranked score is the
//! median of the per-prompt raw ratio-of-means, serial-anchored, with NO no-op normalization,
//! per-pair plausibility bound 8.0 applied BEFORE aggregation, floor 0.90 / ceiling 5.0 on the
//! median. The numeric bounds + aggregation are the public 3.8 record
//! (`benchmark.qwen-mtp.json@3655c68d`); the DRAFT ranked merge STRUCTURE is
//! `qwen-mtp-ranked-benchmark.yml@064c0ff2:2108-2225`. The live-vs-draft diff is B-4, so the
//! aggregation choice carries `// UNVERIFIED(B-4)`.
//!
//! F1 CHANGE 2 — the overlay now consumes BOTH measurement shapes (see [`ResultsShape`]):
//!
//! * SINGLE-STREAM (`teacher_forced_v1`, `free_run_v1_1`) — one timed stream per pool prompt,
//!   `per_prompt` populated. Published score: the median regime described above, UNCHANGED.
//! * COHORT (`batched_free_run_v1_2_b8`) — the whole pool timed concurrently in ONE shared window,
//!   so `per_prompt` is empty by construction and the pool's identities live in
//!   `per_cohort[].members`. Published score: the sealed SHARED-WINDOW COMPOSITE
//!   (`prefill_gain^0.25 * decode_gain^0.75`), which is what a batched run measures — there are no
//!   per-prompt ratios to take a median of.
//!
//! This closes a seam that had never been runnable: the overlay knew only the two single-stream
//! regimes, so it refused its own measure-job's batched artifacts outright.
//!
//! The shape is read off the SERIES TAG, and the body must agree with it — a file cannot claim one
//! shape in its tag and carry the other's records. Everything the single-stream path enforces is
//! enforced on the cohort path too, over the unit that was actually measured: the gates-side harness
//! identity gate, the die-5 `candidate_accepted` verdict, the per-pair plausibility bound, the
//! sealed-median tamper detector (recomputed from the accepted pairs' cohort ratios), the pool
//! cardinality / identity / distinctness grid (over the members), and the 0.90 floor — applied to
//! the composite through the SAME reused `score_paired_decode_only` gate, so there is no second
//! scoring formula. The composite is additionally re-derived from its own sealed gains under the
//! RULED exponents before it is published.
//!
//! Merge semantics mirror the Gemma-era overlay reference
//! `mlxfast-challenge-dev/.github/scripts/overlay-paired-timing.sh@67699fc4`: flip
//! `partial_result → false` (`:159`), floor-fail null score + error prefix (`:161-168`), and
//! re-anchor integrity `score_sha256` over the merged bytes (`:177-181`).

use bench_core::constants::{QWEN_MTP_DECODE_SPEEDUP_CEILING, QWEN_MTP_DECODE_SPEEDUP_FLOOR};
use bench_core::score::{clears_floor, score_paired_decode_only, speedup, PairedDecodeFailure};
use bench_core::stats::even_n_median;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::iterate::HarnessIdentity;
use crate::score::ScorePayload;

/// finding R18 — the sealed-median agreement tolerance (Y:2646-2666): the overlay-recomputed
/// published median must agree with the sealed `aggregate.raw_decode_speedup_median` to within this
/// epsilon (`|recomputed - sealed| < 1e-7`), else the run is rejected as a wrapper tamper.
pub const SEALED_MEDIAN_AGREEMENT_EPS: f64 = 1e-7;

/// The scoring-mode discriminator sealed into the merged `score.json` (finding 11): a consumer
/// reads this to know the score is the paired DECODE-ONLY median regime and NOT the generic
/// `ds^0.75·ps^0.25` timed score. Same string the measure-job / track fixture use for the mode.
pub const SCORING_MODE: &str = "qwen-native-mtp-paired-decode-only";

/// The aggregation discriminator (DRAFT-WF `@064c0ff2:2205` `mode`; PUB38-BM
/// `scoring.aggregationNote`): names the 3.8 median regime so no consumer re-derives a formula.
pub const AGGREGATION: &str = "median_of_per_prompt_raw_ratio_of_means";

/// F1 CHANGE 2 — the aggregation discriminator for a COHORT run. A batched run's published score is
/// NOT the per-prompt median: it is the sealed shared-window composite. The two regimes must never
/// share a label, because a consumer reading `AGGREGATION` off a cohort score would believe it was
/// looking at a median of per-prompt ratios that the artifact does not even contain.
pub const AGGREGATION_COHORT_COMPOSITE: &str = "shared_window_composite_prefill_decode_gain";

/// F1 CHANGE 2 — the relative tolerance for the composite COHERENCE check: the sealed
/// `composite_score` must agree with `prefill_gain^e_p * decode_gain^e_d` recomputed from its own
/// sealed gains and the RULED exponents. Relative rather than absolute because the recompute goes
/// through `powf`, whose last ulp is not guaranteed identical across the producer's libm and the
/// overlay's. This is the composite's analogue of [`SEALED_MEDIAN_AGREEMENT_EPS`]: the overlay does
/// not publish a number it cannot re-derive from the parts the producer sealed beside it.
pub const COMPOSITE_COHERENCE_REL_EPS: f64 = 1e-9;

/// F1 CHANGE 2 — which measurement SHAPE a `results.json` carries. The two shapes seal DIFFERENT
/// pool records, so the overlay's pool predicates have to know which one it is holding.
///
/// The discriminator is the SERIES TAG, not the presence of a field. That is deliberate: the series
/// is already the load-bearing identity on this seam (`validate_series` refuses an unknown one, and
/// §5 makes baselines, floors and bands per-series), so deriving the shape from it means a file
/// cannot claim one shape in its tag and another in its body — the cross-check in
/// [`validate_results`] refuses exactly that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultsShape {
    /// One timed stream per pool prompt: `per_prompt` populated, `per_cohort` absent. The regimes
    /// `teacher_forced_v1` and `free_run_v1_1`.
    SingleStream,
    /// One batched COHORT of `batch_size` concurrent streams sharing a single timed window:
    /// `per_cohort` populated, `per_prompt` EMPTY. The regime `batched_free_run_v1_2_b8`.
    Cohort { batch_size: u32 },
}

/// The shape [`ResultsShape`] describes, read off the candidate leg's series tag.
///
/// Safe to read either leg: [`validate_series`] requires the two legs to be identical (homogeneous)
/// before any caller reaches a shape decision, so they cannot disagree here.
///
/// EXHAUSTIVE, not defaulted. This used to classify anything that was not the b8 tag as
/// [`ResultsShape::SingleStream`] through an `else` fallback, which made an UNKNOWN series tag
/// silently score as a single-stream run: `per_prompt` populated is all the single-stream arm of
/// [`validate_results`] demands, and a future regime whose tag nobody taught this function would
/// have been aggregated by the wrong rule rather than refused. That was safe only by CALL ORDERING
/// — every production caller happens to run [`validate_series`]'s known-tag refusal first — and the
/// function is `pub`, so the ordering is a convention, not a guarantee. It now matches the known
/// tags explicitly and REFUSES anything else, which makes the fail-closed property local to the
/// function instead of a property of where it is called from.
pub fn results_shape(results: &ResultsView) -> Result<ResultsShape, String> {
    let tag = results.timed_series.candidate_leg_timed_mode.trim();
    match tag {
        t if t == bench_core::free_run::TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8 => {
            Ok(ResultsShape::Cohort {
                batch_size: crate::measure_job::SCORED_BATCH_SIZE_B8,
            })
        }
        t if t == bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1
            || t == bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1 =>
        {
            Ok(ResultsShape::SingleStream)
        }
        unknown => Err(format!(
            "results.json timed_series.candidate_leg_timed_mode ({unknown:?}) is not a known timed \
             series ({:?} | {:?} | {:?}): the overlay cannot tell which measurement SHAPE the file \
             carries, so it refuses to pick one — a defaulted shape would aggregate an unknown \
             regime by the single-stream rule",
            bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1,
            bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1,
            bench_core::free_run::TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8,
        )),
    }
}

/// finding R17 — the expected pool SHAPE the overlay validates `results.json` against. The CLI
/// resolves it fail-closed: `pool_size` from env `MLXFAST_QWEN_MTP_POOL_SIZE`, else the `--contract`
/// fixture's `timed_prompt_pool | length` (unknown ⇒ hard error, no score); `min_per_prompt` from
/// env `MLXFAST_QWEN_MTP_MIN_PAIRS_PER_PROMPT` (default 1). The run-total floor is
/// `min_pairs = pool_size * min_per_prompt`.
#[derive(Debug, Clone, Copy)]
pub struct PoolExpectation {
    pub pool_size: usize,
    /// The minimum accepted pairs per MEASURED UNIT. On the single-stream shape the unit is a pool
    /// prompt (hence the name); on the cohort shape the unit is the one cohort, which IS the pool.
    /// See [`PoolExpectation::min_pairs_for`] for how each shape turns this into a run-total floor.
    pub min_per_prompt: usize,
}

impl PoolExpectation {
    /// The run-total accepted-pair floor for `shape`.
    ///
    /// SINGLE-STREAM: `pool_size * min_per_prompt` — each pool prompt is timed on its own, so the
    /// run owes `min_per_prompt` pairs for each of them.
    ///
    /// COHORT: `min_per_prompt` — a batched run times the WHOLE pool inside every pair, so there is
    /// exactly ONE measured unit and the same per-unit minimum applies to it once. This is not a
    /// weakening of the floor, it is the same floor counted over the thing that was measured;
    /// multiplying by `pool_size` here would demand `pool_size`x the pairs the regime produces and
    /// refuse every honest cohort artifact.
    ///
    /// Takes the shape rather than defaulting to one of them, so a caller cannot silently apply the
    /// wrong floor to the wrong artifact.
    pub fn min_pairs_for(&self, shape: ResultsShape) -> usize {
        match shape {
            ResultsShape::SingleStream => self.pool_size.saturating_mul(self.min_per_prompt),
            ResultsShape::Cohort { .. } => self.min_per_prompt,
        }
    }
}

// ---------------------------------------------------------------------------
// results.json — the slim VIEW the overlay consumes (measure_job::Results superset)
// ---------------------------------------------------------------------------

/// The measure-job `results.json` fields the overlay reads. A slim `Deserialize` view (rather than
/// re-using `measure_job::Results`, which is `Serialize`-only) so the overlay depends only on the
/// exact seam-2 fields it consumes: the raw per-side means, the per-pair raw ratios (for the 8.0
/// per-pair bound), and the per-prompt raw ratio-of-means (for the median). `aggregate` is
/// REQUIRED (a `results.json` without it fails to parse → fail-closed).
#[derive(Debug, Clone, Deserialize)]
pub struct ResultsView {
    /// finding R12 — the SEALED CONSTANT track id. REQUIRED (no `#[serde(default)]`): a
    /// results.json omitting `track_id` ERRORS at parse (fail-closed). `validate_results`
    /// additionally requires it non-empty AND, when an expected track is known, equal to it —
    /// the ranked yml gate is `.track_id == $track`, so an arbitrary track_id is refused.
    pub track_id: String,
    /// W3 — the sealed top-level SERIES DESCRIPTOR (`teacher_forced_v1`, `free_run_v1_1`, or the
    /// explicit MIXED descriptor). REQUIRED (no `#[serde(default)]`): a results.json that omits the
    /// series ERRORS at parse rather than being aggregated as an unknown-quantity number. §5 makes
    /// series identity load-bearing — a free-run seconds-per-token and a teacher-forced one are
    /// different physical quantities — so the overlay must never score a file that will not say
    /// which it is.
    pub timed_mode: String,
    /// W3 — the sealed per-leg SERIES DESCRIPTOR block. REQUIRED (fail-closed as above): the overlay
    /// cross-checks it against EVERY pair's per-leg tags, so a file whose legs disagree with the
    /// descriptor (or with each other) is refused instead of aggregated.
    pub timed_series: TimedSeriesView,
    #[serde(default)]
    pub parity_all_ok: bool,
    #[serde(default)]
    pub accepted_pair_count: usize,
    /// finding R2 — the measure-job's sealed die-5 verdict. REQUIRED (no `#[serde(default)]`): a
    /// forged `results.json` that omits it ERRORS at parse rather than defaulting to a passing
    /// bool. The overlay refuses to score a run that die-5'd (`candidate_accepted=false`).
    pub candidate_accepted: bool,
    /// finding R2 — the `--min-pairs` threshold the run required, sealed by the measure-job.
    /// REQUIRED (no `#[serde(default)]`) so the overlay re-checks `accepted_pair_count >= min_pairs`
    /// against the fact; a forged file that omits it ERRORS rather than defaulting to 0.
    pub min_pairs: usize,
    #[serde(default)]
    pub per_prompt: Vec<PerPromptView>,
    /// finding R17 — the sealed pool cardinality. REQUIRED (no `#[serde(default)]`): the ranked
    /// predicate is `prompt_count == pool_size == per_prompt.len()`; a results.json omitting it
    /// ERRORS at parse (fail-closed) rather than defaulting to 0.
    pub prompt_count: usize,
    /// finding R3 — REQUIRED (no `#[serde(default)]`): a forged `results.json` with NO `pairs`
    /// field must ERROR at parse, not default to an empty vec that then merges to a passing score.
    /// `validate_results` additionally enforces `pairs.len() == accepted_pair_count` and non-empty.
    pub pairs: Vec<PairView>,
    pub aggregate: AggregateView,
    /// Minor (overlay input identity cross-check) — the run commit + weights identity the
    /// measure-job sealed. The overlay requires these to MATCH the gates-score.json's
    /// `metrics.commit` / `metrics.weights_hash` (and be non-empty) so the two seam inputs are
    /// provably from the SAME run. Kept `#[serde(default)]` for tolerant parse; a MISSING/empty
    /// value fails the non-empty+equal cross-check (fail-closed), never fabricates a match.
    #[serde(default)]
    pub commit: String,
    #[serde(default)]
    pub weights_hash: String,
    /// F1 CHANGE 2 — the COHORT records. `#[serde(default)]` (empty) so a single-stream
    /// `results.json`, which omits the field entirely, still parses; the shape dispatch in
    /// [`validate_results`] then REQUIRES it non-empty on a cohort run and REQUIRES it empty on a
    /// single-stream one, so the default can never stand in for a missing cohort seal.
    #[serde(default)]
    pub per_cohort: Vec<PerCohortView>,
    /// The sealed cohort WIDTH (the measure-job's `scored_batch_size`). Carried so the width the
    /// artifact claims can be cross-checked against the one its own series tag encodes, rather than
    /// parsing B back out of the tag. `#[serde(default)]` (`None`) on a single-stream file.
    #[serde(default)]
    pub scored_batch_size: Option<u32>,
}

/// F1 CHANGE 2 — one COHORT's record (`measure_job::PerCohort`), the batched counterpart of
/// [`PerPromptView`]. A batched run seals exactly one of these: the whole pool is timed
/// concurrently in a single shared window, so the unit of measurement is the cohort, not the prompt.
///
/// The pool's prompt identities live in `members` rather than in `per_prompt` — which is why the
/// overlay's pool-shape predicates have to look here on a cohort run instead of refusing an empty
/// `per_prompt`.
#[derive(Debug, Clone, Deserialize)]
pub struct PerCohortView {
    /// The recomputable digest over `members` — the cohort's identity.
    #[serde(default)]
    pub cohort_sha256: String,
    /// The pool prompts in SLOT ORDER; `members[i]` is stream slot `i`.
    #[serde(default)]
    pub members: Vec<CohortMemberView>,
    /// The cohort width this record was measured at.
    #[serde(default)]
    pub batch_size: u32,
    #[serde(default)]
    pub parity_ok: bool,
    #[serde(default)]
    pub accepted_pair_count: usize,
    #[serde(default)]
    pub serial_seconds_per_token_mean: f64,
    /// NAME TRAP: the cohort record calls the candidate mean `candidate_seconds_per_token_mean`,
    /// where [`PerPromptView`] calls it `mtp_seconds_per_token_mean` and [`AggregateView`] calls it
    /// `candidate_mtp_seconds_per_token_mean`. Three names, one quantity.
    #[serde(default)]
    pub candidate_seconds_per_token_mean: f64,
    #[serde(default)]
    pub raw_ratio_of_means: f64,
    /// The exponent pair the composite was actually raised to, sealed beside it. Cross-checked
    /// against the RULED constants: a run does not get to choose its own scoring exponents.
    #[serde(default)]
    pub composite_scored_exponents: Option<ScoredExponentsView>,
    /// THE PUBLISHED SCORE of a cohort run — the sealed shared-window composite. `Option` because
    /// the producer omits it (and seals `composite_absent_reason` instead) when it refused to
    /// compute one; the overlay REQUIRES it, so a cohort run without a composite publishes nothing.
    #[serde(default)]
    pub composite: Option<CompositeView>,
    /// Why the composite is absent, when it is. Echoed in the refusal so an operator does not have
    /// to go back to the results.json to find out.
    #[serde(default)]
    pub composite_absent_reason: Option<String>,
}

/// One pool prompt's slot in a cohort (`measure_job::CohortMember`).
#[derive(Debug, Clone, Deserialize)]
pub struct CohortMemberView {
    #[serde(default)]
    pub slot_index: usize,
    #[serde(default)]
    pub prompt_sha256: String,
}

/// The sealed exponent pair (`measure_job::ScoredExponents`).
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct ScoredExponentsView {
    #[serde(default)]
    pub prefill_gain_exponent: f64,
    #[serde(default)]
    pub decode_gain_exponent: f64,
}

/// The sealed shared-window composite (`measure_job::CompositeCohortScore`) — the ratio of SUMMED
/// parent-clocked phase windows across the accepted pairs, serial-anchored, combined as
/// `prefill_gain^0.25 * decode_gain^0.75`.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct CompositeView {
    #[serde(default)]
    pub prefill_gain: f64,
    #[serde(default)]
    pub decode_gain: f64,
    #[serde(default)]
    pub composite_score: f64,
    #[serde(default)]
    pub composite_speedup_floor: f64,
    #[serde(default)]
    pub composite_speedup_floor_met: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AggregateView {
    #[serde(default)]
    pub baseline_serial_seconds_per_token_mean: f64,
    #[serde(default)]
    pub candidate_mtp_seconds_per_token_mean: f64,
    /// finding R17 — the per-PAIR lower-median diagnostic (NAME-TRAP: NOT the published score).
    /// REQUIRED (no `#[serde(default)]`): the ranked pool predicate requires it to be a number, so
    /// a results.json omitting it ERRORS at parse (fail-closed) rather than defaulting to 0.
    pub mtp_decode_speedup_median: f64,
    /// finding R17 — the minimum per-pair ratio diagnostic. REQUIRED (fail-closed as above).
    pub mtp_decode_speedup_min: f64,
    /// finding R18 — the SEALED published median (even-n median of the per-prompt raw ratios). The
    /// overlay RECOMPUTES the median from the per_prompt means and requires agreement with THIS
    /// sealed number within 1e-7 (the wrapper-tamper detector); it then uses the RECOMPUTED median
    /// as the published score, never this field trusted blindly. REQUIRED (no `#[serde(default)]`):
    /// a results.json omitting it ERRORS at parse — the tamper check must have a sealed value.
    pub raw_decode_speedup_median: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PerPromptView {
    /// R17 — per-prompt SHA (the pool binds by bytes). Validated `^[0-9a-f]{64}$` and required
    /// DISTINCT across the pool. `#[serde(default)]` (empty) tolerantly parses; the empty default
    /// fails the hex predicate (fail-closed).
    #[serde(default)]
    pub prompt_sha256: String,
    /// R17 — this prompt's parity verdict; the pool predicate requires it true.
    #[serde(default)]
    pub parity_ok: bool,
    /// R17 — accepted pairs for THIS prompt; the pool predicate requires `>= min_per_prompt`.
    #[serde(default)]
    pub accepted_pair_count: usize,
    /// R17/R18 — the per-prompt serial seconds-per-token mean (>0). R18 recomputes the per-prompt
    /// ratio as `serial_seconds_per_token_mean / mtp_seconds_per_token_mean` from THESE means.
    #[serde(default)]
    pub serial_seconds_per_token_mean: f64,
    /// R17/R18 — the per-prompt candidate MTP seconds-per-token mean (>0).
    #[serde(default)]
    pub mtp_seconds_per_token_mean: f64,
    /// R17 — the per-prompt sealed raw ratio-of-means (>0). Cross-checked against the R18 recompute
    /// from the means at the aggregate level (the sealed median agreement).
    #[serde(default)]
    pub raw_ratio_of_means: f64,
    /// R17 — the exactly-one positive no-op reference decode speedup for this prompt (>0). Sealed
    /// as `Option<f64>` by the measure-job (OMITTED when absent); the pool predicate requires it
    /// present AND >0, so a default 0.0 (absent) fails (fail-closed).
    #[serde(default)]
    pub noop_reference_decode_speedup: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PairView {
    #[serde(default)]
    pub raw_ratio: f64,
    /// W3 — the §5 series tag of each leg of THIS pair. REQUIRED (no `#[serde(default)]`): the
    /// per-pair tags are what make the sealed descriptor falsifiable, so a pair record that omits
    /// them ERRORS at parse rather than inheriting the descriptor's claim.
    pub serial_timed_mode: String,
    pub candidate_timed_mode: String,
}

/// W3 — the sealed per-leg SERIES DESCRIPTOR the overlay reads (the measure-job's `TimedSeries`).
#[derive(Debug, Clone, Deserialize)]
pub struct TimedSeriesView {
    pub serial_leg_timed_mode: String,
    pub candidate_leg_timed_mode: String,
    pub homogeneous: bool,
    /// The sealed §5 comparability verdict between the two legs. The overlay RECOMPUTES this with
    /// `bench_core::free_run::timed_modes_comparable` and refuses a file whose sealed verdict
    /// disagrees — a "comparable: true" stamped onto a cross-series pairing is exactly the lie the
    /// comparability rule exists to catch.
    pub legs_comparable: bool,
}

impl ResultsView {
    /// Parse a `results.json` bytes, FAIL-CLOSED on malformed JSON / missing `aggregate`.
    pub fn parse(bytes: &[u8]) -> Result<ResultsView, String> {
        serde_json::from_slice(bytes).map_err(|e| format!("results.json parse failed: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Fail-closed validation of the two inputs
// ---------------------------------------------------------------------------

/// A well-formed `harness_hash` is a SHA256 hex digest — 64 LOWERCASE hex characters
/// (`^[0-9a-f]{64}$`), the shape the engine Swift `harnessHash()` emits. This is the same grid
/// [`crate::measure_job::validate_prompt_sha256`] enforces on the per-prompt SHAs; kept as a local
/// predicate so the gates/seam-1 harness-identity refusal states its own shape inline.
fn is_well_formed_harness_hash(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Validate the sealed `gates-score.json` (seam 1 out) is a PASSING gates score before the merge
/// (DRAFT-WF `@1493-1506` predicates): `passed==true`, `partial_result==true` (it is the gates
/// half awaiting the timed overlay), `metrics.error==""`, `passed_correctness==true`. Any breach
/// REFUSES the merge — the overlay never seals a ranked score onto an unvalidated / failed gate.
///
/// B2 (#223 review gap) — additionally refuse a gates/seam-1 score whose official baseline pair
/// (`metrics.baseline_{decode,prefill}_seconds_per_token`) is missing, non-finite or non-positive.
/// The pair is the scoring denominator; `ScoreMetrics` is `#[serde(default)]`, so an OMITTED key
/// deserializes to `0.0` and used to pass unexamined.
///
/// RULED SCOPE-DOWN (David) — additionally refuse a gates/seam-1 score whose `metrics.harness_hash`
/// is EMPTY or MALFORMED (not the 64-lowercase-hex SHA256 shape `harnessHash()` produces). This is a
/// SINGLE-LEG harness-identity gate on the gates score itself. It was ORIGINALLY scoped down to
/// exactly that — "NOT a cross-leg equality between seams (deferred while only one harness_hash leg
/// exists)" — because before F1 benchd had no harness identity of its own to compare against.
///
/// DEFERRAL LIFTED — **David ruling 2026-08-26**. F1 (PR #197) removed the deferral's premise: benchd
/// now computes the 9-root harness identity TRUSTED-SIDE ([`HarnessIdentity`]) and a second leg
/// therefore exists. The cross-leg EQUALITY is now enforced, but it lives at the SEAL — see
/// [`validate_harness_identity_cross_leg`] and [`merge_overlay`] — not here, and this single-leg gate
/// is deliberately UNCHANGED: it stays the EARLIER, CHEAPER refusal that rejects an empty/malformed
/// identity before the seal-time recompute is even attempted, so the pre-F1 refusal texts and their
/// precedence are preserved exactly.
///
/// The parity differ's deliberately-empty benchd `harness_hash` (§13 EXPECT_DIFFER waiver) is a
/// DISTINCT path (`parity.rs` ROSTER → `Env`) and is untouched: this refusal only guards the
/// overlay's gates input.
pub fn validate_gates(gates: &ScorePayload) -> Result<(), String> {
    if !gates.passed {
        return Err("gates-score.json is not a passing gates score (passed != true)".to_string());
    }
    // Minor (validate_gates requires score == null): a gates-only artifact must NOT already carry a
    // score — the overlay computes it. A non-null score means the input is a full/already-scored
    // artifact, not the gates half awaiting the timed overlay; refuse it.
    if gates.score.is_some() {
        return Err(format!(
            "gates-score.json already carries a non-null score ({:?}): the overlay computes the \
             score; refusing to overlay onto an already-scored artifact",
            gates.score
        ));
    }
    if !gates.metrics.partial_result {
        return Err(
            "gates-score.json partial_result != true: not a gates-only score awaiting the timed \
             overlay (a full score must not be overlaid again)"
                .to_string(),
        );
    }
    if !gates.metrics.error.is_empty() {
        return Err(format!(
            "gates-score.json carries a non-empty metrics.error ({:?}); refusing to overlay onto a \
             failed gate",
            gates.metrics.error
        ));
    }
    if !gates.metrics.passed_correctness {
        return Err(
            "gates-score.json passed_correctness != true: refusing to overlay timing onto a gate \
             that did not pass correctness"
                .to_string(),
        );
    }
    // B2 (#223 review gap) — the SCORING DENOMINATOR must be present. `ScoreMetrics` is
    // `#[serde(default)]`, so a `gates-score.json` that OMITS `baseline_decode_seconds_per_token`
    // or `baseline_prefill_seconds_per_token` deserializes them as `0.0` — and nothing here used to
    // look. The overlay then merged a ranked score onto a gate whose denominator was silently
    // absent, which reads exactly like a gate that had one. A real gates run resolves the pair
    // before it seals anything (`official::official_gates_only`), so a missing half is a malformed
    // or hand-edited input, never a legitimate state.
    //
    // FAIL-CLOSED and BY NAME on each half independently: present, finite, and positive.
    for (field, value) in [
        (
            "baseline_decode_seconds_per_token",
            gates.metrics.baseline_decode_seconds_per_token,
        ),
        (
            "baseline_prefill_seconds_per_token",
            gates.metrics.baseline_prefill_seconds_per_token,
        ),
    ] {
        if !(value.is_finite() && value > 0.0) {
            return Err(format!(
                "gates-score.json metrics.{field} is {value}: the official baseline pair is the \
                 scoring denominator and must be present, finite and positive. ScoreMetrics is \
                 serde(default), so an OMITTED key reads back as 0.0 — an absent denominator that \
                 looks like a declared one; refusing to overlay a ranked score onto it"
            ));
        }
    }
    // RULED SCOPE-DOWN (David) — the gates/seam-1 harness-identity gate. The gates score carries the
    // producer's REAL harness hash; a gate whose identity is missing (empty) or not a well-formed
    // SHA256 digest (malformed) is refused before it can be overlaid into a ranked score. Only the
    // hash LENGTH is named on the malformed branch — the value itself is never echoed.
    if gates.metrics.harness_hash.is_empty() {
        return Err(
            "gates-score.json carries an empty metrics.harness_hash: the gates/seam-1 score must \
             carry the producer harness identity (a 64-char lowercase-hex SHA256); refusing to \
             overlay a gate with no harness identity"
                .to_string(),
        );
    }
    if !is_well_formed_harness_hash(&gates.metrics.harness_hash) {
        return Err(format!(
            "gates-score.json carries a malformed metrics.harness_hash (expected 64 lowercase hex \
             characters, got length {}): refusing to overlay a gate whose harness identity is not \
             a well-formed SHA256 digest",
            gates.metrics.harness_hash.len()
        ));
    }
    Ok(())
}

/// How many leading hex characters of a harness digest a refusal QUOTES. Enough to identify which
/// two digests disagreed in an operator's log, short enough that the refusal is not itself a
/// transcript of the sealed identity.
const HARNESS_DIGEST_PREVIEW_LEN: usize = 12;

/// The first [`HARNESS_DIGEST_PREVIEW_LEN`] characters of `digest`, or the whole thing when it is
/// shorter (both callers have already passed the well-formedness gate, so in practice it is 64 hex).
fn digest_preview(digest: &str) -> &str {
    &digest[..digest.len().min(HARNESS_DIGEST_PREVIEW_LEN)]
}

/// **David ruling 2026-08-26** — the CROSS-LEG HARNESS-IDENTITY EQUALITY at the seal.
///
/// `gates.metrics.harness_hash` is the harness identity as of the GATES phase (seam 1). `seal_leg`
/// is the identity benchd resolves for itself, TRUSTED-SIDE, at the moment it merges — after seam 2
/// has run. The two must be equal.
///
/// ## The window this closes
///
/// The roster-of-eight content check is die-8, and it runs PRE-GPU. `harnessHashRoots`
/// substantially overlaps that roster, so an engine that mutates a roster path DURING the run —
/// after the pre-GPU pass, before the seal — is a BETWEEN-PHASE TOCTOU the pre-GPU check cannot
/// see by construction: it looked at a tree that no longer exists by the time the score is authored.
/// Re-resolving the identity at the seal and demanding equality with the gates leg is the per-phase
/// re-verify posture (#148 ON) applied to the harness identity itself.
///
/// ## Why the legs are comparable (the CWD contract)
///
/// A harness hash covers the workspace's ABSOLUTE location as well as its bytes
/// (`bench_core::harness_hash` module doc, item 1: "an identity of *this tree at this path on this
/// box*, NOT a portable content digest"). Equality is therefore only meaningful when both legs
/// resolved the SAME root. Under the ruling the TRUSTED DRIVER pins that: `scripts/official-paired.sh`
/// captures the gates leg's workspace root at gates time (`GATES_WS`) and `cd`s to exactly that root
/// before invoking `benchd overlay-timing`, so this recompute — which is CWD-relative, exactly as
/// the reference's own production resolution is — resolves the gates leg's tree.
///
/// ## The parity oracle this doubles as
///
/// On the DEFAULT ranked flow (`GATES_PRODUCER=benchmark-sh`, RULING Q1a) the gates leg is the
/// ENGINE's SWIFT `QwenRuntime.harnessHash()` and this leg is benchd's RUST
/// `bench_core::harness_hash` port, over the same tree at the same absolute root. So the equality is
/// simultaneously a PER-RUN Swift↔Rust cross-implementation parity oracle on the port — live on
/// every ranked run, not just in the port's own test vectors. A port drift and a between-phase
/// mutation both surface here; both are refusals, and both are things a published score must not
/// carry.
///
/// FATAL, never a warning: a score whose harness identity is not the one that was gated is exactly
/// the artifact this repo must not publish.
pub fn validate_harness_identity_cross_leg(
    gates: &ScorePayload,
    seal_leg: &HarnessIdentity,
) -> Result<(), String> {
    if gates.metrics.harness_hash == seal_leg.as_str() {
        return Ok(());
    }
    Err(format!(
        "overlay cross-leg harness-identity mismatch: the gates/seam-1 score was sealed under \
         harness {gates_preview}… but the harness resolved AT THE SEAL is {seal_preview}… — the \
         harness tree CHANGED BETWEEN PHASES (the roster/harness content check is pre-GPU, so a \
         root mutated after that pass and before this seal is invisible to it), or the two legs \
         did not resolve the same workspace root. Refusing to publish a score whose harness \
         identity is not the one that was gated.",
        gates_preview = digest_preview(&gates.metrics.harness_hash),
        seal_preview = digest_preview(seal_leg.as_str()),
    ))
}

/// Resolve the SEAL-TIME harness identity, FAIL-CLOSED, reusing F1's production resolution
/// ([`HarnessIdentity::resolve_from_current_dir`]) — there is no second resolution path here.
///
/// A resolution failure at the seal is a REFUSAL, not a skip: "benchd could not identify the harness
/// it is about to publish a score for" must never soften into "so publish it unchecked". Same
/// discipline (and the same underlying `harnessHash root missing from disk: …` text) as F1's
/// refusal at the top of `iterate`.
///
/// THIS RECOMPUTE IS NOT REDUNDANT WORK — do not "optimise" it into a pass-through of the value
/// `iterate` already sealed. `gates.metrics.harness_hash` IS that sealed value, and it is the thing
/// this leg exists to disagree with: [`validate_harness_identity_cross_leg`] compares the two, and
/// two copies of one number can never disagree. The walk happens in a SEPARATE PROCESS, at a
/// SEPARATE TIME (after seam 2), which is precisely what makes it able to see a roster tree mutated
/// between the pre-GPU content check and the seal — and, on the default ranked flow, what makes the
/// equality a live Swift↔Rust parity oracle on the port. Reusing the gates leg's value would delete
/// both. The walk itself is the one in `bench_core::harness_hash`, which reads ahead off the
/// hashing thread; that is where this leg's cost is reduced, not here.
fn resolve_seal_harness_identity() -> Result<HarnessIdentity, String> {
    HarnessIdentity::resolve_from_current_dir().map_err(seal_resolution_refusal)
}

/// The seal-time resolution refusal text. Split from [`resolve_seal_harness_identity`] so the
/// message discipline is unit-testable WITHOUT mutating the process CWD — which a parallel test
/// suite may not do, and which is precisely why this is the only seam between the two.
fn seal_resolution_refusal(cause: String) -> String {
    format!(
        "the harness identity could not be resolved AT THE SEAL ({cause}); benchd overlay-timing \
         must run with its working directory at the SAME engine workspace root the seam-1 gates \
         producer resolved (the trusted driver pins this), and refuses to publish a score it cannot \
         cross-check the harness identity of"
    )
}

/// W3 — the SERIES FENCE (PROTOCOL-v1.1.md §5, machine-checked rather than conventional). A
/// `results.json` carries an all-teacher-forced run or an all-free-run run, and the overlay must
/// never AGGREGATE across series, SCORE a cross-series ratio, or score a file whose series claims do
/// not hold together. Fail-closed on any of:
///
/// 1. an UNKNOWN series tag on either leg (the overlay scores the two series it knows; a third tag
///    is a file from a regime it cannot reason about);
/// 2. a PAIR whose per-leg tags disagree with the sealed descriptor — including the case that made
///    the fence necessary: pairs from two different candidate regimes pooled into one `pairs[]`,
///    whose ratios would then be a median over two physical quantities;
/// 3. a sealed `homogeneous` / `legs_comparable` that disagrees with the RECOMPUTED verdict
///    (`timed_modes_comparable`) — a run stamped "comparable" whose legs are not is the §5 lie
///    itself;
/// 4. LEGS THAT ARE NOT §5-COMPARABLE — a cross-series ratio, i.e. the
///    [`crate::measure_job::MIXED_SERIES_DESCRIPTOR`]
///    shape. §5: "reusing a v1 teacher-forced baseline (or vice versa) is a scoring bug, not a
///    conservative choice", so the honest MIXED seal is refused rather than scored;
/// 5. a top-level `timed_mode` that does not match its own descriptor (the single tag when
///    homogeneous, the explicit MIXED descriptor otherwise);
/// 6. when `expected_series` is known, a file sealed for a DIFFERENT series — the §5 rule that
///    baselines/floors/bands are per-series means a v1.1 run must not be scored against v1
///    calibration or ranked beside v1 numbers.
///
/// The comparability decision is `bench_core::free_run::timed_modes_comparable` — THE decision
/// function, the same one `measure_job::enforce_calibration_series_fence` uses on the calibration
/// pre-read and the same one the seal computes. One series story across calibration, seal and
/// overlay.
///
/// DEFENSE IN DEPTH — check (4) is now unreachable from measure-job: under the Fable same-series
/// ruling the serial control runs the candidate's regime, so every run measure-job produces is
/// homogeneous. The fence keeps refusing crossed files anyway; the overlay validates the file in
/// front of it rather than trusting the producer that was supposed to have made one.
pub fn validate_series(results: &ResultsView, expected_series: Option<&str>) -> Result<(), String> {
    // F1 CHANGE 2 — the batched cohort regime joins the two single-stream regimes as a series the
    // overlay can reason about. It is an EXACT match on the b8 tag, not a prefix match on
    // `batched_free_run_v1_2_b`: `ScoredBatchPoint::certify` only ever certifies B=8, so any other
    // width is a regime no producer can honestly have sealed and the fence keeps refusing it. B is
    // encoded IN the tag, so `timed_modes_comparable`'s plain string equality already refuses a b1
    // leg paired with a b8 one — no batch-aware gate logic anywhere.
    let known = |tag: &str| {
        tag == bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1
            || tag == bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1
            || tag == bench_core::free_run::TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8
    };
    let serial_tag = results.timed_series.serial_leg_timed_mode.trim();
    let candidate_tag = results.timed_series.candidate_leg_timed_mode.trim();
    for (leg, tag) in [("serial", serial_tag), ("candidate", candidate_tag)] {
        if !known(tag) {
            return Err(format!(
                "results.json timed_series.{leg}_leg_timed_mode ({tag:?}) is not a known timed \
                 series ({:?} | {:?} | {:?}): refusing to score a regime the overlay cannot reason \
                 about",
                bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1,
                bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1,
                bench_core::free_run::TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8,
            ));
        }
    }
    // (2) every pair must belong to the series the descriptor claims. This is the check that refuses
    // to AGGREGATE mismatched series: a `pairs[]` mixing candidate regimes cannot pass it.
    for (i, p) in results.pairs.iter().enumerate() {
        if p.serial_timed_mode.trim() != serial_tag {
            return Err(format!(
                "results.json pairs[{i}].serial_timed_mode ({:?}) != timed_series\
                 .serial_leg_timed_mode ({serial_tag:?}): the sealed series does not describe this \
                 pair — refusing to aggregate mismatched series",
                p.serial_timed_mode
            ));
        }
        if p.candidate_timed_mode.trim() != candidate_tag {
            return Err(format!(
                "results.json pairs[{i}].candidate_timed_mode ({:?}) != timed_series\
                 .candidate_leg_timed_mode ({candidate_tag:?}): the sealed series does not describe \
                 this pair — refusing to aggregate mismatched series",
                p.candidate_timed_mode
            ));
        }
    }
    // (3) the sealed verdicts must equal the RECOMPUTED ones (never trusted as stamped).
    let comparable = bench_core::free_run::timed_modes_comparable(serial_tag, candidate_tag);
    if results.timed_series.legs_comparable != comparable {
        return Err(format!(
            "results.json timed_series.legs_comparable ({}) disagrees with the recomputed §5 \
             comparability of {serial_tag:?} vs {candidate_tag:?} ({comparable}): refusing a sealed \
             comparability claim the rule does not support",
            results.timed_series.legs_comparable
        ));
    }
    if results.timed_series.homogeneous != (serial_tag == candidate_tag) {
        return Err(format!(
            "results.json timed_series.homogeneous ({}) disagrees with its own per-leg tags \
             ({serial_tag:?} vs {candidate_tag:?})",
            results.timed_series.homogeneous
        ));
    }
    // (4) §5 — a cross-series ratio is not scoreable. `raw_ratio` divides the serial leg's
    // seconds-per-token by the candidate's; when the two legs measured different quantities that
    // quotient is not a speedup, whatever it is stamped. Refused here rather than published with a
    // `legs_comparable: false` caveat nothing downstream is obliged to read.
    if !comparable {
        return Err(format!(
            "results.json legs are NOT §5-comparable (serial {serial_tag:?} vs candidate \
             {candidate_tag:?}): every pair's raw_ratio divides two DIFFERENT measured quantities, \
             so the median is not a speedup — refusing to score a cross-series ratio. PROTOCOL-v1.1 \
             §5: a v1.1 score's baseline MUST be measured in v1.1 free-run mode; reusing a v1 \
             teacher-forced baseline (or vice versa) is a scoring bug, not a conservative choice. \
             Re-measure with the serial control in the candidate's series."
        ));
    }
    // (5) the top-level descriptor must match the block it summarizes.
    let expected_descriptor = if serial_tag == candidate_tag {
        serial_tag
    } else {
        crate::measure_job::MIXED_SERIES_DESCRIPTOR
    };
    if results.timed_mode.trim() != expected_descriptor {
        return Err(format!(
            "results.json timed_mode ({:?}) != the descriptor its own timed_series implies \
             ({expected_descriptor:?}): the top-level series label and the per-leg tags disagree",
            results.timed_mode
        ));
    }
    // (6) the series the overlay was told to score, when known.
    if let Some(expected) = expected_series.map(str::trim).filter(|s| !s.is_empty()) {
        if results.timed_mode.trim() != expected {
            return Err(format!(
                "results.json timed_mode ({:?}) != expected series ({expected:?}): §5 makes \
                 baselines, floors and bands PER-SERIES, so a run of one series is never scored as \
                 another",
                results.timed_mode
            ));
        }
    }
    Ok(())
}

/// Validate the measure-job `results.json` (seam 2 out) before the merge (DRAFT-WF `@2145-2153`):
/// finding R12 (`track_id` present, non-empty, and — when `expected_track` is known — equal to it),
/// `parity_all_ok==true`, `accepted_pair_count>=1`, the die-5 seal holds (finding R2:
/// `candidate_accepted==true` AND `accepted_pair_count>=min_pairs`), and `per_prompt` non-empty.
/// Fail-closed otherwise. `expected_track` is the track the overlay was told to score (env
/// `MLXFAST_QWEN_MTP_TRACK_ID`, or a track the gates-score carries); `None` skips the equality
/// check but still requires a non-empty sealed `track_id` (never trust an arbitrary one).
///
/// W3 — plus the SERIES FENCE (PROTOCOL-v1.1.md §5): see [`validate_series`]. `expected_series` is
/// the series the overlay was told to score (env `MLXFAST_QWEN_MTP_TIMED_SERIES`); `None` skips the
/// equality check but never skips the internal coherence checks.
pub fn validate_results(
    results: &ResultsView,
    expected_track: Option<&str>,
    expected_series: Option<&str>,
    pool: &PoolExpectation,
    scored: &bench_core::contract::Contract,
) -> Result<(), String> {
    validate_series(results, expected_series)?;
    // finding R12 — the ranked yml gate is `.track_id == $track`. Refuse an empty sealed track_id,
    // and (when the expected track is known) any that does not equal it — fail-closed, no score.
    let track_id = results.track_id.trim();
    if track_id.is_empty() {
        return Err(
            "results.json track_id is empty: the sealed track_id constant is required (the ranked \
             gate is .track_id == $track)"
                .to_string(),
        );
    }
    if let Some(expected) = expected_track.map(str::trim).filter(|s| !s.is_empty()) {
        if track_id != expected {
            return Err(format!(
                "results.json track_id ({track_id:?}) != expected track ({expected:?}): refusing \
                 to overlay a run sealed for a different track"
            ));
        }
    }
    // The THIRD leg of `constant≡contract≡env`. The two checks above only make the run agree with
    // ITSELF: an operator env and a results.json that both name a FOREIGN track pass them. The
    // gates half of this merge was sealed by THIS tree, against THIS track's official baseline
    // pair, so merging it under another track's `track_id` mis-attributes that pair — and the
    // commit/weights_hash cross-checks in `validate_identity` cannot see it, because both halves
    // really are from the same run. Refuse by name, naming both values.
    // PAIRED-FLOW ENTRY FENCE — the overlay half of the same refusal `measure-job` carries: a
    // track whose sole scored path is the single-leg `iterate --mode official` never has a paired
    // results.json to overlay, and must not have one merged for it. Refuses BY NAME, before the
    // tree fence, for the same reason.
    bench_core::contract::enforce_paired_flow_available(scored, track_id)?;
    bench_core::constants::enforce_declared_track(track_id)?;
    // B2 — the SCORING-TIME REGIME FENCE. The track has to declare WHAT it scores — the batch size
    // of its scored point and the composite exponents — before this merge produces a ranked score
    // for it (`docs/scored-regime-and-prefill-window.md`). A PENDING track refuses by name here, the
    // same way an uncaptured baseline pair does.
    //
    // The line above already pins `track_id` to `TRACK_ID`, whose regime IS declared, so on this
    // branch this is satisfied by construction. It stops the next release branch that sets its own
    // `TRACK_ID` and declares no regime, not anything reachable today.
    bench_core::contract::scored_regime(scored, track_id)?;
    if !results.parity_all_ok {
        return Err(
            "results.json parity_all_ok != true: the paired run had a parity failure".to_string(),
        );
    }
    if results.accepted_pair_count < 1 {
        return Err(format!(
            "results.json accepted_pair_count ({}) < 1: no accepted pairs to score",
            results.accepted_pair_count
        ));
    }
    // finding R2 — the die-5 verdict is a FACT the overlay must honor: a run that die-5'd
    // (candidate rejected) must NEVER be overlaid to a passing score. Require BOTH the sealed
    // `candidate_accepted` verdict AND the numeric `accepted_pair_count >= min_pairs` predicate.
    if !results.candidate_accepted {
        return Err(
            "results.json candidate_accepted != true: the measure-job die-5'd (candidate \
             rejected); refusing to overlay a passing score onto a rejected run"
                .to_string(),
        );
    }
    if results.accepted_pair_count < results.min_pairs {
        return Err(format!(
            "results.json accepted_pair_count ({}) < min_pairs ({}): the run did not reach the \
             required pair count (die 5)",
            results.accepted_pair_count, results.min_pairs
        ));
    }
    // F1 CHANGE 2 — the SHAPE CROSS-CHECK. The shape comes from the series tag; the body must
    // match it. A single-stream file still needs `per_prompt` (the old refusal, unchanged for that
    // shape) and must NOT carry cohort records; a cohort file needs `per_cohort` and must NOT carry
    // per-prompt records. Neither shape may claim to be both, and neither may be empty of the
    // records its own regime is defined by.
    let shape = results_shape(results)?;
    match shape {
        ResultsShape::SingleStream => {
            if results.per_prompt.is_empty() {
                return Err(
                    "results.json per_prompt is empty: nothing to aggregate the median over"
                        .to_string(),
                );
            }
            if !results.per_cohort.is_empty() {
                return Err(format!(
                    "results.json seals {} per_cohort record(s) on a SINGLE-STREAM series ({:?}): \
                     the series tag and the body disagree about what was measured",
                    results.per_cohort.len(),
                    results.timed_mode
                ));
            }
        }
        ResultsShape::Cohort { .. } => {
            if results.per_cohort.is_empty() {
                return Err(format!(
                    "results.json per_cohort is empty on a COHORT series ({:?}): a batched run's \
                     unit of measurement is the cohort, so there is nothing to score",
                    results.timed_mode
                ));
            }
            if !results.per_prompt.is_empty() {
                return Err(format!(
                    "results.json seals {} per_prompt record(s) on a COHORT series ({:?}): the \
                     pool prompts of a batched run are sealed as per_cohort[].members, so a \
                     populated per_prompt means the series tag and the body disagree",
                    results.per_prompt.len(),
                    results.timed_mode
                ));
            }
        }
    }
    // finding R3 — match the measure-job's OWN seal predicates (measure_job.rs:746 doc +
    // debug_assert `accepted_pair_count == pairs.len()`). A looser overlay let a forged NO-PAIRS
    // results.json merge to passed=true: require `pairs` non-empty AND `pairs.len() ==
    // accepted_pair_count`, and both aggregate means strictly positive (the per-pair bound and the
    // ratio orientation are otherwise scored off fabricated/empty inputs).
    if results.pairs.is_empty() {
        return Err(
            "results.json pairs is empty: no accepted per-pair records to score (forged/empty seal)"
                .to_string(),
        );
    }
    if results.pairs.len() != results.accepted_pair_count {
        return Err(format!(
            "results.json pairs.len ({}) != accepted_pair_count ({}): inconsistent seal (the \
             measure-job invariant is pairs.len == accepted_pair_count)",
            results.pairs.len(),
            results.accepted_pair_count
        ));
    }
    let base_mean = results.aggregate.baseline_serial_seconds_per_token_mean;
    if !base_mean.is_finite() || base_mean <= 0.0 {
        return Err(format!(
            "results.json aggregate.baseline_serial_seconds_per_token_mean ({base_mean}) is not a \
             finite value > 0: no valid baseline timing to anchor the ratio"
        ));
    }
    let cand_mean = results.aggregate.candidate_mtp_seconds_per_token_mean;
    if !cand_mean.is_finite() || cand_mean <= 0.0 {
        return Err(format!(
            "results.json aggregate.candidate_mtp_seconds_per_token_mean ({cand_mean}) is not a \
             finite value > 0: no valid candidate timing to anchor the ratio"
        ));
    }
    // finding R17 — the FULL live ranked pool-shape predicate set (Y:2595-2635). benchd's LOCAL
    // parity merge asserts every predicate the organizer's ranked seal (OPEN-2) asserts, so a local
    // run matches what the organizer would seal (or fails identically). Fail-closed: any breach ⇒
    // reject (no score, nonzero exit).
    let PoolExpectation {
        pool_size,
        min_per_prompt,
    } = *pool;
    // (3) run-total accepted-pair floor. SHAPE-DEPENDENT, and the reason it has to be: a
    // single-stream run measures each pool prompt on its own, so the run needs `pool_size *
    // min_per_prompt` pairs; a batched run measures the WHOLE pool in each pair, so one cohort
    // times `min_per_prompt` is the same requirement expressed over the unit that was actually
    // measured. Applying the single-stream product to a cohort run would demand 8x the pairs the
    // regime can produce and refuse every honest batched artifact.
    let min_pairs = pool.min_pairs_for(shape);
    if results.accepted_pair_count < min_pairs {
        return Err(match shape {
            ResultsShape::SingleStream => format!(
                "results.json accepted_pair_count ({}) < pool min_pairs ({min_pairs} = pool_size \
                 {pool_size} * min_per_prompt {min_per_prompt}): the pool did not reach its \
                 accepted floor",
                results.accepted_pair_count
            ),
            ResultsShape::Cohort { batch_size } => format!(
                "results.json accepted_pair_count ({}) < cohort min_pairs ({min_pairs} = one \
                 cohort of {batch_size} * min_per_unit {min_per_prompt}): the cohort did not reach \
                 its accepted floor",
                results.accepted_pair_count
            ),
        });
    }
    // (6) the aggregate per-pair diagnostics must be numbers (parse required them; enforce finite).
    let mtp_median = results.aggregate.mtp_decode_speedup_median;
    if !mtp_median.is_finite() {
        return Err(format!(
            "results.json aggregate.mtp_decode_speedup_median ({mtp_median}) is not a finite number"
        ));
    }
    let mtp_min = results.aggregate.mtp_decode_speedup_min;
    if !mtp_min.is_finite() {
        return Err(format!(
            "results.json aggregate.mtp_decode_speedup_min ({mtp_min}) is not a finite number"
        ));
    }
    // (10) the sealed published median must be a number (parse required it; enforce finite).
    let sealed_median = results.aggregate.raw_decode_speedup_median;
    if !sealed_median.is_finite() {
        return Err(format!(
            "results.json aggregate.raw_decode_speedup_median ({sealed_median}) is not a finite number"
        ));
    }
    // (7) pool CARDINALITY. The per-record half is shape-specific and is checked FIRST, exactly
    // where the single-stream `per_prompt.len()` check has always sat — the predicate SET is
    // unchanged and so is its precedence, so an artifact that was refused with a cardinality
    // message before F1 is refused with the same message after it.
    match shape {
        ResultsShape::SingleStream => {
            if results.per_prompt.len() != pool_size {
                return Err(format!(
                    "results.json per_prompt.len ({}) != pool_size ({pool_size}): the pool is the \
                     wrong size",
                    results.per_prompt.len()
                ));
            }
        }
        ResultsShape::Cohort { .. } => {
            // A batched run times the WHOLE pool in one shared window, so it seals exactly one
            // cohort whose members ARE the pool. Checked here so `per_cohort[0]` is safe below.
            if results.per_cohort.len() != 1 {
                return Err(format!(
                    "results.json per_cohort.len ({}) != 1: a batched run times the WHOLE pool in \
                     one shared window, so it seals exactly one cohort",
                    results.per_cohort.len()
                ));
            }
            if results.per_cohort[0].members.len() != pool_size {
                return Err(format!(
                    "results.json per_cohort[0].members.len ({}) != pool_size ({pool_size}): the \
                     cohort does not carry the whole pool",
                    results.per_cohort[0].members.len()
                ));
            }
        }
    }
    // `prompt_count == pool_size` holds on BOTH shapes — it counts distinct pool prompts TIMED,
    // not per-prompt records — so it stays shared, in its original position.
    if results.prompt_count != pool_size {
        return Err(format!(
            "results.json prompt_count ({}) != pool_size ({pool_size}): the sealed pool count \
             disagrees with the expected pool",
            results.prompt_count
        ));
    }
    // (8)/(9) the per-record predicates, per shape.
    match shape {
        ResultsShape::SingleStream => validate_single_stream_pool(results, pool)?,
        ResultsShape::Cohort { batch_size } => validate_cohort_pool(results, pool, batch_size)?,
    }
    Ok(())
}

/// The pool prompt-sha256 DISTINCTNESS predicate — finding R17 (9) — for BOTH shapes.
///
/// The two shapes seal the pool's prompt identities under different fields: a single-stream run in
/// `per_prompt[]`, a cohort run in `per_cohort[0].members[]` (see [`ResultsShape`]). The predicate
/// over them is the same one — `unique.len == pool_size`, a duplicated prompt means the pool was
/// not actually covered — so it lives here once and takes only what differs between the two: the
/// `sha256` field to read off each record, the sealed `field_path` the refusal names it under, and
/// the `noun` for the unit the duplicate was found in. Both refusals stay byte-identical to the
/// per-shape copies this replaces.
fn validate_pool_shas_distinct<T>(
    records: &[T],
    sha256: fn(&T) -> &str,
    pool_size: usize,
    field_path: &str,
    noun: &str,
) -> Result<(), String> {
    let mut distinct: Vec<&str> = records.iter().map(sha256).collect();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() != pool_size {
        return Err(format!(
            "results.json {field_path} prompt_sha256 are not all distinct (unique {} != pool_size \
             {pool_size}): a duplicated prompt in {noun}",
            distinct.len()
        ));
    }
    Ok(())
}

/// The SINGLE-STREAM pool-record predicates — finding R17 (8) and (9), unchanged and unweakened.
/// Lifted out of [`validate_results`] verbatim so the cohort arm sits beside it rather than inside
/// it: every predicate below still applies to every teacher-forced and free-run artifact exactly as
/// before F1.
fn validate_single_stream_pool(
    results: &ResultsView,
    pool: &PoolExpectation,
) -> Result<(), String> {
    let PoolExpectation {
        pool_size,
        min_per_prompt,
    } = *pool;
    // (8) every per-prompt record must satisfy the full predicate set.
    for (i, p) in results.per_prompt.iter().enumerate() {
        if crate::measure_job::validate_prompt_sha256(&p.prompt_sha256).is_err() {
            return Err(format!(
                "results.json per_prompt[{i}].prompt_sha256 ({:?}) is not 64 lowercase hex",
                p.prompt_sha256
            ));
        }
        if !p.parity_ok {
            return Err(format!(
                "results.json per_prompt[{i}].parity_ok != true: a prompt had a parity failure"
            ));
        }
        if p.accepted_pair_count < min_per_prompt {
            return Err(format!(
                "results.json per_prompt[{i}].accepted_pair_count ({}) < min_per_prompt ({min_per_prompt})",
                p.accepted_pair_count
            ));
        }
        if !(p.serial_seconds_per_token_mean.is_finite() && p.serial_seconds_per_token_mean > 0.0) {
            return Err(format!(
                "results.json per_prompt[{i}].serial_seconds_per_token_mean ({}) is not > 0",
                p.serial_seconds_per_token_mean
            ));
        }
        if !(p.mtp_seconds_per_token_mean.is_finite() && p.mtp_seconds_per_token_mean > 0.0) {
            return Err(format!(
                "results.json per_prompt[{i}].mtp_seconds_per_token_mean ({}) is not > 0",
                p.mtp_seconds_per_token_mean
            ));
        }
        if !(p.raw_ratio_of_means.is_finite() && p.raw_ratio_of_means > 0.0) {
            return Err(format!(
                "results.json per_prompt[{i}].raw_ratio_of_means ({}) is not > 0",
                p.raw_ratio_of_means
            ));
        }
        if !(p.noop_reference_decode_speedup.is_finite() && p.noop_reference_decode_speedup > 0.0) {
            return Err(format!(
                "results.json per_prompt[{i}].noop_reference_decode_speedup ({}) is not > 0 \
                 (exactly-one positive no-op reference required)",
                p.noop_reference_decode_speedup
            ));
        }
    }
    // (9) all per-prompt sha256 DISTINCT: `unique.len == pool_size`.
    validate_pool_shas_distinct(
        &results.per_prompt,
        |p| p.prompt_sha256.as_str(),
        pool_size,
        "per_prompt",
        "the pool",
    )
}

/// F1 CHANGE 2 — the COHORT pool-record predicates: the batched counterpart of
/// [`validate_single_stream_pool`], asserting the same things about the unit that was actually
/// measured.
///
/// The correspondence, predicate for predicate:
/// * pool CARDINALITY — single-stream checks `per_prompt.len() == pool_size`; a cohort's pool
///   prompts are its `members`, so this checks `per_cohort[0].members.len() == pool_size` (and that
///   there is exactly ONE cohort: a batched run times the whole pool in one window).
/// * prompt IDENTITY and DISTINCTNESS — the same 64-lowercase-hex grid and the same
///   all-distinct rule, applied to the members. Plus `slot_index == i`, because slot order is
///   load-bearing on this path (`attestation-slot-i-is-cohort-member-slot-i`).
/// * PARITY — `per_cohort[0].parity_ok`, as `per_prompt[i].parity_ok`.
/// * accepted-pair floor per unit — `>= min_per_prompt` on the cohort, and equal to the run total
///   (one cohort means every accepted pair is this cohort's).
/// * per-unit MEANS strictly positive — the cohort's serial/candidate seconds-per-token and its
///   ratio-of-means, as the per-prompt means are.
/// * the COMPOSITE — REQUIRED, coherent with its own sealed gains under the RULED exponents, and
///   floored. This is the cohort's published score, so it gets the strictest treatment here.
///
/// DELIBERATELY ABSENT: `noop_reference_decode_speedup`. The batched producer seals no no-op
/// reference (there is no per-prompt no-op for a shared window), so requiring one would refuse
/// every honest cohort artifact. Its purpose on the single-stream path — a per-prompt normaliser —
/// has no cohort analogue, and no cohort value is derived from one.
fn validate_cohort_pool(
    results: &ResultsView,
    pool: &PoolExpectation,
    batch_size: u32,
) -> Result<(), String> {
    let PoolExpectation {
        pool_size,
        min_per_prompt,
    } = *pool;

    // Cardinality (`per_cohort.len() == 1`, `members.len() == pool_size`) was established by
    // `validate_results`'s shared cardinality step before this is reached.
    let c = &results.per_cohort[0];

    // The width the artifact claims must equal the width its own series tag encodes, and the
    // top-level `scored_batch_size` (when sealed) must agree with both. B is IN the series tag, so
    // this is the only place the three can be reconciled.
    if c.batch_size != batch_size {
        return Err(format!(
            "results.json per_cohort[0].batch_size ({}) != the width its series tag encodes \
             ({batch_size}): the cohort record and the series disagree about how wide the run was",
            c.batch_size
        ));
    }
    if let Some(sealed_width) = results.scored_batch_size {
        if sealed_width != batch_size {
            return Err(format!(
                "results.json scored_batch_size ({sealed_width}) != the width its series tag \
                 encodes ({batch_size}): the sealed width and the series disagree"
            ));
        }
    }

    if !c.parity_ok {
        return Err(
            "results.json per_cohort[0].parity_ok != true: the cohort had a parity failure"
                .to_string(),
        );
    }
    if c.accepted_pair_count < min_per_prompt {
        return Err(format!(
            "results.json per_cohort[0].accepted_pair_count ({}) < min_per_unit ({min_per_prompt})",
            c.accepted_pair_count
        ));
    }
    if c.accepted_pair_count != results.accepted_pair_count {
        return Err(format!(
            "results.json per_cohort[0].accepted_pair_count ({}) != accepted_pair_count ({}): \
             there is ONE cohort, so every accepted pair is this cohort's",
            c.accepted_pair_count, results.accepted_pair_count
        ));
    }
    if crate::measure_job::validate_prompt_sha256(&c.cohort_sha256).is_err() {
        return Err(format!(
            "results.json per_cohort[0].cohort_sha256 ({:?}) is not 64 lowercase hex",
            c.cohort_sha256
        ));
    }
    for (label, v) in [
        (
            "serial_seconds_per_token_mean",
            c.serial_seconds_per_token_mean,
        ),
        (
            "candidate_seconds_per_token_mean",
            c.candidate_seconds_per_token_mean,
        ),
        ("raw_ratio_of_means", c.raw_ratio_of_means),
    ] {
        if !(v.is_finite() && v > 0.0) {
            return Err(format!(
                "results.json per_cohort[0].{label} ({v}) is not > 0"
            ));
        }
    }

    // Identity over the MEMBERS (the cohort's pool prompts).
    for (i, m) in c.members.iter().enumerate() {
        if crate::measure_job::validate_prompt_sha256(&m.prompt_sha256).is_err() {
            return Err(format!(
                "results.json per_cohort[0].members[{i}].prompt_sha256 ({:?}) is not 64 lowercase \
                 hex",
                m.prompt_sha256
            ));
        }
        if m.slot_index != i {
            return Err(format!(
                "results.json per_cohort[0].members[{i}].slot_index ({}) != {i}: slot order is \
                 load-bearing on the batched path (attestation slot i IS cohort member slot i)",
                m.slot_index
            ));
        }
    }
    validate_pool_shas_distinct(
        &c.members,
        |m| m.prompt_sha256.as_str(),
        pool_size,
        "per_cohort[0].members",
        "the cohort",
    )?;

    validate_cohort_composite(c)
}

/// F1 CHANGE 2 — the COMPOSITE is REQUIRED and must be COHERENT with the parts sealed beside it.
///
/// It is the published score of a cohort run, so it gets the same treatment finding R18 gives the
/// single-stream median: the overlay does not trust a sealed scoring number it cannot re-derive.
/// Here that means recomputing `prefill_gain^e_p * decode_gain^e_d` from the cohort's own sealed
/// gains under the RULED exponents and requiring agreement.
///
/// The exponents are cross-checked against [`crate::measure_job::PREFILL_GAIN_EXPONENT`] /
/// [`crate::measure_job::DECODE_GAIN_EXPONENT`] rather than taken from the artifact: they are ruled
/// constants, not per-run data, and a run that could name its own exponents could name the pair
/// that maximises its score.
/// The RULED composite exponents as [`bench_core::score::composite`] weights.
fn exponents_weights() -> bench_core::score::ScoringWeights {
    bench_core::score::ScoringWeights {
        decode: crate::measure_job::DECODE_GAIN_EXPONENT,
        prefill: crate::measure_job::PREFILL_GAIN_EXPONENT,
    }
}

fn validate_cohort_composite(c: &PerCohortView) -> Result<(), String> {
    let Some(composite) = c.composite else {
        return Err(format!(
            "results.json per_cohort[0].composite is absent (reason: {}): the shared-window \
             composite IS the published score of a batched run — refusing to publish a cohort run \
             that produced none",
            c.composite_absent_reason
                .as_deref()
                .unwrap_or("<none sealed>")
        ));
    };
    if c.composite_absent_reason.is_some() {
        return Err(format!(
            "results.json per_cohort[0] seals BOTH a composite and a composite_absent_reason \
             ({:?}): the producer's invariant is exactly one of the two, so this seal is \
             incoherent",
            c.composite_absent_reason
        ));
    }
    for (label, v) in [
        ("prefill_gain", composite.prefill_gain),
        ("decode_gain", composite.decode_gain),
        ("composite_score", composite.composite_score),
    ] {
        if !(v.is_finite() && v > 0.0) {
            return Err(format!(
                "results.json per_cohort[0].composite.{label} ({v}) is not a finite value > 0"
            ));
        }
    }

    // The RULED exponents — sealed beside the composite, and required to BE the ruled pair.
    let exponents = c.composite_scored_exponents.ok_or(
        "results.json per_cohort[0].composite_scored_exponents is absent: the exponents the \
         composite was raised to must be sealed beside it",
    )?;
    if exponents.prefill_gain_exponent != crate::measure_job::PREFILL_GAIN_EXPONENT
        || exponents.decode_gain_exponent != crate::measure_job::DECODE_GAIN_EXPONENT
    {
        return Err(format!(
            "results.json per_cohort[0].composite_scored_exponents ({}, {}) != the RULED pair ({}, \
             {}): the scoring exponents are ruled constants, not per-run data",
            exponents.prefill_gain_exponent,
            exponents.decode_gain_exponent,
            crate::measure_job::PREFILL_GAIN_EXPONENT,
            crate::measure_job::DECODE_GAIN_EXPONENT
        ));
    }

    // COHERENCE — recompute the composite from its own sealed gains (finding R18's posture, applied
    // to the cohort's published number).
    let recomputed = bench_core::score::composite(
        composite.prefill_gain,
        composite.decode_gain,
        exponents_weights(),
    );
    if !recomputed.is_finite() || recomputed <= 0.0 {
        return Err(format!(
            "overlay composite coherence: the composite recomputed from the sealed gains is not a \
             finite value > 0 ({recomputed}); refusing to score"
        ));
    }
    let rel = (recomputed - composite.composite_score).abs() / recomputed;
    if rel >= COMPOSITE_COHERENCE_REL_EPS {
        return Err(format!(
            "overlay composite coherence failed (tamper): the composite recomputed from the sealed \
             gains ({recomputed}) disagrees with the sealed composite_score ({}) by a relative \
             {rel} (>= {COMPOSITE_COHERENCE_REL_EPS:e}); the cohort's gains and its published \
             composite are inconsistent",
            composite.composite_score
        ));
    }

    // The sealed FLOOR must be the trusted constant, and the sealed VERDICT must agree with the
    // predicate recomputed against it — a wrapper does not get to lower its own floor or stamp a
    // pass onto a score that misses it. (Enforcement of the verdict happens in `merge_overlay`,
    // through the same `score_paired_decode_only` gate the single-stream path goes through.)
    if composite.composite_speedup_floor != QWEN_MTP_DECODE_SPEEDUP_FLOOR {
        return Err(format!(
            "results.json per_cohort[0].composite.composite_speedup_floor ({}) != the track floor \
             ({QWEN_MTP_DECODE_SPEEDUP_FLOOR}): the floor is a constant, not a per-run value",
            composite.composite_speedup_floor
        ));
    }
    let floor_met = clears_floor(composite.composite_score, QWEN_MTP_DECODE_SPEEDUP_FLOOR);
    if composite.composite_speedup_floor_met != floor_met {
        return Err(format!(
            "results.json per_cohort[0].composite.composite_speedup_floor_met ({}) disagrees with \
             the recomputed verdict ({floor_met}) for composite_score {} against floor \
             {QWEN_MTP_DECODE_SPEEDUP_FLOOR}",
            composite.composite_speedup_floor_met, composite.composite_score
        ));
    }
    Ok(())
}

/// Minor (overlay input identity cross-check) — the `gates-score.json` (seam 1) and `results.json`
/// (seam 2) MUST be from the SAME run. Cross-check `commit` and `weights_hash` agree (and are
/// non-empty) between the two inputs; a mismatch (or a missing/empty identity that defaults open)
/// REFUSES the merge, so gates from run A can never be overlaid onto timings from run B. This also
/// closes the fail-open path where a hand-forged minimal gates-score omits its identity fields.
pub fn validate_identity(gates: &ScorePayload, results: &ResultsView) -> Result<(), String> {
    if gates.metrics.commit.is_empty() || results.commit.is_empty() {
        return Err(format!(
            "overlay identity cross-check: empty commit (gates={:?}, results={:?}); the two seam \
             inputs must both carry the run commit",
            gates.metrics.commit, results.commit
        ));
    }
    if gates.metrics.commit != results.commit {
        return Err(format!(
            "overlay identity cross-check: commit mismatch (gates={:?} != results={:?}); the two \
             seam inputs are not from the same run",
            gates.metrics.commit, results.commit
        ));
    }
    if gates.metrics.weights_hash.is_empty() || results.weights_hash.is_empty() {
        return Err(format!(
            "overlay identity cross-check: empty weights_hash (gates={:?}, results={:?}); the two \
             seam inputs must both carry the weights identity",
            gates.metrics.weights_hash, results.weights_hash
        ));
    }
    if gates.metrics.weights_hash != results.weights_hash {
        return Err(format!(
            "overlay identity cross-check: weights_hash mismatch (gates={:?} != results={:?}); the \
             two seam inputs are not from the same run",
            gates.metrics.weights_hash, results.weights_hash
        ));
    }
    Ok(())
}

/// finding R20 — map a paired-decode bound failure to the EXACT live seam-3 `metrics.error` string
/// (Y:2667-2699). The floor prefix `performance floor failed (` is LOAD-BEARING (the redactor bills
/// `floor_failed` off it) — matched verbatim. The ceiling text is deliberately NON-attributable
/// (redactor `runtime_error`, not charged) — also matched verbatim. The per-pair / non-finite
/// classes reject before aggregation and reuse the `bench_core` message (naming the 8.0 bound / the
/// non-finite value).
fn overlay_error_message(failure: &PairedDecodeFailure) -> String {
    match failure {
        PairedDecodeFailure::Floor { median, floor } => format!(
            "performance floor failed (qwen-mtp raw serial-relative median {median} below floor {floor})"
        ),
        PairedDecodeFailure::Ceiling { median, ceiling } => format!(
            "qwen-mtp plausibility ceiling exceeded (median {median} above ceiling {ceiling})"
        ),
        // A per-pair plausibility breach / non-finite median rejects before aggregation; keep the
        // bench_core message (it already names the 8.0 bound / the non-finite value).
        other => other.message(),
    }
}

/// The outcome of a local overlay merge.
#[derive(Debug)]
pub struct OverlayOutcome {
    /// The sealed merged `score.json` bytes (coarsened diagnostics + sorted keys, with the paired
    /// discriminators injected). The integrity anchor is the sha256 of exactly these bytes.
    pub sealed_json: String,
    /// `passed = gates.passed AND median in [floor, ceiling]` (and the per-pair bound held).
    pub passed: bool,
    /// `Some(median)` on pass; `None` on any floor / ceiling / per-pair / non-finite failure.
    pub score: Option<f64>,
}

/// The PURE overlay merge (unit-tested without any IO): validate both inputs fail-closed,
/// aggregate the 3.8 median regime via the REUSED `bench_core::score::score_paired_decode_only`,
/// overlay the timing-derived fields onto the gates metrics, flip `partial_result → false`,
/// recompute ALL floor fields coherently (finding 11), inject the `scoring_mode` discriminator,
/// and seal via the reused [`ScorePayload::to_sealed_json`] coarsening/sort path.
// UNVERIFIED(B-4): the 3.8-median aggregation regime + numeric bounds are the live-as-executes-today
// record (OPEN-1 CLOSED); the live-vs-DRAFT ranked pipeline diff is B-4.
pub fn merge_overlay(
    gates: &ScorePayload,
    results: &ResultsView,
    expected_track: Option<&str>,
    expected_series: Option<&str>,
    pool: &PoolExpectation,
    scored: &bench_core::contract::Contract,
) -> Result<OverlayOutcome, String> {
    // **David ruling 2026-08-26** — resolve benchd's OWN harness identity HERE, at the seal, and
    // fail closed. This is the public entry EVERY production seal goes through (the single
    // `execute_overlay_timing` call site, local and parity alike), so the cross-leg equality is
    // asserted once, at the funnel, rather than being a discipline each call site must remember.
    // `merge_overlay_against_harness` below is the pure funnel; the ONLY thing this wrapper adds is
    // the fail-closed resolution, which is IO and therefore cannot live in the pure half.
    let seal_leg = resolve_seal_harness_identity()?;
    merge_overlay_against_harness(
        gates,
        results,
        expected_track,
        expected_series,
        pool,
        scored,
        &seal_leg,
    )
}

/// [`merge_overlay`] with the seal-time harness leg SUPPLIED rather than resolved — the pure funnel,
/// unit-testable without a harness workspace at the process CWD (which a parallel test suite cannot
/// establish). Production reaches this only through [`merge_overlay`], which resolves `seal_leg`
/// from the CWD fail-closed; nothing else may call it with a hand-picked identity.
fn merge_overlay_against_harness(
    gates: &ScorePayload,
    results: &ResultsView,
    expected_track: Option<&str>,
    expected_series: Option<&str>,
    pool: &PoolExpectation,
    scored: &bench_core::contract::Contract,
    seal_leg: &HarnessIdentity,
) -> Result<OverlayOutcome, String> {
    validate_gates(gates)?;
    // **David ruling 2026-08-26** — the CROSS-LEG equality, immediately AFTER the single-leg
    // well-formedness gate above and never in place of it: an empty/malformed gates identity is
    // still refused by the earlier, cheaper `validate_gates` message, and only a WELL-FORMED gates
    // leg ever reaches this comparison. See `validate_harness_identity_cross_leg` for the
    // between-phase TOCTOU window this closes and the driver-pinned CWD contract that makes the two
    // legs comparable. SHAPE-INDEPENDENT by construction: it reads the gates leg, which both
    // per_prompt and per_cohort runs carry identically.
    validate_harness_identity_cross_leg(gates, seal_leg)?;
    validate_results(results, expected_track, expected_series, pool, scored)?;
    validate_identity(gates, results)?;

    // finding R18 — sealed-median AGREEMENT (the seam-3 wrapper-tamper detector, Y:2646-2666). The
    // overlay does NOT trust the sealed `aggregate.raw_decode_speedup_median` blindly: it RECOMPUTES
    // the published median FROM the per-prompt means (median over per_prompt of
    // `serial_seconds_per_token_mean / mtp_seconds_per_token_mean`, even-n = mean of the two central
    // order statistics via the SHARED bench_core rule) and requires agreement with the sealed value
    // within 1e-7. A disagreement means the wrapper's per-prompt means and its published median were
    // authored inconsistently (tamper) ⇒ REJECT. The published score the overlay uses below is the
    // RECOMPUTED median, never the sealed field. (This is the per-PROMPT even-n median; do NOT
    // conflate with the per-PAIR lower-median diagnostic.)
    // F1 CHANGE 2 — the agreement check is SHAPE-AWARE, because the two shapes seal different
    // sample sets. Neither shape loses it: on both, the overlay recomputes the sealed
    // `aggregate.raw_decode_speedup_median` from the artifact's own per-unit numbers and refuses a
    // file whose parts and published aggregate were authored inconsistently.
    let shape = results_shape(results)?;
    let recomputed_median = match shape {
        // SINGLE-STREAM: the median over per_prompt of `serial / mtp`, from the per-prompt MEANS.
        ResultsShape::SingleStream => {
            let per_prompt_ratios: Vec<f64> = results
                .per_prompt
                .iter()
                .map(|p| {
                    speedup(
                        p.serial_seconds_per_token_mean,
                        p.mtp_seconds_per_token_mean,
                    )
                })
                .collect();
            even_n_median(&per_prompt_ratios)
        }
        // COHORT: `per_prompt` is empty by construction, and the producer's sealed median is the
        // even-n median over the accepted PAIRS' cohort ratios. Recomputing from `pairs[].raw_ratio`
        // keeps the identical tamper detector on data the batched shape actually carries — the
        // check is relocated to the right samples, not dropped.
        ResultsShape::Cohort { .. } => {
            let per_pair: Vec<f64> = results.pairs.iter().map(|p| p.raw_ratio).collect();
            even_n_median(&per_pair)
        }
    };
    let sealed_median = results.aggregate.raw_decode_speedup_median;
    if !recomputed_median.is_finite() {
        return Err(format!(
            "overlay sealed-median agreement: the median recomputed from the run's own per-unit \
             means is not finite ({recomputed_median}); refusing to score"
        ));
    }
    if (recomputed_median - sealed_median).abs() >= SEALED_MEDIAN_AGREEMENT_EPS {
        return Err(format!(
            "overlay sealed-median agreement failed (tamper): recomputed median {recomputed_median} \
             disagrees with the sealed aggregate.raw_decode_speedup_median {sealed_median} by \
             {} (>= {SEALED_MEDIAN_AGREEMENT_EPS:e}); the run's per-unit means and published \
             median are inconsistent",
            (recomputed_median - sealed_median).abs()
        ));
    }

    // Aggregate. The GATE is the same on both shapes and is the SAME reused `bench_core` function:
    // the per-pair plausibility bound (8.0) on EACH pair's raw ratio, then floor 0.90 / ceiling 5.0
    // on the published figure. What differs is only WHICH figure is published:
    //
    // * SINGLE-STREAM — the even-n median of the per-prompt recomputed ratios (3.8 MEDIAN regime,
    //   OPEN-1), unchanged.
    // * COHORT — the sealed SHARED-WINDOW COMPOSITE. A batched run has one shared timed window over
    //   the whole pool, so there are no per-prompt ratios to take a median of; the composite IS the
    //   score (`validate_cohort_composite` has already required it, cross-checked its exponents
    //   against the ruled constants, and re-derived it from its own sealed gains). Passing it as the
    //   single sample makes `score_paired_decode_only` apply the identical floor/ceiling treatment
    //   to it — the median of one value is that value — with no second scoring formula anywhere.
    let per_pair_ratios: Vec<f64> = results.pairs.iter().map(|p| p.raw_ratio).collect();
    let published_samples: Vec<f64> = match shape {
        ResultsShape::SingleStream => results
            .per_prompt
            .iter()
            .map(|p| {
                speedup(
                    p.serial_seconds_per_token_mean,
                    p.mtp_seconds_per_token_mean,
                )
            })
            .collect(),
        ResultsShape::Cohort { .. } => vec![
            results.per_cohort[0]
                .composite
                .expect("validate_cohort_composite requires the composite")
                .composite_score,
        ],
    };
    let paired = score_paired_decode_only(&per_pair_ratios, &published_samples);
    let median = paired.raw_median;

    // Overlay the timing-derived fields onto the sealed gates metrics (start from the gates
    // metrics so the correctness/gate surface carries through unchanged).
    let mut metrics = gates.metrics.clone();
    // Candidate/baseline seconds-per-token MEANS from results.json.aggregate.
    metrics.decode_seconds_per_token = results.aggregate.candidate_mtp_seconds_per_token_mean;
    metrics.baseline_decode_seconds_per_token =
        results.aggregate.baseline_serial_seconds_per_token_mean;
    // The median is the ranking decode_speedup (never coarsened — ScoreMetrics leaves ranking
    // fields precise in with_coarsened_public_diagnostics).
    metrics.decode_speedup = median;
    metrics.decode_speedup_floor = QWEN_MTP_DECODE_SPEEDUP_FLOOR;

    // finding 11 — coherent floor fields for the DECODE-ONLY paired track. Recompute the decode
    // floor against the 0.90 paired floor; NEUTRALIZE the generic prefill floor (this track never
    // scores or floors prefill) so `passed_prefill_speedup_floor` can never contradict `passed`.
    metrics.passed_decode_speedup_floor = clears_floor(median, QWEN_MTP_DECODE_SPEEDUP_FLOOR);
    metrics.prefill_speedup_floor = 0.0; // not-applicable on the paired decode-only track
    metrics.passed_prefill_speedup_floor = true; // vacuously satisfied (prefill is a diagnostic only)

    // Diagnostics (wall/ram) = max across sources where applicable. The measure-job results.json
    // aggregate carries no wall/ram, so the gates diagnostics stand (a max with an absent source).
    metrics.benchmark_wall_seconds = metrics.benchmark_wall_seconds.max(0.0);
    metrics.peak_ram_gb = metrics.peak_ram_gb.max(0.0);

    // partial_result → false: this IS the timed overlay (GEMMA-OVL `@67699fc4:159`).
    metrics.partial_result = false;

    // Verdict: pass iff the gate passed AND the paired gate (median in [floor, ceiling], per-pair
    // bound held). finding R20 — on a bound failure the overlay AUTHORS THE REFUSAL INTO the merged
    // score.json (it does NOT fall back to the gates-score): score is 0 (NOT null) with
    // passed=false, and metrics.error carries the EXACT live floor/ceiling string. The Rust
    // `OverlayOutcome.score` stays `None` (no passing score) so the CLI exits nonzero (1).
    let passed = gates.passed && paired.passed;
    let (payload_score, outcome_score) = if passed {
        (paired.score, paired.score)
    } else {
        if let Some(f) = &paired.failure {
            metrics.error = overlay_error_message(f);
        }
        (Some(0.0), None)
    };

    let passed_ceiling = median.is_finite() && median <= QWEN_MTP_DECODE_SPEEDUP_CEILING;

    // Build via the reused ScorePayload publication path, then inject the paired discriminators.
    // The number seal runs once, after all insertions.
    // as top-level keys (ScoreMetrics is the Swift-parity schema and stays UNTOUCHED — the generic
    // score's roster/parity is unaffected; the discriminators live alongside score/passed/metrics).
    let payload = ScorePayload {
        score: payload_score,
        passed,
        metrics,
    };
    let mut value = payload
        .to_public_unsealed_value()
        .map_err(|e| format!("overlay score serialization failed: {e}"))?;
    if let Value::Object(map) = &mut value {
        map.insert("scoring_mode".to_string(), json!(SCORING_MODE));
        // F1 CHANGE 2 — the aggregation label FOLLOWS THE SHAPE. A cohort score is not a median of
        // per-prompt ratios and must not be labelled as one: a consumer that read `AGGREGATION` off
        // a batched score would believe it was looking at an aggregate the artifact does not
        // contain.
        map.insert(
            "aggregation".to_string(),
            json!(match shape {
                ResultsShape::SingleStream => AGGREGATION,
                ResultsShape::Cohort { .. } => AGGREGATION_COHORT_COMPOSITE,
            }),
        );
        // The ceiling has no ScoreMetrics home (that struct is decode-floor only); carry it and its
        // pass as paired ranking fields at top level (raw, never coarsened).
        map.insert(
            "decode_speedup_ceiling".to_string(),
            json!(QWEN_MTP_DECODE_SPEEDUP_CEILING),
        );
        map.insert(
            "passed_decode_speedup_ceiling".to_string(),
            json!(passed_ceiling),
        );
        // W3 (§5 labelling) — CARRY the series identity into the merged score. A score.json that
        // drops it would let a downstream leaderboard/regression gate compare this number to one of
        // the other series, which is exactly what the comparability rule forbids. The per-leg tags
        // ride along so the mixed shape stays visible after the merge, not just before it.
        map.insert("timed_mode".to_string(), json!(results.timed_mode));
        map.insert(
            "timed_series".to_string(),
            json!({
                "serial_leg_timed_mode": results.timed_series.serial_leg_timed_mode,
                "candidate_leg_timed_mode": results.timed_series.candidate_leg_timed_mode,
                "homogeneous": results.timed_series.homogeneous,
                "legs_comparable": results.timed_series.legs_comparable,
            }),
        );
        // F1 CHANGE 2 — a cohort score carries WHAT WAS SCORED. `decode_speedup` above is the
        // composite; without these fields a reader could not tell which cohort produced it, at what
        // width, or from which two gains — and the composite is not re-derivable from the merged
        // score alone. Same posture as the series identity directly above: carry the discriminators
        // the number is only meaningful with.
        if let ResultsShape::Cohort { batch_size } = shape {
            let c = &results.per_cohort[0];
            let composite = c
                .composite
                .expect("validate_cohort_composite requires the composite");
            map.insert("scored_batch_size".to_string(), json!(batch_size));
            map.insert(
                "cohort".to_string(),
                json!({
                    "cohort_sha256": c.cohort_sha256,
                    "member_count": c.members.len(),
                    "prefill_gain": composite.prefill_gain,
                    "decode_gain": composite.decode_gain,
                    "composite_score": composite.composite_score,
                    "composite_speedup_floor": composite.composite_speedup_floor,
                    "composite_speedup_floor_met": composite.composite_speedup_floor_met,
                    "prefill_gain_exponent": crate::measure_job::PREFILL_GAIN_EXPONENT,
                    "decode_gain_exponent": crate::measure_job::DECODE_GAIN_EXPONENT,
                }),
            );
        }
    }
    let mut private_value = serde_json::to_value(&payload)
        .map_err(|e| format!("overlay private score serialization failed: {e}"))?;
    if let (Value::Object(private), Value::Object(public)) = (&mut private_value, &value) {
        for (key, value) in public {
            if !["metrics", "passed", "score"].contains(&key.as_str()) {
                private.insert(key.clone(), value.clone());
            }
        }
    }
    crate::score::write_private_audit(&private_value)
        .map_err(|e| format!("overlay private audit write failed: {e}"))?;
    crate::score::seal_public_value(&mut value);
    // Sorted + pretty (serde_json Value is a BTreeMap → sorted keys), matching the sealed shape.
    let sealed_json = serde_json::to_string_pretty(&value)
        .map_err(|e| format!("overlay serialize failed: {e}"))?;

    Ok(OverlayOutcome {
        sealed_json,
        passed,
        score: outcome_score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::{sha256_hex, ScoreMetrics};

    /// A passing gates-score.json (seam 1 out): partial_result=true, error empty,
    /// passed_correctness=true, and the generic prefill floor set (which finding 11 must
    /// neutralize on the merge). Timing placeholders are zero (SKIP_TIMED gates).
    fn passing_gates() -> ScorePayload {
        let mut m = ScoreMetrics {
            partial_result: true,
            passed_correctness: true,
            // The generic prefill floor + a FALSE passed flag: the incoherence finding 11 fixes.
            prefill_speedup_floor: 0.95,
            passed_prefill_speedup_floor: false,
            decode_speedup_floor: 0.95,
            passed_decode_speedup_floor: false,
            case_count: 12,
            checked_steps: 34,
            benchmark_wall_seconds: 1.5,
            peak_ram_gb: 20.25,
            commit: "deadbeef".to_string(),
            weights_hash: "w-hash".to_string(),
            // B2 — the official baseline pair a real gates run RESOLVES before it seals
            // (`official::official_gates_only`). The helper used to leave it at the serde default
            // `0.0`, which is the very state `validate_gates` now refuses.
            baseline_decode_seconds_per_token: 0.01385621216015625,
            baseline_prefill_seconds_per_token: 0.00036751938916015625,
            ..ScoreMetrics::default()
        };
        m.error = String::new();
        // A well-formed harness identity (64 lowercase-hex SHA256) — the gates/seam-1 harness-hash
        // gate PASSES on this baseline; the empty/malformed refusal tests mutate it below.
        //
        // David ruling 2026-08-26 — this is now F1's `HarnessIdentity::TEST_HASH` rather than the
        // arbitrary `"a".repeat(64)` it was, so that the fixture carries a MATCHING cross-leg
        // identity: `merge_matched` supplies `HarnessIdentity::test_default()` as the seal leg, and
        // the two are the same digest. The change is a FIXTURE update, not a gate relaxation — the
        // single-leg well-formedness predicate this value has always had to satisfy is unchanged
        // (TEST_HASH is 64 lowercase hex), and the mismatch/empty/malformed twins below still drive
        // their refusals.
        m.harness_hash = HarnessIdentity::TEST_HASH.to_string();
        ScorePayload {
            score: None,
            passed: true,
            metrics: m,
        }
    }

    /// The overlay tests' entry into the merge funnel: [`merge_overlay_against_harness`] with the
    /// seal leg fixed to F1's [`HarnessIdentity::test_default`] — the SAME digest `passing_gates()`
    /// seals, so the David-ruled cross-leg equality is SATISFIED (an honest run) rather than
    /// bypassed. Every pre-existing overlay test routes through here.
    ///
    /// The production entry [`merge_overlay`] is deliberately NOT used: it resolves the seal leg
    /// from the process CWD, and a unit test has no nine-root harness workspace there (nor may a
    /// parallel test suite `chdir` to one). The refusal twins below call
    /// [`merge_overlay_against_harness`] directly with a DIFFERENT leg, which is the only way to
    /// reach the mismatch branch — so the gate is exercised in both directions from this module,
    /// and `merge_overlay`'s own fail-closed resolution is covered by
    /// `seal_time_resolution_failure_refuses_and_names_the_missing_root` plus the
    /// `scripts/test-paired-offline.sh` seam-3 end-to-end.
    fn merge_matched(
        gates: &ScorePayload,
        results: &ResultsView,
        expected_track: Option<&str>,
        expected_series: Option<&str>,
        pool: &PoolExpectation,
    ) -> Result<OverlayOutcome, String> {
        merge_overlay_against_harness(
            gates,
            results,
            expected_track,
            expected_series,
            pool,
            &scored_fixture(),
            HarnessIdentity::test_default(),
        )
    }

    /// The SCORED CONTRACT the merge's two per-track fences read: this tree's own track, its
    /// regime, and the two flow flags. STEP 3 (David 2026-09-15) moved all three from per-track
    /// tables onto the fixture, so the overlay tests hand it one.
    fn scored_fixture() -> bench_core::contract::Contract {
        bench_core::contract::Contract {
            track_id: Some(bench_core::constants::TRACK_ID.to_string()),
            scored_batch_size: Some(1),
            prefill_gain_exponent: Some(0.0),
            decode_gain_exponent: Some(1.0),
            scores_against_live_control_leg: Some(false),
            paired_flow_retired: Some(false),
            ..bench_core::contract::Contract::NONE_DECLARED
        }
    }

    /// RED-first — the pre-existing gap #223's reviewers named. [`ScoreMetrics`] is
    /// `#[serde(default)]`, so a `gates-score.json` that simply OMITS
    /// `baseline_{decode,prefill}_seconds_per_token` deserializes them as `0.0`, and
    /// `validate_gates` never looked at the pair. The overlay then merged a ranked score onto a
    /// gate whose scoring denominator was silently absent.
    ///
    /// The pair must be PRESENT, FINITE and POSITIVE, or the merge refuses BY NAME.
    #[test]
    fn gates_with_a_missing_or_degenerate_baseline_pair_refuses_by_name() {
        // The defect itself: the keys are absent from the JSON and come back as 0.0.
        let mut value = serde_json::to_value(passing_gates()).unwrap();
        let metrics = value.get_mut("metrics").unwrap().as_object_mut().unwrap();
        metrics.remove("baseline_decode_seconds_per_token");
        metrics.remove("baseline_prefill_seconds_per_token");
        let omitted: ScorePayload = serde_json::from_value(value).unwrap();
        assert_eq!(omitted.metrics.baseline_decode_seconds_per_token, 0.0);
        assert_eq!(omitted.metrics.baseline_prefill_seconds_per_token, 0.0);

        let err = validate_gates(&omitted).unwrap_err();
        assert!(
            err.contains("baseline_decode_seconds_per_token"),
            "the refusal must NAME the field: {err}"
        );
        // The message is a multi-line literal, so it needs its `\` continuations. Without them the
        // operator reads runs of padding spaces. Asserted as ONE contiguous phrase, because a
        // fragment check (`contains("scoring denominator")`) still passes on the broken rendering.
        assert!(
            err.contains(
                "the official baseline pair is the scoring denominator and must be present, \
                 finite and positive. ScoreMetrics is serde(default), so an OMITTED key reads \
                 back as 0.0 — an absent denominator that looks like a declared one; refusing to \
                 overlay a ranked score onto it"
            ),
            "exact refusal phrasing: {err}"
        );
        assert!(
            !err.contains("  "),
            "the message must carry no padding-space runs: {err}"
        );

        // Every degenerate spelling of each half, one at a time, so neither is checked by accident
        // through the other.
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut decode = passing_gates();
            decode.metrics.baseline_decode_seconds_per_token = bad;
            let err = validate_gates(&decode).unwrap_err();
            assert!(
                err.contains("baseline_decode_seconds_per_token"),
                "decode {bad}: {err}"
            );

            let mut prefill = passing_gates();
            prefill.metrics.baseline_prefill_seconds_per_token = bad;
            let err = validate_gates(&prefill).unwrap_err();
            assert!(
                err.contains("baseline_prefill_seconds_per_token"),
                "prefill {bad}: {err}"
            );
        }

        // POSITIVE CONTROL — a gates score carrying the pair a real gates run seals validates.
        assert!(validate_gates(&passing_gates()).is_ok());
    }

    /// A pool expectation of `n` prompts, one pair per prompt (the ranked k=1 default).
    fn pooln(n: usize) -> PoolExpectation {
        PoolExpectation {
            pool_size: n,
            min_per_prompt: 1,
        }
    }

    /// The single-prompt pool expectation the single-prompt `results_with` fixtures validate against.
    fn pool1() -> PoolExpectation {
        pooln(1)
    }

    /// A distinct 64-lowercase-hex prompt sha for pool index `i`.
    fn sha(i: usize) -> String {
        format!("{:064x}", i + 1)
    }

    /// One valid per-prompt record whose raw ratio-of-means is `ratio` (and whose serial/mtp means
    /// recompute to the SAME `ratio` for the R18 sealed-median agreement), with a distinct sha,
    /// `parity_ok`, `accepted` pairs, and a positive no-op reference.
    fn per_prompt(i: usize, ratio: f64, accepted: usize) -> PerPromptView {
        PerPromptView {
            prompt_sha256: sha(i),
            parity_ok: true,
            accepted_pair_count: accepted,
            // serial / mtp == ratio EXACTLY (mtp = 1.0 ⇒ IEEE754 division by 1.0 is exact), so the
            // R18 recompute-from-means equals `ratio` (and the sealed median) bit-for-bit.
            serial_seconds_per_token_mean: ratio,
            mtp_seconds_per_token_mean: 1.0,
            raw_ratio_of_means: ratio,
            noop_reference_decode_speedup: 1.0,
        }
    }

    /// W3 — the all-teacher-forced SERIES DESCRIPTOR the legacy (Model-2) fixtures carry: both legs
    /// v1 teacher-forced, homogeneous, and §5-comparable with each other.
    fn tf_series() -> TimedSeriesView {
        TimedSeriesView {
            serial_leg_timed_mode: bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1.to_string(),
            candidate_leg_timed_mode: bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1
                .to_string(),
            homogeneous: true,
            legs_comparable: true,
        }
    }

    /// Fable ruling — the all-free-run SERIES DESCRIPTOR a scored v1.1 run now seals: BOTH legs
    /// `free_run_v1_1` (the serial control free-runs at depth 0), homogeneous and §5-comparable.
    fn free_run_series() -> TimedSeriesView {
        TimedSeriesView {
            serial_leg_timed_mode: bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1.to_string(),
            candidate_leg_timed_mode: bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1.to_string(),
            homogeneous: true,
            legs_comparable: true,
        }
    }

    /// The CROSSED descriptor: teacher-forced serial control + free-run candidate, sealed honestly
    /// as not-homogeneous and not-comparable. measure-job can no longer produce this; the fence must
    /// still refuse it (defense in depth), so it stays as a fixture.
    fn mixed_series() -> TimedSeriesView {
        TimedSeriesView {
            serial_leg_timed_mode: bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1.to_string(),
            candidate_leg_timed_mode: bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1.to_string(),
            homogeneous: false,
            legs_comparable: false,
        }
    }

    /// A pair record belonging to the all-teacher-forced series.
    fn tf_pair(raw_ratio: f64) -> PairView {
        PairView {
            raw_ratio,
            serial_timed_mode: bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1.to_string(),
            candidate_timed_mode: bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1.to_string(),
        }
    }

    /// A pair record belonging to the all-free-run series (both legs v1.1).
    fn free_run_pair(raw_ratio: f64) -> PairView {
        PairView {
            raw_ratio,
            serial_timed_mode: bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1.to_string(),
            candidate_timed_mode: bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1.to_string(),
        }
    }

    /// A pair record from the CROSSED shape (TF serial control, free-run candidate).
    fn mixed_pair(raw_ratio: f64) -> PairView {
        PairView {
            raw_ratio,
            serial_timed_mode: bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1.to_string(),
            candidate_timed_mode: bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1.to_string(),
        }
    }

    /// A results.json view with a single prompt whose raw ratio-of-means is `ratio`, backed by
    /// `pairs` per-pair raw ratios (default: one pair equal to `ratio`). The single-prompt sealed
    /// median equals `ratio`.
    fn results_with(ratio: f64, pairs: Vec<f64>) -> ResultsView {
        let acc = pairs.len().max(1);
        let pair_min = pairs.iter().copied().fold(f64::INFINITY, f64::min);
        ResultsView {
            track_id: bench_core::constants::TRACK_ID.to_string(),
            timed_mode: bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1.to_string(),
            timed_series: tf_series(),
            parity_all_ok: true,
            accepted_pair_count: acc,
            candidate_accepted: true,
            min_pairs: 1,
            per_prompt: vec![per_prompt(0, ratio, acc)],
            prompt_count: 1,
            pairs: pairs.into_iter().map(tf_pair).collect(),
            aggregate: AggregateView {
                baseline_serial_seconds_per_token_mean: 0.038,
                candidate_mtp_seconds_per_token_mean: 0.038 / ratio,
                mtp_decode_speedup_median: ratio,
                mtp_decode_speedup_min: if pair_min.is_finite() {
                    pair_min
                } else {
                    ratio
                },
                // Single-prompt pool: the sealed even-n median IS that one ratio.
                raw_decode_speedup_median: ratio,
            },
            commit: "deadbeef".to_string(),
            weights_hash: "w-hash".to_string(),
            // Single-stream fixtures seal no cohort records — the shape cross-check requires that.
            per_cohort: Vec::new(),
            scored_batch_size: None,
        }
    }

    /// A multi-prompt pool (one pair per prompt): per-prompt raw ratio-of-means = `ratios[i]`, with
    /// the sealed `raw_decode_speedup_median` set to the even-n median of `ratios` (coherent seal).
    fn pool_results(ratios: &[f64]) -> ResultsView {
        let n = ratios.len();
        let per_prompt: Vec<PerPromptView> = ratios
            .iter()
            .enumerate()
            .map(|(i, &r)| per_prompt(i, r, 1))
            .collect();
        let pairs: Vec<PairView> = ratios.iter().map(|&r| tf_pair(r)).collect();
        let median = even_n_median(ratios);
        let mut sorted = ratios.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ResultsView {
            track_id: bench_core::constants::TRACK_ID.to_string(),
            timed_mode: bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1.to_string(),
            timed_series: tf_series(),
            parity_all_ok: true,
            accepted_pair_count: n,
            candidate_accepted: true,
            min_pairs: 1,
            per_prompt,
            prompt_count: n,
            pairs,
            aggregate: AggregateView {
                baseline_serial_seconds_per_token_mean: 0.057,
                candidate_mtp_seconds_per_token_mean: 0.038,
                mtp_decode_speedup_median: median,
                mtp_decode_speedup_min: *sorted.first().unwrap(),
                raw_decode_speedup_median: median,
            },
            commit: "deadbeef".to_string(),
            weights_hash: "w-hash".to_string(),
            // Single-stream fixtures seal no cohort records — the shape cross-check requires that.
            per_cohort: Vec::new(),
            scored_batch_size: None,
        }
    }

    #[test]
    fn identity_merge_passes_and_clears_partial_result() {
        let out = merge_matched(
            &passing_gates(),
            &results_with(1.0, vec![1.0]),
            None,
            None,
            &pool1(),
        )
        .unwrap();
        assert!(out.passed);
        assert_eq!(out.score, Some(1.0));
        let v: Value = serde_json::from_str(&out.sealed_json).unwrap();
        assert_eq!(v["passed"], json!(true));
        assert!((v["score"].as_f64().unwrap() - 1.0).abs() < 1e-12);
        // partial_result flipped to false on the merge.
        assert_eq!(v["metrics"]["partial_result"], json!(false));
        // The scoring_mode discriminator is present.
        assert_eq!(v["scoring_mode"], json!(SCORING_MODE));
        assert_eq!(v["aggregation"], json!(AGGREGATION));
        // Decode floor/ceiling are the paired 0.90/5.0 and both pass.
        assert_eq!(v["metrics"]["decode_speedup_floor"], json!(0.90));
        assert_eq!(v["decode_speedup_ceiling"], json!(5.0));
        assert_eq!(v["metrics"]["passed_decode_speedup_floor"], json!(true));
        assert_eq!(v["passed_decode_speedup_ceiling"], json!(true));
        // Timing overlaid from results.aggregate.
        assert!(
            (v["metrics"]["baseline_decode_seconds_per_token"]
                .as_f64()
                .unwrap()
                - 0.038)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn asymmetric_faster_candidate_scores_above_one_not_inverted() {
        // An ASYMMETRIC case the identity 1.0 test cannot catch: a faster candidate whose
        // per-prompt raw ratio-of-means are [1.5, 1.5, 1.5]. The merged score / decode_speedup
        // must be ≈ 1.5 (> 1), proving the serial/mtp orientation is NOT inverted (an inversion
        // would surface ≈ 1/1.5 ≈ 0.667 and fail the 0.90 floor).
        let results = pool_results(&[1.5, 1.5, 1.5]);
        let out = merge_matched(&passing_gates(), &results, None, None, &pooln(3)).unwrap();
        assert!(
            out.passed,
            "a 1.5 median clears the 0.90 floor / 5.0 ceiling"
        );
        let score = out.score.expect("faster candidate has a score");
        assert!(
            (score - 1.5).abs() < 1e-9,
            "score {score} must be ~1.5, not inverted"
        );
        assert!(score > 1.0, "a faster candidate scores > 1 (not inverted)");
        let v: Value = serde_json::from_str(&out.sealed_json).unwrap();
        assert!((v["metrics"]["decode_speedup"].as_f64().unwrap() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn integrity_anchor_is_sha_of_merged_bytes() {
        let out = merge_matched(
            &passing_gates(),
            &results_with(1.0, vec![1.0]),
            None,
            None,
            &pool1(),
        )
        .unwrap();
        // The integrity re-anchor value is exactly sha256(merged bytes).
        let anchor = sha256_hex(out.sealed_json.as_bytes());
        assert_eq!(anchor.len(), 64);
        // Re-sealing the same inputs is byte-stable (so the anchor is reproducible).
        let out2 = merge_matched(
            &passing_gates(),
            &results_with(1.0, vec![1.0]),
            None,
            None,
            &pool1(),
        )
        .unwrap();
        assert_eq!(out.sealed_json, out2.sealed_json);
        assert_eq!(anchor, sha256_hex(out2.sealed_json.as_bytes()));
    }

    #[test]
    fn floor_fail_authors_score_zero_and_load_bearing_floor_prefix() {
        // finding R20 — median 0.85 < 0.90 floor ⇒ score.json {score:0, passed:false} with the
        // LOAD-BEARING `performance floor failed (` prefix and passed_decode_speedup_floor:false.
        let out = merge_matched(
            &passing_gates(),
            &results_with(0.85, vec![0.85]),
            None,
            None,
            &pool1(),
        )
        .unwrap();
        assert!(!out.passed);
        assert!(
            out.score.is_none(),
            "OverlayOutcome carries no passing score"
        );
        let v: Value = serde_json::from_str(&out.sealed_json).unwrap();
        // R20: the merged score.json score is 0 (NOT null), passed false.
        assert_eq!(v["score"], json!(0.0), "R20: score is 0, not null");
        assert_eq!(v["passed"], json!(false));
        let err = v["metrics"]["error"].as_str().unwrap();
        assert!(
            err.starts_with("performance floor failed ("),
            "floor error must carry the load-bearing `performance floor failed (` prefix: {err}"
        );
        assert!(
            err.contains("qwen-mtp raw serial-relative median"),
            "exact floor text: {err}"
        );
        assert!(err.contains("below floor"), "exact floor text: {err}");
        // Coherence: the decode floor did NOT pass; scoring_mode is the native discriminator.
        assert_eq!(v["metrics"]["passed_decode_speedup_floor"], json!(false));
        assert_eq!(
            v["scoring_mode"],
            json!("qwen-native-mtp-paired-decode-only")
        );
    }

    #[test]
    fn ceiling_fail_authors_score_zero_and_nonattributable_ceiling_text() {
        // finding R20 — median 6.0 > 5.0 ceiling ⇒ score.json {score:0, passed:false} with the
        // NON-attributable `qwen-mtp plausibility ceiling exceeded` text (per-pair bound is 8.0, so
        // a single 6.0 pair clears the per-pair check and reaches the median ceiling).
        let out = merge_matched(
            &passing_gates(),
            &results_with(6.0, vec![6.0]),
            None,
            None,
            &pool1(),
        )
        .unwrap();
        assert!(!out.passed);
        assert!(out.score.is_none());
        let v: Value = serde_json::from_str(&out.sealed_json).unwrap();
        assert_eq!(v["score"], json!(0.0), "R20: score is 0, not null");
        assert_eq!(v["passed"], json!(false));
        let err = v["metrics"]["error"].as_str().unwrap();
        assert!(
            err.starts_with("qwen-mtp plausibility ceiling exceeded"),
            "ceiling error must carry the exact non-attributable text: {err}"
        );
        assert!(err.contains("above ceiling"), "exact ceiling text: {err}");
        assert!(
            !err.chars().any(|c| c.is_ascii_digit()),
            "ceiling error must omit exact numbers: {err}"
        );
    }

    #[test]
    fn finding11_no_contradictory_prefill_floor_with_passed_true() {
        // The generic gates score came in with passed_prefill_speedup_floor=false + a 0.95 prefill
        // floor. After the merge, passed=true must NOT coexist with a false prefill-floor flag.
        let out = merge_matched(
            &passing_gates(),
            &results_with(1.0, vec![1.0]),
            None,
            None,
            &pool1(),
        )
        .unwrap();
        let v: Value = serde_json::from_str(&out.sealed_json).unwrap();
        assert_eq!(v["passed"], json!(true));
        assert_eq!(
            v["metrics"]["passed_prefill_speedup_floor"],
            json!(true),
            "finding 11: no passed=true alongside a contradictory prefill-floor=false"
        );
        // The prefill floor is neutralized to not-applicable (0.0), not the generic 0.95.
        assert_eq!(v["metrics"]["prefill_speedup_floor"], json!(0.0));
        assert_eq!(v["scoring_mode"], json!(SCORING_MODE));
    }

    #[test]
    fn rejects_non_passing_gates_score() {
        // partial_result=false ⇒ a full (already-overlaid) score; refuse to overlay again.
        let mut g = passing_gates();
        g.metrics.partial_result = false;
        assert!(merge_matched(&g, &results_with(1.0, vec![1.0]), None, None, &pool1()).is_err());
        // passed=false ⇒ a failed gate; refuse.
        let mut g = passing_gates();
        g.passed = false;
        assert!(merge_matched(&g, &results_with(1.0, vec![1.0]), None, None, &pool1()).is_err());
        // passed_correctness=false ⇒ refuse.
        let mut g = passing_gates();
        g.metrics.passed_correctness = false;
        assert!(merge_matched(&g, &results_with(1.0, vec![1.0]), None, None, &pool1()).is_err());
        // A non-empty gate error ⇒ refuse.
        let mut g = passing_gates();
        g.metrics.error = "gate boom".to_string();
        assert!(merge_matched(&g, &results_with(1.0, vec![1.0]), None, None, &pool1()).is_err());
    }

    #[test]
    fn rejects_gates_score_with_empty_harness_hash() {
        // RULED SCOPE-DOWN (David) — a gates/seam-1 score whose harness identity is EMPTY is
        // REFUSED before any overlay. Revert-proof mutation: delete the `harness_hash.is_empty()`
        // arm in `validate_gates` and this `unwrap_err()` panics (the merge would proceed), so the
        // assertion is load-bearing on the check itself.
        let mut g = passing_gates();
        g.metrics.harness_hash = String::new();
        let err =
            merge_matched(&g, &results_with(1.0, vec![1.0]), None, None, &pool1()).unwrap_err();
        assert!(
            err.contains("empty metrics.harness_hash"),
            "empty harness_hash must be named: {err}"
        );
        // BASELINE-GATE — the SAME fixture with a well-formed 64-hex harness_hash OVERLAYS (PASS),
        // proving the refusal is attributable to the empty identity and does not false-reject a
        // legit gates score.
        assert!(
            merge_matched(
                &passing_gates(),
                &results_with(1.0, vec![1.0]),
                None,
                None,
                &pool1()
            )
            .is_ok(),
            "a well-formed harness_hash must still overlay"
        );
    }

    #[test]
    fn rejects_gates_score_with_malformed_harness_hash() {
        // RULED SCOPE-DOWN (David) — a harness_hash that is present but not the 64-lowercase-hex
        // SHA256 shape is MALFORMED and REFUSED. Revert-proof mutation: delete the
        // `is_well_formed_harness_hash` arm and every `unwrap_err()` below panics.
        // (a) wrong LENGTH (too short).
        let mut g = passing_gates();
        g.metrics.harness_hash = "abc123".to_string();
        let err =
            merge_matched(&g, &results_with(1.0, vec![1.0]), None, None, &pool1()).unwrap_err();
        assert!(
            err.contains("malformed metrics.harness_hash"),
            "short harness_hash must be named malformed: {err}"
        );
        // (b) correct length (64) but a NON-HEX character.
        let mut g = passing_gates();
        g.metrics.harness_hash = format!("{}z", "a".repeat(63));
        let err =
            merge_matched(&g, &results_with(1.0, vec![1.0]), None, None, &pool1()).unwrap_err();
        assert!(
            err.contains("malformed metrics.harness_hash"),
            "non-hex harness_hash must be named malformed: {err}"
        );
        // (c) 64 hex but UPPERCASE — the shape is lowercase, so uppercase is malformed too.
        let mut g = passing_gates();
        g.metrics.harness_hash = "A".repeat(64);
        assert!(
            merge_matched(&g, &results_with(1.0, vec![1.0]), None, None, &pool1()).is_err(),
            "uppercase hex is not the lowercase SHA256 shape and must be refused"
        );
        // BASELINE-GATE — the well-formed baseline still overlays.
        assert!(
            merge_matched(
                &passing_gates(),
                &results_with(1.0, vec![1.0]),
                None,
                None,
                &pool1()
            )
            .is_ok(),
            "a well-formed harness_hash must still overlay"
        );
    }

    #[test]
    fn rejects_gates_score_that_already_carries_a_score() {
        // Minor (validate_gates requires score == null): a gates-only artifact that already carries
        // a non-null score is an already-scored/full artifact, not the gates half; refuse it.
        let mut g = passing_gates();
        g.score = Some(1.23);
        let err =
            merge_matched(&g, &results_with(1.0, vec![1.0]), None, None, &pool1()).unwrap_err();
        assert!(
            err.contains("non-null score"),
            "must name the pre-set score: {err}"
        );
    }

    #[test]
    fn rejects_cross_run_identity_mismatch() {
        // Minor (overlay input identity cross-check): gates + results must be from the SAME run.
        // A commit mismatch is REFUSED.
        let mut r = results_with(1.0, vec![1.0]);
        r.commit = "feedface".to_string(); // gates commit is "deadbeef"
        let err = merge_matched(&passing_gates(), &r, None, None, &pool1()).unwrap_err();
        assert!(
            err.contains("commit mismatch"),
            "commit mismatch must be named: {err}"
        );

        // A weights_hash mismatch is REFUSED.
        let mut r = results_with(1.0, vec![1.0]);
        r.weights_hash = "other-w".to_string();
        let err = merge_matched(&passing_gates(), &r, None, None, &pool1()).unwrap_err();
        assert!(
            err.contains("weights_hash mismatch"),
            "weights mismatch must be named: {err}"
        );

        // A MISSING/empty identity (the default-open fail path) is REFUSED, not fabricated-match.
        let mut r = results_with(1.0, vec![1.0]);
        r.commit = String::new();
        let err = merge_matched(&passing_gates(), &r, None, None, &pool1()).unwrap_err();
        assert!(
            err.contains("empty commit"),
            "empty identity must fail closed: {err}"
        );

        // A hand-forged gates-score that omits its identity fields (both empty) is REFUSED even
        // against a matching-empty results — the non-empty requirement closes the default-open.
        let mut g = passing_gates();
        g.metrics.commit = String::new();
        g.metrics.weights_hash = String::new();
        let mut r = results_with(1.0, vec![1.0]);
        r.commit = String::new();
        r.weights_hash = String::new();
        assert!(
            merge_matched(&g, &r, None, None, &pool1()).is_err(),
            "empty==empty identity must not fabricate a match"
        );
    }

    #[test]
    fn rejects_unvalidated_results() {
        // parity_all_ok=false ⇒ refuse.
        let mut r = results_with(1.0, vec![1.0]);
        r.parity_all_ok = false;
        assert!(merge_matched(&passing_gates(), &r, None, None, &pool1()).is_err());
        // no accepted pairs ⇒ refuse.
        let mut r = results_with(1.0, vec![1.0]);
        r.accepted_pair_count = 0;
        assert!(merge_matched(&passing_gates(), &r, None, None, &pool1()).is_err());
        // empty per_prompt ⇒ refuse.
        let mut r = results_with(1.0, vec![1.0]);
        r.per_prompt.clear();
        assert!(merge_matched(&passing_gates(), &r, None, None, &pool1()).is_err());
    }

    #[test]
    fn die5_results_are_rejected_no_green_score() {
        // finding R2 — a run that die-5'd must NEVER overlay to a passing score, even though every
        // OTHER field (parity_all_ok, a plausible 1.0 median, non-empty per_prompt) looks green.
        // Case 1: the sealed verdict candidate_accepted=false ⇒ REJECTED (no score).
        let mut r = results_with(1.0, vec![1.0]);
        r.candidate_accepted = false;
        let err = merge_matched(&passing_gates(), &r, None, None, &pool1()).unwrap_err();
        assert!(
            err.contains("candidate_accepted"),
            "die-5 verdict must be named: {err}"
        );

        // Case 2: accepted_pair_count < min_pairs ⇒ REJECTED (the numeric die-5 predicate), even if
        // some stale candidate_accepted flag were true.
        let mut r = results_with(1.0, vec![1.0, 1.0]);
        r.accepted_pair_count = 2;
        r.min_pairs = 3; // accepted 2 < 3 required
        r.candidate_accepted = true;
        let err = merge_matched(&passing_gates(), &r, None, None, &pool1()).unwrap_err();
        assert!(
            err.contains("min_pairs"),
            "the accepted < min_pairs die-5 must be named: {err}"
        );
    }

    #[test]
    fn results_view_parse_fails_closed_without_die5_seal() {
        // finding R2 — the die-5 seal fields are REQUIRED: a results.json omitting either
        // `candidate_accepted` or `min_pairs` ERRORS at parse (fail-closed), never defaulting to a
        // passing verdict. A file carrying the aggregate but not the seal must not parse.
        let no_seal = br#"{"parity_all_ok":true,"accepted_pair_count":1,
            "per_prompt":[{"raw_ratio_of_means":1.0}],"pairs":[{"raw_ratio":1.0}],
            "aggregate":{"baseline_serial_seconds_per_token_mean":0.038,
            "candidate_mtp_seconds_per_token_mean":0.038}}"#;
        assert!(
            ResultsView::parse(no_seal).is_err(),
            "missing die-5 seal must fail parse"
        );
    }

    #[test]
    fn ranking_fields_survive_coarsening_bit_unchanged() {
        // A median with many sig figs must survive the seal uncoarsened (ranking field), and the
        // floor/ceiling are exact.
        let out = merge_matched(
            &passing_gates(),
            &results_with(1.234_567_891_234, vec![1.234_567_891_234]),
            None,
            None,
            &pool1(),
        )
        .unwrap();
        let v: Value = serde_json::from_str(&out.sealed_json).unwrap();
        assert_eq!(
            v["metrics"]["decode_speedup"].as_f64().unwrap(),
            1.234_567_891_234
        );
        assert_eq!(v["score"].as_f64().unwrap(), 1.234_567_891_234);
        assert_eq!(v["metrics"]["decode_speedup_floor"], json!(0.90));
        assert_eq!(v["decode_speedup_ceiling"], json!(5.0));
    }

    #[test]
    fn forged_no_pairs_results_is_rejected_exploit_repro() {
        // finding R3 — LIVE EXPLOIT repro: a forged results.json carrying the die-5 seal, a
        // plausible 4.9 per-prompt median, positive means, accepted_pair_count=1, but NO `pairs`
        // array previously merged to passed=true / score 4.9. With `pairs` REQUIRED it must now
        // ERROR at parse (fail-closed) — no ResultsView, no merge, no green score.
        let forged_no_pairs = br#"{
            "track_id":"qwen3.8-27b-mtp-v1","parity_all_ok":true,"accepted_pair_count":1,
            "candidate_accepted":true,"min_pairs":1,
            "per_prompt":[{"raw_ratio_of_means":4.9}],
            "aggregate":{"baseline_serial_seconds_per_token_mean":0.186,
            "candidate_mtp_seconds_per_token_mean":0.038}}"#;
        assert!(
            ResultsView::parse(forged_no_pairs).is_err(),
            "a forged results.json with NO pairs must ERROR at parse, not default to empty"
        );

        // pairs-len mismatch (present but len != accepted_pair_count) ⇒ validate REJECTS.
        let mut r = results_with(4.9, vec![4.9]);
        r.accepted_pair_count = 3; // pairs.len() == 1 != 3
        let err = merge_matched(&passing_gates(), &r, None, None, &pool1()).unwrap_err();
        assert!(
            err.contains("pairs.len"),
            "the length mismatch must be named: {err}"
        );

        // zero baseline mean ⇒ validate REJECTS (no green score off a zero-anchored ratio).
        let mut r = results_with(4.9, vec![4.9]);
        r.aggregate.baseline_serial_seconds_per_token_mean = 0.0;
        let err = merge_matched(&passing_gates(), &r, None, None, &pool1()).unwrap_err();
        assert!(
            err.contains("baseline_serial_seconds_per_token_mean"),
            "zero mean must be named: {err}"
        );
    }

    #[test]
    fn results_view_parse_fails_closed_without_aggregate() {
        assert!(ResultsView::parse(b"not json").is_err());
        // Missing the required aggregate block ⇒ parse error (fail-closed).
        assert!(ResultsView::parse(b"{\"parity_all_ok\":true}").is_err());
    }

    #[test]
    fn rejects_track_id_mismatch_accepts_match() {
        // finding R12 — the overlay REFUSES a results.json whose sealed track_id != the expected
        // track (the ranked gate is `.track_id == $track`), and ACCEPTS a matching one.
        // results_with seals this tree's own track (bench_core::constants::TRACK_ID).
        let r = results_with(1.0, vec![1.0]);
        // Mismatch → refused.
        let err = merge_matched(
            &passing_gates(),
            &r,
            Some("qwen3.6-27b-mtp-v1"),
            None,
            &pool1(),
        )
        .unwrap_err();
        assert!(
            err.contains("track_id") && err.contains("expected track"),
            "mismatch named: {err}"
        );
        // Match → accepted (a passing 1.0 median).
        let out = merge_matched(
            &passing_gates(),
            &r,
            Some(bench_core::constants::TRACK_ID),
            None,
            &pool1(),
        )
        .unwrap();
        assert!(out.passed, "a matching track scores normally");
        // An EMPTY sealed track_id is refused even with no expected track (never trust an arbitrary).
        let mut empty = results_with(1.0, vec![1.0]);
        empty.track_id = String::new();
        let err = merge_matched(&passing_gates(), &empty, None, None, &pool1()).unwrap_err();
        assert!(
            err.contains("track_id is empty"),
            "empty track_id refused: {err}"
        );
    }

    /// The THIRD leg of `constant≡contract≡env` at the merge. The expected track and the sealed
    /// `track_id` can AGREE with each other and still name a track this tree does not serve; the
    /// commit/weights_hash cross-checks cannot see it, because both halves really are one run.
    /// The gates half was sealed against THIS track's official baseline pair, so merging it under
    /// a foreign track mis-attributes that pair. Refused by name, naming both values.
    #[test]
    fn rejects_a_foreign_track_even_when_both_sides_agree() {
        const FOREIGN: &str = "gemma4-26b-a4b-mlx-v1";
        assert_ne!(FOREIGN, bench_core::constants::TRACK_ID);
        let mut foreign = results_with(1.0, vec![1.0]);
        foreign.track_id = FOREIGN.to_string();
        // The operator names the SAME foreign track, so the R12 equality check passes.
        // F1 — driven through `merge_matched`, the tests' matched-seal-leg entry: the production
        // `merge_overlay` resolves its seal leg from the process CWD, which a unit test has no
        // nine-root workspace for. The track fence is upstream of the harness cross-check either
        // way, so this still proves the fence and not the resolution.
        let err =
            merge_matched(&passing_gates(), &foreign, Some(FOREIGN), None, &pool1()).unwrap_err();
        assert!(err.contains("track_id fence"), "{err}");
        assert!(err.contains(FOREIGN), "must name the declared track: {err}");
        assert!(
            err.contains(bench_core::constants::TRACK_ID),
            "must name the tree's track: {err}"
        );
        // With NO expected track the R12 equality check is skipped entirely — the fence still
        // refuses, so an unset env cannot open the hole.
        let err = merge_matched(&passing_gates(), &foreign, None, None, &pool1()).unwrap_err();
        assert!(err.contains("track_id fence"), "{err}");
    }

    #[test]
    fn results_view_parse_requires_track_id() {
        // finding R12 — `track_id` is REQUIRED: a results.json omitting it ERRORS at parse
        // (fail-closed), never defaulting to an empty/absent track that would slip the gate.
        let no_track = br#"{"parity_all_ok":true,"accepted_pair_count":1,
            "candidate_accepted":true,"min_pairs":1,
            "per_prompt":[{"raw_ratio_of_means":1.0}],"pairs":[{"raw_ratio":1.0}],
            "aggregate":{"baseline_serial_seconds_per_token_mean":0.038,
            "candidate_mtp_seconds_per_token_mean":0.038}}"#;
        assert!(
            ResultsView::parse(no_track).is_err(),
            "missing track_id must fail parse"
        );
    }

    // --- finding R17: the full live ranked pool-shape predicate set ---

    #[test]
    fn r17_clean_eight_prompt_pool_passes() {
        // A clean 8-prompt pool (all parity_ok, distinct shas, positive means/noop) PASSES the full
        // predicate set and scores the even-n median (all 1.0 ⇒ 1.0).
        let ratios = [1.0; 8];
        let out = merge_matched(
            &passing_gates(),
            &pool_results(&ratios),
            None,
            None,
            &pooln(8),
        )
        .unwrap();
        assert!(out.passed, "a clean 8-prompt pool scores");
        assert_eq!(out.score, Some(1.0));
    }

    #[test]
    fn r17_per_prompt_len_ne_pool_size_rejected() {
        // per_prompt.len() (and prompt_count) == 8, but the sealed accepted count exceeds the pool
        // while per_prompt is short: a pool whose cardinality != pool_size is REJECTED. Keep enough
        // accepted pairs to clear the run-total floor so the cardinality predicate is what fires.
        let mut r = pool_results(&[1.0; 8]);
        r.per_prompt.truncate(3); // per_prompt.len() == 3 != pool_size 8 (accepted_pair_count still 8)
        r.prompt_count = 3;
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(
            err.contains("per_prompt.len") && err.contains("pool_size"),
            "len mismatch: {err}"
        );
    }

    #[test]
    fn r17_prompt_count_ne_pool_size_rejected() {
        // A pool whose sealed prompt_count disagrees with per_prompt.len()/pool_size ⇒ REJECTED.
        let mut r = pool_results(&[1.0; 8]);
        r.prompt_count = 7; // per_prompt.len() == 8 == pool_size, but the sealed count is 7
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(
            err.contains("prompt_count"),
            "prompt_count mismatch must be named: {err}"
        );
    }

    #[test]
    fn r17_per_prompt_parity_false_rejected() {
        // One per-prompt parity_ok=false ⇒ REJECTED (even though top-level parity_all_ok is true).
        let mut r = pool_results(&[1.0; 8]);
        r.per_prompt[3].parity_ok = false;
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(
            err.contains("parity_ok"),
            "per-prompt parity failure must be named: {err}"
        );
    }

    #[test]
    fn r17_duplicate_prompt_sha256_rejected() {
        // A duplicated prompt sha (unique.len < pool_size) ⇒ REJECTED.
        let mut r = pool_results(&[1.0; 8]);
        let dup = r.per_prompt[0].prompt_sha256.clone();
        r.per_prompt[5].prompt_sha256 = dup;
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(
            err.contains("distinct"),
            "duplicate prompt sha must be named: {err}"
        );
    }

    #[test]
    fn r17_nonpositive_noop_reference_rejected() {
        // A per-prompt noop_reference_decode_speedup <= 0 (absent/blank) ⇒ REJECTED.
        let mut r = pool_results(&[1.0; 8]);
        r.per_prompt[2].noop_reference_decode_speedup = 0.0;
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(
            err.contains("noop_reference_decode_speedup"),
            "noop<=0 must be named: {err}"
        );
    }

    #[test]
    fn r17_bad_prompt_sha256_hex_rejected() {
        // A per-prompt prompt_sha256 that is not 64 lowercase hex ⇒ REJECTED.
        let mut r = pool_results(&[1.0; 8]);
        r.per_prompt[1].prompt_sha256 = "NOTHEX".to_string();
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(
            err.contains("lowercase hex"),
            "bad sha256 must be named: {err}"
        );
    }

    #[test]
    fn r17_run_total_min_pairs_floor_enforced() {
        // accepted_pair_count below the run-total floor (pool_size * min_per_prompt) ⇒ REJECTED.
        let r = pool_results(&[1.0; 8]);
        // min_per_prompt=2 ⇒ run floor 16; the pool has 8 accepted pairs and each prompt has 1.
        let err = merge_matched(
            &passing_gates(),
            &r,
            None,
            None,
            &PoolExpectation {
                pool_size: 8,
                min_per_prompt: 2,
            },
        )
        .unwrap_err();
        // The per-prompt accepted (1) < min_per_prompt (2) OR the run-total floor both reject.
        assert!(
            err.contains("min_per_prompt") || err.contains("min_pairs"),
            "min-pairs floor must reject: {err}"
        );
    }

    #[test]
    fn r17_aggregate_diagnostics_required_at_parse() {
        // finding R17 — the aggregate per-pair diagnostics + the sealed median are REQUIRED: a
        // results.json whose aggregate omits mtp_decode_speedup_median / _min /
        // raw_decode_speedup_median ERRORS at parse (fail-closed).
        let no_diag = br#"{"track_id":"qwen3.8-27b-mtp-v1","parity_all_ok":true,
            "accepted_pair_count":1,"candidate_accepted":true,"min_pairs":1,"prompt_count":1,
            "per_prompt":[{"prompt_sha256":"aa","parity_ok":true,"accepted_pair_count":1,
            "serial_seconds_per_token_mean":0.038,"mtp_seconds_per_token_mean":0.038,
            "raw_ratio_of_means":1.0,"noop_reference_decode_speedup":1.0}],
            "pairs":[{"raw_ratio":1.0}],
            "aggregate":{"baseline_serial_seconds_per_token_mean":0.038,
            "candidate_mtp_seconds_per_token_mean":0.038}}"#;
        assert!(
            ResultsView::parse(no_diag).is_err(),
            "missing aggregate diagnostics must fail parse"
        );
    }

    #[test]
    fn r17_prompt_count_required_at_parse() {
        // finding R17 — prompt_count is REQUIRED: omitting it ERRORS at parse (fail-closed).
        let no_count = br#"{"track_id":"qwen3.8-27b-mtp-v1","parity_all_ok":true,
            "accepted_pair_count":1,"candidate_accepted":true,"min_pairs":1,
            "per_prompt":[{"prompt_sha256":"aa","parity_ok":true,"accepted_pair_count":1,
            "serial_seconds_per_token_mean":0.038,"mtp_seconds_per_token_mean":0.038,
            "raw_ratio_of_means":1.0,"noop_reference_decode_speedup":1.0}],
            "pairs":[{"raw_ratio":1.0}],
            "aggregate":{"baseline_serial_seconds_per_token_mean":0.038,
            "candidate_mtp_seconds_per_token_mean":0.038,"mtp_decode_speedup_median":1.0,
            "mtp_decode_speedup_min":1.0,"raw_decode_speedup_median":1.0}}"#;
        assert!(
            ResultsView::parse(no_count).is_err(),
            "missing prompt_count must fail parse"
        );
    }

    // --- finding R18: sealed-median agreement (wrapper-tamper detector) ---

    #[test]
    fn r18_sealed_median_tamper_beyond_eps_rejected() {
        // A results.json whose sealed aggregate.raw_decode_speedup_median disagrees with the median
        // recomputed from the per-prompt means by MORE than 1e-7 is a wrapper tamper ⇒ REJECTED.
        let mut r = pool_results(&[1.0; 8]); // recompute median 1.0
        r.aggregate.raw_decode_speedup_median = 1.0 + 1e-6; // sealed disagrees by 1e-6 (> 1e-7)
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(
            err.contains("sealed-median agreement failed") && err.contains("tamper"),
            "sealed-median tamper must be named: {err}"
        );
    }

    #[test]
    fn r18_sealed_median_within_eps_passes() {
        // A sealed median within 1e-7 of the recompute is accepted (finite float noise tolerated).
        let mut r = pool_results(&[1.0; 8]);
        r.aggregate.raw_decode_speedup_median = 1.0 + 5e-8; // within 1e-7
        let out = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap();
        assert!(
            out.passed,
            "a sealed median within 1e-7 of the recompute passes"
        );
    }

    #[test]
    fn r18_overlay_uses_recomputed_median_not_sealed_field() {
        // The published score is the RECOMPUTED median (from the per-prompt means), NOT the sealed
        // field trusted blindly. Build per-prompt means that recompute to 1.5 while the sealed
        // raw_ratio_of_means fields say something else; the score must be the 1.5 recompute.
        let mut r = pool_results(&[1.5, 1.5, 1.5]); // recompute median 1.5, sealed median 1.5
                                                    // Tamper ONLY the per-prompt raw_ratio_of_means (a field the overlay must NOT score off);
                                                    // keep them > 0 so R17 passes. The recompute uses serial/mtp means (still 1.5).
        for p in &mut r.per_prompt {
            p.raw_ratio_of_means = 4.0;
        }
        let out = merge_matched(&passing_gates(), &r, None, None, &pooln(3)).unwrap();
        let score = out.score.expect("passes");
        assert!(
            (score - 1.5).abs() < 1e-12,
            "score must be the 1.5 recompute, got {score}"
        );
        let v: Value = serde_json::from_str(&out.sealed_json).unwrap();
        assert!((v["metrics"]["decode_speedup"].as_f64().unwrap() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn r18_even_n_recompute_uses_mean_of_two_central() {
        // Even-n pool: recompute median = mean of the two central order statistics (NOT lower).
        // ratios [1.0, 2.0, 3.0, 4.0] ⇒ median 2.5. Sealed median 2.5 agrees.
        let r = pool_results(&[1.0, 2.0, 3.0, 4.0]); // all pairs < 8.0 bound, median 2.5 < 5.0 ceiling
        let out = merge_matched(&passing_gates(), &r, None, None, &pooln(4)).unwrap();
        assert!(
            (out.score.unwrap() - 2.5).abs() < 1e-12,
            "even-n recompute median must be 2.5"
        );
    }

    // --- finding R20: floor/ceiling failure artifacts + native scoring_mode ---

    #[test]
    fn r20_scoring_mode_is_the_native_form() {
        // finding R20 — the SCORING_MODE discriminator is the native form aligned with results.json
        // `mode` ("qwen-native-mtp-paired-decode-only"), NOT the old "qwen-mtp-paired-decode-only".
        assert_eq!(SCORING_MODE, "qwen-native-mtp-paired-decode-only");
    }

    #[test]
    fn r20_passing_median_carries_native_scoring_mode() {
        // A passing median ⇒ merged score with the native scoring_mode discriminator.
        let out = merge_matched(
            &passing_gates(),
            &results_with(1.0, vec![1.0]),
            None,
            None,
            &pool1(),
        )
        .unwrap();
        assert!(out.passed);
        let v: Value = serde_json::from_str(&out.sealed_json).unwrap();
        assert_eq!(
            v["scoring_mode"],
            json!("qwen-native-mtp-paired-decode-only")
        );
        // A real score (not 0) is authored on a pass.
        assert_eq!(v["score"], json!(1.0));
    }

    #[test]
    fn r20_floor_fail_exact_artifact_shape() {
        // finding R20 — the FULL floor-fail artifact shape: score 0, passed false, the exact error
        // string, and passed_decode_speedup_floor:false. Mirrors the live score.json.
        let out = merge_matched(
            &passing_gates(),
            &results_with(0.85, vec![0.85]),
            None,
            None,
            &pool1(),
        )
        .unwrap();
        let v: Value = serde_json::from_str(&out.sealed_json).unwrap();
        assert_eq!(v["score"], json!(0.0));
        assert_eq!(v["passed"], json!(false));
        assert_eq!(v["metrics"]["passed_decode_speedup_floor"], json!(false));
        let err = v["metrics"]["error"].as_str().unwrap();
        assert!(
            err.starts_with("performance floor failed ("),
            "load-bearing prefix: {err}"
        );
    }

    #[test]
    fn r20_ceiling_fail_exact_artifact_shape() {
        // finding R20 — the ceiling-fail artifact: score 0, passed false, the exact non-attributable
        // error string.
        let out = merge_matched(
            &passing_gates(),
            &results_with(6.0, vec![6.0]),
            None,
            None,
            &pool1(),
        )
        .unwrap();
        let v: Value = serde_json::from_str(&out.sealed_json).unwrap();
        assert_eq!(v["score"], json!(0.0));
        assert_eq!(v["passed"], json!(false));
        let err = v["metrics"]["error"].as_str().unwrap();
        assert!(
            err.starts_with("qwen-mtp plausibility ceiling exceeded"),
            "exact ceiling text: {err}"
        );
    }

    // -----------------------------------------------------------------------
    // W3 — the §5 SERIES FENCE
    // -----------------------------------------------------------------------

    /// Fable ruling — the shape a scored v1.1 free-run run seals: BOTH legs free-run, on every pair.
    fn free_run_results(ratio: f64, pairs: Vec<f64>) -> ResultsView {
        let mut r = results_with(ratio, pairs);
        r.timed_mode = bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1.to_string();
        r.timed_series = free_run_series();
        r.pairs = r.pairs.iter().map(|p| free_run_pair(p.raw_ratio)).collect();
        r
    }

    /// The CROSSED shape (TF serial control, free-run candidate), sealed honestly. measure-job can
    /// no longer produce it; the fence must still refuse it.
    fn mixed_results(ratio: f64, pairs: Vec<f64>) -> ResultsView {
        let mut r = results_with(ratio, pairs);
        r.timed_mode = crate::measure_job::MIXED_SERIES_DESCRIPTOR.to_string();
        r.timed_series = mixed_series();
        r.pairs = r.pairs.iter().map(|p| mixed_pair(p.raw_ratio)).collect();
        r
    }

    #[test]
    fn fence_accepts_both_legal_shapes() {
        // The two shapes measure-job produces — an all-teacher-forced run and an ALL-FREE-RUN run
        // (Fable ruling: the serial control runs the candidate's series) — both validate and both
        // merge to a real score. The fence gates series identity, not the arithmetic.
        assert!(validate_series(&results_with(1.5, vec![1.5]), None).is_ok());
        assert!(validate_series(&free_run_results(1.5, vec![1.5]), None).is_ok());
        let out = merge_matched(
            &passing_gates(),
            &free_run_results(1.5, vec![1.5]),
            None,
            None,
            &pool1(),
        )
        .unwrap();
        assert!(out.passed);
        // The merged score CARRIES the series identity so it cannot be lost downstream.
        let v: Value = serde_json::from_str(&out.sealed_json).unwrap();
        assert_eq!(v["timed_mode"], json!("free_run_v1_1"));
        assert_eq!(v["timed_series"]["legs_comparable"], json!(true));
        assert_eq!(
            v["timed_series"]["serial_leg_timed_mode"],
            json!("free_run_v1_1")
        );
        assert_eq!(
            v["timed_series"]["candidate_leg_timed_mode"],
            json!("free_run_v1_1")
        );
    }

    #[test]
    fn fence_refuses_a_cross_series_ratio_even_when_sealed_honestly() {
        // §5 / Fable ruling — the CROSSED shape is refused, not scored with a caveat. Its seal is
        // internally coherent (legs_comparable: false is the TRUTH about it), and the earlier
        // posture published it anyway. §5 calls that a scoring bug: the ratio divides a
        // teacher-forced denominator by a free-run numerator, and the 27.2 ms M-5 round-trip floor
        // no longer cancels (N times on one side, once on the other).
        let r = mixed_results(1.5, vec![1.5]);
        let err = validate_series(&r, None).unwrap_err();
        assert!(err.contains("NOT §5-comparable"), "{err}");
        assert!(err.contains("cross-series ratio"), "{err}");
        // The merge refuses it too — no score is published for a crossed run.
        assert!(merge_matched(&passing_gates(), &r, None, None, &pool1()).is_err());
        // DEFENSE IN DEPTH: measure-job's own rule cannot construct this pairing.
        assert_eq!(
            crate::measure_job::serial_control_regime_for(
                crate::measure_job::LegRegime::FreeRunV1_1
            ),
            crate::measure_job::LegRegime::FreeRunV1_1
        );
    }

    #[test]
    fn fence_refuses_pairs_from_two_series_pooled_into_one_file() {
        // THE aggregation guard: a `pairs[]` mixing a teacher-forced candidate pair with a free-run
        // candidate pair would make the median an average over two physical quantities.
        let mut r = mixed_results(1.5, vec![1.5, 1.5]);
        r.pairs[1] = tf_pair(1.5); // one pair from the OTHER series
        let err = validate_series(&r, None).unwrap_err();
        assert!(
            err.contains("refusing to aggregate mismatched series") && err.contains("pairs[1]"),
            "{err}"
        );
        // The merge refuses it too (the fence runs inside validate_results).
        assert!(merge_matched(&passing_gates(), &r, None, None, &pool1()).is_err());
    }

    #[test]
    fn fence_refuses_a_sealed_comparability_lie() {
        // A mixed run stamped `legs_comparable: true` is exactly the §5 lie the rule exists to
        // catch; the overlay RECOMPUTES the verdict and refuses the stamp.
        let mut r = mixed_results(1.5, vec![1.5]);
        r.timed_series.legs_comparable = true;
        let err = validate_series(&r, None).unwrap_err();
        assert!(err.contains("legs_comparable"), "{err}");

        // Same for a `homogeneous: true` on legs that disagree.
        let mut r = mixed_results(1.5, vec![1.5]);
        r.timed_series.homogeneous = true;
        let err = validate_series(&r, None).unwrap_err();
        assert!(err.contains("homogeneous"), "{err}");
    }

    #[test]
    fn fence_refuses_a_top_level_label_that_hides_the_second_series() {
        // Sealing a crossed run under a single tag would let an aggregator read one series where two
        // were measured. Refused in both directions — and note the crossed file is refused EITHER
        // WAY now (check 4 fires on comparability); this asserts the LABEL check specifically, so a
        // future relaxation of check 4 cannot silently un-cover the mislabel.
        let mut r = mixed_results(1.5, vec![1.5]);
        r.timed_mode = "free_run_v1_1".to_string();
        // Recompute-vs-seal (check 3) and comparability (check 4) both hold their own line; the
        // descriptor check is exercised directly on the same file.
        let err = validate_series(&r, None).unwrap_err();
        assert!(
            err.contains("NOT §5-comparable"),
            "a crossed file never reaches scoring: {err}"
        );

        let mut r = results_with(1.5, vec![1.5]); // homogeneous TF
        r.timed_mode = crate::measure_job::MIXED_SERIES_DESCRIPTOR.to_string();
        let err = validate_series(&r, None).unwrap_err();
        assert!(
            err.contains("timed_mode") && err.contains("mixed:"),
            "a homogeneous run must not claim MIXED: {err}"
        );

        // ...and the same for the free-run shape: a homogeneous free-run run mislabelled with the
        // other series' tag is refused by the descriptor check.
        let mut r = free_run_results(1.5, vec![1.5]);
        r.timed_mode = "teacher_forced_v1".to_string();
        let err = validate_series(&r, None).unwrap_err();
        assert!(
            err.contains("the top-level series label and the per-leg tags disagree"),
            "{err}"
        );
    }

    #[test]
    fn fence_refuses_an_unknown_series_tag() {
        let mut r = results_with(1.5, vec![1.5]);
        r.timed_series.candidate_leg_timed_mode = "free_run_v2".to_string();
        r.pairs = vec![PairView {
            raw_ratio: 1.5,
            serial_timed_mode: bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1.to_string(),
            candidate_timed_mode: "free_run_v2".to_string(),
        }];
        let err = validate_series(&r, None).unwrap_err();
        assert!(err.contains("not a known timed series"), "{err}");
    }

    #[test]
    fn fence_refuses_a_file_sealed_for_a_different_expected_series() {
        // §5 makes baselines/floors/bands PER-SERIES: a v1.1 run must not be scored where a v1 run
        // was expected, or vice versa.
        let tf = results_with(1.5, vec![1.5]);
        assert!(validate_series(&tf, Some("teacher_forced_v1")).is_ok());
        let err =
            validate_series(&tf, Some(crate::measure_job::MIXED_SERIES_DESCRIPTOR)).unwrap_err();
        assert!(err.contains("expected series"), "{err}");

        let fr = free_run_results(1.5, vec![1.5]);
        assert!(validate_series(&fr, Some("free_run_v1_1")).is_ok());
        let err = validate_series(&fr, Some("teacher_forced_v1")).unwrap_err();
        assert!(err.contains("expected series"), "{err}");

        // Naming the MIXED descriptor as the expected series never admits a crossed file either:
        // check 4 refuses it before the expectation is even consulted.
        let mixed = mixed_results(1.5, vec![1.5]);
        assert!(
            validate_series(&mixed, Some(crate::measure_job::MIXED_SERIES_DESCRIPTOR)).is_err()
        );
        assert!(validate_series(&mixed, Some("teacher_forced_v1")).is_err());
    }

    #[test]
    fn results_json_without_a_series_fails_to_parse() {
        // A file that will not say which quantity it measured must ERROR at parse, not be scored as
        // an unknown-series number (the same fail-closed posture as track_id / candidate_accepted).
        // (ResultsView is Deserialize-only, so the fixture is written as JSON.)
        let base = json!({
            "track_id": "t",
            "timed_mode": "teacher_forced_v1",
            "timed_series": {
                "serial_leg_timed_mode": "teacher_forced_v1",
                "candidate_leg_timed_mode": "teacher_forced_v1",
                "homogeneous": true,
                "legs_comparable": true
            },
            "candidate_accepted": true,
            "min_pairs": 1,
            "prompt_count": 1,
            "pairs": [{
                "raw_ratio": 1.5,
                "serial_timed_mode": "teacher_forced_v1",
                "candidate_timed_mode": "teacher_forced_v1"
            }],
            "aggregate": {
                "mtp_decode_speedup_median": 1.5,
                "mtp_decode_speedup_min": 1.5,
                "raw_decode_speedup_median": 1.5
            }
        });
        assert!(
            ResultsView::parse(&serde_json::to_vec(&base).unwrap()).is_ok(),
            "the complete shape parses"
        );
        for missing in ["timed_mode", "timed_series"] {
            let mut doc = base.clone();
            doc.as_object_mut().unwrap().remove(missing);
            assert!(
                ResultsView::parse(&serde_json::to_vec(&doc).unwrap()).is_err(),
                "a results.json without {missing} must fail closed at parse"
            );
        }
        // A pair record that omits its per-leg tags is equally fatal — those tags are what make the
        // sealed descriptor falsifiable.
        let mut doc = base.clone();
        doc["pairs"][0]
            .as_object_mut()
            .unwrap()
            .remove("candidate_timed_mode");
        assert!(ResultsView::parse(&serde_json::to_vec(&doc).unwrap()).is_err());
    }
    // =======================================================================
    // F1 CHANGE 2 — the COHORT publication seam
    // =======================================================================

    /// The b8 series descriptor a batched run seals: both legs `batched_free_run_v1_2_b8`,
    /// homogeneous and §5-comparable with each other.
    fn cohort_series() -> TimedSeriesView {
        TimedSeriesView {
            serial_leg_timed_mode: bench_core::free_run::TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8
                .to_string(),
            candidate_leg_timed_mode: bench_core::free_run::TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8
                .to_string(),
            homogeneous: true,
            legs_comparable: true,
        }
    }

    fn cohort_pair(raw_ratio: f64) -> PairView {
        PairView {
            raw_ratio,
            serial_timed_mode: bench_core::free_run::TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8
                .to_string(),
            candidate_timed_mode: bench_core::free_run::TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8
                .to_string(),
        }
    }

    /// The composite the identity cohort produces: `prefill_gain^0.25 * decode_gain^0.75`, computed
    /// through the SAME expression the overlay's coherence check recomputes, so the fixture is
    /// coherent by construction and a test that wants incoherence has to introduce it deliberately.
    fn composite_of(prefill_gain: f64, decode_gain: f64) -> CompositeView {
        let score = prefill_gain.powf(crate::measure_job::PREFILL_GAIN_EXPONENT)
            * decode_gain.powf(crate::measure_job::DECODE_GAIN_EXPONENT);
        CompositeView {
            prefill_gain,
            decode_gain,
            composite_score: score,
            composite_speedup_floor: QWEN_MTP_DECODE_SPEEDUP_FLOOR,
            composite_speedup_floor_met: score >= QWEN_MTP_DECODE_SPEEDUP_FLOOR,
        }
    }

    /// A CONFORMANT cohort-shaped results.json: one cohort of 8 members (slot order, distinct
    /// shas), `per_prompt` EMPTY, `pairs` in the b8 series, and a coherent sealed composite.
    /// `pair_ratios` are the accepted pairs' cohort ratios (the sealed median is theirs).
    fn cohort_results(prefill_gain: f64, decode_gain: f64, pair_ratios: &[f64]) -> ResultsView {
        let members: Vec<CohortMemberView> = (0..8)
            .map(|i| CohortMemberView {
                slot_index: i,
                prompt_sha256: sha(i),
            })
            .collect();
        let acc = pair_ratios.len();
        let median = even_n_median(pair_ratios);
        let mut sorted = pair_ratios.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ResultsView {
            // The track this tree serves. The cohort SHAPE is what these fixtures exercise; the
            // track fence is a separate gate with its own test, so the fixture must not trip it.
            track_id: bench_core::constants::TRACK_ID.to_string(),
            timed_mode: bench_core::free_run::TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8.to_string(),
            timed_series: cohort_series(),
            parity_all_ok: true,
            accepted_pair_count: acc,
            candidate_accepted: true,
            min_pairs: 1,
            // THE COHORT SHAPE: no per-prompt records at all.
            per_prompt: Vec::new(),
            prompt_count: 8,
            pairs: pair_ratios.iter().copied().map(cohort_pair).collect(),
            aggregate: AggregateView {
                baseline_serial_seconds_per_token_mean: 0.040,
                candidate_mtp_seconds_per_token_mean: 0.020,
                mtp_decode_speedup_median: median,
                mtp_decode_speedup_min: *sorted.first().unwrap(),
                raw_decode_speedup_median: median,
            },
            commit: "deadbeef".to_string(),
            weights_hash: "w-hash".to_string(),
            per_cohort: vec![PerCohortView {
                cohort_sha256: sha(99),
                members,
                batch_size: crate::measure_job::SCORED_BATCH_SIZE_B8,
                parity_ok: true,
                accepted_pair_count: acc,
                serial_seconds_per_token_mean: 0.040,
                candidate_seconds_per_token_mean: 0.020,
                raw_ratio_of_means: 2.0,
                composite_scored_exponents: Some(ScoredExponentsView {
                    prefill_gain_exponent: crate::measure_job::PREFILL_GAIN_EXPONENT,
                    decode_gain_exponent: crate::measure_job::DECODE_GAIN_EXPONENT,
                }),
                composite: Some(composite_of(prefill_gain, decode_gain)),
                composite_absent_reason: None,
            }],
            scored_batch_size: Some(crate::measure_job::SCORED_BATCH_SIZE_B8),
        }
    }

    /// The identity cohort: 4x prefill gain, 2x decode gain ⇒ composite `4^0.25 * 2^0.75 = 2^1.25`.
    fn identity_cohort() -> ResultsView {
        cohort_results(4.0, 2.0, &[2.0, 2.0])
    }

    /// THE ACCEPTANCE PATH — a cohort-shaped results.json overlays, and the PUBLISHED SCORE is the
    /// sealed shared-window COMPOSITE (not a median of per-prompt ratios, which this artifact does
    /// not contain). Before F1 this merge was unreachable: the overlay refused the regime name
    /// outright, so the cohort publication seam had never run.
    #[test]
    fn a_cohort_results_json_overlays_and_publishes_the_composite() {
        let r = identity_cohort();
        let expected = 2f64.powf(1.25);
        let out = merge_matched(&passing_gates(), &r, None, None, &pooln(8))
            .expect("a conformant cohort artifact must overlay");
        assert!(out.passed, "cohort merge must pass: {}", out.sealed_json);
        let score = out.score.expect("a passing cohort run publishes a score");
        assert!(
            (score - expected).abs() < 1e-12,
            "the published score must be the composite {expected}, got {score}"
        );

        let v: Value = serde_json::from_str(&out.sealed_json).expect("sealed json parses");
        assert_eq!(v["score"].as_f64().unwrap(), score);
        assert_eq!(v["metrics"]["decode_speedup"].as_f64().unwrap(), score);
        assert_eq!(v["passed"], json!(true));
        assert_eq!(v["metrics"]["partial_result"], json!(false));
        // The aggregation label FOLLOWS the shape — never the median label on a cohort score.
        assert_eq!(v["aggregation"], json!(AGGREGATION_COHORT_COMPOSITE));
        assert_ne!(v["aggregation"], json!(AGGREGATION));
        // What was scored travels with the number.
        assert_eq!(v["scored_batch_size"], json!(8));
        assert_eq!(v["cohort"]["cohort_sha256"], json!(sha(99)));
        assert_eq!(v["cohort"]["member_count"], json!(8));
        assert_eq!(v["cohort"]["prefill_gain"].as_f64().unwrap(), 4.0);
        assert_eq!(v["cohort"]["decode_gain"].as_f64().unwrap(), 2.0);
        assert_eq!(
            v["timed_mode"],
            json!(bench_core::free_run::TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8)
        );
    }

    #[test]
    fn overlay_private_audit_keeps_exact_metrics_and_insertions() {
        if std::env::var_os("BENCHD_OVERLAY_PRIVATE_AUDIT_CHILD").is_some() {
            let out =
                merge_matched(&passing_gates(), &identity_cohort(), None, None, &pooln(8)).unwrap();
            assert!(out.passed);
            return;
        }

        let root = std::env::temp_dir().join(format!(
            "benchd-overlay-private-audit-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("private.json");
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "overlay::tests::overlay_private_audit_keeps_exact_metrics_and_insertions",
                "--test-threads=1",
            ])
            .env("BENCHD_OVERLAY_PRIVATE_AUDIT_CHILD", "1")
            .env("BENCHD_PRIVATE_AUDIT_PATH", &path)
            .status()
            .unwrap();
        assert!(status.success());

        let private: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(private["metrics"]["peak_ram_gb"], json!(20.25));
        assert_eq!(private["cohort"]["composite_score"], json!(2f64.powf(1.25)));
        std::fs::remove_dir_all(root).unwrap();
    }

    /// REFUSAL TWIN of the accepted regime — a batched tag at ANY OTHER WIDTH is still refused. The
    /// producer only ever certifies B=8, so `b1`/`b16` are regimes no honest artifact can carry, and
    /// the acceptance is an exact match rather than a prefix match on `batched_free_run_v1_2_b`.
    #[test]
    fn a_batched_regime_at_another_width_is_still_refused() {
        for tag in ["batched_free_run_v1_2_b1", "batched_free_run_v1_2_b16"] {
            let mut r = identity_cohort();
            r.timed_mode = tag.to_string();
            r.timed_series.serial_leg_timed_mode = tag.to_string();
            r.timed_series.candidate_leg_timed_mode = tag.to_string();
            for p in r.pairs.iter_mut() {
                p.serial_timed_mode = tag.to_string();
                p.candidate_timed_mode = tag.to_string();
            }
            let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
            assert!(
                err.contains("is not a known timed series"),
                "{tag} must still be refused as an unknown series: {err}"
            );
        }
    }

    /// REFUSAL TWIN — a b8 leg paired with a b1 leg is a CROSS-SERIES ratio and stays refused. B is
    /// encoded in the tag, so the existing `timed_modes_comparable` equality does this with no new
    /// gate logic.
    #[test]
    fn a_cohort_leg_crossed_with_another_width_is_refused() {
        let mut r = identity_cohort();
        r.timed_series.serial_leg_timed_mode = "batched_free_run_v1_2_b1".to_string();
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("is not a known timed series"), "{err}");
    }

    /// `results_shape` is EXHAUSTIVE over the known series tags and REFUSES anything else.
    ///
    /// It used to classify every non-b8 tag as `SingleStream` through an `else` fallback, which was
    /// fail-closed only by CALL ORDERING: production reaches it after `validate_series` has already
    /// refused unknown tags. The function is `pub`, so that ordering was a convention. An unknown
    /// tag reaching it directly would have been scored by the single-stream aggregation rule —
    /// silently, because a populated `per_prompt` is all the single-stream arm then asks for.
    ///
    /// REVERT-PROOF: restore the `else` fallback and the unknown-tag assertion below goes red; drop
    /// either known arm and its `Ok` assertion goes red.
    #[test]
    fn results_shape_refuses_a_tag_it_does_not_know() {
        // The three KNOWN tags each classify, and to the right shape.
        for (tag, expected) in [
            (
                bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1,
                ResultsShape::SingleStream,
            ),
            (
                bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1,
                ResultsShape::SingleStream,
            ),
            (
                bench_core::free_run::TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8,
                ResultsShape::Cohort {
                    batch_size: crate::measure_job::SCORED_BATCH_SIZE_B8,
                },
            ),
        ] {
            let mut r = results_with(1.0, vec![1.0]);
            r.timed_series.candidate_leg_timed_mode = tag.to_string();
            assert_eq!(results_shape(&r).unwrap(), expected, "tag {tag}");
        }

        // THE FINDING — an unknown tag is a REFUSAL, not a defaulted single-stream classification.
        // `batched_free_run_v1_2_b16` is the load-bearing case: a batched regime that the b8-only
        // certification can never have produced would have been handed to the per-prompt median.
        for tag in [
            "batched_free_run_v1_2_b16",
            "free_run_v2",
            "",
            "teacher_forced_v1_x",
        ] {
            let mut r = results_with(1.0, vec![1.0]);
            r.timed_series.candidate_leg_timed_mode = tag.to_string();
            let err = results_shape(&r).unwrap_err();
            assert!(
                err.contains("is not a known timed series"),
                "tag {tag:?} must be refused by name: {err}"
            );
        }
    }

    /// The unknown-series REFUSAL MESSAGE must name every tag the known-set actually accepts.
    ///
    /// The message listed only the two single-stream tags while the accepting closure took three,
    /// so an operator debugging a b8-adjacent typo was told the batched regime was not a known
    /// series at all. REVERT-PROOF: drop the b8 tag from either the message or the known-set and
    /// this goes red.
    #[test]
    fn the_unknown_series_refusal_names_every_accepted_tag() {
        let mut r = results_with(1.0, vec![1.0]);
        r.timed_series.candidate_leg_timed_mode = "free_run_v2".to_string();
        let err = validate_series(&r, None).unwrap_err();
        for tag in [
            bench_core::free_run::TIMED_MODE_TEACHER_FORCED_V1,
            bench_core::free_run::TIMED_MODE_FREE_RUN_V1_1,
            bench_core::free_run::TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8,
        ] {
            assert!(
                err.contains(tag),
                "the refusal must name the accepted tag {tag}: {err}"
            );
        }
    }

    /// REFUSAL TWIN of the cohort acceptance — an EMPTY `per_cohort` on a cohort series has nothing
    /// to score.
    #[test]
    fn a_cohort_series_with_no_cohort_records_is_refused() {
        let mut r = identity_cohort();
        r.per_cohort.clear();
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(
            err.contains("per_cohort is empty"),
            "an empty per_cohort must be named: {err}"
        );
    }

    /// REFUSAL TWIN — the shape cross-check runs BOTH ways. A cohort series carrying per-prompt
    /// records, and a single-stream series carrying cohort records, are each refused: the series tag
    /// and the body must agree about what was measured.
    #[test]
    fn a_shape_that_contradicts_its_own_series_tag_is_refused() {
        let mut cohort_with_per_prompt = identity_cohort();
        cohort_with_per_prompt.per_prompt = vec![per_prompt(0, 1.0, 1)];
        let err = merge_matched(
            &passing_gates(),
            &cohort_with_per_prompt,
            None,
            None,
            &pooln(8),
        )
        .unwrap_err();
        assert!(
            err.contains("per_prompt record(s) on a COHORT series"),
            "{err}"
        );

        let mut single_with_cohort = pool_results(&[1.0; 8]);
        single_with_cohort.per_cohort = identity_cohort().per_cohort;
        let err = merge_matched(&passing_gates(), &single_with_cohort, None, None, &pooln(8))
            .unwrap_err();
        assert!(
            err.contains("per_cohort record(s) on a SINGLE-STREAM series"),
            "{err}"
        );
    }

    /// REFUSAL TWIN of the composite acceptance — a cohort run whose composite is ABSENT publishes
    /// NOTHING, and the refusal echoes the producer's own reason.
    #[test]
    fn a_cohort_without_a_composite_is_refused() {
        let mut r = identity_cohort();
        r.per_cohort[0].composite = None;
        r.per_cohort[0].composite_absent_reason =
            Some("zero accepted pairs for the shared window".to_string());
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("composite is absent"), "{err}");
        assert!(
            err.contains("zero accepted pairs for the shared window"),
            "the producer's own reason must be echoed: {err}"
        );

        // And with NO reason sealed either — still refused, never defaulted to a score.
        let mut bare = identity_cohort();
        bare.per_cohort[0].composite = None;
        let err = merge_matched(&passing_gates(), &bare, None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("composite is absent"), "{err}");
    }

    /// The producer's invariant is EXACTLY ONE of composite / composite_absent_reason. A seal
    /// carrying both is incoherent and refused rather than resolved in the run's favour.
    #[test]
    fn a_cohort_sealing_both_a_composite_and_an_absent_reason_is_refused() {
        let mut r = identity_cohort();
        r.per_cohort[0].composite_absent_reason =
            Some("but here is a composite anyway".to_string());
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(
            err.contains("BOTH a composite and a composite_absent_reason"),
            "{err}"
        );
    }

    /// TAMPER — the composite must be re-derivable from the gains sealed beside it. The overlay does
    /// not publish a scoring number it cannot recompute (finding R18's posture, applied to the
    /// cohort's published figure).
    #[test]
    fn a_composite_incoherent_with_its_own_gains_is_refused() {
        let mut r = identity_cohort();
        let c = r.per_cohort[0].composite.as_mut().unwrap();
        c.composite_score *= 1.5; // gains say 2^1.25; the sealed score says otherwise
        c.composite_speedup_floor_met = c.composite_score >= QWEN_MTP_DECODE_SPEEDUP_FLOOR;
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("composite coherence failed"), "{err}");
    }

    /// The scoring EXPONENTS are ruled constants, not per-run data: a run that could name its own
    /// pair could name the one that maximises its score.
    #[test]
    fn a_cohort_declaring_its_own_scoring_exponents_is_refused() {
        let mut r = identity_cohort();
        r.per_cohort[0].composite_scored_exponents = Some(ScoredExponentsView {
            prefill_gain_exponent: 0.75,
            decode_gain_exponent: 0.25,
        });
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("RULED pair"), "{err}");

        // Absent entirely — also refused, never defaulted to the ruled pair.
        let mut absent = identity_cohort();
        absent.per_cohort[0].composite_scored_exponents = None;
        let err = merge_matched(&passing_gates(), &absent, None, None, &pooln(8)).unwrap_err();
        assert!(
            err.contains("composite_scored_exponents is absent"),
            "{err}"
        );
    }

    /// FLOOR ENFORCEMENT, preserved on the cohort path. A composite BELOW the 0.90 track floor
    /// publishes NO score: `passed = false`, `OverlayOutcome.score = None`, and the payload carries
    /// the EXACT live floor error string (the redactor bills `floor_failed` off that prefix).
    #[test]
    fn a_cohort_composite_below_the_floor_publishes_no_score() {
        // gains 0.25 / 0.25 ⇒ composite 0.25, well under 0.90.
        let r = cohort_results(0.25, 0.25, &[2.0, 2.0]);
        let out = merge_matched(&passing_gates(), &r, None, None, &pooln(8))
            .expect("a floor failure is AUTHORED INTO the score, not an Err");
        assert!(!out.passed, "a sub-floor composite must not pass");
        assert!(
            out.score.is_none(),
            "a sub-floor composite publishes no score"
        );
        let v: Value = serde_json::from_str(&out.sealed_json).expect("parses");
        assert_eq!(v["passed"], json!(false));
        assert_eq!(v["score"].as_f64().unwrap(), 0.0);
        let err = v["metrics"]["error"].as_str().unwrap();
        assert!(
            err.starts_with("performance floor failed ("),
            "the live floor prefix is load-bearing: {err}"
        );
        assert_eq!(
            v["metrics"]["passed_decode_speedup_floor"],
            json!(false),
            "the floor flag must agree with the verdict"
        );
    }

    /// A wrapper does not get to LOWER its own floor, or stamp a PASS onto a score that misses it.
    #[test]
    fn a_cohort_that_rewrites_its_own_floor_or_verdict_is_refused() {
        let mut lowered = identity_cohort();
        lowered.per_cohort[0]
            .composite
            .as_mut()
            .unwrap()
            .composite_speedup_floor = 0.10;
        let err = merge_matched(&passing_gates(), &lowered, None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("!= the track floor"), "{err}");

        let mut stamped = cohort_results(0.25, 0.25, &[2.0, 2.0]);
        stamped.per_cohort[0]
            .composite
            .as_mut()
            .unwrap()
            .composite_speedup_floor_met = true; // 0.25 does NOT clear 0.90
        let err = merge_matched(&passing_gates(), &stamped, None, None, &pooln(8)).unwrap_err();
        assert!(
            err.contains("disagrees with the recomputed verdict"),
            "{err}"
        );
    }

    /// PRESERVED REFUSALS on the cohort path — the gates-side harness identity gate, the die-5
    /// verdict, and the per-pair plausibility bound all still fire on a batched artifact. A new
    /// acceptance path must not become a way around the checks the single-stream path enforces.
    #[test]
    fn the_existing_refusals_all_still_fire_on_a_cohort_run() {
        // (1) empty harness identity on the gates half.
        let mut g = passing_gates();
        g.metrics.harness_hash = String::new();
        let err = merge_matched(&g, &identity_cohort(), None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("empty metrics.harness_hash"), "{err}");

        // (2) malformed harness identity.
        let mut g = passing_gates();
        g.metrics.harness_hash = "not-a-digest".to_string();
        let err = merge_matched(&g, &identity_cohort(), None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("malformed metrics.harness_hash"), "{err}");

        // (3) candidate_accepted = false (die 5).
        let mut r = identity_cohort();
        r.candidate_accepted = false;
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("candidate_accepted != true"), "{err}");

        // (4) the per-PAIR plausibility bound (8.0) still applies to a cohort's pairs.
        let r = cohort_results(4.0, 2.0, &[2.0, 99.0]);
        let out = merge_matched(&passing_gates(), &r, None, None, &pooln(8))
            .expect("a bound failure is authored into the score");
        assert!(!out.passed && out.score.is_none());
        let v: Value = serde_json::from_str(&out.sealed_json).expect("parses");
        assert!(
            v["metrics"]["error"]
                .as_str()
                .unwrap()
                .contains("per-pair plausibility bound"),
            "{}",
            v["metrics"]["error"]
        );

        // (5) parity failure on the cohort.
        let mut r = identity_cohort();
        r.per_cohort[0].parity_ok = false;
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("per_cohort[0].parity_ok != true"), "{err}");

        // (6) the identity cross-check between the two seam inputs.
        let mut r = identity_cohort();
        r.commit = "cafebabe".to_string();
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("commit mismatch"), "{err}");
    }

    /// The sealed-median TAMPER DETECTOR is not lost on the cohort path — it is recomputed from the
    /// samples the batched shape actually carries (the accepted pairs' cohort ratios).
    #[test]
    fn a_cohort_whose_sealed_median_disagrees_with_its_pairs_is_refused() {
        let mut r = identity_cohort();
        r.aggregate.raw_decode_speedup_median = 3.5; // the pairs say 2.0
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("sealed-median agreement failed"), "{err}");
    }

    /// The cohort's pool IDENTITY predicates: members are the pool, in slot order, all distinct,
    /// all 64-lowercase-hex — the same grid the single-stream path applies to `per_prompt`.
    #[test]
    fn cohort_member_identity_predicates_are_enforced() {
        let mut short = identity_cohort();
        short.per_cohort[0].members.truncate(3);
        let err = merge_matched(&passing_gates(), &short, None, None, &pooln(8)).unwrap_err();
        assert!(
            err.contains("members.len") && err.contains("pool_size"),
            "{err}"
        );

        let mut dup = identity_cohort();
        dup.per_cohort[0].members[7].prompt_sha256 = sha(0);
        let err = merge_matched(&passing_gates(), &dup, None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("not all distinct"), "{err}");

        let mut bad_sha = identity_cohort();
        bad_sha.per_cohort[0].members[2].prompt_sha256 = "NOTHEX".to_string();
        let err = merge_matched(&passing_gates(), &bad_sha, None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("is not 64 lowercase hex"), "{err}");

        let mut scrambled = identity_cohort();
        scrambled.per_cohort[0].members[4].slot_index = 6;
        let err = merge_matched(&passing_gates(), &scrambled, None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("slot_index"), "{err}");
    }

    /// The sealed WIDTH must agree with the width the series tag encodes, in both places it is
    /// recorded.
    #[test]
    fn a_cohort_whose_width_disagrees_with_its_series_tag_is_refused() {
        let mut r = identity_cohort();
        r.per_cohort[0].batch_size = 4;
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("batch_size"), "{err}");

        let mut r = identity_cohort();
        r.scored_batch_size = Some(4);
        let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
        assert!(err.contains("scored_batch_size"), "{err}");
    }

    /// The accepted-pair FLOOR is counted over the unit that was measured. A cohort run needs
    /// `min_per_unit` pairs of its ONE cohort; applying the single-stream `pool_size *
    /// min_per_prompt` product would demand 8x the pairs the regime produces and refuse every honest
    /// batched artifact. The floor is not weakened — a cohort short of it is still refused.
    #[test]
    fn the_cohort_accepted_pair_floor_is_counted_per_cohort_not_per_prompt() {
        let pool = PoolExpectation {
            pool_size: 8,
            min_per_prompt: 2,
        };
        // TWO accepted pairs of one cohort clears a min-2 floor...
        let ok = cohort_results(4.0, 2.0, &[2.0, 2.0]);
        assert_eq!(
            pool.min_pairs_for(ResultsShape::Cohort { batch_size: 8 }),
            2
        );
        assert_eq!(pool.min_pairs_for(ResultsShape::SingleStream), 16);
        merge_matched(&passing_gates(), &ok, None, None, &pool)
            .expect("two accepted cohort pairs clear a min-2 per-unit floor");

        // ...and ONE does not.
        let short = cohort_results(4.0, 2.0, &[2.0]);
        let err = merge_matched(&passing_gates(), &short, None, None, &pool).unwrap_err();
        assert!(err.contains("cohort min_pairs"), "{err}");
    }

    /// The cohort's per-unit MEANS must be strictly positive, as the per-prompt means are — a blank
    /// or fabricated cohort record cannot merge.
    #[test]
    fn cohort_means_must_be_positive() {
        for mutate in [
            |c: &mut PerCohortView| c.serial_seconds_per_token_mean = 0.0,
            |c: &mut PerCohortView| c.candidate_seconds_per_token_mean = f64::NAN,
            |c: &mut PerCohortView| c.raw_ratio_of_means = -1.0,
        ] {
            let mut r = identity_cohort();
            mutate(&mut r.per_cohort[0]);
            let err = merge_matched(&passing_gates(), &r, None, None, &pooln(8)).unwrap_err();
            assert!(err.contains("is not > 0"), "{err}");
        }
    }

    // -----------------------------------------------------------------------
    // David ruling 2026-08-26 — the CROSS-LEG harness-identity equality at the seal
    // -----------------------------------------------------------------------

    /// A harness identity that is well-formed (so it clears the single-leg gate) but is NOT the one
    /// the seal resolves — the shape a between-phase roster mutation produces.
    fn other_well_formed_identity() -> HarnessIdentity {
        let hash = "b".repeat(64);
        assert_ne!(
            hash,
            HarnessIdentity::TEST_HASH,
            "the mismatch fixture must differ from the matching leg"
        );
        assert!(
            is_well_formed_harness_hash(&hash),
            "the mismatch fixture must still clear the SINGLE-LEG gate, so the refusal under test \
             is the CROSS-LEG one"
        );
        HarnessIdentity::for_test(&hash)
    }

    /// ACCEPTANCE TWIN — the HONEST RUN. The gates leg and the seal leg are the SAME harness, so the
    /// merge publishes. LOAD-BEARING as a negative control on the whole change: if the gate were
    /// wired to reject on equality, or to compare against something the honest flow cannot produce,
    /// this goes red and no amount of refusal coverage would reveal it. Asserted on BOTH SHAPES.
    #[test]
    fn honest_run_matching_harness_legs_publishes_on_both_shapes() {
        // per_prompt (single-stream).
        let single = merge_matched(
            &passing_gates(),
            &results_with(1.0, vec![1.0]),
            None,
            None,
            &pool1(),
        )
        .expect("matching harness legs must publish on the single-stream shape");
        assert!(single.passed, "the honest single-stream run must pass");
        assert!(single.score.is_some(), "the honest run must carry a score");

        // per_cohort (batched).
        let cohort = merge_matched(&passing_gates(), &identity_cohort(), None, None, &pooln(8))
            .expect("matching harness legs must publish on the cohort shape");
        assert!(cohort.passed, "the honest cohort run must pass");
        assert!(cohort.score.is_some(), "the honest run must carry a score");
    }

    /// REFUSAL TWIN — two WELL-FORMED but DIFFERENT legs are refused, on BOTH SHAPES. This is the
    /// between-phase TOCTOU: the gates score was sealed under one harness, the tree changed, and the
    /// seal resolves another. Everything else about both artifacts is the honest fixture, so the
    /// ONLY reason either can fail is the new gate.
    ///
    /// ALSO the ANTI-VACUITY proof. The gate must bind the SEALED seam-1 value; an implementation
    /// that re-derived BOTH sides at the same instant (comparing the seal recompute to itself)
    /// would be trivially satisfied and this test would go GREEN-when-it-must-be-RED. It stays red
    /// only while one side is `gates.metrics.harness_hash` as sealed.
    #[test]
    fn differing_well_formed_harness_legs_are_refused_on_both_shapes() {
        let other = other_well_formed_identity();
        for (shape, results, pool) in [
            ("per_prompt", results_with(1.0, vec![1.0]), pool1()),
            ("per_cohort", identity_cohort(), pooln(8)),
        ] {
            let err = super::merge_overlay_against_harness(
                &passing_gates(),
                &results,
                None,
                None,
                &pool,
                &scored_fixture(),
                &other,
            )
            .expect_err("a seal under a DIFFERENT harness must be refused");
            assert!(
                err.contains("cross-leg harness-identity mismatch"),
                "{shape}: the refusal must be the cross-leg one: {err}"
            );
            // Both digests named, truncated — never the whole sealed identity.
            assert!(
                err.contains(&HarnessIdentity::TEST_HASH[..HARNESS_DIGEST_PREVIEW_LEN]),
                "{shape}: the refusal must name the GATES digest: {err}"
            );
            assert!(
                err.contains(&"b".repeat(HARNESS_DIGEST_PREVIEW_LEN)),
                "{shape}: the refusal must name the SEAL digest: {err}"
            );
            assert!(
                !err.contains(HarnessIdentity::TEST_HASH),
                "{shape}: the refusal must TRUNCATE, not transcribe the identity: {err}"
            );
            // The implication is stated, not just the disagreement.
            assert!(
                err.contains("CHANGED BETWEEN PHASES"),
                "{shape}: the refusal must state the between-phase-mutation implication: {err}"
            );
            assert!(
                err.contains("Refusing to publish"),
                "{shape}: the refusal must be fatal in its own words: {err}"
            );
        }
    }

    /// PRECEDENCE — the pre-existing SINGLE-LEG well-formedness gate stays the EARLIER, CHEAPER
    /// refusal. An empty or malformed gates identity is rejected with its ORIGINAL message, never
    /// with the new cross-leg one, so no pre-F1 refusal text or ordering changed underneath.
    #[test]
    fn single_leg_wellformedness_still_refuses_first_with_its_own_message() {
        for (label, bad, want) in [
            ("empty", String::new(), "empty metrics.harness_hash"),
            (
                "malformed",
                "zz".repeat(32),
                "malformed metrics.harness_hash",
            ),
        ] {
            let mut g = passing_gates();
            g.metrics.harness_hash = bad;
            // Seal leg differs too, so BOTH gates could fire — the single-leg one must win.
            let err = super::merge_overlay_against_harness(
                &g,
                &results_with(1.0, vec![1.0]),
                None,
                None,
                &pool1(),
                &scored_fixture(),
                &other_well_formed_identity(),
            )
            .expect_err("an ill-formed gates identity is still refused");
            assert!(err.contains(want), "{label}: {err}");
            assert!(
                !err.contains("cross-leg harness-identity mismatch"),
                "{label}: the cheap single-leg gate must refuse FIRST: {err}"
            );
        }
    }

    /// FAIL-CLOSED AT THE SEAL — a workspace that is not a harness tree yields NO identity, and the
    /// seal REFUSES rather than publishing unchecked. Driven through the same
    /// `HarnessIdentity::resolve` F1 uses (the production entry `merge_overlay` calls the CWD
    /// variant of it; a parallel test suite may not `chdir`, which is exactly why
    /// `seal_resolution_refusal` is a separate seam).
    #[test]
    fn seal_time_resolution_failure_refuses_and_names_the_cause() {
        // The cause is supplied rather than provoked. This test used to provoke it with an EMPTY
        // directory, back when a missing harnessHash root was a resolution FAILURE; David's
        // skip-not-refuse ruling (the two-engine roster) made a missing root a recorded SKIP, so
        // an empty directory now resolves to a well-formed digest and no cause exists to wrap.
        // What this test pins is the SEAL-side discipline, which is untouched by that ruling: a
        // resolution failure, however it arises, is a REFUSAL that names its cause and says where
        // it fired — never a skip.
        let refusal = seal_resolution_refusal(
            "harnessHash produced a malformed digest (expected 64 lowercase hex characters, got \
             length 3)"
                .to_string(),
        );
        assert!(
            refusal.contains("harnessHash produced a malformed digest"),
            "the seal refusal must carry the cause verbatim: {refusal}"
        );
        assert!(
            refusal.contains("AT THE SEAL"),
            "the refusal must say WHERE it fired: {refusal}"
        );
        assert!(
            refusal.contains("refuses to publish"),
            "resolution failure must be a refusal, never a skip: {refusal}"
        );
    }

    /// The 12-character preview is a PREFIX of the digest and does not panic on a short input.
    #[test]
    fn digest_preview_truncates_to_twelve_and_never_panics() {
        assert_eq!(digest_preview(HarnessIdentity::TEST_HASH).len(), 12);
        assert!(HarnessIdentity::TEST_HASH.starts_with(digest_preview(HarnessIdentity::TEST_HASH)));
        assert_eq!(digest_preview("abc"), "abc");
        assert_eq!(digest_preview(""), "");
    }
}
