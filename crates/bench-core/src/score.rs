//! Score formula, speedups, floors, and acceptance bands.
//!
//! Ported from Sources/MLXFastCore/Score.swift (`BenchmarkScore`, `TimedRunScoreEvaluation`)
//! and Sources/MLXFastCore/AcceptanceBand.swift (`AcceptanceBand`, `AcceptanceBandResult`).
//! The guard / NaN / zero semantics are preserved exactly.

use crate::constants::{
    AcceptanceBands, QWEN_MTP_DECODE_SPEEDUP_CEILING, QWEN_MTP_DECODE_SPEEDUP_FLOOR,
    QWEN_MTP_PER_PAIR_RATIO_BOUND, SCORE_DECODE_SPEEDUP_FLOOR, SCORE_DECODE_WEIGHT,
    SCORE_PREFILL_SPEEDUP_FLOOR, SCORE_PREFILL_WEIGHT,
};

/// THE DEFINITION OF DECODE (David 2026-10-07): decode seconds per token is the DECODE WINDOW —
/// the time of the decode run divided by the N tokens it committed. The seed prefill of the prompt
/// is NOT part of it. Prefill is its own phase, timed on its own.
///
/// Every decode seconds-per-token figure benchd computes, seals, scores, gates, bands or
/// calibrates comes from this one function: the paired official path under every combine rule,
/// the single-leg official path, measure-job (single-stream and cohort), calibrate-baseline and
/// local iterate. There is no other decode figure.
///
/// `decode_run_elapsed_seconds` is the parent clock from the instant the seed prefill closed
/// (`free_decode_begin` returned and its seed token was checked) to the return of the decode run.
/// `decode_tokens` is N (B x N on a cohort).
pub fn decode_window_seconds_per_token(
    decode_run_elapsed_seconds: f64,
    decode_tokens: usize,
) -> f64 {
    decode_run_elapsed_seconds / decode_tokens as f64
}

/// THE DEFINITION OF PREFILL: prefill seconds per token is the prefill phase's time (the median of
/// its timed passes) divided by the prompt's token count. Every prefill seconds-per-token figure
/// benchd computes comes from this one function.
pub fn prefill_seconds_per_token(prefill_elapsed_seconds: f64, prompt_tokens: usize) -> f64 {
    prefill_elapsed_seconds / prompt_tokens as f64
}

/// THE GAIN (`BenchmarkScore.speedup`): control seconds per token / candidate seconds per token, or
/// 0 if either is non-finite or <= 0. A faster candidate has a gain above 1. Every decode gain,
/// prefill gain and per-pair ratio benchd computes comes from this one function.
pub fn speedup(baseline_spt: f64, candidate_spt: f64) -> f64 {
    if !baseline_spt.is_finite()
        || !candidate_spt.is_finite()
        || baseline_spt <= 0.0
        || candidate_spt <= 0.0
    {
        return 0.0;
    }
    baseline_spt / candidate_spt
}

/// `BenchmarkScore.score`: the [`composite`] of the decode and prefill gains, each gain the
/// [`speedup`] of the baseline over the candidate seconds per token.
///
/// Returns `f64::NAN` if a weighted gain is <= 0, or the weights are non-finite / negative / sum
/// to <= 0 (mirrors the Swift `guard ... else { return .nan }`).
pub fn score(
    decode_spt: f64,
    prefill_spt: f64,
    baseline_decode_spt: f64,
    baseline_prefill_spt: f64,
    decode_weight: f64,
    prefill_weight: f64,
) -> f64 {
    composite(
        speedup(baseline_prefill_spt, prefill_spt),
        speedup(baseline_decode_spt, decode_spt),
        ScoringWeights {
            decode: decode_weight,
            prefill: prefill_weight,
        },
    )
}

/// THE COMPOSITE: `prefill_gain ^ (w_p / (w_p + w_d)) * decode_gain ^ (w_d / (w_p + w_d))`, the
/// weighted geometric mean of the two gains. Every composite benchd computes, seals, gates or
/// re-derives comes from this one function: the paired official path (each pair, under both pair
/// rules), the single-leg official path, local iterate and local submit, the measure-job cohort
/// composite, the overlay's coherence check and the per-stream diagnostic. Every declared pair
/// sums to 1, so the exponents are the declared weights themselves.
///
/// * A weight of exactly `0.0` drops its axis: the factor is `1.0` and the gain is not read, so a
///   track that does not measure an axis can leave that gain `NaN`.
/// * An exponent of exactly `1.0` returns the gain itself, with no `powf` round trip.
/// * A weighted gain that is `NaN` or <= 0, or weights that are non-finite, negative or sum to
///   <= 0, give `f64::NAN`. `+inf` is accepted.
pub fn composite(prefill_gain: f64, decode_gain: f64, weights: ScoringWeights) -> f64 {
    let total = weights.decode + weights.prefill;
    if !weights.decode.is_finite()
        || !weights.prefill.is_finite()
        || weights.decode < 0.0
        || weights.prefill < 0.0
        || total.is_nan()
        || total <= 0.0
    {
        return f64::NAN;
    }
    let factor = |gain: f64, weight: f64| -> f64 {
        if weight == 0.0 {
            return 1.0;
        }
        // Reject NaN and non-positive gains. `x.is_nan() || x <= 0.0` is the clippy-clean
        // equivalent of the NaN-catching `!(x > 0.0)` guard (accepts +inf).
        if gain.is_nan() || gain <= 0.0 {
            return f64::NAN;
        }
        let exponent = weight / total;
        if exponent == 1.0 {
            gain
        } else {
            gain.powf(exponent)
        }
    };
    factor(decode_gain, weights.decode) * factor(prefill_gain, weights.prefill)
}

/// THE TWO SCORING WEIGHTS one run combines its gain axes with: the exponents of the weighted
/// geometric mean [`score`] computes (`MLXFastConstants.score{Decode,Prefill}Weight`, 0.75 / 0.25).
///
/// PER PROJECT: the `--contract` track fixture declares them (`score_decode_weight`,
/// `score_prefill_weight`) and `bench_core::contract::scoring_weights` resolves them, refusing a
/// fixture that declares neither. [`ScoringWeights::DEFAULT`] — the [`SCORE_DECODE_WEIGHT`] /
/// [`SCORE_PREFILL_WEIGHT`] constants — is the reference's own pair and rides only on the unscored
/// calibration path. ONE value carries the pair from the fixture to the score a run publishes, so
/// what a run seals is what it weighted.
///
/// The two are ONE declaration: [`score`] divides each by their SUM, so half a pair describes no
/// weighting at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoringWeights {
    pub decode: f64,
    pub prefill: f64,
}

impl ScoringWeights {
    /// The weights the reference carries and every track scored under before the fixture could
    /// declare its own: 0.75 decode, 0.25 prefill.
    pub const DEFAULT: ScoringWeights = ScoringWeights {
        decode: SCORE_DECODE_WEIGHT,
        prefill: SCORE_PREFILL_WEIGHT,
    };
}

/// [`score`] with the pair carried as one value. The guard semantics are [`score`]'s, unchanged:
/// a non-finite, negative or zero-sum pair yields `NaN` rather than a number.
pub fn score_weighted(
    decode_spt: f64,
    prefill_spt: f64,
    baseline_decode_spt: f64,
    baseline_prefill_spt: f64,
    weights: ScoringWeights,
) -> f64 {
    score(
        decode_spt,
        prefill_spt,
        baseline_decode_spt,
        baseline_prefill_spt,
        weights.decode,
        weights.prefill,
    )
}

/// Convenience wrapper using the default 0.75 / 0.25 scoring weights.
pub fn score_default_weights(
    decode_spt: f64,
    prefill_spt: f64,
    baseline_decode_spt: f64,
    baseline_prefill_spt: f64,
) -> f64 {
    score(
        decode_spt,
        prefill_spt,
        baseline_decode_spt,
        baseline_prefill_spt,
        SCORE_DECODE_WEIGHT,
        SCORE_PREFILL_WEIGHT,
    )
}

/// THE TWO SPEEDUP FLOORS one scored run enforces and seals (David ruling 2026-09-09: 0.95 decode
/// AND 0.95 prefill, properly enforced, configurable per project).
///
/// PER PROJECT: the `--contract` track fixture declares them (`decode_speedup_floor`,
/// `prefill_speedup_floor`) and the official path REFUSES a fixture that does not — see
/// `benchd::contract::speedup_floors`. One value carries the pair from that fixture to BOTH the
/// gate ([`evaluate_timed_run`]) and the seal (`metrics.decode_speedup_floor` /
/// `metrics.prefill_speedup_floor`), so what a run seals is what it enforced.
///
/// [`SpeedupFloors::DEFAULT`] — the [`SCORE_DECODE_SPEEDUP_FLOOR`] /
/// [`SCORE_PREFILL_SPEEDUP_FLOOR`] constants — is the LOCAL (no `--contract`) default and nothing
/// else. No scored run may reach it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeedupFloors {
    pub decode: f64,
    pub prefill: f64,
}

impl SpeedupFloors {
    /// The floors a LOCAL run (no `--contract`) uses: the ruled 0.95 / 0.95 constants.
    pub const DEFAULT: SpeedupFloors = SpeedupFloors {
        decode: SCORE_DECODE_SPEEDUP_FLOOR,
        prefill: SCORE_PREFILL_SPEEDUP_FLOOR,
    };
}

/// THE CANDIDATE'S SLOWNESS GATE (David 2026-10-07): a candidate gain is slow enough to refuse
/// only when it is below its floor. The floor is the one slowness gate on the candidate. No
/// candidate-vs-reference band has a slow side. Every pass/fail decision on candidate speed comes
/// from this one predicate: [`passes_speedup_floors`] (the paired path under both pair rules and
/// the single-leg path, through [`evaluate_timed_run`]), the local `passed_*_speedup_floor` flags,
/// the measure-job floor verdict and [`score_paired_decode_only`] (the overlay).
///
/// False if the gain or the floor is not finite. A gain exactly at the floor clears it.
pub fn clears_floor(gain: f64, floor: f64) -> bool {
    gain.is_finite() && floor.is_finite() && gain >= floor
}

/// `BenchmarkScore.passesSpeedupFloors`: both gains [`clears_floor`] on their own axis.
pub fn passes_speedup_floors(
    decode_speedup: f64,
    prefill_speedup: f64,
    decode_floor: f64,
    prefill_floor: f64,
) -> bool {
    clears_floor(decode_speedup, decode_floor) && clears_floor(prefill_speedup, prefill_floor)
}

/// One paired speedup that exceeds a fixture ceiling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeedupCeilingFailure {
    pub axis: &'static str,
    pub value: f64,
    pub ceiling: f64,
}

/// Return the first declared ceiling exceeded by this pair. Absence leaves the old path unchanged;
/// equality passes. Non-finite gains fail closed when that axis has a ceiling.
pub fn first_speedup_above_ceiling(
    decode_speedup: f64,
    prefill_speedup: f64,
    decode_ceiling: Option<f64>,
    prefill_ceiling: Option<f64>,
) -> Option<SpeedupCeilingFailure> {
    [
        ("decode", decode_speedup, decode_ceiling),
        ("prefill", prefill_speedup, prefill_ceiling),
    ]
    .into_iter()
    .find_map(|(axis, value, ceiling)| {
        let ceiling = ceiling?;
        (!value.is_finite() || value > ceiling).then_some(SpeedupCeilingFailure {
            axis,
            value,
            ceiling,
        })
    })
}

#[cfg(test)]
mod speedup_ceiling_tests {
    use super::*;

    #[test]
    fn speedup_ceilings_refuse_only_values_above_them() {
        assert_eq!(
            first_speedup_above_ceiling(5.0, 4.0, Some(5.0), Some(4.0)),
            None
        );
        assert_eq!(
            first_speedup_above_ceiling(4.9, 3.9, Some(5.0), Some(4.0)),
            None
        );
        assert_eq!(first_speedup_above_ceiling(99.0, 99.0, None, None), None);
        let decode = first_speedup_above_ceiling(5.5, 1.0, Some(5.0), None).unwrap();
        assert_eq!(decode.axis, "decode");
        assert_eq!(decode.value, 5.5);
        let prefill = first_speedup_above_ceiling(1.0, 5.5, None, Some(5.0)).unwrap();
        assert_eq!(prefill.axis, "prefill");
    }
}

/// `BenchmarkScore.speedupFloorFailureMessage`: exact POSIX/en_US format, 6 decimals.
pub fn speedup_floor_failure_message(
    decode_speedup: f64,
    prefill_speedup: f64,
    decode_floor: f64,
    prefill_floor: f64,
) -> String {
    // Swift: String(format: "%.6f", locale: en_US_POSIX, value). Rust's {:.6} matches
    // for finite values (the only case this message is produced for in practice).
    format!(
        "performance floor failed: decode_speedup={:.6} floor={:.6} prefill_speedup={:.6} floor={:.6}",
        decode_speedup, decode_floor, prefill_speedup, prefill_floor
    )
}

/// Ported from `AcceptanceBand` (AcceptanceBand.swift).
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptanceBandResult {
    pub passed: bool,
    /// Empty when `passed`; otherwise a human-readable failure reason.
    pub reason: String,
}

impl AcceptanceBandResult {
    fn passed() -> Self {
        AcceptanceBandResult {
            passed: true,
            reason: String::new(),
        }
    }
    fn failed(reason: String) -> Self {
        AcceptanceBandResult {
            passed: false,
            reason,
        }
    }
}

/// THE CANDIDATE'S FAST-SIDE CHECK (the lower side of the Swift `AcceptanceBand.check`). A
/// candidate seconds-per-token figure below `reference * (1 - down_tolerance)` is "improvement too
/// large" or a suspiciously lucky reading, and fails. The check runs only when
/// `enforce_lower_bound` is `true` (the fixture's `*_band_down_enabled`); when it is `false` any
/// finite positive value passes.
///
/// There is NO slow side (David 2026-10-07). A slow candidate is refused by its speedup floor
/// ([`clears_floor`]) and by nothing else. The fixture's `*_band_up_tolerance` is the ceiling of
/// the CONTROL leg's health band (`benchd::baseline::HealthBand::of_contract`) and is not read
/// here. Before this, the band's slow side read the same field, so a candidate prefill had two
/// slowness gates and a floor below `1 / (1 + up)` could never take effect (submission fd79c401).
///
/// A value or reference that is not finite and positive fails on either setting.
pub fn check_fast_side(
    value: f64,
    reference: f64,
    down_tolerance: f64,
    enforce_lower_bound: bool,
    label: &str,
) -> AcceptanceBandResult {
    if !value.is_finite() || value <= 0.0 || !reference.is_finite() || reference <= 0.0 {
        return AcceptanceBandResult::failed(format!(
            "{label} ({value}) and reference ({reference}) must be finite and positive"
        ));
    }
    let lo = reference * (1.0 - down_tolerance);
    if enforce_lower_bound && value < lo {
        return AcceptanceBandResult::failed(format!(
            "{label} {value} below -{}% of reference {reference} (< {lo}): \
improvement too large for one submission (chunk it) or a suspiciously lucky reading",
            down_tolerance * 100.0
        ));
    }
    AcceptanceBandResult::passed()
}

/// Ported from Swift `TimedRunScoreEvaluation`.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedRunScoreEvaluation {
    pub score: f64,
    pub decode_speedup: f64,
    pub prefill_speedup: f64,
    pub passes_floors: bool,
    /// The floors `passes_floors` was decided against, carried so the failure message names the
    /// floors the run actually enforced (never a constant it did not).
    pub floors: SpeedupFloors,
    pub prefill_band: AcceptanceBandResult,
    pub decode_band: AcceptanceBandResult,
}

impl TimedRunScoreEvaluation {
    /// `hasFiniteScore`: score finite && >= 0.
    pub fn has_finite_score(&self) -> bool {
        self.score.is_finite() && self.score >= 0.0
    }

    /// `passesAcceptanceBands`: both bands passed.
    pub fn passes_acceptance_bands(&self) -> bool {
        self.prefill_band.passed && self.decode_band.passed
    }

    /// `firstFailureReason`: same priority order (non-finite score -> floors -> bands).
    pub fn first_failure_reason(&self) -> Option<String> {
        if !self.has_finite_score() {
            return Some("computed score was not finite".to_string());
        }
        if !self.passes_floors {
            return Some(speedup_floor_failure_message(
                self.decode_speedup,
                self.prefill_speedup,
                self.floors.decode,
                self.floors.prefill,
            ));
        }
        if !self.passes_acceptance_bands() {
            let reason = if self.prefill_band.passed {
                &self.decode_band.reason
            } else {
                &self.prefill_band.reason
            };
            return Some(format!("acceptance band failed: {reason}"));
        }
        None
    }
}

/// `BenchmarkScore.evaluateTimedRun`: THE ONE GATE on a candidate's speed, used by the paired
/// official path under both pair rules and by the single-leg official path. The candidate is
/// refused when it is slow by its speedup floors ([`passes_speedup_floors`]), and when it is
/// suspiciously fast by the enabled lower bounds of `bands` ([`check_fast_side`]). The up
/// tolerances of `bands` are not read. `floors` is the run's resolved [`SpeedupFloors`] (the track
/// fixture's pair on a scored run) and the evaluation carries it back, so the caller seals the
/// floors this gate enforced.
pub fn evaluate_timed_run(
    decode_spt: f64,
    prefill_spt: f64,
    baseline_decode_spt: f64,
    baseline_prefill_spt: f64,
    bands: AcceptanceBands,
    floors: SpeedupFloors,
    weights: ScoringWeights,
) -> TimedRunScoreEvaluation {
    let s = score_weighted(
        decode_spt,
        prefill_spt,
        baseline_decode_spt,
        baseline_prefill_spt,
        weights,
    );
    let decode_speedup = speedup(baseline_decode_spt, decode_spt);
    let prefill_speedup = speedup(baseline_prefill_spt, prefill_spt);
    // THE FAST SIDE ONLY. Each lower bound runs when the fixture enables it
    // (`*_band_down_enabled`). Both Nemotron tracks and the paired design disable both, because
    // the control leg is measured live and carries its own health band. The SLOW side is the
    // speedup floors and nothing else.
    let prefill_band = check_fast_side(
        prefill_spt,
        baseline_prefill_spt,
        bands.prefill_down_tolerance,
        bands.prefill_down_enabled,
        "prefill",
    );
    let decode_band = check_fast_side(
        decode_spt,
        baseline_decode_spt,
        bands.decode_down_tolerance,
        bands.decode_down_enabled,
        "decode",
    );
    TimedRunScoreEvaluation {
        score: s,
        decode_speedup,
        prefill_speedup,
        passes_floors: passes_speedup_floors(
            decode_speedup,
            prefill_speedup,
            floors.decode,
            floors.prefill,
        ),
        floors,
        prefill_band,
        decode_band,
    }
}

// ---------------------------------------------------------------------------
// Timed-window liveness (RunTimeout budget)
// ---------------------------------------------------------------------------
//
// A wall-clock deadline for the timed decode round-trips. This is a LIVENESS bound only — it never
// enters the score. (The retired qwen-mtp-paired-decode-only scoring that used to live here went
// with flow B; the single-leg official path scores through `evaluate_timed_run` above.)

/// H3 (cycle-3) — the RunTimeout wall-clock budget for the timed decode round-trips
/// (PROTOCOL-v1.1 §2.2/§4): `N × band_ceiling_spt × margin`. `n` is the token count, `band_ceiling_spt`
/// the upper acceptance/latency band bound (seconds-per-token) for the series, `margin` a fixed
/// slack factor ([`crate::constants::RUN_TIMEOUT_MARGIN`]). The budget is a LIVENESS bound only; it
/// never enters the score.
///
/// #108 (M2) — FAIL-CLOSED on every degenerate input (`n == 0`, non-finite / non-positive ceiling or
/// margin, non-finite / non-positive product): an `Err`, never a `None` that DISARMS the deadline.
/// This function previously returned `None` there and the caller armed no deadline at all, on the
/// reasoning that "a missing budget falls back to the blocking read — safe, not a fake timeout".
/// That is only true when the degenerate input is benchd's own absent configuration. It is NOT true
/// when the input is ATTACKER-CHOSEN: the ceiling is `calibration.serial_mean × band_high`, both
/// read from the `BASELINE_CALIBRATION` file, so a `band_high` of `0.0` made the product
/// non-positive and turned the §2.2 wall-clock bound off through a config file. A hung or looping
/// engine then wedged benchd inside the timed window with nothing to abort it. The caller turns this
/// `Err` into a leg failure with its own reject class, so the condition is loud and the run dies
/// rather than running unbounded.
pub fn run_timeout_budget(
    n: usize,
    band_ceiling_spt: f64,
    margin: f64,
) -> Result<std::time::Duration, String> {
    if n == 0 {
        return Err(
            "RunTimeout budget: token count N is 0, so N × ceiling × margin is not a \
                    positive wall-clock bound (§2.2)"
                .to_string(),
        );
    }
    if !band_ceiling_spt.is_finite() || band_ceiling_spt <= 0.0 {
        return Err(format!(
            "RunTimeout budget: band ceiling ({band_ceiling_spt} s/tok) is not finite and positive \
             — the §2.2 deadline (N × ceiling × margin) cannot be armed from it, and benchd REFUSES \
             to run the timed window unbounded instead (the ceiling is calibration-derived: \
             serial_mean × serial_band_high)"
        ));
    }
    if !margin.is_finite() || margin <= 0.0 {
        return Err(format!(
            "RunTimeout budget: margin ({margin}) is not finite and positive — the §2.2 deadline \
             cannot be armed from it"
        ));
    }
    let secs = n as f64 * band_ceiling_spt * margin;
    if !secs.is_finite() || secs <= 0.0 {
        return Err(format!(
            "RunTimeout budget: N ({n}) × ceiling ({band_ceiling_spt}) × margin ({margin}) = \
             {secs}, which is not a finite positive number of seconds — refusing to run the timed \
             window with no wall-clock bound (§2.2)"
        ));
    }
    Ok(std::time::Duration::from_secs_f64(secs))
}

/// Which bound a paired decode-only run failed (the score is null and `error` names it).
#[derive(Debug, Clone, PartialEq)]
pub enum PairedDecodeFailure {
    /// A single pair ratio exceeded the per-pair plausibility bound (8.0) — rejected before
    /// aggregation. `ratio` is the offending pair value.
    PerPairBound { ratio: f64, bound: f64 },
    /// The raw median was non-finite (a blank/implausible pair leaked through).
    NonFiniteMedian { median: f64 },
    /// The raw median fell below the submission floor (0.90) — a regression worse than -10%.
    Floor { median: f64, floor: f64 },
    /// The raw median exceeded the ceiling (5.0) — a measurement fault or an escape.
    Ceiling { median: f64, ceiling: f64 },
}

impl PairedDecodeFailure {
    /// A human-readable message that NAMES the failing bound (goes into `metrics.error`).
    pub fn message(&self) -> String {
        match self {
            PairedDecodeFailure::PerPairBound { ratio, bound } => format!(
                "paired decode-only per-pair plausibility bound exceeded: pair ratio={ratio} > bound={bound}"
            ),
            PairedDecodeFailure::NonFiniteMedian { median } => {
                format!("paired decode-only median is not finite: raw_median={median}")
            }
            PairedDecodeFailure::Floor { median, floor } => format!(
                "paired decode-only floor failed: raw_median={median} < floor={floor}"
            ),
            PairedDecodeFailure::Ceiling { median, ceiling } => format!(
                "paired decode-only ceiling failed: raw_median={median} > ceiling={ceiling}"
            ),
        }
    }
}

/// The outcome of the qwen-mtp-paired-decode-only score gate.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedDecodeOnlyScore {
    /// The even-n median of the per-prompt raw ratios (ALWAYS reported, full precision — it is
    /// the ranking figure even when a bound fails, for the results.json `decode_speedup`).
    pub raw_median: f64,
    /// `Some(raw_median)` when every bound passed; `None` on any per-pair / floor / ceiling /
    /// non-finite failure.
    pub score: Option<f64>,
    /// True iff `score.is_some()`.
    pub passed: bool,
    /// The failing bound (and its message) when `!passed`.
    pub failure: Option<PairedDecodeFailure>,
}

/// Apply the paired decode-only gate to a run's per-pair ratios (for the per-pair plausibility
/// bound) and per-prompt raw ratios (for the median floor/ceiling). The two slices coincide when
/// there is one pair per prompt (the ranked k=1 default), but are kept separate so the per-pair
/// bound is checked on EACH pair, not on the aggregated per-prompt mean.
///
/// Priority: per-pair plausibility bound (8.0) → non-finite median → floor (0.90) → ceiling (5.0).
pub fn score_paired_decode_only(
    per_pair_ratios: &[f64],
    per_prompt_raw_ratios: &[f64],
) -> PairedDecodeOnlyScore {
    // The EVEN-N median of the per-prompt raw ratios (track fixture
    // `scoring_semantics.median_rule = even_n_mean_of_two_central_order_statistics`).
    let raw_median = crate::stats::even_n_median(per_prompt_raw_ratios);
    let fail = |f: PairedDecodeFailure| PairedDecodeOnlyScore {
        raw_median,
        score: None,
        passed: false,
        failure: Some(f),
    };
    // Per-pair plausibility: any single pair above the bound (or non-finite/≤0) rejects the run
    // before aggregation (box wrapper MAX_PLAUSIBLE_PUBLISHED_SPEEDUP).
    for &r in per_pair_ratios {
        // A 0/negative ratio is an implausible/blank pair (docs classify it PerPairBound, NOT a
        // Floor fail) — reject it here before the median aggregation.
        if !r.is_finite() || r <= 0.0 || r > QWEN_MTP_PER_PAIR_RATIO_BOUND {
            return fail(PairedDecodeFailure::PerPairBound {
                ratio: r,
                bound: QWEN_MTP_PER_PAIR_RATIO_BOUND,
            });
        }
    }
    if !raw_median.is_finite() {
        return fail(PairedDecodeFailure::NonFiniteMedian { median: raw_median });
    }
    if !clears_floor(raw_median, QWEN_MTP_DECODE_SPEEDUP_FLOOR) {
        return fail(PairedDecodeFailure::Floor {
            median: raw_median,
            floor: QWEN_MTP_DECODE_SPEEDUP_FLOOR,
        });
    }
    if raw_median > QWEN_MTP_DECODE_SPEEDUP_CEILING {
        return fail(PairedDecodeFailure::Ceiling {
            median: raw_median,
            ceiling: QWEN_MTP_DECODE_SPEEDUP_CEILING,
        });
    }
    PairedDecodeOnlyScore {
        raw_median,
        score: Some(raw_median),
        passed: true,
        failure: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_timeout_budget_is_n_times_band_ceiling_times_margin() {
        // H3 (cycle-3) — the RunTimeout budget = N × band-ceiling × margin (§2.2/§4).
        let d = run_timeout_budget(128, 0.04, 4.0).unwrap();
        assert!((d.as_secs_f64() - (128.0 * 0.04 * 4.0)).abs() < 1e-9);
        // #108 (M2) — every degenerate input is an ERROR, never a `None` that DISARMS the §2.2
        // deadline. The ceiling is calibration-derived (serial_mean × serial_band_high), so a
        // silently-disarmed deadline was reachable from a config file.
        for (n, ceiling, margin, what) in [
            (0usize, 0.04, 4.0, "N==0"),
            (128, 0.0, 4.0, "non-positive ceiling"),
            (128, -1.0, 4.0, "negative ceiling"),
            (128, 0.04, 0.0, "non-positive margin"),
            (128, f64::NAN, 4.0, "non-finite ceiling"),
            (128, 0.04, f64::INFINITY, "non-finite margin"),
        ] {
            let err = run_timeout_budget(n, ceiling, margin)
                .expect_err(&format!("{what} must not silently disarm the deadline"));
            assert!(err.contains("RunTimeout budget"), "{what}: {err}");
        }
    }

    /// THE ONE COMPOSITE. `score` is `composite` of the two gains; the ruled 0.25 / 0.75 pair is
    /// the raw exponent form bit for bit; a zero-weight axis drops out without reading its gain;
    /// an exponent of 1 is the gain itself; a weighted gain <= 0 or NaN is NaN.
    #[test]
    fn composite_is_the_one_weighted_geometric_mean_every_score_reads() {
        let w = ScoringWeights::DEFAULT;
        let (p, d) = (1.37_f64, 1.46_f64);
        assert_eq!(
            composite(p, d, w).to_bits(),
            (d.powf(0.75) * p.powf(0.25)).to_bits()
        );
        assert_eq!(
            score(0.5, 0.25, 0.5 * d, 0.25 * p, 0.75, 0.25).to_bits(),
            composite(speedup(0.25 * p, 0.25), speedup(0.5 * d, 0.5), w).to_bits()
        );
        let decode_only = ScoringWeights {
            decode: 1.0,
            prefill: 0.0,
        };
        assert_eq!(composite(f64::NAN, d, decode_only).to_bits(), d.to_bits());
        assert!(composite(p, 0.0, w).is_nan());
        assert!(composite(f64::NAN, d, w).is_nan());
        assert!(composite(
            p,
            d,
            ScoringWeights {
                decode: 0.0,
                prefill: 0.0
            }
        )
        .is_nan());
    }

    #[test]
    fn speedup_equal_is_one() {
        assert_eq!(speedup(0.1, 0.1), 1.0);
    }

    #[test]
    fn speedup_twice_as_fast_is_two() {
        // candidate half the seconds-per-token -> 2x speedup.
        assert_eq!(speedup(0.2, 0.1), 2.0);
    }

    #[test]
    fn speedup_guards_return_zero() {
        assert_eq!(speedup(0.0, 0.1), 0.0);
        assert_eq!(speedup(0.1, 0.0), 0.0);
        assert_eq!(speedup(-1.0, 0.1), 0.0);
        assert_eq!(speedup(f64::NAN, 0.1), 0.0);
        assert_eq!(speedup(f64::INFINITY, 0.1), 0.0);
    }

    #[test]
    fn score_equal_speedups_is_one() {
        // baseline == candidate on both axes -> both speedups 1.0 -> score 1.0.
        let s = score_default_weights(0.1, 0.2, 0.1, 0.2);
        assert!((s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn score_decode_two_prefill_one_is_two_pow_075() {
        // decode_speedup = 2.0, prefill_speedup = 1.0 -> 2^0.75 * 1^0.25 = 2^0.75.
        let s = score_default_weights(0.05, 0.2, 0.1, 0.2);
        assert!((s - 2f64.powf(0.75)).abs() < 1e-12);
    }

    #[test]
    fn score_zero_baseline_is_nan() {
        let s = score_default_weights(0.1, 0.2, 0.0, 0.2);
        assert!(s.is_nan());
    }

    #[test]
    fn score_negative_weight_is_nan() {
        let s = score(0.05, 0.2, 0.1, 0.2, -0.1, 0.25);
        assert!(s.is_nan());
    }

    #[test]
    fn floors_at_exactly_095_pass() {
        assert!(passes_speedup_floors(0.95, 0.95, 0.95, 0.95));
    }

    #[test]
    fn floors_below_fail() {
        assert!(!passes_speedup_floors(0.9499, 1.0, 0.95, 0.95));
        assert!(!passes_speedup_floors(1.0, 0.9499, 0.95, 0.95));
    }

    #[test]
    fn floors_nonfinite_fail() {
        assert!(!passes_speedup_floors(f64::NAN, 1.0, 0.95, 0.95));
    }

    #[test]
    fn floor_message_format() {
        let m = speedup_floor_failure_message(0.9, 0.8, 0.95, 0.95);
        assert_eq!(
            m,
            "performance floor failed: decode_speedup=0.900000 floor=0.950000 \
prefill_speedup=0.800000 floor=0.950000"
        );
    }

    #[test]
    fn fast_side_edge_inclusive_pass() {
        let reference = 100.0;
        // lo = 95 (down 5%). Exactly on the edge passes.
        assert!(check_fast_side(95.0, reference, 0.05, true, "x").passed);
        assert!(check_fast_side(100.0, reference, 0.05, true, "x").passed);
    }

    #[test]
    fn fast_side_beyond_edge_fails() {
        let below = check_fast_side(94.9999, 100.0, 0.05, true, "x");
        assert!(!below.passed);
        assert!(below.reason.contains("improvement too large"));
    }

    /// THE CANDIDATE HAS NO SLOW SIDE (David 2026-10-07). A value far ABOVE the reference passes
    /// the band on either lower-bound setting: the speedup floor is the only slowness gate.
    #[test]
    fn fast_side_check_never_refuses_a_slow_value() {
        for enforce in [true, false] {
            let slow = check_fast_side(1000.0, 100.0, 0.05, enforce, "prefill");
            assert!(slow.passed, "{}", slow.reason);
        }
    }

    /// The MTP timed leg disables the decode DOWN band: a value far BELOW the lower edge (an
    /// "improvement too large") PASSES when `enforce_lower_bound = false`, and the same value is
    /// refused with the bound on.
    #[test]
    fn fast_side_disabled_accepts_a_large_improvement() {
        let reference = 100.0;
        let fast = check_fast_side(50.0, reference, 0.05, false, "decode");
        assert!(
            fast.passed,
            "lower bound disabled must accept a large improvement"
        );
        let fast_two_sided = check_fast_side(50.0, reference, 0.05, true, "decode");
        assert!(!fast_two_sided.passed);
        assert!(fast_two_sided.reason.contains("improvement too large"));
    }

    #[test]
    fn band_nonfinite_value_fails() {
        let r = check_fast_side(f64::NAN, 100.0, 0.05, true, "prefill");
        assert!(!r.passed);
        assert!(r.reason.contains("must be finite and positive"));
    }

    /// Test-only bands: a symmetric prefill health gate and a two-sided decode band. Values are
    /// arbitrary; the captured bands live in `constants::OFFICIAL_BASELINE`. `decode_down_enabled`
    /// is `true` here so these generic evaluate_timed_run tests exercise the full two-sided gate.
    const TEST_BANDS: AcceptanceBands = AcceptanceBands {
        prefill_up_tolerance: 0.03,
        prefill_down_tolerance: 0.03,
        decode_up_tolerance: 0.01,
        decode_down_tolerance: 0.025,
        decode_down_enabled: true,
        prefill_down_enabled: true,
    };

    /// PAIRED DESIGN: a candidate prefill 13% faster than the live serial control passes the
    /// prefill band when the lower bound is disabled (`prefill_down_enabled == false`), and is
    /// refused as "improvement too large" by the same evaluation with the bound on. The floors
    /// are unchanged either way.
    #[test]
    fn evaluate_timed_run_prefill_lower_bound_disabled_accepts_large_prefill_gain() {
        let control_decode = 0.0661;
        let control_prefill = 0.0022372;
        let candidate_prefill = 0.0019238; // the refused 2026-09-10 measurement: -14%
        let paired = AcceptanceBands {
            prefill_down_enabled: false,
            decode_down_enabled: false,
            ..TEST_BANDS
        };
        let e = evaluate_timed_run(
            control_decode * 0.85,
            candidate_prefill,
            control_decode,
            control_prefill,
            paired,
            SpeedupFloors::DEFAULT,
            ScoringWeights::DEFAULT,
        );
        assert!(e.prefill_band.passed, "{}", e.prefill_band.reason);
        assert!(e.passes_acceptance_bands());
        assert!(e.passes_floors);
        let two_sided = evaluate_timed_run(
            control_decode * 0.85,
            candidate_prefill,
            control_decode,
            control_prefill,
            AcceptanceBands {
                decode_down_enabled: false,
                ..TEST_BANDS
            },
            SpeedupFloors::DEFAULT,
            ScoringWeights::DEFAULT,
        );
        assert!(!two_sided.prefill_band.passed);
        assert!(two_sided
            .prefill_band
            .reason
            .contains("improvement too large"));
        // A prefill 4% slower than the control passes the band: the floor is the only slowness
        // gate, and 1 / 1.04 clears 0.95.
        let slow = evaluate_timed_run(
            control_decode * 0.85,
            control_prefill * 1.04,
            control_decode,
            control_prefill,
            paired,
            SpeedupFloors::DEFAULT,
            ScoringWeights::DEFAULT,
        );
        assert!(slow.prefill_band.passed, "{}", slow.prefill_band.reason);
        assert_eq!(slow.first_failure_reason(), None);
    }

    /// REGRESSION fd79c401 (run 37713077396): David set the CUDA prefill floor to 0.80, and the
    /// run was still refused by the band's slow side, "prefill 0.000318502 exceeds +5% of
    /// reference 0.000276407". The floor is the candidate's one slowness gate: with the up
    /// tolerance at 0.05, a prefill gain of 0.85 (or the measured 0.868) passes a 0.80 floor, and
    /// 0.79 is refused BY THE FLOOR, with the floor message. Both tolerances set to any value
    /// change nothing on the slow side.
    #[test]
    fn regression_fd79c401_prefill_floor_is_the_only_slowness_gate() {
        let nemotron = AcceptanceBands {
            prefill_up_tolerance: 0.05,
            prefill_down_tolerance: 0.05,
            decode_up_tolerance: 0.02,
            decode_down_tolerance: 0.05,
            decode_down_enabled: false,
            prefill_down_enabled: false,
        };
        let floors = SpeedupFloors {
            decode: 0.95,
            prefill: 0.80,
        };
        let control_decode = 1.0 / 75.0;
        let candidate_decode = 1.0 / 122.0;
        let control_prefill = 0.000276407;
        let eval = |candidate_prefill: f64, bands: AcceptanceBands| {
            evaluate_timed_run(
                candidate_decode,
                candidate_prefill,
                control_decode,
                control_prefill,
                bands,
                floors,
                ScoringWeights::DEFAULT,
            )
        };
        let wide_up = AcceptanceBands {
            prefill_up_tolerance: 1e9,
            decode_up_tolerance: 1e9,
            ..nemotron
        };
        for bands in [nemotron, wide_up] {
            // The measured fd79c401 pair 1.
            let measured = eval(0.000318502, bands);
            assert_eq!(measured.first_failure_reason(), None);
            // Gain 0.85.
            let at_085 = eval(control_prefill / 0.85, bands);
            assert!((at_085.prefill_speedup - 0.85).abs() < 1e-12);
            assert!(at_085.passes_floors);
            assert!(at_085.passes_acceptance_bands());
            assert_eq!(at_085.first_failure_reason(), None);
            // Gain 0.79: the floor refuses it, by the floor message.
            let at_079 = eval(control_prefill / 0.79, bands);
            assert!(!at_079.passes_floors);
            assert!(at_079.passes_acceptance_bands());
            let reason = at_079.first_failure_reason().unwrap();
            assert!(reason.starts_with("performance floor failed:"), "{reason}");
            assert!(
                reason.contains("prefill_speedup=0.790000 floor=0.800000"),
                "{reason}"
            );
        }
    }

    #[test]
    fn evaluate_timed_run_all_pass() {
        // decode & prefill at baseline -> speedups 1.0, in-band, floors pass, score 1.0.
        let e = evaluate_timed_run(
            0.1336139485703125,
            0.010605031949609375,
            0.1336139485703125,
            0.010605031949609375,
            TEST_BANDS,
            SpeedupFloors::DEFAULT,
            ScoringWeights::DEFAULT,
        );
        assert!((e.score - 1.0).abs() < 1e-12);
        assert!(e.passes_floors);
        assert!(e.passes_acceptance_bands());
        assert!(e.has_finite_score());
        assert_eq!(e.first_failure_reason(), None);
    }

    #[test]
    fn evaluate_timed_run_floor_failure_reported() {
        // Candidate far slower on decode: speedup below floor.
        let e = evaluate_timed_run(
            1.0,
            0.010605031949609375,
            0.1336139485703125,
            0.010605031949609375,
            TEST_BANDS,
            SpeedupFloors::DEFAULT,
            ScoringWeights::DEFAULT,
        );
        assert!(!e.passes_floors);
        let reason = e.first_failure_reason().unwrap();
        assert!(reason.starts_with("performance floor failed:"));
    }

    #[test]
    fn evaluate_timed_run_nonfinite_score_first() {
        let e = evaluate_timed_run(
            0.1,
            0.2,
            0.0,
            0.2,
            TEST_BANDS,
            SpeedupFloors::DEFAULT,
            ScoringWeights::DEFAULT,
        );
        assert!(!e.has_finite_score());
        assert_eq!(
            e.first_failure_reason().as_deref(),
            Some("computed score was not finite")
        );
    }

    /// THE FLOORS ARE THE RUN'S OWN (David 2026-09-09, per-project fixture floors): the gate is
    /// decided against the floors the caller passed, the evaluation CARRIES them, and the failure
    /// message names them — no constant is consulted anywhere in between.
    ///
    /// REVERT-PROOF: put `SCORE_*_SPEEDUP_FLOOR` back into `passes_speedup_floors` or into
    /// `first_failure_reason` and the 0.90 arms below go red.
    #[test]
    fn evaluate_timed_run_enforces_the_floors_it_is_given() {
        // A decode speedup of exactly 0.949 against the ruled 0.95 floor: refused, and the
        // message names 0.950000.
        let below = evaluate_timed_run(
            1.0,
            1.0,
            0.949,
            1.0,
            TEST_BANDS,
            SpeedupFloors::DEFAULT,
            ScoringWeights::DEFAULT,
        );
        assert!(!below.passes_floors);
        assert!(below
            .first_failure_reason()
            .unwrap()
            .contains("decode_speedup=0.949000 floor=0.950000"));
        // Exactly AT the floor passes (>=, not >).
        let at = evaluate_timed_run(
            1.0,
            1.0,
            0.95,
            1.0,
            TEST_BANDS,
            SpeedupFloors::DEFAULT,
            ScoringWeights::DEFAULT,
        );
        assert!(at.passes_floors);
        // The SAME 0.949 decode passes a project whose fixture declares 0.90 — the floors are the
        // fixture's, not the constants'.
        let looser = SpeedupFloors {
            decode: 0.90,
            prefill: 0.90,
        };
        let e = evaluate_timed_run(
            1.0,
            1.0,
            0.949,
            1.0,
            TEST_BANDS,
            looser,
            ScoringWeights::DEFAULT,
        );
        assert!(e.passes_floors);
        assert_eq!(e.floors, looser);
        // Prefill is gated on its own axis, against its own floor.
        let prefill_below = evaluate_timed_run(
            1.0,
            1.0,
            1.0,
            0.949,
            TEST_BANDS,
            SpeedupFloors::DEFAULT,
            ScoringWeights::DEFAULT,
        );
        assert!(!prefill_below.passes_floors);
        assert!(prefill_below
            .first_failure_reason()
            .unwrap()
            .contains("prefill_speedup=0.949000 floor=0.950000"));
        assert!(
            evaluate_timed_run(
                1.0,
                1.0,
                1.0,
                0.949,
                TEST_BANDS,
                looser,
                ScoringWeights::DEFAULT
            )
            .passes_floors
        );
    }

    /// The no-contract default is the ruled pair, and it is the ONLY place the constants enter.
    #[test]
    fn default_floors_are_the_ruled_pair() {
        assert_eq!(SpeedupFloors::DEFAULT.decode, 0.95);
        assert_eq!(SpeedupFloors::DEFAULT.prefill, 0.95);
    }
}
