//! NEAR-TIE STATS SEAL — the REPORT-ONLY measurement behind the (b) tolerance gate's
//! batched-oracle rounding channel (David's ruling "hold 10% + seal near-tie stats", 2026-08-25;
//! the audit's final "measured, not argued" conditional).
//!
//! WHY THIS EXISTS. The ≤10% per-stream tolerance gate
//! ([`crate::cohort_tolerance::evaluate_cohort_token_tolerance`]) judges the candidate's committed
//! tokens against the TRUSTED oracle's `sequential_argmax`. That oracle replays the cohort
//! BATCHED, so cross-stream floating-point geometry can perturb the reference logits by a hair. At
//! a position where the reference's top-1 and top-2 are separated by less than that perturbation,
//! the argmax the oracle reports is not robust — a "flippable near-tie". The audit rated this
//! channel REAL BUT LOW-EXPLOITABILITY, bounded by the raw-mismatch budget, and flagged the bound
//! as ARGUED rather than MEASURED. This module MEASURES it: it counts, per run, how many positions
//! are flippable near-ties and how they overlap the positions that actually mismatched, so the
//! organizer can confirm empirically that the flippable set stays small relative to the
//! [`crate::constants::COHORT_TOKEN_TOLERANCE_PER_THOUSAND`] budget — and gets an EARLY WARNING
//! when an honest submission creeps toward the thin margin (the honest drama case is 9.38% against
//! a 10% bar: 12 mismatches of 128).
//!
//! REPORT-ONLY, STRUCTURALLY. Every function here is TOTAL over well-shaped inputs and returns a
//! value, never a verdict: there is no pass/fail, no threshold comparison that any caller could
//! read as a decision, and no error type that could reach a reject path. The one fallible entry
//! point ([`near_tie_stats`]) fails ONLY on shapes the tolerance gate itself already refuses
//! upstream ([`crate::cohort_tolerance::CohortToleranceError`]), and its caller folds that into a
//! NAMED seal refusal instead of a rejection. Nothing in `candidate_accepted` or any
//! `RejectClass` may ever read these numbers.
//!
//! ── NEAR-TIE DEFINITION (the gap index and the envelope semantics) ─────────────────────────────
//!
//! The oracle's per-position `ranked_relative_gaps` is, per the ENGINE's own definition
//! (`mlxfast-gemma4-26b-a4b-engine@6aed6594`,
//! `Sources/MLXFastHarness/QwenRuntimeCohortReferenceReplay.swift:103-104` — "Relative ratio of
//! each rank's gap from top-1: `(top1 - rank_i)/max(1,|top1|)`" — produced by
//! `Sources/MLXFastHarness/QwenRuntimeWidthProbe.swift:215-216,238-253`
//! `rankedReferenceCharacterization`):
//!
//! ```text
//!     ranked_relative_gaps[i] = (top1_logit - rank_i_logit) / max(1, |top1_logit|)
//! ```
//!
//! rank-0 first, descending logit. So `ranked_relative_gaps[0] == 0.0` ALWAYS (top-1 against
//! itself), and **index [`NEAR_TIE_GAP_INDEX`] == 1 is the top-1→top-2 relative gap** — the only
//! index that can state "how close was the runner-up to winning". That index, and no other, is
//! what this module reads.
//!
//! The envelope is the report-level `rel_envelope`, the SAME scale the engine already compares
//! these gaps against: `within_envelope_depth` is defined as "Count of ranks whose relative ratio
//! `<=` the envelope (top-1 included, so >= 1)" (`QwenRuntimeWidthProbe.swift:217-219,251`). A
//! position is therefore a NEAR-TIE here iff
//!
//! ```text
//!     ranked_relative_gaps[1] <= rel_envelope
//! ```
//!
//! using the engine's own INCLUSIVE `<=`, which makes this predicate EXACTLY equivalent to the
//! engine's `within_envelope_depth >= 2`. Choosing `<` instead, or index 0 (always 0.0 ⇒ every
//! position "near-tie") or index 2 (the top-1→top-3 gap ⇒ an over-count), all produce different
//! numbers; the seal tests pin the values so those mutations are visible.
//!
//! ── WHAT THE NUMBERS MEAN ─────────────────────────────────────────────────────────────────────
//!
//! Per stream, over its committed positions:
//!  * `mismatches` — committed != `sequential_argmax`. THE gate's numerator (recounted here from
//!    the same per-position inputs so the seal is self-contained; it must agree with the gate's
//!    tally by construction — [`crate::cohort_tolerance`] counts the identical predicate).
//!  * `near_tie_positions` — positions where the REFERENCE's top-2 gap is inside the envelope,
//!    i.e. where batch geometry could plausibly have flipped the reported argmax. This is a
//!    property of the REFERENCE distribution alone; it does not depend on what the candidate
//!    committed.
//!  * `near_tie_mismatches` — the OVERLAP: mismatches that ARE near-ties. These are the ones the
//!    rounding channel could account for.
//!  * `non_near_tie_mismatches` — mismatches that are NOT near-ties: the reference was CONFIDENT
//!    and the candidate still diverged. **This is the interesting signal** — genuine divergence,
//!    unattributable to batch geometry.
//!  * `min_committed_relative_gap_on_mismatch` / `median_committed_relative_gap_on_mismatch` — how
//!    deep in the reference's distribution the committed token sat, at the positions that
//!    mismatched. Near 0 ⇒ the candidate picked something the reference nearly picked; large ⇒ the
//!    candidate picked something the reference thought poorly of.
//!
//! And per cohort: the same quantities totalled, plus the HEADROOM stat —
//! `max_stream_mismatch_per_thousand` against `budget_per_thousand` — the single number the
//! organizer watches creep toward the bar.

use serde::Serialize;
use std::fmt;

/// The index into `ranked_relative_gaps` that carries the TOP-1→TOP-2 relative gap.
///
/// Index 0 is the top-1 rank's gap from itself and is identically `0.0` by the engine's formula
/// (see the module header's citation), so index **1** is the first index that says anything about
/// how close the runner-up was. Sealed on every seal so a reader never has to assume it.
pub const NEAR_TIE_GAP_INDEX: usize = 1;

/// The engine's relative gap of one logit from the top-1 logit, as the module header defines it:
/// `(top1_logit - logit) / max(1, |top1_logit|)`. The single-stream paired path reads the
/// `top_logits` of the correctness verbs and computes the gap here, with the same formula.
pub fn relative_gap(top1_logit: f64, logit: f64) -> f64 {
    (top1_logit - logit) / top1_logit.abs().max(1.0)
}

/// One replayed position's gap readout, extracted from the oracle report by the caller.
///
/// All four fields are REQUIRED here: the caller is responsible for turning the protocol's
/// OPTIONAL audit fields into these (or for refusing the seal by name when an engine does not emit
/// them). This keeps the core total — it never has to decide what a missing gap means.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionGaps {
    /// The candidate's committed token at this position (N2-verified upstream).
    pub committed_token: i64,
    /// The trusted reference's argmax at this position — the gate's comparand.
    pub sequential_argmax: i64,
    /// `ranked_relative_gaps[NEAR_TIE_GAP_INDEX]` — the reference's TOP-1→TOP-2 relative gap.
    pub top2_relative_gap: f64,
    /// `committed_relative_gap` — the committed token's own relative gap from the reference top-1
    /// (`(top1 - logit(committed)) / max(1,|top1|)`; 0 exactly when committed IS the argmax).
    pub committed_relative_gap: f64,
}

/// One stream's near-tie tally. VALUES ONLY — no verdict, no threshold comparison.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StreamNearTieStats {
    /// The cohort slot (SLOT ORDER) this tally describes.
    pub slot: usize,
    /// The stream's committed-token count — the per-stream denominator the gate also uses.
    pub committed: usize,
    /// Positions where `committed_token != sequential_argmax` (the gate's numerator).
    pub mismatches: usize,
    /// `mismatches * 1000 / committed` as an EXACT float rate, for reading only.
    ///
    /// NOT the gate's arithmetic: the gate decides in pure integers
    /// (`mismatches * 1000 > tolerance * committed`, strict `>`), precisely so no float rounding
    /// touches the 10% boundary. This float is the same quantity rendered for a human, and a
    /// consumer that wants the decision must redo the integer comparison from `mismatches` and
    /// `committed`, both of which are sealed right here.
    pub mismatch_per_thousand: f64,
    /// Positions where the REFERENCE's top-2 gap is within the envelope — the flippable set
    /// (independent of what the candidate committed).
    pub near_tie_positions: usize,
    /// The OVERLAP: mismatched positions that ARE near-ties (attributable to the rounding channel).
    pub near_tie_mismatches: usize,
    /// Mismatched positions that are NOT near-ties: GENUINE divergence against a confident
    /// reference. `near_tie_mismatches + non_near_tie_mismatches == mismatches`, always.
    pub non_near_tie_mismatches: usize,
    /// The MINIMUM `committed_relative_gap` over the MISMATCHED positions. `None` when the stream
    /// had no mismatch — an empty statistic is omitted, never fabricated as `0.0` (which would
    /// read as "the committed token was the reference top-1", the opposite of a mismatch).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_committed_relative_gap_on_mismatch: Option<f64>,
    /// The MEDIAN `committed_relative_gap` over the MISMATCHED positions, under the house even-n
    /// rule (mean of the two central order statistics on an even count, the middle element on an
    /// odd one — the same rule as `bench_core::score::paired_decode_only_median`). `None` on zero
    /// mismatches, same non-fabrication rule as the min.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub median_committed_relative_gap_on_mismatch: Option<f64>,
}

/// The whole cohort replay's near-tie measurement: the per-stream vectors, the cohort totals, and
/// the headroom stat. VALUES ONLY (see the module header) — nothing here is a decision.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NearTieStats {
    /// The report-level `rel_envelope` the near-tie predicate was evaluated at, VERBATIM from the
    /// oracle report. Sealed so the predicate is recomputable from the artifact alone.
    pub rel_envelope: f64,
    /// [`NEAR_TIE_GAP_INDEX`] — WHICH `ranked_relative_gaps` index was read (1 = top-1→top-2).
    pub near_tie_gap_index: usize,
    /// The machine-readable statement of the predicate, so no consumer has to infer it from the
    /// two fields above ([`NEAR_TIE_PREDICATE`]).
    pub near_tie_predicate: &'static str,
    /// One tally per stream, in SLOT ORDER.
    pub per_stream: Vec<StreamNearTieStats>,
    /// Σ `committed` over the streams.
    pub committed_total: usize,
    /// Σ `mismatches`.
    pub mismatches_total: usize,
    /// Σ `near_tie_positions` — the size of the flippable set across the whole cohort.
    pub near_tie_positions_total: usize,
    /// Σ `near_tie_mismatches`.
    pub near_tie_mismatches_total: usize,
    /// Σ `non_near_tie_mismatches` — the cohort-wide GENUINE-divergence count.
    pub non_near_tie_mismatches_total: usize,
    /// The gate's budget ([`crate::constants::COHORT_TOKEN_TOLERANCE_PER_THOUSAND`]), restated
    /// here so `max_stream_mismatch_per_thousand` has its scale in the same object.
    pub budget_per_thousand: u32,
    /// HEADROOM — the WORST stream's `mismatch_per_thousand`. The gate rejects when this exceeds
    /// `budget_per_thousand`; a run sealing e.g. 93.75 against 100 is the honest-drama warning
    /// this seal exists to surface. (Read-only: the gate reached its own verdict in integers,
    /// upstream and independently.)
    pub max_stream_mismatch_per_thousand: f64,
    /// `budget_per_thousand - max_stream_mismatch_per_thousand`. Negative on a run the gate
    /// rejected; sealed signed rather than clamped so the sign carries the information.
    pub headroom_per_thousand: f64,
}

/// The near-tie predicate, stated for the artifact. See the module header for the derivation and
/// the engine citation.
pub const NEAR_TIE_PREDICATE: &str = "ranked_relative_gaps[1] <= rel_envelope";

/// STRUCTURAL faults in the near-tie inputs. NOT a verdict and NEVER a rejection: every shape
/// named here is one the tolerance gate ALREADY refuses upstream (see
/// [`crate::cohort_tolerance::CohortToleranceError`]), so reaching one of these means the seal is
/// being asked to describe a run that has no tolerance decision either. The caller folds it into a
/// NAMED seal refusal.
#[derive(Debug, Clone, PartialEq)]
pub enum NearTieError {
    /// A stream has zero committed positions (denominator 0).
    EmptyStream { slot: usize },
    /// The report described no streams at all.
    NoStreams,
    /// `rel_envelope` was not a finite, non-negative number — the predicate has no meaning against
    /// it, so no near-tie count is stated rather than a fabricated one.
    UnusableEnvelope { rel_envelope: f64 },
}

impl fmt::Display for NearTieError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NearTieError::EmptyStream { slot } => write!(
                f,
                "near-tie seal: stream {slot} has zero committed positions (no denominator — \
                 nothing to characterize)"
            ),
            NearTieError::NoStreams => write!(
                f,
                "near-tie seal: the oracle report described no streams (nothing to characterize)"
            ),
            NearTieError::UnusableEnvelope { rel_envelope } => write!(
                f,
                "near-tie seal: the oracle reported rel_envelope {rel_envelope}, which is not a \
                 finite non-negative envelope (the near-tie predicate has no meaning against it)"
            ),
        }
    }
}

impl std::error::Error for NearTieError {}

/// The house even-n median over an already-materialised f64 slice: mean of the two central order
/// statistics on an even count, the middle element on an odd one. Mirrors
/// `bench_core::score::paired_decode_only_median`'s rule (kept local rather than reused so the
/// score module's function keeps its single scored meaning; the RULE is deliberately the same).
/// `None` on an empty slice — an empty statistic is omitted, never fabricated.
fn even_n_median(values: &[f64]) -> Option<f64> {
    let n = values.len();
    if n == 0 {
        return None;
    }
    let mut sorted = values.to_vec();
    // Total order over a materialised copy; the inputs are engine-reported gaps, and a non-finite
    // one sorts last rather than panicking `partial_cmp`.
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
    if n % 2 == 1 {
        Some(sorted[n / 2])
    } else {
        Some((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
    }
}

/// Compute the cohort's near-tie measurement over the per-stream position readouts.
///
/// `positions_by_stream[s]` is stream `s`'s positions in emission order (SLOT ORDER on the outer
/// vector, the same order the tolerance gate pinned). `rel_envelope` is the report-level envelope,
/// `budget_per_thousand` the gate's tolerance constant — carried in rather than read from
/// `constants` so the seal states the budget the gate ACTUALLY applied on this run.
///
/// REPORT-ONLY: this returns numbers. It renders no verdict and its `Err` shapes are the ones the
/// gate refuses upstream anyway (see [`NearTieError`]).
pub fn near_tie_stats(
    positions_by_stream: &[Vec<PositionGaps>],
    rel_envelope: f64,
    budget_per_thousand: u32,
) -> Result<NearTieStats, NearTieError> {
    if positions_by_stream.is_empty() {
        return Err(NearTieError::NoStreams);
    }
    if !rel_envelope.is_finite() || rel_envelope < 0.0 {
        return Err(NearTieError::UnusableEnvelope { rel_envelope });
    }
    let mut per_stream = Vec::with_capacity(positions_by_stream.len());
    for (slot, positions) in positions_by_stream.iter().enumerate() {
        let committed = positions.len();
        if committed == 0 {
            return Err(NearTieError::EmptyStream { slot });
        }
        let mut mismatches = 0usize;
        let mut near_tie_positions = 0usize;
        let mut near_tie_mismatches = 0usize;
        // The committed-token gaps AT THE MISMATCHED POSITIONS ONLY — the min/median support.
        let mut mismatch_committed_gaps: Vec<f64> = Vec::new();
        for p in positions {
            // NEAR-TIE is a property of the REFERENCE alone: the top-1→top-2 gap inside the
            // envelope, INCLUSIVE `<=` (the engine's own `within_envelope_depth` comparison — see
            // the module header). Evaluated for EVERY position, matched or not.
            let near_tie = p.top2_relative_gap <= rel_envelope;
            if near_tie {
                near_tie_positions += 1;
            }
            // The gate's own predicate, recounted from the same inputs.
            if p.committed_token != p.sequential_argmax {
                mismatches += 1;
                if near_tie {
                    near_tie_mismatches += 1;
                }
                mismatch_committed_gaps.push(p.committed_relative_gap);
            }
        }
        // The complement is DERIVED, never counted separately: the two halves cannot drift apart.
        let non_near_tie_mismatches = mismatches - near_tie_mismatches;
        let min_committed_relative_gap_on_mismatch =
            mismatch_committed_gaps
                .iter()
                .copied()
                .fold(None::<f64>, |acc, g| {
                    Some(match acc {
                        // `f64::min` propagates the non-NaN side, which would silently hide a NaN gap;
                        // an explicit comparison keeps a NaN visible as itself.
                        Some(m) if m <= g => m,
                        _ => g,
                    })
                });
        per_stream.push(StreamNearTieStats {
            slot,
            committed,
            mismatches,
            mismatch_per_thousand: (mismatches as f64) * 1000.0 / (committed as f64),
            near_tie_positions,
            near_tie_mismatches,
            non_near_tie_mismatches,
            min_committed_relative_gap_on_mismatch,
            median_committed_relative_gap_on_mismatch: even_n_median(&mismatch_committed_gaps),
        });
    }
    let committed_total = per_stream.iter().map(|s| s.committed).sum();
    let mismatches_total = per_stream.iter().map(|s| s.mismatches).sum();
    let near_tie_positions_total = per_stream.iter().map(|s| s.near_tie_positions).sum();
    let near_tie_mismatches_total = per_stream.iter().map(|s| s.near_tie_mismatches).sum();
    let non_near_tie_mismatches_total = per_stream.iter().map(|s| s.non_near_tie_mismatches).sum();
    // The HEADROOM stat: the WORST stream, because the gate is per-stream (a cohort average would
    // be the exact statistic the gate was designed NOT to use).
    let max_stream_mismatch_per_thousand = per_stream
        .iter()
        .map(|s| s.mismatch_per_thousand)
        .fold(0.0f64, f64::max);
    Ok(NearTieStats {
        rel_envelope,
        near_tie_gap_index: NEAR_TIE_GAP_INDEX,
        near_tie_predicate: NEAR_TIE_PREDICATE,
        per_stream,
        committed_total,
        mismatches_total,
        near_tie_positions_total,
        near_tie_mismatches_total,
        non_near_tie_mismatches_total,
        budget_per_thousand,
        max_stream_mismatch_per_thousand,
        headroom_per_thousand: (budget_per_thousand as f64) - max_stream_mismatch_per_thousand,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::COHORT_TOKEN_TOLERANCE_PER_THOUSAND;

    const ENV: f64 = 0.05;

    /// One position. `argmax_offset != 0` makes it a MISMATCH; `top2` is the reference's
    /// top-1→top-2 relative gap (< ENV ⇒ near-tie); `committed_gap` is the committed token's own
    /// relative gap.
    fn pos(token: i64, argmax_offset: i64, top2: f64, committed_gap: f64) -> PositionGaps {
        PositionGaps {
            committed_token: token,
            sequential_argmax: token + argmax_offset,
            top2_relative_gap: top2,
            committed_relative_gap: committed_gap,
        }
    }

    #[test]
    fn overlap_splits_mismatches_into_near_tie_and_genuine_divergence() {
        // Slot 0, 8 positions, hand-built so every cell of the 2x2 (mismatch x near-tie) is
        // populated with a DIFFERENT count — a swapped predicate or a swapped overlap produces
        // different numbers, not a coincidence.
        //   near-tie & mismatch    : 3   (gaps 0.01, 0.02, 0.03 — all <= 0.05)
        //   near-tie & match       : 2   (gaps 0.00, 0.04)
        //   not-near-tie & mismatch: 2   (gaps 0.50, 0.90)
        //   not-near-tie & match   : 1   (gap 0.70)
        let positions = vec![
            pos(10, 1, 0.01, 0.11),
            pos(11, 1, 0.02, 0.33),
            pos(12, 1, 0.03, 0.22),
            pos(13, 0, 0.00, 0.0),
            pos(14, 0, 0.04, 0.0),
            pos(15, 1, 0.50, 0.44),
            pos(16, 1, 0.90, 0.55),
            pos(17, 0, 0.70, 0.0),
        ];
        let stats = near_tie_stats(&[positions], ENV, COHORT_TOKEN_TOLERANCE_PER_THOUSAND)
            .expect("well-shaped");
        let s = &stats.per_stream[0];
        assert_eq!(s.slot, 0);
        assert_eq!(s.committed, 8);
        assert_eq!(s.mismatches, 5);
        assert_eq!(
            s.near_tie_positions, 5,
            "near-tie is a REFERENCE property: 3 mismatched + 2 matched positions are near-ties"
        );
        assert_eq!(s.near_tie_mismatches, 3, "the OVERLAP");
        assert_eq!(
            s.non_near_tie_mismatches, 2,
            "GENUINE divergence: the reference was confident and the candidate still diverged"
        );
        assert_eq!(
            s.near_tie_mismatches + s.non_near_tie_mismatches,
            s.mismatches,
            "the two halves must partition the mismatches"
        );
        // min/median of committed_relative_gap over the MISMATCHED positions only:
        // {0.11, 0.33, 0.22, 0.44, 0.55} -> sorted {0.11,0.22,0.33,0.44,0.55}, odd count -> 0.33.
        // A bug that took the gaps over ALL positions would see the 0.0s and report min 0.0.
        assert_eq!(s.min_committed_relative_gap_on_mismatch, Some(0.11));
        assert_eq!(s.median_committed_relative_gap_on_mismatch, Some(0.33));
        // 5/8 = 625 per thousand.
        assert_eq!(s.mismatch_per_thousand, 625.0);
        assert_eq!(stats.rel_envelope, ENV);
        assert_eq!(stats.near_tie_gap_index, 1);
        assert_eq!(
            stats.near_tie_predicate,
            "ranked_relative_gaps[1] <= rel_envelope"
        );
    }

    #[test]
    fn envelope_boundary_is_inclusive_matching_within_envelope_depth() {
        // A gap EXACTLY at the envelope is a near-tie (the engine's `within_envelope_depth`
        // counts ranks at relative ratio `<=` the envelope). A gap one ulp above is not.
        let at = pos(1, 1, ENV, 0.5);
        let above = pos(2, 1, ENV + 1e-12, 0.5);
        let below = pos(3, 1, ENV - 1e-12, 0.5);
        let stats = near_tie_stats(
            &[vec![at, above, below]],
            ENV,
            COHORT_TOKEN_TOLERANCE_PER_THOUSAND,
        )
        .unwrap();
        assert_eq!(
            stats.per_stream[0].near_tie_positions, 2,
            "`<=` (inclusive): the AT-envelope position counts, the above-envelope one does not"
        );
    }

    #[test]
    fn zero_mismatch_stream_omits_the_gap_statistics_rather_than_fabricating_zero() {
        // Every position matches. min/median have no support — they must be absent, NOT 0.0
        // (which would read as "the committed token was the reference top-1 at a mismatch").
        let positions: Vec<PositionGaps> = (0..4).map(|i| pos(100 + i, 0, 0.01, 0.0)).collect();
        let stats = near_tie_stats(&[positions], ENV, COHORT_TOKEN_TOLERANCE_PER_THOUSAND).unwrap();
        let s = &stats.per_stream[0];
        assert_eq!(s.mismatches, 0);
        assert_eq!(
            s.near_tie_positions, 4,
            "still near-ties — just no mismatches"
        );
        assert_eq!(s.min_committed_relative_gap_on_mismatch, None);
        assert_eq!(s.median_committed_relative_gap_on_mismatch, None);
        assert_eq!(s.mismatch_per_thousand, 0.0);
        // Absent statistics are OMITTED from the seal, never null-or-zero.
        let json = serde_json::to_value(s).unwrap();
        assert!(json.get("min_committed_relative_gap_on_mismatch").is_none());
        assert!(json
            .get("median_committed_relative_gap_on_mismatch")
            .is_none());
    }

    #[test]
    fn headroom_is_the_worst_stream_not_the_cohort_average() {
        // Slot 2 at 12/128 = 93.75 per-thousand (the honest-drama 9.375%); every other slot clean.
        // A cohort AVERAGE would read 12/(4*128) = 23.4 and hide the creep entirely.
        let mut streams: Vec<Vec<PositionGaps>> = Vec::new();
        for slot in 0..4usize {
            let mismatches = if slot == 2 { 12 } else { 0 };
            let base = 1000 * (slot as i64 + 1);
            streams.push(
                (0..128)
                    .map(|i| {
                        let offset = if (i as usize) < mismatches { 1 } else { 0 };
                        pos(base + i, offset, 0.5, 0.4)
                    })
                    .collect(),
            );
        }
        let stats = near_tie_stats(&streams, ENV, COHORT_TOKEN_TOLERANCE_PER_THOUSAND).unwrap();
        assert_eq!(stats.per_stream[2].mismatches, 12);
        assert_eq!(stats.max_stream_mismatch_per_thousand, 93.75);
        assert_eq!(
            stats.headroom_per_thousand,
            100.0 - 93.75,
            "6.25 per-thousand of headroom left against the 100/1000 budget"
        );
        assert_eq!(stats.committed_total, 4 * 128);
        assert_eq!(stats.mismatches_total, 12);
        assert_eq!(
            stats.near_tie_positions_total, 0,
            "every gap here is 0.5, far outside the 0.05 envelope"
        );
        assert_eq!(
            stats.non_near_tie_mismatches_total, 12,
            "all 12 are GENUINE divergence — the rounding channel explains none of them"
        );
        // The cohort AVERAGE is far under the bar — proving the headroom stat did not average.
        let average = (stats.mismatches_total as f64) * 1000.0 / (stats.committed_total as f64);
        assert!(average < 25.0, "the average hides the creep: {average}");
    }

    #[test]
    fn even_count_median_averages_the_two_middles() {
        // 4 mismatches with committed gaps {0.1, 0.2, 0.3, 0.4} -> even-n median 0.25.
        let positions = vec![
            pos(1, 1, 0.9, 0.4),
            pos(2, 1, 0.9, 0.1),
            pos(3, 1, 0.9, 0.3),
            pos(4, 1, 0.9, 0.2),
        ];
        let stats = near_tie_stats(&[positions], ENV, COHORT_TOKEN_TOLERANCE_PER_THOUSAND).unwrap();
        let s = &stats.per_stream[0];
        assert_eq!(s.min_committed_relative_gap_on_mismatch, Some(0.1));
        assert_eq!(
            s.median_committed_relative_gap_on_mismatch,
            Some(0.25),
            "even-n rule: mean of the two central order statistics"
        );
    }

    #[test]
    fn later_slot_stats_are_computed_on_their_own_positions() {
        // Mutation guard against "reads slot 0 and multiplies": each slot gets a DIFFERENT
        // mismatch count and a DIFFERENT near-tie count.
        let mut streams: Vec<Vec<PositionGaps>> = Vec::new();
        for slot in 0..3usize {
            let base = 10_000 * (slot as i64 + 1);
            streams.push(
                (0..10)
                    .map(|i| {
                        // slot s: s+1 mismatches, and 2*s+1 near-ties.
                        let offset = if (i as usize) < slot + 1 { 1 } else { 0 };
                        let top2 = if (i as usize) < 2 * slot + 1 {
                            0.01
                        } else {
                            0.9
                        };
                        pos(base + i, offset, top2, 0.6)
                    })
                    .collect(),
            );
        }
        let stats = near_tie_stats(&streams, ENV, COHORT_TOKEN_TOLERANCE_PER_THOUSAND).unwrap();
        for (slot, s) in stats.per_stream.iter().enumerate() {
            assert_eq!(s.slot, slot);
            assert_eq!(
                s.mismatches,
                slot + 1,
                "slot {slot} judged on its OWN positions"
            );
            assert_eq!(s.near_tie_positions, 2 * slot + 1);
        }
        assert_eq!(stats.mismatches_total, 1 + 2 + 3);
        assert_eq!(stats.near_tie_positions_total, 1 + 3 + 5);
    }

    #[test]
    fn structural_faults_are_named_never_silently_zeroed() {
        assert_eq!(
            near_tie_stats(&[], ENV, COHORT_TOKEN_TOLERANCE_PER_THOUSAND).unwrap_err(),
            NearTieError::NoStreams
        );
        assert_eq!(
            near_tie_stats(&[vec![]], ENV, COHORT_TOKEN_TOLERANCE_PER_THOUSAND).unwrap_err(),
            NearTieError::EmptyStream { slot: 0 }
        );
        let bad = near_tie_stats(
            &[vec![pos(1, 0, 0.0, 0.0)]],
            f64::NAN,
            COHORT_TOKEN_TOLERANCE_PER_THOUSAND,
        )
        .unwrap_err();
        assert!(matches!(bad, NearTieError::UnusableEnvelope { .. }));
        assert_eq!(
            near_tie_stats(
                &[vec![pos(1, 0, 0.0, 0.0)]],
                -0.1,
                COHORT_TOKEN_TOLERANCE_PER_THOUSAND
            )
            .unwrap_err(),
            NearTieError::UnusableEnvelope { rel_envelope: -0.1 }
        );
    }
}
