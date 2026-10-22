//! Protocol v1.1 oracle-verified free-run timed decode: the scoring-series identity, the
//! §2.6 consistency TRIPLE, and the AUDIT (`audit_spec_*`) derived metrics.
//!
//! This is the judge-less, scoring-adjacent half of the benchd v1.1 implementation
//! (`cudafast-engine/docs/PROTOCOL-v1.1.md`, SIGNED 2026-08-17 incl. Amendment 4). The
//! wire-driving half (issuing `free_decode_begin` / `free_decode_run`, oracle-verifying
//! every committed token, timing the wall clock) lives in `bench-runner`; this module owns
//! the pure, GPU-free rules it depends on so they are unit-testable without a transport:
//!
//! - the **series tag** (§5): a v1.1 free-run number is a NEW SERIES and MUST NEVER be
//!   compared to a v1 teacher-forced number ([`timed_modes_comparable`]);
//! - the **consistency TRIPLE** (§2.6): `R == acceptance_lengths.len()`,
//!   `sum(acceptance_lengths) == N`, `completed_work == R + 1`, plus
//!   `committed_total == N == tokens.len()` and `drafted_total >= accepted_total`
//!   ([`verify_consistency`]);
//! - the **AUDIT** derived metrics (§3): the flat `audit_spec_*` family, computed from the
//!   engine's self-reported acceptance counters, all explicitly NON-scored ([`FreeRunAudit`]).
//!
//! # v1.2 — the COHORT (batched) generalization
//!
//! The batch-8 design brief moves the unit of measurement from the STREAM to the COHORT: B streams,
//! one window, one number. This module gains the batched twins —
//! [`CohortFreeRunResponse`] / [`verify_cohort_consistency`] / [`CohortFreeRunAudit`] — and the
//! per-batch series tag ([`timed_mode_batched_free_run`]). Three properties are worth stating
//! rather than leaving to be re-derived:
//!
//! - **`acceptance_lengths` stays a SINGLE vector.** CBv2 commits ONE COMMON WIDTH per round, taken
//!   as the minimum across rows, so per-round committed counts are identical across streams by
//!   construction. `sum(acceptance_lengths) == N` therefore holds unchanged in FORM at B > 1.
//! - **`completed_work` stays a SCALAR `R + 1`.** A round is one engine forward regardless of B; the
//!   counter counts forwards, never stream-rounds.
//! - **The scalar equations are single-sourced** ([`check_shared_invariants`]), so `B = 1` through
//!   the cohort verifier runs literally the same checks as the v1.1 verifier — the batched path
//!   cannot drift into a looser regime.

/// Series tag for a v1.1 oracle-verified free-run timed score (§5). Every v1.1 score record
/// MUST carry this so no downstream aggregation, leaderboard, or regression gate silently
/// mixes it with the v1 teacher-forced series.
pub const TIMED_MODE_FREE_RUN_V1_1: &str = "free_run_v1_1";

/// Series tag for a v1 teacher-forced timed score. The counterpart of
/// [`TIMED_MODE_FREE_RUN_V1_1`]; the two series measure different physical quantities
/// (forced-single-step vs free-run-with-verify) and are never comparable.
pub const TIMED_MODE_TEACHER_FORCED_V1: &str = "teacher_forced_v1";

/// Prefix of the v1.2 BATCHED (cohort) free-run series tag; the cohort width B is appended
/// ([`timed_mode_batched_free_run`]).
///
/// D5 (batch-8 design brief) — calibration is per `(series, track, window, BATCH SIZE)`, and the
/// mechanism is this tag rather than a new `batch_size` field with a new fence. Because
/// [`timed_modes_comparable`] is plain string equality and `enforce_calibration_series_fence` runs
/// on the calibration PRE-READ before any measuring, encoding B in the tag inherits the ENTIRE
/// cross-batch comparison prohibition with ZERO new gate code: a B=1 baseline can never band a B=8
/// run, and the two can never be ranked side by side. "Seconds per token is not comparable across
/// token counts" is benchd's own argument for the window check; it is equally not comparable across
/// cohort widths, because the denominator is a physically different arrangement of work.
pub const TIMED_MODE_BATCHED_FREE_RUN_PREFIX: &str = "batched_free_run_v1_2_b";

/// The series tag for the ONE ruled batch point, B = 8 (`"batched_free_run_v1_2_b8"`).
/// Equal to `timed_mode_batched_free_run(8)`; named so call sites and fixtures can refer to the
/// scored series by symbol.
pub const TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8: &str = "batched_free_run_v1_2_b8";

/// The v1.2 batched free-run series tag for cohort width `batch_size` (D5). One series per batch
/// point, by construction: `b1` and `b8` are different strings, so the existing series fence
/// refuses to compare them without a single line of new gate logic.
pub fn timed_mode_batched_free_run(batch_size: u32) -> String {
    format!("{TIMED_MODE_BATCHED_FREE_RUN_PREFIX}{batch_size}")
}

/// The comparability rule (§5): two timed numbers are comparable ONLY if they carry the same
/// series tag. A v1.1 free-run number and a v1 teacher-forced number are different measurement
/// regimes — a v1.1 `decode_seconds_per_token` can be dramatically lower purely because the
/// regime lets MTP acceptance count, which is a category change, not a speedup. Baselines,
/// speedup floors, and acceptance bands are all per-series; a v1.1 run is gated only against
/// v1.1 calibration.
///
/// WHERE THIS IS ENFORCED (#105 cycle-5): `benchd::measure_job::enforce_calibration_series_fence`
/// calls this predicate on the BASELINE_CALIBRATION pre-read, before any measuring and therefore
/// before any banding — a calibration whose series is not comparable to the run's is die-6 and
/// never reaches `evaluate_serial_band`. That is the production caller; the rule is checked on the
/// path that gates a run, not merely stamped on records. Named here so the claim stays falsifiable:
/// if the fence is ever removed, this paragraph is wrong and the tag is decoration again.
pub fn timed_modes_comparable(a: &str, b: &str) -> bool {
    a == b
}

/// The parsed `free_decode_run` response counters benchd checks (PROTOCOL-v1.1.md §2.1/§3).
/// `tokens_len` is the length of the committed `tokens[]` array the runner received; the
/// per-token oracle exact-match is done in the runner, this type carries only the counts the
/// §2.6 triple cross-checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeRunResponse {
    /// Number of committed token IDs the `free_decode_run` response carried.
    pub tokens_len: usize,
    /// Per verify-round committed count (AUDIT, persisted verbatim). Length = round count R.
    pub acceptance_lengths: Vec<u32>,
    /// Total draft tokens proposed across all rounds (self-reported, `>= accepted_total`).
    pub drafted_total: u64,
    /// Total drafts that passed internal verification and were committed (self-reported).
    pub accepted_total: u64,
    /// Total committed tokens; MUST equal N and `tokens_len`.
    pub committed_total: u64,
    /// OPTIONAL spec-decode AUDIT (`WorkerResponse::verify_replay_disagreements`): the number of
    /// verify rounds whose BATCHED verify argmax differed from the ONE-ROW replay argmax of the
    /// same position. `None` when the engine did not report it — NOT REPORTED, which is not the
    /// same claim as `0`.
    pub verify_replay_disagreements: Option<u64>,
    /// OPTIONAL spec-decode AUDIT: how the engine verified the drafts (see [`VerificationReport`]).
    /// `None` when the engine did not report it — NOT REPORTED.
    pub verification: Option<VerificationReport>,
}

/// The engine's self-reported VERIFY PATH for a free-run window (`verification_mode`,
/// `rectangular_verification_rounds`, `serial_verification_rounds` on the wire). `rectangular` is
/// one target forward over the whole 1+k candidate window with the recurrent state captured per
/// position and rolled back to the accepted one; `serial` is one target forward per candidate
/// token, the fallback oracle that costs (1+k) forwards per round. AUDIT-only, never scored; sealed
/// so an artifact can state which path produced its decode number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationReport {
    /// `"rectangular"` or `"serial"`; the runner replaces other wire text with a stable seal.
    pub mode: String,
    /// Rounds that ran the rectangular path, when reported.
    pub rectangular_rounds: Option<u64>,
    /// Rounds that fell back to the serial oracle, when reported.
    pub serial_rounds: Option<u64>,
}

impl VerificationReport {
    /// Build from the three optional wire fields. `None` exactly when the engine reported no
    /// `verification_mode` — the counters alone do not name a path.
    pub fn from_wire(
        mode: Option<&str>,
        rectangular_rounds: Option<u64>,
        serial_rounds: Option<u64>,
    ) -> Option<Self> {
        mode.map(|m| Self {
            mode: m.to_string(),
            rectangular_rounds,
            serial_rounds,
        })
    }
}

/// A way the §2.6 consistency TRIPLE (and its companion count invariants) can fail. Each is a
/// hard, fail-closed failure of the free-run phase: a doctored acceptance histogram is
/// internally falsifiable against counters benchd already trusts (the anti-cheat crux, §4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FreeRunConsistencyError {
    /// `tokens.len() != N` — the response did not carry exactly N committed token IDs.
    TokenCount { expected: usize, got: usize },
    /// `committed_total != N` (§2.4).
    CommittedTotal { n: u32, committed_total: u64 },
    /// `sum(acceptance_lengths) != N` — the per-round splits do not sum to the N verified
    /// tokens (triple equation 2).
    AcceptanceSum { n: u32, sum: u64 },
    /// `completed_work != R + 1` — the verify-round forward counter (seed + R rounds) is
    /// inconsistent with the reported round count (triple equation 3).
    RoundCounter { rounds: usize, completed_work: i64 },
    /// `drafted_total < accepted_total` — an impossible acceptance claim (§2.4 invariant).
    DraftedLessThanAccepted { drafted: u64, accepted: u64 },
    /// `verify_replay_disagreements > drafted_total - accepted_total` — an impossible disagreement
    /// claim. ONLY a REJECTING round runs the one-row replay that can disagree, and the rejected
    /// drafts are exactly `drafted_total - accepted_total`, so a larger count describes rounds that
    /// did not happen. Reported ⇒ checked; absent ⇒ nothing to check.
    VerifyReplayDisagreementsExceedRejected {
        disagreements: u64,
        drafted: u64,
        accepted: u64,
    },
    /// v1.2 COHORT: `tokens_by_stream.len() != B` — the response did not carry exactly B streams.
    CohortWidth { batch_size: u32, got: usize },
    /// v1.2 COHORT: one slot's committed stream is not exactly N tokens long.
    CohortStreamTokenCount {
        slot: usize,
        expected: usize,
        got: usize,
    },
    /// v1.2 COHORT: `committed_total != B * N` — the cohort-sum committed count is wrong.
    CohortCommittedTotal { expected: u64, committed_total: u64 },
    /// v1.2 COHORT: the self-reported `rounds` disagrees with `acceptance_lengths.len()`. The
    /// redundancy is deliberate — a response that contradicts itself is refused, not reconciled.
    CohortRoundsDisagree {
        rounds: u32,
        acceptance_lengths: usize,
    },
    /// v1.2 COHORT: `natural_accepted_by_stream.len() != B`.
    CohortNaturalWidth { batch_size: u32, got: usize },
    /// v1.2 COHORT: one slot's natural accept-walk vector is not exactly R long.
    CohortNaturalRoundCount {
        slot: usize,
        expected: usize,
        got: usize,
    },
    /// v1.2 COHORT: a row's PRE-`min` natural accept-walk is SHORTER than the common committed
    /// width that round. The cohort commits `min` across rows, so no row can have walked less than
    /// what was committed — this is the cross-check that keeps the audit vectors anchored to the
    /// scored histogram instead of being free-floating engine prose.
    CohortNaturalBelowCommitted {
        round: usize,
        slot: usize,
        natural: u32,
        committed: u32,
    },
    /// v1.2 COHORT: `active_streams_by_round.len() != R`.
    CohortActiveStreamsLength { rounds: usize, got: usize },
    /// v1.2 COHORT: an `active_streams_by_round` entry exceeds the cohort width B (or is 0 while
    /// rounds are still being run).
    CohortActiveStreamsOutOfRange {
        round: usize,
        active: u32,
        batch_size: u32,
    },
    /// v1.2 COHORT: `active_streams_by_round` INCREASED. Under the closed fixed-N cohort (no
    /// refill, no admission mid-window) a stream can only leave, never join.
    CohortActiveStreamsIncreased {
        round: usize,
        previous: u32,
        active: u32,
    },
}

impl core::fmt::Display for FreeRunConsistencyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FreeRunConsistencyError::TokenCount { expected, got } => write!(
                f,
                "free_decode_run returned {got} committed tokens, expected N={expected}"
            ),
            FreeRunConsistencyError::CommittedTotal { n, committed_total } => write!(
                f,
                "free_decode_run committed_total {committed_total} != N {n}"
            ),
            FreeRunConsistencyError::AcceptanceSum { n, sum } => write!(
                f,
                "free_decode_run sum(acceptance_lengths) {sum} != N {n}"
            ),
            FreeRunConsistencyError::RoundCounter {
                rounds,
                completed_work,
            } => write!(
                f,
                "free-run completed_work {completed_work} != R+1 (R={rounds} rounds, expected {})",
                *rounds as i64 + 1
            ),
            FreeRunConsistencyError::DraftedLessThanAccepted { drafted, accepted } => write!(
                f,
                "free_decode_run drafted_total {drafted} < accepted_total {accepted}"
            ),
            FreeRunConsistencyError::VerifyReplayDisagreementsExceedRejected {
                disagreements,
                drafted,
                accepted,
            } => write!(
                f,
                "free_decode_run verify_replay_disagreements {disagreements} > rejected drafts {} \
                 (drafted_total {drafted} - accepted_total {accepted}); only a rejecting round can \
                 disagree",
                drafted.saturating_sub(*accepted)
            ),
            FreeRunConsistencyError::CohortWidth { batch_size, got } => write!(
                f,
                "free_decode_run returned {got} token streams, expected B={batch_size}"
            ),
            FreeRunConsistencyError::CohortStreamTokenCount {
                slot,
                expected,
                got,
            } => write!(
                f,
                "free_decode_run stream {slot} returned {got} committed tokens, expected N={expected}"
            ),
            FreeRunConsistencyError::CohortCommittedTotal {
                expected,
                committed_total,
            } => write!(
                f,
                "free_decode_run committed_total {committed_total} != B*N {expected}"
            ),
            FreeRunConsistencyError::CohortRoundsDisagree {
                rounds,
                acceptance_lengths,
            } => write!(
                f,
                "free_decode_run rounds {rounds} != acceptance_lengths.len() {acceptance_lengths}"
            ),
            FreeRunConsistencyError::CohortNaturalWidth { batch_size, got } => write!(
                f,
                "free_decode_run natural_accepted_by_stream has {got} streams, expected B={batch_size}"
            ),
            FreeRunConsistencyError::CohortNaturalRoundCount {
                slot,
                expected,
                got,
            } => write!(
                f,
                "free_decode_run natural_accepted_by_stream[{slot}] has {got} rounds, expected R={expected}"
            ),
            FreeRunConsistencyError::CohortNaturalBelowCommitted {
                round,
                slot,
                natural,
                committed,
            } => write!(
                f,
                "free_decode_run round {round} committed a common width of {committed} but stream \
                 {slot} reports a natural accept walk of only {natural} (the committed width is the \
                 MINIMUM across rows, so no row can be below it)"
            ),
            FreeRunConsistencyError::CohortActiveStreamsLength { rounds, got } => write!(
                f,
                "free_decode_run active_streams_by_round has {got} entries, expected R={rounds}"
            ),
            FreeRunConsistencyError::CohortActiveStreamsOutOfRange {
                round,
                active,
                batch_size,
            } => write!(
                f,
                "free_decode_run active_streams_by_round[{round}] = {active}, outside 1..=B \
                 (B={batch_size})"
            ),
            FreeRunConsistencyError::CohortActiveStreamsIncreased {
                round,
                previous,
                active,
            } => write!(
                f,
                "free_decode_run active_streams_by_round rose from {previous} to {active} at round \
                 {round}; the cohort is CLOSED (fixed N, no refill), so a stream can only leave"
            ),
        }
    }
}

impl std::error::Error for FreeRunConsistencyError {}

/// Enforce the §2.6 consistency TRIPLE plus the §2.4 count invariants on a `free_decode_run`
/// response, returning the [`FreeRunAudit`] on success.
///
/// `n` is the requested committed-token count (benchd's `free_decode_run.count`), and
/// `completed_work` is the counter the engine reported on the phase-close `phase_diagnostics`.
/// All checks are all-or-nothing (fail-closed): because N is fixed by benchd's request and
/// every committed token is exact-matched against the golden elsewhere, the three triple
/// equations are mutually cross-checking, so a fabricated `acceptance_lengths` cannot pass
/// without breaking one of them (§4, closed lever 3).
pub fn verify_consistency(
    resp: &FreeRunResponse,
    n: u32,
    completed_work: i64,
) -> Result<FreeRunAudit, FreeRunConsistencyError> {
    // §2.4: committed_total MUST equal N and tokens.len().
    if resp.tokens_len != n as usize {
        return Err(FreeRunConsistencyError::TokenCount {
            expected: n as usize,
            got: resp.tokens_len,
        });
    }
    if resp.committed_total != n as u64 {
        return Err(FreeRunConsistencyError::CommittedTotal {
            n,
            committed_total: resp.committed_total,
        });
    }
    check_shared_invariants(
        &resp.acceptance_lengths,
        n,
        resp.drafted_total,
        resp.accepted_total,
        completed_work,
    )?;
    check_verify_replay_disagreements(
        resp.verify_replay_disagreements,
        resp.drafted_total,
        resp.accepted_total,
    )?;
    Ok(FreeRunAudit {
        acceptance_lengths: resp.acceptance_lengths.clone(),
        drafted_total: resp.drafted_total,
        accepted_total: resp.accepted_total,
        verified_token_count: n as usize,
        verify_replay_disagreements: resp.verify_replay_disagreements,
        verification: resp.verification.clone(),
    })
}

/// THE BOUND ON `verify_replay_disagreements`, checked wherever a [`FreeRunAudit`] is built — the
/// ONLY way the sealed `spec_verify_replay_disagreements` key can come into existence — so an
/// impossible count can never reach a sealed score.
///
/// The engine's depth-1 MTP cycle runs its one-row replay ONLY on a REJECTING round, and the
/// rejected drafts over the window are exactly `drafted_total - accepted_total` (the sibling
/// invariant [`check_shared_invariants`] has already refused `drafted < accepted`, so the
/// subtraction is well-defined here). A count larger than that describes rounds that did not
/// happen: an engine defect, refused by name rather than sealed.
///
/// ABSENT IS NOT ZERO. `None` means the engine reported nothing, so there is nothing to check and
/// nothing to seal — a pre-field engine is unaffected.
fn check_verify_replay_disagreements(
    disagreements: Option<u64>,
    drafted_total: u64,
    accepted_total: u64,
) -> Result<(), FreeRunConsistencyError> {
    let Some(disagreements) = disagreements else {
        return Ok(());
    };
    if disagreements > drafted_total.saturating_sub(accepted_total) {
        return Err(
            FreeRunConsistencyError::VerifyReplayDisagreementsExceedRejected {
                disagreements,
                drafted: drafted_total,
                accepted: accepted_total,
            },
        );
    }
    Ok(())
}

/// The equations the single-stream (v1.1) and cohort (v1.2) verifiers share, in ONE place, so the
/// batched path cannot quietly become a second, looser regime:
///
/// - `drafted_total >= accepted_total` (§2.4);
/// - `sum(acceptance_lengths) == N` — the PER-STREAM committed budget. Unchanged in FORM at B > 1
///   because the cohort commits one COMMON width per round (the minimum across rows), so
///   `acceptance_lengths` stays a single vector whose entries every stream shares;
/// - `completed_work == R + 1` — the seed forward plus R verify rounds. A round is ONE ENGINE
///   FORWARD regardless of B, so this counter does NOT scale with the cohort width.
///
/// Returns R (`acceptance_lengths.len()`) on success.
fn check_shared_invariants(
    acceptance_lengths: &[u32],
    n: u32,
    drafted_total: u64,
    accepted_total: u64,
    completed_work: i64,
) -> Result<usize, FreeRunConsistencyError> {
    if drafted_total < accepted_total {
        return Err(FreeRunConsistencyError::DraftedLessThanAccepted {
            drafted: drafted_total,
            accepted: accepted_total,
        });
    }
    let sum: u64 = acceptance_lengths.iter().map(|&x| x as u64).sum();
    if sum != n as u64 {
        return Err(FreeRunConsistencyError::AcceptanceSum { n, sum });
    }
    // R == acceptance_lengths.len() is definitional here — R IS the array length — so the
    // substantive forward-counter cross-check is completed_work == R+1.
    let rounds = acceptance_lengths.len();
    if completed_work != rounds as i64 + 1 {
        return Err(FreeRunConsistencyError::RoundCounter {
            rounds,
            completed_work,
        });
    }
    Ok(rounds)
}

/// The AUDIT view of a verified free-run decode phase (§3). Every field here is
/// **informational and NEVER scored** — the wall clock benchd measures is the only input to
/// the score. Two of the derived metrics (`verified_token_count` = N, `rounds` = R) are
/// externally anchored; the acceptance-rate / drafted-total family is purely self-reported and
/// retained only for engine debugging and cross-run trend analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct FreeRunAudit {
    acceptance_lengths: Vec<u32>,
    drafted_total: u64,
    accepted_total: u64,
    verified_token_count: usize,
    verify_replay_disagreements: Option<u64>,
    verification: Option<VerificationReport>,
}

impl FreeRunAudit {
    /// The raw per-round `acceptance_lengths[]`, persisted VERBATIM into the run's metrics
    /// (RULED OQ4 — not just the aggregates), so cross-run analysis and the §2.6 triple have
    /// the full histogram. The single-leg seal is `ScoreMetrics::acceptance_lengths`
    /// (`benchd::score`), written from this accessor; the paired flow persists it through its own
    /// results record.
    pub fn acceptance_lengths(&self) -> &[u32] {
        &self.acceptance_lengths
    }

    /// Total draft tokens the engine proposed across all rounds (self-reported, `>= accepted`).
    /// AUDIT-only, never scored; sealed as `spec_drafted_total`.
    pub fn drafted_total(&self) -> u64 {
        self.drafted_total
    }

    /// Total draft tokens the target ACCEPTED across all rounds (self-reported). AUDIT-only, never
    /// scored; sealed as `spec_accepted_total`.
    pub fn accepted_total(&self) -> u64 {
        self.accepted_total
    }

    /// The engine's `verify_replay_disagreements`, when it reported one. AUDIT-only, never scored;
    /// sealed as `spec_verify_replay_disagreements`. `None` means NOT REPORTED — never `0` by
    /// default, because "the engine measured none" and "the engine cannot say" are different facts.
    /// Bounded by the rejected drafts at construction
    /// ([`FreeRunConsistencyError::VerifyReplayDisagreementsExceedRejected`]).
    pub fn verify_replay_disagreements(&self) -> Option<u64> {
        self.verify_replay_disagreements
    }

    /// The engine's self-reported verify path, when it reported one. AUDIT-only, never scored;
    /// sealed as `spec_verification_mode` / `spec_rectangular_verification_rounds` /
    /// `spec_serial_verification_rounds`. `None` means NOT REPORTED.
    pub fn verification(&self) -> Option<&VerificationReport> {
        self.verification.as_ref()
    }

    /// N — the number of externally verified committed tokens the clock covered.
    pub fn verified_token_count(&self) -> usize {
        self.verified_token_count
    }

    /// R = number of MTP verify rounds (`acceptance_lengths.len()`).
    pub fn rounds(&self) -> usize {
        self.acceptance_lengths.len()
    }

    /// Mean tokens committed per verify round.
    pub fn mean_acceptance_length(&self) -> f64 {
        if self.acceptance_lengths.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.acceptance_lengths.iter().map(|&x| x as u64).sum();
        sum as f64 / self.acceptance_lengths.len() as f64
    }

    /// Median / p50 of `acceptance_lengths` (mean of the two middles on an even count).
    pub fn median_acceptance_length(&self) -> f64 {
        if self.acceptance_lengths.is_empty() {
            return 0.0;
        }
        let mut sorted = self.acceptance_lengths.clone();
        sorted.sort_unstable();
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 1 {
            sorted[mid] as f64
        } else {
            (sorted[mid - 1] as f64 + sorted[mid] as f64) / 2.0
        }
    }

    /// N / R — externally derivable effective committed tokens per verify round.
    pub fn effective_tokens_per_forward(&self) -> f64 {
        let r = self.rounds();
        if r == 0 {
            return 0.0;
        }
        self.verified_token_count as f64 / r as f64
    }

    /// The EFFECTIVE mean draft length benchd computes from the per-round `acceptance_lengths` it
    /// collected — NOT an engine echo (`docs/spec-config-design.md` step 3). This is the mean number
    /// of committed tokens per verify round (`mean_acceptance_length`), the drafter's realized
    /// tokens-per-forward. AUDIT-only, never scored.
    pub fn effective_mean_draft_len(&self) -> f64 {
        self.mean_acceptance_length()
    }

    /// The count of NON-DRAFTING rounds benchd computes from the per-round `acceptance_lengths`: a
    /// round that committed only the base-model fallback token (length <= 1) — the drafter produced
    /// nothing usable that round. AUDIT-only, never scored.
    pub fn non_drafting_round_count(&self) -> usize {
        self.acceptance_lengths
            .iter()
            .filter(|&&len| len <= 1)
            .count()
    }

    /// `accepted_total / drafted_total`, when `drafted_total > 0` (self-reported).
    pub fn acceptance_rate(&self) -> Option<f64> {
        if self.drafted_total == 0 {
            None
        } else {
            Some(self.accepted_total as f64 / self.drafted_total as f64)
        }
    }

    /// The flat `audit_spec_*` derived metrics (§3, RULED OQ4 — flat prefix, no nested object),
    /// as `(key, value)` pairs. All are explicitly non-scored; `audit_spec_acceptance_rate` is
    /// present only when `drafted_total > 0`. The raw per-round histogram is persisted
    /// separately via [`acceptance_lengths`](Self::acceptance_lengths).
    pub fn to_metrics(&self) -> Vec<(String, f64)> {
        let mut metrics = vec![
            (
                "audit_spec_mean_acceptance_length".to_string(),
                self.mean_acceptance_length(),
            ),
            (
                "audit_spec_median_acceptance_length".to_string(),
                self.median_acceptance_length(),
            ),
            (
                "audit_spec_verified_token_count".to_string(),
                self.verified_token_count as f64,
            ),
            ("audit_spec_rounds".to_string(), self.rounds() as f64),
            (
                "audit_spec_drafted_total".to_string(),
                self.drafted_total as f64,
            ),
            (
                "audit_spec_accepted_total".to_string(),
                self.accepted_total as f64,
            ),
            (
                "audit_spec_effective_tokens_per_forward".to_string(),
                self.effective_tokens_per_forward(),
            ),
            (
                "audit_spec_effective_mean_draft_len".to_string(),
                self.effective_mean_draft_len(),
            ),
            (
                "audit_spec_non_drafting_round_count".to_string(),
                self.non_drafting_round_count() as f64,
            ),
        ];
        if let Some(rate) = self.acceptance_rate() {
            metrics.push(("audit_spec_acceptance_rate".to_string(), rate));
        }
        metrics
    }
}

/// The parsed COHORT (`v1.2`) `free_decode_run` response counters benchd checks. The batched
/// counterpart of [`FreeRunResponse`]: the cohort of `batch_size` streams is ONE measurement, so
/// this carries the whole window's accounting rather than one stream's.
///
/// `tokens_len_by_stream` is the per-slot length of the committed `tokens_by_stream[]` arrays the
/// runner received; the per-token oracle exact-match is done in the runner, this type carries only
/// the counts the §2.6 QUADRUPLE cross-checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CohortFreeRunResponse {
    /// The cohort width B the engine echoed (already compared to the request by the runner).
    pub batch_size: u32,
    /// Per slot, the number of committed token IDs that slot's stream carried. Length must be B,
    /// each entry must be N.
    pub tokens_len_by_stream: Vec<usize>,
    /// The per-round COMMON committed width (the minimum across rows), length R. A single vector
    /// even at B > 1 — that is the §2.4 simplification, not an omission.
    pub acceptance_lengths: Vec<u32>,
    /// Per slot, that row's PRE-`min` natural accept-walk length per round (B x R). AUDIT-only.
    pub natural_accepted_by_stream: Vec<Vec<u32>>,
    /// Streams still generating at each round, length R. Non-increasing under the closed cohort.
    pub active_streams_by_round: Vec<u32>,
    /// The engine's self-reported round count; cross-checked against `acceptance_lengths.len()`.
    pub rounds: u32,
    /// Cohort-sum draft tokens proposed across all rounds and streams (`>= accepted_total`).
    pub drafted_total: u64,
    /// Cohort-sum drafts that passed internal verification and were committed.
    pub accepted_total: u64,
    /// Cohort-sum committed tokens; MUST equal `B * N`.
    pub committed_total: u64,
    /// The engine's sealed depth-clamp reason histogram over the window. AUDIT-only.
    pub depth_clamp_reasons: std::collections::BTreeMap<String, u64>,
}

/// Enforce the COHORT consistency QUADRUPLE on a batched `free_decode_run` response, returning the
/// [`CohortFreeRunAudit`] on success.
///
/// `n` is the requested PER-STREAM committed-token count (`free_decode_run.count`) and
/// `completed_work` is the phase-close counter. The four groups, generalizing the §2.6 triple:
///
/// 1. `tokens_len_by_stream.len() == B` and every entry `== N` (the cohort really returned the
///    B x N committed rectangle benchd asked for);
/// 2. `committed_total == B * N` (the cohort-sum accounting agrees with the rectangle);
/// 3. `sum(acceptance_lengths) == N` — unchanged in FORM from the single-stream case — plus
///    `natural_accepted_by_stream` is B x R and no row's natural walk is below the committed
///    common width that round;
/// 4. `completed_work == R + 1` (SCALAR: a round is one engine forward regardless of B), plus
///    `rounds == R`, `active_streams_by_round` is length R, within `1..=B`, and non-increasing;
///    plus `drafted_total >= accepted_total`.
///
/// All checks are all-or-nothing (fail-closed). The anti-cheat property carries over unchanged:
/// because N and B are fixed by benchd's request and every committed token is exact-matched
/// against the golden elsewhere, the equations are mutually cross-checking, so a fabricated
/// histogram — per-round OR per-stream — cannot pass without breaking one of them.
pub fn verify_cohort_consistency(
    resp: &CohortFreeRunResponse,
    n: u32,
    completed_work: i64,
) -> Result<CohortFreeRunAudit, FreeRunConsistencyError> {
    let batch_size = resp.batch_size;
    // (1) the B x N committed rectangle.
    if resp.tokens_len_by_stream.len() != batch_size as usize {
        return Err(FreeRunConsistencyError::CohortWidth {
            batch_size,
            got: resp.tokens_len_by_stream.len(),
        });
    }
    for (slot, &len) in resp.tokens_len_by_stream.iter().enumerate() {
        if len != n as usize {
            return Err(FreeRunConsistencyError::CohortStreamTokenCount {
                slot,
                expected: n as usize,
                got: len,
            });
        }
    }
    // (2) cohort-sum committed accounting.
    let expected_committed = batch_size as u64 * n as u64;
    if resp.committed_total != expected_committed {
        return Err(FreeRunConsistencyError::CohortCommittedTotal {
            expected: expected_committed,
            committed_total: resp.committed_total,
        });
    }
    // (3)/(4) the shared scalar equations — the SAME code the v1.1 verifier runs.
    let rounds = check_shared_invariants(
        &resp.acceptance_lengths,
        n,
        resp.drafted_total,
        resp.accepted_total,
        completed_work,
    )?;
    if resp.rounds as usize != rounds {
        return Err(FreeRunConsistencyError::CohortRoundsDisagree {
            rounds: resp.rounds,
            acceptance_lengths: rounds,
        });
    }
    // (3, cohort half) the natural accept-walk rectangle, anchored to the committed widths.
    if resp.natural_accepted_by_stream.len() != batch_size as usize {
        return Err(FreeRunConsistencyError::CohortNaturalWidth {
            batch_size,
            got: resp.natural_accepted_by_stream.len(),
        });
    }
    for (slot, natural) in resp.natural_accepted_by_stream.iter().enumerate() {
        if natural.len() != rounds {
            return Err(FreeRunConsistencyError::CohortNaturalRoundCount {
                slot,
                expected: rounds,
                got: natural.len(),
            });
        }
        for (round, (&walked, &committed)) in natural
            .iter()
            .zip(resp.acceptance_lengths.iter())
            .enumerate()
        {
            // The committed width is `min` across rows, so `walked >= committed` always. The
            // reverse (min over rows == committed) is deliberately NOT asserted: the final round
            // can be truncated by the per-stream budget N, which legitimately commits fewer tokens
            // than every row had walked. That case is surfaced as an AUDIT count
            // (`audit_cohort_budget_truncated_round_count`), not as a refusal.
            if walked < committed {
                return Err(FreeRunConsistencyError::CohortNaturalBelowCommitted {
                    round,
                    slot,
                    natural: walked,
                    committed,
                });
            }
        }
    }
    // (4, cohort half) the closed-cohort tail.
    if resp.active_streams_by_round.len() != rounds {
        return Err(FreeRunConsistencyError::CohortActiveStreamsLength {
            rounds,
            got: resp.active_streams_by_round.len(),
        });
    }
    let mut previous: Option<u32> = None;
    for (round, &active) in resp.active_streams_by_round.iter().enumerate() {
        if active == 0 || active > batch_size {
            return Err(FreeRunConsistencyError::CohortActiveStreamsOutOfRange {
                round,
                active,
                batch_size,
            });
        }
        if let Some(previous) = previous {
            if active > previous {
                return Err(FreeRunConsistencyError::CohortActiveStreamsIncreased {
                    round,
                    previous,
                    active,
                });
            }
        }
        previous = Some(active);
    }
    Ok(CohortFreeRunAudit {
        base: FreeRunAudit {
            acceptance_lengths: resp.acceptance_lengths.clone(),
            drafted_total: resp.drafted_total,
            accepted_total: resp.accepted_total,
            // The per-STREAM budget: `sum(acceptance_lengths) == N` holds per stream because the
            // committed width is common. The cohort total is carried separately below.
            verified_token_count: n as usize,
            // NOT REPORTED on the cohort form: the verify/replay disagreement counter is a
            // single-stream `free_decode_run` audit, and inventing `0` here would state a
            // measurement the batched response never made.
            verify_replay_disagreements: None,
            verification: None,
        },
        batch_size,
        cohort_committed_total: expected_committed,
        natural_accepted_by_stream: resp.natural_accepted_by_stream.clone(),
        active_streams_by_round: resp.active_streams_by_round.clone(),
        depth_clamp_reasons: resp.depth_clamp_reasons.clone(),
    })
}

/// The AUDIT view of a verified COHORT free-run decode phase. Every field here is
/// **informational and NEVER scored** — the wall clock benchd measures over the batched window,
/// divided by `B * N`, is the only input to the score.
///
/// In particular the PER-STREAM and PER-ROUND vectors are SEALED DIAGNOSTICS, not samples: inside
/// one batched window a stream's throughput is a function of its cohort-mates (and, under the
/// common-width commit, the per-round committed counts are identical across streams by
/// construction), so B numbers out of one window are B correlated readings, never B independent
/// per-prompt measurements. Nothing in this type may be aggregated into a score.
#[derive(Debug, Clone, PartialEq)]
pub struct CohortFreeRunAudit {
    /// The per-round common-width statistics — the SAME [`FreeRunAudit`] the single-stream regime
    /// produces, so the `audit_spec_*` family keeps one definition across both regimes.
    base: FreeRunAudit,
    batch_size: u32,
    cohort_committed_total: u64,
    natural_accepted_by_stream: Vec<Vec<u32>>,
    active_streams_by_round: Vec<u32>,
    depth_clamp_reasons: std::collections::BTreeMap<String, u64>,
}

impl CohortFreeRunAudit {
    /// The per-round common-width AUDIT view, shared with the single-stream regime.
    pub fn base(&self) -> &FreeRunAudit {
        &self.base
    }

    /// B — the cohort width this window ran.
    pub fn batch_size(&self) -> u32 {
        self.batch_size
    }

    /// `B * N` — the total committed tokens the scored window covered (the score's divisor).
    pub fn cohort_committed_total(&self) -> u64 {
        self.cohort_committed_total
    }

    /// R — the number of verify rounds (`acceptance_lengths.len()`).
    pub fn rounds(&self) -> usize {
        self.base.rounds()
    }

    /// The raw per-stream, per-round natural accept walks (B x R), persisted VERBATIM.
    pub fn natural_accepted_by_stream(&self) -> &[Vec<u32>] {
        &self.natural_accepted_by_stream
    }

    /// The raw per-round active-stream counts (length R), persisted VERBATIM.
    pub fn active_streams_by_round(&self) -> &[u32] {
        &self.active_streams_by_round
    }

    /// The sealed depth-clamp reason histogram. Safe keys stay verbatim, and other keys use stable
    /// text seals. This is the evidence for whether the window ACTUALLY SPECULATED: a cohort
    /// clamped to depth zero (e.g.
    /// `automatic_rectangular_limit`, `tail_depth`) is a legitimate engine outcome, and the report
    /// has to be able to say so rather than presenting a target-only window as a speculative one.
    pub fn depth_clamp_reasons(&self) -> &std::collections::BTreeMap<String, u64> {
        &self.depth_clamp_reasons
    }

    /// Rounds where at least one row's natural walk EXCEEDED the committed common width — i.e. the
    /// cohort was held back by its slowest row that round. The straggler-throttling diagnostic.
    pub fn throttled_round_count(&self) -> usize {
        self.count_rounds(|committed, naturals| naturals.iter().any(|&w| w > committed))
    }

    /// Rounds where EVERY row's natural walk exceeded the committed common width — the signature of
    /// the per-stream budget N truncating the tail rather than a straggler throttling the cohort.
    pub fn budget_truncated_round_count(&self) -> usize {
        self.count_rounds(|committed, naturals| {
            !naturals.is_empty() && naturals.iter().all(|&w| w > committed)
        })
    }

    fn count_rounds(&self, predicate: impl Fn(u32, &[u32]) -> bool) -> usize {
        (0..self.rounds())
            .filter(|&round| {
                let naturals: Vec<u32> = self
                    .natural_accepted_by_stream
                    .iter()
                    .filter_map(|s| s.get(round).copied())
                    .collect();
                predicate(self.base.acceptance_lengths()[round], &naturals)
            })
            .count()
    }

    /// Mean natural accept-walk length over every (stream, round) cell. `0.0` on an empty cohort.
    pub fn mean_natural_accepted_length(&self) -> f64 {
        let (sum, count) =
            self.natural_accepted_by_stream
                .iter()
                .fold((0u64, 0usize), |(sum, count), stream| {
                    (
                        sum + stream.iter().map(|&x| x as u64).sum::<u64>(),
                        count + stream.len(),
                    )
                });
        if count == 0 {
            return 0.0;
        }
        sum as f64 / count as f64
    }

    /// The total depth-clamp events the engine reported. Reported as ONE scalar rather than one
    /// metric per reason key: the histogram keys are ENGINE-CONTROLLED strings, and the metric
    /// namespace is not a place to let an engine mint names. The full sealed histogram is available
    /// via [`depth_clamp_reasons`](Self::depth_clamp_reasons).
    pub fn depth_clamp_total(&self) -> u64 {
        self.depth_clamp_reasons.values().sum()
    }

    /// The flat metrics for a cohort window: the shared `audit_spec_*` family (per-round common
    /// width) plus the `audit_cohort_*` family. All explicitly non-scored.
    pub fn to_metrics(&self) -> Vec<(String, f64)> {
        let mut metrics = self.base.to_metrics();
        metrics.extend([
            (
                "audit_cohort_batch_size".to_string(),
                self.batch_size as f64,
            ),
            (
                "audit_cohort_committed_total".to_string(),
                self.cohort_committed_total as f64,
            ),
            (
                "audit_cohort_mean_natural_accepted_length".to_string(),
                self.mean_natural_accepted_length(),
            ),
            (
                "audit_cohort_throttled_round_count".to_string(),
                self.throttled_round_count() as f64,
            ),
            (
                "audit_cohort_budget_truncated_round_count".to_string(),
                self.budget_truncated_round_count() as f64,
            ),
            (
                "audit_cohort_min_active_streams".to_string(),
                self.active_streams_by_round
                    .iter()
                    .copied()
                    .min()
                    .unwrap_or(0) as f64,
            ),
            (
                "audit_cohort_depth_clamp_total".to_string(),
                self.depth_clamp_total() as f64,
            ),
        ]);
        metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A conformant N=4 response: R=2 rounds committing 3 then 1 token (sum=4).
    fn conformant(n: u32) -> FreeRunResponse {
        FreeRunResponse {
            tokens_len: n as usize,
            acceptance_lengths: vec![3, 1],
            drafted_total: 5,
            accepted_total: 2,
            committed_total: n as u64,
            verify_replay_disagreements: None,
            verification: None,
        }
    }

    #[test]
    fn series_never_comparable_across_regimes() {
        // §5: the machine-checked comparability rule — free-run vs teacher-forced never mix.
        assert!(timed_modes_comparable(
            TIMED_MODE_FREE_RUN_V1_1,
            TIMED_MODE_FREE_RUN_V1_1
        ));
        assert!(!timed_modes_comparable(
            TIMED_MODE_FREE_RUN_V1_1,
            TIMED_MODE_TEACHER_FORCED_V1
        ));
    }

    #[test]
    fn positive_control_triple_holds() {
        // R=2, so completed_work must be R+1 = 3.
        let audit = verify_consistency(&conformant(4), 4, 3).unwrap();
        assert_eq!(audit.rounds(), 2);
        assert_eq!(audit.verified_token_count(), 4);
        assert_eq!(audit.acceptance_lengths(), &[3, 1]);
        assert_eq!(audit.mean_acceptance_length(), 2.0);
        assert_eq!(audit.median_acceptance_length(), 2.0);
        assert_eq!(audit.effective_tokens_per_forward(), 2.0);
        assert_eq!(audit.acceptance_rate(), Some(2.0 / 5.0));
    }

    #[test]
    fn negative_control_sum_not_n() {
        // Doctored histogram: sums to 5, not N=4.
        let mut r = conformant(4);
        r.acceptance_lengths = vec![3, 2];
        let err = verify_consistency(&r, 4, 3).unwrap_err();
        assert_eq!(err, FreeRunConsistencyError::AcceptanceSum { n: 4, sum: 5 });
    }

    #[test]
    fn negative_control_completed_work_not_r_plus_1() {
        // R=2 rounds but completed_work claims 2 (should be 3).
        let err = verify_consistency(&conformant(4), 4, 2).unwrap_err();
        assert_eq!(
            err,
            FreeRunConsistencyError::RoundCounter {
                rounds: 2,
                completed_work: 2
            }
        );
    }

    #[test]
    fn negative_control_committed_total_not_n() {
        let mut r = conformant(4);
        r.committed_total = 3;
        let err = verify_consistency(&r, 4, 3).unwrap_err();
        assert_eq!(
            err,
            FreeRunConsistencyError::CommittedTotal {
                n: 4,
                committed_total: 3
            }
        );
    }

    #[test]
    fn negative_control_token_count_not_n() {
        let mut r = conformant(4);
        r.tokens_len = 3;
        let err = verify_consistency(&r, 4, 3).unwrap_err();
        assert_eq!(
            err,
            FreeRunConsistencyError::TokenCount {
                expected: 4,
                got: 3
            }
        );
    }

    #[test]
    fn negative_control_drafted_less_than_accepted() {
        let mut r = conformant(4);
        r.drafted_total = 1;
        r.accepted_total = 2;
        let err = verify_consistency(&r, 4, 3).unwrap_err();
        assert_eq!(
            err,
            FreeRunConsistencyError::DraftedLessThanAccepted {
                drafted: 1,
                accepted: 2
            }
        );
    }

    #[test]
    fn audit_metrics_are_flat_and_non_scored() {
        let audit = verify_consistency(&conformant(4), 4, 3).unwrap();
        let metrics = audit.to_metrics();
        // Every key carries the flat `audit_spec_` prefix (RULED OQ4 — no nested object).
        assert!(metrics.iter().all(|(k, _)| k.starts_with("audit_spec_")));
        let get = |name: &str| metrics.iter().find(|(k, _)| k == name).map(|(_, v)| *v);
        assert_eq!(get("audit_spec_verified_token_count"), Some(4.0));
        assert_eq!(get("audit_spec_rounds"), Some(2.0));
        assert_eq!(get("audit_spec_effective_tokens_per_forward"), Some(2.0));
        assert_eq!(get("audit_spec_drafted_total"), Some(5.0));
    }

    #[test]
    fn benchd_computes_effective_draft_metrics_from_per_round_data() {
        // Step 3: effective_mean_draft_len / non_drafting_round_count are computed by benchd from the
        // per-round acceptance_lengths it collected, NOT from an engine echo. R=3 rounds: 3,1,1.
        let r = FreeRunResponse {
            tokens_len: 5,
            acceptance_lengths: vec![3, 1, 1],
            drafted_total: 6,
            accepted_total: 2,
            committed_total: 5,
            verify_replay_disagreements: None,
            verification: None,
        };
        let audit = verify_consistency(&r, 5, 4).unwrap();
        // mean committed per round = 5/3.
        assert!((audit.effective_mean_draft_len() - 5.0 / 3.0).abs() < 1e-9);
        // two rounds committed only the fallback (length 1) => 2 non-drafting rounds.
        assert_eq!(audit.non_drafting_round_count(), 2);
        let metrics = audit.to_metrics();
        assert!(metrics
            .iter()
            .any(|(k, v)| k == "audit_spec_non_drafting_round_count" && *v == 2.0));
        assert!(metrics
            .iter()
            .any(|(k, _)| k == "audit_spec_effective_mean_draft_len"));
    }

    /// The disagreement counter rides through the audit VERBATIM when the engine reports it, and
    /// stays `None` — NOT REPORTED, never `0` — when it does not.
    #[test]
    fn verify_replay_disagreements_ride_through_the_audit_or_stay_absent() {
        // conformant(): drafted 5, accepted 2 => 3 rejected drafts.
        let audit = verify_consistency(&conformant(4), 4, 3).unwrap();
        assert_eq!(audit.verify_replay_disagreements(), None);

        let reporting = FreeRunResponse {
            verify_replay_disagreements: Some(3),
            verification: None,
            ..conformant(4)
        };
        let audit = verify_consistency(&reporting, 4, 3).unwrap();
        assert_eq!(
            audit.verify_replay_disagreements(),
            Some(3),
            "exactly as many disagreements as rejected drafts is possible, and is carried verbatim"
        );
    }

    /// AN IMPOSSIBLE COUNT REFUSES BY NAME. Only a REJECTING round runs the one-row replay that can
    /// disagree, so the count cannot exceed `drafted_total - accepted_total`. This is the negative
    /// control for the bound: one past it fails closed, and the message names the field.
    #[test]
    fn verify_replay_disagreements_above_the_rejected_drafts_refuse() {
        // drafted 5, accepted 2 => 3 rejected; 4 describes a round that did not happen.
        let impossible = FreeRunResponse {
            verify_replay_disagreements: Some(4),
            verification: None,
            ..conformant(4)
        };
        let err = verify_consistency(&impossible, 4, 3).unwrap_err();
        assert_eq!(
            err,
            FreeRunConsistencyError::VerifyReplayDisagreementsExceedRejected {
                disagreements: 4,
                drafted: 5,
                accepted: 2,
            }
        );
        assert!(err.to_string().contains("verify_replay_disagreements 4"));

        // A SERIAL leg drafted nothing, so it has no rejected drafts and cannot have disagreed.
        let serial = FreeRunResponse {
            tokens_len: 1,
            acceptance_lengths: vec![1],
            drafted_total: 0,
            accepted_total: 0,
            committed_total: 1,
            verify_replay_disagreements: Some(1),
            verification: None,
        };
        assert!(verify_consistency(&serial, 1, 2).is_err());
    }

    #[test]
    fn acceptance_rate_absent_when_no_drafts() {
        // drafted_total == 0: rate is None and the metric key is omitted.
        let r = FreeRunResponse {
            tokens_len: 1,
            acceptance_lengths: vec![1],
            drafted_total: 0,
            accepted_total: 0,
            committed_total: 1,
            verify_replay_disagreements: None,
            verification: None,
        };
        let audit = verify_consistency(&r, 1, 2).unwrap();
        assert_eq!(audit.acceptance_rate(), None);
        assert!(!audit
            .to_metrics()
            .iter()
            .any(|(k, _)| k == "audit_spec_acceptance_rate"));
    }

    // ---- v1.2 COHORT ------------------------------------------------------

    /// A conformant cohort: B streams x N=4 committed tokens, R=2 rounds of common width 3 then 1.
    /// Every row's natural walk equals the committed width (no straggler, no truncation).
    fn conformant_cohort(batch_size: u32) -> CohortFreeRunResponse {
        let n = 4usize;
        let acceptance_lengths = vec![3u32, 1];
        let rounds = acceptance_lengths.len();
        CohortFreeRunResponse {
            batch_size,
            tokens_len_by_stream: vec![n; batch_size as usize],
            natural_accepted_by_stream: vec![acceptance_lengths.clone(); batch_size as usize],
            active_streams_by_round: vec![batch_size; rounds],
            rounds: rounds as u32,
            // One base-model fallback per round is not an accepted draft.
            drafted_total: (batch_size as u64) * n as u64,
            accepted_total: (batch_size as u64) * (n - rounds) as u64,
            committed_total: batch_size as u64 * n as u64,
            acceptance_lengths,
            depth_clamp_reasons: [("tail_depth".to_string(), 1u64)].into_iter().collect(),
        }
    }

    #[test]
    fn cohort_series_tag_is_per_batch_size() {
        // D5: one series per batch point, so the EXISTING string-equality fence refuses a
        // cross-batch comparison with no new gate code.
        assert_eq!(
            timed_mode_batched_free_run(8),
            TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8
        );
        assert!(timed_modes_comparable(
            &timed_mode_batched_free_run(8),
            TIMED_MODE_BATCHED_FREE_RUN_V1_2_B8
        ));
        assert!(!timed_modes_comparable(
            &timed_mode_batched_free_run(1),
            &timed_mode_batched_free_run(8)
        ));
        // And a batched number is never comparable to the single-stream free-run series.
        assert!(!timed_modes_comparable(
            &timed_mode_batched_free_run(8),
            TIMED_MODE_FREE_RUN_V1_1
        ));
    }

    #[test]
    fn cohort_positive_control_quadruple_holds() {
        // R=2 ⇒ completed_work = R+1 = 3, a SCALAR that does NOT scale with B.
        let audit = verify_cohort_consistency(&conformant_cohort(8), 4, 3).unwrap();
        assert_eq!(audit.batch_size(), 8);
        assert_eq!(audit.rounds(), 2);
        assert_eq!(audit.cohort_committed_total(), 32);
        assert_eq!(audit.base().acceptance_lengths(), &[3, 1]);
        // No row walked past the committed width: no throttling, no budget truncation.
        assert_eq!(audit.throttled_round_count(), 0);
        assert_eq!(audit.budget_truncated_round_count(), 0);
        assert_eq!(audit.depth_clamp_total(), 1);
    }

    #[test]
    fn cohort_at_b1_runs_the_same_scalar_equations_as_v1_1() {
        // The merge gate, at the pure-rules layer: B=1 through the cohort verifier must accept
        // exactly what the v1.1 verifier accepts and reject exactly what it rejects.
        let cohort = conformant_cohort(1);
        let single = FreeRunResponse {
            tokens_len: 4,
            acceptance_lengths: cohort.acceptance_lengths.clone(),
            drafted_total: cohort.drafted_total,
            accepted_total: cohort.accepted_total,
            committed_total: cohort.committed_total,
            verify_replay_disagreements: None,
            verification: None,
        };
        let cohort_audit = verify_cohort_consistency(&cohort, 4, 3).unwrap();
        let single_audit = verify_consistency(&single, 4, 3).unwrap();
        // The shared `audit_spec_*` family is IDENTICAL — same acceptance data, same derivations.
        assert_eq!(cohort_audit.base(), &single_audit);
        assert_eq!(cohort_audit.cohort_committed_total(), 4);

        // Negative controls agree, error for error, across both verifiers.
        for (mutate_cohort, mutate_single) in [
            (
                Box::new(|c: &mut CohortFreeRunResponse| c.acceptance_lengths = vec![3, 2])
                    as Box<dyn Fn(&mut CohortFreeRunResponse)>,
                Box::new(|s: &mut FreeRunResponse| s.acceptance_lengths = vec![3, 2])
                    as Box<dyn Fn(&mut FreeRunResponse)>,
            ),
            (
                Box::new(|c: &mut CohortFreeRunResponse| {
                    c.drafted_total = 1;
                    c.accepted_total = 2;
                }),
                Box::new(|s: &mut FreeRunResponse| {
                    s.drafted_total = 1;
                    s.accepted_total = 2;
                }),
            ),
        ] {
            let mut c = conformant_cohort(1);
            mutate_cohort(&mut c);
            let mut s = single.clone();
            mutate_single(&mut s);
            assert_eq!(
                verify_cohort_consistency(&c, 4, 3).unwrap_err(),
                verify_consistency(&s, 4, 3).unwrap_err(),
                "B=1 cohort and v1.1 verifiers must reject identically"
            );
        }
        // completed_work != R+1 is the same rejection in both regimes.
        assert_eq!(
            verify_cohort_consistency(&cohort, 4, 2).unwrap_err(),
            verify_consistency(&single, 4, 2).unwrap_err()
        );
    }

    #[test]
    fn cohort_negative_control_wrong_width() {
        let mut c = conformant_cohort(8);
        c.tokens_len_by_stream.pop();
        assert_eq!(
            verify_cohort_consistency(&c, 4, 3).unwrap_err(),
            FreeRunConsistencyError::CohortWidth {
                batch_size: 8,
                got: 7
            }
        );
    }

    #[test]
    fn cohort_negative_control_short_stream_and_wrong_cohort_sum() {
        let mut c = conformant_cohort(8);
        c.tokens_len_by_stream[3] = 3;
        assert_eq!(
            verify_cohort_consistency(&c, 4, 3).unwrap_err(),
            FreeRunConsistencyError::CohortStreamTokenCount {
                slot: 3,
                expected: 4,
                got: 3
            }
        );
        // committed_total must be the COHORT sum B*N, not N.
        let mut c = conformant_cohort(8);
        c.committed_total = 4;
        assert_eq!(
            verify_cohort_consistency(&c, 4, 3).unwrap_err(),
            FreeRunConsistencyError::CohortCommittedTotal {
                expected: 32,
                committed_total: 4
            }
        );
    }

    #[test]
    fn cohort_negative_control_self_contradicting_round_count() {
        // `rounds` is redundant with acceptance_lengths.len() BY DESIGN: a response that
        // disagrees with itself is refused, never reconciled.
        let mut c = conformant_cohort(8);
        c.rounds = 3;
        assert_eq!(
            verify_cohort_consistency(&c, 4, 3).unwrap_err(),
            FreeRunConsistencyError::CohortRoundsDisagree {
                rounds: 3,
                acceptance_lengths: 2
            }
        );
    }

    #[test]
    fn cohort_negative_control_natural_walk_below_committed_width() {
        // The committed width is the MINIMUM across rows, so a row claiming it walked LESS than
        // what was committed is arithmetically impossible — the cross-check that anchors the
        // audit vectors to the scored histogram.
        let mut c = conformant_cohort(8);
        c.natural_accepted_by_stream[5][0] = 2; // committed width that round is 3
        assert_eq!(
            verify_cohort_consistency(&c, 4, 3).unwrap_err(),
            FreeRunConsistencyError::CohortNaturalBelowCommitted {
                round: 0,
                slot: 5,
                natural: 2,
                committed: 3
            }
        );
        // Wrong natural rectangle shape is also refused.
        let mut c = conformant_cohort(8);
        c.natural_accepted_by_stream[0] = vec![3];
        assert_eq!(
            verify_cohort_consistency(&c, 4, 3).unwrap_err(),
            FreeRunConsistencyError::CohortNaturalRoundCount {
                slot: 0,
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn cohort_negative_control_active_streams_tail() {
        // Length must be R.
        let mut c = conformant_cohort(8);
        c.active_streams_by_round = vec![8];
        assert_eq!(
            verify_cohort_consistency(&c, 4, 3).unwrap_err(),
            FreeRunConsistencyError::CohortActiveStreamsLength { rounds: 2, got: 1 }
        );
        // A closed cohort can only shed streams, never gain them.
        let mut c = conformant_cohort(8);
        c.active_streams_by_round = vec![7, 8];
        assert_eq!(
            verify_cohort_consistency(&c, 4, 3).unwrap_err(),
            FreeRunConsistencyError::CohortActiveStreamsIncreased {
                round: 1,
                previous: 7,
                active: 8
            }
        );
        // And never exceed B.
        let mut c = conformant_cohort(8);
        c.active_streams_by_round = vec![9, 8];
        assert_eq!(
            verify_cohort_consistency(&c, 4, 3).unwrap_err(),
            FreeRunConsistencyError::CohortActiveStreamsOutOfRange {
                round: 0,
                active: 9,
                batch_size: 8
            }
        );
    }

    #[test]
    fn cohort_audit_separates_straggler_throttling_from_budget_truncation() {
        // Round 0: rows walked 3,3,5 but only 3 committed ⇒ ONE row held the cohort back.
        // Round 1: every row walked 2 but only 1 committed ⇒ the per-stream budget N truncated it.
        let mut c = conformant_cohort(3);
        c.natural_accepted_by_stream = vec![vec![3, 2], vec![3, 2], vec![5, 2]];
        let audit = verify_cohort_consistency(&c, 4, 3).unwrap();
        assert_eq!(audit.throttled_round_count(), 2);
        assert_eq!(audit.budget_truncated_round_count(), 1);
        let metrics = audit.to_metrics();
        let get = |name: &str| metrics.iter().find(|(k, _)| k == name).map(|(_, v)| *v);
        assert_eq!(get("audit_cohort_throttled_round_count"), Some(2.0));
        assert_eq!(get("audit_cohort_budget_truncated_round_count"), Some(1.0));
        assert_eq!(get("audit_cohort_batch_size"), Some(3.0));
        assert_eq!(get("audit_cohort_committed_total"), Some(12.0));
        // mean natural over 6 cells: (3+2+3+2+5+2)/6
        assert!(
            (get("audit_cohort_mean_natural_accepted_length").unwrap() - 17.0 / 6.0).abs() < 1e-9
        );
    }

    #[test]
    fn cohort_metrics_are_audit_only_and_carry_no_engine_minted_keys() {
        // Every key is prefixed and non-scored, and the ENGINE-CONTROLLED depth_clamp_reasons keys
        // never reach the metric namespace — only their total does. The histogram itself is sealed
        // verbatim through the accessor.
        let mut c = conformant_cohort(2);
        c.depth_clamp_reasons = [
            ("automatic_rectangular_limit".to_string(), 2u64),
            ("smuggled_metric_key".to_string(), 5u64),
        ]
        .into_iter()
        .collect();
        let audit = verify_cohort_consistency(&c, 4, 3).unwrap();
        let metrics = audit.to_metrics();
        assert!(metrics
            .iter()
            .all(|(k, _)| k.starts_with("audit_spec_") || k.starts_with("audit_cohort_")));
        assert!(!metrics.iter().any(|(k, _)| k.contains("smuggled")));
        assert_eq!(
            metrics
                .iter()
                .find(|(k, _)| k == "audit_cohort_depth_clamp_total")
                .map(|(_, v)| *v),
            Some(7.0)
        );
        assert_eq!(audit.depth_clamp_reasons()["smuggled_metric_key"], 5);
    }

    #[test]
    fn median_even_count_averages_two_middles() {
        let r = FreeRunResponse {
            tokens_len: 10,
            acceptance_lengths: vec![1, 2, 3, 4],
            drafted_total: 10,
            accepted_total: 6,
            committed_total: 10,
            verify_replay_disagreements: None,
            verification: None,
        };
        let audit = verify_consistency(&r, 10, 5).unwrap();
        // sorted [1,2,3,4] -> (2+3)/2 = 2.5
        assert_eq!(audit.median_acceptance_length(), 2.5);
    }
}
