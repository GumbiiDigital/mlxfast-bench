//! Engine conformance kit skeleton (docs/architecture.md §2).
//!
//! Point it at an `EngineHandle` + a validated golden; it runs the semantic-layer
//! anchor and free-run gates and reports pass/fail per case plus an overall result.
//! `EngineHandle` is the minimal local surface the anchor/free-run gates need. Top-k
//! entries reuse the wire type `bench_protocol::CorrectnessTraceLogit` (S3) rather than
//! a duplicate local struct.

use crate::constants::{
    COHORT_TOKEN_TOLERANCE_PER_THOUSAND, CORRECTNESS_LOGIT_TIE_TOLERANCE, CORRECTNESS_TOP_LOGITS,
};
use crate::golden::{GoldenAnchorCase, GoldenCase, GoldenFixture, GoldenFreeRunCase, Token};
use crate::BenchError;
use std::collections::HashSet;

/// One entry of a returned top-k logit distribution (the wire type; `token`/`logit`).
pub use bench_protocol::CorrectnessTraceLogit as TopLogit;

/// Output of a teacher-forced single forward over a full context.
#[derive(Debug, Clone, PartialEq)]
pub struct AnchorOutput {
    /// The engine's materialized argmax token.
    pub token: Token,
    /// Top-k logits (architecture.md: `top_logits[8]`), ordered by the engine.
    pub top_logits: Vec<TopLogit>,
}

/// Minimal engine surface for the anchor/free-run gates. See docs/architecture.md §2.
pub trait EngineHandle {
    /// Teacher-forced single forward over a full context; returns argmax + top-k logits.
    fn anchor_forward(&mut self, context_tokens: &[Token]) -> Result<AnchorOutput, BenchError>;

    /// Greedy free run: prompt + `steps` steps -> generated token ids.
    fn free_run(&mut self, prompt_tokens: &[Token], steps: usize)
        -> Result<Vec<Token>, BenchError>;

    /// Teacher-forced primary-case forward (B3 / Swift `compareTeacherForcedWithWorker`):
    /// begin with `prompt_tokens`, then feed each `forced_tokens[i]` (the GOLDEN token, not
    /// the model's own output) and return the engine's predicted token + top-k logits at
    /// each of `steps` positions. The returned vector MUST have exactly `steps` entries.
    fn teacher_forced(
        &mut self,
        prompt_tokens: &[Token],
        forced_tokens: &[Token],
        steps: usize,
    ) -> Result<Vec<AnchorOutput>, BenchError>;

    /// Allocator-drain hook (docs/architecture.md §2 anti-cheat). Default: no-op.
    fn drain_allocator(&mut self) -> Result<(), BenchError> {
        Ok(())
    }
}

/// Canonical argmax / tie-break: LOWEST token id among the max-logit set
/// (docs/architecture.md §2 "canonical argmax tie-break (lowest token id)").
/// Returns `None` for an empty slice.
/// The engine's token as a failure message names it.
fn produced(token: Option<Token>) -> String {
    token.map_or_else(|| "no token".to_string(), |t| format!("token {t}"))
}

/// A diagnostic value as a failure message names it.
fn or_unknown<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "unknown".to_string(), |v| v.to_string())
}

pub fn canonical_argmax(logits: &[TopLogit]) -> Option<Token> {
    let mut best: Option<TopLogit> = None;
    for &cur in logits {
        match best {
            None => best = Some(cur),
            Some(b) => {
                if cur.logit > b.logit || (cur.logit == b.logit && cur.token < b.token) {
                    best = Some(cur);
                }
            }
        }
    }
    best.map(|b| b.token)
}

/// 1-based rank of `token` within `logits` ordered by logit desc, tie-break lowest id.
/// `None` if `token` is not present in the returned top-k.
fn rank_of(logits: &[TopLogit], token: Token) -> Option<usize> {
    if !logits.iter().any(|l| l.token == token) {
        return None;
    }
    let mut sorted: Vec<TopLogit> = logits.to_vec();
    // C7: total_cmp gives a TOTAL order even under NaN (partial_cmp+unwrap_or(Equal) is
    // non-total, yielding arbitrary ranks and a potential driftsort panic on Rust >=1.81).
    // C1's finiteness gate is the primary defense; this keeps the sort sound regardless.
    sorted.sort_by(|a, b| b.logit.total_cmp(&a.logit).then(a.token.cmp(&b.token)));
    sorted.iter().position(|l| l.token == token).map(|i| i + 1)
}

fn max_logit(logits: &[TopLogit]) -> Option<f64> {
    logits
        .iter()
        .map(|l| l.logit)
        .fold(None, |acc, v| match acc {
            None => Some(v),
            Some(m) => Some(if v > m { v } else { m }),
        })
}

fn logit_of(logits: &[TopLogit], token: Token) -> Option<f64> {
    logits.iter().find(|l| l.token == token).map(|l| l.logit)
}

/// Exact port of Swift `correctnessTokenAccepted` (QwenRuntimeSupport.swift): accept
/// `actual_token` for `accepted_token` if they are equal, or `accepted_token`'s logit is
/// within the canonical tie tolerance of the TOP (first-returned) logit. Some Apple
/// GPU/Metal combinations break exact argmax ties differently, so a true top-logit tie is
/// accepted — this is the per-accepted-token leniency the anchor loop applies.
fn correctness_token_accepted(
    accepted_token: Token,
    actual_token: Token,
    top_logits: &[TopLogit],
) -> bool {
    if actual_token == accepted_token {
        return true;
    }
    let Some(top) = top_logits.first().map(|l| l.logit) else {
        return false;
    };
    let Some(accepted_logit) = top_logits
        .iter()
        .find(|l| l.token == accepted_token)
        .map(|l| l.logit)
    else {
        return false;
    };
    top - accepted_logit <= CORRECTNESS_LOGIT_TIE_TOLERANCE
}

/// Per-anchor-case result.
#[derive(Debug, Clone, PartialEq)]
pub struct AnchorCaseResult {
    pub name: String,
    pub passed: bool,
    pub argmax: Option<Token>,
    pub expected_token: Token,
    pub expected_rank: Option<usize>,
    pub top_logit_delta: Option<f64>,
    pub reason: String,
    /// Swift `CorrectnessTokenComparison.checkedSteps` for an anchor: ALWAYS 1, on both the
    /// pass and the fail path (`compareAnchorToken`, QwenRuntimeCorrectnessCompare.swift:472/480).
    pub checked_steps: usize,
}

/// Per-free-run-case result.
#[derive(Debug, Clone, PartialEq)]
pub struct FreeRunCaseResult {
    pub name: String,
    pub passed: bool,
    /// First index where the generated token diverges from the expected prefix.
    pub first_mismatch_step: Option<usize>,
    pub reason: String,
    /// Swift `CorrectnessTokenComparison.checkedSteps` for a free-run case
    /// (`compareFreeRunTokens`, QwenRuntimeCorrectnessCompare.swift:499): PASS → the enforced
    /// prefix length (`exact_prefix_tokens ?? expected_tokens.len()`), FAIL → `step + 1`.
    pub checked_steps: usize,
}

/// Per-primary-case result (B3): a teacher-forced base `GoldenCase`.
///
/// SHAPE OF THE MISMATCH SUMMARY (David's blanket-10% ruling, 2026-09-02): `mismatch_count`
/// is the TOTAL number of non-accepted teacher-forced steps in the case, and `compared_steps`
/// is the number of steps compared (the budget denominator). `passed` is the BUDGET verdict,
/// not `mismatch_count == 0`: a case can pass with `mismatch_count > 0`, and
/// `first_mismatch_step` stays populated so the first divergence is still visible. On a
/// refusal `reason` states the first mismatch step, the mismatch count, the compared steps,
/// and the budget.
#[derive(Debug, Clone, PartialEq)]
pub struct TeacherForcedCaseResult {
    pub name: String,
    pub passed: bool,
    /// First teacher-forced step whose predicted token was not accepted.
    pub first_mismatch_step: Option<usize>,
    pub expected_token: Option<Token>,
    pub actual_token: Option<Token>,
    /// Total non-accepted teacher-forced steps in this case (the budget numerator).
    pub mismatch_count: usize,
    /// Teacher-forced steps actually compared in this case (the budget denominator).
    pub compared_steps: usize,
    pub reason: String,
    /// Swift `CorrectnessTokenComparison.checkedSteps` for a teacher-forced base case
    /// (`compareTeacherForcedCached`, QwenRuntimeCorrectnessCompare.swift:366/394): PASS → the
    /// full teacher-forced window `steps`, FAIL → `first_mismatch_step + 1`.
    pub checked_steps: usize,
}

/// David's blanket-10% token-mismatch budget, in INTEGER arithmetic, applied PER CASE:
/// `mismatches * 1000 <= COHORT_TOKEN_TOLERANCE_PER_THOUSAND * steps`.
///
/// Same constant and same arithmetic as the batched cohort path
/// ([`crate::cohort_tolerance::evaluate_cohort_token_tolerance`]). The denominator convention
/// [`COHORT_TOKEN_TOLERANCE_PER_THOUSAND`] prescribes is "the stream's OWN token count", which
/// for a teacher-forced case is the number of compared steps. Per-thousand integers keep the
/// boundary exact: 10% of the steps passes; one more mismatch refuses.
pub fn within_teacher_forced_mismatch_budget(mismatches: usize, steps: usize) -> bool {
    (mismatches as u64) * 1000 <= (COHORT_TOKEN_TOLERANCE_PER_THOUSAND as u64) * (steps as u64)
}

/// Overall conformance report. `base_cases` are the golden's primary teacher-forced
/// `cases[]` (B3): before this they were validated at load but never evaluated, so
/// benchd passed a strictly weaker exam than Swift's layered correctness.
#[derive(Debug, Clone, PartialEq)]
pub struct ConformanceReport {
    pub base_cases: Vec<TeacherForcedCaseResult>,
    pub anchors: Vec<AnchorCaseResult>,
    pub free_run: Vec<FreeRunCaseResult>,
    pub passed: bool,
}

impl ConformanceReport {
    /// The real per-case checked-step SUM, byte-for-byte with Swift's
    /// `runLayeredCorrectness` accumulator (QwenRuntimeCorrectness.swift:192-306): walk the
    /// gates in Swift's evaluation order — base teacher-forced cases, THEN anchors, THEN
    /// free-run — adding each case's `checked_steps`, and STOP after the FIRST failing case
    /// (its steps included, matching Swift's `checkedSteps + comparison.checkedSteps` return).
    /// A fully-passing report therefore sums every evaluated case; a mid-correctness failure
    /// yields the partial sum through the failing gate.
    ///
    /// BEHAVIOR/GPQA GAP (same as official §8 / MINOR-1): bench-core conformance does not run
    /// the behavior gates, so for a behavior-bearing golden this SUM under-counts vs Swift by
    /// exactly the behavior contribution — the pre-existing B-3 gap, not a new divergence.
    pub fn checked_steps(&self) -> i64 {
        let mut total: i64 = 0;
        for c in &self.base_cases {
            total += c.checked_steps as i64;
            if !c.passed {
                return total;
            }
        }
        for a in &self.anchors {
            total += a.checked_steps as i64;
            if !a.passed {
                return total;
            }
        }
        for f in &self.free_run {
            total += f.checked_steps as i64;
            if !f.passed {
                return total;
            }
        }
        total
    }
}

/// Evaluate a single anchor case against an engine output.
///
/// PASS if argmax ∈ accepted_tokens (or, when accepted_tokens is absent, argmax ==
/// expected_token), OR expected-token `rank <= max_expected_rank` AND
/// `(top_logit - expected_logit) <= max_top_logit_delta`. The rank/delta path can
/// only pass when BOTH tolerances are present in the case.
/// `vocab_size` is the TRACK's vocabulary bound ([`crate::constants::TrackModelIdentity`]): a
/// worker top-logit entry outside it is untrustworthy, and the bound is a property of the
/// checkpoint the track pins, never a global.
pub fn evaluate_anchor_case(
    case: &GoldenAnchorCase,
    output: &AnchorOutput,
    vocab_size: usize,
) -> AnchorCaseResult {
    // C1: fail CLOSED on any non-finite logit. Comparisons against NaN are all false, so
    // a spoofed `{token: expected, logit: NaN}` entry is never displaced by the
    // argmax/rank comparisons — a NaN-poisoned distribution could otherwise satisfy a
    // gate that must fail. Reject before evaluating anything.
    if output.top_logits.iter().any(|l| !l.logit.is_finite()) {
        return AnchorCaseResult {
            name: case.name.clone(),
            passed: false,
            argmax: None,
            expected_token: case.expected_token,
            expected_rank: None,
            top_logit_delta: None,
            reason: format!(
                "anchor {} failed: non-finite logit in engine top_logits",
                case.name
            ),
            checked_steps: 1,
        };
    }

    // Enforce the canonical tie-break from the returned distribution rather than
    // trusting the engine's self-reported token.
    let argmax = canonical_argmax(&output.top_logits);

    // C2 (EXACT port of Swift `anchorTokenAccepted`): the accepted set always includes
    // expected_token, and each accepted token is checked with `correctness_token_accepted`
    // — argmax equals it, OR its logit is within the canonical tie tolerance of the top
    // (first-returned) logit (Swift's per-accepted-token leniency loop). Matching Swift
    // exactly avoids a parity trap in the WS1-10 score diff: a golden that passes in Swift
    // only via this leniency must not fail in Rust.
    let mut accepted_set: Vec<Token> = case.accepted_tokens.clone().unwrap_or_default();
    if !accepted_set.contains(&case.expected_token) {
        accepted_set.push(case.expected_token);
    }
    // Gate the tie-leniency path with the same validatedWorkerTopLogits check as the
    // teacher-forced path: a malformed/unsorted distribution degrades to the strict
    // (exact argmax) comparison, so it can never WIDEN acceptance.
    let pass_accepted = match argmax {
        Some(a) => {
            let tie_logits: &[TopLogit] =
                if top_logits_trustworthy(&output.top_logits, a, vocab_size) {
                    &output.top_logits
                } else {
                    &[]
                };
            accepted_set
                .iter()
                .any(|&acc| correctness_token_accepted(acc, a, tie_logits))
        }
        None => false,
    };

    let expected_rank = rank_of(&output.top_logits, case.expected_token);
    let top_logit_delta = match (
        max_logit(&output.top_logits),
        logit_of(&output.top_logits, case.expected_token),
    ) {
        (Some(top), Some(exp)) => Some(top - exp),
        _ => None,
    };

    // C2: the rank/delta path activates on `max_expected_rank` ALONE (Swift only guards on
    // maxExpectedRank); `max_top_logit_delta` defaults to the canonical tie tolerance when
    // absent (Swift `?? correctnessLogitTieTolerance`).
    let pass_rank_delta = match case.max_expected_rank {
        Some(max_rank) => {
            let max_delta = case
                .max_top_logit_delta
                .unwrap_or(CORRECTNESS_LOGIT_TIE_TOLERANCE);
            let rank_ok = expected_rank.map(|r| r <= max_rank).unwrap_or(false);
            let delta_ok = top_logit_delta.map(|d| d <= max_delta).unwrap_or(false);
            rank_ok && delta_ok
        }
        None => false,
    };

    let passed = pass_accepted || pass_rank_delta;
    let reason = if passed {
        String::new()
    } else {
        format!(
            "anchor {} failed: the engine chose {}, which is not an accepted token, and the \
expected token is outside the tolerance (its rank is {}, its logit is {} below the top logit)",
            case.name,
            produced(argmax),
            or_unknown(expected_rank),
            or_unknown(top_logit_delta)
        )
    };

    AnchorCaseResult {
        name: case.name.clone(),
        passed,
        argmax,
        expected_token: case.expected_token,
        expected_rank,
        top_logit_delta,
        reason,
        checked_steps: 1,
    }
}

/// Evaluate a free-run case: the first `exact_prefix_tokens` generated tokens
/// (or all expected tokens when unset) must exactly match the expected greedy
/// continuation. Reports the first mismatch step.
pub fn evaluate_free_run_case(case: &GoldenFreeRunCase, generated: &[Token]) -> FreeRunCaseResult {
    let required = case
        .exact_prefix_tokens
        .unwrap_or(case.expected_tokens.len());
    let mut first_mismatch: Option<usize> = None;
    for step in 0..required {
        let expected = case.expected_tokens.get(step).copied();
        let actual = generated.get(step).copied();
        if expected != actual {
            first_mismatch = Some(step);
            break;
        }
    }
    let passed = first_mismatch.is_none();
    let reason = if passed {
        String::new()
    } else {
        let step = first_mismatch.unwrap();
        format!(
            "free-run {} diverged at step {}: the engine produced {}, which is not the token \
the golden expects at this step",
            case.name,
            step,
            produced(generated.get(step).copied())
        )
    };
    // Swift compareFreeRunTokens: PASS → the enforced prefix length, FAIL → step + 1.
    let checked_steps = match first_mismatch {
        Some(step) => step + 1,
        None => required,
    };
    FreeRunCaseResult {
        name: case.name.clone(),
        passed,
        first_mismatch_step: first_mismatch,
        reason,
        checked_steps,
    }
}

/// Are these worker-reported top logits internally consistent enough to trust for the
/// tie-tolerance path? Port of the essential `validatedWorkerTopLogits` checks: every
/// logit finite AND the top (first-returned) entry is the reported token. A malformed
/// distribution degrades to the STRICT comparison (exact match only) — never to a wider
/// acceptance — exactly as Swift discards bad worker logits before the tie check.
fn top_logits_trustworthy(
    top_logits: &[TopLogit],
    reported_token: Token,
    vocab_size: usize,
) -> bool {
    // EXACT port of Swift validatedWorkerTopLogits: non-empty, at most CORRECTNESS_TOP_LOGITS
    // entries, leads with `reported_token`, and every entry is in-vocab, finite, unique, and
    // in STRICT logit-descending order with the lowest-token tie-break. Any violation is
    // untrustworthy -> the caller degrades to the strict (exact-only) comparison, NEVER to a
    // wider acceptance. This closes the malformed/unsorted-distribution cheat (e.g.
    // [{99,1.0},{31,9.0}] would otherwise let the higher-logit 31 be tie-accepted against the
    // first-listed, lower-logit token).
    if top_logits.is_empty() || top_logits.len() > CORRECTNESS_TOP_LOGITS {
        return false;
    }
    if top_logits[0].token != reported_token {
        return false;
    }
    let mut seen: HashSet<Token> = HashSet::with_capacity(top_logits.len());
    for (i, item) in top_logits.iter().enumerate() {
        if item.token < 0 || item.token >= vocab_size as i64 {
            return false;
        }
        if !item.logit.is_finite() {
            return false;
        }
        if !seen.insert(item.token) {
            return false;
        }
        if i > 0 {
            let prev = &top_logits[i - 1];
            let descending =
                prev.logit > item.logit || (prev.logit == item.logit && prev.token < item.token);
            if !descending {
                return false;
            }
        }
    }
    true
}

/// Evaluate a primary teacher-forced case (B3). At each of `steps` positions the engine
/// predicted `outputs[step].token` after being fed the golden prefix; accept it if it
/// equals `expected_tokens[step]` OR is a true top-logit tie (`correctness_token_accepted`
/// over trustworthy worker logits). Ports Swift `compareTeacherForcedWithWorker` (same
/// acceptance rule as the anchor tie path).
///
/// NOT exact-match: the case passes iff the non-accepted steps stay inside David's blanket
/// 10% token-mismatch budget (ruling 2026-09-02), evaluated by
/// [`within_teacher_forced_mismatch_budget`] over the WHOLE case. Teacher forcing feeds the
/// GOLDEN token at every position, so every step stays comparable after a mismatch; the walk
/// therefore continues to the end and the count is complete. `first_mismatch_step` /
/// `expected_token` / `actual_token` still report the FIRST non-accepted step, on both the
/// pass and the refuse path.
pub fn evaluate_teacher_forced_case(
    case: &GoldenCase,
    outputs: &[AnchorOutput],
    steps: usize,
    vocab_size: usize,
) -> TeacherForcedCaseResult {
    let mut mismatch_count: usize = 0;
    let mut first: Option<(usize, Token, Token)> = None;
    let mut compared_steps: usize = 0;
    for (step, output) in outputs.iter().enumerate().take(steps) {
        compared_steps += 1;
        let actual = output.token;
        let expected = case.expected_tokens[step];
        if actual != expected {
            // Degrade to strict (empty slice) when the worker logits are untrustworthy,
            // so a malformed distribution can never WIDEN acceptance.
            let logits: &[TopLogit] =
                if top_logits_trustworthy(&output.top_logits, actual, vocab_size) {
                    &output.top_logits
                } else {
                    &[]
                };
            if !correctness_token_accepted(expected, actual, logits) {
                mismatch_count += 1;
                if first.is_none() {
                    first = Some((step, expected, actual));
                }
            }
        }
    }

    let passed = within_teacher_forced_mismatch_budget(mismatch_count, compared_steps);
    let (first_mismatch_step, expected_token, actual_token) = match first {
        Some((step, expected, actual)) => (Some(step), Some(expected), Some(actual)),
        None => (None, None, None),
    };
    let reason = match (passed, first) {
        (true, _) => String::new(),
        (false, Some((step, _, actual))) => format!(
            "teacher-forced {} mismatch at step {step}: the engine produced token {actual}, \
which is not the token the golden expects at this step ({mismatch_count} mismatches over \
{compared_steps} steps exceeds the {COHORT_TOKEN_TOLERANCE_PER_THOUSAND}-per-thousand budget)",
            case.name
        ),
        // Unreachable in practice: zero mismatches is always inside the budget.
        (false, None) => format!("teacher-forced {} failed the mismatch budget", case.name),
    };
    // Swift compareTeacherForcedCached: PASS -> the full teacher-forced window; FAIL ->
    // first_mismatch_step + 1. The budget changes WHEN a case fails, never this shape.
    let checked_steps = match (passed, first_mismatch_step) {
        (false, Some(step)) => step + 1,
        _ => steps,
    };
    TeacherForcedCaseResult {
        name: case.name.clone(),
        passed,
        first_mismatch_step,
        expected_token,
        actual_token,
        mismatch_count,
        compared_steps,
        reason,
        checked_steps,
    }
}

/// Which correctness gates [`run_conformance`] evaluates.
///
/// Swift's `--local-iterate` path (`QwenRuntime.localIterate` →
/// `runLocalIterateCheckedTiming`) judges correctness SOLELY from the primary
/// teacher-forced `cases.first` timing stream — it never evaluates
/// `correctness_gates.anchors` / `.free_run` / `.behavior`. benchd historically
/// ran the full superset (base cases + anchors + free-run), so a golden's anchor or
/// free-run corruption FAILED benchd where Swift PASSES. `BaseCasesOnly` restores
/// Swift-exact behavior; `Full` keeps benchd's superset (opt-in via `--strict`,
/// and always for official/submit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrectnessScope {
    /// Swift-exact: evaluate ONLY the primary teacher-forced `cases[]` (the timing
    /// stream). The anchor / free-run gates are skipped. (The `behavior` gate is not
    /// wired into the runner in EITHER scope today — a pre-existing gap, filed separately.)
    BaseCasesOnly,
    /// benchd superset: base cases PLUS the anchor and free-run gates.
    Full,
}

/// Top-level runner: drains the allocator before each sequence, evaluates the primary
/// teacher-forced cases (B3) then — when `scope` is [`CorrectnessScope::Full`] — the
/// anchor and free-run gates, and returns the aggregate report. `steps` is the
/// teacher-forced window per primary case (Swift `correctnessSteps`). Under
/// [`CorrectnessScope::BaseCasesOnly`] the anchor/free-run vectors are left empty
/// (Swift `--local-iterate` parity), so `passed` reflects the base cases alone.
pub fn run_conformance<E: EngineHandle>(
    engine: &mut E,
    golden: &GoldenFixture,
    steps: usize,
    scope: CorrectnessScope,
    vocab_size: usize,
) -> Result<ConformanceReport, BenchError> {
    // B3: the golden's primary `cases[]` (teacher-forced) ran in Swift's layered
    // correctness but were skipped here entirely — evaluate them FIRST, matching the
    // Swift order (cases -> anchors -> free-run). Each is a fresh sequence, so it drains.
    let mut base_cases = Vec::new();
    for case in &golden.cases {
        // Bounds guard (matches Swift compareTeacherForcedWithWorker's `expectedTokens.count
        // >= steps` throw): the loader validates this for loaded goldens, but a directly
        // constructed fixture — or a future `steps` change — must fail closed with a
        // BenchError, never index-panic in evaluate_teacher_forced_case.
        if case.expected_tokens.len() < steps {
            return Err(BenchError::InvalidInput(format!(
                "primary case {} has {} expected_tokens; teacher-forced correctness needs at least {steps}",
                case.name,
                case.expected_tokens.len()
            )));
        }
        engine.drain_allocator()?;
        let outputs = engine.teacher_forced(&case.prompt_tokens, &case.expected_tokens, steps)?;
        if outputs.len() != steps {
            return Err(BenchError::InvalidInput(format!(
                "teacher_forced returned {} outputs for {} steps (case {})",
                outputs.len(),
                steps,
                case.name
            )));
        }
        base_cases.push(evaluate_teacher_forced_case(
            case, &outputs, steps, vocab_size,
        ));
    }

    // Swift `--local-iterate` parity (`CorrectnessScope::BaseCasesOnly`): skip the
    // anchor/free-run gates entirely, leaving both vectors empty so `passed` is the base
    // cases alone. `Full` keeps benchd's superset (official/submit + `--strict`).
    let gates = match scope {
        CorrectnessScope::Full => golden.correctness_gates.as_ref(),
        CorrectnessScope::BaseCasesOnly => None,
    };

    // C3: PROTOCOL.md (:100-106, normative) requires the allocator drain at the start of
    // EVERY new correctness sequence, not once per phase — otherwise cases 2..n run
    // undrained and a state-caching engine slips the anti-memoization gate. Drain inside
    // each loop, before each sequence.
    let mut anchors = Vec::new();
    if let Some(gates) = gates {
        for case in gates.anchor_cases() {
            engine.drain_allocator()?;
            let output = engine.anchor_forward(&case.context_tokens)?;
            anchors.push(evaluate_anchor_case(case, &output, vocab_size));
        }
    }

    let mut free_run = Vec::new();
    if let Some(gates) = gates {
        for case in gates.free_run_cases() {
            engine.drain_allocator()?;
            let generated = engine.free_run(&case.prompt_tokens, case.expected_tokens.len())?;
            free_run.push(evaluate_free_run_case(case, &generated));
        }
    }

    let passed = base_cases.iter().all(|r| r.passed)
        && anchors.iter().all(|r| r.passed)
        && free_run.iter().all(|r| r.passed);
    Ok(ConformanceReport {
        base_cases,
        anchors,
        free_run,
        passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::VOCAB_SIZE;
    use crate::golden::{GoldenCorrectnessGates, GoldenFixture};

    fn tl(token: Token, logit: f64) -> TopLogit {
        TopLogit { token, logit }
    }

    #[test]
    fn canonical_argmax_lowest_id_on_tie() {
        let logits = vec![tl(5, 1.0), tl(2, 1.0), tl(9, 0.5)];
        assert_eq!(canonical_argmax(&logits), Some(2));
    }

    #[test]
    fn canonical_argmax_empty_none() {
        assert_eq!(canonical_argmax(&[]), None);
    }

    fn anchor(
        name: &str,
        expected: Token,
        accepted: Option<Vec<Token>>,
        rank: Option<usize>,
        delta: Option<f64>,
    ) -> GoldenAnchorCase {
        GoldenAnchorCase {
            name: name.into(),
            context_tokens: vec![1, 2, 3],
            expected_token: expected,
            accepted_tokens: accepted,
            max_expected_rank: rank,
            max_top_logit_delta: delta,
        }
    }

    #[test]
    fn anchor_argmax_in_accepted_passes() {
        let case = anchor("a", 100, Some(vec![7, 8]), None, None);
        let out = AnchorOutput {
            token: 7,
            top_logits: vec![tl(7, 3.0), tl(8, 2.0), tl(100, 1.0)],
        };
        let r = evaluate_anchor_case(&case, &out, VOCAB_SIZE);
        assert!(r.passed);
        assert_eq!(r.argmax, Some(7));
    }

    #[test]
    fn anchor_argmax_missed_but_rank_delta_within_tolerance_passes() {
        // argmax is 7 (not accepted={8}), but expected=100 is rank 2 with delta 1.0.
        let case = anchor("a", 100, Some(vec![8]), Some(2), Some(1.5));
        let out = AnchorOutput {
            token: 7,
            top_logits: vec![tl(7, 3.0), tl(100, 2.0), tl(8, 1.0)],
        };
        let r = evaluate_anchor_case(&case, &out, VOCAB_SIZE);
        assert!(r.passed);
        assert_eq!(r.expected_rank, Some(2));
        assert_eq!(r.top_logit_delta, Some(1.0));
    }

    #[test]
    fn anchor_rank_exceeds_max_fails() {
        // expected=100 is rank 3 but max_expected_rank=2; argmax not accepted.
        let case = anchor("a", 100, Some(vec![8]), Some(2), Some(10.0));
        let out = AnchorOutput {
            token: 7,
            top_logits: vec![tl(7, 3.0), tl(9, 2.5), tl(100, 2.0)],
        };
        let r = evaluate_anchor_case(&case, &out, VOCAB_SIZE);
        assert!(!r.passed);
        assert_eq!(r.expected_rank, Some(3));
        assert!(!r.reason.contains("100"), "{}", r.reason);
    }

    #[test]
    fn anchor_delta_exceeds_max_fails() {
        let case = anchor("a", 100, Some(vec![8]), Some(2), Some(0.5));
        let out = AnchorOutput {
            token: 7,
            top_logits: vec![tl(7, 3.0), tl(100, 2.0)],
        };
        // rank ok (2) but delta 1.0 > 0.5
        let r = evaluate_anchor_case(&case, &out, VOCAB_SIZE);
        assert!(!r.passed);
    }

    #[test]
    fn anchor_no_accepted_falls_back_to_exact_argmax() {
        let case = anchor("a", 7, None, None, None);
        let out = AnchorOutput {
            token: 7,
            top_logits: vec![tl(7, 3.0), tl(8, 2.0)],
        };
        assert!(evaluate_anchor_case(&case, &out, VOCAB_SIZE).passed);

        let case2 = anchor("a", 8, None, None, None);
        assert!(!evaluate_anchor_case(&case2, &out, VOCAB_SIZE).passed);
    }

    #[test]
    fn free_run_exact_prefix_match_passes() {
        let case = GoldenFreeRunCase {
            name: "fr".into(),
            prompt_tokens: vec![1, 2, 3],
            expected_tokens: vec![10, 11, 12, 13],
            exact_prefix_tokens: Some(2),
        };
        // first 2 match; later divergence is beyond the enforced prefix.
        let r = evaluate_free_run_case(&case, &[10, 11, 99, 99]);
        assert!(r.passed);
        assert_eq!(r.first_mismatch_step, None);
    }

    #[test]
    fn free_run_mismatch_fails_with_step() {
        let case = GoldenFreeRunCase {
            name: "fr".into(),
            prompt_tokens: vec![1, 2, 3],
            expected_tokens: vec![10, 11, 12, 13],
            exact_prefix_tokens: None,
        };
        let r = evaluate_free_run_case(&case, &[10, 11, 99, 13]);
        assert!(!r.passed);
        assert_eq!(r.first_mismatch_step, Some(2));
        assert!(
            r.reason.contains("99") && !r.reason.contains("12"),
            "{}",
            r.reason
        );
    }

    // --- fake engine + run_conformance ---

    struct FakeEngine {
        drain_calls: usize,
        anchor_out: AnchorOutput,
        free_run_out: Vec<Token>,
        /// Per-step teacher-forced outputs returned for EACH primary case (B3).
        teacher_forced_out: Vec<AnchorOutput>,
    }

    impl EngineHandle for FakeEngine {
        fn anchor_forward(&mut self, _ctx: &[Token]) -> Result<AnchorOutput, BenchError> {
            Ok(self.anchor_out.clone())
        }
        fn free_run(&mut self, _p: &[Token], _steps: usize) -> Result<Vec<Token>, BenchError> {
            Ok(self.free_run_out.clone())
        }
        fn teacher_forced(
            &mut self,
            _p: &[Token],
            _forced: &[Token],
            _steps: usize,
        ) -> Result<Vec<AnchorOutput>, BenchError> {
            Ok(self.teacher_forced_out.clone())
        }
        fn drain_allocator(&mut self) -> Result<(), BenchError> {
            self.drain_calls += 1;
            Ok(())
        }
    }

    fn fixture_with_gates(gates: GoldenCorrectnessGates) -> GoldenFixture {
        GoldenFixture {
            model_provenance: None,
            model_type: None,
            cases: vec![],
            correctness_gates: Some(gates),
            benchmark: None,
            sha256: "0".repeat(64),
            byte_len: 0,
        }
    }

    #[test]
    fn run_conformance_passes_and_drains_per_sequence() {
        // C3: TWO anchor cases + one free-run. Per-sequence drain => 3 drains; the old
        // per-phase drain would be only 2 (this test fails before the fix, passes after).
        let gates = GoldenCorrectnessGates {
            anchors: Some(vec![
                anchor("a1", 7, Some(vec![7]), None, None),
                anchor("a2", 7, Some(vec![7]), None, None),
            ]),
            free_run: Some(vec![GoldenFreeRunCase {
                name: "fr".into(),
                prompt_tokens: vec![1, 2, 3],
                expected_tokens: vec![10, 11],
                exact_prefix_tokens: None,
            }]),
            behavior: None,
        };
        let fx = fixture_with_gates(gates);
        let mut engine = FakeEngine {
            drain_calls: 0,
            anchor_out: AnchorOutput {
                token: 7,
                top_logits: vec![tl(7, 3.0), tl(8, 2.0)],
            },
            free_run_out: vec![10, 11],
            teacher_forced_out: vec![],
        };
        let report = run_conformance(
            &mut engine,
            &fx,
            crate::constants::CORRECTNESS_STEPS,
            CorrectnessScope::Full,
            VOCAB_SIZE,
        )
        .unwrap();
        assert!(report.passed);
        assert_eq!(report.anchors.len(), 2);
        assert_eq!(report.free_run.len(), 1);
        // drain called before EVERY sequence: 2 anchors + 1 free-run = 3.
        assert_eq!(engine.drain_calls, 3);
    }

    #[test]
    fn base_cases_only_skips_failing_anchor_and_free_run() {
        // Swift `--local-iterate` parity: `BaseCasesOnly` evaluates ONLY the primary
        // teacher-forced cases. A golden whose anchor AND free-run gates would FAIL under
        // the superset must PASS here (no cases[] present → nothing to fail), and neither
        // gate is even executed (no anchor/free-run drains).
        let gates = GoldenCorrectnessGates {
            // anchor expects 999 but the engine argmax is 7 → would FAIL under `Full`.
            anchors: Some(vec![anchor("bad", 999, Some(vec![999]), None, None)]),
            // free-run expects [10,11] but the engine returns [10,99] → would FAIL under `Full`.
            free_run: Some(vec![GoldenFreeRunCase {
                name: "fr".into(),
                prompt_tokens: vec![1, 2, 3],
                expected_tokens: vec![10, 11],
                exact_prefix_tokens: None,
            }]),
            behavior: None,
        };
        let fx = fixture_with_gates(gates);
        let mut engine = FakeEngine {
            drain_calls: 0,
            anchor_out: AnchorOutput {
                token: 7,
                top_logits: vec![tl(7, 3.0), tl(8, 2.0)],
            },
            free_run_out: vec![10, 99],
            teacher_forced_out: vec![],
        };
        let report = run_conformance(
            &mut engine,
            &fx,
            crate::constants::CORRECTNESS_STEPS,
            CorrectnessScope::BaseCasesOnly,
            VOCAB_SIZE,
        )
        .unwrap();
        assert!(report.passed, "base-cases-only skips the failing gates");
        assert!(report.anchors.is_empty());
        assert!(report.free_run.is_empty());
        // No sequences ran (no cases[], gates skipped) → no drains.
        assert_eq!(engine.drain_calls, 0);

        // Sanity: the SAME fixture FAILS under the superset (proves the gates are real).
        let mut engine_full = FakeEngine {
            drain_calls: 0,
            anchor_out: AnchorOutput {
                token: 7,
                top_logits: vec![tl(7, 3.0), tl(8, 2.0)],
            },
            free_run_out: vec![10, 99],
            teacher_forced_out: vec![],
        };
        let full = run_conformance(
            &mut engine_full,
            &fx,
            crate::constants::CORRECTNESS_STEPS,
            CorrectnessScope::Full,
            VOCAB_SIZE,
        )
        .unwrap();
        assert!(
            !full.passed,
            "superset evaluates and fails the corrupt gates"
        );
    }

    #[test]
    fn anchor_non_finite_logit_fails_closed() {
        // C1: expected token carries a NaN logit, with a higher finite logit present.
        // A NaN-blind argmax/rank could pass this; the finiteness gate must fail it.
        let case = anchor("nan", 100, Some(vec![100]), Some(1), Some(1e9));
        let out = AnchorOutput {
            token: 100,
            top_logits: vec![tl(100, f64::NAN), tl(7, 3.0)],
        };
        let r = evaluate_anchor_case(&case, &out, VOCAB_SIZE);
        assert!(!r.passed);
        assert!(r.reason.contains("non-finite logit"));
        // +inf is also rejected.
        let out2 = AnchorOutput {
            token: 100,
            top_logits: vec![tl(100, f64::INFINITY), tl(7, 3.0)],
        };
        assert!(!evaluate_anchor_case(&case, &out2, VOCAB_SIZE).passed);
    }

    #[test]
    fn anchor_expected_token_always_accepted() {
        // C2: expected_token ∉ accepted_tokens, but argmax == expected_token (bit-exact).
        // Before the fix this failed (expected not in accepted, no rank/delta tolerances);
        // after, expected_token is implicitly in the accepted set.
        let case = anchor("exact", 100, Some(vec![8]), None, None);
        let out = AnchorOutput {
            token: 100,
            top_logits: vec![tl(100, 5.0), tl(8, 2.0)],
        };
        let r = evaluate_anchor_case(&case, &out, VOCAB_SIZE);
        assert!(r.passed);
        assert_eq!(r.argmax, Some(100));
    }

    #[test]
    fn anchor_rank_only_uses_tie_tolerance_default() {
        // C2: max_expected_rank set but max_top_logit_delta absent -> delta defaults to
        // the tie tolerance (1e-6). Before the fix the rank/delta path required BOTH and
        // stayed inactive, failing this case.
        let case = anchor("rankonly", 100, Some(vec![8]), Some(2), None);
        // expected=100 at rank 2, delta 5e-7 (< 1e-6 tie tolerance) -> passes.
        let out = AnchorOutput {
            token: 7,
            top_logits: vec![tl(7, 3.0), tl(100, 3.0 - 5e-7)],
        };
        let r = evaluate_anchor_case(&case, &out, VOCAB_SIZE);
        assert!(r.passed);
        assert_eq!(r.expected_rank, Some(2));

        // Same rank but delta 1e-3 (> tie tolerance) -> fails.
        let out_far = AnchorOutput {
            token: 7,
            top_logits: vec![tl(7, 3.0), tl(100, 3.0 - 1e-3)],
        };
        assert!(!evaluate_anchor_case(&case, &out_far, VOCAB_SIZE).passed);
    }

    #[test]
    fn anchor_passes_only_via_per_accepted_token_leniency() {
        // C2 exact Swift port: argmax (7) is NOT in accepted∪expected={50,100}, no
        // rank/delta tolerances are set, and expected (100) is far from the top — the
        // ONLY acceptance path is the per-accepted-token leniency loop: accepted token 50
        // sits within the tie tolerance (5e-7 < 1e-6) of the top logit, so
        // correctness_token_accepted(50, 7, ..) is true. Fails on the pre-port strict
        // code (membership-only), passes after.
        let case = anchor("leniency", 100, Some(vec![50]), None, None);
        let out = AnchorOutput {
            token: 7,
            top_logits: vec![tl(7, 5.0), tl(50, 5.0 - 5e-7), tl(100, 2.0)],
        };
        let r = evaluate_anchor_case(&case, &out, VOCAB_SIZE);
        assert!(
            r.passed,
            "must pass via the per-accepted-token tie leniency"
        );
        assert_eq!(r.argmax, Some(7));

        // Control: move token 50 outside the tie tolerance -> no path accepts -> fails.
        let out_far = AnchorOutput {
            token: 7,
            top_logits: vec![tl(7, 5.0), tl(50, 5.0 - 1e-3), tl(100, 2.0)],
        };
        assert!(!evaluate_anchor_case(&case, &out_far, VOCAB_SIZE).passed);
    }

    #[test]
    fn rank_of_total_order_under_nan_does_not_panic() {
        // C7: total_cmp gives a total order; a NaN entry must not panic the sort.
        let logits = vec![tl(1, f64::NAN), tl(2, 3.0), tl(3, 1.0)];
        // total_cmp is a total order: NaN sorts highest (rank 1), then 3.0 (token 2),
        // then 1.0 (token 3). Deterministic and panic-free.
        assert_eq!(rank_of(&logits, 2), Some(2));
        assert_eq!(rank_of(&logits, 3), Some(3));
        assert!(rank_of(&logits, 99).is_none());
    }

    #[test]
    fn run_conformance_reports_failure() {
        let gates = GoldenCorrectnessGates {
            anchors: None,
            free_run: Some(vec![GoldenFreeRunCase {
                name: "fr".into(),
                prompt_tokens: vec![1, 2, 3],
                expected_tokens: vec![10, 11],
                exact_prefix_tokens: None,
            }]),
            behavior: None,
        };
        let fx = fixture_with_gates(gates);
        let mut engine = FakeEngine {
            drain_calls: 0,
            anchor_out: AnchorOutput {
                token: 0,
                top_logits: vec![],
            },
            free_run_out: vec![10, 99],
            teacher_forced_out: vec![],
        };
        let report = run_conformance(
            &mut engine,
            &fx,
            crate::constants::CORRECTNESS_STEPS,
            CorrectnessScope::Full,
            VOCAB_SIZE,
        )
        .unwrap();
        assert!(!report.passed);
        assert_eq!(report.free_run[0].first_mismatch_step, Some(1));
    }

    // --- B3: primary teacher-forced cases ---

    fn case_with_expected(name: &str, expected: Vec<Token>) -> GoldenCase {
        GoldenCase {
            name: name.into(),
            prompt_tokens: vec![1, 2, 3],
            expected_tokens: expected,
        }
    }

    fn fixture_with_cases(cases: Vec<GoldenCase>) -> GoldenFixture {
        GoldenFixture {
            model_provenance: None,
            model_type: None,
            cases,
            correctness_gates: None,
            benchmark: None,
            sha256: "0".repeat(64),
            byte_len: 0,
        }
    }

    #[test]
    fn b3_primary_cases_are_evaluated_and_can_fail() {
        // The primary `cases[]` were skipped entirely before B3. Here the engine's
        // teacher-forced predictions (30, 99, 32) diverge from the golden (30, 31, 32)
        // at step 1 with no tie cover, so the run must FAIL on the primary case —
        // proving the exam is no longer vacuous.
        let fx = fixture_with_cases(vec![case_with_expected("c1", vec![30, 31, 32])]);
        let mut engine = FakeEngine {
            drain_calls: 0,
            anchor_out: AnchorOutput {
                token: 0,
                top_logits: vec![],
            },
            free_run_out: vec![],
            teacher_forced_out: vec![
                AnchorOutput {
                    token: 30,
                    top_logits: vec![tl(30, 5.0), tl(31, 1.0)],
                },
                AnchorOutput {
                    token: 99,
                    top_logits: vec![tl(99, 5.0), tl(31, 1.0)],
                },
                AnchorOutput {
                    token: 32,
                    top_logits: vec![tl(32, 5.0)],
                },
            ],
        };
        let report =
            run_conformance(&mut engine, &fx, 3, CorrectnessScope::Full, VOCAB_SIZE).unwrap();
        assert!(!report.passed);
        assert_eq!(report.base_cases.len(), 1);
        assert_eq!(report.base_cases[0].first_mismatch_step, Some(1));
        assert_eq!(report.base_cases[0].actual_token, Some(99));
        // one drain per primary sequence.
        assert_eq!(engine.drain_calls, 1);
    }

    #[test]
    fn b3_primary_case_passes_on_exact_and_tie() {
        // step 0 exact (40==40); step 1 the engine returns 98 but 41's logit ties the
        // top within tolerance -> accepted; step 2 exact. Whole case passes.
        let fx = fixture_with_cases(vec![case_with_expected("c1", vec![40, 41, 42])]);
        let mut engine = FakeEngine {
            drain_calls: 0,
            anchor_out: AnchorOutput {
                token: 0,
                top_logits: vec![],
            },
            free_run_out: vec![],
            teacher_forced_out: vec![
                AnchorOutput {
                    token: 40,
                    top_logits: vec![tl(40, 5.0), tl(41, 4.99)],
                },
                AnchorOutput {
                    token: 98,
                    top_logits: vec![tl(98, 5.0), tl(41, 5.0 - 1e-7)],
                },
                AnchorOutput {
                    token: 42,
                    top_logits: vec![tl(42, 5.0)],
                },
            ],
        };
        let report =
            run_conformance(&mut engine, &fx, 3, CorrectnessScope::Full, VOCAB_SIZE).unwrap();
        assert!(report.passed, "exact + true tie must pass");
    }

    // --- 1a: validatedWorkerTopLogits parity (malformed/unsorted distributions) ---

    #[test]
    fn top_logits_trustworthy_mirrors_swift_validation() {
        // valid: strictly descending, lead == reported, in-vocab, unique.
        assert!(top_logits_trustworthy(
            &[tl(5, 9.0), tl(3, 8.0), tl(7, 7.0)],
            5,
            VOCAB_SIZE
        ));
        // equal logits must tie-break lowest-token-first.
        assert!(top_logits_trustworthy(
            &[tl(3, 9.0), tl(5, 9.0)],
            3,
            VOCAB_SIZE
        ));
        assert!(!top_logits_trustworthy(
            &[tl(5, 9.0), tl(3, 9.0)],
            5,
            VOCAB_SIZE
        ));
        // the review scenario: UNSORTED (2nd entry has a higher logit) -> untrustworthy.
        assert!(!top_logits_trustworthy(
            &[tl(99, 1.0), tl(31, 9.0)],
            99,
            VOCAB_SIZE
        ));
        // lead != reported token.
        assert!(!top_logits_trustworthy(
            &[tl(5, 9.0), tl(3, 8.0)],
            3,
            VOCAB_SIZE
        ));
        // more than CORRECTNESS_TOP_LOGITS entries.
        let many: Vec<TopLogit> = (0..=(CORRECTNESS_TOP_LOGITS as i64))
            .map(|i| tl(1000 - i, 100.0 - i as f64))
            .collect();
        assert!(!top_logits_trustworthy(&many, many[0].token, VOCAB_SIZE));
        // duplicate token, out-of-vocab token, non-finite logit.
        assert!(!top_logits_trustworthy(
            &[tl(5, 9.0), tl(5, 8.0)],
            5,
            VOCAB_SIZE
        ));
        assert!(!top_logits_trustworthy(
            &[tl(VOCAB_SIZE as i64, 9.0), tl(3, 8.0)],
            VOCAB_SIZE as i64,
            VOCAB_SIZE
        ));
        assert!(!top_logits_trustworthy(
            &[tl(5, f64::INFINITY), tl(3, 8.0)],
            5,
            VOCAB_SIZE
        ));
        assert!(top_logits_trustworthy(&[tl(5, 9.0)], 5, VOCAB_SIZE)); // singleton is fine.
    }

    #[test]
    fn teacher_forced_unsorted_top_logits_do_not_rescue_a_mismatch() {
        // The review's cheat: the engine reports 99 but presents [{99,1.0},{31,9.0}] so the
        // golden token 31 (higher logit) looks tie-acceptable against the first-listed 99.
        // The strict-descending check rejects the unsorted list -> strict -> 99 != 31 -> FAIL.
        let case = case_with_expected("c", vec![31, 0, 0]);
        let outputs = vec![
            AnchorOutput {
                token: 99,
                top_logits: vec![tl(99, 1.0), tl(31, 9.0)],
            },
            AnchorOutput {
                token: 0,
                top_logits: vec![tl(0, 9.0)],
            },
            AnchorOutput {
                token: 0,
                top_logits: vec![tl(0, 9.0)],
            },
        ];
        let r = evaluate_teacher_forced_case(&case, &outputs, 3, VOCAB_SIZE);
        assert!(
            !r.passed,
            "unsorted top_logits must not rescue a teacher-forced mismatch"
        );
        assert_eq!(r.first_mismatch_step, Some(0));
        assert_eq!(r.actual_token, Some(99));
    }

    #[test]
    fn anchor_unsorted_top_logits_do_not_rescue_a_mismatch() {
        // The anchor expects 50 (NOT the canonical argmax, which is 99). An unsorted
        // distribution puts a low-logit token first so 50 looks tie-close to it; the
        // strict-descending check rejects the list -> strict -> 50 is not accepted -> FAIL.
        let case = anchor("a", 50, None, None, None);
        let out = AnchorOutput {
            token: 99,
            top_logits: vec![tl(31, 0.5), tl(99, 5.0), tl(50, 0.5)],
        };
        let r = evaluate_anchor_case(&case, &out, VOCAB_SIZE);
        assert!(
            !r.passed,
            "unsorted top_logits must not rescue an anchor mismatch"
        );
    }

    #[test]
    fn run_conformance_guards_short_expected_tokens() {
        // A directly-constructed fixture whose primary case has fewer expected_tokens than
        // the requested steps must fail closed with a BenchError, not index-panic (2.1;
        // matches Swift compareTeacherForcedWithWorker's guard).
        let fx = fixture_with_cases(vec![case_with_expected("short", vec![1, 2])]);
        let mut engine = FakeEngine {
            drain_calls: 0,
            anchor_out: AnchorOutput {
                token: 0,
                top_logits: vec![],
            },
            free_run_out: vec![],
            teacher_forced_out: vec![],
        };
        let err =
            run_conformance(&mut engine, &fx, 3, CorrectnessScope::Full, VOCAB_SIZE).unwrap_err(); // steps=3 > 2 expected
        assert!(
            matches!(&err, BenchError::InvalidInput(m) if m.contains("expected_tokens") && m.contains("short")),
            "expected a bounds BenchError, got {err:?}"
        );
    }
    // --- David's blanket-10% teacher-forced mismatch budget (ruling 2026-09-02) ---

    /// Teacher-forced outputs that match `expected` everywhere except at `mismatch_indices`,
    /// where the engine returns a token no tie can rescue (a well-formed single-entry
    /// distribution that leads with the engine's own token and never lists the golden one).
    fn tf_outputs(expected: &[Token], mismatch_indices: &[usize]) -> Vec<AnchorOutput> {
        expected
            .iter()
            .enumerate()
            .map(|(i, &token)| {
                let emitted = if mismatch_indices.contains(&i) {
                    900 + i as Token
                } else {
                    token
                };
                AnchorOutput {
                    token: emitted,
                    top_logits: vec![tl(emitted, 5.0)],
                }
            })
            .collect()
    }

    #[test]
    fn budget_arithmetic_is_integer_and_exact_at_ten_percent() {
        // The SAME arithmetic the batched cohort path uses, restated at the boundary:
        // `mismatches * 1000 <= COHORT_TOKEN_TOLERANCE_PER_THOUSAND * steps`.
        assert_eq!(COHORT_TOKEN_TOLERANCE_PER_THOUSAND, 100);
        assert!(within_teacher_forced_mismatch_budget(6, 64)); // 6000 <= 6400
        assert!(!within_teacher_forced_mismatch_budget(7, 64)); // 7000 >  6400
        assert!(within_teacher_forced_mismatch_budget(1, 10)); // 1000 <= 1000, exactly 10%
        assert!(!within_teacher_forced_mismatch_budget(2, 10)); // 2000 >  1000
        assert!(within_teacher_forced_mismatch_budget(0, 0));
    }

    #[test]
    fn base_case_passes_inside_the_ten_percent_budget() {
        // 64 steps, 6 non-accepted tokens: 6 * 1000 = 6000 <= 100 * 64 = 6400 -> PASS.
        // The first mismatch step and the COUNT stay reported on the pass path.
        let expected = vec![2 as Token; 64];
        let case = case_with_expected("c64", expected.clone());
        let outputs = tf_outputs(&expected, &[2, 9, 17, 30, 41, 63]);
        let r = evaluate_teacher_forced_case(&case, &outputs, 64, VOCAB_SIZE);
        assert!(r.passed, "6 of 64 is inside the budget: {}", r.reason);
        assert_eq!(r.mismatch_count, 6);
        assert_eq!(r.compared_steps, 64);
        assert_eq!(r.first_mismatch_step, Some(2));
        assert_eq!(r.checked_steps, 64, "PASS keeps the full window");
    }

    #[test]
    fn base_case_refuses_one_mismatch_over_the_budget() {
        // 64 steps, 7 non-accepted tokens: 7 * 1000 = 7000 > 6400 -> REFUSE. The refusal
        // reports the FIRST failing step, the count, the steps, and the budget.
        let expected = vec![2 as Token; 64];
        let case = case_with_expected("c64", expected.clone());
        let outputs = tf_outputs(&expected, &[2, 9, 17, 30, 41, 55, 63]);
        let r = evaluate_teacher_forced_case(&case, &outputs, 64, VOCAB_SIZE);
        assert!(!r.passed, "7 of 64 breaks the budget");
        assert_eq!(r.mismatch_count, 7);
        assert_eq!(r.compared_steps, 64);
        assert_eq!(r.first_mismatch_step, Some(2), "FIRST failing step is kept");
        assert_eq!(r.expected_token, Some(2));
        assert_eq!(r.actual_token, Some(902));
        assert_eq!(
            r.checked_steps, 3,
            "Swift FAIL shape: first mismatch step + 1"
        );
        assert!(r.reason.contains("at step 2"), "reason: {}", r.reason);
        assert!(r.reason.contains("7 mismatches"), "reason: {}", r.reason);
        assert!(r.reason.contains("over 64 steps"), "reason: {}", r.reason);
        assert!(
            r.reason.contains("100-per-thousand"),
            "reason: {}",
            r.reason
        );
    }

    #[test]
    fn base_case_with_no_mismatch_passes_and_reports_none() {
        let expected = vec![2 as Token; 64];
        let case = case_with_expected("c64", expected.clone());
        let outputs = tf_outputs(&expected, &[]);
        let r = evaluate_teacher_forced_case(&case, &outputs, 64, VOCAB_SIZE);
        assert!(r.passed);
        assert_eq!(r.mismatch_count, 0);
        assert_eq!(r.first_mismatch_step, None);
        assert_eq!(r.reason, "");
    }

    #[test]
    fn base_case_budget_scales_with_the_case_length() {
        // A 10-step case: 1 mismatch is EXACTLY 10% and passes; 2 refuses.
        let expected = vec![2 as Token; 10];
        let case = case_with_expected("c10", expected.clone());
        let one = evaluate_teacher_forced_case(&case, &tf_outputs(&expected, &[4]), 10, VOCAB_SIZE);
        assert!(one.passed, "1 of 10 is exactly the budget: {}", one.reason);
        assert_eq!(one.mismatch_count, 1);
        let two =
            evaluate_teacher_forced_case(&case, &tf_outputs(&expected, &[4, 7]), 10, VOCAB_SIZE);
        assert!(!two.passed, "2 of 10 breaks the budget");
        assert_eq!(two.mismatch_count, 2);
        assert_eq!(two.first_mismatch_step, Some(4));
    }

    #[test]
    fn base_case_walk_continues_past_a_mismatch_so_the_count_is_complete() {
        // Teacher forcing feeds the GOLDEN token at every position, so every step stays
        // comparable after a mismatch. The walk must NOT abort at the first one, or a case
        // that breaks the budget later would be counted as a single mismatch and pass.
        let expected = vec![2 as Token; 64];
        let case = case_with_expected("c64", expected.clone());
        let indices: Vec<usize> = (0..20).collect();
        let r =
            evaluate_teacher_forced_case(&case, &tf_outputs(&expected, &indices), 64, VOCAB_SIZE);
        assert_eq!(
            r.mismatch_count, 20,
            "every step after the first is counted"
        );
        assert!(!r.passed);
    }
}
