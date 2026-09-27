//! TIMED-WINDOW DIVERGENCE REPLAY (GumbiiDigital fork, Nemotron 3.5 Lightning track, 2026-09-27).
//!
//! WHY. The timed free-run decode window is exact-matched against the golden continuation. On a
//! checkpoint whose logits come out of the `lm_head` in BF16, the reference's own top-1 and top-2
//! TIE exactly, or sit one BF16 step apart, several times in every 128-token window (16 near-ties
//! on the Nemotron track's live prompt). A legitimately faster kernel that moves the last bit of
//! those logits, or a speculative verify whose FP8 activation scales are computed over several
//! tokens at once, then commits a different, equally valid token, and the exact match fails the
//! run although the engine computed the model. Recording every tie resolution in the golden does
//! not scale: the continuation after each flip has near-ties of its own.
//!
//! WHAT. When a track fixture declares a [`TimedReplayPolicy`], a candidate's timed divergence is
//! not a verdict. benchd records the committed stream and, after the leg, REPLAYS it teacher-forced
//! through the REFERENCE engine. Every committed token from the first divergence on must be one the
//! reference itself nearly chose at that position:
//!
//! * when it is in the reference's top-8 at that position, its logit is at most `max_logit_gap`
//!   below the reference's top logit there, and
//! * at most `off_argmax_per_thousand` of the window's positions may be OFF-ARGMAX: more than one
//!   BF16 step below the top logit, or outside the reference's top-8 altogether. A tie or a
//!   one-step near-tie counts as on-argmax.
//!
//! THE NUMBERS ARE MEASURED, NOT ARGUED. Replaying the reference's OWN speculative streams (MTP-3
//! and MTP-6 on all 8 pool prompts, 2026-09-27) through its serial engine gave a worst in-top-8 gap
//! of 1.625 logits, at most 9 off-argmax positions in 129, and one token outside the top-8: its
//! multi-token verify quantizes FP8 activations over several tokens at once, which moves logits.
//! The Nemotron fixture declares 2.0 logits and 100 per thousand.
//!
//! An engine cannot meet that without computing the model: knowing which tokens are near-optimal
//! at every position requires the logits. The frozen formats and the correctness gate still hold
//! the engine to the model itself; this only stops exact-match brittleness from failing honest
//! numerics. A fixture that declares no policy keeps the exact match, unchanged.

/// The replay tolerance a track fixture declares (`timed_replay_max_logit_gap`,
/// `timed_replay_off_argmax_per_thousand`). Both are declared together or not at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimedReplayPolicy {
    /// The largest gap, in raw logits, between the reference's top logit and the committed
    /// token's logit at any replayed position.
    pub max_logit_gap: f64,
    /// How many replayed positions per thousand may be off-argmax (more than one BF16 step below
    /// the top logit).
    pub off_argmax_per_thousand: u32,
}

/// The largest `max_logit_gap` a fixture may declare. Beyond this the check stops meaning "the
/// reference nearly chose it".
pub const MAX_DECLARABLE_LOGIT_GAP: f64 = 4.0;

impl TimedReplayPolicy {
    /// Refuse a policy that measures nothing or that is too loose to mean anything.
    pub fn validate(&self) -> Result<(), String> {
        if !(self.max_logit_gap.is_finite()
            && self.max_logit_gap > 0.0
            && self.max_logit_gap <= MAX_DECLARABLE_LOGIT_GAP)
        {
            return Err(format!(
                "timed_replay_max_logit_gap must be finite, above 0 and at most \
                 {MAX_DECLARABLE_LOGIT_GAP}; got {}",
                self.max_logit_gap
            ));
        }
        if self.off_argmax_per_thousand > 1000 {
            return Err(format!(
                "timed_replay_off_argmax_per_thousand must be at most 1000; got {}",
                self.off_argmax_per_thousand
            ));
        }
        Ok(())
    }

    /// How many off-argmax positions a replay of `positions` positions may contain.
    pub fn off_argmax_budget(&self, positions: usize) -> usize {
        positions * self.off_argmax_per_thousand as usize / 1000
    }
}

/// One BF16 step (unit in the last place) at the magnitude of `x`: BF16 keeps 8 significand bits,
/// so the step is `2^(floor(log2|x|) - 7)`.
pub fn bf16_step(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return f64::MIN_POSITIVE;
    }
    2f64.powi(x.abs().log2().floor() as i32 - 7)
}

/// The reference's verdict on one committed token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionJudgement {
    /// Top logit minus the committed token's logit (0 for the argmax or an exact tie), or `None`
    /// when the token is outside the reference's top-k and its logit is unknown.
    pub gap: Option<f64>,
    /// More than one BF16 step below the top logit, or outside the top-k.
    pub off_argmax: bool,
}

/// Judge one committed `token` against the reference's `top` logits at that position. A token
/// outside the top-k is OFF-ARGMAX (the caller's budget decides); `Err` is a token inside the top-k
/// whose gap is beyond the declared maximum, or a reference that returned no logits.
pub fn judge_position(
    top: &[(i64, f64)],
    token: i64,
    policy: &TimedReplayPolicy,
) -> Result<PositionJudgement, String> {
    let max = top
        .iter()
        .map(|&(_, l)| l)
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return Err("the reference returned no finite top logit".to_string());
    }
    let Some(&(_, logit)) = top.iter().find(|&&(t, _)| t == token) else {
        return Ok(PositionJudgement {
            gap: None,
            off_argmax: true,
        });
    };
    let gap = max - logit;
    if gap > policy.max_logit_gap {
        return Err(format!(
            "token {token} is {gap:.4} logits below the reference's top logit, above the declared \
             {} maximum",
            policy.max_logit_gap
        ));
    }
    Ok(PositionJudgement {
        gap: Some(gap),
        off_argmax: gap > bf16_step(max),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICY: TimedReplayPolicy = TimedReplayPolicy {
        max_logit_gap: 2.0,
        off_argmax_per_thousand: 100,
    };

    #[test]
    fn bf16_step_is_one_unit_in_the_last_place() {
        assert_eq!(bf16_step(16.0), 0.125);
        assert_eq!(bf16_step(20.75), 0.125);
        assert_eq!(bf16_step(15.9375), 0.0625);
        assert_eq!(bf16_step(-3.0), 2f64.powi(-6));
    }

    #[test]
    fn an_exact_tie_and_a_one_step_near_tie_are_on_argmax() {
        let top = [(10, 16.0), (20, 16.0), (30, 15.875), (40, 15.0)];
        for t in [10, 20, 30] {
            let j = judge_position(&top, t, &POLICY).unwrap();
            assert!(!j.off_argmax, "token {t}: {j:?}");
        }
        let j = judge_position(&top, 40, &POLICY).unwrap();
        assert!(j.off_argmax);
        assert_eq!(j.gap, Some(1.0));
    }

    #[test]
    fn outside_the_top_is_off_argmax_and_beyond_the_gap_is_refused() {
        let top = [(10, 20.0), (20, 17.5)];
        let j = judge_position(&top, 99, &POLICY).unwrap();
        assert_eq!(
            j,
            PositionJudgement {
                gap: None,
                off_argmax: true
            }
        );
        let e = judge_position(&top, 20, &POLICY).unwrap_err();
        assert!(e.contains("above the declared"), "{e}");
    }

    #[test]
    fn the_policy_refuses_what_measures_nothing() {
        let mut p = POLICY;
        p.max_logit_gap = 0.0;
        assert!(p.validate().is_err());
        p.max_logit_gap = 4.5;
        assert!(p.validate().is_err());
        p.max_logit_gap = 2.0;
        p.off_argmax_per_thousand = 1001;
        assert!(p.validate().is_err());
        assert!(POLICY.validate().is_ok());
        assert_eq!(POLICY.off_argmax_budget(129), 12);
    }
}
