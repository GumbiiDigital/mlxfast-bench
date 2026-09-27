//! Scoring + golden-schema constants, ported verbatim from the Swift
//! `MLXFastConstants` enum (Sources/MLXFastCore/Constants.swift).
//!
//! This module is the single source of truth for the values that are triplicated
//! across the Swift codebase (MLXFastConstants, benchmark.yml, overlay-paired-timing.sh).
//! Only the scoring + golden-validation subset is ported here (this crate's scope).
//!
//! NOTE: a track scores against a baseline in ONE of two ways, and the two never mix.
//!
//! * A LIVE-CONTROL-LEG track (the fixture's `scores_against_live_control_leg`, read through
//!   [`crate::contract::scores_against_live_control_leg`]) stores NO pair at all. Its ranked run measures a
//!   serial-control leg on the reference tree on the same box, in the same job, and that live
//!   measurement is the denominator (David 2026-09-08, `docs/track-release-branches.md`). The two
//!   Qwen 3.8 125B-A6B tracks are these tracks.
//! * A STORED-PAIR track keeps its pair and its acceptance bands as ONE captured unit
//!   ([`OfficialBaseline`]) in the fixture's own
//!   `official_baseline_{prefill,decode}_seconds_per_token`, read through
//!   [`crate::contract::official_baseline`], which refuses BY NAME for a track that declares
//!   neither ([`OFFICIAL_BASELINE_PENDING`]). Values are carried at full precision to stay bit-identical
//!   with the source they were captured from. The legacy `qwen3.8-27b-mtp-v1` and
//!   `gemma4-26b-a4b-mlx-v1` tracks are these tracks.
//!
//! No number anywhere here is a placeholder: a track with no pair carries no bytes a score could
//! consume.

// --- Scoring subset (MLXFastConstants.score*, *BandTolerance, officialBaseline*) ---

// NONE OF THE FOUR VALUES BELOW IS A SCORED INPUT ANY MORE (STEP 3 of the constants→contract
// migration, David 2026-09-15). Every scored value comes from the `--contract` track fixture, and
// every resolver refuses a fixture that declares none. What is left here is the pair the UNSCORED
// calibration path carries — `calibrate-baseline` writes a capture record and seals no score, so
// it needs a well-formed `ScoringInputs` that reaches no verdict — plus the reference's own
// spellings, which the score proptests check the composite's algebra against.

/// `MLXFastConstants.scoreDecodeWeight`. NOT the weight any run scores under: a scored run reads
/// `score_decode_weight` from its track fixture and refuses absence
/// ([`crate::contract::scoring_weights`]). This is [`crate::score::ScoringWeights::DEFAULT`]'s
/// value, which rides only on the unscored calibration path.
pub const SCORE_DECODE_WEIGHT: f64 = 0.75;
/// `MLXFastConstants.scorePrefillWeight`. The other half of [`SCORE_DECODE_WEIGHT`], and not a
/// scored input for the same reason.
pub const SCORE_PREFILL_WEIGHT: f64 = 0.25;

/// `MLXFastConstants.scoreDecodeSpeedupFloor` (David ruling 2026-09-09: 0.95 decode AND 0.95
/// prefill, enforced, configurable per project).
///
/// NOT a floor any run is judged against: a scored run reads `decode_speedup_floor` from its track
/// fixture and refuses a fixture that declares none, and STEP 3 made `--contract` required on every
/// `benchd iterate` mode — so there is no "local run without a fixture" left for this to be the
/// default of. It survives as [`crate::score::SpeedupFloors::DEFAULT`]'s value on the UNSCORED
/// calibration path, which seals no score for a floor to gate.
pub const SCORE_DECODE_SPEEDUP_FLOOR: f64 = 0.95;
/// `MLXFastConstants.scorePrefillSpeedupFloor`. Not a scored input, exactly as
/// [`SCORE_DECODE_SPEEDUP_FLOOR`].
pub const SCORE_PREFILL_SPEEDUP_FLOOR: f64 = 0.95;

// --- qwen-mtp-paired-decode-only scoring (track qwen3.8-27b-mtp-v1) ---
//
// The authoritative paired score for the MTP spec-decode track (benchmark.json `scoring`
// mode `qwen-mtp-paired-decode-only`, mirrored in the track fixture
// qwen3_8_27b_mtp_track.json `scoring_semantics`). This is DECODE-ONLY and serial-anchored
// (serial control = 1.0, no normalization): per prompt the raw ratio is
// `mean(serial depth-0 decode s/tok) / mean(candidate decode s/tok)` over that prompt's
// accepted pairs, and the published score is the EVEN-N median of the per-prompt raw ratios.
// These constants REPLACE the generic 0.95 decode/prefill speedup floors for the paired
// score; the generic `SCORE_*_SPEEDUP_FLOOR` path (ds^0.75·ps^0.25) is untouched.

/// Paired decode-only submission floor on the RAW median (ranked workflow
/// `MLXFAST_QWEN_MTP_DECODE_SPEEDUP_FLOOR`). Operator decision 2026-08-14: 0.90 — "do not
/// regress serial by more than 10%". A candidate that cannot beat serial should stop drafting
/// and take 1.0. Below this the run floor-fails (score null).
///
/// #117 — this floor governs the `free_run_v1_1` series TOO, by David's ruling on #109 (comment
/// 5353123259, 2026-08-20): "floor stays 0.90, no sub-floor bootstrap governance built — the stock
/// free-run median landing below 0.90 'shouldn't happen; ignore the case.'" The ruling is the
/// AUTHORITY there; the ~0.935 calibration that justified 0.90 for the teacher-forced series is
/// NOT inherited into the free-run series (#109 comment 5350423826, §5). Sealed on the free-run
/// measure-job path as `measure_job::FREE_RUN_DECODE_SPEEDUP_FLOOR`, which aliases this constant so
/// the seal and the ranked overlay floor cannot diverge.
pub const QWEN_MTP_DECODE_SPEEDUP_FLOOR: f64 = 0.90;
/// Paired decode-only ceiling on the RAW median (ranked workflow
/// `MLXFAST_QWEN_MTP_DECODE_SPEEDUP_CEILING`). Raised 3.0→5.0 by operator decision 2026-08-17.
/// Above this the median is a measurement fault or an escape and the run ceiling-fails.
pub const QWEN_MTP_DECODE_SPEEDUP_CEILING: f64 = 5.0;
/// Per-PAIR plausibility bound (box wrapper `MAX_PLAUSIBLE_PUBLISHED_SPEEDUP` /
/// `QMTP_MAX_PLAUSIBLE`): any single pair ratio above this is rejected before aggregation.
/// Raised 5.0→8.0 by operator decision 2026-08-17 so it stays strictly looser than the 5.0
/// median ceiling.
pub const QWEN_MTP_PER_PAIR_RATIO_BOUND: f64 = 8.0;

/// The timed-run acceptance bands (`MLXFastConstants.prefillBand{Up,Down}Tolerance`,
/// `decodeBand{Up,Down}Tolerance`): multiplicative tolerances around the official baseline pair.
/// They are FIXED LITERALS carried inside [`OfficialBaseline`] and pending together with the pair
/// (they are NOT re-derived from a capture CV at seal time — the scored path reads exactly these).
///
/// Band SHAPE for the single-leg MTP-on-the-timed-leg regime (David ruling): prefill ±5% symmetric;
/// decode +2% UP; decode DOWN-band DISABLED. The values land at calibration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcceptanceBands {
    pub prefill_up_tolerance: f64,
    pub prefill_down_tolerance: f64,
    pub decode_up_tolerance: f64,
    pub decode_down_tolerance: f64,
    /// Whether the decode band enforces its LOWER bound (`value < reference*(1-down_tolerance)` =
    /// "improvement too large"). `true` for a normal two-sided band; `false` for the MTP timed leg,
    /// where the ruling is: decode DOWN-band DISABLED. MTP spec-decode decode is legitimately much
    /// faster than the serial baseline, so Laguna's "-5% improvement too large" lower guard would
    /// WRONGLY fail a healthy MTP run — the 0.95 decode speedup FLOOR is the only lower guard the
    /// decode axis needs. When `false`, [`crate::score::evaluate_timed_run`]/`check` skip the
    /// decode lower-bound test and keep the decode UP bound. `decode_down_tolerance` is then inert.
    pub decode_down_enabled: bool,
    /// Whether the prefill band enforces its LOWER bound. `false` on the paired design: the
    /// serial-control leg is measured live in the same run and carries its own health band
    /// (`benchd::baseline::check_band`), so a candidate prefill far below the control's is a
    /// faster engine, not a lottery, and the -tolerance% "improvement too large" guard would refuse
    /// exactly the submissions the track exists to reward. The prefill UP bound stays.
    pub prefill_down_enabled: bool,
}

/// The official serial baseline for the LOCAL-ITERATE / OFFICIAL scoring denominator
/// (`MLXFastConstants.officialBaseline{Prefill,Decode}SecondsPerToken`, #127 ruling: the local
/// legs score against these, so a stale or foreign value is a live scoring defect) together with
/// its acceptance bands. Captured on the ranked box per docs/qwen38-125b-a6b-baseline-capture.md
/// and mirrored bit-identical from the reference engine's `Constants.swift`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OfficialBaseline {
    pub prefill_seconds_per_token: f64,
    pub decode_seconds_per_token: f64,
    pub bands: AcceptanceBands,
}

/// The engine PLATFORM a track runs on. ONE bench tree serves both engines (David 2026-08-27:
/// a shared benchd for cuda and mlx), so every platform-specific fact — the reference model,
/// the official baseline and its pending sentinel — is keyed by this enum and resolved from the
/// TRACK ID (`{model}{ver}-{params}-{platform}-v{N}`), never from a branch name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Mlx,
    Cuda,
}

/// The track-id token for [`Platform::Mlx`].
pub const PLATFORM_KEY_MLX: &str = "mlx";
/// The track-id token for [`Platform::Cuda`].
pub const PLATFORM_KEY_CUDA: &str = "cuda";

/// The MLX (Mac) track's local pre-timing cool-gate temperature (C). A Mac idles well below
/// this, so the gate blocks only a genuinely warm GPU.
pub const COOL_GATE_TEMP_C_MLX: f64 = 40.0;
/// The CUDA (GB10) track's local pre-timing cool-gate temperature (C). GumbiiDigital fork,
/// 2026-09-26 (Nemotron 3.5 Lightning track): 60 C, raised from upstream's 50 C.
///
/// WHY. On driver 580.159.03 ANY CUDA context holds the GB10 in P0 (~10.7 W, 2.4 GHz at zero
/// utilization), so a leg's loaded resident plateaus at 49–52 C on a quiet box. A 50 C gate
/// therefore sat on the plateau: every measured pass was admitted at exactly 50.0 C after
/// 70–390 s of idle, and 4 of 7 boxes were refused ("not cooling down") stuck at 51.0 C.
///
/// THE THROTTLE LIMIT IS NOT 55 C. `nvidia-smi -q` reports `GPU T.Limit Temp` as the MARGIN below
/// the maximum operating temperature (53 C of headroom at 43 C, so the limit is ~96 C), and the
/// box's `SW/HW Thermal Slowdown` counters stayed at 0 us across runs that reached 64 C; the only
/// active limiter was the SW power cap. 60 C keeps a 30+ C margin below any thermal slowdown and
/// still refuses a box that stays genuinely hot.
pub const COOL_GATE_TEMP_C_CUDA: f64 = 60.0;

impl Platform {
    /// Every platform, for tests and mirrors that must cover the whole table.
    pub const ALL: [Platform; 2] = [Platform::Mlx, Platform::Cuda];

    /// The platform's track-id token.
    pub fn key(self) -> &'static str {
        match self {
            Platform::Mlx => PLATFORM_KEY_MLX,
            Platform::Cuda => PLATFORM_KEY_CUDA,
        }
    }

    /// Resolve the platform from a track id of the canonical shape
    /// `{model}{ver}-{params}-{platform}-v{N}`: the token before the trailing `v{N}` segment.
    /// Anything else refuses by name — a track id that names no platform can key no fact.
    pub fn from_track_id(track_id: &str) -> Result<Platform, String> {
        let segments: Vec<&str> = track_id.trim().split('-').collect();
        let platform = match segments.as_slice() {
            [.., platform, version]
                if version.len() > 1
                    && version.starts_with('v')
                    && version[1..].bytes().all(|b| b.is_ascii_digit()) =>
            {
                *platform
            }
            _ => {
                return Err(format!(
                    "track_id {track_id:?} does not end in `-{{platform}}-v{{N}}`, so it names no \
                     platform; the platform keys the reference model and the official baseline \
                     and must be resolvable from the track id"
                ))
            }
        };
        Platform::ALL
            .into_iter()
            .find(|p| p.key() == platform)
            .ok_or_else(|| {
                format!(
                    "track_id {track_id:?} names platform {platform:?}, which is not one of \
                     {:?}",
                    Platform::ALL.map(Platform::key)
                )
            })
    }

    /// The platform's local pre-timing GPU cool-gate temperature (C). Finding R21 originally
    /// froze this at a single, non-parameterizable 40 C constant; David 2026-08-30 ruled that
    /// rigidity out of scope under his "no adversarial hardening" stance (the GB10 idle of
    /// 40–43 C against a 40 C gate makes the gate unusable). The threshold is therefore now a
    /// trusted PER-PLATFORM value keyed here — like every other platform fact — never a
    /// contract/candidate-supplied input. See [`COOL_GATE_TEMP_C_MLX`] / [`COOL_GATE_TEMP_C_CUDA`].
    pub fn cool_gate_temp_c(self) -> f64 {
        match self {
            Platform::Mlx => COOL_GATE_TEMP_C_MLX,
            Platform::Cuda => COOL_GATE_TEMP_C_CUDA,
        }
    }
}

/// `MLXFastConstants.publicDiagnosticSignificantFigures`
pub const PUBLIC_DIAGNOSTIC_SIGNIFICANT_FIGURES: u32 = 2;

// --- Timed-window liveness (RunTimeout budget) ---

/// H3 (cycle-3) — RunTimeout liveness safeguard (PROTOCOL-v1.1 §2.2/§4). benchd arms a wall-clock
/// timeout on the timed decode round-trips equal to `N × band-ceiling × margin`; this is the fixed
/// `margin` slack factor. It is a LIVENESS bound, never an input to the score — a passing run
/// finishes well inside the budget; the margin only exists so normal jitter never trips it. On
/// timeout benchd raises `RunTimeout`, discards the session (fail-closed), and the pair fails.
pub const RUN_TIMEOUT_MARGIN: f64 = 4.0;
/// H3 (cycle-3) — fallback per-token latency `band-ceiling` (seconds-per-token) for the RunTimeout
/// budget when no `BASELINE_CALIBRATION` is available (e.g. the free-run path or `BASELINE_BAND_ENFORCE=0`).
/// With calibration present, the band-ceiling is `calibration.serial_mean × calibration.band_high`
/// (the upper acceptance/latency band bound); absent it, this deliberately-generous constant
/// bounds a hung engine without ever tripping a healthy run.
///
/// #127 — this used to ALIAS the official DECODE baseline (now one pair per `track_id`),
/// which was carrying the RETIRED `mlxfast-challenge-dev` fork's Gemma-era value. Correcting that constant to the
/// reference's Qwen value (below) would have tightened this liveness ceiling ~9.6×, to BELOW the
/// decode seconds-per-token a healthy candidate actually measures (the §8 window measured
/// ~0.0347 s/token against a 0.01386 s/token reference baseline — a candidate is SLOWER than the
/// reference-runner baseline, which is the whole point of the speedup denominator). A liveness
/// bound that a passing run trips is not a liveness bound, so the ceiling keeps its own literal:
/// numerically unchanged, no longer coupled to a scoring denominator it never meant to track.
pub const RUN_TIMEOUT_DEFAULT_BAND_CEILING_SECONDS_PER_TOKEN: f64 = 0.1336139485703125;

/// The `track_id` this RELEASE BRANCH serves (`docs/track-release-branches.md`): the
/// grandfathered qwen 3.8 27B MLX track, which runs on `main` and seals `qwen3.8-27b-mtp-v1`.
///
/// It is the key every per-track fact in this module is looked up by. A new track cuts its own
/// release branch and sets its own value here; the branch that does so gets NO baseline until its
/// `--contract` fixture declares one, because the lookup refuses by name instead of falling back
/// (see [`crate::contract::official_baseline`]).
pub const TRACK_ID: &str = "qwen3.8-27b-mtp-v1";

/// The two-sided acceptance-band shape every track carried before the bands moved INSIDE
/// [`OfficialBaseline`]. They were four global constants
/// (`MLXFastConstants.prefillBand{Up,Down}Tolerance` / `decodeBand{Up,Down}Tolerance`), which is
/// exactly the defect the per-track table exists to fix — so they survive as ONE named band shape
/// that the tracks calibrated under them name explicitly, never as a fallback anything inherits.
pub const PREFILL_BAND_UP_TOLERANCE: f64 = 0.03;
/// See [`PREFILL_BAND_UP_TOLERANCE`].
pub const PREFILL_BAND_DOWN_TOLERANCE: f64 = 0.03;
/// See [`PREFILL_BAND_UP_TOLERANCE`].
pub const DECODE_BAND_UP_TOLERANCE: f64 = 0.01;
/// See [`PREFILL_BAND_UP_TOLERANCE`].
pub const DECODE_BAND_DOWN_TOLERANCE: f64 = 0.025;

/// The EXACT-MATCH name of the state "this track has no captured official baseline". It is a
/// NAME, never a value: nothing numeric stands in for an uncaptured pair, and every refusal
/// quotes this string so an operator can grep for the one condition that stopped the run.
pub const OFFICIAL_BASELINE_PENDING: &str = "OFFICIAL-BASELINE-PENDING-CAPTURE";

/// The ADMISSION rule for a pair entering a fixture's
/// `official_baseline_{prefill,decode}_seconds_per_token`: the maximum SAMPLE
/// coefficient of variation, in percent, that a calibration's legs may show on EITHER axis.
///
/// David's rule "baselines match the official measurement path": the pinned pair is the mean of N
/// legs measured by the scored path itself (fresh serve + warm-up leg, then the timed legs), and a
/// spread wider than this says the box was not quiet enough for the mean to describe it. It is a
/// FIXED value, not a flag: a calibration that could loosen its own gate proves nothing.
pub const CALIBRATION_MAX_CV_PERCENT: f64 = 1.0;

/// The EXACT-MATCH name of the refusal "this calibration's legs are too noisy to pin a pair".
/// A NAME, so an operator can grep for the one condition that stopped the calibration.
pub const CALIBRATION_CV_EXCEEDED: &str = "CALIBRATION-CV-EXCEEDED";

/// The captured official baseline of every track this tree scores. ONE pair per `track_id`.
///
/// NOT a resolution surface. [`crate::contract::official_baseline`] is the ONE accessor: it is the
/// only form that
/// refuses an uncaptured track by name. This table and `official_baseline_declared` are private
/// to this module so no caller can read a pair out of them and skip that refusal — the guarded
/// form is the only form there is.
///
/// A track is in this table only once its pair is CAPTURED on that track's own benchmark
/// hardware. A track that is ABSENT is [`OFFICIAL_BASELINE_PENDING`], and so is a track whose
/// entry is `None` — the platform constants' own pending state, carried here unchanged rather
/// than flattened, so an absent row and a `None` row can never disagree about whether a pair
/// exists. Neither shape is a placeholder a new track could inherit a number
/// from. This is the whole point of the table: the pair used to be two GLOBAL constants, which
/// a new track silently scored against.
///
/// The THIRD leg of the `constant≡contract≡env` rule: the track a run DECLARES must be the track
/// this tree serves ([`TRACK_ID`]).
///
/// The other two legs already agree with each other — `measure_job::resolve_track_id` refuses an
/// env/`--contract` disagreement — but both could name a track this tree cannot measure. The
/// baseline table is per-track, so a `main`-built benchd driven by another track's contract would
/// seal THIS track's baseline pair under THAT track's `track_id`, and every later cross-check
/// (commit, weights hash) would still agree. Nothing downstream can detect that: the pair and the
/// name are both internally consistent, they just describe different tracks.
///
/// So the run refuses here, by name, naming BOTH values.
///
/// Not every path has a declared track id to fence. `benchd iterate` (local legs and the
/// official run) takes no `--contract` and reads no track env, so [`TRACK_ID`] is its ONLY source
/// and there is nothing to cross-check — do not invent a second source there.
pub fn enforce_declared_track(declared_track_id: &str) -> Result<(), String> {
    let declared = declared_track_id.trim();
    if declared == TRACK_ID {
        return Ok(());
    }
    Err(format!(
        "track_id fence: this benchd tree serves track {TRACK_ID:?} but the run declares track \
         {declared:?}. The workflow-declared track id must be ONE value (constant≡contract≡env), \
         and the official baseline pair is per-track — measuring or sealing {declared:?} with \
         {TRACK_ID:?}'s constants would mis-attribute the baseline. Run this track on its own \
         release branch (docs/track-release-branches.md); refusing"
    ))
}

/// The EXACT-MATCH name of the refusal "this track does not run the PAIRED flow". A NAME, so an
/// operator can grep for the one condition that stopped a `measure-job` / `overlay-timing` run.
pub const PAIRED_FLOW_RETIRED_FOR_TRACK: &str = "PAIRED-FLOW-RETIRED-FOR-TRACK";

/// The PAIRED-FLOW entry fence: a track whose sole scored path is the single leg refuses
/// `measure-job` / `overlay-timing` BY NAME, naming the track and the path it must use instead.
///
/// Called at the entry of both flow-B verbs, BEFORE [`enforce_declared_track`], so the specific
/// message ("this track scores through the single leg") reaches the operator rather than the
/// generic tree-serves-another-track fence.
///
/// Pure and total over the RESOLVED verdict, so the message lives in one place whether the verdict
/// came from the fixture or from the table.
pub fn refuse_retired_paired_flow(track_id: &str, retired: bool) -> Result<(), String> {
    if !retired {
        return Ok(());
    }
    let declared = track_id.trim();
    Err(format!(
        "{PAIRED_FLOW_RETIRED_FOR_TRACK}: track_id {declared:?} scores through the SINGLE-LEG \
         path only (`benchd iterate --mode official`, MTP on the timed leg against the pinned \
         per-platform baseline). The paired measure-job/overlay-timing seam is retired for this \
         track and will not measure or seal it; refusing"
    ))
}

// --- Scored regime, one per track ---

/// WHAT ONE TRACK SCORES: the batch size its scored point is measured at, and the two exponents
/// that combine the two gain axes into the published composite.
///
/// The composite is `prefill_gain ^ prefill_gain_exponent * decode_gain ^ decode_gain_exponent`.
/// An exponent of exactly `0.0` means the axis carries NO WEIGHT — the track does not score it,
/// and the machinery that certifies it is not armed (see
/// [`crate::prefill_window::certify_prefill_window`]).
///
/// DECLARED, NOT YET COMPUTED: [`crate::score::composite_score`] has no production call site. The
/// declaration states what a track scores and ARMS the prefill certification; the published figure
/// still comes from [`crate::score::score_paired_decode_only`].
///
/// RAISING A PREFILL EXPONENT IS GATED. Certification binds the SUM of the two window halves, not
/// where the work sits inside them, so an engine that defers seed-prefill work past
/// `free_decode_begin` inflates the prefill gain while every check still passes. No track may
/// declare a nonzero `prefill_gain_exponent` on `main` until a work-placement invariant exists —
/// `docs/scored-regime-and-prefill-window.md` §3 states the two forms that would close it.
///
/// The three numbers are ONE declaration — a batch size without its exponents describes no score
/// — so they are one value here and can never be half-replaced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredRegime {
    /// THE BATCH SIZE the scored point is measured at, declared by the track fixture (David
    /// 2026-09-15: batching is part of the configuration). `1` selects the SINGLE-STREAM measured
    /// path; any greater width selects the BATCHED COHORT path, whose series tag is assigned with
    /// the width at `benchd::measure_job::ScoredBatchPoint::certify`.
    pub scored_batch_size: usize,
    /// The exponent on the prefill gain. `0.0` ⇒ prefill is not scored on this track.
    pub prefill_gain_exponent: f64,
    /// The exponent on the decode gain.
    pub decode_gain_exponent: f64,
}

impl ScoredRegime {
    /// True when the prefill axis carries NONZERO weight in the composite, i.e. the track's score
    /// moves when the prefill window moves. This is the ARMING predicate for prefill-window
    /// certification: a track whose prefill exponent is exactly `0.0` measures the window for the
    /// record and enforces nothing on it, because no enforcement could protect a number that is
    /// not an input to the score.
    ///
    /// A NEGATIVE or non-finite exponent is not "zero weight" — it is a malformed declaration, and
    /// it reads as ARMED here so the certification refuses rather than silently disarming. The
    /// declaration itself is checked by `declared_regimes_are_well_formed`.
    pub fn prefill_is_scored(&self) -> bool {
        self.prefill_gain_exponent != 0.0
    }
}

/// The EXACT-MATCH name of the state "this track has not declared what it scores". A NAME, never a
/// value: no exponent pair stands in for an undeclared regime, and every refusal quotes this
/// string so an operator can grep for the one condition that stopped the run.
pub const SCORED_REGIME_PENDING: &str = "SCORED-REGIME-PENDING-DECLARATION";

// --- Golden-schema validation subset (MLXFastConstants.*) ---

/// The required golden `model_type` (Swift `QwenRuntime.requiredGoldenModelType`).
/// benchd's golden loader requires it exactly, matching the Swift benchmark/correctness
/// path — a golden without it, or with a different value, is rejected byte-for-byte as Swift
/// does. This is a bench-core-level identity fact (not a CLI detail), single-sourced here so
/// the loader and any consumer share one definition and it falls under the loader-parity
/// corpus.
///
/// NAMING: `qwen4_exp_text` is the model's INTERNAL ARCHITECTURE id as the weights declare it —
/// it is NOT a track name and does NOT track a release version. Track names are the `track_id`
/// strings in `docs/track-release-branches.md` (`{model}{ver}-{params}-{platform}-v{N}`); this
/// constant is pinned to the tower this tree loads. Do not "correct" it to match a track
/// version.
///
/// PER-TRACK, RESOLVED. The identity is NO LONGER per-BRANCH: the `--contract` fixture declares it
/// (`golden_model_type`, `vocab_size`, `num_hidden_layers`, `seed_tokens`), the way it declares the
/// baseline pair, and
/// [`crate::golden::load_golden_fixture`] takes the resolved [`TrackModelIdentity`]. One `main`
/// therefore loads a golden of ANY declared track, and a golden whose `model_type` is another
/// track's is refused.
///
/// This constant SURVIVES as the 125B row's value for the sites where a compile-time value is
/// unavoidable — test fixtures, the 125B-generated loader-parity / fuzz corpora, and the
/// `record-correctness-golden` recorder, which authors 125B goldens. It is NOT a fallback: no
/// production path reads it, they read [`crate::contract::model_identity`].
pub const REQUIRED_GOLDEN_MODEL_TYPE: &str = "qwen4_exp_text";

/// `MLXFastConstants.vocabSize`, read off the pinned checkpoint's own `config.json`. Was
/// `262_144` — the gemma vocabulary; the tape/golden token-range validation would have
/// admitted token ids in `248_320..262_144` that this model cannot emit.
///
/// LOCKSTEP HAZARD: this bound exists in TWO copies that must move together — the engine's
/// `MLXFastConstants.vocabSize`, which the reference-tape recorder applies as its emit-time
/// pre-check, and THIS constant, which the benchd tape/golden loaders apply at load time. If
/// they diverge, the recorder's fail-early guarantee is defeated: a token the recorder happily
/// emits is refused only later, at benchd load. benchd was the stale copy when the two last
/// diverged.
pub const VOCAB_SIZE: usize = 248_320;

/// The MODEL IDENTITY of one track: the facts a golden, a timed-prompt tape and the sealed
/// audit metrics must all agree with. One row per `track_id`.
///
/// It exists for the same reason [`OfficialBaseline`] does. These four values used to be
/// GLOBAL constants ([`REQUIRED_GOLDEN_MODEL_TYPE`], [`VOCAB_SIZE`], `NUM_HIDDEN_LAYERS` in
/// `benchd::iterate`, and the seed-length constants), so the tree carried exactly ONE model
/// identity while the baseline table carried four tracks: a golden of any other track was
/// refused BY THIS TREE rather than by its own track's rule, and the only way to run a second
/// track was to cut a release branch and re-pin the constants there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackModelIdentity {
    /// The `model_type` this track's goldens declare (Swift `QwenRuntime.requiredGoldenModelType`).
    /// The model's INTERNAL ARCHITECTURE id as the weights declare it — never a track name and
    /// never a release version.
    ///
    /// OWNED (`String`), and that is what took `Copy` from this struct. A track fixture supplies
    /// the string, so it cannot be a `&'static str`: an owned string is the simplest shape that
    /// lets fixture bytes be the source, and the consumers all take `&TrackModelIdentity` or clone
    /// it explicitly. The alternatives — `Cow<'static, str>`, a `'a` lifetime parameter, or
    /// interning into a leaked `&'static str` — each buy `Copy` back at the price of a signature
    /// or a leak, and David ruled for the simplest shape (2026-09-15).
    pub golden_model_type: String,
    /// `MLXFastConstants.vocabSize` for this track's checkpoint: the token-id bound the golden and
    /// tape loaders apply (`0..<vocab_size`) and the bound the conformance path judges worker
    /// top-logit distributions against.
    ///
    /// LOCKSTEP HAZARD (unchanged by the move to a table): this bound exists in TWO copies that
    /// must move together — the engine's `MLXFastConstants.vocabSize`, which the reference-tape
    /// recorder applies as its emit-time pre-check, and this one, which benchd applies at load
    /// time. If they diverge, a token the recorder happily emits is refused only later, at load.
    pub vocab_size: usize,
    /// `MLXFastConstants.numHiddenLayers` — the checkpoint's decoder-layer count. Sealed as
    /// `metrics.num_layers`, a DETERMINISTIC parity field (benchd and the Swift score.json must
    /// agree); it feeds no score, floor or band.
    pub num_hidden_layers: i64,
    /// The track's SEED LENGTH: `MLXFastConstants.correctnessPromptTokens`,
    /// `benchmarkPrefillPromptTokens` and `benchmarkDecodeSeedTokens`, which are ONE value.
    /// David's 2026-08-24 seed-length ruling ("Seed becomes 1024") moved all three together, and
    /// every lineage carries them equal — 1024 on the Gemma 4 and Qwen 3.8 125B-A6B tracks, 512
    /// on the grandfathered Qwen 3.8 27B track, which was cut before the ruling.
    pub seed_tokens: usize,
}

impl TrackModelIdentity {
    /// The identity of one track, from its four values.
    ///
    /// The values come from the track fixture (`bench_core::contract::model_identity`); this is
    /// only the constructor, so a caller that has the four numbers does not have to spell the
    /// `String` conversion at every site.
    pub fn new(
        golden_model_type: &str,
        vocab_size: usize,
        num_hidden_layers: i64,
        seed_tokens: usize,
    ) -> TrackModelIdentity {
        TrackModelIdentity {
            golden_model_type: golden_model_type.to_string(),
            vocab_size,
            num_hidden_layers,
            seed_tokens,
        }
    }
}

/// The EXACT-MATCH name of the state "this track declares no model identity". A NAME, never a
/// value: nothing stands in for an undeclared identity, and the refusal quotes this string so an
/// operator can grep for the one condition that stopped the load.
pub const MODEL_IDENTITY_UNDECLARED: &str = "MODEL-IDENTITY-UNDECLARED-FOR-TRACK";

/// The EXACT-MATCH name of the state "this track declares no measurement window". A NAME, never a
/// value: a window nothing declared is a window nothing can judge.
pub const WINDOW_SHAPE_UNDECLARED: &str = "WINDOW-SHAPE-UNDECLARED-FOR-TRACK";

/// The EXACT-MATCH name of the state "this track declares no acceptance band shape". A NAME, never
/// a value: the band is what gates the timed run, so an absent one is a refusal, not a default.
pub const ACCEPTANCE_BANDS_UNDECLARED: &str = "ACCEPTANCE-BANDS-UNDECLARED-FOR-TRACK";

/// The model identity of every track this tree can load a golden for. ONE row per `track_id`.
///
/// NOT a resolution surface, and private for that reason: [`crate::contract::model_identity`] is
/// the ONE accessor,
/// and it is the only form that refuses an undeclared track BY NAME. A caller that could read a
/// row out of this table would be able to skip that refusal.
///
/// PROVENANCE — each row is the value that track's own release branch carries, read from git
/// history rather than inferred:
///
///   * `qwen3.8-27b-mtp-v1` — `main` before the reconverge merge (`73c6b30`):
///     `REQUIRED_GOLDEN_MODEL_TYPE = "qwen3_5_text"`, `VOCAB_SIZE = 248_320`,
///     `CORRECTNESS_PROMPT_TOKENS = 512` (pre-ruling seed), and `NUM_HIDDEN_LAYERS = 64` in
///     `crates/benchd/src/iterate.rs`.
///   * `gemma4-26b-a4b-mlx-v1` — branch `gemma4-26b-a4b-mlx-v1`: `"gemma4_text"`,
///     `VOCAB_SIZE = 262_144`, `CORRECTNESS_PROMPT_TOKENS = 1_024`, `NUM_HIDDEN_LAYERS = 64`.
///   * `qwen3.8-125b-a6b-{mlx,cuda}-v1` — branch `qwen3.8-125b-a6b-v1`, which is what `main`
///     carries today: `"qwen4_exp_text"`, `VOCAB_SIZE = 248_320`,
///     `CORRECTNESS_PROMPT_TOKENS = 1_024`, `NUM_HIDDEN_LAYERS = 48` (48 decoder layers read off
///     BOTH pinned checkpoints' `config.json`; `layer_types` = 12 x [3 linear_attention,
///     1 full_attention]). The two platforms share ONE checkpoint architecture, so they share a
///     row value; they are listed separately because the key is the track, not the model.
///
/// `MLXFastConstants.correctnessSteps`
pub const CORRECTNESS_STEPS: usize = 64;
/// `MLXFastConstants.correctnessPromptTokens`
///
/// 1024 (was 512): David's 2026-08-24 seed-length ruling for the Gemma 4 track — "Seed becomes
/// 1024". The decode window is unchanged ([`BENCHMARK_DECODE_STEPS`] stays 128); golden shape
/// becomes 1024 `prompt_tokens` + 129 `expected_tokens` (seed next-token + 128 checked steps).
/// The 1024-token versions of the hidden pool prompts must be uploaded and referenced from the
/// Gemma benchmark branch, and every golden regenerated at the new seed, before scoring arms.
pub const CORRECTNESS_PROMPT_TOKENS: usize = 1_024;
/// `MLXFastConstants.correctnessTopLogits`
pub const CORRECTNESS_TOP_LOGITS: usize = 8;
/// `MLXFastConstants.correctnessLogitTieTolerance` — the default top-logit delta the
/// anchor rank/delta path uses when a case sets `max_expected_rank` but no explicit
/// `max_top_logit_delta` (Swift `anchor.maxTopLogitDelta ?? correctnessLogitTieTolerance`).
pub const CORRECTNESS_LOGIT_TIE_TOLERANCE: f64 = 1e-6;

/// `MLXFastConstants.correctnessMaxAnchorContextTokens`
pub const CORRECTNESS_MAX_ANCHOR_CONTEXT_TOKENS: usize = 1_024;
/// `MLXFastConstants.correctnessMaxFreeRunSteps`
pub const CORRECTNESS_MAX_FREE_RUN_STEPS: usize = 256;
/// `MLXFastConstants.correctnessMaxBehaviorPromptTokens`
pub const CORRECTNESS_MAX_BEHAVIOR_PROMPT_TOKENS: usize = 2_048;
/// `MLXFastConstants.correctnessMaxBehaviorSteps`
pub const CORRECTNESS_MAX_BEHAVIOR_STEPS: usize = 64;

/// `MLXFastConstants.benchmarkPrefillPromptTokens`
///
/// 1024 (was 512): moves with [`CORRECTNESS_PROMPT_TOKENS`] under the 2026-08-24 seed-length
/// ruling — the timed prefill leg is now 8 x 1024 tokens per cohort. Any baseline/calibration
/// value derived at the 512-token prefill window is invalidated and must be re-derived.
pub const BENCHMARK_PREFILL_PROMPT_TOKENS: usize = 1_024;
/// `MLXFastConstants.benchmarkDecodeSeedTokens`
///
/// 1024 (was 512): moves with [`CORRECTNESS_PROMPT_TOKENS`] under the 2026-08-24 seed-length
/// ruling.
pub const BENCHMARK_DECODE_SEED_TOKENS: usize = 1_024;
/// `MLXFastConstants.benchmarkDecodeSteps`
pub const BENCHMARK_DECODE_STEPS: usize = 128;
/// `MLXFastConstants.localIterateBenchmarkDecodeSteps` — the checked decode window the
/// participant edit loop (`--local-iterate`) uses.
///
/// This is `benchmarkDecodeSteps` on the reference tree, NOT a shorter window: the reference
/// states the reason inline — "Local iterate charges the same seed prefill as the
/// official decode window" ([`BENCHMARK_DECODE_SEED_TOKENS`], 1024 since the 2026-08-24
/// seed-length ruling; 512 at the time of the quoted reference), "so it must use the same
/// denominator to produce a comparable decode seconds-per-token estimate"
/// (`mlxfast-qwen-38-27b-mtp-engine/Sources/MLXFastCore/Constants.swift@6279c7a:197-201`).
///
/// It was ported as `16` from the retired Laguna/DFlash fork
/// (`mlxfast-challenge-dev/Sources/MLXFastCore/Constants.swift:71`), which is the tree the
/// original local-iterate port read; the Qwen 3.8 engine — this challenge's reference —
/// carries `= benchmarkDecodeSteps`. Same class of stale-reference drift as the golden
/// `model_provenance` row (#112/#114): benchd was LOOSER than the reference because it held
/// the old fork's value.
pub const LOCAL_ITERATE_BENCHMARK_DECODE_STEPS: usize = BENCHMARK_DECODE_STEPS;

/// `MLXFastConstants.defaultMaxTransformedWeightsBytes` (Constants.swift:133) — the
/// default transformed-weights size cap (25 GiB) enforced by weights preflight, overridable
/// by `MLXFAST_MAX_WEIGHTS_BYTES` (`0`/`none`/`unlimited` disable it).
pub const DEFAULT_MAX_TRANSFORMED_WEIGHTS_BYTES: u64 = 25 * 1024 * 1024 * 1024;

/// (b) admission — the PER-STREAM token-tolerance threshold, in tokens-per-thousand (David's
/// blanket-10% ruling, 2026-08-25). Each cohort stream may differ from the trusted reference argmax
/// on at most this many of every 1000 of its OWN committed tokens; expressed per-thousand so the gate
/// is pure INTEGER arithmetic (`mismatches * 1000 <= COHORT_TOKEN_TOLERANCE_PER_THOUSAND *
/// committed_len`), with no float ratio and no rounding at the 10% boundary — exactly 10% passes.
///
/// PER-STREAM, never a cohort average: ANY single stream over the threshold rejects the WHOLE run
/// ([`crate::cohort_tolerance::evaluate_cohort_token_tolerance`]). The reference argmax comes from the
/// organizer's TRUSTED oracle replaying the candidate's own committed tokens over the pinned reference
/// weights, so the candidate can only choose WHICH tokens it commits, not steer the reference.
///
/// Anti-gaming caveats (stated for the verdict): David accepted that a UNIFORMLY degraded model wrong
/// on ≤10% of tokens per stream passes and can win on speed — (b) is a similar-output speedup bar, not
/// a lossless-correctness one. CONCENTRATION gaming (pushing all divergence into one stream) is closed
/// by the per-stream rule: one stream over 10% fails the run regardless of the others. The value lives
/// ONLY here (never in the JSON fixture — config-carries-no-prose).
pub const COHORT_TOKEN_TOLERANCE_PER_THOUSAND: u32 = 100;

/// `MLXFastConstants.benchmarkPrefillWarmupRuns` — zero: the timed benchmark runs
/// cold (the correctness gate must not warm the measured path), and the official
/// baseline was calibrated the same way. See Constants.swift.
pub const BENCHMARK_PREFILL_WARMUP_RUNS: usize = 0;
/// `MLXFastConstants.benchmarkPrefillTimedRuns` — one measured prefill run.
pub const BENCHMARK_PREFILL_TIMED_RUNS: usize = 1;

/// THE SHAPE OF THE WINDOW a run measures and checks: how many decode steps each mode times, how
/// many steps the correctness gate checks, and how many unmeasured prefill passes the official
/// timed session runs before its one timed prefill.
///
/// These were GLOBAL constants ([`CORRECTNESS_STEPS`], [`BENCHMARK_DECODE_STEPS`], a submit-path
/// decode depth) plus one PER-PLATFORM warm-up count, read straight off the module by every
/// consumer. Bundling them is what lets a run RESOLVE the window once —
/// from its `--contract` track fixture when the fixture declares one
/// (`bench_core::contract::window_shape`, STEP 1 of the constants→contract migration, David
/// 2026-09-15) — and then measure and check under that ONE shape, instead of each consumer
/// reaching for a constant of its own.
///
/// They travel together because they describe ONE window: a decode depth without its checked-step
/// count describes a window nothing judges, and the warm-up count is how that window starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowShape {
    /// [`CORRECTNESS_STEPS`]: the checked decode steps the correctness gate evaluates, and the
    /// golden-loader arity the official path requires.
    pub correctness_steps: usize,
    /// [`BENCHMARK_DECODE_STEPS`]: the timed decode depth of the official and local-iterate modes.
    pub benchmark_decode_steps: usize,
    /// The long continuous checked decode the submit path (`--local-submit`) times — 1023 steps on
    /// the reference tree (`MLXFastConstants.localSubmitBenchmarkDecodeSteps`). It reuses the
    /// local-iterate checked-timing machinery over one decode of `cases[0]`.
    pub local_submit_benchmark_decode_steps: usize,
    /// The fixture's `official_prefill_warmup_runs`: the unmeasured prefill passes the OFFICIAL timed
    /// session runs before its one timed prefill. PER PLATFORM, because each platform's official
    /// baseline pair was captured that way, and the local modes keep
    /// [`BENCHMARK_PREFILL_WARMUP_RUNS`] instead.
    ///
    /// WHY IT EXISTS AT ALL (David 2026-09-07): the timed leg attaches a fresh worker session, and
    /// a session's FIRST prefill pays the drain-at-accept and any page-cache state the previous
    /// session left (on the Qwen 3.8 125B ranked MLX box: 0.68-0.91 ms/token across boots against
    /// 0.64 ±0.2 % from the second pass on), so a one-pass timed prefill refused the ±5 % band at
    /// random even after the process was warm. ONE unmeasured pass, then the timed one, mirrors
    /// what every serving engine does before it measures. Fixed count, no settling loop. CUDA
    /// declares ZERO: it runs against a resident engine that serve-up boots and health-checks
    /// before the window, so its first prefill is already a steady reading, its official baseline
    /// pair was captured with no warm-up pass, and the ds4 adapter fails closed on a second
    /// `prefill` opener without a `phase_diagnostics` barrier between them — a warm-up pass there
    /// is a protocol error, not a warmer number.
    pub official_prefill_warmup_runs: usize,
    /// The fixture's `official_prefill_timed_runs`: how many MEASURED prefill passes the official
    /// timed session runs after its warm-up passes. The phase's prefill time is their MEDIAN.
    /// Optional in the fixture: absent means [`BENCHMARK_PREFILL_TIMED_RUNS`] (1), which is every
    /// track but the Nemotron one, and with one run the median is that run.
    ///
    /// WHY (GumbiiDigital fork, 2026-09-27): on the Nemotron 3.5 Lightning track, with a 4096-token
    /// seed, one warm-up pass and a 60 C gate, ONE timed prefill per calibration pass still varied
    /// 1.0-2.7% pass to pass on half the GB10 boxes, against the fixed 1% calibration maximum. The
    /// median of several passes damps the per-sample noise and drops a single slow pass.
    pub official_prefill_timed_runs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The third leg of `constant≡contract≡env`: a declared track that is not this tree's track
    /// refuses BY NAME, naming BOTH values. This is the leg the env/contract cross-check cannot
    /// cover — those two can agree with each other and still both be foreign.
    #[test]
    fn a_foreign_declared_track_refuses_naming_both() {
        const FOREIGN: &str = "gemma4-26b-a4b-mlx-v1";
        assert_ne!(FOREIGN, TRACK_ID);
        let err = enforce_declared_track(FOREIGN).unwrap_err();
        assert!(err.contains("track_id fence"), "{err}");
        assert!(err.contains(FOREIGN), "must name the declared track: {err}");
        assert!(err.contains(TRACK_ID), "must name the tree's track: {err}");

        // This tree's own track passes, with or without surrounding whitespace — fixture-inert
        // for every path that already declares it.
        assert!(enforce_declared_track(TRACK_ID).is_ok());
        assert!(enforce_declared_track(&format!("  {TRACK_ID}  ")).is_ok());
        // A near-miss is foreign: the match is exact.
        assert!(enforce_declared_track(&format!("{TRACK_ID}-v2")).is_err());
        assert!(enforce_declared_track("").is_err());
    }

    #[test]
    fn platform_resolves_from_the_canonical_track_id_shape() {
        assert_eq!(
            Platform::from_track_id("qwen3.8-125b-a6b-mlx-v1").unwrap(),
            Platform::Mlx
        );
        assert_eq!(
            Platform::from_track_id("qwen3.8-125b-a6b-cuda-v1").unwrap(),
            Platform::Cuda
        );
        assert_eq!(
            Platform::from_track_id(" qwen3.8-125b-a6b-cuda-v12 ").unwrap(),
            Platform::Cuda
        );
        for bad in [
            "",
            "qwen3.8-125b-a6b-v1",
            "qwen3.8-125b-a6b-spark-v1",
            "qwen3.8-125b-a6b-mlx",
            "qwen3.8-125b-a6b-mlx-v",
            "qwen3.8-125b-a6b-mlx-vX",
            "gemma4-26b-a4b-MLX-v1",
        ] {
            let err = Platform::from_track_id(bad).unwrap_err();
            assert!(err.contains("platform"), "{bad:?}: {err}");
        }
    }

    #[test]
    fn cool_gate_temp_is_per_platform_mac_40_gb10_60() {
        // R21 lift (David 2026-08-30): the gate temperature is a trusted per-platform value, not a
        // frozen 40 C constant. Mac/MLX idles cool → 40 C. GB10/CUDA: a loaded resident holds the
        // die at 49–52 C in P0, so this fork re-sites the gate to 60 C (see the constant).
        assert_eq!(Platform::Mlx.cool_gate_temp_c(), 40.0);
        assert_eq!(Platform::Cuda.cool_gate_temp_c(), 60.0);
        assert!(
            Platform::Cuda.cool_gate_temp_c() > Platform::Mlx.cool_gate_temp_c(),
            "the GB10 gate is raised above the Mac gate, keyed by platform"
        );
        // At least 30 C below the GB10 maximum operating temperature (~96 C: `GPU T.Limit Temp` is a
        // margin, 53 C at 43 C), so the gate never admits a box near thermal slowdown.
        assert!(Platform::Cuda.cool_gate_temp_c() <= 96.0 - 30.0);
    }
}
