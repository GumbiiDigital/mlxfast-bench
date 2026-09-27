//! The `--contract` track fixture and the David 2026-08-26 ARM GATE.
//!
//! Moved out of the retired measure-job (`measure_job.rs`) so the SOLE scored path — flow A,
//! `benchd iterate --mode official` — inherits the arm gate. The struct and the gate behavior
//! are byte-identical to the measure-job originals. This is the ONE typed view of a track
//! fixture: the measure-job used to keep a second, four-field `Contract` of its own and parse the
//! same bytes into both, so a fixture had two schemas that could disagree within one file. serde
//! ignores every other fixture key exactly as before (no `deny_unknown_fields`), so the real track
//! fixtures — which still carry `calibration`, `hidden_correctness_golden`, `reference_model`, …
//! — parse unchanged.
//!
//! A fixture is READ ONCE per run, through [`load`], which digests the bytes it parsed
//! ([`LoadedContract`]). That digest is sealed as `metrics.contract_sha256` — the MIGRATION PIN:
//! the scored artifact records WHICH fixture bytes decided its arm state, its floors and its pair
//! count, the same way `metrics.golden_hash` records which golden bytes it was judged against.

use crate::constants::{
    AcceptanceBands, OfficialBaseline, ScoredRegime, TrackModelIdentity, WindowShape,
    ACCEPTANCE_BANDS_UNDECLARED, MODEL_IDENTITY_UNDECLARED, OFFICIAL_BASELINE_PENDING,
    SCORED_REGIME_PENDING, WINDOW_SHAPE_UNDECLARED,
};
use crate::score::{ScoringWeights, SpeedupFloors};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// One entry of the contract's `timed_prompt_pool[]`. `sha256`, `bytes` and
/// `noop_decode_speedup` are read (the pin identity plus the per-prompt no-op ref carried into
/// the superset `results.json`); the remaining entry keys (`r2_path`,
/// `noop_decode_speedup_pairs`, …) are ignored by serde.
// UNVERIFIED(B-4): the `timed_prompt_pool[].{sha256,bytes,noop_decode_speedup}` field paths.
#[derive(Debug, Clone, Deserialize)]
pub struct PromptPoolEntry {
    pub sha256: String,
    /// #112 (L3) — the entry's declared BYTE COUNT. A canonical golden is identified by
    /// `sha256` + `bytes` TOGETHER; the live pool entries carry both, and this half was being
    /// parsed past and never checked. OPTIONAL because older/offline contract fixtures pin by
    /// sha alone — when it is PRESENT it is ENFORCED (`validate_goldens_pinned`, die-8), and
    /// when it is absent the sha-only pin stands exactly as before. A present-but-wrong `bytes`
    /// is never treated as "close enough": the two halves must agree or the run refuses.
    #[serde(default)]
    pub bytes: Option<u64>,
    #[serde(default)]
    pub noop_decode_speedup: Option<f64>,
}

/// The parsed `--contract` track fixture — the ONE typed view of one. Only the fields benchd
/// consumes are modelled; serde ignores the rest.
#[derive(Debug, Clone, Deserialize)]
pub struct Contract {
    /// The track fixture's own workflow-declared track id (e.g. `qwen3.8-125b-a6b-mlx-v1`).
    #[serde(default)]
    pub track_id: Option<String>,
    /// R12 — the optional human track name; sealed as `track_name` when present (env override
    /// wins). The track ID above is the CONSTANT; this is the label.
    #[serde(default)]
    pub track_name: Option<String>,
    /// The track's TIMED PROMPT POOL — the pins a `--golden` must resolve to
    /// (`validate_goldens_pinned`) and the per-prompt no-op references the coverage gate reads
    /// (`validate_timed_pool_coverage`).
    #[serde(default)]
    pub timed_prompt_pool: Vec<PromptPoolEntry>,
    /// David ruling (2026-08-26) — the track's ARM STATE, and the one contract field that decides
    /// whether benchd may seal a SCORING artifact for this track at all
    /// ([`enforce_official_scoring_enabled`]).
    ///
    /// `Option<bool>` rather than `bool`, deliberately: ABSENT and `false` are BOTH refusals, but
    /// they are DIFFERENT diagnoses (a track that has not been armed yet vs. a fixture that never
    /// declares an arm state at all) and the refusal says which. Collapsing them into
    /// `#[serde(default)] bool` would make a fixture that forgot the key indistinguishable from one
    /// that deliberately declared `false` — and, worse, would make ABSENCE look like a decision.
    /// Absence is never armed.
    #[serde(default)]
    pub official_scoring_enabled: Option<bool>,
    /// David ruling (2026-08-26) — the track's ALLOWED-MODES LIST: the spec modes a submission may
    /// declare on THIS track (`resolve_allowed_modes`, `enforce_track_allowed_modes`).
    ///
    /// `Option<Vec<String>>` rather than `#[serde(default)] Vec<String>`, for the same reason
    /// [`Contract::official_scoring_enabled`] is a tri-state: ABSENT means "this fixture has no
    /// opinion, use the default list" and is the state every OTHER track is in, while an EMPTY
    /// list is a fixture that declared the field and listed nothing — a refusal, not a default.
    /// Collapsing the two would make a botched edit read as a decision.
    ///
    /// Absence is the OTHER-TRACK PROTECTION: adding `dflash` to the gemma4 fixture cannot widen
    /// qwen3.8 or laguna, whose calibration bands, floors and leaderboards were never measured for
    /// it. And because the list is DATA, arming a further mode on gemma4 later is a fixture edit —
    /// no benchd rebuild — which is the whole point of making the fence contract-driven.
    #[serde(default)]
    pub allowed_modes: Option<Vec<String>>,
    /// PAIRS PER SCORED RUN on the paired per-box path (David 2026-09-09: "2 pairs on both mlx and
    /// cuda"; "1 pair is not sufficient"). Each pair is one serial-control leg on the reference
    /// tree followed by one candidate leg, same prompt, same box. The fixture is the ONLY source
    /// of this count: no flag, no environment, no default — so a box cannot silently run fewer
    /// pairs than the track declares.
    #[serde(default)]
    pub official_pairs: Option<u32>,
    /// THE DECODE SPEEDUP FLOOR this project's scored run must clear (David 2026-09-09: 0.95).
    /// The fixture is the ONLY source on the scoring path — no flag, no environment, no default —
    /// so a track cannot be scored against a floor it never declared, and each project sets its
    /// own. See [`speedup_floors`].
    #[serde(default)]
    pub decode_speedup_floor: Option<f64>,
    /// THE PREFILL SPEEDUP FLOOR this project's scored run must clear (David 2026-09-09: 0.95).
    /// Declared and enforced exactly as [`Contract::decode_speedup_floor`]; the prefill axis is a
    /// floor of its own, not a decode side effect.
    #[serde(default)]
    pub prefill_speedup_floor: Option<f64>,
    /// THE BATCH SIZE this track's scored point is measured at, and the value that SELECTS the
    /// measured path (David 2026-09-15: "Benchd should support batching. Batching as part of the
    /// configuration.").
    ///
    /// `1` selects the SINGLE-STREAM path — one `free_decode_begin` / `free_decode_run` round trip
    /// per leg, one stream, one prompt. A width greater than 1 selects the BATCHED COHORT path: the
    /// engine free-runs the whole pinned pool concurrently through the cohort form of the same
    /// verbs and benchd times ONE window over the cohort
    /// (`benchd::measure_job::LegRegime::BatchedFreeRunV1_2`). The width is a PINNED IDENTITY read
    /// out of the fixture and echoed by the engine on the wire — never a CLI flag, which could
    /// silently differ between legs.
    ///
    /// Certified at PARSE ([`Contract::certify`]) as a count of streams — at least 1 — so a
    /// malformed width lands on the file rather than on a run that has already spent box time.
    #[serde(default)]
    pub scored_batch_size: Option<u32>,
    /// The exponent the composite raises this track's PREFILL gain to
    /// ([`ScoredRegime::prefill_gain_exponent`]). STEP 1 of the constants→contract migration
    /// (David 2026-09-15): declared here, the fixture supplies the regime; absent, the track
    /// measures its own denominator on the box and seals `scored_regime: "device"`.
    ///
    /// The regime's two exponents arrive TOGETHER with [`Contract::scored_batch_size`]: a batch
    /// size without its exponents describes no score, so a fixture that declares one exponent must
    /// declare all three ([`Contract::certify`]).
    #[serde(default)]
    pub prefill_gain_exponent: Option<f64>,
    /// The exponent the composite raises this track's DECODE gain to
    /// ([`ScoredRegime::decode_gain_exponent`]). Declared, certified and defaulted exactly as
    /// [`Contract::prefill_gain_exponent`].
    #[serde(default)]
    pub decode_gain_exponent: Option<f64>,
    /// Whether this track's ranked run MEASURES ITS OWN DENOMINATOR — a serial-control leg on the
    /// organizer-staged reference tree, same box, same job (David 2026-09-08) — instead of scoring
    /// against a stored pair.
    ///
    /// The fixture decides; absent, the run is REFUSED BY NAME. The
    /// two states are mutually exclusive with a declared official baseline pair, and
    /// [`Contract::certify`] refuses a fixture that declares both.
    #[serde(default)]
    pub scores_against_live_control_leg: Option<bool>,
    /// Whether the PAIRED `measure-job`/`overlay-timing` seam is RETIRED for this track, i.e. its
    /// sole scored path is the single leg (David 2026-08-30).
    ///
    /// The fixture decides; absent, the run is REFUSED BY NAME.
    #[serde(default)]
    pub paired_flow_retired: Option<bool>,
    /// The track's CAPTURED OFFICIAL BASELINE — the serial prefill seconds-per-token half of the
    /// pair a stored-pair track's scored run divides by.
    ///
    /// Declared, the fixture supplies the denominator; absent, the run is refused BY NAME as
    /// `OFFICIAL-BASELINE-PENDING-CAPTURE`.
    ///
    /// The two halves are ONE captured pair and arrive TOGETHER ([`Contract::certify`]): a prefill
    /// figure without its decode figure is not a baseline, and a run that mixed one captured half
    /// with one table half would score against a pair no capture ever produced.
    #[serde(default)]
    pub official_baseline_prefill_seconds_per_token: Option<f64>,
    /// The DECODE half of the track's captured official baseline pair. Declared, certified and
    /// defaulted exactly as [`Contract::official_baseline_prefill_seconds_per_token`].
    #[serde(default)]
    pub official_baseline_decode_seconds_per_token: Option<f64>,
    /// The track's PREFILL acceptance band, upper tolerance
    /// ([`AcceptanceBands::prefill_up_tolerance`]).
    ///
    /// The SIX band values are ONE shape and arrive together; absent, the run is refused BY NAME as
    /// [`crate::constants::ACCEPTANCE_BANDS_UNDECLARED`]. The band is what gates the timed run, so
    /// benchd never invents one.
    #[serde(default)]
    pub prefill_band_up_tolerance: Option<f64>,
    /// The track's PREFILL acceptance band, lower tolerance. INERT while
    /// [`Contract::prefill_band_down_enabled`] is `false`.
    #[serde(default)]
    pub prefill_band_down_tolerance: Option<f64>,
    /// The track's DECODE acceptance band, upper tolerance.
    #[serde(default)]
    pub decode_band_up_tolerance: Option<f64>,
    /// The track's DECODE acceptance band, lower tolerance. INERT while
    /// [`Contract::decode_band_down_enabled`] is `false`.
    #[serde(default)]
    pub decode_band_down_tolerance: Option<f64>,
    /// Whether the DECODE band enforces its lower bound. `false` on the MTP timed leg, where
    /// spec-decode decode is legitimately much faster than the serial reference and an
    /// "improvement too large" guard would fail a healthy run.
    #[serde(default)]
    pub decode_band_down_enabled: Option<bool>,
    /// Whether the PREFILL band enforces its lower bound. `false` on the paired design, where the
    /// control leg is measured live and carries its own health band.
    #[serde(default)]
    pub prefill_band_down_enabled: Option<bool>,
    /// The DECODE weight of the published composite ([`ScoringWeights::decode`], 0.75 on every
    /// track scored so far).
    ///
    /// The two weights are ONE declaration — [`crate::score::score`] divides each by their sum —
    /// and arrive together. A fixture that declares neither is refused by name.
    #[serde(default)]
    pub score_decode_weight: Option<f64>,
    /// The PREFILL weight of the published composite. Declared, certified and defaulted exactly as
    /// [`Contract::score_decode_weight`].
    #[serde(default)]
    pub score_prefill_weight: Option<f64>,
    /// The CHECKED DECODE STEPS the correctness gate evaluates ([`WindowShape::correctness_steps`],
    /// `CORRECTNESS_STEPS`).
    ///
    /// The FOUR window values are ONE shape and arrive together. A fixture that declares none of
    /// them is refused by name: there is no in-tree window to fall back to.
    #[serde(default)]
    pub correctness_steps: Option<u32>,
    /// The TIMED DECODE DEPTH of the official and local-iterate modes
    /// ([`WindowShape::benchmark_decode_steps`], `BENCHMARK_DECODE_STEPS`).
    #[serde(default)]
    pub benchmark_decode_steps: Option<u32>,
    /// The long continuous checked decode the submit path times
    /// ([`WindowShape::local_submit_benchmark_decode_steps`]).
    #[serde(default)]
    pub local_submit_benchmark_decode_steps: Option<u32>,
    /// The UNMEASURED prefill passes the official timed session runs before its one timed prefill
    /// ([`WindowShape::official_prefill_warmup_runs`]).
    ///
    /// `0` is a real declaration, not an absence: the CUDA track runs no warm-up pass because its
    /// resident engine is already warm and its adapter refuses a second opener.
    #[serde(default)]
    pub official_prefill_warmup_runs: Option<u32>,
    /// The MEASURED prefill passes the official timed session runs after its warm-up passes; the
    /// phase's prefill time is their median ([`WindowShape::official_prefill_timed_runs`]).
    /// OPTIONAL, unlike the four window values: absent means one timed pass, the behaviour every
    /// fixture that predates the field already describes. `0` is refused.
    #[serde(default)]
    pub official_prefill_timed_runs: Option<u32>,
    /// The track checkpoint's VOCABULARY BOUND ([`TrackModelIdentity::vocab_size`]): the range
    /// `0..vocab_size` every token id in a golden or a tape must fall in, and the bound the
    /// conformance path judges worker top-logit distributions against.
    ///
    /// The model-shape values (`golden_model_type`, `vocab_size`, `num_hidden_layers`,
    /// `seed_tokens`) are ONE declaration and arrive together; absent, the run refuses the
    /// undeclared track BY NAME.
    ///
    /// LOCKSTEP HAZARD, unchanged by the move: this bound exists in TWO copies that must move
    /// together — the engine's `MLXFastConstants.vocabSize`, which the reference-tape recorder
    /// applies as its emit-time pre-check, and this one, which benchd applies at load time.
    #[serde(default)]
    pub vocab_size: Option<u32>,
    /// The checkpoint's DECODER-LAYER COUNT ([`TrackModelIdentity::num_hidden_layers`]). Sealed as
    /// `metrics.num_layers`, a DETERMINISTIC parity field; it feeds no score, floor or band.
    #[serde(default)]
    pub num_hidden_layers: Option<u32>,
    /// The track's SEED LENGTH ([`TrackModelIdentity::seed_tokens`]) — the correctness prompt, the
    /// timed prefill prompt and the decode seed, which are ONE value (David's 2026-08-24
    /// seed-length ruling moved all three together).
    #[serde(default)]
    pub seed_tokens: Option<u32>,
    /// The `model_type` this track's goldens declare ([`TrackModelIdentity::golden_model_type`]):
    /// the model's INTERNAL ARCHITECTURE id as the weights declare it, never a track name and
    /// never a release version.
    ///
    /// The FOURTH member of the model-shape declaration, and the one STEP 1 could not move: it was
    /// a `&'static str` on `TrackModelIdentity`, which is what kept that struct `Copy`. David ruled
    /// the simplest shape (2026-09-15), so the struct owns a `String`, drops `Copy`, and the
    /// fixture supplies the value. Declared with the other three or not at all.
    #[serde(default)]
    pub golden_model_type: Option<String>,
}

impl Contract {
    /// A contract that DECLARES NOTHING: every value is absent.
    ///
    /// It is what a run WITHOUT `--contract` resolves against, so the contract-first resolvers have
    /// one shape to read and the no-fixture case is the same code path as a fixture that is silent
    /// about a group — not a second, parallel branch. It is NOT a default set of values: nothing
    /// numeric lives here, and every resolver that reads it refuses by name, except the two groups
    /// the device measures for itself ([`ContractSource::Device`]).
    pub const NONE_DECLARED: Contract = Contract {
        track_id: None,
        track_name: None,
        timed_prompt_pool: Vec::new(),
        official_scoring_enabled: None,
        allowed_modes: None,
        official_pairs: None,
        decode_speedup_floor: None,
        prefill_speedup_floor: None,
        scored_batch_size: None,
        prefill_gain_exponent: None,
        decode_gain_exponent: None,
        scores_against_live_control_leg: None,
        paired_flow_retired: None,
        official_baseline_prefill_seconds_per_token: None,
        official_baseline_decode_seconds_per_token: None,
        prefill_band_up_tolerance: None,
        prefill_band_down_tolerance: None,
        decode_band_up_tolerance: None,
        decode_band_down_tolerance: None,
        decode_band_down_enabled: None,
        prefill_band_down_enabled: None,
        score_decode_weight: None,
        score_prefill_weight: None,
        correctness_steps: None,
        benchmark_decode_steps: None,
        local_submit_benchmark_decode_steps: None,
        official_prefill_warmup_runs: None,
        official_prefill_timed_runs: None,
        vocab_size: None,
        num_hidden_layers: None,
        seed_tokens: None,
        golden_model_type: None,
    };
}

/// [`Contract::NONE_DECLARED`] at a `'static` address, and DEFINED FROM IT so there is still one
/// spelling of "no fixture". The struct owns a `Vec` (`timed_prompt_pool`), so the const cannot be
/// rvalue-promoted behind a reference; anything that needs to BORROW the no-fixture contract for
/// longer than a statement borrows this.
pub static NO_FIXTURE: Contract = Contract::NONE_DECLARED;

/// The `--contract` fixture a run resolves its values against: the loaded one, or
/// [`Contract::NONE_DECLARED`] when the run was given none.
///
/// ONE function so no call site invents its own "no fixture" stand-in.
pub fn declared(loaded: Option<&LoadedContract>) -> &Contract {
    match loaded {
        Some(loaded) => &loaded.contract,
        None => &NO_FIXTURE,
    }
}

/// WHERE a resolved value came from — the half of the constants→contract migration a sealed
/// artifact has to record.
///
/// A value is either DECLARED by the `--contract` track fixture or MEASURED ON THE DEVICE that
/// runs the benchmark. Sealing this beside `metrics.contract_sha256` is what makes that legible:
/// the digest says WHICH fixture bytes were in force, and this says WHICH GROUPS of the scored
/// regime those bytes decided and which groups the box answered for itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContractSource {
    /// The `--contract` track fixture declared the value, and the run used it.
    Contract,
    /// The fixture declares nothing, because THIS DEVICE MEASURES THE VALUE FOR ITSELF.
    ///
    /// The 125B tracks leave the scored regime and the official baseline pair undeclared. Each
    /// device tracks its own reference benchmark: it measures the serial-control leg on its own
    /// box, in the same job as the scored leg, and scores the live ratio. benchd does not upload
    /// that reference and does not save it to a centralized repository, so there is no shared
    /// document for the seal to name — the device is the source.
    Device,
    /// The fixture declared nothing, so the run used an in-tree per-track table.
    ///
    /// NEVER EMITTED. STEP 3 deleted the tables, so no run this binary seals can carry it: every
    /// resolver refuses an absent declaration, and an undeclared group that reaches a seal is a
    /// device-measured one ([`ContractSource::Device`]). The variant is KEPT so an artifact sealed
    /// by a STEP-1 binary still deserializes and still reads as what it was — the seal is
    /// evidence, and evidence does not get rewritten when the code moves on.
    Table,
}

impl ContractSource {
    /// The sealed spelling, which is also the serde spelling.
    pub fn key(self) -> &'static str {
        match self {
            ContractSource::Contract => "contract",
            ContractSource::Device => "device",
            ContractSource::Table => "table",
        }
    }
}

/// A resolved value AND the source that produced it, inseparably.
///
/// The two travel together for the same reason [`LoadedContract`] carries its digest: a source is
/// only worth sealing if it describes the value the run actually used. There is no way to get a
/// value out of a resolver without its source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resolved<T> {
    /// The value, identical in type and meaning to what the table accessor returns.
    pub value: T,
    /// Where [`Resolved::value`] came from.
    pub source: ContractSource,
}

/// WHICH SOURCE decided each group of a run's scored regime. Sealed as `metrics.contract_sources`.
///
/// One field per GROUP, not per value: the groups are the units the tables are keyed by and the
/// units a fixture declares atomically, so a per-value record would say the same thing many times
/// and could not say anything a per-group record cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContractSources {
    /// The scored regime: batch size and the two composite exponents ([`scored_regime`]).
    pub scored_regime: ContractSource,
    /// Whether the track measures its own denominator ([`scores_against_live_control_leg`]).
    pub live_control_leg: ContractSource,
    /// Whether the paired flow is retired for the track ([`paired_flow_retired`]).
    pub paired_flow: ContractSource,
    /// The captured serial prefill/decode seconds-per-token pair ([`official_baseline`]).
    pub official_baseline: ContractSource,
    /// The acceptance-band shape ([`acceptance_bands`]).
    pub acceptance_bands: ContractSource,
    /// The two composite weights ([`scoring_weights`]).
    pub scoring_weights: ContractSource,
    /// The measured/checked window's shape ([`window_shape`]).
    pub window_shape: ContractSource,
    /// The track checkpoint's model SHAPE: vocabulary bound, layer count, seed length
    /// ([`model_identity`]).
    pub model_shape: ContractSource,
}

impl ContractSources {
    /// EVERY GROUP FROM A TABLE: what [`sources`] returned for a run whose fixture declared
    /// nothing, before STEP 3 deleted the tables.
    ///
    /// No run this binary seals can carry it any more, for the reason [`ContractSource::Table`]
    /// gives, so its remaining job is to be the ALL-TABLE payload the seal tests build — the shape
    /// of an artifact a STEP-1 binary sealed. Named once here, beside [`Contract::NONE_DECLARED`],
    /// every reader of it agrees about what "all table" is.
    pub const ALL_TABLE: ContractSources = ContractSources {
        scored_regime: ContractSource::Table,
        live_control_leg: ContractSource::Table,
        paired_flow: ContractSource::Table,
        official_baseline: ContractSource::Table,
        acceptance_bands: ContractSource::Table,
        scoring_weights: ContractSource::Table,
        window_shape: ContractSource::Table,
        model_shape: ContractSource::Table,
    };
}

/// THE SCORED REGIME, CONTRACT-FIRST: the fixture's declaration, or a refusal by name when it
/// makes none.
///
/// The three numbers are ONE declaration — `scored_batch_size`, `prefill_gain_exponent` and
/// `decode_gain_exponent` — so the fixture declares all three or none. A fixture that declares an EXPONENT
/// without the others is refused at the parse ([`Contract::certify`]), never half-resolved against
/// the table.
///
/// THE DECLARED WIDTH SELECTS THE MEASURED PATH (David 2026-09-15: batching is part of the
/// configuration). `scored_batch_size: 1` is the single-stream point; a greater width is the
/// batched cohort point, whose series tag is assigned with the width at
/// `benchd::measure_job::ScoredBatchPoint::certify`. The resolver CARRIES the declared width —
/// there is no measured-width constant to check it against any more.
pub fn scored_regime(
    contract: &Contract,
    track_id: &str,
) -> Result<Resolved<ScoredRegime>, String> {
    match declared_scored_regime(contract) {
        Some(regime) => Ok(Resolved {
            value: regime,
            source: ContractSource::Contract,
        }),
        None => Err(format!(
            "scored regime for track_id {track_id:?} is {SCORED_REGIME_PENDING}: the --contract \
             track fixture declares no scored_batch_size, prefill_gain_exponent and \
             decode_gain_exponent; refusing to score"
        )),
    }
}

/// The regime the fixture DECLARES, or `None` when it declares none of it. Shape-checked by
/// [`certify_scored_regime`], so a partial declaration never reaches here.
fn declared_scored_regime(contract: &Contract) -> Option<ScoredRegime> {
    match (
        contract.scored_batch_size,
        contract.prefill_gain_exponent,
        contract.decode_gain_exponent,
    ) {
        (Some(batch), Some(prefill), Some(decode)) => Some(ScoredRegime {
            scored_batch_size: batch as usize,
            prefill_gain_exponent: prefill,
            decode_gain_exponent: decode,
        }),
        _ => None,
    }
}

/// THE REGIME DECLARATION IS ALL-OR-NOTHING, checked at the parse.
///
/// `scored_batch_size` ALONE is not a partial regime, and that asymmetry is deliberate: the field
/// predates this migration as a CROSS-CHECK (see [`Contract::scored_batch_size`]) and the live
/// engine fixtures already carry it, so reading it as half a regime would refuse fixtures that are
/// correct today. An EXPONENT is new and regime-specific, so declaring either one commits the
/// fixture to the whole declaration.
fn certify_scored_regime(contract: &Contract, track_id: &str) -> Result<(), String> {
    let exponents = [
        ("prefill_gain_exponent", contract.prefill_gain_exponent),
        ("decode_gain_exponent", contract.decode_gain_exponent),
    ];
    if exponents.iter().all(|(_, v)| v.is_none()) {
        return Ok(());
    }
    for (field, declared) in exponents {
        match declared {
            Some(v) if v.is_finite() && v >= 0.0 => {}
            Some(v) => {
                return Err(format!(
                    "the --contract track fixture for {track_id:?} declares {field}: {v}; a \
                     composite exponent must be finite and non-negative (0 means the axis carries \
                     no weight)"
                ))
            }
            None => {
                return Err(format!(
                    "the --contract track fixture for {track_id:?} declares a scored regime but no \
                     {field}; the batch size and the two composite exponents are ONE declaration — \
                     a batch size without its exponents describes no score — so a fixture declares \
                     all three (scored_batch_size, prefill_gain_exponent, decode_gain_exponent) or \
                     none of them"
                ))
            }
        }
    }
    if contract.scored_batch_size.is_none() {
        return Err(format!(
            "the --contract track fixture for {track_id:?} declares composite exponents but no \
             scored_batch_size; the batch size and the two exponents are ONE declaration — a \
             regime that does not say what point it is measured at scores nothing — so a fixture \
             declares all three or none of them"
        ));
    }
    if contract.prefill_gain_exponent.unwrap_or(0.0) + contract.decode_gain_exponent.unwrap_or(0.0)
        <= 0.0
    {
        return Err(format!(
            "the --contract track fixture for {track_id:?} declares prefill_gain_exponent 0 and \
             decode_gain_exponent 0; a regime that weights neither axis scores nothing"
        ));
    }
    Ok(())
}

/// WHETHER THE TRACK MEASURES ITS OWN DENOMINATOR, CONTRACT-FIRST: the fixture's declaration, or a
/// refusal by name when it makes none.
pub fn scores_against_live_control_leg(
    contract: &Contract,
    track_id: &str,
) -> Result<Resolved<bool>, String> {
    match contract.scores_against_live_control_leg {
        Some(declared) => Ok(Resolved {
            value: declared,
            source: ContractSource::Contract,
        }),
        None => Err(format!(
            "the --contract track fixture for {track_id:?} declares no \
             scores_against_live_control_leg; a track either measures its own denominator or \
             scores against a stored pair, and benchd will not pick one for it; refusing"
        )),
    }
}

/// WHETHER THE PAIRED FLOW IS RETIRED for the track, CONTRACT-FIRST: the fixture's declaration, or
/// a refusal by name when it makes none.
pub fn paired_flow_retired(contract: &Contract, track_id: &str) -> Result<Resolved<bool>, String> {
    match contract.paired_flow_retired {
        Some(declared) => Ok(Resolved {
            value: declared,
            source: ContractSource::Contract,
        }),
        None => Err(format!(
            "the --contract track fixture for {track_id:?} declares no paired_flow_retired; \
             whether the measure-job/overlay-timing seam is retired for a track is the track's \
             declaration, not benchd's default; refusing"
        )),
    }
}

/// The PAIRED-FLOW entry fence, CONTRACT-FIRST: a track whose sole scored path is the single leg
/// refuses `measure-job` / `overlay-timing` BY NAME. The verdict is
/// `bench_core::constants::enforce_paired_flow_available`'s, over the resolved value.
pub fn enforce_paired_flow_available(contract: &Contract, track_id: &str) -> Result<(), String> {
    let resolved = paired_flow_retired(contract, track_id)?;
    crate::constants::refuse_retired_paired_flow(track_id, resolved.value)
}

/// EVERY GROUP'S SOURCE for one run, resolved from ONE contract and ONE track id.
///
/// A resolver is a pure function of `(contract, track_id)`, so the source this records is the
/// source every call site of that group in the same run computes — which is what makes sealing it
/// once, here, a true statement about the whole run rather than about this call.
pub fn sources(contract: &Contract, _track_id: &str) -> ContractSources {
    // A group the fixture declares seals `contract`. A group it leaves undeclared seals `device`:
    // the scored regime and the official baseline pair on a live-control-leg track, which the box
    // measures for itself in the same job. `table` is never emitted — STEP 3 deleted the tables,
    // and every resolver of a group that is neither declared nor device-measured refuses before a
    // seal exists.
    let declared = |present: bool| {
        if present {
            ContractSource::Contract
        } else {
            ContractSource::Device
        }
    };
    ContractSources {
        scored_regime: declared(declared_scored_regime(contract).is_some()),
        live_control_leg: declared(contract.scores_against_live_control_leg.is_some()),
        paired_flow: declared(contract.paired_flow_retired.is_some()),
        official_baseline: declared(declared_baseline_pair(contract).is_some()),
        acceptance_bands: declared(declared_bands(contract).is_some()),
        scoring_weights: declared(
            contract.score_decode_weight.is_some() && contract.score_prefill_weight.is_some(),
        ),
        window_shape: declared(declared_window_shape(contract).is_some()),
        model_shape: declared(declared_model_shape(contract).is_some()),
    }
}

/// THE TRACK'S MODEL IDENTITY, with its SHAPE resolved CONTRACT-FIRST: the vocabulary bound, the
/// layer count and the seed length from the fixture when it declares them, and a refusal by name
/// when it does not.
///
/// THE `golden_model_type` MOVES WITH THEM (David 2026-09-15). It was a `&'static str` on
/// [`TrackModelIdentity`], which is what kept that struct `Copy`; the struct now owns a `String`,
/// drops `Copy`, and the fixture supplies the value. All four travel together.
pub fn model_identity(
    contract: &Contract,
    track_id: &str,
) -> Result<Resolved<TrackModelIdentity>, String> {
    match declared_model_shape(contract) {
        Some(shape) => Ok(Resolved {
            value: TrackModelIdentity {
                golden_model_type: shape.golden_model_type,
                vocab_size: shape.vocab_size,
                num_hidden_layers: shape.num_hidden_layers,
                seed_tokens: shape.seed_tokens,
            },
            source: ContractSource::Contract,
        }),
        None => Err(format!(
            "model identity for track_id {track_id:?} is {MODEL_IDENTITY_UNDECLARED}: the \
             --contract track fixture declares no golden_model_type, vocab_size, \
             num_hidden_layers and seed_tokens; refusing to load a golden"
        )),
    }
}

/// The model shape the fixture declares, as one value: the three numbers and the `model_type`
/// string that travels with them.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelShape {
    vocab_size: usize,
    num_hidden_layers: i64,
    seed_tokens: usize,
    golden_model_type: String,
}

/// The model shape the fixture DECLARES, or `None` when it declares none of it. A PARTIAL shape
/// never reaches here — [`certify_model_shape`] refuses it at the parse.
fn declared_model_shape(contract: &Contract) -> Option<ModelShape> {
    Some(ModelShape {
        vocab_size: contract.vocab_size? as usize,
        num_hidden_layers: contract.num_hidden_layers? as i64,
        seed_tokens: contract.seed_tokens? as usize,
        golden_model_type: contract.golden_model_type.clone()?,
    })
}

/// THE MODEL SHAPE IS ONE DECLARATION: all FOUR values or none. Each number must be strictly
/// positive — a vocabulary of zero admits no token, a checkpoint of zero layers decodes nothing,
/// and a seed of zero tokens primes nothing — and the `model_type` must be a non-empty string,
/// because an empty one names no architecture and would match no golden.
fn certify_model_shape(contract: &Contract, track_id: &str) -> Result<(), String> {
    let values = [
        ("vocab_size", contract.vocab_size),
        ("num_hidden_layers", contract.num_hidden_layers),
        ("seed_tokens", contract.seed_tokens),
    ];
    let model_type = contract.golden_model_type.as_deref();
    if values.iter().all(|(_, v)| v.is_none()) && model_type.is_none() {
        return Ok(());
    }
    for (field, declared) in values {
        match declared {
            Some(n) if n >= 1 => {}
            Some(n) => {
                return Err(format!(
                    "the --contract track fixture for {track_id:?} declares {field}: {n}; a \
                     vocabulary bound, a layer count and a seed length must each be at least 1"
                ))
            }
            None => {
                return Err(format!(
                    "the --contract track fixture for {track_id:?} declares a model shape but no \
                     {field}; the golden model_type, the vocabulary bound, the layer count and \
                     the seed length are ONE declaration (they describe one checkpoint) and are \
                     declared together — a half-declared shape describes no checkpoint at all"
                ))
            }
        }
    }
    match model_type {
        Some(t) if !t.trim().is_empty() => Ok(()),
        Some(_) => Err(format!(
            "the --contract track fixture for {track_id:?} declares an empty golden_model_type; \
             the model_type names the checkpoint's own architecture, so an empty one names nothing \
             and matches no golden"
        )),
        None => Err(format!(
            "the --contract track fixture for {track_id:?} declares a model shape but no \
             golden_model_type; the golden model_type, the vocabulary bound, the layer count and \
             the seed length are ONE declaration (they describe one checkpoint) and are declared \
             together — a half-declared shape describes no checkpoint at all"
        )),
    }
}

/// THE WINDOW SHAPE, CONTRACT-FIRST: the four values the fixture declares, or a refusal.
///
/// The four arrive together or not at all, and the warm-up count belongs to the track rather than
/// to a platform: a fixture that states its own warm-up count has stated it for the track. A
/// fixture that states none of the four is refused by name — benchd measures no window it was not
/// given.
pub fn window_shape(contract: &Contract, track_id: &str) -> Result<Resolved<WindowShape>, String> {
    match declared_window_shape(contract) {
        Some(window) => Ok(Resolved {
            value: window,
            source: ContractSource::Contract,
        }),
        None => Err(format!(
            "measurement window for track_id {track_id:?} is {WINDOW_SHAPE_UNDECLARED}: the \
             --contract track fixture declares no correctness_steps, benchmark_decode_steps, \
             local_submit_benchmark_decode_steps and official_prefill_warmup_runs; refusing to \
             measure a window it was not given"
        )),
    }
}

/// The window the fixture DECLARES, or `None` when it declares none of it. A PARTIAL window never
/// reaches here — [`certify_window_shape`] refuses it at the parse.
fn declared_window_shape(contract: &Contract) -> Option<WindowShape> {
    Some(WindowShape {
        correctness_steps: contract.correctness_steps? as usize,
        benchmark_decode_steps: contract.benchmark_decode_steps? as usize,
        local_submit_benchmark_decode_steps: contract.local_submit_benchmark_decode_steps? as usize,
        official_prefill_warmup_runs: contract.official_prefill_warmup_runs? as usize,
        official_prefill_timed_runs: contract
            .official_prefill_timed_runs
            .map_or(crate::constants::BENCHMARK_PREFILL_TIMED_RUNS, |n| {
                n as usize
            }),
    })
}

/// THE WINDOW IS ONE SHAPE: all four values or none, and each STEP COUNT strictly positive — a
/// window of zero decode steps measures nothing and a gate of zero checked steps checks nothing.
///
/// The WARM-UP count is the exception that may be zero, and legitimately is on CUDA.
fn certify_window_shape(contract: &Contract, track_id: &str) -> Result<(), String> {
    if contract.official_prefill_timed_runs == Some(0) {
        return Err(format!(
            "the --contract track fixture for {track_id:?} declares official_prefill_timed_runs: \
             0; a prefill of zero timed runs measures nothing"
        ));
    }
    let counts = [
        ("correctness_steps", contract.correctness_steps),
        ("benchmark_decode_steps", contract.benchmark_decode_steps),
        (
            "local_submit_benchmark_decode_steps",
            contract.local_submit_benchmark_decode_steps,
        ),
    ];
    let warmup = (
        "official_prefill_warmup_runs",
        contract.official_prefill_warmup_runs,
    );
    if counts.iter().all(|(_, v)| v.is_none()) && warmup.1.is_none() {
        return Ok(());
    }
    for (field, declared) in counts {
        match declared {
            Some(n) if n >= 1 => {}
            Some(n) => {
                return Err(format!(
                    "the --contract track fixture for {track_id:?} declares {field}: {n}; a window \
                     of zero steps measures and checks nothing"
                ))
            }
            None => return Err(partial_window_shape(track_id, field)),
        }
    }
    if warmup.1.is_none() {
        return Err(partial_window_shape(track_id, warmup.0));
    }
    Ok(())
}

/// The one refusal a PARTIAL window shape gets, naming the missing field.
fn partial_window_shape(track_id: &str, field: &str) -> String {
    format!(
        "the --contract track fixture for {track_id:?} declares a measurement window but no \
         {field}; the four window values are ONE shape (correctness_steps, benchmark_decode_steps, \
         local_submit_benchmark_decode_steps and official_prefill_warmup_runs) and are declared \
         together — a decode depth without its checked-step count describes a window nothing \
         judges"
    )
}

/// THE TRACK'S OFFICIAL BASELINE, CONTRACT-FIRST: the captured pair the fixture declares, or a
/// refusal.
///
/// An undeclared pair is NOT a value: the resolver refuses BY NAME and the refusal carries the
/// `OFFICIAL_BASELINE_PENDING` sentinel. The BANDS the pair carries resolve through
/// [`acceptance_bands`], so a fixture may declare a pair, a band shape, both, or neither.
pub fn official_baseline(
    contract: &Contract,
    track_id: &str,
) -> Result<Resolved<OfficialBaseline>, String> {
    match declared_baseline_pair(contract) {
        Some((prefill, decode)) => {
            // The band shape a DECLARED pair carries: the fixture's when it declares one. There is
            // no table row to borrow bands from here — a fixture that supplies its own denominator
            // must also say what band gates it, or there is nothing to gate the timed run with.
            let bands = declared_bands(contract).ok_or_else(|| {
                format!(
                    "the --contract track fixture for {track_id:?} declares an official baseline \
                     pair but no acceptance band shape; the pair is the reference the timed run is \
                     banded against, so a declared pair must carry its own six band values \
                     (prefill/decode up and down tolerances and the two *_down_enabled flags)"
                )
            })?;
            Ok(Resolved {
                value: OfficialBaseline {
                    prefill_seconds_per_token: prefill,
                    decode_seconds_per_token: decode,
                    bands,
                },
                source: ContractSource::Contract,
            })
        }
        None => Err(format!(
            "official baseline for track_id {track_id:?} is {OFFICIAL_BASELINE_PENDING}: the \
             --contract track fixture declares no \
             official_baseline_prefill_seconds_per_token / \
             official_baseline_decode_seconds_per_token pair; refusing to score"
        )),
    }
}

/// THE TRACK'S ACCEPTANCE BAND SHAPE, CONTRACT-FIRST: the fixture's six values when it declares
/// them, and a refusal by name when it does not.
///
/// There is no shape to borrow when the fixture declares none: the band is what gates the timed
/// run, so a track that states no band has nothing to gate it with.
pub fn acceptance_bands(
    contract: &Contract,
    track_id: &str,
) -> Result<Resolved<AcceptanceBands>, String> {
    match declared_bands(contract) {
        Some(bands) => Ok(Resolved {
            value: bands,
            source: ContractSource::Contract,
        }),
        None => Err(format!(
            "acceptance bands for track_id {track_id:?} are {ACCEPTANCE_BANDS_UNDECLARED}: the \
             --contract track fixture declares none of the six band values \
             (prefill/decode up and down tolerances and the two *_down_enabled flags); the band \
             is what gates the timed run, so benchd will not invent one; refusing"
        )),
    }
}

/// THE TWO COMPOSITE WEIGHTS, CONTRACT-FIRST: the fixture's pair when it declares one, else a
/// refusal.
///
/// [`ScoringWeights::DEFAULT`] (the `SCORE_*_WEIGHT` constants) is the 0.75/0.25 weighting every
/// track scored under before a fixture could declare its own. It is the reference's own pair, not
/// a fallback: this resolver never reads it, and a fixture that declares neither weight is refused
/// by name.
pub fn scoring_weights(
    contract: &Contract,
    track_id: &str,
) -> Result<Resolved<ScoringWeights>, String> {
    match (contract.score_decode_weight, contract.score_prefill_weight) {
        (Some(decode), Some(prefill)) => Ok(Resolved {
            value: ScoringWeights { decode, prefill },
            source: ContractSource::Contract,
        }),
        _ => Err(format!(
            "the --contract track fixture for {track_id:?} declares no score_decode_weight and \
             score_prefill_weight; the composite divides by their sum, so a run cannot be \
             weighted by a pair it was never given; refusing"
        )),
    }
}

/// The baseline pair the fixture DECLARES, or `None` when it declares neither half. A HALF pair
/// never reaches here — [`certify_official_baseline`] refuses it at the parse.
fn declared_baseline_pair(contract: &Contract) -> Option<(f64, f64)> {
    contract
        .official_baseline_prefill_seconds_per_token
        .zip(contract.official_baseline_decode_seconds_per_token)
}

/// The band shape the fixture DECLARES, or `None` when it declares none of it. A PARTIAL shape
/// never reaches here — [`certify_acceptance_bands`] refuses it at the parse.
fn declared_bands(contract: &Contract) -> Option<AcceptanceBands> {
    Some(AcceptanceBands {
        prefill_up_tolerance: contract.prefill_band_up_tolerance?,
        prefill_down_tolerance: contract.prefill_band_down_tolerance?,
        decode_up_tolerance: contract.decode_band_up_tolerance?,
        decode_down_tolerance: contract.decode_band_down_tolerance?,
        decode_down_enabled: contract.decode_band_down_enabled?,
        prefill_down_enabled: contract.prefill_band_down_enabled?,
    })
}

/// THE PAIR IS ONE CAPTURE, and its two halves must be finite and positive — a seconds-per-token
/// denominator of 0, a negative one or a non-finite one scores nothing.
///
/// A fixture that declares a pair AND `scores_against_live_control_leg: true` is refused too: those
/// tracks measure their own denominator and store none (David 2026-09-08), so a fixture that says
/// both has described two mutually exclusive designs and benchd cannot tell which one the operator
/// meant.
fn certify_official_baseline(contract: &Contract, track_id: &str) -> Result<(), String> {
    let halves = [
        (
            "official_baseline_prefill_seconds_per_token",
            contract.official_baseline_prefill_seconds_per_token,
        ),
        (
            "official_baseline_decode_seconds_per_token",
            contract.official_baseline_decode_seconds_per_token,
        ),
    ];
    if halves.iter().all(|(_, v)| v.is_none()) {
        return Ok(());
    }
    for (field, declared) in halves {
        match declared {
            Some(v) if v.is_finite() && v > 0.0 => {}
            Some(v) => {
                return Err(format!(
                    "the --contract track fixture for {track_id:?} declares {field}: {v}; a \
                     seconds-per-token baseline must be finite and greater than 0"
                ))
            }
            None => {
                return Err(format!(
                    "the --contract track fixture for {track_id:?} declares half an official \
                     baseline pair and no {field}; the prefill and decode seconds-per-token are \
                     ONE capture and are declared together — mixing a declared half with a table \
                     half would score against a pair no capture produced"
                ))
            }
        }
    }
    if contract.scores_against_live_control_leg == Some(true) {
        return Err(format!(
            "the --contract track fixture for {track_id:?} declares an official baseline pair AND \
             scores_against_live_control_leg: true; a track that measures its own denominator on \
             the box stores no pair (David 2026-09-08), so these two declarations describe \
             different designs and benchd will not pick one"
        ));
    }
    Ok(())
}

/// THE BAND SHAPE IS ONE DECLARATION: all six values or none. A tolerance must be finite and
/// non-negative; the `*_down_enabled` flags carry no range.
///
/// All six together, including the flags, because a tolerance whose flag is missing states a bound
/// without stating whether it is enforced — and `false` there is a real decision on both the MTP
/// timed leg and the paired design, never a default.
fn certify_acceptance_bands(contract: &Contract, track_id: &str) -> Result<(), String> {
    let tolerances = [
        (
            "prefill_band_up_tolerance",
            contract.prefill_band_up_tolerance,
        ),
        (
            "prefill_band_down_tolerance",
            contract.prefill_band_down_tolerance,
        ),
        (
            "decode_band_up_tolerance",
            contract.decode_band_up_tolerance,
        ),
        (
            "decode_band_down_tolerance",
            contract.decode_band_down_tolerance,
        ),
    ];
    let flags = [
        (
            "decode_band_down_enabled",
            contract.decode_band_down_enabled,
        ),
        (
            "prefill_band_down_enabled",
            contract.prefill_band_down_enabled,
        ),
    ];
    let any = tolerances.iter().any(|(_, v)| v.is_some()) || flags.iter().any(|(_, v)| v.is_some());
    if !any {
        return Ok(());
    }
    for (field, declared) in tolerances {
        match declared {
            Some(v) if v.is_finite() && v >= 0.0 => {}
            Some(v) => {
                return Err(format!(
                    "the --contract track fixture for {track_id:?} declares {field}: {v}; an \
                     acceptance-band tolerance must be finite and non-negative"
                ))
            }
            None => return Err(partial_band_shape(track_id, field)),
        }
    }
    for (field, declared) in flags {
        if declared.is_none() {
            return Err(partial_band_shape(track_id, field));
        }
    }
    Ok(())
}

/// The one refusal a PARTIAL band shape gets, naming the missing field.
fn partial_band_shape(track_id: &str, field: &str) -> String {
    format!(
        "the --contract track fixture for {track_id:?} declares an acceptance band shape but no \
         {field}; the six band values are ONE shape (prefill/decode up and down tolerances and the \
         two *_down_enabled flags) and are declared together — a tolerance without its enforcement \
         flag states a bound without saying whether it is enforced"
    )
}

/// THE TWO WEIGHTS ARE ONE DECLARATION: both or neither, each finite and non-negative, and their
/// SUM strictly positive — [`crate::score::score`] divides each weight by that sum.
fn certify_scoring_weights(contract: &Contract, track_id: &str) -> Result<(), String> {
    let pair = [
        ("score_decode_weight", contract.score_decode_weight),
        ("score_prefill_weight", contract.score_prefill_weight),
    ];
    if pair.iter().all(|(_, v)| v.is_none()) {
        return Ok(());
    }
    for (field, declared) in pair {
        match declared {
            Some(v) if v.is_finite() && v >= 0.0 => {}
            Some(v) => {
                return Err(format!(
                    "the --contract track fixture for {track_id:?} declares {field}: {v}; a \
                     scoring weight must be finite and non-negative"
                ))
            }
            None => {
                return Err(format!(
                    "the --contract track fixture for {track_id:?} declares half a scoring-weight \
                     pair and no {field}; the composite divides each weight by their SUM, so half \
                     a pair describes no weighting at all"
                ))
            }
        }
    }
    if contract.score_decode_weight.unwrap_or(0.0) + contract.score_prefill_weight.unwrap_or(0.0)
        <= 0.0
    {
        return Err(format!(
            "the --contract track fixture for {track_id:?} declares score_decode_weight 0 and \
             score_prefill_weight 0; the composite divides by their sum, so a zero-sum pair scores \
             nothing"
        ));
    }
    Ok(())
}

/// A `--contract` file AS IT WAS LOADED: the parsed, certified contract and the sha256 of the
/// EXACT bytes it was parsed from.
///
/// The two travel together for the whole run because the digest is only worth sealing if it covers
/// the bytes the decisions were actually made on. [`load`] is the ONE constructor — there is no way
/// to pair a contract with a digest of some other read of the file.
#[derive(Debug, Clone)]
pub struct LoadedContract {
    /// The parsed and certified fixture.
    pub contract: Contract,
    /// `sha256` of the exact file bytes [`load`] read and [`Contract::parse`] parsed. Sealed as
    /// `metrics.contract_sha256` — THE MIGRATION PIN (David 2026-09-15). The fixture decides the
    /// run's arm state, its speedup floors and its pair count; without this digest the sealed
    /// artifact records the verdicts but not the bytes that produced them, so a later reader cannot
    /// tell which fixture was in force. `metrics.golden_hash` records the golden the same way.
    pub sha256: String,
}

/// Read, digest, parse and CERTIFY a `--contract` file. ONE read of ONE file: every
/// contract-derived decision of a run, and the digest the run seals, describe the same bytes.
pub fn load(path: &Path) -> Result<LoadedContract, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("--contract read failed ({}): {e}", path.display()))?;
    let sha256 = crate::hash::sha256_hex(&bytes);
    Ok(LoadedContract {
        contract: Contract::parse(&bytes)?,
        sha256,
    })
}

/// The scored run's speedup floors, or the refusal naming what the fixture must declare.
///
/// Both floors are REQUIRED and each is refused on its own: an absent floor is not 0.95, and a
/// fixture that declares one axis has still not declared the other. A declared value must be
/// finite and positive — a floor of 0, NaN or a negative number gates nothing.
pub fn speedup_floors(contract: &Contract, track_id: &str) -> Result<SpeedupFloors, String> {
    Ok(SpeedupFloors {
        decode: one_floor(
            contract.decode_speedup_floor,
            "decode_speedup_floor",
            track_id,
        )?,
        prefill: one_floor(
            contract.prefill_speedup_floor,
            "prefill_speedup_floor",
            track_id,
        )?,
    })
}

/// One axis of [`speedup_floors`].
fn one_floor(declared: Option<f64>, field: &str, track_id: &str) -> Result<f64, String> {
    match declared {
        Some(v) if v.is_finite() && v > 0.0 => Ok(v),
        Some(v) => Err(format!(
            "the --contract track fixture for {track_id:?} declares {field}: {v}; a speedup floor \
             must be finite and greater than 0 (David 2026-09-09 ruled 0.95/0.95)"
        )),
        None => Err(format!(
            "the --contract track fixture for {track_id:?} declares no {field}; the official \
             scored run refuses to guess a speedup floor (David 2026-09-09 ruled 0.95/0.95, \
             configurable per project) — pin it in the fixture"
        )),
    }
}

/// The paired path's pair count, or the refusal naming what the fixture must declare.
pub fn official_pairs(contract: &Contract, track_id: &str) -> Result<usize, String> {
    match contract.official_pairs {
        Some(n) if n >= 1 => Ok(n as usize),
        Some(n) => Err(format!(
            "the --contract track fixture for {track_id:?} declares official_pairs: {n}; the paired \
             official run needs at least 1 pair (David 2026-09-09 ruled 2)"
        )),
        None => Err(format!(
            "the --contract track fixture for {track_id:?} declares no official_pairs; the paired \
             official run refuses to guess a pair count (David 2026-09-09 ruled 2 on both \
             platforms) — pin it in the fixture"
        )),
    }
}

/// THE BATCH SIZE IS A CONFIGURED WIDTH, not a constant benchd checks a fixture against (David
/// 2026-09-15: "Benchd should support batching. Batching as part of the configuration.").
///
/// The rule is therefore a RANGE rule and nothing more: a declared width must be a whole number of
/// streams, so it must be at least 1. `1` is the single-stream point; any greater width is the
/// batched cohort point. Which measured path that selects is [`scored_regime`]'s answer, not this
/// function's — certification asks only whether the declared value is usable.
///
/// Absent declares nothing and is not refused HERE; the resolver refuses absence by name.
fn certify_scored_batch_size(declared: Option<u32>, track_id: &str) -> Result<(), String> {
    match declared {
        None => Ok(()),
        Some(n) if n >= 1 => Ok(()),
        Some(n) => Err(format!(
            "the --contract track fixture for {track_id:?} declares scored_batch_size {n}; a \
             scored batch size is a count of concurrent streams and must be at least 1 (1 is the \
             single-stream point, a greater width is the batched cohort point); refusing"
        )),
    }
}

impl Contract {
    /// Parse a `--contract` file's bytes, FAIL-CLOSED on malformed JSON (never fall open), and
    /// CERTIFY the shape of everything the fixture declares ([`Contract::certify`]).
    ///
    /// Certification happens HERE, at the parse, rather than at each use. A fixture that declares a
    /// floor of `0`, a pair count of `0` or a batch width benchd cannot measure is refused when the
    /// file is read — before a worker spawns, before a gate runs, before any box time is spent —
    /// and the refusal names the file's own value. Deferring the same check to the use site would
    /// scatter it across the resolvers and let a malformed fixture reach a GPU window.
    pub fn parse(bytes: &[u8]) -> Result<Contract, String> {
        let contract: Contract =
            serde_json::from_slice(bytes).map_err(|e| format!("--contract parse failed: {e}"))?;
        contract.certify()?;
        Ok(contract)
    }

    /// RANGE AND SHAPE, once, over everything the fixture DECLARES.
    ///
    /// Certification never asks whether a field is PRESENT — presence is the resolvers' question
    /// ([`speedup_floors`], [`official_pairs`]), and each of them refuses absence with its own
    /// named diagnosis. This asks only whether a declared value is USABLE, and each rule lives in
    /// exactly one function so the parse-time check and the use-time check can never disagree.
    pub fn certify(&self) -> Result<(), String> {
        // The fixture's OWN declared track id names the file in these refusals: certification runs
        // at the parse, where the run's resolved track id does not exist yet.
        let track_id = self
            .track_id
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .unwrap_or("<no declared track_id>");
        if self.decode_speedup_floor.is_some() {
            one_floor(self.decode_speedup_floor, "decode_speedup_floor", track_id)?;
        }
        if self.prefill_speedup_floor.is_some() {
            one_floor(
                self.prefill_speedup_floor,
                "prefill_speedup_floor",
                track_id,
            )?;
        }
        if self.official_pairs.is_some() {
            official_pairs(self, track_id)?;
        }
        certify_scored_batch_size(self.scored_batch_size, track_id)?;
        certify_scored_regime(self, track_id)?;
        certify_official_baseline(self, track_id)?;
        certify_acceptance_bands(self, track_id)?;
        certify_scoring_weights(self, track_id)?;
        certify_window_shape(self, track_id)?;
        certify_model_shape(self, track_id)?;
        Ok(())
    }
}

/// David ruling (2026-08-26) — the ARM GATE: refuse a SCORING/ranked run whose `--contract`
/// track fixture does not declare `official_scoring_enabled: true`.
///
/// `scoring_mode` is the SAME signal every other scoring-vs-local decision keys on: a run is a
/// scoring run here exactly when it is an official (non `--local-dev`) run. It is deliberately NOT
/// a second, parallel notion of "official".
///
/// The three refusable states are kept DISTINCT in the message because they need different actions:
///
/// * `Some(true)` — armed. Proceed; this is the ONLY accepting state.
/// * `Some(false)` — declared UNARMED. The track exists and is being brought up; the fix is to
///   iterate with a local mode (or wait for the arm), never to edit the fixture locally.
/// * `None` — the fixture declares no arm state. FAIL-CLOSED, identically to `false`: a contract
///   that never says it is armed is not armed. This is the half that matters most — the flag was
///   invisible to benchd for its whole life, so "the key is simply missing" is the likeliest way a
///   track would otherwise slip into scoring unarmed.
///
/// Pure and total: it reads three values and returns a verdict, so the whole truth table is unit
/// testable without a box, a GPU, or a contract file.
pub fn enforce_official_scoring_enabled(
    scoring_mode: bool,
    official_scoring_enabled: Option<bool>,
    track_id: &str,
) -> Result<(), String> {
    // LOCAL modes are untouched, on purpose and load-bearing: the whole point of the unarmed
    // period is that participants and organizers can iterate against the real harness before the
    // track opens. Gating a local run would make the flag's `false` state mean "this track is
    // unusable", which is the opposite of what it is for.
    if !scoring_mode {
        return Ok(());
    }
    match official_scoring_enabled {
        Some(true) => Ok(()),
        Some(false) => Err(format!(
            "official scoring is not enabled for this track: the --contract track fixture for \
             {track_id:?} declares official_scoring_enabled: false, so benchd refuses to seal an \
             official/ranked scoring artifact for it. This is the track's ARM STATE and only the \
             track fixture may change it — pass --local-dev to iterate against the unarmed track \
             (no scoring seal), or wait for the track to be armed."
        )),
        None => Err(format!(
            "official scoring is not enabled for this track: the --contract track fixture for \
             {track_id:?} declares NO official_scoring_enabled at all, and an absent arm state is \
             NOT an armed one (fail-closed) — benchd refuses to seal an official/ranked scoring \
             artifact for it. Add official_scoring_enabled: true to the track fixture to arm it, \
             or pass --local-dev to iterate against the unarmed track (no scoring seal)."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The contract parses `official_scoring_enabled` as a TRI-STATE, and the three states stay
    /// DISTINCT: ABSENT survives as `None` (not collapse into `false`) so the refusal can tell the
    /// two apart.
    #[test]
    fn contract_parses_official_scoring_enabled_as_a_tri_state() {
        let armed =
            Contract::parse(br#"{"track_id":"t","official_scoring_enabled":true}"#).unwrap();
        assert_eq!(armed.official_scoring_enabled, Some(true));

        let unarmed =
            Contract::parse(br#"{"track_id":"t","official_scoring_enabled":false}"#).unwrap();
        assert_eq!(unarmed.official_scoring_enabled, Some(false));

        // ABSENT stays ABSENT — it must not read back as `false`, because the two refusals say
        // different things.
        let silent = Contract::parse(br#"{"track_id":"t"}"#).unwrap();
        assert_eq!(silent.official_scoring_enabled, None);
    }

    /// A fixture carrying the FULL measure-job schema still parses — serde ignores the keys the
    /// surviving official path does not model (no `deny_unknown_fields`), so a real track fixture
    /// with `timed_prompt_pool`/`allowed_modes`/`calibration`/`hidden_correctness_golden` reads its
    /// arm state exactly as a trimmed one.
    #[test]
    fn contract_ignores_unmodelled_fixture_keys() {
        let full = Contract::parse(
            br#"{"track_id":"t","official_scoring_enabled":true,
                 "timed_prompt_pool":[{"sha256":"ab","bytes":3}],
                 "allowed_modes":["mtp"],"scored_batch_size":1,
                 "calibration":{"expected_raw_median":0.994,"band_pct":2.0},
                 "hidden_correctness_golden":{"sha256":"cd","bytes":7},
                 "reference_model":{"repository":"r","revision":"v"}}"#,
        )
        .unwrap();
        assert_eq!(full.official_scoring_enabled, Some(true));
        assert_eq!(full.track_id.as_deref(), Some("t"));
        // `scored_batch_size` is the ONE key this struct started modelling: it is certified, not
        // ignored. Its RANGE rule is `a_declared_batch_width_is_any_positive_stream_count`; the
        // width the fixture states is the width the run measures at.
        assert_eq!(full.scored_batch_size, Some(1));
    }

    /// Malformed JSON is a FAIL-CLOSED parse error, never a fall-open default.
    #[test]
    fn contract_parse_fails_closed_on_malformed() {
        assert!(Contract::parse(br#"{"track_id": "#).is_err());
        assert!(Contract::parse(b"not json").is_err());
    }

    /// ARM GATE — the whole truth table of [`enforce_official_scoring_enabled`], the pure decision
    /// benchd's two pre-GPU call sites — the measure-job's and the official path's — are thin
    /// wrappers over. This is the ONE copy of the gate and the ONE truth table for it: the
    /// measure-job carried a byte-identical second implementation, with a second copy of this
    /// table, so both copies looked independently proven while neither could catch them drifting.
    ///
    /// REVERT-PROOF three ways. Delete the gate (always `Ok`) and both scoring refusals go red.
    /// Invert it (accept `false`/absent, refuse `true`) and every arm of this table goes red.
    /// Make it warn-only (`eprintln!` + `Ok`) and the two `is_err()` arms go red.
    #[test]
    fn official_scoring_arm_gate_truth_table() {
        // ARMED — the ONLY accepting scoring state.
        assert!(enforce_official_scoring_enabled(true, Some(true), "t").is_ok());

        // DECLARED UNARMED — refuses, names the flag, and points at the local escape hatch.
        let declared_false = enforce_official_scoring_enabled(true, Some(false), "gemma4-track")
            .expect_err("a scoring run over an unarmed track must refuse");
        assert!(
            declared_false.contains("official scoring is not enabled for this track"),
            "the refusal must lead with the ruled wording: {declared_false}"
        );
        assert!(
            declared_false.contains("official_scoring_enabled")
                && declared_false.contains("gemma4-track"),
            "the refusal must NAME the flag and the track: {declared_false}"
        );
        assert!(
            declared_false.contains("--local-dev"),
            "the refusal must name the un-gated local path: {declared_false}"
        );

        // ABSENT — fail-closed, identically refused, but diagnosed differently: this fixture never
        // declared an arm state, so the remedy is to ADD the key, not to wait for a flip.
        let absent = enforce_official_scoring_enabled(true, None, "silent-track")
            .expect_err("absence is not armed");
        assert!(
            absent.contains("official scoring is not enabled for this track")
                && absent.contains("official_scoring_enabled"),
            "the absent-case refusal must carry the same named verdict: {absent}"
        );
        assert_ne!(
            absent, declared_false,
            "absent and false must not produce the SAME message — they need different actions"
        );

        // LOCAL — the load-bearing NEGATIVE control. A non-scoring run is not gated, so NONE of the
        // three contract states may refuse it: iterating against an unarmed track is the entire
        // purpose of the unarmed period, and a gate that blocked it would invert the flag's meaning
        // from "not scoring yet" into "unusable".
        for state in [Some(true), Some(false), None] {
            assert!(
                enforce_official_scoring_enabled(false, state, "t").is_ok(),
                "a non-scoring run must be unaffected by official_scoring_enabled = {state:?}"
            );
        }
    }
}

#[cfg(test)]
mod official_pairs_tests {
    use super::*;

    #[test]
    fn the_pair_count_comes_from_the_fixture_alone() {
        let two =
            Contract::parse(br#"{"official_scoring_enabled": true, "official_pairs": 2}"#).unwrap();
        assert_eq!(official_pairs(&two, "t"), Ok(2));
        let absent = Contract::parse(br#"{"official_scoring_enabled": true}"#).unwrap();
        let err = official_pairs(&absent, "qwen3.8-125b-a6b-cuda-v1").unwrap_err();
        assert!(err.contains("declares no official_pairs"), "{err}");
        assert!(err.contains("qwen3.8-125b-a6b-cuda-v1"), "{err}");
        // A declared pair count of 0 is refused AT THE PARSE (`Contract::certify`), naming the
        // fixture's own value.
        let err = Contract::parse(br#"{"official_scoring_enabled": true, "official_pairs": 0}"#)
            .unwrap_err();
        assert!(err.contains("official_pairs: 0"), "{err}");
    }
}

#[cfg(test)]
mod speedup_floor_tests {
    use super::*;

    /// The floors come from the FIXTURE alone, one axis at a time, and an absent or unusable
    /// value is a refusal that names the field and the track — never a silent 0.95.
    #[test]
    fn the_floors_come_from_the_fixture_alone() {
        let ruled = Contract::parse(
            br#"{"official_scoring_enabled": true, "official_pairs": 2,
                 "decode_speedup_floor": 0.95, "prefill_speedup_floor": 0.95}"#,
        )
        .unwrap();
        assert_eq!(
            speedup_floors(&ruled, "t"),
            Ok(SpeedupFloors {
                decode: 0.95,
                prefill: 0.95
            })
        );

        // A project may declare its own pair; benchd enforces what the fixture says.
        let per_project =
            Contract::parse(br#"{"decode_speedup_floor": 0.90, "prefill_speedup_floor": 0.80}"#)
                .unwrap();
        assert_eq!(
            speedup_floors(&per_project, "t"),
            Ok(SpeedupFloors {
                decode: 0.90,
                prefill: 0.80
            })
        );

        // BOTH are required, and each refusal names its own field.
        let decode_only = Contract::parse(br#"{"decode_speedup_floor": 0.95}"#).unwrap();
        let err = speedup_floors(&decode_only, "qwen3.8-125b-a6b-cuda-v1").unwrap_err();
        assert!(err.contains("declares no prefill_speedup_floor"), "{err}");
        assert!(err.contains("qwen3.8-125b-a6b-cuda-v1"), "{err}");

        let neither = Contract::parse(br#"{"official_scoring_enabled": true}"#).unwrap();
        let err = speedup_floors(&neither, "t").unwrap_err();
        assert!(err.contains("declares no decode_speedup_floor"), "{err}");
        assert!(err.contains("David 2026-09-09"), "{err}");

        // Non-finite / non-positive declarations gate nothing, so they are refused — AT THE PARSE
        // now (`Contract::certify`), which is the whole point of certifying shape at load: the file
        // is refused when it is read, not when a resolver reaches for the value.
        for body in [
            &br#"{"decode_speedup_floor": 0.0, "prefill_speedup_floor": 0.95}"#[..],
            &br#"{"decode_speedup_floor": -0.5, "prefill_speedup_floor": 0.95}"#[..],
            &br#"{"decode_speedup_floor": 0.95, "prefill_speedup_floor": 0.0}"#[..],
        ] {
            let err = Contract::parse(body).unwrap_err();
            assert!(err.contains("finite and greater than 0"), "{err}");
        }
        // JSON has no NaN literal; a non-finite value reaches the resolver only as a struct.
        let nan = Contract {
            decode_speedup_floor: Some(f64::NAN),
            prefill_speedup_floor: Some(0.95),
            ..Contract::NONE_DECLARED
        };
        assert!(speedup_floors(&nan, "t")
            .unwrap_err()
            .contains("finite and greater than 0"));
    }
}

#[cfg(test)]
mod certification_tests {
    use super::*;

    /// CERTIFICATION IS AT THE PARSE. Every declared value the fixture carries is range-checked
    /// when the file is read, so a malformed fixture is refused before a worker spawns.
    ///
    /// REVERT-PROOF: delete the `contract.certify()?` line from [`Contract::parse`] and every
    /// `unwrap_err` below goes red.
    #[test]
    fn a_declared_value_is_range_checked_at_the_parse() {
        // A well-formed fixture parses, and certification changes none of its values.
        let ok = Contract::parse(
            br#"{"track_id":"qwen3.8-125b-a6b-mlx-v1","official_scoring_enabled":true,
                 "official_pairs":2,"decode_speedup_floor":0.95,"prefill_speedup_floor":0.95,
                 "scored_batch_size":1}"#,
        )
        .expect("a well-formed fixture certifies");
        assert_eq!(ok.official_pairs, Some(2));
        assert_eq!(ok.scored_batch_size, Some(1));
        assert_eq!(
            speedup_floors(&ok, "qwen3.8-125b-a6b-mlx-v1"),
            Ok(SpeedupFloors {
                decode: 0.95,
                prefill: 0.95
            })
        );

        // The refusal names the FIXTURE'S OWN track id, because at the parse there is no resolved
        // run track id to name.
        let err = Contract::parse(br#"{"track_id":"gemma4-26b-a4b-mlx-v1","official_pairs":0}"#)
            .unwrap_err();
        assert!(err.contains("gemma4-26b-a4b-mlx-v1"), "{err}");

        // …and a fixture that declares no track id still names the condition rather than nothing.
        let err = Contract::parse(br#"{"official_pairs":0}"#).unwrap_err();
        assert!(err.contains("<no declared track_id>"), "{err}");
    }

    /// The declared BATCH WIDTH is a COUNT OF STREAMS: any whole number of streams parses, and
    /// only a width below one refuses. David 2026-09-15 made batching part of the configuration,
    /// so the width the fixture states is the width the run measures at — there is no in-binary
    /// width for it to have to agree with.
    #[test]
    fn a_declared_batch_width_is_any_positive_stream_count() {
        assert!(Contract::parse(br#"{"scored_batch_size":1}"#).is_ok());
        assert!(
            Contract::parse(br#"{"track_id":"t","scored_batch_size":8}"#).is_ok(),
            "a batched width is a configuration, not a defect"
        );
        assert!(
            Contract::parse(br#"{"official_scoring_enabled":true}"#).is_ok(),
            "an absent scored_batch_size declares nothing and must not refuse at the parse"
        );
        let err = Contract::parse(br#"{"track_id":"t","scored_batch_size":0}"#).unwrap_err();
        assert!(err.contains("scored_batch_size 0"), "{err}");
        assert!(err.contains("at least 1"), "{err}");
    }

    /// THE MIGRATION PIN (David 2026-09-15). [`load`] digests the EXACT bytes it parsed, so the
    /// digest a run seals and the values a run enforced come from one read of one file.
    ///
    /// REVERT-PROOF: digest anything other than the parsed bytes — a second read, a normalised
    /// re-serialisation — and the equality below goes red.
    #[test]
    fn the_loaded_digest_covers_the_exact_parsed_bytes() {
        let dir = std::env::temp_dir().join(format!(
            "benchd-contract-load-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("track.json");
        // Trailing newline and indentation included on purpose: the digest covers the FILE, not a
        // re-rendering of the parsed values.
        let bytes = b"{\n  \"track_id\": \"qwen3.8-125b-a6b-mlx-v1\",\n  \"official_scoring_enabled\": true,\n  \"official_pairs\": 2,\n  \"decode_speedup_floor\": 0.95,\n  \"prefill_speedup_floor\": 0.95\n}\n";
        std::fs::write(&path, bytes).unwrap();

        let loaded = load(&path).expect("a well-formed fixture loads");
        assert_eq!(loaded.sha256, crate::hash::sha256_hex(bytes));
        assert_eq!(loaded.sha256.len(), 64);
        assert_eq!(loaded.contract.official_pairs, Some(2));

        // One edited byte moves the digest — which is the whole reason to seal it.
        let mut edited = bytes.to_vec();
        let at = edited
            .windows(4)
            .position(|w| w == b"0.95")
            .expect("the floor literal is in the fixture");
        edited[at + 3] = b'0';
        std::fs::write(&path, &edited).unwrap();
        let reloaded = load(&path).expect("0.90 is still a usable floor");
        assert_ne!(reloaded.sha256, loaded.sha256);

        // An unreadable fixture is a fail-closed refusal, never an empty digest.
        std::fs::remove_file(&path).unwrap();
        assert!(load(&path).unwrap_err().contains("--contract read failed"));
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod fixture_oracle_tests {
    use super::*;

    /// Every checked-in JSON file in this repository that declares a `track_id` — the shape of a
    /// `--contract` track fixture — must PARSE and CERTIFY.
    ///
    /// This is the ORACLE for the constants→contract migration: certification may only ever refuse
    /// a fixture this tree does not ship. It walks the tree at test time rather than naming files,
    /// so a fixture added later is covered without anyone remembering to add it here.
    ///
    /// SCOPE, stated because it bounds what this test can prove: the REAL track fixtures
    /// (`qwen3_8_125b_a6b_track.json`, `qwen3_8_27b_mtp_track.json`, `gemma4_26b_a4b_track.json`,
    /// `laguna_xs_2_1_dflash_track.json`) live in the ENGINE repositories, not here — benchd ships
    /// none of them. The per-track and per-platform tables in `bench_core::constants` are therefore
    /// still the only in-tree declaration of those tracks' baselines, regimes and identities, and
    /// this oracle cannot stand in for one over fixtures it cannot see.
    #[test]
    fn every_checked_in_track_fixture_parses_and_certifies() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("crates/bench-core is two levels below the repository root")
            .to_path_buf();

        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name();
                let name = name.to_string_lossy();
                // `target/` is build output and `dist/` holds prebuilt binaries: neither is source.
                if path.is_dir() {
                    if name != "target" && name != "dist" && name != ".git" {
                        walk(&path, out);
                    }
                } else if name.ends_with(".json") {
                    out.push(path);
                }
            }
        }

        let mut files = Vec::new();
        walk(&root, &mut files);
        assert!(
            !files.is_empty(),
            "the oracle found no JSON at all under {} — the walk is broken, not the tree",
            root.display()
        );

        let mut certified = Vec::new();
        for path in files {
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            // Contract-SHAPED means: a JSON object with a string `track_id`. Every other JSON in
            // the tree (goldens, fuzz corpora, manifests) is not a track fixture.
            let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
                continue;
            };
            if !value
                .get("track_id")
                .is_some_and(serde_json::Value::is_string)
            {
                continue;
            }
            let parsed = Contract::parse(&bytes).unwrap_or_else(|e| {
                panic!(
                    "the track fixture this tree ships does not certify: {} — {e}",
                    path.display()
                )
            });
            // …and a group this fixture is SILENT about now RESOLVES TO A REFUSAL, by name. STEP 3
            // (David 2026-09-15) deleted the per-track tables, so silence is no longer answered by
            // an in-tree value — it is answered by a message that says what the file did not
            // declare. Checked here, over every contract-shaped fixture this tree ships, so a
            // resolver that quietly reintroduced a default would go red on the first silent group.
            if let Some(track_id) = parsed.track_id.as_deref() {
                if parsed.prefill_gain_exponent.is_none() {
                    let err = scored_regime(&parsed, track_id).unwrap_err();
                    assert!(
                        err.contains(crate::constants::SCORED_REGIME_PENDING),
                        "{}: {err}",
                        path.display()
                    );
                }
                if parsed.scores_against_live_control_leg.is_none() {
                    assert!(scores_against_live_control_leg(&parsed, track_id).is_err());
                }
                if parsed.paired_flow_retired.is_none() {
                    assert!(paired_flow_retired(&parsed, track_id).is_err());
                }
                if parsed.official_baseline_prefill_seconds_per_token.is_none() {
                    let err = official_baseline(&parsed, track_id).unwrap_err();
                    assert!(
                        err.contains(crate::constants::OFFICIAL_BASELINE_PENDING),
                        "{}: {err}",
                        path.display()
                    );
                }
                if parsed.prefill_band_up_tolerance.is_none() {
                    let err = acceptance_bands(&parsed, track_id).unwrap_err();
                    assert!(
                        err.contains(crate::constants::ACCEPTANCE_BANDS_UNDECLARED),
                        "{}: {err}",
                        path.display()
                    );
                }
                if parsed.benchmark_decode_steps.is_none() {
                    let err = window_shape(&parsed, track_id).unwrap_err();
                    assert!(
                        err.contains(crate::constants::WINDOW_SHAPE_UNDECLARED),
                        "{}: {err}",
                        path.display()
                    );
                }
                if parsed.vocab_size.is_none() {
                    let err = model_identity(&parsed, track_id).unwrap_err();
                    assert!(
                        err.contains(crate::constants::MODEL_IDENTITY_UNDECLARED),
                        "{}: {err}",
                        path.display()
                    );
                }
                if parsed.score_decode_weight.is_none() {
                    assert!(scoring_weights(&parsed, track_id).is_err());
                }
            }
            certified.push(path);
        }
        assert!(
            !certified.is_empty(),
            "no contract-shaped fixture was found; if benchd stopped shipping one, say so here \
             rather than letting the oracle pass vacuously"
        );
    }
}

#[cfg(test)]
mod round_trip_oracle_tests {
    use super::*;
    use crate::constants;

    /// The directory holding one FULL fixture per track this tree carries a reference copy of.
    /// Named here once: the engine-repository lane copies these files verbatim, so the path is part
    /// of the migration's interface, not a test detail.
    const FULL_FIXTURE_DIR: &str = "crates/benchd/tests/fixtures/contract-full";

    /// The repository root, two levels above this crate.
    fn repo_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("crates/bench-core is two levels below the repository root")
            .to_path_buf()
    }

    /// WHAT ONE TRACK'S FIXTURE MUST RESOLVE TO, written out here as DATA.
    ///
    /// These numbers are the per-track tables' own, copied into this file BEFORE the tables were
    /// deleted, so the values survive the deletion as TEST DATA rather than as a second production
    /// source. `None` is a group the fixture declares nothing for and the resolver therefore
    /// REFUSES — which is a pinned outcome here, not a skipped one.
    struct Expected {
        track_id: &'static str,
        regime: Option<(usize, f64, f64)>,
        live_control_leg: bool,
        paired_flow_retired: bool,
        baseline: Option<(f64, f64)>,
        /// `(prefill_up, prefill_down, decode_up, decode_down, decode_down_on, prefill_down_on)`.
        bands: (f64, f64, f64, f64, bool, bool),
        /// `(decode, prefill)`.
        weights: (f64, f64),
        /// `(correctness, benchmark_decode, local_submit_decode, official_prefill_warmup,
        /// official_prefill_timed)`.
        window: Option<(usize, usize, usize, usize, usize)>,
        /// `(golden_model_type, vocab_size, num_hidden_layers, seed_tokens)`.
        model: (&'static str, usize, i64, usize),
    }

    /// THE EXPECTATIONS, one per checked-in full fixture.
    ///
    /// PROVENANCE — every value below is what the deleted table gave that track:
    ///
    /// * `qwen3.8-27b-mtp-v1` — `SCORED_REGIMES_BY_TRACK`'s one row (the SINGLE-STREAM decode-only
    ///   point: `prefill_gain_exponent 0.0`, `decode_gain_exponent 1.0`, so the composite over any
    ///   prefill gain whatsoever is the decode gain itself, which is the number this track
    ///   publishes); `OFFICIAL_BASELINES_BY_TRACK`'s captured pair, carried at full precision from
    ///   the reference `Constants.swift:255-256` @ `b26f76f`; the legacy TWO-SIDED acceptance bands
    ///   (prefill ±3 %, decode +1 %/-2.5 %, both down bands enabled);
    ///   `MODEL_IDENTITIES_BY_TRACK`'s pre-ruling 512-token seed. It named NO platform
    ///   (`Platform::from_track_id` refuses its pre-canonical id), so it had no table window and
    ///   its fixture declares none.
    /// * `qwen3.8-125b-a6b-{mlx,cuda}-v1` — `LIVE_CONTROL_LEG_TRACKS` and `SINGLE_LEG_ONLY_TRACKS`
    ///   both listed them, so each measures its own denominator and stores NO pair;
    ///   `MTP_SINGLE_LEG_BANDS` (prefill +/-5% symmetric, decode +2% up with the down band
    ///   disabled); the window constants, whose warm-up count was the one PER-PLATFORM value — 1 on
    ///   MLX, 0 on CUDA, because the CUDA adapter's resident engine is already warm.
    ///
    /// * `nemotron3.5-lightning-30b-a3b-cuda-v1` — the first track declared contract-first from
    ///   the start: the 125B CUDA regime (live control leg, the same bands and weights) over the
    ///   Nemotron checkpoint's own shape (`nemotron_h`, vocabulary 131072, 52 decoder layers). Its
    ///   window differs on purpose: a 4096-token seed, ONE unmeasured prefill warm-up pass and
    ///   the MEDIAN of FIVE timed prefill passes,
    ///   because a single post-cool-gate 1024-token prefill on this 3B-active model is
    ///   bandwidth-bound and varied 3–8% pass to pass (above the 1% calibration maximum). No
    ///   table ever held it.
    ///
    /// `gemma4-26b-a4b-mlx-v1` is DELIBERATELY ABSENT. It does not migrate: it runs its own channel
    /// benchd, and this tree carries none of its values any more (David 2026-09-15, "gemma branch
    /// is untouched"). Its reference fixture was deleted with the tables.
    const EXPECTED: &[Expected] = &[
        Expected {
            track_id: "qwen3.8-27b-mtp-v1",
            regime: Some((1, 0.0, 1.0)),
            live_control_leg: false,
            paired_flow_retired: false,
            baseline: Some((0.00036751938916015625, 0.01385621216015625)),
            bands: (0.03, 0.03, 0.01, 0.025, true, true),
            weights: (0.75, 0.25),
            window: None,
            model: ("qwen3_5_text", 248_320, 64, 512),
        },
        Expected {
            track_id: "qwen3.8-125b-a6b-mlx-v1",
            regime: None,
            live_control_leg: true,
            paired_flow_retired: true,
            baseline: None,
            bands: (0.05, 0.05, 0.02, 0.05, false, false),
            weights: (0.75, 0.25),
            window: Some((64, 128, 1023, 1, 1)),
            model: ("qwen4_exp_text", 248_320, 48, 1_024),
        },
        Expected {
            track_id: "qwen3.8-125b-a6b-cuda-v1",
            regime: None,
            live_control_leg: true,
            paired_flow_retired: true,
            baseline: None,
            bands: (0.05, 0.05, 0.02, 0.05, false, false),
            weights: (0.75, 0.25),
            window: Some((64, 128, 1023, 0, 1)),
            model: ("qwen4_exp_text", 248_320, 48, 1_024),
        },
        Expected {
            track_id: "nemotron3.5-lightning-30b-a3b-cuda-v1",
            regime: None,
            live_control_leg: true,
            paired_flow_retired: true,
            baseline: None,
            bands: (0.05, 0.05, 0.02, 0.05, false, false),
            weights: (0.75, 0.25),
            window: Some((64, 128, 1023, 1, 5)),
            model: ("nemotron_h", 131_072, 52, 4_096),
        },
    ];

    /// THE ORACLE, converted: each checked-in full fixture RESOLVES TO THE CHECKED-IN EXPECTED
    /// VALUES, group by group, value for value and refusal for refusal.
    ///
    /// It used to compare the fixture against the per-track tables. The tables are gone, so the
    /// numbers they held live HERE, as the expectation. That is the whole conversion: the same
    /// proof, with the reference moved from a production table into test data, so nothing in the
    /// binary can agree with the fixture by construction.
    ///
    /// REVERT-PROOF: change any value in a fixture and its group goes red; change any expectation
    /// and the same group goes red; make a resolver return something of its own and every declared
    /// group goes red; delete a fixture and the enumeration goes red.
    #[test]
    fn every_full_fixture_resolves_to_its_expected_values() {
        let dir = repo_root().join(FULL_FIXTURE_DIR);

        let mut seen = Vec::new();
        for want in EXPECTED {
            let track_id = want.track_id;
            let path = dir.join(format!("{track_id}.json"));
            let bytes = std::fs::read(&path).unwrap_or_else(|e| {
                panic!(
                    "{track_id:?} has no full fixture at {} ({e})",
                    path.display()
                )
            });
            let full =
                Contract::parse(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert_eq!(
                full.track_id.as_deref(),
                Some(track_id),
                "{} must declare its own track_id",
                path.display()
            );

            // 1. THE SCORED REGIME.
            match want.regime {
                Some((batch, prefill, decode)) => {
                    let got = scored_regime(&full, track_id)
                        .unwrap_or_else(|e| panic!("{track_id}: scored_regime: {e}"));
                    assert_eq!(got.value.scored_batch_size, batch, "{track_id}: batch size");
                    assert_eq!(
                        got.value.prefill_gain_exponent, prefill,
                        "{track_id}: prefill exponent"
                    );
                    assert_eq!(
                        got.value.decode_gain_exponent, decode,
                        "{track_id}: decode exponent"
                    );
                    assert_eq!(got.source, ContractSource::Contract);
                }
                None => {
                    let err = scored_regime(&full, track_id)
                        .expect_err("{track_id}: an undeclared regime must refuse");
                    assert!(
                        err.contains(constants::SCORED_REGIME_PENDING),
                        "{track_id}: {err}"
                    );
                }
            }

            // 2. THE TWO FLOW SELECTORS.
            let leg = scores_against_live_control_leg(&full, track_id)
                .unwrap_or_else(|e| panic!("{track_id}: live_control_leg: {e}"));
            assert_eq!(leg.value, want.live_control_leg, "{track_id}: control leg");
            assert_eq!(leg.source, ContractSource::Contract);
            let paired = paired_flow_retired(&full, track_id)
                .unwrap_or_else(|e| panic!("{track_id}: paired_flow: {e}"));
            assert_eq!(
                paired.value, want.paired_flow_retired,
                "{track_id}: paired flow"
            );
            assert_eq!(paired.source, ContractSource::Contract);

            // 3. THE BASELINE PAIR.
            match want.baseline {
                Some((prefill, decode)) => {
                    let got = official_baseline(&full, track_id)
                        .unwrap_or_else(|e| panic!("{track_id}: official_baseline: {e}"));
                    assert_eq!(
                        got.value.prefill_seconds_per_token, prefill,
                        "{track_id}: baseline prefill"
                    );
                    assert_eq!(
                        got.value.decode_seconds_per_token, decode,
                        "{track_id}: baseline decode"
                    );
                    assert_eq!(got.source, ContractSource::Contract);
                }
                None => {
                    let err = official_baseline(&full, track_id)
                        .expect_err("a track that measures its own denominator stores no pair");
                    assert!(
                        err.contains(constants::OFFICIAL_BASELINE_PENDING),
                        "{track_id}: {err}"
                    );
                }
            }

            // 4. THE BAND SHAPE.
            let bands = acceptance_bands(&full, track_id)
                .unwrap_or_else(|e| panic!("{track_id}: acceptance_bands: {e}"));
            let (pu, pd, du, dd, dd_on, pd_on) = want.bands;
            assert_eq!(
                (
                    bands.value.prefill_up_tolerance,
                    bands.value.prefill_down_tolerance,
                    bands.value.decode_up_tolerance,
                    bands.value.decode_down_tolerance,
                    bands.value.decode_down_enabled,
                    bands.value.prefill_down_enabled,
                ),
                (pu, pd, du, dd, dd_on, pd_on),
                "{track_id}: band shape"
            );
            assert_eq!(bands.source, ContractSource::Contract);

            // 5. THE COMPOSITE WEIGHTS.
            let weights = scoring_weights(&full, track_id)
                .unwrap_or_else(|e| panic!("{track_id}: scoring_weights: {e}"));
            assert_eq!(
                (weights.value.decode, weights.value.prefill),
                want.weights,
                "{track_id}: composite weights"
            );
            assert_eq!(weights.source, ContractSource::Contract);

            // 6. THE MEASUREMENT WINDOW.
            match want.window {
                Some((correctness, decode_steps, submit_steps, warmup, timed)) => {
                    let got = window_shape(&full, track_id)
                        .unwrap_or_else(|e| panic!("{track_id}: window_shape: {e}"));
                    assert_eq!(
                        (
                            got.value.correctness_steps,
                            got.value.benchmark_decode_steps,
                            got.value.local_submit_benchmark_decode_steps,
                            got.value.official_prefill_warmup_runs,
                            got.value.official_prefill_timed_runs,
                        ),
                        (correctness, decode_steps, submit_steps, warmup, timed),
                        "{track_id}: window"
                    );
                    assert_eq!(got.source, ContractSource::Contract);
                }
                None => {
                    let err = window_shape(&full, track_id)
                        .expect_err("an undeclared window must refuse");
                    assert!(
                        err.contains(constants::WINDOW_SHAPE_UNDECLARED),
                        "{track_id}: {err}"
                    );
                }
            }

            // 7. THE MODEL SHAPE.
            let model = model_identity(&full, track_id)
                .unwrap_or_else(|e| panic!("{track_id}: model_identity: {e}"));
            let (model_type, vocab, layers, seed) = want.model;
            assert_eq!(
                model.value.golden_model_type, model_type,
                "{track_id}: type"
            );
            assert_eq!(model.value.vocab_size, vocab, "{track_id}: vocab");
            assert_eq!(model.value.num_hidden_layers, layers, "{track_id}: layers");
            assert_eq!(model.value.seed_tokens, seed, "{track_id}: seed");
            assert_eq!(model.source, ContractSource::Contract);

            seen.push(format!("{track_id}.json"));
        }

        // The directory holds fixtures for EXACTLY the expected tracks: a stale file nothing
        // states an expectation for is a fixture this oracle proves nothing about.
        let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
            .expect("the full-fixture directory exists")
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".json"))
            .collect();
        on_disk.sort();
        seen.sort();
        assert_eq!(
            on_disk, seen,
            "the full-fixture directory must hold exactly one fixture per expected track"
        );
    }

    /// A SILENT FIXTURE RESOLVES NOTHING. Every group refuses BY NAME, and the name is the same
    /// sentinel the per-track table used to refuse an undeclared track with — the refusals moved,
    /// they were not invented.
    ///
    /// This is the half of STEP 3 that changes behaviour: STEP 1's fall-back arm answered silence
    /// with an in-tree value, and there is no in-tree value left to answer it with.
    #[test]
    fn a_silent_fixture_refuses_every_group_by_name() {
        const TRACK: &str = "qwen3.8-125b-a6b-mlx-v1";
        let silent = &Contract::NONE_DECLARED;
        for (group, err) in [
            (
                constants::SCORED_REGIME_PENDING,
                scored_regime(silent, TRACK).unwrap_err(),
            ),
            (
                constants::OFFICIAL_BASELINE_PENDING,
                official_baseline(silent, TRACK).unwrap_err(),
            ),
            (
                constants::ACCEPTANCE_BANDS_UNDECLARED,
                acceptance_bands(silent, TRACK).unwrap_err(),
            ),
            (
                constants::WINDOW_SHAPE_UNDECLARED,
                window_shape(silent, TRACK).unwrap_err(),
            ),
            (
                constants::MODEL_IDENTITY_UNDECLARED,
                model_identity(silent, TRACK).unwrap_err(),
            ),
        ] {
            assert!(err.contains(group), "the refusal must name {group}: {err}");
            assert!(
                err.contains(TRACK),
                "the refusal must name the track: {err}"
            );
            assert!(
                !err.contains("  "),
                "the message must carry no padding-space runs: {err}"
            );
        }
        // The two flow selectors and the weights refuse too; they carry no sentinel of their own
        // because there is no "pending" state to name — a fixture either declares them or it does
        // not.
        assert!(scores_against_live_control_leg(silent, TRACK).is_err());
        assert!(paired_flow_retired(silent, TRACK).is_err());
        assert!(scoring_weights(silent, TRACK).is_err());
    }

    /// THE SEALED SPELLINGS ROUND-TRIP, AND `table` STILL READS. [`ContractSource::key`] and serde
    /// agree on every variant, and an artifact a STEP-1 binary sealed — which wrote `"table"` —
    /// still deserializes to the variant it named. Only `contract` and `device` are ever emitted
    /// ([`sources`]); `table` is read-only evidence.
    #[test]
    fn every_source_spelling_round_trips_and_table_still_reads() {
        for want in [
            ContractSource::Contract,
            ContractSource::Device,
            ContractSource::Table,
        ] {
            let sealed = serde_json::to_string(&want).unwrap();
            assert_eq!(sealed, format!("\"{}\"", want.key()));
            assert_eq!(
                serde_json::from_str::<ContractSource>(&sealed).unwrap(),
                want,
                "an artifact spelling {sealed} must read back as the variant it named"
            );
        }
    }

    /// A SEALED SOURCE IS `contract` OR `device`, NEVER `table`. There is no table left for a run
    /// to read: a group the fixture declares seals `contract`, and the two groups the 125B fixtures
    /// leave undeclared are measured on the box, so they seal `device`.
    #[test]
    fn a_full_fixture_seals_every_group_as_contract_or_device() {
        let dir = repo_root().join(FULL_FIXTURE_DIR);
        for want in EXPECTED {
            let bytes = std::fs::read(dir.join(format!("{}.json", want.track_id))).unwrap();
            let full = Contract::parse(&bytes).unwrap();
            let got = sources(&full, want.track_id);
            // The groups this tree's reference fixtures declare. The 125B fixtures declare no
            // regime and no pair — each of those devices measures its own reference benchmark — so
            // those two seal `device`, and they are pinned as such rather than glossed.
            assert_eq!(
                got.scored_regime,
                if want.regime.is_some() {
                    ContractSource::Contract
                } else {
                    ContractSource::Device
                }
            );
            assert_eq!(
                got.official_baseline,
                if want.baseline.is_some() {
                    ContractSource::Contract
                } else {
                    ContractSource::Device
                }
            );
            for (group, source) in [
                ("live_control_leg", got.live_control_leg),
                ("paired_flow", got.paired_flow),
                ("acceptance_bands", got.acceptance_bands),
                ("scoring_weights", got.scoring_weights),
                ("model_shape", got.model_shape),
            ] {
                assert_eq!(
                    source,
                    ContractSource::Contract,
                    "{}: {group} must seal as the contract",
                    want.track_id
                );
            }
        }
    }

    /// THE FIXTURE IS THE ONLY AUTHORITY, and it says what the in-tree lists used to say the
    /// opposite of. A fixture may declare either flow flag either way, for any track, and the
    /// resolver carries the declaration — there is no list left to override.
    ///
    /// This is the negative control for the oracle above, whose expectations were copied FROM the
    /// deleted tables: without it, a resolver that still had a table wired into it could pass every
    /// assertion there. Both flags are flipped against what the tables said for both tracks, so a
    /// hard-wired answer goes red on all four.
    #[test]
    fn the_fixture_decides_both_flow_flags_for_any_track() {
        // `qwen3.8-125b-a6b-mlx-v1` WAS in both in-tree lists (live-control-leg, single-leg-only).
        // A fixture that says otherwise is believed.
        const WAS_LISTED: &str = "qwen3.8-125b-a6b-mlx-v1";
        let flipped = Contract::parse(
            br#"{"track_id":"qwen3.8-125b-a6b-mlx-v1","scores_against_live_control_leg":false,
                 "paired_flow_retired":false}"#,
        )
        .unwrap();
        assert!(
            !scores_against_live_control_leg(&flipped, WAS_LISTED)
                .unwrap()
                .value
        );
        assert!(!paired_flow_retired(&flipped, WAS_LISTED).unwrap().value);
        assert!(enforce_paired_flow_available(&flipped, WAS_LISTED).is_ok());

        // `qwen3.8-27b-mtp-v1` was in NEITHER list. A fixture that opts it into both is believed
        // too, and the paired-flow fence then refuses that track BY NAME.
        const WAS_UNLISTED: &str = "qwen3.8-27b-mtp-v1";
        let opted_in = Contract::parse(
            br#"{"track_id":"qwen3.8-27b-mtp-v1","scores_against_live_control_leg":true,
                 "paired_flow_retired":true}"#,
        )
        .unwrap();
        assert!(
            scores_against_live_control_leg(&opted_in, WAS_UNLISTED)
                .unwrap()
                .value
        );
        let err = enforce_paired_flow_available(&opted_in, WAS_UNLISTED)
            .expect_err("a fixture may retire the paired flow for its own track");
        assert!(
            err.contains(constants::PAIRED_FLOW_RETIRED_FOR_TRACK),
            "{err}"
        );

        // …and a track NO table ever spoke for resolves from its fixture like any other. Before
        // STEP 3 this was the one case the table could not answer at all.
        const UNKNOWN: &str = "qwen4.0-9b-mlx-v1";
        let fresh = Contract::parse(
            br#"{"track_id":"qwen4.0-9b-mlx-v1","scored_batch_size":4,
                 "prefill_gain_exponent":0.25,"decode_gain_exponent":0.75,
                 "scores_against_live_control_leg":true,"paired_flow_retired":true}"#,
        )
        .unwrap();
        let regime = scored_regime(&fresh, UNKNOWN).expect("the fixture declares the regime");
        assert_eq!(regime.value.scored_batch_size, 4);
        assert_eq!(regime.value.prefill_gain_exponent, 0.25);
        assert_eq!(regime.source, ContractSource::Contract);
        assert!(
            scores_against_live_control_leg(&fresh, UNKNOWN)
                .unwrap()
                .value
        );
    }

    /// THE REGIME DECLARATION IS ALL-OR-NOTHING, and the refusal lands at the PARSE.
    ///
    /// REVERT-PROOF: drop `certify_scored_regime` from [`Contract::certify`] and every
    /// `unwrap_err` below goes red — a half-declared regime would then resolve half from the
    /// fixture and half from the table, which is the one outcome this migration must never have.
    #[test]
    fn a_half_declared_regime_is_refused_at_the_parse() {
        for body in [
            &br#"{"track_id":"t","prefill_gain_exponent":0.0}"#[..],
            &br#"{"track_id":"t","decode_gain_exponent":1.0}"#[..],
            &br#"{"track_id":"t","scored_batch_size":1,"decode_gain_exponent":1.0}"#[..],
        ] {
            let err = Contract::parse(body).unwrap_err();
            assert!(err.contains("ONE declaration"), "{err}");
            assert!(err.contains("\"t\""), "{err}");
        }
        // A regime that weights NEITHER axis scores nothing — the table's own well-formedness rule.
        let err = Contract::parse(
            br#"{"track_id":"t","scored_batch_size":1,"prefill_gain_exponent":0.0,
                 "decode_gain_exponent":0.0}"#,
        )
        .unwrap_err();
        assert!(err.contains("weights neither axis"), "{err}");
        // A negative exponent is malformed, not "no weight".
        let err = Contract::parse(
            br#"{"track_id":"t","scored_batch_size":1,"prefill_gain_exponent":-1.0,
                 "decode_gain_exponent":1.0}"#,
        )
        .unwrap_err();
        assert!(err.contains("finite and non-negative"), "{err}");
        // `scored_batch_size` ALONE stays what it has always been: a cross-check, not half a
        // regime. The live engine fixtures carry it, so reading it as a partial declaration would
        // refuse files that are correct today.
        let batch_only = Contract::parse(br#"{"track_id":"t","scored_batch_size":1}"#)
            .expect("a bare scored_batch_size is a cross-check, not a partial regime");
        assert!(declared_scored_regime(&batch_only).is_none());
        // A DECLARED BATCHED regime RESOLVES (David 2026-09-15: batching is configuration). The
        // width is carried through, not checked against a measured-width constant.
        let batched = Contract {
            scored_batch_size: Some(8),
            prefill_gain_exponent: Some(0.0),
            decode_gain_exponent: Some(1.0),
            ..Contract::NONE_DECLARED
        };
        let resolved = scored_regime(&batched, "t").expect("a batched width is a declaration");
        assert_eq!(resolved.value.scored_batch_size, 8);
        assert_eq!(resolved.source, ContractSource::Contract);
    }
}

#[cfg(test)]
mod baseline_and_weights_tests {
    use super::*;
    use crate::constants;
    use crate::score::ScoringWeights;

    /// A DECLARED baseline pair, band shape and weight pair RESOLVE, for a track nothing in this
    /// binary has ever heard of, and each group is independent of the others: a fixture may declare
    /// its band shape without its pair (the live-control-leg design) or its weights alone.
    ///
    /// The negative control for the oracle, whose expectations were copied from the deleted tables:
    /// the track below appears in no expectation, no fixture and no former table row, so a resolver
    /// with anything wired into it could not answer for it at all.
    #[test]
    fn a_declared_pair_bands_and_weights_resolve_for_an_unknown_track() {
        const UNKNOWN: &str = "qwen3.9-27b-mlx-v1";
        let declares = Contract::parse(
            br#"{"track_id":"qwen3.9-27b-mlx-v1",
                 "official_baseline_prefill_seconds_per_token":0.0004,
                 "official_baseline_decode_seconds_per_token":0.014,
                 "prefill_band_up_tolerance":0.05,"prefill_band_down_tolerance":0.05,
                 "decode_band_up_tolerance":0.02,"decode_band_down_tolerance":0.05,
                 "decode_band_down_enabled":false,"prefill_band_down_enabled":false,
                 "score_decode_weight":0.6,"score_prefill_weight":0.4}"#,
        )
        .unwrap();
        let resolved = official_baseline(&declares, UNKNOWN).expect("the fixture declares a pair");
        assert_eq!(resolved.source, ContractSource::Contract);
        assert_eq!(resolved.value.prefill_seconds_per_token, 0.0004);
        assert_eq!(resolved.value.decode_seconds_per_token, 0.014);
        assert_eq!(resolved.value.bands.decode_up_tolerance, 0.02);
        assert!(!resolved.value.bands.decode_down_enabled);
        let w = scoring_weights(&declares, UNKNOWN).expect("the fixture declares both weights");
        assert_eq!(w.source, ContractSource::Contract);
        assert_eq!(
            w.value,
            ScoringWeights {
                decode: 0.6,
                prefill: 0.4
            }
        );

        // BANDS ALONE — the live-control-leg design: no stored pair, its own band shape.
        let bands_only = Contract::parse(
            br#"{"track_id":"qwen3.8-125b-a6b-mlx-v1","scores_against_live_control_leg":true,
                 "prefill_band_up_tolerance":0.01,"prefill_band_down_tolerance":0.01,
                 "decode_band_up_tolerance":0.01,"decode_band_down_tolerance":0.01,
                 "decode_band_down_enabled":true,"prefill_band_down_enabled":true}"#,
        )
        .unwrap();
        let resolved = acceptance_bands(&bands_only, "qwen3.8-125b-a6b-mlx-v1")
            .expect("the fixture declares its own band shape");
        assert_eq!(resolved.source, ContractSource::Contract);
        assert!(resolved.value.decode_down_enabled);
        assert_eq!(resolved.value.decode_up_tolerance, 0.01);
        // …and it still stores no pair: the pair refuses BY NAME.
        let err = official_baseline(&bands_only, "qwen3.8-125b-a6b-mlx-v1").unwrap_err();
        assert!(err.contains(constants::OFFICIAL_BASELINE_PENDING), "{err}");

        // WEIGHTS ALONE — absence is now a REFUSAL, not the 0.75/0.25 constants. The composite
        // divides by their sum, so a run weighted by a pair it was never given is a number
        // attributed to a rule nobody stated.
        let err = scoring_weights(&Contract::NONE_DECLARED, UNKNOWN).unwrap_err();
        assert!(err.contains("score_decode_weight"), "{err}");
        assert!(err.contains(UNKNOWN), "{err}");
    }

    /// EACH NEW GROUP IS ALL-OR-NOTHING, refused AT THE PARSE, and the two contradictory designs
    /// cannot both be declared.
    ///
    /// REVERT-PROOF: drop any of `certify_official_baseline`, `certify_acceptance_bands` or
    /// `certify_scoring_weights` from [`Contract::certify`] and the matching `unwrap_err` goes red
    /// — a half-declared group would then resolve half from the fixture and half from the table.
    #[test]
    fn a_half_declared_pair_band_shape_or_weight_pair_is_refused_at_the_parse() {
        // HALF A PAIR.
        for body in [
            &br#"{"track_id":"t","official_baseline_prefill_seconds_per_token":0.0004}"#[..],
            &br#"{"track_id":"t","official_baseline_decode_seconds_per_token":0.014}"#[..],
        ] {
            let err = Contract::parse(body).unwrap_err();
            assert!(err.contains("half an official baseline pair"), "{err}");
            assert!(err.contains("\"t\""), "{err}");
        }
        // A non-positive or non-finite denominator scores nothing.
        let err = Contract::parse(
            br#"{"track_id":"t","official_baseline_prefill_seconds_per_token":0.0,
                 "official_baseline_decode_seconds_per_token":0.014}"#,
        )
        .unwrap_err();
        assert!(err.contains("finite and greater than 0"), "{err}");

        // A PARTIAL BAND SHAPE — including a tolerance with no enforcement flag.
        let err =
            Contract::parse(br#"{"track_id":"t","prefill_band_up_tolerance":0.05}"#).unwrap_err();
        assert!(err.contains("ONE shape"), "{err}");
        let err = Contract::parse(
            br#"{"track_id":"t","prefill_band_up_tolerance":0.05,
                 "prefill_band_down_tolerance":0.05,"decode_band_up_tolerance":0.02,
                 "decode_band_down_tolerance":0.05,"decode_band_down_enabled":false}"#,
        )
        .unwrap_err();
        assert!(err.contains("prefill_band_down_enabled"), "{err}");
        let err = Contract::parse(
            br#"{"track_id":"t","prefill_band_up_tolerance":-0.05,
                 "prefill_band_down_tolerance":0.05,"decode_band_up_tolerance":0.02,
                 "decode_band_down_tolerance":0.05,"decode_band_down_enabled":false,
                 "prefill_band_down_enabled":false}"#,
        )
        .unwrap_err();
        assert!(err.contains("finite and non-negative"), "{err}");

        // HALF A WEIGHT PAIR, and a zero-sum pair.
        let err = Contract::parse(br#"{"track_id":"t","score_decode_weight":0.75}"#).unwrap_err();
        assert!(err.contains("half a scoring-weight pair"), "{err}");
        let err = Contract::parse(
            br#"{"track_id":"t","score_decode_weight":0.0,"score_prefill_weight":0.0}"#,
        )
        .unwrap_err();
        assert!(err.contains("zero-sum pair"), "{err}");

        // THE TWO DESIGNS ARE MUTUALLY EXCLUSIVE: a stored pair and a live control leg cannot both
        // be declared for one track (David 2026-09-08).
        let err = Contract::parse(
            br#"{"track_id":"t","scores_against_live_control_leg":true,
                 "official_baseline_prefill_seconds_per_token":0.0004,
                 "official_baseline_decode_seconds_per_token":0.014,
                 "prefill_band_up_tolerance":0.05,"prefill_band_down_tolerance":0.05,
                 "decode_band_up_tolerance":0.02,"decode_band_down_tolerance":0.05,
                 "decode_band_down_enabled":false,"prefill_band_down_enabled":false}"#,
        )
        .unwrap_err();
        assert!(err.contains("different designs"), "{err}");

        // A DECLARED PAIR MUST CARRY ITS BANDS: there is no table shape for it to borrow, because
        // the shape the table would give belongs to the pair the table stores, not to this one.
        let no_bands = Contract::parse(
            br#"{"track_id":"qwen3.8-27b-mtp-v1",
                 "official_baseline_prefill_seconds_per_token":0.0004,
                 "official_baseline_decode_seconds_per_token":0.014}"#,
        )
        .expect("a pair without bands parses; the refusal is at resolution");
        let err = official_baseline(&no_bands, "qwen3.8-27b-mtp-v1").unwrap_err();
        assert!(err.contains("no acceptance band shape"), "{err}");
    }
}

#[cfg(test)]
mod window_shape_tests {
    use super::*;
    use crate::constants::{self, WindowShape};

    /// THE WINDOW IS THE FIXTURE'S, and it is no longer keyed by PLATFORM at all. The warm-up count
    /// used to be the one per-platform value in the window (1 on MLX, 0 on CUDA); a track that
    /// states its own window states that count too, so `window_shape` takes no platform — and one
    /// fixture therefore resolves one window whatever platform the track names.
    #[test]
    fn a_declared_window_is_the_windows_only_source() {
        let declared = Contract::parse(
            br#"{"track_id":"qwen3.8-125b-a6b-cuda-v1","correctness_steps":32,
                 "benchmark_decode_steps":256,"local_submit_benchmark_decode_steps":2047,
                 "official_prefill_warmup_runs":3}"#,
        )
        .unwrap();
        let resolved = window_shape(&declared, "qwen3.8-125b-a6b-cuda-v1")
            .expect("the fixture declares the whole window");
        assert_eq!(resolved.source, ContractSource::Contract);
        assert_eq!(
            resolved.value,
            WindowShape {
                correctness_steps: 32,
                benchmark_decode_steps: 256,
                local_submit_benchmark_decode_steps: 2047,
                official_prefill_warmup_runs: 3,
                official_prefill_timed_runs: 1,
            }
        );
        // The SAME fixture resolves the SAME window under a track naming the other platform: the
        // per-platform warm-up count went with the table.
        assert_eq!(
            window_shape(&declared, "qwen3.8-125b-a6b-mlx-v1")
                .unwrap()
                .value,
            resolved.value
        );

        // A SILENT fixture resolves NOTHING, by name.
        let err = window_shape(&Contract::NONE_DECLARED, "t").unwrap_err();
        assert!(err.contains(constants::WINDOW_SHAPE_UNDECLARED), "{err}");
        assert!(err.contains("\"t\""), "{err}");
    }

    /// THE TIMED PREFILL COUNT is optional: absent resolves to one pass (every fixture that predates
    /// the field), a declared count resolves as declared, and zero is refused at the parse.
    #[test]
    fn the_timed_prefill_count_defaults_to_one_and_refuses_zero() {
        let window = br#"{"track_id":"t","correctness_steps":64,"benchmark_decode_steps":128,
                  "local_submit_benchmark_decode_steps":1023,"official_prefill_warmup_runs":1"#;
        let with = |tail: &str| {
            let mut body = window.to_vec();
            body.extend_from_slice(tail.as_bytes());
            body
        };
        let absent = Contract::parse(&with("}")).unwrap();
        assert_eq!(
            window_shape(&absent, "t")
                .unwrap()
                .value
                .official_prefill_timed_runs,
            constants::BENCHMARK_PREFILL_TIMED_RUNS
        );
        let five = Contract::parse(&with(r#","official_prefill_timed_runs":5}"#)).unwrap();
        assert_eq!(
            window_shape(&five, "t")
                .unwrap()
                .value
                .official_prefill_timed_runs,
            5
        );
        let err = Contract::parse(&with(r#","official_prefill_timed_runs":0}"#)).unwrap_err();
        assert!(err.contains("zero timed runs measures nothing"), "{err}");
    }

    /// THE WINDOW IS ONE SHAPE, refused AT THE PARSE when it is half-declared, and a step count of
    /// zero is refused outright. The WARM-UP count may be zero — CUDA's is.
    ///
    /// REVERT-PROOF: drop `certify_window_shape` from [`Contract::certify`] and every `unwrap_err`
    /// below goes red.
    #[test]
    fn a_half_declared_window_is_refused_at_the_parse() {
        for body in [
            &br#"{"track_id":"t","benchmark_decode_steps":128}"#[..],
            &br#"{"track_id":"t","official_prefill_warmup_runs":1}"#[..],
            &br#"{"track_id":"t","correctness_steps":64,"benchmark_decode_steps":128,
                  "local_submit_benchmark_decode_steps":1023}"#[..],
        ] {
            let err = Contract::parse(body).unwrap_err();
            assert!(err.contains("ONE shape"), "{err}");
            assert!(err.contains("\"t\""), "{err}");
        }
        let err = Contract::parse(
            br#"{"track_id":"t","correctness_steps":0,"benchmark_decode_steps":128,
                 "local_submit_benchmark_decode_steps":1023,"official_prefill_warmup_runs":1}"#,
        )
        .unwrap_err();
        assert!(
            err.contains("zero steps measures and checks nothing"),
            "{err}"
        );
        // ZERO WARM-UP PASSES is a real declaration, not an absence: the CUDA track runs none.
        let cuda = Contract::parse(
            br#"{"track_id":"t","correctness_steps":64,"benchmark_decode_steps":128,
                 "local_submit_benchmark_decode_steps":1023,"official_prefill_warmup_runs":0}"#,
        )
        .expect("a zero warm-up count is a declaration");
        assert_eq!(
            window_shape(&cuda, "t")
                .unwrap()
                .value
                .official_prefill_warmup_runs,
            0
        );
    }
}

#[cfg(test)]
mod model_shape_tests {
    use super::*;
    use crate::constants;

    /// A DECLARED model shape overrides ALL FOUR of the table's values, the `golden_model_type`
    /// included — the one inventory item STEP 1 left behind, moved by David's 2026-09-15 ruling.
    #[test]
    fn a_declared_shape_overrides_all_four_values() {
        const TRACK: &str = "qwen3.8-125b-a6b-mlx-v1";
        // The identity this track's table row used to give — kept as TEST DATA so the
        // assertions below compare against something other than what the fixture declares.
        let was_table = constants::TrackModelIdentity::new("qwen4_exp_text", 248_320, 48, 1_024);
        let declared = Contract::parse(
            br#"{"track_id":"qwen3.8-125b-a6b-mlx-v1","vocab_size":1024,
                 "num_hidden_layers":7,"seed_tokens":33,
                 "golden_model_type":"declared_text"}"#,
        )
        .unwrap();
        let resolved = model_identity(&declared, TRACK).unwrap();
        assert_eq!(resolved.source, ContractSource::Contract);
        assert_eq!(resolved.value.vocab_size, 1024);
        assert_eq!(resolved.value.num_hidden_layers, 7);
        assert_eq!(resolved.value.seed_tokens, 33);
        assert_eq!(resolved.value.golden_model_type, "declared_text");
        // …and every one really did differ from the table, so this is not a tautology.
        assert_ne!(was_table.vocab_size, 1024);
        assert_ne!(was_table.num_hidden_layers, 7);
        assert_ne!(was_table.seed_tokens, 33);
        assert_ne!(was_table.golden_model_type, "declared_text");

        // SILENT refuses BY NAME — for THIS track, which the table used to answer for, and for a
        // track no table ever knew. Deleting the table made the two cases one case.
        for track in [TRACK, "qwen3.9-27b-mlx-v1"] {
            let err = model_identity(&Contract::NONE_DECLARED, track).unwrap_err();
            assert!(err.contains(constants::MODEL_IDENTITY_UNDECLARED), "{err}");
            assert!(err.contains(track), "{err}");
        }
        // …and a fixture that DECLARES the shape answers for either of them.
        assert_eq!(
            model_identity(&declared, "qwen3.9-27b-mlx-v1")
                .unwrap()
                .source,
            ContractSource::Contract
        );
    }

    /// AN EMPTY `golden_model_type` is not a declaration: it names no architecture and would match
    /// no golden, so it refuses at the parse beside the zero-valued numbers.
    #[test]
    fn an_empty_model_type_is_refused_at_the_parse() {
        let err = Contract::parse(
            br#"{"track_id":"t","vocab_size":1024,"num_hidden_layers":7,
                 "seed_tokens":33,"golden_model_type":"  "}"#,
        )
        .unwrap_err();
        assert!(err.contains("empty golden_model_type"), "{err}");
        assert!(err.contains("matches no golden"), "{err}");
    }

    /// THE SHAPE IS ONE DECLARATION, refused AT THE PARSE when half-declared, and each value must
    /// be at least 1.
    ///
    /// REVERT-PROOF: drop `certify_model_shape` from [`Contract::certify`] and every `unwrap_err`
    /// below goes red — a shape half read from the fixture and half from the table describes no
    /// checkpoint at all.
    #[test]
    fn a_half_declared_model_shape_is_refused_at_the_parse() {
        for body in [
            &br#"{"track_id":"t","vocab_size":1024}"#[..],
            &br#"{"track_id":"t","num_hidden_layers":48}"#[..],
            &br#"{"track_id":"t","vocab_size":1024,"seed_tokens":1024}"#[..],
            &br#"{"track_id":"t","vocab_size":1024,"num_hidden_layers":48,"seed_tokens":1024}"#[..],
            &br#"{"track_id":"t","golden_model_type":"qwen4_exp_text"}"#[..],
        ] {
            let err = Contract::parse(body).unwrap_err();
            assert!(err.contains("ONE declaration"), "{err}");
            assert!(err.contains("\"t\""), "{err}");
        }
        for body in [
            &br#"{"track_id":"t","vocab_size":0,"num_hidden_layers":48,"seed_tokens":1024}"#[..],
            &br#"{"track_id":"t","vocab_size":1024,"num_hidden_layers":0,"seed_tokens":1024}"#[..],
            &br#"{"track_id":"t","vocab_size":1024,"num_hidden_layers":48,"seed_tokens":0}"#[..],
        ] {
            let err = Contract::parse(body).unwrap_err();
            assert!(err.contains("must each be at least 1"), "{err}");
        }
    }
}

#[cfg(test)]
mod batch_size_tests {
    use super::*;

    /// The checked-in TEST-ONLY fixture that declares the BATCHED width.
    fn batched_fixture() -> Contract {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("crates/bench-core is two levels below the repository root")
            .join("crates/benchd/tests/fixtures/contract-batched/b8.json");
        let bytes = std::fs::read(&path).expect("the batched test fixture is checked in");
        Contract::parse(&bytes).expect("the batched test fixture parses and certifies")
    }

    /// BATCHING IS CONFIGURATION (David 2026-09-15). A fixture declaring `scored_batch_size: 8`
    /// PARSES, and the regime it resolves carries the declared width — benchd no longer holds a
    /// measured-width constant a fixture has to agree with.
    #[test]
    fn a_declared_batch_of_eight_parses_and_resolves_to_the_batched_width() {
        let contract = batched_fixture();
        assert_eq!(contract.scored_batch_size, Some(8));
        let resolved = scored_regime(&contract, "batched-b8-test-v1").expect("8 is a usable width");
        assert_eq!(resolved.value.scored_batch_size, 8);
        // The SEAL says the fixture decided the regime, not a table.
        assert_eq!(resolved.source, ContractSource::Contract);
        assert_eq!(
            sources(&contract, "batched-b8-test-v1").scored_regime,
            ContractSource::Contract
        );
    }

    /// The SINGLE-STREAM width is the same declaration, with a different value: 1 resolves to 1.
    /// The two widths differ only in the number the fixture states, which is what makes the width
    /// the SELECTOR of the measured path rather than a flag beside it.
    #[test]
    fn a_declared_batch_of_one_resolves_to_the_single_stream_width() {
        let single = Contract {
            scored_batch_size: Some(1),
            prefill_gain_exponent: Some(0.25),
            decode_gain_exponent: Some(0.75),
            ..Contract::NONE_DECLARED
        };
        let resolved = scored_regime(&single, "t").expect("1 is a usable width");
        assert_eq!(resolved.value.scored_batch_size, 1);
        assert_eq!(resolved.source, ContractSource::Contract);
    }

    /// ZERO IS NOT A WIDTH, and the refusal lands at the PARSE — on the file — naming the track and
    /// the value the file states.
    #[test]
    fn a_batch_size_of_zero_is_refused_at_the_parse() {
        let err = Contract::parse(br#"{"track_id":"t","scored_batch_size":0}"#).unwrap_err();
        assert!(err.contains("scored_batch_size 0"), "{err}");
        assert!(err.contains("at least 1"), "{err}");
        assert!(
            err.contains("\"t\""),
            "the refusal must name the track: {err}"
        );
        assert!(
            !err.contains("  "),
            "the message must carry no padding-space runs: {err}"
        );
    }

    /// A NEGATIVE width is refused too, one layer earlier: the field is a count of streams, so the
    /// type admits no negative value and the parse itself fails fail-closed rather than truncating
    /// or wrapping it.
    #[test]
    fn a_negative_batch_size_is_refused_at_the_parse() {
        let err = Contract::parse(br#"{"track_id":"t","scored_batch_size":-1}"#).unwrap_err();
        assert!(err.contains("--contract parse failed"), "{err}");
    }
}
