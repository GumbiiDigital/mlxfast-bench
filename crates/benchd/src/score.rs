//! Sealed `score.json` payload — a faithful port of the Swift
//! `ScorePayload` / `ScoreMetrics` (Sources/MLXFastCore/Score.swift).
//!
//! Field names, JSON keys, nesting, and null semantics match the Swift Codable
//! types so a benchd-written score parses/diffs against `benchmark.sh --local-iterate`
//! (the M1 / WS1-10 gate). Diagnostic real-valued fields are coarsened to
//! `PUBLIC_DIAGNOSTIC_SIGNIFICANT_FIGURES` (2) sig figs before writing, exactly like
//! Swift `withCoarsenedPublicDiagnostics`; ranking/floor fields stay precise.
//!
//! benchd is the SOLE writer of the score (no discard/reseal), and writes a
//! `.sha256` sidecar of the exact score bytes.

use bench_core::constants::PUBLIC_DIAGNOSTIC_SIGNIFICANT_FIGURES;
use serde::{Deserialize, Serialize};

/// Port of Swift `ScorePayload`: `{ score: Double?, passed: Bool, metrics: {...} }`.
///
/// `Deserialize` is derived (in addition to the sealed-write `Serialize`) so the A-3 overlay
/// (`overlay-timing`) can READ a sealed `gates-score.json` back into a typed `ScorePayload`,
/// validate it, and overlay the measured timing onto its metrics. Deserialization is additive —
/// it does not change the sealed bytes or the Swift-parity schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScorePayload {
    /// Finite score, or `null` on any failure (Swift encodes nil explicitly).
    pub score: Option<f64>,
    pub passed: bool,
    pub metrics: ScoreMetrics,
}

/// Outcome of a local phase; a later failure does not erase an earlier result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PhaseStatus {
    #[default]
    NotRun,
    Passed,
    Failed,
}

/// Local diagnostics separate the untimed correctness gate from checked timing.
/// These fields never participate in scoring or replace the legacy Swift audit fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalPhases {
    pub correctness: PhaseStatus,
    /// Actual conformance steps, not the legacy timed `metrics.checked_steps` count.
    /// Null when the gate started but returned no report (e.g. a transport error).
    pub correctness_checked_steps: Option<i64>,
    /// Passed only after both timed phases complete. NotRun means the first timing
    /// gate was never cleared; Failed includes an interrupted/partial measurement.
    pub timing: PhaseStatus,
}

impl Default for LocalPhases {
    fn default() -> Self {
        Self {
            correctness: PhaseStatus::NotRun,
            correctness_checked_steps: Some(0),
            timing: PhaseStatus::NotRun,
        }
    }
}

/// Port of Swift `ScoreMetrics`. JSON keys match the Swift `CodingKeys`.
///
/// The output is emitted sorted-key + pretty (see [`ScorePayload::to_sealed_json`]),
/// so struct declaration order does not affect the bytes; it is kept in Swift order
/// for auditability. The five `first_failing_*` / token fields and the top-level
/// `score` are the only nullable fields and are emitted as JSON `null` when absent
/// (no `skip_serializing_if`), matching Swift `encodeNil`.
///
/// `Default` lets the parity verdict tool enumerate the serde field names (a
/// `serde_json::to_value(ScoreMetrics::default())` object) so its bucket roster is checked
/// against the ACTUAL schema at `cargo test` time (§T1 exhaustiveness).
///
/// `Deserialize` + container `#[serde(default)]` let the A-3 overlay read a sealed
/// `gates-score.json` back into this type. `default` is fail-CLOSED for the overlay's validation:
/// a gates score that omits `passed_correctness` / `partial_result` deserializes them as `false`,
/// which the overlay's gate check REJECTS (it never fabricates a passing gate from an absent field).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ScoreMetrics {
    /// Additive local-only diagnostics. Absent on official paths and old artifacts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_phases: Option<LocalPhases>,
    /// The engine-reported peak RAM, rounded to the nearest 0.5 GB when it is set
    /// ([`crate::iterate::apply_timing_metrics`]); the raw value goes to benchd's stderr.
    pub peak_ram_gb: f64,
    pub bandwidth_gb_per_token: f64,
    pub decode_seconds_per_token: f64,
    pub prefill_seconds_per_token: f64,
    pub baseline_decode_seconds_per_token: f64,
    pub baseline_prefill_seconds_per_token: f64,
    pub decode_speedup: f64,
    pub prefill_speedup: f64,
    pub decode_speedup_floor: f64,
    pub prefill_speedup_floor: f64,
    pub passed_decode_speedup_floor: bool,
    pub passed_prefill_speedup_floor: bool,
    pub benchmark_wall_seconds: f64,
    pub preflight_seconds: f64,
    pub correctness_seconds: f64,
    pub timed_benchmark_seconds: f64,
    pub gpqa_ttft_passed: bool,
    pub gpqa_ttft_pass_count: i64,
    pub gpqa_ttft_case_count: i64,
    pub gpqa_ttft_seconds: f64,
    pub gpqa_ttft_p50_seconds: f64,
    pub gpqa_ttft_max_seconds: f64,
    pub gpqa_ttft_source: String,
    pub semantic_gpqa_passed: bool,
    pub semantic_gpqa_pass_count: i64,
    pub semantic_gpqa_case_count: i64,
    pub semantic_gpqa_model: String,
    pub process_resident_memory_gb: f64,
    pub passed_correctness: bool,
    pub num_layers: i64,
    pub checked_steps: i64,
    pub case_count: i64,
    pub expert_cache_hits: u64,
    pub expert_cache_misses: u64,
    pub expert_cache_evictions: u64,
    pub expert_bytes_read: u64,
    pub expert_read_seconds: f64,
    pub expert_peak_cached_tensors: u64,
    pub expert_hit_rate: f64,
    pub first_failing_layer: Option<i64>,
    pub first_failing_case: Option<String>,
    pub first_failing_step: Option<i64>,
    pub expected_token: Option<i64>,
    pub actual_token: Option<i64>,
    pub max_abs_diff: f64,
    pub golden_hash: String,
    /// ADDITIVE, benchd-only — THE MIGRATION PIN (David 2026-09-15): the sha256 of the EXACT
    /// `--contract` track-fixture bytes this run loaded.
    ///
    /// The fixture is a SCORING INPUT, not documentation: it decides whether the track is armed at
    /// all (`official_scoring_enabled`), the two speedup floors the run is gated against, and the
    /// pair count the paired path measures. `golden_hash` above records which golden bytes judged
    /// correctness; this records which fixture bytes set the scoring rules, so a reader of a sealed
    /// artifact can tell the two fixtures apart instead of inferring them from the verdicts.
    ///
    /// ABSENT (never `null`) on a run that loaded no fixture — the local modes without
    /// `--contract` — so those runs' sealed bytes are unchanged. Like `per_prompt` this is
    /// UNROSTERED: the Swift reference emits no such key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contract_sha256: Option<String>,
    /// ADDITIVE, benchd-only — WHICH SOURCE DECIDED EACH GROUP of this run's scored regime (STEP 1
    /// of the constants→contract migration, David 2026-09-15).
    ///
    /// `contract_sha256` above names the fixture BYTES; this names, per group, whether those bytes
    /// declared the group (`"contract"`) or whether THIS DEVICE measured the value for itself
    /// (`"device"`). A live-control-leg track leaves the scored regime and the official baseline
    /// pair undeclared and measures its own reference benchmark on the box, in the same job; benchd
    /// does not upload that reference and does not save it to a centralized repository. One binary
    /// scores tracks of both kinds, so an artifact that recorded only the digest would leave a
    /// reader unable to tell which groups the fixture actually set.
    ///
    /// `"table"` is read-only history. STEP 3 deleted the in-tree tables, so no run this binary
    /// seals emits that word; an artifact a STEP-1 binary sealed still reads as what it was.
    ///
    /// ABSENT (never `null`) on a run that resolves no regime at all, so those runs' sealed bytes
    /// are unchanged. Like `contract_sha256` this is UNROSTERED: the Swift reference emits no such
    /// key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contract_sources: Option<bench_core::contract::ContractSources>,
    pub bandwidth_source: String,
    pub error: String,
    pub commit: String,
    pub timestamp: String,
    pub harness_hash: String,
    pub weights_hash: String,
    pub weights_byte_count: i64,
    pub weights_file_count: i64,
    pub runtime: String,
    pub partial_result: bool,
    /// ADDITIVE, benchd-only — the per-timed-prompt records the challenge BOARD reads for its MTP
    /// column and its per-prompt decode readout (see [`ScorePerPrompt`]). One entry per timed prompt
    /// the run actually measured; EMPTY on every path that measured none.
    ///
    /// OMITTED FROM THE JSON WHEN EMPTY (`skip_serializing_if`), so no existing score.json key moves
    /// and no path that does not populate it changes by a byte. That is also why it carries no
    /// `parity.rs` ROSTER bucket: the roster is the SWIFT-PARITY surface, the reference emits no
    /// such key, and a rostered key that is absent on both sides would hard-fail the differ as
    /// SCHEMA-DRIFT-MISSING. `per_prompt_is_the_only_unrostered_metric` pins that this stays the
    /// ONE additive exception.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub per_prompt: Vec<ScorePerPrompt>,
    /// ADDITIVE, benchd-only — THE SPECULATIVE-DECODE SEAL. The mode the engine ECHOED as
    /// `effective_spec` on the timed `free_decode_begin` (`"serial"` / `"mtp"`), already validated
    /// EQUAL to what benchd requested (spec-never-ignored, `bench_protocol::spec_echo_honors_request`).
    /// It answers "what did the scored leg actually run?" from the engine's own words, not from the
    /// operator's intent.
    ///
    /// ABSENT (never `null`) on a leg that carried no spec — today's serial default — so a serial
    /// run's sealed bytes are unchanged. Like `per_prompt` this is UNROSTERED: the Swift reference
    /// emits no such key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_spec_mode: Option<String>,
    /// ADDITIVE — the DEPTH inside that echoed spec: the engine-resolved `mtp.depth`, or `0` for
    /// `serial` (serial has no drafter, so zero is its true depth, not a placeholder). Absent
    /// exactly when [`effective_spec_mode`](Self::effective_spec_mode) is absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_spec_depth: Option<i64>,
    /// ADDITIVE — R, the number of verify rounds the timed free-run window ran
    /// (`acceptance_lengths.len()`). Externally anchored: the phase-close `completed_work` counter
    /// must equal R+1 or the leg is refused. AUDIT-only, never scored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_rounds: Option<u64>,
    /// ADDITIVE — total draft tokens the engine PROPOSED across the window (self-reported).
    /// AUDIT-only, never scored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_drafted_total: Option<u64>,
    /// ADDITIVE — total draft tokens the target ACCEPTED across the window (self-reported).
    /// AUDIT-only, never scored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_accepted_total: Option<u64>,
    /// ADDITIVE — `spec_accepted_total / spec_drafted_total`. ABSENT when nothing was drafted (a
    /// serial leg, or an mtp leg whose drafter proposed nothing): a zero denominator has no rate,
    /// and `0.0` would read as "drafted plenty, accepted none". AUDIT-only, never scored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_acceptance_rate: Option<f64>,
    /// ADDITIVE — the engine's `verify_replay_disagreements`: how many verify rounds of the timed
    /// window had the BATCHED verify forward and the ONE-ROW replay of the same position choose a
    /// DIFFERENT argmax. The replay is the token that was committed; the divergence is a property
    /// of a tower that is not batch-invariant, so it is COUNTED, never a refusal.
    ///
    /// ABSENT ⇒ NOT REPORTED, never `0`. An engine that does not put the counter on the wire seals
    /// no key here, and "the engine measured none" stays distinguishable from "the engine cannot
    /// say". Bounded at audit-construction time by the rejected drafts
    /// (`spec_drafted_total - spec_accepted_total`), so an impossible count refuses the leg instead
    /// of reaching these bytes. AUDIT-only, never scored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_verify_replay_disagreements: Option<u64>,
    /// ADDITIVE — the engine's self-reported VERIFY PATH for the timed window: `"rectangular"`
    /// (one target forward over the 1+k candidate window, recurrent state captured per position
    /// and rolled back to the accepted one) or `"serial"` (one forward per candidate token, the
    /// fallback oracle). Present only when the engine reported it. AUDIT-only, never scored — it
    /// exists so a sealed decode number states which path produced it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_verification_mode: Option<String>,
    /// ADDITIVE — verify rounds that ran the rectangular path (self-reported). AUDIT-only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_rectangular_verification_rounds: Option<u64>,
    /// ADDITIVE — verify rounds that fell back to the serial oracle (self-reported). AUDIT-only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_serial_verification_rounds: Option<u64>,
    /// ADDITIVE — the per-round `acceptance_lengths[]` histogram of the timed window, SUMMARISED
    /// as `{count, mean}`. The raw vector is engine-chosen (one value per round, after the engine
    /// has seen the hidden prompt) and this record goes back to the participant, so sealing it
    /// verbatim would be a channel of several hundred bytes; the full vector goes to benchd's
    /// stderr instead. Absent on a leg with no free-run audit. AUDIT-only, never scored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acceptance_lengths: Option<AcceptanceLengthsSummary>,
    /// ADDITIVE — THE ENGINE IDENTITY SEAL, taken from the TIMED worker's `hello` (the worker whose
    /// leg is scored). `hello.backend` as `sha256=<hex> bytes=<bucket>` of the text, never the text: a
    /// later pair's engine has seen the hidden prompts, so hello free text would be a read-back
    /// channel. The same holds for `engine_device`, `runner_id`, `runner_model_type`,
    /// `runner_build` and `spec_verification_mode`; the plain text goes to benchd's stderr. Absent when no timed worker ran or
    /// the engine sent none. AUDIT-only, never scored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine_backend: Option<String>,
    /// ADDITIVE — the timed worker's `hello.device` VERBATIM (e.g. `"cuda sm_121"`). AUDIT-only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine_device: Option<String>,
    /// ADDITIVE — the timed worker's `hello.protocol_version`. The session handshake already
    /// refuses a version benchd does not speak; this records WHICH one answered. AUDIT-only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine_protocol_version: Option<u32>,
    /// ADDITIVE — the VERDICT on the timed worker's loaded-head digest
    /// (`hello.head_provenance.sha256`, #106), never the digest itself (256 engine-chosen bits):
    /// `match` / `mismatch` against the reference leg's head digest, `valid` when the run has no
    /// reference leg, `invalid` when it is not 64 lowercase hex. The raw digest goes to benchd's
    /// stderr. The board's custom-head reader looks for it here and on each `per_prompt` entry.
    /// Absent for an engine that echoes no head provenance. AUDIT-only, never scored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_provenance_sha256: Option<String>,
    /// ADDITIVE — the timed worker's runner id (`hello.runner.id`, e.g. `"layr/qwen4exp-125b-a6b"`).
    /// Absent for a worker that echoes no runner identity. IDENTITY, never an input to the score.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runner_id: Option<String>,
    /// ADDITIVE — the `config.json` model type the timed worker loaded (`hello.runner.model_type`).
    /// IDENTITY, never an input to the score.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runner_model_type: Option<String>,
    /// ADDITIVE — the digest of the timed worker's CANONICAL runner manifest
    /// (`hello.runner.manifest_sha256`), sealed only as `valid` (64 lowercase hex) or `invalid`;
    /// the raw value goes to benchd's stderr. IDENTITY, never an input to the score;
    /// the conformance kit, not the scorer, is what compares it against a declared manifest.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runner_manifest_sha256: Option<String>,
    /// ADDITIVE — the worker build the runner identity was cut from (`hello.runner.build`).
    /// IDENTITY, never an input to the score.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runner_build: Option<String>,
    /// ADDITIVE — the process id of the RESIDENT process the timed worker ATTACHED to
    /// (`hello.resident.pid`). NO LONGER SEALED: it is an engine-chosen number, so benchd logs it
    /// to stderr and leaves this key absent. Kept in the schema so older records still parse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resident_pid: Option<u32>,
    /// ADDITIVE — the resident process's load stamp (`hello.resident.load_epoch`), taken when its
    /// load ended and constant for the life of that process. Every phase of one window seals the
    /// SAME value; the official path REFUSES a window whose phases report different resident
    /// identities, because that is a reload inside the window (weights-load-once). NO LONGER
    /// SEALED, like `resident_pid`: logged to stderr, key absent, the check runs on the in-memory
    /// hello. Kept in the schema so older records still parse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resident_load_epoch: Option<u64>,
    /// ADDITIVE — THE PAIRED-BASELINE SEAL (David 2026-09-08). WHERE the denominator came from:
    /// `"serial-control-leg"` says it was MEASURED, on this box, in this job, on the reference
    /// tree, immediately before the candidate leg. Absent on every path that did not measure a
    /// control leg, so an absent key is not a claim about one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_source: Option<String>,
    /// ADDITIVE — the ranked BOX the paired run measured both legs on (the runner name the
    /// calibration file names). IDENTITY, never an input to the score.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_box: Option<String>,
    /// ADDITIVE — the digest of the per-box calibration FILE this run checked its control leg
    /// against. The file is a health band, never a denominator; the digest states which band.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_calibration_sha256: Option<String>,
    /// ADDITIVE — the digest of the GOLDEN the serial-control leg verified its decode tokens
    /// against. The control leg is serial, so on a track that carries per-depth oracle tapes it
    /// reads a different golden than the candidate leg (`--control-golden`); this states which one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_golden_sha256: Option<String>,
    /// ADDITIVE — the NAME of the prompt this run scored, on a track that draws its scored prompt
    /// per job from the contract's `live_golden_rotation.pool` (`--live-prompt`). Public (one of
    /// the pool) so results are auditable; absent on a track that does not rotate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_prompt: Option<String>,
    /// ADDITIVE — the reference tree's engine commit the calibration was captured at.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_reference_commit: Option<String>,
    /// ADDITIVE — whether the measured control leg sat inside this box's band. It is `true`
    /// wherever it is present: a leg outside the band seals no score at all, so `false` never
    /// reaches a sealed artifact. It is sealed so a reader can see the gate ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_band_passed: Option<bool>,
    /// ADDITIVE — the SERIAL-CONTROL leg's measured prefill seconds-per-token. The same value
    /// [`ScoreMetrics::baseline_prefill_seconds_per_token`] carries, named for what it is.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_leg_prefill_seconds_per_token: Option<f64>,
    /// ADDITIVE — the SERIAL-CONTROL leg's measured decode seconds-per-token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_leg_decode_seconds_per_token: Option<f64>,
    /// ADDITIVE — the CANDIDATE leg's measured prefill seconds-per-token, READ BACK from
    /// [`ScoreMetrics::prefill_seconds_per_token`] so the two cannot drift. Absent when the
    /// candidate leg produced no timing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_leg_prefill_seconds_per_token: Option<f64>,
    /// ADDITIVE — the CANDIDATE leg's measured decode seconds-per-token, READ BACK from
    /// [`ScoreMetrics::decode_seconds_per_token`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_leg_decode_seconds_per_token: Option<f64>,
    /// ADDITIVE, REPORT-ONLY — the per-role MEANS over `paired_legs` of the free-run window
    /// SPLIT's decode half per token (`PairedLegRecord::*_decode_window_seconds_per_token`):
    /// the decode-only rate, without the seed forward the enforced `decode_seconds_per_token`
    /// charges to its window. THE BOARD READS THESE for its decode tok/s readout when present.
    /// Absent when no pair carried a split (teacher-forced windows, or no candidate timing).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_leg_decode_window_seconds_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_leg_decode_window_seconds_per_token: Option<f64>,
    /// ADDITIVE, REPORT-ONLY — the same means for the seed-prefill half per seed token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_leg_seed_prefill_window_seconds_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_leg_seed_prefill_window_seconds_per_token: Option<f64>,
    /// ADDITIVE — the timed token tolerance the paired run applied, in tokens per thousand
    /// (`timed_token_tolerance_per_thousand` in the track fixture). Absent when the fixture
    /// declares none: then one different token fails the run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timed_token_tolerance_per_thousand: Option<u32>,
    /// ADDITIVE — the near-tie limit the paired run applied (`timed_token_near_tie_relative_gap`
    /// in the track fixture). Absent when the fixture declares none: then every different token
    /// counts against the tolerance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timed_token_near_tie_relative_gap: Option<f64>,
    /// ADDITIVE — the SCORED pair's [`PairedLegRecord::token_mismatch_count`]. Absent on a path
    /// with no paired legs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_mismatch_count: Option<i64>,
    /// ADDITIVE — the SCORED pair's [`PairedLegRecord::token_mismatch_first_step`]. Absent when
    /// that pair has no mismatch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_mismatch_first_step: Option<i64>,
    /// ADDITIVE — the SCORED pair's [`PairedLegRecord::token_mismatch_near_tie_count`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_mismatch_near_tie_count: Option<i64>,
    /// ADDITIVE, REPORT-ONLY — the SCORED pair's
    /// [`PairedLegRecord::token_mismatch_second_choice_count`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_mismatch_second_choice_count: Option<i64>,
    /// ADDITIVE, REPORT-ONLY — the SCORED pair's
    /// [`PairedLegRecord::token_mismatch_second_choice_max_relative_gap`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_mismatch_second_choice_max_relative_gap: Option<f64>,
    /// PAIRED PATH audit trail (David 2026-09-09, `official_pairs` in the track fixture): every
    /// pair this run measured, in order, both legs' per-token times as measured. The enforced
    /// `baseline_leg_*` / `candidate_leg_*` fields above are ONE of these rows — the pair whose
    /// composite is the lower median over the pairs (David 2026-09-17: pairs are never averaged).
    /// OMITTED when empty so the single-leg and local payloads keep their key set.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub paired_legs: Vec<PairedLegRecord>,
    /// GATE LOG (David 2026-09-17): every GATE POINT of the run, in run order, each naming where
    /// it applied and what BOTH gates read. A paired run of two pairs seals eight points — two
    /// pairs x two legs (`control`, `candidate`) x two phases (`prefill`, `decode`) — so a reader
    /// can see that EVERY timed window opened on an idle, cool box, not just one of them. A gate
    /// that REFUSED never reaches a seal: the run stops instead, so only `passed` and `skipped`
    /// appear here. Calibration files carry the same records per PASS
    /// (`crate::baseline::BaselineCalibration::gates`).
    ///
    /// ENVIRONMENTAL: the waits depend on how hot and how busy the box was, so two honest runs
    /// never agree on them. `parity-diff` waives the key for that reason.
    /// OMITTED when empty, so the single-leg and local payloads keep their key set.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gates: Vec<crate::quiescegate::GateRecord>,
}

/// One measured pair of the paired official run: the serial-control leg and the candidate leg on
/// the same box and prompt, as measured (never coarsened — this is the audit trail).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedLegRecord {
    pub pair: i64,
    /// The sha256 of the candidate golden this pair measured. Both legs of a pair measure the same
    /// prompt. `default` lets this benchd read a score sealed before the key existed.
    #[serde(default)]
    pub prompt_sha256: String,
    pub control_prefill_seconds_per_token: f64,
    pub control_decode_seconds_per_token: f64,
    pub candidate_prefill_seconds_per_token: f64,
    pub candidate_decode_seconds_per_token: f64,
    /// ADDITIVE, REPORT-ONLY — the same legs' free-run window SPLIT
    /// (`docs/scored-regime-and-prefill-window.md`), per token: the seed-prefill half over the
    /// seed length and the decode half over N. `*_decode_seconds_per_token` above divides the
    /// WHOLE window (seed forward included) and is what scores; these name the decode-only rate
    /// a reader would otherwise have to back out. Absent on a teacher-forced (v1) window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_seed_prefill_window_seconds_per_token: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_decode_window_seconds_per_token: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_seed_prefill_window_seconds_per_token: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_decode_window_seconds_per_token: Option<f64>,
    /// The candidate leg's timed tokens (the seed token, then the decode tokens) that differ from
    /// the reference engine's choice for the same prefix. `0` when the tokens equal the golden's
    /// oracle. `null` only when the replay that counts them did not complete.
    #[serde(default)]
    pub token_mismatch_count: Option<i64>,
    /// The position of the first such token: `0` is the seed token, `i` is decode token `i - 1`.
    /// `null` when there is no mismatch.
    #[serde(default)]
    pub token_mismatch_first_step: Option<i64>,
    /// How many of the mismatches are NEAR TIES under the fixture's
    /// `timed_token_near_tie_relative_gap`: the candidate's token is the reference engine's second
    /// choice, and the relative gap between its first and second choice is at most the limit.
    /// `null` when the fixture declares no limit, or when the replay did not complete.
    #[serde(default)]
    pub token_mismatch_near_tie_count: Option<i64>,
    /// REPORT-ONLY — how many of the mismatches are positions where the candidate's token is the
    /// reference engine's second choice. No decision reads it.
    #[serde(default)]
    pub token_mismatch_second_choice_count: Option<i64>,
    /// REPORT-ONLY — the largest relative gap between the reference engine's first and second
    /// choice over those positions (`bench_core::near_tie::relative_gap`). `null` when there is
    /// no such position. No decision reads it.
    #[serde(default)]
    pub token_mismatch_second_choice_max_relative_gap: Option<f64>,
}

/// The SEALED form of a candidate-chosen count: `0` stays `0`, anything else rounds UP to the next
/// power of two (1, 2, 4, 8, …), so a sealed count says "at most this many" and no more. The same
/// power-of-two bucketing as `bench_runner::seal_byte_len`, kept numeric so the key keeps its type.
pub fn sealed_count_bucket(n: i64) -> i64 {
    if n <= 0 {
        n
    } else {
        (n as u64).next_power_of_two() as i64
    }
}

/// The label of the timed-oracle failure class, which names no prompt and is sealed as is.
const BENCHMARK_CASE: &str = "benchmark";

/// The SEALED name of a correctness case: `case-sha256:<first 16 hex>` of its name, except the
/// fixed timed-oracle label. A live golden's case is named after its prompt, and pool names hint at
/// the hidden prompts; the organizer maps the digest back from the golden.
pub fn sealed_case(name: &str) -> String {
    if name == BENCHMARK_CASE {
        return name.to_string();
    }
    format!("case-sha256:{}", &sha256_hex(name.as_bytes())[..16])
}

/// The SEALED failure text. A correctness case's failure text names the case and carries what
/// the engine produced (its token, the step, a rank or a logit gap), so it is sealed as its class
/// and the case digest only; benchd logged the full text when it failed. Every other error is
/// sealed as it stands (it was scrubbed at `iterate::failed_score`).
pub fn sealed_error(error: &str, case: Option<&str>) -> String {
    match case {
        Some(name) if name != BENCHMARK_CASE => {
            let class = ["anchor", "free-run", "teacher-forced"]
                .into_iter()
                .find(|c| error.starts_with(&format!("{c} ")))
                .unwrap_or("correctness");
            format!(
                "{class} case {} did not match the golden",
                sealed_case(name)
            )
        }
        _ => error.to_string(),
    }
}

/// The SEALED form of a candidate-influenced gap: 2 decimals.
pub fn sealed_gap(gap: f64) -> f64 {
    (gap * 100.0).round() / 100.0
}

impl PairedLegRecord {
    /// This record as SEALED: the token counts bucketed ([`sealed_count_bucket`]), the first
    /// mismatch step dropped, the gap at 2 decimals. The leg times and `prompt_sha256` (the
    /// organizer's golden digest, which the engine cannot choose) are carried as measured.
    pub fn seal_token_counts_coarse(&self) -> PairedLegRecord {
        PairedLegRecord {
            token_mismatch_count: self.token_mismatch_count.map(sealed_count_bucket),
            token_mismatch_first_step: None,
            token_mismatch_near_tie_count: self
                .token_mismatch_near_tie_count
                .map(sealed_count_bucket),
            token_mismatch_second_choice_count: self
                .token_mismatch_second_choice_count
                .map(sealed_count_bucket),
            token_mismatch_second_choice_max_relative_gap: self
                .token_mismatch_second_choice_max_relative_gap
                .map(sealed_gap),
            ..self.clone()
        }
    }
}

/// One timed prompt's board-facing record, sealed in [`ScoreMetrics::per_prompt`].
///
/// THE READER IS THE BOARD, and it reads every field INDEPENDENTLY and OPTIONALLY
/// (yukon `apps/challenges-ui/shared/lib/throughput-metrics.ts`, PR #622; `mlxfast/lib/format.ts`
/// on master): the MTP column shows the MEAN of `effective_mean_draft_len` over the entries and a
/// dash when the array is absent or empty; the decode readout uses `mtp_seconds_per_token_mean`,
/// skipping any entry whose value is missing or <= 0.
///
/// NOT [`crate::measure_job::PerPrompt`]. That type is the PAIRED flow's results.json record: its
/// `parity_ok`, `accepted_pair_count`, `serial_seconds_per_token_mean` and `raw_ratio_of_means` are
/// all mandatory and all describe a SERIAL-vs-CANDIDATE pair, which a single-leg run does not have.
/// Reusing it here would mean inventing values for the pair half. This type carries only the fields
/// the board reads, under the identical JSON key names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ScorePerPrompt {
    /// The timed prompt's identity, BOUND BY BYTES: the sha256 of the golden this run timed
    /// ([`bench_core::golden::GoldenFixture::sha256`]) — the same `golden_hash` this score already
    /// seals, and the same rule the paired flow applies for its own records
    /// (`measure_job.rs`, "the prompt IDENTITY is the sha256 of THIS golden's bytes").
    pub prompt_sha256: String,
    /// The mean number of committed tokens per verify round, exactly as the free-run audit computes
    /// it ([`bench_core::free_run::FreeRunAudit::effective_mean_draft_len`]). `0` is a REAL measured
    /// value, never a placeholder. AUDIT-ONLY — never a scoring input.
    pub effective_mean_draft_len: f64,
    /// This prompt's ENFORCED whole-window decode seconds-per-token — the SAME number
    /// [`ScoreMetrics::decode_seconds_per_token`] carries. On a single timed prompt the two are
    /// equal by construction. It is deliberately NOT a decode-only figure: see the RED-TEAM REVERT
    /// notes in `bench-runner/src/timing.rs`, which exist because an earlier revision redefined this
    /// quantity to exclude the seed forward.
    pub mtp_seconds_per_token_mean: f64,
    /// ADDITIVE — this prompt's verify-round count R. Absent on a leg with no free-run audit, so an
    /// entry that has none seals the historical three keys exactly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_rounds: Option<u64>,
    /// ADDITIVE — this prompt's total PROPOSED draft tokens (self-reported). AUDIT-only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_drafted_total: Option<u64>,
    /// ADDITIVE — this prompt's total ACCEPTED draft tokens (self-reported). AUDIT-only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec_accepted_total: Option<u64>,
    /// ADDITIVE — the timed worker's loaded-head digest VERDICT, MIRRORED here because the board's
    /// custom-head reader reads it off the per-prompt entry. Same value as
    /// [`ScoreMetrics::head_provenance_sha256`]. AUDIT-only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_provenance_sha256: Option<String>,
}

/// The sealed stand-in for a timed window's `acceptance_lengths[]`: how many verify rounds, and
/// their mean committed length rounded to 3 decimals.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AcceptanceLengthsSummary {
    pub count: u64,
    pub mean: f64,
}

impl AcceptanceLengthsSummary {
    /// `None` for an empty histogram (nothing to summarise, so no key is sealed).
    pub fn of(lengths: &[u32]) -> Option<Self> {
        if lengths.is_empty() {
            return None;
        }
        let sum: u64 = lengths.iter().map(|&x| u64::from(x)).sum();
        let mean = sum as f64 / lengths.len() as f64;
        Some(AcceptanceLengthsSummary {
            count: lengths.len() as u64,
            mean: (mean * 1000.0).round() / 1000.0,
        })
    }
}

/// Port of Swift `roundedToSignificantFigures`: monotone sig-fig rounding via a
/// formatted round-trip so the result is the clean nearest double to the N-sig-fig
/// decimal. Non-finite / zero / non-positive `figures` pass through unchanged.
pub fn rounded_to_significant_figures(value: f64, figures: u32) -> f64 {
    if !value.is_finite() || value == 0.0 || figures == 0 {
        return value;
    }
    // printf `%.*g` keeps `figures` significant digits. `{:.*e}` with `figures-1`
    // fractional mantissa digits is the same significant-figure grid, and parsing
    // the scientific string yields the clean nearest double (drops float noise).
    let formatted = format!("{:.*e}", (figures - 1) as usize, value);
    formatted.parse::<f64>().unwrap_or(value)
}

impl ScoreMetrics {
    /// Port of Swift `withCoarsenedPublicDiagnostics`: round the diagnostic
    /// (non-ranking) real-valued fields to `figures` sig figs; leave the ranking /
    /// floor / int / bool / string fields untouched. Re-clamps the ordering pairs
    /// (wall >= timed, ttft_max >= p50) after rounding, as Swift does.
    pub fn with_coarsened_public_diagnostics(&self, figures: u32) -> ScoreMetrics {
        let r = |v: f64| rounded_to_significant_figures(v, figures);

        let rounded_timed = r(self.timed_benchmark_seconds);
        let rounded_wall = r(self.benchmark_wall_seconds).max(rounded_timed);
        let rounded_p50 = r(self.gpqa_ttft_p50_seconds);
        let rounded_ttft_max = r(self.gpqa_ttft_max_seconds).max(rounded_p50);

        ScoreMetrics {
            local_phases: self.local_phases.clone(),
            peak_ram_gb: r(self.peak_ram_gb),
            bandwidth_gb_per_token: r(self.bandwidth_gb_per_token),
            decode_seconds_per_token: self.decode_seconds_per_token,
            prefill_seconds_per_token: self.prefill_seconds_per_token,
            baseline_decode_seconds_per_token: self.baseline_decode_seconds_per_token,
            baseline_prefill_seconds_per_token: self.baseline_prefill_seconds_per_token,
            decode_speedup: self.decode_speedup,
            prefill_speedup: self.prefill_speedup,
            decode_speedup_floor: self.decode_speedup_floor,
            prefill_speedup_floor: self.prefill_speedup_floor,
            passed_decode_speedup_floor: self.passed_decode_speedup_floor,
            passed_prefill_speedup_floor: self.passed_prefill_speedup_floor,
            benchmark_wall_seconds: rounded_wall,
            preflight_seconds: r(self.preflight_seconds),
            correctness_seconds: r(self.correctness_seconds),
            timed_benchmark_seconds: rounded_timed,
            gpqa_ttft_passed: self.gpqa_ttft_passed,
            gpqa_ttft_pass_count: self.gpqa_ttft_pass_count,
            gpqa_ttft_case_count: self.gpqa_ttft_case_count,
            gpqa_ttft_seconds: r(self.gpqa_ttft_seconds),
            gpqa_ttft_p50_seconds: rounded_p50,
            gpqa_ttft_max_seconds: rounded_ttft_max,
            gpqa_ttft_source: self.gpqa_ttft_source.clone(),
            semantic_gpqa_passed: self.semantic_gpqa_passed,
            semantic_gpqa_pass_count: self.semantic_gpqa_pass_count,
            semantic_gpqa_case_count: self.semantic_gpqa_case_count,
            semantic_gpqa_model: self.semantic_gpqa_model.clone(),
            process_resident_memory_gb: r(self.process_resident_memory_gb),
            passed_correctness: self.passed_correctness,
            num_layers: self.num_layers,
            checked_steps: self.checked_steps,
            case_count: self.case_count,
            expert_cache_hits: self.expert_cache_hits,
            expert_cache_misses: self.expert_cache_misses,
            expert_cache_evictions: self.expert_cache_evictions,
            expert_bytes_read: self.expert_bytes_read,
            expert_read_seconds: r(self.expert_read_seconds),
            expert_peak_cached_tensors: self.expert_peak_cached_tensors,
            expert_hit_rate: r(self.expert_hit_rate),
            first_failing_layer: self.first_failing_layer,
            first_failing_case: self.first_failing_case.as_deref().map(sealed_case),
            // The step a failure happened at is the engine's choice: bucketed. The engine's own
            // token is never sealed (both go to benchd's stderr at `iterate::failed_score`).
            first_failing_step: self.first_failing_step.map(sealed_count_bucket),
            // The golden's token is never published. The case and the step locate the failure.
            expected_token: None,
            actual_token: None,
            max_abs_diff: r(self.max_abs_diff),
            golden_hash: self.golden_hash.clone(),
            contract_sha256: self.contract_sha256.clone(),
            contract_sources: self.contract_sources,
            bandwidth_source: self.bandwidth_source.clone(),
            error: sealed_error(&self.error, self.first_failing_case.as_deref()),
            commit: self.commit.clone(),
            timestamp: self.timestamp.clone(),
            harness_hash: self.harness_hash.clone(),
            weights_hash: self.weights_hash.clone(),
            weights_byte_count: self.weights_byte_count,
            weights_file_count: self.weights_file_count,
            runtime: self.runtime.clone(),
            partial_result: self.partial_result,
            // Carried VERBATIM: `mtp_seconds_per_token_mean` must stay byte-equal to
            // `decode_seconds_per_token`, which is a ranking field and is not coarsened either.
            per_prompt: self.per_prompt.clone(),
            // The SPEC and ENGINE-IDENTITY seals are carried VERBATIM. They are counts, an echoed
            // mode/depth, a ratio and identity strings — facts about what ran, not diagnostic
            // real-valued measurements, so coarsening them would only lose information.
            effective_spec_mode: self.effective_spec_mode.clone(),
            effective_spec_depth: self.effective_spec_depth,
            spec_rounds: self.spec_rounds,
            spec_drafted_total: self.spec_drafted_total,
            spec_accepted_total: self.spec_accepted_total,
            spec_acceptance_rate: self.spec_acceptance_rate,
            spec_verify_replay_disagreements: self.spec_verify_replay_disagreements,
            spec_verification_mode: self.spec_verification_mode.clone(),
            spec_rectangular_verification_rounds: self.spec_rectangular_verification_rounds,
            spec_serial_verification_rounds: self.spec_serial_verification_rounds,
            acceptance_lengths: self.acceptance_lengths,
            engine_backend: self.engine_backend.clone(),
            engine_device: self.engine_device.clone(),
            engine_protocol_version: self.engine_protocol_version,
            head_provenance_sha256: self.head_provenance_sha256.clone(),
            runner_id: self.runner_id.clone(),
            runner_model_type: self.runner_model_type.clone(),
            runner_manifest_sha256: self.runner_manifest_sha256.clone(),
            runner_build: self.runner_build.clone(),
            resident_pid: self.resident_pid,
            resident_load_epoch: self.resident_load_epoch,
            // The PAIRED-BASELINE seal is carried VERBATIM. Its two leg pairs mirror the ranking
            // fields `baseline_*_seconds_per_token` / `*_seconds_per_token`, which are not
            // coarsened either, and the rest is identity.
            baseline_source: self.baseline_source.clone(),
            baseline_box: self.baseline_box.clone(),
            baseline_calibration_sha256: self.baseline_calibration_sha256.clone(),
            baseline_golden_sha256: self.baseline_golden_sha256.clone(),
            live_prompt: self.live_prompt.clone(),
            baseline_reference_commit: self.baseline_reference_commit.clone(),
            baseline_band_passed: self.baseline_band_passed,
            baseline_leg_prefill_seconds_per_token: self.baseline_leg_prefill_seconds_per_token,
            baseline_leg_decode_seconds_per_token: self.baseline_leg_decode_seconds_per_token,
            candidate_leg_prefill_seconds_per_token: self.candidate_leg_prefill_seconds_per_token,
            candidate_leg_decode_seconds_per_token: self.candidate_leg_decode_seconds_per_token,
            baseline_leg_decode_window_seconds_per_token: self
                .baseline_leg_decode_window_seconds_per_token,
            candidate_leg_decode_window_seconds_per_token: self
                .candidate_leg_decode_window_seconds_per_token,
            baseline_leg_seed_prefill_window_seconds_per_token: self
                .baseline_leg_seed_prefill_window_seconds_per_token,
            candidate_leg_seed_prefill_window_seconds_per_token: self
                .candidate_leg_seed_prefill_window_seconds_per_token,
            // The fixture's limits are carried VERBATIM. The token COUNTS are the candidate's
            // choice (it picked which tokens to commit), so they are sealed only as a bucket, the
            // first mismatch step not at all, and the gap at 2 decimals
            // ([`PairedLegRecord::seal_token_counts_coarse`]). Pass/fail is the run's own `passed` + named error.
            timed_token_tolerance_per_thousand: self.timed_token_tolerance_per_thousand,
            timed_token_near_tie_relative_gap: self.timed_token_near_tie_relative_gap,
            token_mismatch_count: self.token_mismatch_count.map(sealed_count_bucket),
            token_mismatch_first_step: None,
            token_mismatch_near_tie_count: self
                .token_mismatch_near_tie_count
                .map(sealed_count_bucket),
            token_mismatch_second_choice_count: self
                .token_mismatch_second_choice_count
                .map(sealed_count_bucket),
            token_mismatch_second_choice_max_relative_gap: self
                .token_mismatch_second_choice_max_relative_gap
                .map(sealed_gap),
            paired_legs: self
                .paired_legs
                .iter()
                .map(PairedLegRecord::seal_token_counts_coarse)
                .collect(),
            // The GATE LOG is carried VERBATIM: a wait in seconds and a temperature the gate
            // actually read are facts about the run, not diagnostics to round.
            gates: self.gates.clone(),
        }
    }
}

impl ScorePayload {
    /// Serialize to the sealed JSON bytes: coarsen diagnostics, then encode
    /// pretty + sorted-key (serde_json's default `Map` is a `BTreeMap`, so routing
    /// through `to_value` sorts keys, matching Swift's `.sortedKeys`). No trailing
    /// newline (Swift `data.write` writes the encoder output verbatim).
    pub fn to_sealed_json(&self) -> Result<String, serde_json::Error> {
        let published = ScorePayload {
            score: self.score,
            passed: self.passed,
            metrics: self
                .metrics
                .with_coarsened_public_diagnostics(PUBLIC_DIAGNOSTIC_SIGNIFICANT_FIGURES),
        };
        let value = serde_json::to_value(&published)?;
        serde_json::to_string_pretty(&value)
    }
}

/// Lowercase-hex sha256 of `bytes` (for the `.sha256` sidecar).
///
/// #58: re-exported from [`bench_core::hash`] rather than reimplemented — benchd and the
/// golden loader must agree byte-for-byte on what a digest of the same bytes is, so there is
/// exactly one implementation. Kept exposed here because `crate::score::sha256_hex` is the
/// name the sidecar/score writers already call.
pub use bench_core::hash::sha256_hex;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_sig_two_figures_matches_printf_g() {
        assert_eq!(rounded_to_significant_figures(18.0, 2), 18.0);
        assert_eq!(rounded_to_significant_figures(20.25, 2), 20.0);
        assert_eq!(rounded_to_significant_figures(0.384, 2), 0.38);
        // 0.0106 -> "1.1e-2" -> 0.011
        assert_eq!(rounded_to_significant_figures(0.0106, 2), 0.011);
    }

    #[test]
    fn round_sig_passthrough_edge_cases() {
        assert_eq!(rounded_to_significant_figures(0.0, 2), 0.0);
        assert!(rounded_to_significant_figures(f64::NAN, 2).is_nan());
        assert_eq!(rounded_to_significant_figures(5.0, 0), 5.0);
    }

    #[test]
    fn ranking_fields_are_not_coarsened() {
        let mut m = zero_metrics();
        m.decode_seconds_per_token = 0.1336139485703125;
        m.prefill_seconds_per_token = 0.010605031949609375;
        m.decode_speedup = 1.234567;
        m.peak_ram_gb = 20.25;
        let c = m.with_coarsened_public_diagnostics(2);
        // ranking fields untouched, diagnostics coarsened
        assert_eq!(c.decode_seconds_per_token, 0.1336139485703125);
        assert_eq!(c.prefill_seconds_per_token, 0.010605031949609375);
        assert_eq!(c.decode_speedup, 1.234567);
        assert_eq!(c.peak_ram_gb, 20.0);
    }

    #[test]
    fn coarsen_reclamps_wall_at_least_timed() {
        let mut m = zero_metrics();
        m.timed_benchmark_seconds = 0.049; // -> 0.049
        m.benchmark_wall_seconds = 0.051; // r -> 0.051, but must be >= r(timed)
        let c = m.with_coarsened_public_diagnostics(2);
        assert!(c.benchmark_wall_seconds >= c.timed_benchmark_seconds);
    }

    #[test]
    fn sealed_json_is_sorted_and_nested() {
        let payload = ScorePayload {
            score: Some(1.5),
            passed: true,
            metrics: zero_metrics(),
        };
        let json = payload.to_sealed_json().unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v.get("metrics").unwrap().is_object());
        assert_eq!(v.get("passed").unwrap(), &serde_json::json!(true));
        // sorted keys: top-level order is metrics, passed, score
        let top_keys: Vec<&str> = v.as_object().unwrap().keys().map(|s| s.as_str()).collect();
        assert_eq!(top_keys, vec!["metrics", "passed", "score"]);
        // null nullable fields present, not omitted
        assert!(json.contains("\"first_failing_layer\": null"));
    }

    #[test]
    fn null_score_is_emitted() {
        let payload = ScorePayload {
            score: None,
            passed: false,
            metrics: zero_metrics(),
        };
        let json = payload.to_sealed_json().unwrap();
        assert!(json.contains("\"score\": null"));
    }

    /// A zeroed metrics block used across tests. Every field of it is that field's own `Default`
    /// — `ScoreMetrics` derives `Default` — so the 88-line literal this used to spell out was a
    /// hand-maintained copy of the derive, and a new field added to the struct silently diverged
    /// from it.
    pub(crate) fn zero_metrics() -> ScoreMetrics {
        ScoreMetrics::default()
    }

    // ---------------------------------------------------------------------------------------
    // `metrics.per_prompt` — the ADDITIVE board array (see [`ScorePerPrompt`]).
    // ---------------------------------------------------------------------------------------

    /// The 56 `metrics.*` keys the sealed score carried BEFORE `per_prompt` was added. Pinned
    /// VERBATIM (not derived from the struct) so this is a real before/after snapshot: any future
    /// edit that renames, drops or reorders an existing key fails here.
    const METRICS_KEYS_BEFORE_PER_PROMPT: &[&str] = &[
        "actual_token",
        "bandwidth_gb_per_token",
        "bandwidth_source",
        "baseline_decode_seconds_per_token",
        "baseline_prefill_seconds_per_token",
        "benchmark_wall_seconds",
        "case_count",
        "checked_steps",
        "commit",
        "correctness_seconds",
        "decode_seconds_per_token",
        "decode_speedup",
        "decode_speedup_floor",
        "error",
        "expected_token",
        "expert_bytes_read",
        "expert_cache_evictions",
        "expert_cache_hits",
        "expert_cache_misses",
        "expert_hit_rate",
        "expert_peak_cached_tensors",
        "expert_read_seconds",
        "first_failing_case",
        "first_failing_layer",
        "first_failing_step",
        "golden_hash",
        "gpqa_ttft_case_count",
        "gpqa_ttft_max_seconds",
        "gpqa_ttft_p50_seconds",
        "gpqa_ttft_pass_count",
        "gpqa_ttft_passed",
        "gpqa_ttft_seconds",
        "gpqa_ttft_source",
        "harness_hash",
        "max_abs_diff",
        "num_layers",
        "partial_result",
        "passed_correctness",
        "passed_decode_speedup_floor",
        "passed_prefill_speedup_floor",
        "peak_ram_gb",
        "preflight_seconds",
        "prefill_seconds_per_token",
        "prefill_speedup",
        "prefill_speedup_floor",
        "process_resident_memory_gb",
        "semantic_gpqa_case_count",
        "semantic_gpqa_model",
        "semantic_gpqa_pass_count",
        "semantic_gpqa_passed",
        "timed_benchmark_seconds",
        "timestamp",
        "weights_byte_count",
        "weights_file_count",
        "weights_hash",
        "runtime",
    ];

    fn sealed_metrics_keys(metrics: ScoreMetrics) -> std::collections::BTreeSet<String> {
        let json = ScorePayload {
            score: Some(1.0),
            passed: true,
            metrics,
        }
        .to_sealed_json()
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        v["metrics"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
    }

    #[test]
    fn per_prompt_is_purely_additive_and_moves_no_existing_key() {
        let before: std::collections::BTreeSet<String> = METRICS_KEYS_BEFORE_PER_PROMPT
            .iter()
            .map(|k| (*k).to_string())
            .collect();

        // A score that measured no timed prompt is byte-for-byte the OLD key set: an empty
        // `per_prompt` is omitted from the JSON entirely.
        assert_eq!(
            sealed_metrics_keys(zero_metrics()),
            before,
            "an unpopulated per_prompt must not add a key"
        );

        // A score that DID measure one adds exactly `per_prompt` — nothing else moves.
        let mut populated = zero_metrics();
        populated.per_prompt = vec![ScorePerPrompt {
            prompt_sha256: "ab".repeat(32),
            effective_mean_draft_len: 4.0,
            mtp_seconds_per_token_mean: 0.125,
            ..Default::default()
        }];
        let after = sealed_metrics_keys(populated);
        let added: Vec<_> = after.difference(&before).collect();
        let removed: Vec<_> = before.difference(&after).collect();
        assert_eq!(added, vec!["per_prompt"], "one added key, and only one");
        assert!(removed.is_empty(), "existing keys removed: {removed:?}");
    }

    /// The ONE sanctioned ADDITIVE key set — `per_prompt` plus the speculative-decode seal and the
    /// engine identity (backend/device/protocol version, the loaded-head digest, the runner
    /// identity, and the resident-process identity), and the timed token tolerance with its counts.
    /// Every entry is omitted-when-unset, so a run that does not produce it seals
    /// byte-identically to before it existed.
    const ADDITIVE_METRICS_KEYS: &[&str] = &[
        "acceptance_lengths",
        "effective_spec_depth",
        "effective_spec_mode",
        "engine_backend",
        "engine_device",
        "engine_protocol_version",
        "head_provenance_sha256",
        "local_phases",
        "per_prompt",
        "resident_load_epoch",
        "resident_pid",
        "runner_build",
        "runner_id",
        "runner_manifest_sha256",
        "runner_model_type",
        "spec_acceptance_rate",
        "spec_accepted_total",
        "spec_drafted_total",
        "spec_rectangular_verification_rounds",
        "spec_rounds",
        "spec_serial_verification_rounds",
        "spec_verification_mode",
        "spec_verify_replay_disagreements",
        "timed_token_near_tie_relative_gap",
        "timed_token_tolerance_per_thousand",
        "token_mismatch_count",
        // `token_mismatch_first_step` is never sealed: the candidate chose where its stream went.
        "token_mismatch_near_tie_count",
        "token_mismatch_second_choice_count",
        "token_mismatch_second_choice_max_relative_gap",
    ];

    /// KEY-SET SNAPSHOT. The 56 pre-existing keys are UNCHANGED — none renamed, dropped or
    /// reordered — and a FULLY populated metrics block adds exactly the sanctioned additive set and
    /// nothing else.
    #[test]
    fn the_spec_and_identity_seals_are_purely_additive() {
        let before: std::collections::BTreeSet<String> = METRICS_KEYS_BEFORE_PER_PROMPT
            .iter()
            .map(|k| (*k).to_string())
            .collect();
        assert_eq!(
            before.len(),
            56,
            "the pinned pre-existing key set is 56 keys"
        );

        // Nothing populated: byte-for-byte the historical key set.
        assert_eq!(sealed_metrics_keys(zero_metrics()), before);

        let populated = ScoreMetrics {
            local_phases: Some(LocalPhases::default()),
            per_prompt: vec![ScorePerPrompt {
                prompt_sha256: "ab".repeat(32),
                effective_mean_draft_len: 2.0,
                mtp_seconds_per_token_mean: 0.125,
                spec_rounds: Some(64),
                spec_drafted_total: Some(64),
                spec_accepted_total: Some(32),
                head_provenance_sha256: Some("cd".repeat(32)),
            }],
            effective_spec_mode: Some("mtp".to_string()),
            effective_spec_depth: Some(1),
            spec_rounds: Some(64),
            spec_drafted_total: Some(64),
            spec_accepted_total: Some(32),
            spec_acceptance_rate: Some(0.5),
            spec_verify_replay_disagreements: Some(7),
            spec_verification_mode: Some("rectangular".to_string()),
            spec_rectangular_verification_rounds: Some(7),
            spec_serial_verification_rounds: Some(0),
            acceptance_lengths: AcceptanceLengthsSummary::of(&[2; 64]),
            engine_backend: Some("ds4-dfm-rs@abc".to_string()),
            engine_device: Some("cuda sm_121".to_string()),
            engine_protocol_version: Some(1),
            head_provenance_sha256: Some("cd".repeat(32)),
            runner_id: Some("layr/qwen4exp-125b-a6b".to_string()),
            runner_model_type: Some("qwen4_exp".to_string()),
            runner_manifest_sha256: Some("ef".repeat(32)),
            runner_build: Some("c4089870".to_string()),
            resident_pid: Some(4242),
            resident_load_epoch: Some(1_756_944_000),
            timed_token_tolerance_per_thousand: Some(100),
            timed_token_near_tie_relative_gap: Some(0.05),
            token_mismatch_count: Some(3),
            token_mismatch_first_step: Some(17),
            token_mismatch_near_tie_count: Some(2),
            token_mismatch_second_choice_count: Some(2),
            token_mismatch_second_choice_max_relative_gap: Some(0.0125),
            ..zero_metrics()
        };
        let after = sealed_metrics_keys(populated);
        let added: Vec<String> = after.difference(&before).cloned().collect();
        let removed: Vec<String> = before.difference(&after).cloned().collect();
        assert_eq!(added, ADDITIVE_METRICS_KEYS, "the additive set, exactly");
        assert!(removed.is_empty(), "existing keys removed: {removed:?}");
    }

    /// THE TOKEN COUNTS ARE THE CANDIDATE'S CHOICE: the sealed record carries them only bucketed
    /// (0 stays 0, else the next power of two), never the first mismatch step, and the gap at 2
    /// decimals — on the flat metrics and on every `paired_legs` row. In memory they stay exact.
    #[test]
    fn the_sealed_record_holds_no_exact_token_counts() {
        let record = PairedLegRecord {
            pair: 1,
            prompt_sha256: "ab".repeat(32),
            control_prefill_seconds_per_token: 0.001,
            control_decode_seconds_per_token: 0.03,
            candidate_prefill_seconds_per_token: 0.001,
            candidate_decode_seconds_per_token: 0.02,
            control_seed_prefill_window_seconds_per_token: None,
            control_decode_window_seconds_per_token: None,
            candidate_seed_prefill_window_seconds_per_token: None,
            candidate_decode_window_seconds_per_token: None,
            token_mismatch_count: Some(13),
            token_mismatch_first_step: Some(77),
            token_mismatch_near_tie_count: Some(5),
            token_mismatch_second_choice_count: Some(0),
            token_mismatch_second_choice_max_relative_gap: Some(0.012_345),
        };
        let payload = ScorePayload {
            score: None,
            passed: false,
            metrics: ScoreMetrics {
                token_mismatch_count: Some(13),
                token_mismatch_first_step: Some(77),
                token_mismatch_near_tie_count: Some(5),
                token_mismatch_second_choice_count: Some(0),
                token_mismatch_second_choice_max_relative_gap: Some(0.012_345),
                paired_legs: vec![record],
                ..zero_metrics()
            },
        };
        let json = payload.to_sealed_json().unwrap();
        for raw in ["13", "77", "0.012345", "0.0123"] {
            assert!(!json.contains(raw), "{raw} sealed: {json}");
        }
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let m = &v["metrics"];
        assert_eq!(m["token_mismatch_count"], 16);
        assert_eq!(m["token_mismatch_near_tie_count"], 8);
        assert_eq!(m["token_mismatch_second_choice_count"], 0);
        assert_eq!(m["token_mismatch_second_choice_max_relative_gap"], 0.01);
        assert!(m.get("token_mismatch_first_step").is_none());
        let row = &m["paired_legs"][0];
        assert_eq!(row["token_mismatch_count"], 16);
        assert!(row["token_mismatch_first_step"].is_null());
        assert_eq!(row["token_mismatch_near_tie_count"], 8);
        assert_eq!(row["token_mismatch_second_choice_max_relative_gap"], 0.01);
        // The organizer's golden digest is not the engine's choice: carried as is.
        assert_eq!(row["prompt_sha256"], "ab".repeat(32));
        // In memory the exact values remain for the judge and the log.
        assert_eq!(payload.metrics.token_mismatch_count, Some(13));
        for (n, b) in [(0, 0), (1, 1), (2, 2), (3, 4), (13, 16), (129, 256)] {
            assert_eq!(sealed_count_bucket(n), b, "{n}");
        }
    }

    /// The per-prompt entry keeps its historical THREE keys when the run drafted nothing and the
    /// engine announced no head, and grows only the four additive ones when it did.
    #[test]
    fn per_prompt_additive_keys_are_omitted_when_unset() {
        let entry_keys = |pp: ScorePerPrompt| -> Vec<String> {
            let json = ScorePayload {
                score: Some(1.0),
                passed: true,
                metrics: ScoreMetrics {
                    per_prompt: vec![pp],
                    ..zero_metrics()
                },
            }
            .to_sealed_json()
            .unwrap();
            let v: serde_json::Value = serde_json::from_str(&json).unwrap();
            v["metrics"]["per_prompt"][0]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect()
        };

        assert_eq!(
            entry_keys(ScorePerPrompt {
                prompt_sha256: "ab".repeat(32),
                effective_mean_draft_len: 1.0,
                mtp_seconds_per_token_mean: 0.125,
                ..Default::default()
            }),
            vec![
                "effective_mean_draft_len",
                "mtp_seconds_per_token_mean",
                "prompt_sha256"
            ]
        );
        assert_eq!(
            entry_keys(ScorePerPrompt {
                prompt_sha256: "ab".repeat(32),
                effective_mean_draft_len: 2.0,
                mtp_seconds_per_token_mean: 0.125,
                spec_rounds: Some(64),
                spec_drafted_total: Some(64),
                spec_accepted_total: Some(32),
                head_provenance_sha256: Some("cd".repeat(32)),
            }),
            vec![
                "effective_mean_draft_len",
                "head_provenance_sha256",
                "mtp_seconds_per_token_mean",
                "prompt_sha256",
                "spec_accepted_total",
                "spec_drafted_total",
                "spec_rounds",
            ]
        );
    }

    #[test]
    fn per_prompt_json_keys_round_trip_under_the_names_the_board_reads() {
        let metrics = ScoreMetrics {
            decode_seconds_per_token: 0.125,
            per_prompt: vec![ScorePerPrompt {
                prompt_sha256: "cd".repeat(32),
                effective_mean_draft_len: 0.0,
                mtp_seconds_per_token_mean: 0.125,
                ..Default::default()
            }],
            ..zero_metrics()
        };
        let json = ScorePayload {
            score: Some(1.0),
            passed: true,
            metrics: metrics.clone(),
        }
        .to_sealed_json()
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();

        // The reader's exact key names (yukon throughput-metrics.ts): the array is under
        // `metrics.per_prompt`, and each entry carries these three, no more.
        let entry = &v["metrics"]["per_prompt"][0];
        let keys: Vec<&str> = entry
            .as_object()
            .unwrap()
            .keys()
            .map(|s| s.as_str())
            .collect();
        assert_eq!(
            keys,
            vec![
                "effective_mean_draft_len",
                "mtp_seconds_per_token_mean",
                "prompt_sha256",
            ]
        );
        assert_eq!(entry["prompt_sha256"], serde_json::json!("cd".repeat(32)));
        assert_eq!(entry["effective_mean_draft_len"], serde_json::json!(0.0));
        assert_eq!(
            entry["mtp_seconds_per_token_mean"],
            serde_json::json!(0.125)
        );

        // Round-trips back into the typed payload unchanged (the overlay deserializes sealed
        // scores), and the coarsening pass leaves the array verbatim.
        let back: ScorePayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.metrics.per_prompt, metrics.per_prompt);
        assert_eq!(
            metrics.with_coarsened_public_diagnostics(2).per_prompt,
            metrics.per_prompt,
            "per_prompt mirrors the ranking fields: never coarsened"
        );
    }
}

#[cfg(test)]
mod migration_pin_tests {
    use super::*;

    /// THE MIGRATION PIN CHANGES NOTHING ELSE. A run that loaded no `--contract` seals BYTE-IDENTICAL
    /// JSON to what it sealed before the field existed, and a run that loaded one seals exactly the
    /// same bytes PLUS `"contract_sha256"` — no key moves, no number changes.
    ///
    /// This is the acceptance evidence for "the replay produces the same score numbers before and
    /// after": the payload writer is the only thing between a measured run and its artifact, and it
    /// is proved here to be additive-and-omitted-when-unset.
    ///
    /// REVERT-PROOF: drop the `skip_serializing_if` and the first assertion goes red (a
    /// contract-less run's bytes would gain a `"contract_sha256":null`). Seal the digest into any
    /// other field and the second assertion goes red.
    #[test]
    fn the_sealed_bytes_change_by_exactly_one_omitted_when_unset_key() {
        let payload = |digest: Option<&str>| ScorePayload {
            score: Some(2.88),
            passed: true,
            metrics: ScoreMetrics {
                decode_seconds_per_token: 0.0345,
                prefill_seconds_per_token: 0.000_367,
                decode_speedup: 1.21,
                prefill_speedup: 1.03,
                golden_hash: "f".repeat(64),
                contract_sha256: digest.map(str::to_string),
                ..Default::default()
            },
        };

        let without = payload(None).to_sealed_json().unwrap();
        assert!(
            !without.contains("contract_sha256"),
            "a run that loaded no fixture must seal the bytes it always sealed: {without}"
        );

        let digest = "a".repeat(64);
        let with = payload(Some(&digest)).to_sealed_json().unwrap();
        let (bare, pinned): (serde_json::Value, serde_json::Value) = (
            serde_json::from_str(&without).unwrap(),
            serde_json::from_str(&with).unwrap(),
        );
        let mut stripped = pinned.clone();
        assert_eq!(
            stripped["metrics"]
                .as_object_mut()
                .unwrap()
                .remove("contract_sha256"),
            Some(serde_json::Value::String(digest)),
            "the digest is sealed under `metrics.contract_sha256`, beside `golden_hash`"
        );
        assert_eq!(
            bare, stripped,
            "sealing the contract digest must change EXACTLY that one key — every score number, \
             floor, band and identity is byte-for-byte what it was"
        );
    }

    /// THE PER-GROUP SOURCE SEAL CHANGES NOTHING ELSE either. A payload that records no sources
    /// seals byte-identical JSON to what it sealed before the field existed, and a payload that
    /// records sources seals exactly the same bytes PLUS `"contract_sources"`. The ALL-TABLE shape
    /// a STEP-1 binary sealed is the payload here, because it is the one every group of it fills.
    ///
    /// This is the acceptance evidence the brief asks for: the sealed artifact differs from the
    /// pre-migration one by exactly the new key, so no score, floor, band or identity moved.
    ///
    /// REVERT-PROOF: drop the `skip_serializing_if` and the first assertion goes red; seal the
    /// sources into any other field and the second goes red; change a resolved VALUE anywhere in
    /// the payload and the final equality goes red.
    #[test]
    fn the_all_table_source_seal_changes_by_exactly_one_omitted_when_unset_key() {
        let all_table = bench_core::contract::ContractSources::ALL_TABLE;
        let payload = |sources: Option<bench_core::contract::ContractSources>| ScorePayload {
            score: Some(2.88),
            passed: true,
            metrics: ScoreMetrics {
                decode_seconds_per_token: 0.0345,
                prefill_seconds_per_token: 0.000_367,
                decode_speedup: 1.21,
                prefill_speedup: 1.03,
                golden_hash: "f".repeat(64),
                contract_sources: sources,
                ..Default::default()
            },
        };

        let without = payload(None).to_sealed_json().unwrap();
        assert!(
            !without.contains("contract_sources"),
            "a payload that records no sources must seal the bytes it always sealed: {without}"
        );

        let with = payload(Some(all_table)).to_sealed_json().unwrap();
        let (bare, sealed): (serde_json::Value, serde_json::Value) = (
            serde_json::from_str(&without).unwrap(),
            serde_json::from_str(&with).unwrap(),
        );
        let mut stripped = sealed.clone();
        assert_eq!(
            stripped["metrics"]
                .as_object_mut()
                .unwrap()
                .remove("contract_sources"),
            Some(serde_json::json!({
                "scored_regime": "table",
                "live_control_leg": "table",
                "paired_flow": "table",
                "official_baseline": "table",
                "acceptance_bands": "table",
                "scoring_weights": "table",
                "window_shape": "table",
                "model_shape": "table",
            })),
            "the sources are sealed under `metrics.contract_sources`, one key per group"
        );
        assert_eq!(
            bare, stripped,
            "recording the per-group sources must change EXACTLY that one key — every score \
             number, floor, band and identity is byte-for-byte what it was"
        );
    }

    /// NAMING THE DEVICE SOURCE MOVES ONE WORD. A 125B run leaves the scored regime and the
    /// official baseline pair undeclared, because the device measures its own reference benchmark.
    /// That artifact used to seal `"table"` for those two groups and now seals `"device"`. Seal the
    /// SAME payload under both spellings: the two documents differ in exactly those two values, and
    /// in nothing else.
    ///
    /// REVERT-PROOF: emit `table` again and the first two assertions go red; change any other
    /// sealed value and the final equality goes red.
    #[test]
    fn naming_the_device_source_changes_exactly_the_one_word() {
        use bench_core::contract::{ContractSource, ContractSources};
        // What a 125B fixture declares: everything but the regime and the pair.
        let declared = ContractSources {
            scored_regime: ContractSource::Contract,
            live_control_leg: ContractSource::Contract,
            paired_flow: ContractSource::Contract,
            official_baseline: ContractSource::Contract,
            acceptance_bands: ContractSource::Contract,
            scoring_weights: ContractSource::Contract,
            window_shape: ContractSource::Contract,
            model_shape: ContractSource::Contract,
        };
        let mut was = declared;
        was.scored_regime = ContractSource::Table;
        was.official_baseline = ContractSource::Table;
        let mut now = declared;
        now.scored_regime = ContractSource::Device;
        now.official_baseline = ContractSource::Device;

        let seal = |sources: ContractSources| -> serde_json::Value {
            let payload = ScorePayload {
                score: Some(2.88),
                passed: true,
                metrics: ScoreMetrics {
                    decode_seconds_per_token: 0.0345,
                    prefill_seconds_per_token: 0.000_367,
                    decode_speedup: 1.21,
                    prefill_speedup: 1.03,
                    golden_hash: "f".repeat(64),
                    contract_sources: Some(sources),
                    ..Default::default()
                },
            };
            serde_json::from_str(&payload.to_sealed_json().unwrap()).unwrap()
        };
        let (before, after) = (seal(was), seal(now));

        assert_eq!(
            after["metrics"]["contract_sources"]["scored_regime"],
            serde_json::json!("device"),
            "the device-measured regime seals the device"
        );
        assert_eq!(
            after["metrics"]["contract_sources"]["official_baseline"],
            serde_json::json!("device"),
            "the device-measured baseline pair seals the device"
        );

        let strip = |mut doc: serde_json::Value| -> serde_json::Value {
            let sources = doc["metrics"]["contract_sources"].as_object_mut().unwrap();
            sources.remove("scored_regime");
            sources.remove("official_baseline");
            doc
        };
        assert_eq!(
            strip(before),
            strip(after),
            "the rename must move EXACTLY those two words — every other sealed byte is what it was"
        );
    }
}

#[cfg(test)]
mod sealed_key_pin_tests {
    use super::*;

    fn fully_populated_paired_leg() -> PairedLegRecord {
        PairedLegRecord {
            pair: 1,
            prompt_sha256: "v140".to_string(),
            control_prefill_seconds_per_token: 2.5,
            control_decode_seconds_per_token: 3.5,
            candidate_prefill_seconds_per_token: 4.5,
            candidate_decode_seconds_per_token: 5.5,
            control_seed_prefill_window_seconds_per_token: None,
            control_decode_window_seconds_per_token: None,
            candidate_seed_prefill_window_seconds_per_token: None,
            candidate_decode_window_seconds_per_token: None,
            token_mismatch_count: Some(141),
            token_mismatch_first_step: Some(143),
            token_mismatch_near_tie_count: Some(159),
            token_mismatch_second_choice_count: Some(145),
            token_mismatch_second_choice_max_relative_gap: Some(147.5),
        }
    }

    /// ONE fully populated GATE POINT: both gates, every reading present. The skip side of the
    /// shape (`skip_reason`, no readings) is pinned by the gate log's own tests.
    fn fully_populated_gate_record() -> crate::quiescegate::GateRecord {
        crate::quiescegate::GateRecord {
            pair: Some(1),
            pass: Some(2),
            leg: Some("control".to_string()),
            phase: "decode".to_string(),
            quiescence: crate::quiescegate::QuiescenceReading {
                state: "passed".to_string(),
                waited_seconds: 45,
                load: Some(1.25),
                gpu_util: Some(0.03),
                skip_reason: Some("v137".to_string()),
            },
            cool: crate::quiescegate::CoolReading {
                state: "passed".to_string(),
                waited_seconds: 30,
                gpu_temp_c: Some(39.5),
                skip_reason: Some("v139".to_string()),
            },
        }
    }

    fn fully_populated_per_prompt() -> ScorePerPrompt {
        ScorePerPrompt {
            prompt_sha256: "v125".to_string(),
            effective_mean_draft_len: 126.5,
            mtp_seconds_per_token_mean: 127.5,
            spec_rounds: Some(129),
            spec_drafted_total: Some(131),
            spec_accepted_total: Some(133),
            head_provenance_sha256: Some("v135".to_string()),
        }
    }

    fn fully_populated_payload() -> ScorePayload {
        ScorePayload {
            score: Some(0.5),
            passed: true,
            metrics: ScoreMetrics {
                local_phases: Some(LocalPhases::default()),
                peak_ram_gb: 1.5,
                bandwidth_gb_per_token: 2.5,
                decode_seconds_per_token: 3.5,
                prefill_seconds_per_token: 4.5,
                baseline_decode_seconds_per_token: 5.5,
                baseline_prefill_seconds_per_token: 6.5,
                decode_speedup: 7.5,
                prefill_speedup: 8.5,
                decode_speedup_floor: 9.5,
                prefill_speedup_floor: 10.5,
                passed_decode_speedup_floor: true,
                passed_prefill_speedup_floor: true,
                benchmark_wall_seconds: 13.5,
                preflight_seconds: 14.5,
                correctness_seconds: 15.5,
                timed_benchmark_seconds: 16.5,
                gpqa_ttft_passed: true,
                gpqa_ttft_pass_count: 18,
                gpqa_ttft_case_count: 19,
                gpqa_ttft_seconds: 20.5,
                gpqa_ttft_p50_seconds: 21.5,
                gpqa_ttft_max_seconds: 22.5,
                gpqa_ttft_source: "v23".to_string(),
                semantic_gpqa_passed: true,
                semantic_gpqa_pass_count: 25,
                semantic_gpqa_case_count: 26,
                semantic_gpqa_model: "v27".to_string(),
                process_resident_memory_gb: 28.5,
                passed_correctness: true,
                num_layers: 30,
                checked_steps: 31,
                case_count: 32,
                expert_cache_hits: 33,
                expert_cache_misses: 34,
                expert_cache_evictions: 35,
                expert_bytes_read: 36,
                expert_read_seconds: 37.5,
                expert_peak_cached_tensors: 38,
                expert_hit_rate: 39.5,
                first_failing_layer: Some(41),
                first_failing_case: Some("v43".to_string()),
                first_failing_step: Some(45),
                expected_token: Some(47),
                actual_token: Some(49),
                max_abs_diff: 50.5,
                golden_hash: "v51".to_string(),
                contract_sha256: Some("v53".to_string()),
                contract_sources: Some(bench_core::contract::ContractSources::ALL_TABLE),
                bandwidth_source: "v54".to_string(),
                error: "v55".to_string(),
                commit: "v56".to_string(),
                timestamp: "v57".to_string(),
                harness_hash: "v58".to_string(),
                weights_hash: "v59".to_string(),
                weights_byte_count: 60,
                weights_file_count: 61,
                runtime: "v62".to_string(),
                partial_result: true,
                per_prompt: vec![fully_populated_per_prompt()],
                effective_spec_mode: Some("v65".to_string()),
                effective_spec_depth: Some(67),
                spec_rounds: Some(69),
                spec_drafted_total: Some(71),
                spec_accepted_total: Some(73),
                spec_acceptance_rate: Some(75.5),
                spec_verify_replay_disagreements: Some(77),
                spec_verification_mode: Some("v79".to_string()),
                spec_rectangular_verification_rounds: Some(81),
                spec_serial_verification_rounds: Some(83),
                acceptance_lengths: AcceptanceLengthsSummary::of(&[1, 2, 3]),
                engine_backend: Some("v86".to_string()),
                engine_device: Some("v88".to_string()),
                engine_protocol_version: Some(90),
                head_provenance_sha256: Some("v92".to_string()),
                runner_id: Some("v94".to_string()),
                runner_model_type: Some("v96".to_string()),
                runner_manifest_sha256: Some("v98".to_string()),
                runner_build: Some("v100".to_string()),
                resident_pid: Some(102),
                resident_load_epoch: Some(104),
                baseline_source: Some("v106".to_string()),
                baseline_box: Some("v108".to_string()),
                baseline_calibration_sha256: Some("v110".to_string()),
                baseline_golden_sha256: Some("v112".to_string()),
                live_prompt: Some("v113".to_string()),
                baseline_reference_commit: Some("v114".to_string()),
                baseline_band_passed: Some(true),
                baseline_leg_prefill_seconds_per_token: Some(118.5),
                baseline_leg_decode_seconds_per_token: Some(120.5),
                candidate_leg_prefill_seconds_per_token: Some(122.5),
                candidate_leg_decode_seconds_per_token: Some(124.5),
                paired_legs: vec![fully_populated_paired_leg()],
                gates: vec![fully_populated_gate_record()],
                baseline_leg_decode_window_seconds_per_token: None,
                candidate_leg_decode_window_seconds_per_token: None,
                baseline_leg_seed_prefill_window_seconds_per_token: None,
                candidate_leg_seed_prefill_window_seconds_per_token: None,
                timed_token_tolerance_per_thousand: Some(149),
                timed_token_near_tie_relative_gap: Some(0.25),
                token_mismatch_count: Some(151),
                token_mismatch_first_step: Some(153),
                token_mismatch_near_tie_count: Some(161),
                token_mismatch_second_choice_count: Some(155),
                token_mismatch_second_choice_max_relative_gap: Some(157.5),
            },
        }
    }

    /// EVERY SEALED BYTE OF A FULLY-POPULATED SCORE, PINNED.
    ///
    /// The 96 `#[serde(rename = "x")]` attributes this file used to carry all named the field
    /// they decorated, so none of them could move a key. This test is the evidence: it seals a
    /// value in which every field — including every `Option` and every nested record — is
    /// populated, and compares the whole artifact byte-for-byte. Removing a no-op rename leaves
    /// it green; removing a load-bearing one would rename a key and turn it red.
    ///
    /// REVERT-PROOF: add `#[serde(rename = "…")]` naming anything but the field, and the key it
    /// decorates moves in the sorted output.
    #[test]
    fn a_fully_populated_score_seals_these_exact_bytes() {
        assert_eq!(
            fully_populated_payload().to_sealed_json().unwrap(),
            SEALED_BYTES,
            "the sealed artifact is the board's contract: no key may be added, dropped or renamed"
        );
    }

    /// The bytes [`a_fully_populated_score_seals_these_exact_bytes`] pins. Captured from the tree
    /// as it stood before the no-op renames were deleted.
    const SEALED_BYTES: &str = r#"{
  "metrics": {
    "acceptance_lengths": {
      "count": 3,
      "mean": 2.0
    },
    "actual_token": null,
    "bandwidth_gb_per_token": 2.5,
    "bandwidth_source": "v54",
    "baseline_band_passed": true,
    "baseline_box": "v108",
    "baseline_calibration_sha256": "v110",
    "baseline_decode_seconds_per_token": 5.5,
    "baseline_golden_sha256": "v112",
    "baseline_leg_decode_seconds_per_token": 120.5,
    "baseline_leg_prefill_seconds_per_token": 118.5,
    "baseline_prefill_seconds_per_token": 6.5,
    "baseline_reference_commit": "v114",
    "baseline_source": "v106",
    "benchmark_wall_seconds": 16.0,
    "candidate_leg_decode_seconds_per_token": 124.5,
    "candidate_leg_prefill_seconds_per_token": 122.5,
    "case_count": 32,
    "checked_steps": 31,
    "commit": "v56",
    "contract_sha256": "v53",
    "contract_sources": {
      "acceptance_bands": "table",
      "live_control_leg": "table",
      "model_shape": "table",
      "official_baseline": "table",
      "paired_flow": "table",
      "scored_regime": "table",
      "scoring_weights": "table",
      "window_shape": "table"
    },
    "correctness_seconds": 16.0,
    "decode_seconds_per_token": 3.5,
    "decode_speedup": 7.5,
    "decode_speedup_floor": 9.5,
    "effective_spec_depth": 67,
    "effective_spec_mode": "v65",
    "engine_backend": "v86",
    "engine_device": "v88",
    "engine_protocol_version": 90,
    "error": "correctness case case-sha256:04247d855818b509 did not match the golden",
    "expected_token": null,
    "expert_bytes_read": 36,
    "expert_cache_evictions": 35,
    "expert_cache_hits": 33,
    "expert_cache_misses": 34,
    "expert_hit_rate": 40.0,
    "expert_peak_cached_tensors": 38,
    "expert_read_seconds": 38.0,
    "first_failing_case": "case-sha256:04247d855818b509",
    "first_failing_layer": 41,
    "first_failing_step": 64,
    "gates": [
      {
        "cool": {
          "gpu_temp_c": 39.5,
          "skip_reason": "v139",
          "state": "passed",
          "waited_seconds": 30
        },
        "leg": "control",
        "pair": 1,
        "pass": 2,
        "phase": "decode",
        "quiescence": {
          "gpu_util": 0.03,
          "load": 1.25,
          "skip_reason": "v137",
          "state": "passed",
          "waited_seconds": 45
        }
      }
    ],
    "golden_hash": "v51",
    "gpqa_ttft_case_count": 19,
    "gpqa_ttft_max_seconds": 22.0,
    "gpqa_ttft_p50_seconds": 22.0,
    "gpqa_ttft_pass_count": 18,
    "gpqa_ttft_passed": true,
    "gpqa_ttft_seconds": 20.0,
    "gpqa_ttft_source": "v23",
    "harness_hash": "v58",
    "head_provenance_sha256": "v92",
    "live_prompt": "v113",
    "local_phases": {
      "correctness": "not_run",
      "correctness_checked_steps": 0,
      "timing": "not_run"
    },
    "max_abs_diff": 50.0,
    "num_layers": 30,
    "paired_legs": [
      {
        "candidate_decode_seconds_per_token": 5.5,
        "candidate_prefill_seconds_per_token": 4.5,
        "control_decode_seconds_per_token": 3.5,
        "control_prefill_seconds_per_token": 2.5,
        "pair": 1,
        "prompt_sha256": "v140",
        "token_mismatch_count": 256,
        "token_mismatch_first_step": null,
        "token_mismatch_near_tie_count": 256,
        "token_mismatch_second_choice_count": 256,
        "token_mismatch_second_choice_max_relative_gap": 147.5
      }
    ],
    "partial_result": true,
    "passed_correctness": true,
    "passed_decode_speedup_floor": true,
    "passed_prefill_speedup_floor": true,
    "peak_ram_gb": 1.5,
    "per_prompt": [
      {
        "effective_mean_draft_len": 126.5,
        "head_provenance_sha256": "v135",
        "mtp_seconds_per_token_mean": 127.5,
        "prompt_sha256": "v125",
        "spec_accepted_total": 133,
        "spec_drafted_total": 131,
        "spec_rounds": 129
      }
    ],
    "prefill_seconds_per_token": 4.5,
    "prefill_speedup": 8.5,
    "prefill_speedup_floor": 10.5,
    "preflight_seconds": 14.0,
    "process_resident_memory_gb": 28.0,
    "resident_load_epoch": 104,
    "resident_pid": 102,
    "runner_build": "v100",
    "runner_id": "v94",
    "runner_manifest_sha256": "v98",
    "runner_model_type": "v96",
    "runtime": "v62",
    "semantic_gpqa_case_count": 26,
    "semantic_gpqa_model": "v27",
    "semantic_gpqa_pass_count": 25,
    "semantic_gpqa_passed": true,
    "spec_acceptance_rate": 75.5,
    "spec_accepted_total": 73,
    "spec_drafted_total": 71,
    "spec_rectangular_verification_rounds": 81,
    "spec_rounds": 69,
    "spec_serial_verification_rounds": 83,
    "spec_verification_mode": "v79",
    "spec_verify_replay_disagreements": 77,
    "timed_benchmark_seconds": 16.0,
    "timed_token_near_tie_relative_gap": 0.25,
    "timed_token_tolerance_per_thousand": 149,
    "timestamp": "v57",
    "token_mismatch_count": 256,
    "token_mismatch_near_tie_count": 256,
    "token_mismatch_second_choice_count": 256,
    "token_mismatch_second_choice_max_relative_gap": 157.5,
    "weights_byte_count": 60,
    "weights_file_count": 61,
    "weights_hash": "v59"
  },
  "passed": true,
  "score": 0.5
}"#;
}
