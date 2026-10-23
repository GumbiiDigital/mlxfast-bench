//! Engine Protocol v1 — the single normative definition of the benchmarker <-> engine wire.
//!
//! NDJSON over stdio (co-located) or localhost TCP (via bench-agent / compose network).
//! Only token IDs, top-K logits, and RSS cross this boundary; all timing is parent-side.
//! Message kinds: hello, prefill, decode_begin, decode_step, correctness,
//! correctness_begin/_step, phase_diagnostics. See docs/architecture.md §2 and PROTOCOL.md.
//!
//! These structs are a faithful port of the Swift Codable types that define the wire today:
//! - [`WorkerRequest`]  <- `RuntimeWorkerRequest`  (Sources/MLXFastHarness/QwenRuntimeWorker.swift)
//! - [`WorkerResponse`] <- `RuntimeWorkerResponse` (Sources/MLXFastHarness/QwenRuntimeWorker.swift)
//! - [`CorrectnessTraceLogit`] <- `CorrectnessTraceLogit` (Sources/MLXFastHarness/QwenRuntime.swift)
//! - [`ExpertStreamingStats`]  <- `ExpertStreamingStats`  (Sources/MLXFastCore/ExpertStreamingStats.swift)
//!
//! Serde fidelity contract (matches Swift's auto-synthesized Codable):
//! - Optional fields carry `skip_serializing_if = "Option::is_none"` so a `nil`/`None`
//!   field is OMITTED from the JSON, exactly like Swift's `encodeIfPresent`, and
//!   `default` so a missing key deserializes to `None`.
//! - Rust field/JSON-key order matches the Swift declaration order, so parse-then-
//!   reserialize of a canonical (declaration-ordered, no-nil-keys) line is byte-identical.
//! - The Phase-0 `hello` fields (`protocol_version`, `backend`, `device`) are appended
//!   last so pre-Phase-0 messages still round-trip unchanged.

use serde::{Deserialize, Serialize};

/// Engine Protocol version implemented by this crate. Reported on the `hello`.
pub const PROTOCOL_VERSION: u32 = 1;

/// Top-k logit count carried by `correctness_begin` / `correctness_step` responses
/// (`top_logits[8]`, PROTOCOL.md). The runner rejects a response whose `top_logits`
/// length differs from this.
pub const TOP_LOGITS_K: usize = 8;

/// Speculative-config mode string for the plain serial (no drafter) path.
/// See [`SpecConfig`] and `docs/spec-config-design.md`.
pub const SPEC_MODE_SERIAL: &str = "serial";
/// Speculative-config mode string for the native multi-token-prediction (MTP) path.
pub const SPEC_MODE_MTP: &str = "mtp";
/// Speculative-config mode string for the DFlash draft-model path.
pub const SPEC_MODE_DFLASH: &str = "dflash";
/// Speculative-config mode string for the (reserved) DSpark path.
pub const SPEC_MODE_DSPARK: &str = "dspark";
/// Documented speculative-config mode strings.
pub const SPEC_MODES: [&str; 4] = [
    SPEC_MODE_SERIAL,
    SPEC_MODE_MTP,
    SPEC_MODE_DFLASH,
    SPEC_MODE_DSPARK,
];

/// Documented wire values for [`WorkerResponse::verification_mode`].
pub const VERIFICATION_MODE_RECTANGULAR: &str = "rectangular";
pub const VERIFICATION_MODE_SERIAL: &str = "serial";
pub const VERIFICATION_MODES: [&str; 2] = [VERIFICATION_MODE_RECTANGULAR, VERIFICATION_MODE_SERIAL];

/// Documented wire value for `CohortReferenceReplayReport::logit_provenance`.
pub const LOGIT_PROVENANCE_POST_SOFTCAP: &str = "post_softcap";

/// Protocol v1.1 capability advertised on `hello.capabilities` when the engine supports
/// the oracle-verified free-run timed decode mode (`free_decode_begin` / `free_decode_run`).
///
/// v1.1 is WIRE-ADDITIVE: `hello.protocol_version` stays `1` (PROTOCOL-v1.1.md §2.1,
/// RULED OQ1); the mode is advertised by this capability flag, and benchd REFUSES to
/// issue the `free_decode_*` kinds to an engine that does not advertise it (an unadvertised
/// capability is a hard protocol error, never a silent fallback). A single engine binary can
/// thus serve both the v1 teacher-forced and the v1.1 free-run regimes.
pub const CAPABILITY_FREE_RUN_DECODE: &str = "free_run_decode";

/// Protocol v1.2 capability advertised on `hello.capabilities` when the engine supports the
/// BATCHED (cohort-shaped) form of the oracle-verified free-run timed decode mode — the same
/// `free_decode_begin` / `free_decode_run` verbs, carrying the cohort fields.
///
/// v1.2 is WIRE-ADDITIVE on top of v1.1 and does NOT introduce a parallel verb family: the batched
/// path IS the general path and single-stream is `B = 1` of it (batch-8 design brief, D6 alt (b)
/// rejected). So `hello.protocol_version` stays `1`, the request kinds stay `free_decode_begin` /
/// `free_decode_run`, and the cohort shape is selected by the additive
/// [`WorkerRequest::batch_size`] / [`WorkerRequest::seed_tokens_by_stream`] fields — gated by this
/// capability. benchd REFUSES to issue the cohort form to an engine that does not advertise it,
/// BEFORE the cool gate and BEFORE the clock, exactly as v1.1 refuses `free_run_decode` (an
/// unadvertised capability is a hard protocol error, never a silent fallback to the single-stream
/// regime — that would silently swap the measured quantity class).
pub const CAPABILITY_BATCHED_FREE_RUN_DECODE: &str = "batched_free_run_decode";

/// Per-stream timing instrumentation capability (per-stream-instrumentation-spec.md step 1),
/// advertised on `hello.capabilities` when the engine's batched `free_decode_begin` /
/// `free_decode_run` responses carry the additive [`WorkerResponse::prefill_ns_by_stream`] /
/// [`WorkerResponse::decode_ns_by_stream`] per-slot monotonic-ns vectors. Advertise-before-use,
/// same posture as [`CAPABILITY_BATCHED_FREE_RUN_DECODE`]: benchd refuses to request per-stream
/// scoring against an engine that does not advertise this.
pub const CAPABILITY_PER_STREAM_TIMING: &str = "per_stream_timing";

/// (b) admission — TRUSTED-ORACLE capability advertised on `hello.capabilities` when the engine
/// serves the `cohort_reference_replay` verb (engine #35, d528f6ac). The verb replays each cohort
/// stream TEACHER-FORCED on the candidate's OWN committed tokens over the ORGANIZER's pinned
/// reference weights and reports the reference argmax per position.
///
/// SECURITY (N1): ONLY the organizer's TRUSTED build advertises this flag; benchd REFUSES to issue
/// `cohort_reference_replay` to any worker whose hello did not carry it (advertise-before-use, the
/// same fail-closed posture as [`CAPABILITY_BATCHED_FREE_RUN_DECODE`] — an unadvertised capability is
/// a hard protocol error, never a silent fallback). The UNTRUSTED candidate worker never serves the
/// verb, so the reference argmax can never be produced by the candidate's editable forward code.
pub const CAPABILITY_COHORT_REFERENCE_REPLAY: &str = "cohort_reference_replay";

/// (b) admission (engine #37) — the ONE replay width benchd ever requests: the reference replays
/// the B streams BATCHED at the scored candidate's own cohort geometry, so the comparison is
/// like-for-like and a batch-geometry difference cannot show up as a token divergence. The engine
/// also accepts `"canonical"` (the per-stream width-1 diagnostic) and REFUSES any other value;
/// benchd never sends that one — the scored gate is cohort-width by ruling.
///
/// Sent EXPLICITLY on every request ([`WorkerRequest::replay_width`]) rather than relying on the
/// engine-side default, which happens to be the same value today: an enforced reference geometry
/// that benchd does not state is one benchd cannot verify.
pub const REPLAY_WIDTH_COHORT: &str = "cohort";

/// The Draft 2020-12 JSON Schema for Engine Protocol v1, embedded from
/// `schema/engine-protocol-v1.schema.json`.
pub const JSON_SCHEMA: &str = include_str!("../schema/engine-protocol-v1.schema.json");

/// Per-module speculative configuration — the wire-additive `spec` tagged union
/// (`docs/spec-config-design.md`, David-ruled 2026-08-19).
///
/// Exactly one `mode` per object; the per-mode config block is nested UNDER the mode key
/// (`{"mode":"mtp","mtp":{"depth":2}}`). `serial` carries no block. Depth is a MODULE field
/// (`mtp.depth`), never a top-level benchmarker flag. The benchmarker forwards this block as
/// opaque module input on `decode_begin` and seals ONLY the engine's echoed
/// [`WorkerResponse::effective_spec`]; the module owns its own validation/defaults engine-side.
///
/// The envelope stays CLOSED (`deny_unknown_fields`), so an unknown top-level key is a hard
/// error. Cross-module keys (`{"mode":"mtp","dflash":{…}}`) are legal on the wire here (both
/// blocks are optional) but the engine module rejects them fail-closed; benchd's own
/// never-ignored check (runner) compares the echo to the request, so a divergent echo rejects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecConfig {
    /// The selected mode: `serial` | `mtp` | `dflash` | `dspark`.
    pub mode: String,
    /// The `mtp` module block, present only when `mode == "mtp"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mtp: Option<MtpSpec>,
    /// The `dflash` module block (opaque to benchd), present only when `mode == "dflash"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dflash: Option<serde_json::Value>,
    /// The `dspark` module block (opaque to benchd; RESERVED), present only when `mode == "dspark"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dspark: Option<serde_json::Value>,
}

impl SpecConfig {
    /// The plain serial spec — no drafter module, no depth (`{"mode":"serial"}`).
    pub fn serial() -> Self {
        Self {
            mode: SPEC_MODE_SERIAL.to_string(),
            mtp: None,
            dflash: None,
            dspark: None,
        }
    }

    /// The native-MTP spec at the given draft depth (`{"mode":"mtp","mtp":{"depth":D}}`).
    pub fn mtp(depth: u32) -> Self {
        Self {
            mode: SPEC_MODE_MTP.to_string(),
            mtp: Some(MtpSpec { depth: Some(depth) }),
            dflash: None,
            dspark: None,
        }
    }

    /// The engine-decides MTP spec (`{"mode":"mtp","mtp":{}}`) — David ruling 2026-08-27: depth is
    /// the PARTICIPANT'S variable (their drafter code sets it), so when the operator names no depth
    /// benchd must not inject its own default request. The mtp block stays present (mode↔block
    /// coherence), the `depth` key is omitted, and the engine's envelope resolves/clamps its own
    /// value and reports it in the `effective_spec` echo — which [`spec_echo_honors_request`] then
    /// accepts as the module filling a field the request left to it.
    pub fn mtp_engine_default() -> Self {
        Self {
            mode: SPEC_MODE_MTP.to_string(),
            mtp: Some(MtpSpec { depth: None }),
            dflash: None,
            dspark: None,
        }
    }

    /// The DFlash spec carrying the arm's draft-depth lever
    /// (`{"mode":"dflash","dflash":{"depth":D}}`). The `dflash` block stays an OPAQUE module input
    /// (benchd forwards it verbatim and the module owns its own validation); `depth` is the one
    /// lever benchd later projects back out with [`SpecConfig::dflash_depth`] for the metrics seal.
    pub fn dflash(depth: u32) -> Self {
        Self {
            mode: SPEC_MODE_DFLASH.to_string(),
            mtp: None,
            dflash: Some(serde_json::json!({ "depth": depth })),
            dspark: None,
        }
    }

    /// The DFlash arm's draft-depth lever (`dflash.depth`), STRUCTURALLY decoded out of the
    /// otherwise-opaque `dflash` module block via serde. Returns `Some(depth)` EXACTLY when
    /// `mode == "dflash"` AND the block carries a numeric `depth`; `None` otherwise (a non-dflash
    /// spec, or a dflash block that omits `depth`). This is a READ-ONLY projection: the block stays
    /// opaque module input, no other key is interpreted, and the MTP depth constant is NEVER reused
    /// for the dflash arm.
    pub fn dflash_depth(&self) -> Option<u32> {
        if self.mode != SPEC_MODE_DFLASH {
            return None;
        }
        /// The minimal projection benchd reads out of the opaque `dflash` block: just the `depth`
        /// lever. No `deny_unknown_fields`, so extra module keys are ignored — the block is not
        /// re-schema'd here, only its one benchd-visible lever is decoded.
        #[derive(Deserialize)]
        struct DflashDepthProjection {
            depth: u32,
        }
        let block = self.dflash.as_ref()?;
        serde_json::from_value::<DflashDepthProjection>(block.clone())
            .ok()
            .map(|p| p.depth)
    }
}

/// The `mtp` module config block. Depth is the MODULE's field (`docs/spec-config-design.md` §2):
/// the engine module owns its clamp/default; benchd only carries it and bounds-checks it against
/// the anti-DDoS cap before spawning.
///
/// `depth` is OPTIONAL on the wire (David ruling 2026-08-27): an absent depth means "the engine's
/// drafter decides" — the module resolves its own default/clamp and reports the operating value in
/// the `effective_spec` echo. The engine already parses it exactly this way
/// (`RuntimeWorkerSpecConfig.swift`: `decodeIfPresent`, "the module parses its own block, fills its
/// own defaults"). A PRESENT depth remains a hard request the echo must honor verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MtpSpec {
    /// The MTP draft depth the module runs; `None` = the engine's drafter decides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<u32>,
}

/// Whether an engine's `effective_spec` ECHO honors the REQUEST — the spec-never-ignored rule with
/// engine-decides semantics (David ruling 2026-08-27):
///
/// Every field the request SPECIFIED must appear in the echo with an identical value (a divergent
/// value is still a discard — a requested depth 4 echoed as 3 rejects exactly as before). Fields
/// the request left ABSENT INSIDE ITS MODULE BLOCK are the module's to fill, and the echo REPORTS
/// them — a resolved `mtp.depth`, a dflash `draft` identity block. That tolerance is scoped to
/// WITHIN the mode's own block:
///
/// - The MODE must be identical, and the TOP-LEVEL MODULE-KEY SETS must match exactly — an echo
///   may not introduce a module block the request did not carry (a serial request echoed with a
///   stray `mtp`/`dflash` block refuses, exactly as the pre-ruling byte-equality did) nor drop one
///   the request carried.
/// - Inside the mode's block, the request's keys must echo verbatim (recursively); echo-added keys
///   are the module reporting.
/// - The echo must REPORT a resolved depth: an engine-decides request whose echo also omits
///   `mtp.depth` (or a dflash echo without a numeric `depth`) refuses — with the seal's `mtp_depth`
///   now absent-by-design on engine-decides runs, the echo is the one place the operating depth is
///   recorded, and a silent echo would leave the run with no depth on record at all.
pub fn spec_echo_honors_request(requested: &SpecConfig, effective: &SpecConfig) -> bool {
    if requested.mode != effective.mode {
        return false;
    }
    // Top-level module-key sets match exactly (presence, per block).
    if requested.mtp.is_some() != effective.mtp.is_some()
        || requested.dflash.is_some() != effective.dflash.is_some()
        || requested.dspark.is_some() != effective.dspark.is_some()
    {
        return false;
    }
    // mtp block: a requested depth echoes verbatim; an engine-decides request (depth absent)
    // requires the echo to carry the module-RESOLVED depth.
    if let (Some(req_mtp), Some(echo_mtp)) = (&requested.mtp, &effective.mtp) {
        match req_mtp.depth {
            Some(d) => {
                if echo_mtp.depth != Some(d) {
                    return false;
                }
            }
            None => {
                if echo_mtp.depth.is_none() {
                    return false;
                }
            }
        }
    }
    // dflash block (opaque): request keys must echo verbatim, echo may add module-filled keys
    // (`draft` identity, a resolved `depth`) — and the echo must carry a numeric `depth`.
    if let (Some(req_block), Some(echo_block)) = (&requested.dflash, &effective.dflash) {
        if !json_subset(req_block, echo_block) {
            return false;
        }
        if requested.mode == SPEC_MODE_DFLASH && effective.dflash_depth().is_none() {
            return false;
        }
    }
    // dspark block (opaque, reserved): request keys must echo verbatim.
    if let (Some(req_block), Some(echo_block)) = (&requested.dspark, &effective.dspark) {
        if !json_subset(req_block, echo_block) {
            return false;
        }
    }
    true
}

/// Project an accepted spec echo onto the request for an audit record.
///
/// The never-ignored gate can accept added keys in opaque blocks. Audit records keep only keys
/// that the request named, while preserving the echo's values for those keys.
pub fn project_spec_echo_for_audit(requested: &SpecConfig, effective: &SpecConfig) -> SpecConfig {
    let mut projected = effective.clone();
    projected.dflash = project_opaque_block(requested.dflash.as_ref(), effective.dflash.as_ref());
    projected.dspark = project_opaque_block(requested.dspark.as_ref(), effective.dspark.as_ref());
    projected
}

fn project_opaque_block(
    requested: Option<&serde_json::Value>,
    effective: Option<&serde_json::Value>,
) -> Option<serde_json::Value> {
    match (requested, effective) {
        (Some(requested), Some(effective)) => Some(project_json(requested, effective)),
        _ => None,
    }
}

fn project_json(requested: &serde_json::Value, effective: &serde_json::Value) -> serde_json::Value {
    match (requested, effective) {
        (serde_json::Value::Object(requested), serde_json::Value::Object(effective)) => {
            serde_json::Value::Object(
                requested
                    .iter()
                    .filter_map(|(key, requested_value)| {
                        effective
                            .get(key)
                            .map(|value| (key.clone(), project_json(requested_value, value)))
                    })
                    .collect(),
            )
        }
        _ => effective.clone(),
    }
}

/// `a` ⊆ `b`, recursively: objects require every key of `a` present in `b` with a subset value;
/// everything else requires equality. Arrays are compared by equality — no element-wise subset —
/// because an order/length divergence in a requested array is a real divergence.
fn json_subset(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    match (a, b) {
        (serde_json::Value::Object(ao), serde_json::Value::Object(bo)) => ao
            .iter()
            .all(|(k, av)| bo.get(k).is_some_and(|bv| json_subset(av, bv))),
        _ => a == b,
    }
}

/// Request kind carried by [`WorkerRequest::kind`] — the benchmarker -> engine message
/// kinds, as a typed helper over the raw wire string.
///
/// PROTOCOL.md ("Message kinds") enumerates eight kinds: `hello` is the *unsolicited*
/// `id = 0` **response** the engine emits once at startup, so it is NOT a request kind.
/// The seven variants below are exactly the kinds the benchmarker sends (see also the
/// docs/architecture.md §2 protocol table).
///
/// The wire field stays a `String` on [`WorkerRequest`] (the serialized envelope is
/// frozen — `deny_unknown_fields`, byte-for-byte round-trip). This enum does NOT change
/// the wire; it is the single place that (a) names the canonical wire strings and
/// (b) defines which kinds are *timed steps* ([`RequestKind::is_timed_step`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RequestKind {
    /// `prefill` — force full evaluation of `prompt_tokens`; returns a `token`.
    Prefill,
    /// `decode_begin` — the single seed forward that opens the timed decode window.
    DecodeBegin,
    /// `decode_step` — one timed decode step.
    DecodeStep,
    /// `correctness` — free-run greedy generation; returns `tokens[]`.
    Correctness,
    /// `correctness_begin` — first teacher-forced forward; opens the anchor-gate session.
    CorrectnessBegin,
    /// `correctness_step` — one teacher-forced timed step.
    CorrectnessStep,
    /// `phase_diagnostics` — closes a timed phase (barrier + completed-work counter).
    PhaseDiagnostics,
    /// `free_decode_begin` — v1.1 (additive): the single seed forward that opens an
    /// oracle-verified free-run decode phase. Same contract as `decode_begin`
    /// (`seed_tokens[]` in, `seed_token` out); issued only inside a v1.1 free-run phase.
    ///
    /// v1.2 (additive): the SAME kind carries the COHORT form when
    /// [`WorkerRequest::seed_tokens_by_stream`] + [`WorkerRequest::batch_size`] are present
    /// (B seeds in, [`WorkerResponse::seed_token_by_stream`] + `effective_batch_size` out),
    /// gated by [`CAPABILITY_BATCHED_FREE_RUN_DECODE`]. No new verb.
    FreeDecodeBegin,
    /// `free_decode_run` — v1.1 (additive): the engine free-runs its own MTP loop until it
    /// has committed `count` (= N) tokens, returning all N committed token IDs plus the
    /// AUDIT acceptance counters. PROTOCOL-v1.1.md §2.1.
    ///
    /// v1.2 (additive): the SAME kind carries the COHORT form when
    /// [`WorkerRequest::batch_size`] is present — `count` is then N PER STREAM and the response
    /// carries [`WorkerResponse::tokens_by_stream`] (B x N) instead of `tokens`.
    FreeDecodeRun,
    /// `cohort_reference_replay` — (b) admission (engine #35, d528f6ac): the TRUSTED-ORACLE verb.
    /// Replays each cohort stream TEACHER-FORCED on the candidate's OWN committed tokens
    /// ([`WorkerRequest::replay_seeds_by_stream`] + [`WorkerRequest::committed_by_stream`]) over the
    /// organizer's pinned reference weights and reports the reference `sequential_argmax` per position
    /// ([`WorkerResponse::cohort_reference_replay`]). Served on a PLAIN spawn (NOT behind the
    /// `--speculative-protocol` gate) but capability-gated by
    /// [`CAPABILITY_COHORT_REFERENCE_REPLAY`] — only the trusted build advertises it (N1). NOT a
    /// `completed_work` timed step: it is an UNTIMED post-run correctness oracle, never inside a
    /// scored window.
    CohortReferenceReplay,
}

impl RequestKind {
    /// The exact wire string written to [`WorkerRequest::kind`] for this kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            RequestKind::Prefill => "prefill",
            RequestKind::DecodeBegin => "decode_begin",
            RequestKind::DecodeStep => "decode_step",
            RequestKind::Correctness => "correctness",
            RequestKind::CorrectnessBegin => "correctness_begin",
            RequestKind::CorrectnessStep => "correctness_step",
            RequestKind::PhaseDiagnostics => "phase_diagnostics",
            RequestKind::FreeDecodeBegin => "free_decode_begin",
            RequestKind::FreeDecodeRun => "free_decode_run",
            RequestKind::CohortReferenceReplay => "cohort_reference_replay",
        }
    }

    /// Parse a wire `kind` string into a [`RequestKind`], or `None` if it is not one of
    /// the seven request kinds (e.g. the response-only `hello`, or an unknown string).
    pub fn from_wire(kind: &str) -> Option<RequestKind> {
        match kind {
            "prefill" => Some(RequestKind::Prefill),
            "decode_begin" => Some(RequestKind::DecodeBegin),
            "decode_step" => Some(RequestKind::DecodeStep),
            "correctness" => Some(RequestKind::Correctness),
            "correctness_begin" => Some(RequestKind::CorrectnessBegin),
            "correctness_step" => Some(RequestKind::CorrectnessStep),
            "phase_diagnostics" => Some(RequestKind::PhaseDiagnostics),
            "free_decode_begin" => Some(RequestKind::FreeDecodeBegin),
            "free_decode_run" => Some(RequestKind::FreeDecodeRun),
            "cohort_reference_replay" => Some(RequestKind::CohortReferenceReplay),
            _ => None,
        }
    }

    /// Is this a *timed step* whose completion the phase-close barrier guards?
    ///
    /// True for exactly `decode_begin | decode_step | correctness_begin |
    /// correctness_step` — the kinds that fall inside a timed decode / correctness
    /// window and are counted toward the engine's completed-work counter
    /// (docs/architecture.md §2, phase-close barrier). `prefill` and `correctness` are
    /// setup / free-run, and `phase_diagnostics` is the barrier itself, so none of them
    /// count.
    ///
    /// This is the SINGLE source of truth for the timed-step set. The session's
    /// `issued_steps` increment and the mock engine's completed-work counter both derive
    /// from it, so the previously hand-synced kind lists collapse to this one definition.
    ///
    /// Amendment 4 (PROTOCOL-v1.1.md) locks this `completed_work` timed-step set to exactly
    /// these four. The v1.1 `free_decode_begin` / `free_decode_run` kinds are NOT in it: a
    /// free-run decode phase's `completed_work` is `R + 1` (seed forward + R MTP verify
    /// rounds), which benchd validates against the §2.6 consistency TRIPLE in the free-run
    /// driver — NOT by counting issued steps here. So these two kinds return `false`.
    ///
    /// v1.2 (cohort) does not move that line either, and the reason is NORMATIVE, not
    /// incidental: **a round is ONE engine forward regardless of B**, so a cohort free-run
    /// phase's `completed_work` is still the SCALAR `R + 1` — it does NOT scale with the batch
    /// width. `completed_work` counts forwards, never stream-rounds. Since v1.2 reuses the two
    /// free-run kinds rather than adding a verb family, this set is unchanged by the batch work.
    pub fn is_timed_step(&self) -> bool {
        matches!(
            self,
            RequestKind::DecodeBegin
                | RequestKind::DecodeStep
                | RequestKind::CorrectnessBegin
                | RequestKind::CorrectnessStep
        )
    }
}

/// Benchmarker -> engine request.
///
/// Port of Swift `RuntimeWorkerRequest` (Sources/MLXFastHarness/QwenRuntimeWorker.swift).
/// Field order and JSON keys match the Swift `CodingKeys`. `id` and `kind` are always
/// present; the remaining fields are populated only for the kinds that use them:
/// `prompt_tokens` (prefill / correctness / correctness_begin), `token`
/// (decode_step / correctness_step), `seed_tokens` (decode_begin), `steps` (correctness).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
// C5: CLOSED envelope. The JSON Schema (additionalProperties:false) and PROTOCOL.md
// both declare the wire closed; deny_unknown_fields makes the serde structs agree, so
// all three normative validators match. An open envelope would be a side-channel for
// engine-side smuggled fields — the anti-cheat posture wants fail-closed.
#[serde(deny_unknown_fields)]
pub struct WorkerRequest {
    /// Monotonic request id, echoed back on the matching response.
    pub id: i64,
    /// Message kind, e.g. `"prefill"`, `"decode_begin"`, `"phase_diagnostics"`.
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<Vec<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_tokens: Option<Vec<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<i64>,
    /// v1.1 (additive): the number of committed tokens N the engine must free-run and
    /// return. Used only by `free_decode_run`. Appended last so pre-v1.1 request lines
    /// round-trip unchanged. (PROTOCOL-v1.1.md §2.1.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    /// Wire-additive (`docs/spec-config-design.md`): the per-module speculative configuration for
    /// the timed decode window. Carried on `decode_begin` (and its v1.1 `free_decode_begin`
    /// counterpart); absent ⇒ the engine default (a v1 engine that ignores it stays valid, but
    /// benchd's never-ignored check rejects a leg whose echo diverges). Appended last so pre-spec
    /// request lines round-trip unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec: Option<SpecConfig>,
    /// v1.2 (additive, COHORT): the per-stream seed token IDs for a batched `free_decode_begin` —
    /// `B` inner arrays, one per cohort slot, in SLOT ORDER. The single-stream v1.1 form keeps
    /// using [`seed_tokens`](Self::seed_tokens); a request carries one or the other, never both.
    /// Appended last so pre-v1.2 request lines round-trip unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_tokens_by_stream: Option<Vec<Vec<i64>>>,
    /// v1.2 (additive, COHORT): the EXPLICIT cohort width B, carried on both `free_decode_begin`
    /// and `free_decode_run`. B is never inferred from an array length alone — the engine must
    /// echo it back as [`WorkerResponse::effective_batch_size`] and benchd rejects a divergent
    /// echo fail-closed, so the cohort width is a never-ignored identity rather than a
    /// configuration hint. Presence of this field is what selects the cohort form of the verb.
    /// Appended last.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<u32>,
    /// (b) admission (engine #35, d528f6ac): the per-stream SEED token lists for a
    /// `cohort_reference_replay` — `B` inner arrays in SLOT ORDER. A DEDICATED field, distinct from
    /// the free-run [`seed_tokens_by_stream`](Self::seed_tokens_by_stream): the oracle replays
    /// teacher-forced and needs each stream's seed context. Rides ONLY on `cohort_reference_replay`.
    /// Appended last so pre-(b) request lines round-trip unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_seeds_by_stream: Option<Vec<Vec<i64>>>,
    /// (b) admission (engine #35): the candidate's COMMITTED token journals to replay teacher-forced —
    /// `B` inner arrays in SLOT ORDER, with `committed_by_stream.len() == replay_seeds_by_stream.len()`.
    /// Rides ONLY on `cohort_reference_replay`. Appended last.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub committed_by_stream: Option<Vec<Vec<i64>>>,
    /// (b) admission (engine #35): the oracle's logit top-K depth (engine default 16). AUDIT-only
    /// under (b) — benchd consumes only `sequential_argmax` — but carried so the wire matches the
    /// merged verb exactly. Appended last.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_top_k: Option<u32>,
    /// (b) admission (engine #35): the oracle's relative-envelope threshold (engine default 0.05).
    /// AUDIT-only under (b). Appended last.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rel_envelope: Option<f64>,
    /// (b) admission (engine #37): the reference's replay WIDTH for `cohort_reference_replay` —
    /// [`REPLAY_WIDTH_COHORT`] or `"canonical"`. Rides ONLY on `cohort_reference_replay`.
    /// Appended last.
    ///
    /// benchd sends this EXPLICITLY on every replay (never `None`): the oracle's geometry is the
    /// ENFORCED reference the tolerance gate judges against, so it must be PINNED by the request
    /// rather than inherited from whatever the engine happens to default to. An engine-side
    /// default is a value benchd neither states nor verifies, i.e. exactly the kind of implicit
    /// enforced parameter this codebase pins everywhere else.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_width: Option<String>,
}

impl WorkerRequest {
    /// A minimal request carrying only the required `id` and `kind`.
    pub fn new(id: i64, kind: impl Into<String>) -> Self {
        Self {
            id,
            kind: kind.into(),
            ..Self::default()
        }
    }
}

/// Engine -> benchmarker response.
///
/// Port of Swift `RuntimeWorkerResponse` (Sources/MLXFastHarness/QwenRuntimeWorker.swift).
/// Field order and JSON keys match the Swift `CodingKeys`. `id` and `ok` are always
/// present (`id` is `-1` for a response to an unparseable line). The remaining fields
/// are populated per kind (see PROTOCOL.md).
///
/// `protocol_version`, `backend`, and `device` are NEW in v1 (docs/architecture.md §2),
/// appended last so pre-Phase-0 messages round-trip unchanged. They are meaningful only
/// on the `hello` (id=0) message emitted after in-engine weight/config validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
// C5: CLOSED envelope (see WorkerRequest) — deny_unknown_fields matches the schema's
// additionalProperties:false and PROTOCOL.md; fail-closed against smuggled fields.
#[serde(deny_unknown_fields)]
pub struct WorkerResponse {
    /// Echoes the request id; `-1` when the request line could not be parsed.
    pub id: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nonce: Option<String>,
    /// Success flag. On `false` the session for that phase is discarded.
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logits: Option<Vec<CorrectnessTraceLogit>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_token: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<i64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_stats: Option<ExpertStreamingStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_ram_gb: Option<f64>,
    /// NEW in v1: engine's implemented protocol version. Meaningful only on the `hello`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_version: Option<u32>,
    /// NEW in v1: compute backend, e.g. `"mlx"` / `"cuda"`. Meaningful only on the `hello`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// NEW in v1: device identity, e.g. `"m5"` / `"gb10"`. Meaningful only on the `hello`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// NEW in v1: monotonic count of timed step-requests the engine has completed
    /// in the current phase. Reported on `phase_diagnostics`; the runner fails the
    /// run if it does not equal the number of timed steps issued (architecture §3,
    /// phase-close barrier). Meaningful only on `phase_diagnostics`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_work: Option<i64>,
    /// NEW in v1 (#54): the MLX allocator's free-buffer *cache* size in bytes at the
    /// phase boundary, mirroring Swift `Memory.cacheMemory` as read inside
    /// `resetRuntimeWorkerAllocatorForPhaseStart` (QwenRuntimeWorker.swift:99-115).
    /// Swift fails the run CLOSED unless it is exactly `0` after `Memory.clearCache()`,
    /// so a conformant engine reports `0` here; the parent asserts the drain on
    /// `phase_diagnostics`. Appended last so pre-#54 engines that omit it round-trip
    /// unchanged (back-compat: absent ⇒ not asserted; present ⇒ MUST be 0).
    /// Meaningful only on `phase_diagnostics`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_memory: Option<i64>,
    /// v1.1 (additive): capability flags the engine advertises. Meaningful only on the
    /// `hello`; carries `["free_run_decode"]` for an engine that implements the v1.1
    /// oracle-verified free-run timed decode mode ([`CAPABILITY_FREE_RUN_DECODE`]).
    /// Appended last, back-compat: a v1-only engine omits it (absent ⇒ no v1.1 capability).
    /// PROTOCOL-v1.1.md §2.1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<String>>,
    /// v1.1 (additive): per verify-round committed count from `free_decode_run`. Length is
    /// the round count R. AUDIT-only (never scored); persisted verbatim into the run's
    /// metrics and cross-checked by the §2.6 consistency TRIPLE. PROTOCOL-v1.1.md §3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptance_lengths: Option<Vec<u32>>,
    /// v1.1 (additive): total draft tokens proposed across all rounds (`>= accepted_total`;
    /// the round-ending base-model fallback position counts as drafted, RULED OQ5). AUDIT-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drafted_total: Option<u64>,
    /// v1.1 (additive): total drafts that passed internal verification and were committed.
    /// AUDIT-only, self-reported and distrusted for scoring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accepted_total: Option<u64>,
    /// v1.1 (additive): total committed tokens; MUST equal N and `tokens.len()`
    /// (PROTOCOL-v1.1.md §2.4). Externally anchored by benchd's exact-match verification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub committed_total: Option<u64>,
    /// Wire-additive (`docs/spec-config-design.md`): the module-parsed, default-filled speculative
    /// config the engine WILL ACTUALLY RUN, echoed on the `decode_begin` (or `free_decode_begin`)
    /// response. benchd seals ONLY this echo as the leg's effective spec and REJECTS a leg whose
    /// echo diverges from the request (never-ignored, fail-closed). Appended last, back-compat: a
    /// pre-spec engine omits it (absent ⇒ no echo; a spec'd request against such an engine rejects).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_spec: Option<SpecConfig>,
    /// Wire-additive (`docs/spec-config-design.md`): the RUNNABLE speculative modes the engine
    /// advertises, meaningful only on the `hello`. A stub module is visible in code but never listed
    /// here. Appended last; a pre-spec engine omits it (absent ⇒ only the default path is runnable).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_modes: Option<Vec<String>>,
    /// #106 (passthrough MODEL): the engine's loaded-head provenance (sha256 + byte size + shard
    /// file count), emitted on the `hello`. MODELED as an optional field so benchd stops rejecting
    /// the real engine line under `deny_unknown_fields`, keeping the CLOSED posture. Appended last;
    /// a pre-#106 engine omits it (absent ⇒ no head provenance echoed on the wire).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_provenance: Option<HeadProvenance>,
    /// #106 (passthrough MODEL): the MLX allocator's ACTIVE (in-use) memory in bytes, read PRE-DRAIN
    /// on `phase_diagnostics` (distinct from the existing POST-drain [`Self::cache_memory`], which a
    /// conformant engine reports as 0). AUDIT-only, never scored. Appended last; a pre-#106 engine
    /// omits it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mlx_active_memory_bytes: Option<u64>,
    /// #106 (passthrough MODEL): the MLX allocator's free-buffer CACHE memory in bytes, read
    /// PRE-DRAIN on `phase_diagnostics`. Distinct from the post-drain [`Self::cache_memory`] (which must be
    /// 0 after `Memory.clearCache()`); this is the cache size BEFORE the drain. AUDIT-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mlx_cache_memory_bytes: Option<u64>,
    /// #106 (passthrough MODEL): the MLX allocator's PEAK memory watermark in bytes, read PRE-DRAIN
    /// on `phase_diagnostics`. AUDIT-only, never scored. Appended last.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mlx_peak_memory_bytes: Option<u64>,
    /// #106 (passthrough MODEL): the ALWAYS-present top-logit margin (the gap between the top and
    /// second logit) on the `correctness_*` teacher-forced steps. AUDIT-only. Appended last; a
    /// pre-#106 engine omits it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logit_margin: Option<f64>,
    /// #106 (passthrough MODEL): the CONDITIONAL logit of the expected (teacher-forced) token on the
    /// `correctness_*` steps — present only when the engine has an expected token to report.
    /// AUDIT-only. Appended last.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_token_logit: Option<f64>,
    /// #106 (passthrough MODEL): the CONDITIONAL rank of the expected (teacher-forced) token in the
    /// engine's logit ordering on the `correctness_*` steps — present only alongside
    /// [`Self::expected_token_logit`]. AUDIT-only. Appended last.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_token_rank: Option<u32>,
    /// Decoder-neutral spec-decode ABSTRACTION (additive): the name of the speculative decoder that
    /// produced this round's data — e.g. `"mtp"` / `"dflash"` / `"dspark"`. Emitted on the
    /// `free_decode_run` response so a scored run's per-round data is self-describing about WHICH
    /// decoder generated it, independent of the request's `spec.mode`. AUDIT-only, never scored.
    /// Appended last, back-compat: an engine that does not set it round-trips fine
    /// (absent ⇒ `None`, no decoder identity echoed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_decoder: Option<String>,
    /// v1.2 (additive, COHORT): the largest cohort width B the engine can serve, advertised on the
    /// `hello`. Lets benchd refuse an over-wide cohort PRE-GPU (before the cool gate and before the
    /// clock) rather than discovering it inside a timed window. Absent ⇒ not advertised; benchd
    /// then relies on the capability flag alone and the engine's own refusal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_batch_size: Option<u32>,
    /// v1.2 (additive, COHORT): the B seed-forward token IDs from a batched `free_decode_begin`,
    /// one per cohort slot in SLOT ORDER. Each is oracle-checked against that slot's expected seed
    /// token. The single-stream v1.1 form keeps using [`seed_token`](Self::seed_token).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_token_by_stream: Option<Vec<i64>>,
    /// v1.2 (additive, COHORT): the cohort width the engine WILL ACTUALLY RUN, echoed on the
    /// `free_decode_begin` (and `free_decode_run`) response. NEVER-IGNORED, exactly like
    /// [`effective_spec`](Self::effective_spec): benchd compares it to the requested
    /// [`WorkerRequest::batch_size`] and DISCARDS the leg fail-closed on divergence, so a silently
    /// narrowed cohort can never be sealed as a B=8 measurement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_batch_size: Option<u32>,
    /// v1.2 (additive, COHORT): the committed token IDs from a batched `free_decode_run` — `B`
    /// inner arrays of `N` tokens each, in SLOT ORDER. Every token is exact-matched benchd-side
    /// against that slot's golden continuation. Replaces (never accompanies)
    /// [`tokens`](Self::tokens) in the cohort form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_by_stream: Option<Vec<Vec<i64>>>,
    /// v1.2 (additive, COHORT): each row's PRE-`min` natural accept-walk length per round —
    /// `B` inner arrays of `R` counts, in SLOT ORDER. AUDIT-ONLY, never scored.
    ///
    /// Why it exists: CBv2 commits ONE COMMON WIDTH per round, taken as the minimum across rows,
    /// so [`acceptance_lengths`](Self::acceptance_lengths) stays a single vector even at B > 1.
    /// That common-width commit is exactly what makes a single straggler throttle the whole
    /// cohort, and this field is what makes that throttling VISIBLE instead of silently folded
    /// into the committed width.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub natural_accepted_by_stream: Option<Vec<Vec<u32>>>,
    /// v1.2 (additive, COHORT): the round count R the engine ran. Redundant with
    /// `acceptance_lengths.len()` BY DESIGN — benchd cross-checks the two, so a response that
    /// disagrees with itself is refused rather than silently reconciled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rounds: Option<u32>,
    /// v1.2 (additive, COHORT): the number of streams still generating at each round, length R.
    /// Makes the fixed-cohort tail auditable: under the closed-cohort policy (identical per-stream
    /// budget N, no refill, no EOS exit) this must be non-increasing. AUDIT-only, never scored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_streams_by_round: Option<Vec<u32>>,
    /// v1.2 (additive, COHORT): a histogram of the engine's depth-clamp reasons over the window
    /// (e.g. `automatic_rectangular_limit`, `tail_depth`, `batch_gate`, `step_kv_headroom`,
    /// `rectangular_cache_unsupported`). AUDIT-only and never scored. Safe snake-case keys seal
    /// verbatim; other keys use a fixed engine-text-withheld marker.
    ///
    /// This is what makes "did it actually speculate?" a checkable question rather than a trusted
    /// claim: a cohort that reports zero draft rounds under a clamp reason is a legitimate engine
    /// outcome, and the report has to be able to say so. A `BTreeMap` so the key order — and hence
    /// the serialized bytes — is deterministic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_clamp_reasons: Option<std::collections::BTreeMap<String, u32>>,
    /// Per-stream timing instrumentation (per-stream-instrumentation-spec.md step 1, additive
    /// COHORT): per-slot monotonic nanoseconds from cohort-prefill start to that slot's seed
    /// commit, one per cohort slot in SLOT ORDER, on a batched `free_decode_begin` response.
    /// Gated by [`CAPABILITY_PER_STREAM_TIMING`] on the hello — benchd refuses to request
    /// per-stream scoring against an engine that does not advertise it. RAW engine clock reads
    /// only (no sums, ratios, or seconds conversions engine-side); UNTRUSTED for scoring until
    /// this crate's attestation module admits them (engine-reported-time-untrusted doctrine —
    /// parent-clock doctrine). Currently REPORT-ONLY: computed and sealed as a diagnostic
    /// verdict, never enforced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_ns_by_stream: Option<Vec<u64>>,
    /// Per-stream timing instrumentation (per-stream-instrumentation-spec.md step 1, additive
    /// COHORT): per-slot monotonic nanoseconds from decode-phase start to that slot's
    /// final-token commit, one per cohort slot in SLOT ORDER, on a batched `free_decode_run`
    /// response. Same capability gate and untrusted-until-attested posture as
    /// [`prefill_ns_by_stream`](Self::prefill_ns_by_stream).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_ns_by_stream: Option<Vec<u64>>,
    /// (b) admission (engine #35, d528f6ac): the TRUSTED oracle's `cohort_reference_replay` report —
    /// the per-stream reference argmax for each replayed position. Present ONLY on a
    /// `cohort_reference_replay` response; benchd applies the ≤10% per-stream token-tolerance gate over
    /// it ([`CohortReferenceReplayReport`]). Appended last so every other response round-trips
    /// unchanged (absent ⇒ not a replay response).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cohort_reference_replay: Option<CohortReferenceReplayReport>,
    /// Spec-decode AUDIT (additive): the number of verify rounds in which the engine's BATCHED
    /// verify forward and its ONE-ROW replay of the same position chose a DIFFERENT argmax,
    /// reported on the `free_decode_run` response.
    ///
    /// WHY AN ENGINE REPORTS IT. A depth-1 MTP cycle verifies the draft in a two-row forward and,
    /// on a REJECTING round, re-runs that position as a one-row forward to produce the committed
    /// token. The tower is not batch-invariant, so the two argmaxes can differ; the REPLAY stands
    /// (it is the row shape the serial path would have run), and the divergence is COUNTED rather
    /// than refused. This field is that count, so a scored artifact can state how often the two
    /// shapes disagreed instead of leaving it invisible.
    ///
    /// AUDIT-ONLY, never scored. OPTIONAL, appended last: an engine that does not report it
    /// round-trips unchanged (absent ⇒ NOT REPORTED, which is not the same claim as `0`).
    /// Only a REJECTING round can disagree, so a reported count is bounded by the rejected drafts
    /// (`drafted_total - accepted_total`); benchd refuses a leg that reports more
    /// (`bench_core::free_run::FreeRunConsistencyError::VerifyReplayDisagreementsExceedRejected`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verify_replay_disagreements: Option<u64>,
    /// Spec-decode AUDIT (additive, optional): HOW the engine verified the drafts on this
    /// `free_decode_run` — `"rectangular"` (one target forward over the whole 1+k candidate window,
    /// recurrent state captured per position and rolled back to the accepted one) or `"serial"`
    /// (one target forward per candidate token, the fallback oracle). AUDIT-only, never scored.
    /// Absent ⇒ NOT REPORTED (a pre-field engine). Sealed as `spec_verification_mode`.
    /// See [`VERIFICATION_MODES`] for the documented wire values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_mode: Option<String>,
    /// Spec-decode AUDIT (additive, optional): verify rounds that ran the rectangular path.
    /// AUDIT-only, never scored. Sealed as `spec_rectangular_verification_rounds`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rectangular_verification_rounds: Option<u64>,
    /// Spec-decode AUDIT (additive, optional): verify rounds that fell back to the serial oracle.
    /// AUDIT-only, never scored. Sealed as `spec_serial_verification_rounds`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serial_verification_rounds: Option<u64>,
    /// Runner contract §13 (additive): the RUNNER IDENTITY the worker serves, emitted on
    /// the `hello`. Names the runner, the loaded `model_type`, the digest of the runner's canonical
    /// manifest, and the build the worker was cut from ([`RunnerIdentity`]). MODELED exactly like
    /// [`head_provenance`](Self::head_provenance): optional, hello-only, AUDIT/provenance only, never
    /// a scoring input. Appended last, back-compat: a worker that omits it round-trips unchanged
    /// (absent ⇒ no runner identity echoed on the wire).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runner: Option<RunnerIdentity>,
    /// Resident-worker identity (additive), emitted on the `hello` ONLY: the process a worker
    /// ATTACHED to instead of loading its own weights ([`ResidentIdentity`]).
    ///
    /// WHO EMITS IT. The MLX `bench-worker resident` — the twin of `ds4-resident` — keeps one
    /// loaded process alive and lets each window's worker attach to it. Such a worker emits this
    /// object; a worker that loaded the weights itself omits it.
    ///
    /// WHAT IT SAYS. `pid` is the resident process's id and `load_epoch` a monotonic stamp taken
    /// when that process finished its load. The stamp is CONSTANT for the life of the process, so
    /// every phase of one window seals the same identity, and a CHANGED value is a reload that
    /// should not have happened.
    ///
    /// AUDIT-only, never scored. Appended last, back-compat: a worker that omits it round-trips
    /// unchanged (absent ⇒ the worker attached to no resident process).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resident: Option<ResidentIdentity>,
}

impl WorkerResponse {
    /// A minimal successful response carrying only the required `id` and `ok = true`.
    pub fn ok(id: i64) -> Self {
        Self {
            id,
            ok: true,
            ..Self::default()
        }
    }

    /// A failure response: `ok = false` with an `error` string.
    pub fn error(id: i64, message: impl Into<String>) -> Self {
        Self {
            id,
            ok: false,
            error: Some(message.into()),
            ..Self::default()
        }
    }
}

/// A single top-K logit entry.
///
/// Port of Swift `CorrectnessTraceLogit` (Sources/MLXFastHarness/QwenRuntime.swift).
/// The Swift struct has NO `CodingKeys`, so the JSON keys are literally `token` / `logit`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct CorrectnessTraceLogit {
    /// Token ID.
    pub token: i64,
    /// Logit value.
    pub logit: f64,
}

impl CorrectnessTraceLogit {
    pub fn new(token: i64, logit: f64) -> Self {
        Self { token, logit }
    }
}

/// (b) admission (engine #35, d528f6ac) — the TRUSTED oracle's `cohort_reference_replay` report.
///
/// The oracle re-decodes each cohort stream TEACHER-FORCED on the candidate's OWN committed tokens
/// over the ORGANIZER's pinned reference weights and reports, per position, the reference argmax
/// benchd compares the candidate's committed token against. Under (b) benchd consumes ONLY
/// [`CohortReferenceReplayPosition::committed_token`] (the N2 echo-integrity check) and
/// [`CohortReferenceReplayPosition::sequential_argmax`] (the reference argmax for the ≤10% gate); the
/// logit / gap / envelope fields ride along AUDIT-ONLY and are tolerated but unused. The verb renders
/// NO verdict — benchd applies the tolerance gate.
///
/// NOT `deny_unknown_fields`: the engine emits a richer payload than (b) reads, and an added audit
/// field must never fail the decode. The two load-bearing position fields ARE required.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CohortReferenceReplayReport {
    /// The logit provenance tag the oracle used (e.g. `"post_softcap"`). AUDIT-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_provenance: Option<String>,
    /// The logit top-K the oracle ran. AUDIT-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_topk: Option<u32>,
    /// The relative envelope the oracle ran. AUDIT-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rel_envelope: Option<f64>,
    /// The replay WIDTH the oracle actually ran at, ECHOED back — NOT audit-only: when present it
    /// is ASSERTED against the width the request pinned (`bench_runner`'s
    /// `Session::cohort_reference_replay`), fail-closed, exactly like `effective_batch_size`'s
    /// never-ignored echo on the free-run cohort.
    ///
    /// `None` against TODAY's engine, which parses `replay_width` on the request but does not
    /// stamp it on the report (verified against the engine's `CohortReferenceReplayReport`
    /// CodingKeys: `logit_provenance` / `logit_topk` / `rel_envelope` / `streams`). The assertion
    /// is therefore presence-conditioned and ARMS ITSELF the moment the engine-side stamp lands —
    /// no benchd change needed then. Until it does, the request-side pin is what makes the
    /// geometry explicit, and the engine refuses any width value it does not recognize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_width: Option<String>,
    /// The per-stream replay results, in SLOT ORDER.
    pub streams: Vec<CohortReferenceReplayStream>,
}

/// (b) admission — one cohort stream's replay results: its slot index plus per-position reference
/// data. See [`CohortReferenceReplayReport`]. NOT `deny_unknown_fields` (tolerate a richer payload).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CohortReferenceReplayStream {
    /// The cohort slot this stream occupies (SLOT ORDER identity).
    pub slot: i64,
    /// The per-position replay results for this stream.
    pub positions: Vec<CohortReferenceReplayPosition>,
}

/// (b) admission — one replayed position's reference data. `committed_token` + `sequential_argmax`
/// are REQUIRED (the two fields (b) consumes); everything else is AUDIT-ONLY and optional so a richer
/// engine payload decodes without failing. NOT `deny_unknown_fields`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CohortReferenceReplayPosition {
    /// The candidate's committed token the oracle replayed at this position, ECHOED back. benchd
    /// verifies this equals the candidate's own `tokens_by_stream` (byte/id) BEFORE counting
    /// mismatches (N2 integrity): a divergence means the oracle replayed a DIFFERENT journal than the
    /// candidate committed — a hard integrity error, never a tolerance decision.
    pub committed_token: i64,
    /// The reference argmax at this position (`== ranked_tokens[0]`) — THE reference token benchd
    /// compares the committed token against for the ≤10% per-stream tolerance gate.
    pub sequential_argmax: i64,
    /// AUDIT-ONLY (unused by (b)): the full ranked token list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ranked_tokens: Option<Vec<i64>>,
    /// AUDIT-ONLY: the ranked logits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ranked_logits: Option<Vec<f64>>,
    /// AUDIT-ONLY: the ranked relative gaps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ranked_relative_gaps: Option<Vec<f64>>,
    /// AUDIT-ONLY: the committed token's logit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub committed_token_logit: Option<f64>,
    /// AUDIT-ONLY: the committed token's relative gap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub committed_relative_gap: Option<f64>,
    /// AUDIT-ONLY: the within-envelope depth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within_envelope_depth: Option<i64>,
}

/// Expert-streaming counters reported alongside diagnostics.
///
/// Port of Swift `ExpertStreamingStats` (Sources/MLXFastCore/ExpertStreamingStats.swift).
/// JSON keys match the Swift `CodingKeys` (`expert_*`). The dense RAM-resident Qwen
/// runtime always reports the zero struct; the field shape is retained for schema
/// stability across submissions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ExpertStreamingStats {
    #[serde(rename = "expert_cache_hits")]
    pub cache_hits: u64,
    #[serde(rename = "expert_cache_misses")]
    pub cache_misses: u64,
    #[serde(rename = "expert_cache_evictions")]
    pub cache_evictions: u64,
    #[serde(rename = "expert_bytes_read")]
    pub bytes_read: u64,
    #[serde(rename = "expert_read_seconds")]
    pub read_seconds: f64,
    #[serde(rename = "expert_peak_cached_tensors")]
    pub peak_cached_tensors: u64,
}

impl ExpertStreamingStats {
    /// The all-zero stats reported by the dense RAM-resident runtime.
    /// Mirrors Swift `ExpertStreamingStats.zero`.
    pub fn zero() -> Self {
        Self::default()
    }
}

/// #106 (passthrough MODEL): the engine's loaded-head provenance, emitted on the `hello`
/// (`WorkerResponse::head_provenance`). The `sha256` identifies the head bytes the engine actually
/// loaded; `bytes` is the total on-disk size and `file_count` the number of shard files. All three
/// are REQUIRED inside the object (a `hello` that omits `head_provenance` entirely is still valid;
/// but a present object carries all three). benchd MODELS but does not score these — they are
/// AUDIT/provenance only, sealed as the engine ECHO of the head it loaded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct HeadProvenance {
    /// The sha256 of the head bytes the engine loaded.
    pub sha256: String,
    /// Total on-disk byte size of the head.
    pub bytes: u64,
    /// Number of shard files comprising the head.
    pub file_count: u32,
}

/// Runner contract §6.1 (additive): the runner identity a worker emits on the `hello`
/// (`WorkerResponse::runner`). `id` is the manifest's vendor-namespaced runner id, `model_type` the
/// `config.json` model type the worker actually loaded, `manifest_sha256` the digest of the runner's
/// CANONICAL manifest JSON (64 lowercase hex, `bench_core::runner_manifest`), and `build` the
/// worker build the identity was cut from. All four are REQUIRED inside the object: a `hello` that
/// omits `runner` entirely is valid, but a present object carries all four. The object is CLOSED —
/// an unknown key is a hard error. benchd MODELS but never scores these; they are AUDIT/provenance
/// only, and the conformance kit compares them against the declared manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RunnerIdentity {
    /// The manifest's vendor-namespaced runner id, e.g. `"layr/qwen4exp-125b-a6b"`.
    pub id: String,
    /// The `config.json` model type the worker loaded.
    pub model_type: String,
    /// Sha256 of the runner's canonical manifest JSON, 64 lowercase hex.
    pub manifest_sha256: String,
    /// The worker build the identity was cut from.
    pub build: String,
}

/// The resident-process identity a worker emits on the `hello` (`WorkerResponse::resident`) when it
/// ATTACHED to an already-loaded process instead of loading the weights itself. The MLX
/// `bench-worker resident` — the twin of `ds4-resident` — is that process.
///
/// `pid` is the resident process's id. `load_epoch` is a monotonic stamp taken when that process
/// finished its load; it is CONSTANT for the life of the process, so every phase of one window
/// seals the same identity and a CHANGED value is a reload that should not have happened. Both keys
/// are REQUIRED inside the object: a `hello` that omits `resident` entirely is valid, but a present
/// object carries both. The object is CLOSED — an unknown key is a hard error. benchd MODELS but
/// never scores these; they are AUDIT/provenance only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ResidentIdentity {
    /// The resident process's id.
    pub pid: u32,
    /// A monotonic stamp taken when the resident process finished its load; constant for the life
    /// of that process.
    pub load_epoch: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// David ruling 2026-08-27 — the engine-decides spec serializes with the mtp block present and
    /// the depth key ABSENT (`{"mode":"mtp","mtp":{}}`), which is exactly the shape the engine's
    /// module parser documents ("the mtp block has no depth key → the envelope resolves its own").
    #[test]
    fn engine_default_spec_serializes_without_a_depth_key() {
        let spec = SpecConfig::mtp_engine_default();
        assert_eq!(
            serde_json::to_string(&spec).unwrap(),
            r#"{"mode":"mtp","mtp":{}}"#
        );
        // ...and the explicit form is unchanged.
        assert_eq!(
            serde_json::to_string(&SpecConfig::mtp(2)).unwrap(),
            r#"{"mode":"mtp","mtp":{"depth":2}}"#
        );
    }

    /// The echo-honors-request rule: requested fields must come back identical; module-filled
    /// fields the request omitted are reported, not divergence.
    #[test]
    fn spec_echo_honors_request_semantics() {
        // Fully specified request: exact equality passes (the pre-ruling behavior, unchanged)...
        assert!(spec_echo_honors_request(
            &SpecConfig::mtp(2),
            &SpecConfig::mtp(2)
        ));
        // ...and a CHANGED requested value still rejects — a requested 4 echoed as 3 is divergence.
        assert!(!spec_echo_honors_request(
            &SpecConfig::mtp(4),
            &SpecConfig::mtp(3)
        ));
        // Engine-decides request: the module resolving a depth the request omitted is ACCEPTED.
        assert!(spec_echo_honors_request(
            &SpecConfig::mtp_engine_default(),
            &SpecConfig::mtp(3)
        ));
        // The mode is always load-bearing: a serial request echoed as mtp rejects.
        assert!(!spec_echo_honors_request(
            &SpecConfig::serial(),
            &SpecConfig::mtp(3)
        ));
        // A missing module block in the ECHO of an engine-decides request still rejects — the
        // engine must report what it resolved, not stay silent (`{"mtp":{}}` ⊄ no block).
        let mut bare = SpecConfig::mtp_engine_default();
        bare.mtp = None;
        assert!(!spec_echo_honors_request(
            &SpecConfig::mtp_engine_default(),
            &bare
        ));
        // DFlash: a requested depth honored while the engine adds its draft-identity block passes —
        // the added block is the module reporting, not divergence.
        let dflash_req = SpecConfig::dflash(1);
        let mut dflash_echo = SpecConfig::dflash(1);
        dflash_echo.dflash = Some(serde_json::json!({
            "depth": 1,
            "draft": { "artifact": "dflash-head", "sha256": "ab" }
        }));
        assert!(spec_echo_honors_request(&dflash_req, &dflash_echo));
        // ...but a changed dflash depth rejects.
        assert!(!spec_echo_honors_request(
            &SpecConfig::dflash(2),
            &dflash_echo
        ));
    }

    /// Review gate (48/aa item 3) — the module-block tolerance is scoped WITHIN the mode's block:
    /// an echo may not INTRODUCE a module block the request did not carry. Pre-ruling byte-equality
    /// refused a serial echo with a stray speculative block; the subset rule must not readmit it.
    #[test]
    fn stray_module_blocks_in_the_echo_refuse() {
        let mut serial_with_mtp = SpecConfig::serial();
        serial_with_mtp.mtp = Some(MtpSpec { depth: Some(2) });
        assert!(!spec_echo_honors_request(
            &SpecConfig::serial(),
            &serial_with_mtp
        ));
        let mut serial_with_dflash = SpecConfig::serial();
        serial_with_dflash.dflash = Some(serde_json::json!({ "depth": 1 }));
        assert!(!spec_echo_honors_request(
            &SpecConfig::serial(),
            &serial_with_dflash
        ));
        // An mtp echo sprouting a dflash block refuses the same way.
        let mut mtp_with_dflash = SpecConfig::mtp(2);
        mtp_with_dflash.dflash = Some(serde_json::json!({ "depth": 1 }));
        assert!(!spec_echo_honors_request(
            &SpecConfig::mtp(2),
            &mtp_with_dflash
        ));
    }

    /// Review gate (48/aa item 4) — the echo must REPORT a resolved depth. With the seal's
    /// `mtp_depth` absent-by-design on engine-decides runs, the echo is the only place the
    /// operating depth is recorded; an echo as silent as the request leaves the run with no depth
    /// on record anywhere, and refuses.
    #[test]
    fn engine_decides_echo_must_carry_the_resolved_depth() {
        assert!(!spec_echo_honors_request(
            &SpecConfig::mtp_engine_default(),
            &SpecConfig::mtp_engine_default()
        ));
        // A dflash echo without a numeric depth refuses for the same reason.
        let req = SpecConfig {
            mode: SPEC_MODE_DFLASH.to_string(),
            mtp: None,
            dflash: Some(serde_json::json!({})),
            dspark: None,
        };
        let mut echo_no_depth = req.clone();
        echo_no_depth.dflash = Some(serde_json::json!({
            "draft": { "artifact": "dflash-head", "sha256": "cd" }
        }));
        assert!(!spec_echo_honors_request(&req, &echo_no_depth));
    }

    /// Review gate (48/aa item 2) — the VERBATIM PRODUCTION REQUEST for the DFlash arm
    /// (`{"mode":"dflash","dflash":{}}`, the wrapper's engine-decides form) against an
    /// engine-shaped echo: resolved depth plus the draft identity block with a real 64-hex digest.
    /// This exact shape is what the byte-equality gate refused on every served dflash session; it
    /// must be exercised by a committed test, not only by specified-field variants.
    #[test]
    fn production_engine_decides_dflash_request_accepts_the_engine_echo() {
        let req: SpecConfig = serde_json::from_str(r#"{"mode":"dflash","dflash":{}}"#).unwrap();
        let echo: SpecConfig = serde_json::from_str(&format!(
            r#"{{"mode":"dflash","dflash":{{"depth":3,"draft":{{"artifact":"dflash-head","sha256":"{}"}}}}}}"#,
            "ab".repeat(32)
        ))
        .unwrap();
        assert!(spec_echo_honors_request(&req, &echo));
        // And the same echo with the depth dropped refuses (the item-4 rule on the real shape).
        let echo_no_depth: SpecConfig = serde_json::from_str(&format!(
            r#"{{"mode":"dflash","dflash":{{"draft":{{"artifact":"dflash-head","sha256":"{}"}}}}}}"#,
            "ab".repeat(32)
        ))
        .unwrap();
        assert!(!spec_echo_honors_request(&req, &echo_no_depth));
    }

    /// Review contract (48): a REQUEST-SPECIFIED drafter identity is verbatim-match-or-refuse.
    /// The module-filled tolerance covers only fields the request left ABSENT — a request that
    /// declares `draft.sha256` has specified it, and an echo carrying any other digest is a
    /// wrong-drafter divergence, discarded exactly like a changed depth. (The engine's own
    /// declared-vs-recomputed identity refusal — `RuntimeWorkerSpecConfig.swift`, "refusing rather
    /// than echoing a draft identity this run is not using" — sits in front of this and is
    /// untouched by this change; this is benchd's independent half of the same guarantee.)
    #[test]
    fn request_specified_drafter_identity_is_verbatim_or_refused() {
        let mut req = SpecConfig::dflash(1);
        req.dflash = Some(serde_json::json!({
            "depth": 1,
            "draft": { "sha256": "aa" }
        }));
        // Echo honors the declared digest (and fills artifact, a field the request omitted): OK.
        let mut echo_honored = SpecConfig::dflash(1);
        echo_honored.dflash = Some(serde_json::json!({
            "depth": 1,
            "draft": { "artifact": "dflash-head", "sha256": "aa" }
        }));
        assert!(spec_echo_honors_request(&req, &echo_honored));
        // A DIFFERENT digest in the echo is a wrong drafter: refused.
        let mut echo_wrong = echo_honored.clone();
        echo_wrong.dflash = Some(serde_json::json!({
            "depth": 1,
            "draft": { "artifact": "dflash-head", "sha256": "bb" }
        }));
        assert!(!spec_echo_honors_request(&req, &echo_wrong));
        // An echo that DROPS the declared digest entirely is equally a refusal — silence about a
        // specified field is not honoring it.
        let mut echo_silent = echo_honored.clone();
        echo_silent.dflash = Some(serde_json::json!({
            "depth": 1,
            "draft": { "artifact": "dflash-head" }
        }));
        assert!(!spec_echo_honors_request(&req, &echo_silent));
        // Wrong MODE stays refused whatever the blocks say — for the engine-decides request too.
        assert!(!spec_echo_honors_request(
            &SpecConfig::mtp_engine_default(),
            &SpecConfig::serial()
        ));
    }

    /// Parse a canonical NDJSON line, re-serialize, and assert byte-identical output.
    fn assert_request_roundtrip(canonical: &str) {
        let parsed: WorkerRequest = serde_json::from_str(canonical).unwrap();
        let reserialized = serde_json::to_string(&parsed).unwrap();
        assert_eq!(reserialized, canonical);
    }

    fn assert_response_roundtrip(canonical: &str) {
        let parsed: WorkerResponse = serde_json::from_str(canonical).unwrap();
        let reserialized = serde_json::to_string(&parsed).unwrap();
        assert_eq!(reserialized, canonical);
    }

    #[test]
    fn request_roundtrip_all_kinds() {
        // prefill: prompt_tokens
        assert_request_roundtrip(r#"{"id":1,"kind":"prefill","prompt_tokens":[10,20,30]}"#);
        // decode_begin: seed_tokens
        assert_request_roundtrip(r#"{"id":2,"kind":"decode_begin","seed_tokens":[1,2,3,4]}"#);
        // decode_step: token
        assert_request_roundtrip(r#"{"id":3,"kind":"decode_step","token":42}"#);
        // correctness: prompt_tokens + steps
        assert_request_roundtrip(
            r#"{"id":4,"kind":"correctness","prompt_tokens":[5,6],"steps":8}"#,
        );
        // correctness_begin: prompt_tokens
        assert_request_roundtrip(r#"{"id":5,"kind":"correctness_begin","prompt_tokens":[7,8,9]}"#);
        // correctness_step: token
        assert_request_roundtrip(r#"{"id":6,"kind":"correctness_step","token":99}"#);
        // phase_diagnostics: no extra fields
        assert_request_roundtrip(r#"{"id":7,"kind":"phase_diagnostics"}"#);
    }

    #[test]
    fn response_roundtrip_hello_with_phase0_fields() {
        // hello: id=0, nonce, ok, expert_stats (zero), plus the new v1 fields.
        let canonical = r#"{"id":0,"nonce":"abc123","ok":true,"expert_stats":{"expert_cache_hits":0,"expert_cache_misses":0,"expert_cache_evictions":0,"expert_bytes_read":0,"expert_read_seconds":0.0,"expert_peak_cached_tensors":0},"protocol_version":1,"backend":"mlx","device":"m5"}"#;
        assert_response_roundtrip(canonical);
    }

    #[test]
    fn response_roundtrip_prefill_token() {
        assert_response_roundtrip(r#"{"id":1,"nonce":"n","ok":true,"token":1234}"#);
    }

    #[test]
    fn response_roundtrip_decode_begin_seed_token() {
        assert_response_roundtrip(r#"{"id":2,"nonce":"n","ok":true,"seed_token":567}"#);
    }

    #[test]
    fn response_roundtrip_correctness_begin_full() {
        // correctness_begin: token, top_logits[8], expert_stats, peak_ram_gb.
        let canonical = r#"{"id":5,"nonce":"n","ok":true,"token":11,"top_logits":[{"token":11,"logit":9.5},{"token":12,"logit":8.25},{"token":13,"logit":7.0},{"token":14,"logit":6.5},{"token":15,"logit":5.0},{"token":16,"logit":4.5},{"token":17,"logit":3.0},{"token":18,"logit":2.5}],"expert_stats":{"expert_cache_hits":0,"expert_cache_misses":0,"expert_cache_evictions":0,"expert_bytes_read":0,"expert_read_seconds":0.0,"expert_peak_cached_tensors":0},"peak_ram_gb":18.5}"#;
        assert_response_roundtrip(canonical);
    }

    #[test]
    fn response_roundtrip_correctness_freerun() {
        assert_response_roundtrip(
            r#"{"id":4,"nonce":"n","ok":true,"tokens":[100,101,102],"expert_stats":{"expert_cache_hits":0,"expert_cache_misses":0,"expert_cache_evictions":0,"expert_bytes_read":0,"expert_read_seconds":0.0,"expert_peak_cached_tensors":0},"peak_ram_gb":18.0}"#,
        );
    }

    #[test]
    fn response_roundtrip_phase_diagnostics() {
        assert_response_roundtrip(
            r#"{"id":7,"nonce":"n","ok":true,"expert_stats":{"expert_cache_hits":3,"expert_cache_misses":1,"expert_cache_evictions":2,"expert_bytes_read":4096,"expert_read_seconds":1.5,"expert_peak_cached_tensors":7},"peak_ram_gb":20.25}"#,
        );
    }

    #[test]
    fn response_roundtrip_phase_diagnostics_with_completed_work() {
        // phase_diagnostics with the new completed_work counter appended last.
        assert_response_roundtrip(
            r#"{"id":7,"nonce":"n","ok":true,"expert_stats":{"expert_cache_hits":0,"expert_cache_misses":0,"expert_cache_evictions":0,"expert_bytes_read":0,"expert_read_seconds":0.0,"expert_peak_cached_tensors":0},"peak_ram_gb":20.25,"completed_work":16}"#,
        );
    }

    #[test]
    fn response_roundtrip_phase_diagnostics_with_cache_memory() {
        // #54: cache_memory (Swift Memory.cacheMemory) appended AFTER completed_work; a
        // conformant drained engine reports 0. Round-trips byte-identically.
        assert_response_roundtrip(
            r#"{"id":7,"nonce":"n","ok":true,"expert_stats":{"expert_cache_hits":0,"expert_cache_misses":0,"expert_cache_evictions":0,"expert_bytes_read":0,"expert_read_seconds":0.0,"expert_peak_cached_tensors":0},"peak_ram_gb":20.25,"completed_work":16,"cache_memory":0}"#,
        );
    }

    #[test]
    fn request_roundtrip_free_decode_kinds() {
        // v1.1 (additive): free_decode_begin reuses seed_tokens; free_decode_run carries count.
        assert_request_roundtrip(r#"{"id":8,"kind":"free_decode_begin","seed_tokens":[1,2,3]}"#);
        assert_request_roundtrip(r#"{"id":9,"kind":"free_decode_run","count":128}"#);
    }

    #[test]
    fn request_roundtrip_decode_begin_with_spec() {
        // Wire-additive: decode_begin carries the mtp spec (config nested under the mode key),
        // appended after the v1/v1.1 fields. Round-trips byte-identically.
        assert_request_roundtrip(
            r#"{"id":2,"kind":"decode_begin","seed_tokens":[1,2,3],"spec":{"mode":"mtp","mtp":{"depth":2}}}"#,
        );
        // serial spec carries no module block.
        assert_request_roundtrip(
            r#"{"id":2,"kind":"decode_begin","seed_tokens":[1,2,3],"spec":{"mode":"serial"}}"#,
        );
    }

    #[test]
    fn response_roundtrip_decode_begin_with_effective_spec() {
        assert_response_roundtrip(
            r#"{"id":2,"nonce":"n","ok":true,"seed_token":567,"effective_spec":{"mode":"mtp","mtp":{"depth":2}}}"#,
        );
    }

    #[test]
    fn response_roundtrip_hello_with_spec_modes() {
        assert_response_roundtrip(
            r#"{"id":0,"nonce":"n","ok":true,"protocol_version":1,"backend":"mock","device":"test","spec_modes":["serial","mtp"]}"#,
        );
    }

    #[test]
    fn spec_absent_deserializes_to_none_backcompat() {
        // A pre-spec line omits spec / effective_spec / spec_modes: all deserialize to None.
        let req: WorkerRequest =
            serde_json::from_str(r#"{"id":2,"kind":"decode_begin","seed_tokens":[1]}"#).unwrap();
        assert_eq!(req.spec, None);
        let resp: WorkerResponse = serde_json::from_str(r#"{"id":0,"ok":true}"#).unwrap();
        assert_eq!(resp.effective_spec, None);
        assert_eq!(resp.spec_modes, None);
        // And a minimal message never emits the spec keys.
        assert!(
            !serde_json::to_string(&WorkerRequest::new(1, "decode_begin"))
                .unwrap()
                .contains("spec")
        );
        let resp_json = serde_json::to_string(&WorkerResponse::ok(0)).unwrap();
        assert!(!resp_json.contains("effective_spec"));
        assert!(!resp_json.contains("spec_modes"));
    }

    #[test]
    fn spec_config_rejects_unknown_and_constructs() {
        // Closed spec envelope: an unknown key inside the spec is a hard error.
        let bad: std::result::Result<SpecConfig, _> =
            serde_json::from_str(r#"{"mode":"mtp","mtp":{"depth":2},"bogus":1}"#);
        assert!(bad.is_err(), "unknown spec key must be rejected");
        // Constructors round-trip to the canonical wire shape.
        assert_eq!(
            serde_json::to_string(&SpecConfig::serial()).unwrap(),
            r#"{"mode":"serial"}"#
        );
        assert_eq!(
            serde_json::to_string(&SpecConfig::mtp(4)).unwrap(),
            r#"{"mode":"mtp","mtp":{"depth":4}}"#
        );
        assert_eq!(
            serde_json::to_string(&SpecConfig::dflash(5)).unwrap(),
            r#"{"mode":"dflash","dflash":{"depth":5}}"#
        );
    }

    #[test]
    fn dflash_depth_projects_the_lever_out_of_the_opaque_block() {
        // The `dflash` block stays opaque, but benchd structurally decodes the ONE lever it seals.
        assert_eq!(SpecConfig::dflash(5).dflash_depth(), Some(5));
        // Decoded off a wire echo carrying EXTRA module keys — the projection ignores them (the
        // block is not re-schema'd), and still reads `depth`.
        let echo: SpecConfig =
            serde_json::from_str(r#"{"mode":"dflash","dflash":{"depth":7,"drafter":"z-lab"}}"#)
                .unwrap();
        assert_eq!(echo.dflash_depth(), Some(7));
        // A non-dflash spec has no dflash lever — never the mtp depth standing in for it.
        assert_eq!(SpecConfig::mtp(4).dflash_depth(), None);
        assert_eq!(SpecConfig::serial().dflash_depth(), None);
        // A dflash block without a numeric `depth` yields None (honest — never a fabricated depth).
        let no_depth: SpecConfig =
            serde_json::from_str(r#"{"mode":"dflash","dflash":{"drafter":"z-lab"}}"#).unwrap();
        assert_eq!(no_depth.dflash_depth(), None);
    }

    #[test]
    fn response_roundtrip_hello_with_capabilities() {
        // v1.1 (additive): capabilities appended AFTER the v1 phase-0 fields; a v1.1-capable
        // engine advertises ["free_run_decode"]. Round-trips byte-identically.
        let canonical = r#"{"id":0,"nonce":"abc123","ok":true,"expert_stats":{"expert_cache_hits":0,"expert_cache_misses":0,"expert_cache_evictions":0,"expert_bytes_read":0,"expert_read_seconds":0.0,"expert_peak_cached_tensors":0},"protocol_version":1,"backend":"cuda","device":"gb10","capabilities":["free_run_decode"]}"#;
        assert_response_roundtrip(canonical);
    }

    #[test]
    fn response_roundtrip_free_decode_run() {
        // v1.1 (additive): free_decode_run response — tokens[] plus the AUDIT counters,
        // appended after cache_memory. N=4 example (R=2 rounds, sum(acceptance)=4).
        assert_response_roundtrip(
            r#"{"id":9,"nonce":"n","ok":true,"tokens":[700,701,702,703],"acceptance_lengths":[3,1],"drafted_total":5,"accepted_total":2,"committed_total":4}"#,
        );
    }

    #[test]
    fn v1_1_fields_absent_deserialize_to_none_backcompat() {
        // A v1-only line omits every v1.1 field: request `count`, response capabilities and
        // the free_decode_run AUDIT counters all deserialize to None (wire-additive).
        let req: WorkerRequest = serde_json::from_str(r#"{"id":9,"kind":"prefill"}"#).unwrap();
        assert_eq!(req.count, None);
        let resp: WorkerResponse = serde_json::from_str(r#"{"id":0,"ok":true}"#).unwrap();
        assert_eq!(resp.capabilities, None);
        assert_eq!(resp.acceptance_lengths, None);
        assert_eq!(resp.drafted_total, None);
        assert_eq!(resp.accepted_total, None);
        assert_eq!(resp.committed_total, None);
        // And a minimal message never emits any v1.1 key.
        let req_json = serde_json::to_string(&WorkerRequest::new(1, "prefill")).unwrap();
        assert!(!req_json.contains("count"));
        let resp_json = serde_json::to_string(&WorkerResponse::ok(0)).unwrap();
        assert!(!resp_json.contains("capabilities"));
        assert!(!resp_json.contains("committed_total"));
    }

    /// The disagreement counter is WIRE-ADDITIVE. An engine that omits it — every engine built
    /// before the field existed — still conforms to the CLOSED envelope and parses to `None`, which
    /// means NOT REPORTED and never `0`; a minimal response emits no such key; and an engine that
    /// DOES send it is no longer rejected as an unknown field.
    #[test]
    fn verification_mode_fields_absent_deserialize_to_none_and_round_trip_when_present() {
        let older: WorkerResponse = serde_json::from_str(
            r#"{"id":4,"ok":true,"tokens":[1,2],"acceptance_lengths":[1,1],"drafted_total":2,"accepted_total":0,"committed_total":2}"#,
        )
        .unwrap();
        assert_eq!(older.verification_mode, None);
        assert_eq!(older.rectangular_verification_rounds, None);
        assert_eq!(older.serial_verification_rounds, None);
        assert!(!serde_json::to_string(&older)
            .unwrap()
            .contains("verification"));
        let reporting: WorkerResponse = serde_json::from_str(
            r#"{"id":4,"ok":true,"tokens":[1,2],"acceptance_lengths":[1,1],"drafted_total":2,"accepted_total":0,"committed_total":2,"verification_mode":"rectangular","rectangular_verification_rounds":2,"serial_verification_rounds":0}"#,
        )
        .unwrap();
        assert_eq!(reporting.verification_mode.as_deref(), Some("rectangular"));
        assert_eq!(reporting.rectangular_verification_rounds, Some(2));
        assert_eq!(reporting.serial_verification_rounds, Some(0));
    }

    #[test]
    fn verify_replay_disagreements_absent_deserializes_to_none_backcompat() {
        let older: WorkerResponse = serde_json::from_str(
            r#"{"id":4,"ok":true,"tokens":[1,2],"acceptance_lengths":[1,1],"drafted_total":2,"accepted_total":0,"committed_total":2}"#,
        )
        .unwrap();
        assert_eq!(older.verify_replay_disagreements, None);
        assert!(!serde_json::to_string(&WorkerResponse::ok(0))
            .unwrap()
            .contains("verify_replay_disagreements"));

        let reporting: WorkerResponse = serde_json::from_str(
            r#"{"id":4,"ok":true,"tokens":[1,2],"acceptance_lengths":[1,1],"drafted_total":2,"accepted_total":0,"committed_total":2,"verify_replay_disagreements":2}"#,
        )
        .expect("the reporting engine's line must decode under deny_unknown_fields");
        assert_eq!(reporting.verify_replay_disagreements, Some(2));
    }

    #[test]
    fn cache_memory_absent_deserializes_to_none_backcompat() {
        // Back-compat: a pre-#54 engine that omits cache_memory parses to None (the parent
        // then does NOT assert the drain), and a minimal response never emits the key.
        let resp: WorkerResponse =
            serde_json::from_str(r#"{"id":7,"ok":true,"completed_work":16}"#).unwrap();
        assert_eq!(resp.cache_memory, None);
        assert!(!serde_json::to_string(&WorkerResponse::ok(0))
            .unwrap()
            .contains("cache_memory"));
    }

    #[test]
    fn response_roundtrip_hello_with_head_provenance() {
        // #106: head_provenance appended AFTER spec_modes; round-trips byte-identically.
        assert_response_roundtrip(
            r#"{"id":0,"nonce":"n","ok":true,"protocol_version":1,"backend":"mlx","device":"m5","head_provenance":{"sha256":"abcd","bytes":1048576,"file_count":3}}"#,
        );
    }

    #[test]
    fn response_roundtrip_hello_with_runner_identity() {
        // Runner contract §13: `runner` is appended LAST and round-trips byte-identically on a
        // hello that carries it alongside the head provenance.
        assert_response_roundtrip(
            r#"{"id":0,"nonce":"n","ok":true,"protocol_version":1,"backend":"mlx","device":"m5","head_provenance":{"sha256":"abcd","bytes":1048576,"file_count":3},"runner":{"id":"layr/qwen4exp-125b-a6b","model_type":"qwen4_exp","manifest_sha256":"0000000000000000000000000000000000000000000000000000000000000000","build":"c4089870"}}"#,
        );
    }

    #[test]
    fn runner_identity_absent_deserializes_to_none_backcompat() {
        // A worker that omits `runner` round-trips unchanged, and a minimal message never emits
        // the key.
        let resp: WorkerResponse =
            serde_json::from_str(r#"{"id":0,"nonce":"n","ok":true,"protocol_version":1}"#).unwrap();
        assert_eq!(resp.runner, None);
        assert!(!serde_json::to_string(&WorkerResponse::ok(0))
            .unwrap()
            .contains("runner"));
        // The runner object stays CLOSED: an unknown key is a hard error.
        let bad: std::result::Result<RunnerIdentity, _> = serde_json::from_str(
            r#"{"id":"r","model_type":"m","manifest_sha256":"s","build":"b","bogus":1}"#,
        );
        assert!(bad.is_err(), "unknown runner key must be rejected");
        // ...and all four fields are required.
        let missing: std::result::Result<RunnerIdentity, _> =
            serde_json::from_str(r#"{"id":"r","model_type":"m","manifest_sha256":"s"}"#);
        assert!(missing.is_err(), "runner build is required");
    }

    #[test]
    fn response_roundtrip_hello_with_resident_identity() {
        // `resident` is appended LAST, after `runner`, and round-trips byte-identically on a hello
        // that carries it alongside the head provenance and the runner identity.
        assert_response_roundtrip(
            r#"{"id":0,"nonce":"n","ok":true,"protocol_version":1,"backend":"mlx","device":"m5","head_provenance":{"sha256":"abcd","bytes":1048576,"file_count":3},"runner":{"id":"layr/qwen4exp-125b-a6b","model_type":"qwen4_exp","manifest_sha256":"0000000000000000000000000000000000000000000000000000000000000000","build":"c4089870"},"resident":{"pid":4242,"load_epoch":1756944000}}"#,
        );
        // ...and a hello WITHOUT it round-trips byte-identically too (the attach is optional).
        assert_response_roundtrip(
            r#"{"id":0,"nonce":"n","ok":true,"protocol_version":1,"backend":"mlx","device":"m5"}"#,
        );
    }

    #[test]
    fn resident_identity_absent_deserializes_to_none_backcompat() {
        // A worker that loaded its own weights omits `resident`, and a minimal message never emits
        // the key.
        let resp: WorkerResponse =
            serde_json::from_str(r#"{"id":0,"nonce":"n","ok":true,"protocol_version":1}"#).unwrap();
        assert_eq!(resp.resident, None);
        assert!(!serde_json::to_string(&WorkerResponse::ok(0))
            .unwrap()
            .contains("resident"));
        // The resident object stays CLOSED: an unknown key is a hard error.
        let bad: std::result::Result<ResidentIdentity, _> =
            serde_json::from_str(r#"{"pid":1,"load_epoch":2,"bogus":3}"#);
        assert!(bad.is_err(), "unknown resident key must be rejected");
        // ...and both fields are required.
        let missing: std::result::Result<ResidentIdentity, _> =
            serde_json::from_str(r#"{"pid":1}"#);
        assert!(missing.is_err(), "resident load_epoch is required");
    }

    #[test]
    fn resident_on_a_non_hello_response_is_tolerated_like_runner() {
        // `resident` is HELLO-ONLY by contract, exactly like `runner`, and the wire is permissive
        // about it in exactly the same way: a non-hello response that carries the key still decodes
        // under `deny_unknown_fields`, because the field is declared on the one envelope. Nothing
        // in the protocol REFUSES it there; what makes it hello-only is that the session reads it
        // only off the hello (`bench_runner::Hello`), so a stray copy on a later response is
        // dropped rather than sealed. This test pins that shared posture: if a hello-only refusal
        // is ever added, `runner` and `resident` must gain it together.
        let runner_on_step: WorkerResponse = serde_json::from_str(
            r#"{"id":3,"nonce":"n","ok":true,"token":1,"runner":{"id":"r","model_type":"m","manifest_sha256":"0000000000000000000000000000000000000000000000000000000000000000","build":"b"}}"#,
        )
        .expect("runner on a non-hello response decodes");
        assert!(runner_on_step.runner.is_some());
        let resident_on_step: WorkerResponse = serde_json::from_str(
            r#"{"id":3,"nonce":"n","ok":true,"token":1,"resident":{"pid":4242,"load_epoch":7}}"#,
        )
        .expect("resident on a non-hello response decodes, mirroring runner");
        assert_eq!(
            resident_on_step.resident,
            Some(ResidentIdentity {
                pid: 4242,
                load_epoch: 7
            })
        );
    }

    #[test]
    fn schema_admits_resident_on_the_hello_and_keeps_the_envelope_closed() {
        // Schema validation of the additive field: the response envelope declares `resident` as a
        // `$ref` to a CLOSED `$defs.ResidentIdentity` requiring both keys, and the envelope itself
        // stays closed, so the addition admits the new hello without opening the wire.
        let schema: serde_json::Value = serde_json::from_str(JSON_SCHEMA).unwrap();
        let response = &schema["$defs"]["WorkerResponse"];
        assert_eq!(
            response["additionalProperties"],
            serde_json::Value::Bool(false),
            "the response envelope must stay closed"
        );
        assert_eq!(
            response["properties"]["resident"]["$ref"],
            "#/$defs/ResidentIdentity"
        );
        let resident = &schema["$defs"]["ResidentIdentity"];
        assert_eq!(
            resident["additionalProperties"],
            serde_json::Value::Bool(false)
        );
        assert_eq!(
            resident["required"],
            serde_json::json!(["pid", "load_epoch"])
        );
    }

    #[test]
    fn response_roundtrip_phase_diagnostics_with_mlx_memory() {
        // #106: the PRE-drain mlx_* memory ints appended after head_provenance; distinct from the
        // POST-drain cache_memory (0). Round-trips byte-identically.
        assert_response_roundtrip(
            r#"{"id":7,"nonce":"n","ok":true,"peak_ram_gb":20.25,"completed_work":16,"cache_memory":0,"mlx_active_memory_bytes":123,"mlx_cache_memory_bytes":456,"mlx_peak_memory_bytes":789}"#,
        );
    }

    #[test]
    fn response_roundtrip_correctness_step_with_logit_margin_fields() {
        // #106: top_logit_margin ALWAYS + the conditional expected_token_logit / expected_token_rank.
        assert_response_roundtrip(
            r#"{"id":6,"nonce":"n","ok":true,"token":11,"top_logit_margin":1.25,"expected_token_logit":8.5,"expected_token_rank":2}"#,
        );
        // top_logit_margin present, expected_* omitted (the unconditional-only shape).
        assert_response_roundtrip(
            r#"{"id":6,"nonce":"n","ok":true,"token":11,"top_logit_margin":1.25}"#,
        );
    }

    #[test]
    fn passthrough_106_fields_absent_deserialize_to_none_backcompat() {
        // A pre-#106 line omits every passthrough field: all deserialize to None.
        let resp: WorkerResponse = serde_json::from_str(r#"{"id":0,"ok":true}"#).unwrap();
        assert_eq!(resp.head_provenance, None);
        assert_eq!(resp.mlx_active_memory_bytes, None);
        assert_eq!(resp.mlx_cache_memory_bytes, None);
        assert_eq!(resp.mlx_peak_memory_bytes, None);
        assert_eq!(resp.top_logit_margin, None);
        assert_eq!(resp.expected_token_logit, None);
        assert_eq!(resp.expected_token_rank, None);
        // A minimal message never emits any passthrough key.
        let resp_json = serde_json::to_string(&WorkerResponse::ok(0)).unwrap();
        assert!(!resp_json.contains("head_provenance"));
        assert!(!resp_json.contains("mlx_active_memory_bytes"));
        assert!(!resp_json.contains("top_logit_margin"));
        // The head_provenance object stays CLOSED: an unknown key is a hard error.
        let bad: std::result::Result<HeadProvenance, _> =
            serde_json::from_str(r#"{"sha256":"h","bytes":1,"file_count":1,"bogus":1}"#);
        assert!(bad.is_err(), "unknown head_provenance key must be rejected");
    }

    #[test]
    fn response_roundtrip_free_run_with_spec_decoder() {
        // Decoder-neutral spec-decode abstraction: spec_decoder is appended LAST and round-trips
        // byte-identically on a free_decode_run response that names which decoder produced the round.
        assert_response_roundtrip(
            r#"{"id":5,"nonce":"n","ok":true,"tokens":[1,2,3,4],"acceptance_lengths":[3,1],"drafted_total":5,"accepted_total":2,"committed_total":4,"spec_decoder":"mtp"}"#,
        );
    }

    #[test]
    fn spec_decoder_absent_deserializes_to_none_backcompat() {
        // Additive + Option: an engine that does NOT set spec_decoder round-trips fine under the
        // CLOSED deny_unknown_fields envelope (absent ⇒ None), and a minimal response never emits it.
        let resp: WorkerResponse =
            serde_json::from_str(r#"{"id":5,"ok":true,"tokens":[1,2,3,4]}"#).unwrap();
        assert_eq!(resp.spec_decoder, None);
        // Present ⇒ carried through as the decoder identity.
        let set: WorkerResponse =
            serde_json::from_str(r#"{"id":5,"ok":true,"spec_decoder":"dflash"}"#).unwrap();
        assert_eq!(set.spec_decoder.as_deref(), Some("dflash"));
        // A minimal ok response never emits the key.
        assert!(!serde_json::to_string(&WorkerResponse::ok(0))
            .unwrap()
            .contains("spec_decoder"));
    }

    #[test]
    fn request_roundtrip_v1_2_cohort_kinds() {
        // v1.2 (additive, COHORT): the SAME two verbs, now carrying seed_tokens_by_stream +
        // batch_size (begin) and count + batch_size (run). Appended last, so both round-trip
        // byte-identically after the v1/v1.1/spec fields.
        assert_request_roundtrip(
            r#"{"id":8,"kind":"free_decode_begin","spec":{"mode":"mtp","mtp":{"depth":1}},"seed_tokens_by_stream":[[1,2],[3,4]],"batch_size":2}"#,
        );
        assert_request_roundtrip(r#"{"id":9,"kind":"free_decode_run","count":128,"batch_size":8}"#);
    }

    #[test]
    fn response_roundtrip_v1_2_cohort_begin_and_run() {
        // hello: the cohort capability + max_batch_size.
        assert_response_roundtrip(
            r#"{"id":0,"nonce":"n","ok":true,"protocol_version":1,"backend":"mlx","device":"m5","capabilities":["free_run_decode","batched_free_run_decode"],"max_batch_size":8}"#,
        );
        // free_decode_begin (cohort): B seed tokens + the never-ignored effective_batch_size echo
        // + (per-stream timing instrumentation) the per-slot cohort-prefill elapsed-ns vector.
        assert_response_roundtrip(
            r#"{"id":8,"nonce":"n","ok":true,"effective_spec":{"mode":"mtp","mtp":{"depth":1}},"seed_token_by_stream":[10,11],"effective_batch_size":2,"prefill_ns_by_stream":[50000,70000]}"#,
        );
        // free_decode_run (cohort): B x N tokens_by_stream, the SINGLE common-width
        // acceptance_lengths vector, cohort-sum totals, the audit-only cohort vectors, and
        // (per-stream timing instrumentation) the per-slot decode-phase elapsed-ns vector.
        assert_response_roundtrip(
            r#"{"id":9,"nonce":"n","ok":true,"acceptance_lengths":[2,2],"drafted_total":6,"accepted_total":2,"committed_total":8,"spec_decoder":"mtp","effective_batch_size":2,"tokens_by_stream":[[700,701,702,703],[800,801,802,803]],"natural_accepted_by_stream":[[2,3],[2,2]],"rounds":2,"active_streams_by_round":[2,2],"depth_clamp_reasons":{"automatic_rectangular_limit":1,"tail_depth":3},"decode_ns_by_stream":[320000,340000]}"#,
        );
    }

    #[test]
    fn depth_clamp_reasons_key_order_is_deterministic() {
        // A BTreeMap, so the serialized key order is sorted regardless of the order the engine
        // wrote them — the byte-round-trip fidelity contract survives an arbitrary input order.
        let resp: WorkerResponse = serde_json::from_str(
            r#"{"id":9,"ok":true,"depth_clamp_reasons":{"tail_depth":3,"batch_gate":1,"automatic_rectangular_limit":2}}"#,
        )
        .unwrap();
        assert_eq!(
            serde_json::to_string(&resp).unwrap(),
            r#"{"id":9,"ok":true,"depth_clamp_reasons":{"automatic_rectangular_limit":2,"batch_gate":1,"tail_depth":3}}"#
        );
    }

    #[test]
    fn v1_2_fields_absent_deserialize_to_none_backcompat() {
        // A v1.1 line omits every v1.2 cohort field: all deserialize to None, so a single-stream
        // engine keeps round-tripping unchanged under the CLOSED envelope.
        let req: WorkerRequest =
            serde_json::from_str(r#"{"id":9,"kind":"free_decode_run","count":8}"#).unwrap();
        assert_eq!(req.seed_tokens_by_stream, None);
        assert_eq!(req.batch_size, None);
        let resp: WorkerResponse =
            serde_json::from_str(r#"{"id":9,"ok":true,"tokens":[1,2]}"#).unwrap();
        assert_eq!(resp.max_batch_size, None);
        assert_eq!(resp.seed_token_by_stream, None);
        assert_eq!(resp.effective_batch_size, None);
        assert_eq!(resp.tokens_by_stream, None);
        assert_eq!(resp.natural_accepted_by_stream, None);
        assert_eq!(resp.rounds, None);
        assert_eq!(resp.active_streams_by_round, None);
        assert_eq!(resp.depth_clamp_reasons, None);
        assert_eq!(resp.prefill_ns_by_stream, None);
        assert_eq!(resp.decode_ns_by_stream, None);
        // A minimal message never emits any v1.2 key.
        let req_json = serde_json::to_string(&WorkerRequest::new(1, "free_decode_run")).unwrap();
        assert!(!req_json.contains("batch_size"));
        assert!(!req_json.contains("seed_tokens_by_stream"));
        let resp_json = serde_json::to_string(&WorkerResponse::ok(0)).unwrap();
        assert!(!resp_json.contains("batch_size"));
        assert!(!resp_json.contains("tokens_by_stream"));
        assert!(!resp_json.contains("depth_clamp_reasons"));
        assert!(!resp_json.contains("prefill_ns_by_stream"));
        assert!(!resp_json.contains("decode_ns_by_stream"));
    }

    #[test]
    fn cohort_reference_replay_request_roundtrips_and_kind_maps() {
        // (b) admission — the trusted oracle request kind maps to its exact wire string both ways,
        // and is NOT a completed_work timed step (it is an untimed post-run oracle).
        assert_eq!(
            RequestKind::CohortReferenceReplay.as_str(),
            "cohort_reference_replay"
        );
        assert_eq!(
            RequestKind::from_wire("cohort_reference_replay"),
            Some(RequestKind::CohortReferenceReplay)
        );
        assert!(!RequestKind::CohortReferenceReplay.is_timed_step());

        // The dedicated replay fields round-trip under the CLOSED request envelope.
        let mut req = WorkerRequest::new(7, RequestKind::CohortReferenceReplay.as_str());
        req.replay_seeds_by_stream = Some(vec![vec![1, 2], vec![3, 4]]);
        req.committed_by_stream = Some(vec![vec![10, 11], vec![12, 13]]);
        let line = serde_json::to_string(&req).unwrap();
        let back: WorkerRequest = serde_json::from_str(&line).unwrap();
        assert_eq!(back, req);
        assert!(line.contains("\"replay_seeds_by_stream\""));
        assert!(line.contains("\"committed_by_stream\""));
    }

    #[test]
    fn replay_width_rides_the_request_as_cohort_and_is_absent_by_default() {
        // (b) admission (engine #37) — the ENFORCED reference geometry is pinned BY THE REQUEST.
        // The exact wire spelling matters: the engine refuses any value it does not recognize.
        assert_eq!(REPLAY_WIDTH_COHORT, "cohort");
        let mut req = WorkerRequest::new(9, RequestKind::CohortReferenceReplay.as_str());
        req.replay_seeds_by_stream = Some(vec![vec![1, 2]]);
        req.committed_by_stream = Some(vec![vec![10, 11]]);
        req.replay_width = Some(REPLAY_WIDTH_COHORT.to_string());
        let line = serde_json::to_string(&req).unwrap();
        assert!(
            line.contains(r#""replay_width":"cohort""#),
            "the request must carry the width explicitly: {line}"
        );
        assert_eq!(
            serde_json::from_str::<WorkerRequest>(&line).unwrap(),
            req,
            "replay_width round-trips under the CLOSED request envelope"
        );

        // BACK-COMPAT both ways: absent on every other verb (omitted, never null), and a request
        // line WITHOUT the field still decodes — a pre-#37 recording is not invalidated.
        let plain = WorkerRequest::new(1, RequestKind::FreeDecodeRun.as_str());
        assert!(!serde_json::to_string(&plain)
            .unwrap()
            .contains("replay_width"));
        let old_line = r#"{"id":7,"kind":"cohort_reference_replay","replay_seeds_by_stream":[[1]],"committed_by_stream":[[2]]}"#;
        assert_eq!(
            serde_json::from_str::<WorkerRequest>(old_line)
                .unwrap()
                .replay_width,
            None
        );
    }

    #[test]
    fn replay_report_width_echo_is_optional_and_decodes_when_present() {
        // TODAY's engine stamps no width on the report, so the field decodes as `None` and the
        // benchd-side echo assertion stays dormant (see `Session::cohort_reference_replay`).
        let today = r#"{"id":3,"ok":true,"cohort_reference_replay":{"streams":[]}}"#;
        let resp: WorkerResponse = serde_json::from_str(today).unwrap();
        assert_eq!(
            resp.cohort_reference_replay.unwrap().replay_width,
            None,
            "the current engine emits no width stamp"
        );
        // When the engine-side stamp lands, it decodes — and that is what arms the assertion.
        let stamped = r#"{"id":3,"ok":true,"cohort_reference_replay":{"replay_width":"cohort","streams":[]}}"#;
        let resp: WorkerResponse = serde_json::from_str(stamped).unwrap();
        assert_eq!(
            resp.cohort_reference_replay
                .unwrap()
                .replay_width
                .as_deref(),
            Some(REPLAY_WIDTH_COHORT)
        );
    }

    #[test]
    fn cohort_reference_replay_report_decodes_required_fields_and_tolerates_extras() {
        // The engine emits a richer payload than (b) reads; the position struct is NOT
        // deny_unknown_fields, so the two REQUIRED fields (committed_token + sequential_argmax)
        // decode while every audit field (present or absent) is tolerated.
        let canonical = r#"{"id":3,"nonce":"n","ok":true,"cohort_reference_replay":{"logit_provenance":"post_softcap","logit_topk":16,"rel_envelope":0.05,"streams":[{"slot":0,"positions":[{"committed_token":700,"sequential_argmax":700,"ranked_tokens":[700,42],"ranked_logits":[1.0,0.5],"ranked_relative_gaps":[0.0,0.5],"committed_token_logit":1.0,"committed_relative_gap":0.0,"within_envelope_depth":1}]}]}}"#;
        let resp: WorkerResponse = serde_json::from_str(canonical).unwrap();
        let report = resp.cohort_reference_replay.as_ref().unwrap();
        assert_eq!(report.streams.len(), 1);
        assert_eq!(report.streams[0].slot, 0);
        let pos = &report.streams[0].positions[0];
        assert_eq!(pos.committed_token, 700);
        assert_eq!(pos.sequential_argmax, 700);

        // A MINIMAL payload carrying only the two required fields also decodes (audit fields None).
        let minimal = r#"{"id":3,"ok":true,"cohort_reference_replay":{"streams":[{"slot":0,"positions":[{"committed_token":5,"sequential_argmax":6}]}]}}"#;
        let resp: WorkerResponse = serde_json::from_str(minimal).unwrap();
        let pos = &resp.cohort_reference_replay.as_ref().unwrap().streams[0].positions[0];
        assert_eq!((pos.committed_token, pos.sequential_argmax), (5, 6));
        assert_eq!(pos.ranked_tokens, None);
    }

    #[test]
    fn cohort_reference_replay_capability_constant_is_distinct() {
        // (b) admission — the trusted-oracle capability is its own flag, distinct from every
        // free-run capability; only the trusted build advertises it (N1, wire-level half).
        assert_eq!(
            CAPABILITY_COHORT_REFERENCE_REPLAY,
            "cohort_reference_replay"
        );
        assert_ne!(
            CAPABILITY_COHORT_REFERENCE_REPLAY,
            CAPABILITY_BATCHED_FREE_RUN_DECODE
        );
        assert_ne!(
            CAPABILITY_COHORT_REFERENCE_REPLAY,
            CAPABILITY_FREE_RUN_DECODE
        );
        assert_ne!(
            CAPABILITY_COHORT_REFERENCE_REPLAY,
            CAPABILITY_PER_STREAM_TIMING
        );
    }

    #[test]
    fn per_stream_timing_capability_constant_is_distinct_from_the_batched_form() {
        // The per-stream timing surface is gated by its OWN capability: a v1.2 engine that
        // advertises only batched_free_run_decode must not be treated as per-stream-timing
        // capable — this is the wire-level half of "advertise-before-use".
        assert_eq!(CAPABILITY_PER_STREAM_TIMING, "per_stream_timing");
        assert_ne!(
            CAPABILITY_PER_STREAM_TIMING,
            CAPABILITY_BATCHED_FREE_RUN_DECODE
        );
        assert_ne!(CAPABILITY_PER_STREAM_TIMING, CAPABILITY_FREE_RUN_DECODE);
    }

    #[test]
    fn cohort_capability_constant_is_distinct_from_v1_1() {
        // The cohort form is gated by its OWN capability: a v1.1 engine that advertises only
        // `free_run_decode` must not be treated as batch-capable.
        assert_eq!(
            CAPABILITY_BATCHED_FREE_RUN_DECODE,
            "batched_free_run_decode"
        );
        assert_ne!(
            CAPABILITY_BATCHED_FREE_RUN_DECODE,
            CAPABILITY_FREE_RUN_DECODE
        );
    }

    #[test]
    fn v1_2_adds_no_verb_family() {
        // D6 alternative (b) — a wholly new `batch_decode_*` kind family — is REJECTED: the batched
        // path IS the general path. Lock that: the kind set is unchanged by v1.2, and the two
        // free-run kinds stay OUT of the completed_work timed-step set (a round is one engine
        // forward regardless of B, so completed_work is the SCALAR R+1).
        assert_eq!(RequestKind::from_wire("batched_free_decode_begin"), None);
        assert_eq!(RequestKind::from_wire("batch_decode_begin"), None);
        assert_eq!(ALL_REQUEST_KINDS.len(), 9);
        assert!(!RequestKind::FreeDecodeBegin.is_timed_step());
        assert!(!RequestKind::FreeDecodeRun.is_timed_step());
    }

    #[test]
    fn response_roundtrip_error_id_minus_one() {
        // Unparseable-line error path: id=-1, ok=false, error string.
        assert_response_roundtrip(
            r#"{"id":-1,"nonce":"n","ok":false,"error":"runtime worker protocol line was not valid JSON"}"#,
        );
    }

    #[test]
    fn nil_optionals_are_omitted() {
        // A minimal request must serialize with no optional keys at all.
        let req = WorkerRequest::new(1, "phase_diagnostics");
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"id":1,"kind":"phase_diagnostics"}"#);
        assert!(!json.contains("prompt_tokens"));
        assert!(!json.contains("token"));
        assert!(!json.contains("seed_tokens"));
        assert!(!json.contains("steps"));

        // A minimal ok response has only id + ok.
        let resp = WorkerResponse::ok(0);
        let json = serde_json::to_string(&resp).unwrap();
        assert_eq!(json, r#"{"id":0,"ok":true}"#);
        assert!(!json.contains("nonce"));
        assert!(!json.contains("protocol_version"));
    }

    #[test]
    fn missing_optional_keys_deserialize_to_none() {
        let req: WorkerRequest = serde_json::from_str(r#"{"id":9,"kind":"prefill"}"#).unwrap();
        assert_eq!(req.prompt_tokens, None);
        assert_eq!(req.token, None);
        assert_eq!(req.seed_tokens, None);
        assert_eq!(req.steps, None);

        let resp: WorkerResponse = serde_json::from_str(r#"{"id":0,"ok":true}"#).unwrap();
        assert_eq!(resp.nonce, None);
        assert_eq!(resp.error, None);
        assert_eq!(resp.token, None);
        assert_eq!(resp.top_logits, None);
        assert_eq!(resp.seed_token, None);
        assert_eq!(resp.tokens, None);
        assert_eq!(resp.expert_stats, None);
        assert_eq!(resp.peak_ram_gb, None);
        assert_eq!(resp.protocol_version, None);
        assert_eq!(resp.backend, None);
        assert_eq!(resp.device, None);
        assert_eq!(resp.completed_work, None);
    }

    #[test]
    fn unknown_keys_are_rejected() {
        // C5: CLOSED envelope — an unknown key is a hard error, not silently ignored.
        // Fail-closed against engine-side smuggled fields (matches the schema's
        // additionalProperties:false and PROTOCOL.md).
        let req: std::result::Result<WorkerRequest, _> =
            serde_json::from_str(r#"{"id":1,"kind":"prefill","future_field":true}"#);
        assert!(req.is_err(), "unknown request key must be rejected");
        let resp: std::result::Result<WorkerResponse, _> =
            serde_json::from_str(r#"{"id":0,"ok":true,"smuggled":123}"#);
        assert!(resp.is_err(), "unknown response key must be rejected");
    }

    #[test]
    fn embedded_schema_is_valid_json() {
        let value: serde_json::Value = serde_json::from_str(JSON_SCHEMA).unwrap();
        assert!(value.is_object());
        assert_eq!(
            value["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
    }

    #[test]
    fn protocol_version_constant() {
        assert_eq!(PROTOCOL_VERSION, 1);
    }

    /// Every request kind: the exact wire string and an `as_str` -> `from_wire` round-trip.
    const ALL_REQUEST_KINDS: [RequestKind; 9] = [
        RequestKind::Prefill,
        RequestKind::DecodeBegin,
        RequestKind::DecodeStep,
        RequestKind::Correctness,
        RequestKind::CorrectnessBegin,
        RequestKind::CorrectnessStep,
        RequestKind::PhaseDiagnostics,
        RequestKind::FreeDecodeBegin,
        RequestKind::FreeDecodeRun,
    ];

    #[test]
    fn request_kind_as_str_exact_wire_strings() {
        assert_eq!(RequestKind::Prefill.as_str(), "prefill");
        assert_eq!(RequestKind::DecodeBegin.as_str(), "decode_begin");
        assert_eq!(RequestKind::DecodeStep.as_str(), "decode_step");
        assert_eq!(RequestKind::Correctness.as_str(), "correctness");
        assert_eq!(RequestKind::CorrectnessBegin.as_str(), "correctness_begin");
        assert_eq!(RequestKind::CorrectnessStep.as_str(), "correctness_step");
        assert_eq!(RequestKind::PhaseDiagnostics.as_str(), "phase_diagnostics");
        assert_eq!(RequestKind::FreeDecodeBegin.as_str(), "free_decode_begin");
        assert_eq!(RequestKind::FreeDecodeRun.as_str(), "free_decode_run");
    }

    #[test]
    fn request_kind_as_str_from_wire_roundtrip() {
        for kind in ALL_REQUEST_KINDS {
            assert_eq!(
                RequestKind::from_wire(kind.as_str()),
                Some(kind),
                "round-trip failed for {:?}",
                kind
            );
        }
    }

    #[test]
    fn request_kind_from_wire_rejects_non_request_kinds() {
        // `hello` is the response-only id=0 kind — not a request kind.
        assert_eq!(RequestKind::from_wire("hello"), None);
        assert_eq!(RequestKind::from_wire("bogus"), None);
        assert_eq!(RequestKind::from_wire(""), None);
        assert_eq!(RequestKind::from_wire("Prefill"), None); // case-sensitive
    }

    #[test]
    fn request_kind_is_timed_step_locks_the_set() {
        // The single timed-step definition (architecture §3): exactly these four.
        assert!(RequestKind::DecodeBegin.is_timed_step());
        assert!(RequestKind::DecodeStep.is_timed_step());
        assert!(RequestKind::CorrectnessBegin.is_timed_step());
        assert!(RequestKind::CorrectnessStep.is_timed_step());
        // Setup / free-run / barrier kinds are NOT timed steps.
        assert!(!RequestKind::Prefill.is_timed_step());
        assert!(!RequestKind::Correctness.is_timed_step());
        assert!(!RequestKind::PhaseDiagnostics.is_timed_step());
        // v1.1 free-run kinds are NOT `completed_work` timed steps (Amendment 4): the
        // free-run phase's counter is validated by the §2.6 triple, not by this set.
        assert!(!RequestKind::FreeDecodeBegin.is_timed_step());
        assert!(!RequestKind::FreeDecodeRun.is_timed_step());
    }

    /// S4: schema/struct drift guard. Serialize a FULLY-populated instance of each wire
    /// type (every field present) and assert its JSON key set exactly equals the schema
    /// `$defs` `properties` set — both directions, so adding a struct field without the
    /// schema (or vice versa) fails CI.
    #[test]
    fn schema_properties_match_struct_fields() {
        use std::collections::BTreeSet;
        let schema: serde_json::Value = serde_json::from_str(JSON_SCHEMA).unwrap();
        let defs = &schema["$defs"];
        let schema_props = |name: &str| -> BTreeSet<String> {
            defs[name]["properties"]
                .as_object()
                .unwrap_or_else(|| panic!("schema $defs.{name}.properties missing"))
                .keys()
                .cloned()
                .collect()
        };
        let json_keys = |v: &serde_json::Value| -> BTreeSet<String> {
            v.as_object().unwrap().keys().cloned().collect()
        };

        let req = WorkerRequest {
            id: 1,
            kind: "x".into(),
            prompt_tokens: Some(vec![1]),
            token: Some(1),
            seed_tokens: Some(vec![1]),
            steps: Some(1),
            count: Some(1),
            spec: Some(SpecConfig::mtp(2)),
            seed_tokens_by_stream: Some(vec![vec![1]]),
            batch_size: Some(1),
            replay_seeds_by_stream: Some(vec![vec![1]]),
            committed_by_stream: Some(vec![vec![1]]),
            logit_top_k: Some(16),
            rel_envelope: Some(0.05),
            replay_width: Some(REPLAY_WIDTH_COHORT.into()),
        };
        assert_eq!(
            json_keys(&serde_json::to_value(&req).unwrap()),
            schema_props("WorkerRequest"),
            "WorkerRequest struct/schema field drift"
        );

        let resp = WorkerResponse {
            id: 1,
            nonce: Some("n".into()),
            ok: true,
            error: Some("e".into()),
            token: Some(1),
            top_logits: Some(vec![CorrectnessTraceLogit::new(1, 1.0)]),
            seed_token: Some(1),
            tokens: Some(vec![1]),
            expert_stats: Some(ExpertStreamingStats::zero()),
            peak_ram_gb: Some(1.0),
            protocol_version: Some(1),
            backend: Some("b".into()),
            device: Some("d".into()),
            completed_work: Some(1),
            cache_memory: Some(0),
            capabilities: Some(vec!["free_run_decode".into()]),
            acceptance_lengths: Some(vec![1]),
            drafted_total: Some(1),
            accepted_total: Some(1),
            committed_total: Some(1),
            effective_spec: Some(SpecConfig::serial()),
            spec_modes: Some(vec!["serial".into()]),
            head_provenance: Some(HeadProvenance {
                sha256: "h".into(),
                bytes: 1,
                file_count: 1,
            }),
            mlx_active_memory_bytes: Some(1),
            mlx_cache_memory_bytes: Some(1),
            mlx_peak_memory_bytes: Some(1),
            top_logit_margin: Some(1.0),
            expected_token_logit: Some(1.0),
            expected_token_rank: Some(1),
            spec_decoder: Some("mtp".into()),
            max_batch_size: Some(8),
            seed_token_by_stream: Some(vec![1]),
            effective_batch_size: Some(1),
            tokens_by_stream: Some(vec![vec![1]]),
            natural_accepted_by_stream: Some(vec![vec![1]]),
            rounds: Some(1),
            active_streams_by_round: Some(vec![1]),
            depth_clamp_reasons: Some([("tail_depth".to_string(), 1u32)].into_iter().collect()),
            prefill_ns_by_stream: Some(vec![1]),
            decode_ns_by_stream: Some(vec![1]),
            cohort_reference_replay: Some(CohortReferenceReplayReport {
                logit_provenance: Some("post_softcap".into()),
                logit_topk: Some(16),
                rel_envelope: Some(0.05),
                replay_width: Some(REPLAY_WIDTH_COHORT.into()),
                streams: vec![CohortReferenceReplayStream {
                    slot: 0,
                    positions: vec![CohortReferenceReplayPosition {
                        committed_token: 1,
                        sequential_argmax: 1,
                        ..Default::default()
                    }],
                }],
            }),
            verify_replay_disagreements: Some(1),
            verification_mode: Some("rectangular".into()),
            rectangular_verification_rounds: Some(1),
            serial_verification_rounds: Some(0),
            runner: Some(RunnerIdentity {
                id: "r".into(),
                model_type: "m".into(),
                manifest_sha256: "s".into(),
                build: "b".into(),
            }),
            resident: Some(ResidentIdentity {
                pid: 1,
                load_epoch: 1,
            }),
        };
        assert_eq!(
            json_keys(&serde_json::to_value(&resp).unwrap()),
            schema_props("WorkerResponse"),
            "WorkerResponse struct/schema field drift"
        );

        // #106: the passthrough $defs.HeadProvenance property set matches the struct's keys.
        assert_eq!(
            json_keys(
                &serde_json::to_value(HeadProvenance {
                    sha256: "h".into(),
                    bytes: 1,
                    file_count: 1
                })
                .unwrap()
            ),
            schema_props("HeadProvenance"),
            "HeadProvenance struct/schema field drift"
        );

        // Runner contract: the $defs.RunnerIdentity property set matches the struct's keys.
        assert_eq!(
            json_keys(
                &serde_json::to_value(RunnerIdentity {
                    id: "r".into(),
                    model_type: "m".into(),
                    manifest_sha256: "s".into(),
                    build: "b".into()
                })
                .unwrap()
            ),
            schema_props("RunnerIdentity"),
            "RunnerIdentity struct/schema field drift"
        );

        // The $defs.ResidentIdentity property set matches the struct's keys.
        assert_eq!(
            json_keys(
                &serde_json::to_value(ResidentIdentity {
                    pid: 1,
                    load_epoch: 1
                })
                .unwrap()
            ),
            schema_props("ResidentIdentity"),
            "ResidentIdentity struct/schema field drift"
        );

        assert_eq!(
            json_keys(&serde_json::to_value(CorrectnessTraceLogit::new(1, 1.0)).unwrap()),
            schema_props("CorrectnessTraceLogit"),
            "CorrectnessTraceLogit struct/schema field drift"
        );
        assert_eq!(
            json_keys(&serde_json::to_value(ExpertStreamingStats::zero()).unwrap()),
            schema_props("ExpertStreamingStats"),
            "ExpertStreamingStats struct/schema field drift"
        );
    }

    /// The shared conformance transcript (`schema/engine-wire-v1-adapter.ndjson`), a whole
    /// session emitted by the `bench-adapter` loop: the hello, then every verb the adapter
    /// serves as a request line and the response line it produced, then one refusal. The
    /// Swift bench-worker REPLAYS THESE BYTES, so this crate — which owns the wire — holds
    /// the file to the wire it defines.
    ///
    /// Every line must:
    ///
    ///   * decode into its wire type under `deny_unknown_fields`, which is the serde half of
    ///     the schema's `additionalProperties:false` plus its declared types (a response
    ///     carries no `kind`, and that is what tells a response from a request on the wire);
    ///   * carry only keys the schema declares for that type; and
    ///   * RE-SERIALIZE BYTE-IDENTICALLY. That last one is what a replayer needs and what a
    ///     schema alone cannot give: it fixes key order and number formatting, so a
    ///     transcript recorded from the Rust adapter is the same bytes any conforming worker
    ///     must produce.
    ///
    /// The structs are held to the schema field-for-field by
    /// [`schema_properties_match_struct_fields`], so a line that satisfies the structs
    /// satisfies the schema's property set.
    #[test]
    fn the_adapter_transcript_conforms_to_this_wire() {
        use std::collections::BTreeSet;
        const TRANSCRIPT: &str = include_str!("../schema/engine-wire-v1-adapter.ndjson");
        let schema: serde_json::Value = serde_json::from_str(JSON_SCHEMA).unwrap();
        let schema_props = |name: &str| -> BTreeSet<String> {
            schema["$defs"][name]["properties"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect()
        };
        let request_props = schema_props("WorkerRequest");
        let response_props = schema_props("WorkerResponse");

        let mut requests = 0;
        let mut responses = 0;
        for (i, line) in TRANSCRIPT.lines().enumerate() {
            let value: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("transcript line {i} is not JSON: {e}"));
            let keys: BTreeSet<String> = value.as_object().unwrap().keys().cloned().collect();
            let (declared, reserialized) = if value.get("kind").is_some() {
                requests += 1;
                let req: WorkerRequest = serde_json::from_str(line)
                    .unwrap_or_else(|e| panic!("transcript line {i} is not a WorkerRequest: {e}"));
                (&request_props, serde_json::to_string(&req).unwrap())
            } else {
                responses += 1;
                let resp: WorkerResponse = serde_json::from_str(line)
                    .unwrap_or_else(|e| panic!("transcript line {i} is not a WorkerResponse: {e}"));
                (&response_props, serde_json::to_string(&resp).unwrap())
            };
            let undeclared: Vec<&String> = keys.difference(declared).collect();
            assert!(
                undeclared.is_empty(),
                "transcript line {i} carries keys the schema does not declare: {undeclared:?}"
            );
            assert_eq!(
                reserialized, line,
                "transcript line {i} is not byte-canonical; the Swift bench-worker replays \
                 these bytes"
            );
        }
        assert!(
            requests > 0 && responses == requests + 1,
            "one hello plus one response per request: {responses} responses, {requests} requests"
        );
    }
}
