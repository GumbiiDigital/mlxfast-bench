//! Engine session lifecycle: hello handshake, id/nonce validation, typed requests,
//! and the phase-close barrier + completed-work counter.
//!
//! Client semantics are a faithful port of the Swift `RuntimeWorkerClient`
//! (`Sources/MLXFastHarness/QwenRuntimeWorker.swift`, ~lines 1073–1330): the hello
//! handshake, `send`, and `readResponseLine` (skip non-JSON preamble lines, echo id,
//! validate nonce). See docs/architecture.md §2 for the normative invariants.

use bench_core::free_run::{
    verify_cohort_consistency, verify_consistency, CohortFreeRunAudit, CohortFreeRunResponse,
    FreeRunAudit, FreeRunResponse,
};
use bench_protocol::{
    CohortReferenceReplayReport, RequestKind, WorkerRequest, WorkerResponse,
    CAPABILITY_BATCHED_FREE_RUN_DECODE, CAPABILITY_COHORT_REFERENCE_REPLAY,
    CAPABILITY_FREE_RUN_DECODE, CAPABILITY_PER_STREAM_TIMING, PROTOCOL_VERSION,
    REPLAY_WIDTH_COHORT, TOP_LOGITS_K,
};

use crate::error::{Result, RunnerError};
use crate::transport::{LineTransport, ReadOutcome};

/// The decoded, validated `hello` (id=0) fields callers care about.
#[derive(Debug, Clone, PartialEq)]
pub struct Hello {
    /// Session nonce every subsequent response must echo.
    pub nonce: String,
    /// Engine's implemented protocol version (`hello` only).
    pub protocol_version: Option<u32>,
    /// Compute backend, e.g. `"mlx"` / `"cuda"` (`hello` only).
    pub backend: Option<String>,
    /// Device identity, e.g. `"m5"` / `"gb10"` (`hello` only).
    pub device: Option<String>,
    /// v1.1 capability flags the engine advertised (`hello` only). Empty for a v1-only engine.
    pub capabilities: Vec<String>,
    /// Runnable speculative modes the engine advertised (`hello.spec_modes`,
    /// `docs/spec-config-design.md`). Empty for a pre-spec engine (only the default path is runnable).
    pub spec_modes: Vec<String>,
    /// #106 (passthrough MODEL): the engine's loaded-head provenance echoed on the `hello`
    /// (`WorkerResponse::head_provenance`). `None` for a pre-#106 engine that omits it. AUDIT-only —
    /// benchd surfaces it for sealing/provenance, never scores it.
    pub head_provenance: Option<bench_protocol::HeadProvenance>,
    /// v1.2 (COHORT): the largest cohort width B the engine advertised on the `hello`
    /// (`WorkerResponse::max_batch_size`). `None` for an engine that omits it. benchd uses it to
    /// refuse an over-wide cohort PRE-GPU (before the cool gate and before the clock).
    pub max_batch_size: Option<u32>,
    /// Runner contract §6.1: the runner identity the worker echoed on the `hello`
    /// (`WorkerResponse::runner`). `None` for a worker that omits it. AUDIT-only — benchd surfaces
    /// it for sealing/provenance and the conformance kit checks it against the declared manifest,
    /// but it is never a scoring input.
    pub runner: Option<bench_protocol::RunnerIdentity>,
    /// The resident-process identity the worker echoed on the `hello`
    /// (`WorkerResponse::resident`). `None` for a worker that loaded the weights itself rather than
    /// attaching to a resident process. AUDIT-only — benchd surfaces it for sealing/provenance, and
    /// a changed `load_epoch` across one window is a reload that should not have happened, but it is
    /// never a scoring input and it is not a manifest fact.
    pub resident: Option<bench_protocol::ResidentIdentity>,
}

impl Hello {
    /// Does the engine advertise the v1.1 oracle-verified free-run timed decode capability?
    /// benchd REFUSES to issue `free_decode_*` unless this holds (PROTOCOL-v1.1.md §2.1).
    pub fn supports_free_run_decode(&self) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == CAPABILITY_FREE_RUN_DECODE)
    }

    /// Does the engine advertise the v1.2 BATCHED (cohort) form of the free-run verbs?
    /// benchd REFUSES to issue the cohort form unless this holds — an engine advertising only
    /// `free_run_decode` is single-stream, never treated as batch-capable.
    pub fn supports_batched_free_run_decode(&self) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == CAPABILITY_BATCHED_FREE_RUN_DECODE)
    }

    /// Does the engine advertise the per-stream timing instrumentation capability
    /// (`per_stream_timing`, per-stream-instrumentation-spec.md step 1) — i.e. promise the
    /// additive `prefill_ns_by_stream` / `decode_ns_by_stream` per-slot vectors on its batched
    /// free-run responses? REPORT-ONLY this increment: unlike the two free-run capabilities above
    /// this gates no request refusal here — it is recorded so downstream attestation can tell
    /// "not advertised" apart from "advertised but absent". Parent clocks remain the sole scored
    /// timing source (engine-reported-time-untrusted / parent-clock doctrine).
    pub fn supports_per_stream_timing(&self) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == CAPABILITY_PER_STREAM_TIMING)
    }

    /// Does the engine advertise `mode` as a RUNNABLE speculative mode (`hello.spec_modes`)?
    pub fn supports_spec_mode(&self, mode: &str) -> bool {
        self.spec_modes.iter().any(|m| m == mode)
    }

    /// (b) admission — does this engine advertise the TRUSTED-ORACLE `cohort_reference_replay`
    /// capability? benchd REFUSES to issue the verb unless this holds (N1): only the organizer's
    /// trusted build advertises it, so this is the wire-level proof that the reference argmax will be
    /// produced by trusted, non-candidate forward code.
    pub fn supports_cohort_reference_replay(&self) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == CAPABILITY_COHORT_REFERENCE_REPLAY)
    }

    /// The canonical-manifest digest the worker declared on the `hello`
    /// (`runner.manifest_sha256`), or `None` when the worker echoed no runner identity. The
    /// conformance kit compares this against the digest it computes over the manifest file.
    pub fn runner_manifest_sha256(&self) -> Option<&str> {
        self.runner.as_ref().map(|r| r.manifest_sha256.as_str())
    }
}

/// Does this line look like a JSON response object?
///
/// Mirrors Swift `runtimeWorkerLineLooksLikeJSONResponse`: skip leading spaces/tabs/CR,
/// and treat the line as a response iff the first non-whitespace byte is `{`. Anything
/// else (blank lines, plain-text engine logs) is skipped by the read loop.
fn line_looks_like_json_response(line: &str) -> bool {
    for &byte in line.as_bytes() {
        if byte == 0x20 || byte == 0x09 || byte == 0x0d {
            continue;
        }
        return byte == 0x7b; // '{'
    }
    false
}

/// A live engine session over one [`LineTransport`].
///
/// The session assigns monotonic request ids starting at 1, validates each response's
/// nonce and id, and discards itself (fail-closed, §3) on any error so no further
/// request is accepted after a fault.
pub struct Session<T: LineTransport> {
    transport: T,
    nonce: String,
    next_id: i64,
    /// Timed step-requests issued in the current phase; checked at `close_phase`.
    issued_steps: i64,
    /// v1.1 capability flags the engine advertised on the hello; gates `free_decode_*`.
    capabilities: Vec<String>,
    /// Runnable speculative modes the engine advertised on the hello (`hello.spec_modes`); gates a
    /// spec'd `decode_begin` (a non-default mode not listed here is refused before the timed seed
    /// forward).
    spec_modes: Vec<String>,
    /// v1.2 (COHORT): the `max_batch_size` the engine advertised on the hello, if any; used to
    /// refuse an over-wide cohort request pre-GPU.
    max_batch_size: Option<u32>,
    discarded: bool,
    /// H3 (cycle-3) — the RunTimeout deadline for the CURRENT timed round-trips (§2.2/§4): the
    /// wall-clock instant the wait must not pass, and the configured budget seconds (for the error).
    /// When `Some`, every response wait is bounded by it; a wait that passes the deadline raises
    /// `RunTimeout` and discards the session (fail-closed). `None` means no bound (the untimed
    /// default). Armed/disarmed around the timed window by the timing layer.
    run_deadline: Option<(std::time::Instant, f64)>,
}

impl<T: LineTransport> Session<T> {
    /// Perform the hello handshake and construct a session.
    ///
    /// Reads lines, skipping any that do not look like a JSON response object (Swift
    /// `readResponseLine` behavior), decodes the first that does as the `hello`, and
    /// requires `id == 0`, `ok == true`, and a non-empty `nonce`.
    pub fn connect(transport: T) -> Result<(Self, Hello)> {
        Self::connect_inner(transport).inspect_err(crate::error::log_engine_data)
    }

    fn connect_inner(mut transport: T) -> Result<(Self, Hello)> {
        // The hello handshake is untimed (no RunTimeout deadline armed yet).
        let resp = read_response_line(&mut transport, None, "hello")?;
        if resp.id != 0 {
            // The id is engine-chosen: the log gets it, the sealed text does not.
            eprintln!("bench-runner: engine hello id {}", resp.id);
            return Err(RunnerError::Protocol("hello had a non-zero id".to_string()));
        }
        if !resp.ok {
            return Err(RunnerError::ProtocolEngineText {
                detail: "hello was not ok".to_string(),
                engine_text: resp.error.unwrap_or_else(|| "unknown error".to_string()),
            });
        }
        let nonce = match resp.nonce {
            Some(ref n) if !n.is_empty() => n.clone(),
            _ => {
                return Err(RunnerError::Protocol(
                    "hello did not carry a non-empty nonce".to_string(),
                ))
            }
        };
        // C4: reject an engine speaking a different protocol version (or none at all).
        // The hello is the only place the version is meaningful; refuse to drive an
        // engine we cannot guarantee wire-compatibility with.
        if resp.protocol_version != Some(PROTOCOL_VERSION) {
            eprintln!(
                "bench-runner: engine hello protocol_version {:?}",
                resp.protocol_version
            );
            return Err(RunnerError::Protocol(format!(
                "hello protocol_version does not match supported {PROTOCOL_VERSION}"
            )));
        }
        let capabilities = resp.capabilities.unwrap_or_default();
        let spec_modes = resp.spec_modes.unwrap_or_default();
        let hello = Hello {
            nonce: nonce.clone(),
            protocol_version: resp.protocol_version,
            backend: resp.backend,
            device: resp.device,
            capabilities: capabilities.clone(),
            spec_modes: spec_modes.clone(),
            head_provenance: resp.head_provenance,
            max_batch_size: resp.max_batch_size,
            runner: resp.runner,
            resident: resp.resident,
        };
        let session = Session {
            transport,
            nonce,
            next_id: 1,
            issued_steps: 0,
            capabilities,
            spec_modes,
            max_batch_size: resp.max_batch_size,
            discarded: false,
            run_deadline: None,
        };
        Ok((session, hello))
    }

    /// H3 (cycle-3) — arm the RunTimeout deadline for the coming timed round-trips: every response
    /// wait until [`disarm_run_deadline`](Self::disarm_run_deadline) is bounded by `deadline`. The
    /// timing layer arms this at the start of the timed window with `now + (N × band-ceiling ×
    /// margin)` and disarms it after, so the untimed barrier/setup reads are never bounded.
    /// `budget_seconds` is the configured budget, carried for the `RunTimeout` error only.
    pub fn arm_run_deadline(&mut self, deadline: std::time::Instant, budget_seconds: f64) {
        self.run_deadline = Some((deadline, budget_seconds));
    }

    /// H3 (cycle-3) — clear the RunTimeout deadline (subsequent reads block unbounded again).
    pub fn disarm_run_deadline(&mut self) {
        self.run_deadline = None;
    }

    /// The session nonce established at hello.
    pub fn nonce(&self) -> &str {
        &self.nonce
    }

    /// Whether the session has been discarded by a prior error.
    pub fn is_discarded(&self) -> bool {
        self.discarded
    }

    /// Timed steps issued since the last `begin_phase`.
    pub fn issued_steps(&self) -> i64 {
        self.issued_steps
    }

    /// Send a request, assigning its id, and validate the response.
    ///
    /// Order of checks mirrors the Swift client: nonce, then id, then `ok`. Any error
    /// discards the session (fail-closed, §3).
    fn send(&mut self, mut req: WorkerRequest) -> Result<WorkerResponse> {
        // Phase-close barrier bookkeeping (§3): count exactly the timed-step kinds
        // toward `issued_steps`. `RequestKind::is_timed_step` is the ONE definition of
        // that set, so no typed method hand-codes whether it bumps the counter.
        if RequestKind::from_wire(&req.kind).is_some_and(|k| k.is_timed_step()) {
            self.issued_steps += 1;
        }
        if self.discarded {
            return Err(RunnerError::SessionDiscarded);
        }
        let id = self.next_id;
        self.next_id += 1;
        req.id = id;

        // From here on, any failure taints the session.
        let result = self.send_inner(&req, id);
        if let Err(e) = &result {
            self.discarded = true;
            crate::error::log_engine_data(e);
        }
        result
    }

    fn send_inner(&mut self, req: &WorkerRequest, id: i64) -> Result<WorkerResponse> {
        let line = serde_json::to_string(req)?;
        self.transport.write_line(&line)?;
        // H3 (cycle-3) — bound the response wait by the armed RunTimeout deadline (if any); a hung
        // engine that never responds raises `RunTimeout` (session discarded by `send`), not a hang.
        let resp = read_response_line(&mut self.transport, self.run_deadline, &req.kind)?;

        // C8: the spec's unparseable-line response is `{id: -1, ok: false, error}` — the
        // engine could not parse our request. Surface the engine's own error (as an
        // Engine fault) BEFORE the nonce/id checks, which would otherwise mask it as a
        // NonceMismatch or id-mismatch. Either way the session is discarded (fail-closed).
        if resp.id == -1 && !resp.ok {
            return Err(RunnerError::Engine {
                kind: req.kind.clone(),
                message: resp
                    .error
                    .clone()
                    .unwrap_or_else(|| "engine could not parse the request line".to_string()),
            });
        }

        // 1. nonce
        if resp.nonce.as_deref() != Some(self.nonce.as_str()) {
            return Err(RunnerError::NonceMismatch {
                expected: self.nonce.clone(),
                got: resp.nonce.clone(),
            });
        }
        // 2. id echo
        if resp.id != id {
            eprintln!("bench-runner: engine response id {}", resp.id);
            return Err(RunnerError::Protocol(format!(
                "response id did not match request id {id}"
            )));
        }
        // 3. ok
        if !resp.ok {
            return Err(RunnerError::Engine {
                kind: req.kind.clone(),
                message: resp
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown error".to_string()),
            });
        }
        Ok(resp)
    }

    // ---- typed requests ---------------------------------------------------

    /// `prefill` — force full evaluation of `prompt_tokens`, returns a `token`.
    /// Not a timed step (§3): prefill is setup, not part of the timed decode window.
    pub fn prefill(&mut self, prompt_tokens: &[i64]) -> Result<WorkerResponse> {
        let mut req = WorkerRequest::new(0, RequestKind::Prefill.as_str());
        req.prompt_tokens = Some(prompt_tokens.to_vec());
        self.send(req)
    }

    /// `decode_begin` — the single seed forward. Counts as one timed step (§3): the
    /// decode clock starts before this call, so the seed forward is inside the window.
    /// (`issued_steps` is bumped centrally in `send` via `RequestKind::is_timed_step`.)
    ///
    /// The no-spec form; a spec'd decode window uses [`decode_begin_spec`](Self::decode_begin_spec).
    pub fn decode_begin(&mut self, seed_tokens: &[i64]) -> Result<WorkerResponse> {
        self.decode_begin_spec(seed_tokens, None)
    }

    /// `decode_begin` carrying the per-module speculative `spec` (`docs/spec-config-design.md`).
    /// When a `spec` is supplied, benchd enforces SPEC-NEVER-IGNORED (§6): the response MUST echo an
    /// `effective_spec` EQUAL to the request, else the session is discarded fail-closed
    /// ([`RunnerError::SpecEchoDivergence`]) — the same posture as [`Self::require_free_run_capability`].
    /// A `None` spec issues the legacy no-spec `decode_begin` and does not check the echo.
    pub fn decode_begin_spec(
        &mut self,
        seed_tokens: &[i64],
        spec: Option<&bench_protocol::SpecConfig>,
    ) -> Result<WorkerResponse> {
        self.require_spec_mode_runnable(spec)?;
        let mut req = WorkerRequest::new(0, RequestKind::DecodeBegin.as_str());
        req.seed_tokens = Some(seed_tokens.to_vec());
        req.spec = spec.cloned();
        let resp = self.send(req)?;
        self.require_spec_echo(spec, &resp)?;
        Ok(resp)
    }

    /// Medium (#105) — ENFORCEMENT of `hello.spec_modes` before the timed seed forward (cycle-5
    /// finding 6: NOT "pre-clock" — `measure_decode` takes its `Instant::now()` before calling in
    /// here): when a spec requests a NON-default
    /// speculative mode (anything but `serial`), the engine MUST have advertised that mode as runnable
    /// on its hello, else the spec'd decode is refused BEFORE the timed seed forward is issued and the
    /// session is discarded fail-closed ([`RunnerError::SpecModeNotRunnable`]). This wires the
    /// otherwise-dead [`Hello::supports_spec_mode`] into the live path, so the "rejected before any
    /// timed work" posture is REAL, not a claim. `serial` (the default path) is always runnable and is
    /// never gated (a pre-spec engine advertises no spec_modes yet still runs serial). `None` spec ⇒
    /// no check.
    fn require_spec_mode_runnable(
        &mut self,
        spec: Option<&bench_protocol::SpecConfig>,
    ) -> Result<()> {
        let Some(spec) = spec else { return Ok(()) };
        if spec.mode == bench_protocol::SPEC_MODE_SERIAL {
            return Ok(());
        }
        if self.spec_modes.iter().any(|m| m == &spec.mode) {
            return Ok(());
        }
        self.discarded = true;
        let e = RunnerError::SpecModeNotRunnable {
            mode: spec.mode.clone(),
            advertised: self.spec_modes.clone(),
        };
        crate::error::log_engine_data(&e);
        Err(e)
    }

    /// Spec-never-ignored enforcement (§6): when `requested` is `Some`, the engine's echoed
    /// `effective_spec` must be present and HONOR it — every field the request specified must come
    /// back identical, while fields the request left absent are the module's to fill and the echo
    /// reports them ([`bench_protocol::spec_echo_honors_request`]; David ruling 2026-08-27 — an
    /// engine-decides request like `{"mode":"mtp","mtp":{}}` accepts the module-resolved depth in
    /// the echo). A missing echo, or one that CHANGES a requested value, discards the session and
    /// raises [`RunnerError::SpecEchoDivergence`]. `None` requested ⇒ no check.
    fn require_spec_echo(
        &mut self,
        requested: Option<&bench_protocol::SpecConfig>,
        resp: &WorkerResponse,
    ) -> Result<()> {
        let Some(requested) = requested else {
            return Ok(());
        };
        if resp
            .effective_spec
            .as_ref()
            .is_some_and(|e| bench_protocol::spec_echo_honors_request(requested, e))
        {
            return Ok(());
        }
        self.discarded = true;
        let e = RunnerError::SpecEchoDivergence {
            requested: serde_json::to_string(requested).unwrap_or_default(),
            effective: resp
                .effective_spec
                .as_ref()
                .map(|e| serde_json::to_string(e).unwrap_or_default()),
        };
        crate::error::log_engine_data(&e);
        Err(e)
    }

    /// `decode_step` — one timed decode step (§3).
    pub fn decode_step(&mut self, token: i64) -> Result<WorkerResponse> {
        let mut req = WorkerRequest::new(0, RequestKind::DecodeStep.as_str());
        req.token = Some(token);
        self.send(req)
    }

    /// Whether the engine advertised the v1.1 `free_run_decode` capability on its hello.
    pub fn supports_free_run_decode(&self) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == CAPABILITY_FREE_RUN_DECODE)
    }

    /// Refuse to issue a `free_decode_*` request when the engine did not advertise the
    /// `free_run_decode` capability. An unadvertised capability is a HARD protocol error, never
    /// a silent fallback (PROTOCOL-v1.1.md §2.1); the session is discarded fail-closed.
    fn require_free_run_capability(&mut self) -> Result<()> {
        if !self.supports_free_run_decode() {
            self.discarded = true;
            return Err(RunnerError::CapabilityNotAdvertised {
                capability: CAPABILITY_FREE_RUN_DECODE.to_string(),
            });
        }
        Ok(())
    }

    /// `free_decode_begin` — v1.1 (additive): the single seed forward that opens an
    /// oracle-verified free-run decode phase. Same wire contract as `decode_begin`
    /// (`seed_tokens[]` in, `seed_token` out). NOT a `completed_work` timed step (Amendment 4):
    /// the free-run phase's counter is `R + 1`, validated by the §2.6 triple in the driver.
    /// REFUSES if the engine did not advertise `free_run_decode`.
    pub fn free_decode_begin(&mut self, seed_tokens: &[i64]) -> Result<WorkerResponse> {
        self.free_decode_begin_spec(seed_tokens, None)
    }

    /// `free_decode_begin` carrying the per-module speculative `spec`. Enforces SPEC-NEVER-IGNORED
    /// on the echo exactly like [`decode_begin_spec`](Self::decode_begin_spec).
    pub fn free_decode_begin_spec(
        &mut self,
        seed_tokens: &[i64],
        spec: Option<&bench_protocol::SpecConfig>,
    ) -> Result<WorkerResponse> {
        self.require_free_run_capability()?;
        self.require_spec_mode_runnable(spec)?;
        let mut req = WorkerRequest::new(0, RequestKind::FreeDecodeBegin.as_str());
        req.seed_tokens = Some(seed_tokens.to_vec());
        req.spec = spec.cloned();
        let resp = self.send(req)?;
        self.require_spec_echo(spec, &resp)?;
        Ok(resp)
    }

    /// `free_decode_run` — v1.1 (additive): the engine free-runs its own MTP loop until it has
    /// committed `count` (= N) tokens, returning all N committed token IDs plus the AUDIT
    /// acceptance counters (PROTOCOL-v1.1.md §2.1). REFUSES if `free_run_decode` was not
    /// advertised. The wall clock and the per-token oracle verification are the driver's job.
    pub fn free_decode_run(&mut self, count: u32) -> Result<WorkerResponse> {
        self.require_free_run_capability()?;
        let mut req = WorkerRequest::new(0, RequestKind::FreeDecodeRun.as_str());
        req.count = Some(count);
        self.send(req)
    }

    // ---- v1.2 COHORT (batched free-run) ------------------------------------

    /// Whether the engine advertised the v1.2 `batched_free_run_decode` capability on its hello.
    pub fn supports_batched_free_run_decode(&self) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == CAPABILITY_BATCHED_FREE_RUN_DECODE)
    }

    /// Whether the engine advertised the `per_stream_timing` capability on its hello
    /// (per-stream-instrumentation-spec.md step 1). Same semantics as
    /// [`Hello::supports_per_stream_timing`]; REPORT-ONLY — consulted to RECORD the advertisement
    /// alongside the carried per-slot ns vectors, never to gate a request or feed a scored value.
    pub fn supports_per_stream_timing(&self) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == CAPABILITY_PER_STREAM_TIMING)
    }

    /// The `max_batch_size` the engine advertised on its hello (`None` if omitted). Lets a caller
    /// refuse an over-wide cohort PRE-GPU rather than discovering it inside a timed window.
    pub fn max_batch_size(&self) -> Option<u32> {
        self.max_batch_size
    }

    /// Refuse to issue the COHORT form of a `free_decode_*` request when the engine did not
    /// advertise `batched_free_run_decode`. Mirrors [`require_free_run_capability`]: an
    /// unadvertised capability is a HARD protocol error, never a silent fallback to the
    /// single-stream regime (which would silently swap the measured quantity class); the session
    /// is discarded fail-closed.
    fn require_batched_free_run_capability(&mut self) -> Result<()> {
        if !self.supports_batched_free_run_decode() {
            self.discarded = true;
            return Err(RunnerError::CapabilityNotAdvertised {
                capability: CAPABILITY_BATCHED_FREE_RUN_DECODE.to_string(),
            });
        }
        Ok(())
    }

    /// BATCH-NEVER-IGNORED (v1.2, the `effective_spec` posture applied to the cohort width): the
    /// engine's echoed `effective_batch_size` must be PRESENT and EQUAL the requested B. A missing
    /// or divergent echo discards the session and raises [`RunnerError::BatchEchoDivergence`] — a
    /// silently narrowed (or widened) cohort can never be sealed as a B=8 measurement.
    ///
    /// Unlike the spec echo (checked only when a spec was requested), this check is UNCONDITIONAL
    /// on the cohort path: the cohort form always carries an explicit `batch_size`, so there is
    /// always an identity to hold the engine to.
    fn require_batch_echo(&mut self, requested: u32, resp: &WorkerResponse) -> Result<()> {
        if resp.effective_batch_size == Some(requested) {
            return Ok(());
        }
        self.discarded = true;
        Err(RunnerError::BatchEchoDivergence {
            requested,
            effective: resp.effective_batch_size,
        }
        .logged())
    }

    /// `free_decode_begin` in its v1.2 COHORT form: B seed forwards, one per cohort slot in SLOT
    /// ORDER (`seed_tokens_by_stream`), with the EXPLICIT `batch_size` carried on the wire (B is
    /// never inferred from the array length alone). Enforces, fail-closed:
    ///
    /// - the `batched_free_run_decode` capability ([`Self::require_batched_free_run_capability`]);
    /// - SPEC-NEVER-IGNORED on the echoed `effective_spec` — ONE spec for the whole cohort
    ///   (per-stream spec is deliberately not offered; the engine forbids mixed depths in a plan);
    /// - BATCH-NEVER-IGNORED on the echoed `effective_batch_size` ([`Self::require_batch_echo`]).
    ///
    /// Like its v1.1 counterpart, NOT a `completed_work` timed step: the cohort free-run phase's
    /// counter is the SCALAR `R + 1` (a round is one engine forward regardless of B), validated by
    /// the cohort consistency quadruple in [`Self::close_batched_free_run_phase`].
    pub fn free_decode_begin_batched(
        &mut self,
        seed_tokens_by_stream: &[Vec<i64>],
        batch_size: u32,
        spec: Option<&bench_protocol::SpecConfig>,
    ) -> Result<WorkerResponse> {
        self.require_batched_free_run_capability()?;
        self.require_spec_mode_runnable(spec)?;
        let mut req = WorkerRequest::new(0, RequestKind::FreeDecodeBegin.as_str());
        req.seed_tokens_by_stream = Some(seed_tokens_by_stream.to_vec());
        req.batch_size = Some(batch_size);
        req.spec = spec.cloned();
        let resp = self.send(req)?;
        self.require_spec_echo(spec, &resp)?;
        self.require_batch_echo(batch_size, &resp)?;
        Ok(resp)
    }

    /// `free_decode_run` in its v1.2 COHORT form: the engine free-runs its own loop until EVERY
    /// stream has committed `count` (= N per stream) tokens, returning the B x N
    /// `tokens_by_stream` rectangle plus the cohort AUDIT counters. Carries the explicit
    /// `batch_size` again; if the engine echoes `effective_batch_size` here too, it must still
    /// equal the request (never-ignored) — the response's own rectangle shape is separately pinned
    /// to B by the cohort consistency quadruple.
    pub fn free_decode_run_batched(
        &mut self,
        count: u32,
        batch_size: u32,
    ) -> Result<WorkerResponse> {
        self.require_batched_free_run_capability()?;
        let mut req = WorkerRequest::new(0, RequestKind::FreeDecodeRun.as_str());
        req.count = Some(count);
        req.batch_size = Some(batch_size);
        let resp = self.send(req)?;
        if let Some(echoed) = resp.effective_batch_size {
            if echoed != batch_size {
                self.discarded = true;
                return Err(RunnerError::BatchEchoDivergence {
                    requested: batch_size,
                    effective: Some(echoed),
                }
                .logged());
            }
        }
        Ok(resp)
    }

    // ---- (b) admission COHORT REFERENCE REPLAY (trusted oracle) ------------

    /// (b) admission — whether THIS session's engine advertised the TRUSTED-ORACLE
    /// `cohort_reference_replay` capability on its hello. Only the organizer's trusted build does.
    pub fn supports_cohort_reference_replay(&self) -> bool {
        self.capabilities
            .iter()
            .any(|c| c == CAPABILITY_COHORT_REFERENCE_REPLAY)
    }

    /// Refuse to issue `cohort_reference_replay` to a worker that did not advertise
    /// [`CAPABILITY_COHORT_REFERENCE_REPLAY`]. This is the wire half of N1: the UNTRUSTED candidate
    /// worker never advertises the capability, so benchd never asks it for a reference argmax (an
    /// unadvertised capability is a HARD protocol error, never a silent fallback). The session is
    /// discarded fail-closed. The organizer wires this verb ONLY onto a session spawned from the
    /// TRUSTED build (`benchd` oracle spawn path).
    fn require_cohort_reference_replay_capability(&mut self) -> Result<()> {
        if !self.supports_cohort_reference_replay() {
            self.discarded = true;
            return Err(RunnerError::CapabilityNotAdvertised {
                capability: CAPABILITY_COHORT_REFERENCE_REPLAY.to_string(),
            });
        }
        Ok(())
    }

    /// `cohort_reference_replay` (engine #35, d528f6ac) — issue the TRUSTED oracle verb: replay each
    /// cohort stream TEACHER-FORCED on the candidate's OWN committed tokens (`committed_by_stream`,
    /// with the per-stream seed context `replay_seeds_by_stream`) and return the per-stream reference
    /// argmax report. REFUSES if the (trusted) hello did not advertise
    /// [`CAPABILITY_COHORT_REFERENCE_REPLAY`] (N1). NOT a `completed_work` timed step — it is an
    /// UNTIMED post-run oracle, so it is never charged to any scored window.
    ///
    /// The caller is responsible for the WEIGHTS PROVENANCE: this method sends only the tokens; the
    /// organizer's reference weights are fixed at SPAWN time (`--weights`), never carried here, so the
    /// candidate cannot substitute the oracle's weights through this request.
    ///
    /// GEOMETRY IS PINNED BY THE REQUEST (engine #37): the width is sent EXPLICITLY as
    /// [`REPLAY_WIDTH_COHORT`], never left to the engine-side default. The reference's replay
    /// geometry is an ENFORCED parameter — the tolerance gate judges the candidate against
    /// whatever this oracle produced — and an enforced parameter benchd does not state is one
    /// benchd cannot verify. See [`cohort_reference_replay_request`].
    pub fn cohort_reference_replay(
        &mut self,
        replay_seeds_by_stream: &[Vec<i64>],
        committed_by_stream: &[Vec<i64>],
    ) -> Result<CohortReferenceReplayReport> {
        self.require_cohort_reference_replay_capability()?;
        let req = cohort_reference_replay_request(replay_seeds_by_stream, committed_by_stream);
        let resp = self.send(req)?;
        let report = resp.cohort_reference_replay.ok_or_else(|| {
            RunnerError::Protocol(
                "cohort_reference_replay response carried no cohort_reference_replay report"
                    .to_string(),
            )
        })?;
        assert_replay_width_echo(REPLAY_WIDTH_COHORT, &report)?;
        Ok(report)
    }

    /// `correctness` — free-run greedy generation. Not a timed step (§3): it is a
    /// single free-run request, not a per-step timed forward guarded by the barrier.
    pub fn correctness(&mut self, prompt_tokens: &[i64], steps: i64) -> Result<WorkerResponse> {
        let mut req = WorkerRequest::new(0, RequestKind::Correctness.as_str());
        req.prompt_tokens = Some(prompt_tokens.to_vec());
        req.steps = Some(steps);
        self.send(req)
    }

    /// `correctness_begin` — first teacher-forced forward. Counts as one timed step (§3).
    pub fn correctness_begin(&mut self, prompt_tokens: &[i64]) -> Result<WorkerResponse> {
        let mut req = WorkerRequest::new(0, RequestKind::CorrectnessBegin.as_str());
        req.prompt_tokens = Some(prompt_tokens.to_vec());
        let resp = self.send(req)?;
        self.require_top_logits_k(&resp)?;
        Ok(resp)
    }

    /// `correctness_step` — one teacher-forced timed step (§3).
    pub fn correctness_step(&mut self, token: i64) -> Result<WorkerResponse> {
        let mut req = WorkerRequest::new(0, RequestKind::CorrectnessStep.as_str());
        req.token = Some(token);
        let resp = self.send(req)?;
        self.require_top_logits_k(&resp)?;
        Ok(resp)
    }

    /// S2: teacher-forced responses must carry exactly `TOP_LOGITS_K` top-logit entries
    /// (PROTOCOL.md `top_logits[8]`). A wrong length is a conformance violation — fail
    /// closed and discard the session.
    fn require_top_logits_k(&mut self, resp: &WorkerResponse) -> Result<()> {
        let n = resp.top_logits.as_ref().map(|v| v.len()).unwrap_or(0);
        if n != TOP_LOGITS_K {
            self.discarded = true;
            return Err(RunnerError::Protocol(format!(
                "expected {TOP_LOGITS_K} top_logits, got {n}"
            )));
        }
        Ok(())
    }

    /// `phase_diagnostics` — raw send, without the completed-work barrier check.
    /// Used internally by `close_phase`; exposed for callers that need the raw response.
    /// Not a timed step.
    pub fn phase_diagnostics_raw(&mut self) -> Result<WorkerResponse> {
        let req = WorkerRequest::new(0, RequestKind::PhaseDiagnostics.as_str());
        self.send(req)
    }

    // ---- phase-close barrier (WS1-7) --------------------------------------

    /// Reset the timed-step counter at the start of a phase.
    pub fn begin_phase(&mut self) {
        self.issued_steps = 0;
    }

    /// Close the phase: send `phase_diagnostics` and enforce the completed-work barrier
    /// (§3 / §8 cycle 2). The engine's `completed_work` must equal the number of timed
    /// steps issued since `begin_phase`; otherwise the run fails and the session is
    /// discarded.
    pub fn close_phase(&mut self) -> Result<WorkerResponse> {
        let resp = self.phase_diagnostics_raw()?;
        // #54 — allocator-drain assertion. Swift's worker fails the run CLOSED at every
        // phase boundary unless `Memory.cacheMemory == 0` after `Memory.clearCache()`
        // (resetRuntimeWorkerAllocatorForPhaseStart, QwenRuntimeWorker.swift:99-115). The
        // engine surfaces that value as `cache_memory` on phase_diagnostics; the parent
        // asserts the same drain here. Back-compat: a pre-#54 engine that omits the field
        // (None) is not asserted; a present non-zero fails and discards the session.
        // Checked BEFORE the completed-work barrier so an undrained cache is reported as
        // such regardless of the work count.
        if let Some(reported) = resp.cache_memory {
            if reported != 0 {
                self.discarded = true;
                return Err(RunnerError::AllocatorCacheNotDrained { reported }.logged());
            }
        }
        match resp.completed_work {
            Some(reported) if reported == self.issued_steps => {
                // C6: a conformant engine resets its completed-work counter after
                // phase_diagnostics, so the runner must too — otherwise a second phase
                // opened without an explicit begin_phase() carries a stale count and
                // spuriously fails an honest run. begin_phase()'s reset stays as
                // belt-and-suspenders.
                self.issued_steps = 0;
                Ok(resp)
            }
            other => {
                self.discarded = true;
                Err(RunnerError::CompletedWorkMismatch {
                    issued: self.issued_steps,
                    reported: other,
                }
                .logged())
            }
        }
    }

    /// The prologue both free-run phase closers run, VERBATIM and in this order: send
    /// `phase_diagnostics` (outside the timed window), assert the allocator drained, read the
    /// engine's `completed_work` counter, and reset the timed-step counter across the phase
    /// boundary. Every failure discards the session fail-closed, exactly as before.
    ///
    /// Returns the raw diagnostics response (the caller needs it for `peak_ram_gb`) and the
    /// `completed_work` counter the caller hands to its consistency check.
    fn close_free_run_prologue(&mut self) -> Result<(WorkerResponse, i64)> {
        let diag = self.phase_diagnostics_raw()?;
        // Allocator-drain assertion (#54), same fail-closed rule as `close_phase`.
        if let Some(reported) = diag.cache_memory {
            if reported != 0 {
                self.discarded = true;
                return Err(RunnerError::AllocatorCacheNotDrained { reported }.logged());
            }
        }
        let completed_work = match diag.completed_work {
            Some(cw) => cw,
            None => {
                self.discarded = true;
                return Err(RunnerError::FreeRunConsistency {
                    detail: "phase_diagnostics carried no completed_work counter".to_string(),
                });
            }
        };
        // Reset the timed-step counter across the phase boundary (parity with `close_phase`).
        self.issued_steps = 0;
        Ok((diag, completed_work))
    }

    /// Close a v1.1 free-run decode phase (PROTOCOL-v1.1.md §2.6): send `phase_diagnostics`
    /// (outside the timed window), assert the allocator drain, and enforce the consistency
    /// TRIPLE against the `free_decode_run` response counts (`resp`) and the requested N.
    ///
    /// Unlike [`Self::close_phase`], the barrier is NOT `completed_work == issued_steps` (the free-run
    /// kinds are not `is_timed_step`, so `issued_steps` is 0 here); instead the engine's
    /// `completed_work` must equal `R + 1` (seed forward + R verify rounds), cross-checked with
    /// `sum(acceptance_lengths) == N` and `committed_total == N == tokens.len()` via
    /// [`bench_core::free_run::verify_consistency`]. Any failure discards the session fail-closed.
    /// Returns the AUDIT view plus the raw diagnostics response (for `peak_ram_gb`).
    pub fn close_free_run_phase(
        &mut self,
        resp: &FreeRunResponse,
        n: u32,
    ) -> Result<(FreeRunAudit, WorkerResponse)> {
        let (diag, completed_work) = self.close_free_run_prologue()?;
        match verify_consistency(resp, n, completed_work) {
            Ok(audit) => Ok((audit, diag)),
            Err(e) => {
                self.discarded = true;
                Err(RunnerError::FreeRunConsistency {
                    detail: e.to_string(),
                }
                .logged())
            }
        }
    }

    /// Close a v1.2 COHORT free-run decode phase: send `phase_diagnostics` (outside the timed
    /// window), assert the allocator drain, and enforce the cohort consistency QUADRUPLE
    /// ([`bench_core::free_run::verify_cohort_consistency`]) against the batched
    /// `free_decode_run` response counts and the requested per-stream N.
    ///
    /// The barrier is the same shape as [`Self::close_free_run_phase`]'s: the free-run kinds are not
    /// `is_timed_step`, and the engine's `completed_work` must equal the SCALAR `R + 1` — a round
    /// is ONE engine forward regardless of B, so the counter does not scale with the cohort
    /// width. Any failure discards the session fail-closed. Returns the cohort AUDIT view plus
    /// the raw diagnostics response (for `peak_ram_gb`).
    pub fn close_batched_free_run_phase(
        &mut self,
        resp: &CohortFreeRunResponse,
        n: u32,
    ) -> Result<(CohortFreeRunAudit, WorkerResponse)> {
        let (diag, completed_work) = self.close_free_run_prologue()?;
        match verify_cohort_consistency(resp, n, completed_work) {
            Ok(audit) => Ok((audit, diag)),
            Err(e) => {
                self.discarded = true;
                Err(RunnerError::FreeRunConsistency {
                    detail: e.to_string(),
                }
                .logged())
            }
        }
    }
}

/// Read the next line that looks like a JSON response, decode it, or fail.
///
/// Skips non-JSON preamble lines (engine logs) exactly like Swift `readResponseLine`.
/// EOF before any JSON line is a protocol violation (Swift: "closed stdout before
/// returning a response").
///
/// H3 (cycle-3) — when `deadline` is `Some`, the WHOLE wait (across skipped preamble lines) is
/// bounded by it: if it passes before a JSON response arrives, benchd raises `RunTimeout` (the
/// caller `send` discards the session). `phase` names the request kind for the error. `deadline ==
/// None` blocks unbounded (the untimed default).
fn read_response_line<T: LineTransport>(
    transport: &mut T,
    deadline: Option<(std::time::Instant, f64)>,
    phase: &str,
) -> Result<WorkerResponse> {
    let deadline_at = deadline.map(|(at, _)| at);
    loop {
        match transport.read_line_deadline(deadline_at)? {
            ReadOutcome::TimedOut => {
                // A hung/looping engine did not respond within the RunTimeout budget.
                return Err(RunnerError::RunTimeout {
                    phase: phase.to_string(),
                    budget_seconds: deadline.map(|(_, b)| b).unwrap_or(0.0),
                });
            }
            ReadOutcome::Eof => {
                // #134 — the bare form of this message is what blocked Proof A: every leg died
                // here and the run record carried nothing that could localise it. Ask the
                // transport for its post-mortem (child wait status + retained redacted worker
                // stderr tail) and APPEND it, so the engine's own last words travel with the
                // error into whatever record seals it. The leading clause is unchanged, so
                // anything keying on this signature still matches.
                return Err(with_failure_diagnostic(
                    transport,
                    "engine closed the stream before returning a response".to_string(),
                ));
            }
            ReadOutcome::Line(line) => {
                if !line_looks_like_json_response(&line) {
                    continue;
                }
                // #134 — an engine that dies MID-LINE delivers a truncated final line (the reader
                // yields whatever preceded EOF), which looks like a response and then fails to
                // decode. That is the same "engine died on the way down" event as the EOF arm and
                // needs the same post-mortem: without it the operator sees only a bare serde
                // message. Reported as `Protocol` so the diagnostic can ride along — the
                // measure-job classifier maps both this and `Json` to `RejectClass::Infra`, so the
                // reject class is unchanged.
                match serde_json::from_str::<WorkerResponse>(&line) {
                    Ok(resp) => return Ok(resp),
                    // serde's Display quotes the offending value, which is engine-chosen text:
                    // only its category and position are benchd's own.
                    Err(e) => {
                        return Err(with_failure_diagnostic(
                            transport,
                            format!(
                                "engine response line could not be decoded: {}",
                                crate::error::serde_error_summary(&e)
                            ),
                        ));
                    }
                }
            }
        }
    }
}

/// Attach the transport's post-mortem (child wait status + retained worker stderr tail) to a
/// benchd-authored `detail`. The post-mortem is engine-controlled, so it rides as
/// [`RunnerError::ProtocolEngineText`]: sealed records get its digest, the log gets the text.
fn with_failure_diagnostic<T: LineTransport>(transport: &mut T, detail: String) -> RunnerError {
    match transport.failure_diagnostic() {
        Some(engine_text) => RunnerError::ProtocolEngineText {
            detail,
            engine_text,
        },
        None => RunnerError::Protocol(detail),
    }
}

/// (b) admission (engine #35/#37) — build the TRUSTED-ORACLE request: the per-stream seed context,
/// the candidate's committed journals, and the EXPLICIT replay width.
///
/// Split out from [`Session::cohort_reference_replay`] so the exact bytes benchd puts on the wire
/// are unit-testable without an engine: the width is an ENFORCED parameter of the reference the
/// tolerance gate judges against, and "we send it" has to be a tested fact, not a code-reading.
fn cohort_reference_replay_request(
    replay_seeds_by_stream: &[Vec<i64>],
    committed_by_stream: &[Vec<i64>],
) -> WorkerRequest {
    let mut req = WorkerRequest::new(0, RequestKind::CohortReferenceReplay.as_str());
    req.replay_seeds_by_stream = Some(replay_seeds_by_stream.to_vec());
    req.committed_by_stream = Some(committed_by_stream.to_vec());
    // PINNED, never the engine-side default — see `REPLAY_WIDTH_COHORT`.
    req.replay_width = Some(REPLAY_WIDTH_COHORT.to_string());
    req
}

/// (b) admission (engine #37) — the replay-width ECHO check, in the never-ignored spirit of
/// `effective_batch_size`: if the oracle STATES the width it ran at, it must be the width the
/// request pinned, or the report is refused fail-closed (a reference replayed at a geometry benchd
/// did not ask for is not the reference the tolerance gate is defined over).
///
/// PRESENCE-CONDITIONED, deliberately. Today's trusted engine parses `replay_width` on the request
/// but stamps nothing on the report, so `None` is the honest current state and is accepted: benchd
/// does not invent an echo the wire never carried. The engine-side stamp is a follow-up in flight;
/// when it lands, this assertion ARMS ITSELF with no benchd change. Until then the request-side pin
/// plus the engine's own refusal of unrecognized width values is what fixes the geometry.
fn assert_replay_width_echo(requested: &str, report: &CohortReferenceReplayReport) -> Result<()> {
    match report.replay_width.as_deref() {
        None => Ok(()),
        Some(echoed) if echoed == requested => Ok(()),
        Some(echoed) => Err(RunnerError::Protocol(format!(
            "cohort_reference_replay ran at replay_width '{echoed}' but the request pinned \
             '{requested}' — the trusted oracle's replay geometry is the reference the token \
             tolerance gate judges against; refusing the report rather than gating against a \
             geometry benchd did not request"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cohort_reference_replay_request_pins_the_cohort_width() {
        // The ENFORCED reference geometry rides on the request, explicitly. Before engine #37
        // benchd sent nothing here and the width came from an engine-side default — a parameter
        // benchd neither stated nor could verify.
        let req = cohort_reference_replay_request(&[vec![1, 2]], &[vec![10, 11]]);
        assert_eq!(req.kind, "cohort_reference_replay");
        assert_eq!(req.replay_width.as_deref(), Some("cohort"));
        let line = serde_json::to_string(&req).unwrap();
        assert!(
            line.contains(r#""replay_width":"cohort""#),
            "the wire line must carry the pinned width: {line}"
        );
    }

    #[test]
    fn replay_width_echo_refuses_a_divergent_geometry_and_tolerates_an_absent_one() {
        let report = |width: Option<&str>| CohortReferenceReplayReport {
            replay_width: width.map(str::to_string),
            ..Default::default()
        };
        // TODAY: no stamp on the report ⇒ accepted (benchd never invents an echo).
        assert!(assert_replay_width_echo(REPLAY_WIDTH_COHORT, &report(None)).is_ok());
        // WHEN THE STAMP LANDS: matching ⇒ accepted, diverging ⇒ refused by name. This is the
        // assertion arming itself with no further benchd change.
        assert!(assert_replay_width_echo(REPLAY_WIDTH_COHORT, &report(Some("cohort"))).is_ok());
        let err = assert_replay_width_echo(REPLAY_WIDTH_COHORT, &report(Some("canonical")))
            .expect_err("a divergent width must be refused fail-closed");
        let msg = err.to_string();
        assert!(msg.contains("canonical") && msg.contains("cohort"), "{msg}");
    }

    #[test]
    fn json_line_detection_matches_swift() {
        assert!(line_looks_like_json_response("{\"id\":0}"));
        assert!(line_looks_like_json_response("   \t {\"id\":0}"));
        assert!(!line_looks_like_json_response("loading weights..."));
        assert!(!line_looks_like_json_response(""));
        assert!(!line_looks_like_json_response("   "));
        assert!(!line_looks_like_json_response("[1,2,3]"));
    }
}
