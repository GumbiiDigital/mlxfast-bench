//! Captured engine-wire crosscheck — benchd's INDEPENDENT reference of what the real engine
//! emits, pinned by a sha256 both repos hold, re-parsed under benchd's own CLOSED
//! [`WorkerResponse`].
//!
//! The captured bytes ([`ENGINE_WIRE_V1_FIXTURE`]) are CAPTURED from the engine's own Codable
//! encoder (engine test `emitEngineWireFixture`, `.sortedKeys` canonical) — NOT a hand-written
//! literal — and UNTRIMMED: they carry every field the engine emits (the `phase_diagnostics`
//! `mlx_*` memory ints, the hello `head_provenance` / `capabilities` / `spec_modes`, the
//! `free_decode_begin` seam). Both repos pin the same [`ENGINE_WIRE_V1_SHA256`]; regenerate via
//! the engine's `emitEngineWireFixture` and repin BOTH repos intentionally when the wire surface
//! changes.
//!
//! [`verify_captured_engine_wire`] is the ONE crosscheck body: it hashes the captured bytes
//! against the mirror-integrity reference sha256, then re-parses every line under benchd's
//! `deny_unknown_fields` [`WorkerResponse`]. It is shared by the cargo-test crosscheck and by
//! `benchd measure-job` (which previously ran neither at measure time), so a single definition
//! backs both the offline assertion and the pre-GPU gate.

use bench_core::hash::sha256_hex;
use bench_protocol::{
    WorkerResponse, CAPABILITY_BATCHED_FREE_RUN_DECODE, CAPABILITY_PER_STREAM_TIMING,
};

/// The sha256 of [`ENGINE_WIRE_V1_FIXTURE`] — the MIRROR-INTEGRITY REFERENCE both repos pin the
/// captured engine-wire bytes to. Referenced by symbol everywhere; the literal lives only here.
///
/// REPINNED 2026-08-23 (per-stream timing instrumentation spec step 1, engine PR
/// lane/gemma4-per-stream-timing): line count stays 11 — no line added or removed. The gate-on
/// hello (line 5) gains `per_stream_timing` in its `capabilities` list; the batched cohort begin
/// lines (6, 10) each gain `prefill_ns_by_stream`; the batched cohort run lines (7, 11) each gain
/// `decode_ns_by_stream`. Every other line, and every other field on 6/7/10/11, is
/// BYTE-IDENTICAL to the prior repin (`718799e3...`). Both new vectors are derived through the
/// engine's own `commitTimestampNs` pure helper over representative synthetic per-slot commit
/// histories (engine fixture generator is model-free by design), not hand-typed ns deltas.
/// Reproduce: engine `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test
/// --force-resolved-versions --filter emitEngineWireFixture` (3031 bytes).
pub const ENGINE_WIRE_V1_SHA256: &str =
    "4d7e2657b801d23eb20b3d9aecbcd2a543ce83b56eea326552515ed1fe323a7f";

/// The captured engine-wire reference bytes, EMBEDDED into the library so the crosscheck can run
/// at measure time (`benchd measure-job`) and not only under `cargo test`.
pub const ENGINE_WIRE_V1_FIXTURE: &str = include_str!("fixtures/engine-wire-v1.jsonl");

/// Crosscheck CAPTURED engine-wire `bytes` against the mirror-integrity `reference_sha256`, then
/// re-parse every non-empty line under benchd's CLOSED [`WorkerResponse`] (`deny_unknown_fields`).
///
/// Fail-closed, in order:
///   1. the sha256 of `bytes` MUST equal `reference_sha256` (the captured bytes are the ones both
///      repos pinned) — a disagreement here is the tampered / drifted-capture case;
///   2. every line MUST decode into `WorkerResponse` under `deny_unknown_fields` — a field the
///      engine emits that benchd's schema would reject (or vice-versa) is caught here.
///
/// Returns the parsed lines on success so a caller that also wants to assert their contents does
/// not re-parse. The measure-job gate discards them; the offline crosscheck test asserts on them.
pub fn verify_captured_engine_wire(
    bytes: &[u8],
    reference_sha256: &str,
) -> Result<Vec<WorkerResponse>, String> {
    let actual = sha256_hex(bytes);
    if !actual.eq_ignore_ascii_case(reference_sha256.trim()) {
        return Err(format!(
            "captured engine-wire bytes hash {actual} disagree with the mirror-integrity reference \
             {reference_sha256}"
        ));
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|e| format!("captured engine-wire bytes are not valid utf-8: {e}"))?;
    let mut parsed = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let resp = serde_json::from_str::<WorkerResponse>(line).map_err(|e| {
            format!(
                "captured engine-wire line rejected by benchd WorkerResponse \
                 (deny_unknown_fields): {}",
                crate::error::serde_error_summary(&e)
            )
        })?;
        parsed.push(resp);
    }
    Ok(parsed)
}

/// Whether the EMBEDDED captured fixture actually carries the v1.2 BATCHED (cohort) wire lines —
/// the coverage the B=8 scored window is interlocked on (orchestrator angle-6 BINDING ruling:
/// "no scored window before the cohort crosscheck exists" is a STRUCTURAL refusal, not a
/// schedule). `ScoredBatchPoint::certify` (benchd `measure_job`) consults this and refuses
/// `scored_batch_size = 8` when it returns `false`.
///
/// Fail-closed on every branch: a fixture that does not verify against its pin (or does not
/// parse under the CLOSED `WorkerResponse`) covers NOTHING. Coverage requires all three of:
///
/// 1. a gate-on hello advertising [`CAPABILITY_BATCHED_FREE_RUN_DECODE`] WITH `max_batch_size`
///    (the pre-GPU width-refusal surface);
/// 2. a batched `free_decode_begin` line (`seed_token_by_stream` + the never-ignored
///    `effective_batch_size` echo);
/// 3. a batched `free_decode_run` line (the `tokens_by_stream` rectangle plus the cohort AUDIT
///    vectors the consistency QUADRUPLE consumes: `natural_accepted_by_stream`, `rounds`,
///    `active_streams_by_round`, `depth_clamp_reasons`).
///
/// Deliberately UNCHANGED by per-stream timing instrumentation (spec step 1): this is the
/// existing B=8 scored-window interlock (`ScoredBatchPoint::certify`), and the enforced whole-
/// window metric it gates does not change this increment (spec binding constraint #2). Per-
/// stream-timing coverage is a SEPARATE, additional check —
/// [`captured_fixture_covers_per_stream_timing`] below — consulted only by the new (report-only)
/// attestation module, never by this scored-window gate. Coupling the two would make an engine
/// that has not yet landed per-stream timing unable to certify a B=8 window it could certify
/// today, which is exactly the report-only module reaching backward into enforcement this spec
/// forbids.
pub fn captured_fixture_covers_cohort_wire() -> bool {
    let lines =
        match verify_captured_engine_wire(ENGINE_WIRE_V1_FIXTURE.as_bytes(), ENGINE_WIRE_V1_SHA256)
        {
            Ok(lines) => lines,
            Err(_) => return false,
        };
    let covers_hello = lines.iter().any(|l| {
        l.max_batch_size.is_some()
            && l.capabilities
                .as_deref()
                .is_some_and(|c| c.iter().any(|x| x == CAPABILITY_BATCHED_FREE_RUN_DECODE))
    });
    let covers_begin = lines
        .iter()
        .any(|l| l.seed_token_by_stream.is_some() && l.effective_batch_size.is_some());
    let covers_run = lines.iter().any(|l| {
        l.tokens_by_stream.is_some()
            && l.natural_accepted_by_stream.is_some()
            && l.rounds.is_some()
            && l.active_streams_by_round.is_some()
            && l.depth_clamp_reasons.is_some()
    });
    covers_hello && covers_begin && covers_run
}

/// Whether the EMBEDDED captured fixture carries the per-stream timing instrumentation wire
/// lines (spec step 1): a hello advertising [`CAPABILITY_PER_STREAM_TIMING`], a batched begin
/// line carrying `prefill_ns_by_stream`, and a batched run line carrying `decode_ns_by_stream`.
///
/// This is the coverage precondition for the ATTESTATION module (`attestation.rs`) to have real
/// per-slot data to compute over — deliberately SEPARATE from [`captured_fixture_covers_cohort_wire`]
/// (the existing, unrelated B=8 scored-window interlock; see its doc comment for why the two must
/// not be coupled). Fail-closed on every branch, same posture as its sibling: a fixture that does
/// not verify against its pin, or does not parse, covers nothing.
pub fn captured_fixture_covers_per_stream_timing() -> bool {
    let lines =
        match verify_captured_engine_wire(ENGINE_WIRE_V1_FIXTURE.as_bytes(), ENGINE_WIRE_V1_SHA256)
        {
            Ok(lines) => lines,
            Err(_) => return false,
        };
    let covers_hello = lines.iter().any(|l| {
        l.capabilities
            .as_deref()
            .is_some_and(|c| c.iter().any(|x| x == CAPABILITY_PER_STREAM_TIMING))
    });
    let covers_begin = lines.iter().any(|l| l.prefill_ns_by_stream.is_some());
    let covers_run = lines.iter().any(|l| l.decode_ns_by_stream.is_some());
    covers_hello && covers_begin && covers_run
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_reference_matches_its_pinned_sha_and_parses() {
        let lines =
            verify_captured_engine_wire(ENGINE_WIRE_V1_FIXTURE.as_bytes(), ENGINE_WIRE_V1_SHA256)
                .expect(
                "the embedded captured reference matches its pin and parses under WorkerResponse",
            );
        assert_eq!(lines.len(), 11);
    }

    #[test]
    fn embedded_reference_covers_the_cohort_wire() {
        // The interlock's positive half: the 2026-08-23 capture carries the batched hello,
        // begin and run lines, so the coverage check the B=8 certify consults passes on the
        // real embedded fixture.
        assert!(
            captured_fixture_covers_cohort_wire(),
            "the embedded captured fixture must carry the v1.2 cohort lines \
             (batched hello + begin + run) — the B=8 scored window is interlocked on this"
        );
    }

    #[test]
    fn embedded_reference_covers_per_stream_timing() {
        // Per-stream timing instrumentation's own (separate) coverage check passes on the
        // current embedded fixture — the per-stream-timing repin landed all three pieces
        // (hello capability, begin vector, run vector) together.
        assert!(
            captured_fixture_covers_per_stream_timing(),
            "the embedded captured fixture must carry the per-stream timing lines \
             (per_stream_timing capability + prefill_ns_by_stream + decode_ns_by_stream)"
        );
    }

    #[test]
    fn tampered_bytes_fail_the_sha_gate_before_any_parse() {
        // Tamper a byte INSIDE a string value (the nonce): every line stays valid JSON and still
        // decodes under `WorkerResponse`, so ONLY the sha gate can reject these bytes — proving the
        // sha crosscheck fires independently of the parse crosscheck.
        let tampered = ENGINE_WIRE_V1_FIXTURE.replace("session-nonce", "session-xonce");
        assert_ne!(sha256_hex(tampered.as_bytes()), ENGINE_WIRE_V1_SHA256);
        let err = verify_captured_engine_wire(tampered.as_bytes(), ENGINE_WIRE_V1_SHA256)
            .expect_err("a captured-bytes sha disagreement must be refused");
        assert!(
            err.contains("mirror-integrity reference"),
            "the sha gate must name the reference: {err}"
        );
    }
}
