//! Cross-repo wire crosscheck: benchd parses the FORK ENGINE's REAL encoder bytes.
//!
//! `src/fixtures/engine-wire-v1.jsonl` (embedded via `bench_runner::ENGINE_WIRE_V1_FIXTURE` so
//! the crosscheck also runs at measure time) is CAPTURED from the engine's own Codable
//! encoder (engine test `emitEngineWireFixture`, `.sortedKeys` canonical) — NOT a
//! hand-written literal, and UNTRIMMED: it carries every field the engine emits,
//! notably the `phase_diagnostics` `mlx_*` memory ints and the hello
//! `head_provenance` / `capabilities` / `spec_modes` that benchd's
//! `deny_unknown_fields` used to REJECT — the exact interop break #102 cycle-4
//! flagged, and which the prior trimmed-literal test hid. Both repos pin the same
//! sha256; regenerate via the engine's `emitEngineWireFixture` and repin BOTH repos
//! intentionally when the wire surface changes.
//!
//! #109 W3 finding 6 — the capture now also carries a `free_decode_begin` line, so the
//! PROTOCOL-v1.1 §2.2 begin/run SEAM is pinned in real encoder bytes rather than left as a
//! convention each side could drift on alone: `seed_token` is 699 and `free_decode_run`'s
//! `tokens[0]` is 700, the first of the N tokens that come AFTER it.

use bench_core::free_run::{
    verify_cohort_consistency, verify_consistency, CohortFreeRunResponse, FreeRunResponse,
};
use bench_protocol::{
    WorkerResponse, CAPABILITY_BATCHED_FREE_RUN_DECODE, CAPABILITY_PER_STREAM_TIMING,
};
use bench_runner::{
    captured_fixture_covers_cohort_wire, captured_fixture_covers_per_stream_timing,
    verify_captured_engine_wire, ENGINE_WIRE_V1_FIXTURE, ENGINE_WIRE_V1_SHA256,
};

/// Parse the sha-pinned captured fixture through the SAME crosscheck body the measure-job gate
/// uses (`verify_captured_engine_wire`): the captured bytes hash to the mirror-integrity
/// reference AND every line decodes into benchd's `WorkerResponse` under `deny_unknown_fields`.
fn captured_engine_lines() -> Vec<WorkerResponse> {
    verify_captured_engine_wire(ENGINE_WIRE_V1_FIXTURE.as_bytes(), ENGINE_WIRE_V1_SHA256).expect(
        "engine wire fixture drifted — regenerate via emitEngineWireFixture and repin BOTH repos",
    )
}

#[test]
fn engine_real_hello_and_phase_diagnostics_parse_into_benchd() {
    let lines = captured_engine_lines();
    assert_eq!(
        lines.len(),
        11,
        "fixture carries the 4 single-stream lines (hello + phase_diagnostics + \
         free_decode_begin + free_decode_run) plus the 4 v1.2 SERIAL cohort lines (batched \
         hello + batched begin + batched run + the cohort phase_diagnostics) plus the MTP \
         arm's mtp effective-spec echo line plus the round-execution lane's 2 v1.2 MTP cohort \
         lines (batched begin + batched run, spec mtp)"
    );

    // Line 0 — gated-on hello: the fields benchd's deny_unknown_fields used to REJECT.
    let hello = &lines[0];
    assert_eq!(hello.protocol_version, Some(1));
    assert_eq!(hello.backend.as_deref(), Some("mlx"));
    assert_eq!(
        hello.spec_modes.as_deref(),
        Some(["serial".to_string(), "mtp".to_string()].as_slice()),
        "the MTP arm (2026-08-23): this worker loaded an assistant head, so spec_modes \
         advertises mtp again"
    );
    assert!(hello
        .capabilities
        .as_deref()
        .is_some_and(|c| c.iter().any(|x| x == "free_run_decode")));
    let hp = hello
        .head_provenance
        .as_ref()
        .expect("hello carries head_provenance");
    assert_eq!(hp.bytes, 849_398_784); // the real reference MTP head's byte count
    assert_eq!(hp.file_count, 2);
    assert_eq!(hp.sha256.len(), 64);

    // Line 1 — phase_diagnostics: the FATAL case. The engine ALWAYS emits these
    // mlx_* ints; before #106 they were unmodeled and blew up every phase close.
    let phase = &lines[1];
    assert_eq!(phase.mlx_active_memory_bytes, Some(1));
    assert_eq!(phase.mlx_cache_memory_bytes, Some(0));
    assert_eq!(phase.mlx_peak_memory_bytes, Some(2));
    assert_eq!(phase.completed_work, Some(3)); // R+1 for the R=2 free-run below
    assert_eq!(phase.cache_memory, Some(0)); // drained
}

#[test]
fn engine_real_begin_run_seam_puts_the_run_after_the_seed() {
    // #109 W3 finding 6 — PROTOCOL-v1.1 §2.2, pinned in the engine's OWN encoder bytes:
    //
    //   free_decode_begin(seed_tokens)   -> seed_token
    //       verify seed_token == expected_decode_seed_token
    //   free_decode_run(count = N)       -> { tokens[N], … }
    //       for i in 0..N: require tokens[i] == expected_decode_tokens[i]
    //
    // The seed token is verified on its OWN line against its OWN oracle field; §2.1 says the begin
    // "establishes the last-committed state" and the run commits N MORE. So `tokens[0]` is the
    // token AFTER the seed — never the seed itself. Window 3's engine re-emitted it, which put
    // every following token one position late (0/16 against the golden; 16/16 under the shift).
    let lines = captured_engine_lines();
    let begin = &lines[2];
    let run = &lines[3];
    let seed = begin
        .seed_token
        .expect("free_decode_begin returns seed_token");
    assert_eq!(seed, 699);
    assert_eq!(
        begin.effective_spec.as_ref().map(|s| s.mode.as_str()),
        Some("serial"),
        "the gate-on begin echoes the spec it resolved (serial-only engine: the gemma \
         track's MTP arm is a deferred follow-up; repin this echo when it lands)"
    );
    let tokens = run
        .tokens
        .as_deref()
        .expect("free_decode_run returns tokens");
    assert_eq!(
        tokens[0], 700,
        "tokens[0] is expected_decode_tokens[0] — the token AFTER the seed"
    );
    assert!(
        !tokens.contains(&seed),
        "the run must not re-emit the token free_decode_begin already returned (#109 W3 finding 6)"
    );
    // And the §2.6 counters describe that same window: N committed tokens over R rounds, none of
    // them the seed (whose forward is the separate `+1` in `completed_work == R + 1`).
    assert_eq!(run.committed_total, Some(tokens.len() as u64));
    assert_eq!(lines[1].completed_work, Some(3)); // R=2 rounds + the seed forward
}

#[test]
fn engine_real_free_decode_run_satisfies_the_triple() {
    let lines = captured_engine_lines();
    let run = &lines[3];
    assert_eq!(run.tokens.as_deref(), Some([700, 701, 702, 703].as_slice()));
    assert_eq!(
        run.acceptance_lengths.as_deref(),
        Some([3u32, 1].as_slice())
    );
    assert_eq!(run.committed_total, Some(4));

    let fr = FreeRunResponse {
        tokens_len: run.tokens.as_ref().unwrap().len(),
        acceptance_lengths: run.acceptance_lengths.clone().unwrap(),
        drafted_total: run.drafted_total.unwrap(),
        accepted_total: run.accepted_total.unwrap(),
        committed_total: run.committed_total.unwrap(),
        verify_replay_disagreements: run.verify_replay_disagreements,
        verification: bench_core::free_run::VerificationReport::from_wire(
            run.verification_mode.as_deref(),
            run.rectangular_verification_rounds,
            run.serial_verification_rounds,
        ),
    };
    // N=4 committed, completed_work == R+1 == 3 (from the phase_diagnostics above).
    let audit = verify_consistency(&fr, 4, 3)
        .expect("engine free_decode_run satisfies benchd's §2.6 triple");
    assert_eq!(audit.rounds(), 2); // R == acceptance_lengths.len()
    assert_eq!(audit.acceptance_lengths(), &[3, 1]);
    assert_eq!(audit.verified_token_count(), 4); // committed_total == N == tokens.len()

    // Fail-closed: a wrong completed_work (not R+1) is rejected.
    assert!(verify_consistency(&fr, 4, 2).is_err());
}

#[test]
fn engine_real_batched_hello_advertises_the_cohort_surface() {
    // JOINT cohort extension (2026-08-23): line 4 is the engine's REAL gate-on hello with the
    // v1.2 cohort surface — the `batched_free_run_decode` capability plus the `max_batch_size`
    // ceiling benchd uses to refuse an over-wide cohort PRE-GPU. Parsed here under
    // deny_unknown_fields, which is the point: an engine field benchd's schema would reject is
    // caught in this capture, before any live session.
    let lines = captured_engine_lines();
    let hello = &lines[4];
    assert_eq!(hello.protocol_version, Some(1));
    assert_eq!(
        hello.spec_modes.as_deref(),
        Some(["serial".to_string(), "mtp".to_string()].as_slice()),
        "the batched hello shares the single-stream hello's registry (decided once at \
         startup) — it advertises mtp too, even though the cohort driver's round EXECUTION \
         does not accept it yet (see the mtp echo test below vs. this hello's advertisement)"
    );
    assert!(
        hello
            .capabilities
            .as_deref()
            .is_some_and(|c| c.iter().any(|x| x == CAPABILITY_BATCHED_FREE_RUN_DECODE)),
        "the batched hello advertises batched_free_run_decode"
    );
    assert!(
        hello
            .capabilities
            .as_deref()
            .is_some_and(|c| c.iter().any(|x| x == "free_run_decode")),
        "the cohort form is v1.1-plus-width: the batched hello still advertises free_run_decode"
    );
    assert_eq!(
        hello.max_batch_size,
        Some(8),
        "the engine's cohort-width ceiling (runtimeWorkerMaxCohortBatchSize, the ruled B=8)"
    );
    assert!(
        hello
            .capabilities
            .as_deref()
            .is_some_and(|c| c.iter().any(|x| x == CAPABILITY_PER_STREAM_TIMING)),
        "per-stream timing instrumentation (spec step 1): the batched hello advertises \
         per_stream_timing alongside batched_free_run_decode"
    );
}

#[test]
fn engine_real_batched_begin_run_seam_and_cohort_quadruple() {
    // The cohort begin/run seam in REAL encoder bytes (per slot, the batched generalization of
    // the single-stream §2.2 seam): each slot's committed row starts AFTER that slot's seed and
    // never re-emits it. Serial closed cohort B=2, N=3, assembled by the engine's own
    // `assembleSerialCohortFreeRun`.
    let lines = captured_engine_lines();
    let begin = &lines[5];
    let run = &lines[6];
    let phase = &lines[7];

    let seeds = begin
        .seed_token_by_stream
        .as_deref()
        .expect("batched free_decode_begin returns seed_token_by_stream");
    assert_eq!(seeds, [800, 900], "B seed-forward argmaxes in SLOT ORDER");
    assert_eq!(
        begin.effective_spec.as_ref().map(|s| s.mode.as_str()),
        Some("serial"),
        "the batched begin echoes the resolved spec (serial-only engine in this increment)"
    );
    assert_eq!(
        begin.effective_batch_size,
        Some(2),
        "the never-ignored width echo rides on the batched begin"
    );
    assert_eq!(
        begin.prefill_ns_by_stream.as_deref(),
        Some([50_000u64, 70_000].as_slice()),
        "per-stream timing instrumentation: per-slot cohort-prefill elapsed ns, SLOT ORDER"
    );

    let rectangle = run
        .tokens_by_stream
        .as_deref()
        .expect("batched free_decode_run returns tokens_by_stream");
    assert_eq!(rectangle, [vec![801, 802, 803], vec![901, 902, 903]]);
    for (slot, row) in rectangle.iter().enumerate() {
        assert!(
            !row.contains(&seeds[slot]),
            "slot {slot} must not re-emit the seed its batched begin already returned"
        );
    }
    assert_eq!(run.effective_batch_size, Some(2));
    assert_eq!(
        run.decode_ns_by_stream.as_deref(),
        Some([320_000u64, 340_000].as_slice()),
        "per-stream timing instrumentation: per-slot decode-phase elapsed ns, SLOT ORDER"
    );

    // The cohort QUADRUPLE (`verify_cohort_consistency`) over the CAPTURED counters, with the
    // captured phase line's `completed_work` — the same shape the runner enforces live. N=3 per
    // stream; the serial cohort's R == N.
    let resp = CohortFreeRunResponse {
        batch_size: run.effective_batch_size.unwrap(),
        tokens_len_by_stream: rectangle.iter().map(Vec::len).collect(),
        acceptance_lengths: run.acceptance_lengths.clone().unwrap(),
        natural_accepted_by_stream: run.natural_accepted_by_stream.clone().unwrap(),
        active_streams_by_round: run.active_streams_by_round.clone().unwrap(),
        rounds: run.rounds.unwrap(),
        drafted_total: run.drafted_total.unwrap(),
        accepted_total: run.accepted_total.unwrap(),
        committed_total: run.committed_total.unwrap(),
        depth_clamp_reasons: run
            .depth_clamp_reasons
            .clone()
            .unwrap()
            .into_iter()
            .map(|(key, count)| (key, u64::from(count)))
            .collect(),
    };
    let completed_work = phase
        .completed_work
        .expect("the cohort phase_diagnostics line pins completed_work");
    assert_eq!(completed_work, 4, "SCALAR R + 1 for the R=3 serial cohort");
    let audit = verify_cohort_consistency(&resp, 3, completed_work)
        .expect("the captured cohort lines satisfy benchd's consistency QUADRUPLE");
    assert_eq!(audit.rounds(), 3);
    assert_eq!(audit.base().acceptance_lengths(), &[1, 1, 1]);

    // Serial totals: nothing drafted, nothing accepted-from-a-draft, committed == B * N.
    assert_eq!(run.drafted_total, Some(0));
    assert_eq!(run.accepted_total, Some(0));
    assert_eq!(run.committed_total, Some(6));
    assert!(
        run.depth_clamp_reasons.as_ref().unwrap().is_empty(),
        "the serial route clamps nothing — sealed EMPTY, not omitted"
    );

    // Fail-closed: a wrong completed_work (not R+1) is rejected.
    assert!(verify_cohort_consistency(&resp, 3, 3).is_err());
}

#[test]
fn engine_real_mtp_effective_spec_echo_parses_into_benchd() {
    // The MTP arm (2026-08-23): line 8 is a decode_begin-shaped response exercising the REAL
    // {"mtp":{"depth":...}} effective-spec echo (RuntimeWorkerSpecRegistry.resolveEffectiveSpec,
    // not a hand-built literal). Depth 3 is the engine's pinned ceiling
    // (Gemma4MTPEnvelope.maxDraftTokens) — the captured request carried no explicit depth, so it
    // resolved to "use everything this arm supports". This particular capture point is
    // decode_begin's echo (route-independent of round execution by construction); the ROUND
    // EXECUTION coverage this comment used to say was missing is now the two tests below.
    let lines = captured_engine_lines();
    let mtp_line = &lines[8];
    let spec = mtp_line
        .effective_spec
        .as_ref()
        .expect("the mtp line carries an effective_spec");
    assert_eq!(spec.mode, "mtp");
    assert_eq!(
        spec.mtp.as_ref().and_then(|m| m.depth),
        Some(3),
        "no depth was requested, so the echo carries the pinned ceiling"
    );
}

#[test]
fn engine_real_batched_mtp_begin_run_seam_and_cohort_quadruple() {
    // Round execution (2026-08-23): lines 9/10 are the batched (v1.2 cohort) free_decode_begin /
    // free_decode_run pair at spec `mtp`, assembled by the engine's own
    // `assembleMTPCohortFreeRun` over a representative per-slot round history — B=2, N=3, the
    // SAME shape as the serial cohort pair (lines 5/6) above but through the mtp route. This is
    // the batched-mtp line PR #9's fixture comment named as deliberately absent ("the cohort
    // driver's round execution has no drafter binding yet, so a batched mtp line would capture
    // bytes the engine cannot actually produce") — it now exists because that binding does.
    let lines = captured_engine_lines();
    let begin = &lines[9];
    let run = &lines[10];

    let seeds = begin
        .seed_token_by_stream
        .as_deref()
        .expect("batched mtp free_decode_begin returns seed_token_by_stream");
    assert_eq!(seeds, [800, 900], "B seed-forward argmaxes in SLOT ORDER");
    assert_eq!(
        begin.effective_spec.as_ref().map(|s| s.mode.as_str()),
        Some("mtp"),
        "the batched begin echoes the resolved mtp spec — this is the line that used to be \
         refused (\"batched free-run mtp round execution is not yet wired\")"
    );
    assert_eq!(begin.effective_batch_size, Some(2));
    assert_eq!(
        begin.prefill_ns_by_stream.as_deref(),
        Some([60_000u64, 80_000].as_slice()),
        "per-stream timing instrumentation, mtp leg: per-slot cohort-prefill elapsed ns"
    );

    let rectangle = run
        .tokens_by_stream
        .as_deref()
        .expect("batched mtp free_decode_run returns tokens_by_stream");
    assert_eq!(rectangle, [vec![801, 802, 803], vec![901, 902, 903]]);
    for (slot, row) in rectangle.iter().enumerate() {
        assert!(
            !row.contains(&seeds[slot]),
            "slot {slot} must not re-emit the seed its batched begin already returned"
        );
    }
    assert_eq!(run.effective_batch_size, Some(2));
    assert_eq!(
        run.decode_ns_by_stream.as_deref(),
        Some([410_000u64, 430_000].as_slice()),
        "per-stream timing instrumentation, mtp leg: per-slot decode-phase elapsed ns"
    );

    let resp = CohortFreeRunResponse {
        batch_size: run.effective_batch_size.unwrap(),
        tokens_len_by_stream: rectangle.iter().map(Vec::len).collect(),
        acceptance_lengths: run.acceptance_lengths.clone().unwrap(),
        natural_accepted_by_stream: run.natural_accepted_by_stream.clone().unwrap(),
        active_streams_by_round: run.active_streams_by_round.clone().unwrap(),
        rounds: run.rounds.unwrap(),
        drafted_total: run.drafted_total.unwrap(),
        accepted_total: run.accepted_total.unwrap(),
        committed_total: run.committed_total.unwrap(),
        depth_clamp_reasons: run
            .depth_clamp_reasons
            .clone()
            .unwrap()
            .into_iter()
            .map(|(key, count)| (key, u64::from(count)))
            .collect(),
    };
    // completed_work isn't separately captured for this pair (no dedicated phase_diagnostics
    // line rides with it in this fixture); R=2 rounds (acceptance_lengths [1, 2]) so
    // completed_work == R + 1 == 3 is what a live session's phase close would report.
    let audit = verify_cohort_consistency(&resp, 3, 3)
        .expect("the captured batched mtp lines satisfy benchd's consistency QUADRUPLE");
    assert_eq!(audit.rounds(), 2);
    assert_eq!(audit.base().acceptance_lengths(), &[1, 2]);

    // MTP totals: real drafting happened (drafted >= accepted > 0), unlike the serial pair's
    // all-zero totals, and the depth-clamp histogram is NON-EMPTY — a speculative cohort can
    // genuinely clamp depth, which the serial route never can (it clamps nothing to clamp).
    assert_eq!(run.drafted_total, Some(5));
    assert_eq!(run.accepted_total, Some(3));
    assert_eq!(run.committed_total, Some(6));
    assert_eq!(
        run.depth_clamp_reasons.as_ref().unwrap().get("tail_depth"),
        Some(&1)
    );

    // Fail-closed: a wrong completed_work (not R+1) is rejected, exactly as the serial pair's
    // equivalent check.
    assert!(verify_cohort_consistency(&resp, 3, 2).is_err());
}

#[test]
fn captured_fixture_cohort_coverage_holds() {
    // The angle-6 interlock's data prerequisite, asserted where the fixture lives: the embedded
    // capture COVERS the cohort wire, so `ScoredBatchPoint::certify(8)` (benchd) can pass its
    // coverage gate. If this ever regresses (a repin that drops the cohort lines), the B=8
    // scored window refuses structurally — that refusal is the ruling, this test just makes the
    // regression loud at `cargo test` time too.
    assert!(captured_fixture_covers_cohort_wire());
}

#[test]
fn captured_fixture_per_stream_timing_coverage_holds() {
    // Per-stream timing instrumentation's own (separate) coverage precondition, asserted where
    // the fixture lives: the embedded capture carries the per_stream_timing capability plus a
    // prefill_ns_by_stream and a decode_ns_by_stream line, so the attestation module has real
    // per-slot data to compute over.
    assert!(captured_fixture_covers_per_stream_timing());
}

#[test]
fn benchd_serial_free_run_triple_is_all_ones() {
    // The honest serial control (constructed, not a wire literal): each round commits
    // exactly one token, so acceptance_lengths is [1]*N, R == N, completed_work == N+1.
    let fr = FreeRunResponse {
        tokens_len: 8,
        acceptance_lengths: vec![1; 8],
        drafted_total: 0,
        accepted_total: 0,
        committed_total: 8,
        verify_replay_disagreements: None,
        verification: None,
    };
    let audit = verify_consistency(&fr, 8, 9).expect("serial triple holds (R=N=8, completed=9)");
    assert_eq!(audit.rounds(), 8);
    assert_eq!(audit.acceptance_lengths(), &[1, 1, 1, 1, 1, 1, 1, 1]);
    assert!(verify_consistency(&fr, 8, 8).is_err()); // wrong completed_work fails closed
}
