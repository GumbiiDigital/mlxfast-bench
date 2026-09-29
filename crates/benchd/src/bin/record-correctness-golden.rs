//! `record-correctness-golden` — produce a valid hidden-correctness [`GoldenDocument`] by driving
//! the engine adapter GREEDY teacher-forced (temperature 0) over one or more 1024-token prompts.
//!
//! WHAT IT PRODUCES. A `bench_core::golden::GoldenDocument` — the exact object a track contract pins
//! as its `hidden_correctness_golden` (sha256 + bytes). The document's shape and semantics are the
//! authority of `crates/bench-core/src/golden.rs`:
//! * `version` — always `1`;
//! * `model_type` — always [`bench_core::constants::REQUIRED_GOLDEN_MODEL_TYPE`]
//!   (`"qwen4_exp_text"`), the reference model
//!   family this track scores;
//! * `model_provenance` — the pinned reference MODEL identity (`repository` + 40-hex `revision`),
//!   defaulting to the qwen3.8-125b-a6b CUDA target pin;
//! * `cases[]` — one entry per prompt: `{name, prompt_tokens(1024), expected_tokens(--steps)}`.
//!
//! HOW IT DRIVES THE ENGINE. `expected_tokens` is the GREEDY teacher-forced continuation of the
//! prompt, produced by the correctness gate verbs (`correctness_begin` / `correctness_step`, the
//! teacher-forced `Step {token, top_logits[8]}` API — the same verbs `record-reference-tape` drives).
//! `correctness_begin(prompt)` yields `expected_tokens[0]` (the seed argmax = the first emitted
//! token), then each `correctness_step(prev)` yields the next token; feeding each greedy pick forward
//! IS the greedy continuation. This reuses `measure-noop`'s spawn/protocol machinery: a
//! freshly-spawned engine ([`ChildStdioTransport`], or the in-process mock under `--backend mock`)
//! driven through a [`bench_runner::Session`]. Each case gets its OWN cold engine so no case's KV
//! state leaks into the next — the same "fresh spawn per pass" discipline the tape recorder uses.
//!
//! MIRRORS THE GEMMA HIDDEN-CORRECTNESS GOLDEN. The gemma track ships
//! `correctness_prompts/public_longcopy_gate_english_1024_{256,1024}.json`: version 1, one case named
//! `longcopy-gate-english-1024`, `prompt_tokens` length 1024, `expected_tokens` length 256 (public)
//! or 1024 (full-window), `model_provenance` present. This tool emits the SAME shape for the qwen
//! target: same case count/name (operator-supplied), same 1024 prompt length, `expected_tokens`
//! length `--steps` (default 1024, the full-window variant), differing ONLY in the tokens (qwen, not
//! gemma) and the `model_provenance` (the qwen target pin). bench-core enforces only a MINIMUM
//! `expected_tokens` length ([`CORRECTNESS_STEPS`] = 64, `expected_tokens.len() >= CORRECTNESS_STEPS`),
//! not an exact one, so both the 256- and 1024-step variants load; `--steps` picks which.
//!
//! VALIDATION BEFORE FINALIZE. The emitted JSON is loaded back through
//! [`bench_core::golden::load_golden_fixture`] — with [`CORRECTNESS_STEPS`],
//! [`bench_core::constants::CORRECTNESS_PROMPT_TOKENS`], the required model_type, and the
//! provenance pin — BEFORE `--out` is
//! written; a golden that would not pass the scoring loader (or would not attest against the pin) is
//! never produced.
//!
//! DETERMINISM. Greedy temp-0 generation over the same prompts and engine is deterministic, and the
//! serialization is deterministic (fixed field order, trailing newline), so two records of the same
//! prompts are BYTE-IDENTICAL — the mechanical review artifact the tests pin.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use bench_core::constants::{BENCHMARK_DECODE_STEPS, CORRECTNESS_STEPS};
use bench_core::free_run::FreeRunResponse;
use bench_core::golden::{
    load_golden_fixture, BenchmarkGolden, GoldenCase, GoldenDocument, GoldenModelProvenance,
    ReferenceModelPin,
};
use bench_core::hash::sha256_hex;
use bench_runner::{ChildStdioTransport, LineTransport, Session};

/// Default `expected_tokens` length: the full-window gemma variant
/// (`public_longcopy_gate_english_1024_1024.json`), so the hidden oracle covers the full scored
/// decode trajectory with margin. `--steps 256` produces the public-length variant; any value
/// `>= CORRECTNESS_STEPS` (64) loads.
const DEFAULT_STEPS: usize = 1_024;

/// Default `--benchmark-steps`: the TIMED decode window benchd scores FOR THIS TRACK — the
/// official / local-iterate decode length ([`BENCHMARK_DECODE_STEPS`] = 128, which
/// [`bench_core::constants::LOCAL_ITERATE_BENCHMARK_DECODE_STEPS`] equals; see
/// `iterate.rs` `Mode::decode_steps`). The captured decode-seconds-per-token is the official
/// baseline DENOMINATOR, so the free-run oracle must cover exactly the window that denominator is
/// measured over — NOT the 1024-token correctness `--steps`. A shorter oracle cannot verify the
/// window it must gate, so [`BENCHMARK_DECODE_STEPS`] is also the enforced FLOOR (below).
const DEFAULT_BENCHMARK_STEPS: usize = BENCHMARK_DECODE_STEPS;

/// The qwen3.8-125b-a6b CUDA track's pinned reference MODEL (target). Defaults for
/// `--model-provenance-repo` / `--model-provenance-rev`; mirrors
/// `fixtures/qwen3_8_125b_a6b_track.json`'s `target.upstream_model_id` / `upstream_revision`.
const DEFAULT_PROVENANCE_REPO: &str = "RadixArk/Qwen3.8-Flash-Next-NVFP4";
const DEFAULT_PROVENANCE_REV: &str = "7b719225242aacd3dbd3f9407468c2ee9a9d2594";

// ----------------------------------------------------------------------------------------------
// CLI
// ----------------------------------------------------------------------------------------------

/// How the engine is driven.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    /// Spawn `--worker-bin` (the cuda-engine adapter) and speak the protocol over its stdio. The
    /// real, box path (a `--features vllm-engine` build drives live vLLM; a default build drives the
    /// adapter's own mock backend).
    Live,
    /// Drive the in-process deterministic mock engine ([`bench_runner::mock::MockEngine`]) — no
    /// binary, no GPU. Exercises the identical spawn/protocol machinery for tests and local smoke.
    Mock,
}

/// One prompt to record: its 1024 token ids plus the case name it becomes in the golden.
#[derive(Debug)]
struct CaseInput {
    name: String,
    prompt_tokens: Vec<i64>,
}

#[derive(Debug)]
struct Args {
    backend: Backend,
    worker_bin: Option<String>,
    weights: Option<PathBuf>,
    cases: Vec<CaseInput>,
    /// The MODEL IDENTITY of the track this golden is recorded for
    /// (`bench_core::constants::model_identity`): the `model_type` the document declares, the
    /// vocabulary its ids must fall in, and the prompt length each case must carry. Resolved from
    /// `--track` / `MLXFAST_QWEN_MTP_TRACK_ID` — never a compile-time default.
    identity: bench_core::constants::TrackModelIdentity,
    steps: usize,
    /// The tokens that end an answer (`--stop-token`). A case stops at the first of them, and
    /// `steps` is then the maximum length. Empty: every case is exactly `steps` long.
    stop_tokens: Vec<i64>,
    /// When set, ALSO author the golden's `benchmark` block by driving the engine's FREE-RUN
    /// trajectory (`free_decode_begin` / `free_decode_run`) over `cases[0]`'s prompt — the
    /// author-with-what-you-replay fix (a8/David ruling): the TIMED leg replays free-run, so its
    /// oracle must be authored free-run, not teacher-forced.
    benchmark_free_run: bool,
    /// The free-run oracle length N (`benchmark.expected_decode_tokens`). Defaults to
    /// [`DEFAULT_BENCHMARK_STEPS`]; must be `>= BENCHMARK_DECODE_STEPS` (the scored decode window).
    benchmark_steps: usize,
    provenance_repo: String,
    provenance_rev: String,
    out: PathBuf,
}

const USAGE: &str = "\
record-correctness-golden — produce a hidden-correctness GoldenDocument by driving the engine
adapter GREEDY teacher-forced (temp=0) over one or more 1024-token prompts.

USAGE:
    record-correctness-golden --worker-bin <PATH> --weights <TARGET_DIR> \\
        --prompt-tokens <FILE> --case-name <NAME> [--prompt-tokens <FILE> --case-name <NAME> ...] \\
        [--steps 1024] [--stop-token <ID> ...] [--benchmark-free-run [--benchmark-steps 128]] \\
        --out <GOLDEN.json> [--backend live|mock] \\
        [--model-provenance-repo <REPO>] [--model-provenance-rev <40-HEX>]

FLAGS:
    --worker-bin <PATH>   cuda-engine adapter executable (spawned as `<bin> runtime-worker
                          --weights <DIR>`; the adapter ignores argv). Required for --backend live.
    --weights <DIR>       TARGET (backbone) weights directory. Required for --backend live.
    --prompt-tokens <FILE>  File of token ids (JSON array or whitespace/comma-separated ints). Must
                          carry exactly CORRECTNESS_PROMPT_TOKENS (1024) ids. Repeatable; each must
                          be paired with a --case-name.
    --case-name <NAME>    The golden case name for the preceding --prompt-tokens. Repeatable.
    --steps <N>           expected_tokens length per case (default 1024). Must be >= 64
                          (CORRECTNESS_STEPS). With --stop-token it is the MAXIMUM length.
    --stop-token <ID>     A token that ends an answer. Repeatable. A case stops at the first stop
                          token, which is kept as its last expected token. An answer that ends
                          before CORRECTNESS_STEPS, or inside the free-run window, is refused.
    --benchmark-free-run  ALSO author the golden's `benchmark` block by driving the engine FREE-RUN
                          (free_decode_begin/free_decode_run) over cases[0]'s prompt, so the TIMED
                          leg's oracle matches the free-run it replays (author-with-what-you-replay).
    --benchmark-steps <N> free-run oracle length (default 128 = BENCHMARK_DECODE_STEPS, the SCORED
                          decode window benchd times FOR THIS TRACK — the official baseline
                          denominator; equals LOCAL_ITERATE_BENCHMARK_DECODE_STEPS). Must be
                          >= BENCHMARK_DECODE_STEPS: an oracle shorter than the window it must verify
                          is invalid. Only meaningful with --benchmark-free-run.
    --track <TRACK-ID>    The track this golden is recorded FOR. Its model identity — model_type,
                          vocabulary bound and seed length — is what the golden declares and is
                          validated against. Falls back to env MLXFAST_QWEN_MTP_TRACK_ID; an
                          undeclared track is refused by name.
    --model-provenance-repo <REPO>  Reference model repository (default the qwen target pin).
    --model-provenance-rev <HEX>    Reference model revision, 40 lowercase hex (default the pin).
    --out <PATH>          Where to write the golden JSON (written only after it loads clean).
    --backend <live|mock> Engine driver (default live). `mock` uses the in-process mock (no GPU).
";

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut backend = Backend::Live;
    let mut worker_bin = None;
    let mut weights = None;
    let mut steps = DEFAULT_STEPS;
    let mut stop_tokens: Vec<i64> = Vec::new();
    let mut benchmark_free_run = false;
    let mut benchmark_steps = DEFAULT_BENCHMARK_STEPS;
    let mut track: Option<String> = None;
    let mut contract: Option<PathBuf> = None;
    let mut provenance_repo = DEFAULT_PROVENANCE_REPO.to_string();
    let mut provenance_rev = DEFAULT_PROVENANCE_REV.to_string();
    let mut out = None;

    // --prompt-tokens and --case-name are paired POSITIONALLY: a --prompt-tokens must be followed
    // by its --case-name before the next --prompt-tokens. We buffer a pending prompt-file until its
    // name arrives, so the two flags stay explicitly associated.
    // The prompt FILES are buffered, not read, until the track identity is resolved below: the
    // required prompt length is the TRACK's seed length, so a file cannot be judged before the
    // track is known.
    let mut case_files: Vec<(String, PathBuf)> = Vec::new();
    let mut pending_prompt: Option<PathBuf> = None;

    let mut it = argv.iter();
    while let Some(flag) = it.next() {
        let mut value = || {
            it.next()
                .cloned()
                .ok_or_else(|| format!("flag {flag} needs a value"))
        };
        match flag.as_str() {
            "--backend" => {
                backend = match value()?.as_str() {
                    "live" => Backend::Live,
                    "mock" => Backend::Mock,
                    other => return Err(format!("--backend must be live or mock, got {other:?}")),
                }
            }
            "--worker-bin" => worker_bin = Some(value()?),
            "--weights" => weights = Some(PathBuf::from(value()?)),
            "--prompt-tokens" => {
                if pending_prompt.is_some() {
                    return Err(
                        "each --prompt-tokens must be followed by its --case-name before the next \
                         --prompt-tokens"
                            .to_string(),
                    );
                }
                pending_prompt = Some(PathBuf::from(value()?));
            }
            "--case-name" => {
                let name = value()?;
                let prompt_path = pending_prompt
                    .take()
                    .ok_or("--case-name must come after the --prompt-tokens it names")?;
                case_files.push((name, prompt_path));
            }
            "--steps" => steps = value()?.parse().map_err(|e| format!("--steps: {e}"))?,
            "--stop-token" => {
                stop_tokens.push(value()?.parse().map_err(|e| format!("--stop-token: {e}"))?)
            }
            "--benchmark-free-run" => benchmark_free_run = true,
            "--benchmark-steps" => {
                benchmark_steps = value()?
                    .parse()
                    .map_err(|e| format!("--benchmark-steps: {e}"))?
            }
            "--track" => track = Some(value()?),
            "--contract" => contract = Some(PathBuf::from(value()?)),
            "--model-provenance-repo" => provenance_repo = value()?,
            "--model-provenance-rev" => provenance_rev = value()?,
            "--out" => out = Some(PathBuf::from(value()?)),
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown flag {other}\n\n{USAGE}")),
        }
    }
    if pending_prompt.is_some() {
        return Err("a --prompt-tokens is missing its --case-name".to_string());
    }
    if case_files.is_empty() {
        return Err(
            "provide at least one case: --prompt-tokens <FILE> --case-name <NAME>".to_string(),
        );
    }
    if steps < CORRECTNESS_STEPS {
        return Err(format!(
            "--steps {steps} is below the minimum {CORRECTNESS_STEPS} (CORRECTNESS_STEPS); the \
             loader requires expected_tokens.len() >= CORRECTNESS_STEPS"
        ));
    }
    // AUTHOR-TIME FLOOR (a8/David ruling): the free-run oracle must be at least as long as the
    // TIMED decode window it verifies. The scored window FOR THIS TRACK is BENCHMARK_DECODE_STEPS
    // (128), so an oracle shorter than that could never gate the window benchd times — refuse it
    // here rather than emitting a golden the scoring loader (validate_benchmark_golden) rejects.
    if benchmark_free_run && benchmark_steps < BENCHMARK_DECODE_STEPS {
        return Err(format!(
            "--benchmark-steps {benchmark_steps} is below the scored decode window \
             {BENCHMARK_DECODE_STEPS} (BENCHMARK_DECODE_STEPS): a free-run oracle shorter than the \
             window it must verify is invalid"
        ));
    }
    // The recorder AUTHORS a golden, so it must know which track's identity to author under —
    // there is no default: an unset or undeclared track refuses by name.
    let track_id = track
        .or_else(|| std::env::var("MLXFAST_QWEN_MTP_TRACK_ID").ok())
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .ok_or(
            "no track: pass --track <TRACK-ID> or set env MLXFAST_QWEN_MTP_TRACK_ID (the golden \
             is authored under that track's model identity)",
        )?;
    // STEP 3 (David 2026-09-15) — the identity the golden is AUTHORED under comes from the track
    // fixture; there is no per-track table left to read it from, so the recorder requires
    // `--contract` exactly as every measuring path does.
    let contract_path = contract.ok_or(
        "no --contract: the golden is authored under the track fixture's declared model identity, \
         and there is no in-tree table to fall back to",
    )?;
    let loaded = bench_core::contract::load(&contract_path)?;
    let identity = bench_core::contract::model_identity(&loaded.contract, &track_id)?.value;
    let mut cases: Vec<CaseInput> = Vec::with_capacity(case_files.len());
    for (name, prompt_path) in case_files {
        let prompt_tokens = read_prompt_tokens_file(&prompt_path, identity.seed_tokens)?;
        cases.push(CaseInput {
            name,
            prompt_tokens,
        });
    }
    Ok(Args {
        backend,
        worker_bin,
        cases,
        identity,
        weights,
        steps,
        stop_tokens,
        benchmark_free_run,
        benchmark_steps,
        provenance_repo,
        provenance_rev,
        out: out.ok_or("--out is required")?,
    })
}

// ----------------------------------------------------------------------------------------------
// PROMPT RESOLUTION
// ----------------------------------------------------------------------------------------------

/// Parse a prompt-tokens file: a JSON array, or whitespace/comma-separated integers (`[` `]` `,` are
/// treated as separators, so all three forms parse uniformly). Requires exactly
/// [`bench_core::constants::CORRECTNESS_PROMPT_TOKENS`] ids — the golden loader rejects any other
/// prompt length, so this
/// names the defect at the input rather than at the load-back.
fn read_prompt_tokens_file(path: &Path, seed_tokens: usize) -> Result<Vec<i64>, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("--prompt-tokens read failed ({}): {e}", path.display()))?;
    let normalized: String = raw
        .chars()
        .map(|c| if matches!(c, '[' | ']' | ',') { ' ' } else { c })
        .collect();
    let mut ids = Vec::new();
    for tok in normalized.split_whitespace() {
        let id: i64 = tok.parse().map_err(|e| {
            format!(
                "--prompt-tokens {}: {tok:?} is not an integer: {e}",
                path.display()
            )
        })?;
        ids.push(id);
    }
    if ids.len() != seed_tokens {
        return Err(format!(
            "--prompt-tokens {} has {} ids; a correctness golden case for this track needs \
             exactly {} (the track's seed length)",
            path.display(),
            ids.len(),
            seed_tokens
        ));
    }
    Ok(ids)
}

// ----------------------------------------------------------------------------------------------
// RECORD — greedy teacher-forced generation
// ----------------------------------------------------------------------------------------------

/// Drive the engine to one case's `expected_tokens`: the greedy teacher-forced continuation of
/// `prompt`, `steps` tokens long. Generic over the spawn factory so the real run drives a
/// freshly-spawned engine child ([`ChildStdioTransport`]) and the tests drive the in-process mock —
/// each `spawn()` yields a cold, connected session, so no prior case's KV state carries over.
///
/// `expected_tokens[0]` is `correctness_begin(prompt)`'s argmax (the seed token = the first emitted
/// token); each subsequent `correctness_step(prev)` appends the next greedy pick. Producing `steps`
/// tokens is therefore one begin + `steps - 1` steps.
///
/// With `stop_tokens`, the case ends at the first stop token, which is its last expected token,
/// and `steps` is the maximum length. An answer shorter than [`CORRECTNESS_STEPS`] is refused: the
/// loader needs that many tokens.
fn record_case<T, F>(
    spawn: &mut F,
    name: &str,
    prompt: &[i64],
    steps: usize,
    stop_tokens: &[i64],
) -> Result<GoldenCase, String>
where
    T: LineTransport,
    F: FnMut() -> bench_runner::Result<Session<T>>,
{
    if steps == 0 {
        return Err(format!("case {name:?}: steps must be positive"));
    }
    let mut engine = spawn().map_err(|e| format!("case {name:?}: spawn: {e}"))?;
    engine.begin_phase();

    let begin = engine
        .correctness_begin(prompt)
        .map_err(|e| format!("case {name:?}: correctness_begin (seed forward): {e}"))?;
    let seed_token = begin
        .token
        .ok_or_else(|| format!("case {name:?}: correctness_begin returned no token"))?;

    let mut expected_tokens: Vec<i64> = Vec::with_capacity(steps);
    expected_tokens.push(seed_token);
    let mut prev = seed_token;
    // steps - 1 further teacher-forced steps: expected_tokens[0] is already the seed argmax.
    for step in 0..steps - 1 {
        if stop_tokens.contains(&prev) {
            break;
        }
        let resp = engine
            .correctness_step(prev)
            .map_err(|e| format!("case {name:?}: correctness_step {step}: {e}"))?;
        let token = resp
            .token
            .ok_or_else(|| format!("case {name:?}: correctness_step {step} returned no token"))?;
        expected_tokens.push(token);
        prev = token;
    }
    engine
        .close_phase()
        .map_err(|e| format!("case {name:?}: phase close barrier: {e}"))?;
    if expected_tokens.len() < CORRECTNESS_STEPS {
        return Err(format!(
            "case {name:?}: the answer ends after {} tokens, and the loader needs at least \
             {CORRECTNESS_STEPS} (CORRECTNESS_STEPS)",
            expected_tokens.len()
        ));
    }

    Ok(GoldenCase {
        name: name.to_string(),
        prompt_tokens: prompt.to_vec(),
        expected_tokens,
    })
}

/// Drive the engine's FREE-RUN trajectory over `seed_prompt` and assemble the golden's `benchmark`
/// block — the author-with-what-you-replay fix (a8/David ruling). The TIMED decode leg replays the
/// engine's incremental free-run (`free_decode_begin` + `free_decode_run`); on a stateless serve
/// that free-run trajectory DETERMINISTICALLY forks from the teacher-forced (per-step re-prefill)
/// `cases[]` continuation at near-tie argmax, so the timed leg must verify against a FREE-RUN oracle,
/// not the teacher-forced one.
///
/// `expected_prefill_token` and `expected_decode_seed_token` are BOTH the `free_decode_begin` seed
/// forward (the seed's first greedy token — regime-independent), and `expected_decode_tokens` is the
/// `free_decode_run(steps)` greedy continuation. `prefill_prompt_tokens` = `decode_seed_tokens` =
/// `seed_prompt` (`cases[0]`'s prompt), so the authored oracle covers exactly the workload official
/// scores. No baselines (calibration is a separate capture).
///
/// A FRESH cold engine (its own `spawn()`), same discipline as [`record_case`]. The free-run phase
/// is closed through the §2.6 consistency triple ([`Session::close_free_run_phase`]) exactly as the
/// scoring path does — an engine whose audit counters do not reconcile cannot author an oracle. Greedy
/// (`free_decode_run` forces temperature 0) + deterministic serialization keeps re-records
/// byte-identical.
fn record_benchmark_free_run<T, F>(
    spawn: &mut F,
    seed_prompt: &[i64],
    steps: usize,
) -> Result<BenchmarkGolden, String>
where
    T: LineTransport,
    F: FnMut() -> bench_runner::Result<Session<T>>,
{
    if steps == 0 {
        return Err("benchmark free-run: steps must be positive".to_string());
    }
    let mut engine = spawn().map_err(|e| format!("benchmark free-run: spawn: {e}"))?;
    // A teacher-forced-only engine cannot author a free-run oracle — refuse rather than silently
    // falling back to a per-step regime (the very fork this fix removes).
    if !engine.supports_free_run_decode() {
        return Err(
            "benchmark free-run: the engine did not advertise the free_run_decode capability, so its \
             free-run trajectory cannot be recorded"
                .to_string(),
        );
    }
    engine.begin_phase();

    let begin = engine
        .free_decode_begin(seed_prompt)
        .map_err(|e| format!("benchmark free-run: free_decode_begin (seed forward): {e}"))?;
    let seed_token = begin.seed_token.ok_or_else(|| {
        "benchmark free-run: free_decode_begin returned no seed_token".to_string()
    })?;

    let n = steps as u32;
    let run = engine
        .free_decode_run(n)
        .map_err(|e| format!("benchmark free-run: free_decode_run({steps}): {e}"))?;
    let tokens = run
        .tokens
        .clone()
        .ok_or_else(|| "benchmark free-run: free_decode_run returned no tokens".to_string())?;
    if tokens.len() != steps {
        return Err(format!(
            "benchmark free-run: free_decode_run returned {} committed tokens; need exactly {steps}",
            tokens.len()
        ));
    }

    // Close the free-run phase through the §2.6 consistency triple (drain + counts), exactly as
    // `measure_free_run_decode` does at score time.
    let fr = FreeRunResponse {
        tokens_len: tokens.len(),
        acceptance_lengths: run.acceptance_lengths.clone().ok_or_else(|| {
            "benchmark free-run: free_decode_run response missing acceptance_lengths".to_string()
        })?,
        drafted_total: run.drafted_total.ok_or_else(|| {
            "benchmark free-run: free_decode_run response missing drafted_total".to_string()
        })?,
        accepted_total: run.accepted_total.ok_or_else(|| {
            "benchmark free-run: free_decode_run response missing accepted_total".to_string()
        })?,
        committed_total: run.committed_total.ok_or_else(|| {
            "benchmark free-run: free_decode_run response missing committed_total".to_string()
        })?,
        // OPTIONAL — absent means NOT REPORTED, never a fault.
        verify_replay_disagreements: run.verify_replay_disagreements,
        verification: bench_core::free_run::VerificationReport::from_wire(
            run.verification_mode.as_deref(),
            run.rectangular_verification_rounds,
            run.serial_verification_rounds,
        ),
    };
    engine
        .close_free_run_phase(&fr, n)
        .map_err(|e| format!("benchmark free-run: phase-close barrier: {e}"))?;

    Ok(BenchmarkGolden {
        prefill_prompt_tokens: seed_prompt.to_vec(),
        expected_prefill_token: seed_token,
        decode_seed_tokens: seed_prompt.to_vec(),
        expected_decode_seed_token: seed_token,
        expected_decode_tokens: tokens,
        baseline_prefill_seconds_per_token: None,
        baseline_decode_seconds_per_token: None,
    })
}

/// Assemble the [`GoldenDocument`] from recorded cases, the provenance pin, and an OPTIONAL
/// free-run `benchmark` block ([`record_benchmark_free_run`]). `correctness_gates` is absent
/// (a hidden-correctness golden carries only base cases). `benchmark` is `Some` only under
/// `--benchmark-free-run`; a participant/correctness-only golden keeps it `None`, matching the
/// gemma golden's shape.
fn build_document(
    cases: Vec<GoldenCase>,
    repo: &str,
    rev: &str,
    benchmark: Option<BenchmarkGolden>,
    identity: &bench_core::constants::TrackModelIdentity,
) -> GoldenDocument {
    GoldenDocument {
        version: Some(1),
        model_type: Some(identity.golden_model_type.to_string()),
        model_provenance: Some(GoldenModelProvenance {
            repository: repo.to_string(),
            revision: rev.to_string(),
        }),
        cases,
        correctness_gates: None,
        benchmark,
    }
}

/// Serialize a golden to its on-disk bytes. Deterministic (fixed field order via the struct layout,
/// deterministic number formatting), so re-recording the same prompts is byte-identical.
fn serialize_golden(doc: &GoldenDocument) -> Result<Vec<u8>, String> {
    let mut bytes = serde_json::to_vec_pretty(doc).map_err(|e| format!("serialize golden: {e}"))?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Load the emitted bytes back through the strict scoring loader BEFORE `--out` is written: version,
/// model_type, the exact 1024 prompt length, the `>= CORRECTNESS_STEPS` expected length, token
/// ranges, AND the provenance identity against the pin. A golden that would not pass the loader (or
/// would not attest against the target pin) is never produced.
fn validate_golden(
    bytes: &[u8],
    repo: &str,
    rev: &str,
    identity: &bench_core::constants::TrackModelIdentity,
) -> Result<(), String> {
    let pin = ReferenceModelPin {
        repository: repo.to_string(),
        revision: rev.to_string(),
    };
    load_golden_fixture(
        bytes,
        CORRECTNESS_STEPS,
        identity.seed_tokens,
        identity,
        Some(&identity.golden_model_type),
        None,
        Some(&pin),
    )
    .map_err(|e| {
        format!("INTERNAL: the emitted golden failed its own loader (would not be scorable): {e}")
    })?;
    Ok(())
}

// ----------------------------------------------------------------------------------------------

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = parse_args(&argv)?;

    eprintln!(
        "record-correctness-golden: {} case(s), {} expected_tokens each{} [backend={}]",
        args.cases.len(),
        args.steps,
        if args.benchmark_free_run {
            format!(
                " + free-run benchmark block ({} decode tokens over cases[0]'s prompt)",
                args.benchmark_steps
            )
        } else {
            String::new()
        },
        match args.backend {
            Backend::Live => "live",
            Backend::Mock => "mock",
        }
    );

    let (cases, benchmark) = match args.backend {
        Backend::Live => {
            let worker_bin = args
                .worker_bin
                .as_deref()
                .ok_or("--worker-bin is required for --backend live")?;
            let weights = args
                .weights
                .as_ref()
                .ok_or("--weights is required for --backend live")?
                .to_string_lossy()
                .to_string();
            // cuda-engine argv: NO extra flags — the MTP head is embedded and the adapter ignores
            // argv (it reads its vLLM endpoint from the environment). The transport leads the argv
            // with `runtime-worker --weights <DIR>`.
            let extra: Vec<String> = Vec::new();
            let mut spawn = || -> bench_runner::Result<Session<ChildStdioTransport>> {
                let transport = ChildStdioTransport::spawn(worker_bin, &weights, &extra)?;
                let (session, _hello) = Session::connect(transport)?;
                Ok(session)
            };
            let cases = record_all(&mut spawn, &args.cases, args.steps, &args.stop_tokens)?;
            let benchmark = author_benchmark(&mut spawn, &args, &cases)?;
            (cases, benchmark)
        }
        Backend::Mock => {
            use bench_runner::mock::MockEngine;
            // The mock advertises `free_run_decode` (like the real engine) so the same spawn drives
            // both the teacher-forced cases[] and the free-run benchmark block.
            let mut spawn = || -> bench_runner::Result<Session<MockEngine>> {
                let (session, _hello) = Session::connect(MockEngine::new().free_run_capable())?;
                Ok(session)
            };
            let cases = record_all(&mut spawn, &args.cases, args.steps, &args.stop_tokens)?;
            let benchmark = author_benchmark(&mut spawn, &args, &cases)?;
            (cases, benchmark)
        }
    };

    let doc = build_document(
        cases,
        &args.provenance_repo,
        &args.provenance_rev,
        benchmark,
        &args.identity,
    );
    let bytes = serialize_golden(&doc)?;
    validate_golden(
        &bytes,
        &args.provenance_repo,
        &args.provenance_rev,
        &args.identity,
    )?;

    std::fs::write(&args.out, &bytes)
        .map_err(|e| format!("--out write failed ({}): {e}", args.out.display()))?;
    eprintln!(
        "record-correctness-golden: wrote {} ({} bytes, sha256 {})",
        args.out.display(),
        bytes.len(),
        sha256_hex(&bytes)
    );
    Ok(())
}

/// Author the OPTIONAL free-run `benchmark` block from `cases[0]`'s prompt (`--benchmark-free-run`),
/// on a FRESH engine from the SAME `spawn`. `Ok(None)` when the flag is absent (a participant /
/// correctness-only golden). `cases` is non-empty (the CLI requires at least one case), so
/// `cases[0]` is the designated seed — the prompt the timed leg replays.
fn author_benchmark<T, F>(
    spawn: &mut F,
    args: &Args,
    cases: &[GoldenCase],
) -> Result<Option<BenchmarkGolden>, String>
where
    T: LineTransport,
    F: FnMut() -> bench_runner::Result<Session<T>>,
{
    if !args.benchmark_free_run {
        return Ok(None);
    }
    let answer = &cases[0].expected_tokens;
    let ended = answer.last().is_some_and(|t| args.stop_tokens.contains(t));
    if ended && answer.len() <= args.benchmark_steps {
        return Err(format!(
            "case {:?}: the answer ends after {} tokens, inside the free-run window of {} tokens; \
             the window would time tokens that are not part of the answer",
            cases[0].name,
            answer.len(),
            args.benchmark_steps
        ));
    }
    let seed_prompt = &cases[0].prompt_tokens;
    let benchmark = record_benchmark_free_run(spawn, seed_prompt, args.benchmark_steps)?;
    Ok(Some(benchmark))
}

/// Record every case in order, each on its own cold engine (fresh `spawn()`), rejecting a duplicate
/// case name early (the loader rejects them too, but naming it here points at the CLI input).
fn record_all<T, F>(
    spawn: &mut F,
    cases: &[CaseInput],
    steps: usize,
    stop_tokens: &[i64],
) -> Result<Vec<GoldenCase>, String>
where
    T: LineTransport,
    F: FnMut() -> bench_runner::Result<Session<T>>,
{
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(cases.len());
    for case in cases {
        if !seen.insert(case.name.clone()) {
            return Err(format!("duplicate --case-name {:?}", case.name));
        }
        out.push(record_case(
            spawn,
            &case.name,
            &case.prompt_tokens,
            steps,
            stop_tokens,
        )?);
    }
    Ok(out)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("record-correctness-golden: {msg}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    /// The Qwen 3.8 125B-A6B (MLX) row — the identity this recorder's fixtures are written to.
    /// Resolved through the ONE accessor, never restated, so a fixture cannot drift from
    /// the `--contract` fixture's declared model shape.
    fn identity_125b() -> bench_core::constants::TrackModelIdentity {
        bench_core::constants::TrackModelIdentity::new("qwen4_exp_text", 248_320, 48, 1_024)
    }

    const CORRECTNESS_PROMPT_TOKENS: usize = 1_024;
    const REQUIRED_GOLDEN_MODEL_TYPE: &str = "qwen4_exp_text";

    use super::*;
    use bench_runner::mock::MockEngine;

    /// A fresh in-process mock spawn factory (deterministic; each call a cold session).
    fn mock_spawn() -> impl FnMut() -> bench_runner::Result<Session<MockEngine>> {
        || {
            let (session, _hello) = Session::connect(MockEngine::new())?;
            Ok(session)
        }
    }

    /// A fresh FREE-RUN-CAPABLE mock spawn factory (advertises `free_run_decode`, like the real
    /// engine), for the `--benchmark-free-run` authoring path.
    fn mock_spawn_free_run() -> impl FnMut() -> bench_runner::Result<Session<MockEngine>> {
        || {
            let (session, _hello) = Session::connect(MockEngine::new().free_run_capable())?;
            Ok(session)
        }
    }

    /// A well-formed 1024-id prompt (values inside the vocab range 0..VOCAB_SIZE).
    fn prompt_1024() -> Vec<i64> {
        (0..CORRECTNESS_PROMPT_TOKENS as i64).collect()
    }

    /// The mechanical review artifact: a golden recorded over the mock LOADS through the strict
    /// scoring loader (with the provenance pin), and a SECOND record on the same prompts is
    /// BYTE-IDENTICAL (greedy + serialization determinism).
    #[test]
    fn records_a_loadable_and_byte_identical_golden_over_the_mock() {
        // At least CORRECTNESS_STEPS so the loader's `expected_tokens.len() >= CORRECTNESS_STEPS`
        // gate is exercised; small enough to stay a fast CPU test.
        const STEPS: usize = CORRECTNESS_STEPS;
        let cases = vec![CaseInput {
            name: "longcopy-gate-english-1024".to_string(),
            prompt_tokens: prompt_1024(),
        }];

        let mut spawn1 = mock_spawn();
        let recorded1 = record_all(&mut spawn1, &cases, STEPS, &[]).expect("record 1");
        let doc1 = build_document(
            recorded1,
            DEFAULT_PROVENANCE_REPO,
            DEFAULT_PROVENANCE_REV,
            None,
            &identity_125b(),
        );
        let bytes1 = serialize_golden(&doc1).expect("serialize 1");

        // Shape mirrors the gemma golden: one case, 1024 prompt tokens, STEPS expected tokens.
        assert_eq!(doc1.cases.len(), 1);
        assert_eq!(doc1.cases[0].name, "longcopy-gate-english-1024");
        assert_eq!(doc1.cases[0].prompt_tokens.len(), CORRECTNESS_PROMPT_TOKENS);
        assert_eq!(doc1.cases[0].expected_tokens.len(), STEPS);

        // LOADS through the strict validator (the scoring gate), provenance pinned to the target.
        validate_golden(
            &bytes1,
            DEFAULT_PROVENANCE_REPO,
            DEFAULT_PROVENANCE_REV,
            &identity_125b(),
        )
        .expect("emitted golden must load and attest against the pin");
        let pin = ReferenceModelPin {
            repository: DEFAULT_PROVENANCE_REPO.to_string(),
            revision: DEFAULT_PROVENANCE_REV.to_string(),
        };
        let fx = load_golden_fixture(
            &bytes1,
            CORRECTNESS_STEPS,
            CORRECTNESS_PROMPT_TOKENS,
            &identity_125b(),
            Some(REQUIRED_GOLDEN_MODEL_TYPE),
            None,
            Some(&pin),
        )
        .expect("load fixture");
        assert_eq!(fx.model_type.as_deref(), Some(REQUIRED_GOLDEN_MODEL_TYPE));
        assert_eq!(fx.cases.len(), 1);
        let prov = fx.model_provenance.expect("provenance carried through");
        assert_eq!(prov.repository, DEFAULT_PROVENANCE_REPO);
        assert_eq!(prov.revision, DEFAULT_PROVENANCE_REV);

        // A SECOND record on the same prompts is BYTE-IDENTICAL.
        let mut spawn2 = mock_spawn();
        let recorded2 = record_all(&mut spawn2, &cases, STEPS, &[]).expect("record 2");
        let doc2 = build_document(
            recorded2,
            DEFAULT_PROVENANCE_REPO,
            DEFAULT_PROVENANCE_REV,
            None,
            &identity_125b(),
        );
        let bytes2 = serialize_golden(&doc2).expect("serialize 2");
        assert_eq!(
            bytes1, bytes2,
            "re-recording the same prompts must be byte-identical (greedy + serialize determinism)"
        );
    }

    /// `expected_tokens[0]` is the `correctness_begin` argmax (the seed token = first emitted), and
    /// the length is exactly `--steps`.
    #[test]
    fn expected_tokens_seed_is_begin_argmax() {
        let prompt = prompt_1024();
        let mut spawn = mock_spawn();

        // The mock returns `5000 + req.id` for correctness_begin/step; the begin is the first
        // request of a cold session, so seed == 5000 + <begin id>. We assert the STRUCTURE
        // (length + that [0] equals a standalone begin's token), not the exact id.
        let case = record_case(&mut spawn, "c", &prompt, CORRECTNESS_STEPS, &[]).expect("record");
        assert_eq!(case.expected_tokens.len(), CORRECTNESS_STEPS);

        let mut spawn2 = mock_spawn();
        let mut engine = spawn2().expect("spawn");
        engine.begin_phase();
        let begin = engine.correctness_begin(&prompt).expect("begin");
        assert_eq!(case.expected_tokens[0], begin.token.expect("begin token"));
    }

    /// A stop token ends the case. It is the last expected token, and `steps` is the maximum.
    #[test]
    fn a_stop_token_ends_the_case_and_steps_is_the_maximum() {
        let prompt = prompt_1024();
        let full = record_case(&mut mock_spawn(), "c", &prompt, 200, &[]).expect("record");
        assert_eq!(full.expected_tokens.len(), 200);

        let stop = full.expected_tokens[99];
        let ended = record_case(&mut mock_spawn(), "c", &prompt, 200, &[stop]).expect("record");
        assert_eq!(ended.expected_tokens, full.expected_tokens[..100]);

        let absent = record_case(&mut mock_spawn(), "c", &prompt, 200, &[-1]).expect("record");
        assert_eq!(absent.expected_tokens.len(), 200);
    }

    /// An answer that ends before the loader's minimum is refused by name.
    #[test]
    fn an_answer_shorter_than_the_correctness_steps_is_refused() {
        let prompt = prompt_1024();
        let full = record_case(&mut mock_spawn(), "c", &prompt, 200, &[]).expect("record");
        let stop = full.expected_tokens[CORRECTNESS_STEPS - 2];
        let err = record_case(&mut mock_spawn(), "c", &prompt, 200, &[stop]).unwrap_err();
        assert!(err.contains("the answer ends after"), "{err}");
    }

    /// Two cases record independently (each a fresh engine) and both land in the document in order.
    #[test]
    fn multiple_cases_recorded_in_order() {
        let cases = vec![
            CaseInput {
                name: "case-a".to_string(),
                prompt_tokens: prompt_1024(),
            },
            CaseInput {
                name: "case-b".to_string(),
                prompt_tokens: prompt_1024(),
            },
        ];
        let mut spawn = mock_spawn();
        let recorded = record_all(&mut spawn, &cases, CORRECTNESS_STEPS, &[]).expect("record");
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].name, "case-a");
        assert_eq!(recorded[1].name, "case-b");
        let doc = build_document(
            recorded,
            DEFAULT_PROVENANCE_REPO,
            DEFAULT_PROVENANCE_REV,
            None,
            &identity_125b(),
        );
        let bytes = serialize_golden(&doc).expect("serialize");
        validate_golden(
            &bytes,
            DEFAULT_PROVENANCE_REPO,
            DEFAULT_PROVENANCE_REV,
            &identity_125b(),
        )
        .expect("two-case golden must load");
    }

    /// A duplicate case name is rejected at the CLI layer (the loader rejects it too).
    #[test]
    fn duplicate_case_name_rejected() {
        let cases = vec![
            CaseInput {
                name: "dup".to_string(),
                prompt_tokens: prompt_1024(),
            },
            CaseInput {
                name: "dup".to_string(),
                prompt_tokens: prompt_1024(),
            },
        ];
        let mut spawn = mock_spawn();
        let err = record_all(&mut spawn, &cases, CORRECTNESS_STEPS, &[]).unwrap_err();
        assert!(err.contains("duplicate --case-name"), "{err}");
    }

    /// STEP 2 (a8/David ruling): `--benchmark-free-run` authors a `benchmark` block from cases[0]'s
    /// prompt by driving the engine FREE-RUN. The emitted golden LOADS through the strict scoring
    /// loader (so `validate_benchmark_golden` accepts the block), the block's shape matches the
    /// author-with-what-you-replay contract (prefill prompt = decode seed = cases[0]'s prompt;
    /// expected_prefill = expected_decode_seed = the free_decode_begin seed; expected_decode_tokens
    /// = the free_decode_run continuation, `--benchmark-steps` long; no baselines), and a SECOND
    /// record is BYTE-IDENTICAL.
    #[test]
    fn benchmark_free_run_block_is_authored_loadable_and_byte_identical() {
        let prompt = prompt_1024();
        let cases_input = vec![CaseInput {
            name: "longcopy-gate-english-1024".to_string(),
            prompt_tokens: prompt.clone(),
        }];
        let args = Args {
            identity: identity_125b(),
            backend: Backend::Mock,
            worker_bin: None,
            weights: None,
            cases: cases_input,
            steps: CORRECTNESS_STEPS,
            stop_tokens: Vec::new(),
            benchmark_free_run: true,
            benchmark_steps: BENCHMARK_DECODE_STEPS,
            provenance_repo: DEFAULT_PROVENANCE_REPO.to_string(),
            provenance_rev: DEFAULT_PROVENANCE_REV.to_string(),
            out: PathBuf::from("unused"),
        };

        let mut spawn1 = mock_spawn_free_run();
        let cases1 = record_all(&mut spawn1, &args.cases, args.steps, &[]).expect("record cases 1");
        let benchmark1 = author_benchmark(&mut spawn1, &args, &cases1)
            .expect("author benchmark 1")
            .expect("--benchmark-free-run authors a Some(benchmark)");

        // Author-with-what-you-replay shape.
        assert_eq!(benchmark1.prefill_prompt_tokens, prompt);
        assert_eq!(benchmark1.decode_seed_tokens, prompt);
        assert_eq!(
            benchmark1.expected_prefill_token, benchmark1.expected_decode_seed_token,
            "prefill oracle and decode-seed oracle are both the free_decode_begin seed forward"
        );
        assert_eq!(
            benchmark1.expected_decode_tokens.len(),
            BENCHMARK_DECODE_STEPS
        );
        assert!(benchmark1.baseline_prefill_seconds_per_token.is_none());
        assert!(benchmark1.baseline_decode_seconds_per_token.is_none());

        let doc1 = build_document(
            cases1,
            DEFAULT_PROVENANCE_REPO,
            DEFAULT_PROVENANCE_REV,
            Some(benchmark1),
            &identity_125b(),
        );
        let bytes1 = serialize_golden(&doc1).expect("serialize 1");
        // LOADS through the strict scoring loader (which runs validate_benchmark_golden on the block).
        validate_golden(
            &bytes1,
            DEFAULT_PROVENANCE_REPO,
            DEFAULT_PROVENANCE_REV,
            &identity_125b(),
        )
        .expect("golden with a free-run benchmark block must load through the scoring loader");
        let pin = ReferenceModelPin {
            repository: DEFAULT_PROVENANCE_REPO.to_string(),
            revision: DEFAULT_PROVENANCE_REV.to_string(),
        };
        let fx = load_golden_fixture(
            &bytes1,
            CORRECTNESS_STEPS,
            CORRECTNESS_PROMPT_TOKENS,
            &identity_125b(),
            Some(REQUIRED_GOLDEN_MODEL_TYPE),
            None,
            Some(&pin),
        )
        .expect("load fixture");
        assert!(
            fx.benchmark.is_some(),
            "the loaded golden carries the authored benchmark block"
        );

        // A SECOND record on the same prompt is BYTE-IDENTICAL (greedy + serialize determinism).
        let mut spawn2 = mock_spawn_free_run();
        let cases2 = record_all(&mut spawn2, &args.cases, args.steps, &[]).expect("record cases 2");
        let benchmark2 = author_benchmark(&mut spawn2, &args, &cases2)
            .expect("author benchmark 2")
            .expect("Some");
        let doc2 = build_document(
            cases2,
            DEFAULT_PROVENANCE_REPO,
            DEFAULT_PROVENANCE_REV,
            Some(benchmark2),
            &identity_125b(),
        );
        let bytes2 = serialize_golden(&doc2).expect("serialize 2");
        assert_eq!(
            bytes1, bytes2,
            "re-recording the same prompt (cases + free-run benchmark) must be byte-identical"
        );
    }

    /// Without `--benchmark-free-run`, `author_benchmark` returns `None` — a participant /
    /// correctness-only golden keeps `benchmark: None` (unchanged behaviour).
    #[test]
    fn no_benchmark_flag_leaves_the_block_absent() {
        let cases_input = vec![CaseInput {
            name: "c".to_string(),
            prompt_tokens: prompt_1024(),
        }];
        let args = Args {
            identity: identity_125b(),
            backend: Backend::Mock,
            worker_bin: None,
            weights: None,
            cases: cases_input,
            steps: CORRECTNESS_STEPS,
            stop_tokens: Vec::new(),
            benchmark_free_run: false,
            benchmark_steps: BENCHMARK_DECODE_STEPS,
            provenance_repo: DEFAULT_PROVENANCE_REPO.to_string(),
            provenance_rev: DEFAULT_PROVENANCE_REV.to_string(),
            out: PathBuf::from("unused"),
        };
        let mut spawn = mock_spawn_free_run();
        let cases = record_all(&mut spawn, &args.cases, args.steps, &[]).expect("record");
        let benchmark = author_benchmark(&mut spawn, &args, &cases).expect("author");
        assert!(
            benchmark.is_none(),
            "no --benchmark-free-run ⇒ benchmark: None"
        );
    }

    /// An answer that ends inside the free-run window is refused: the window would time tokens
    /// that are not part of the answer.
    #[test]
    fn an_answer_that_ends_inside_the_free_run_window_is_refused() {
        let prompt = prompt_1024();
        let full = record_case(&mut mock_spawn(), "c", &prompt, 200, &[]).expect("record");
        let args = Args {
            identity: identity_125b(),
            backend: Backend::Mock,
            worker_bin: None,
            weights: None,
            cases: vec![CaseInput {
                name: "c".to_string(),
                prompt_tokens: prompt,
            }],
            steps: 200,
            stop_tokens: vec![full.expected_tokens[99]],
            benchmark_free_run: true,
            benchmark_steps: BENCHMARK_DECODE_STEPS,
            provenance_repo: DEFAULT_PROVENANCE_REPO.to_string(),
            provenance_rev: DEFAULT_PROVENANCE_REV.to_string(),
            out: PathBuf::from("unused"),
        };
        let mut spawn = mock_spawn_free_run();
        let cases =
            record_all(&mut spawn, &args.cases, args.steps, &args.stop_tokens).expect("record");
        assert_eq!(cases[0].expected_tokens.len(), 100);
        let err = author_benchmark(&mut spawn, &args, &cases).unwrap_err();
        assert!(err.contains("inside the free-run window"), "{err}");
    }

    /// Build a `parse_args` argv that carries a real prompt file (required before the floor check)
    /// plus the given `--benchmark-steps`. Returns the argv and the temp prompt path (caller cleans
    /// it up).
    fn floor_argv(benchmark_steps: usize) -> (Vec<String>, PathBuf) {
        let prompt_path =
            std::env::temp_dir().join(format!("rcg-floor-{benchmark_steps}-prompt.tokens"));
        let ids: Vec<String> = (0..CORRECTNESS_PROMPT_TOKENS)
            .map(|i| i.to_string())
            .collect();
        std::fs::write(&prompt_path, ids.join(" ")).expect("write prompt file");
        let argv = vec![
            "--backend".to_string(),
            "mock".to_string(),
            // The recorder authors UNDER a track: the identity is required, never defaulted, and
            // since STEP 3 (David 2026-09-15) it comes from that track's FIXTURE.
            "--track".to_string(),
            "qwen3.8-125b-a6b-mlx-v1".to_string(),
            "--contract".to_string(),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/contract-full/qwen3.8-125b-a6b-mlx-v1.json"
            )
            .to_string(),
            "--prompt-tokens".to_string(),
            prompt_path.to_string_lossy().to_string(),
            "--case-name".to_string(),
            "c".to_string(),
            "--benchmark-free-run".to_string(),
            "--benchmark-steps".to_string(),
            benchmark_steps.to_string(),
            "--out".to_string(),
            "g.json".to_string(),
        ];
        (argv, prompt_path)
    }

    /// AUTHOR-TIME FLOOR: `--benchmark-steps` below the scored decode window
    /// (`BENCHMARK_DECODE_STEPS`) is refused at parse time — an oracle shorter than the window it
    /// must verify is invalid.
    #[test]
    fn benchmark_steps_below_scored_window_refused() {
        let (argv, prompt_path) = floor_argv(BENCHMARK_DECODE_STEPS - 1);
        let err = parse_args(&argv).expect_err("below-floor --benchmark-steps must be refused");
        let _ = std::fs::remove_file(&prompt_path);
        assert!(
            err.contains("--benchmark-steps") && err.contains("below the scored decode window"),
            "{err}"
        );
    }

    /// At the floor, `--benchmark-steps` is accepted (the default equals the floor).
    #[test]
    fn benchmark_steps_at_scored_window_accepted() {
        let (argv, prompt_path) = floor_argv(BENCHMARK_DECODE_STEPS);
        let parsed = parse_args(&argv).expect("floor value accepted");
        let _ = std::fs::remove_file(&prompt_path);
        assert!(parsed.benchmark_free_run);
        assert_eq!(parsed.benchmark_steps, BENCHMARK_DECODE_STEPS);
    }
}
