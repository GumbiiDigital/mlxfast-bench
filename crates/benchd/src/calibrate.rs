//! `benchd calibrate-baseline` — measure ONE box's serial-control band, once, on that box, for
//! each prompt a ranked run measures.
//!
//! Under the paired ranked design (David 2026-09-08) a ranked run measures its OWN denominator: a
//! SERIAL-CONTROL leg on the organizer-staged reference tree, on the box, in the same job. Nothing
//! is pinned in the constants, in a fixture or in a golden. What a box still needs is a HEALTH
//! BAND: a statement of what that box's control leg costs when the box is well, so a ranked run can
//! refuse a leg that is not.
//!
//! This verb writes that statement. For each `--golden`, in the order given, it runs the SAME
//! function the ranked leg 1 runs ([`crate::official::run_serial_control_leg`]) `--passes` times,
//! under the full official
//! methodology — the per-platform prefill warm-up, the unmeasured warmup leg, one resident worker
//! per pass, the cool gate before every timed phase, the live golden's own oracle — then writes the
//! per-box calibration file, with one entry per prompt. It refuses by name
//! ([`bench_core::constants::CALIBRATION_CV_EXCEEDED`]) when the passes vary by more than the fixed
//! maximum: a box that noisy has no mean that describes it, so it has no band either.
//!
//! It is an ORGANIZER step, run once per ranked box (and again after an organizer re-baseline
//! moves the reference tree). It writes NO score and NO integrity sidecar.

use crate::baseline;
/// The shared CLI flag-value reader (`crate::flag_value`), under the local name every parse
/// loop in the binary uses.
use crate::flag_value as value;
use crate::iterate::Mode;
use bench_runner::{ChildStdioTransport, RunnerError, Session};
use std::path::{Path, PathBuf};

pub const USAGE: &str = "\
benchd calibrate-baseline — measure this box's serial-control health band

USAGE:
    benchd calibrate-baseline --baseline-workspace <DIR> --engine <PATH> --golden <PATH>...
                              --contract <FILE> --out <FILE> [--weights <DIR>] [--passes N]
                              [--box <RUNNER>]

For each --golden, in the order given, the verb runs the RANKED path's own serial-control leg
`--passes` times on the reference tree. It writes ONE per-box calibration file with one entry per
prompt, and the ranked path checks each control leg against the entry of the prompt it measured.
Run it ON the ranked box, and again whenever the organizer re-baselines the reference tree.

REQUIRED:
    --baseline-workspace <DIR>   The built REFERENCE tree on this box (default: env
                                 MLXFAST_BASELINE_WORKSPACE).
    --engine <PATH>              The engine executable INSIDE that tree, as a path relative to the
                                 workspace root (the same relative path a ranked run gives for the
                                 candidate).
    --golden <PATH>              Repeatable. A LIVE golden of the track: one prompt a ranked run
                                 measures. Give each golden a ranked run on this box measures.
    --warmup-golden <PATH>       Required when the contract declares warmup_golden; used only for
                                 warm-up and never scored.
    --contract <FILE>            The track fixture. It declares the model shape the golden loads
                                 under, the measurement window the legs run and the health band
                                 the file records, so the verb cannot run without it.
    --out <FILE>                 Where to write the calibration file.

OPTIONS:
    --weights <DIR>              The transformed weights the control leg loads. Default:
                                 <baseline-workspace>/weights, the reference tree's OWN transform
                                 output. Name another directory only for a track whose weights are
                                 an organizer-staged tree outside every checkout.
    --passes <N>                 Control legs to measure (default 4; at least 2, because the file
                                 records a coefficient of variation).
    --box <RUNNER>               The runner name this box answers to (default: env RUNNER_NAME).
                                 The ranked run refuses a calibration captured on another box.
    --track <TRACK-ID>           The track being calibrated (default: env
                                 MLXFAST_QWEN_MTP_TRACK_ID).
    --prompt <NAME>              Repeatable. The prompt name recorded for the --golden in the
                                 same position (default: each golden's file stem). Give it once
                                 per --golden, or not at all.
    --reference-commit <SHA40>   The reference tree's engine commit (default: `git -C
                                 <baseline-workspace> rev-parse HEAD`).
    --benchd-source-commit <SHA40>
                                 The benchd commit that measured the legs (default: env
                                 MLXFAST_BENCHD_SOURCE_COMMIT). benchd cannot resolve its own
                                 source from a deployed binary, so one of the two must be given.
    --engine-resource <NAME=PATH>
                                 Repeatable out-of-checkpoint input, passed to every worker spawn
                                 exactly as `benchd iterate` passes it.
    --no-cool-gate               Skip the pre-phase GPU cool gate (dev only; a calibration that
                                 skipped it does not describe a cool box).
    -h, --help                   Show this help

REFUSALS (by name):
    BASELINE-WORKSPACE-MISSING     no reference tree was named, or it is not a directory
    BASELINE-WORKSPACE-NO-ENGINE   the tree holds no engine at the given relative path
    BASELINE-WORKSPACE-NO-WEIGHTS  the tree holds no transform output to measure against
    BASELINE-BOX-UNRESOLVED        neither RUNNER_NAME nor --box names this box
    GOLDEN-COUNT-MISMATCH          --prompt was given, but not once per --golden
    BASELINE-CALIBRATION-INVALID   two goldens name the same prompt
    SERIAL-CONTROL-LEG-FAILED      a control leg did not complete
    WARMUP-GOLDEN-REQUIRED         fixture declares a warm-up golden but no path was given
    WARMUP-GOLDEN-UNDECLARED       a path was given but the fixture declares no warm-up golden
    WARMUP-GOLDEN-PIN-MISMATCH     warm-up bytes do not match the fixture pin
    PROMPT-REUSE-REFUSED           warm-up or composed timed tokens collide
    CALIBRATION-CV-EXCEEDED        the legs vary by more than the track's maximum
";

/// The environment variable naming the benchd source commit, for boxes that run a deployed binary.
pub const BENCHD_SOURCE_COMMIT_ENV: &str = "MLXFAST_BENCHD_SOURCE_COMMIT";

/// The parsed command line.
#[derive(Debug)]
struct Args {
    baseline_workspace: PathBuf,
    engine: String,
    weights: PathBuf,
    /// Each golden to calibrate, in the order given, with the prompt name its entry records.
    goldens: Vec<(PathBuf, String)>,
    out: PathBuf,
    passes: u32,
    box_name: String,
    track_id: String,
    contract: PathBuf,
    reference_commit: String,
    benchd_source_commit: String,
    engine_resources: Vec<crate::engine_resource::EngineResource>,
    cool_gate: bool,
    warmup_golden: Option<PathBuf>,
}

pub fn run(args: &[String]) -> std::process::ExitCode {
    match execute(args) {
        Ok(None) => {
            print!("{USAGE}");
            std::process::ExitCode::SUCCESS
        }
        Ok(Some(())) => std::process::ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("benchd calibrate-baseline: {msg}");
            std::process::ExitCode::from(1)
        }
    }
}

fn parse(args: &[String]) -> Result<Option<Args>, String> {
    let mut workspace_flag: Option<PathBuf> = None;
    let mut engine: Option<String> = None;
    let mut weights: Option<PathBuf> = None;
    let mut goldens: Vec<PathBuf> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut passes: u32 = 4;
    let mut box_flag: Option<String> = None;
    let mut track_flag: Option<String> = None;
    let mut contract_flag: Option<PathBuf> = None;
    let mut prompt_flags: Vec<String> = Vec::new();
    let mut reference_commit_flag: Option<String> = None;
    let mut benchd_commit_flag: Option<String> = None;
    let mut engine_resources: Vec<crate::engine_resource::EngineResource> = Vec::new();
    let mut cool_gate = true;
    let mut warmup_golden: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => return Ok(None),
            "--baseline-workspace" => {
                workspace_flag = Some(PathBuf::from(value(args, i, "--baseline-workspace")?));
                i += 2;
            }
            "--engine" => {
                engine = Some(value(args, i, "--engine")?.to_string());
                i += 2;
            }
            "--contract" => {
                contract_flag = Some(PathBuf::from(value(args, i, "--contract")?));
                i += 2;
            }
            "--weights" => {
                weights = Some(PathBuf::from(value(args, i, "--weights")?));
                i += 2;
            }
            "--golden" => {
                goldens.push(PathBuf::from(value(args, i, "--golden")?));
                i += 2;
            }
            "--warmup-golden" => {
                warmup_golden = Some(PathBuf::from(value(args, i, "--warmup-golden")?));
                i += 2;
            }
            "--out" => {
                out = Some(PathBuf::from(value(args, i, "--out")?));
                i += 2;
            }
            "--passes" => {
                let v = value(args, i, "--passes")?;
                passes = v
                    .parse()
                    .map_err(|_| format!("invalid u32 for --passes: {v:?}"))?;
                i += 2;
            }
            "--box" => {
                box_flag = Some(value(args, i, "--box")?.to_string());
                i += 2;
            }
            "--track" => {
                track_flag = Some(value(args, i, "--track")?.to_string());
                i += 2;
            }
            "--prompt" => {
                prompt_flags.push(value(args, i, "--prompt")?.to_string());
                i += 2;
            }
            "--reference-commit" => {
                reference_commit_flag = Some(value(args, i, "--reference-commit")?.to_string());
                i += 2;
            }
            "--benchd-source-commit" => {
                benchd_commit_flag = Some(value(args, i, "--benchd-source-commit")?.to_string());
                i += 2;
            }
            crate::engine_resource::ENGINE_RESOURCE_FLAG => {
                crate::engine_resource::push_engine_resource(
                    &mut engine_resources,
                    value(args, i, crate::engine_resource::ENGINE_RESOURCE_FLAG)?,
                )?;
                i += 2;
            }
            "--no-cool-gate" => {
                cool_gate = false;
                i += 1;
            }
            other => return Err(format!("unknown flag {other:?}")),
        }
    }

    let baseline_workspace = baseline::resolve_workspace(
        workspace_flag.as_deref(),
        std::env::var(baseline::BASELINE_WORKSPACE_ENV)
            .ok()
            .as_deref(),
    )?;
    let engine = engine.ok_or(
        "missing required --engine (the engine's path relative to the reference workspace root)",
    )?;
    // DEFAULT: the reference tree's OWN transform output. A control leg must never load a
    // participant-editable transform's output, and the reference tree is the organizer's.
    let weights = weights.unwrap_or_else(|| baseline_workspace.join(baseline::TREE_WEIGHTS_DIR));
    if !weights.is_dir() {
        return Err(format!(
            "{}: the control leg's weights directory {} is not a directory; the default is the \
             reference tree's own transform output",
            baseline::BASELINE_WORKSPACE_NO_WEIGHTS,
            weights.display()
        ));
    }
    if goldens.is_empty() {
        return Err("missing required --golden".to_string());
    }
    let out = out.ok_or("missing required --out")?;
    if passes < 2 {
        return Err(format!(
            "--passes is {passes}: the calibration records a coefficient of variation, which needs \
             at least 2 legs"
        ));
    }
    let box_name = baseline::resolve_box_name(
        box_flag.as_deref(),
        std::env::var(baseline::RUNNER_NAME_ENV).ok().as_deref(),
    )?;
    let track_id = track_flag
        .or_else(crate::env_track_id)
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .ok_or(
            "no track_id: pass --track or set MLXFAST_QWEN_MTP_TRACK_ID to the track this box is \
             calibrated for",
        )?;
    // ONE rule for the prompt name, shared with the ranked run's own check: the calibrator
    // records the golden it measured, and the ranked run names the golden it is measuring, and the
    // two must be the same name. `--prompt` names the golden in the same position.
    let prompts = if prompt_flags.is_empty() {
        goldens
            .iter()
            .map(|g| {
                baseline::golden_prompt_name(g).ok_or_else(|| {
                    format!(
                        "--golden {} has no file name to take a prompt name from; pass --prompt",
                        g.display()
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?
    } else if prompt_flags.len() == goldens.len() {
        prompt_flags
    } else {
        return Err(format!(
            "{}: --prompt was given {} time(s) and --golden {} time(s); give one --prompt per \
             --golden, in the same order, or none",
            baseline::GOLDEN_COUNT_MISMATCH,
            prompt_flags.len(),
            goldens.len()
        ));
    };
    // The file holds one entry per prompt, so two goldens of one prompt are refused here, before
    // any leg runs, and not after every pass has run.
    for (i, prompt) in prompts.iter().enumerate() {
        if prompts[..i].contains(prompt) {
            return Err(format!(
                "{}: two goldens name prompt {prompt:?}; the file holds one entry per prompt",
                baseline::BASELINE_CALIBRATION_INVALID
            ));
        }
    }
    let reference_commit = match reference_commit_flag {
        Some(c) => c.trim().to_string(),
        None => crate::git_rev_parse_head(Some(&baseline_workspace), false).ok_or_else(|| {
            format!(
                "the reference tree {} has no readable git HEAD; pass --reference-commit <SHA40>",
                baseline_workspace.display()
            )
        })?,
    };
    let benchd_source_commit = benchd_commit_flag
        .or_else(|| std::env::var(BENCHD_SOURCE_COMMIT_ENV).ok())
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .ok_or(
            "no benchd source commit: a deployed benchd cannot resolve its own source, so pass \
             --benchd-source-commit <SHA40> or set MLXFAST_BENCHD_SOURCE_COMMIT",
        )?;

    Ok(Some(Args {
        baseline_workspace,
        engine,
        weights,
        goldens: goldens.into_iter().zip(prompts).collect(),
        out,
        passes,
        box_name,
        track_id,
        contract: contract_flag.ok_or(
            "no --contract: the track fixture declares the model shape the golden loads under and \
             the window the legs measure, and there is no in-tree table to fall back to",
        )?,
        reference_commit,
        benchd_source_commit,
        engine_resources,
        cool_gate,
        warmup_golden,
    }))
}

fn calibration_shapes<'a>(
    goldens: &[bench_core::golden::GoldenFixture],
    warmup: Option<&'a bench_core::golden::GoldenFixture>,
    offset: Option<usize>,
) -> Result<
    (
        Vec<bench_core::golden::BenchmarkGolden>,
        Option<&'a bench_core::golden::BenchmarkGolden>,
    ),
    String,
> {
    let shift = offset.unwrap_or(0);
    if let Some(shift) = offset {
        crate::official::validate_decode_golden_offset(shift, goldens.len())?;
    }
    let sources: Vec<_> = goldens
        .iter()
        .map(|golden| {
            golden
                .benchmark
                .as_ref()
                .ok_or_else(|| "benchmark golden file must contain a benchmark oracle".to_string())
        })
        .collect::<Result<_, _>>()?;
    let composed: Vec<_> = (0..sources.len())
        .map(|g| {
            crate::official::compose_benchmark_golden(
                sources[g],
                sources[(g + shift) % sources.len()],
            )
        })
        .collect();
    let warmup_benchmark = warmup
        .map(|golden| {
            golden
                .benchmark
                .as_ref()
                .ok_or_else(|| "warm-up golden file must contain a benchmark oracle".to_string())
        })
        .transpose()?;
    crate::official::validate_prompt_distinctness(
        &sources,
        &sources,
        &composed,
        &composed,
        warmup_benchmark,
        offset.is_some(),
    )?;
    Ok((composed, warmup_benchmark))
}

fn execute(args: &[String]) -> Result<Option<()>, String> {
    let args = match parse(args)? {
        Some(a) => a,
        None => return Ok(None),
    };
    let platform = bench_core::constants::Platform::from_track_id(&args.track_id)?;
    // STEP 3 (David 2026-09-15) — the model shape and the window come from the track fixture, and
    // there is no in-tree table left to fall back to, so this verb REQUIRES `--contract` like every
    // other path that resolves a per-track value. One read of one file, through the one resolver.
    let loaded = bench_core::contract::load(&args.contract)?;
    let identity = bench_core::contract::model_identity(&loaded.contract, &args.track_id)?.value;
    let window = bench_core::contract::window_shape(&loaded.contract, &args.track_id)?.value;
    // Every golden is loaded and checked before the first pass runs.
    let mut goldens = Vec::with_capacity(args.goldens.len());
    for (path, _) in &args.goldens {
        let golden = crate::load_golden_checked(
            path,
            None,
            Mode::Official.golden_required_steps(&window),
            None,
            &args.track_id,
            &identity,
        )?;
        // A calibration is measured on the SAME golden a ranked run measures, so the same refusal
        // applies: a golden carrying a stored pair belongs to the retired design.
        baseline::refuse_golden_with_stored_pair(&golden)?;
        goldens.push(golden);
    }
    let warmup_golden = match (&loaded.contract.warmup_golden, &args.warmup_golden) {
        (Some(_), None) => {
            return Err(
                "WARMUP-GOLDEN-REQUIRED: the contract declares warmup_golden; pass \
                 --warmup-golden"
                    .to_string(),
            )
        }
        (None, Some(path)) => {
            return Err(format!(
                "WARMUP-GOLDEN-UNDECLARED: --warmup-golden {} was given but the contract \
                 declares no warmup_golden",
                path.display()
            ))
        }
        (None, None) => None,
        (Some(pin), Some(path)) => {
            let golden = crate::load_golden_checked(
                path,
                None,
                Mode::Official.golden_required_steps(&window),
                None,
                &args.track_id,
                &identity,
            )?;
            if !golden.sha256.eq_ignore_ascii_case(&pin.sha256)
                || pin.bytes.is_some_and(|bytes| bytes != golden.byte_len)
            {
                return Err(format!(
                    "WARMUP-GOLDEN-PIN-MISMATCH: --warmup-golden {} does not match the contract pin",
                    path.display()
                ));
            }
            Some(golden)
        }
    };
    let validated_private_dir = if loaded.contract.require_private_dir == Some(true) {
        let mut paths: Vec<PathBuf> = args.goldens.iter().map(|(path, _)| path.clone()).collect();
        paths.extend(args.warmup_golden.iter().cloned());
        Some(crate::require_paired_private_dir(
            std::env::var_os("MLXFAST_PRIVATE_DIR").as_deref(),
            &paths,
            &[],
            std::env::var_os("MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE").is_some(),
            std::env::var("MLXFAST_NO_SANDBOX").is_ok_and(|value| value == "1"),
        )?)
    } else {
        None
    };
    let (composed, warmup_benchmark) = calibration_shapes(
        &goldens,
        warmup_golden.as_ref(),
        loaded.contract.decode_golden_offset,
    )?;

    // The engine lives INSIDE the reference tree, at the relative path the operator gave.
    let engine =
        baseline::reference_engine_path(&args.engine, Path::new(""), &args.baseline_workspace)?;
    let engine_str = engine.to_string_lossy().to_string();
    let weights_str = args.weights.to_string_lossy().to_string();
    // The Seatbelt plan is built PER PASS, not once: its one `network-outbound` allowance names
    // the socket THIS pass's resident answers on, and that socket exists only in the pass's
    // spawn-env overrides (see `spawn_resident_socket`). One plan up front would name this
    // process's socket — which a calibrating box does not have — and every pass would die on
    // `Operation not permitted`.
    let sandboxed = cfg!(target_os = "macos");
    // The calibration passes run the RANKED leg-1 shape exactly, including its per-leg engine
    // lifecycle: each pass boots the reference tree's OWN resident and tears it down again, on
    // both platforms. An inherited socket is refused for the same reason the ranked path refuses
    // it — a calibration measured against another tree's resident describes another tree.
    crate::legserve::refuse_inherited_socket(
        platform,
        std::env::var(crate::legserve::DS4_RESIDENT_SOCKET_ENV)
            .ok()
            .as_deref(),
        std::env::var(crate::legserve::BENCH_WORKER_RESIDENT_SOCKET_ENV)
            .ok()
            .as_deref(),
    )?;
    // EVERY calibration pass runs behind BOTH gates, quiescence first then cool (David
    // 2026-09-17). The one `--cool-gate` switch turns both on or both off: a pass measured on a
    // busy box describes the box, not the reference tree.
    let gates_on = args.cool_gate;
    // THE GATE LOG of this calibration (David 2026-09-17): every gate point of every pass, named
    // by the pass it belongs to, sealed into the calibration file below.
    let gate_log = std::rc::Rc::new(crate::quiescegate::GateLog::new());
    let log_for_gates = std::rc::Rc::clone(&gate_log);
    let mut phase_gates = move |phase: &str| -> Result<(), RunnerError> {
        if !gates_on {
            return Ok(());
        }
        crate::quiescegate::timed_phase_gates_logged(phase, platform, &log_for_gates)
    };

    // One pair of leg lists per golden. Pass numbers run on across the goldens, so each gate
    // point names one pass of the whole calibration.
    let mut measured_legs: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(goldens.len());
    let mut pass_number = 0;
    for (golden_index, (golden_path, prompt)) in args.goldens.iter().enumerate() {
        let benchmark = &composed[golden_index];
        let mut legs: Vec<bench_runner::TimingResult> = Vec::with_capacity(args.passes as usize);
        for pass in 1..=args.passes {
            pass_number += 1;
            // NAME THE GATE POINTS this pass is about to make.
            gate_log.enter_pass(pass_number);
            // ALWAYS SERIAL: a control leg is the serial denominator, so the resident boots serial.
            let serve = crate::legserve::boot_leg(
                &args.baseline_workspace,
                None,
                "serial-control",
                platform,
            )?;
            let leg_env = serve.spawn_env();
            let sandbox = if sandboxed {
                // The engine is the REFERENCE tree's own, resolved from --baseline-workspace, so
                // the `MLXFAST_RUNTIME_WORKER_EXECUTABLE` override is deliberately not honoured.
                Some(crate::resolve_official_sandbox_from_env(
                    &engine_str,
                    golden_path,
                    false,
                    validated_private_dir.as_deref(),
                    crate::spawn_resident_socket(&leg_env, None).as_deref(),
                )?)
            } else {
                None
            };
            let spawn = || -> bench_runner::Result<Session<ChildStdioTransport>> {
                crate::connect_official_worker(
                    sandbox.as_ref(),
                    &engine_str,
                    &weights_str,
                    &crate::free_run_spawn_args(&args.engine_resources),
                    &leg_env,
                )
            };
            let measured = crate::official::run_serial_control_leg(
                Some(benchmark),
                warmup_benchmark,
                &window,
                spawn,
                &mut phase_gates,
            );
            // The pass's resident goes down before the next pass's comes up, on success and
            // failure alike — one resident at a time, exactly as the ranked run holds one leg at a
            // time.
            let mut leg = match measured {
                Ok(leg) => leg,
                Err(error) => {
                    drop(serve);
                    return Err(format!(
                        "prompt {prompt:?} pass {pass}/{}: {error}",
                        args.passes
                    ));
                }
            };
            leg.decode_seconds_per_token =
                match crate::official::decode_seconds_per_token_for_scoring(
                    &leg,
                    loaded.contract.decode_window_includes_seed == Some(true),
                    pass as usize,
                    args.passes as usize,
                    "control",
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        drop(serve);
                        return Err(format!(
                            "prompt {prompt:?} pass {pass}/{}: {error}",
                            args.passes
                        ));
                    }
                };
            serve.stop().map_err(|error| {
                format!(
                    "prompt {prompt:?} pass {pass}/{}: LEG-TEARDOWN-FAILED: {}",
                    args.passes,
                    bench_runner::scrub_reason_for_seal(&error)
                )
            })?;
            eprintln!(
                "benchd calibrate-baseline: prompt {prompt:?} pass {pass}/{} measured prefill {} \
                 s/tok, decode scoring window {} s/tok",
                args.passes, leg.prefill_seconds_per_token, leg.decode_seconds_per_token
            );
            legs.push(leg);
        }
        measured_legs.push(baseline::control_leg_seconds(&legs));
    }

    let captured_at = crate::iterate::iso8601_now();
    let prompt_passes: Vec<baseline::PromptPasses<'_>> = args
        .goldens
        .iter()
        .zip(&measured_legs)
        .map(
            |((_, prompt), (prefill_legs, decode_legs))| baseline::PromptPasses {
                prompt,
                prefill_legs,
                decode_legs,
            },
        )
        .collect();
    let calibration = baseline::calibration_from_passes(
        &baseline::CalibrationIdentity {
            track_id: &args.track_id,
            box_name: &args.box_name,
            reference_commit: &args.reference_commit,
            benchd_source_commit: &args.benchd_source_commit,
            captured_at: &captured_at,
        },
        &prompt_passes,
        baseline::HealthBand::of_contract(&loaded.contract),
        gate_log.records(),
        baseline::max_cv_percent_of_contract(&loaded.contract),
    )?;
    let sha256 = baseline::write_calibration(&args.out, &calibration)?;
    eprintln!(
        "benchd calibrate-baseline: wrote {} (sha256 {sha256}) — box {:?}, track {:?}; no score \
         was written",
        args.out.display(),
        calibration.box_name,
        calibration.track_id,
    );
    for entry in &calibration.prompts {
        eprintln!(
            "benchd calibrate-baseline: prompt {:?}: {} passes, prefill mean {} s/tok (CV \
             {:.4}%), decode scoring-window mean {} s/tok (CV {:.4}%)",
            entry.prompt,
            entry.passes,
            entry.prefill_seconds_per_token_mean,
            entry.prefill_cv * 100.0,
            entry.decode_seconds_per_token_mean,
            entry.decode_cv * 100.0,
        );
    }
    Ok(Some(()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_calibrate_entry_charges_seed_when_the_contract_opts_in() {
        #[cfg(target_os = "macos")]
        {
            let probe = std::process::Command::new("/usr/bin/sandbox-exec")
                .args(["-p", "(version 1) (allow default)", "/usr/bin/true"])
                .output()
                .unwrap();
            if !probe.status.success() {
                eprintln!(
                    "skipping production calibrate entry test: sandbox-exec is blocked in this test environment"
                );
                return;
            }
        }
        let root = std::env::temp_dir().join(format!(
            "benchd-calibrate-entry.{}.{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let workspace = root.join("reference");
        std::fs::create_dir_all(workspace.join("weights")).unwrap();
        std::fs::create_dir_all(workspace.join("tools")).unwrap();

        let engine_source = root.join("engine.rs");
        let engine = workspace.join("engine");
        std::fs::write(
            &engine_source,
            r#"use std::io::{self, BufRead, Write};
fn number(line: &str, key: &str) -> i64 {
    line.split_once(key).unwrap().1.chars().skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap()
}
fn main() {
    let mut out = io::stdout().lock();
    writeln!(out, "{{\"id\":0,\"nonce\":\"test\",\"ok\":true,\"protocol_version\":1,\"capabilities\":[\"free_run_decode\"]}}").unwrap();
    out.flush().unwrap();
    let mut work = 0i64;
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let id = number(&line, "\"id\"");
        let response = if line.contains("\"kind\":\"prefill\"") {
            std::thread::sleep(std::time::Duration::from_millis(40));
            format!("{{\"id\":{id},\"nonce\":\"test\",\"ok\":true,\"token\":5}}")
        } else if line.contains("\"kind\":\"free_decode_begin\"") {
            std::thread::sleep(std::time::Duration::from_millis(10)); work = 1;
            format!("{{\"id\":{id},\"nonce\":\"test\",\"ok\":true,\"seed_token\":6}}")
        } else if line.contains("\"kind\":\"free_decode_run\"") {
            std::thread::sleep(std::time::Duration::from_millis(10));
            let n = number(&line, "\"count\"") as usize; work += n as i64;
            let tokens = vec!["7"; n].join(","); let accepts = vec!["1"; n].join(",");
            format!("{{\"id\":{id},\"nonce\":\"test\",\"ok\":true,\"tokens\":[{tokens}],\"acceptance_lengths\":[{accepts}],\"drafted_total\":{n},\"accepted_total\":0,\"committed_total\":{n}}}")
        } else if line.contains("\"kind\":\"phase_diagnostics\"") {
            let done = work; work = 0;
            format!("{{\"id\":{id},\"nonce\":\"test\",\"ok\":true,\"completed_work\":{done},\"cache_memory\":0,\"peak_ram_gb\":1.0}}")
        } else {
            format!("{{\"id\":{id},\"nonce\":\"test\",\"ok\":true}}")
        };
        writeln!(out, "{response}").unwrap(); out.flush().unwrap();
    }
}
"#,
        )
        .unwrap();
        let compile = std::process::Command::new("rustc")
            .args(["--edition=2021", "-O"])
            .arg(&engine_source)
            .arg("-o")
            .arg(&engine)
            .output()
            .unwrap();
        assert!(
            compile.status.success(),
            "{}",
            String::from_utf8_lossy(&compile.stderr)
        );

        let serve = workspace.join(crate::legserve::MLX_LEG_SERVE_SCRIPT);
        std::fs::write(
            &serve,
            "#!/bin/sh\nwhile [ $# -gt 0 ]; do\n case \"$1\" in --socket-out) printf '%s\\n' '/tmp/benchd-calibrate-test-no-resident.sock' > \"$2\"; shift 2;; *) shift;; esac\ndone\nexit 0\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&serve, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        let golden = root.join("botany.json");
        std::fs::write(&golden, crate::testgolden::TestGolden::new().bytes()).unwrap();
        let window = crate::iterate::test_window();
        let run = |includes_seed: bool, suffix: &str| {
            let contract = root.join(format!("contract-{suffix}.json"));
            let out = root.join(format!("calibration-{suffix}.json"));
            let fixture = serde_json::json!({
                "track_id": "qwen3.8-125b-a6b-mlx-v1",
                "vocab_size": 248320,
                "num_hidden_layers": 48,
                "seed_tokens": 1024,
                "golden_model_type": "qwen4_exp_text",
                "correctness_steps": window.correctness_steps,
                "benchmark_decode_steps": window.benchmark_decode_steps,
                "local_submit_benchmark_decode_steps": window.local_submit_benchmark_decode_steps,
                "official_prefill_warmup_runs": window.official_prefill_warmup_runs,
                "official_prefill_timed_runs": 1,
                "decode_window_includes_seed": includes_seed,
                "prefill_band_up_tolerance": 1.0,
                "prefill_band_down_tolerance": 0.5,
                "decode_band_up_tolerance": 1.0,
                "decode_band_down_tolerance": 0.5,
                "decode_band_down_enabled": true,
                "prefill_band_down_enabled": true,
                "calibration_max_cv_percent": 100.0
            });
            std::fs::write(&contract, serde_json::to_vec(&fixture).unwrap()).unwrap();
            let args = vec![
                "--baseline-workspace".to_string(),
                workspace.display().to_string(),
                "--engine".to_string(),
                "engine".to_string(),
                "--weights".to_string(),
                workspace.join("weights").display().to_string(),
                "--golden".to_string(),
                golden.display().to_string(),
                "--contract".to_string(),
                contract.display().to_string(),
                "--out".to_string(),
                out.display().to_string(),
                "--passes".to_string(),
                "2".to_string(),
                "--box".to_string(),
                "test-box".to_string(),
                "--track".to_string(),
                "qwen3.8-125b-a6b-mlx-v1".to_string(),
                "--reference-commit".to_string(),
                "a".repeat(40),
                "--benchd-source-commit".to_string(),
                "b".repeat(40),
                "--no-cool-gate".to_string(),
            ];
            execute(&args).unwrap();
            let document: serde_json::Value =
                serde_json::from_slice(&std::fs::read(out).unwrap()).unwrap();
            document["prompts"][0]["decode_seconds_per_token_mean"]
                .as_f64()
                .unwrap()
        };

        let legacy = run(false, "legacy");
        let charged = run(true, "charged");
        assert!(charged > legacy * 1.5, "legacy={legacy}, charged={charged}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn guarded_calibration_uses_composed_shape_and_warmup_golden() {
        let mut goldens: Vec<_> = (0..3)
            .map(|_| crate::testgolden::TestGolden::new().fixture())
            .collect();
        for (index, golden) in goldens.iter_mut().enumerate() {
            let benchmark = golden.benchmark.as_mut().unwrap();
            benchmark.prefill_prompt_tokens.fill(100 + index as i64);
            benchmark.decode_seed_tokens.fill(200 + index as i64);
        }
        let mut warmup = crate::testgolden::TestGolden::new().fixture();
        let warmup_expected = {
            let benchmark = warmup.benchmark.as_mut().unwrap();
            benchmark.prefill_prompt_tokens.fill(900);
            benchmark.decode_seed_tokens.fill(901);
            benchmark.clone()
        };
        let (shapes, passed_warmup) = calibration_shapes(&goldens, Some(&warmup), Some(1)).unwrap();
        for index in 0..3 {
            assert_eq!(
                shapes[index].prefill_prompt_tokens,
                goldens[index]
                    .benchmark
                    .as_ref()
                    .unwrap()
                    .prefill_prompt_tokens
            );
            assert_eq!(
                shapes[index].decode_seed_tokens,
                goldens[(index + 1) % 3]
                    .benchmark
                    .as_ref()
                    .unwrap()
                    .decode_seed_tokens
            );
        }
        assert_eq!(passed_warmup, Some(&warmup_expected));
    }

    #[test]
    fn absent_guard_keys_keep_the_calibration_leg_shape_byte_identical() {
        let golden = crate::testgolden::TestGolden::new().fixture();
        let original = golden.benchmark.as_ref().unwrap().clone();
        let (shapes, warmup) = calibration_shapes(&[golden], None, None).unwrap();
        assert_eq!(shapes, [original]);
        assert!(warmup.is_none());
    }

    #[test]
    fn help_and_unknown_flags_are_answered_at_parse() {
        assert!(parse(&["-h".to_string()]).unwrap().is_none());
        assert!(parse(&["--help".to_string()]).unwrap().is_none());
        let err = parse(&["--nope".to_string()]).unwrap_err();
        assert!(err.contains("--nope"), "{err}");
        let err = parse(&["--passes".to_string()]).unwrap_err();
        assert!(err.contains("--passes"), "{err}");
    }

    /// `--golden` and `--prompt` repeat and match by position; a `--prompt` count that does not
    /// match the `--golden` count, and two goldens of one prompt, refuse before any leg runs.
    #[test]
    fn several_goldens_parse_with_their_prompts_matched_by_position() {
        let dir =
            std::env::temp_dir().join(format!("benchd-calibrate-parse.{}", std::process::id()));
        std::fs::create_dir_all(dir.join("weights")).unwrap();
        let workspace = dir.to_string_lossy().to_string();
        let reference = "a".repeat(40);
        let benchd = "b".repeat(40);
        let argv = |extra: &[&str]| -> Vec<String> {
            [
                "--baseline-workspace",
                &workspace,
                "--engine",
                "bin/engine",
                "--contract",
                "track.json",
                "--out",
                "cal.json",
                "--box",
                "box-a",
                "--track",
                "track-a",
                "--reference-commit",
                &reference,
                "--benchd-source-commit",
                &benchd,
                "--golden",
                "/g/botany.golden.json",
                "--golden",
                "/g/kelp.mtp1.golden.json",
            ]
            .iter()
            .chain(extra)
            .map(|s| s.to_string())
            .collect()
        };
        let named =
            |args: &Args| -> Vec<String> { args.goldens.iter().map(|(_, p)| p.clone()).collect() };

        let parsed = parse(&argv(&[])).unwrap().unwrap();
        assert_eq!(named(&parsed), ["botany", "kelp"]);
        assert_eq!(
            parsed.goldens[1].0,
            PathBuf::from("/g/kelp.mtp1.golden.json")
        );
        let parsed = parse(&argv(&["--prompt", "p1", "--prompt", "p2"]))
            .unwrap()
            .unwrap();
        assert_eq!(named(&parsed), ["p1", "p2"]);
        let parsed = parse(&argv(&["--warmup-golden", "/g/warmup.golden.json"]))
            .unwrap()
            .unwrap();
        assert_eq!(
            parsed.warmup_golden,
            Some(PathBuf::from("/g/warmup.golden.json"))
        );

        let err = parse(&argv(&["--prompt", "p1"])).unwrap_err();
        assert!(err.contains(baseline::GOLDEN_COUNT_MISMATCH), "{err}");
        let err = parse(&argv(&["--prompt", "p1", "--prompt", "p1"])).unwrap_err();
        assert!(
            err.contains(baseline::BASELINE_CALIBRATION_INVALID),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The usage text states every refusal the verb can raise BY NAME, so an operator can grep the
    /// help for the message their run stopped on.
    #[test]
    fn the_usage_names_every_refusal() {
        for name in [
            baseline::BASELINE_WORKSPACE_MISSING,
            baseline::BASELINE_WORKSPACE_NO_ENGINE,
            baseline::BASELINE_BOX_UNRESOLVED,
            baseline::GOLDEN_COUNT_MISMATCH,
            crate::official::SERIAL_CONTROL_LEG_FAILED,
            crate::official::PROMPT_REUSE_REFUSED,
            "WARMUP-GOLDEN-REQUIRED",
            "WARMUP-GOLDEN-UNDECLARED",
            "WARMUP-GOLDEN-PIN-MISMATCH",
            bench_core::constants::CALIBRATION_CV_EXCEEDED,
        ] {
            assert!(USAGE.contains(name), "the usage must name {name}");
        }
    }
}
