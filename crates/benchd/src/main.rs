//! benchd — the benchmarker CLI: setup | transform | iterate | submit | official.
//!
//! Absorbs benchmark.sh and the Swift harness targets. Only writer of score artifacts.
//! Emits score.json honoring the Yukon contract (finite `score`, optional metrics).
//!
//! This wave (WS1-8) implements `iterate`: drive a live engine (spawned over
//! `ChildStdioTransport`) through the correctness gate + WS1-6 parent-side timing and
//! write a sealed `score.json` (+ `.sha256` sidecar). `transform`/`submit`/`official`
//! are stubs.

mod baseline;
mod byte_budget;
mod calibrate;
mod capture;
mod coolgate;
mod correctness;
mod editable_divergence;
mod engine_resource;
mod file_digest;
mod iterate;
mod legserve;
mod measure_job;
mod official;
mod overlay;
mod parity;
mod prefill_decompose;
mod quiescegate;
mod score;
/// #63: the shared golden-document builder for unit tests (test builds only).
#[cfg(test)]
mod testgolden;
#[cfg(test)]
mod testtmp;
mod trusted_scope;
mod weights_preflight;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use bench_core::constants::CORRECTNESS_STEPS;
/// The `--contract` TRACK FIXTURE, and the values a run reads out of it. It moved into `bench-core`
/// beside the per-track tables it resolves against, so the resolvers can live next to both and the
/// runner crate's own call sites can reach them. Spelled `contract::` here exactly as it was when
/// the module was `crate::contract`.
use bench_core::contract;
use bench_core::golden::{
    hidden_correctness_golden_pin_from_contract, load_golden_fixture,
    reference_model_pin_from_contract, verify_correctness_golden_attestation,
    verify_golden_integrity, CorrectnessGoldenPin, GoldenFixture, GoldenIntegrityPin,
    ReferenceModelPin,
};
use bench_runner::{
    resolve_official_sandbox, ChildStdioTransport, OfficialSandboxInputs, OfficialSandboxPlan,
    RunnerError, Session, SANDBOX_EXEC_PATH,
};

use crate::iterate::{
    dir_digest, dir_digest_weights, iterate_flow_windowed, DirDigest, HarnessIdentity, Mode,
    RunDigests, ScoringInputs,
};
use crate::measure_job::CohortSeams;
use crate::score::{sha256_hex, ScorePayload};

/// The per-track MODEL IDENTITY: `contract::model_identity` is the ONE accessor — the
/// `--contract` fixture's declared model SHAPE, and a refusal by name when it declares none
/// (David 2026-09-15) — and every production golden/tape load in this file
/// resolves it from the run's track id rather than reading a compile-time constant. An undeclared
/// track refuses BY NAME before any golden is trusted.
use bench_core::constants::{TrackModelIdentity, WindowShape};

/// The value a CLI flag at `args[i]` carries: `args[i + 1]`, or a refusal naming the flag.
///
/// One reader for every hand-written parse loop in the binary (`parse_measure_job_args`,
/// `parse_overlay_timing_args`, `parse_iterate_args`, `calibrate::parse`,
/// `prefill_decompose::parse`). Each of those used to hold its own byte-identical copy, so a
/// change to the refusal text reached only the loop it was made in.
fn flag_value<'a>(args: &'a [String], i: usize, name: &str) -> Result<&'a str, String> {
    args.get(i + 1)
        .map(|s| s.as_str())
        .ok_or_else(|| format!("flag {name} requires a value"))
}

/// How a flag carries its value on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlagArity {
    /// Present or absent, no value: `--preflight-only`.
    Switch,
    /// Takes one value; a repeat REPLACES the earlier one, which is what a loop that assigned into
    /// an `Option` did.
    Value,
    /// Takes one value and may be given more than once; every occurrence is kept, in order:
    /// `--golden`, `--engine-resource`.
    Repeatable,
    /// A RETIRED spelling, refused by name. The refusal is the flag AS TYPED followed by this
    /// suffix, so a renamed flag names itself and an alias keeps its own spelling.
    Refused(&'static str),
}

/// One flag of one subcommand's CLI table.
///
/// The table IS the grammar: [`parse_flags`] walks the command line against it, so a new flag is a
/// table row rather than another copy of a `while i < args.len()` loop. Every refusal the table
/// produces is the refusal the hand-written loops produced, byte for byte; the `*_USAGE` consts
/// stay the hand-written prose, and a test pins each table to its usage text.
struct FlagSpec {
    /// The canonical spelling first, then any aliases (`--min-pairs`, `--min-pairs-per-prompt`).
    /// A lookup matches any of them; a refusal quotes back the spelling the operator typed.
    names: &'static [&'static str],
    arity: FlagArity,
    /// Whether the subcommand refuses without it. The refusal is RAISED by
    /// [`ParsedFlags::missing_required`] where the built args need the value, so the order in which
    /// several missing flags are reported stays the build's and does not become the table's.
    required: bool,
    /// Flags this one may not be combined with. The refusal fires on the SECOND of the pair and
    /// names it first (`--no-cool-gate conflicts with --cool-gate`).
    conflicts_with: &'static [&'static str],
}

impl FlagSpec {
    const fn new(names: &'static [&'static str], arity: FlagArity) -> Self {
        Self {
            names,
            arity,
            required: false,
            conflicts_with: &[],
        }
    }

    /// A flag that takes one value.
    const fn value(names: &'static [&'static str]) -> Self {
        Self::new(names, FlagArity::Value)
    }

    /// A valueless flag.
    const fn switch(names: &'static [&'static str]) -> Self {
        Self::new(names, FlagArity::Switch)
    }

    /// A value flag that may be given more than once.
    const fn repeatable(names: &'static [&'static str]) -> Self {
        Self::new(names, FlagArity::Repeatable)
    }

    /// A retired spelling: `<flag as typed><suffix>` is the refusal.
    const fn refused(names: &'static [&'static str], suffix: &'static str) -> Self {
        Self::new(names, FlagArity::Refused(suffix))
    }

    const fn required(mut self) -> Self {
        self.required = true;
        self
    }

    const fn conflicts_with(mut self, flags: &'static [&'static str]) -> Self {
        self.conflicts_with = flags;
        self
    }
}

/// One flag occurrence read off the command line.
struct FlagHit {
    /// Index into the subcommand's table — the flag's canonical identity.
    spec: usize,
    /// The spelling the operator typed (canonical or alias), so `--min-pairs-per-prompt xx` refuses
    /// as itself rather than as `--min-pairs`.
    typed: &'static str,
    /// The value of a `Value`/`Repeatable` flag; `None` for a switch.
    value: Option<String>,
}

/// Every flag one command line carried, in the order it was typed.
struct ParsedFlags {
    specs: &'static [FlagSpec],
    hits: Vec<FlagHit>,
}

impl ParsedFlags {
    /// The table row for a flag. A name the table does not carry is a BUG in the build function,
    /// never operator input (`parse_flags` refuses an unknown flag long before this), so it panics
    /// rather than returning a refusal — and the per-subcommand table test walks every row.
    fn index_of(&self, name: &str) -> usize {
        self.specs
            .iter()
            .position(|s| s.names.contains(&name))
            .unwrap_or_else(|| panic!("{name} is not in this subcommand's flag table"))
    }

    /// Was the flag given at all — the switch reader.
    fn present(&self, name: &str) -> bool {
        let spec = self.index_of(name);
        self.hits.iter().any(|h| h.spec == spec)
    }

    /// The LAST value given, matching a loop that assigned into an `Option` on every occurrence.
    fn last(&self, name: &str) -> Option<&str> {
        let spec = self.index_of(name);
        self.hits
            .iter()
            .rev()
            .find(|h| h.spec == spec)
            .and_then(|h| h.value.as_deref())
    }

    fn string(&self, name: &str) -> Option<String> {
        self.last(name).map(str::to_string)
    }

    fn path(&self, name: &str) -> Option<PathBuf> {
        self.last(name).map(PathBuf::from)
    }

    /// Every value of a repeatable flag, in the order given.
    fn values(&self, name: &str) -> Vec<&str> {
        let spec = self.index_of(name);
        self.hits
            .iter()
            .filter(|h| h.spec == spec)
            .filter_map(|h| h.value.as_deref())
            .collect()
    }

    fn paths(&self, name: &str) -> Vec<PathBuf> {
        self.values(name).into_iter().map(PathBuf::from).collect()
    }

    /// Convert the flag's value, refusing on the FIRST occurrence that does not convert and
    /// returning the LAST one that does — exactly what a loop that converted as it walked did, so a
    /// `--min-pairs 1 --min-pairs zz` line still refuses the `zz`.
    ///
    /// `convert` is handed the spelling the operator typed and the value, because the refusal text
    /// names the spelling.
    fn mapped<T>(
        &self,
        name: &str,
        convert: impl Fn(&str, &str) -> Result<T, String>,
    ) -> Result<Option<T>, String> {
        let spec = self.index_of(name);
        let mut out = None;
        for hit in self.hits.iter().filter(|h| h.spec == spec) {
            out = Some(convert(
                hit.typed,
                hit.value.as_deref().unwrap_or_default(),
            )?);
        }
        Ok(out)
    }

    fn usize_flag(&self, name: &str) -> Result<Option<usize>, String> {
        self.mapped(name, |typed, v| {
            v.parse::<usize>()
                .map_err(|_| format!("invalid usize for {typed}: {v:?}"))
        })
    }

    fn f64_flag(&self, name: &str) -> Result<Option<f64>, String> {
        self.mapped(name, |typed, v| {
            v.parse::<f64>()
                .map_err(|_| format!("invalid f64 for {typed}: {v:?}"))
        })
    }

    /// The `missing required <flag>` refusal, raised where the build needs the value. A repeatable
    /// flag names itself as one (`missing required --golden (repeatable)`).
    fn missing_required(&self, name: &'static str) -> String {
        let spec = &self.specs[self.index_of(name)];
        let note = if matches!(spec.arity, FlagArity::Repeatable) {
            " (repeatable)"
        } else {
            ""
        };
        format!("missing required {name}{note}")
    }
}

/// Walk a command line against a subcommand's flag table.
///
/// `Ok(None)` is `-h`/`--help` — the caller prints its usage and exits 0. The three refusals this
/// raises (`unknown flag`, the [`flag_value`] missing-value one, and a retired flag's own message)
/// are the ones the hand-written loops raised, byte for byte.
fn parse_flags(args: &[String], specs: &'static [FlagSpec]) -> Result<Option<ParsedFlags>, String> {
    let mut hits: Vec<FlagHit> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "-h" || arg == "--help" {
            return Ok(None);
        }
        let Some(index) = specs.iter().position(|s| s.names.contains(&arg)) else {
            return Err(format!("unknown flag {arg:?}"));
        };
        let spec = &specs[index];
        // The `'static` spelling the operator typed: refusals quote it, and a `FlagHit` outlives
        // the `args` slice's borrow.
        let typed = *spec
            .names
            .iter()
            .find(|n| **n == arg)
            .expect("the table row that matched carries the spelling");
        for other in spec.conflicts_with {
            if let Some(earlier) = hits.iter().find(|h| specs[h.spec].names.contains(other)) {
                return Err(format!("{typed} conflicts with {}", earlier.typed));
            }
        }
        match spec.arity {
            FlagArity::Refused(suffix) => return Err(format!("{typed}{suffix}")),
            FlagArity::Switch => {
                hits.push(FlagHit {
                    spec: index,
                    typed,
                    value: None,
                });
                i += 1;
            }
            FlagArity::Value | FlagArity::Repeatable => {
                hits.push(FlagHit {
                    spec: index,
                    typed,
                    value: Some(flag_value(args, i, typed)?.to_string()),
                });
                i += 2;
            }
        }
    }
    Ok(Some(ParsedFlags { specs, hits }))
}

/// A subcommand's whole command-line surface: what it is called, the usage text it prints, the flag
/// table it is parsed against, and the typed parse that builds its arguments out of that table.
struct CliSpec<A: 'static> {
    /// The name in `benchd <name>: <refusal>`.
    name: &'static str,
    /// The hand-written usage text `-h`/`--help` prints and a refusal echoes.
    usage: &'static str,
    /// The table [`parse_flags`] walks. `parse` reads it from here, so one spec names everything
    /// [`run_cli`] and the parse both need.
    flags: &'static [FlagSpec],
    /// `Ok(None)` = the operator asked for help.
    parse: fn(&[String]) -> Result<Option<A>, String>,
}

/// Run one subcommand: parse, print the usage on `-h`/`--help`, refuse with exit 2 + usage on a
/// parse error, else execute.
///
/// `exec` returns the process exit code of a completed run, or `(exit, message)` for a failure that
/// carries its own honest exit (measure-job's die-8 / die-6 / exit-1 table). The three subcommands
/// used to hold three copies of this, differing only in the strings they were given.
fn run_cli<A>(
    spec: &CliSpec<A>,
    args: &[String],
    exec: impl Fn(&A) -> Result<u8, (u8, String)>,
) -> ExitCode {
    let parsed = match (spec.parse)(args) {
        Ok(Some(parsed)) => parsed,
        Ok(None) => {
            print!("{}", spec.usage);
            return ExitCode::SUCCESS;
        }
        Err(msg) => {
            eprintln!("benchd {}: {msg}", spec.name);
            eprint!("{}", spec.usage);
            return ExitCode::from(2);
        }
    };
    match exec(&parsed) {
        Ok(code) => ExitCode::from(code),
        Err((code, msg)) => {
            eprintln!("benchd {}: {msg}", spec.name);
            ExitCode::from(code)
        }
    }
}

/// THE GIT BENCHD RUNS, named by ABSOLUTE PATH rather than found on `PATH`.
///
/// One rule for all three call sites — the two fork-point gates
/// ([`editable_divergence::verify_no_write_outside_editable_from_git`],
/// [`byte_budget`]'s commit-side sum) and [`git_rev_parse_head`] — which is the rule the third
/// already had.
///
/// WHY THE ABSOLUTE PATH, not `PATH`. It resolves on both ranked fleets. The MLX box runs on the
/// Command Line Tools (docs/runbook-box-setup-mlx.md §6), which put git at `/usr/bin/git`; the
/// CUDA box runs the stock Ubuntu 24.04 GB10 image, which ships git at `/usr/bin/git` and which
/// §1 of docs/runbook-box-setup-cuda.md forbids installing packages over. So the absolute path
/// buys everything `PATH` lookup would, and costs one environment variable out of what decides a
/// gate: `PATH` on the ranked path is a value in the runner service's `.env`
/// (docs/box-setup-runbook.md, troubleshooting table), so under `PATH` lookup an edit to that
/// file moves WHICH git judges a submission. A box where this path is wrong fails the gates
/// CLOSED, with the refusal they already raise — which reads as the runbook's PATH line unmet.
const GIT: &str = "/usr/bin/git";

/// `git -C <repo_root> <args…>` for a FAIL-CLOSED GATE: spawn, refuse a non-zero exit, decode
/// stdout as UTF-8. Every failure is an `Err`; nothing here falls open.
///
/// The two fork-point gates carried a byte-identical copy of this block and differed only in the
/// words they refuse with, so the words are the parameters:
///
/// * `prefix` — the gate's refusal prefix, in front of all three messages.
/// * `invocation` — how the NON-ZERO-EXIT refusal names the command, with its base sha
///   (`git diff <base>..HEAD`, `git ls-tree <base>`).
/// * `output_shape` — how the NON-UTF-8 refusal names it, by the flags that shape its output
///   (`git diff --name-status -z`, `git ls-tree -z`).
/// * `spawn_hint` — text the SPAWN refusal appends, or `""`.
///
/// Each message is byte-identical to the copy it replaces, so a refusal a box has seen before
/// reads the same.
fn git_gate_stdout(
    repo_root: &Path,
    args: &[&str],
    prefix: &str,
    invocation: &str,
    output_shape: &str,
    spawn_hint: &str,
) -> Result<String, String> {
    let output = std::process::Command::new(GIT)
        .arg("-C")
        .arg(repo_root)
        .args(args)
        .output()
        .map_err(|e| {
            format!(
                "{prefix}: cannot run git in {}: {e}{spawn_hint}",
                repo_root.display()
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "{prefix}: {invocation} failed in {}: {}",
            repo_root.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|e| format!("{prefix}: {output_shape} emitted non-UTF-8 output: {e}"))
}

/// `git [-C <dir>] rev-parse [--short] HEAD`, trimmed; `None` on any failure (Swift's
/// `(try? …) ?? ""`). `dir` is `None` for the process's own working directory.
///
/// NOT [`git_gate_stdout`]: this is best-effort PROVENANCE, not a gate — every failure is a
/// `None` the caller records as an empty commit, never a refusal.
fn git_rev_parse_head(dir: Option<&Path>, short: bool) -> Option<String> {
    let mut cmd = std::process::Command::new(GIT);
    if let Some(dir) = dir {
        cmd.arg("-C").arg(dir);
    }
    cmd.arg("rev-parse");
    if short {
        cmd.arg("--short");
    }
    let out = cmd.arg("HEAD").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// The benchd→worker spawn flag (`--speculative-protocol v1.1`) that opts the engine into
/// advertising the v1.1 speculative surface, so its unsolicited hello carries the
/// `free_run_decode` capability ([`bench_protocol::CAPABILITY_FREE_RUN_DECODE`]).
///
/// WHY EVERY FLOW-A ENGINE SPAWN CARRIES IT. The worker speaks FIRST — the hello cannot be
/// negotiated — so the MLX worker gates the v1.1 surface at SPAWN on this flag
/// (`RuntimeWorkerGenericDispatch.swift`: `runtimeWorkerSpeculativeProtocolFlag` /
/// `runtimeWorkerAdvertisesSpeculativeProtocol`). Absent, the MLX worker emits a v1-only hello
/// WITHOUT `free_run_decode`. Every scored/capture/local timed run drives the free-run decode
/// verbs (`free_decode_begin` / `free_decode_run`, `timing::measure_free_run_decode`), which the
/// runner REFUSES on a worker whose hello lacks the capability
/// (`Session::require_free_run_capability`) — the ~16 s pre-load refusal that blocked the MLX
/// calibration. The ONE resident worker of a window also serves the timed decode leg, so the
/// correctness/window spawn needs the flag too, not just the timed spawn.
///
/// This RESTORES the flag the retired flow-B `measure_job::leg_spawn_args` / `leg_extra_args`
/// always carried on a free-run leg (deleted in commit 1a90c3a, "Retire paired flow B"); flow A
/// lost it in that deletion.
///
/// ONE SHAPE SERVES BOTH PLATFORMS (a8 Rider 1). The CUDA `cuda-engine` adapter never reads its
/// argv (`bin/cuda-engine.rs` main reads config from the environment and NDJSON from stdin) and
/// advertises `free_run_decode` on its hello UNCONDITIONALLY (`adapter.rs`), so passing this flag
/// is inert there — a benchd-only change, no adapter edit.
///
/// MEASURED-NUMBER SAFETY. The flag only unblocks the capability the timed leg already intends to
/// use; it changes no timing computation (the clock is parent-side wall time). The plain-prefill
/// and correctness legs send `spec = None`, and `Session::require_spec_echo(None, …)` skips the
/// echo check, so a v1.1 hello — which adds only the additive `effective_spec` echo and the
/// `spec_modes`/`capabilities` hello fields — does not perturb those legs.
///
/// THE RESOURCES RIDE IN FRONT OF THE GATE. `declared` is the run's `--engine-resource` list
/// ([`engine_resource`]); each becomes `--resource NAME=PATH` BEFORE the gate flag, because the
/// runner needs the resource to LOAD the model, not to serve a v1.1 verb. A spawn that speaks
/// strict v1 (the correctness gate) therefore carries the resources and NOT this flag.
fn free_run_spawn_args(declared: &[engine_resource::EngineResource]) -> Vec<String> {
    let mut args = engine_resource::spawn_args(declared);
    // The gate itself lives in ONE place (`measure_job::speculative_protocol_args`), so the two
    // spawn-argv builders cannot drift: this one used to push the flag unconditionally while the
    // measure-job builder gated it on the leg's regime.
    //
    // EVERY spawn built here IS a free-run spawn, so the gate resolves to the flag. The
    // iterate/official timed decode leg drives the v1.1 free-run verbs, and that SAME session
    // serves those runs' correctness gate too, so their correctness spawner advertises
    // `free_run_decode` as well. (The standalone
    // `correctness` VERB is the strict-v1 spawn named above: it builds its argv from
    // `engine_resource::spawn_args` alone and never comes through here.) A teacher-forced
    // measure-job leg takes its argv from `measure_job::leg_spawn_args`, which withholds the
    // flag — that gate-off spawn IS the standing v1-compat proof.
    args.extend(measure_job::speculative_protocol_args(
        measure_job::LegRegime::FreeRunV1_1,
    ));
    args
}

/// Sandbox provenance value sealed when the worker ran under the macOS Seatbelt sandbox.
const SANDBOX_PROVENANCE_SEATBELT: &str = "seatbelt";
/// Sandbox provenance value for an OFFICIAL run on a host with no Seatbelt (the linux-aarch64
/// CUDA box): the worker ran unsandboxed, and the seal says so honestly (a8 ruling b).
const SANDBOX_PROVENANCE_NONE_LINUX: &str = "none (linux)";
/// Sandbox provenance value for a run that is never sandboxed by design (the local modes, on any
/// platform).
const SANDBOX_PROVENANCE_NONE: &str = "none";

/// Spawn one official worker ([`spawn_official_worker`]) and connect its session, DISCARDING
/// the hello. The shape every correctness-gate spawner uses: the gate session's identity is not
/// the one the score seals ([`connect_retaining_hello`] is), and on a window that runs every phase
/// over one worker it IS the timed session anyway.
fn connect_official_worker(
    plan: Option<&OfficialSandboxPlan>,
    engine: &str,
    weights_path: &str,
    extra_args: &[String],
    leg_env: &[(String, String)],
) -> bench_runner::Result<Session<ChildStdioTransport>> {
    let transport = spawn_official_worker(plan, engine, weights_path, extra_args, leg_env)?;
    let (session, _hello) = Session::connect(transport)?;
    Ok(session)
}

/// Connect a session over an already-spawned `transport` and RETAIN its hello in `retained`.
///
/// The timed worker's hello is the only place the engine states its own identity — backend build
/// string, device, protocol version, loaded-head digest — and the scored leg runs on THIS worker,
/// so this is the hello the score seals (`official::seal_engine_identity`). Every timed phase of
/// one window must report the SAME resident identity; a change is a mid-window reload and is
/// refused by name (`resident_identity_changed_within_window`).
fn connect_retaining_hello(
    transport: ChildStdioTransport,
    retained: &std::cell::RefCell<Option<bench_runner::Hello>>,
) -> bench_runner::Result<Session<ChildStdioTransport>> {
    let (session, hello) = Session::connect(transport)?;
    official::retain_timed_hello(&mut retained.borrow_mut(), hello)
        .map_err(|e| RunnerError::Protocol(e.to_string()))?;
    Ok(session)
}

/// The sandbox provenance a run seals into its integrity sidecar (a8 ruling b), as a pure decision
/// so it is testable on any host regardless of `target_os`.
///
/// * a resolved Seatbelt plan (`sandbox_plan_resolved`) → [`SANDBOX_PROVENANCE_SEATBELT`]. Only
///   macOS official resolves one, so this is the macOS official value.
/// * an official run with NO plan → the host has no Seatbelt (a non-macOS box), so the worker ran
///   unsandboxed: [`SANDBOX_PROVENANCE_NONE_LINUX`]. (macOS official always resolves a plan or
///   fails closed before this point, so this arm is only reached off macOS.)
/// * anything else (the local modes) → never sandboxed by design: [`SANDBOX_PROVENANCE_NONE`].
fn sandbox_provenance(is_official: bool, sandbox_plan_resolved: bool) -> &'static str {
    if sandbox_plan_resolved {
        SANDBOX_PROVENANCE_SEATBELT
    } else if is_official {
        SANDBOX_PROVENANCE_NONE_LINUX
    } else {
        SANDBOX_PROVENANCE_NONE
    }
}

/// Spawn the OFFICIAL runtime worker. Under Seatbelt when a sandbox `plan` was resolved (macOS
/// official); otherwise UNSANDBOXED with worker-stderr forwarding forced OFF — the a8 ruling-(b)
/// linux-official path, where `/usr/bin/sandbox-exec` does not exist. Either way `extra_args` is
/// the engine argv verbatim and the child env is sanitized identically. Only HOW the worker is
/// wrapped differs — the measured number is unaffected.
///
/// EVERY official spawn goes through here, the measure-job legs included: the `None` arm is the
/// only path to an unsandboxed official worker, so a spawn site that called
/// `spawn_official_sandboxed` itself could not run on a host without Seatbelt at all.
///
/// The caller passes the argv because the two families build it differently: the iterate/official
/// spawns take [`free_run_spawn_args`] (engine resources + the v1.1 gate), while a measure-job leg
/// takes `measure_job::leg_spawn_args` (its head dirs + the gate ONLY on a free-run leg).
fn spawn_official_worker(
    plan: Option<&OfficialSandboxPlan>,
    engine: &str,
    weights_path: &str,
    extra_args: &[String],
    leg_env: &[(String, String)],
) -> std::io::Result<ChildStdioTransport> {
    match plan {
        Some(plan) => {
            ChildStdioTransport::spawn_official_sandboxed(plan, weights_path, extra_args, leg_env)
        }
        None => ChildStdioTransport::spawn_unsandboxed_official(
            engine,
            weights_path,
            extra_args,
            leg_env,
        ),
    }
}

const ITERATE_USAGE: &str = "\
benchd iterate — run the engine end-to-end and write a sealed score.json

USAGE:
    benchd iterate --engine <PATH> --weights <DIR> --golden <PATH>... [OPTIONS]

REQUIRED:
    --engine <PATH>              Engine executable (spawned as `<engine> runtime-worker --weights <DIR>`)
    --weights <DIR>              Transformed weights directory
    --golden <PATH>              GoldenDocument JSON (loaded + validated by bench-core). Repeatable
                                 on the PAIRED PATH only: pair k (1-based) measures golden
                                 (k - 1) mod N, in the order given, and both legs of a pair
                                 measure that golden's prompt. The fixture's official_pairs must
                                 be a multiple of N (else OFFICIAL-PAIRS-NOT-A-MULTIPLE-OF-GOLDENS),
                                 and the calibration file must hold an entry for each golden's
                                 prompt (else BASELINE-CALIBRATION-PROMPT-MISMATCH). The run scores
                                 the lower-median pair over all pairs. Every other path refuses a
                                 second --golden (MULTIPLE-GOLDENS-WITHOUT-PAIRED-PATH).

OPTIONS:
    --baseline-prefill-spt <F>   STORED-PAIR TRACKS ONLY, on --mode official. Prefill baseline
    --baseline-decode-spt <F>    seconds/token (trusted override; both required together; else the
                                 golden's declared pair). IGNORED on local-iterate/local-submit:
                                 those legs score against the track's CONSTANTS (#127). REFUSED BY
                                 NAME on the ranked paired path, which measures its denominator.
    --baseline-workspace <DIR>   PAIRED PATH (env MLXFAST_BASELINE_WORKSPACE). The
                                 organizer-staged, built REFERENCE tree on this box. Leg 1 — the
                                 serial-control leg — runs the same root-relative engine AND
                                 weights paths inside this tree that leg 2 runs inside the
                                 submission tree, so the control leg never loads the candidate's
                                 participant-editable transform output.
    --baseline-calibration <F>   PAIRED PATH (env MLXFAST_BASELINE_CALIBRATION). This box's
                                 calibration file (`benchd calibrate-baseline` writes it). It holds
                                 one band per prompt. The band of a pair's prompt is the HEALTH
                                 BAND for that pair's leg 1 and never a denominator: a leg outside
                                 the band refuses by name and seals no score.
                                 REQUIRED on --mode official for a track that scores against a
                                 live control leg; both refuse by name when absent. On
                                 local-iterate/local-submit they are OPTIONAL: give BOTH to run
                                 the full paired path locally, or neither to run the CANDIDATE LEG
                                 ONLY and seal no score (real timings, real correctness,
                                 score=null, baseline_source=\"none (local mode: unscored)\").
    --box <RUNNER>               PAIRED PATH. The runner name this box answers to, for the
                                 calibration file's `box` check. RUNNER_NAME wins when set.
    --control-golden <PATH>      PAIRED PATH. Repeatable: give it once per --golden, in the same
                                 order, or not at all (else GOLDEN-COUNT-MISMATCH). The golden the
                                 SERIAL-CONTROL leg (leg 1) of that golden's pairs verifies its
                                 decode tokens against. ABSENT means leg 1 uses --golden, which
                                 is only correct when the timed oracle at the declared draft depth
                                 is byte-identical to the serial tape. A track that carries
                                 PER-DEPTH oracle tapes must pass its serial live golden here: leg
                                 1 is serial by construction, so a per-depth tape refuses it at the
                                 first step the two tapes disagree on. It must name the SAME prompt
                                 as its --golden, and it is refused by name on a run that measures
                                 no control leg.
    --control-golden-sha256 <HEX>
                                 Integrity pin for --control-golden: refuse it unless its sha256
                                 matches. Repeatable: once per --control-golden, in the same
                                 order, or not at all (else GOLDEN-COUNT-MISMATCH).
    --control-golden-bytes <N>   Integrity pin for --control-golden: refuse it unless its byte
                                 count matches (both pin flags must be given together; checked
                                 before parse). Repeatable, as --control-golden-sha256.
    --mode <local-iterate|local-submit|official>
                                 Decode window: 128 (local-iterate, default), 1023 (local-submit), 128 (official)
    --score-path <OUT>           Output score path (default: score.local-iterate.json for
                                 local-iterate; score.json for local-submit/official)
    --golden-sha256 <HEX>        Integrity pin: refuse the golden unless its sha256 matches.
                                 Repeatable: once per --golden, in the same order, or not at all
                                 (else GOLDEN-COUNT-MISMATCH).
    --golden-bytes <N>           Integrity pin: refuse the golden unless its byte count matches
                                 (both pin flags must be given together; checked before parse).
                                 Repeatable, as --golden-sha256.
    --cool-gate                  Force the local GPU cool-down gate ON before each timed phase.
    --no-cool-gate               Force the cool-down gate OFF (overrides the per-mode default,
                                 e.g. local-submit's default-ON).
                                 local-iterate defaults OFF (this opts in); local-submit ON.
    --strict                     local-iterate/local-submit: also evaluate the golden's
                                 anchor/free-run gates (benchd superset). Default is Swift-exact: the
                                 correctness gate checks only the primary teacher-forced
                                 cases[]. No effect on official mode.
    --capture-baseline <REC>     CAPTURE MODE, local-iterate ONLY (every other mode refuses at
                                 parse). Runs the checked-timing legs WITHOUT resolving this
                                 track's official baseline and appends the measured pair to the
                                 capture record <REC> (identity-checked; sample CVs; temp+rename).
                                 Writes NO score and NO integrity sidecar. A track whose official
                                 baseline is already captured refuses the mode BY NAME, so it can
                                 never double as a scoring bypass.
    --mtp-depth <N>              Request the native-MTP speculative leg at draft depth N on the
                                 TIMED free-run decode window: the wire spec becomes
                                 {\"mode\":\"mtp\",\"mtp\":{\"depth\":N}} and the engine's effective_spec
                                 echo MUST match it (spec-never-ignored) or the leg is discarded.
                                 ABSENT is SERIAL: no spec goes on the wire and the engine runs its
                                 default path, which is exactly today's behaviour. N must be >= 1
                                 (depth 0 IS serial — omit the flag) and within the draft-depth cap
                                 (32 on the official path, where MLXFAST_MAX_DRAFT_DEPTH is
                                 ignored). Mutually exclusive with --candidate-spec. The
                                 correctness gate is teacher-forced and never carries a spec.
    --candidate-spec <JSON>      The EXPLICIT spec object for the timed decode window, e.g.
                                 {\"mode\":\"mtp\",\"mtp\":{\"depth\":2}}. The envelope is CLOSED (an
                                 unknown key is refused). Mutually exclusive with --mtp-depth,
                                 which is only the convenience that builds this object.
    --engine-resource <NAME=PATH>
                                 Repeatable. Pass one out-of-checkpoint input to the engine as
                                 `--resource NAME=PATH` on every worker spawn (for example
                                 qwen4exp.ngramRowSource=<dir>, the Qwen 3.8 Flash-Next n-gram row
                                 source directory). The runner needs it to LOAD the model, so it
                                 rides on every spawn. NAME uses ASCII letters, digits, '.', '_'
                                 and '-'; a repeated NAME, an empty NAME or an empty PATH refuses at
                                 parse. PATH is passed through as given: the worker validates that
                                 it exists and has the shape the resource needs, and refuses by
                                 name. The value comes from this command line only, never from a
                                 file inside the submission.
    --contract <PATH>            OFFICIAL ONLY (required). The track fixture whose
                                 `official_scoring_enabled` ARM STATE gates the scored seal: an
                                 official run over a fixture that does not declare it `true` (false
                                 or ABSENT) refuses, pre-GPU, before any score is written. Ignored
                                 on the local modes (they seal no scored artifact).
                                 The fixture also sets the TIMED TOKEN RULE of the paired path.
                                 Without timed_token_tolerance_per_thousand, one timed token that
                                 differs from the golden's oracle fails the run. With N (0 to
                                 1000), the candidate leg runs its whole decode window, the
                                 reference tree's engine replays the candidate's own tokens
                                 teacher-forced after the last pair, and a pair fails
                                 (TIMED-DIVERGENCE-OVER-TOLERANCE) when more than N per thousand of
                                 its tokens differ from the reference engine's choice for the same
                                 prefix. The serial-control leg stays exact. With a tolerance, a
                                 ranked run does not prove that the candidate's output is the
                                 reference output. With timed_token_near_tie_relative_gap G as
                                 well (the near-tie rule), a different token is tolerated only
                                 when it is the reference engine's second choice and the relative
                                 gap between its first and second choice is at most G; any other
                                 different token fails the run (TIMED-DIVERGENCE-NOT-A-NEAR-TIE),
                                 and near ties still count against N. The near-tie rule forgives
                                 a change of winner between two tokens the reference engine scores
                                 almost the same, and nothing else. A local run without a
                                 reference tree cannot replay, so it stays exact.
    -h, --help                  Show this help

    --capture-baseline <REC>     CAPTURE MODE (local-iterate only; refuses every other mode). Runs
                                 the checked-timing leg normally while the track's official baseline
                                 is PENDING, appends this run's prefill/decode seconds-per-token to
                                 the capture record at <REC> (identity-checked; sample CVs
                                 recomputed) and writes NO score and no integrity sidecar. A
                                 captured track refuses the mode by name.
    --capture-timed-only         Only with --capture-baseline (a8 ruling). SKIP the teacher-forced
                                 correctness gate and run ONLY the timed prefill+decode pass, then
                                 append the pair. The engine cannot change between passes of the
                                 organizer's own calibration, so the PLE-SSD-bound gate runs ONCE per
                                 prompt per window (the first capture pass, WITHOUT this flag) and
                                 every remaining pass (A passes 2-4 and all B passes) uses this flag.
                                 Refused without --capture-baseline; never reaches a scored run.
    --capture-passes <SPEC>      Only with --capture-baseline (a8 Option-A restructure). SPEC is a
                                 comma list of per-pass record LABELS, e.g. W,A,A,B,B (warmup + A1 A2
                                 + B1 B2). This ONE invocation runs EVERY listed pass over ONE
                                 persistent model residency, so on MLX the model loads ONCE per
                                 prompt window instead of once per pass. NO pass runs the
                                 teacher-forced correctness gate (dropped from calibration entirely,
                                 David 2026-08-31); EVERY pass is timed-only, with the free-run
                                 oracle token-match as the correctness evidence. Each pass's pair is
                                 appended to <base>.<label>.<ext> off
                                 the --capture-baseline base path; a repeated label (the two A's)
                                 merges into one record. The per-pass MEASURED number is
                                 byte-identical to a standalone single-pass invocation. Conflicts
                                 with --capture-timed-only (subsumed); refused without
                                 --capture-baseline.

    --weights-digest <S:B:C>     Only with --capture-baseline (Option B digest-hoist). SKIP the
                                 per-pass weights re-hash and construct the run's weights digest
                                 from this <sha256>:<bytes>:<files> triple instead — the value
                                 `benchd weights-digest --weights <DIR>` prints. It is
                                 BYTE-IDENTICAL to the per-pass `dir_digest` of the same immutable
                                 tree; the window computes it ONCE at the start and passes it to
                                 every remaining pass so the ~105 GB tree is hashed once, not once
                                 per pass. Refused at PARSE without --capture-baseline: an official
                                 (scored) run always hashes for itself; a passed-in digest can never
                                 reach a scored seal.

ENV:
    MLXFAST_QWEN_MTP_TRACK_ID    REQUIRED. The track this run scores under (for example
                                 qwen3.8-125b-a6b-mlx-v1). Its `-{platform}-v{N}` suffix keys the
                                 official baseline pair + acceptance bands; while that platform's
                                 capture is pending the run refuses by name before any engine spawn.
";

const TOP_USAGE: &str = "\
benchd — benchmarker CLI

USAGE:
    benchd <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
    iterate         Run the engine end-to-end and write a sealed score.json
    correctness     Run the correctness gate end-to-end; exit 0 = pass, 1 = fail
    validate-golden Integrity-pin + load-validate a golden (no engine); exit 0 if accepted
    validate-weights Preflight the transformed-weights dir + size cap (no engine); exit 0 if accepted
    parity-diff     Diff two score.json (benchd vs swift); exit 0 = PARITY: PASS
    prefill-decompose  Fit prefill elapsed_ms = c + m*n across sizes (M-5 residual attribution)
    measure-job     Option-A seam 2: PAIRED timing over candidate/baseline workspaces → results.json
    overlay-timing  Option-A seam 3 (LOCAL): merge gates-score.json + results.json → LOCAL/parity score.json (organizer owns the ranked seal)
    harness-hash    Print the 9-root harness identity of the PROCESS CWD (read-only; the seal's own resolution)
    weights-digest  Print the weights-tree digest as <sha256>:<bytes>:<files> (the window's once-per-run digest)
    calibrate-baseline  Measure this box's serial-control health band and write its calibration
    transform       (not implemented in this wave)
    submit          (not implemented in this wave)
    official        run the official benchmark (alias: `iterate --mode official`)
";

const VALIDATE_GOLDEN_USAGE: &str = "\
benchd validate-golden — integrity-pin + load-validate a golden (no engine spawned)

USAGE:
    benchd validate-golden --golden <PATH> --track <TRACK-ID> --contract <PATH>
                             [--golden-sha256 <HEX> --golden-bytes <N>] [--gates-only]

--track <TRACK-ID> names the track whose MODEL IDENTITY the golden is judged against: its
model_type, vocabulary bound and seed length. It falls back to env MLXFAST_QWEN_MTP_TRACK_ID,
and an undeclared track is refused BY NAME — this command never guesses an identity.

--contract <PATH> is REQUIRED: the track fixture DECLARES that identity, and there is no in-tree
table to fall back to.

Exit 0 if the golden passes the pin (when given) AND the bench-core loader accepts it;
non-zero with the rejection reason on stderr otherwise. Used by the loader-parity harness
(Rust side) and as a standalone integrity check.

The same fixture supplies the track's REFERENCE MODEL (#114): when it declares
`target.upstream_model_id` / `target.upstream_revision`, a golden carrying a `model_provenance`
block must NAME that model — the same value gate the Swift reference applies from its
compile-time constants. When it declares no target, the block is validated for SHAPE only, and
the command says so rather than implying the values were checked.

By default the golden MUST carry a benchmark oracle block (byte-consistent with Swift
preflight, which rejects a benchmark-less golden with 'benchmark golden file must contain
a benchmark oracle'). Pass --gates-only to SKIP that requirement — validating only the
structure + correctness gates — for internal fixtures that legitimately lack a benchmark
oracle.
";

const CORRECTNESS_USAGE: &str = "\
benchd correctness — run the correctness gate end-to-end (no score.json)

USAGE:
    benchd correctness --engine <PATH> --weights <DIR> --golden <PATH> [OPTIONS]

Spawns the engine (`<engine> runtime-worker --weights <DIR>`) and runs the FULL correctness
set — base teacher-forced cases THEN the golden's anchor / free-run gates (Swift
`runCorrectness` → `runLayeredCorrectness`, checkGates:true). A concise JSON verdict is
printed to stdout; the EXIT CODE is authoritative: 0 = pass, 1 = fail (byte-matching Swift
`mlxfast-swift correctness`, `return report.passed ? 0 : 1`).

The golden need NOT carry a benchmark oracle (Swift `checkCorrectnessArtifacts`,
requiresBenchmarkOracle:false) — correctness is oracle-optional, unlike `benchmark`/`official`.

REQUIRED:
    --engine <PATH>              Engine executable
    --weights <DIR>              Transformed weights directory
    --golden <PATH>              GoldenDocument JSON (loaded + validated by bench-core)

OPTIONS:
    --golden-sha256 <HEX>        Integrity pin (with --golden-bytes): refuse a non-matching golden
    --golden-bytes <N>           Integrity pin (both pin flags must be given together)
    --manifest <PATH>            RunnerManifest JSON. The kit checks the engine's hello against it
                                 BEFORE the correctness set runs: the advertised spec modes are a
                                 subset of the manifest decoders, the capabilities and the cohort
                                 ceiling are the ones the manifest derives, the backend matches, and
                                 the hello carries a runner identity whose manifest_sha256 is the
                                 canonical digest of this file. Each failed check is reported by
                                 name. Exit 1 on a failed check, 2 on an unreadable or malformed
                                 manifest.
    --trusted                    The kit runs as the TRUSTED build, so the manifest derives the
                                 `cohort_reference_replay` capability. Only with --manifest.
    --engine-resource <NAME=PATH>
                                 Repeatable. Pass one out-of-checkpoint input to the engine as
                                 `--resource NAME=PATH` on the worker spawn (for example
                                 qwen4exp.ngramRowSource=<dir>). The runner needs it to LOAD the
                                 model. NAME uses ASCII letters, digits, '.', '_' and '-'; a
                                 repeated NAME, an empty NAME or an empty PATH refuses at parse.
                                 The value comes from this command line only, never from a file
                                 inside the submission.
    -h, --help                   Show this help
";

const VALIDATE_WEIGHTS_USAGE: &str = "\
benchd validate-weights — preflight the transformed-weights directory (no engine spawned)

USAGE:
    benchd validate-weights --weights <DIR> [--golden <PATH>]

Mirrors the WEIGHTS half of Swift `BenchmarkPreflight`: the weights path is a real directory
(not a symlink), `config.json` and `model.safetensors.index.json` are present as regular
files, and the directory byte-count (symlinks / non-regular files REJECTED) is enforced
against the size cap. The cap is read from `MLXFAST_MAX_WEIGHTS_BYTES` (empty ⇒ 25 GiB
default; `0` / `none` / `unlimited` ⇒ uncapped; a positive integer ⇒ that cap).

Exit 0 = accepted, 1 = rejected (with the reason on stderr), 2 = usage, 3 = IO error.
`validate-golden` covers the golden half; pass `--golden` here only to also require its
presence (Swift `requiredFiles`).
";

const WEIGHTS_DIGEST_USAGE: &str = "\
benchd weights-digest — print the weights-tree digest (Option B digest-hoist)

USAGE:
    benchd weights-digest --weights <DIR>

Computes the digest over the transformed-weights directory with the SAME `dir_digest` an
`iterate` pass uses, and prints ONE line to stdout in the stable, parseable form:

    <sha256>:<byte_count>:<file_count>

The value is byte-identical to what any pass would compute for the same immutable tree. A
calibration window computes it ONCE at window start and hands it to passes 2-N via
`benchd iterate --weights-digest <sha256>:<bytes>:<files> --capture-baseline ...`, so the
~105 GB tree is hashed once per window instead of once per pass. Read-only: it spawns no
engine, resolves no baseline, and writes no artifact.

Exit 0 = printed; non-zero with the reason on stderr on a bad flag or IO error.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sub = match args.first() {
        Some(s) => s.as_str(),
        None => {
            eprint!("{TOP_USAGE}");
            return ExitCode::from(2);
        }
    };
    match sub {
        "iterate" => run_iterate(&args[1..]),
        // #90 item 2: the standalone correctness gate (Swift `mlxfast-swift correctness`).
        // Exit 0 = pass, 1 = fail (byte-matching Swift `report.passed ? 0 : 1`).
        "correctness" => run_correctness(&args[1..]),
        "validate-golden" => run_validate_golden(&args[1..]),
        // #90 item 3: the WEIGHTS-half preflight (Swift `BenchmarkPreflight` weights checks +
        // `MLXFAST_MAX_WEIGHTS_BYTES`). `validate-golden` covers only the golden half.
        "validate-weights" => run_validate_weights(&args[1..]),
        // §T: the parity verdict tool (Rust port of scripts/parity-diff.py). Shares the real
        // ScoreMetrics type; the bucket roster is pinned to the schema by a cargo test.
        "parity-diff" => parity::run(&args[1..]),
        // M-5 (#68): prefill-DECOMPOSITION diagnostic — fit elapsed_ms = c + m*n across
        // synthetic prompt sizes to attribute the +2.70% single-shot prefill residual
        // (RULING A3) to the protocol/spawn floor vs per-token compute. Timing-only, no
        // oracle, no score artifact — it does NOT touch the scoring/timing production path.
        "prefill-decompose" => prefill_decompose::run(&args[1..]),
        // A-1: the Option-A measure-job component (seam 2) — the PAIRED/batched flow-B timing over
        // candidate/baseline workspaces → results.json; no gates (seam 1), no score.json (seam 3).
        // The Qwen 3.8 125B-A6B tracks do NOT run it: their sole scored path is
        // `iterate --mode official` (`execute_measure_job` refuses them by name).
        "measure-job" => run_measure_job_cli(&args[1..]),
        // A-3: the Option-A overlay component (seam 3, LOCAL). Merges the seam-1 gates-score.json
        // with the measure-job results.json into a sealed ranked score.json (3.8 median regime).
        // On the RANKED path the organizer authors score.json (OPEN-2); this is the LOCAL merge +
        // the verifiable seam-3 parity reference.
        "overlay-timing" => run_overlay_timing_cli(&args[1..]),
        // David ruling 2026-08-26 — the read-only harness-identity printer. Same resolution as the
        // seal-time cross-leg gate; lets a driver/test obtain the identity without reimplementing
        // the 9-root algorithm in shell.
        "calibrate-baseline" => calibrate::run(&args[1..]),
        "harness-hash" => run_harness_hash(),
        // Option B digest-hoist: compute the weights-tree digest ONCE per calibration window and
        // print it as <sha256>:<bytes>:<files> for the driver to pass to passes 2-N via
        // `iterate --weights-digest`. Uses the SAME `dir_digest` a pass would, so the value is
        // byte-identical to a per-pass recompute — never a shell-side sha reimplementation.
        "weights-digest" => run_weights_digest(&args[1..]),
        // Cool-gate helper: Swift `runLocalPhaseCoolGate` dispatches to
        // `<MLXFAST_LOCAL_COOL_GATE_HELPER> --local-cool-gate-only` with the phase in
        // `MLXFAST_LOCAL_COOL_GATE_PHASE`. Pointing that helper at benchd gives the Swift
        // leg the SAME macmon gate benchd runs natively — identical thermal semantics on
        // both sides. Exit 0 = passed/skipped (Swift's ok contract); non-zero = thermal abort.
        "--local-cool-gate-only" => run_cool_gate_only(),
        // Official runs via `iterate --mode official` (B-2): timed-first, three fresh
        // sandboxed workers, full correctness set, official gating + sealing. The bare
        // `official` subcommand points there rather than being a second entrypoint.
        "official" => {
            eprintln!(
                "benchd official: run the official benchmark via `benchd iterate --mode official`"
            );
            ExitCode::from(2)
        }
        "transform" | "submit" => {
            eprintln!("benchd {sub}: not implemented in this wave");
            ExitCode::from(2)
        }
        "-h" | "--help" | "help" => {
            print!("{TOP_USAGE}");
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("benchd: unknown subcommand {other:?}");
            eprint!("{TOP_USAGE}");
            ExitCode::from(2)
        }
    }
}

/// `--local-cool-gate-only`: run the two timed-phase gates — quiescence, then cool-down — for the
/// phase named in `MLXFAST_LOCAL_COOL_GATE_PHASE`, then exit. Mirrors benchmark.sh's
/// `--local-cool-gate-only` so the Swift harness can dispatch its `runLocalPhaseCoolGate` to
/// benchd, and gives that leg BOTH gates the native path runs (David 2026-09-17).
/// `benchd harness-hash` — print the 9-root WORKSPACE HARNESS IDENTITY of the PROCESS CWD.
///
/// A read-only diagnostic over [`HarnessIdentity::resolve_from_current_dir`] — the SAME resolution
/// `iterate` seals with and the SAME one the overlay's David-ruled cross-leg gate recomputes at the
/// seal. It exists so a shell can obtain the identity WITHOUT a second implementation of the
/// algorithm: `scripts/test-paired-offline.sh` stamps its mock gates-score with the real identity of
/// the tree its seam-3 invocation runs from, so that test exercises the real equality instead of a
/// doctored one. CWD-only by design (no `--workspace` flag): the CWD invariant is the property worth
/// being able to check from the outside.
///
/// Feeds NO enforced value and authors NO artifact — it prints a digest and exits. It hashes over
/// the roster roots that exist (a root absent on this engine surface is logged to stderr and
/// skipped); it exits 1 only if the current working directory itself cannot be resolved.
fn run_harness_hash() -> ExitCode {
    match HarnessIdentity::resolve_from_current_dir() {
        Ok(identity) => {
            println!("{}", identity.as_str());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!(
                "benchd harness-hash: {e}; run it with the working directory at the engine \
                 workspace root"
            );
            ExitCode::from(1)
        }
    }
}

fn run_cool_gate_only() -> ExitCode {
    let phase =
        std::env::var("MLXFAST_LOCAL_COOL_GATE_PHASE").unwrap_or_else(|_| "local".to_string());
    match quiescegate::timed_phase_gates(&phase, cool_gate_platform_from_env()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("benchd --local-cool-gate-only: {e}");
            ExitCode::from(1)
        }
    }
}

const MEASURE_JOB_USAGE: &str = "\
benchd measure-job — Option-A seam 2: paired ranked TIMING over two workspaces

USAGE:
    benchd measure-job --candidate <WS> --baseline <WS> --golden <PATH> [--golden <PATH> ...] \\
        --contract <PATH> --min-pairs <N> --target-pairs <N> --tag <S> --out <DIR> \\
        [--correctness-golden <PATH>] \\
        [--tokens 512] [--mtp-depth <N>] [--weights <DIR>] [--exactness-probe once] \\
        [--prompt <PATH> --prompt-sha256 <HEX> --target-id <ID>] [--preflight-only] \\
        [--calibration-bootstrap] [--gates-producer <NAME>]

Runs the alternating serial-control / candidate PAIR LOOP and emits <out>/results.json
(+ .sha256 sidecar, bare-basename body) and <out>/benchmark-integrity.results.json. Runs NO
correctness/gates (those come from the seam-1 GATES PRODUCER — by default the reference
benchmark.sh, per ruling Q1a) and authors NO score.json (that is the overlay, seam 3). Fails
closed (exit nonzero, die 5) when fewer than --min-pairs pairs are accepted.

REQUIRED:
    --candidate <WS>   Candidate workspace (cloned on-box; the MTP spec-decode engine). Its engine
                       is resolved as <WS>/.build/release/<bin> (bin defaults to `mlxfast-runtime-worker`,
                       overridable via MLXFAST_MEASURE_WORKER_BIN); fail-closed if absent.
    --baseline <WS>    Baseline workspace (the serial-control / depth-0 engine); same resolution.
    --golden <PATH>    The timed prompt, in EITHER accepted shape, routed by required-key signature:
                         * TIMED-PROMPT TAPE (what the live timed_prompt_pool PINS): keys
                           seed_tokens / reference_seed_token / rows [+ reference_self_consistent,
                           emitted_tokens]. The legs decode from seed_tokens, oracle the seed forward
                           against reference_seed_token, then check rows[i].sequential_argmax.
                         * GoldenDocument JSON (legacy): keys version / model_type / cases /
                           correctness_gates / benchmark; the benchmark oracle drives the legs.
                       REPEATABLE — pass once per golden; a DUPLICATE DIGEST (same bytes twice) is
                       fatal (die 8). Every golden's sha256 must resolve to EXACTLY ONE --contract
                       timed_prompt_pool entry with a positive noop_decode_speedup (die 8, pre-GPU).
                       The per-golden loop is R7; this component measures the FIRST golden.
    --contract <PATH>  Track fixture (timed_prompt_pool + track_id). Also the review-gated authority
                       for the hidden correctness golden (its `hidden_correctness_golden` sha256+bytes
                       SIBLING pin, LANE 2a) — a SIBLING of timed_prompt_pool that never changes N.

OPTIONAL (LANE 2a):
    --correctness-golden <PATH>  The run's correctness-golden ATTESTATION: the staged hidden
                       correctness golden this run verified token-fidelity against. benchd hashes it
                       (sha256 + bytes) and refuses (die 8, pre-GPU) any run whose identity does not
                       CITE the --contract fixture's hidden_correctness_golden pin. FAIL-CLOSED both
                       ways: a fixture that pins the golden REQUIRES this flag; passing it against a
                       fixture that pins none is refused. Omit only on offline/legacy tracks whose
                       fixture declares no hidden_correctness_golden.

    --min-pairs <N>        Per-prompt floor (>= 1); fail closed (die 5) below it.
                           Alias: --min-pairs-per-prompt.
    --target-pairs <N>     Per-prompt target; stop accepting once reached (>= --min-pairs).
                           Alias: --pairs-per-prompt.
    --tag <S>          Per-run identity sealed into results.json (NOT the track_id).
    --out <DIR>        Output directory for results.json (+ sidecars)

OPTIONAL:
    --tokens <N>       Depth-0 decode window both legs time (default 512).
    --mtp-depth <N>    Convenience for the candidate spec {\"mode\":\"mtp\",\"mtp\":{\"depth\":N}}.
                       OMITTED = the engine's drafter decides ({\"mode\":\"mtp\",\"mtp\":{}} — no
                       depth is requested or sealed; the echo reports the operating value).
                       Depth is a MODULE field now. MUTUALLY EXCLUSIVE with
                       --candidate-spec (pass one). Bounds-checked against the readonly 32
                       draft-depth cap.
    --candidate-spec <JSON>  Explicit per-module speculative spec for the candidate leg (e.g.
                       '{\"mode\":\"mtp\",\"mtp\":{\"depth\":4}}'), recorded spec_source cli-override.
    --baseline-spec <JSON>   Explicit baseline spec (default {\"mode\":\"serial\"}). MUST be
                       mode==serial: the baseline is the serial DENOMINATOR and is not
                       CLI-steerable off serial (a non-serial baseline is a hard error).
    --weights <DIR>    OVERRIDE for the transformed weights directory. Spawned as `<engine>
                       runtime-worker --weights <DIR>` for BOTH legs (backbone/identity case).
                       The approved draft measure-job CLI carries NO --weights: weights are
                       DERIVED on-box from the env `QMTP_TARGET_DIR` (the draft's source).
                       When --weights is omitted the env is used; when NEITHER is set this fails
                       closed with a clear message.
    --gates-producer <NAME>    WHICH seam-1 gates producer the driver used (`benchmark-sh` = the
                               organizer's reference chain and the DEFAULT per ruling Q1a,
                               `facade` = benchd's own --official, `direct-swift` = the weightless
                               fallback). SEALED VERBATIM into benchmark-integrity.results.json.
                               measure-job is seam 2 and cannot observe seam 1, so the driver
                               DECLARES this and measure-job records the declaration; omitted seals
                               `undeclared`, which is the answer for a standalone run with no seam
                               1, not a gap. Any name is accepted (provenance, not a policy gate);
                               whitespace/control characters are refused because the value lands in
                               an artifact. NOTE: a DECLARATION, not an independent verification —
                               see #140.
    --exactness-probe <MODE>   none|once|per-prompt|per-pair (default once). The untimed mtp-verify
                               gate that consumes it is R15; here it is parsed + validated + stored.
    --prompt <PATH>            An explicit prompt file. ALL-THREE-OR-NONE with --prompt-sha256 and
    --prompt-sha256 <HEX>      --target-id: the sha is 64 lowercase hex, the target-id matches
    --target-id <ID>           [A-Za-z0-9._-]+, and the prompt file's sha256 MUST equal
                               --prompt-sha256 (die 8 on mismatch).
    --write-gate-base <SHA>    Judge the write-outside gate + growth budget against the candidate's
                               own committed diff (<SHA>..HEAD, its fork point from harness main)
                               instead of the staged --baseline tree. ABSENT = the legacy staged-
                               --baseline tree-diff, unchanged.
    --preflight-only           Run the pre-GPU prereq/quiesce checks then exit 0 without measuring.
    --calibration-bootstrap    Skip the BASELINE_CALIBRATION serial-band check and mark for authoring.
    --local-dev                LOCAL-DEV mode: a failed pair retries up to a budget (target-pairs x4)
                               and MLXFAST_MAX_DRAFT_DEPTH may raise the --mtp-depth cap. ABSENT
                               (default) = OFFICIAL: a failed pair is an immediate die 5 and the
                               depth cap is the readonly submission-proof constant (env ignored).
                               It is also the ONLY way to run the paired harness against a track
                               whose --contract fixture is not armed (see below).

ARM STATE:
    An OFFICIAL (non --local-dev) measuring run REFUSES (die 8, pre-GPU) unless the --contract track
    fixture declares `official_scoring_enabled: true`. `false` and ABSENT both refuse — an absent arm
    state is not an armed one. --preflight-only is NOT gated (it seals nothing). Only the track
    fixture arms a track; benchd never overrides it.

ALLOWED MODES:
    The modes a submission may declare are the --contract track fixture's `allowed_modes` when it
    declares one, and [serial, mtp] when it does not. ABSENT is not a widening: a fixture opts IN,
    so declaring `dflash` on one track cannot enable it anywhere else. A declared list must contain
    `serial` (the baseline denominator is pinned serial), must not repeat an entry, and may only
    name serial | mtp | dflash — `dspark` is reserved and refuses by name. A candidate declaring a
    mode outside the list REFUSES die 8, pre-GPU.

    `dflash` is SINGLE-STREAM ONLY (the engine's cohort driver refuses it by name), so a dflash
    candidate keeps the single-stream free-run series even on a fixture that pins
    `scored_batch_size`. The regime it ran is sealed in results.timed_mode, and the overlay's §5
    series fence keeps the single-stream and batched-cohort series from ever being pooled.

ENV (R14):
    QMTP_TARGET_DIR            Backbone/target weights dir (the --weights fallback).
    QMTP_HEAD_DIR              Pinned native MTP head (serial leg); existence-checked when set (die 8).
    QMTP_CANDIDATE_HEAD_DIR    Candidate-leg BYO head; defaults to QMTP_HEAD_DIR when unset.
    QMTP_DFLASH_HEAD_DIR       Pinned DFlash drafter (serial leg); existence-checked when set (die 8).
                               REQUIRED when the candidate declares mode dflash — without it both
                               legs would fall back to the engine's CWD-relative ./dflash-head and
                               load the SAME drafter regardless of workspace.
    QMTP_CANDIDATE_DFLASH_HEAD_DIR
                               Candidate-leg DFlash drafter; defaults to QMTP_DFLASH_HEAD_DIR.
    BASELINE_CALIBRATION       JSON calibration file; the pooled serial mean is band-checked against
                               it after measuring (die 6 on drift or decode_tokens mismatch).
    BASELINE_BAND_ENFORCE      Default 1: a MISSING calibration fails closed (die 6). Set 0 to allow.
    GPU_LOADED_UTIL            Telemetry loaded/steady util threshold (default 0.70; env-driven).

The dropped `MLXFAST_PAIRED_BASELINE_*` env / `--baseline-*` flags (the baseline is a
WORKSPACE now) are a hard mutual-exclusion error if present alongside this subcommand.
";

const OVERLAY_TIMING_USAGE: &str = "\
benchd overlay-timing — Option-A seam 3 (LOCAL): merge gates + timing into a LOCAL/parity score.json (the organizer owns the published ranked seal)

USAGE:
    benchd overlay-timing --gates-score <gates-score.json> --results <results.json> \\
        --score-path <score.json> [--integrity <benchmark-integrity.json>] [--contract <fixture.json>]

benchd's LOCAL merge and the verifiable seam-3 PARITY reference: it overlays the measure-job
<results.json> (seam 2) onto the seam-1 producer's sealed <gates-score.json> and seals the ranked
<score.json> (+ bare-basename .sha256). Aggregation is the 3.8 MEDIAN regime (median of the
per-prompt raw ratio-of-means; per-pair bound 8.0; floor 0.90 / ceiling 5.0). On the RANKED path
the organizer's trusted shell authors score.json (OPEN-2); this subcommand is LOCAL-only.

Flips partial_result → false, recomputes ALL floor fields coherently for the decode-only paired
track (finding 11), stamps a `scoring_mode` discriminator, and re-anchors integrity `score_sha256`
over the merged bytes (into --integrity when given, else a fresh benchmark-integrity.json).

Exit 0 when the merged score PASSES (median in [floor, ceiling]); nonzero when a floor/ceiling
bound fails (score=null); 2 on a usage error; 1 on a load/validation/IO error.

REQUIRED:
    --gates-score <PATH>  The seam-1 producer's sealed gates-score.json (partial_result=true)
    --results <PATH>      The measure-job results.json (seam 2 superset)
    --score-path <PATH>   Output LOCAL/parity score.json (+ .sha256 sidecar; organizer owns the ranked seal)

OPTIONS:
    --integrity <PATH>    An existing benchmark-integrity.json to RE-ANCHOR (score_sha256/score_path
                          rewritten over the merged bytes). Absent ⇒ a fresh sidecar next to the score.
    --contract <PATH>     R17 pool-shape source: the fixture whose `timed_prompt_pool | length` sets
                          the expected pool_size when MLXFAST_QWEN_MTP_POOL_SIZE is unset (fail-closed).
    -h, --help            Show this help
";

/// `gates_producer` when no driver declared one — a standalone `measure-job` with no seam 1.
///
/// A DECLARED sentinel rather than an empty string, for the #132/F3 reason: a reader must never
/// have to guess whether benchd failed to record the producer or there was genuinely none.
const GATES_PRODUCER_UNDECLARED: &str = "undeclared";

/// Validate a `--gates-producer` value before it is sealed into an artifact.
///
/// Deliberately NOT an allowlist of the three known producer names. measure-job does not own the
/// driver's producer vocabulary, and hardcoding it here would mean a new producer could not be
/// recorded without changing this file — the seal is PROVENANCE, not a policy gate.
///
/// What is enforced is that the value can be read back as what it claims to be: non-empty, and no
/// whitespace or control characters, so a declaration cannot smuggle a second field, a newline or a
/// terminal escape into a sealed record that a human or a parser later reads.
fn validate_gates_producer(raw: &str) -> Result<String, String> {
    if raw.is_empty() {
        return Err(
            "--gates-producer must not be empty (omit the flag to record 'undeclared')".into(),
        );
    }
    if let Some(bad) = raw.chars().find(|c| c.is_whitespace() || c.is_control()) {
        return Err(format!(
            "--gates-producer {raw:?} contains {bad:?}: the value is sealed into an artifact, \
             so whitespace and control characters are refused"
        ));
    }
    Ok(raw.to_string())
}

/// The SHARED sealed-write for the measure-job `results.json` (+ `.sha256` sidecar), used by
/// `measure-job` (A-1, finding 14 — one sealed-write recipe, not a forked copy). UNLIKE the
/// [`write_score`] sidecar — whose BODY carries the full score path to byte-match benchmark.sh
/// (`printf '%s  %s\n' "${score_hash}" "${SCORE_PATH}"`, benchmark.sh:1269-1270), pinned by the
/// failing-run sidecar test — this sidecar uses the BARE BASENAME (`<hex>  results.json\n`), so
/// `shasum -c results.json.sha256` verifies the downloaded artifact in its own directory.
/// Returns the sealed digest so the caller can anchor it inside `benchmark-integrity` (finding 10).
fn write_results_json(results_path: &Path, json: &str) -> Result<String, String> {
    if let Some(parent) = results_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
        }
    }
    std::fs::write(results_path, json.as_bytes())
        .map_err(|e| format!("could not write {}: {e}", results_path.display()))?;
    let basename = results_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    let sidecar = results_path.with_file_name(format!("{basename}.sha256"));
    let digest = sha256_hex(json.as_bytes());
    std::fs::write(&sidecar, format!("{digest}  {basename}\n"))
        .map_err(|e| format!("could not write {}: {e}", sidecar.display()))?;
    Ok(digest)
}

/// Re-anchor the integrity `score_sha256` over the merged score bytes. When `--integrity` names an
/// existing sidecar (e.g. the measure-job's `benchmark-integrity.results.json`), its
/// `score_sha256`/`score_path` are rewritten in place and every other field is preserved. Absent,
/// a fresh minimal `benchmark-integrity.json` is written next to the ranked score.
fn reanchor_overlay_integrity(args: &OverlayTimingArgs, score_sha256: &str) -> Result<(), String> {
    // F-5 — relativise at seal: the score path is provenance and must not carry a home directory.
    let score_path_str = relativize_for_seal(&args.score_path);
    if let Some(integrity_path) = args.integrity.as_ref() {
        let bytes = std::fs::read(integrity_path).map_err(|e| {
            format!(
                "--integrity read failed ({}): {e}",
                integrity_path.display()
            )
        })?;
        let mut value: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("--integrity parse failed: {e}"))?;
        let obj = value
            .as_object_mut()
            .ok_or("--integrity is not a JSON object")?;
        obj.insert(
            "score_sha256".to_string(),
            serde_json::Value::String(score_sha256.to_string()),
        );
        obj.insert(
            "score_path".to_string(),
            serde_json::Value::String(score_path_str),
        );
        let json = serde_json::to_string_pretty(&value)
            .map_err(|e| format!("--integrity serialization failed: {e}"))?;
        std::fs::write(integrity_path, format!("{json}\n"))
            .map_err(|e| format!("could not write {}: {e}", integrity_path.display()))?;
    } else {
        let path = args.score_path.with_file_name("benchmark-integrity.json");
        let sidecar = OverlayIntegrity {
            score_path: score_path_str,
            score_sha256: score_sha256.to_string(),
        };
        let json = serde_json::to_string_pretty(&sidecar)
            .map_err(|e| format!("integrity serialization failed: {e}"))?;
        std::fs::write(&path, format!("{json}\n"))
            .map_err(|e| format!("could not write {}: {e}", path.display()))?;
    }
    Ok(())
}

/// Load ONE `measure-job --golden` as EITHER document shape, routed by REQUIRED-KEY SIGNATURE.
///
/// The live `timed_prompt_pool` pins TEACHER-FORCING TAPES (`{seed_tokens,
/// reference_seed_token, rows, reference_self_consistent, emitted_tokens}`), not
/// `GoldenDocument`s — so `--golden` accepts both, and the loader must decide which is in front
/// of it BEFORE parsing.
///
/// DETECTION IS BY REQUIRED-KEY SIGNATURE, never by "whatever parses first"
/// ([`bench_core::tape::classify_golden_input`]). The two signatures are disjoint — a tape must
/// carry `seed_tokens`/`reference_seed_token`/`rows`, a GoldenDocument must carry `cases`, and
/// both structs are `deny_unknown_fields`, so neither document can parse as the other — which is
/// why no `--golden-kind` flag is needed. Signature routing also keeps the DIAGNOSTIC honest: a
/// tape with one broken row is reported as a broken TAPE, naming the real defect, instead of
/// being retried as a GoldenDocument and reported as "unknown field `emitted_tokens`" (exactly
/// the misleading message the 20260819 window hit). A file matching NEITHER signature is
/// refused naming BOTH shapes rather than guessed at.
///
/// Both branches hash the RAW BYTES they were handed, so the identity R4 pins against the
/// contract pool is the same quantity for either shape.
///
/// #114 — `reference_model` is the track contract's declared reference-model identity. It only
/// bites on the GoldenDocument branch: a `model_provenance` block is a GoldenDocument key, and a
/// timed-prompt tape carries no model identity at all (its identity is the pool's sha256+bytes
/// pin). Passing it here rather than checking after the load keeps the reference's ordering — a
/// golden naming the wrong model is refused by the LOADER, not accepted and re-judged later.
fn load_timed_prompt_checked(
    path: &Path,
    reference_model: Option<&ReferenceModelPin>,
    track_id: &str,
    identity: &TrackModelIdentity,
) -> Result<measure_job::TimedPrompt, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("golden read failed ({}): {e}", path.display()))?;
    match bench_core::tape::classify_golden_input(&bytes) {
        bench_core::tape::GoldenInputKind::TimedPromptTape => {
            // Pin `None` here: measure-job's pin is the contract's `timed_prompt_pool` (R4,
            // exactly-one, fail-closed), enforced on these same raw bytes right after loading.
            bench_core::tape::load_timed_prompt_tape(&bytes, identity, None)
                .map(measure_job::TimedPrompt::Tape)
                .map_err(|e| format!("timed-prompt tape load failed ({}): {e}", path.display()))
        }
        // #112 (L2) — SINGLE-READ, like the tape branch above: the bytes are already in hand, so
        // the classification, the loader's sha256 and the parsed document all describe the SAME
        // read. (Pin `None` for the same reason as the tape branch: measure-job's pin is the
        // contract pool, enforced on these bytes right after loading.)
        // Arity: the loader DEFAULT (`CORRECTNESS_STEPS`) — a measure-job golden is a timed
        // prompt source, not a local-iterate checked-decode window.
        bench_core::tape::GoldenInputKind::GoldenDocument => load_golden_bytes_checked(
            &bytes,
            None,
            CORRECTNESS_STEPS,
            reference_model,
            track_id,
            identity,
        )
        .map(measure_job::TimedPrompt::Golden)
        .map_err(|e| format!("{e} ({})", path.display())),
        bench_core::tape::GoldenInputKind::Unrecognized => Err(format!(
            "--golden {} matches NEITHER accepted shape: a {} needs {:?}, a {} needs {:?} \
             (the live timed_prompt_pool pins tapes)",
            path.display(),
            measure_job::PROMPT_KIND_TAPE,
            bench_core::tape::TAPE_REQUIRED_KEYS,
            measure_job::PROMPT_KIND_GOLDEN,
            bench_core::tape::GOLDEN_DOCUMENT_REQUIRED_KEYS,
        )),
    }
}

/// Per-shape counts of the loaded `--golden` set, for the `--preflight-only` line: an operator
/// proving satisfiability offline should be able to READ which shape was accepted, not infer it.
fn golden_kind_summary(prompts: &[measure_job::TimedPrompt]) -> String {
    let tapes = prompts
        .iter()
        .filter(|p| p.kind() == measure_job::PROMPT_KIND_TAPE)
        .count();
    format!(
        "{}={} {}={}",
        measure_job::PROMPT_KIND_TAPE,
        tapes,
        measure_job::PROMPT_KIND_GOLDEN,
        prompts.len() - tapes,
    )
}

/// Assemble `benchmark-integrity.results.json` from the declared args + the run-derived values.
///
/// EXTRACTED FROM [`execute_measure_job`] DELIBERATELY. This is the only production code that puts
/// the declared `--gates-producer` into an artifact, and while it lived inline in
/// `execute_measure_job` — which needs a GPU, two workspaces and a real pair loop to reach — no
/// test could execute it. A mutation hardcoding the sealed producer left the whole workspace suite
/// AND the offline driver suite green, because the driver suite's sidecar comes from a bash stub
/// that re-implements the behaviour: stub agreeing with stub.
///
/// Pulling the assembly out makes the seal a pure function over its inputs, so the ruling-Q1a
/// audit trail is pinned by a test that runs the REAL line rather than a copy of it.
fn build_measure_job_integrity(
    args: &MeasureJobArgs,
    run: MeasureJobSealInputs,
) -> MeasureJobIntegrity {
    MeasureJobIntegrity {
        // F-5 — relativise every path at seal so the anchor carries no operator home directory.
        results_path: relativize_for_seal(Path::new(&run.results_path)),
        results_sha256: run.results_sha256,
        candidate_workspace: relativize_for_seal(&args.candidate),
        baseline_workspace: relativize_for_seal(&args.baseline),
        candidate_executable: relativize_for_seal(Path::new(&run.candidate_executable)),
        baseline_executable: relativize_for_seal(Path::new(&run.baseline_executable)),
        gates_producer: args.gates_producer.clone(),
        candidate_workspace_sha256: run.candidate_workspace_sha256,
        baseline_workspace_sha256: run.baseline_workspace_sha256,
        golden_sha256: run.golden_sha256,
        contract_sha256: run.contract_sha256,
        weights_dir: relativize_for_seal(&args.weights),
        weights_sha256: run.weights_sha256,
        weights_file_count: run.weights_file_count,
        weights_byte_count: run.weights_byte_count,
    }
}

/// A fresh overlay integrity sidecar (when no `--integrity` to re-anchor is given): the ranked
/// score path + its sha256 over the merged bytes.
#[derive(serde::Serialize)]
struct OverlayIntegrity {
    score_path: String,
    score_sha256: String,
}

/// The measure-job integrity sidecar (finding 10): the sealed `results.json` digest lives
/// INSIDE this anchor, alongside the workspace/executable provenance.
#[derive(serde::Serialize)]
struct MeasureJobIntegrity {
    results_path: String,
    results_sha256: String,
    candidate_workspace: String,
    baseline_workspace: String,
    candidate_executable: String,
    baseline_executable: String,
    /// CANDIDATE-IDENTITY provenance: the sha256 of the candidate WORKSPACE tree (the built engine
    /// source). Distinct from the weights identity below.
    candidate_workspace_sha256: String,
    /// BASELINE-IDENTITY provenance (integrity-anchor minor): the sha256 of the baseline WORKSPACE
    /// tree, so the seal pins BOTH legs' built-engine sources, not only the candidate's.
    baseline_workspace_sha256: String,
    /// GOLDEN-IDENTITY provenance (integrity-anchor minor): the sha256 of the actual `--golden`
    /// bytes (== `GoldenFixture::sha256`) — the prompt oracle this run measured against.
    golden_sha256: String,
    /// CONTRACT-IDENTITY provenance (integrity-anchor minor): the sha256 of the `--contract` track
    /// fixture bytes — the thresholds/pool the run was configured from.
    contract_sha256: String,
    /// WEIGHTS-IDENTITY provenance: the `--weights` DIR and its digest (the transformed weights
    /// both legs load). `weights_sha256`/`weights_file_count`/`weights_byte_count` digest THIS dir,
    /// not the workspace.
    weights_dir: String,
    weights_sha256: String,
    weights_file_count: i64,
    weights_byte_count: i64,
    /// RULING Q1a — WHICH seam-1 gates producer made the gates this run was scored against
    /// (`benchmark-sh` = the organizer's reference chain and the default, `facade` = benchd's own
    /// `--official`, `direct-swift` = the weightless fallback), or
    /// [`GATES_PRODUCER_UNDECLARED`] when no driver declared one.
    ///
    /// WHY THIS ARTIFACT. The producer is a seam-1 fact, but seam 1's own output (`gates-score.json`)
    /// is written BY the producer — so it cannot be trusted to name itself. This sidecar is the
    /// first artifact in the chain that benchd writes and the driver anchors, which makes it the
    /// earliest honest home for the declaration.
    ///
    /// WHY IT MATTERS. The opt-in is an ENVIRONMENT variable, so an exported `GATES_PRODUCER=facade`
    /// can select the parity-test producer for a scoring run without appearing in any command line.
    /// Sealing it here does not prevent that — it makes it AUDITABLE after the fact, which is what
    /// lets the env-var opt-in stand instead of forcing an argv-only interface.
    gates_producer: String,
}

/// The RUN-DERIVED half of the measure-job integrity seal — everything measured or digested
/// during the run, as opposed to declared on the command line.
///
/// Separate from [`MeasureJobArgs`] so [`build_measure_job_integrity`] states exactly which fields
/// come from the operator and which come from the run.
struct MeasureJobSealInputs {
    results_path: String,
    results_sha256: String,
    candidate_executable: String,
    baseline_executable: String,
    candidate_workspace_sha256: String,
    baseline_workspace_sha256: String,
    golden_sha256: String,
    contract_sha256: String,
    weights_sha256: String,
    weights_file_count: i64,
    weights_byte_count: i64,
}

/// R6/R14: the env the draft passes the transformed-weights dir through on-box (draft@064c0ff2:2084).
/// `--weights` is an OPTIONAL OVERRIDE of this; when neither is set, measure-job fails closed. R14 —
/// RENAMED from the dead `QWEN_MTP_TARGET_DIR` to `QMTP_TARGET_DIR` (live wrapper W:370).
const WEIGHTS_ENV_VAR: &str = "QMTP_TARGET_DIR";

/// Parsed `measure-job` flags (A-1, seam 2).
#[derive(Debug)]
struct MeasureJobArgs {
    candidate: PathBuf,
    baseline: PathBuf,
    weights: PathBuf,
    /// R13 — `--golden` is REPEATABLE → a Vec (non-empty). R7 — the pair loop measures EVERY
    /// golden in the pool (one per_prompt record per golden), with the dup-digest guard up front.
    goldens: Vec<PathBuf>,
    contract: PathBuf,
    /// LANE 2a — `--correctness-golden <path>`: the run's correctness-golden ATTESTATION. The staged
    /// hidden correctness golden the run verified token-fidelity against; benchd hashes it (sha256 +
    /// bytes) and verifies that identity CITES the `--contract` fixture's `hidden_correctness_golden`
    /// SIBLING pin (fail-closed both directions). Absent on offline/legacy tracks whose fixture pins
    /// no correctness golden; REQUIRED once the fixture declares one.
    correctness_golden: Option<PathBuf>,
    /// R13/W3 — `--tokens`. Default 512 (`DEFAULT_TOKENS`) on the teacher-forced path;
    /// `FREE_RUN_DECODE_TOKENS` (128, RULED) on the v1.1 free-run path, where any other explicit
    /// value is a usage error.
    tokens: usize,
    /// W3 — the candidate leg's TIMED REGIME, derived from `candidate_spec` by
    /// `measure_job::candidate_regime_for_spec` (the single production rule). The serial control leg
    /// is always teacher-forced.
    candidate_regime: measure_job::LegRegime,
    /// R13 — `--mtp-depth` (replaces `--depth`): candidate MTP depth. Derived from
    /// `candidate_spec.mtp.depth`; sealed as the `mtp_depth` mirror (`Some(0)` for a non-mtp
    /// candidate, matching the serial-control vocabulary). `None` = the operator requested no
    /// depth and the ENGINE'S DRAFTER DECIDES (David ruling 2026-08-27) — the seal then carries no
    /// `mtp_depth` at all rather than presenting benchd's retired default as if it described the
    /// run; the operating depth is what the engine's `effective_spec` echo reports.
    mtp_depth: Option<usize>,
    /// spec (docs/spec-config-design.md) — the resolved candidate/baseline declared specs + sources.
    candidate_spec: bench_protocol::SpecConfig,
    baseline_spec: bench_protocol::SpecConfig,
    candidate_spec_source: String,
    baseline_spec_source: String,
    /// R13 — `--min-pairs` (alias `--min-pairs-per-prompt`): PER-PROMPT floor, >= 1.
    min_pairs: usize,
    /// R13 — `--target-pairs` (alias `--pairs-per-prompt`): PER-PROMPT target, >= min.
    target_pairs: usize,
    tag: String,
    out: PathBuf,
    /// R13 — the `--prompt`/`--prompt-sha256`/`--target-id` trio (all-three-or-none). Parsed +
    /// validated + RECOGNISED here; the sealed `evaluation_target` shape is R16 (DEFERRED).
    prompt: Option<PathBuf>,
    prompt_sha256: Option<String>,
    target_id: Option<String>,
    /// R13 — `--exactness-probe` mode (default `once`), STORED. The untimed `mtp-verify` gate that
    /// consumes it is R15 (DEFERRED — parse + validate + store only).
    exactness_probe: measure_job::ExactnessProbe,
    /// R13 — `--preflight-only`: run the pre-GPU prereq/quiesce checks then exit 0 without measuring.
    preflight_only: bool,
    /// R13 — `--calibration-bootstrap`: skip the R14 serial-band check + mark the run for authoring.
    calibration_bootstrap: bool,
    /// H6/H3 (cycle-3) — `--local-dev`: enable the pair-attempt budget loop + honor the
    /// `MLXFAST_MAX_DRAFT_DEPTH` cap override. Absent (default) = OFFICIAL: immediate die-5 on a
    /// failed pair, readonly submission-proof depth cap.
    local_dev: bool,
    /// Ruling Q1a — `--gates-producer`: WHICH seam-1 gates producer the driver actually used, for
    /// the seal. measure-job is seam 2 and does not run the gates, so it cannot observe this; the
    /// driver DECLARES it and measure-job records the declaration verbatim.
    ///
    /// Absent = [`GATES_PRODUCER_UNDECLARED`], which is the ANSWER, not a gap: a standalone
    /// measure-job (no driver, no seam 1) genuinely has no producer, and #132/F3's lesson is that
    /// an empty string must never be ambiguous.
    gates_producer: String,
    /// David ruling 2026-08-27 — `--write-gate-base <SHA>`: when present, the write-outside gate and
    /// the growth budget judge the candidate's own committed diff/state at `<sha>..HEAD` instead of
    /// tree-diffing the staged `--baseline`; the staged workspace remains the timing baseline only.
    ///
    /// ABSENT is the LEGACY behavior, byte-identical (the qwen38 track lane runs without the flag
    /// and must not move): both gates keep tree-diffing the staged baseline workspace.
    write_gate_base: Option<String>,
}

/// The refusal tail shared by the four DROPPED paired-baseline flags (finding 3): the baseline is a
/// WORKSPACE now, so an engine / seconds-per-token override is a hard mutual-exclusion error rather
/// than a silently-ignored flag. The refusal names the flag as typed, then this.
const MEASURE_JOB_DROPPED_BASELINE_FLAG: &str =
    " is not a measure-job flag: the baseline is a WORKSPACE (--baseline <WS>) now, \
     not an engine/seconds-per-token override";

/// The `measure-job` command line. [`MEASURE_JOB_USAGE`] is its prose; this is its grammar.
static MEASURE_JOB_FLAGS: &[FlagSpec] = &[
    FlagSpec::value(&["--candidate"]).required(),
    FlagSpec::value(&["--baseline"]).required(),
    // David ruling 2026-08-27 — the SUBMISSION'S OWN fork point from harness main. When given,
    // the write-outside gate and the growth budget judge `<SHA>..HEAD` inside the --candidate
    // repo and stop consulting the staged --baseline tree (which stays the paired TIMING
    // baseline). The sha shape is validated at the gates, which is where the fail-closed
    // refusal belongs.
    FlagSpec::value(&["--write-gate-base"]),
    FlagSpec::value(&["--weights"]),
    // R13 — REPEATABLE: each `--golden` appends to the set (the per-golden loop is R7).
    FlagSpec::repeatable(&["--golden"]).required(),
    FlagSpec::value(&["--contract"]).required(),
    // LANE 2a — the run's correctness-golden attestation (single, not repeatable like
    // `--golden`): the staged hidden correctness golden benchd hashes and verifies cites the
    // fixture's `hidden_correctness_golden` pin.
    FlagSpec::value(&["--correctness-golden"]),
    FlagSpec::value(&["--tokens"]),
    // R13 — `--mtp-depth` replaces `--depth`. Since the spec re-home
    // (docs/spec-config-design.md) depth is a MODULE field: `--mtp-depth D` is a convenience
    // that builds the default candidate spec `{"mode":"mtp","mtp":{"depth":D}}`; it is no
    // longer a scored knob (the spec is), and a `--candidate-spec` override supersedes it.
    FlagSpec::value(&["--mtp-depth"]),
    // spec (docs/spec-config-design.md, step 5) — explicit per-module spec overrides, recorded
    // as spec_source "cli-override". `--candidate-spec` supersedes `--mtp-depth`; the baseline
    // defaults to {"mode":"serial"} and `--baseline-spec` overrides it.
    FlagSpec::value(&["--candidate-spec"]),
    FlagSpec::value(&["--baseline-spec"]),
    // R13 — `--depth` was RENAMED to `--mtp-depth`; a helpful hard error, not a silent accept.
    FlagSpec::refused(
        &["--depth"],
        " was renamed to --mtp-depth (candidate MTP depth, >= 2; serial \
         control is the depth-0 constant)",
    ),
    // R13 — `--min-pairs`/`--target-pairs` and their per-prompt aliases set the SAME floor and
    // target; an alias keeps its own spelling in every refusal.
    FlagSpec::value(&["--min-pairs", "--min-pairs-per-prompt"]).required(),
    FlagSpec::value(&["--target-pairs", "--pairs-per-prompt"]).required(),
    FlagSpec::value(&["--tag"]).required(),
    // Ruling Q1a — the seam-1 producer the DRIVER used, recorded into the integrity seal.
    FlagSpec::value(&["--gates-producer"]),
    FlagSpec::value(&["--out"]).required(),
    // R13 — the `--prompt`/`--prompt-sha256`/`--target-id` trio (all-three-or-none, validated
    // in the build).
    FlagSpec::value(&["--prompt"]),
    FlagSpec::value(&["--prompt-sha256"]),
    FlagSpec::value(&["--target-id"]),
    // R13 — `--exactness-probe {none|once|per-prompt|per-pair}` (parse + validate; store).
    FlagSpec::value(&["--exactness-probe"]),
    // R13 — boolean flags (no value).
    FlagSpec::switch(&["--preflight-only"]),
    FlagSpec::switch(&["--calibration-bootstrap"]),
    // H6/H3 (cycle-3) — LOCAL-DEV mode: enables the pair-attempt BUDGET LOOP (up to
    // target_pairs × 4) and honors the `MLXFAST_MAX_DRAFT_DEPTH` env override of the depth cap.
    // ABSENT (the default) = OFFICIAL/ranked: a failed pair is an immediate die-5 and the depth
    // cap is the readonly submission-proof constant (env ignored).
    FlagSpec::switch(&["--local-dev"]),
    // finding 3: the dropped paired-baseline flags are a HARD mutual-exclusion error (the
    // baseline is a WORKSPACE now), not a silently-ignored flag.
    FlagSpec::refused(
        &["--baseline-prefill-spt"],
        MEASURE_JOB_DROPPED_BASELINE_FLAG,
    ),
    FlagSpec::refused(
        &["--baseline-decode-spt"],
        MEASURE_JOB_DROPPED_BASELINE_FLAG,
    ),
    FlagSpec::refused(&["--baseline-engine"], MEASURE_JOB_DROPPED_BASELINE_FLAG),
    FlagSpec::refused(&["--engine"], MEASURE_JOB_DROPPED_BASELINE_FLAG),
];

/// The `measure-job` subcommand: its name, its usage prose, its flag table and its parse.
static MEASURE_JOB_CLI: CliSpec<MeasureJobArgs> = CliSpec {
    name: "measure-job",
    usage: MEASURE_JOB_USAGE,
    flags: MEASURE_JOB_FLAGS,
    parse: parse_measure_job_args,
};

/// A-1: the Option-A MEASURE-JOB subcommand (seam 2). Parses the workspace CLI, runs the
/// alternating pair loop, and seals `<out>/results.json` (+ bare-basename `.sha256` and the
/// `benchmark-integrity.results.json` anchor). Exit 0 when the candidate is accepted
/// (`accepted >= --min-pairs`), else exit 5 (die 5 — candidate rejected); 2 on a usage error.
fn run_measure_job_cli(args: &[String]) -> ExitCode {
    run_cli(&MEASURE_JOB_CLI, args, |parsed| {
        // The verdict→exit mapping is extracted so it is unit-testable (a real ExitCode end-to-end
        // needs a live pair loop): accepted / preflight-ok → 0, die-5 (candidate rejected) → 5.
        // Finding R19 — a thermal-gate timeout is NOT a distinct exit; it folds into die-5 like
        // every reject class. A [`MeasureJobFailure`] carries its OWN honest exit (die-8 golden/sha
        // prereq, die-6 calibration, exit-1 IO/load), never collapsed to a single error code.
        execute_measure_job(parsed)
            .map(measure_job_exit_status)
            .map_err(|f| (f.exit, f.message))
    })
}

/// A measure-job execution failure that carries its OWN honest process exit code (R13/R19 exit
/// table): die-8 (prereq / golden / sha), die-6 (baseline/calibration drift — R14), or the generic
/// exit-1 load/IO path. A plain `String` error auto-converts to exit-1 via `From`, so the many
/// `?`-propagated IO errors keep the exit-1 path while the honest hard-die sites construct explicitly.
#[derive(Debug)]
struct MeasureJobFailure {
    exit: u8,
    message: String,
}

impl MeasureJobFailure {
    /// die-8 — a PRE-GPU prereq failure (golden dup-digest, `--prompt` file hash ≠ `--prompt-sha256`,
    /// a missing/unresolvable prerequisite, a missing head dir). R19 exit table: `8 prereq/golden/sha`.
    fn die8(message: impl Into<String>) -> Self {
        Self {
            exit: 8,
            message: message.into(),
        }
    }

    /// die-6 — a baseline/calibration failure caught BEFORE measuring (a malformed `BASELINE_CALIBRATION`
    /// file). The POST-measure serial-band drift is the [`MeasureJobVerdict::CalibrationDrift`] verdict
    /// (results.json sealed), also exit 6. R19 exit table: `6 baseline/calibration drift`.
    fn die6(message: impl Into<String>) -> Self {
        Self {
            exit: 6,
            message: message.into(),
        }
    }
}

impl From<String> for MeasureJobFailure {
    /// A bare error string is the generic exit-1 load/IO path (unchanged behaviour).
    fn from(message: String) -> Self {
        Self { exit: 1, message }
    }
}

/// The terminal verdicts of a measure-job run. Finding R19 (reverts R8) — a thermal-gate timeout no
/// longer produces a distinct exit-2 hard die: EVERY reject class (thermal, parity, implausible,
/// row-accounting, spawn/protocol infra) is retried once and folds into die-5 on persistence, so
/// the only rejection verdict is `RejectedDie5`. The exit `2` is reserved for a genuine usage/parse
/// error caught PRE-execution in `run_measure_job_cli`, never a mid-pair thermal event.
///
/// Honest exit table (measure-job): 0 accepted · 5 candidate rejected (pair fail incl.
/// thermal-after-retry, floor, accept-count) · 2 usage/parse error (pre-execution). (Die 6
/// calibration, die 8 prereq/golden/sha, die 9 lock, etc. are OTHER findings — hooks not
/// implemented here.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MeasureJobVerdict {
    /// `accepted_pairs >= --min-pairs`.
    Accepted,
    /// R13 — `--preflight-only`: the pre-GPU prereq/quiesce checks all passed and the run EXITED
    /// WITHOUT measuring (exit 0). Distinct from `Accepted` (no pair loop ran).
    PreflightOk,
    /// Fewer than `--min-pairs` accepted (die 5) — the candidate is rejected. Finding R19 — this is
    /// the ONLY candidate-rejection verdict; a persistent thermal-gate timeout folds into it (not exit-2).
    RejectedDie5,
    /// R14 — the POOLED serial mean drifted outside the `BASELINE_CALIBRATION` band (or a required
    /// calibration was missing under `BASELINE_BAND_ENFORCE`) after measuring: die-6. results.json is
    /// sealed (the calibration provenance records what was checked); the process exits 6.
    CalibrationDrift,
}

/// The verdict→process-exit contract for a measure-job run: an accepted candidate (or a passing
/// preflight-only run) exits 0; a die-5 rejected candidate exits 5 (finding R19 — the sole
/// candidate-rejection exit, thermal-after-retry included). Distinct from the exit-1 load/IO error
/// path, the exit-2 usage/parse path, and the die-8 prereq / die-6 calibration paths ([`MeasureJobFailure`]).
fn measure_job_exit_status(verdict: MeasureJobVerdict) -> u8 {
    match verdict {
        MeasureJobVerdict::Accepted => 0,
        MeasureJobVerdict::PreflightOk => 0,
        MeasureJobVerdict::RejectedDie5 => 5,
        MeasureJobVerdict::CalibrationDrift => 6,
    }
}

/// Parse a `measure-job` command line against [`MEASURE_JOB_CLI`].
fn parse_measure_job_args(args: &[String]) -> Result<Option<MeasureJobArgs>, String> {
    match parse_flags(args, MEASURE_JOB_CLI.flags)? {
        Some(flags) => build_measure_job_args(&flags).map(Some),
        // `-h`/`--help`.
        None => Ok(None),
    }
}

/// Build the typed `measure-job` arguments out of the parsed table: the per-flag conversions first
/// (in the order this function reads them, where the hand-written loop converted in argv order, so
/// a command line with two separately invalid values names the one the build reaches first), then
/// the cross-flag validation in the order the refusals have always been raised.
fn build_measure_job_args(flags: &ParsedFlags) -> Result<MeasureJobArgs, String> {
    // W3 — `--tokens` stays an Option here: its DEFAULT is regime-dependent (teacher-forced 512 =
    // the live wrapper window; v1.1 free-run 128 = the RULED N), so it is resolved AFTER the
    // candidate spec (and therefore the regime) is known, below.
    let tokens_flag = flags.usize_flag("--tokens")?;
    let mtp_depth = flags.usize_flag("--mtp-depth")?;
    let candidate_spec_json = flags.string("--candidate-spec");
    let baseline_spec_json = flags.string("--baseline-spec");
    let min_pairs = flags.usize_flag("--min-pairs")?;
    let target_pairs = flags.usize_flag("--target-pairs")?;
    let gates_producer = flags.mapped("--gates-producer", |_, v| validate_gates_producer(v))?;
    let exactness_probe = flags.mapped("--exactness-probe", |_, v| {
        measure_job::ExactnessProbe::parse(v)
    })?;
    let prompt = flags.path("--prompt");
    let prompt_sha256 = flags.string("--prompt-sha256");
    let target_id = flags.string("--target-id");
    let local_dev = flags.present("--local-dev");

    // finding 3: the trusted paired-baseline ENV override must not coexist with measure-job
    // either — the baseline is a workspace, so a present env is an operator wiring error.
    for key in [
        "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN",
        "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN",
    ] {
        if std::env::var(key)
            .ok()
            .is_some_and(|v| !v.trim().is_empty())
        {
            return Err(format!(
                "{key} is set, but measure-job takes the baseline as a WORKSPACE (--baseline <WS>); \
                 unset the paired-baseline env override (mutually exclusive)"
            ));
        }
    }

    let candidate = flags
        .path("--candidate")
        .ok_or_else(|| flags.missing_required("--candidate"))?;
    let baseline = flags
        .path("--baseline")
        .ok_or_else(|| flags.missing_required("--baseline"))?;
    // R6: the approved draft measure-job CLI (draft@064c0ff2:2088-2098) carries NO --weights; the
    // draft passes the weights dir on-box as env QMTP_TARGET_DIR (R14 rename). So --weights is an
    // OPTIONAL OVERRIDE: when omitted we DERIVE the weights dir from QMTP_TARGET_DIR; when
    // NEITHER is provided we fail closed with a clear message rather than guessing a path.
    // UNVERIFIED(measure-job): QMTP_TARGET_DIR is the draft's on-box weights source.
    let weights = match flags.path("--weights") {
        Some(w) => w,
        None => {
            let env_dir = std::env::var(WEIGHTS_ENV_VAR)
                .ok()
                .filter(|s| !s.trim().is_empty());
            match env_dir {
                Some(dir) => PathBuf::from(dir.trim()),
                None => {
                    return Err(format!(
                        "no weights directory: pass --weights <DIR>, or set the env \
                         {WEIGHTS_ENV_VAR} (the draft's on-box weights source) — neither is set"
                    ));
                }
            }
        }
    };
    let goldens = flags.paths("--golden");
    if goldens.is_empty() {
        return Err(flags.missing_required("--golden"));
    }
    let contract = flags
        .path("--contract")
        .ok_or_else(|| flags.missing_required("--contract"))?;
    // Medium (#105) — `--mtp-depth` is a CONVENIENCE that builds the default candidate spec; a
    // `--candidate-spec` override SUPERSEDES it. Silently discarding an explicit `--mtp-depth` when
    // `--candidate-spec` is also given hid an operator wiring conflict, so it is now an EXPLICIT hard
    // error (mutually exclusive) rather than a quiet no-op.
    if candidate_spec_json.is_some() && mtp_depth.is_some() {
        return Err(
            "--mtp-depth and --candidate-spec are mutually exclusive: --candidate-spec is the \
             explicit spec, and --mtp-depth is only the convenience that builds the default one — \
             pass exactly one (drop --mtp-depth, or fold the depth into --candidate-spec)"
                .to_string(),
        );
    }
    // David ruling 2026-08-27 — NO built-in default depth. Depth is the participant's variable
    // (their drafter code sets it); an omitted `--mtp-depth` therefore builds the engine-decides
    // spec `{"mode":"mtp","mtp":{}}` rather than injecting benchd's retired DEFAULT_MTP_DEPTH as a
    // request the engine is then forced to echo. The flag stays for explicit experiments.
    let min_pairs = min_pairs.ok_or_else(|| flags.missing_required("--min-pairs"))?;
    let target_pairs = target_pairs.ok_or_else(|| flags.missing_required("--target-pairs"))?;
    let tag = flags
        .string("--tag")
        .ok_or_else(|| flags.missing_required("--tag"))?;
    let out = flags
        .path("--out")
        .ok_or_else(|| flags.missing_required("--out"))?;
    // R13 — candidate --mtp-depth must be >= 2 (depth 0 = serial control, depth 1 = diagnostic).
    // David ruling (cycle-3) — plus a DEFENSIVE upper CAP (> cap rejected before GPU work),
    // SUBMISSION-PROOF like the engine: OFFICIAL (default) uses the readonly constant 32 and IGNORES
    // MLXFAST_MAX_DRAFT_DEPTH; --local-dev honors the env override.
    let max_draft_depth_cap = measure_job::resolve_max_draft_depth_cap(
        local_dev,
        std::env::var(measure_job::MAX_DRAFT_DEPTH_ENV)
            .ok()
            .as_deref(),
    );

    // spec (docs/spec-config-design.md, steps 4/5) — resolve the per-leg declared specs.
    // Candidate: a `--candidate-spec` JSON override (spec_source "cli-override"), else `--mtp-depth D`
    // builds `{"mode":"mtp","mtp":{"depth":D}}` (spec_source "mtp-depth-flag"), else the
    // ENGINE-DECIDES default `{"mode":"mtp","mtp":{}}` (spec_source "mtp-engine-default") — David
    // ruling 2026-08-27: depth is the participant's variable, so an omitted flag names NO depth and
    // the engine's module resolves its own, reporting it in the effective_spec echo. Baseline: a
    // `--baseline-spec` override, else `{"mode":"serial"}` (spec_source "serial-default").
    let (candidate_spec, candidate_spec_source) = match (&candidate_spec_json, mtp_depth) {
        (Some(json), _) => (
            measure_job::parse_spec_override(json)?,
            measure_job::SPEC_SOURCE_CLI_OVERRIDE.to_string(),
        ),
        (None, Some(depth)) => (
            // Medium (#105) — u32 TRUNCATION GUARD: `mtp.depth` is a u32 module field, so a usize
            // `--mtp-depth` that does not fit u32 must ERROR, never wrap silently to a small value
            // that would sneak under the depth cap. (A cast `as u32` truncates; try_from does not.)
            bench_protocol::SpecConfig::mtp(u32::try_from(depth).map_err(|_| {
                format!(
                    "--mtp-depth {depth} does not fit a u32 (mtp.depth is a u32 module field); \
                     a plausible depth is a small integer bounded by the {} draft-depth cap",
                    measure_job::DEFAULT_MAX_DRAFT_DEPTH_CAP
                )
            })?),
            measure_job::SPEC_SOURCE_MTP_DEPTH_FLAG.to_string(),
        ),
        (None, None) => (
            bench_protocol::SpecConfig::mtp_engine_default(),
            measure_job::SPEC_SOURCE_MTP_ENGINE_DEFAULT.to_string(),
        ),
    };
    let (baseline_spec, baseline_spec_source) = match &baseline_spec_json {
        Some(json) => (
            measure_job::parse_spec_override(json)?,
            measure_job::SPEC_SOURCE_CLI_OVERRIDE.to_string(),
        ),
        None => (
            bench_protocol::SpecConfig::serial(),
            measure_job::SPEC_SOURCE_SERIAL_DEFAULT.to_string(),
        ),
    };
    // #105 H-B — the BASELINE is the SERIAL DENOMINATOR; it must NOT be CLI-steerable off serial.
    // A non-serial `--baseline-spec` is a HARD ERROR (pre-GPU): the serial control anchors the whole
    // ratio (serial = 1.0), so a caller cannot swap the denominator for a faster/slower regime and
    // inflate the speedup. `--candidate-spec` stays free; only the baseline is pinned.
    measure_job::validate_baseline_is_serial(&baseline_spec)?;
    // Depth-0-via-serial-mode: candidate validation keys on the MODE being in the track's allowed
    // list (not a depth-int floor). The 32 cap is re-homed as a bounds-check on the module's
    // `mtp.depth`. Both are pre-GPU usage errors.
    //
    // David ruling (2026-08-26) — the ALLOWED-MODES half of that check has MOVED to
    // `execute_measure_job`, immediately after `Contract::parse`. It had to: the list is now
    // CONTRACT DATA (`Contract::allowed_modes`) and the contract file is not read here, which is
    // exactly why the override `DEFAULT_ALLOWED_MODES` had always advertised never existed. What
    // stays here is the half that is CONTRACT-INDEPENDENT and therefore still a pure usage error:
    // the mode↔module COHERENCE shape (`{"mode":"mtp"}` with no mtp block, an `mtp(0)` candidate, a
    // cross-module `{"mode":"mtp","dflash":{…}}`) — malformed on every track, whatever any fixture
    // declares. The allowed-list refusal becomes a die-8 pre-GPU prereq instead of an exit-2 usage
    // error, which is the honest classification: "this track does not admit this mode" is a fact
    // about the track fixture, not about how the operator typed the command.
    measure_job::validate_spec_module_coherent(&candidate_spec)?;
    measure_job::validate_spec_module_coherent(&baseline_spec)?;
    measure_job::validate_spec_capped(&candidate_spec, max_draft_depth_cap)?;
    measure_job::validate_spec_capped(&baseline_spec, max_draft_depth_cap)?;
    // Keep the sealed `mtp_depth` mirror consistent with the candidate spec's module depth: a
    // requested depth seals verbatim; a non-mtp candidate seals `Some(0)` (the serial-control
    // vocabulary); an engine-decides mtp candidate (depth left to the module) seals NOTHING —
    // `None` — because a depth benchd never requested must not be presented as if it described the
    // run (David ruling 2026-08-27). The operating value lives in the engine's echoed
    // `effective_spec`, sealed per leg.
    let mtp_depth = match &candidate_spec.mtp {
        Some(m) => m.depth.map(|d| d as usize),
        None => Some(0),
    };

    // W3 — the CANDIDATE LEG'S TIMED REGIME, derived from the declared candidate spec by the single
    // production rule (`candidate_regime_for_spec`): a speculating candidate (mtp today, dflash when
    // it lands) is scored in the v1.1 FREE-RUN regime, because teacher forcing structurally cannot
    // execute speculation. A serial candidate stays teacher-forced. There is deliberately NO
    // separate `--free-run` flag: a second switch could drift from the declared spec.
    let candidate_regime = measure_job::candidate_regime_for_spec(&candidate_spec);
    // W3 — the decode window N. Its DEFAULT is regime-dependent, and on the free-run path an
    // EXPLICIT `--tokens` that is not the RULED N is a hard usage error rather than a silent
    // re-window: N divides the scored seconds-per-token, and the v1.1 series is defined at
    // N = BENCHMARK_DECODE_STEPS (PROTOCOL-v1.1 OQ3, RULED).
    // A zero window is invalid in EVERY regime, and its message must not be shadowed by the
    // regime-specific one below (an operator who typed `--tokens 0` needs to be told that, not told
    // about N).
    if tokens_flag == Some(0) {
        return Err("--tokens must be > 0 (a zero decode window is invalid)".to_string());
    }
    let tokens = match (candidate_regime.is_free_run(), tokens_flag) {
        (true, None) => measure_job::FREE_RUN_DECODE_TOKENS,
        (true, Some(t)) if t != measure_job::FREE_RUN_DECODE_TOKENS => {
            return Err(format!(
                "--tokens {t} is invalid for the v1.1 free-run series: PROTOCOL-v1.1 RULES N = {} \
                 (BENCHMARK_DECODE_STEPS). A speculating --candidate-spec/--mtp-depth selects the \
                 free-run regime, whose window is fixed; drop --tokens, or pass a serial \
                 --candidate-spec to measure the teacher-forced series at your own window.",
                measure_job::FREE_RUN_DECODE_TOKENS
            ));
        }
        (true, Some(t)) => t,
        // R13 — teacher-forced default 512 (the live wrapper window).
        (false, other) => other.unwrap_or(measure_job::DEFAULT_TOKENS),
    };
    if tokens == 0 {
        return Err("--tokens must be > 0 (a zero decode window is invalid)".to_string());
    }
    if min_pairs == 0 {
        return Err("--min-pairs (per prompt) must be >= 1".to_string());
    }
    if target_pairs < min_pairs {
        return Err(format!(
            "--target-pairs ({target_pairs}) must be >= --min-pairs ({min_pairs}) [per prompt]"
        ));
    }

    // R13 — the `--prompt`/`--prompt-sha256`/`--target-id` trio is ALL-THREE-OR-NONE; when present,
    // the sha is 64-lowercase-hex and the target-id matches [A-Za-z0-9._-]+ (the file-hash ==
    // --prompt-sha256 check is done in execute, where the file bytes are read → die-8 on mismatch).
    let trio_present = [
        prompt.is_some(),
        prompt_sha256.is_some(),
        target_id.is_some(),
    ];
    let present_count = trio_present.iter().filter(|p| **p).count();
    if present_count != 0 && present_count != 3 {
        return Err(
            "--prompt, --prompt-sha256 and --target-id are ALL-THREE-OR-NONE (an explicit prompt \
             must carry its pinned sha256 and target-id)"
                .to_string(),
        );
    }
    if let Some(s) = prompt_sha256.as_deref() {
        measure_job::validate_prompt_sha256(s)?;
    }
    if let Some(t) = target_id.as_deref() {
        measure_job::validate_target_id(t)?;
    }

    Ok(MeasureJobArgs {
        candidate,
        baseline,
        weights,
        goldens,
        contract,
        correctness_golden: flags.path("--correctness-golden"),
        tokens,
        candidate_regime,
        mtp_depth,
        candidate_spec,
        baseline_spec,
        candidate_spec_source,
        baseline_spec_source,
        min_pairs,
        target_pairs,
        tag,
        out,
        prompt,
        prompt_sha256,
        target_id,
        exactness_probe: exactness_probe.unwrap_or_default(),
        preflight_only: flags.present("--preflight-only"),
        calibration_bootstrap: flags.present("--calibration-bootstrap"),
        local_dev,
        gates_producer: gates_producer.unwrap_or_else(|| GATES_PRODUCER_UNDECLARED.to_string()),
        write_gate_base: flags.string("--write-gate-base"),
    })
}

/// R14/H6 — resolve the `BASELINE_CALIBRATION` file into a
/// [`measure_job::ResolvedCalibration`] (for the pre-
/// measure ceiling + the AFTER-measure serial-band die-6), FAIL-CLOSED on a malformed file or a
/// declared-but-missing `--target-id` entry. H6/H2 (cycle-3): under `--calibration-bootstrap` this
/// run AUTHORS the band, so it does NOT pre-read or require the file — a MISSING file is fine
/// (returns `None`), mirroring the wrapper's bootstrap early-return (W:1423-1426). Pure over its
/// inputs (path string + target-id) so the bootstrap-skips-missing-file behavior is unit-testable.
fn resolve_calibration_env(
    calibration_bootstrap: bool,
    calibration_path: Option<&str>,
    target_id: Option<&str>,
    run_timed_mode: &str,
    run_track_id: &str,
) -> Result<Option<measure_job::ResolvedCalibration>, MeasureJobFailure> {
    if calibration_bootstrap {
        // Authoring run: skip the pre-read entirely. write_calibration_bootstrap reads/merges the
        // file itself after an accepted+parity run; a missing file here is expected, not a die-6.
        return Ok(None);
    }
    let path = match calibration_path.map(str::trim).filter(|s| !s.is_empty()) {
        Some(p) => p,
        None => return Ok(None),
    };
    let bytes = std::fs::read(path).map_err(|e| {
        MeasureJobFailure::die6(format!("BASELINE_CALIBRATION read failed ({path}): {e}"))
    })?;
    let parsed =
        measure_job::BaselineCalibration::parse(&bytes).map_err(MeasureJobFailure::die6)?;
    // #105 cycle-5 (HIGH) — the SERIES FENCE runs FIRST, before the file is resolved to a band and
    // long before the pooled serial mean is banded against it: a calibration measured in another
    // series (or authored for another track) must never reach `evaluate_serial_band` at all. die-6.
    measure_job::enforce_calibration_series_fence(&parsed, run_timed_mode, run_track_id)
        .map_err(MeasureJobFailure::die6)?;
    // FAIL-CLOSED: a declared --target-id with no matching per-target entry is a miswired rotation
    // (die-6), never a fallback to the top-level baseline.
    Ok(Some(
        parsed.resolve(target_id).map_err(MeasureJobFailure::die6)?,
    ))
}

/// Read the dispatch sha record (`candidate.sha`) the in-repo dispatch script authored, if any.
/// The path is threaded via `MLXFAST_CANDIDATE_SHA_FILE`, which `official-paired.sh`'s seam-2
/// invocation sets when it records the dispatched sha (`run-paired-window.sh` reaches this only by
/// invoking `official-paired.sh` — it sets nothing itself). The dispatched sha itself comes from
/// the CI/yukon dispatch context (`MLXFAST_CANDIDATE_SHA`/`GITHUB_SHA`); wiring an outer dispatch to
/// EXPORT that context on the live scoring box is a separate dispatch-lane item, so on a scoring run
/// where the env is unset the seal fails closed in [`official::author_sealed_commit`] rather than
/// falling back to git. Presence of the env var means "a dispatch promised a record", so a
/// SET-but-unreadable path is a die-8 refuse. Unset ⇒ `None`. The trimmed contents are validated by
/// [`official::author_sealed_commit`], not here (that is the single seal authority).
fn read_dispatch_sha_record() -> Result<Option<String>, MeasureJobFailure> {
    let path = match std::env::var("MLXFAST_CANDIDATE_SHA_FILE") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => return Ok(None),
    };
    let raw = std::fs::read_to_string(&path).map_err(|e| {
        MeasureJobFailure::die8(format!(
            "author-at-seal: MLXFAST_CANDIDATE_SHA_FILE={path} is unreadable ({e}); refusing to \
             seal against an unidentified dispatch"
        ))
    })?;
    Ok(Some(raw.trim().to_string()))
}

/// Execute the measure-job: load inputs, run the pair loop over sandboxed workspace workers,
/// and seal the results. Returns `Ok(candidate_accepted)`. The live spawn wiring (workspace →
/// sandboxed worker) is un-mirrored and unit-tested via the pure `measure_job::run_measure_job`
/// core rather than here.
// UNVERIFIED(measure-job): the workspace→sandboxed-worker spawn wiring.
fn execute_measure_job(args: &MeasureJobArgs) -> Result<MeasureJobVerdict, MeasureJobFailure> {
    // #114 — the --contract track fixture is READ FIRST, before any golden, because the contract
    // is where this track's REFERENCE-MODEL IDENTITY is declared and the golden loader needs that
    // pin as an input (RULED: the contract is the pin authority, not a compiled-in constant). Read
    // ONCE here; the pool/thermal parse below and the sealed contract digest both reuse these same
    // bytes, so every contract-derived decision in this run describes one read of one file.
    let contract_bytes = std::fs::read(&args.contract)
        .map_err(|e| format!("--contract read failed ({}): {e}", args.contract.display()))?;
    // FAIL-CLOSED: an unreadable contract, or one declaring only half a reference-model pin, is an
    // error — never silently "this track pins no reference model".
    let reference_model = reference_model_pin_from_contract(&contract_bytes)
        .map_err(|e| MeasureJobFailure::die8(format!("--contract reference-model pin: {e}")))?;

    // R13 — `--golden` is REPEATABLE. Load EVERY golden fail-closed (loading also hashes its bytes),
    // then reject a DUPLICATE DIGEST (die-8, pre-GPU) — the same golden bytes passed twice. R7 — the
    // whole set is validated + dup-guarded up front, then the pair loop measures EVERY golden (one
    // per_prompt record per golden, bound by bytes).
    //
    // Each `--golden` is EITHER the live pool's teacher-forcing TAPE or a legacy GoldenDocument,
    // routed by required-key signature ([`load_timed_prompt_checked`]). The pool pins tapes, so the
    // tape is the shape a ranked run actually passes; the GoldenDocument path stays for the offline
    // fixtures and harnesses built on it.
    //
    // The TRACK is settled BEFORE the first golden, for the same reason the reference-model pin
    // is: the per-track MODEL IDENTITY (model_type, vocabulary bound, seed length) is an INPUT to
    // both loaders, so nothing can be validated until it is known. `resolve_track_id` stays the
    // ONE resolution — env `MLXFAST_QWEN_MTP_TRACK_ID` or the `--contract` fixture's own
    // `track_id`, a hard error when the two disagree — and its answer is the value sealed as
    // `track_id` further down. This is also the typed parse of the contract bytes read at the top
    // of this function, so every contract-derived decision still describes ONE read of ONE file.
    //
    // ONE PARSE. The measure-job used to parse these bytes TWICE — once into a four-field
    // `measure_job::Contract` shadow and once into the migrated schema — so one file had two typed
    // views that could disagree. The four fields moved onto the migrated schema; this is the only
    // parse, and every contract-derived decision below reads that one value.
    let contract = bench_core::contract::Contract::parse(&contract_bytes)?;
    let track_id = measure_job::resolve_track_id(
        std::env::var("MLXFAST_QWEN_MTP_TRACK_ID").ok().as_deref(),
        &contract,
    )?;
    let identity = bench_core::contract::model_identity(&contract, &track_id)
        .map(|r| r.value)
        .map_err(|e| MeasureJobFailure::die8(format!("--golden model identity: {e}")))?;

    let mut golden_fixtures = Vec::with_capacity(args.goldens.len());
    for g in &args.goldens {
        golden_fixtures.push(
            load_timed_prompt_checked(g, reference_model.as_ref(), &track_id, &identity)
                .map_err(MeasureJobFailure::die8)?,
        );
    }
    let golden_digests: Vec<String> = golden_fixtures
        .iter()
        .map(|g| g.sha256().to_string())
        .collect();
    measure_job::check_golden_digests(&golden_digests).map_err(MeasureJobFailure::die8)?;
    // R7 — the pair loop measures the WHOLE pool (`&golden_fixtures`), one per_prompt record per
    // golden bound BY BYTES. `golden`/`golden_path` below name goldens[0] only for the single-golden
    // sandbox read-grant + the integrity-anchor golden_sha256 (the worker gets its tokens over the
    // protocol, not by re-reading any golden file; per-golden sandbox read-grants are a later on-box
    // refinement, R15).
    let golden = &golden_fixtures[0];
    let golden_path = &args.goldens[0];

    // R13 — the `--prompt`/`--prompt-sha256`/`--target-id` trio is validated at parse (all-three-or-
    // none, sha/target-id shape). Here the file bytes are read: the `--prompt` file's sha256 MUST
    // equal the pinned `--prompt-sha256` (die-8 on mismatch — a pre-GPU prereq/integrity failure).
    if let (Some(pf), Some(pinned)) = (args.prompt.as_ref(), args.prompt_sha256.as_deref()) {
        let bytes = std::fs::read(pf).map_err(|e| {
            MeasureJobFailure::die8(format!("--prompt read failed ({}): {e}", pf.display()))
        })?;
        let actual = sha256_hex(&bytes);
        if actual != pinned {
            return Err(MeasureJobFailure::die8(format!(
                "--prompt file {} hashes to {actual}, but --prompt-sha256 pins {pinned} (die 8)",
                pf.display()
            )));
        }
    }

    // David ruling (2026-08-26) — THE TRACK MODE FENCE, contract-driven, die-8 PRE-GPU.
    //
    // "why the hell do we reject dflash" — because `DEFAULT_ALLOWED_MODES` was the ONLY list that
    // existed and it was consulted at CLI-parse time, before this file was read. The list is now
    // the fixture's `allowed_modes` when it declares one, and `DEFAULT_ALLOWED_MODES` when it does
    // not, so gemma4 can admit `dflash` without widening a single other track.
    //
    // FIRST of the contract-derived checks, immediately after the parse and BEFORE
    // `effective_candidate_regime` below: the regime resolution now takes the candidate's MODE as an
    // input, and resolving a regime for a mode this track never admitted would be answering the
    // second question before the first. It is also the cheapest refusal in the file — three string
    // comparisons, no filesystem — so an inadmissible mode costs nothing to reject.
    let allowed_modes = measure_job::enforce_track_allowed_modes(
        &args.candidate_spec,
        &args.baseline_spec,
        contract.allowed_modes.as_deref(),
    )
    .map_err(MeasureJobFailure::die8)?;
    // Medium (cycle-3) — every --golden must be PINNED: its sha256 resolves to EXACTLY ONE
    // timed_prompt_pool entry with a POSITIVE noop_decode_speedup, else die-8 BEFORE any GPU work
    // (wrapper noop_reference_for_golden W:663-679). An unpinned golden would otherwise burn gated
    // box time only to seal a results.json the ranked jq rejects (missing/<=0 per-prompt noop).
    measure_job::validate_goldens_pinned(&golden_fixtures, &contract.timed_prompt_pool)
        .map_err(MeasureJobFailure::die8)?;
    // Anti-lottery ≥N-DISTINCT COVERAGE gate (die-8, pre-GPU) — benchd is the FINAL validator. The
    // published ranked score is the MEDIAN over the pool of each prompt's raw serial-relative
    // ratio-of-means (docs/measure-job-contract.md@fe2da64, evicted; the all-8-median aggregation of
    // docs/parity-completion-gate.md §3), which is only well-defined when the run's TIMED coverage is
    // EXACTLY the full DISTINCT pinned pool. `validate_goldens_pinned` above accepts a SUBSET (each
    // golden pins individually); this gate additionally requires FULL, DISTINCT coverage of the
    // fixture's timed_prompt_pool — refusing a subset, a duplicate (<N distinct), or a substitution
    // (a timed prompt matching no pin) — so a scoring run can never publish a median over a
    // hand-picked support. Same pre-GPU point, same exit-8 path.
    measure_job::validate_timed_pool_coverage(&golden_fixtures, &contract.timed_prompt_pool)
        .map_err(MeasureJobFailure::die8)?;
    // COHORT (batch-8 brief D9, amended by David 2026-09-15 — batching is part of the
    // configuration) — the fixture's `scored_batch_size`, read off the MIGRATED schema, is the
    // value that SELECTS the measured path: absent or `1` keeps the spec-derived single-stream
    // regime, and a width greater than 1 asks for the batched cohort regime (admitted through the
    // one certify-match, which pairs a width with its ruled §5 series tag). Resolved HERE — before the calibration pre-read,
    // the config, and the spawn surfaces — so every regime-derived decision below (the b8 series
    // tag the calibration is fenced against, the v1.1 spawn gate, the closure selection) describes
    // the one regime this run measures.
    //
    // David ruling (2026-08-26) — the resolution now takes the candidate's MODE, because the
    // cohort upgrade is mode-aware: a SINGLE-STREAM-ONLY mode (`dflash`, which the engine's cohort
    // driver refuses by name) keeps its single-stream regime even under a fixture that pins a
    // width. Without that, gemma4's `scored_batch_size: 8` would have kept the track structurally
    // closed to dflash even after the mode fence admitted it — the refusal would just have moved
    // from benchd to the engine, one spawn and one chunk of gated box time later.
    let candidate_regime = measure_job::effective_candidate_regime(
        &args.candidate_spec.mode,
        args.candidate_regime,
        contract.scored_batch_size,
    )
    .map_err(MeasureJobFailure::die8)?;
    // Say it out loud when a PINNED cohort width was NOT applied. The regime itself is sealed in
    // `results.timed_mode` and the overlay's §5 series fence keeps the two regimes from ever being
    // pooled or compared — so this note changes no decision — but an operator reading a run of a
    // b8-pinned track must not have to infer from a series tag that this one measured a single
    // stream. Derived from the RETURNED regime, so it cannot disagree with what was resolved.
    if contract.scored_batch_size.is_some_and(|w| w > 1)
        && candidate_regime.scored_batch_point().is_none()
    {
        eprintln!(
            "benchd measure-job: --contract pins scored_batch_size {:?}, but the candidate mode \
             {:?} is SINGLE-STREAM ONLY, so this run measures the single-stream series and not the \
             batched cohort. Admitted modes for this track: {allowed_modes:?}.",
            contract.scored_batch_size, args.candidate_spec.mode,
        );
    }
    // Orchestrator ruling (2026-08-23) — the composite score's exponent pair is ALSO a
    // FIXTURE-PINNED IDENTITY, exactly like `scored_batch_size` above, and declared beside it on
    // the SAME migrated schema: consulted (and REQUIRED) ONLY on the batched cohort regime — a
    // single-stream run never reads the exponents at all, matching how it never reads
    // `scored_batch_size` beyond the regime selection above.
    // Certified HERE, alongside `candidate_regime`, so the config built below never carries an
    // uncertified value.
    let scored_exponents = match candidate_regime.scored_batch_point() {
        Some(_) => Some(
            measure_job::ScoredExponents::certify(
                contract
                    .prefill_gain_exponent
                    .zip(contract.decode_gain_exponent),
            )
            .map_err(MeasureJobFailure::die8)?,
        ),
        None => None,
    };
    // COHORT (D2) — on the batched path, the cohort-membership gate (die-8, pre-GPU): the cohort
    // is EXACTLY the fixture-pinned pool, in POOL ORDER, every slot pinned by sha256 AND bytes.
    // Produces the SEALED member list `per_cohort[].members` carries. `None` on the single-stream
    // path.
    let cohort_members = match candidate_regime.scored_batch_point() {
        Some(point) => Some(
            measure_job::validate_cohort_membership(
                &golden_fixtures,
                &contract.timed_prompt_pool,
                // The CERTIFIED width the regime carries — the same fixture data
                // `effective_candidate_regime` admitted, read back from its one certify point.
                point.batch_size(),
            )
            .map_err(MeasureJobFailure::die8)?,
        ),
        None => None,
    };
    // LANE 2a — the correctness-golden ATTESTATION gate (die-8, pre-GPU), a SEPARATE authority from
    // the anti-lottery timed-pool coverage above. The fixture pins the hidden correctness golden as
    // a SIBLING of `timed_prompt_pool` (engine PR #41), sourced here from the SAME contract bytes via
    // `hidden_correctness_golden_pin_from_contract` — the one place that field path is spelled — so
    // it never perturbs the anti-lottery cardinality (`timed_prompt_pool | length`). The run's
    // attestation is `--correctness-golden`: benchd HASHES the staged bytes (sha256 + bytes), never
    // trusting a self-declared digest, and refuses (fail-closed both directions) any run whose
    // attested identity does not CITE the fixture pin. The golden's NAME appears nowhere — the pin is
    // the only identity.
    let fixture_correctness_pin = hidden_correctness_golden_pin_from_contract(&contract_bytes)
        .map_err(|e| MeasureJobFailure::die8(format!("--contract correctness-golden pin: {e}")))?;
    let attested_correctness_pin = match args.correctness_golden.as_ref() {
        Some(path) => {
            let bytes = std::fs::read(path).map_err(|e| {
                MeasureJobFailure::die8(format!(
                    "--correctness-golden read failed ({}): {e}",
                    path.display()
                ))
            })?;
            Some(CorrectnessGoldenPin {
                sha256: sha256_hex(&bytes),
                bytes: bytes.len() as u64,
            })
        }
        None => None,
    };
    verify_correctness_golden_attestation(
        attested_correctness_pin.as_ref(),
        fixture_correctness_pin.as_ref(),
    )
    .map_err(|e| MeasureJobFailure::die8(e.to_string()))?;
    // #142 — the CAPTURED engine-wire crosscheck now runs AT MEASURE TIME, not only under
    // `cargo test`: benchd re-verifies its embedded captured engine-wire reference against the
    // mirror-integrity reference sha256 and re-parses it under its own CLOSED `WorkerResponse`, so a
    // drifted capture or a schema divergence dies pre-GPU (die-8) rather than being trusted on the
    // contract's self-declared pin alone. Independent of `validate_goldens_pinned`'s contract-pin.
    measure_job::crosscheck_captured_engine_wire(
        bench_runner::ENGINE_WIRE_V1_FIXTURE.as_bytes(),
        bench_runner::ENGINE_WIRE_V1_SHA256,
    )
    .map_err(MeasureJobFailure::die8)?;
    // R14 — `loaded_util` is env-driven (`GPU_LOADED_UTIL`, default 0.70, W:403); GATE_TEMP/COOL_TIMEOUT
    // stay fixed wrapper constants (R21). Resolve the util (fail-closed on an invalid value) and thread
    // it + its honest source into the thermal thresholds.
    let (loaded_util, loaded_util_source) =
        measure_job::resolve_loaded_util(std::env::var("GPU_LOADED_UTIL").ok().as_deref())?;
    let thermal = measure_job::thermal_thresholds(loaded_util, loaded_util_source);

    // finding 2 + WORKSPACE fix: `--candidate`/`--baseline` are WORKSPACE DIRECTORIES; each leg's
    // runtime-worker executable is resolved as `<ws>/.build/release/<bin>` (bin defaults to
    // `mlxfast-runtime-worker`, overridable via MLXFAST_MEASURE_WORKER_BIN), FAIL-CLOSED if absent.
    // MLXFAST_RUNTIME_WORKER_EXECUTABLE remains an override that must not CONFLICT with the
    // workspace-resolved path. The engine is spawned as `<engine> runtime-worker --weights <DIR>`
    // where `<DIR>` is the SEPARATE `--weights` argument, never the workspace.
    let worker_override = std::env::var("MLXFAST_RUNTIME_WORKER_EXECUTABLE").ok();
    let worker_bin = std::env::var("MLXFAST_MEASURE_WORKER_BIN")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| measure_job::DEFAULT_MEASURE_WORKER_BIN.to_string());
    let candidate_exec = measure_job::resolve_workspace_engine(
        &args.candidate.to_string_lossy(),
        &worker_bin,
        worker_override.as_deref(),
    )?;
    let baseline_exec = measure_job::resolve_workspace_engine(
        &args.baseline.to_string_lossy(),
        &worker_bin,
        worker_override.as_deref(),
    )?;
    // #42 box-leg — Metal loads `mlx.metallib` from BESIDE the resolved worker. Verify the sibling
    // exists next to EACH resolved leg engine HERE (pre-GPU, and on the `--preflight-only` path): a
    // missing metallib does not fail at spawn — it kills the run LATE, at the first MLXArray inside
    // the GPU window, after gated box time is spent. Resolution is unchanged; this only asserts the
    // adjacency the run will silently depend on.
    measure_job::verify_worker_metallib_sibling(&candidate_exec)
        .map_err(MeasureJobFailure::die8)?;
    measure_job::verify_worker_metallib_sibling(&baseline_exec).map_err(MeasureJobFailure::die8)?;

    // DECIDE-1 — the general trusted-source-scope freeze, homed HERE (benchd measure-job). The
    // BASELINE workspace IS the trusted ref (sub-decision 2), so its `benchmark.json` editable
    // surface is the contract to freeze, and the roster-of-EIGHT trusted paths are resolved against
    // that same trusted root (sub-decision 3). A manifest whose editablePaths / optionalEditablePaths
    // / exemptPaths overlap ANY roster-of-eight trusted path — directly, cased, or via an inode-
    // identical spelling — is REFUSED here (die-8, pre-GPU), before any gated box time is spent. The
    // eighth roster path is `benchmark.json` itself: a manifest may not declare its own file editable.
    //
    // The manifest is read from the trusted ref, never from a submission, so a candidate can not
    // steer this check. When the trusted ref carries NO `benchmark.json` (an engine-only tree
    // declares its editable surface elsewhere), there is no editable-surface declaration to freeze
    // and nothing to refuse — the freeze binds exactly when the manifest is present, which is the
    // ranked reality; a present-but-overlapping OR malformed manifest is a hard refusal.
    //
    // The absent-manifest SKIP is correct-by-construction (the manifest is `args.baseline`, an
    // operator-controlled arg; a candidate can not suppress it), but the skip is SILENT, so an audit
    // can not tell "checked and passed" from "did not bind". Emit a one-line stderr NOTICE on the
    // skip path so the two are distinguishable in a log.
    let trusted_manifest = args.baseline.join("benchmark.json");
    if trusted_manifest.is_file() {
        let manifest_bytes = std::fs::read(&trusted_manifest).map_err(|e| {
            MeasureJobFailure::die8(format!(
                "trusted-ref benchmark.json read failed ({}): {e}",
                trusted_manifest.display()
            ))
        })?;
        trusted_scope::verify_editable_surface_within_trusted_scope(
            &args.baseline,
            &manifest_bytes,
        )
        .map_err(MeasureJobFailure::die8)?;

        // WIRE-1 item 1a — the AUTHORITATIVE editable-surface BYTE BUDGET (native Rust, executes NO
        // engine-repo code). The caps + editablePaths are read from the TRUSTED --baseline manifest
        // (a candidate can not steer its own budget); the surface WALKED is the --candidate workspace
        // (the submission whose surface we bound). Ported from
        // EditableSurfaceByteBudget.swift@736781ea and pinned against it by tests/byte_budget_parity.rs.
        // An overshoot of maxTotalBytes / maxFileBytes / exemptPathMaxBytes is a die-8 refusal, pre-GPU.
        match byte_budget::verify_byte_budget_over(&manifest_bytes, &args.candidate) {
            byte_budget::BudgetVerification::Verified { .. } => {}
            // Skipped can only arise from a missing contract, impossible here (the bytes are in
            // hand); the variant exists for the Swift-parity test surface, so its arm is test-gated.
            #[cfg(test)]
            byte_budget::BudgetVerification::Skipped(_) => {}
            byte_budget::BudgetVerification::Exceeded(reason) => {
                return Err(MeasureJobFailure::die8(format!(
                    "editable-surface byte budget: {reason}"
                )));
            }
        }
        // WIRE-1 item 1a (growth) — the benchd-native maxGrowthBytes bound the launch-time Swift
        // enforcer resolves but can NOT consume (no review base at launch). benchd HAS the base:
        // growth = candidate − baseline editable code bytes. An overshoot is a die-8 refusal, pre-GPU.
        //
        // David ruling 2026-08-27 — WHICH base: with `--write-gate-base <SHA>` the base is the
        // candidate repo's OWN state at its fork point from harness main, so the bound stops moving
        // when the organizer moves main. Without the flag this is byte-identical legacy behavior
        // (the staged --baseline workspace), which the qwen38 track lane depends on.
        let growth = match &args.write_gate_base {
            Some(base) => {
                byte_budget::verify_growth_over_from_git(&manifest_bytes, &args.candidate, base)
            }
            None => {
                byte_budget::verify_growth_over(&manifest_bytes, &args.baseline, &args.candidate)
            }
        };
        match growth {
            byte_budget::BudgetVerification::Verified { .. } => {}
            #[cfg(test)]
            byte_budget::BudgetVerification::Skipped(_) => {}
            byte_budget::BudgetVerification::Exceeded(reason) => {
                return Err(MeasureJobFailure::die8(format!(
                    "editable-surface growth: {reason}"
                )));
            }
        }
        // WIRE-1 item 1b — the AUTHORITATIVE write-outside-editablePaths gate. Any file changed,
        // added or deleted OUTSIDE editablePaths is a die-8 refusal, pre-GPU (the same overlap
        // discipline as #147's trusted-scope: casefold + device:inode, not substring).
        //
        // David ruling 2026-08-27 — WHICH diff: with `--write-gate-base <SHA>` the gate judges the
        // SUBMISSION'S OWN committed diff (<SHA>..HEAD in the --candidate repo), so an organizer
        // commit to harness main — which sits BELOW the fork point — can no longer be charged to
        // every submission at once (the 2026-08-27 stale-staging refusals). Without the flag this is
        // byte-identical legacy behavior: the candidate tree versus the staged --baseline tree.
        match &args.write_gate_base {
            Some(base) => editable_divergence::verify_no_write_outside_editable_from_git(
                &manifest_bytes,
                &args.candidate,
                base,
            ),
            None => editable_divergence::verify_no_write_outside_editable(
                &manifest_bytes,
                &args.baseline,
                &args.candidate,
            ),
        }
        .map_err(MeasureJobFailure::die8)?;
    } else {
        // #150 — the absent-manifest SKIP is silent, so emit a one-line stderr NOTICE so an audit can
        // tell "checked and passed" from "did not bind" (the manifest is operator-controlled, so the
        // skip is correct-by-construction).
        eprintln!(
            "NOTICE trusted-scope freeze: no benchmark.json under the trusted ref {} — not binding \
             (editable surface declared elsewhere; the manifest is operator-controlled, so this \
             skip is correct-by-construction)",
            args.baseline.display()
        );
    }

    // R14 — `BASELINE_CALIBRATION` is a JSON FILE path (env), REPLACING the dead scalar
    // `MLXFAST_QWEN_MTP_SERIAL_CALIBRATION_SPT`. Parse FAIL-CLOSED (a malformed file is die-6, pre-
    // measure) and RESOLVE it for `--target-id` (or the top-level default). The die-6 serial-band
    // ENFORCEMENT against the pooled serial mean runs AFTER measuring. `BASELINE_BAND_ENFORCE`
    // (default 1) makes a MISSING calibration fail closed (die-6). `--calibration-bootstrap` skips it.
    // H6/H2 (cycle-3) — an EMPTY-STRING `BASELINE_BAND_ENFORCE=""` must map to ENFORCED (fail-closed),
    // same as unset; only an explicit `"0"` disables. (The old parse treated "" as disabled.)
    let band_enforce =
        measure_job::band_enforce_from_env(std::env::var("BASELINE_BAND_ENFORCE").ok().as_deref());
    // #105 cycle-5 — the calibration PRE-READ now happens further down, once `track_id` is resolved:
    // the series fence cross-checks the file's `timed_mode`/`track_id` against this run's, so the
    // read cannot precede the track resolution.

    // R14 — resolve the per-leg native-MTP head dirs (QMTP_HEAD_DIR = pinned serial head;
    // QMTP_CANDIDATE_HEAD_DIR = candidate BYO head, defaulting to the pinned head). Existence-check
    // both when present (die-8 prereq). The actual head-into-timed-verb spawn wiring is R15.
    // UNVERIFIED(measure-job): the on-box head-into-verb spawn use (R15).
    let head_dirs = measure_job::resolve_head_dirs(
        std::env::var("QMTP_HEAD_DIR").ok().as_deref(),
        std::env::var("QMTP_CANDIDATE_HEAD_DIR").ok().as_deref(),
    );
    // David ruling (2026-08-26) — the DFlash drafter's OWN per-leg pair, resolved by the SAME
    // function with the SAME defaulting rule (candidate BYO falls back to the pinned dir). Optional
    // here for every mode; REQUIRED for a `dflash` candidate, enforced below by
    // `enforce_dflash_head_present` at the point the MTP head's own unset-refusal lives.
    let dflash_head_dirs = measure_job::resolve_head_dirs(
        std::env::var("QMTP_DFLASH_HEAD_DIR").ok().as_deref(),
        std::env::var("QMTP_CANDIDATE_DFLASH_HEAD_DIR")
            .ok()
            .as_deref(),
    );
    // The DFlash pair is only VALIDATED for a `dflash` candidate, the same scope its own die-8
    // requirement (`enforce_dflash_head_present`) and its spawn-flag emission
    // (`paired_leg_spawn_args`) use. The env is resolved mode-independently above, so an operator
    // shell still carrying a `QMTP_DFLASH_HEAD_DIR` export from an earlier dflash run — pointing at
    // a workspace that has since been swept — would otherwise die-8 an unrelated mtp run for a
    // drafter that run never loads. The MTP-family pair stays unscoped: it is the standing
    // behaviour on every track and this lane does not widen or narrow it.
    let validated_head_dirs = [
        Some((
            head_dirs.as_ref(),
            ("QMTP_HEAD_DIR", "QMTP_CANDIDATE_HEAD_DIR"),
        )),
        (args.candidate_spec.mode == bench_protocol::SPEC_MODE_DFLASH).then_some((
            dflash_head_dirs.as_ref(),
            ("QMTP_DFLASH_HEAD_DIR", "QMTP_CANDIDATE_DFLASH_HEAD_DIR"),
        )),
    ];
    for (hd, labels) in validated_head_dirs.into_iter().flatten() {
        let Some(hd) = hd else { continue };
        for (label, dir) in [(labels.0, &hd.head_dir), (labels.1, &hd.candidate_head_dir)] {
            if !Path::new(dir).is_dir() {
                return Err(MeasureJobFailure::die8(format!(
                    "{label} does not exist or is not a directory: {dir} (die 8)"
                )));
            }
        }
    }

    // WEIGHTS-IDENTITY provenance: digest the `--weights` DIR (the transformed weights loaded by
    // both legs) — this is the weights identity carried as `weights_hash` in results.json and
    // `weights_sha256` in the integrity anchor. Distinct from the candidate WORKSPACE digest below.
    let weights_digest = dir_digest_weights(&args.weights)
        .map_err(|e| format!("--weights digest failed ({}): {e}", args.weights.display()))?;
    // CANDIDATE-IDENTITY provenance: digest the candidate WORKSPACE (the built engine source) —
    // recorded as `candidate_workspace_sha256` in the integrity anchor. A workspace and a weights
    // dir are different things; each carries its own provenance field.
    let candidate_ws_digest = dir_digest(&args.candidate).map_err(|e| {
        format!(
            "candidate workspace digest failed ({}): {e}",
            args.candidate.display()
        )
    })?;
    // Integrity-anchor minor: the seal must cover the BASELINE workspace too (not only the
    // candidate), so both legs' built-engine sources are pinned. Digest it here.
    let baseline_ws_digest = dir_digest(&args.baseline).map_err(|e| {
        format!(
            "baseline workspace digest failed ({}): {e}",
            args.baseline.display()
        )
    })?;
    // Integrity-anchor minor: pin the golden IDENTITY (the sha of the actual --golden bytes, ==
    // `GoldenFixture::sha256`) and the CONTRACT digest (sha of the --contract fixture bytes) so the
    // seal covers the exact prompt oracle + track fixture this run measured against.
    let golden_sha256 = golden.sha256().to_string();
    let contract_sha256 = sha256_hex(&contract_bytes);

    // AUTHOR-AT-SEAL (DECIDE-3) — the sealed `metrics.commit` is AUTHORED from the sha the in-repo
    // dispatch script RECORDED (candidate.sha, from the CI/yukon dispatch context), never from
    // participant git state (unusable under the ranked sandbox) and never trusting a
    // competitor-proposed commit. `MLXFAST_COMMIT_SHA` (the engine's `commitIdentifier` emission)
    // stays DEFENCE-IN-DEPTH: present-and-disagreeing is a die-8 refuse. On a SCORING run
    // (`!args.local_dev`, the same signal `cfg.local_pair_budget` keys on) an ABSENT record fails
    // closed — never git; only `--local-dev` keeps the un-bound resolution.
    let dispatch_record = read_dispatch_sha_record()?;
    let commit_env = std::env::var("MLXFAST_COMMIT_SHA").ok();
    let commit = official::author_sealed_commit(
        dispatch_record.as_deref(),
        commit_env.as_deref(),
        !args.local_dev,
    )
    .map_err(MeasureJobFailure::die8)?;

    // R12 — the SEALED CONSTANT `track_id` (the workflow-declared track id) is resolved from env
    // `MLXFAST_QWEN_MTP_TRACK_ID` or the `--contract` fixture's own `track_id` (constant≡contract≡env:
    // a present env≠contract is a HARD ERROR; neither present fails closed). It is NOT `--tag`: the
    // per-run `--tag` is sealed SEPARATELY as `tag`. `track_name` is optional (env/contract/omit).
    // `track_id` was resolved above, before the goldens loaded — the per-track model identity is
    // a loader INPUT, so it could not wait until here. This is the same one value, sealed.
    let track_name = measure_job::resolve_track_name(
        std::env::var("MLXFAST_QWEN_MTP_TRACK_NAME").ok().as_deref(),
        contract.track_name.as_deref(),
    );

    // H6/H2 (cycle-3) — under `--calibration-bootstrap` this run AUTHORS the band; it must NOT
    // pre-read or require the existing file. The wrapper's `serial_band_check` returns immediately
    // in bootstrap mode (W:1423-1426) and `write_calibration_bootstrap` reads/merges the file
    // itself afterwards — so a MISSING calibration file in bootstrap mode is fine, not a die-6.
    // #105 cycle-5 (HIGH) — the read carries this run's SERIES and resolved `track_id` so
    // `enforce_calibration_series_fence` can die-6 a cross-series / cross-track calibration here,
    // BEFORE any measuring and therefore before any banding.
    //
    // W3 (fence reconciliation) — the series passed here is THIS RUN'S OWN series, not the
    // hardcoded teacher-forced tag. The band divides the run's pooled SERIAL mean, and under the
    // Fable ruling the serial control runs the run's series ([`measure_job::run_timed_mode`]), so a
    // free-run run bands only against free-run calibration and a TF run only against TF
    // calibration. This is the SAME decision function (`timed_modes_comparable`) the overlay's §5
    // fence uses on results/score — one series story across calibration, overlay and seal.
    let run_timed_mode = measure_job::run_timed_mode(candidate_regime);
    let calibration = resolve_calibration_env(
        args.calibration_bootstrap,
        std::env::var("BASELINE_CALIBRATION").ok().as_deref(),
        args.target_id.as_deref(),
        run_timed_mode,
        &track_id,
    )?;

    let cfg = measure_job::MeasureJobConfig {
        scored_regime: bench_core::contract::scored_regime(&contract, &track_id)
            .map_err(MeasureJobFailure::die8)?
            .value,
        track_id,
        track_name,
        tag: args.tag.clone(),
        tokens: args.tokens,
        mtp_depth: args.mtp_depth,
        candidate_spec: args.candidate_spec.clone(),
        baseline_spec: args.baseline_spec.clone(),
        candidate_spec_source: args.candidate_spec_source.clone(),
        baseline_spec_source: args.baseline_spec_source.clone(),
        min_pairs: args.min_pairs,
        target_pairs: args.target_pairs,
        prompt_pool: contract.timed_prompt_pool.clone(),
        thermal,
        candidate_executable: candidate_exec.clone(),
        baseline_executable: baseline_exec.clone(),
        calibration: calibration.clone(),
        band_enforce,
        // R16 (medium cycle-3) — the sealed top-level `timestamp` (date -u), stamped now.
        run_timestamp: iterate::iso8601_now(),
        calibration_bootstrap: args.calibration_bootstrap,
        target_id: args.target_id.clone(),
        prompt_sha256: args.prompt_sha256.clone(),
        exactness_probe: args.exactness_probe,
        // H6/H3 — OFFICIAL by default (immediate die-5 on a failed pair); `--local-dev` enables the
        // budget loop.
        local_pair_budget: args.local_dev,
        // W3 — the candidate leg's timed regime: the spec-derived regime, upgraded to the batched
        // cohort regime when the fixture pins `scored_batch_size` (COHORT, D9).
        candidate_regime,
        // Orchestrator ruling (2026-08-23) — the CERTIFIED composite exponent pair, resolved
        // above alongside `candidate_regime`; `None` off the batched regime.
        scored_exponents,
    };
    // W3 — refuse an incoherent regime/spec/window combination BEFORE any GPU work (the pair loop
    // re-checks it, but a pre-GPU refusal costs no gated box time).
    measure_job::validate_candidate_regime_coherent(&cfg).map_err(MeasureJobFailure::die8)?;
    // #112 (M1) — and refuse a golden that cannot ORACLE that window, also pre-GPU. The
    // rows-vs-window rule used to live only inside the pair loop's per-prompt `timing_params`,
    // which runs AFTER the `--preflight-only` return below: a tape too short for the window
    // therefore passed preflight and died on the first prompt of the real run instead. Same
    // function, every loaded golden, now on both paths. `cfg.tokens` is the RULED window — the
    // check above has already pinned it to N = FREE_RUN_DECODE_TOKENS in the free-run series;
    // teacher-forced runs use `--tokens` as given.
    // 2b box-leg — a golden routed to the ranked GATES phase as a legacy GoldenDocument must carry
    // the `.benchmark` oracle (a benchmark/official window is TIMED against it). Refuse EARLY and
    // CLEARLY here — naming the engine's weightless attach-benchmark-oracle remedy — BEFORE the
    // generic per-prompt window refusal below, which would otherwise frame a missing oracle as a
    // token-count shortfall. Same die-8 pre-GPU point; actionable message.
    measure_job::validate_gates_goldens_carry_oracle(&golden_fixtures)
        .map_err(MeasureJobFailure::die8)?;
    measure_job::validate_prompt_windows(&golden_fixtures, cfg.tokens)
        .map_err(MeasureJobFailure::die8)?;

    // Each leg/phase spawns a FRESH worker from its workspace engine (fail-closed; SANDBOXED
    // wherever Seatbelt exists — see the plan resolution below),
    // loading the SHARED `--weights` DIR — the proven official spawn `<engine> runtime-worker
    // --weights <DIR>`. The engine (workspace-resolved) and the weights (the `--weights` DIR) are
    // DIFFERENT paths: the old code passed the workspace as both, so no real run could load weights.
    // UNVERIFIED(measure-job): the sandboxed-workspace spawn recipe (first exercised on-box).
    // UNVERIFIED(B-4): both legs share `--weights` for the backbone/identity case; per-side MTP
    // head weights (QMTP_HEAD_DIR / QMTP_CANDIDATE_HEAD_DIR) are a later refinement (R15).
    // R13 — `--preflight-only`: every pre-GPU prereq/quiesce check above (golden load + dup-digest,
    // prompt-hash, contract parse, workspace-engine resolution, weights/workspace digests, track-id
    // resolution, regime coherence, and — #112 (M1) — the window each golden must be able to
    // oracle) has passed. Exit 0 WITHOUT measuring — no pair loop, no results.json.
    if args.preflight_only {
        eprintln!(
            "benchd measure-job: --preflight-only OK ({} golden(s) [{}], candidate={}, baseline={}) — \
             pre-GPU checks passed, not measuring",
            args.goldens.len(),
            golden_kind_summary(&golden_fixtures),
            candidate_exec,
            baseline_exec,
        );
        return Ok(MeasureJobVerdict::PreflightOk);
    }
    // ARM GATE (David ruling 2026-08-26) — the track fixture's `official_scoring_enabled` is a REAL
    // gate: a SCORING run (`!args.local_dev`, the same signal author-at-seal and the pair budget key
    // on) over a fixture that does not declare `true` REFUSES here, die-8, naming the flag. Absent
    // and `false` both refuse — an absent arm state is not an armed one.
    //
    // FIRST of the post-preflight pre-GPU checks, ahead of F-6, on purpose: whether the track is
    // ARMED AT ALL dominates every other precondition, and "official scoring is not enabled for
    // this track" is a far more actionable verdict than the calibration/head-dir refusals that
    // would otherwise be reported for an unarmed track that also happens to be missing a band.
    //
    // AFTER the `--preflight-only` return, also on purpose, and for the reason F-6 states just
    // below for itself: preflight opens no GPU window and SEALS NOTHING, and it is the tool the
    // track is brought up WITH — during exactly the period when the flag is legitimately false.
    // The ruling is about refusing to SEAL an official score, not about refusing to look at a
    // workspace. The gate still costs no gated box time: it fires before the first spawn.
    //
    // This one call site covers the whole ranked chain. `benchd overlay-timing` is LOCAL-only by
    // design (the organizer's trusted shell authors the published score.json, OPEN-2) and its
    // `--contract` is OPTIONAL, but it REQUIRES `--results` — a measure-job artifact — so no ranked
    // score.json can exist without a measure-job that passed this gate.
    bench_core::contract::enforce_official_scoring_enabled(
        !args.local_dev,
        contract.official_scoring_enabled,
        &cfg.track_id,
    )
    .map_err(MeasureJobFailure::die8)?;
    // F-6 — fail fast, PRE-GPU, on a missing baseline calibration under enforcement. The
    // post-measure band check (below, after `run_measure_job`) already die-6s when
    // `BASELINE_BAND_ENFORCE=1` (the default) and no calibration was resolved — but only AFTER the
    // GPU window has opened and both legs have measured. A missing calibration is knowable now, so
    // discovering it here costs no gated box time. Enforcement SEMANTICS are unchanged: the same
    // condition, the same die-6, only earlier. `--preflight-only` returned above and is deliberately
    // NOT gated on this (it opens no GPU window); a real run reaches here. A set-but-unreadable or
    // malformed `BASELINE_CALIBRATION` is ALREADY a pre-measure die-6 in `resolve_calibration_env`;
    // this closes the remaining gap — the env UNSET.
    if band_enforce && !args.calibration_bootstrap && calibration.is_none() {
        return Err(MeasureJobFailure::die6(
            "no BASELINE_CALIBRATION but BASELINE_BAND_ENFORCE=1 (default) — cannot validate the \
             serial baseline; failing closed PRE-GPU (die 6) before opening the timed window. Set \
             BASELINE_BAND_ENFORCE=0 or pass --calibration-bootstrap to author one."
                .to_string(),
        ));
    }
    // R15 — a real measure run needs the PINNED native-MTP head (`QMTP_HEAD_DIR`): the serial leg
    // loads it, and it is the default for the candidate leg's BYO head. Fail closed (die-8) if the
    // pinned head is unset once we are actually measuring (preflight-only already returned above).
    let head_dirs = head_dirs.ok_or_else(|| {
        MeasureJobFailure::die8(
            "QMTP_HEAD_DIR is unset: the pinned native-MTP head is required for a measure run (the \
             serial leg loads it; the candidate leg defaults to it) — die 8"
                .to_string(),
        )
    })?;
    // David ruling (2026-08-26) — the SAME refusal for the DFlash drafter, but only when the
    // candidate actually declares mode `dflash`. Placed HERE, beside the MTP head's refusal and
    // after the `--preflight-only` return, for the reason that return exists: preflight opens no
    // GPU window and seals nothing, and it is the tool a track is brought up with — during exactly
    // the period when the drafter may not be staged yet.
    measure_job::enforce_dflash_head_present(&args.candidate_spec.mode, dflash_head_dirs.as_ref())
        .map_err(MeasureJobFailure::die8)?;

    // The paired measure-job already resolves its two executables itself, so neither leg takes
    // the `MLXFAST_RUNTIME_WORKER_EXECUTABLE` override: honouring it would point both legs at one
    // binary. (The override was already inert here in practice — both calls passed their own
    // executable — so this states the existing intent rather than changing it.)
    // The paired measure-job runs under ONE resident for the window, so both legs' profiles name
    // the process env's socket — the legacy single-resident topology, unchanged.
    let window_socket = process_resident_socket();
    // a8 ruling (b) — Seatbelt is a macOS facility. On a non-macOS host (the linux-aarch64 CUDA
    // box) `/usr/bin/sandbox-exec` does not exist, so the legs resolve NO plan and spawn
    // UNSANDBOXED through `spawn_official_worker`, exactly as the official and iterate paths
    // already do. macOS keeps the fail-closed resolution byte-unchanged: the plan is resolved
    // here, before any leg exists, and a missing prerequisite still refuses the run up front.
    let sandboxed = cfg!(target_os = "macos");
    let serial_plan = sandboxed
        .then(|| {
            resolve_official_sandbox_from_env(
                &baseline_exec,
                golden_path,
                false,
                window_socket.as_deref(),
            )
        })
        .transpose()?;
    let candidate_plan = sandboxed
        .then(|| {
            resolve_official_sandbox_from_env(
                &candidate_exec,
                golden_path,
                false,
                window_socket.as_deref(),
            )
        })
        .transpose()?;
    let serial_weights = args.weights.to_string_lossy().to_string();
    let candidate_weights = args.weights.to_string_lossy().to_string();
    // R15 — per-side heads passed to the ONE spawned worker per leg: the serial control loads the
    // PINNED head, the candidate the DECLARED BYO head. The head is loaded on BOTH legs (residency
    // charges the denominator), so `--mtp-head` is passed on each.
    //
    // VERIFIED-on-box(window-2, #109) — the spawned verb is the engine's GENERIC `runtime-worker`,
    // and its option surface is exactly `{--weights, --mtp-head, --speculative-protocol}`
    // (`Sources/MLXFastRuntimeWorkerCLI/main.swift`, `requireOnly(values:)`). Window 2 isolated this
    // FOUR WAYS on a live box: `--weights W --mtp-head H` → exit 0 with a real hello;
    // `… --speculative-protocol v1.1` → exit 0 with `spec_modes`/`capabilities`/`head_provenance`;
    // `… --mtp-depth 0 --mtp-report P` (benchd's then-argv) → exit 1 *"unexpected participant worker
    // option --mtp-depth"*; `… --mtp-report P` alone → exit 1 on `--mtp-report`. The verb exits on the
    // FIRST unknown option, BEFORE the hello, which is why every pair of every leg died pre-GPU as
    // *"engine closed the stream before returning a response"*. Both retired flags belonged elsewhere:
    // `--mtp-depth` to a DIFFERENT binary's verb (`mlxfast-swift mtp-timed`) and `--mtp-report` to no
    // verb at all. The surviving argv is fenced by `measure_job::RUNTIME_WORKER_ACCEPTED_FLAGS`.
    //
    // Both retired channels already exist on the wire: DEPTH as the `decode_begin` /
    // `free_decode_begin` `spec` (echoed back as `effective_spec`, spec-never-ignored), and the report
    // facts as the hello's `head_provenance` plus benchd's OWN free-run histogram math. H1 is
    // untouched: benchd's parent-side wall clock (`run_decode_phase_fresh` /
    // `run_free_run_decode_phase_fresh`) was already the ONLY scored value, and is now the only
    // parent-clock number in existence.
    // #105 (Engine-can't-speculate-on-TF) — the SERIAL CONTROL leg always times a SERIAL decode
    // window: it is the depth-0 control (benchd feeds each token, or free-runs with no drafter), so
    // its wire spec is the serial spec.
    //
    // W3 — the CANDIDATE leg's wire spec follows its REGIME:
    //   * teacher-forced (a serial candidate, or the legacy Model-2 shape) → the DOWNGRADED serial
    //     spec, because sealing an mtp regime that cannot have run under teacher forcing is refused
    //     downstream (`tf_regime_is_serial`);
    //   * v1.1 free-run → the DECLARED candidate spec verbatim. The free-run window is exactly where
    //     that spec CAN run, so downgrading it here would measure a serial engine and seal it as the
    //     candidate's number. The runner enforces SPEC-NEVER-IGNORED on the echo, and the seal
    //     refuses a serial echo on a free-run candidate leg (`free_run_regime_is_speculative`).
    // The DECLARED candidate_spec/baseline_spec stay as results.json provenance (cfg carries them
    // into build_results) either way.
    //
    // Coordinator ruling (#109, leg B) — these two values are what a leg WOULD request. Whether it
    // requests anything at all is `measure_job::requested_wire_spec`, applied at the timed window
    // below: on a TF pair the answer is None for both legs (gate-off spawn ⇒ no spec, no echo, and
    // the serial regime sealed from the spawn surface), so the downgraded serial spec above is
    // computed and then deliberately not sent. It stays here because the TF branch of
    // `candidate_wire_spec` is what makes the downgrade explicit at the point a reader looks for it.
    let serial_wire_spec = measure_job::timed_decode_wire_spec();
    let candidate_wire_spec = if candidate_regime.is_free_run() {
        cfg.candidate_spec.clone()
    } else {
        measure_job::timed_decode_wire_spec()
    };

    // #105 cycle-5 finding 4, closed the subtractive way (#109 window-2 finding 3) — the spawn argv
    // no longer carries a depth AT ALL, so the two depth channels it used to have to reconcile are
    // one: the wire `spec` above. The argv's `--mtp-depth` was never a channel the spawned verb
    // could even read (it rejects the flag), so tying it to the wire spec only made a
    // never-honoured value consistent; removing it makes the wire spec the single source by
    // construction, and the runner's spec-never-ignored echo check its only guard.
    //
    // Window-prep gap (engine-train review): the engine gates ALL v1.1 wire fields behind
    // `--speculative-protocol v1.1` at spawn. Free-run legs (BOTH of them — the depth-0 serial
    // control speaks the same v1.1 session) must carry the flag; teacher-forced legs must not
    // (their gate-off spawn is the standing v1-compat proof).
    //
    // David ruling (2026-08-26) — the DFlash drafter is passed the SAME way and with the SAME
    // per-leg split: the serial control gets the PINNED drafter, the candidate its own. This is the
    // whole point of the `--dflash-head` channel — before it, the engine resolved a bare relative
    // `./dflash-head` against the WORKER's CWD, both workers inherit benchd's CWD (the spawn sets
    // no `current_dir`), and so both legs loaded ONE directory no matter which workspace they were
    // measuring.
    //
    // BOTH legs' argv now come from ONE call. The four hand-written field accesses this replaces —
    // pinned/BYO x mtp/dflash — were the only remaining place a leg could be handed the OTHER leg's
    // head, and they lived in `execute_measure_job`, whose spawn wiring is explicitly
    // `UNVERIFIED(measure-job)` and therefore covered by no test at all. A mutation that swapped
    // the candidate leg's drafter for the pinned one passed the whole suite; against
    // `paired_leg_spawn_args` it does not.
    //
    // The candidate MODE is passed in because the DFlash channel is scoped to a `dflash` candidate:
    // the env behind `dflash_head_dirs` is resolved mode-independently, so a stale
    // `QMTP_DFLASH_HEAD_DIR` export would otherwise put `--dflash-head` on both legs of an mtp pair
    // and kill it pre-hello against any engine build that predates the flag.
    let (serial_base_args, candidate_base_args) = measure_job::paired_leg_spawn_args(
        &head_dirs,
        dflash_head_dirs.as_ref(),
        candidate_regime,
        &args.candidate_spec.mode,
    );
    // #109 window-2 finding 3 — fence BOTH legs' argv against the verb's accepted option surface
    // before any worker is spawned, so a flag the engine would reject dies here, naming itself,
    // rather than as one opaque "engine closed the stream" infra reject per leg per pair.
    for base_args in [&serial_base_args, &candidate_base_args] {
        measure_job::validate_spawn_argv(base_args).map_err(MeasureJobFailure::die8)?;
    }

    // H3 (cycle-3) — arm the RunTimeout budget for the timed decode round-trips (§2.2/§4):
    // `N × band-ceiling × margin`. The band-ceiling (upper acceptance/latency band bound, s/tok) is
    // `calibration.serial_mean × calibration.band_high` when a BASELINE_CALIBRATION is present, else
    // the deliberately-generous fallback constant. A hung/looping engine then aborts as `RunTimeout`
    // (session discarded) instead of wedging benchd inside the timed window. Liveness bound only —
    // never a score input.
    //
    // #108 (M2) — the budget is FAIL-CLOSED, not optional. A degenerate `N × ceiling × margin` used
    // to yield `None`, arming NO deadline; since the ceiling is calibration-derived, a
    // `BASELINE_CALIBRATION` file could disarm the §2.2 bound entirely. Now the arithmetic returns
    // an `Err`, and the leg fails under `RejectClass::RunTimeoutBudgetInvalid` rather than running
    // the timed window unbounded. (The band bounds are ALSO validated at parse, so a well-formed
    // calibration cannot reach this at all; this is the second fence, on the arithmetic itself.)
    let run_timeout_result = {
        let band_ceiling_spt = match calibration.as_ref() {
            Some(cal) if cal.serial_mean.is_finite() && cal.serial_mean > 0.0 => {
                cal.serial_mean * cal.band_high
            }
            _ => bench_core::constants::RUN_TIMEOUT_DEFAULT_BAND_CEILING_SECONDS_PER_TOKEN,
        };
        bench_core::score::run_timeout_budget(
            args.tokens,
            band_ceiling_spt,
            bench_core::constants::RUN_TIMEOUT_MARGIN,
        )
    };

    // R15 — each leg is ONE `runtime-worker` invocation: ONE fresh sandboxed worker, ONE cool gate
    // (finding R21: the FIXED wrapper constant 40 °C, W:422-429 — recorded in `provenance.thermal`),
    // then ONE timed decode window (the seed prefill INSIDE it, `prefill_component: "none"`). H1
    // (cycle-3): benchd's OWN parent wall clock is the scored `benchd_seconds_per_token`.
    //
    // #109 window-2 finding 3 — every ECHO/AUDIT fact now comes off the WIRE, because under the
    // generic verb there is no report file to read (and asking for one killed the spawn):
    //   * `effective_spec` — the `decode_begin` / `free_decode_begin` echo the runner captured and
    //     validated never-ignored, threaded through as `wire_effective_spec`;
    //   * `head_provenance.sha256` — the engine's `hello` echo, captured below and threaded through
    //     as `wire_head_provenance`;
    //   * draft statistics — benchd's OWN histogram math over the free-run §3 audit (already the
    //     case per W3; the TF-only report requirement is retired, since teacher forcing feeds every
    //     token and no round can draft).
    // `validate_leg_report` still FAILS CLOSED on each missing echo — nothing is ever fabricated.
    //
    // W3 — the leg's REGIME selects which runner entry point drives it:
    //   * `LegRegime::TeacherForcedV1` → `run_decode_phase_fresh` (v1: benchd feeds each golden
    //     token, N forced single-token forwards);
    //   * `LegRegime::FreeRunV1_1` → `run_free_run_decode_phase_fresh` (v1.1: the engine drives its
    //     own recurrence; benchd clocks the batched `free_decode_begin` + `free_decode_run(N)` round
    //     trip exactly as PROTOCOL-v1.1 §2.2 specifies, exact-matches every committed token, and
    //     enforces the §2.6 triple at the phase close).
    // Both paths take benchd's OWN parent clock as the only scored number (H1) and arm the same
    // §2.2 RunTimeout budget; the free-run path additionally REFUSES an engine that does not
    // advertise `free_run_decode` before the cool gate and before the clock (§2.1).
    let measure_leg = |plan: Option<&OfficialSandboxPlan>,
                       engine: &str,
                       weights: &str,
                       base_args: &[String],
                       leg_spec: &bench_protocol::SpecConfig,
                       regime: measure_job::LegRegime,
                       params: &bench_runner::TimingParams|
     -> bench_runner::Result<measure_job::LegInvocation> {
        // #109 window-2 finding 3 — the spawn argv is EXACTLY the accepted surface: the transport
        // prepends `runtime-worker --weights W`, and `base_args` carries `--mtp-head H` (+ the v1.1
        // spawn gate on a free-run leg). Nothing is appended here — the per-attempt `--mtp-report`
        // path that used to be is retired with the flag.
        let extra_args = base_args.to_vec();

        let mut recorded = coolgate::GateState::SkippedNoReader;
        let mut recorded_quiesce = quiescegate::QuiesceRecord::default();
        // #109 window-2 finding 3 — capture the hello's `head_provenance` (the engine's echo of the
        // head bytes it loaded) from the LAST spawn this leg made: the retired report file was the
        // only other channel that ever carried the candidate's head identity. Same capture pattern
        // as the cool-gate state above.
        let wire_head_provenance = std::cell::RefCell::new(None);
        let mut spawn = || -> bench_runner::Result<Session<ChildStdioTransport>> {
            let transport = spawn_official_worker(plan, engine, weights, &extra_args, &[])?;
            let (session, hello) = Session::connect(transport)?;
            *wire_head_provenance.borrow_mut() = hello.head_provenance.clone();
            Ok(session)
        };
        let mut gate = |phase: &str| -> bench_runner::Result<()> {
            // BOTH gates guard the timed window, quiescence first: the box must be idle before the
            // GPU temperature is even read (David 2026-09-17).
            let (quiesced, cooled) =
                quiescegate::timed_phase_gates_report(phase, bench_core::constants::Platform::Mlx)?;
            recorded_quiesce = quiesced;
            recorded = cooled.state;
            Ok(())
        };
        // H1 (cycle-3) — benchd's OWN parent-side wall clock, measured here, is the SCORED spt.
        // H3 (cycle-3) — arm the RunTimeout budget over this leg's timed decode window.
        // spec (docs/spec-config-design.md) — carry the leg's declared spec on the timed decode
        // window. The runner enforces SPEC-NEVER-IGNORED: it discards the session fail-closed
        // (RunnerError::SpecEchoDivergence → a retryable reject) if the engine's echoed effective_spec
        // is absent or diverges from what was requested, so a leg can never silently run a different
        // (or default) spec than the one declared. The echoed spec is surfaced on `timing.effective_spec`.
        // #108 (M2) — a leg NEVER opens its timed window without a §2.2 deadline: an unarmable
        // budget fails THIS leg (own reject class) instead of disarming the only wall-clock bound.
        let run_timeout =
            run_timeout_result
                .as_ref()
                .map_err(|detail| RunnerError::RunTimeoutBudgetInvalid {
                    detail: detail.clone(),
                })?;
        // Coordinator ruling (#109, leg B) — the spec is requested ONLY on a free-run leg
        // (`measure_job::requested_wire_spec`). A TF leg is spawned gate-off, and a gate-off worker
        // speaks strict v1: it rejects any wire `spec` at the session's spec guard and runs its
        // teacher-forced kinds serially regardless. Asking one for a spec would discard every TF
        // session for an echo the worker is gated out of producing; the gate-off spawn is itself the
        // proof of serial semantics, so nothing is asked for and nothing is expected back.
        let params = params
            .clone()
            .with_run_timeout(Some(*run_timeout))
            .with_spec(measure_job::requested_wire_spec(leg_spec, regime));
        // W3 — one timed window per leg, driven by the leg's regime. `free_run_audit` is `Some` only
        // on a v1.1 leg, and only after the runner's §2.6 triple passed at the phase-close barrier.
        // `_peak_ram_gb`: the worker's phase-close `phase_diagnostics` peak RAM. It reached the seal
        // only through the retired report struct's audit-only `peak_ram_gb`, which no consumer ever
        // read (no pair record, per-prompt or aggregate field is derived from it) — retired with the
        // struct. The wire still carries it for any future seal that wants it.
        // B2 — `phase_window` is benchd's own SPLIT of the free-run leg's timed window at the
        // `free_decode_begin` / `free_decode_run` boundary. A teacher-forced leg drives neither verb,
        // so it has no prefill window and reports `None` rather than a fabricated one.
        let (seconds_per_token, _peak_ram_gb, wire_effective_spec, free_run_audit, phase_window) =
            match regime {
                measure_job::LegRegime::TeacherForcedV1 => {
                    let t = bench_runner::run_decode_phase_fresh(&mut spawn, &mut gate, &params)?;
                    (
                        t.seconds_per_token,
                        t.peak_ram_gb,
                        t.effective_spec,
                        None,
                        None,
                    )
                }
                measure_job::LegRegime::FreeRunV1_1 => {
                    let t = bench_runner::run_free_run_decode_phase_fresh(
                        &mut spawn, &mut gate, &params,
                    )?;
                    (
                        t.seconds_per_token,
                        t.peak_ram_gb,
                        t.effective_spec,
                        Some(t.audit),
                        Some(t.phase_window),
                    )
                }
                // COHORT — a batched leg times ONE window over the whole cohort and is driven by
                // the COHORT measure closure (`measure_cohort_leg`, CohortTimingParams) on the
                // batched branch of this function; this single-stream closure can never
                // legitimately receive the batched regime, so reaching it is a wiring defect,
                // refused fail-closed rather than silently timed as a single stream (which would
                // swap the measured quantity).
                measure_job::LegRegime::BatchedFreeRunV1_2(_) => {
                    return Err(RunnerError::Protocol(
                        "batched cohort legs are driven by the cohort measure closure \
                         (CohortTimingParams), never the single-stream one — wiring defect"
                            .to_string(),
                    ));
                }
            };
        // R16 — the on-box per-block sampled telemetry stream (GPU temp / steady freq) is not wired
        // into this path yet; with no sample available the top-level `telemetry` seal is OMITTED
        // honestly, never fabricated from the gate state.
        // UNVERIFIED(measure-job): the on-box telemetry-sample stream is an engine/gate-protocol
        // addition; until it exists, benchd observes no sample and omits `telemetry`.
        Ok(measure_job::LegInvocation {
            // H1 (cycle-3) — the ONLY scored number: benchd's own parent clock.
            benchd_seconds_per_token: seconds_per_token,
            gate_state: recorded,
            quiesce: recorded_quiesce,
            telemetry: None,
            // The WIRE engine-echoed effective_spec benchd's runner captured + validated (equal to the
            // request; a divergence would have already discarded the session above). Sealed per leg.
            wire_effective_spec,
            // #109 window-2 finding 3 — the WIRE head echo from this leg's hello.
            wire_head_provenance: wire_head_provenance.into_inner(),
            // W3 — the regime this leg actually ran, and its §3 AUDIT when it free-ran.
            regime,
            free_run_audit,
            // B2 — the clock split benchd measured on this leg (free-run legs only).
            phase_window,
            // COHORT — never produced by the single-stream closure (the batched regime is refused
            // above); the cohort closure on the batched branch fills it.
            cohort_audit: None,
            // COMPOSITE (Gemma cohort scoring) — the phase-split window is a batched-regime-only
            // channel; this closure never drives the batched regime (refused above), so it has
            // none to report.
            cohort_phase_windows: None,
            // Per-stream timing (gap G2) — a batched-only wire channel; none exists here.
            per_stream_timing: None,
            // (b) admission — the committed-token journal is a batched-regime-only channel; the
            // single-stream regime enforces token correctness inline in the runner, so there is no
            // journal to surface for a trusted-oracle gate here.
            cohort_committed_tokens_by_stream: None,
        })
    };
    let measure_serial = |params: &bench_runner::TimingParams| {
        measure_leg(
            serial_plan.as_ref(),
            &baseline_exec,
            &serial_weights,
            &serial_base_args,
            &serial_wire_spec,
            // Fable ruling (same-series serial control) — the control runs THE SAME REGIME as the
            // candidate, at depth 0 (its wire spec above is the serial spec either way). Both legs
            // therefore share the same verb, the same N, the same RunTimeout arithmetic and the same
            // parent clock, so the ratio divides two numbers of one measured quantity.
            measure_job::serial_control_regime_for(candidate_regime),
            params,
        )
    };
    let measure_candidate = |params: &bench_runner::TimingParams| {
        measure_leg(
            candidate_plan.as_ref(),
            &candidate_exec,
            &candidate_weights,
            &candidate_base_args,
            &candidate_wire_spec,
            candidate_regime,
            params,
        )
    };

    // COHORT (batch-8 brief §4.5) — one closure per leg for the BATCHED cohort window, mirroring
    // `measure_leg` line for line except that it drives the batched runner entry point with the
    // COHORT params (one fresh worker, one cool gate, ONE timed window over all B streams) and
    // fills the cohort audit channel. The RunTimeout budget scales to the window it bounds: a
    // cohort window commits B*N tokens under a per-committed-token band ceiling, so the budget is
    // `B*N × ceiling × margin` — the single-stream `N × ceiling × margin` would under-bound an
    // honest cohort window by a factor of B and trip on every run.
    let measure_cohort_leg = |plan: Option<&OfficialSandboxPlan>,
                              engine: &str,
                              weights: &str,
                              base_args: &[String],
                              leg_spec: &bench_protocol::SpecConfig,
                              regime: measure_job::LegRegime,
                              params: &bench_runner::CohortTimingParams|
     -> bench_runner::Result<measure_job::LegInvocation> {
        let extra_args = base_args.to_vec();
        let mut recorded = coolgate::GateState::SkippedNoReader;
        let mut recorded_quiesce = quiescegate::QuiesceRecord::default();
        let wire_head_provenance = std::cell::RefCell::new(None);
        let mut spawn = || -> bench_runner::Result<Session<ChildStdioTransport>> {
            let transport = spawn_official_worker(plan, engine, weights, &extra_args, &[])?;
            let (session, hello) = Session::connect(transport)?;
            *wire_head_provenance.borrow_mut() = hello.head_provenance.clone();
            Ok(session)
        };
        let mut gate = |phase: &str| -> bench_runner::Result<()> {
            // BOTH gates guard the timed window, quiescence first: the box must be idle before the
            // GPU temperature is even read (David 2026-09-17).
            let (quiesced, cooled) =
                quiescegate::timed_phase_gates_report(phase, bench_core::constants::Platform::Mlx)?;
            recorded_quiesce = quiesced;
            recorded = cooled.state;
            Ok(())
        };
        let band_ceiling_spt = match calibration.as_ref() {
            Some(cal) if cal.serial_mean.is_finite() && cal.serial_mean > 0.0 => {
                cal.serial_mean * cal.band_high
            }
            _ => bench_core::constants::RUN_TIMEOUT_DEFAULT_BAND_CEILING_SECONDS_PER_TOKEN,
        };
        let run_timeout = bench_core::score::run_timeout_budget(
            args.tokens * params.batch_size as usize,
            band_ceiling_spt,
            bench_core::constants::RUN_TIMEOUT_MARGIN,
        )
        .map_err(|detail| RunnerError::RunTimeoutBudgetInvalid { detail })?;
        let params = params
            .clone()
            .with_run_timeout(Some(run_timeout))
            .with_spec(measure_job::requested_wire_spec(leg_spec, regime));
        let t =
            bench_runner::run_batched_free_run_decode_phase_fresh(&mut spawn, &mut gate, &params)?;
        // COMPOSITE (Gemma cohort scoring) — the phase-split window, straight off the runner's
        // new fields (benchd's own parent clock; the engine reports nothing new here).
        let cohort_phase_windows = measure_job::CohortPhaseWindows::from(&t);
        // Per-stream timing (gap G2, REPORT-ONLY) — the PR-A carry, lifted VERBATIM for the
        // attestation seal. Untrusted for scoring; nothing enforced reads it.
        let per_stream_timing = measure_job::PerStreamTimingCarry::from(&t);
        Ok(measure_job::LegInvocation {
            benchd_seconds_per_token: t.seconds_per_token,
            gate_state: recorded,
            quiesce: recorded_quiesce,
            telemetry: None,
            wire_effective_spec: t.effective_spec,
            wire_head_provenance: wire_head_provenance.into_inner(),
            regime,
            free_run_audit: None,
            // B2 — the batched cohort's clock split rides on `cohort_phase_windows` below; the
            // single-stream `phase_window` channel is a v1.1 free-run fact and has none here.
            phase_window: None,
            cohort_audit: Some(t.audit),
            cohort_phase_windows: Some(cohort_phase_windows),
            per_stream_timing: Some(per_stream_timing),
            // (b) admission — surface the candidate's committed rectangle UNJUDGED for benchd's
            // trusted-oracle tolerance gate. Surfaced on BOTH cohort legs; `validate_leg_report`
            // keeps it only on the candidate leg (the serial control is not token-judged).
            cohort_committed_tokens_by_stream: Some(t.tokens_by_stream),
        })
    };

    // (b) admission — the TRUSTED-ORACLE closure passed to `run_cohort_measure_job`. It is the ONLY
    // place the oracle's build + weights are fixed, and it fixes BOTH to organizer-controlled sources
    // the candidate cannot touch:
    //   * BUILD (N1): the trusted worker bin comes from `resolve_trusted_oracle_worker_bin`, which
    //     reads ONLY `MLXFAST_TRUSTED_ORACLE_WORKER_BIN` and FAILS CLOSED if unset — it shares NO
    //     fallback with the candidate/baseline resolver, so the oracle can never be the candidate
    //     build. The oracle's forward AND weight-load code are therefore the organizer's.
    //   * WEIGHTS: `--weights` is the ORGANIZER's reference weights dir (`args.weights` — the SAME
    //     organizer source both measured legs already use, main.rs ~1563), fixed HERE at spawn, NEVER
    //     derived from the candidate's response / journal / env. `run_cohort_measure_job` passes this
    //     closure only TOKENS (organizer replay seeds + the candidate journal to JUDGE), never a
    //     weights path.
    // So even a candidate that loaded rogue weights in its OWN cohort run only makes its journal
    // diverge MORE from this organizer-weights reference (→ rejected), never helps it pass.
    let oracle_weights = args.weights.to_string_lossy().to_string();
    let oracle = |replay_seeds_by_stream: &[Vec<i64>],
                  committed_by_stream: &[Vec<i64>]|
     -> bench_runner::Result<bench_protocol::CohortReferenceReplayReport> {
        // TRUSTED BUILD, FAIL-CLOSED — never a fallback to the candidate worker bin.
        let trusted_bin =
            measure_job::resolve_trusted_oracle_worker_bin().map_err(RunnerError::Protocol)?;
        // Spawn the trusted worker over the ORGANIZER reference weights on a PLAIN runtime-worker
        // argv (the verb is NOT behind the --speculative-protocol gate).
        let transport = ChildStdioTransport::spawn(&trusted_bin, &oracle_weights, &[])?;
        let (mut session, hello) = Session::connect(transport)?;
        // N1 wire half — REFUSE unless the (trusted) hello advertised the capability. The UNTRUSTED
        // candidate worker never advertises it, so benchd never asks it for a reference argmax.
        if !hello.supports_cohort_reference_replay() {
            return Err(RunnerError::CapabilityNotAdvertised {
                capability: bench_protocol::CAPABILITY_COHORT_REFERENCE_REPLAY.to_string(),
            });
        }
        session.cohort_reference_replay(replay_seeds_by_stream, committed_by_stream)
    };

    let outcome = match cohort_members {
        // COHORT — the batched cohort pair loop: same alternation/retry/die-5 machinery, cohort
        // params, cohort seal.
        Some(members) => measure_job::run_cohort_measure_job(
            &golden_fixtures,
            members,
            &weights_digest,
            &commit,
            &cfg,
            CohortSeams {
                measure_serial: |params: &bench_runner::CohortTimingParams| {
                    measure_cohort_leg(
                        serial_plan.as_ref(),
                        &baseline_exec,
                        &serial_weights,
                        &serial_base_args,
                        &serial_wire_spec,
                        measure_job::serial_control_regime_for(candidate_regime),
                        params,
                    )
                },
                measure_candidate: |params: &bench_runner::CohortTimingParams| {
                    measure_cohort_leg(
                        candidate_plan.as_ref(),
                        &candidate_exec,
                        &candidate_weights,
                        &candidate_base_args,
                        &candidate_wire_spec,
                        candidate_regime,
                        params,
                    )
                },
                oracle,
            },
        )?,
        None => measure_job::run_measure_job(
            &golden_fixtures,
            &weights_digest,
            &commit,
            &cfg,
            measure_serial,
            measure_candidate,
        )?,
    };

    // Seal results.json (+ bare-basename .sha256) and anchor its digest inside the integrity
    // sidecar (finding 10 — digest INSIDE benchmark-integrity, name derived from the results
    // stem, no hardcoded clobbering sibling).
    let results_json = outcome
        .results
        .to_sealed_json()
        .map_err(|e| format!("results.json serialization failed: {e}"))?;
    let results_path = args.out.join("results.json");
    let results_sha256 = write_results_json(&results_path, &results_json)?;

    let integrity_name = format!(
        "benchmark-integrity.{}.json",
        results_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
    );
    let integrity_path = args.out.join(integrity_name);
    let integrity = build_measure_job_integrity(
        args,
        MeasureJobSealInputs {
            results_path: results_path.display().to_string(),
            results_sha256,
            candidate_executable: candidate_exec,
            baseline_executable: baseline_exec,
            candidate_workspace_sha256: candidate_ws_digest.sha256.clone(),
            baseline_workspace_sha256: baseline_ws_digest.sha256.clone(),
            golden_sha256,
            contract_sha256,
            weights_sha256: weights_digest.sha256.clone(),
            weights_file_count: weights_digest.file_count,
            weights_byte_count: weights_digest.byte_count,
        },
    );
    let integrity_json = serde_json::to_string_pretty(&integrity)
        .map_err(|e| format!("integrity serialization failed: {e}"))?;
    std::fs::write(&integrity_path, format!("{integrity_json}\n"))
        .map_err(|e| format!("could not write {}: {e}", integrity_path.display()))?;

    // finding R19 — there is no mid-pair hard die: a thermal-gate timeout (and every other reject
    // class) was retried once inside the pair loop and, on persistence, simply left the pair
    // unaccepted; too few accepted pairs is the die-5 verdict below. results.json is already sealed.
    eprintln!(
        "benchd measure-job: wrote {} (accepted_pairs={}, candidate_accepted={})",
        results_path.display(),
        outcome.results.accepted_pair_count,
        outcome.candidate_accepted,
    );

    // R14 — the serial-band verdict (results.json is ALREADY sealed with the calibration provenance).
    // `--calibration-bootstrap` SKIPS the band check (authoring mode). Otherwise: a MISSING calibration
    // under BASELINE_BAND_ENFORCE fails closed (die-6); a present calibration enforces the pooled serial
    // mean / calibration mean band + decode_tokens match (die-6). A drifted BASELINE invalidates the
    // comparison, so die-6 takes PRECEDENCE over the die-5 candidate verdict.
    if args.calibration_bootstrap {
        // R13/R14 — bootstrap AUTHORS the band (it does not check it). Author the per-target entry
        // ONLY after a fully-accepted, parity-true run, merging into any existing BASELINE_CALIBRATION
        // file (other targets preserved) and installing it atomically. A rejected/parity-false run,
        // or a bootstrap without --target-id / without a destination, authors nothing (logged, not fatal).
        if measure_job::should_author_bootstrap(
            outcome.candidate_accepted,
            outcome.results.parity_all_ok,
        ) {
            match (
                args.target_id.as_deref(),
                std::env::var("BASELINE_CALIBRATION")
                    .ok()
                    .filter(|s| !s.trim().is_empty()),
            ) {
                (Some(tid), Some(path)) => {
                    let path = std::path::PathBuf::from(path.trim());
                    let existing = std::fs::read(&path).ok();
                    let json = measure_job::build_bootstrap_calibration(
                        existing.as_deref(),
                        &measure_job::BootstrapAuthorInput {
                            target_id: tid,
                            // #105 cycle-5 — author the file's REQUIRED series + track identity from
                            // the run that measured the band, so the authored file passes its own
                            // fence on the next (same-series, same-track) run and dies on any other.
                            // W3 — the run's OWN series, not the hardcoded TF tag: a free-run
                            // bootstrap that stamped `teacher_forced_v1` would author a file that
                            // die-6s every subsequent free-run run against its own band.
                            timed_mode: measure_job::run_timed_mode(candidate_regime),
                            track_id: &outcome.results.track_id,
                            pooled_serial_mean: outcome
                                .results
                                .aggregate
                                .baseline_serial_seconds_per_token_mean,
                            tokens: args.tokens,
                            mtp_depth: args.mtp_depth,
                            serial_control_depth: measure_job::SERIAL_CONTROL_DEPTH,
                            pairs_total: outcome.results.accepted_pair_count,
                        },
                    )
                    .map_err(MeasureJobFailure::die6)?;
                    measure_job::write_bootstrap_calibration(&path, &json)
                        .map_err(MeasureJobFailure::die6)?;
                    eprintln!(
                        "benchd measure-job: --calibration-bootstrap authored targets[{tid}] in {} \
                         (serial_mean={}, decode_tokens={})",
                        path.display(),
                        outcome.results.aggregate.baseline_serial_seconds_per_token_mean,
                        args.tokens,
                    );
                }
                _ => eprintln!(
                    "benchd measure-job: --calibration-bootstrap needs both --target-id and a \
                     BASELINE_CALIBRATION destination path to author; skipping the write."
                ),
            }
        } else {
            eprintln!(
                "benchd measure-job: --calibration-bootstrap SKIPPED authoring — the run was not \
                 fully accepted + parity-true (candidate_accepted={}, parity_all_ok={}).",
                outcome.candidate_accepted, outcome.results.parity_all_ok,
            );
        }
    } else {
        match calibration.as_ref() {
            None => {
                if band_enforce {
                    return Err(MeasureJobFailure::die6(
                        "no BASELINE_CALIBRATION but BASELINE_BAND_ENFORCE=1 (default) — cannot \
                         validate the serial baseline; failing closed (die 6). Set \
                         BASELINE_BAND_ENFORCE=0 or pass --calibration-bootstrap to author one."
                            .to_string(),
                    ));
                }
            }
            Some(cal) => {
                let pooled_serial_mean = outcome
                    .results
                    .aggregate
                    .baseline_serial_seconds_per_token_mean;
                // Only meaningful once the candidate is accepted (a valid pooled serial mean); a
                // rejected candidate is the die-5 verdict below, not a baseline-drift die-6.
                if outcome.candidate_accepted {
                    if let Err(reason) = measure_job::enforce_serial_band(
                        pooled_serial_mean,
                        args.tokens,
                        cal,
                        band_enforce,
                    ) {
                        eprintln!("benchd measure-job: {reason}");
                        return Ok(MeasureJobVerdict::CalibrationDrift);
                    }
                }
            }
        }
    }

    Ok(if outcome.candidate_accepted {
        MeasureJobVerdict::Accepted
    } else {
        MeasureJobVerdict::RejectedDie5
    })
}

/// Parsed `overlay-timing` flags (A-3, seam 3 LOCAL).
struct OverlayTimingArgs {
    gates_score: PathBuf,
    results: PathBuf,
    score_path: PathBuf,
    integrity: Option<PathBuf>,
    /// R17 — the contract fixture whose `timed_prompt_pool | length` gives the expected pool_size
    /// when env `MLXFAST_QWEN_MTP_POOL_SIZE` is unset. Optional; if neither is available the overlay
    /// fails closed (pool_size unknown ⇒ no score).
    contract: Option<PathBuf>,
}

/// The `overlay-timing` command line. [`OVERLAY_TIMING_USAGE`] is its prose; this is its grammar.
static OVERLAY_TIMING_FLAGS: &[FlagSpec] = &[
    FlagSpec::value(&["--gates-score"]).required(),
    FlagSpec::value(&["--results"]).required(),
    FlagSpec::value(&["--score-path"]).required(),
    FlagSpec::value(&["--integrity"]),
    // R17 — the fixture whose `timed_prompt_pool | length` gives the expected pool_size.
    FlagSpec::value(&["--contract"]),
];

/// The `overlay-timing` subcommand: its name, its usage prose, its flag table and its parse.
static OVERLAY_TIMING_CLI: CliSpec<OverlayTimingArgs> = CliSpec {
    name: "overlay-timing",
    usage: OVERLAY_TIMING_USAGE,
    flags: OVERLAY_TIMING_FLAGS,
    parse: parse_overlay_timing_args,
};

/// A-3: the Option-A OVERLAY subcommand (seam 3, LOCAL). Parses the flags, merges the sealed
/// gates-score.json with the measure-job results.json (3.8 median regime), seals the ranked
/// score.json (+ bare-basename .sha256), and re-anchors integrity `score_sha256` over the merged
/// bytes. Exit 0 when the merged score PASSES, nonzero (1) when a floor/ceiling bound fails; 2 on
/// a usage error; 1 on a load/validation/IO error.
fn run_overlay_timing_cli(args: &[String]) -> ExitCode {
    run_cli(&OVERLAY_TIMING_CLI, args, |parsed| {
        // The merged score encodes any bound failure; a non-pass exits nonzero so callers notice
        // (a floor/ceiling fail sets passed=false, score=null), matching the iterate contract.
        execute_overlay_timing(parsed)
            .map(iterate_exit_status)
            .map_err(|msg| (1, msg))
    })
}

/// Parse an `overlay-timing` command line against [`OVERLAY_TIMING_CLI`].
fn parse_overlay_timing_args(args: &[String]) -> Result<Option<OverlayTimingArgs>, String> {
    let Some(flags) = parse_flags(args, OVERLAY_TIMING_CLI.flags)? else {
        // `-h`/`--help`.
        return Ok(None);
    };
    Ok(Some(OverlayTimingArgs {
        gates_score: flags
            .path("--gates-score")
            .ok_or_else(|| flags.missing_required("--gates-score"))?,
        results: flags
            .path("--results")
            .ok_or_else(|| flags.missing_required("--results"))?,
        score_path: flags
            .path("--score-path")
            .ok_or_else(|| flags.missing_required("--score-path"))?,
        integrity: flags.path("--integrity"),
        contract: flags.path("--contract"),
    }))
}

/// R17 — resolve the expected pool SHAPE fail-closed. `pool_size` comes from env
/// `MLXFAST_QWEN_MTP_POOL_SIZE` when set, else the `--contract` fixture's `timed_prompt_pool |
/// length`; if NEITHER is available the overlay refuses to score (pool_size unknown). `min_per_prompt`
/// is env `MLXFAST_QWEN_MTP_MIN_PAIRS_PER_PROMPT` (default 1).
fn resolve_pool_expectation(contract: Option<&Path>) -> Result<overlay::PoolExpectation, String> {
    let pool_size = match std::env::var("MLXFAST_QWEN_MTP_POOL_SIZE") {
        Ok(v) if !v.trim().is_empty() => v.trim().parse::<usize>().map_err(|e| {
            format!("MLXFAST_QWEN_MTP_POOL_SIZE ({v:?}) is not a valid pool size: {e}")
        })?,
        _ => {
            let path = contract.ok_or(
                "pool_size unknown: set MLXFAST_QWEN_MTP_POOL_SIZE or pass --contract <fixture> \
                 (fail-closed, no score)",
            )?;
            let bytes = std::fs::read(path)
                .map_err(|e| format!("--contract read failed ({}): {e}", path.display()))?;
            let value: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|e| format!("--contract parse failed: {e}"))?;
            let pool = value
                .get("timed_prompt_pool")
                .and_then(|p| p.as_array())
                .ok_or("--contract fixture has no `timed_prompt_pool` array")?;
            pool.len()
        }
    };
    if pool_size == 0 {
        return Err("resolved pool_size is 0: the pool must have at least one prompt".to_string());
    }
    let min_per_prompt = match std::env::var("MLXFAST_QWEN_MTP_MIN_PAIRS_PER_PROMPT") {
        Ok(v) if !v.trim().is_empty() => v.trim().parse::<usize>().map_err(|e| {
            format!("MLXFAST_QWEN_MTP_MIN_PAIRS_PER_PROMPT ({v:?}) is not a valid count: {e}")
        })?,
        _ => 1,
    };
    if min_per_prompt == 0 {
        return Err(
            "MLXFAST_QWEN_MTP_MIN_PAIRS_PER_PROMPT is 0: at least 1 pair per prompt \
                    is required"
                .to_string(),
        );
    }
    Ok(overlay::PoolExpectation {
        pool_size,
        min_per_prompt,
    })
}

/// Execute the overlay: load + validate the two inputs (fail-closed), merge them (pure
/// `overlay::merge_overlay`), seal the ranked score.json via the SHARED bare-basename sealed-write
/// (finding 14 — same recipe the measure-job results.json uses), and re-anchor integrity
/// `score_sha256` over the merged bytes. Returns `Ok(passed)`.
fn execute_overlay_timing(args: &OverlayTimingArgs) -> Result<bool, String> {
    // Load + deserialize the sealed gates-score.json into a typed ScorePayload.
    let gates_bytes = std::fs::read(&args.gates_score).map_err(|e| {
        format!(
            "--gates-score read failed ({}): {e}",
            args.gates_score.display()
        )
    })?;
    let gates: crate::score::ScorePayload = serde_json::from_slice(&gates_bytes)
        .map_err(|e| format!("--gates-score parse failed: {e}"))?;

    // Load + deserialize the measure-job results.json (fail-closed on a missing aggregate).
    let results_bytes = std::fs::read(&args.results)
        .map_err(|e| format!("--results read failed ({}): {e}", args.results.display()))?;
    let results = overlay::ResultsView::parse(&results_bytes)?;

    // R12 — the EXPECTED track the overlay was told to score: env `MLXFAST_QWEN_MTP_TRACK_ID` when
    // set (the same var the ranked yml passes as `$track`). When set, the merge REJECTS a
    // results.json sealed for a different track; when unset, the sealed track_id must still be
    // non-empty (never trust an arbitrary one).
    let expected_track = std::env::var("MLXFAST_QWEN_MTP_TRACK_ID")
        .ok()
        .filter(|s| !s.trim().is_empty());

    // W3 (§5) — the EXPECTED TIMED SERIES the overlay was told to score: env
    // `MLXFAST_QWEN_MTP_TIMED_SERIES` when set. §5 makes baselines/floors/bands PER-SERIES, so when
    // the operator states the series, a results.json sealed for a different one is REFUSED. When
    // unset, the fence still enforces the file's INTERNAL series coherence (known tags, per-pair
    // agreement, recomputed comparability) — it just does not pin which series is expected.
    let expected_series = std::env::var("MLXFAST_QWEN_MTP_TIMED_SERIES")
        .ok()
        .filter(|s| !s.trim().is_empty());

    // R17 — resolve the expected pool SHAPE fail-closed (env / --contract fixture length).
    let pool = resolve_pool_expectation(args.contract.as_deref())?;

    // STEP 3 (David 2026-09-15) — the merge's two per-track fences (the paired-flow entry fence and
    // the scored-regime fence) read the TRACK FIXTURE, and there is no in-tree table left for them
    // to fall back to, so `--contract` is REQUIRED here as it is on every other scored path.
    let contract_path = args.contract.as_deref().ok_or(
        "no --contract: the overlay's paired-flow and scored-regime fences read the track fixture,          and there is no in-tree table to fall back to",
    )?;
    let scored_contract = bench_core::contract::load(contract_path)?.contract;

    // The pure merge: validation is fail-closed inside merge_overlay.
    let outcome = overlay::merge_overlay(
        &gates,
        &results,
        expected_track.as_deref(),
        expected_series.as_deref(),
        &pool,
        &scored_contract,
    )?;

    // Seal the ranked score.json (+ bare-basename `.sha256`) via the SHARED sealed-write, and use
    // the returned digest as the integrity re-anchor value (over the merged bytes).
    let score_sha256 = write_results_json(&args.score_path, &outcome.sealed_json)?;

    // Emit the sealed payload to STDOUT (parity with the iterate/benchmark.sh cat) — no trailing
    // newline; the write_results_json already wrote the exact bytes to disk.
    print!("{}", outcome.sealed_json);

    // Re-anchor integrity `score_sha256` over the merged bytes (GEMMA-OVL `@67699fc4:177-181`).
    // With --integrity, REWRITE the existing anchor's score_sha256/score_path in place (preserving
    // the measure-job's weights/workspace provenance); else write a fresh sidecar next to the score.
    reanchor_overlay_integrity(args, &score_sha256)?;

    eprintln!(
        "benchd overlay-timing: wrote {} (passed={}, score={})",
        args.score_path.display(),
        outcome.passed,
        outcome
            .score
            .map(|s| s.to_string())
            .unwrap_or_else(|| "null".to_string()),
    );
    Ok(outcome.passed)
}

/// Reduce a filesystem path to a WORKSPACE-RELATIVE form before it is sealed into an integrity
/// artifact, so no operator home directory (`/Users/<home>/…`) travels with a run.
///
/// A sealed `benchmark-integrity.*.json` is an artifact and travels with the run, and the
/// secret-tier rule keeps home/box paths out of every sealed artifact. These path fields are
/// PROVENANCE only — the sha256 digests beside them carry identity, and no consumer resolves the
/// strings back to files — so relativising changes what a human reads, never what a gate checks.
///
/// A relative path is already leak-free and returned unchanged. An absolute path is made relative
/// to the run's current working directory (its workspace root) when it lies under it; otherwise the
/// operator's own `$HOME` prefix is stripped; and as a final guard any residual `/Users/<user>/` or
/// `/home/<user>/` head is dropped. The result therefore never begins with a user-home segment. A
/// path that is absolute but outside any home (e.g. `/opt/…`) is kept as-is — it carries no home to
/// leak.
fn relativize_for_seal(path: &Path) -> String {
    if path.is_relative() {
        return path.display().to_string();
    }
    // (1) workspace-relative: under the run's current working directory.
    if let Ok(cwd) = std::env::current_dir() {
        if let Ok(rel) = path.strip_prefix(&cwd) {
            return rel_or_dot(rel);
        }
    }
    // (2) home-relative: strip the operator's own $HOME (drops the username with it).
    if let Some(home) = std::env::var_os("HOME").filter(|h| !h.is_empty()) {
        if let Ok(rel) = path.strip_prefix(PathBuf::from(home)) {
            return rel_or_dot(rel);
        }
    }
    // (3) final guard: a foreign home ($HOME unset, or a path under another user's home) still must
    // not seal a `/Users/<user>/` or `/home/<user>/` head.
    drop_home_head(path)
}

/// `rel` rendered, or `"."` when stripping the prefix left it empty (the path WAS the prefix).
fn rel_or_dot(rel: &Path) -> String {
    if rel.as_os_str().is_empty() {
        ".".to_string()
    } else {
        rel.display().to_string()
    }
}

/// Drop a leading `/Users/<user>/` or `/home/<user>/` head, returning the remaining tail; the path
/// unchanged when it has no such head. The last safety net for [`relativize_for_seal`].
fn drop_home_head(path: &Path) -> String {
    use std::path::Component;
    let comps: Vec<Component> = path.components().collect();
    if comps.len() >= 3 {
        if let (Component::RootDir, Component::Normal(top), Component::Normal(_user)) =
            (&comps[0], &comps[1], &comps[2])
        {
            if *top == "Users" || *top == "home" {
                let tail: PathBuf = comps[3..].iter().collect();
                return rel_or_dot(&tail);
            }
        }
    }
    path.display().to_string()
}

/// Parsed `iterate` flags.
struct IterateArgs {
    engine: String,
    weights: PathBuf,
    /// `--golden <PATH>` (repeatable), in the order given. Never empty. Only the paired path
    /// measures more than one: pair `k` (1-based) measures golden `(k - 1) mod N`.
    goldens: Vec<PathBuf>,
    /// The `--golden-sha256` / `--golden-bytes` pins, matched to `goldens` by position. Empty, or
    /// one per golden.
    golden_pins: Vec<GoldenIntegrityPin>,
    baseline_prefill_spt: Option<f64>,
    baseline_decode_spt: Option<f64>,
    mode: Mode,
    score_path: PathBuf,
    /// Tri-state local GPU cool gate (#60.3): `None` = per-mode default (local-iterate OFF,
    /// local-submit ON, official n/a); `Some(true)` = `--cool-gate` forces ON; `Some(false)` =
    /// `--no-cool-gate` forces OFF (needed to force submit's default-ON gate off). Per David's
    /// ruling (2026-08-17) the facade always passes `--cool-gate` to match benchmark.sh.
    cool_gate: Option<bool>,
    /// R3: opt into benchd's correctness SUPERSET for local-iterate. Default OFF makes
    /// the gate Swift-exact (primary teacher-forced `cases[]` only); `--strict` also
    /// evaluates the golden's anchor/free-run gates. No effect on official mode.
    strict: bool,
    /// B5 `--capture-baseline <RECORD>`: the OFFICIAL-BASELINE CAPTURE MODE on the local-iterate
    /// path. See `capture.rs` — it runs the checked-timing legs without resolving this track's
    /// official baseline, writes only the capture record, and refuses on a track whose pair is
    /// already captured. Local-iterate only; parse refuses it by name on every other mode.
    capture_baseline: Option<PathBuf>,
    /// `--capture-timed-only` (a8/David ruling): in `--capture-baseline` mode, SKIP the
    /// teacher-forced correctness gate and run ONLY the timed prefill+decode pass, appending the
    /// pair to the capture record. The engine cannot change between passes of the organizer's own
    /// calibration, so the PLE-SSD-bound gate is run ONCE per prompt per window (the first capture
    /// pass, WITHOUT this flag) and every remaining pass carries it. Requires `--capture-baseline`
    /// (refused otherwise at parse); it never touches a scored/official run.
    capture_timed_only: bool,
    /// `--capture-passes <spec>` (a8 Option-A capture restructure, David 2026-08-31): a comma list
    /// of per-pass record LABELS (e.g. `W,A,A,B,B` — warmup + A1 A2 + B1 B2). When present, this ONE
    /// invocation runs EVERY listed pass over ONE persistent model residency (`run_capture_passes`),
    /// so on MLX the model loads ONCE per prompt window instead of once per pass. NO pass runs the
    /// teacher-forced correctness gate (dropped from calibration entirely, David 2026-08-31); EVERY
    /// pass is timed-only, with the free-run oracle token-match as the correctness evidence.
    /// Each pass's pair is appended to a per-label record derived from the `--capture-baseline` base
    /// path (`<base-stem>.<label>.<ext>`); a repeated label (the two `A`s) merges into one record,
    /// exactly as two separate `--capture-baseline …A.json` invocations do today. `Some` only with
    /// `--capture-baseline`, and mutually exclusive with `--capture-timed-only` (that per-invocation
    /// modifier is subsumed — the per-pass timed-only decision is positional here). Refused otherwise
    /// at parse; it never touches a scored/official run (official refuses `--capture-baseline`).
    capture_passes: Option<Vec<String>>,
    /// `--weights-digest <sha256>:<bytes>:<files>` (Option B digest-hoist): a pre-computed weights
    /// digest to USE instead of re-hashing the immutable ~105 GB tree. The window computes it ONCE
    /// via `benchd weights-digest` and passes it to every remaining pass; it is byte-identical to
    /// a per-pass `dir_digest`. `Some` only in `--capture-baseline` mode — refused otherwise at
    /// parse (RIDER 1), so a passed-in digest NEVER reaches an official/scored seal, which always
    /// hashes for itself.
    weights_digest: Option<DirDigest>,
    /// `--contract <PATH>` — the track fixture whose `official_scoring_enabled` ARM STATE gates
    /// the SOLE scored path (`--mode official`). REQUIRED on official (the arm gate refuses without
    /// it); ignored on the local modes. Moved here from the retired measure-job when flow A became
    /// the only sealed/scored path, so the David 2026-08-26 arm gate travels with the seal.
    contract: Option<PathBuf>,
    /// The per-module SPECULATIVE CONFIG the TIMED free-run decode legs request on the wire,
    /// resolved at parse from `--mtp-depth N` (the convenience that builds
    /// `{"mode":"mtp","mtp":{"depth":N}}`) or `--candidate-spec <JSON>` (the explicit override) —
    /// the SAME two flags, with the same mutual exclusion and the same depth cap, that the
    /// measure-job surface already carries.
    ///
    /// `None` — neither flag given — is TODAY'S BEHAVIOUR BYTE-FOR-BYTE: `free_decode_begin` goes
    /// out with no `spec`, the engine resolves its default (serial by protocol), and nothing is
    /// echo-checked. `Some` arms spec-never-ignored on every timed leg that carries it.
    spec: Option<bench_protocol::SpecConfig>,
    /// `--baseline-workspace <DIR>` — THE REFERENCE TREE (David 2026-09-08). REQUIRED on the
    /// ranked paired path (`--mode official` on a
    /// [`bench_core::contract::scores_against_live_control_leg`] track): the organizer-staged, built
    /// reference tree this box runs the SERIAL-CONTROL leg on. Falls back to the
    /// [`baseline::BASELINE_WORKSPACE_ENV`] runner variable; absent from both, the run refuses by
    /// name. Ignored on every other path, which measures no control leg.
    baseline_workspace: Option<PathBuf>,
    /// `--baseline-calibration <FILE>` — THIS BOX's calibration file. REQUIRED on the ranked
    /// paired path, falling back to [`baseline::BASELINE_CALIBRATION_ENV`]. It is a HEALTH BAND
    /// for the control leg and NEVER a denominator: no number in it reaches the score.
    baseline_calibration: Option<PathBuf>,
    /// `--box <RUNNER>` — the box this run is on, for the calibration file's `box` check. The
    /// job's own `RUNNER_NAME` wins when it is set (Actions sets it); this flag is how an operator
    /// names the box off Actions. A run that can name neither refuses by name.
    box_name: Option<String>,
    /// `--control-golden <PATH>` (repeatable) — the golden the SERIAL-CONTROL leg (leg 1) verifies
    /// its decode tokens against. Control golden `i` belongs to golden `i`, so it is empty or has
    /// one entry per golden. Empty means leg 1 verifies against `--golden`, which is correct ONLY when
    /// the timed oracle at the declared draft depth is byte-identical to the serial tape. A track
    /// that ships PER-DEPTH oracle tapes passes its serial live golden here: leg 1 is serial by
    /// construction, and on the MLX engine the per-depth tape diverges from the serial one at step
    /// 1, so a shared golden kills leg 1. Paired path only — refused BY NAME on a run that
    /// measures no control leg, never silently ignored.
    control_goldens: Vec<PathBuf>,
    /// The `--control-golden-sha256` / `--control-golden-bytes` integrity pins, matched to
    /// `control_goldens` by position and checked on the raw bytes before the parse exactly as
    /// `golden_pins` are.
    control_golden_pins: Vec<GoldenIntegrityPin>,
    /// `--engine-resource NAME=PATH` (repeatable) — the out-of-checkpoint inputs the runner needs
    /// to LOAD the model (runner contract §8.1/§13b). Each becomes `--resource NAME=PATH`
    /// on EVERY engine spawn this run makes. THE VALUE COMES FROM THIS COMMAND LINE, as the engine
    /// repository's wrapper invokes benchd — never from a manifest, fixture or any other
    /// submission-editable file. Empty is today's behaviour byte for byte. See
    /// [`engine_resource`].
    engine_resources: Vec<engine_resource::EngineResource>,
}

/// The `iterate` command line. [`ITERATE_USAGE`] is its prose; this is its grammar.
static ITERATE_FLAGS: &[FlagSpec] = &[
    FlagSpec::value(&["--engine"]).required(),
    FlagSpec::value(&["--weights"]).required(),
    // Repeatable for the paired path, which cycles its pairs over the goldens. Each pin flag
    // matches `--golden` by position.
    FlagSpec::repeatable(&["--golden"]).required(),
    FlagSpec::value(&["--baseline-prefill-spt"]),
    FlagSpec::value(&["--baseline-decode-spt"]),
    FlagSpec::value(&["--mode"]),
    FlagSpec::value(&["--score-path"]),
    FlagSpec::repeatable(&["--golden-sha256"]),
    FlagSpec::repeatable(&["--golden-bytes"]),
    // The tri-state local GPU cool gate (#60.3): each spelling forces one side, and the pair is
    // refused together so a wiring conflict never resolves silently to the last one written.
    FlagSpec::switch(&["--cool-gate"]).conflicts_with(&["--no-cool-gate"]),
    FlagSpec::switch(&["--no-cool-gate"]).conflicts_with(&["--cool-gate"]),
    FlagSpec::switch(&["--strict"]),
    FlagSpec::value(&["--capture-baseline"]),
    FlagSpec::switch(&["--capture-timed-only"]),
    FlagSpec::value(&["--capture-passes"]),
    FlagSpec::value(&["--weights-digest"]),
    FlagSpec::value(&["--contract"]),
    // THE RANKED PAIRED PATH's two runner inputs (David 2026-09-08). Both are also read from
    // the runner environment; the flags are how an operator drives the path by hand.
    FlagSpec::value(&["--baseline-workspace"]),
    FlagSpec::value(&["--baseline-calibration"]),
    FlagSpec::value(&["--box"]),
    // LEG 1's OWN GOLDEN. The control leg is serial, so on a track that ships per-depth oracle
    // tapes it cannot be verified against the candidate's tape.
    FlagSpec::repeatable(&["--control-golden"]),
    FlagSpec::repeatable(&["--control-golden-sha256"]),
    FlagSpec::repeatable(&["--control-golden-bytes"]),
    // Repeatable resource passthrough. The value is taken from THIS command line and is never
    // read out of a manifest, fixture or other submission-editable file.
    FlagSpec::repeatable(&[engine_resource::ENGINE_RESOURCE_FLAG]),
    FlagSpec::value(&["--mtp-depth"]),
    FlagSpec::value(&["--candidate-spec"]),
];

/// The `iterate` subcommand: its name, its usage prose, its flag table and its parse.
static ITERATE_CLI: CliSpec<IterateArgs> = CliSpec {
    name: "iterate",
    usage: ITERATE_USAGE,
    flags: ITERATE_FLAGS,
    parse: parse_iterate_args,
};

fn run_iterate(args: &[String]) -> ExitCode {
    run_cli(&ITERATE_CLI, args, |parsed| {
        // `Ok` means the run REACHED ITS END, not that a score exists. On a scored run a score
        // was written (it encodes any failure), and a run that did not pass exits nonzero so
        // callers notice — a paired REJECT, a floor/ceiling fail, a serial-band breach, all of
        // which set `passed = false`. On a `--capture-baseline` run NO score is written at all:
        // that mode returns `Ok(true)` once the capture record is on disk, and every way it can
        // fail is an `Err` below.
        execute_iterate(parsed)
            .map(iterate_exit_status)
            .map_err(|msg| (1, msg))
    })
}

/// The boolean→process-exit contract for an iterate run: a passing run exits 0, a run that did
/// NOT pass (any fail-closed verdict — paired REJECT, floor/ceiling fail, serial-band breach)
/// exits 1 so callers notice. Extracted so the exact mapping is unit-testable (constructing a
/// real `ExitCode` end-to-end needs a live engine).
fn iterate_exit_status(passed: bool) -> u8 {
    if passed {
        0
    } else {
        1
    }
}

/// The LOCAL arm of an iterate run (`local-iterate` / `local-submit`): wire the
/// fresh-engine-per-timed-phase spawner and the per-mode cool gate, spawn the shared correctness
/// engine ONLY on the gated path, and hand all of it to `iterate::iterate_flow_windowed`.
///
/// #65: lifted out of `execute_iterate`'s baseline match. That match's job is choosing HOW a
/// run ends — preflight refusal, gates-only, official, or local — and this arm's engine
/// lifecycle detail buried the choice.
///
/// a8 (finish #236): a `timed_only` pass (capture `--capture-timed-only`) SKIPS the correctness
/// gate, so it no longer spawns the shared correctness session at all — that worker was launched
/// and hello-handshaked only to be dropped unused. The gated path (every scored/checked run, plus
/// the first per-prompt capture pass) still spawns it exactly as before.
fn run_local_iterate(
    args: &IterateArgs,
    golden: &GoldenFixture,
    digests: RunDigests<'_>,
    window: &WindowShape,
    scoring: ScoringInputs,
    timed_only: bool,
) -> Result<ScorePayload, String> {
    let weights_str = args.weights.to_string_lossy().to_string();

    // §A — the timed-only pass opens its ONE window worker here. The call launches a new
    // `runtime-worker` child and completes the hello handshake; the hello is retained (see below).
    // A spawn/handshake failure surfaces as a RunnerError into the timing path.
    // `--speculative-protocol v1.1` opts the worker into advertising `free_run_decode` so the
    // timed decode leg's free-run verbs are not refused (see `free_run_spawn_args`).
    // THE TIMED WORKER'S HELLO IS RETAINED (see `official::seal_engine_identity`): the local modes
    // seal the same engine identity the official path does, from the same place — the worker that
    // ran the timed leg.
    let timed_hello = std::cell::RefCell::new(None);
    let spawn_timed = || -> Result<Session<ChildStdioTransport>, RunnerError> {
        let transport = ChildStdioTransport::spawn(
            &args.engine,
            &weights_str,
            &free_run_spawn_args(&args.engine_resources),
        )?;
        // Every timed phase of one window must report the SAME resident identity; a change is a
        // mid-window reload and is refused by name (`resident_identity_changed_within_window`).
        connect_retaining_hello(transport, &timed_hello)
    };

    // Per-mode cool-gate default (David 2026-08-17): local-iterate OFF unless
    // `--cool-gate`; local-submit ON; official never calls it. `--no-cool-gate` forces
    // OFF regardless of mode (#60.3 tri-state). Disabled → a no-op closure, so the gate
    // machinery stays wired for the facade (which always passes it).
    let cool_gate_fn = local_cool_gate(
        args.cool_gate
            .unwrap_or_else(|| args.mode.cool_gate_on_by_default()),
    );

    // `timed_only` is the a8 capture-timed-only skip: a `FlowParams` with `timed_only = true`
    // runs ONLY the timed pass. The normal path (scored runs AND the first per-prompt capture
    // pass) sets it false, and its gate behaviour is byte-unchanged. Both arms consume the same
    // `spawn_timed` closure; only one executes.
    // The gated path's shared correctness session hello — on MLX that session ALSO runs the timed
    // legs, so it is the timed worker's hello when `spawn_timed` never fires.
    let mut session_hello = None;
    let mut payload = if timed_only {
        // a8 (finish #236): a `--capture-timed-only` pass SKIPS the correctness gate, so the shared
        // correctness session would be spawned + hello-handshaked and then never used. Do NOT spawn
        // it — the gate is gone from this path, so the worker it needed is too. This removes one of
        // the per-pass worker spawns on the ~56 timed-only passes of a calibration window. The timed
        // prefill/decode phases run over ONE worker opened by `spawn_timed`, and nothing measured
        // changes: no correctness value was ever read here.
        iterate_flow_windowed(
            None,
            iterate::FlowParams {
                golden,
                scoring,
                mode: args.mode,
                strict: args.strict,
                timed_only: true,
                digests,
                // The resolved `--mtp-depth`/`--candidate-spec` spec (None = no spec on the wire).
                spec: args.spec.clone(),
                window,
            },
            spawn_timed,
            cool_gate_fn,
        )
    } else {
        // Gated path (scored/checked runs AND the first per-prompt capture pass): spawn the shared
        // engine for the CORRECTNESS gate and connect (the shared session, §A). This SAME session
        // then serves the timed phases (load-once). `--speculative-protocol v1.1`: this resident
        // session runs the free-run timed decode leg, so it must advertise `free_run_decode` (see
        // `free_run_spawn_args`).
        let transport = ChildStdioTransport::spawn(
            &args.engine,
            &weights_str,
            &free_run_spawn_args(&args.engine_resources),
        )
        .map_err(|e| format!("failed to spawn engine {:?}: {e}", args.engine))?;
        let (mut session, hello) = Session::connect(transport)
            .map_err(|e| format!("engine hello handshake failed: {e}"))?;
        session_hello = Some(hello);
        iterate_flow_windowed(
            Some(&mut session),
            iterate::FlowParams {
                golden,
                scoring,
                mode: args.mode,
                strict: args.strict,
                timed_only: false,
                digests,
                // The resolved `--mtp-depth`/`--candidate-spec` spec (None = no spec on the wire).
                spec: args.spec.clone(),
                window,
            },
            spawn_timed,
            cool_gate_fn,
        )
    };
    // The timed legs run on the CORRECTNESS session opened just above, so `spawn_timed` may never
    // fire; that session's own hello is then the timed worker's. `session_hello` is `Some` only on
    // the gated path, where such a session exists.
    let identity_hello = timed_hello.borrow_mut().take().or(session_hello);
    if let Some(hello) = identity_hello.as_ref() {
        official::seal_engine_identity(&mut payload.metrics, hello);
    }
    Ok(payload)
}

/// --capture-passes (a8 Option-A capture restructure, David 2026-08-31): run EVERY listed pass over
/// ONE persistent model residency, appending each pass's parent-measured pair to its own per-label
/// capture record.
///
/// The model is loaded ONCE for the whole invocation: this opens a SINGLE worker session and hands
/// it to [`iterate::run_capture_passes_over_session`], which drives every pass over it — the gate
/// pass and every timed leg reuse that one ~90 GiB residency. So a calibration window is 8 loads
/// (one per prompt) instead of 40 — the whole point of the restructure. The per-pass MEASURED number is
/// byte-identical to a standalone single-pass invocation (same `iterate_flow_windowed`, same #28
/// per-phase reset between passes); see the helper's doc.
///
/// The capture GATES are applied to EACH pass's payload exactly as the single-pass path applies them
/// to its one payload: a failed correctness gate (only reachable on the FIRST, gated pass) refuses
/// the whole capture by name, and the per-leg gates run inside `capture::capture_run_from`, which is
/// what PRODUCES the pair this loop records — a failed timed phase refuses with the real cause, and
/// a leg whose engine speculated on its own refuses at `CALIBRATION-SPEC-ARMED`. A repeated label
/// merges its passes into one record; the merge's own identity + finite-positive checks stay as the
/// backstop.
fn run_capture_passes(
    args: &IterateArgs,
    golden: &GoldenFixture,
    digests: RunDigests<'_>,
    window: &WindowShape,
    destination: capture::CaptureDestination<'_>,
) -> Result<bool, String> {
    let capture::CaptureDestination {
        base: capture_base,
        labels,
        identity,
    } = destination;
    let weights_str = args.weights.to_string_lossy().to_string();

    // The timed-phase spawner, IDENTICAL to run_local_iterate's. Every pass reuses the one resident
    // worker opened below, so this never fires.
    // `--speculative-protocol v1.1` opts the worker into `free_run_decode` for the free-run timed
    // decode leg (see `free_run_spawn_args`).
    let spawn_timed = || -> Result<Session<ChildStdioTransport>, RunnerError> {
        let transport = ChildStdioTransport::spawn(
            &args.engine,
            &weights_str,
            &free_run_spawn_args(&args.engine_resources),
        )?;
        let (session, _hello) = Session::connect(transport)?;
        Ok(session)
    };
    // Per-mode cool gate, IDENTICAL to run_local_iterate's (capture is local-iterate only).
    let cool_gate_fn = local_cool_gate(
        args.cool_gate
            .unwrap_or_else(|| args.mode.cool_gate_on_by_default()),
    );

    // Open the ONE resident worker for the whole window (load-once). It serves the first pass's
    // correctness gate and, on MLX, every pass's timed legs; the model loads exactly once here.
    // `--speculative-protocol v1.1`: this resident session drives the free-run capture/timed decode
    // legs, so it must advertise `free_run_decode` (see `free_run_spawn_args`).
    let transport = ChildStdioTransport::spawn(
        &args.engine,
        &weights_str,
        &free_run_spawn_args(&args.engine_resources),
    )
    .map_err(|e| format!("failed to spawn engine {:?}: {e}", args.engine))?;
    let (mut session, _hello) =
        Session::connect(transport).map_err(|e| format!("engine hello handshake failed: {e}"))?;

    // Drive every pass over that one residency. INERT (0.0, 0.0) baselines, exactly as the
    // single-pass capture path uses — the capture record reads only the parent-measured pair and
    // the correctness verdict, never a score.
    let payloads = iterate::run_capture_passes_over_session(
        &mut session,
        iterate::FlowParams {
            golden,
            // INERT (0.0, 0.0) baselines, exactly as the single-pass capture path uses — the
            // capture record reads only the parent-measured pair and the correctness verdict,
            // never a score.
            scoring: ScoringInputs::local(0.0, 0.0),
            mode: args.mode,
            strict: args.strict,
            // GATELESS: every capture pass is timed-only; the loop sets this itself.
            timed_only: true,
            digests,
            // CALIBRATION authors the SERIAL reference pair; it never requests a spec.
            spec: None,
            window,
        },
        labels.len(),
        spawn_timed,
        cool_gate_fn,
    );

    // Apply the capture gates + merge per pass, in label order — the SAME gates the single-pass path
    // applies to its one payload.
    for (label, payload) in labels.iter().zip(payloads.iter()) {
        if !payload.metrics.passed_correctness {
            return Err(format!(
                "--capture-baseline refuses to record this run: the correctness gate failed \
                 ({:?}) on capture pass {label:?}; a baseline is captured only from a healthy \
                 stock run",
                payload.metrics.error
            ));
        }
        // The per-leg gates PRODUCE the pair (`capture::capture_run_from`): a leg whose timing
        // failed, or whose engine speculated on its own, yields no `CaptureRun` at all, so this
        // recorder cannot proceed without them having run.
        let run = capture::capture_run_from(&payload.metrics)?;
        let record_path = capture_pass_record_path(capture_base, label);
        let record = capture::record_run(&record_path, identity.clone(), run)?;
        eprintln!(
            "benchd iterate: capture pass {label:?} -> record {} now carries {} run(s) (prefill \
             CV {:?}%, decode CV {:?}%); no score was written",
            record_path.display(),
            record.run_count,
            record.prefill_cv_percent,
            record.decode_cv_percent,
        );
    }
    eprintln!(
        "benchd iterate: --capture-passes ran {} pass(es) over one model residency; no score \
         was written",
        labels.len()
    );
    Ok(true)
}

/// Tokens per second from seconds per token, for a HUMAN-FACING line (David: every human-facing
/// surface shows tok/s; seconds-per-token stays the internal representation). A non-positive or
/// non-finite reading has no rate, and prints as `0.0` rather than an infinity.
fn tokens_per_second(seconds_per_token: f64) -> f64 {
    if seconds_per_token.is_finite() && seconds_per_token > 0.0 {
        1.0 / seconds_per_token
    } else {
        0.0
    }
}

/// Whether BOTH runner inputs of the paired path are present — from the flags, else from the
/// runner environment. It is the switch a LOCAL mode takes to run the full paired path instead of
/// a candidate-only unscored run; the ranked path requires them either way and refuses by name.
fn paired_inputs_present(args: &IterateArgs) -> bool {
    let from_env = |name: &str| {
        std::env::var(name)
            .ok()
            .is_some_and(|v| !v.trim().is_empty())
    };
    (args.baseline_workspace.is_some() || from_env(baseline::BASELINE_WORKSPACE_ENV))
        && (args.baseline_calibration.is_some() || from_env(baseline::BASELINE_CALIBRATION_ENV))
}

/// The resident socket a SPAWN's Seatbelt profile must allow: the LEG's own, taken from that
/// spawn's env overrides, else the process environment's.
///
/// The two sources are the two topologies. A PAIRED run boots one resident per leg and hands each
/// spawn its leg's socket in the overrides — the process env has none, and must not be consulted,
/// because a stale one would name the wrong resident. The legacy SINGLE-resident path (a local
/// unscored run under the measure script's own wrapper) has no overrides and its socket is exactly
/// the process env's. Pure, so the selection is testable without a sandbox or a box.
fn spawn_resident_socket(
    leg_env: &[(String, String)],
    process_env: Option<&str>,
) -> Option<String> {
    let non_empty = |value: &str| (!value.trim().is_empty()).then(|| value.to_string());
    // A leg that supplied the name AT ALL has answered for itself, even with an empty value:
    // falling back to the process env there would hand that leg a DIFFERENT resident, which is
    // the whole failure this function exists to stop. An empty leg value means no socket.
    if let Some((_, value)) = leg_env
        .iter()
        .find(|(name, _)| name == legserve::BENCH_WORKER_RESIDENT_SOCKET_ENV)
    {
        return non_empty(value);
    }
    process_env.and_then(non_empty)
}

/// The `BENCH_WORKER_RESIDENT_SOCKET` this process inherited, if any.
fn process_resident_socket() -> Option<String> {
    std::env::var(legserve::BENCH_WORKER_RESIDENT_SOCKET_ENV).ok()
}

fn execute_iterate(args: &IterateArgs) -> Result<bool, String> {
    // The `--capture-baseline` ARMING GATE is NOT here. It used to be: `main` served ONE track,
    // so "is this track's pair captured?" was decidable from a compile-time table before a golden
    // load. This tree serves FOUR, each declaring its own pair, and the pair a capture run must
    // not overwrite is the one the run RESOLVES — its PLATFORM's, on the single-leg official path.
    // Keying it on `TRACK_ID` here would refuse every capture run on this tree, including the
    // tracks whose pair is genuinely pending. The gate therefore fires beside the capture branch
    // below, once the platform is resolved (`capture::refuse_unless_pending`), still before any
    // engine spawns and still before any record is written.

    // F1 — resolve the WORKSPACE HARNESS IDENTITY, fail-closed, before anything else happens.
    //
    // Every payload this run can produce seals `metrics.harness_hash` from this value (the single
    // `iterate::base_metrics` funnel), on ALL modes — official, official gates-only, local-iterate
    // and local-submit alike. Before F1 that field was a `String::new()` stub, and the seam-3
    // overlay correctly refuses to publish a gates-score with no harness identity, so no benchd
    // score could ever be published. The value is computed HERE, trusted-side, over the workspace
    // benchd drives — never read off the engine wire, because the worker is participant-built and
    // a wire-reported hash would be attacker-controlled.
    //
    // The workspace is the process CWD, which is exactly how the reference resolves its own roster
    // roots (the reference implementations only produce this hash when the benchmark process runs
    // with CWD == the repo/workspace root) and how this file already treats the CWD elsewhere
    // (`relativize_for_seal`: "the run's current working directory (its workspace root)").
    //
    // Resolved FIRST on purpose. The hash covers the roster roots that EXIST — the roster spans
    // both the MLX and CUDA engine surfaces, and a root absent on this engine is logged and skipped
    // (David's ruling), so this never refuses on a missing root. Resolving at the top still means a
    // run whose CWD cannot even be read produces NO artifacts rather than dying after minutes of
    // weights hashing and correctness work. There is no path from here that seals `""`.
    let harness = HarnessIdentity::resolve_from_current_dir().map_err(|e| {
        format!(
            "harness identity could not be resolved for this workspace ({e}); benchd iterate \
             must run with its working directory at the engine workspace root"
        )
    })?;

    // The TRACK keys both the platform (the official baseline; ONE bench tree serves both
    // engines) and the MODEL IDENTITY (the golden's model_type, vocabulary bound and seed
    // length). Both are resolved from the workflow-declared track id, the same env measure-job
    // seals, and BOTH are resolved HERE — above the golden load — because the golden cannot be
    // validated at all until the identity it is judged against is known. An undeclared track
    // refuses by name before any golden bytes are trusted.
    let track_id_env = env_track_id();
    let platform = iterate_platform(track_id_env.as_deref())?;
    // `iterate_platform` already refused an absent track id, so this names the resolved track.
    let track_id = track_id_env.clone().unwrap_or_default();

    // ONE READ of the `--contract` file, and every contract-derived decision of this run — the arm
    // gate, the speedup floors, the pair count, and every group of the constants→contract migration
    // — reads THOSE bytes. STEP 3 (David 2026-09-15) made the file REQUIRED on every mode: the
    // per-track tables are gone, so a run without one cannot resolve a window, a model shape, a
    // denominator, a band or a weight, and it refuses BY NAME at the first of them rather than
    // measuring under values nobody declared. `contract::load` also digests them, and the digest is sealed below as
    // `metrics.contract_sha256` (THE MIGRATION PIN, David 2026-09-15), so the artifact records
    // which fixture set the rules it was scored under. Loading here also certifies the fixture's
    // declared ranges BEFORE any worker spawns.
    //
    // It sits ABOVE the golden load because the run's WINDOW — the decode depth the golden's oracle
    // must be long enough for, and the checked-step arity the loader demands — is one of the values
    // the fixture may declare. A golden cannot be validated against a window the run has not
    // resolved yet.
    let loaded_contract = args.contract.as_deref().map(contract::load).transpose()?;
    // THE FIXTURE EVERY RESOLVER READS FIRST (STEP 1 of the constants→contract migration, David
    // 2026-09-15). A run given no `--contract` resolves against `Contract::NONE_DECLARED` — a
    // fixture that declares nothing — so "no fixture" and "a fixture silent about this group" are
    // ONE code path, and neither is a set of default values: STEP 3 made both REFUSE, by name.
    let declared_contract = contract::declared(loaded_contract.as_ref());
    // THE WINDOW this run measures and checks, resolved ONCE: the timed decode depth, the checked
    // step count and the unmeasured prefill passes. Every phase below is driven from this one
    // value rather than from the module constants, so a declared window cannot reach some call
    // sites and miss others.
    let window = contract::window_shape(declared_contract, &track_id)?.value;
    // …and the MODEL SHAPE the golden is judged against: the vocabulary bound, the layer count and
    // the seed length, from the fixture when it declares them (STEP 1, David 2026-09-15). It is
    // resolved BELOW the contract load for the same reason the window is — the fixture may declare
    // it — and ABOVE the golden load because it is a loader INPUT.
    let identity = iterate_model_identity(declared_contract, track_id_env.as_deref())?;

    // Integrity-pin (when given) + load + validate the golden. The pin is checked on the
    // raw bytes BEFORE the parse, so a golden that does not match the caller's pin is
    // refused before benchd ever trusts its contents.
    //
    // The arity is MODE-PARAMETRIC, exactly as the reference's own load call is: Swift
    // `QwenRuntime.localIterate` demands `benchmarkDecodeSteps + 1` expected tokens (seed +
    // one per decode step) at LOAD time, so a golden that cannot cover this mode's decode
    // window is refused BEFORE the correctness gate runs and before the golden is hashed —
    // `case_count = 0`, `checked_steps = 0`, `golden_hash = ""`. Passing the flat
    // `CORRECTNESS_STEPS` here made benchd accept goldens the reference rejects.
    //
    // #114 — reference-model pin `None`: `iterate` has NO `--contract` surface, and the ruling put
    // the reference-model identity in the track contract, so this command has no pin to apply.
    // That is a SCOPED residual, not a hidden one — recorded as the
    // contract-less half of the #114 row (the ranked path, `measure-job`, always carries one).
    //
    // EVERY golden is loaded here, each with its own pin, so a bad golden refuses before any GPU
    // work. Only the paired path measures more than the first one (refused by name below).
    let goldens = args
        .goldens
        .iter()
        .enumerate()
        .map(|(i, path)| {
            load_golden_checked(
                path,
                args.golden_pins.get(i),
                args.mode.golden_required_steps(&window),
                None,
                &track_id,
                &identity,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let golden = &goldens[0];

    // B-2: an OFFICIAL run FAILS CLOSED on any missing sandbox prerequisite (worker disabled,
    // MLXFAST_NO_SANDBOX=1, no engine exe, no derivable profile, no sandbox-exec) — resolved
    // UP FRONT, before any score is written (Swift `runtimeWorkerOptions` throws before the
    // worker spawns). Surfaced as a hard error (exit 1, no artifacts), verbatim the Swift
    // message. Local modes never resolve a sandbox.
    //
    // a8 ruling (b): the Seatbelt sandbox is a macOS facility (`/usr/bin/sandbox-exec`). On a
    // NON-macOS host (the linux-aarch64 CUDA box) it does not exist, so an official run there does
    // NOT resolve or require a sandbox — the worker is spawned unsandboxed and the seal records
    // `sandbox: none (linux)`. macOS keeps the fail-closed Seatbelt behavior BYTE-UNCHANGED
    // (including the `MLXFAST_NO_SANDBOX` refusal, which only reaches the resolver on macOS).
    let official_sandbox: Option<OfficialSandboxPlan> =
        if args.mode == Mode::Official && cfg!(target_os = "macos") {
            // The SHAPE of the official plan (is there one at all, and which executable does it
            // name) — resolved here, before any leg exists. The paired path re-resolves per spawn
            // with that leg's own resident socket; every other path spawns with this plan and the
            // process env's socket, which is the topology it runs under.
            Some(resolve_official_sandbox_from_env(
                &args.engine,
                &args.goldens[0],
                true,
                process_resident_socket().as_deref(),
            )?)
        } else {
            None
        };

    // Sandbox provenance sealed into the integrity sidecar (a8 ruling b): a resolved Seatbelt plan
    // → `seatbelt` (macOS official); an official run with no plan → the host has no Seatbelt
    // (non-macOS), sealed honestly as `none (linux)`; local modes are never sandboxed on any
    // platform → `none`.
    let sandbox_provenance =
        sandbox_provenance(args.mode == Mode::Official, official_sandbox.is_some());

    // Digest the weights directory. #64: this runs AFTER the cheap arg/preflight validation
    // above, because it is the expensive step — a full recursive walk that streams every
    // safetensors byte through sha256 (tens of GB). Everything before it can hard-reject the
    // run (bad/unpinned golden, missing sandbox prerequisite), and there is no reason to spend
    // minutes hashing weights for a run that was never going to start. It stays BELOW the
    // sandbox resolution (Swift resolves `runtimeWorkerOptions` first) and ABOVE the §F2
    // baseline resolution, which is the reference's own order — on doubly-broken input Swift
    // reports the weights-digest failure, not the baseline one
    // (QwenRuntimeBenchmark.swift:406→428, QwenRuntimeLocalIterate.swift:77→100 @242b19d).
    // It also stays BEFORE the payload branch below, which needs the digest for both the
    // preflight-failed score and the real run.
    // Option B digest-hoist (RIDER 1): in --capture-baseline mode the driver may pass the window's
    // once-computed digest via --weights-digest, and this run REUSES it instead of re-hashing the
    // immutable ~105 GB tree — byte-identical to the per-pass recompute, minutes saved per pass.
    // The flag is refused at parse outside --capture-baseline (which official refuses at parse), so
    // an official/scored seal NEVER reaches the reuse branch: it always falls through to dir_digest
    // and hashes for itself. #64's ordering (below the cheap rejects, below sandbox resolution,
    // above the baseline resolution) is preserved — only the SOURCE of the value changes.
    let weights_digest = resolve_weights_digest(args)
        .map_err(|e| format!("weights digest failed ({}): {e}", args.weights.display()))?;

    // The two digests benchd computes and seals for this run: WHICH WEIGHTS it measured and WHICH
    // HARNESS produced the result. Bundled once here so every payload builder below — passing,
    // failing, gates-only, preflight-refused — seals the same pair.
    let digests = RunDigests {
        weights: &weights_digest,
        harness: &harness,
        model: &identity,
    };

    // #123 — RULED (David 2026-08-20): pin the RUNNER, not just the inputs. Resolved here, before
    // anything spawns, so the sidecar names the binary that actually ran and an unreadable engine
    // refuses the run instead of surfacing after a score exists. On official the spawned binary is
    // the sandbox plan's, which `MLXFAST_RUNTIME_WORKER_EXECUTABLE` can redirect away from
    // `--engine`.
    let runner_engine = official_sandbox
        .as_ref()
        .map(|p| p.executable_path.clone())
        .unwrap_or_else(|| args.engine.clone());
    let runner = resolve_runner_identity(&runner_engine);

    // Gates-only OFFICIAL run (seam 1): MLXFAST_BENCHMARK_SKIP_TIMED=1 skips the timed phases and
    // runs ONLY the correctness gates, sealing a partial_result=true gates-score the paired
    // overlay later completes. This is benchd's parity of the reference mlxfast-swift SKIP_TIMED
    // path — env read at `main.swift@b26f76f:386` (`MLXFAST_BENCHMARK_SKIP_TIMED`, defaulting to
    // "1"), threaded into the options at `:397`, consumed at
    // `QwenRuntimeBenchmark.swift@b26f76f:457`. (#132/F-7: this used to cite `main.swift:321-322`,
    // which is the LOCAL branch's `QwenRuntime.localIterate` call — an unrelated code path.)
    // No timed phase runs here, so no paired baselines are resolved.

    // ARM GATE (David 2026-08-26) — the SOLE scored path inherits the gate the retired measure-job
    // used to carry: an OFFICIAL (scoring) run REFUSES, pre-GPU and BEFORE any score is sealed,
    // unless the --contract track fixture declares `official_scoring_enabled: true`. `false` and
    // ABSENT both refuse (an absent arm state is not an armed one). The local modes never reach
    // here as a scoring run, so they are untouched — the gate keys on `Mode::Official` exactly as
    // measure-job keyed on `!--local-dev`. --capture-baseline returns below without a score, but it
    // is local-iterate-only (refused on official at parse), so it never bypasses this gate. This is
    // the one call site that covers the whole scored chain now that flow B is gone.
    //
    // It reads the fixture LOADED ONCE at the top of this function, above the golden — the same
    // bytes every other contract-derived decision of this run reads, and the same bytes the digest
    // sealed as `metrics.contract_sha256` covers.
    if args.mode == Mode::Official {
        enforce_official_arm_gate(loaded_contract.as_ref(), track_id_env.as_deref())?;
    }
    // SPEEDUP FLOORS (David 2026-09-09: 0.95 decode AND 0.95 prefill, enforced, configurable per
    // project). The pinned track fixture is the ONLY source on the scored path — no flag, no
    // environment, no default — so a track cannot be scored against a floor it never declared. The
    // arm gate above already parsed the fixture for `--mode official`.
    //
    // STEP 3 (David 2026-09-15) — there is NO LOCAL DEFAULT any more. A local run used to fall back
    // to the `SCORE_*_SPEEDUP_FLOOR` constants and say so; deleting the per-track tables made
    // `--contract` structurally required on EVERY mode anyway (a local run must still resolve a
    // window and a model shape, and neither has a fallback left), so a floors-only default would be
    // unreachable code claiming a behaviour the binary no longer has. ONE value from here reaches
    // every gate and every seal of this run.
    let floors = contract::speedup_floors(declared_contract, &track_id)?;
    // THE COMPOSITE WEIGHTS (STEP 1 of the constants→contract migration, David 2026-09-15). Same
    // shape as the floors above: the pinned track fixture declares them
    // (`score_decode_weight` / `score_prefill_weight`) and every gate and every seal of this run
    // reads the ONE value resolved here. STEP 3: absent is a REFUSAL — the composite divides by
    // their sum, so a run weighted by a pair it was never given is a number attributed to a rule
    // nobody stated.
    let weights = contract::scoring_weights(declared_contract, &track_id)?.value;
    // THE ACCEPTANCE BAND SHAPE (STEP 3, David 2026-09-15). The shape used to be an in-tree
    // constant on the paired arm; the table is gone, so the track fixture declares its own six
    // band values and an absent declaration refuses BY NAME. Resolved here, beside the floors and
    // the weights, so one shape reaches the gate and the seal.
    let paired_bands = contract::acceptance_bands(declared_contract, &track_id)?.value;
    // THE RANKED PAIRED PATH's fences, PRE-GPU (David 2026-09-08). A live-control-leg track
    // measures its own denominator, so every STORED-pair door is closed before anything spawns:
    // the trusted env override, the `--baseline-*` flags, and a golden that still declares a pair.
    // Each refuses BY NAME. They are checked for the whole of `--mode official` — including the
    // gates-only seam, which seals no denominator of its own — so an operator who wires one of
    // them is told once, at the door, rather than after a GPU window.
    // THE PAIRED PATH is taken by a live-control-leg track when it CAN measure two legs: always
    // on the ranked path (`--mode official`, where the two runner inputs are REQUIRED and each
    // refuses by name when absent), and on a LOCAL mode when the box supplies both of them. A
    // participant on a laptop has no reference tree, so the local modes fall through to the
    // CANDIDATE-ONLY, UNSCORED run below (David: the local benchmark must keep working).
    let live_control_leg_track =
        contract::scores_against_live_control_leg(declared_contract, &track_id)?.value;
    let paired_inputs_present = paired_inputs_present(args);
    let paired_track =
        live_control_leg_track && (args.mode == Mode::Official || paired_inputs_present);
    // `--control-golden` is LEG 1's golden, so it has meaning only on a run that measures leg 1.
    // A run that measures none is refused BY NAME rather than dropping the flag: an operator who
    // wired the serial tape and got a candidate-only run must be told so, not left reading a
    // verdict the flag never reached.
    if let Some(control) = args.control_goldens.first() {
        if !paired_inputs_present {
            return Err(format!(
                "{}: --control-golden {} names the golden the serial-control leg verifies against, \
                 but this run measures no control leg — a reference workspace and a calibration \
                 file must both be in reach (--baseline-workspace/--baseline-calibration, or the \
                 runner environment) before leg 1 exists",
                baseline::CONTROL_GOLDEN_WITHOUT_PAIRED_PATH,
                control.display(),
            ));
        }
    }
    // Only the paired path cycles its pairs over several goldens. Every other path measures one
    // golden, so a second `--golden` there is refused BY NAME rather than ignored.
    if args.goldens.len() > 1 && !paired_track {
        return Err(format!(
            "{}: --golden was given {} times, and only the paired path measures more than one \
             golden; this run measures no control leg",
            baseline::MULTIPLE_GOLDENS_WITHOUT_PAIRED_PATH,
            args.goldens.len()
        ));
    }
    if live_control_leg_track && args.mode == Mode::Official {
        baseline::refuse_stored_baseline_override(
            std::env::var("MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN")
                .ok()
                .as_deref(),
            std::env::var("MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN")
                .ok()
                .as_deref(),
            args.baseline_prefill_spt.is_some() || args.baseline_decode_spt.is_some(),
        )?;
        for golden in &goldens {
            baseline::refuse_golden_with_stored_pair(golden)?;
        }
    }
    // --capture-baseline (David round-4, the capture-circularity fix): author the official
    // baseline's CAPTURE RECORD from a normal checked-timing run WITHOUT resolving the official
    // baseline — the pending state must not block the capture that ends it, and a CAPTURED state
    // refuses the mode by name (capture::refuse_unless_pending). This branch returns BEFORE the
    // score/integrity writers below: a capture run produces no artifact a scored run would.
    if let Some(capture_path) = args.capture_baseline.as_ref() {
        // A live-control-leg track stores no pair, so there is nothing to capture; every other
        // track arms the mode exactly while its own table row is absent.
        capture::refuse_live_control_leg_track(declared_contract, &track_id)?;
        capture::refuse_unless_pending(
            &track_id,
            contract::official_baseline(declared_contract, &track_id).is_ok(),
            bench_core::constants::OFFICIAL_BASELINE_PENDING,
        )?;
        capture::refuse_unresolved_engine(
            &runner_engine,
            &runner.candidate_executable_resolution,
            &runner.candidate_executable_sha256,
        )?;
        // The identity every pass of this capture stamps: one engine / golden / track / mode. A pass
        // whose engine or golden differs refuses to merge into the record (`capture::merge`).
        let identity = capture::CaptureIdentity {
            track_id: track_id_env
                .as_deref()
                .unwrap_or_default()
                .trim()
                .to_string(),
            mode: args.mode.mode_name().to_string(),
            // The window the pair was measured over, sealed as its own value: `mode` implies it
            // today, but the pair depends on the window, not on the name.
            decode_steps: args.mode.decode_steps(&window) as i64,
            engine_sha256: runner.candidate_executable_sha256.clone(),
            // The weights digest this run already computed above — two passes over different
            // checkpoints must never average into one pair.
            weights_sha256: weights_digest.sha256.clone(),
            golden_sha256: golden.sha256.clone(),
            // benchd does the parent-side timing, so the harness that measured is part of the
            // measurement and belongs in the identity.
            benchd_sha256: runner.benchd_executable_sha256.clone(),
        };
        // --capture-passes (a8 Option-A restructure): run EVERY listed pass over ONE persistent
        // model residency and append each pass's pair to its own per-label record. This is the ONLY
        // multi-pass branch; it returns before the single-pass path below. Byte-identical measured
        // number per pass — see `iterate::run_capture_passes_over_session`.
        if let Some(labels) = args.capture_passes.as_ref() {
            return run_capture_passes(
                args,
                golden,
                digests,
                &window,
                capture::CaptureDestination {
                    base: capture_path,
                    labels,
                    identity: &identity,
                },
            );
        }
        // The pair below is INERT and never leaves this scope: iterate_core needs a denominator
        // to assemble its in-memory payload, but the capture record reads ONLY the
        // parent-measured timing fields and the correctness verdict. (0.0, 0.0) additionally
        // makes the in-memory score invalid by construction, so even the discarded payload never
        // carries a plausible-looking score. Nothing derived from this pair is written anywhere.
        let payload = run_local_iterate(
            args,
            golden,
            digests,
            &window,
            ScoringInputs {
                baseline_prefill_spt: 0.0,
                baseline_decode_spt: 0.0,
                floors,
                weights,
            },
            args.capture_timed_only,
        )?;
        capture::refuse_unless_correctness_passed(&payload.metrics)?;
        // A TIMING-phase failure (cool-gate stall abort, token mismatch, worker spawn error)
        // passes the check above — the correctness gate had already passed — and arrives here
        // with a zeroed pair and the real cause in `metrics.error`. `capture_run_from` refuses it
        // BY NAME, so the operator reads the cause, and refuses a leg whose engine speculated on
        // its own; the merge's non-finite refusal stays as the backstop. The gates PRODUCE the
        // pair, so this recorder cannot run without them.
        let run = capture::capture_run_from(&payload.metrics)?;
        let record = capture::record_run(capture_path, identity, run)?;
        eprintln!(
            "benchd iterate: capture record {} now carries {} run(s) (prefill CV {:?}%, decode \
             CV {:?}%); no score was written",
            capture_path.display(),
            record.run_count,
            record.prefill_cv_percent,
            record.decode_cv_percent,
        );
        return Ok(true);
    }
    // WHERE THIS RUN'S PAIR COMES FROM, decided once. `RunBaselines` is `Copy`, so the arms
    // below read it without re-deciding — and without the decision drifting between them.
    let baseline_decision = run_baselines(
        declared_contract,
        args.mode,
        golden,
        args.baseline_prefill_spt.zip(args.baseline_decode_spt),
        track_id_env.as_deref(),
    )?;
    let payload = if args.mode == Mode::Official && official_gates_only_from_env() {
        // macOS resolves a Seatbelt plan; a non-macOS official run has none and spawns unsandboxed
        // (a8 ruling b, `spawn_official_worker`).
        let plan = official_sandbox.as_ref();
        let weights_str = args.weights.to_string_lossy().to_string();
        let commit_env = std::env::var("MLXFAST_COMMIT_SHA").ok();
        let commit = official::commit_identifier(commit_env.as_deref());
        // One official worker spawn shape (carries `--speculative-protocol v1.1`): identical to the
        // full-run timed/correctness spawns below, so the gated worker is spawned the same way here.
        let spawn_correctness = || -> bench_runner::Result<Session<ChildStdioTransport>> {
            connect_official_worker(
                plan,
                &args.engine,
                &weights_str,
                &free_run_spawn_args(&args.engine_resources),
                &[],
            )
        };
        // The gates-only partial seals the same resolved pair the full run would: the trusted env
        // override, else the golden's declared pair, else the official constants — which refuse
        // by name while this track's capture is pending.
        let flag_override = args.baseline_prefill_spt.zip(args.baseline_decode_spt);
        let effective_override = official::paired_baseline_override()?.or(flag_override);
        let (baseline_prefill, baseline_decode) = gates_only_baselines(
            declared_contract,
            track_id_env.as_deref(),
            effective_override,
            golden,
        )?;
        official::official_gates_only(
            golden,
            ScoringInputs {
                baseline_prefill_spt: baseline_prefill,
                baseline_decode_spt: baseline_decode,
                floors,
                weights,
            },
            digests,
            &commit,
            spawn_correctness,
        )?
    } else if paired_track {
        // THE RANKED PAIRED PATH (David 2026-09-08). This track stores no baseline pair anywhere,
        // so the denominator is MEASURED here: a SERIAL-CONTROL leg on the organizer-staged
        // reference tree, on this box, in this job, immediately before the candidate leg. The
        // per-box calibration file is a HEALTH BAND for that control leg and never a denominator.
        //
        // The two runner inputs are REQUIRED and each refuses BY NAME when it is absent or does
        // not match this track and this box (`baseline.rs`). The stored-pair doors — the
        // `MLXFAST_PAIRED_BASELINE_*` env, the `--baseline-*` flags, and a golden carrying
        // `benchmark.baseline_*_seconds_per_token` — were already refused pre-GPU above.
        let workspace = baseline::resolve_workspace(
            args.baseline_workspace.as_deref(),
            std::env::var(baseline::BASELINE_WORKSPACE_ENV)
                .ok()
                .as_deref(),
        )?;
        let calibration = baseline::load_calibration(
            args.baseline_calibration.as_deref(),
            std::env::var(baseline::BASELINE_CALIBRATION_ENV)
                .ok()
                .as_deref(),
        )?;
        let box_name = baseline::resolve_box_name(
            args.box_name.as_deref(),
            std::env::var(baseline::RUNNER_NAME_ENV).ok().as_deref(),
        )?;
        // PAIR COUNT (David 2026-09-09): the pinned track fixture is the only source on the ranked
        // path. The arm gate above already parsed it for `--mode official`. A LOCAL paired run
        // (both per-box inputs present, no scoring seal) takes the count from --contract when one
        // is given and otherwise runs one pair: local numbers are directional, and the ranked
        // count is pinned where the ranked run reads it.
        let pairs = match loaded_contract.as_ref() {
            Some(loaded) => contract::official_pairs(&loaded.contract, &track_id)?,
            None => {
                eprintln!(
                    "benchd iterate: local paired run without --contract runs 1 pair (the ranked \
                     path takes official_pairs from the track fixture)"
                );
                1
            }
        };
        // Pair `k` measures golden `(k - 1) mod N`, so every golden gets the same number of pairs
        // only when the count is a multiple of N.
        baseline::check_pairs_cover_goldens(pairs, goldens.len())?;
        // The prompt each golden MEASURES, named from the golden's path — the same name the
        // calibrator recorded from the golden it measured. The calibration must hold an entry for
        // each one.
        let prompts = args
            .goldens
            .iter()
            .map(|path| {
                let prompt = baseline::golden_prompt_name(path).ok_or_else(|| {
                    format!(
                        "{}: --golden {} has no file name to take a prompt name from, so the \
                         calibration's own prompt cannot be checked against it",
                        baseline::BASELINE_CALIBRATION_PROMPT_MISMATCH,
                        path.display()
                    )
                })?;
                calibration
                    .calibration
                    .check_identity(&track_id, &box_name, &prompt)?;
                Ok(prompt)
            })
            .collect::<Result<Vec<_>, String>>()?;

        // LEG 1's OWN GOLDENS. The control leg is SERIAL, so it must verify its decode tokens
        // against the SERIAL tape. On a track whose timed oracle at the declared depth is
        // byte-identical to the serial tape that IS `--golden`; on a track that ships PER-DEPTH
        // oracle tapes the two disagree (on MLX from step 1 on), and leg 1 dies against the
        // candidate's tape. `--control-golden` `i` is golden `i`'s serial live golden, loaded with
        // the same arity, the same track and the same model identity the candidate golden is
        // loaded with, and pinned by its own two flags.
        //
        // BOTH LEGS OF A PAIR MEASURE ONE PROMPT. A control golden naming another prompt would
        // make the ratio a comparison of two different prompts, so it is refused BY NAME.
        let mut control_goldens = Vec::with_capacity(args.control_goldens.len());
        for (i, path) in args.control_goldens.iter().enumerate() {
            let golden_path = &args.goldens[i];
            let prompt = &prompts[i];
            let control_prompt = baseline::golden_prompt_name(path).ok_or_else(|| {
                format!(
                    "{}: --control-golden {} has no file name to take a prompt name from, so it \
                     cannot be checked against --golden {}",
                    baseline::CONTROL_GOLDEN_PROMPT_MISMATCH,
                    path.display(),
                    golden_path.display()
                )
            })?;
            if &control_prompt != prompt {
                return Err(format!(
                    "{}: --control-golden {} measures prompt {control_prompt:?} but --golden {} \
                     measures prompt {prompt:?}; the two legs of a pair measure ONE prompt, so \
                     their ratio would compare two different prompts",
                    baseline::CONTROL_GOLDEN_PROMPT_MISMATCH,
                    path.display(),
                    golden_path.display()
                ));
            }
            control_goldens.push(load_golden_checked(
                path,
                args.control_golden_pins.get(i),
                args.mode.golden_required_steps(&window),
                None,
                &track_id,
                &identity,
            )?);
        }
        // ABSENT ⇒ leg 1 verifies against the candidate golden, which is today's behaviour and is
        // correct exactly when that golden's oracle is the serial tape.
        let paired_goldens: Vec<official::PairedGoldens<'_>> = goldens
            .iter()
            .enumerate()
            .map(|(i, candidate)| official::PairedGoldens {
                candidate,
                control: control_goldens.get(i).unwrap_or(candidate),
                prompt: &prompts[i],
            })
            .collect();
        // The golden PATHS the current pair's two legs read, for the per-spawn sandbox plan. The
        // leg openers set the index at the start of each leg.
        let pair_golden = std::cell::Cell::new(0usize);
        let control_golden_path =
            |i: usize| args.control_goldens.get(i).unwrap_or(&args.goldens[i]);

        // THE TWO ROOTS. The reference leg runs the SAME root-relative engine path inside the
        // organizer's tree that the candidate leg runs inside the submission tree, so the two legs
        // differ by their tree and by nothing else. The re-rooting starts from the RESOLVED
        // candidate executable, which is what the candidate leg will actually spawn.
        let workspace_root = std::env::current_dir()
            .map_err(|e| format!("the run's workspace root could not be resolved: {e}"))?;
        let reference_engine =
            baseline::reference_engine_path(&runner_engine, &workspace_root, &workspace)?;
        let reference_engine_str = reference_engine.to_string_lossy().to_string();
        // …and the same for the WEIGHTS. The transform that produces them is
        // PARTICIPANT-EDITABLE, so the control leg loads the REFERENCE tree's own transform
        // output, never the candidate's (`baseline::reference_weights_path` carries the three
        // rules and the one case where both legs legitimately share an organizer-staged tree).
        let reference_weights =
            baseline::reference_weights_path(&args.weights, &workspace_root, &workspace)?;
        let reference_weights_str = reference_weights.path().to_string_lossy().to_string();
        // The two legs are wrapped IDENTICALLY: a leg resolves a Seatbelt plan exactly when the
        // official plan resolved one. A run that sandboxed one leg and not the other would compare
        // two differently-wrapped processes — and a LOCAL paired run, which resolves no official
        // plan at all, would otherwise sandbox only the control leg.
        let sandboxed = official_sandbox.is_some();
        let weights_str = args.weights.to_string_lossy().to_string();
        let commit_env = std::env::var("MLXFAST_COMMIT_SHA").ok();
        let commit = official::commit_identifier(commit_env.as_deref());

        // PER-LEG RESIDENT ENGINES, on BOTH platforms. The model is owned by a RESIDENT process
        // on either engine — a `ds4-resident` on CUDA, a `bench-worker resident` holding the whole
        // checkpoint on MLX — and that process belongs to ONE leg's tree. So benchd boots that
        // leg's resident from that leg's OWN tree, one at a time, through the fixed
        // `--boot`/`--stop` convention (`legserve.rs`), and puts the socket into THAT leg's worker
        // spawns only. benchd's own environment is never mutated, so the two legs cannot bleed
        // into each other.
        //
        // An INHERITED socket is refused here: the measure scripts used to wrap the whole benchd
        // invocation in their own resident wrapper, which boots ONE resident for the window. On
        // MLX that reached the worker as a refusal — the resident held the candidate tree's
        // weights and the reference leg asked for the reference tree's (ranked run 34230122059).
        legserve::refuse_inherited_socket(
            platform,
            std::env::var(legserve::DS4_RESIDENT_SOCKET_ENV)
                .ok()
                .as_deref(),
            std::env::var(legserve::BENCH_WORKER_RESIDENT_SOCKET_ENV)
                .ok()
                .as_deref(),
        )?;
        let leg_env: std::cell::RefCell<Vec<(String, String)>> =
            std::cell::RefCell::new(Vec::new());
        let candidate_spec = args.spec.clone();
        // THE PLAN IS PER SPAWN, because the SOCKET is per leg. Seatbelt classifies an AF_UNIX
        // connect as network, and the profile's one `network-outbound` allowance names ONE
        // path-literal — so it has to be built from the socket THIS leg's worker will connect to,
        // which exists only in that leg's spawn-env overrides. Resolving it once up front from
        // this process's environment left a per-leg worker with no allowance at all
        // (`cannot reach the resident … Operation not permitted`, box-4 calibration on #296).
        //
        // Each leg therefore gets an allowance for ITS OWN socket and nothing else: the reference
        // leg cannot connect to the candidate's resident, and the candidate cannot connect to the
        // reference's. The rest of the profile is unchanged, and its `(allow default)` base means
        // the reference leg reads the reference tree's own weights and metallib without any
        // checkout-rooted allowance — the deny rules name only the private golden and the private
        // dir.
        //
        // The GOLDEN the profile denies reads of is likewise per spawn: leg 1 reads the SERIAL
        // golden and leg 2 the candidate's, and a leg must be denied the file it is verified
        // against, not the other leg's.
        let leg_sandbox_plan = |engine: &str,
                                golden: &Path,
                                honor_executable_override: bool|
         -> bench_runner::Result<Option<OfficialSandboxPlan>> {
            if !sandboxed {
                return Ok(None);
            }
            let socket = spawn_resident_socket(&leg_env.borrow(), None);
            resolve_official_sandbox_from_env(
                engine,
                golden,
                honor_executable_override,
                socket.as_deref(),
            )
            .map(Some)
            .map_err(bench_runner::RunnerError::Protocol)
        };
        let open_baseline_leg = |golden_index: usize| -> Result<legserve::LegServe, String> {
            pair_golden.set(golden_index);
            // ALWAYS SERIAL, whatever the submission declares: this is the control.
            let serve = legserve::boot_leg(&workspace, None, "serial-control", platform)?;
            *leg_env.borrow_mut() = serve.spawn_env();
            Ok(serve)
        };
        let open_candidate_leg = |golden_index: usize| -> Result<legserve::LegServe, String> {
            pair_golden.set(golden_index);
            let serve = legserve::boot_leg(
                &workspace_root,
                candidate_spec.as_ref(),
                "candidate",
                platform,
            )?;
            *leg_env.borrow_mut() = serve.spawn_env();
            Ok(serve)
        };

        eprintln!(
            "benchd iterate: the serial-control leg loads {} ({}); the candidate leg loads {}",
            reference_weights_str,
            match reference_weights {
                baseline::ReferenceWeights::ReferenceTree(_) => "the reference tree's own weights",
                baseline::ReferenceWeights::SharedOutOfTree(_) =>
                    "an organizer-staged tree outside every checkout, shared by both legs",
            },
            weights_str,
        );

        // Leg 1's worker, rooted at the REFERENCE tree. Its hello is retained SEPARATELY from the
        // candidate's: the two legs are two different engine builds, so one shared retention slot
        // would refuse the run as a mid-window resident change. The identity the score seals is
        // the CANDIDATE's — the leg that is scored.
        let baseline_hello = std::cell::RefCell::new(None);
        let spawn_baseline = || -> bench_runner::Result<Session<ChildStdioTransport>> {
            let plan = leg_sandbox_plan(
                &reference_engine_str,
                control_golden_path(pair_golden.get()),
                false,
            )?;
            let transport = spawn_official_worker(
                plan.as_ref(),
                &reference_engine_str,
                &reference_weights_str,
                &free_run_spawn_args(&args.engine_resources),
                &leg_env.borrow(),
            )?;
            connect_retaining_hello(transport, &baseline_hello)
        };
        // Leg 2's workers, rooted at the SUBMISSION tree — byte-for-byte the single-leg path's.
        let timed_hello = std::cell::RefCell::new(None);
        let spawn_timed = || -> bench_runner::Result<Session<ChildStdioTransport>> {
            let plan = leg_sandbox_plan(&args.engine, &args.goldens[pair_golden.get()], true)?;
            let transport = spawn_official_worker(
                plan.as_ref(),
                &args.engine,
                &weights_str,
                &free_run_spawn_args(&args.engine_resources),
                &leg_env.borrow(),
            )?;
            connect_retaining_hello(transport, &timed_hello)
        };
        // THE GATE LOG of this run (David 2026-09-17): the closure records every gate point on it,
        // the paired core names the pair and the leg of each point, and the payload seals the whole
        // array below — including on a run that stopped part-way.
        let gate_log = std::rc::Rc::new(quiescegate::GateLog::new());
        let cool_gate_fn = official_cool_gate(
            args.cool_gate
                .unwrap_or_else(|| args.mode.cool_gate_on_by_default()),
            std::rc::Rc::clone(&gate_log),
        );
        eprintln!(
            "benchd iterate: paired official run on box {box_name:?}, {pairs} pair(s) over {} \
             golden(s) — in each pair leg 1 is the serial-control leg on the reference tree {}, \
             leg 2 the candidate, both on the pair's golden; the score is the live ratio of the \
             two legs of the lower-median pair (calibration {} is the band on every leg 1, never a \
             denominator)",
            goldens.len(),
            workspace.display(),
            calibration.path.display(),
        );
        let mut payload = official::official_core_paired(
            &paired_goldens,
            &calibration.calibration,
            official::PairedBaselineSeal {
                box_name: &box_name,
                calibration_sha256: &calibration.sha256,
                reference_commit: &calibration.calibration.reference_commit,
                // Both are filled in by the paired core from what it measured; the caller states
                // nothing about a leg that has not run.
                band_passed: false,
                leg: None,
            },
            digests,
            &commit,
            official::PairedLegs {
                open_baseline_leg,
                open_candidate_leg,
                spawn_baseline,
                spawn_timed,
            },
            official::PairedWindow {
                // THE BAND SHAPE COMES FROM THE TRACK FIXTURE, resolved once above as
                // `paired_bands` (`contract::acceptance_bands`). There is no in-tree shape left to
                // fall back to — STEP 3 deleted the tables, so a fixture that declares none of the
                // six values refuses BY NAME. What the bands are applied TO is this arm's own
                // difference: a live control-leg measurement, not a stored pair.
                window,
                bands: paired_bands,
                // A per-leg resident engine accepts ONE connection, so every phase of a leg runs
                // over ONE attached worker: the load-once window, on both platforms.
                spec: args.spec.clone(),
                cool_gate: cool_gate_fn,
                gate_log: std::rc::Rc::clone(&gate_log),
                pairs,
                floors,
                weights,
                // The fixture is the only source. Absent keeps the exact rule.
                token_tolerance_per_thousand: declared_contract.timed_token_tolerance_per_thousand,
                near_tie_relative_gap: declared_contract.timed_token_near_tie_relative_gap,
            },
        );
        if let Some(hello) = timed_hello.borrow().as_ref() {
            official::seal_engine_identity(&mut payload.metrics, hello);
        }
        // `metrics.gates` is sealed by the paired core itself, on every payload it can return.
        payload
    } else if let RunBaselines::Decided {
        prefill,
        decode,
        flags_ignored,
    } = baseline_decision
    {
        eprintln!(
            "benchd iterate: {} baseline prefill_seconds_per_token={prefill} \
             decode_seconds_per_token={decode} (official-runner constants; local \
             speedups are directional)",
            args.mode.mode_name()
        );
        if flags_ignored {
            eprintln!(
                "benchd iterate: --baseline-prefill-spt/--baseline-decode-spt are IGNORED on \
                 {} (#127: local baselines come from the constants, as the reference's \
                 localIterate does); the run scores against the constants above",
                args.mode.mode_name()
            );
        }
        // The non-capture local path is never timed-only: it always runs the correctness gate.
        let mut payload = run_local_iterate(
            args,
            golden,
            digests,
            &window,
            ScoringInputs {
                baseline_prefill_spt: prefill,
                baseline_decode_spt: decode,
                floors,
                weights,
            },
            false,
        )?;
        explain_local_exact_token_rule(
            &mut payload,
            declared_contract.timed_token_tolerance_per_thousand,
        );
        payload
    } else if baseline_decision == RunBaselines::Unscored {
        // THE UNSCORED LOCAL RUN (David 2026-09-08). This track measures its denominator on the
        // ranked box against the organizer's reference tree, and this box has no reference tree,
        // so there is nothing to divide by. The run is the CANDIDATE LEG, in full: the real timed
        // prefill and decode, the real correctness gate, the real sealed timing surface — and NO
        // score. A participant iterating on a laptop gets a working benchmark; nobody gets a
        // number that looks like a rank.
        //
        // The `(0.0, 0.0)` pair IS the "no denominator" statement, and `seal_local_unscored`
        // finishes it: `score` stays null, `baseline_source` says why, and the placeholder
        // "score is not finite" text is cleared for a run whose correctness passed.
        let mut payload = run_local_iterate(
            args,
            golden,
            digests,
            &window,
            ScoringInputs {
                baseline_prefill_spt: 0.0,
                baseline_decode_spt: 0.0,
                floors,
                weights,
            },
            false,
        )?;
        explain_local_exact_token_rule(
            &mut payload,
            declared_contract.timed_token_tolerance_per_thousand,
        );
        iterate::seal_local_unscored(&mut payload);
        let m = &payload.metrics;
        eprintln!(
            "benchd iterate: {} UNSCORED on track {track_id} — it scores against a serial-control \
             leg on the organizer's reference tree, and none is named here (set \
             {} and {} to run the paired path locally). Candidate leg: prefill {:.1} tok/s, \
             decode {:.1} tok/s; correctness {}. No score was written.",
            args.mode.mode_name(),
            baseline::BASELINE_WORKSPACE_ENV,
            baseline::BASELINE_CALIBRATION_ENV,
            tokens_per_second(m.prefill_seconds_per_token),
            tokens_per_second(m.decode_seconds_per_token),
            if m.passed_correctness {
                "passed"
            } else {
                "FAILED"
            },
        );
        payload
    } else {
        // §F2 (OFFICIAL only): resolve the REQUIRED paired baselines. Explicit --baseline flags
        // override; else the golden's benchmark must carry both. Missing → a preflight-failed
        // score, NO engine run (Swift throws during validation, before spawning the worker).
        let flag_override = args.baseline_prefill_spt.zip(args.baseline_decode_spt);
        // #61: the trusted `MLXFAST_PAIRED_BASELINE_*` env override takes precedence over the
        // flags (the measure-job contract: the reference is measured on the same runner
        // immediately before the candidate). Resolved FAIL-CLOSED on a half-set/invalid pair — a
        // hard error (exit 1, NO artifacts), verbatim the Swift message
        // (BenchmarkSupport.swift PairedBaselineOverride.fromEnvironment). Local modes never read
        // it, and since #127 they never reach this branch at all. (The old two-leg `--paired`
        // monolith that bypassed this is REMOVED — it is now the standalone `benchd measure-job`
        // subcommand, seam 2.)
        let effective_override = official::paired_baseline_override()?.or(flag_override);
        // A STORED-PAIR track's official run needs its captured pair: the acceptance BANDS gate
        // the timed run, and the #74 early-refuse record carries the pair. A track with no row
        // refuses by name. (A live-control-leg track never reaches here — it took the paired arm
        // above, which measures its own denominator and reads no table.)
        // CONTRACT-FIRST (STEP 1, David 2026-09-15): the captured pair AND its band shape from the
        // fixture when it declares them, from the per-track table when it does not — which still
        // refuses BY NAME while the capture is pending.
        let official = contract::official_baseline(declared_contract, &track_id)?.value;
        match resolve_paired_baselines(effective_override, golden) {
            None => iterate::preflight_failed_payload(
                args.mode,
                golden,
                digests,
                iterate::missing_paired_baselines_error(args.mode),
                ScoringInputs {
                    baseline_prefill_spt: official.prefill_seconds_per_token,
                    baseline_decode_spt: official.decode_seconds_per_token,
                    floors,
                    weights,
                },
            ),
            // B-2 OFFICIAL: timed-first, three fresh SANDBOXED workers, full correctness set,
            // benchmark-oracle checks, official floor/band/finite gating, and a stamped commit.
            // This is now the ONLY resolved arm — the local modes branch off above (#127).
            Some((baseline_prefill, baseline_decode)) => {
                // macOS resolves a Seatbelt plan (three fresh SANDBOXED workers); a non-macOS
                // official run has no plan and spawns the workers UNSANDBOXED (a8 ruling b,
                // `spawn_official_worker`) — same env sanitization + stderr suppression, no
                // `sandbox-exec` wrapper the platform cannot provide.
                let plan = official_sandbox.as_ref();
                let weights_str = args.weights.to_string_lossy().to_string();
                // metrics.commit: valid-hex MLXFAST_COMMIT_SHA, else `git rev-parse --short HEAD`.
                let commit_env = std::env::var("MLXFAST_COMMIT_SHA").ok();
                let commit = official::commit_identifier(commit_env.as_deref());
                // The window opens ONE worker (macOS: under `sandbox-exec -f <profile>`), worker
                // stderr forwarding forced OFF (redacted + retained, never echoed).
                // `--speculative-protocol v1.1` on every official spawn: the timed decode leg drives
                // the free-run verbs and the timed session is also the correctness session, so the
                // spawner advertises `free_run_decode` (see `free_run_spawn_args`).
                // THE TIMED WORKER'S HELLO IS RETAINED, not discarded. It is the only place the
                // engine states its own identity — backend build string, device, protocol version,
                // loaded-head digest — and the scored leg runs on THIS worker, so this is the hello
                // the score seals (`seal_engine_identity` below).
                let timed_hello = std::cell::RefCell::new(None);
                let spawn_timed = || -> bench_runner::Result<Session<ChildStdioTransport>> {
                    let transport = spawn_official_worker(
                        plan,
                        &args.engine,
                        &weights_str,
                        &free_run_spawn_args(&args.engine_resources),
                        &[],
                    )?;
                    // Every timed phase of one window must report the SAME resident identity; a
                    // change is a mid-window reload, refused by name
                    // (`resident_identity_changed_within_window`).
                    connect_retaining_hello(transport, &timed_hello)
                };
                // The ranked path's per-phase cool gate (David 2026-09-06): ON by default for
                // Mode::Official, `--no-cool-gate` turns it off for a local dry run. Official
                // fails CLOSED without a temperature reader: a ranked run that silently skipped
                // the gate would seal a number the contract does not cover.
                // The SINGLE-LEG official arm measures one leg, so its gate points are pair 1 of
                // the candidate leg. It seals the same `gates` array the paired arm does.
                let gate_log = std::rc::Rc::new(quiescegate::GateLog::new());
                gate_log.enter_leg(1, quiescegate::LEG_CANDIDATE);
                let cool_gate_fn = official_cool_gate(
                    args.cool_gate
                        .unwrap_or_else(|| args.mode.cool_gate_on_by_default()),
                    std::rc::Rc::clone(&gate_log),
                );
                let mut payload = official::official_core_windowed(
                    official::OfficialParams {
                        golden,
                        scoring: ScoringInputs {
                            baseline_prefill_spt: baseline_prefill,
                            baseline_decode_spt: baseline_decode,
                            floors,
                            weights,
                        },
                        digests,
                        commit: &commit,
                        spec: args.spec.clone(),
                        window: &window,
                    },
                    official.bands,
                    spawn_timed,
                    cool_gate_fn,
                );
                if let Some(hello) = timed_hello.borrow().as_ref() {
                    official::seal_engine_identity(&mut payload.metrics, hello);
                }
                payload.metrics.gates = gate_log.records();
                payload
            }
        }
    };

    // THE MIGRATION PIN (David 2026-09-15) — seal the sha256 of the `--contract` bytes this run
    // loaded, alongside `metrics.golden_hash`. The fixture is a scoring input (arm state, both
    // speedup floors, the pair count), so the artifact must name the bytes those verdicts came
    // from. Stamped on every arm of the payload match above, from the ONE load at the top of the
    // run, so a seal can never quote a fixture the run did not read. Absent when no fixture was
    // given, leaving the local modes' sealed bytes unchanged.
    let mut payload = payload;
    payload.metrics.contract_sha256 = loaded_contract.as_ref().map(|l| l.sha256.clone());
    // WHICH SOURCE DECIDED EACH GROUP (STEP 1, David 2026-09-15). The digest above names the bytes;
    // this names which groups of the scored regime those bytes actually decided and which still
    // came from the in-tree per-track table. A sealed score therefore says what regime it was
    // scored under, not merely which fixture was present. Every resolver is a pure function of
    // `(contract, track_id)`, so the one resolution recorded here is the source every call site of
    // that group in this run computed.
    payload.metrics.contract_sources = Some(contract::sources(declared_contract, &track_id));

    // Write the sealed score + sha256 sidecar (benchd is the sole writer).
    let json = payload
        .to_sealed_json()
        .map_err(|e| format!("score serialization failed: {e}"))?;
    let score_sha256 = write_score(&args.score_path, &json)?;

    // Emit the sealed payload to STDOUT (Swift binary `emitScorePayloadToStdout`; benchmark.sh
    // `cat "${SCORE_PATH}"`) — no trailing newline, matching the on-disk bytes.
    print!("{json}");

    // Integrity sidecar (benchmark.sh benchmark-integrity.*.json). golden_sha256 =
    // sha256 of the raw golden bytes (== `shasum -a 256 GOLDEN`); transform_source_sha256 =
    // the `<weights>/.benchmark-source.sha256` marker content, or "" if the marker is absent
    // (benchd cannot recompute the Swift source hash without the source tree).
    //
    // The digest is the one `load_golden_checked` already recorded from the bytes it VALIDATED
    // and PARSED, not a second `fs::read` of the same path. Two reads are two different files
    // whenever the golden is swapped between them — the sidecar would then pin an identity the
    // run never measured against — which is the same TOCTOU `load_golden_bytes_checked` exists
    // to close upstream (#112/L2). Byte-identical value: `GoldenFixture::sha256` IS
    // `sha256_hex` of the raw file bytes.
    let golden_sha256 = golden.sha256.clone();
    let transform_source_sha256 =
        std::fs::read_to_string(args.weights.join(".benchmark-source.sha256"))
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
    write_integrity_sidecar(
        &args.score_path,
        args.mode,
        &IntegritySidecar {
            // F-5 — relativise every path at seal so the sidecar carries no operator home directory
            // (the same helper the measure-job anchor uses, so the two records agree).
            score_path: relativize_for_seal(&args.score_path),
            score_sha256,
            weights_path: relativize_for_seal(&args.weights),
            weights_sha256: weights_digest.sha256.clone(),
            weights_file_count: weights_digest.file_count,
            weights_byte_count: weights_digest.byte_count,
            golden_path: "[private]".to_string(),
            golden_sha256,
            transform_source_sha256,
            candidate_executable: relativize_for_seal(Path::new(&runner.candidate_executable)),
            candidate_executable_sha256: runner.candidate_executable_sha256,
            baseline_executable: String::new(),
            baseline_executable_sha256: String::new(),
            candidate_executable_resolution: runner.candidate_executable_resolution,
            benchd_executable: relativize_for_seal(Path::new(&runner.benchd_executable)),
            benchd_executable_sha256: runner.benchd_executable_sha256,
            candidate_workspace_sha256: String::new(),
            sandbox: sandbox_provenance.to_string(),
        },
    )?;

    eprintln!(
        "benchd iterate: wrote {} (passed={}, score={})",
        args.score_path.display(),
        payload.passed,
        payload
            .score
            .map(|s| s.to_string())
            .unwrap_or_else(|| "null".to_string())
    );
    Ok(payload.passed)
}

/// A local run with no reference tree has no reference engine to replay a different token on, so it
/// judges every timed decode token exactly. When the track declares a timed token tolerance, a
/// timed token refusal says so, so the participant does not read it as the ranked rule.
fn explain_local_exact_token_rule(payload: &mut ScorePayload, tolerance: Option<u32>) {
    let Some(n) = tolerance else {
        return;
    };
    let error = &payload.metrics.error;
    if error.contains("decode token mismatch") || error.contains("decode seed token mismatch") {
        payload.metrics.error = format!(
            "{error}; the ranked run tolerates up to {n} per thousand different timed tokens, but \
             a local run without a reference tree cannot apply that rule, so it judges every \
             token exactly"
        );
    }
}

/// Where THIS run's baseline pair comes from — the #127 decision seam, in one place both the
/// runner and its test go through.
#[derive(Debug, Clone, Copy, PartialEq)]
enum RunBaselines {
    /// The LOCAL legs: the pair is already decided, and it is the compile-time official-runner
    /// constants. Nothing downstream may reconsider it.
    Decided {
        prefill: f64,
        decode: f64,
        /// The caller passed `--baseline-*` and they are being ignored — worth saying out loud.
        flags_ignored: bool,
    },
    /// A LOCAL leg of a LIVE-CONTROL-LEG track with no reference tree in reach: there is no
    /// denominator, by design, so the run measures the CANDIDATE LEG ONLY and seals NO score.
    /// David requires the local benchmark to keep working for a participant on a laptop, and a
    /// score with no denominator is not a score — so the run is UNSCORED rather than refused.
    Unscored,
    /// OFFICIAL: golden-authoritative, resolved downstream from the trusted env/flag override
    /// ahead of the golden's own declaration.
    ResolveFromOverrideOrGolden,
}

/// The LOCAL modes' per-phase gates (`local-iterate`, `local-submit`, and the capture passes):
/// QUIESCENCE first, then COOL (David 2026-09-17). Disabled → a no-op for both, so the gate
/// machinery stays wired for the facade, which always passes `--cool-gate`. The one switch turns
/// both gates on or off; there is no second knob.
///
/// The gate temperature is keyed by the platform resolved from the track-id env, defaulting to
/// the Mac/MLX 40 C gate (R21 lift, David 2026-08-30). The quiescence thresholds are fixed
/// ([`quiescegate::LOAD_MAX`] / [`quiescegate::GPU_UTIL_MAX`]).
fn local_cool_gate(enabled: bool) -> impl Fn(&str) -> Result<(), RunnerError> {
    move |phase: &str| {
        if enabled {
            quiescegate::timed_phase_gates(phase, cool_gate_platform_from_env())
        } else {
            Ok(())
        }
    }
}

/// The RANKED path's per-phase gates: QUIESCENCE first, then COOL (David 2026-09-06, extended by
/// the 2026-09-17 quiescence ruling). Official fails CLOSED without a reader for EITHER gate: a
/// ranked run that silently skipped a gate would seal a number the contract does not cover — a
/// measurement on a busy box is as far outside the contract as one on a hot GPU.
/// `--no-cool-gate` turns both off for a local dry run.
fn official_cool_gate(
    enabled: bool,
    gate_log: std::rc::Rc<quiescegate::GateLog>,
) -> impl Fn(&str) -> bench_runner::Result<()> {
    move |phase: &str| {
        if !enabled {
            return Ok(());
        }
        // BOTH gates, quiescence first, and official fails CLOSED on either skip.
        let (quiescence, cool) = quiescegate::run_timed_phase_gates(
            || {
                match quiescegate::quiesce_gate_report(phase)? {
                r if r.state == quiescegate::QuiesceState::SkippedNoReader => {
                    Err(bench_runner::RunnerError::GateRejected {
                        phase: phase.to_string(),
                        reason: "official mode requires a load and GPU utilization reader for the quiescence gate (install macmon or set MLXFAST_MACMON_BIN)".to_string(),
                    })
                }
                r => Ok(r),
            }
            },
            || {
                let cool = coolgate::cool_gate_report(phase, cool_gate_platform_from_env())?;
                if cool.state == coolgate::GateState::SkippedNoReader {
                    return Err(bench_runner::RunnerError::GateRejected {
                        phase: phase.to_string(),
                        reason: "official mode requires a GPU temperature reader for the cool gate (install macmon or set MLXFAST_MACMON_BIN)".to_string(),
                    });
                }
                Ok(cool)
            },
        )?;
        // ONE RECORD PER GATE POINT: the paired core has already named the pair and the leg.
        gate_log.record(phase, &quiescence, &cool);
        Ok(())
    }
}

/// The #127 routing decision, extracted so it is TESTABLE rather than only observable through a
/// live engine run.
///
/// **Why it is its own function (#132/F2).** The ruling's whole content is *which source wins on
/// which leg*, and that lived inline in `execute_iterate` — reachable only by spawning an engine.
/// The merged test injected baselines into `iterate_core` directly, which exercises what the run
/// DOES with a pair, not where the pair came from: reverting the routing left the suite green.
/// Now the runner and `run_baselines_ignores_the_goldens_pair_on_the_local_legs` call the same
/// function, so a revert fails a test.
///
/// RULED (David 2026-08-20, "MIRROR REFERENCE"): the LOCAL legs take the constants, full stop.
/// The golden's declared `benchmark.baseline_*_seconds_per_token` is INERT LEGACY DATA here — not
/// cross-checked, not required, not consulted; a stale pair no longer refuses, it is simply
/// ignored. That mirrors the reference's `localIterate`, which reads
/// `MLXFastConstants.officialBaseline*` directly and never reaches the `resolvedBaseline*`
/// accessor (`QwenRuntimeLocalIterate.swift@b26f76f:34,36`, used at `:291,317,382,386`;
/// `grep -c resolvedBaseline` over BOTH harness copies of that file returns 0). BOTH local modes
/// take this path because the reference routes both through that one function — `--local-submit`
/// and `--local-iterate` differ only by decode steps, repeats and labels
/// (`main.swift@b26f76f:315-322`).
///
/// SCOPED DELIBERATELY: OFFICIAL stays golden-authoritative via `resolvedBaseline*`
/// (`Golden.swift@b26f76f:220-226`, consumed at `QwenRuntimeBenchmark.swift@b26f76f:155-157,443-445`).
/// The reference is asymmetric between its own two paths; the ruling mirrors that asymmetry rather
/// than fixing it.
///
/// `golden` is taken and deliberately unused on the local arm: the signature is the claim. A
/// future edit that wants the golden's pair back has to reach for it explicitly, in a function
/// whose doc comment says why it must not.
/// The platform an `iterate` run scores under, from the workflow-declared track id
/// (`MLXFAST_QWEN_MTP_TRACK_ID`, the env measure-job seals as `track_id`). `iterate` carries no
/// track fixture, so the env is the ONE declaration; absent, the run refuses by name rather than
/// guessing which platform's baseline to score against.
/// Resolve the run platform for the cool gate from the workflow-declared track id
/// (`MLXFAST_QWEN_MTP_TRACK_ID`), defaulting to MLX (the Mac 40 C gate) when no track id is set —
/// a bare local run on a Mac. The SCORED measure-job path never uses this: it passes the platform
/// it already resolved from the contract≡env track id. Only the local dev helpers
/// (`--local-cool-gate-only`, `--mode local-iterate`) fall back to this env resolution, where a GB10
/// operator sets the track id env to key the 50 C GB10 gate. The gate temperature is per-platform
/// (`Platform::cool_gate_temp_c`), never a contract/candidate value.
fn cool_gate_platform_from_env() -> bench_core::constants::Platform {
    std::env::var("MLXFAST_QWEN_MTP_TRACK_ID")
        .ok()
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|t| bench_core::constants::Platform::from_track_id(t).ok())
        .unwrap_or(bench_core::constants::Platform::Mlx)
}

/// The workflow-declared track id (`MLXFAST_QWEN_MTP_TRACK_ID`), trimmed, or `None` when it is
/// unset or blank. Every verb that has no `--contract` to read a track from resolves through this
/// one reader, so "which track is this run?" has ONE answer per process.
fn env_track_id() -> Option<String> {
    std::env::var("MLXFAST_QWEN_MTP_TRACK_ID")
        .ok()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
}

/// Resolve the run's per-track MODEL IDENTITY from the workflow-declared track id, exactly where
/// and how [`iterate_platform`] resolves the platform: ONE track id, read once, keying every
/// per-track fact. FAIL-CLOSED both ways — no track id refuses, and a track whose fixture declares
/// no model shape refuses BY NAME rather than falling back to any tree default.
fn iterate_model_identity(
    contract: &contract::Contract,
    env_track_id: Option<&str>,
) -> Result<TrackModelIdentity, String> {
    let track_id = env_track_id
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            "no track_id: set env MLXFAST_QWEN_MTP_TRACK_ID to the track this run scores under \
             (its model identity — golden model_type, vocabulary bound and seed length — is what \
             the golden is validated against)"
                .to_string()
        })?;
    contract::model_identity(contract, track_id).map(|r| r.value)
}

fn iterate_platform(env_track_id: Option<&str>) -> Result<bench_core::constants::Platform, String> {
    let track_id = env_track_id
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            "no track_id: set env MLXFAST_QWEN_MTP_TRACK_ID to the track this run scores under \
             (its `-{platform}-v{N}` suffix keys the official baseline)"
                .to_string()
        })?;
    bench_core::constants::Platform::from_track_id(track_id)
}

/// ARM GATE (David 2026-08-26) — the SOLE scored path (`--mode official`) inherits the gate the
/// retired measure-job carried: REFUSE, pre-GPU and before any score is sealed, unless the
/// `--contract` track fixture declares `official_scoring_enabled: true`. `false` and ABSENT both
/// refuse (an absent arm state is not an armed one). A missing `--contract` also refuses,
/// fail-closed; an UNREADABLE or uncertifiable one has already refused at [`contract::load`].
/// `env_track_id` NAMES the track in the refusal (falling back to the fixture's own `track_id` when
/// the env is unset). Extracted from `execute_iterate` so the gate is behaviorally testable — the
/// verdict over an already-loaded fixture — without a live engine.
fn enforce_official_arm_gate<'a>(
    loaded: Option<&'a contract::LoadedContract>,
    env_track_id: Option<&str>,
) -> Result<&'a contract::Contract, String> {
    let loaded = loaded.ok_or_else(|| {
        "--mode official requires --contract <track-fixture.json> for the arm gate".to_string()
    })?;
    let contract = &loaded.contract;
    // The env track id is the one flow A resolves the platform from; fall back to the fixture's own
    // declared track_id purely to NAME the track in the refusal when the env is unset.
    let track_id = env_track_id
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or(contract.track_id.as_deref())
        .unwrap_or("<unset>");
    contract::enforce_official_scoring_enabled(true, contract.official_scoring_enabled, track_id)?;
    Ok(contract)
}

/// WHETHER THIS TRACK SCORES AGAINST A STORED PAIR at all — the arm choice, asked of the fixture.
///
/// A track whose fixture declares a scored regime is scored against a stored denominator; one that
/// declares none has nothing for this arm to resolve, and the arm's own fence would refuse it
/// anyway. Keying on the regime is the narrower and correct question, and it is the one the arm
/// ends in: "does this track declare a regime" and "can this arm answer for it" are one question.
///
/// STEP 3 (David 2026-09-15): the fixture is the ONLY source. The predicate used to distinguish a
/// fixture declaration from a per-track table row; there is no table left, so it asks the fixture.
fn declares_a_scored_regime(contract: &contract::Contract, track_id: &str) -> bool {
    contract::scored_regime(contract, track_id).is_ok()
}

fn run_baselines(
    contract: &contract::Contract,
    mode: Mode,
    golden: &GoldenFixture,
    flag_override: Option<(f64, f64)>,
    track_id: Option<&str>,
) -> Result<RunBaselines, String> {
    // ONE TABLE. A declared-regime track resolves through `local_mode_baselines` (which fences
    // on the regime); every other track resolves through the table's own guarded accessor. A
    // LIVE-CONTROL-LEG track has no row, so it refuses BY NAME — and the refusal names the path
    // that does have a denominator for it, because the local modes never measure a control leg.
    // The refusal rides on the `official` VALUE rather than on an early return, because
    // `run_baselines_with` consumes it only on the LOCAL arm: official must keep deferring
    // (`ResolveFromOverrideOrGolden`) so `execute_iterate`'s paired arm can measure its own.
    let trimmed = track_id.map(str::trim).filter(|t| !t.is_empty());
    // A LIVE-CONTROL-LEG track has no stored pair to decide from. On a LOCAL leg that is not a
    // refusal: it is the UNSCORED run (`RunBaselines::Unscored`) — the candidate leg, its real
    // timings and its real correctness gate, and no score. On OFFICIAL the paired arm in
    // `execute_iterate` has already taken the run, and this function keeps deferring.
    if trimmed.is_some_and(|t| {
        contract::scores_against_live_control_leg(contract, t).is_ok_and(|r| r.value)
    }) {
        return Ok(if mode.is_local_checked_timing() {
            RunBaselines::Unscored
        } else {
            RunBaselines::ResolveFromOverrideOrGolden
        });
    }
    let official = match trimmed {
        Some(t) if declares_a_scored_regime(contract, t) => {
            iterate::local_mode_baselines(contract, t)
        }
        Some(t) => contract::official_baseline(contract, t).map(|r| r.value),
        None => Err(
            "no track_id: the local baseline pair is keyed by the track this run scores under \
             (set MLXFAST_QWEN_MTP_TRACK_ID)"
                .to_string(),
        ),
    };
    run_baselines_with(mode, golden, flag_override, official)
}

/// The GATES-ONLY (`MLXFAST_BENCHMARK_SKIP_TIMED=1`) official run's baseline pair. Split out of
/// `execute_iterate` so the arm choice is unit-testable without a golden on disk and an engine to
/// spawn — the same seam, and the same predicate, as [`run_baselines`].
///
/// A DECLARED-REGIME track resolves in the reference's own order through
/// `official::official_resolved_baselines` (env override, else the golden's declared pair, else
/// the track's captured pair), because this is the seam-1 producer for the paired overlay and must
/// resolve the way the overlay that completes it does.
///
/// A LIVE-CONTROL-LEG track resolves NOTHING: a gates-only run measures no leg, and that track's
/// only denominator is a leg. It seals the zero placeholders the timing fields already carry on
/// this path, rather than a number no run produced. (David 2026-09-08 — this is where
/// `Platform::official_baseline` used to hand a stored pair to these tracks.)
///
/// Every other track takes the trusted override, else the golden's declared pair.
fn gates_only_baselines(
    contract: &contract::Contract,
    track_id: Option<&str>,
    effective_override: Option<(f64, f64)>,
    golden: &GoldenFixture,
) -> Result<(f64, f64), String> {
    let trimmed = track_id.map(str::trim).filter(|t| !t.is_empty());
    if trimmed.is_some_and(|t| {
        contract::scores_against_live_control_leg(contract, t).is_ok_and(|r| r.value)
    }) {
        return Ok((0.0, 0.0));
    }
    match trimmed.filter(|t| declares_a_scored_regime(contract, t)) {
        Some(track_id) => official::official_resolved_baselines(contract, golden, track_id),
        None => resolve_paired_baselines(effective_override, golden).ok_or_else(|| {
            "no baseline pair for the gates-only official run: neither \
             MLXFAST_PAIRED_BASELINE_{PREFILL,DECODE}_SECONDS_PER_TOKEN, nor the --baseline-* \
             flags, nor the golden's declared pair supplied one"
                .to_string()
        }),
    }
}

/// The #127 rule with the official baseline as a PARAMETER (the seam the captured-state test
/// drives): the local legs score against the official-runner constants and ONLY those — never the
/// golden's declared pair, never a `--baseline` flag. While the capture is pending the accessor's
/// refusal is passed through, so a local run stops here: no engine spawn, no record, no score.
fn run_baselines_with(
    mode: Mode,
    _golden: &GoldenFixture,
    flag_override: Option<(f64, f64)>,
    official: Result<bench_core::constants::OfficialBaseline, String>,
) -> Result<RunBaselines, String> {
    if mode.is_local_checked_timing() {
        let baseline = official?;
        return Ok(RunBaselines::Decided {
            prefill: baseline.prefill_seconds_per_token,
            decode: baseline.decode_seconds_per_token,
            flags_ignored: flag_override.is_some(),
        });
    }
    Ok(RunBaselines::ResolveFromOverrideOrGolden)
}

/// §F2: resolve the REQUIRED paired baselines for an OFFICIAL run. `flag_override` (the
/// `MLXFAST_PAIRED_BASELINE_*` env pair, else both `--baseline-*` flags, already validated as a
/// pair at parse time) is a trusted override; otherwise the golden's benchmark must carry both.
/// `Some` when resolved, `None` when neither supplies them (→ a preflight-failed score).
///
/// **#127 (F8) — the old doc comment here was wrong and is corrected.** It justified the
/// requirement by appeal to Swift `requiredGoldenBenchmarkBaselines` ("no fallback to the official
/// constant"), which is a symbol of the RETIRED `mlxfast-challenge-dev` fork. The reference does
/// not work that way in either direction:
///
/// - its OFFICIAL path is golden-authoritative but FALLS BACK to the constants when the golden
///   declares no pair — `resolvedBaseline*` is `baseline* ?? officialBaseline*`
///   (`Golden.swift@b26f76f:220-226`), consumed at
///   `QwenRuntimeBenchmark.swift@b26f76f:155-157,443-445`;
/// - its LOCAL path never reads the golden's pair at all.
///
/// So this function is now OFFICIAL-ONLY (the local callers were removed by the #127 ruling), and
/// what it still enforces beyond the reference — refusing an official run whose golden declares no
/// pair and whose caller passed no override, where the reference would fall back to the constants
/// — is benchd being STRICTER on the ranked path, deliberately: the ranked runner is required to
/// measure its baseline in the same session (#61), so an official run with no pair in sight is a
/// missing measurement, not a cue to score against a cached constant. Recorded on #127 and NOT
/// changed by that ruling, which scoped itself to the local leg.
fn resolve_paired_baselines(
    flag_override: Option<(f64, f64)>,
    golden: &GoldenFixture,
) -> Option<(f64, f64)> {
    flag_override.or_else(|| {
        golden.benchmark.as_ref().and_then(|b| {
            match (
                b.baseline_prefill_seconds_per_token,
                b.baseline_decode_seconds_per_token,
            ) {
                (Some(p), Some(d)) => Some((p, d)),
                _ => None,
            }
        })
    })
}

/// Whether this OFFICIAL run is a gates-only (seam 1) run: `MLXFAST_BENCHMARK_SKIP_TIMED=1`
/// skips the timed phases so only the correctness gates run, producing a `partial_result=true`
/// gates-score. Mirrors the reference `mlxfast-swift` env contract
/// (`main.swift@b26f76f:386` reads it, defaulting to "1"; `:397` threads it into the options;
/// `QwenRuntimeBenchmark.swift@b26f76f:457` consumes it — #132/F-7 corrected this from
/// `main.swift:321-322`, which is the local branch); the
/// paired driver sets it alongside `MLXFAST_BENCHMARK_CHECK_GATES=1`. Only `"1"` is truthy.
fn official_gates_only_from_env() -> bool {
    std::env::var("MLXFAST_BENCHMARK_SKIP_TIMED")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Resolve the official-run Seatbelt sandbox from the process env + the engine/golden paths,
/// FAIL-CLOSED (Swift `runtimeWorkerOptions`, main.swift:1143-1219). Reads the `MLXFAST_*`
/// knobs, probes `/usr/bin/sandbox-exec` for executability, and returns the resolved plan or
/// the verbatim Swift refusal message. Official forces worker-stderr forwarding OFF.
///
/// Also reads `BENCH_WORKER_RESIDENT_SOCKET`: when set, the derived profile allows the ONE
/// outbound Unix-socket connect the worker needs to attach to the resident bench-worker.
fn resolve_official_sandbox_from_env(
    engine: &str,
    golden: &Path,
    honor_executable_override: bool,
    resident_socket: Option<&str>,
) -> Result<OfficialSandboxPlan, String> {
    let use_rw = std::env::var("MLXFAST_USE_RUNTIME_WORKER").ok();
    let no_sb = std::env::var("MLXFAST_NO_SANDBOX").ok();
    // `honor_executable_override` is FALSE for the REFERENCE leg of a paired run: that leg's
    // executable is derived by re-rooting the resolved candidate into the reference workspace, and
    // an env override that pointed both legs at ONE binary would silently collapse the two roots
    // into one and price the candidate against itself.
    let exec_ov = honor_executable_override
        .then(|| std::env::var("MLXFAST_RUNTIME_WORKER_EXECUTABLE").ok())
        .flatten();
    let prof_ov = std::env::var("MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE").ok();
    let priv_dir = std::env::var("MLXFAST_PRIVATE_DIR").ok();
    // THE RESIDENT SOCKET IS THE SPAWN'S, NOT THE PROCESS'S. Seatbelt counts an AF_UNIX
    // connect as network, so the profile's one `network-outbound` allowance must name exactly
    // the socket THIS spawn's worker will connect to. On the paired path that socket lives in
    // the LEG's spawn-env overrides and nowhere else — reading it from this process's own env
    // gave a per-leg spawn no allowance at all, and the box-4 calibration died on
    // `cannot reach the resident … Operation not permitted`. The caller passes the leg's
    // socket; the legacy single-resident path passes the process env's.
    let golden_str = golden.to_string_lossy().to_string();
    let inputs = OfficialSandboxInputs {
        use_runtime_worker: use_rw.as_deref(),
        no_sandbox: no_sb.as_deref(),
        executable_override: exec_ov.as_deref(),
        profile_override: prof_ov.as_deref(),
        private_dir: priv_dir.as_deref(),
        resident_socket,
        fallback_executable: engine,
        golden_path: &golden_str,
        sandbox_exec_available: sandbox_exec_is_executable(),
    };
    // forwards_worker_stderr = false: official never echoes worker stderr (the plan also
    // forces it off, so this is belt-and-braces).
    resolve_official_sandbox(&inputs, false).map_err(|e| e.to_string())
}

/// Whether `/usr/bin/sandbox-exec` exists and is executable (Swift
/// `FileManager.isExecutableFile`). On unix, requires an execute bit; elsewhere, existence.
fn sandbox_exec_is_executable() -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(SANDBOX_EXEC_PATH)
            .map(|m| m.is_file() && (m.permissions().mode() & 0o111) != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        std::fs::metadata(SANDBOX_EXEC_PATH)
            .map(|m| m.is_file())
            .unwrap_or(false)
    }
}

/// Build the optional integrity pin from CLI flag strings. Both or neither must be given.
fn parse_golden_pin(
    sha256: Option<String>,
    bytes: Option<String>,
) -> Result<Option<GoldenIntegrityPin>, String> {
    parse_named_golden_pin(sha256, bytes, "--golden-sha256", "--golden-bytes")
}

/// The pin rule for a REPEATABLE golden flag: each pin flag is given once per golden, matched by
/// position, or not at all. `names` are the sha flag, the bytes flag and the golden flag, so each
/// golden surface names its own flags when it refuses.
fn parse_golden_pins(
    sha256: Vec<&str>,
    bytes: Vec<&str>,
    goldens: usize,
    names: [&str; 3],
) -> Result<Vec<GoldenIntegrityPin>, String> {
    let [sha_flag, bytes_flag, golden_flag] = names;
    if sha256.is_empty() || bytes.is_empty() {
        // Neither is no pin. One without the other is half a pin, refused by the one-pin rule.
        return parse_named_golden_pin(
            sha256.first().map(|v| v.to_string()),
            bytes.first().map(|v| v.to_string()),
            sha_flag,
            bytes_flag,
        )
        .map(|_| Vec::new());
    }
    if sha256.len() != goldens || bytes.len() != goldens {
        return Err(format!(
            "{}: {sha_flag} was given {} time(s), {bytes_flag} {} time(s) and {golden_flag} \
             {goldens} time(s); give each pin once per {golden_flag}, in the same order, or not \
             at all",
            baseline::GOLDEN_COUNT_MISMATCH,
            sha256.len(),
            bytes.len()
        ));
    }
    sha256
        .into_iter()
        .zip(bytes)
        .map(|(sha, n)| {
            parse_named_golden_pin(
                Some(sha.to_string()),
                Some(n.to_string()),
                sha_flag,
                bytes_flag,
            )
            .map(|pin| pin.expect("both halves of the pin were given"))
        })
        .collect()
}

/// The pin rule itself, stated once: both flags together or neither, and the byte count parses as
/// a u64. `sha_flag` / `bytes_flag` are the names the refusal reports, so each golden surface
/// names its own flags.
fn parse_named_golden_pin(
    sha256: Option<String>,
    bytes: Option<String>,
    sha_flag: &str,
    bytes_flag: &str,
) -> Result<Option<GoldenIntegrityPin>, String> {
    match (sha256, bytes) {
        (None, None) => Ok(None),
        (Some(sha256), Some(bytes)) => {
            let bytes: u64 = bytes
                .parse()
                .map_err(|_| format!("invalid u64 for {bytes_flag}: {bytes:?}"))?;
            Ok(Some(GoldenIntegrityPin { sha256, bytes }))
        }
        _ => Err(format!(
            "{sha_flag} and {bytes_flag} must be given together"
        )),
    }
}

/// Read + integrity-pin (when given) + load-validate a golden. The pin is checked on the
/// RAW BYTES BEFORE any parse (port of verify-correctness-golden.sh), so an unexpected,
/// tampered, or unknown-provenance golden fails closed before its contents are trusted.
///
/// `required_steps` is the caller's `expected_tokens` arity, because the REFERENCE's arity is
/// per-consumer, not global: `QwenRuntime.localIterate` loads with `benchmarkDecodeSteps + 1`
/// while the standalone `correctness`/`benchmark` paths take the `correctnessSteps` default
/// (see [`Mode::golden_required_steps`]).
///
/// `track_id` + `identity` are the run's TRACK and the model identity that track declares. The
/// identity supplies the vocabulary bound, the seed arity and the required `model_type`, so one
/// tree loads a golden of any declared track — and a golden of ANOTHER track is refused naming
/// both the track and the two identities.
fn load_golden_checked(
    path: &Path,
    pin: Option<&GoldenIntegrityPin>,
    required_steps: usize,
    reference_model: Option<&ReferenceModelPin>,
    track_id: &str,
    identity: &TrackModelIdentity,
) -> Result<GoldenFixture, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("golden read failed ({}): {e}", path.display()))?;
    load_golden_bytes_checked(
        &bytes,
        pin,
        required_steps,
        reference_model,
        track_id,
        identity,
    )
}

/// #112 (L2) — the BYTES half of [`load_golden_checked`], for a caller that ALREADY HOLDS the
/// file's bytes. The pin, the sha256 the loader records, and the parsed document then all come
/// from ONE read of ONE byte string: a second `std::fs::read` of the same path could observe a
/// DIFFERENT file (a golden swapped between the two reads), which would let the identity checked
/// upstream and the document actually loaded diverge. The tape branch of
/// [`load_timed_prompt_checked`] already worked this way; this makes the GoldenDocument branch
/// match it.
fn load_golden_bytes_checked(
    bytes: &[u8],
    pin: Option<&GoldenIntegrityPin>,
    required_steps: usize,
    reference_model: Option<&ReferenceModelPin>,
    track_id: &str,
    identity: &TrackModelIdentity,
) -> Result<GoldenFixture, String> {
    // The pin (when given) is now folded INTO the loader: a single call checks it on the raw
    // bytes BEFORE the parse, so this "load for use" path can no longer forget the pin. The
    // mismatch reason (byte-count / sha256) is emitted verbatim by the loader.
    //
    // #114 — `reference_model` is the SAME arrangement for the track contract's reference-model
    // identity: a caller that parsed a `--contract` hands its pin here and the loader applies
    // the reference's own value gate; a caller with no contract passes `None` and gets the
    // shape-only check.
    load_golden_fixture(
        bytes,
        required_steps,
        identity.seed_tokens,
        identity,
        Some(&identity.golden_model_type),
        pin,
        reference_model,
    )
    // NAME BOTH SIDES. The loader's own message is the reference's (Swift-parity) wording and
    // names the two model types; the TRACK is what this wrapper knows, and without it a
    // cross-track golden reads as a bare type mismatch rather than "this golden belongs to
    // another track".
    .map_err(|e| {
        format!(
            "golden load failed (track {track_id:?}, model identity {}, vocab {}, seed {}): {e}",
            identity.golden_model_type, identity.vocab_size, identity.seed_tokens
        )
    })
}

/// The `validate-golden` command line. [`VALIDATE_GOLDEN_USAGE`] is its prose; this is its grammar.
static VALIDATE_GOLDEN_FLAGS: &[FlagSpec] = &[
    FlagSpec::value(&["--golden"]).required(),
    FlagSpec::value(&["--golden-sha256"]),
    FlagSpec::value(&["--golden-bytes"]),
    // STEP 3 (David 2026-09-15) — the identity the golden is judged against comes from the track
    // FIXTURE, and there is no in-tree table to fall back to, so the fixture is REQUIRED.
    FlagSpec::value(&["--contract"]).required(),
    FlagSpec::value(&["--track"]),
    FlagSpec::switch(&["--gates-only"]),
];

/// The `validate-golden` arguments, built out of [`VALIDATE_GOLDEN_FLAGS`].
struct ValidateGoldenArgs {
    golden: PathBuf,
    /// The optional integrity pin: `--golden-sha256` + `--golden-bytes`, both or neither.
    pin: Option<GoldenIntegrityPin>,
    contract: PathBuf,
    /// `--track` AS TYPED. The env fallback (`MLXFAST_QWEN_MTP_TRACK_ID`) is resolved at run time,
    /// not here, so the parse stays a pure function of the command line.
    track: Option<String>,
    gates_only: bool,
}

/// The `validate-golden` subcommand: its name, its usage prose, its flag table and its parse.
static VALIDATE_GOLDEN_CLI: CliSpec<ValidateGoldenArgs> = CliSpec {
    name: "validate-golden",
    usage: VALIDATE_GOLDEN_USAGE,
    flags: VALIDATE_GOLDEN_FLAGS,
    parse: parse_validate_golden_args,
};

fn parse_validate_golden_args(args: &[String]) -> Result<Option<ValidateGoldenArgs>, String> {
    let Some(flags) = parse_flags(args, VALIDATE_GOLDEN_CLI.flags)? else {
        // `-h`/`--help`.
        return Ok(None);
    };
    build_validate_golden_args(&flags).map(Some)
}

fn build_validate_golden_args(flags: &ParsedFlags) -> Result<ValidateGoldenArgs, String> {
    Ok(ValidateGoldenArgs {
        golden: flags
            .path("--golden")
            .ok_or_else(|| flags.missing_required("--golden"))?,
        pin: parse_golden_pin(
            flags.string("--golden-sha256"),
            flags.string("--golden-bytes"),
        )?,
        contract: flags
            .path("--contract")
            .ok_or_else(|| flags.missing_required("--contract"))?,
        track: flags.string("--track"),
        gates_only: flags.present("--gates-only"),
    })
}

/// `validate-golden`: integrity-pin + load-validate a golden, no engine. Exit 0 = accepted,
/// 1 = rejected (pin or schema), 2 = usage error. The loader-parity harness compares this
/// accept/reject against `mlxfast-swift preflight` on the same fixture corpus.
fn run_validate_golden(args: &[String]) -> ExitCode {
    run_cli(&VALIDATE_GOLDEN_CLI, args, execute_validate_golden)
}

fn execute_validate_golden(args: &ValidateGoldenArgs) -> Result<u8, (u8, String)> {
    // The MODEL IDENTITY the golden is judged against is the TRACK's, resolved from `--track` or
    // the workflow env — never a compile-time default. A missing or undeclared track is a USAGE
    // error (exit 2): the caller's ARGUMENT is unusable, which is not the same event as the golden
    // being rejected, and answering "accepted" under a guessed identity is the defect this
    // command exists to catch.
    let track_id = args
        .track
        .as_ref()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .or_else(env_track_id)
        .ok_or_else(|| {
            (
                2,
                "no track: pass --track <TRACK-ID> or set env MLXFAST_QWEN_MTP_TRACK_ID (the \
                 track's model identity is what the golden is judged against)"
                    .to_string(),
            )
        })?;
    // STEP 3 (David 2026-09-15) — the model identity the golden is judged against comes from the
    // TRACK FIXTURE. There is no in-tree table left, so `--contract` is REQUIRED (the table row
    // says so): a missing one is a USAGE error (exit 2), like a missing `--track`, because the
    // caller's arguments do not say which identity to judge under.
    let loaded_contract = contract::load(&args.contract).map_err(|e| (2, e))?;
    let identity = contract::model_identity(&loaded_contract.contract, &track_id)
        .map(|r| r.value)
        .map_err(|e| (2, e))?;
    // #114 — the track contract's declared reference-model identity. A contract that cannot be
    // READ is exit 3 (IO, same as an unreadable golden); one that cannot be PARSED, or that
    // declares only half a pin, is exit 2 — the caller's ARGUMENT is invalid, which is not the
    // same event as the golden being rejected, and collapsing the two would let a broken contract
    // read as "this golden is fine".
    let contract_bytes = std::fs::read(&args.contract).map_err(|e| {
        (
            3,
            format!("IO ERROR reading contract {}: {e}", args.contract.display()),
        )
    })?;
    let reference_model = reference_model_pin_from_contract(&contract_bytes).map_err(|e| {
        (
            2,
            format!("contract {} is unusable: {e}", args.contract.display()),
        )
    })?;
    // Distinct exit codes so a harness can tell an IO failure apart from a rejection:
    //   0 = accepted, 1 = rejected (integrity pin or schema), 2 = usage, 3 = IO error.
    let bytes = std::fs::read(&args.golden).map_err(|e| {
        (
            3,
            format!("IO ERROR reading {}: {e}", args.golden.display()),
        )
    })?;
    // validate-golden keeps the pin as an EXPLICIT step (not folded into the loader) because it
    // has a bespoke contract the folded form cannot express: a distinct "(integrity pin)" reject
    // label vs "(schema)", and it already owns the raw bytes for the IO-vs-reject exit-code split
    // (3 = IO, 1 = reject). The loader is therefore called unpinned (`None`) here; the pin is
    // still enforced, just at this call site with richer diagnostics.
    if let Some(pin) = args.pin.as_ref() {
        if let Err(e) = verify_golden_integrity(&bytes, pin) {
            return Err((
                1,
                format!("REJECT {} (integrity pin): {e}", args.golden.display()),
            ));
        }
    }
    // #114 — the reference-model pin, unlike the integrity pin above, is passed INTO the loader
    // rather than checked as a separate step: the reference applies it mid-load (after the shape
    // check, before the case validation), so folding it in keeps benchd's evaluation ORDER equal
    // to Swift's, and the reject carries the reference's own message. It is deliberately NOT given
    // its own reject label — Swift raises it as a plain loader `invalidInput` too, so labelling it
    // "(schema)" is what makes the two loaders' stderr comparable.
    match load_golden_fixture(
        &bytes,
        CORRECTNESS_STEPS,
        identity.seed_tokens,
        &identity,
        Some(&identity.golden_model_type),
        None,
        reference_model.as_ref(),
    ) {
        Ok(fx) => {
            // #77: by default a benchmark golden MUST carry a benchmark oracle block, byte-
            // consistent with Swift preflight (which rejects a benchmark-less golden with the
            // same message). `--gates-only` skips this for internal structural fixtures that
            // legitimately lack a benchmark oracle (structure + gates already validated above).
            if !args.gates_only && fx.benchmark.is_none() {
                return Err((
                    1,
                    format!(
                        "REJECT {} (schema): benchmark golden file must contain a benchmark oracle",
                        args.golden.display()
                    ),
                ));
            }
            eprintln!(
                "benchd validate-golden: ACCEPT {} (sha256={}, cases={}, gate_cases={})",
                args.golden.display(),
                fx.sha256,
                fx.cases.len(),
                fx.correctness_gates
                    .as_ref()
                    .map(|g| g.total_case_count())
                    .unwrap_or(0)
            );
            Ok(0)
        }
        Err(e) => Err((1, format!("REJECT {} (schema): {e}", args.golden.display()))),
    }
}

/// The `correctness` command line. [`CORRECTNESS_USAGE`] is its prose; this is its grammar.
static CORRECTNESS_FLAGS: &[FlagSpec] = &[
    FlagSpec::value(&["--engine"]).required(),
    FlagSpec::value(&["--weights"]).required(),
    FlagSpec::value(&["--golden"]).required(),
    FlagSpec::value(&["--golden-sha256"]),
    FlagSpec::value(&["--golden-bytes"]),
    // Runner contract §13: the manifest the engine's hello is checked against, and the kit's
    // trusted posture (only a trusted build derives `cohort_reference_replay`).
    FlagSpec::value(&["--manifest"]),
    FlagSpec::switch(&["--trusted"]),
    // §8.1/§13b resource passthrough. The correctness spawn LOADS the model, so it needs the same
    // resources the timed spawns do. The values come from THIS command line, never from a
    // submission-editable file.
    FlagSpec::repeatable(&[engine_resource::ENGINE_RESOURCE_FLAG]),
];

/// The `correctness` arguments, built out of [`CORRECTNESS_FLAGS`].
struct CorrectnessArgs {
    engine: String,
    weights: PathBuf,
    golden: PathBuf,
    /// The optional integrity pin: `--golden-sha256` + `--golden-bytes`, both or neither.
    pin: Option<GoldenIntegrityPin>,
    manifest: Option<PathBuf>,
    trusted: bool,
    engine_resources: Vec<engine_resource::EngineResource>,
}

/// The `correctness` subcommand: its name, its usage prose, its flag table and its parse.
static CORRECTNESS_CLI: CliSpec<CorrectnessArgs> = CliSpec {
    name: "correctness",
    usage: CORRECTNESS_USAGE,
    flags: CORRECTNESS_FLAGS,
    parse: parse_correctness_args,
};

fn parse_correctness_args(args: &[String]) -> Result<Option<CorrectnessArgs>, String> {
    let Some(flags) = parse_flags(args, CORRECTNESS_CLI.flags)? else {
        // `-h`/`--help`.
        return Ok(None);
    };
    build_correctness_args(&flags).map(Some)
}

fn build_correctness_args(flags: &ParsedFlags) -> Result<CorrectnessArgs, String> {
    let mut engine_resources: Vec<engine_resource::EngineResource> = Vec::new();
    for value in flags.values(engine_resource::ENGINE_RESOURCE_FLAG) {
        engine_resource::push_engine_resource(&mut engine_resources, value)?;
    }
    Ok(CorrectnessArgs {
        engine: flags
            .string("--engine")
            .ok_or_else(|| flags.missing_required("--engine"))?,
        weights: flags
            .path("--weights")
            .ok_or_else(|| flags.missing_required("--weights"))?,
        golden: flags
            .path("--golden")
            .ok_or_else(|| flags.missing_required("--golden"))?,
        pin: parse_golden_pin(
            flags.string("--golden-sha256"),
            flags.string("--golden-bytes"),
        )?,
        manifest: flags.path("--manifest"),
        trusted: flags.present("--trusted"),
        engine_resources,
    })
}

/// `correctness` (#90 item 2): spawn the engine, run the FULL correctness set, print a JSON
/// verdict, and exit 0 (pass) / 1 (fail). Byte-matches Swift `mlxfast-swift correctness`'s
/// exit contract. `2` on usage error. The golden is loaded oracle-optional (Swift
/// `checkCorrectnessArtifacts`).
fn run_correctness(args: &[String]) -> ExitCode {
    run_cli(&CORRECTNESS_CLI, args, execute_correctness)
}

fn execute_correctness(args: &CorrectnessArgs) -> Result<u8, (u8, String)> {
    // Load the golden ORACLE-OPTIONAL (Swift checkCorrectnessArtifacts): load_golden_checked
    // integrity-pins + structurally validates but does NOT require a benchmark oracle. Arity is
    // the loader DEFAULT here — Swift `correctness` calls `loadGoldenFixture(from:)` with no
    // `requiredSteps:` override (`QwenRuntimeCorrectness.swift:80`), i.e. `correctnessSteps`.
    // #114 — reference-model pin `None`: `correctness` takes no `--contract` (same scoped residual
    // as `iterate`).
    // The MODEL IDENTITY the golden is loaded under, and the vocabulary bound the conformance
    // gates judge worker logits against, are the TRACK's. `correctness` takes no `--contract`, so
    // the workflow env is its only source; an unset or undeclared track refuses (exit 2) rather
    // than falling back to a tree default.
    let track_id = env_track_id().ok_or_else(|| {
        (
            2,
            "no track_id: set env MLXFAST_QWEN_MTP_TRACK_ID to the track this golden belongs to \
             (its model identity is what the golden is validated against)"
                .to_string(),
        )
    })?;
    // These verbs take no `--contract`, so they resolve the identity against a fixture that
    // declares nothing (STEP 1, David 2026-09-15): the in-tree table, exactly as before — but
    // through the ONE resolver, so a declared shape reaches them the day a fixture carries one.
    let identity = contract::model_identity(&contract::Contract::NONE_DECLARED, &track_id)
        .map(|r| r.value)
        .map_err(|e| (2, e))?;
    let golden_fx = load_golden_checked(
        &args.golden,
        args.pin.as_ref(),
        CORRECTNESS_STEPS,
        None,
        &track_id,
        &identity,
    )
    .map_err(|e| (1, e))?;
    let weights_str = args.weights.to_string_lossy().to_string();
    // The correctness gate is TEACHER-FORCED and spawns GATE-OFF (strict v1, the standing
    // v1-compat proof), so the argv carries the resources and NOT `--speculative-protocol`.
    let spawn_args = engine_resource::spawn_args(&args.engine_resources);
    let transport = ChildStdioTransport::spawn(&args.engine, &weights_str, &spawn_args)
        .map_err(|e| (1, format!("failed to spawn engine {:?}: {e}", args.engine)))?;
    let (mut session, hello) = Session::connect(transport)
        .map_err(|e| (1, format!("engine hello handshake failed: {e}")))?;
    // Runner contract §13: the hello-against-manifest gate runs BEFORE the correctness set, so a
    // worker that misdeclares itself is refused without spending the gate.
    if let Some(path) = args.manifest.as_deref() {
        match correctness::check_manifest(path, &hello, args.trusted) {
            Ok(()) => eprintln!(
                "benchd correctness: manifest {} matches the hello",
                path.display()
            ),
            Err(correctness::ManifestGateError::Input(m)) => {
                return Err((2, format!("manifest {m}")));
            }
            Err(correctness::ManifestGateError::Failed(failures)) => {
                for failure in &failures {
                    eprintln!("benchd correctness: manifest check FAILED {failure}");
                }
                return Ok(1);
            }
        }
    } else if args.trusted {
        return Err((2, "--trusted needs --manifest".to_string()));
    }
    let outcome = correctness::correctness_core(&mut session, &golden_fx, &identity);
    // Emit the JSON verdict to stdout; the exit code is the authoritative pass/fail.
    print!("{}", outcome.to_json());
    eprintln!(
        "benchd correctness: passed={} case_count={}{}",
        outcome.passed,
        outcome.case_count,
        if outcome.passed {
            String::new()
        } else {
            format!(" error={:?}", outcome.error)
        }
    );
    if outcome.passed {
        Ok(0)
    } else {
        Ok(1)
    }
}

/// The `validate-weights` command line. [`VALIDATE_WEIGHTS_USAGE`] is its prose; this is its
/// grammar.
static VALIDATE_WEIGHTS_FLAGS: &[FlagSpec] = &[
    FlagSpec::value(&["--weights"]).required(),
    // The golden half is `validate-golden`'s; pass it here only to also require its PRESENCE
    // (Swift `requiredFiles`).
    FlagSpec::value(&["--golden"]),
];

/// The `validate-weights` arguments, built out of [`VALIDATE_WEIGHTS_FLAGS`].
struct ValidateWeightsArgs {
    weights: PathBuf,
    golden: Option<PathBuf>,
}

/// The `validate-weights` subcommand: its name, its usage prose, its flag table and its parse.
static VALIDATE_WEIGHTS_CLI: CliSpec<ValidateWeightsArgs> = CliSpec {
    name: "validate-weights",
    usage: VALIDATE_WEIGHTS_USAGE,
    flags: VALIDATE_WEIGHTS_FLAGS,
    parse: parse_validate_weights_args,
};

fn parse_validate_weights_args(args: &[String]) -> Result<Option<ValidateWeightsArgs>, String> {
    let Some(flags) = parse_flags(args, VALIDATE_WEIGHTS_CLI.flags)? else {
        // `-h`/`--help`.
        return Ok(None);
    };
    build_validate_weights_args(&flags).map(Some)
}

fn build_validate_weights_args(flags: &ParsedFlags) -> Result<ValidateWeightsArgs, String> {
    Ok(ValidateWeightsArgs {
        weights: flags
            .path("--weights")
            .ok_or_else(|| flags.missing_required("--weights"))?,
        golden: flags.path("--golden"),
    })
}

/// `validate-weights` (#90 item 3): the WEIGHTS-half preflight. Exit 0 = accepted,
/// 1 = rejected, 2 = usage, 3 = IO error (mirrors `validate-golden`'s distinct codes).
fn run_validate_weights(args: &[String]) -> ExitCode {
    run_cli(&VALIDATE_WEIGHTS_CLI, args, execute_validate_weights)
}

fn execute_validate_weights(args: &ValidateWeightsArgs) -> Result<u8, (u8, String)> {
    // Resolve the size cap from MLXFAST_MAX_WEIGHTS_BYTES (Swift parseTransformedWeightsByteLimit).
    let cap = weights_preflight::weights_byte_limit_from_env(
        std::env::var("MLXFAST_MAX_WEIGHTS_BYTES").ok().as_deref(),
    )
    .map_err(|m| (2, m))?;
    match weights_preflight::validate_weights(&args.weights, args.golden.as_deref(), cap) {
        Ok(report) => {
            eprintln!(
                "benchd validate-weights: ACCEPT {} (bytes={}, files={}, cap={})",
                args.weights.display(),
                report.weights_byte_count,
                report.file_count,
                report
                    .max_weights_byte_count
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "unlimited".to_string()),
            );
            Ok(0)
        }
        Err(e) => Err((1, format!("REJECT {}: {e}", args.weights.display()))),
    }
}

/// The `weights-digest` command line. [`WEIGHTS_DIGEST_USAGE`] is its prose; this is its grammar.
static WEIGHTS_DIGEST_FLAGS: &[FlagSpec] = &[FlagSpec::value(&["--weights"]).required()];

/// The `weights-digest` arguments, built out of [`WEIGHTS_DIGEST_FLAGS`].
struct WeightsDigestArgs {
    weights: PathBuf,
}

/// The `weights-digest` subcommand: its name, its usage prose, its flag table and its parse.
static WEIGHTS_DIGEST_CLI: CliSpec<WeightsDigestArgs> = CliSpec {
    name: "weights-digest",
    usage: WEIGHTS_DIGEST_USAGE,
    flags: WEIGHTS_DIGEST_FLAGS,
    parse: parse_weights_digest_args,
};

fn parse_weights_digest_args(args: &[String]) -> Result<Option<WeightsDigestArgs>, String> {
    let Some(flags) = parse_flags(args, WEIGHTS_DIGEST_CLI.flags)? else {
        // `-h`/`--help`.
        return Ok(None);
    };
    build_weights_digest_args(&flags).map(Some)
}

fn build_weights_digest_args(flags: &ParsedFlags) -> Result<WeightsDigestArgs, String> {
    Ok(WeightsDigestArgs {
        weights: flags
            .path("--weights")
            .ok_or_else(|| flags.missing_required("--weights"))?,
    })
}

/// `benchd weights-digest --weights <DIR>` — the Option B digest-hoist producer.
///
/// Computes the weights digest with the SAME [`dir_digest`] an `iterate` pass uses and prints it to
/// stdout as `<sha256>:<byte_count>:<file_count>` (one line, no extra text). The window runs this
/// ONCE at start and passes the value to passes 2-N via `iterate --weights-digest`, so the
/// immutable ~105 GB tree is hashed once per window instead of once per pass. Because it calls the
/// SAME `dir_digest`, the printed value is byte-identical to a per-pass recompute — the property the
/// hoist relies on. Read-only: no engine, no baseline, no artifact.
fn run_weights_digest(args: &[String]) -> ExitCode {
    run_cli(&WEIGHTS_DIGEST_CLI, args, execute_weights_digest)
}

fn execute_weights_digest(args: &WeightsDigestArgs) -> Result<u8, (u8, String)> {
    match dir_digest_weights(&args.weights) {
        Ok(d) => {
            // The stable, parseable form `iterate --weights-digest` parses back with
            // `parse_weights_digest`. `dir_digest` renders sha256 through `hex_lower`, so this
            // string round-trips to a byte-identical `DirDigest`.
            println!("{}:{}:{}", d.sha256, d.byte_count, d.file_count);
            Ok(0)
        }
        Err(e) => Err((
            3,
            format!("digest failed ({}): {e}", args.weights.display()),
        )),
    }
}

/// Parse a `--weights-digest` value of the form `<sha256>:<byte_count>:<file_count>` — the exact
/// string `benchd weights-digest` prints — into a [`DirDigest`]. The sha256 must be 64 lowercase
/// hex chars (the `hex_lower` form `dir_digest` emits); both counts are non-negative i64. Building
/// the digest from this string is byte-identical to `dir_digest` computing it over the same
/// immutable tree, which is the whole point of the hoist.
fn parse_weights_digest(v: &str) -> Result<DirDigest, String> {
    let parts: Vec<&str> = v.split(':').collect();
    if parts.len() != 3 {
        return Err(format!(
            "invalid --weights-digest {v:?}: expected <sha256>:<byte_count>:<file_count>"
        ));
    }
    let sha256 = parts[0];
    if sha256.len() != 64
        || !sha256
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(format!(
            "invalid --weights-digest sha256 {sha256:?}: expected 64 lowercase hex chars (the \
             `benchd weights-digest` form)"
        ));
    }
    let byte_count: i64 = parts[1]
        .parse()
        .map_err(|_| format!("invalid --weights-digest byte_count {:?}", parts[1]))?;
    let file_count: i64 = parts[2]
        .parse()
        .map_err(|_| format!("invalid --weights-digest file_count {:?}", parts[2]))?;
    if byte_count < 0 || file_count < 0 {
        return Err(format!(
            "invalid --weights-digest {v:?}: byte_count and file_count must be non-negative"
        ));
    }
    Ok(DirDigest {
        sha256: sha256.to_string(),
        byte_count,
        file_count,
    })
}

/// Resolve the weights digest for an iterate run: the pre-computed `--weights-digest` value when
/// present (Option B digest-hoist — the window's once-per-run digest, byte-identical to a per-pass
/// recompute), otherwise a fresh [`dir_digest`] of the weights tree. When a digest is passed the
/// tree is NOT touched — that is the whole saving. `--weights-digest` is refused at parse outside
/// `--capture-baseline`, so this reuse branch is unreachable on an official/scored run, which always
/// hashes for itself. Extracted from `execute_iterate` so the skip-recompute selection is unit-
/// testable without a live engine.
fn resolve_weights_digest(args: &IterateArgs) -> std::io::Result<DirDigest> {
    match args.weights_digest.as_ref() {
        Some(passed) => Ok(passed.clone()),
        None => dir_digest_weights(&args.weights),
    }
}

/// Write the sealed score + its `.sha256` sidecar. Returns the score sha256 (for the
/// integrity sidecar). The sidecar format is the shasum two-space form
/// `"<hex>  <score_path>\n"`, byte-matching benchmark.sh
/// (`printf '%s  %s\n' "${score_hash}" "${SCORE_PATH}"`, benchmark.sh:1269-1270), not the
/// bare `"<hex>\n"` benchd used before.
fn write_score(score_path: &Path, json: &str) -> Result<String, String> {
    if let Some(parent) = score_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
        }
    }
    std::fs::write(score_path, json.as_bytes())
        .map_err(|e| format!("could not write {}: {e}", score_path.display()))?;
    let sidecar = score_path.with_file_name(format!(
        "{}.sha256",
        score_path.file_name().unwrap_or_default().to_string_lossy()
    ));
    let digest = sha256_hex(json.as_bytes());
    std::fs::write(&sidecar, format!("{digest}  {}\n", score_path.display()))
        .map_err(|e| format!("could not write {}: {e}", sidecar.display()))?;
    Ok(digest)
}

/// The benchmark integrity sidecar (Swift/`benchmark.sh` `benchmark-integrity.*.json`,
/// benchmark.sh:1289-1308). `weights_*_count` are JSON numbers.
///
/// The first nine fields are the reference's, in the reference's `jq -n` order. Everything after
/// them is benchd's RUNNER-IDENTITY superset.
///
/// **#123 — RULED (David 2026-08-20): EXTEND THE SIDECAR.** The sidecar sealed the score, the
/// weights and the golden but named no executable, so the runner behind a local-iterate number was
/// not pinned and an E2 parity claim rested on inference (window 4: the engine binaries had been
/// rebuilt ~2h before the run, which makes them the overwhelmingly likely ones — but "likely" is
/// not a seal). Every `benchmark-integrity.results.json` the measure-job legs write DOES pin them;
/// this brings the local sidecar up to that bar. The issue recommended a separate
/// `benchmark-runners.*.json` to preserve the 9-field byte-match; David chose the superset, so
/// **the byte-match row is RETIRED and re-graded VERIFIED (superset)** —
/// the sidecar is a strict superset of the reference's object, not a byte-for-byte twin.
///
/// BACKWARD-READABLE by construction: the nine reference fields keep their names, types, order and
/// values, so a consumer that reads them still reads them.
#[derive(serde::Serialize)]
struct IntegritySidecar {
    score_path: String,
    score_sha256: String,
    weights_path: String,
    weights_sha256: String,
    weights_file_count: i64,
    weights_byte_count: i64,
    golden_path: String,
    golden_sha256: String,
    transform_source_sha256: String,
    /// The engine executable this run actually spawned, and the sha256 of its BYTES. Named as
    /// measure-job names it (`candidate_executable`) so one vocabulary covers both surfaces.
    /// Resolved and digested BEFORE the run, so the seal names the binary that ran and a
    /// missing/unreadable engine fails early rather than after the score exists.
    candidate_executable: String,
    candidate_executable_sha256: String,
    /// Empty by construction on this command, and empty is the ANSWER, not a gap: `iterate` runs
    /// ONE engine. There is no baseline leg to pin — on the local legs the baseline is the
    /// compile-time constant pair (#127), and on official it is the golden's declaration or the
    /// `MLXFAST_PAIRED_BASELINE_*` override, neither of which is a runner. A two-leg measurement
    /// pins its baseline runner in `benchmark-integrity.results.json` instead.
    baseline_executable: String,
    baseline_executable_sha256: String,
    /// Whether `candidate_executable` above is a canonical path with a real digest
    /// (`"canonical"`), or a path benchd could not resolve or read, recorded verbatim with an
    /// empty digest (`"unresolved"`) — #132/F3.
    ///
    /// This exists so an empty `candidate_executable_sha256` is never AMBIGUOUS. Without it, a
    /// reader cannot tell "benchd sealed no engine digest" from "the engine had no digest to
    /// seal", and the honest answer to a PATH-resolved bare name or an exec-but-not-readable
    /// binary is the second one. Sealing the weaker identity beats refusing the run: #123 exists
    /// to make artifacts say MORE about the runner, not to make runs that used to work fail.
    candidate_executable_resolution: String,
    /// benchd's own executable — the OTHER half of the runner identity, and the half no artifact
    /// pinned before. `metrics.commit` names a source revision; this names the binary.
    benchd_executable: String,
    benchd_executable_sha256: String,
    /// Empty on this command: `iterate` takes no workspace argument, so there is no built-engine
    /// SOURCE tree to digest — only the built binary above. Carried as a declared empty rather
    /// than omitted, so the field roster matches measure-job's and a consumer can tell "this run
    /// had no workspace" from "this run forgot to record one". Populating it would need a new
    /// `--candidate-workspace` flag, which the ruling did not ask for.
    candidate_workspace_sha256: String,
    /// HOW the runtime worker was sandboxed for this run (a8 ruling b): `seatbelt` when it ran
    /// under the macOS Seatbelt profile, `none (linux)` when an official run on a host without
    /// Seatbelt ran it unsandboxed, `none` for the never-sandboxed local modes. Honest provenance
    /// so a reader knows the isolation an official Linux seal was produced under, without inferring
    /// it from the host. See [`sandbox_provenance`].
    sandbox: String,
}

/// The runner identity a `benchd iterate` run seals into its integrity sidecar (#123).
struct RunnerIdentity {
    candidate_executable: String,
    candidate_executable_sha256: String,
    candidate_executable_resolution: String,
    benchd_executable: String,
    benchd_executable_sha256: String,
}

/// `candidate_executable_resolution` when the engine path canonicalised AND read.
const ENGINE_RESOLUTION_CANONICAL: &str = "canonical";
/// `candidate_executable_resolution` when it did not: the path is sealed as GIVEN, with no digest.
const ENGINE_RESOLUTION_UNRESOLVED: &str = "unresolved";

/// Resolve + digest the executables behind an `iterate` run.
///
/// `engine` is the path this run will spawn: `args.engine` on the local legs, and the sandbox
/// plan's `executable_path` on official (where `MLXFAST_RUNTIME_WORKER_EXECUTABLE` can point
/// somewhere else entirely — sealing `args.engine` there would name a binary that never ran).
///
/// Paths are canonicalised so the seal is location-stable. This is TOTAL — it never fails the run.
///
/// **#132/F3 — it used to.** The first cut hard-errored when `canonicalize` or `read` failed on
/// the engine, reasoning that an unreadable engine is a run about to fail anyway. That reasoning
/// is wrong for two shapes that worked before #123 and stopped working after it, both of them
/// pre-run, before anything was even attempted:
///
/// * a bare name like `mlxfast-engine`, resolved by the spawner through `PATH` — `canonicalize`
///   resolves against the CWD, not `PATH`, so it fails on a name the run would have spawned fine;
/// * a binary that is executable but not readable by this user — `spawn` needs `--x--x--x`,
///   `read` needs `r`.
///
/// Neither is a broken run, and #123 exists to make artifacts say MORE about the runner, not to
/// turn working invocations into exit 1. So an unresolvable engine now seals the WEAKER identity
/// — the path exactly as given, no digest — and `candidate_executable_resolution` says which of
/// the two it is, so an empty digest is never ambiguous. If the engine really is broken, the
/// spawn a moment later reports it, in the words that actually describe it.
///
/// benchd's OWN path is best-effort for the same reason: `current_exe` can legitimately fail (a
/// deleted or relinked image), and losing a run over the identity of the binary that is reporting
/// it would be the wrong trade.
fn resolve_runner_identity(engine: &str) -> RunnerIdentity {
    let resolved = std::fs::canonicalize(engine)
        .ok()
        .and_then(|p| std::fs::read(&p).ok().map(|bytes| (p, bytes)));
    let (candidate_executable, candidate_executable_sha256, candidate_executable_resolution) =
        match resolved {
            Some((path, bytes)) => (
                path.display().to_string(),
                sha256_hex(&bytes),
                ENGINE_RESOLUTION_CANONICAL.to_string(),
            ),
            None => (
                engine.to_string(),
                String::new(),
                ENGINE_RESOLUTION_UNRESOLVED.to_string(),
            ),
        };
    let (benchd_executable, benchd_executable_sha256) = std::env::current_exe()
        .and_then(std::fs::canonicalize)
        .ok()
        .map(|p| {
            let sha = std::fs::read(&p)
                .map(|b| sha256_hex(&b))
                .unwrap_or_default();
            (p.display().to_string(), sha)
        })
        .unwrap_or_default();
    RunnerIdentity {
        candidate_executable,
        candidate_executable_sha256,
        candidate_executable_resolution,
        benchd_executable,
        benchd_executable_sha256,
    }
}

/// Write the integrity sidecar next to the score, named as benchmark.sh does per mode
/// (`benchmark-integrity.local-iterate.json` for local-iterate ONLY; `benchmark-integrity.json`
/// for local-submit and official). `jq -n` emits pretty (2-space) + a trailing newline; serde_json pretty
/// matches the 2-space form, and we append the newline.
fn write_integrity_sidecar(
    score_path: &Path,
    mode: Mode,
    sidecar: &IntegritySidecar,
) -> Result<(), String> {
    // Only local-ITERATE gets the `.local-iterate` suffix (benchmark.sh:135-137). local-SUBMIT
    // writes the DEFAULT `benchmark-integrity.json` (benchmark.sh:92-95); official likewise.
    let name = match mode {
        Mode::LocalIterate => "benchmark-integrity.local-iterate.json",
        Mode::LocalSubmit | Mode::Official => "benchmark-integrity.json",
    };
    let path = score_path.with_file_name(name);
    let json = serde_json::to_string_pretty(sidecar)
        .map_err(|e| format!("integrity serialization failed: {e}"))?;
    std::fs::write(&path, format!("{json}\n"))
        .map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(())
}

/// Parse `iterate` flags. `Ok(None)` means `--help` was requested.
/// Parse an `iterate` command line against [`ITERATE_CLI`].
fn parse_iterate_args(args: &[String]) -> Result<Option<IterateArgs>, String> {
    match parse_flags(args, ITERATE_CLI.flags)? {
        Some(flags) => build_iterate_args(&flags).map(Some),
        // `-h`/`--help`.
        None => Ok(None),
    }
}

/// Build the typed `iterate` arguments out of the parsed table: the per-flag conversions first (in
/// the order this function reads them, where the hand-written loop converted in argv order, so a
/// command line with two separately invalid values names the one the build reaches first), then the
/// cross-flag validation in the order the refusals have always been raised.
fn build_iterate_args(flags: &ParsedFlags) -> Result<IterateArgs, String> {
    let baseline_prefill_spt = flags.f64_flag("--baseline-prefill-spt")?;
    let baseline_decode_spt = flags.f64_flag("--baseline-decode-spt")?;
    let mode = flags
        .mapped("--mode", |_, v| {
            Mode::parse(v).ok_or_else(|| {
                format!("invalid --mode {v:?} (expected local-iterate, local-submit, or official)")
            })
        })?
        .unwrap_or(Mode::LocalIterate);
    // Tri-state (#60.3): the table already refused the two spellings together, so at most one is
    // present and the reader is a plain lookup.
    let cool_gate = if flags.present("--cool-gate") {
        Some(true)
    } else if flags.present("--no-cool-gate") {
        Some(false)
    } else {
        None
    };
    let capture_baseline = flags.path("--capture-baseline");
    let capture_timed_only = flags.present("--capture-timed-only");
    let capture_passes = flags.mapped("--capture-passes", |_, v| parse_capture_passes(v))?;
    let weights_digest = flags.mapped("--weights-digest", |_, v| parse_weights_digest(v))?;
    let contract = flags.path("--contract");
    let mut engine_resources: Vec<engine_resource::EngineResource> = Vec::new();
    for value in flags.values(engine_resource::ENGINE_RESOURCE_FLAG) {
        engine_resource::push_engine_resource(&mut engine_resources, value)?;
    }
    let mtp_depth = flags.mapped("--mtp-depth", |_, v| {
        v.parse::<u32>().map_err(|_| {
            format!(
                "invalid --mtp-depth {v:?}: mtp.depth is a u32 module field, so the value \
                 must be a non-negative integer that fits a u32"
            )
        })
    })?;
    let candidate_spec_json = flags.string("--candidate-spec");

    // Baselines are a PAIRED override (Swift `MLXFAST_PAIRED_BASELINE_*` must be provided
    // together); a lone `--baseline-*` is a usage error (exit 2 + usage). CONTRACT CHANGE:
    // pre-PR a single `--baseline-*` flag was accepted (it paired with a golden/constant
    // fallback); F2 removed the fallback, so the pair is now required together.
    if baseline_prefill_spt.is_some() != baseline_decode_spt.is_some() {
        return Err(
            "--baseline-prefill-spt and --baseline-decode-spt must be given together".to_string(),
        );
    }
    // B5 MODE FENCE. `--capture-baseline` is the LOCAL-ITERATE capture mode and nothing else. It is
    // refused at PARSE, by name, for every other mode: the mode's whole safety argument is that it
    // runs the local checked-timing legs without resolving an official baseline and writes no
    // score, and neither half of that holds on `local-submit` (a different decode window, so a
    // different pair) or on `official` (the ranked, sealed path).
    if capture_baseline.is_some() && mode != Mode::LocalIterate {
        return Err(format!(
            "--capture-baseline is the local-iterate CAPTURE MODE (it authors the official \
             baseline's capture record from the checked-timing leg and writes no score); mode \
             {:?} is refused by name",
            mode.mode_name()
        ));
    }
    // --capture-timed-only is a MODIFIER of --capture-baseline (a8 ruling): it skips the
    // teacher-forced correctness gate for passes 2-4/B, so it is meaningless — and refused by
    // name — without --capture-baseline. This keeps the timed-only skip scoped strictly to the
    // capture mode; it can never reach a scored/official run.
    if capture_timed_only && capture_baseline.is_none() {
        return Err(
            "--capture-timed-only requires --capture-baseline: it is the capture mode's \
             correctness-gate skip for the passes AFTER the first per-prompt gate (a8 ruling), \
             and has no meaning outside --capture-baseline"
                .to_string(),
        );
    }
    // --capture-passes is a MODIFIER of --capture-baseline (a8 Option-A restructure), gated
    // identically to --capture-timed-only: refused by name WITHOUT --capture-baseline, so the
    // multi-pass-over-one-residency path can never reach a scored/official run (official refuses
    // --capture-baseline at parse above).
    if capture_passes.is_some() && capture_baseline.is_none() {
        return Err(
            "--capture-passes requires --capture-baseline: it runs the listed calibration passes \
             over ONE persistent model residency (a8 Option-A restructure), authoring the official \
             baseline's per-label capture records, and has no meaning outside --capture-baseline"
                .to_string(),
        );
    }
    // --capture-passes SUBSUMES --capture-timed-only: in multi-pass mode EVERY pass is already
    // timed-only (gateless — no TF gate on any pass), so the single-invocation --capture-timed-only
    // modifier is meaningless alongside it and is refused rather than silently ignored.
    if capture_passes.is_some() && capture_timed_only {
        return Err(
            "--capture-passes conflicts with --capture-timed-only: --capture-passes decides the \
             per-pass correctness-gate skip positionally (first pass gated, the rest timed-only), \
             so the single-invocation --capture-timed-only flag has no meaning alongside it"
                .to_string(),
        );
    }
    // RIDER 1 (Option B digest-hoist) — --weights-digest is a MODIFIER of --capture-baseline,
    // gated identically to --capture-timed-only above: refused by name WITHOUT --capture-baseline.
    // A passed-in weights digest may NEVER reach a scored seal: an official (scoring) run always
    // hashes for itself, and official already refuses --capture-baseline at parse, so this refusal
    // keeps the injected digest scoped strictly to the capture window.
    if weights_digest.is_some() && capture_baseline.is_none() {
        return Err(
            "--weights-digest requires --capture-baseline: it injects the window's once-computed \
             weights digest into a capture pass to skip the per-pass re-hash (Option B), and has \
             no meaning outside --capture-baseline — an official/scored run always hashes for \
             itself"
                .to_string(),
        );
    }
    // ARM GATE prerequisite (David 2026-08-26): the SOLE scored path (`--mode official`) REQUIRES a
    // --contract track fixture so the arm gate can read `official_scoring_enabled`. Refuse at parse,
    // fail-closed — an official run with no contract has no arm state to consult, and an absent arm
    // state is never armed. The local modes never seal a score, so they do not require it.
    if mode == Mode::Official && contract.is_none() {
        return Err(
            "--mode official requires --contract <track-fixture.json>: the arm gate refuses to seal \
             an official/ranked score for a track whose fixture does not declare \
             official_scoring_enabled: true (David 2026-08-26)"
                .to_string(),
        );
    }

    // SPEC RESOLUTION — the SAME contract the measure-job surface already carries, reused rather
    // than re-invented, so an operator (and the engine's measure-and-score.sh) writes one spelling
    // for both. `--mtp-depth N` is the CONVENIENCE that builds `{"mode":"mtp","mtp":{"depth":N}}`;
    // `--candidate-spec <JSON>` is the EXPLICIT override. They are MUTUALLY EXCLUSIVE: silently
    // discarding one would hide an operator wiring conflict.
    if candidate_spec_json.is_some() && mtp_depth.is_some() {
        return Err(
            "--mtp-depth and --candidate-spec are mutually exclusive: --candidate-spec is the \
             explicit spec, and --mtp-depth is only the convenience that builds the default one — \
             pass exactly one (drop --mtp-depth, or fold the depth into --candidate-spec)"
                .to_string(),
        );
    }
    let spec =
        match (&candidate_spec_json, mtp_depth) {
            (Some(json), _) => Some(measure_job::parse_spec_override(json)?),
            // Depth 0 is the SERIAL control, not an mtp candidate: `{"mode":"mtp","mtp":{"depth":0}}`
            // is refused by the module-coherence gate below, so `--mtp-depth 0` is spelled as what it
            // is — no flag at all. Naming it explicitly is clearer than an opaque coherence refusal.
            (None, Some(0)) => return Err(
                "--mtp-depth 0 is the SERIAL leg, which is what benchd runs when no spec flag is \
                 given: omit --mtp-depth rather than requesting depth 0 (an mtp spec at depth 0 is \
                 refused as incoherent)"
                    .to_string(),
            ),
            (None, Some(depth)) => Some(bench_protocol::SpecConfig::mtp(depth)),
            // ABSENT ⇒ NO SPEC ON THE WIRE — today's behaviour, byte for byte.
            (None, None) => None,
        };
    if let Some(spec) = spec.as_ref() {
        // The DEFENSIVE draft-depth cap, resolved by the measure-job rule: the OFFICIAL/scored path
        // uses the readonly constant and IGNORES MLXFAST_MAX_DRAFT_DEPTH (an env override can never
        // widen a scored submission); the LOCAL dev modes honour the override.
        let cap = measure_job::resolve_max_draft_depth_cap(
            mode != Mode::Official,
            std::env::var(measure_job::MAX_DRAFT_DEPTH_ENV)
                .ok()
                .as_deref(),
        );
        measure_job::validate_spec_capped(spec, cap)?;
        // Shape fail-closed: exactly the ONE module block matching the mode, and an mtp block with
        // a depth >= 1. This is the shape gate only — WHICH modes a track admits is the contract's
        // allowed-modes decision on the paired surface, and the engine's own `hello.spec_modes`
        // gate refuses an unrunnable mode at the runner seam before the timed seed forward.
        measure_job::validate_spec_module_coherent(spec)?;
    }
    // The SERIAL-LEG refusal at the door the spec comes in by. A capture leg authors the official
    // baseline, which is the SERIAL denominator every scored speculative leg is divided by, so a
    // spec-armed capture would pin a speculative pair as the serial reference. `--capture-passes`
    // already forces `None` on the wire; the SINGLE-pass capture path threads `args.spec` straight
    // into the timed leg, so that combination is the one that could seal a speculative pair. It
    // refuses here, by name, rather than being silently dropped. (The engine's own arming — a serve
    // started with speculation on — is caught after the leg by `capture::refuse_spec_armed_engine`,
    // which reads the effective_spec ECHO; this gate covers the REQUEST.)
    if capture_baseline.is_some() && spec.is_some() {
        return Err(format!(
            "{}: --capture-baseline refuses --mtp-depth/--candidate-spec. The captured pair \
             BECOMES the official baseline — the serial denominator every scored speculative leg \
             is divided by — so a capture leg is always serial; drop the spec flag",
            capture::CALIBRATION_SPEC_ARMED
        ));
    }

    let engine = flags
        .string("--engine")
        .ok_or_else(|| flags.missing_required("--engine"))?;
    let weights = flags
        .path("--weights")
        .ok_or_else(|| flags.missing_required("--weights"))?;
    let goldens = flags.paths("--golden");
    if goldens.is_empty() {
        return Err(flags.missing_required("--golden"));
    }
    let golden_pins = parse_golden_pins(
        flags.values("--golden-sha256"),
        flags.values("--golden-bytes"),
        goldens.len(),
        ["--golden-sha256", "--golden-bytes", "--golden"],
    )?;
    // Control golden `i` belongs to golden `i`, so there is one per golden or none.
    let control_goldens = flags.paths("--control-golden");
    if !control_goldens.is_empty() && control_goldens.len() != goldens.len() {
        return Err(format!(
            "{}: --control-golden was given {} time(s) and --golden {} time(s); give one \
             --control-golden per --golden, in the same order, or none",
            baseline::GOLDEN_COUNT_MISMATCH,
            control_goldens.len(),
            goldens.len()
        ));
    }
    // The control goldens are pinned by the SAME rule as the candidate goldens.
    let control_golden_pins = parse_golden_pins(
        flags.values("--control-golden-sha256"),
        flags.values("--control-golden-bytes"),
        control_goldens.len(),
        [
            "--control-golden-sha256",
            "--control-golden-bytes",
            "--control-golden",
        ],
    )?;
    // Default score name mirrors benchmark.sh: local-ITERATE writes `score.local-iterate.json`
    // (benchmark.sh:92-95); local-SUBMIT writes the DEFAULT `score.json`, as does official.
    let score_path = flags.path("--score-path").unwrap_or_else(|| {
        PathBuf::from(match mode {
            Mode::LocalIterate => "score.local-iterate.json",
            Mode::LocalSubmit | Mode::Official => "score.json",
        })
    });

    Ok(IterateArgs {
        engine,
        weights,
        goldens,
        golden_pins,
        baseline_prefill_spt,
        baseline_decode_spt,
        mode,
        score_path,
        cool_gate,
        strict: flags.present("--strict"),
        capture_baseline,
        capture_timed_only,
        capture_passes,
        weights_digest,
        contract,
        spec,
        baseline_workspace: flags.path("--baseline-workspace"),
        baseline_calibration: flags.path("--baseline-calibration"),
        box_name: flags.string("--box"),
        control_goldens,
        control_golden_pins,
        engine_resources,
    })
}

/// Parse a `--capture-passes` spec: a comma list of per-pass record LABELS (e.g. `W,A,A,B,B`).
///
/// Each label names the record file that pass's pair is appended to (`<base>.<label>.<ext>` off the
/// `--capture-baseline` base path), so a repeated label (the two `A`s) merges its passes into ONE
/// record — exactly as two separate `--capture-baseline …A.json` invocations do today. Labels are
/// restricted to `[A-Za-z0-9_-]+` so every label is a safe single filename component (no separators,
/// no `.`/`..` traversal). An empty spec, an empty label, or an out-of-charset label is refused.
fn parse_capture_passes(spec: &str) -> Result<Vec<String>, String> {
    let labels: Vec<String> = spec.split(',').map(|s| s.trim().to_string()).collect();
    if labels.iter().any(|l| l.is_empty()) {
        return Err(format!(
            "invalid --capture-passes {spec:?}: expected a comma list of non-empty pass labels \
             (e.g. W,A,A,B,B); an empty label is not allowed"
        ));
    }
    for label in &labels {
        if !label
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(format!(
                "invalid --capture-passes label {label:?}: a pass label must be a single filename \
                 component of [A-Za-z0-9_-] (it names the per-pass record file <base>.<label>.json)"
            ));
        }
    }
    Ok(labels)
}

/// The per-label capture record path for one pass: insert `.<label>` before the `--capture-baseline`
/// base path's final extension, matching the driver's existing `baseline.<prompt-id>.<label>.json`
/// naming. `baseline.p1.json` + `A` → `baseline.p1.A.json`; a base with no extension appends
/// `.<label>`. `label` is charset-validated by [`parse_capture_passes`], so it is always one safe
/// filename component.
fn capture_pass_record_path(base: &Path, label: &str) -> PathBuf {
    let stem = base
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("baseline");
    let file_name = match base.extension().and_then(|e| e.to_str()) {
        Some(ext) => format!("{stem}.{label}.{ext}"),
        None => format!("{stem}.{label}"),
    };
    base.with_file_name(file_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE WINDOW THESE TESTS MEASURE UNDER — the 125B MLX track's own, as TEST DATA. STEP 3
    /// (David 2026-09-15) deleted `WindowShape::table`, so a test that needs a window states one;
    /// this is the same window `crate::iterate::test_window` states, named once.
    fn test_window() -> bench_core::constants::WindowShape {
        crate::iterate::test_window()
    }

    /// A local run with no reference tree stays exact. When the track declares a tolerance, its
    /// timed token refusal says that the ranked run tolerates up to N per thousand and that this
    /// run cannot apply the rule. Nothing else changes.
    #[test]
    fn a_local_token_refusal_names_the_ranked_tolerance() {
        let refused = |error: &str| ScorePayload {
            score: None,
            passed: false,
            metrics: crate::score::ScoreMetrics {
                error: error.to_string(),
                ..Default::default()
            },
        };
        let mismatch = "benchmark free-run decode token mismatch at step 3";
        let mut payload = refused(mismatch);
        explain_local_exact_token_rule(&mut payload, Some(100));
        let error = &payload.metrics.error;
        assert!(error.starts_with(mismatch), "{error}");
        assert!(
            error.contains("tolerates up to 100 per thousand"),
            "{error}"
        );
        assert!(
            error.contains("a local run without a reference tree cannot apply that rule"),
            "{error}"
        );

        let mut exact = refused(mismatch);
        explain_local_exact_token_rule(&mut exact, None);
        assert_eq!(exact.metrics.error, mismatch);

        let other = "acceptance band failed";
        let mut unrelated = refused(other);
        explain_local_exact_token_rule(&mut unrelated, Some(100));
        assert_eq!(unrelated.metrics.error, other);
    }

    /// THE NO-FIXTURE CONTRACT: a run given no `--contract` declares nothing, so every resolver
    /// REFUSES (STEP 3, David 2026-09-15). Tests that need a value hand in a fixture; this is the
    /// exact behaviour these tests pinned before STEP 1 existed. A test that wants the CONTRACT arm
    /// builds its own fixture and says so.
    static NO_FIXTURE: &contract::Contract = &contract::NO_FIXTURE;

    /// A FIXTURE FOR A STORED-PAIR TRACK, as test data: `qwen3.8-27b-mtp-v1`'s own declaration.
    ///
    /// STEP 3 (David 2026-09-15) deleted the per-track tables, so the tests that used to lean on a
    /// table row state the same values here. The pair is the captured one, at the precision it was
    /// captured at, and it matches `tests/fixtures/contract-full/qwen3.8-27b-mtp-v1.json` — the
    /// oracle pins that file against the same numbers.
    fn stored_pair_fixture() -> contract::Contract {
        contract::Contract {
            track_id: Some(bench_core::constants::TRACK_ID.to_string()),
            scored_batch_size: Some(1),
            prefill_gain_exponent: Some(0.0),
            decode_gain_exponent: Some(1.0),
            scores_against_live_control_leg: Some(false),
            paired_flow_retired: Some(false),
            official_baseline_prefill_seconds_per_token: Some(0.00036751938916015625),
            official_baseline_decode_seconds_per_token: Some(0.01385621216015625),
            prefill_band_up_tolerance: Some(0.03),
            prefill_band_down_tolerance: Some(0.03),
            decode_band_up_tolerance: Some(0.01),
            decode_band_down_tolerance: Some(0.025),
            decode_band_down_enabled: Some(true),
            prefill_band_down_enabled: Some(true),
            score_decode_weight: Some(0.75),
            score_prefill_weight: Some(0.25),
            ..contract::Contract::NONE_DECLARED
        }
    }

    /// A FIXTURE FOR A LIVE-CONTROL-LEG TRACK: it measures its own denominator, so it declares the
    /// two flow flags and its own band shape and NO stored pair. `track` names which one.
    fn live_control_leg_fixture(track: &str) -> contract::Contract {
        contract::Contract {
            track_id: Some(track.to_string()),
            scores_against_live_control_leg: Some(true),
            paired_flow_retired: Some(true),
            prefill_band_up_tolerance: Some(0.05),
            prefill_band_down_tolerance: Some(0.05),
            decode_band_up_tolerance: Some(0.02),
            decode_band_down_tolerance: Some(0.05),
            decode_band_down_enabled: Some(false),
            prefill_band_down_enabled: Some(false),
            score_decode_weight: Some(0.75),
            score_prefill_weight: Some(0.25),
            ..contract::Contract::NONE_DECLARED
        }
    }

    /// The two tracks that measure their own denominator, as test data (they were
    /// `LIVE_CONTROL_LEG_TRACKS`).
    const LIVE_CONTROL_LEG_TRACKS: [&str; 2] =
        ["qwen3.8-125b-a6b-mlx-v1", "qwen3.8-125b-a6b-cuda-v1"];

    /// M-3 — FAILING-RUN ARTIFACT COMPLETENESS. A run that fails (correctness or preflight:
    /// `passed=false`, `score=null`) must write the SAME artifact set as a passing run —
    /// `score.json`, its `.sha256` sidecar, AND the `benchmark-integrity.*.json` — byte-shaped
    /// identically; only the payload's `passed`/`score` differ. This locks the invariant that
    /// benchd's fail path reaches the same `write_score`/`write_integrity_sidecar` as the
    /// pass path (no early-return skips an artifact), matching the Swift reference: Swift
    /// `localIterate` catches every failure into a `failedScore` payload (never throws), so
    /// `writeScorePayload`+`emitScorePayloadToStdout` always run and the binary exits 0; then
    /// `benchmark.sh` writes BOTH sidecars UNCONDITIONALLY (`:1269-1270` `.sha256`, `:1289-1308`
    /// integrity JSON) and only THEN exits 1 on `.passed != true` (`:1310-1312`). The failing
    /// artifacts are written before that exit, exactly as benchd writes them before returning
    /// `Ok(false)` → exit 1.
    #[test]
    fn failing_iterate_writes_full_byte_shaped_artifact_set() {
        use crate::iterate::{preflight_failed_payload, DirDigest, Mode};
        use crate::score::ScorePayload;

        // A genuine FAILING run: the §F2 missing-baseline PREFLIGHT failure (score=null,
        // passed=false, passed_correctness=false) — the same payload execute_iterate feeds to
        // the common writers when the golden carries no baselines and no --baseline flags.
        // A benchmark-less golden: this test only needs a loadable fixture to hang the
        // preflight refusal on, and that refusal is precisely "no baselines".
        let doc = crate::testgolden::TestGolden::new().without_benchmark();
        let bytes = doc.bytes();
        let golden = doc.fixture();
        let weights = DirDigest::empty();
        let failing = preflight_failed_payload(
            Mode::LocalIterate,
            &golden,
            RunDigests::for_test(&weights),
            "local-iterate requires external Qwen benchmark baselines".to_string(),
            ScoringInputs::local(
                crate::testgolden::TEST_BASELINE.prefill_seconds_per_token,
                crate::testgolden::TEST_BASELINE.decode_seconds_per_token,
            ),
        );
        assert!(!failing.passed, "precondition: this is a FAILING payload");
        assert!(
            failing.score.is_none(),
            "precondition: a failing run has no score"
        );

        // A passing-shaped twin (same metrics, flipped verdict) proves the SAME three
        // filenames are produced whether the run passed or failed — the completeness claim.
        let passing = ScorePayload {
            score: Some(1.5),
            passed: true,
            metrics: failing.metrics.clone(),
        };

        // Exercise the EXACT common write sequence execute_iterate runs after building the
        // payload: to_sealed_json -> write_score (score.json + .sha256) -> write_integrity_sidecar.
        let write_artifacts = |payload: &ScorePayload, dir: &Path| -> (PathBuf, String) {
            let score_path = dir.join("score.local-iterate.json");
            let json = payload.to_sealed_json().unwrap();
            let score_sha256 = write_score(&score_path, &json).unwrap();
            let golden_sha256 = sha256_hex(&bytes);
            write_integrity_sidecar(
                &score_path,
                Mode::LocalIterate,
                &IntegritySidecar {
                    score_path: score_path.display().to_string(),
                    score_sha256: score_sha256.clone(),
                    weights_path: "weights".to_string(),
                    weights_sha256: weights.sha256.clone(),
                    weights_file_count: weights.file_count,
                    weights_byte_count: weights.byte_count,
                    golden_path: "[private]".to_string(),
                    golden_sha256,
                    transform_source_sha256: String::new(),
                    candidate_executable: "engine".to_string(),
                    candidate_executable_sha256: "e0".to_string(),
                    baseline_executable: String::new(),
                    baseline_executable_sha256: String::new(),
                    candidate_executable_resolution: ENGINE_RESOLUTION_CANONICAL.to_string(),
                    benchd_executable: "benchd".to_string(),
                    benchd_executable_sha256: "b0".to_string(),
                    candidate_workspace_sha256: String::new(),
                    sandbox: SANDBOX_PROVENANCE_NONE.to_string(),
                },
            )
            .unwrap();
            (score_path, json)
        };

        let fail_dir = std::env::temp_dir().join(format!(
            "benchd-m3-fail-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let pass_dir = std::env::temp_dir().join(format!(
            "benchd-m3-pass-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&fail_dir);
        let _ = std::fs::remove_dir_all(&pass_dir);
        std::fs::create_dir_all(&fail_dir).unwrap();
        std::fs::create_dir_all(&pass_dir).unwrap();

        let (fail_score_path, fail_json) = write_artifacts(&failing, &fail_dir);
        let (_pass_score_path, _pass_json) = write_artifacts(&passing, &pass_dir);

        // The three artifact filenames present after a FAILING run.
        let score = fail_dir.join("score.local-iterate.json");
        let sidecar = fail_dir.join("score.local-iterate.json.sha256");
        let integrity = fail_dir.join("benchmark-integrity.local-iterate.json");
        assert!(score.exists(), "FAILING run must still write score.json");
        assert!(
            sidecar.exists(),
            "FAILING run must still write the .sha256 sidecar"
        );
        assert!(
            integrity.exists(),
            "FAILING run must still write the integrity JSON sidecar"
        );

        // Completeness: the passing twin writes the SAME three filenames — no artifact is
        // skipped on failure (nor added).
        let names_in = |dir: &Path| -> Vec<String> {
            let mut v: Vec<String> = std::fs::read_dir(dir)
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
                .collect();
            v.sort();
            v
        };
        assert_eq!(
            names_in(&fail_dir),
            names_in(&pass_dir),
            "a FAILING run must write the identical artifact SET as a passing run"
        );

        // 1. score.json bytes == the sealed JSON, and it encodes the failure verbatim.
        let on_disk = std::fs::read_to_string(&score).unwrap();
        assert_eq!(
            on_disk, fail_json,
            "score.json is the sealed JSON byte-for-byte"
        );
        assert!(
            on_disk.contains("\"passed\": false"),
            "failing score.json records passed=false"
        );
        assert!(
            on_disk.contains("\"score\": null"),
            "failing score.json records score=null"
        );

        // 2. The .sha256 sidecar is the shasum two-space form `<hex>  <score_path>\n` over the
        //    ON-DISK score bytes (byte-matching benchmark.sh:1269-1270), unchanged by failure.
        let sidecar_bytes = std::fs::read_to_string(&sidecar).unwrap();
        let expected_hex = sha256_hex(fail_json.as_bytes());
        assert_eq!(
            sidecar_bytes,
            format!("{expected_hex}  {}\n", fail_score_path.display()),
            "failing .sha256 is `<hex>  <path>\\n` over the score bytes"
        );

        // 3. The integrity sidecar is well-formed on failure: exactly the 9 fields, in order,
        //    and score_sha256 pins the SAME digest the .sha256 sidecar carries.
        let integrity_bytes = std::fs::read_to_string(&integrity).unwrap();
        assert!(
            integrity_bytes.ends_with("}\n"),
            "integrity JSON ends with the jq trailing newline"
        );
        // The 9 fields appear in benchmark.sh insertion order (a parsed Value would re-sort,
        // so assert order over the raw serialized bytes).
        let field_order = [
            "score_path",
            "score_sha256",
            "weights_path",
            "weights_sha256",
            "weights_file_count",
            "weights_byte_count",
            "golden_path",
            "golden_sha256",
            "transform_source_sha256",
        ];
        let mut cursor = 0usize;
        for field in field_order {
            let needle = format!("\"{field}\":");
            let at = integrity_bytes[cursor..].find(&needle).unwrap_or_else(|| {
                panic!("failing integrity sidecar missing {field} in benchmark.sh order")
            });
            cursor += at + needle.len();
        }
        let v: serde_json::Value = serde_json::from_str(&integrity_bytes).unwrap();
        let obj = v.as_object().unwrap();
        // #123: the 9 reference fields + the runner-identity roster (8 since #132/F3 added the
        // resolution sentinel). The roster itself is pinned by
        // `integrity_sidecar_is_a_superset_of_the_jq_pretty_object`, and its LENGTH is read from
        // the same single-source file rather than restated, so this count cannot drift from it.
        // The claim HERE is only that a FAILING run writes the same shape a passing one does.
        const ROSTER_DOC: &str =
            include_str!("../../../scripts/fixtures/integrity-runner-keys.json");
        let roster_len = serde_json::from_str::<serde_json::Value>(ROSTER_DOC).unwrap()["keys"]
            .as_array()
            .unwrap()
            .len();
        assert_eq!(
            obj.len(),
            9 + roster_len,
            "failing integrity sidecar must carry the 9 reference fields + the full runner roster"
        );
        assert_eq!(
            obj["score_sha256"].as_str().unwrap(),
            expected_hex,
            "integrity score_sha256 matches the score bytes' digest on a failing run"
        );

        let _ = std::fs::remove_dir_all(&fail_dir);
        let _ = std::fs::remove_dir_all(&pass_dir);
    }

    /// #123 (RULED David 2026-08-20, EXTEND THE SIDECAR) — the sidecar is now a strict SUPERSET
    /// of `benchmark.sh`'s `jq -n` object, not a byte-for-byte twin, and the old byte-match test
    /// is re-graded to say exactly that.
    ///
    /// The claim it still enforces is the one the superset has to keep: the reference's nine
    /// fields come FIRST, in the reference's `jq -n` order, with the reference's names, types and
    /// values — so the reference's object is a literal PREFIX of benchd's bytes, and a consumer
    /// reading those nine reads them unchanged. What was a whole-document equality is now a
    /// prefix equality plus an exhaustive roster for the extension, so neither half can drift
    /// silently: a reordered/renamed reference field breaks the prefix, and an extra field nobody
    /// declared breaks the roster.
    #[test]
    fn integrity_sidecar_is_a_superset_of_the_jq_pretty_object() {
        let s = IntegritySidecar {
            score_path: "score.local-iterate.json".into(),
            score_sha256: "aaa".into(),
            weights_path: "weights".into(),
            weights_sha256: "fde4f615".into(),
            weights_file_count: 14,
            weights_byte_count: 15_159_954_417,
            golden_path: "[private]".into(),
            golden_sha256: "32045f7e".into(),
            transform_source_sha256: String::new(),
            candidate_executable: "/opt/engine/mlxfast-engine".into(),
            candidate_executable_sha256: "c0ffee".into(),
            baseline_executable: String::new(),
            baseline_executable_sha256: String::new(),
            candidate_executable_resolution: ENGINE_RESOLUTION_CANONICAL.into(),
            benchd_executable: "/opt/bin/benchd".into(),
            benchd_executable_sha256: "b0b0".into(),
            candidate_workspace_sha256: String::new(),
            sandbox: SANDBOX_PROVENANCE_NONE.into(),
        };
        let got = format!("{}\n", serde_json::to_string_pretty(&s).unwrap());

        // The reference's object, verbatim, minus its closing brace — benchd's bytes must open
        // with exactly this.
        let reference_prefix = "{\n  \"score_path\": \"score.local-iterate.json\",\n  \"score_sha256\": \"aaa\",\n  \"weights_path\": \"weights\",\n  \"weights_sha256\": \"fde4f615\",\n  \"weights_file_count\": 14,\n  \"weights_byte_count\": 15159954417,\n  \"golden_path\": \"[private]\",\n  \"golden_sha256\": \"32045f7e\",\n  \"transform_source_sha256\": \"\"";
        assert!(
            got.starts_with(reference_prefix),
            "the reference's 9 fields must remain a byte-exact PREFIX of the sidecar; got:\n{got}"
        );

        // …and the extension is exactly the declared runner-identity roster, nothing more, in
        // order. Read off the EMITTED BYTES rather than a parsed map: `serde_json::Value` sorts
        // its keys, which would silently discard the very ordering this test exists to hold.
        let keys: Vec<&str> = got
            .lines()
            .filter_map(|l| l.trim().strip_prefix('"'))
            .filter_map(|l| l.split_once("\":"))
            .map(|(k, _)| k)
            .collect();
        const REFERENCE_KEYS: [&str; 9] = [
            "score_path",
            "score_sha256",
            "weights_path",
            "weights_sha256",
            "weights_file_count",
            "weights_byte_count",
            "golden_path",
            "golden_sha256",
            "transform_source_sha256",
        ];
        assert_eq!(
            &keys[..REFERENCE_KEYS.len()],
            &REFERENCE_KEYS[..],
            "the reference's 9 fields changed name or order"
        );

        // C3 — the runner roster is NOT restated here. It is read from the SAME file the two live
        // parity legs read (`facade-leg.sh`, `official-parity.sh`), so the shell and Rust encodings
        // cannot drift apart: a key added to the struct but not the roster fails here, and a key
        // added to the roster but not the struct fails here too. What this test still states in its
        // own words is the ORDER, which the roster file deliberately does not describe.
        const ROSTER: &str = include_str!("../../../scripts/fixtures/integrity-runner-keys.json");
        let roster_doc: serde_json::Value = serde_json::from_str(ROSTER).unwrap();
        let roster: Vec<String> = roster_doc["keys"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        // #132/F6 — the file documents itself as SORTED, because both shell legs compare it
        // against `jq 'keys'` output, which is sorted. That was documentation only; an unsorted
        // edit would have broken the live legs at a distance instead of here.
        let mut sorted = roster.clone();
        sorted.sort();
        assert_eq!(
            roster, sorted,
            "scripts/fixtures/integrity-runner-keys.json must stay sorted — the shell legs \
             compare it against jq 'keys' output, which is"
        );
        let mut surplus: Vec<String> = keys[REFERENCE_KEYS.len()..]
            .iter()
            .map(|k| k.to_string())
            .collect();
        surplus.sort();
        assert_eq!(
            surplus, roster,
            "the sidecar's runner roster and scripts/fixtures/integrity-runner-keys.json disagree — \
             the live parity legs read that file, so this is the shell check going out of sync"
        );
        assert_eq!(
            keys.len(),
            REFERENCE_KEYS.len() + roster.len(),
            "the sidecar roster changed without the parity-matrix row changing with it"
        );

        // Backward-readability, mechanically: the only in-repo reader round-trips through
        // `serde_json::Value` and rewrites two keys. Prove the extension survives that.
        let mut round: serde_json::Value = serde_json::from_str(&got).unwrap();
        let obj = round.as_object_mut().unwrap();
        obj.insert("score_sha256".into(), serde_json::json!("rewritten"));
        obj.insert("score_path".into(), serde_json::json!("elsewhere.json"));
        let after = serde_json::to_string_pretty(&round).unwrap();
        let after: serde_json::Value = serde_json::from_str(&after).unwrap();
        assert_eq!(
            after["candidate_executable_sha256"],
            serde_json::json!("c0ffee")
        );
        assert_eq!(
            after["benchd_executable"],
            serde_json::json!("/opt/bin/benchd")
        );
        assert_eq!(after["weights_sha256"], serde_json::json!("fde4f615"));
    }

    #[test]
    fn resolve_paired_baselines_requires_golden_or_flags() {
        // §F2: a golden must carry both benchmark baselines (or explicit flags override);
        // otherwise resolution yields None → a preflight-failed score (Swift's behavior).
        let golden = |with_baselines: bool| {
            let g = crate::testgolden::TestGolden::new();
            let g = if with_baselines {
                g.baselines(0.01, 0.1)
            } else {
                g
            };
            g.fixture()
        };
        let with = golden(true);
        let without = golden(false);
        // Golden carries the paired baselines → resolved from the golden (no override).
        assert_eq!(resolve_paired_baselines(None, &with), Some((0.01, 0.1)));
        // Golden lacks them, no override → None (preflight fail; no official-constant fallback).
        assert_eq!(resolve_paired_baselines(None, &without), None);
        // An explicit paired override wins even over a baseline-less golden.
        assert_eq!(
            resolve_paired_baselines(Some((0.02, 0.2)), &without),
            Some((0.02, 0.2))
        );
    }

    #[test]
    fn half_set_baseline_flags_are_a_usage_error() {
        // Review fix: a lone --baseline-* is a USAGE error at parse (→ exit 2 + usage), not a
        // silent exit-1. Both-together or neither is fine.
        let mk = |extra: &[&str]| -> Vec<String> {
            let mut v: Vec<String> = ["--engine", "e", "--weights", "w", "--golden", "g"]
                .iter()
                .map(|s| s.to_string())
                .collect();
            v.extend(extra.iter().map(|s| s.to_string()));
            v
        };
        assert!(parse_iterate_args(&mk(&["--baseline-prefill-spt", "0.01"])).is_err());
        assert!(parse_iterate_args(&mk(&["--baseline-decode-spt", "0.1"])).is_err());
        assert!(parse_iterate_args(&mk(&[
            "--baseline-prefill-spt",
            "0.01",
            "--baseline-decode-spt",
            "0.1"
        ]))
        .unwrap()
        .is_some());
        assert!(parse_iterate_args(&mk(&[])).unwrap().is_some());
    }

    /// THE LIVE-CONTROL-LEG TRACKS, on every path that is NOT the paired one (David 2026-09-08).
    ///
    /// These two tracks store no pair anywhere, so every path that would have read one now answers
    /// for itself, and each answer is asserted BY VALUE or BY NAME:
    ///
    /// * the LOCAL legs resolve to `Unscored` — they measure the candidate leg and seal no score,
    ///   so a participant on a laptop keeps a working benchmark and nobody gets a number that
    ///   looks like a rank;
    /// * the GATES-ONLY official seam resolves to the ZERO placeholders, because it measures no
    ///   leg and there is no stored pair to fill in;
    /// * `--capture-baseline` refuses BY NAME: there is no pair to capture, and the verb that
    ///   replaced it is `calibrate-baseline`.
    ///
    /// NEGATIVE CONTROLS: the STORED-PAIR tracks are untouched on all three.
    #[test]
    fn the_live_control_leg_tracks_store_no_pair_on_any_other_path() {
        use crate::iterate::Mode;

        // A golden that declares NO pair: it cannot mask which source answered.
        let golden = crate::testgolden::TestGolden::new().fixture();
        assert_eq!(
            resolve_paired_baselines(None, &golden),
            None,
            "precondition: the golden declares no pair"
        );

        for track in LIVE_CONTROL_LEG_TRACKS {
            let fixture = live_control_leg_fixture(track);
            assert!(
                bench_core::contract::official_baseline(&fixture, track).is_err(),
                "precondition: {track} stores no pair"
            );
            for mode in [Mode::LocalIterate, Mode::LocalSubmit] {
                assert_eq!(
                    run_baselines(&fixture, mode, &golden, None, Some(track)).unwrap(),
                    RunBaselines::Unscored,
                    "{track}/{}: a local leg measures the candidate and seals no score",
                    mode.mode_name()
                );
                // Not even a `--baseline-*` flag turns it into a scored run: the flags are a
                // STORED pair, and this track has no source for one.
                assert_eq!(
                    run_baselines(&fixture, mode, &golden, Some((0.5, 0.6)), Some(track)).unwrap(),
                    RunBaselines::Unscored,
                    "{track}/{}: a stored pair must not score this track",
                    mode.mode_name()
                );
            }
            assert_eq!(
                gates_only_baselines(&fixture, Some(track), None, &golden).unwrap(),
                (0.0, 0.0),
                "{track}: a gates-only run measured no leg, so it seals no pair"
            );
            // Even a trusted override does not put a denominator on this track's gates-only seam:
            // the timed path refuses that override outright, so honouring it here would be the one
            // place a stored pair still reached these tracks.
            assert_eq!(
                gates_only_baselines(&fixture, Some(track), Some((0.5, 0.6)), &golden).unwrap(),
                (0.0, 0.0),
                "{track}: gates-only must not absorb a stored override"
            );
            let err = capture::refuse_live_control_leg_track(&fixture, track)
                .expect_err("there is no pair to capture on a live-control-leg track");
            assert!(
                err.contains(capture::CAPTURE_RETIRED_FOR_LIVE_CONTROL_LEG),
                "{track}: {err}"
            );
            assert!(err.contains("calibrate-baseline"), "{track}: {err}");
            // OFFICIAL still DEFERS in `run_baselines` — the paired arm in `execute_iterate` is
            // what answers, and it answers by measuring.
            assert_eq!(
                run_baselines(&fixture, Mode::Official, &golden, None, Some(track)).unwrap(),
                RunBaselines::ResolveFromOverrideOrGolden
            );
        }

        // NEGATIVE CONTROLS — a stored-pair track is untouched on all three.
        const STORED: &str = bench_core::constants::TRACK_ID;
        let stored_fixture = stored_pair_fixture();
        assert!(
            !bench_core::contract::scores_against_live_control_leg(&stored_fixture, STORED)
                .unwrap()
                .value
        );
        assert!(capture::refuse_live_control_leg_track(&stored_fixture, STORED).is_ok());
        assert!(run_baselines(
            &stored_fixture,
            Mode::LocalIterate,
            &golden,
            None,
            Some(STORED)
        )
        .is_ok());
        assert_ne!(
            gates_only_baselines(&stored_fixture, Some(STORED), None, &golden).unwrap(),
            (0.0, 0.0)
        );
    }

    /// THE TWO LOCAL BRANCHES of a live-control-leg track, at the switch that chooses between
    /// them.
    ///
    /// With NO reference tree in reach a local run is UNSCORED: it measures the candidate leg and
    /// seals no score. With BOTH runner inputs present it takes the full PAIRED path — the same
    /// two-leg measurement the ranked path runs. The switch is `paired_inputs_present`, and it
    /// reads the flags first and the runner environment second; HALF the inputs is not the paired
    /// path, because one leg cannot be checked against a band that is not there.
    #[test]
    fn a_local_run_pairs_only_when_both_runner_inputs_are_present() {
        fn args_with(workspace: Option<&str>, calibration: Option<&str>) -> IterateArgs {
            let mut argv: Vec<String> = ["--engine", "e", "--weights", "w", "--golden", "g"]
                .iter()
                .map(|s| s.to_string())
                .collect();
            if let Some(w) = workspace {
                argv.push("--baseline-workspace".to_string());
                argv.push(w.to_string());
            }
            if let Some(c) = calibration {
                argv.push("--baseline-calibration".to_string());
                argv.push(c.to_string());
            }
            parse_iterate_args(&argv).unwrap().unwrap()
        }

        // The FLAGS decide, and both are needed.
        assert!(paired_inputs_present(&args_with(
            Some("/ref/tree"),
            Some("/ref/cal.json")
        )));
        assert!(!paired_inputs_present(&args_with(Some("/ref/tree"), None)));
        assert!(!paired_inputs_present(&args_with(
            None,
            Some("/ref/cal.json")
        )));
        assert!(!paired_inputs_present(&args_with(None, None)));
    }

    /// THE UNSCORED SEAL. A local run of a live-control-leg track carries the real timing surface
    /// and the real correctness verdict, states WHY it has no denominator, and seals `score:
    /// null` — and a REAL failure keeps its own error rather than being tidied away by the
    /// unscored conversion.
    #[test]
    fn an_unscored_local_run_seals_its_timings_and_no_score() {
        let golden = crate::testgolden::TestGolden::new().fixture();
        // The shape `local_iterate_score` produces with the `(0.0, 0.0)` no-denominator pair:
        // real timings, correctness passed, no score, and the placeholder text.
        let timing = bench_runner::TimingResult {
            prefill_seconds_per_token: 0.0004,
            decode_seconds_per_token: 0.02,
            decode_steps: 128,
            prefill_prompt_tokens: 512,
            prefill_elapsed_seconds: 0.2048,
            decode_elapsed_seconds: 2.56,
            peak_ram_gb: 20.0,
            effective_spec: None,
            free_run_audit: None,
            phase_window: None,
            emitted_tokens: Vec::new(),
        };
        let mut payload = iterate::local_iterate_score(
            Mode::LocalIterate,
            &timing,
            ScoringInputs::local(0.0, 0.0),
            &golden,
            iterate::RunDigests::for_test(&DirDigest::empty()),
            &test_window(),
        );
        assert!(
            !payload.passed,
            "precondition: the zero pair reports no score"
        );
        assert_eq!(
            payload.metrics.error,
            iterate::INVALID_LOCAL_SCORE_ERROR,
            "precondition: the placeholder text is what the zero pair leaves"
        );

        iterate::seal_local_unscored(&mut payload);
        assert!(payload.score.is_none(), "an unscored run seals no score");
        assert!(payload.passed, "a healthy unscored run is not a failure");
        assert_eq!(payload.metrics.error, "", "the placeholder text is cleared");
        assert_eq!(
            payload.metrics.baseline_source.as_deref(),
            Some(iterate::BASELINE_SOURCE_LOCAL_UNSCORED)
        );
        assert_eq!(
            payload.metrics.baseline_source.as_deref(),
            Some("none (local mode: unscored)")
        );
        // The RAW timings are sealed, and they are the ones measured.
        assert_eq!(payload.metrics.prefill_seconds_per_token, 0.0004);
        assert_eq!(payload.metrics.decode_seconds_per_token, 0.02);
        assert!(payload.metrics.passed_correctness);
        // No denominator was invented for it.
        assert_eq!(payload.metrics.baseline_prefill_seconds_per_token, 0.0);
        assert_eq!(payload.metrics.baseline_decode_seconds_per_token, 0.0);
        // The human-facing rate, from the same numbers.
        assert!((tokens_per_second(0.02) - 50.0).abs() < 1e-9);
        assert_eq!(tokens_per_second(0.0), 0.0);
        assert_eq!(tokens_per_second(f64::NAN), 0.0);
        // …and it reaches the sealed JSON as a marker, not as a number.
        let sealed: serde_json::Value =
            serde_json::from_str(&payload.to_sealed_json().unwrap()).unwrap();
        assert!(sealed["score"].is_null());
        assert_eq!(
            sealed["metrics"]["baseline_source"].as_str(),
            Some("none (local mode: unscored)")
        );

        // A REAL failure is NOT converted: its error and its verdict survive.
        let mut failed = iterate::local_iterate_score(
            Mode::LocalIterate,
            &timing,
            ScoringInputs::local(0.0, 0.0),
            &golden,
            iterate::RunDigests::for_test(&DirDigest::empty()),
            &test_window(),
        );
        failed.metrics.passed_correctness = false;
        failed.metrics.error = "correctness failed: case-a step 3".to_string();
        iterate::seal_local_unscored(&mut failed);
        assert!(!failed.passed, "a correctness failure stays a failure");
        assert_eq!(failed.metrics.error, "correctness failed: case-a step 3");
        assert!(failed.score.is_none());
        assert_eq!(
            failed.metrics.baseline_source.as_deref(),
            Some(iterate::BASELINE_SOURCE_LOCAL_UNSCORED),
            "an unscored run says so even when it failed"
        );
    }

    /// THE STORED-PAIR DOORS ARE SHUT on the ranked paired path, each BY NAME. This is the fence
    /// `execute_iterate` applies pre-GPU: a golden that still declares a pair, the trusted
    /// `MLXFAST_PAIRED_BASELINE_*` env, and the `--baseline-*` flags. Each is a denominator, and
    /// the paired path has exactly one denominator: the leg it measures.
    #[test]
    fn the_paired_path_refuses_every_stored_denominator() {
        let declaring = crate::testgolden::TestGolden::new()
            .baselines(0.000123456, 0.00987654)
            .fixture();
        let err = baseline::refuse_golden_with_stored_pair(&declaring).unwrap_err();
        assert!(
            err.contains(baseline::GOLDEN_CARRIES_STORED_BASELINE),
            "{err}"
        );
        // A golden with no declared pair passes the same gate.
        let clean = crate::testgolden::TestGolden::new().fixture();
        assert!(baseline::refuse_golden_with_stored_pair(&clean).is_ok());

        // The flags are still PARSED (the stored-pair tracks use them); they are refused at the
        // paired path's door, not at the door of every run.
        let args: Vec<String> = [
            "--engine",
            "e",
            "--weights",
            "w",
            "--golden",
            "g",
            "--mode",
            "official",
            "--contract",
            "c.json",
            "--baseline-prefill-spt",
            "0.0006",
            "--baseline-decode-spt",
            "0.03",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let parsed = parse_iterate_args(&args).unwrap().unwrap();
        assert_eq!(parsed.baseline_prefill_spt, Some(0.0006));
        let err = baseline::refuse_stored_baseline_override(
            None,
            None,
            parsed.baseline_prefill_spt.is_some() || parsed.baseline_decode_spt.is_some(),
        )
        .unwrap_err();
        assert!(
            err.contains(baseline::STORED_BASELINE_OVERRIDE_REFUSED),
            "{err}"
        );
    }

    /// THE LEG'S OWN SOCKET, AND ONLY ITS OWN, REACHES THAT LEG'S SEATBELT PROFILE.
    ///
    /// Seatbelt counts an AF_UNIX connect as network, so the profile's one `network-outbound`
    /// allowance must name the socket THIS spawn's worker will connect to. With per-leg residents
    /// that socket lives in the leg's spawn-env overrides and nowhere else — the process
    /// environment has none — and reading it from the process env left a leg worker with no
    /// allowance at all (`cannot reach the resident … Operation not permitted`, box-4 calibration).
    ///
    /// The profile is built here through the same pure resolver `resolve_official_sandbox_from_env`
    /// delegates to, with `sandbox_exec_available` injected, so the assertion holds on any host.
    #[test]
    fn a_leg_spawns_profile_allows_exactly_its_own_resident_socket() {
        const REFERENCE_SOCKET: &str = "/tmp/bench-worker-resident.reference.sock";
        const CANDIDATE_SOCKET: &str = "/tmp/bench-worker-resident.candidate.sock";
        const PROCESS_SOCKET: &str = "/tmp/bench-worker-resident.window.sock";

        let leg_env = |socket: &str| {
            vec![(
                legserve::BENCH_WORKER_RESIDENT_SOCKET_ENV.to_string(),
                socket.to_string(),
            )]
        };
        let profile_for = |socket: Option<&str>| -> String {
            let plan = bench_runner::resolve_official_sandbox(
                &bench_runner::OfficialSandboxInputs {
                    resident_socket: socket,
                    fallback_executable: "/ref/tree/.build/release/bench-worker",
                    golden_path: "/private/goldens/botany.golden.json",
                    sandbox_exec_available: true,
                    ..Default::default()
                },
                false,
            )
            .expect("the plan resolves with a sandbox-exec available");
            match plan.profile {
                bench_runner::SandboxProfile::Generated(source) => source,
                other => panic!("expected a generated profile, got {other:?}"),
            }
        };

        // THE SELECTION: the leg's override wins, and the process env is not consulted when a leg
        // named one. A leg with no override falls back to the process env — the legacy
        // single-resident topology, which is the only one that has a process socket.
        let reference = spawn_resident_socket(&leg_env(REFERENCE_SOCKET), Some(PROCESS_SOCKET));
        assert_eq!(reference.as_deref(), Some(REFERENCE_SOCKET));
        let candidate = spawn_resident_socket(&leg_env(CANDIDATE_SOCKET), Some(PROCESS_SOCKET));
        assert_eq!(candidate.as_deref(), Some(CANDIDATE_SOCKET));
        assert_eq!(
            spawn_resident_socket(&[], Some(PROCESS_SOCKET)).as_deref(),
            Some(PROCESS_SOCKET),
            "the legacy single-resident path keeps its process-env socket"
        );
        // An empty value is no socket at all — and a leg that named the variable at all has
        // ANSWERED, so it never falls back to a process socket that names a different resident.
        assert_eq!(spawn_resident_socket(&[], Some("  ")), None);
        assert_eq!(
            spawn_resident_socket(&leg_env(""), Some(PROCESS_SOCKET)),
            None
        );

        // THE PROFILE: each leg's profile allows exactly its own socket — not the other leg's, and
        // not the process env's.
        let reference_profile = profile_for(reference.as_deref());
        assert!(
            reference_profile.contains(&format!(
                "(allow network-outbound (remote unix-socket (path-literal \"{REFERENCE_SOCKET}\")))"
            )),
            "{reference_profile}"
        );
        assert!(
            !reference_profile.contains(CANDIDATE_SOCKET),
            "the reference leg must not be allowed to reach the candidate's resident: \
             {reference_profile}"
        );
        assert!(
            !reference_profile.contains(PROCESS_SOCKET),
            "the reference leg must not be allowed to reach a window resident: {reference_profile}"
        );

        let candidate_profile = profile_for(candidate.as_deref());
        assert!(
            candidate_profile.contains(&format!(
                "(allow network-outbound (remote unix-socket (path-literal \"{CANDIDATE_SOCKET}\")))"
            )),
            "{candidate_profile}"
        );
        assert!(
            !candidate_profile.contains(REFERENCE_SOCKET),
            "{candidate_profile}"
        );
        assert!(
            !candidate_profile.contains(PROCESS_SOCKET),
            "{candidate_profile}"
        );
        // Exactly ONE allowance, on each side.
        for profile in [&reference_profile, &candidate_profile] {
            assert_eq!(
                profile.matches("network-outbound").count(),
                1,
                "one allowance and no more: {profile}"
            );
        }

        // NO SOCKET ANYWHERE: no unix-socket allowance at all, and the blanket network deny stands.
        assert_eq!(spawn_resident_socket(&[], None), None);
        let bare = profile_for(None);
        assert!(
            !bare.contains("unix-socket") && !bare.contains("network-outbound"),
            "a run with no resident must widen nothing: {bare}"
        );
        assert!(bare.contains("(deny network*)"), "{bare}");

        // THE READ SURFACE. The profile's base is `(allow default)` and its only read denials name
        // the private golden and the private dir — nothing is rooted at the candidate checkout. So
        // the REFERENCE leg reads the reference tree's own weights and metallib, which live
        // outside that checkout, with no extra allowance needed.
        for profile in [&reference_profile, &bare] {
            assert!(profile.contains("(allow default)"), "{profile}");
            let denied_reads: Vec<&str> = profile
                .lines()
                .filter(|l| l.starts_with("(deny file-read*"))
                .collect();
            assert_eq!(
                denied_reads,
                vec!["(deny file-read* (literal \"/private/goldens/botany.golden.json\"))"],
                "the only denied read is the private golden: {profile}"
            );
        }
    }

    /// LEG 1's OWN GOLDEN parses, and it is pinned by the SAME rule the candidate golden is:
    /// both pin flags together or neither, each refusal naming its own two flags.
    #[test]
    fn the_control_golden_and_its_pin_parse_as_flags() {
        let args: Vec<String> = [
            "--engine",
            "e",
            "--weights",
            "w",
            "--golden",
            "/goldens/botany.mtp1.golden.json",
            "--control-golden",
            "/goldens/botany.golden.json",
            "--control-golden-sha256",
            "abc",
            "--control-golden-bytes",
            "10",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let parsed = parse_iterate_args(&args).unwrap().unwrap();
        assert_eq!(
            parsed.control_goldens,
            [PathBuf::from("/goldens/botany.golden.json")]
        );
        let pin = &parsed.control_golden_pins[0];
        assert_eq!(pin.sha256, "abc");
        assert_eq!(pin.bytes, 10);

        // ABSENT ⇒ leg 1 verifies against `--golden`, which is today's behaviour.
        let bare: Vec<String> = ["--engine", "e", "--weights", "w", "--golden", "g"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let parsed = parse_iterate_args(&bare).unwrap().unwrap();
        assert!(parsed.control_goldens.is_empty());
        assert!(parsed.control_golden_pins.is_empty());

        // HALF A PIN is refused BY ITS OWN FLAG NAMES: an operator holding a control-golden sha
        // with no byte count must not be told about `--golden-bytes`.
        let names = [
            "--control-golden-sha256",
            "--control-golden-bytes",
            "--control-golden",
        ];
        for half in [
            parse_golden_pins(vec!["abc"], Vec::new(), 1, names),
            parse_golden_pins(Vec::new(), vec!["10"], 1, names),
        ] {
            let err = half.expect_err("half a pin must be refused");
            assert!(
                err.contains("--control-golden-sha256") && err.contains("--control-golden-bytes"),
                "a half pin must name BOTH of its own flags: {err}"
            );
        }
    }

    /// SEVERAL GOLDENS parse in order, and each pin flag and `--control-golden` matches `--golden`
    /// by position. A count that does not agree refuses by name.
    #[test]
    fn several_goldens_parse_with_their_pins_matched_by_position() {
        let argv = |extra: &[&str]| -> Vec<String> {
            [
                "--engine",
                "e",
                "--weights",
                "w",
                "--golden",
                "a.json",
                "--golden",
                "b.json",
            ]
            .iter()
            .chain(extra)
            .map(|s| s.to_string())
            .collect()
        };
        let parsed = parse_iterate_args(&argv(&[
            "--golden-sha256",
            "aaa",
            "--golden-bytes",
            "1",
            "--golden-sha256",
            "bbb",
            "--golden-bytes",
            "2",
            "--control-golden",
            "a-serial.json",
            "--control-golden",
            "b-serial.json",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(
            parsed.goldens,
            [PathBuf::from("a.json"), PathBuf::from("b.json")]
        );
        assert_eq!(parsed.golden_pins[1].sha256, "bbb");
        assert_eq!(parsed.golden_pins[1].bytes, 2);
        assert_eq!(
            parsed.control_goldens,
            [
                PathBuf::from("a-serial.json"),
                PathBuf::from("b-serial.json")
            ]
        );
        assert!(parsed.control_golden_pins.is_empty());

        for extra in [
            // One pin for two goldens.
            &["--golden-sha256", "aaa", "--golden-bytes", "1"][..],
            // One control golden for two goldens.
            &["--control-golden", "a-serial.json"][..],
            // Two control goldens, one control pin.
            &[
                "--control-golden",
                "a-serial.json",
                "--control-golden",
                "b-serial.json",
                "--control-golden-sha256",
                "aaa",
                "--control-golden-bytes",
                "1",
            ][..],
        ] {
            let Err(err) = parse_iterate_args(&argv(extra)) else {
                panic!("{extra:?} must be refused");
            };
            assert!(
                err.contains(baseline::GOLDEN_COUNT_MISMATCH),
                "{extra:?}: {err}"
            );
        }
    }

    /// THE PAIRED PATH's two runner inputs are FLAGS as well as environment variables, and both
    /// resolve the same way: the flag when it is given, else the runner variable.
    #[test]
    fn the_paired_runner_inputs_parse_as_flags() {
        let args: Vec<String> = [
            "--engine",
            "e",
            "--weights",
            "w",
            "--golden",
            "g",
            "--mode",
            "official",
            "--contract",
            "c.json",
            "--baseline-workspace",
            "/ref/tree",
            "--baseline-calibration",
            "/ref/calibration.json",
            "--box",
            "m5-max-128gb-4-qwen38-125b-a6b-mlx",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let parsed = parse_iterate_args(&args).unwrap().unwrap();
        assert_eq!(
            parsed.baseline_workspace.as_deref(),
            Some(Path::new("/ref/tree"))
        );
        assert_eq!(
            parsed.baseline_calibration.as_deref(),
            Some(Path::new("/ref/calibration.json"))
        );
        assert_eq!(
            parsed.box_name.as_deref(),
            Some("m5-max-128gb-4-qwen38-125b-a6b-mlx")
        );
        // Absent from the command line, the flags are absent — the environment answers instead,
        // and a run with neither refuses by name (`baseline::resolve_workspace`).
        let bare: Vec<String> = ["--engine", "e", "--weights", "w", "--golden", "g"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let parsed = parse_iterate_args(&bare).unwrap().unwrap();
        assert!(parsed.baseline_workspace.is_none());
        assert!(parsed.baseline_calibration.is_none());
        assert!(parsed.box_name.is_none());
        let err = baseline::resolve_workspace(None, None).unwrap_err();
        assert!(err.contains(baseline::BASELINE_WORKSPACE_MISSING), "{err}");
        let err = baseline::load_calibration(None, None).unwrap_err();
        assert!(
            err.contains(baseline::BASELINE_CALIBRATION_MISSING),
            "{err}"
        );
    }

    /// The other direction: the track that DOES declare a regime still takes the TABLE arm, so the
    /// narrowed predicate did not simply disable the table resolution for everyone.
    #[test]
    fn the_declared_regime_track_still_resolves_through_the_table_arm() {
        use crate::iterate::Mode;
        use bench_core::constants::TRACK_ID;

        // `TestGolden::new()` carries the oracle but declares NO baseline pair.
        let golden = crate::testgolden::TestGolden::new().fixture();
        let fixture = stored_pair_fixture();
        assert!(declares_a_scored_regime(&fixture, TRACK_ID));
        let want = bench_core::contract::official_baseline(&fixture, TRACK_ID)
            .unwrap()
            .value;
        // The DISCRIMINATOR: this track is not a live-control-leg track, so its declared pair is
        // the denominator where the paired arm measures one for those.
        assert!(
            !bench_core::contract::scores_against_live_control_leg(&fixture, TRACK_ID)
                .unwrap()
                .value
        );

        match run_baselines(&fixture, Mode::LocalIterate, &golden, None, Some(TRACK_ID)).unwrap() {
            RunBaselines::Decided {
                prefill, decode, ..
            } => {
                assert_eq!(prefill, want.prefill_seconds_per_token);
                assert_eq!(decode, want.decode_seconds_per_token);
            }
            other => panic!("the declared-regime track must decide its pair, got {other:?}"),
        }
        assert_eq!(
            gates_only_baselines(&fixture, Some(TRACK_ID), None, &golden).unwrap(),
            (
                want.prefill_seconds_per_token,
                want.decode_seconds_per_token
            )
        );
    }

    #[test]
    fn run_baselines_takes_only_the_official_constants_on_the_local_legs() {
        use crate::iterate::Mode;
        use bench_core::constants::Platform;
        let golden = crate::testgolden::TestGolden::new()
            .baselines(0.000123456, 0.00987654)
            .fixture();
        assert_eq!(
            resolve_paired_baselines(None, &golden),
            Some((0.000123456, 0.00987654)),
            "precondition: the OFFICIAL resolver really would have taken the golden's pair"
        );
        // NO TRACK ID at all: the local legs have nothing to key a pair by, so they refuse
        // before anything runs — the golden's declared pair and a `--baseline` flag are both in
        // reach and neither stands in for it.
        for mode in [Mode::LocalIterate, Mode::LocalSubmit] {
            for flags in [None, Some((0.5_f64, 0.6_f64))] {
                let err = run_baselines(NO_FIXTURE, mode, &golden, flags, None)
                    .expect_err("a local leg must not score without a track to key the pair by");
                assert!(
                    err.contains("MLXFAST_QWEN_MTP_TRACK_ID"),
                    "{}: the refusal must name the declaration it needs: {err}",
                    mode.mode_name()
                );
            }
        }
        assert_eq!(
            run_baselines(NO_FIXTURE, Mode::Official, &golden, None, None).unwrap(),
            RunBaselines::ResolveFromOverrideOrGolden,
            "official must stay golden-authoritative (#127 scoped itself to the local leg)"
        );
        // A track with no row in the table refuses BY NAME through the table's own accessor.
        for mode in [Mode::LocalIterate, Mode::LocalSubmit] {
            let err = run_baselines(NO_FIXTURE, mode, &golden, None, Some("qwen3.9-27b-mlx-v1"))
                .expect_err("an uncaptured track must refuse before anything runs");
            assert!(
                err.contains(bench_core::constants::OFFICIAL_BASELINE_PENDING),
                "{}: {err}",
                mode.mode_name()
            );
        }
        // The platform comes from the declared track id and nowhere else.
        assert_eq!(
            iterate_platform(Some("qwen3.8-125b-a6b-mlx-v1")).unwrap(),
            Platform::Mlx
        );
        // #127 INERTNESS, captured state (the seam): once a baseline exists, the local legs take
        // exactly that pair — the golden's declared pair and a --baseline flag never win.
        let stand_in = crate::testgolden::TEST_BASELINE;
        for mode in [Mode::LocalIterate, Mode::LocalSubmit] {
            for flags in [None, Some((0.5_f64, 0.6_f64))] {
                match run_baselines_with(mode, &golden, flags, Ok(stand_in)).unwrap() {
                    RunBaselines::Decided {
                        prefill,
                        decode,
                        flags_ignored,
                    } => {
                        assert_eq!(
                            (prefill, decode),
                            (
                                stand_in.prefill_seconds_per_token,
                                stand_in.decode_seconds_per_token
                            ),
                            "{} took a pair that is not the official constants",
                            mode.mode_name()
                        );
                        assert_ne!(
                            prefill,
                            0.000123456,
                            "{}: the golden's pair won",
                            mode.mode_name()
                        );
                        assert_ne!(
                            decode,
                            0.00987654,
                            "{}: the golden's pair won",
                            mode.mode_name()
                        );
                        if let Some((fp, fd)) = flags {
                            assert_ne!(prefill, fp, "{}: a --baseline flag won", mode.mode_name());
                            assert_ne!(decode, fd, "{}: a --baseline flag won", mode.mode_name());
                        }
                        assert_eq!(flags_ignored, flags.is_some(), "{}", mode.mode_name());
                    }
                    other => panic!("{} must decide locally, got {other:?}", mode.mode_name()),
                }
            }
        }
        assert_eq!(
            run_baselines_with(Mode::Official, &golden, None, Ok(stand_in)).unwrap(),
            RunBaselines::ResolveFromOverrideOrGolden
        );
        assert_eq!(
            iterate_platform(Some("qwen3.8-125b-a6b-cuda-v1")).unwrap(),
            Platform::Cuda
        );
        let err = iterate_platform(None).unwrap_err();
        assert!(err.contains("MLXFAST_QWEN_MTP_TRACK_ID"), "{err}");
        // --capture-baseline is a local-iterate-only mode: every other mode refuses at parse.
        let args: Vec<String> = [
            "--engine",
            "e",
            "--weights",
            "w",
            "--golden",
            "g",
            "--mode",
            "local-submit",
            "--capture-baseline",
            "rec.json",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let err = parse_iterate_args(&args)
            .err()
            .expect("a non-local-iterate mode with --capture-baseline must refuse");
        assert!(err.contains("--capture-baseline"), "{err}");
        assert!(err.contains("local-submit"), "{err}");
        let args: Vec<String> = [
            "--engine",
            "e",
            "--weights",
            "w",
            "--golden",
            "g",
            "--mode",
            "local-iterate",
            "--capture-baseline",
            "rec.json",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let parsed = parse_iterate_args(&args).unwrap().unwrap();
        assert_eq!(
            parsed.capture_baseline.as_deref(),
            Some(std::path::Path::new("rec.json"))
        );
        assert!(
            !parsed.capture_timed_only,
            "--capture-timed-only defaults OFF (the first per-prompt pass runs the gate)"
        );

        // a8 ruling — CAPTURE-ONLY SCOPING of --capture-timed-only: the correctness-gate skip is a
        // MODIFIER of --capture-baseline. It is refused by name WITHOUT --capture-baseline, so it can
        // never attach to a scored/official run (official also refuses --capture-baseline itself, and
        // official::official_core's gate is untouched). WITH --capture-baseline it parses and sets
        // the flag; the driver hands it to passes 2-4/B only.
        let base = |extra: &[&str]| -> Vec<String> {
            let mut v = vec![
                "--engine",
                "e",
                "--weights",
                "w",
                "--golden",
                "g",
                "--mode",
                "local-iterate",
            ];
            v.extend_from_slice(extra);
            v.iter().map(|s| s.to_string()).collect()
        };
        // Refused without --capture-baseline.
        let err = parse_iterate_args(&base(&["--capture-timed-only"]))
            .err()
            .expect("--capture-timed-only without --capture-baseline must refuse");
        assert!(err.contains("--capture-timed-only"), "{err}");
        assert!(err.contains("--capture-baseline"), "{err}");
        // Accepted WITH --capture-baseline; the flag is set.
        let parsed = parse_iterate_args(&base(&[
            "--capture-baseline",
            "rec.json",
            "--capture-timed-only",
        ]))
        .unwrap()
        .unwrap();
        assert!(parsed.capture_timed_only);
    }

    /// Option B digest-hoist — RIDER 1 PARSE-LEVEL REFUSAL. `--weights-digest` is a MODIFIER of
    /// `--capture-baseline`, gated identically to `--capture-timed-only` (its sibling above): it is
    /// refused by name WITHOUT `--capture-baseline`, so a passed-in weights digest can never attach
    /// to a scored/official run (official refuses `--capture-baseline` itself). WITH
    /// `--capture-baseline` it parses and carries the exact sha/bytes/count into `IterateArgs`.
    #[test]
    fn weights_digest_flag_refused_outside_capture_baseline() {
        let base = |extra: &[&str]| -> Vec<String> {
            let mut v = vec![
                "--engine",
                "e",
                "--weights",
                "w",
                "--golden",
                "g",
                "--mode",
                "local-iterate",
            ];
            v.extend_from_slice(extra);
            v.iter().map(|s| s.to_string()).collect()
        };
        // A valid digest triple: 64-hex sha + two counts.
        let digest = format!("{}:{}:{}", "a".repeat(64), 12345, 7);

        // Refused WITHOUT --capture-baseline (mirrors the --capture-timed-only sibling refusal).
        let err = parse_iterate_args(&base(&["--weights-digest", &digest]))
            .err()
            .expect("--weights-digest without --capture-baseline must refuse");
        assert!(err.contains("--weights-digest"), "{err}");
        assert!(err.contains("--capture-baseline"), "{err}");

        // Official mode never sees the flag: it refuses --capture-baseline itself, and
        // --weights-digest requires --capture-baseline, so the two can never co-occur on official.
        let official: Vec<String> = [
            "--engine",
            "e",
            "--weights",
            "w",
            "--golden",
            "g",
            "--mode",
            "official",
            "--contract",
            "c.json",
            "--weights-digest",
            &digest,
            "--capture-baseline",
            "rec.json",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let err = parse_iterate_args(&official)
            .err()
            .expect("official + --capture-baseline must refuse before the digest is honored");
        assert!(err.contains("--capture-baseline"), "{err}");

        // Accepted WITH --capture-baseline; the parsed digest carries the exact triple.
        let parsed = parse_iterate_args(&base(&[
            "--capture-baseline",
            "rec.json",
            "--weights-digest",
            &digest,
        ]))
        .unwrap()
        .unwrap();
        let wd = parsed
            .weights_digest
            .expect("--weights-digest parses in capture-baseline mode");
        assert_eq!(wd.sha256, "a".repeat(64));
        assert_eq!(wd.byte_count, 12345);
        assert_eq!(wd.file_count, 7);
    }

    /// Option B digest-hoist — SKIP-RECOMPUTE. With `--weights-digest` set, `resolve_weights_digest`
    /// returns the PASSED digest and never touches the weights path (proven by pointing `--weights`
    /// at a path that does not exist: a recompute would error). The resulting value is exactly the
    /// triple the driver passed — the digest that flows into `RunDigests.weights`.
    #[test]
    fn weights_digest_flag_skips_recompute_and_carries_value() {
        let digest = format!("{}:{}:{}", "b".repeat(64), 99, 3);
        let parsed = parse_iterate_args(
            &[
                "--engine",
                "e",
                "--weights",
                "/nonexistent/weights/path/that/must/not/be/hashed",
                "--golden",
                "g",
                "--mode",
                "local-iterate",
                "--capture-baseline",
                "rec.json",
                "--weights-digest",
                &digest,
            ]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        )
        .unwrap()
        .unwrap();
        // The nonexistent --weights path is never read because the passed digest short-circuits it.
        let resolved = resolve_weights_digest(&parsed)
            .expect("a passed --weights-digest is reused without touching the weights tree");
        assert_eq!(resolved.sha256, "b".repeat(64));
        assert_eq!(resolved.byte_count, 99);
        assert_eq!(resolved.file_count, 3);

        // A malformed triple is refused at parse (not silently accepted into a seal).
        let bad = parse_iterate_args(
            &[
                "--engine",
                "e",
                "--weights",
                "w",
                "--golden",
                "g",
                "--mode",
                "local-iterate",
                "--capture-baseline",
                "rec.json",
                "--weights-digest",
                "NOTHEX:1:1",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        )
        .err()
        .expect("a non-hex sha256 must be refused at parse");
        assert!(bad.contains("--weights-digest"), "{bad}");
    }

    /// Option B digest-hoist — BYTE-IDENTITY. The `weights-digest` subcommand's `<sha>:<bytes>:
    /// <files>` output, parsed back with `parse_weights_digest`, EQUALS the `DirDigest` `dir_digest`
    /// returns for the same directory. This is the property the hoist rests on: the window's
    /// once-computed value is exactly what a per-pass recompute would produce.
    #[test]
    fn weights_digest_subcommand_output_round_trips_to_dir_digest() {
        use crate::iterate::dir_digest;
        let tmp = std::env::temp_dir().join(format!(
            "benchd-wdigest-rt-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("sub")).unwrap();
        std::fs::write(tmp.join("a.bin"), b"hello").unwrap();
        std::fs::write(tmp.join("sub/b.bin"), b"world").unwrap();

        // What `dir_digest` (and therefore any pass) computes for this tree.
        let computed = dir_digest(&tmp).unwrap();
        // The exact one-line string the `weights-digest` subcommand prints.
        let printed = format!(
            "{}:{}:{}",
            computed.sha256, computed.byte_count, computed.file_count
        );
        // Parsed back, it is byte-identical to the computed digest.
        let round_tripped = parse_weights_digest(&printed).unwrap();
        assert_eq!(round_tripped, computed);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The integrity sidecar's `golden_sha256` is now `GoldenFixture::sha256` — the digest the
    /// loader recorded from the bytes it validated — instead of a second `fs::read` of the same
    /// path. That substitution is only value-preserving if the held digest IS `sha256_hex` of the
    /// raw file bytes. Pin it here, so the sidecar can never quietly start pinning something else.
    #[test]
    fn held_golden_digest_equals_the_sha256_of_the_raw_bytes() {
        let g = crate::testgolden::TestGolden::new();
        assert_eq!(g.fixture().sha256, sha256_hex(&g.bytes()));
    }

    /// The LOCAL legs refuse an uncaptured track BY NAME. #127 makes them score against the
    /// official constants and nothing else, so a track with no captured pair has nothing to score
    /// against — it must stop before the engine spawns, never inherit another track's numbers.
    #[test]
    fn local_legs_refuse_an_uncaptured_track_by_name() {
        const UNCAPTURED: &str = "qwen3.9-27b-mlx-v1";
        let golden = crate::testgolden::TestGolden::new().fixture();
        for mode in [Mode::LocalIterate, Mode::LocalSubmit] {
            let err = run_baselines_with(
                mode,
                &golden,
                None,
                bench_core::contract::official_baseline(NO_FIXTURE, UNCAPTURED).map(|r| r.value),
            )
            .unwrap_err();
            assert!(err.contains(UNCAPTURED), "{}: {err}", mode.mode_name());
            assert!(
                err.contains(bench_core::constants::OFFICIAL_BASELINE_PENDING),
                "{}: {err}",
                mode.mode_name()
            );
        }
        // The branch's own track still resolves, on both legs.
        for mode in [Mode::LocalIterate, Mode::LocalSubmit] {
            assert!(run_baselines_with(
                mode,
                &golden,
                None,
                bench_core::contract::official_baseline(
                    &stored_pair_fixture(),
                    bench_core::constants::TRACK_ID
                )
                .map(|r| r.value)
            )
            .is_ok());
        }
    }

    /// #132/F3 — the strong direction: a real, readable engine seals a FULL identity.
    #[test]
    fn runner_identity_seals_a_canonical_engine_in_full() {
        let dir = std::env::temp_dir().join(format!(
            "benchd-f3-ok-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = dir.join("engine-bin");
        let bytes = b"#!/bin/sh\nexit 0\n";
        std::fs::write(&engine, bytes).unwrap();

        // Deliberately a NON-canonical spelling of the same file, so the test proves
        // canonicalisation rather than string passthrough.
        let noisy = dir.join(".").join("engine-bin").display().to_string();
        let id = resolve_runner_identity(&noisy);

        assert_eq!(
            id.candidate_executable_resolution,
            ENGINE_RESOLUTION_CANONICAL
        );
        assert_eq!(id.candidate_executable_sha256, sha256_hex(bytes));
        assert_eq!(
            id.candidate_executable,
            std::fs::canonicalize(&engine)
                .unwrap()
                .display()
                .to_string(),
            "the seal must name the canonical path, not the spelling the caller used"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// #132/F3 — the direction that used to kill the run: an engine benchd cannot canonicalise or
    /// read seals the WEAKER identity and the run PROCEEDS.
    ///
    /// This covers the two shapes that worked before #123 and broke after it — a bare name the
    /// spawner would have resolved through `PATH` (`canonicalize` resolves against the CWD, not
    /// `PATH`), and an executable-but-not-readable binary (`spawn` needs `--x`, `read` needs `r`).
    /// The function is TOTAL, which is the "it still runs" claim: there is no error type left for
    /// it to abort `execute_iterate` with.
    #[test]
    fn runner_identity_falls_back_to_a_sentinel_instead_of_failing_the_run() {
        // A bare name that is not a path relative to the CWD — the PATH-resolved shape.
        let bare = "mlxfast-engine-that-is-not-in-this-directory";
        let id = resolve_runner_identity(bare);
        assert_eq!(
            id.candidate_executable_resolution, ENGINE_RESOLUTION_UNRESOLVED,
            "an unresolvable engine must be SEALED as unresolved, not silently blank"
        );
        assert_eq!(
            id.candidate_executable, bare,
            "the path is sealed exactly as the caller gave it — that is the identity we have"
        );
        assert!(
            id.candidate_executable_sha256.is_empty(),
            "no digest exists, so none is fabricated"
        );

        // An executable-but-unreadable file, on the platforms where that is expressible.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let dir = std::env::temp_dir().join(format!(
                "benchd-f3-noread-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let engine = dir.join("engine-bin");
            std::fs::write(&engine, b"x").unwrap();
            std::fs::set_permissions(&engine, std::fs::Permissions::from_mode(0o111)).unwrap();
            // Skip rather than fail when running as root, where the read succeeds regardless.
            if std::fs::read(&engine).is_err() {
                let id = resolve_runner_identity(&engine.display().to_string());
                assert_eq!(
                    id.candidate_executable_resolution, ENGINE_RESOLUTION_UNRESOLVED,
                    "an exec-but-not-readable engine must seal the sentinel, not abort the run"
                );
                assert!(id.candidate_executable_sha256.is_empty());
            }
            let _ = std::fs::set_permissions(&engine, std::fs::Permissions::from_mode(0o644));
            let _ = std::fs::remove_dir_all(&dir);
        }

        // The sentinel is a DISTINCT value, so an empty digest is never ambiguous.
        assert_ne!(ENGINE_RESOLUTION_UNRESOLVED, ENGINE_RESOLUTION_CANONICAL);
    }

    #[test]
    fn preflight_failed_payload_is_swift_shaped() {
        use crate::iterate::{DirDigest, Mode};
        let golden = crate::testgolden::TestGolden::new()
            .without_benchmark()
            .fixture();
        let err = crate::iterate::missing_paired_baselines_error(Mode::LocalIterate);
        let p = crate::iterate::preflight_failed_payload(
            Mode::LocalIterate,
            &golden,
            RunDigests::for_test(&DirDigest::empty()),
            err.clone(),
            ScoringInputs::local(
                crate::testgolden::TEST_BASELINE.prefill_seconds_per_token,
                crate::testgolden::TEST_BASELINE.decode_seconds_per_token,
            ),
        );
        assert!(!p.passed);
        assert!(p.score.is_none());
        assert!(!p.metrics.passed_correctness);
        // #74 (RULED 2026-08-20): the reference's early-refuse record carries the official
        // baseline CONSTANTS, not zeros — `failedScore`'s baseline parameters default to them
        // and the local refusal site overrides neither. (The pre-ruling assertion pinned 0.0,
        // which described the RETIRED fork.) What the record must be in full is pinned by
        // `iterate::tests::early_refuse_record_byte_matches_the_reference_capture`.
        assert_eq!(
            p.metrics.baseline_prefill_seconds_per_token,
            crate::testgolden::TEST_BASELINE.prefill_seconds_per_token
        );
        assert_eq!(
            p.metrics.baseline_decode_seconds_per_token,
            crate::testgolden::TEST_BASELINE.decode_seconds_per_token
        );
        // The two fields #74 names: nothing ran, so no run is described.
        assert_eq!(p.metrics.golden_hash, "");
        assert_eq!(p.metrics.case_count, 0);
        assert_eq!(p.metrics.checked_steps, 0);
        // #62: the payload carries the message IN FULL. What that message must BE is pinned
        // separately, against the Swift capture, by
        // `iterate::tests::missing_paired_baselines_error_matches_swift_capture` — this
        // assertion would otherwise just restate whatever benchd happens to produce.
        assert_eq!(p.metrics.error, err);
    }

    #[test]
    fn cool_gate_flag_defaults_off_and_opts_in() {
        let base: Vec<String> = ["--engine", "e", "--weights", "w", "--golden", "g"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        // Tri-state (#60.3): unset → None (per-mode default decides).
        assert_eq!(parse_iterate_args(&base).unwrap().unwrap().cool_gate, None);
        // --cool-gate forces ON.
        let mut with = base.clone();
        with.push("--cool-gate".to_string());
        assert_eq!(
            parse_iterate_args(&with).unwrap().unwrap().cool_gate,
            Some(true)
        );
        // --no-cool-gate forces OFF (overrides submit's default-ON).
        let mut without = base.clone();
        without.push("--no-cool-gate".to_string());
        assert_eq!(
            parse_iterate_args(&without).unwrap().unwrap().cool_gate,
            Some(false)
        );
        // Conflicting flags are rejected (either order).
        let mut both = base.clone();
        both.push("--cool-gate".to_string());
        both.push("--no-cool-gate".to_string());
        assert!(parse_iterate_args(&both).is_err());
        // Per-mode default: local-iterate OFF, official ON (David 2026-09-06: the 40 C
        // per-phase contract holds on the ranked path), local-submit ON (P6 RULING).
        assert!(!Mode::LocalIterate.cool_gate_on_by_default());
        assert!(Mode::Official.cool_gate_on_by_default());
        assert!(Mode::LocalSubmit.cool_gate_on_by_default());
    }

    #[test]
    fn local_submit_defaults_to_plain_score_json_and_integrity() {
        // M-6 NAMING: local-submit writes the DEFAULT `score.json` (+ the default
        // `benchmark-integrity.json` sidecar), NOT the `.local-iterate`-suffixed names —
        // only local-iterate carries that suffix (benchmark.sh:92-95,135-137).
        let ok = parse_iterate_args(&[
            "--engine".into(),
            "e".into(),
            "--weights".into(),
            "w".into(),
            "--golden".into(),
            "g".into(),
            "--mode".into(),
            "local-submit".into(),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(ok.mode, Mode::LocalSubmit);
        assert_eq!(ok.score_path, PathBuf::from("score.json"));
        // local-iterate keeps the suffixed default; contrast the two.
        let iter = parse_iterate_args(&[
            "--engine".into(),
            "e".into(),
            "--weights".into(),
            "w".into(),
            "--golden".into(),
            "g".into(),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(iter.score_path, PathBuf::from("score.local-iterate.json"));
    }

    /// Official REFUSES at parse when no --contract is given — the arm gate has no fixture to read,
    /// and an official run must never seal a scored artifact for an unarmed/unnamed track.
    #[test]
    fn official_requires_contract() {
        let res = parse_iterate_args(&[
            "--engine".into(),
            "e".into(),
            "--weights".into(),
            "w".into(),
            "--golden".into(),
            "g".into(),
            "--mode".into(),
            "official".into(),
        ]);
        let err = match res {
            Err(e) => e,
            Ok(_) => panic!("official without --contract must be a usage error"),
        };
        assert!(err.contains("--contract"), "{err}");
    }

    /// LOAD-BEARING PROOF that the arm gate survived the move into flow A: the OFFICIAL path's
    /// `enforce_official_arm_gate` (the file-read + parse + verdict the scored seal is gated on)
    /// REFUSES a fixture that declares `official_scoring_enabled: false` AND one that declares it not
    /// at all, and ACCEPTS one that declares `true`. Ported from the retired measure-job arm-gate
    /// tests to exercise flow A instead. Reds if the gate is deleted (always Ok), inverted, or the
    /// absent case silently reads as armed.
    #[test]
    fn official_arm_gate_refuses_unarmed_and_absent_accepts_armed() {
        let dir = std::env::temp_dir().join(format!("armgate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // The gate now reads an already-LOADED fixture (one read, one digest per run), so each
        // case writes its bytes and loads them exactly as `execute_iterate` does.
        let write = |name: &str, body: &str| {
            let p = dir.join(name);
            std::fs::write(&p, body).unwrap();
            contract::load(&p).expect("the fixture is well-formed")
        };
        let track = Some("qwen3.8-125b-a6b-mlx-v1");

        // ARMED — accepted.
        let armed = write(
            "armed.json",
            r#"{"track_id":"qwen3.8-125b-a6b-mlx-v1","official_scoring_enabled":true}"#,
        );
        assert!(enforce_official_arm_gate(Some(&armed), track).is_ok());

        // DECLARED false — refused, names the flag.
        let unarmed = write(
            "unarmed.json",
            r#"{"track_id":"qwen3.8-125b-a6b-mlx-v1","official_scoring_enabled":false}"#,
        );
        let false_err = enforce_official_arm_gate(Some(&unarmed), track)
            .expect_err("official over an unarmed fixture must refuse before sealing");
        assert!(
            false_err.contains("official scoring is not enabled")
                && false_err.contains("official_scoring_enabled"),
            "{false_err}"
        );

        // ABSENT — refused too, but a DIFFERENT diagnosis than the declared-false case.
        let absent = write("absent.json", r#"{"track_id":"qwen3.8-125b-a6b-mlx-v1"}"#);
        let absent_err = enforce_official_arm_gate(Some(&absent), track)
            .expect_err("an absent arm state is not an armed one");
        assert!(
            absent_err.contains("official scoring is not enabled"),
            "{absent_err}"
        );
        assert_ne!(
            false_err, absent_err,
            "false and absent need different messages"
        );

        // MISSING --contract — fail-closed usage refusal.
        assert!(enforce_official_arm_gate(None, track).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// DAVID 2026-09-09 — THE SCORED PATH TAKES ITS SPEEDUP FLOORS FROM THE TRACK FIXTURE, and an
    /// ARMED fixture that declares none REFUSES. This drives the exact pair of calls
    /// `execute_iterate` makes on `--mode official`: the arm gate parses the fixture, and
    /// `contract::speedup_floors` resolves the floors out of that same parsed contract. Reds if a
    /// default (0.95 or anything else) is ever substituted for an undeclared floor.
    #[test]
    fn official_refuses_a_fixture_that_declares_no_speedup_floors() {
        let dir = std::env::temp_dir().join(format!("floors-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, body: &str| {
            let p = dir.join(name);
            std::fs::write(&p, body).unwrap();
            contract::load(&p).expect("the fixture is well-formed")
        };
        let track = Some("qwen3.8-125b-a6b-mlx-v1");

        // ARMED, but silent about the floors: the run refuses rather than guessing one.
        let silent = write(
            "silent.json",
            r#"{"track_id":"qwen3.8-125b-a6b-mlx-v1","official_scoring_enabled":true,"official_pairs":2}"#,
        );
        let parsed = enforce_official_arm_gate(Some(&silent), track).expect("armed");
        let err = contract::speedup_floors(parsed, "qwen3.8-125b-a6b-mlx-v1")
            .expect_err("an armed fixture with no floors must refuse");
        assert!(err.contains("declares no decode_speedup_floor"), "{err}");
        assert!(err.contains("qwen3.8-125b-a6b-mlx-v1"), "{err}");

        // DECLARED — the ruled pair is what the run enforces and seals.
        let declared = write(
            "declared.json",
            r#"{"track_id":"qwen3.8-125b-a6b-mlx-v1","official_scoring_enabled":true,
                "official_pairs":2,"decode_speedup_floor":0.95,"prefill_speedup_floor":0.95}"#,
        );
        let parsed = enforce_official_arm_gate(Some(&declared), track).expect("armed");
        assert_eq!(
            contract::speedup_floors(parsed, "qwen3.8-125b-a6b-mlx-v1"),
            Ok(bench_core::score::SpeedupFloors {
                decode: 0.95,
                prefill: 0.95
            })
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// B5 MODE FENCE — `--capture-baseline` is accepted ONLY with `--mode local-iterate`, and
    /// every other mode refuses AT PARSE, by name. The refusal names the flag and the mode that
    /// tried to use it, so the operator is not left guessing which half was wrong.
    #[test]
    fn capture_baseline_is_local_iterate_only() {
        let argv = |mode: &str| -> Vec<String> {
            vec![
                "--engine".into(),
                "e".into(),
                "--weights".into(),
                "w".into(),
                "--golden".into(),
                "g".into(),
                "--mode".into(),
                mode.into(),
                "--capture-baseline".into(),
                "rec.json".into(),
            ]
        };
        // POSITIVE control: local-iterate accepts it and carries the operator-named record path.
        let ok = parse_iterate_args(&argv("local-iterate"))
            .expect("local-iterate accepts --capture-baseline")
            .expect("not a help request");
        assert_eq!(ok.mode, Mode::LocalIterate);
        assert_eq!(ok.capture_baseline.as_deref(), Some(Path::new("rec.json")));
        // It is also OFF by default — no flag, no capture mode.
        let plain = parse_iterate_args(&[
            "--engine".into(),
            "e".into(),
            "--weights".into(),
            "w".into(),
            "--golden".into(),
            "g".into(),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(plain.capture_baseline, None);
        // NEGATIVE controls: every other mode refuses BY NAME. The refusal is asserted as one
        // CONTIGUOUS phrase, and against any run of two spaces: a multi-line Rust string literal
        // that loses its `\` continuation still compiles and still satisfies a loose `contains`
        // on either half, but renders with the source indentation embedded in it (the #223
        // defect class).
        for mode in ["local-submit", "official"] {
            let err = parse_iterate_args(&argv(mode))
                .err()
                .expect("a non-local-iterate mode must refuse --capture-baseline");
            assert!(
                err.contains(
                    "--capture-baseline is the local-iterate CAPTURE MODE (it authors the \
                     official baseline's capture record from the checked-timing leg and writes \
                     no score)"
                ),
                "{mode}: {err}"
            );
            assert!(err.contains(mode), "the refusal must name the mode: {err}");
            assert!(
                !err.contains("  "),
                "the refusal must not embed source indentation: {err}"
            );
        }
    }

    #[test]
    fn golden_pin_requires_both_or_neither() {
        assert!(parse_golden_pin(None, None).unwrap().is_none());
        let pin = parse_golden_pin(Some("abc".into()), Some("10".into()))
            .unwrap()
            .unwrap();
        assert_eq!(pin.bytes, 10);
        // NEGATIVE CONTROLS, asserted BY NAME. A bare `is_err()` here would stay green if the
        // refusal were replaced by a generic usage error, leaving an operator holding HALF a pin
        // (a sha with no byte count, or the reverse) with nothing to tell them which half is
        // missing — so each refusal must name the flags it is about.
        for half in [
            parse_golden_pin(Some("abc".into()), None),
            parse_golden_pin(None, Some("10".into())),
        ] {
            let err = half.expect_err("half a pin must be refused");
            assert!(
                err.contains("--golden-sha256") && err.contains("--golden-bytes"),
                "a half pin must name BOTH flags: {err}"
            );
        }
        // A non-numeric byte count names the flag it could not parse AND echoes the value, so the
        // diagnostic is not confusable with the half-pin refusal above.
        let err = parse_golden_pin(Some("abc".into()), Some("nan".into()))
            .expect_err("a non-numeric byte count must be refused");
        assert!(
            err.contains("--golden-bytes") && err.contains("nan"),
            "a bad byte count must name the flag and the value: {err}"
        );
    }

    /// NEGATIVE CONTROL for [`validate_gates_producer`] — the guard on a value that is SEALED
    /// VERBATIM into `benchmark-integrity.results.json`.
    ///
    /// Its doc promises the value can be read back as what it claims to be: non-empty, and free of
    /// whitespace and control characters, so a declaration cannot smuggle a second field, a newline
    /// or a terminal escape into a sealed record. Nothing pinned that promise, so a relaxed guard
    /// would have shipped silently. Each rejected character class is asserted separately, and the
    /// paired POSITIVE control keeps the guard from passing by refusing everything.
    #[test]
    fn gates_producer_refuses_a_value_that_cannot_be_sealed_verbatim() {
        // POSITIVE control: the live producer names, and an unknown future one, are accepted
        // VERBATIM (the seal is provenance, not an allowlist).
        for good in [
            "benchmark-sh",
            "facade",
            "direct-swift",
            "some-future-producer",
        ] {
            assert_eq!(
                validate_gates_producer(good).expect("a well-formed producer name is accepted"),
                good
            );
        }

        // NEGATIVE control 1 — EMPTY. The refusal names the flag and the remedy, because absent and
        // empty are different things: absent seals `undeclared`, which is an ANSWER, not a gap.
        let err = validate_gates_producer("").expect_err("an empty producer must be refused");
        assert!(err.contains("--gates-producer"), "names the flag: {err}");
        assert!(
            err.contains(GATES_PRODUCER_UNDECLARED),
            "names the omit-the-flag remedy: {err}"
        );

        // NEGATIVE control 2 — one case per SMUGGLING SHAPE the guard exists to stop. Each is
        // asserted on its own so a guard narrowed to (say) newlines alone fails here.
        for (why, raw) in [
            ("a space splits the value into two fields", "benchmark sh"),
            ("a tab splits the value into two fields", "benchmark\tsh"),
            (
                "a newline forges a second record line",
                "benchmark-sh\nfacade",
            ),
            (
                "a carriage return overwrites the line",
                "benchmark-sh\rfacade",
            ),
            ("an ESC injects a terminal escape", "benchmark-sh\u{1b}[31m"),
            ("a NUL truncates a C-side reader", "benchmark-sh\u{0}"),
            ("a leading space is not trimmed away", " benchmark-sh"),
            ("a trailing newline is not trimmed away", "benchmark-sh\n"),
        ] {
            let err = validate_gates_producer(raw)
                .expect_err(&format!("must be refused: {why} ({raw:?})"));
            assert!(
                err.contains("--gates-producer"),
                "{why}: the refusal must name the flag: {err}"
            );
            assert!(
                err.contains("refused"),
                "{why}: the refusal must say so: {err}"
            );
        }
    }

    // ------------------------------------------------------------------------------------------
    // F-5 — path relativisation at seal (no operator home directory in any sealed artifact).
    // ------------------------------------------------------------------------------------------

    /// The engine of the fix. A relative path is untouched; an absolute path under $HOME is reduced
    /// to its home-relative tail (dropping the username); a foreign `/Users/<u>/` head is dropped by
    /// the final guard; a non-home absolute path is kept. RED if the helper is reverted to a plain
    /// `.display()`.
    #[test]
    fn relativize_for_seal_strips_home_but_keeps_relative_and_foreign_paths() {
        // Relative stays byte-for-byte (the common CI shape: `--candidate candidate`).
        assert_eq!(relativize_for_seal(Path::new("candidate/x")), "candidate/x");

        // Under the operator's own $HOME → home-relative, no `/Users/<home>`.
        temp_env_home("/Users/operator", || {
            let out = relativize_for_seal(Path::new("/Users/operator/ws/candidate"));
            assert_eq!(out, "ws/candidate");
            assert!(!out.contains("/Users/"), "home stripped: {out}");
        });

        // A foreign home ($HOME elsewhere) still must not seal a `/Users/<user>/` head.
        temp_env_home("/Users/operator", || {
            let out = relativize_for_seal(Path::new("/Users/someoneelse/models/qwen"));
            assert_eq!(out, "models/qwen");
            assert!(!out.contains("/Users/"), "foreign home head dropped: {out}");
        });
        temp_env_home("/home/operator", || {
            let out = relativize_for_seal(Path::new("/home/other/w"));
            assert_eq!(out, "w");
        });

        // Absolute but outside any home carries nothing to leak, so it is left intact.
        temp_env_home("/Users/operator", || {
            assert_eq!(
                relativize_for_seal(Path::new("/opt/weights")),
                "/opt/weights"
            );
        });
    }

    /// RULING C — the STRUCTURAL property this side of the parity relies on: every home-shaped input
    /// reduces to a RELATIVE, leak-free string (no leading `/`, no `/Users/`, no `/home/`). These are
    /// the exact divergence vectors the re-review named; the shell mirror is asserted over the SAME
    /// vectors in test-official-offline.sh. C requires both impls leak-free, NOT byte-equal — so
    /// basename-comparing `weights_path` across the two sidecars can never diverge, and neither seals
    /// a home path. The foreign home ROOT (`/Users/<other>`, no trailing component) is the mandatory
    /// leak case: it must reduce to `.`, never survive as an absolute `/Users/<other>`.
    #[test]
    fn relativize_for_seal_reduces_every_home_shaped_input_to_relative_and_leakfree() {
        fn is_relative_leakfree(s: &str) -> bool {
            !s.starts_with('/') && !s.contains("/Users/") && !s.contains("/home/")
        }
        temp_env_home("/Users/operator", || {
            // (trailing slash, $HOME-exact, foreign+slash, foreign ROOT, a symlinked-CWD-shaped path
            //  that cannot prefix-match the real CWD and therefore reduces via the $HOME/head arms).
            for v in [
                "/Users/operator/models/qwen/",
                "/Users/operator/",
                "/Users/someoneelse/models/qwen/",
                "/Users/someoneelse",
                "/Users/operator/via-symlink/models/qwen",
                "/home/someoneelse",
            ] {
                let out = relativize_for_seal(Path::new(v));
                assert!(
                    is_relative_leakfree(&out),
                    "relativize_for_seal({v:?}) = {out:?} must be relative + leak-free"
                );
            }
        });
    }

    /// Serialise `$HOME` mutation across these tests: the process env is global, so two threads
    /// racing on it would flake. `cargo test` runs a module's tests on separate threads, so the
    /// three home-sensitive tests share this mutex and restore the prior value.
    fn temp_env_home(value: &str, f: impl FnOnce()) {
        use std::sync::Mutex;
        static HOME_LOCK: Mutex<()> = Mutex::new(());
        let _guard = HOME_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var_os("HOME");
        std::env::set_var("HOME", value);
        f();
        match prev {
            Some(v) => std::env::set_var("HOME", v),
            None => std::env::remove_var("HOME"),
        }
    }

    #[test]
    fn iterate_exit_status_maps_pass_to_zero_and_fail_to_nonzero() {
        // The exact boolean→exit contract at the run_iterate mapping site: a passing run exits 0,
        // a run that did NOT pass (a rejected leg / floor / ceiling / serial-band fail all set
        // passed=false) exits 1 (nonzero). run_iterate feeds this into ExitCode::from, so a false
        // verdict drives a nonzero process ExitCode. (Constructing the real ExitCode end-to-end
        // needs a live engine, so the exact mapping is asserted here at its source.)
        assert_eq!(iterate_exit_status(true), 0, "passing run exits 0");
        assert_eq!(
            iterate_exit_status(false),
            1,
            "failing/rejected run exits nonzero"
        );
        assert_ne!(
            iterate_exit_status(false),
            0,
            "a non-pass must never map to success"
        );
    }

    #[test]
    fn parse_rejects_bad_mode_and_unknown_flag() {
        assert!(parse_iterate_args(&[
            "--engine".into(),
            "e".into(),
            "--weights".into(),
            "w".into(),
            "--golden".into(),
            "g".into(),
            "--mode".into(),
            "turbo".into()
        ])
        .is_err());
        assert!(parse_iterate_args(&["--bogus".into()]).is_err());
    }

    /// A minimal valid local-iterate argv, extended per test. `--capture-baseline` is local-iterate
    /// only, so the base mode stays local-iterate.
    fn base_local_iterate_argv() -> Vec<String> {
        vec![
            "--engine".into(),
            "e".into(),
            "--weights".into(),
            "w".into(),
            "--golden".into(),
            "g".into(),
            "--mode".into(),
            "local-iterate".into(),
        ]
    }

    /// `parse_capture_passes` — the spec is a comma list of filename-safe labels; empty specs, empty
    /// labels, and out-of-charset labels are refused (a label names a single record-file component).
    #[test]
    fn parse_capture_passes_accepts_labels_and_refuses_unsafe_ones() {
        assert_eq!(
            parse_capture_passes("W,A,A,B,B").unwrap(),
            vec!["W", "A", "A", "B", "B"]
        );
        // Whitespace around labels is trimmed.
        assert_eq!(parse_capture_passes(" W , A ").unwrap(), vec!["W", "A"]);
        // Empty label (trailing comma) refused.
        assert!(parse_capture_passes("W,A,").is_err());
        // Empty spec refused.
        assert!(parse_capture_passes("").is_err());
        // Path-separator / traversal labels refused (a label is one filename component).
        for bad in ["../x", "a/b", "a.b", "a b"] {
            let err = parse_capture_passes(bad).unwrap_err();
            assert!(err.contains("invalid --capture-passes"), "{bad}: {err}");
        }
    }

    /// `capture_pass_record_path` inserts `.<label>` before the base path's extension, matching the
    /// driver's `baseline.<prompt-id>.<label>.json` naming; a base without an extension appends
    /// `.<label>`.
    #[test]
    fn capture_pass_record_path_inserts_label_before_extension() {
        assert_eq!(
            capture_pass_record_path(Path::new("/cap/baseline.p1.json"), "A"),
            PathBuf::from("/cap/baseline.p1.A.json")
        );
        assert_eq!(
            capture_pass_record_path(Path::new("/cap/baseline.p1.json"), "W"),
            PathBuf::from("/cap/baseline.p1.W.json")
        );
        // No extension → append the label.
        assert_eq!(
            capture_pass_record_path(Path::new("/cap/baseline"), "B"),
            PathBuf::from("/cap/baseline.B")
        );
    }

    /// L12c — the SERIAL-LEG refusal at PARSE. `--capture-passes` puts no spec on the wire by
    /// construction, but the SINGLE-pass capture path threads `args.spec` straight into the timed
    /// leg, so `--capture-baseline --mtp-depth N` would append a SPECULATIVE pair to the record that
    /// becomes the official SERIAL denominator. Refused by name, both spellings.
    ///
    /// NEGATIVE CONTROLS: the same spec flags parse fine WITHOUT `--capture-baseline` (a scored
    /// speculative leg is the normal case), and a plain capture with no spec flag parses fine.
    #[test]
    fn capture_baseline_refuses_the_spec_flags_by_name() {
        for spec in [
            vec!["--mtp-depth".to_string(), "2".to_string()],
            vec![
                "--candidate-spec".to_string(),
                r#"{"mode":"mtp","mtp":{"depth":2}}"#.to_string(),
            ],
        ] {
            let mut argv = base_local_iterate_argv();
            argv.extend(["--capture-baseline".to_string(), "rec.json".to_string()]);
            argv.extend(spec.clone());
            let err = parse_iterate_args(&argv).err().unwrap();
            assert!(err.contains(capture::CALIBRATION_SPEC_ARMED), "{err}");
            assert!(err.contains("a capture leg is always serial"), "{err}");

            // NEGATIVE CONTROL: the same spec flag WITHOUT --capture-baseline is a normal
            // speculative leg.
            let mut scored = base_local_iterate_argv();
            scored.extend(spec);
            assert!(parse_iterate_args(&scored).unwrap().is_some());
        }
        // NEGATIVE CONTROL: a plain capture (no spec flag) still parses.
        let mut plain = base_local_iterate_argv();
        plain.extend(["--capture-baseline".to_string(), "rec.json".to_string()]);
        assert!(parse_iterate_args(&plain).unwrap().is_some());
    }

    /// `--capture-passes` is a MODIFIER of `--capture-baseline`: refused BY NAME without it (mirrors
    /// `--capture-timed-only` / `--weights-digest`), so the multi-pass path can never reach a
    /// scored/official run.
    #[test]
    fn capture_passes_refused_without_capture_baseline() {
        let mut argv = base_local_iterate_argv();
        argv.extend(["--capture-passes".into(), "W,A,A,B,B".into()]);
        let err = parse_iterate_args(&argv).err().unwrap();
        assert!(
            err.contains("--capture-passes requires --capture-baseline"),
            "{err}"
        );
    }

    /// `--capture-passes` SUBSUMES `--capture-timed-only`: giving both is a usage error (the per-pass
    /// timed-only decision is positional under `--capture-passes`), so it is refused, never silently
    /// ignored.
    #[test]
    fn capture_passes_conflicts_with_capture_timed_only() {
        let mut argv = base_local_iterate_argv();
        argv.extend([
            "--capture-baseline".into(),
            "/cap/baseline.p1.json".into(),
            "--capture-passes".into(),
            "W,A".into(),
            "--capture-timed-only".into(),
        ]);
        let err = parse_iterate_args(&argv).err().unwrap();
        assert!(
            err.contains("--capture-passes conflicts with --capture-timed-only"),
            "{err}"
        );
    }

    /// The happy path parses: local-iterate + `--capture-baseline` + `--capture-passes` yields the
    /// label vector on `IterateArgs`.
    #[test]
    fn capture_passes_parses_with_capture_baseline() {
        let mut argv = base_local_iterate_argv();
        argv.extend([
            "--capture-baseline".into(),
            "/cap/baseline.p1.json".into(),
            "--capture-passes".into(),
            "W,A,A,B,B".into(),
        ]);
        let parsed = parse_iterate_args(&argv).unwrap().unwrap();
        assert_eq!(
            parsed.capture_passes.as_deref(),
            Some(["W", "A", "A", "B", "B"].map(String::from).as_slice())
        );
    }

    /// §13b — the ENGINE-RESOURCE PASSTHROUGH reaches the parsed args in command-line order, and
    /// the timed spawn argv carries each one as `--resource NAME=PATH` AHEAD of the v1.1 gate.
    #[test]
    fn engine_resources_parse_in_order_and_reach_the_spawn_argv() {
        let argv = vec![
            "--engine".into(),
            "e".into(),
            "--weights".into(),
            "w".into(),
            "--golden".into(),
            "g".into(),
            "--engine-resource".into(),
            "qwen4exp.ngramRowSource=/data/ngram".into(),
            "--engine-resource".into(),
            "rows-2=/data/other".into(),
        ];
        let parsed = parse_iterate_args(&argv).unwrap().unwrap();
        assert_eq!(
            parsed
                .engine_resources
                .iter()
                .map(|r| (r.name.as_str(), r.path.as_str()))
                .collect::<Vec<_>>(),
            vec![
                ("qwen4exp.ngramRowSource", "/data/ngram"),
                ("rows-2", "/data/other"),
            ]
        );
        assert_eq!(
            free_run_spawn_args(&parsed.engine_resources),
            vec![
                "--resource".to_string(),
                "qwen4exp.ngramRowSource=/data/ngram".to_string(),
                "--resource".to_string(),
                "rows-2=/data/other".to_string(),
                "--speculative-protocol".to_string(),
                "v1.1".to_string(),
            ]
        );
    }

    /// A duplicate resource NAME, and a malformed value, are USAGE errors at parse — before any
    /// engine spawn.
    #[test]
    fn a_duplicate_or_malformed_engine_resource_refuses_at_parse() {
        let base = |extra: &[&str]| -> Vec<String> {
            let mut v: Vec<String> = ["--engine", "e", "--weights", "w", "--golden", "g"]
                .iter()
                .map(|s| s.to_string())
                .collect();
            v.extend(extra.iter().map(|s| s.to_string()));
            v
        };
        let err = parse_iterate_args(&base(&[
            "--engine-resource",
            "rows=/a",
            "--engine-resource",
            "rows=/b",
        ]))
        .err()
        .unwrap();
        assert!(err.contains("declared twice"), "{err}");
        let err = parse_iterate_args(&base(&["--engine-resource", "rows"]))
            .err()
            .unwrap();
        assert!(err.contains("no '='"), "{err}");
    }

    /// The engine repository's wrapper PROBES `benchd iterate --help` for `--engine-resource` and
    /// passes the flag only when it is there
    /// (`tools/qwen38-125b-a6b-measure-and-score.sh`). Renaming or dropping the flag from the usage
    /// text silently disarms the n-gram row source, so the name is pinned here — on both verbs the
    /// contract §13b names.
    #[test]
    fn usage_lists_the_engine_resource_flag_the_engine_script_greps_for() {
        assert!(
            ITERATE_USAGE.contains("--engine-resource"),
            "iterate --help must name --engine-resource: the engine wrapper greps for it"
        );
        assert!(ITERATE_USAGE.contains("--engine-resource <NAME=PATH>"));
        assert!(CORRECTNESS_USAGE.contains("--engine-resource <NAME=PATH>"));
    }

    /// OFFICIAL IS BYTE-UNTOUCHED by `--capture-passes`. `--capture-baseline` (and therefore
    /// `--capture-passes`, which requires it) is refused on `--mode official` at parse, so the
    /// multi-pass path is structurally unreachable from the sole scored mode; and a plain official
    /// parse carries `capture_passes: None`.
    #[test]
    fn capture_passes_never_reaches_official() {
        // official + capture-baseline + capture-passes: refused at parse by the capture-baseline
        // mode gate (official also requires --contract, but the capture-baseline mode refusal is the
        // one that proves capture can't touch official).
        let argv = vec![
            "--engine".into(),
            "e".into(),
            "--weights".into(),
            "w".into(),
            "--golden".into(),
            "g".into(),
            "--mode".into(),
            "official".into(),
            "--contract".into(),
            "c".into(),
            "--capture-baseline".into(),
            "/cap/baseline.p1.json".into(),
            "--capture-passes".into(),
            "W,A".into(),
        ];
        let err = parse_iterate_args(&argv).err().unwrap();
        assert!(
            err.contains("--capture-baseline is the local-iterate CAPTURE MODE"),
            "capture must be refused on official at parse: {err}"
        );

        // A plain official run carries no capture_passes.
        let official = vec![
            "--engine".into(),
            "e".into(),
            "--weights".into(),
            "w".into(),
            "--golden".into(),
            "g".into(),
            "--mode".into(),
            "official".into(),
            "--contract".into(),
            "c".into(),
        ];
        let parsed = parse_iterate_args(&official).unwrap().unwrap();
        assert_eq!(parsed.mode, Mode::Official);
        assert!(parsed.capture_passes.is_none());
    }

    /// THE ENGINE-SIDE CONTRACT. `tools/qwen38-125b-a6b-measure-and-score.sh` greps
    /// `benchd iterate --help` for `--mtp-depth` and REFUSES to run a declared MTP leg against a
    /// benchd that does not list it — rather than silently measuring serial against the MTP
    /// oracle. So the usage text carrying this exact string is a WIRE CONTRACT, not documentation.
    #[test]
    fn iterate_usage_lists_the_mtp_depth_flag_the_engine_script_greps_for() {
        assert!(
            ITERATE_USAGE.contains("--mtp-depth"),
            "measure-and-score.sh refuses a declared MTP leg unless `iterate --help` lists \
             --mtp-depth; the usage text must carry it"
        );
        assert!(ITERATE_USAGE.contains("--mtp-depth <N>"));
        // `-h`/`--help` prints the usage to STDOUT and exits 0 (`run_iterate`'s `Ok(None)` arm).
        assert!(parse_iterate_args(&["--help".to_string()])
            .unwrap()
            .is_none());
        assert!(parse_iterate_args(&["-h".to_string()]).unwrap().is_none());
    }

    /// `--mtp-depth N` builds the module spec the timed free-run decode window requests; ABSENT is
    /// no spec at all, which is today's byte-for-byte serial behaviour.
    #[test]
    fn mtp_depth_builds_the_wire_spec_and_absent_means_no_spec() {
        let mut argv = base_local_iterate_argv();
        argv.extend(["--mtp-depth".into(), "2".into()]);
        let parsed = parse_iterate_args(&argv).unwrap().unwrap();
        assert_eq!(parsed.spec, Some(bench_protocol::SpecConfig::mtp(2)));
        assert_eq!(
            serde_json::to_string(&parsed.spec.unwrap()).unwrap(),
            r#"{"mode":"mtp","mtp":{"depth":2}}"#
        );

        let plain = parse_iterate_args(&base_local_iterate_argv())
            .unwrap()
            .unwrap();
        assert_eq!(plain.spec, None, "no flag ⇒ no spec on the wire");
    }

    /// The OFFICIAL (scored) path takes the flag too — that is the whole point, since the engine's
    /// measure-and-score.sh appends it to an `--mode official` invocation.
    #[test]
    fn mtp_depth_is_accepted_on_the_official_scored_path() {
        let argv = vec![
            "--engine".into(),
            "e".into(),
            "--weights".into(),
            "w".into(),
            "--golden".into(),
            "g".into(),
            "--mode".into(),
            "official".into(),
            "--contract".into(),
            "c".into(),
            "--mtp-depth".into(),
            "1".into(),
        ];
        let parsed = parse_iterate_args(&argv).unwrap().unwrap();
        assert_eq!(parsed.mode, Mode::Official);
        assert_eq!(parsed.spec, Some(bench_protocol::SpecConfig::mtp(1)));
    }

    /// The spec flags are MUTUALLY EXCLUSIVE, depth 0 is named as the serial leg it is, and the
    /// draft-depth CAP is enforced at parse — before any GPU work.
    #[test]
    fn mtp_depth_refusals_are_by_name() {
        let mut both = base_local_iterate_argv();
        both.extend([
            "--mtp-depth".into(),
            "2".into(),
            "--candidate-spec".into(),
            r#"{"mode":"mtp","mtp":{"depth":2}}"#.into(),
        ]);
        assert!(parse_iterate_args(&both)
            .err()
            .unwrap()
            .contains("mutually exclusive"));

        let mut zero = base_local_iterate_argv();
        zero.extend(["--mtp-depth".into(), "0".into()]);
        assert!(parse_iterate_args(&zero)
            .err()
            .unwrap()
            .contains("--mtp-depth 0 is the SERIAL leg"));

        let mut over = base_local_iterate_argv();
        over.extend(["--mtp-depth".into(), "33".into()]);
        assert!(parse_iterate_args(&over)
            .err()
            .unwrap()
            .contains("exceeds the maximum draft depth cap"));

        let mut junk = base_local_iterate_argv();
        junk.extend(["--mtp-depth".into(), "two".into()]);
        assert!(parse_iterate_args(&junk)
            .err()
            .unwrap()
            .contains("invalid --mtp-depth"));

        // The envelope is CLOSED: an unknown key in an explicit spec is refused.
        let mut bad_spec = base_local_iterate_argv();
        bad_spec.extend([
            "--candidate-spec".into(),
            r#"{"mode":"mtp","mtp":{"depth":2},"nope":1}"#.into(),
        ]);
        assert!(parse_iterate_args(&bad_spec)
            .err()
            .unwrap()
            .contains("spec JSON parse failed"));
    }

    /// FIX 1 (a) — benchd's engine spawn argv for the free-run/timed/capture legs carries
    /// `--speculative-protocol v1.1`, the flag the MLX worker gates its `free_run_decode` hello
    /// advertisement on (`RuntimeWorkerGenericDispatch.swift`). Without it the worker emits a
    /// v1-only hello and benchd's own capture path REFUSES the free-run verbs — the ~16 s pre-load
    /// refusal that blocked the MLX calibration. This restores the flag the retired flow-B
    /// `leg_spawn_args` always carried on a free-run leg (commit 1a90c3a).
    ///
    /// The (b) half — a worker hello that advertises `free_run_decode` lets the capture path
    /// proceed, and one that does not is refused fail-closed — is the runner's capability gate,
    /// covered end-to-end via the mock in `bench-runner`'s `session_acceptance` suite
    /// (`free_run_positive_control_round_trip` and `free_run_capability_is_advertised_and_gated`).
    /// ONE GATE, TWO BUILDERS. `free_run_spawn_args` (the iterate/official spawns) and
    /// `measure_job::leg_spawn_args` (the measure-job legs) both take the v1.1 flag from
    /// `measure_job::speculative_protocol_args`, so the gate cannot be applied in one and
    /// forgotten in the other — which is exactly how they had drifted: this builder pushed the
    /// flag unconditionally while the measure-job builder gated it on the leg's regime. A
    /// teacher-forced leg's gate-off spawn IS the standing v1-compat proof, so it must keep
    /// getting no flag.
    #[test]
    fn both_spawn_argv_builders_take_the_v1_1_flag_from_one_gate() {
        assert_eq!(
            measure_job::speculative_protocol_args(measure_job::LegRegime::FreeRunV1_1),
            vec!["--speculative-protocol".to_string(), "v1.1".to_string()],
        );
        assert!(
            measure_job::speculative_protocol_args(measure_job::LegRegime::TeacherForcedV1)
                .is_empty(),
            "a teacher-forced leg must spawn WITHOUT the v1.1 gate"
        );
        // The iterate/official builder is the free-run arm of that one gate, appended to the
        // declared engine resources — nothing pushed on the side.
        let declared = vec![engine_resource::parse_engine_resource("rows=/tmp/rows.bin").unwrap()];
        let mut expected = engine_resource::spawn_args(&declared);
        expected.extend(measure_job::speculative_protocol_args(
            measure_job::LegRegime::FreeRunV1_1,
        ));
        assert_eq!(free_run_spawn_args(&declared), expected);
        // The measure-job builder withholds it on a teacher-forced leg, and that leg's argv is
        // the one `spawn_official_worker` now carries verbatim.
        let tf = measure_job::leg_spawn_args("/h", None, measure_job::LegRegime::TeacherForcedV1);
        assert!(
            !tf.iter().any(|a| a == "--speculative-protocol"),
            "the teacher-forced leg argv grew the v1.1 gate: {tf:?}"
        );
    }

    /// (b) — EVERY official spawn goes through `spawn_official_worker`, and its `None` arm is the
    /// unsandboxed path a host without Seatbelt (the linux CUDA box) runs on. The measure-job legs
    /// used to call `spawn_official_sandboxed` directly, so they could not run there at all. This
    /// drives the `None` arm end-to-end and proves the caller's argv reaches the child VERBATIM,
    /// after the transport's own `runtime-worker --weights <DIR>` lead.
    #[cfg(unix)]
    #[test]
    fn unsandboxed_official_spawn_passes_the_callers_argv_verbatim() {
        use bench_runner::LineTransport;
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!("benchd-nosandbox-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let argv_out = dir.join("argv.txt");
        let engine = dir.join("engine.sh");
        std::fs::write(
            &engine,
            format!(
                "#!/bin/sh\nfor a in \"$@\"; do printf '%s\\n' \"$a\"; done > '{}'\nprintf 'ready\\n'\n",
                argv_out.display()
            ),
        )
        .unwrap();
        let mut perms = std::fs::metadata(&engine).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&engine, perms).unwrap();

        let engine_str = engine.to_string_lossy().to_string();
        let extra = measure_job::leg_spawn_args(
            "/heads/pinned",
            None,
            measure_job::LegRegime::TeacherForcedV1,
        );
        let mut transport =
            spawn_official_worker(None, &engine_str, "/w/qwen", &extra, &[]).unwrap();
        assert_eq!(
            transport.read_line().unwrap(),
            Some("ready".to_string()),
            "the unsandboxed spawn never started"
        );

        let seen: Vec<String> = std::fs::read_to_string(&argv_out)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(seen, ChildStdioTransport::build_args("/w/qwen", &extra));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn free_run_legs_spawn_with_speculative_protocol_v1_1() {
        assert_eq!(
            free_run_spawn_args(&[]),
            vec!["--speculative-protocol".to_string(), "v1.1".to_string()],
            "the free-run spawn args must be exactly the flow-B free-run leg flag+value"
        );
        // The FULL engine argv benchd builds for a free-run leg: the leading `runtime-worker
        // --weights <DIR>` (unchanged) plus the flag, so the worker advertises free_run_decode.
        let argv = ChildStdioTransport::build_args("/w/qwen", &free_run_spawn_args(&[]));
        assert_eq!(
            argv,
            vec![
                "runtime-worker".to_string(),
                "--weights".to_string(),
                "/w/qwen".to_string(),
                "--speculative-protocol".to_string(),
                "v1.1".to_string(),
            ]
        );
    }

    /// FIX 2 — the sandbox provenance sealed into the integrity sidecar (a8 ruling b). A resolved
    /// Seatbelt plan → `seatbelt` (macOS official); an official run with no plan → the host has no
    /// Seatbelt (the linux-aarch64 CUDA box), sealed honestly as `none (linux)`; the never-sandboxed
    /// local modes → `none`. Pure decision, asserted on any host regardless of `target_os`.
    #[test]
    fn sandbox_provenance_covers_seatbelt_linux_and_local() {
        assert_eq!(sandbox_provenance(true, true), "seatbelt");
        assert_eq!(sandbox_provenance(true, false), "none (linux)");
        assert_eq!(sandbox_provenance(false, false), "none");
        // A local run never resolves a plan, so it can never claim seatbelt provenance.
        assert_eq!(sandbox_provenance(false, false), SANDBOX_PROVENANCE_NONE);
    }

    /// FIX 2 — the sidecar actually SERIALIZES the sandbox provenance under the `sandbox` key with
    /// the ruled value, so a Linux official seal names its (absent) isolation honestly. The key's
    /// membership in the runner roster is held by `integrity_sidecar_is_a_superset_of_the_jq_pretty_object`.
    #[test]
    fn integrity_sidecar_seals_sandbox_provenance() {
        let s = IntegritySidecar {
            score_path: "score.json".into(),
            score_sha256: "a".into(),
            weights_path: "w".into(),
            weights_sha256: "b".into(),
            weights_file_count: 1,
            weights_byte_count: 2,
            golden_path: "[private]".into(),
            golden_sha256: "c".into(),
            transform_source_sha256: String::new(),
            candidate_executable: "engine".into(),
            candidate_executable_sha256: "d".into(),
            baseline_executable: String::new(),
            baseline_executable_sha256: String::new(),
            candidate_executable_resolution: ENGINE_RESOLUTION_CANONICAL.into(),
            benchd_executable: "benchd".into(),
            benchd_executable_sha256: "e".into(),
            candidate_workspace_sha256: String::new(),
            sandbox: SANDBOX_PROVENANCE_NONE_LINUX.into(),
        };
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(v["sandbox"], serde_json::json!("none (linux)"));
    }

    // ---------------------------------------------------------------------------------------
    // The per-subcommand FLAG TABLES. One test per subcommand feeds the documented flag set and
    // asserts the parsed struct field by field, so a flag that is dropped from a table (or stops
    // reaching its field) fails here instead of going quietly missing from a run.
    // ---------------------------------------------------------------------------------------

    /// `argv` as the `Vec<String>` every `parse_*` takes.
    fn argv(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    /// Every flag spelling a table carries, aliases included.
    fn table_names(specs: &[FlagSpec]) -> Vec<&'static str> {
        specs.iter().flat_map(|s| s.names.iter().copied()).collect()
    }

    /// The flag spellings an argv mentions (every token the table knows).
    fn argv_names(specs: &[FlagSpec], args: &[String]) -> Vec<&'static str> {
        let known = table_names(specs);
        args.iter()
            .filter_map(|a| known.iter().copied().find(|n| *n == a.as_str()))
            .collect()
    }

    /// Assert the given command lines, together, exercise EVERY flag of the table that a run can
    /// accept — the check that keeps a new table row from being added without a case that parses
    /// it. A `Refused` row is covered by its own refusal assertion instead.
    fn assert_covers_table(specs: &'static [FlagSpec], cases: &[&[String]]) {
        let mut seen: Vec<&str> = cases
            .iter()
            .flat_map(|args| argv_names(specs, args))
            .collect();
        seen.sort_unstable();
        seen.dedup();
        for spec in specs {
            if matches!(spec.arity, FlagArity::Refused(_)) {
                continue;
            }
            for name in spec.names {
                assert!(
                    seen.contains(name),
                    "{name} is in the table but no table test parses it"
                );
            }
        }
    }

    /// Assert every flag the table ACCEPTS is also DOCUMENTED in the subcommand's usage text, so
    /// the two cannot drift apart silently. A `Refused` row is a retired spelling the usage
    /// deliberately no longer advertises; its refusal text is asserted on its own.
    fn assert_usage_documents_table(usage: &str, specs: &'static [FlagSpec]) {
        for spec in specs {
            if matches!(spec.arity, FlagArity::Refused(_)) {
                continue;
            }
            for name in spec.names {
                assert!(
                    usage.contains(name),
                    "{name} is in the table but not in the usage text"
                );
            }
        }
    }

    /// Assert dropping each REQUIRED flag from an otherwise complete command line refuses by name.
    fn assert_required_flags_refuse(
        specs: &'static [FlagSpec],
        parse: impl Fn(&[String]) -> Result<Option<()>, String>,
        full: &[String],
    ) {
        for spec in specs.iter().filter(|s| s.required) {
            let mut trimmed: Vec<String> = Vec::new();
            let mut i = 0;
            while i < full.len() {
                if spec.names.contains(&full[i].as_str()) {
                    i += if matches!(spec.arity, FlagArity::Switch) {
                        1
                    } else {
                        2
                    };
                    continue;
                }
                trimmed.push(full[i].clone());
                i += 1;
            }
            let err = parse(&trimmed).expect_err("a missing required flag refuses");
            let note = if matches!(spec.arity, FlagArity::Repeatable) {
                " (repeatable)"
            } else {
                ""
            };
            assert_eq!(err, format!("missing required {}{note}", spec.names[0]));
        }
    }

    #[test]
    fn measure_job_table_parses_its_whole_flag_set() {
        // The teacher-forced case: a serial --candidate-spec keeps --tokens free (the free-run
        // regime pins N), and carries every optional flag the table admits alongside it.
        let full = argv(&[
            "--candidate",
            "cand-ws",
            "--baseline",
            "base-ws",
            "--write-gate-base",
            "abc1234",
            "--weights",
            "w-dir",
            "--golden",
            "g1.json",
            "--golden",
            "g2.json",
            "--contract",
            "fixture.json",
            "--correctness-golden",
            "hidden.json",
            "--tokens",
            "256",
            "--candidate-spec",
            r#"{"mode":"serial"}"#,
            "--baseline-spec",
            r#"{"mode":"serial"}"#,
            "--min-pairs",
            "2",
            "--target-pairs",
            "5",
            "--tag",
            "run-tag",
            "--gates-producer",
            "benchmark-sh",
            "--out",
            "out-dir",
            "--prompt",
            "p.txt",
            "--prompt-sha256",
            &"a".repeat(64),
            "--target-id",
            "pool.0",
            "--exactness-probe",
            "per-pair",
            "--preflight-only",
            "--calibration-bootstrap",
            "--local-dev",
        ]);
        let parsed = parse_measure_job_args(&full).unwrap().unwrap();
        assert_eq!(parsed.candidate, PathBuf::from("cand-ws"));
        assert_eq!(parsed.baseline, PathBuf::from("base-ws"));
        assert_eq!(parsed.write_gate_base.as_deref(), Some("abc1234"));
        assert_eq!(parsed.weights, PathBuf::from("w-dir"));
        assert_eq!(
            parsed.goldens,
            vec![PathBuf::from("g1.json"), PathBuf::from("g2.json")]
        );
        assert_eq!(parsed.contract, PathBuf::from("fixture.json"));
        assert_eq!(
            parsed.correctness_golden,
            Some(PathBuf::from("hidden.json"))
        );
        assert_eq!(parsed.tokens, 256);
        assert!(!parsed.candidate_regime.is_free_run());
        assert_eq!(parsed.mtp_depth, Some(0));
        assert_eq!(parsed.candidate_spec.mode, "serial");
        assert_eq!(parsed.baseline_spec.mode, "serial");
        assert_eq!(
            parsed.candidate_spec_source,
            measure_job::SPEC_SOURCE_CLI_OVERRIDE
        );
        assert_eq!(
            parsed.baseline_spec_source,
            measure_job::SPEC_SOURCE_CLI_OVERRIDE
        );
        assert_eq!(parsed.min_pairs, 2);
        assert_eq!(parsed.target_pairs, 5);
        assert_eq!(parsed.tag, "run-tag");
        assert_eq!(parsed.out, PathBuf::from("out-dir"));
        assert_eq!(parsed.prompt, Some(PathBuf::from("p.txt")));
        assert_eq!(
            parsed.prompt_sha256.as_deref(),
            Some("a".repeat(64).as_str())
        );
        assert_eq!(parsed.target_id.as_deref(), Some("pool.0"));
        assert_eq!(parsed.exactness_probe, measure_job::ExactnessProbe::PerPair);
        assert!(parsed.preflight_only);
        assert!(parsed.calibration_bootstrap);
        assert!(parsed.local_dev);
        assert_eq!(parsed.gates_producer, "benchmark-sh");

        // The convenience half of the spec pair, and the per-prompt ALIASES, which set the same
        // fields as their canonical spellings.
        let depth = argv(&[
            "--candidate",
            "cand-ws",
            "--baseline",
            "base-ws",
            "--weights",
            "w-dir",
            "--golden",
            "g1.json",
            "--contract",
            "fixture.json",
            "--mtp-depth",
            "4",
            "--min-pairs-per-prompt",
            "1",
            "--pairs-per-prompt",
            "3",
            "--tag",
            "t",
            "--out",
            "o",
        ]);
        let parsed = parse_measure_job_args(&depth).unwrap().unwrap();
        assert_eq!(parsed.mtp_depth, Some(4));
        assert_eq!(parsed.candidate_spec.mode, "mtp");
        assert_eq!(
            parsed.candidate_spec_source,
            measure_job::SPEC_SOURCE_MTP_DEPTH_FLAG
        );
        assert_eq!(parsed.min_pairs, 1);
        assert_eq!(parsed.target_pairs, 3);
        // A speculating candidate is the v1.1 free-run regime at the RULED N.
        assert!(parsed.candidate_regime.is_free_run());
        assert_eq!(parsed.tokens, measure_job::FREE_RUN_DECODE_TOKENS);
        // Omitted --gates-producer / --exactness-probe seal their documented defaults.
        assert_eq!(parsed.gates_producer, GATES_PRODUCER_UNDECLARED);
        assert_eq!(parsed.exactness_probe, measure_job::ExactnessProbe::Once);
        assert!(!parsed.preflight_only);
        assert!(!parsed.calibration_bootstrap);
        assert!(!parsed.local_dev);
        assert_eq!(parsed.write_gate_base, None);
        assert_eq!(parsed.correctness_golden, None);

        assert_covers_table(MEASURE_JOB_FLAGS, &[&full, &depth]);
        assert_usage_documents_table(MEASURE_JOB_USAGE, MEASURE_JOB_FLAGS);
        assert_required_flags_refuse(
            MEASURE_JOB_FLAGS,
            |args| parse_measure_job_args(args).map(|p| p.map(|_| ())),
            &full,
        );
        // The retired spellings refuse by name rather than being accepted or ignored.
        for (flag, want) in [
            (
                "--depth",
                "--depth was renamed to --mtp-depth (candidate MTP depth, >= 2; serial control is \
                 the depth-0 constant)",
            ),
            (
                "--baseline-prefill-spt",
                "--baseline-prefill-spt is not a measure-job flag: the baseline is a WORKSPACE \
                 (--baseline <WS>) now, not an engine/seconds-per-token override",
            ),
            (
                "--baseline-decode-spt",
                "--baseline-decode-spt is not a measure-job flag: the baseline is a WORKSPACE \
                 (--baseline <WS>) now, not an engine/seconds-per-token override",
            ),
            (
                "--baseline-engine",
                "--baseline-engine is not a measure-job flag: the baseline is a WORKSPACE \
                 (--baseline <WS>) now, not an engine/seconds-per-token override",
            ),
            (
                "--engine",
                "--engine is not a measure-job flag: the baseline is a WORKSPACE (--baseline <WS>) \
                 now, not an engine/seconds-per-token override",
            ),
        ] {
            let err = parse_measure_job_args(&argv(&[flag, "x"])).unwrap_err();
            assert_eq!(err, want);
        }
    }

    #[test]
    fn overlay_timing_table_parses_its_whole_flag_set() {
        let full = argv(&[
            "--gates-score",
            "gates-score.json",
            "--results",
            "results.json",
            "--score-path",
            "score.json",
            "--integrity",
            "benchmark-integrity.json",
            "--contract",
            "fixture.json",
        ]);
        let parsed = parse_overlay_timing_args(&full).unwrap().unwrap();
        assert_eq!(parsed.gates_score, PathBuf::from("gates-score.json"));
        assert_eq!(parsed.results, PathBuf::from("results.json"));
        assert_eq!(parsed.score_path, PathBuf::from("score.json"));
        assert_eq!(
            parsed.integrity,
            Some(PathBuf::from("benchmark-integrity.json"))
        );
        assert_eq!(parsed.contract, Some(PathBuf::from("fixture.json")));

        let bare = argv(&[
            "--gates-score",
            "g.json",
            "--results",
            "r.json",
            "--score-path",
            "s.json",
        ]);
        let parsed = parse_overlay_timing_args(&bare).unwrap().unwrap();
        assert_eq!(parsed.integrity, None);
        assert_eq!(parsed.contract, None);

        assert_covers_table(OVERLAY_TIMING_FLAGS, &[&full]);
        assert_usage_documents_table(OVERLAY_TIMING_USAGE, OVERLAY_TIMING_FLAGS);
        assert_required_flags_refuse(
            OVERLAY_TIMING_FLAGS,
            |args| parse_overlay_timing_args(args).map(|p| p.map(|_| ())),
            &full,
        );
    }

    #[test]
    fn iterate_table_parses_its_whole_flag_set() {
        // Two command lines cover the table, because the surface fences two groups apart: the
        // CAPTURE modifiers are local-iterate only and refuse a spec flag, and `--mode official`
        // requires a --contract. Together they parse every flag the table admits.
        let digest = format!("{}:{}:{}", "b".repeat(64), 1234, 7);
        let capture = argv(&[
            "--engine",
            "engine-bin",
            "--weights",
            "w-dir",
            "--golden",
            "g.json",
            "--baseline-prefill-spt",
            "0.001",
            "--baseline-decode-spt",
            "0.002",
            "--mode",
            "local-iterate",
            "--score-path",
            "out.json",
            "--golden-sha256",
            &"c".repeat(64),
            "--golden-bytes",
            "42",
            "--cool-gate",
            "--strict",
            "--capture-baseline",
            "record.json",
            "--capture-passes",
            "W,A,A",
            "--weights-digest",
            &digest,
            "--baseline-workspace",
            "ref-ws",
            "--baseline-calibration",
            "calib.json",
            "--box",
            "box-3",
            "--control-golden",
            "serial.json",
            "--control-golden-sha256",
            &"d".repeat(64),
            "--control-golden-bytes",
            "43",
            "--engine-resource",
            "qwen4exp.ngramRowSource=/rows",
            "--engine-resource",
            "other.thing=/p",
        ]);
        let parsed = parse_iterate_args(&capture).unwrap().unwrap();
        assert_eq!(parsed.engine, "engine-bin");
        assert_eq!(parsed.weights, PathBuf::from("w-dir"));
        assert_eq!(parsed.goldens, [PathBuf::from("g.json")]);
        assert_eq!(parsed.baseline_prefill_spt, Some(0.001));
        assert_eq!(parsed.baseline_decode_spt, Some(0.002));
        assert_eq!(parsed.mode, Mode::LocalIterate);
        assert_eq!(parsed.score_path, PathBuf::from("out.json"));
        let pin = &parsed.golden_pins[0];
        assert_eq!(pin.sha256, "c".repeat(64));
        assert_eq!(pin.bytes, 42);
        assert_eq!(parsed.cool_gate, Some(true));
        assert!(parsed.strict);
        assert_eq!(parsed.capture_baseline, Some(PathBuf::from("record.json")));
        assert!(!parsed.capture_timed_only);
        assert_eq!(
            parsed.capture_passes,
            Some(vec!["W".to_string(), "A".to_string(), "A".to_string()])
        );
        let wd = parsed.weights_digest.as_ref().unwrap();
        assert_eq!(wd.sha256, "b".repeat(64));
        assert_eq!(wd.byte_count, 1234);
        assert_eq!(wd.file_count, 7);
        assert_eq!(parsed.contract, None);
        assert_eq!(parsed.spec, None);
        assert_eq!(parsed.baseline_workspace, Some(PathBuf::from("ref-ws")));
        assert_eq!(
            parsed.baseline_calibration,
            Some(PathBuf::from("calib.json"))
        );
        assert_eq!(parsed.box_name.as_deref(), Some("box-3"));
        assert_eq!(parsed.control_goldens, [PathBuf::from("serial.json")]);
        let ctl = &parsed.control_golden_pins[0];
        assert_eq!(ctl.sha256, "d".repeat(64));
        assert_eq!(ctl.bytes, 43);
        assert_eq!(parsed.engine_resources.len(), 2);
        assert_eq!(parsed.engine_resources[0].name, "qwen4exp.ngramRowSource");
        assert_eq!(parsed.engine_resources[0].path, "/rows");
        assert_eq!(parsed.engine_resources[1].name, "other.thing");

        // The SCORED half: official + contract + the spec pair's convenience spelling, and the
        // cool gate forced OFF.
        let official = argv(&[
            "--engine",
            "engine-bin",
            "--weights",
            "w-dir",
            "--golden",
            "g.json",
            "--mode",
            "official",
            "--contract",
            "fixture.json",
            "--mtp-depth",
            "3",
            "--no-cool-gate",
        ]);
        let parsed = parse_iterate_args(&official).unwrap().unwrap();
        assert_eq!(parsed.mode, Mode::Official);
        assert_eq!(parsed.contract, Some(PathBuf::from("fixture.json")));
        let spec = parsed.spec.as_ref().unwrap();
        assert_eq!(spec.mode, "mtp");
        assert_eq!(spec.mtp.as_ref().unwrap().depth, Some(3));
        assert_eq!(parsed.cool_gate, Some(false));
        // The default score name for a mode that was given no --score-path.
        assert_eq!(parsed.score_path, PathBuf::from("score.json"));

        // The remaining two rows: the explicit spec object and the single-invocation timed-only
        // capture modifier, which the two cases above fence out of each other.
        let explicit = argv(&[
            "--engine",
            "engine-bin",
            "--weights",
            "w-dir",
            "--golden",
            "g.json",
            "--capture-baseline",
            "record.json",
            "--capture-timed-only",
        ]);
        let parsed = parse_iterate_args(&explicit).unwrap().unwrap();
        assert!(parsed.capture_timed_only);
        let spec_only = argv(&[
            "--engine",
            "engine-bin",
            "--weights",
            "w-dir",
            "--golden",
            "g.json",
            "--candidate-spec",
            r#"{"mode":"mtp","mtp":{"depth":2}}"#,
        ]);
        let parsed = parse_iterate_args(&spec_only).unwrap().unwrap();
        assert_eq!(
            parsed.spec.as_ref().unwrap().mtp.as_ref().unwrap().depth,
            Some(2)
        );
        // The DEFAULT mode of an argv that gives no `--mode`, its default score name, and the
        // untouched defaults of every absent flag.
        assert_eq!(parsed.mode, Mode::LocalIterate);
        assert_eq!(parsed.score_path, PathBuf::from("score.local-iterate.json"));
        assert_eq!(parsed.cool_gate, None);
        assert!(!parsed.strict);
        assert!(parsed.engine_resources.is_empty());
        assert!(parsed.golden_pins.is_empty());
        assert!(parsed.control_golden_pins.is_empty());

        assert_covers_table(ITERATE_FLAGS, &[&capture, &official, &explicit, &spec_only]);
        assert_usage_documents_table(ITERATE_USAGE, ITERATE_FLAGS);
        assert_required_flags_refuse(
            ITERATE_FLAGS,
            |args| parse_iterate_args(args).map(|p| p.map(|_| ())),
            &official,
        );
    }

    #[test]
    fn validate_golden_table_parses_its_whole_flag_set() {
        let full = argv(&[
            "--golden",
            "g.json",
            "--golden-sha256",
            &"a".repeat(64),
            "--golden-bytes",
            "17",
            "--contract",
            "fixture.json",
            "--track",
            "track-1",
            "--gates-only",
        ]);
        let parsed = parse_validate_golden_args(&full).unwrap().unwrap();
        assert_eq!(parsed.golden, PathBuf::from("g.json"));
        assert_eq!(parsed.contract, PathBuf::from("fixture.json"));
        assert_eq!(parsed.track.as_deref(), Some("track-1"));
        assert!(parsed.gates_only);
        let pin = parsed.pin.as_ref().expect("both pin flags given");
        assert_eq!(pin.sha256, "a".repeat(64));
        assert_eq!(pin.bytes, 17);

        // The unpinned, untracked line: the pin is absent (not half-built) and `--track` falls to
        // the env at RUN time, so the parse leaves it `None` rather than guessing.
        let bare = argv(&["--golden", "g.json", "--contract", "fixture.json"]);
        let parsed = parse_validate_golden_args(&bare).unwrap().unwrap();
        assert_eq!(parsed.pin, None);
        assert_eq!(parsed.track, None);
        assert!(!parsed.gates_only);

        // Half a pin is refused by the pair rule, naming both flags.
        assert_eq!(
            parse_validate_golden_args(&argv(&[
                "--golden",
                "g.json",
                "--contract",
                "fixture.json",
                "--golden-sha256",
                "abc",
            ]))
            .map(|_| ())
            .unwrap_err(),
            "--golden-sha256 and --golden-bytes must be given together"
        );

        assert_covers_table(VALIDATE_GOLDEN_FLAGS, &[&full]);
        assert_usage_documents_table(VALIDATE_GOLDEN_USAGE, VALIDATE_GOLDEN_FLAGS);
        assert_required_flags_refuse(
            VALIDATE_GOLDEN_FLAGS,
            |args| parse_validate_golden_args(args).map(|p| p.map(|_| ())),
            &full,
        );
    }

    #[test]
    fn correctness_table_parses_its_whole_flag_set() {
        let full = argv(&[
            "--engine",
            "engine-bin",
            "--weights",
            "w-dir",
            "--golden",
            "g.json",
            "--golden-sha256",
            &"b".repeat(64),
            "--golden-bytes",
            "42",
            "--manifest",
            "manifest.json",
            "--trusted",
            engine_resource::ENGINE_RESOURCE_FLAG,
            "qwen4exp.ngramRowSource=/rows",
            engine_resource::ENGINE_RESOURCE_FLAG,
            "aux=/aux",
        ]);
        let parsed = parse_correctness_args(&full).unwrap().unwrap();
        assert_eq!(parsed.engine, "engine-bin");
        assert_eq!(parsed.weights, PathBuf::from("w-dir"));
        assert_eq!(parsed.golden, PathBuf::from("g.json"));
        assert_eq!(parsed.manifest, Some(PathBuf::from("manifest.json")));
        assert!(parsed.trusted);
        let pin = parsed.pin.as_ref().expect("both pin flags given");
        assert_eq!(pin.sha256, "b".repeat(64));
        assert_eq!(pin.bytes, 42);
        // `--engine-resource` is REPEATABLE: every occurrence is kept, in the order typed.
        assert_eq!(
            parsed
                .engine_resources
                .iter()
                .map(|r| r.name.as_str())
                .collect::<Vec<_>>(),
            vec!["qwen4exp.ngramRowSource", "aux"]
        );

        let bare = argv(&["--engine", "e", "--weights", "w", "--golden", "g.json"]);
        let parsed = parse_correctness_args(&bare).unwrap().unwrap();
        assert_eq!(parsed.pin, None);
        assert_eq!(parsed.manifest, None);
        assert!(!parsed.trusted);
        assert!(parsed.engine_resources.is_empty());

        // A repeated resource NAME refuses at parse, as it did in the hand-written loop.
        assert_eq!(
            parse_correctness_args(&argv(&[
                "--engine",
                "e",
                "--weights",
                "w",
                "--golden",
                "g.json",
                engine_resource::ENGINE_RESOURCE_FLAG,
                "a=/x",
                engine_resource::ENGINE_RESOURCE_FLAG,
                "a=/y",
            ]))
            .map(|_| ())
            .unwrap_err(),
            "--engine-resource NAME \"a\" is declared twice: give each resource exactly once"
        );

        assert_covers_table(CORRECTNESS_FLAGS, &[&full]);
        assert_usage_documents_table(CORRECTNESS_USAGE, CORRECTNESS_FLAGS);
        assert_required_flags_refuse(
            CORRECTNESS_FLAGS,
            |args| parse_correctness_args(args).map(|p| p.map(|_| ())),
            &full,
        );
    }

    #[test]
    fn validate_weights_table_parses_its_whole_flag_set() {
        let full = argv(&["--weights", "w-dir", "--golden", "g.json"]);
        let parsed = parse_validate_weights_args(&full).unwrap().unwrap();
        assert_eq!(parsed.weights, PathBuf::from("w-dir"));
        assert_eq!(parsed.golden, Some(PathBuf::from("g.json")));

        let bare = argv(&["--weights", "w-dir"]);
        assert_eq!(
            parse_validate_weights_args(&bare).unwrap().unwrap().golden,
            None
        );

        assert_covers_table(VALIDATE_WEIGHTS_FLAGS, &[&full]);
        assert_usage_documents_table(VALIDATE_WEIGHTS_USAGE, VALIDATE_WEIGHTS_FLAGS);
        assert_required_flags_refuse(
            VALIDATE_WEIGHTS_FLAGS,
            |args| parse_validate_weights_args(args).map(|p| p.map(|_| ())),
            &full,
        );
    }

    #[test]
    fn weights_digest_table_parses_its_whole_flag_set() {
        let full = argv(&["--weights", "w-dir"]);
        let parsed = parse_weights_digest_args(&full).unwrap().unwrap();
        assert_eq!(parsed.weights, PathBuf::from("w-dir"));

        // A repeat keeps the LAST value, which is what assigning into an `Option` did.
        let repeated = argv(&["--weights", "first", "--weights", "second"]);
        assert_eq!(
            parse_weights_digest_args(&repeated)
                .unwrap()
                .unwrap()
                .weights,
            PathBuf::from("second")
        );

        assert_covers_table(WEIGHTS_DIGEST_FLAGS, &[&full]);
        assert_usage_documents_table(WEIGHTS_DIGEST_USAGE, WEIGHTS_DIGEST_FLAGS);
        assert_required_flags_refuse(
            WEIGHTS_DIGEST_FLAGS,
            |args| parse_weights_digest_args(args).map(|p| p.map(|_| ())),
            &full,
        );
    }

    #[test]
    fn every_cli_table_refuses_an_unknown_flag_and_prints_help() {
        // The three shared behaviours the copies used to each own: `-h`/`--help` is `Ok(None)` (the
        // caller prints the usage), an unrecognised flag refuses by name, and a value flag with no
        // value names itself.
        for parse in [
            (|a: &[String]| parse_iterate_args(a).map(|p| p.map(|_| ())))
                as fn(&[String]) -> Result<Option<()>, String>,
            |a: &[String]| parse_measure_job_args(a).map(|p| p.map(|_| ())),
            |a: &[String]| parse_overlay_timing_args(a).map(|p| p.map(|_| ())),
            |a: &[String]| parse_validate_golden_args(a).map(|p| p.map(|_| ())),
            |a: &[String]| parse_correctness_args(a).map(|p| p.map(|_| ())),
            |a: &[String]| parse_validate_weights_args(a).map(|p| p.map(|_| ())),
            |a: &[String]| parse_weights_digest_args(a).map(|p| p.map(|_| ())),
        ] {
            assert!(parse(&argv(&["--help"])).unwrap().is_none());
            assert!(parse(&argv(&["-h"])).unwrap().is_none());
            assert_eq!(
                parse(&argv(&["--bogus"])).unwrap_err(),
                "unknown flag \"--bogus\""
            );
        }
        assert_eq!(
            parse_iterate_args(&argv(&["--engine"])).err().unwrap(),
            "flag --engine requires a value"
        );
        // An ALIAS names ITSELF in the missing-value refusal, not its canonical spelling.
        assert_eq!(
            parse_measure_job_args(&argv(&["--min-pairs-per-prompt"])).unwrap_err(),
            "flag --min-pairs-per-prompt requires a value"
        );
    }

    #[test]
    fn a_repeated_value_flag_keeps_the_last_value_and_still_validates_the_earlier_ones() {
        // What the hand-written loops did by assigning into an `Option` as they walked: the LAST
        // occurrence wins, and an EARLIER occurrence that does not convert still refuses.
        let args = argv(&[
            "--gates-score",
            "first.json",
            "--gates-score",
            "second.json",
            "--results",
            "r.json",
            "--score-path",
            "s.json",
        ]);
        let parsed = parse_overlay_timing_args(&args).unwrap().unwrap();
        assert_eq!(parsed.gates_score, PathBuf::from("second.json"));

        let err = parse_iterate_args(&argv(&[
            "--engine",
            "e",
            "--weights",
            "w",
            "--golden",
            "g",
            "--baseline-prefill-spt",
            "nope",
            "--baseline-prefill-spt",
            "0.1",
        ]))
        .err()
        .unwrap();
        assert_eq!(err, "invalid f64 for --baseline-prefill-spt: \"nope\"");
    }
}
