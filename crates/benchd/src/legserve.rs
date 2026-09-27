//! PER-LEG RESIDENT ENGINES for the paired path — on BOTH platforms.
//!
//! A paired run measures two legs from two trees. On EITHER platform the model is owned by a
//! RESIDENT process, and that process belongs to ONE leg's tree:
//!
//! * CUDA: the worker is a thin adapter over a resident `ds4-resident` (`tools/serve-up.sh`).
//! * MLX: the worker attaches to a resident `bench-worker` that holds the ~113 GB checkpoint
//!   (`tools/resident-up.sh`), because an in-process load per phase is unaffordable.
//!
//! Each engine repo's measure script used to wrap the WHOLE benchd invocation in its own resident
//! wrapper, which boots ONE resident for the window. That shape cannot serve two legs: on M5 #4
//! (ranked run 34230122059) the reference leg attached to the CANDIDATE tree's resident and the
//! worker refused — `resident holds <candidate>/weights but this phase asked for
//! <baseline-workspace>/weights`. So benchd boots and tears down EACH leg's resident itself, from
//! that leg's own tree, one at a time.
//!
//! ## The convention benchd drives, per leg
//!
//! Both commands run with the working directory set to the LEG's own workspace. The argv contract
//! is IDENTICAL on both platforms; only the script name differs
//! ([`leg_serve_script`]):
//!
//! ```text
//! <workspace>/<script> --boot --spec <serial|mtp> --draft-len <N> --socket-out <FILE>
//! <workspace>/<script> --stop --socket <PATH>
//! ```
//!
//! * `--boot` boots exactly ONE resident from that tree, waits until it answers a healthy hello,
//!   writes the resident's Unix-socket path as the FIRST LINE of `<FILE>`, and exits 0 leaving the
//!   resident running. It owns its own health timeout; benchd waits for it.
//! * `--spec serial --draft-len 0` is what LEG 1 always gets: the control leg is serial regardless
//!   of what the submission declares. Leg 2 gets the run's declared spec.
//! * `--stop` tears that resident down and is IDEMPOTENT (a resident that is already gone is not
//!   an error). benchd runs it when the leg's [`LegServe`] is dropped — after the leg, on success
//!   and on failure alike — so a resident never outlives its leg and the two legs never hold GPU
//!   memory at the same time.
//!
//! benchd then puts the socket path into that leg's WORKER SPAWNS ONLY, under the names that
//! platform's worker connects by ([`socket_env_names`]): `BENCH_WORKER_RESIDENT_SOCKET` on both,
//! and `DS4_RESIDENT_SOCKET` additionally on CUDA. benchd's own environment is never mutated, so
//! the two legs cannot bleed into each other.
//!
//! ## The inherited socket
//!
//! On the PAIRED path an inherited socket is REFUSED by name on both platforms: one resident for
//! both legs prices the candidate against itself, and on MLX it does not even reach a number — the
//! reference leg's worker refuses the weights mismatch. The refusal is scoped to the paired path,
//! so the SINGLE-resident shapes keep working untouched: a local unscored run has one tree and one
//! leg, and an inherited resident is exactly right for it.

use bench_core::constants::Platform;
use bench_protocol::SpecConfig;
use std::path::{Path, PathBuf};

/// The CUDA track's per-leg boot/teardown script, relative to a leg's workspace root.
pub const CUDA_LEG_SERVE_SCRIPT: &str = "tools/serve-up.sh";
/// The MLX track's per-leg boot/teardown script, relative to a leg's workspace root.
pub const MLX_LEG_SERVE_SCRIPT: &str = "tools/resident-up.sh";
/// The socket name the ds4 adapter connects by.
pub const DS4_RESIDENT_SOCKET_ENV: &str = "DS4_RESIDENT_SOCKET";
/// The socket name the generic bench worker connects by.
pub const BENCH_WORKER_RESIDENT_SOCKET_ENV: &str = "BENCH_WORKER_RESIDENT_SOCKET";

/// EXACT-MATCH refusal names.
pub const LEG_SERVE_SCRIPT_MISSING: &str = "LEG-SERVE-SCRIPT-MISSING";
/// The boot command failed, or produced no socket.
pub const LEG_SERVE_BOOT_FAILED: &str = "LEG-SERVE-BOOT-FAILED";
/// A resident socket was inherited from the environment on a path that boots its own, per leg.
pub const LEG_SERVE_INHERITED_SOCKET: &str = "LEG-SERVE-INHERITED-SOCKET";

/// The per-leg boot/teardown script THIS platform's trees carry. The argv contract behind the two
/// names is the same; the name differs because the two engine families already had their own
/// resident wrapper and each keeps it.
pub fn leg_serve_script(platform: Platform) -> &'static str {
    match platform {
        Platform::Mlx => MLX_LEG_SERVE_SCRIPT,
        Platform::Cuda => CUDA_LEG_SERVE_SCRIPT,
    }
}

/// The child-environment names a leg's resident socket is injected under.
///
/// `BENCH_WORKER_RESIDENT_SOCKET` on BOTH platforms — it is the generic bench-worker's name, and
/// the MLX worker attaches by it. `DS4_RESIDENT_SOCKET` is added on CUDA, where the ds4 adapter
/// reads that name instead. A platform is never given a name its worker does not read.
pub fn socket_env_names(platform: Platform) -> &'static [&'static str] {
    match platform {
        Platform::Mlx => &[BENCH_WORKER_RESIDENT_SOCKET_ENV],
        Platform::Cuda => &[DS4_RESIDENT_SOCKET_ENV, BENCH_WORKER_RESIDENT_SOCKET_ENV],
    }
}

/// One leg's resident engine: booted from that leg's tree, torn down when this value is dropped.
#[derive(Debug)]
pub struct LegServe {
    workspace: PathBuf,
    socket: String,
    label: &'static str,
    platform: Platform,
}

impl LegServe {
    /// The child-environment entries this leg's worker spawns must carry — the socket, under the
    /// names THIS platform's worker connects by ([`socket_env_names`]).
    pub fn spawn_env(&self) -> Vec<(String, String)> {
        socket_env_names(self.platform)
            .iter()
            .map(|name| ((*name).to_string(), self.socket.clone()))
            .collect()
    }
}

impl Drop for LegServe {
    fn drop(&mut self) {
        let script = self.workspace.join(leg_serve_script(self.platform));
        let outcome = std::process::Command::new(&script)
            .current_dir(&self.workspace)
            .args(["--stop", "--socket", &self.socket])
            .output();
        match outcome {
            Ok(out) if out.status.success() => {}
            Ok(out) => eprintln!(
                "benchd: the {} leg's resident did not stop cleanly ({}): {}",
                self.label,
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ),
            // Teardown runs on the failure path too, so it REPORTS rather than panics: a leg that
            // already failed must not be replaced in the log by its own cleanup.
            Err(e) => eprintln!(
                "benchd: the {} leg's resident teardown could not run ({}): {e}",
                self.label,
                script.display()
            ),
        }
    }
}

/// The `--spec` / `--draft-len` pair for a leg, from the spec that leg requests. The CONTROL leg
/// passes `None` and is booted serial; a `serial` spec is serial too.
pub fn spec_arguments(spec: Option<&SpecConfig>) -> (String, String) {
    match spec.and_then(|s| s.mtp.as_ref()).and_then(|m| m.depth) {
        Some(depth) if depth >= 1 => ("mtp".to_string(), depth.to_string()),
        _ => ("serial".to_string(), "0".to_string()),
    }
}

/// BOOT one leg's resident from `workspace`, at the spec that leg runs. `label` names the leg in
/// every message ("serial-control" / "candidate").
///
/// Fails CLOSED and BY NAME: a tree with no boot script, a boot that exits non-zero, and a boot
/// that leaves no socket behind are three different refusals with three different names.
pub fn boot_leg(
    workspace: &Path,
    spec: Option<&SpecConfig>,
    label: &'static str,
    platform: Platform,
) -> Result<LegServe, String> {
    let script_name = leg_serve_script(platform);
    let script = workspace.join(script_name);
    if !script.is_file() {
        return Err(format!(
            "{LEG_SERVE_SCRIPT_MISSING}: the {label} leg's tree {} holds no {script_name}, and a \
             paired run boots ONE resident PER LEG from that leg's own tree",
            workspace.display()
        ));
    }
    let (spec_mode, draft_len) = spec_arguments(spec);
    // The socket-out file is per PROCESS, per LEG and per BOOT: two legs of one run, and two runs
    // sharing a box, must never read each other's answer.
    let socket_out = std::env::temp_dir().join(format!(
        "benchd-leg-socket.{}.{label}.{}.txt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    let _ = std::fs::remove_file(&socket_out);
    let out = std::process::Command::new(&script)
        .current_dir(workspace)
        .args([
            "--boot",
            "--spec",
            &spec_mode,
            "--draft-len",
            &draft_len,
            "--socket-out",
        ])
        .arg(&socket_out)
        .output()
        .map_err(|e| {
            format!(
                "{LEG_SERVE_BOOT_FAILED}: the {label} leg's boot command {} could not run: {e}",
                script.display()
            )
        })?;
    if !out.status.success() {
        // The boot script's stderr is the participant tree's own text, and a later pair's boot
        // runs after that tree's engine has seen the hidden prompt: the full (scrubbed) text goes
        // to the box-local log, the error — which a sealed record carries — gets only its digest.
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stderr = stderr.trim();
        eprintln!(
            "benchd: the {label} leg's resident boot stderr: {}",
            bench_runner::scrub_reason_for_seal(stderr)
        );
        return Err(format!(
            "{LEG_SERVE_BOOT_FAILED}: the {label} leg's resident did not boot ({}): {}",
            out.status,
            bench_runner::seal_engine_text(stderr)
        ));
    }
    let socket = std::fs::read_to_string(&socket_out)
        .map_err(|e| {
            format!(
                "{LEG_SERVE_BOOT_FAILED}: the {label} leg's boot reported success but wrote no \
                 socket to {}: {e}",
                socket_out.display()
            )
        })?
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    let _ = std::fs::remove_file(&socket_out);
    if socket.is_empty() {
        return Err(format!(
            "{LEG_SERVE_BOOT_FAILED}: the {label} leg's boot wrote an empty socket path"
        ));
    }
    eprintln!(
        "benchd: the {label} leg's resident is up from {} at {socket} ({spec_mode}, draft-len \
         {draft_len})",
        workspace.display()
    );
    Ok(LegServe {
        workspace: workspace.to_path_buf(),
        socket,
        label,
        platform,
    })
}

/// REFUSE an inherited resident socket on a path that boots one PER LEG. One socket in benchd's
/// own environment means one resident for both legs: on CUDA that prices the candidate against
/// itself, and on MLX it does not even reach a number — the reference leg's worker refuses,
/// because the resident holds the candidate tree's weights and the leg asked for the reference
/// tree's (ranked run 34230122059 on M5 #4).
///
/// Only the names THIS platform's worker CONNECTS BY are checked. A name that platform's worker
/// never reads cannot misroute its leg, and refusing on it would stop a box that merely has an
/// unrelated variable set.
pub fn refuse_inherited_socket(
    platform: Platform,
    ds4: Option<&str>,
    bench_worker: Option<&str>,
) -> Result<(), String> {
    let script = leg_serve_script(platform);
    for (name, value) in [
        (DS4_RESIDENT_SOCKET_ENV, ds4),
        (BENCH_WORKER_RESIDENT_SOCKET_ENV, bench_worker),
    ] {
        if !socket_env_names(platform).contains(&name) {
            continue;
        }
        if value.map(str::trim).is_some_and(|v| !v.is_empty()) {
            return Err(format!(
                "{LEG_SERVE_INHERITED_SOCKET}: {name} is set in benchd's environment, and a paired \
                 run boots ONE resident PER LEG from that leg's own tree; do not wrap benchd in \
                 {script} — benchd runs it itself, twice"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A workspace holding a STUB boot/stop script for `platform`. The stub records the argv of
    /// every call in `calls.txt` and answers `--boot` with `socket` on the first line of the
    /// `--socket-out` file — the whole contract benchd depends on, and nothing else.
    fn stub_workspace(platform: Platform, socket: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "benchd-legserve.{}.{}.{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
            platform
        ));
        let script = dir.join(leg_serve_script(platform));
        std::fs::create_dir_all(script.parent().expect("the script sits under tools/")).unwrap();
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$(dirname \"$0\")/../calls.txt\"\n\
                 while [ $# -gt 0 ]; do\n\
                 \x20 case \"$1\" in --socket-out) printf '%s\\n' '{socket}' > \"$2\"; shift 2;; \
                 *) shift;; esac\n\
                 done\nexit 0\n"
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        dir
    }

    fn calls(dir: &Path) -> Vec<String> {
        std::fs::read_to_string(dir.join("calls.txt"))
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    /// THE CONVENTION, END TO END, ON BOTH PLATFORMS. benchd boots a leg's resident from that
    /// leg's own tree with the exact argv the engine repos implement, reads the socket off the
    /// first line of the `--socket-out` file, injects it under the names that platform's worker
    /// connects by, and STOPS it when the guard drops.
    ///
    /// The MLX case is the one this exists for: its measure script wrapped the whole benchd
    /// invocation in ONE resident from the candidate tree, so the reference leg attached to the
    /// candidate's weights and the worker refused (ranked run 34230122059).
    #[test]
    fn a_leg_boots_and_stops_through_its_platforms_script() {
        for (platform, script, want_env) in [
            (
                Platform::Mlx,
                MLX_LEG_SERVE_SCRIPT,
                vec![BENCH_WORKER_RESIDENT_SOCKET_ENV],
            ),
            (
                Platform::Cuda,
                CUDA_LEG_SERVE_SCRIPT,
                vec![DS4_RESIDENT_SOCKET_ENV, BENCH_WORKER_RESIDENT_SOCKET_ENV],
            ),
        ] {
            let socket = format!(
                "/tmp/leg-{}.sock",
                if platform == Platform::Mlx {
                    "mlx"
                } else {
                    "cuda"
                }
            );
            let dir = stub_workspace(platform, &socket);
            assert_eq!(leg_serve_script(platform), script);

            // LEG 1: serial, whatever the run declares.
            let leg = boot_leg(&dir, Some(&SpecConfig::mtp(3)), "serial-control", platform);
            let leg =
                leg.unwrap_or_else(|e| panic!("{platform:?}: the control leg must boot: {e}"));
            let boot = calls(&dir);
            assert_eq!(boot.len(), 1, "{platform:?}: one boot call");
            assert!(
                boot[0].starts_with("--boot --spec mtp --draft-len 3 --socket-out "),
                "{platform:?}: {}",
                boot[0]
            );

            // The socket reaches the leg's spawns under exactly the platform's own names.
            let env = leg.spawn_env();
            assert_eq!(
                env.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
                want_env,
                "{platform:?}: the socket must ride under the names this worker reads"
            );
            assert!(
                env.iter().all(|(_, v)| v == &socket),
                "{platform:?}: every name carries the socket the boot reported"
            );

            // …and the guard STOPS it, with the socket it booted.
            drop(leg);
            let after = calls(&dir);
            assert_eq!(after.len(), 2, "{platform:?}: boot then stop");
            assert_eq!(
                after[1],
                format!("--stop --socket {socket}"),
                "{platform:?}"
            );

            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// The CONTROL leg is booted SERIAL and the CANDIDATE leg at its declared depth — the one
    /// asymmetry between the two boots.
    #[test]
    fn the_control_leg_boots_serial_and_the_candidate_boots_its_declared_depth() {
        assert_eq!(
            spec_arguments(None),
            ("serial".to_string(), "0".to_string())
        );
        assert_eq!(
            spec_arguments(Some(&SpecConfig::serial())),
            ("serial".to_string(), "0".to_string())
        );
        assert_eq!(
            spec_arguments(Some(&SpecConfig::mtp(1))),
            ("mtp".to_string(), "1".to_string())
        );
        assert_eq!(
            spec_arguments(Some(&SpecConfig::mtp(4))),
            ("mtp".to_string(), "4".to_string())
        );

        // …and the control leg really does put `serial` on the boot command line, whatever the
        // candidate declares.
        let dir = stub_workspace(Platform::Mlx, "/tmp/leg-serial.sock");
        let leg = boot_leg(&dir, None, "serial-control", Platform::Mlx).unwrap();
        assert!(calls(&dir)[0].starts_with("--boot --spec serial --draft-len 0 --socket-out "));
        drop(leg);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A tree with no boot script refuses BY NAME, naming THAT platform's script.
    #[test]
    fn a_tree_without_the_boot_script_refuses_by_name() {
        for platform in Platform::ALL {
            let dir = std::env::temp_dir().join(format!(
                "benchd-legserve-bare.{}.{:?}",
                std::process::id(),
                platform
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let err = boot_leg(&dir, None, "serial-control", platform).unwrap_err();
            assert!(
                err.contains(LEG_SERVE_SCRIPT_MISSING),
                "{platform:?}: {err}"
            );
            assert!(
                err.contains(leg_serve_script(platform)),
                "{platform:?}: the refusal must name the script it looked for: {err}"
            );
            assert!(err.contains("serial-control"), "{platform:?}: {err}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// AN INHERITED SOCKET IS REFUSED ON BOTH PLATFORMS, on the name that platform's worker
    /// actually connects by — and only on that name, so an unrelated variable does not stop a box.
    #[test]
    fn an_inherited_resident_socket_refuses_by_name() {
        for platform in Platform::ALL {
            assert!(refuse_inherited_socket(platform, None, None).is_ok());
            assert!(refuse_inherited_socket(platform, Some("  "), Some("")).is_ok());

            // BENCH_WORKER_RESIDENT_SOCKET is read by BOTH workers, so both refuse on it. This is
            // the MLX case: its measure script exports exactly this name.
            let err = refuse_inherited_socket(platform, None, Some("/tmp/bw.sock")).unwrap_err();
            assert!(
                err.contains(LEG_SERVE_INHERITED_SOCKET),
                "{platform:?}: {err}"
            );
            assert!(
                err.contains(BENCH_WORKER_RESIDENT_SOCKET_ENV),
                "{platform:?}: {err}"
            );
            assert!(
                err.contains(leg_serve_script(platform)),
                "{platform:?}: the refusal must name the wrapper not to wrap benchd in: {err}"
            );
        }

        // DS4_RESIDENT_SOCKET is the ds4 adapter's name: CUDA refuses on it…
        let err = refuse_inherited_socket(Platform::Cuda, Some("/tmp/ds4.sock"), None).unwrap_err();
        assert!(err.contains(LEG_SERVE_INHERITED_SOCKET), "{err}");
        assert!(err.contains(DS4_RESIDENT_SOCKET_ENV), "{err}");
        // …and MLX does NOT, because no MLX worker reads it, so it cannot misroute an MLX leg.
        assert!(refuse_inherited_socket(Platform::Mlx, Some("/tmp/ds4.sock"), None).is_ok());
    }

    /// A boot that exits non-zero, and one that reports success with no socket, are two different
    /// refusals — and neither leaves a guard behind that would try to stop nothing.
    #[test]
    fn a_boot_that_fails_or_reports_no_socket_refuses_by_name() {
        for (name, body) in [
            (
                "nonzero",
                "#!/bin/sh\necho 'resident refused: no GPU lock' >&2\nexit 2\n",
            ),
            ("nosocket", "#!/bin/sh\nexit 0\n"),
        ] {
            let dir =
                std::env::temp_dir().join(format!("benchd-legserve-{name}.{}", std::process::id()));
            let script = dir.join(leg_serve_script(Platform::Mlx));
            std::fs::create_dir_all(script.parent().unwrap()).unwrap();
            std::fs::write(&script, body).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            let err = boot_leg(&dir, None, "serial-control", Platform::Mlx).unwrap_err();
            assert!(err.contains(LEG_SERVE_BOOT_FAILED), "{name}: {err}");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// The spawn environment names the socket under every name the platform's worker reads, and
    /// dropping the guard runs the teardown — which REPORTS rather than panics when the tree is
    /// gone, because teardown runs on the failure path too.
    #[test]
    fn the_leg_environment_names_the_platforms_socket_variables() {
        let leg = LegServe {
            workspace: PathBuf::from("/ref/tree"),
            socket: "/tmp/ds4-resident.run.sock".to_string(),
            label: "candidate",
            platform: Platform::Cuda,
        };
        assert_eq!(
            leg.spawn_env(),
            vec![
                (
                    DS4_RESIDENT_SOCKET_ENV.to_string(),
                    "/tmp/ds4-resident.run.sock".to_string()
                ),
                (
                    BENCH_WORKER_RESIDENT_SOCKET_ENV.to_string(),
                    "/tmp/ds4-resident.run.sock".to_string()
                ),
            ]
        );
        drop(leg);

        let leg = LegServe {
            workspace: PathBuf::from("/ref/tree"),
            socket: "/tmp/bench-worker.run.sock".to_string(),
            label: "serial-control",
            platform: Platform::Mlx,
        };
        assert_eq!(
            leg.spawn_env(),
            vec![(
                BENCH_WORKER_RESIDENT_SOCKET_ENV.to_string(),
                "/tmp/bench-worker.run.sock".to_string()
            )]
        );
        drop(leg);
    }
}
