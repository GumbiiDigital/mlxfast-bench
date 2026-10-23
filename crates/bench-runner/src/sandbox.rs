//! B-2 — official-run Seatbelt sandbox + fail-closed spawn policy.
//!
//! Port of the Swift official-run sandbox from
//! `mlxfast-challenge-dev/Sources/MLXFastCLI/main.swift`
//! (`runtimeWorkerOptions` :1143-1219, `writeRuntimeWorkerSandboxProfile` :1221-1258,
//! `seatbeltEscaped` :1270-1274) and `benchmark.sh enforce_official_sandbox` (:666-719).
//!
//! An official benchmark run executes the (untrusted, submitted) engine under
//! `/usr/bin/sandbox-exec -f <profile> <engine> runtime-worker …` with a Seatbelt profile
//! that denies network, fork, exec (except the engine itself), all file writes (except
//! `/dev/null`), reads of the private golden (+ the private dir), and reads or writes of the
//! resolver-supplied evaluator-only paths. The run FAILS CLOSED
//! — refuses to run at all rather than falling back to an unsandboxed/worker-less path — on
//! ANY of: the worker disabled, `MLXFAST_NO_SANDBOX=1`, no engine executable, no derivable
//! sandbox profile, or `/usr/bin/sandbox-exec` missing.
//!
//! ## The one resident-socket exception
//!
//! macOS Seatbelt classifies an AF_UNIX `connect(2)` as a NETWORK operation, so the blanket
//! `(deny network*)` above also denies the sandboxed worker's connect to the resident
//! `bench-worker` Unix domain socket — every phase worker fails with `Operation not
//! permitted`, whatever directory the socket lives in. That made a ranked RESIDENT run
//! impossible on macOS: the worker could not attach, so the checkpoint would have to be
//! re-loaded per phase, breaking the binding "weights load once" rule.
//!
//! When (and only when) `BENCH_WORKER_RESIDENT_SOCKET` names the resident socket, the derived
//! profile therefore gains EXACTLY ONE more rule, placed straight after `(deny network*)` so
//! the later, more specific rule wins:
//!
//! ```text
//! (allow network-outbound (remote unix-socket (path-literal "<socket>")))
//! ```
//!
//! That is the whole widening. It is OUTBOUND only (the worker may connect, never listen), it
//! names ONE path-literal (a connect to any other socket still fails), and it opens no TCP/UDP
//! and no filesystem path. The resident itself is booted OUTSIDE the sandbox by the window
//! tooling before benchd runs, so the sandbox never has to permit loading weights or binding a
//! socket — only the connect to that one already-existing name. The socket path is box
//! configuration (constant across a window, never submission-controlled), and a path that is
//! not absolute or that the Seatbelt escaper cannot represent is REFUSED, never silently
//! skipped — a skipped rule would resurrect the `Operation not permitted` failure at spawn.
//! With the variable unset the profile is byte-identical to the pre-resident one.
//!
//! The predicate spelling was settled BEHAVIOURALLY against `/usr/bin/sandbox-exec` on macOS
//! 26, not from documentation. Several filter forms both compile and work — `(literal …)`,
//! `(remote unix-socket (literal …))`, `(remote unix-socket (path-literal …))` and the
//! `subpath` variants all permit the connect. We take `(remote unix-socket (path-literal …))`
//! as the narrowest and most explicit of them: it says outbound, unix-socket, this one exact
//! path, and it cannot be misread as a subtree or as a TCP filter the way a bare `(literal …)`
//! or a `subpath` form could. Note that a profile which merely COMPILES proves nothing about
//! whether the connect is permitted, which is why `tests/sandbox_unix_socket.rs` exercises a
//! real listener under a real `sandbox-exec` instead of asserting on the profile text.
//!
//! This module is macOS-buildable and unit-tested against a STUB (the pure profile builder
//! and the fail-closed resolver take injected inputs, so no real `sandbox-exec` or GPU is
//! needed). The real sandboxed timed run is exercised on the GPU box in B-3.

use std::path::{Path, PathBuf};

/// The Seatbelt interpreter the official run wraps the engine with (Swift
/// `sandboxExecutable`, main.swift:1223).
pub const SANDBOX_EXEC_PATH: &str = "/usr/bin/sandbox-exec";

/// Escape a path for embedding inside a Seatbelt `(literal "...")` / `(subpath "...")`
/// string. Byte-for-byte port of Swift `seatbeltEscaped` (main.swift:1270-1274): backslash
/// first, then double-quote. Order matters — escaping the quote first would double-escape
/// the backslash it introduces.
pub fn seatbelt_escaped(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

/// The single Seatbelt rule that lets the sandboxed worker connect to the resident
/// `bench-worker` socket, verified BEHAVIOURALLY against `/usr/bin/sandbox-exec` (macOS 26).
/// `(remote unix-socket (path-literal …))` is the narrowest spelling that works: outbound
/// only, unix-socket only, one exact path. Broader spellings (`(literal …)`, `subpath`) also
/// permit the connect, so they are not wrong — they are simply less precise about what is
/// being opened, and a `subpath` form would widen the allowance to a whole directory.
/// `tests/sandbox_unix_socket.rs` proves the rule's EFFECT: allowed path connects, no rule
/// gives EPERM, any other path still gives EPERM.
fn resident_socket_rule(socket_path: &str) -> String {
    format!(
        "(allow network-outbound (remote unix-socket (path-literal \"{}\")))",
        seatbelt_escaped(socket_path)
    )
}

/// Reject a resident-socket path the profile could not faithfully name. The rule is a
/// security boundary, so an unrepresentable path must REFUSE the run rather than be dropped:
/// dropping it would emit a valid-looking profile whose worker then dies on `Operation not
/// permitted`, and a relative path would name the wrong socket (Seatbelt literals are
/// matched against the kernel-resolved absolute path).
///
/// Refused: a relative path, and any path carrying an ASCII control character (`\n`
/// included). [`seatbelt_escaped`] can only escape `\\` and `"`; a control character would
/// either terminate the profile line early or be re-interpreted by the profile reader.
pub fn validate_resident_socket(socket_path: &str) -> Result<(), OfficialSandboxError> {
    if !Path::new(socket_path).is_absolute() {
        return Err(OfficialSandboxError::InvalidResidentSocket);
    }
    if socket_path.chars().any(|c| c.is_control()) {
        return Err(OfficialSandboxError::InvalidResidentSocket);
    }
    Ok(())
}

/// Build the Seatbelt profile source that guards the official runtime worker. This follows
/// Swift `writeRuntimeWorkerSandboxProfile` (main.swift:1237-1256), then adds rules that
/// block process-persistent IPC and preference writes. The golden deny-read rule and optional
/// private-dir subpath deny-read rule follow. Evaluator read+write denies are last so they win
/// over every allow. Lines are joined by `\n` with NO trailing newline.
///
/// `resident_socket`, when present and non-empty, adds the single resident-socket
/// `network-outbound` allowance documented at the top of this module, immediately after
/// `(deny network*)`. `None` (or an empty string) reproduces the pre-resident profile
/// byte-for-byte. Validate the path with [`validate_resident_socket`] BEFORE calling.
///
/// Callers pass ABSOLUTE, symlink-resolved paths ([`absolute_path`]) so the embedded
/// literals match what the kernel resolves the worker's accesses to.
pub fn build_seatbelt_profile(
    engine_path: &str,
    golden_path: &str,
    private_dir: Option<&str>,
    resident_socket: Option<&str>,
    evaluator_paths: &[String],
) -> String {
    let mut denied_read_rules = vec![format!(
        "(deny file-read* (literal \"{}\"))",
        seatbelt_escaped(golden_path)
    )];
    if let Some(dir) = private_dir {
        if !dir.is_empty() {
            denied_read_rules.push(format!(
                "(deny file-read* (subpath \"{}\"))",
                seatbelt_escaped(dir)
            ));
        }
    }
    // The fixed preamble (main.swift:1247-1254), then the deny-read rules. No trailing
    // newline: the Swift `"""` literal closes right after `deniedReadRules.joined(...)`.
    let mut lines = vec![
        "(version 1)".to_string(),
        "(allow default)".to_string(),
        "(deny network*)".to_string(),
    ];
    // The ONE widening (see the module docs): a resident run needs the worker to connect to
    // the already-booted bench-worker over its Unix socket, which Seatbelt classifies as
    // network. Emitted directly after `(deny network*)` so the later, more specific rule wins.
    // Absent `resident_socket`, this vector is byte-identical to the pre-resident profile.
    if let Some(socket) = resident_socket {
        if !socket.is_empty() {
            lines.push(resident_socket_rule(socket));
        }
    }
    lines.extend([
        "(deny process-fork)".to_string(),
        "(deny process-exec*)".to_string(),
        format!(
            "(allow process-exec (literal \"{}\"))",
            seatbelt_escaped(engine_path)
        ),
        // The worker shares an OS account with trusted jobs, so deny signals to siblings,
        // preserve self-signals, and deny scheduling changes.
        "(deny signal)".to_string(),
        "(allow signal (target self))".to_string(),
        "(deny system-sched)".to_string(),
        "(deny ipc-posix-shm*)".to_string(),
        "(deny ipc-sysv*)".to_string(),
        "(deny ipc-posix-sem*)".to_string(),
        "(deny user-preference-write)".to_string(),
        "(deny mach-lookup)".to_string(),
        "(allow mach-lookup (global-name \"com.apple.MTLCompilerService\") (global-name \"com.apple.system.opendirectoryd.libinfo\") (global-name \"com.apple.system.opendirectoryd.membership\") (global-name \"com.apple.logd\") (global-name \"com.apple.diagnosticd\"))".to_string(),
        "(deny file-write*)".to_string(),
        "(allow file-write* (literal \"/dev/null\"))".to_string(),
    ]);
    lines.extend(denied_read_rules);
    // Last, so these evaluator-only denies win over every allow above. `subpath` also
    // covers file paths, matching the resident profile's one rule shape for both.
    lines.extend(evaluator_paths.iter().map(|path| {
        format!(
            "(deny file-read* file-write* (subpath \"{}\"))",
            seatbelt_escaped(path)
        )
    }));
    lines.join("\n")
}

/// Resolve a path to absolute + symlink-resolved form, matching Swift `absolutePath`
/// (main.swift:1260-1268: relative-to-cwd, `standardizedFileURL.resolvingSymlinksInPath`).
/// For a missing target, resolve the deepest existing ancestor, then append the missing suffix.
/// Every error other than `NotFound` is returned so an official sandbox can fail closed.
pub fn absolute_path(path: &str) -> std::io::Result<String> {
    let p = Path::new(path);
    let absolute: PathBuf = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir()?.join(p)
    };
    let mut resolved = PathBuf::new();
    let mut missing = false;
    for component in absolute.components() {
        use std::path::Component::*;
        match component {
            CurDir => {}
            ParentDir => {
                resolved.pop();
            }
            Normal(name) => {
                resolved.push(name);
                if !missing {
                    match std::fs::canonicalize(&resolved) {
                        Ok(path) => resolved = path,
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                            missing = true;
                        }
                        Err(error) => return Err(error),
                    }
                }
            }
            RootDir | Prefix(_) => resolved.push(component.as_os_str()),
        }
    }
    Ok(resolved.to_string_lossy().to_string())
}

/// The fail-closed rejection reasons for an official run, each carrying the EXACT Swift
/// error string (main.swift `runtimeWorkerOptions` / `writeRuntimeWorkerSandboxProfile`).
/// These must byte-match so an operator sees the same refusal on either side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OfficialSandboxError {
    /// `MLXFAST_USE_RUNTIME_WORKER` is `0`/`false` (main.swift:1157-1161).
    WorkerDisabled,
    /// `MLXFAST_NO_SANDBOX=1` (main.swift:1166-1170).
    SandboxDisabled,
    /// No engine executable configured or derivable (main.swift:1178-1183).
    NoExecutable,
    /// `/usr/bin/sandbox-exec` is not an executable file (main.swift:1224-1226).
    SandboxExecNotFound,
    /// No sandbox profile configured or derivable (main.swift:1206-1210).
    NoProfile,
    /// `BENCH_WORKER_RESIDENT_SOCKET` names a path the Seatbelt profile cannot faithfully
    /// express (relative, or carrying a control character). Adapter-only: the Swift original
    /// has no resident topology, so this refusal has no Swift twin to byte-match.
    InvalidResidentSocket,
    /// A sandbox path could not be resolved safely. Adapter-only: fail closed rather than emit a
    /// profile whose path rule may name a different object.
    PathResolution,
}

impl OfficialSandboxError {
    /// The verbatim Swift message for this rejection.
    pub fn message(&self) -> &'static str {
        match self {
            OfficialSandboxError::WorkerDisabled => {
                "official benchmark runs require the runtime worker; unset MLXFAST_USE_RUNTIME_WORKER"
            }
            OfficialSandboxError::SandboxDisabled => {
                "official benchmark runs require the runtime worker sandbox; unset MLXFAST_NO_SANDBOX"
            }
            OfficialSandboxError::NoExecutable => {
                "official benchmark runs require a runtime worker executable; none was configured or derivable"
            }
            OfficialSandboxError::SandboxExecNotFound => {
                "sandbox-exec not found for runtime worker sandbox"
            }
            OfficialSandboxError::NoProfile => {
                "official benchmark runs require a runtime worker sandbox profile; none was configured or derivable"
            }
            OfficialSandboxError::InvalidResidentSocket => {
                "BENCH_WORKER_RESIDENT_SOCKET must be an absolute path with no control characters"
            }
            OfficialSandboxError::PathResolution => {
                "official benchmark sandbox path could not be resolved safely"
            }
        }
    }
}

impl std::fmt::Display for OfficialSandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for OfficialSandboxError {}

/// The env/context inputs the official-sandbox resolver reads (the `MLXFAST_*` knobs Swift
/// `runtimeWorkerOptions` consults). `None` means the variable is unset/empty. Passing them
/// explicitly (rather than reading the process env inside the resolver) makes the
/// fail-closed matrix a pure, exhaustively-testable function.
#[derive(Debug, Clone, Default)]
pub struct OfficialSandboxInputs<'a> {
    /// `MLXFAST_USE_RUNTIME_WORKER` (fallback `"1"`).
    pub use_runtime_worker: Option<&'a str>,
    /// `MLXFAST_NO_SANDBOX` (fallback `"0"`).
    pub no_sandbox: Option<&'a str>,
    /// `MLXFAST_RUNTIME_WORKER_EXECUTABLE` (fallback: `fallback_executable`).
    pub executable_override: Option<&'a str>,
    /// `MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE` — a pre-built profile path.
    pub profile_override: Option<&'a str>,
    /// `MLXFAST_PRIVATE_DIR` — an extra subpath denied to the worker.
    pub private_dir: Option<&'a str>,
    /// `BENCH_WORKER_RESIDENT_SOCKET` — the resident `bench-worker` socket the sandboxed
    /// worker must be allowed to connect to. `None`/empty leaves the profile unwidened.
    pub resident_socket: Option<&'a str>,
    /// The engine path benchd was told to run (Swift's `CommandLine.arguments.first`
    /// fallback for the worker executable).
    pub fallback_executable: &'a str,
    /// The private golden the worker must not read (`blockedGoldenPath`).
    pub golden_path: &'a str,
    /// Absolute, symlink-resolved evaluator-only paths denied for both reads and writes.
    pub evaluator_paths: &'a [String],
    /// Whether `/usr/bin/sandbox-exec` exists + is executable (Swift
    /// `FileManager.isExecutableFile`). Injected so the matrix is testable off-box.
    pub sandbox_exec_available: bool,
}

/// Where the resolved profile comes from: an operator-supplied path or a generated source
/// string the caller must write to a temp `.sb` file before spawning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxProfile {
    /// `MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE` pointed at an existing profile file.
    Override(String),
    /// A freshly-built profile ([`build_seatbelt_profile`]) to be written out.
    Generated(String),
}

/// The resolved official-sandbox plan: the executable to run, the profile to enforce, and
/// whether worker stderr is forwarded (official forces this OFF, main.swift:1217).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfficialSandboxPlan {
    /// Absolute path of the engine executable to `(allow process-exec (literal …))`.
    pub executable_path: String,
    /// The profile to pass to `sandbox-exec -f`.
    pub profile: SandboxProfile,
    /// Whether live worker stderr is echoed to this process (always false on official).
    pub forward_worker_stderr: bool,
}

/// Resolve the official-run sandbox, FAIL-CLOSED on any missing prerequisite. Port of Swift
/// `runtimeWorkerOptions` (main.swift:1143-1219) restricted to `officialRun == true` (the
/// only mode benchd calls this from): every "no worker / no sandbox" exit that Swift turns
/// into a throw on an official run is an `Err` here, never a silent unsandboxed fallback.
///
/// `forwards_worker_stderr` is the caller's requested value; the returned plan's
/// `forward_worker_stderr` is `forwards_worker_stderr && false` (official forces it off —
/// Swift `forwardsWorkerStderr && !officialRun`, main.swift:1217).
pub fn resolve_official_sandbox(
    inputs: &OfficialSandboxInputs<'_>,
    forwards_worker_stderr: bool,
) -> std::result::Result<OfficialSandboxPlan, OfficialSandboxError> {
    // 1. Worker must be enabled (main.swift:1155-1164).
    let enabled = inputs.use_runtime_worker.unwrap_or("1");
    if enabled == "0" || enabled.eq_ignore_ascii_case("false") {
        return Err(OfficialSandboxError::WorkerDisabled);
    }
    // 2. The sandbox must not be explicitly disabled (main.swift:1165-1170).
    if inputs.no_sandbox == Some("1") {
        return Err(OfficialSandboxError::SandboxDisabled);
    }
    // 3. An engine executable must be configured or derivable (main.swift:1171-1184).
    let executable = match inputs.executable_override {
        Some(e) if !e.is_empty() => e,
        _ => inputs.fallback_executable,
    };
    if executable.is_empty() {
        return Err(OfficialSandboxError::NoExecutable);
    }
    let executable_path =
        absolute_path(executable).map_err(|_| OfficialSandboxError::PathResolution)?;

    // 4. Resolve the profile (main.swift:1195-1210). An operator override wins; else, when
    //    the sandbox is not disabled and a golden path is present, generate one — which
    //    requires `/usr/bin/sandbox-exec` to exist (Swift writeRuntimeWorkerSandboxProfile).
    let profile = match inputs.profile_override {
        Some(p) if !p.is_empty() => SandboxProfile::Override(p.to_string()),
        _ => {
            if inputs.no_sandbox != Some("1") && !inputs.golden_path.is_empty() {
                if !inputs.sandbox_exec_available {
                    return Err(OfficialSandboxError::SandboxExecNotFound);
                }
                let golden = absolute_path(inputs.golden_path)
                    .map_err(|_| OfficialSandboxError::PathResolution)?;
                let private_dir = inputs
                    .private_dir
                    .filter(|d| !d.is_empty())
                    .map(absolute_path)
                    .transpose()
                    .map_err(|_| OfficialSandboxError::PathResolution)?;
                // Validate BEFORE absolutizing: `absolute_path` would happily join a
                // relative socket name onto the cwd and hide the misconfiguration behind a
                // rule naming a socket that does not exist.
                let resident_socket = match inputs.resident_socket.filter(|s| !s.is_empty()) {
                    Some(socket) => {
                        validate_resident_socket(socket)?;
                        Some(
                            absolute_path(socket)
                                .map_err(|_| OfficialSandboxError::PathResolution)?,
                        )
                    }
                    None => None,
                };
                SandboxProfile::Generated(build_seatbelt_profile(
                    &executable_path,
                    &golden,
                    private_dir.as_deref(),
                    resident_socket.as_deref(),
                    inputs.evaluator_paths,
                ))
            } else {
                // No override, and no way to generate one → fail closed below.
                return Err(OfficialSandboxError::NoProfile);
            }
        }
    };

    // Official forces worker-stderr forwarding OFF (Swift `forwardsWorkerStderr &&
    // !officialRun`, main.swift:1217; this resolver is the officialRun == true path, so the
    // requested `forwards_worker_stderr` is always suppressed). Named to keep the caller's
    // intent visible even though it can only resolve to false here.
    let _requested_forward = forwards_worker_stderr;
    Ok(OfficialSandboxPlan {
        executable_path,
        profile,
        forward_worker_stderr: false,
    })
}

/// Build the argv the official run spawns: `sandbox-exec -f <profile_path> <engine>
/// runtime-worker --weights <weights> [extra…]`. The engine argv is exactly the one the
/// unsandboxed [`crate::transport::ChildStdioTransport::build_args`] produces, wrapped by
/// the Seatbelt interpreter. Returned as `(program, args)`.
pub fn sandbox_exec_command(
    profile_path: &str,
    engine_path: &str,
    weights_path: &str,
    extra_args: &[String],
) -> (String, Vec<String>) {
    let mut args = vec![
        "-f".to_string(),
        profile_path.to_string(),
        engine_path.to_string(),
    ];
    args.extend(crate::transport::ChildStdioTransport::build_args(
        weights_path,
        extra_args,
    ));
    (SANDBOX_EXEC_PATH.to_string(), args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_inputs<'a>() -> OfficialSandboxInputs<'a> {
        OfficialSandboxInputs {
            use_runtime_worker: None,
            no_sandbox: None,
            executable_override: None,
            profile_override: None,
            private_dir: None,
            resident_socket: None,
            fallback_executable: "/opt/mlxfast/engine",
            golden_path: "/private/golden.json",
            evaluator_paths: &[],
            sandbox_exec_available: true,
        }
    }

    #[test]
    fn seatbelt_escape_backslash_then_quote() {
        // Backslash escaped first, then quote — matching Swift order.
        assert_eq!(seatbelt_escaped(r#"a\b"c"#), r#"a\\b\"c"#);
        assert_eq!(seatbelt_escaped("/plain/path"), "/plain/path");
    }

    #[test]
    fn profile_byte_shape_adds_persistence_denies_to_swift_original() {
        // Exact Swift rule order plus the M14 persistence block after the engine allowance.
        // The golden deny-read remains the last line with NO trailing newline.
        let profile =
            build_seatbelt_profile("/opt/engine", "/private/golden.json", None, None, &[]);
        let expected = "(version 1)\n\
             (allow default)\n\
             (deny network*)\n\
             (deny process-fork)\n\
             (deny process-exec*)\n\
             (allow process-exec (literal \"/opt/engine\"))\n\
             (deny signal)\n\
             (allow signal (target self))\n\
             (deny system-sched)\n\
             (deny ipc-posix-shm*)\n\
             (deny ipc-sysv*)\n\
             (deny ipc-posix-sem*)\n\
             (deny user-preference-write)\n\
             (deny mach-lookup)\n\
             (allow mach-lookup (global-name \"com.apple.MTLCompilerService\") (global-name \"com.apple.system.opendirectoryd.libinfo\") (global-name \"com.apple.system.opendirectoryd.membership\") (global-name \"com.apple.logd\") (global-name \"com.apple.diagnosticd\"))\n\
             (deny file-write*)\n\
             (allow file-write* (literal \"/dev/null\"))\n\
             (deny file-read* (literal \"/private/golden.json\"))";
        assert_eq!(profile, expected);
        assert!(
            !profile.ends_with('\n'),
            "no trailing newline (Swift literal)"
        );
    }

    #[test]
    fn profile_denies_signals_allows_self_and_denies_scheduling() {
        let profile = build_seatbelt_profile("/opt/engine", "/g.json", None, None, &[]);
        let lines: Vec<&str> = profile.lines().collect();
        let exec_allow = lines
            .iter()
            .position(|line| *line == "(allow process-exec (literal \"/opt/engine\"))")
            .expect("engine exec allowance present");

        assert_eq!(lines[exec_allow + 1], "(deny signal)");
        assert_eq!(lines[exec_allow + 2], "(allow signal (target self))");
        assert_eq!(lines[exec_allow + 3], "(deny system-sched)");
        assert!(profile.lines().any(|line| line == "(deny system-sched)"));
    }

    #[test]
    fn profile_appends_private_dir_subpath() {
        // With MLXFAST_PRIVATE_DIR set, a second deny-read subpath rule is appended.
        let profile = build_seatbelt_profile(
            "/opt/engine",
            "/private/golden.json",
            Some("/private/dir"),
            None,
            &[],
        );
        assert!(profile.ends_with(
            "(deny file-read* (literal \"/private/golden.json\"))\n\
             (deny file-read* (subpath \"/private/dir\"))"
        ));
        // An empty private dir does NOT add a rule.
        let profile2 =
            build_seatbelt_profile("/opt/engine", "/private/golden.json", Some(""), None, &[]);
        assert_eq!(
            profile2,
            build_seatbelt_profile("/opt/engine", "/private/golden.json", None, None, &[])
        );
    }

    #[test]
    fn profile_escapes_paths() {
        let profile =
            build_seatbelt_profile(r#"/opt/en"gine"#, r#"/priv/g"n.json"#, None, None, &[]);
        assert!(profile.contains(r#"(allow process-exec (literal "/opt/en\"gine"))"#));
        assert!(profile.contains(r#"(deny file-read* (literal "/priv/g\"n.json"))"#));
    }

    #[test]
    fn evaluator_read_write_denies_are_last() {
        let paths = vec![
            "/private/evaluator-dir".to_string(),
            r#"/private/evaluator-"file"#.to_string(),
        ];
        let profile = build_seatbelt_profile(
            "/opt/engine",
            "/private/golden.json",
            Some("/private/dir"),
            None,
            &paths,
        );
        assert!(profile.ends_with(
            "(deny file-read* file-write* (subpath \"/private/evaluator-dir\"))\n\
             (deny file-read* file-write* (subpath \"/private/evaluator-\\\"file\"))"
        ));
    }

    // ---- Fail-closed matrix (the 5 official refusals) ----

    #[test]
    fn official_happy_path_generates_profile_and_forces_stderr_off() {
        let plan = resolve_official_sandbox(&base_inputs(), true).unwrap();
        assert!(matches!(plan.profile, SandboxProfile::Generated(_)));
        // Even when the caller requests stderr forwarding, official forces it OFF.
        assert!(!plan.forward_worker_stderr);
        assert_eq!(
            plan.executable_path,
            absolute_path("/opt/mlxfast/engine").unwrap()
        );
    }

    #[test]
    fn fail_closed_worker_disabled() {
        for v in ["0", "false", "False", "FALSE"] {
            let mut inp = base_inputs();
            inp.use_runtime_worker = Some(v);
            assert_eq!(
                resolve_official_sandbox(&inp, false).unwrap_err(),
                OfficialSandboxError::WorkerDisabled
            );
        }
    }

    #[test]
    fn fail_closed_sandbox_disabled() {
        let mut inp = base_inputs();
        inp.no_sandbox = Some("1");
        assert_eq!(
            resolve_official_sandbox(&inp, false).unwrap_err(),
            OfficialSandboxError::SandboxDisabled
        );
    }

    #[test]
    fn fail_closed_no_executable() {
        let mut inp = base_inputs();
        inp.fallback_executable = "";
        inp.executable_override = None;
        assert_eq!(
            resolve_official_sandbox(&inp, false).unwrap_err(),
            OfficialSandboxError::NoExecutable
        );
    }

    #[test]
    fn fail_closed_no_sandbox_exec() {
        let mut inp = base_inputs();
        inp.sandbox_exec_available = false;
        assert_eq!(
            resolve_official_sandbox(&inp, false).unwrap_err(),
            OfficialSandboxError::SandboxExecNotFound
        );
    }

    #[test]
    fn fail_closed_no_derivable_profile() {
        // No override, and no golden path to build one from → NoProfile (a generated
        // profile is impossible). sandbox_exec availability is irrelevant here.
        let mut inp = base_inputs();
        inp.golden_path = "";
        inp.profile_override = None;
        assert_eq!(
            resolve_official_sandbox(&inp, false).unwrap_err(),
            OfficialSandboxError::NoProfile
        );
    }

    #[test]
    fn profile_override_bypasses_generation() {
        // An operator-supplied profile path is used verbatim (Override), and it does NOT
        // require sandbox-exec to be present (Swift only touches sandbox-exec when it must
        // WRITE a profile).
        let mut inp = base_inputs();
        inp.profile_override = Some("/etc/mlxfast/worker.sb");
        inp.sandbox_exec_available = false;
        let plan = resolve_official_sandbox(&inp, false).unwrap();
        assert_eq!(
            plan.profile,
            SandboxProfile::Override("/etc/mlxfast/worker.sb".to_string())
        );
    }

    #[test]
    fn all_error_messages_are_swift_verbatim() {
        assert_eq!(
            OfficialSandboxError::WorkerDisabled.message(),
            "official benchmark runs require the runtime worker; unset MLXFAST_USE_RUNTIME_WORKER"
        );
        assert_eq!(
            OfficialSandboxError::SandboxDisabled.message(),
            "official benchmark runs require the runtime worker sandbox; unset MLXFAST_NO_SANDBOX"
        );
        assert_eq!(
            OfficialSandboxError::NoExecutable.message(),
            "official benchmark runs require a runtime worker executable; none was configured or derivable"
        );
        assert_eq!(
            OfficialSandboxError::SandboxExecNotFound.message(),
            "sandbox-exec not found for runtime worker sandbox"
        );
        assert_eq!(
            OfficialSandboxError::NoProfile.message(),
            "official benchmark runs require a runtime worker sandbox profile; none was configured or derivable"
        );
        // Adapter-only refusal (no Swift twin): still a stable operator-facing string.
        assert_eq!(
            OfficialSandboxError::InvalidResidentSocket.message(),
            "BENCH_WORKER_RESIDENT_SOCKET must be an absolute path with no control characters"
        );
    }

    // ---- The resident-socket widening ----

    /// The load-bearing guarantee: with no resident socket the profile is BYTE-IDENTICAL to
    /// the pre-resident one. A non-resident official run must be unaffected by this feature.
    #[test]
    fn profile_without_resident_socket_is_byte_identical() {
        let pre_resident = "(version 1)\n\
             (allow default)\n\
             (deny network*)\n\
             (deny process-fork)\n\
             (deny process-exec*)\n\
             (allow process-exec (literal \"/opt/engine\"))\n\
             (deny signal)\n\
             (allow signal (target self))\n\
             (deny system-sched)\n\
             (deny ipc-posix-shm*)\n\
             (deny ipc-sysv*)\n\
             (deny ipc-posix-sem*)\n\
             (deny user-preference-write)\n\
             (deny mach-lookup)\n\
             (allow mach-lookup (global-name \"com.apple.MTLCompilerService\") (global-name \"com.apple.system.opendirectoryd.libinfo\") (global-name \"com.apple.system.opendirectoryd.membership\") (global-name \"com.apple.logd\") (global-name \"com.apple.diagnosticd\"))\n\
             (deny file-write*)\n\
             (allow file-write* (literal \"/dev/null\"))\n\
             (deny file-read* (literal \"/private/golden.json\"))";
        for socket in [None, Some("")] {
            assert_eq!(
                build_seatbelt_profile("/opt/engine", "/private/golden.json", None, socket, &[]),
                pre_resident,
                "an absent/empty resident socket must not perturb one byte"
            );
        }
        // ... and the same with a private dir in play.
        assert_eq!(
            build_seatbelt_profile("/opt/e", "/g.json", Some("/private/dir"), None, &[]),
            build_seatbelt_profile("/opt/e", "/g.json", Some("/private/dir"), Some(""), &[],)
        );
    }

    #[test]
    fn profile_always_denies_process_persistent_state() {
        let rules = [
            "(deny ipc-posix-shm*)",
            "(deny ipc-sysv*)",
            "(deny ipc-posix-sem*)",
            "(deny user-preference-write)",
            "(deny mach-lookup)",
            "(allow mach-lookup (global-name \"com.apple.MTLCompilerService\") (global-name \"com.apple.system.opendirectoryd.libinfo\") (global-name \"com.apple.system.opendirectoryd.membership\") (global-name \"com.apple.logd\") (global-name \"com.apple.diagnosticd\"))",
        ];
        for (private_dir, resident_socket) in [
            (None, None),
            (Some("/private/dir"), None),
            (None, Some("/tmp/bw.sock")),
            (Some("/private/dir"), Some("/tmp/bw.sock")),
        ] {
            let profile =
                build_seatbelt_profile("/opt/engine", "/g.json", private_dir, resident_socket, &[]);
            for rule in rules {
                assert!(profile.lines().any(|line| line == rule), "missing {rule}");
            }
        }
    }

    #[test]
    fn profile_with_resident_socket_adds_exactly_one_rule_after_deny_network() {
        let without = build_seatbelt_profile("/opt/engine", "/g.json", None, None, &[]);
        let with =
            build_seatbelt_profile("/opt/engine", "/g.json", None, Some("/tmp/bw.sock"), &[]);
        let rule = "(allow network-outbound (remote unix-socket (path-literal \"/tmp/bw.sock\")))";
        assert!(with.contains(rule), "the allow rule is present: {with}");
        // EXACTLY one extra line, and it sits immediately after `(deny network*)` so the
        // later, more specific rule beats the blanket deny.
        let with_lines: Vec<&str> = with.lines().collect();
        assert_eq!(with_lines.len(), without.lines().count() + 1);
        let deny_at = with_lines
            .iter()
            .position(|l| *l == "(deny network*)")
            .expect("(deny network*) present");
        assert_eq!(
            with_lines[deny_at + 1],
            rule,
            "rule follows (deny network*)"
        );
        // Removing that one line reproduces the unwidened profile byte-for-byte.
        let stripped: Vec<&str> = with_lines.iter().copied().filter(|l| *l != rule).collect();
        assert_eq!(stripped.join("\n"), without);
        // Nothing else opened: no inbound, no bind/listen, no extra file or network rule.
        assert!(!with.contains("network-inbound"));
        assert!(!with.contains("network-bind"));
        assert_eq!(with.matches("(allow network").count(), 1);
    }

    #[test]
    fn resident_socket_path_is_seatbelt_escaped() {
        let profile =
            build_seatbelt_profile("/opt/e", "/g.json", None, Some(r#"/tmp/b"w.sock"#), &[]);
        assert!(profile.contains(
            r#"(allow network-outbound (remote unix-socket (path-literal "/tmp/b\"w.sock")))"#
        ));
    }

    #[test]
    fn resident_socket_validation_refuses_unrepresentable_paths() {
        assert!(validate_resident_socket("/tmp/bench-worker.sock").is_ok());
        for bad in [
            "relative.sock",            // not absolute
            "./bw.sock",                // not absolute
            "",                         // not absolute
            "/tmp/bw\n(allow default)", // newline would forge a profile line
            "/tmp/bw\u{7f}.sock",       // DEL
            "/tmp/bw\t.sock",           // tab
        ] {
            assert_eq!(
                validate_resident_socket(bad).unwrap_err(),
                OfficialSandboxError::InvalidResidentSocket,
                "must refuse {bad:?}"
            );
        }
    }

    /// Refuse, do NOT skip: a bad socket path fails the whole official run.
    #[test]
    fn fail_closed_invalid_resident_socket() {
        for bad in ["relative.sock", "/tmp/bw\n(allow default)"] {
            let mut inp = base_inputs();
            inp.resident_socket = Some(bad);
            assert_eq!(
                resolve_official_sandbox(&inp, false).unwrap_err(),
                OfficialSandboxError::InvalidResidentSocket
            );
        }
    }

    #[test]
    fn missing_path_resolves_through_its_deepest_existing_ancestor() {
        let missing = format!("/tmp/benchd-missing-{}/child", std::process::id());
        let resolved = absolute_path(&missing).unwrap();
        let canonical_tmp = std::fs::canonicalize("/tmp").unwrap();
        assert_eq!(
            PathBuf::from(resolved),
            canonical_tmp
                .join(format!("benchd-missing-{}", std::process::id()))
                .join("child")
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlink_is_resolved_before_parent_component() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!(
            "benchd-symlink-parent-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let target = root.join("real/nested");
        std::fs::create_dir_all(&target).unwrap();
        let link = root.join("link");
        symlink(&target, &link).unwrap();

        let resolved = absolute_path(&link.join("../missing").to_string_lossy()).unwrap();
        let expected = std::fs::canonicalize(root.join("real"))
            .unwrap()
            .join("missing");
        assert_eq!(PathBuf::from(resolved), expected);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_suffix_is_normalized_lexically() {
        let input = format!("/tmp/benchd-missing-{}/gone/../child", std::process::id());
        let resolved = absolute_path(&input).unwrap();
        let canonical_tmp = std::fs::canonicalize("/tmp").unwrap();
        assert_eq!(
            PathBuf::from(resolved),
            canonical_tmp
                .join(format!("benchd-missing-{}", std::process::id()))
                .join("child")
        );
    }

    #[test]
    fn path_resolution_fails_closed_on_non_not_found_errors() {
        let error = absolute_path("/tmp/bad\0path").unwrap_err();
        assert_ne!(error.kind(), std::io::ErrorKind::NotFound);

        let mut inputs = base_inputs();
        inputs.fallback_executable = "/tmp/bad\0path";
        assert_eq!(
            resolve_official_sandbox(&inputs, false).unwrap_err(),
            OfficialSandboxError::PathResolution
        );
    }

    #[test]
    fn resolver_threads_the_resident_socket_into_the_generated_profile() {
        let mut inp = base_inputs();
        inp.resident_socket = Some("/tmp/bench-worker.sock");
        let plan = resolve_official_sandbox(&inp, false).unwrap();
        let SandboxProfile::Generated(profile) = plan.profile else {
            panic!("expected a generated profile");
        };
        assert!(profile.contains(&format!(
            "(allow network-outbound (remote unix-socket (path-literal \"{}\")))",
            absolute_path("/tmp/bench-worker.sock").unwrap()
        )));
        // An empty value is treated as unset, not as a refusal.
        let mut empty = base_inputs();
        empty.resident_socket = Some("");
        let SandboxProfile::Generated(unwidened) =
            resolve_official_sandbox(&empty, false).unwrap().profile
        else {
            panic!("expected a generated profile");
        };
        assert!(!unwidened.contains("network-outbound"));
    }

    #[test]
    fn sandbox_exec_command_wraps_engine_argv() {
        let (program, args) =
            sandbox_exec_command("/tmp/x.sb", "/opt/engine", "/weights/qwen", &[]);
        assert_eq!(program, "/usr/bin/sandbox-exec");
        assert_eq!(
            args,
            vec![
                "-f",
                "/tmp/x.sb",
                "/opt/engine",
                "runtime-worker",
                "--weights",
                "/weights/qwen",
            ]
        );
    }
}
