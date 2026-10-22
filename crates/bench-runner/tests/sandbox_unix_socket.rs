//! BEHAVIOURAL proof that the official Seatbelt profile's one resident-socket rule actually
//! works — run against the real `/usr/bin/sandbox-exec`, not against the profile text.
//!
//! Why this test has to exist: the profile text is not self-verifying. `sandbox-exec`
//! accepts a profile whose rules do not have the effect they appear to have — a malformed or
//! mis-targeted filter COMPILES silently and the connect is still denied — so an assertion on
//! the profile string can pass on a profile that denies every phase worker its connection to
//! the resident `bench-worker`. The path also has to survive symlink resolution: Seatbelt
//! matches `path-literal` against the kernel-resolved path, so a rule naming `/tmp/x.sock`
//! does NOT allow a connect to the same socket reached via `/private/tmp/x.sock`. None of
//! that is visible in the text. So this test opens a real `UnixListener` and runs a real
//! client under a real Seatbelt profile derived by `bench_runner::build_seatbelt_profile`,
//! asserting three things:
//!
//!   1. WITH the rule, the connect SUCCEEDS.
//!   2. WITHOUT the rule, the connect fails with EPERM (`Operation not permitted`) — the
//!      exact failure seen on M5 #4.
//!   3. WITH the rule, a connect to a DIFFERENT (existing, live) socket still fails with
//!      EPERM — the allowance is one path-literal, not a hole in `(deny network*)`.
//!
//! Hermetic: no GPU, no engine, no network, no `nc`. The client is THIS test binary
//! re-executed under `sandbox-exec` (see `resident_socket_client`), which is why the derived
//! profile's `(allow process-exec (literal …))` names `current_exe()`. Re-exec (rather than a
//! separate helper crate) also lets the client report the raw errno, so "fails with EPERM" is
//! asserted as EPERM and not merely as a non-zero exit — `nc(1)` is silent about errno, so it
//! cannot tell EPERM apart from ENOENT.
//!
//! macOS-only: `sandbox-exec` and this whole official-run path do not exist elsewhere.
#![cfg(target_os = "macos")]

use bench_runner::build_seatbelt_profile;
use std::io::Write;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Env var that turns the re-executed test binary into the connect client.
const CLIENT_SOCKET_ENV: &str = "BENCH_RUNNER_TEST_CONNECT_SOCKET";
/// Exit code the client uses for a FAILED connect (distinct from 0 and from libtest's 101,
/// so a sandbox refusal of the exec itself cannot be mistaken for a clean connect failure).
const CLIENT_CONNECT_FAILED: i32 = 40;

/// The client half of the behavioural test. It is a `#[test]` only so that re-executing this
/// test binary can reach it; under a normal `cargo test` run `CLIENT_SOCKET_ENV` is unset and
/// this returns immediately without asserting anything.
///
/// When the variable IS set, the process connects to that socket and exits with a code the
/// parent can read: 0 on success, [`CLIENT_CONNECT_FAILED`] on failure, printing `ERRNO=<n>`
/// so the parent can assert on the errno itself. `process::exit` is deliberate — it stops the
/// libtest harness before it can run or report anything else.
#[test]
fn resident_socket_client() {
    let Ok(socket_path) = std::env::var(CLIENT_SOCKET_ENV) else {
        return; // normal test run: not the client
    };
    match UnixStream::connect(&socket_path) {
        Ok(_) => {
            println!("CONNECTED");
            std::process::exit(0);
        }
        Err(err) => {
            println!("ERRNO={}", err.raw_os_error().unwrap_or(-1));
            std::process::exit(CLIENT_CONNECT_FAILED);
        }
    }
}

/// Outcome of one sandboxed connect attempt.
struct ConnectAttempt {
    connected: bool,
    errno: Option<i32>,
    output: String,
}

/// Run this test binary's client under `sandbox-exec -f <profile>` against `socket_path`.
fn connect_under_profile(profile: &str, dir: &Path, socket_path: &Path) -> ConnectAttempt {
    let profile_path = dir.join("profile.sb");
    std::fs::write(&profile_path, profile).expect("write profile");
    let exe = std::env::current_exe().expect("current_exe");
    let out = Command::new("/usr/bin/sandbox-exec")
        .arg("-f")
        .arg(&profile_path)
        .arg(&exe)
        // `--exact` + the test name: run ONLY the client test. `--test-threads=1` keeps the
        // harness from spinning up more threads than the sandbox needs to tolerate.
        .args(["--exact", "resident_socket_client", "--nocapture"])
        .args(["--test-threads", "1"])
        .env(CLIENT_SOCKET_ENV, socket_path)
        .output()
        .expect("spawn sandbox-exec");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // `--nocapture` prints the marker on libtest's own progress line ("test <name> ...
    // ERRNO=1"), so scan for the marker anywhere in the line rather than at its start.
    let errno = text
        .lines()
        .find_map(|l| l.split("ERRNO=").nth(1))
        .and_then(|n| {
            let digits: String = n
                .trim()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            digits.parse::<i32>().ok()
        });
    ConnectAttempt {
        connected: out.status.code() == Some(0) && text.contains("CONNECTED"),
        errno,
        output: text,
    }
}

/// Bind a listener that keeps accepting until dropped, so several attempts can share it.
fn spawn_listener(path: &Path) -> UnixListener {
    let listener = UnixListener::bind(path).expect("bind unix socket");
    let accepting = listener.try_clone().expect("clone listener");
    std::thread::spawn(move || {
        for stream in accepting.incoming() {
            match stream {
                Ok(mut s) => {
                    let _ = s.write_all(b"ok");
                }
                Err(_) => break,
            }
        }
    });
    listener
}

/// A short-enough directory for AF_UNIX paths. `sun_path` is 104 bytes, and a cargo target
/// dir or a long `TMPDIR` blows straight through it — a silent `bind` failure here would look
/// like a sandbox denial, so keep the path tiny and assert it fits.
///
/// The result is CANONICALIZED, and that is not cosmetic: `/tmp` is a symlink to
/// `/private/tmp`, Seatbelt matches `path-literal` against the path the kernel resolves, and a
/// profile naming `/tmp/…` denies a connect to the very socket it meant to allow. Production
/// gets this from `bench_runner::sandbox::absolute_path`; the test must not skip it.
fn short_temp_dir(tag: &str) -> PathBuf {
    let dir = PathBuf::from("/tmp").join(format!("brs-{}-{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    std::fs::canonicalize(&dir).expect("canonicalize temp dir")
}

fn profile_for(exe: &Path, socket: Option<&Path>) -> String {
    build_seatbelt_profile(
        &exe.to_string_lossy(),
        "/nonexistent/golden.json",
        None,
        socket.map(|s| s.to_string_lossy().into_owned()).as_deref(),
        &[],
    )
}

/// The whole behavioural matrix in ONE test: the three cases share one pair of live
/// listeners, and splitting them would let a flaky bind in one case be read as a sandbox
/// verdict in another.
#[test]
fn resident_socket_rule_is_the_only_thing_that_permits_the_connect() {
    if !Path::new("/usr/bin/sandbox-exec").is_file() {
        return; // no Seatbelt: nothing to prove
    }
    let dir = short_temp_dir("sb");
    let allowed = dir.join("a.sock");
    let other = dir.join("b.sock");
    assert!(
        allowed.as_os_str().len() < 100,
        "socket path must fit sun_path: {}",
        allowed.display()
    );
    let _allowed_listener = spawn_listener(&allowed);
    let _other_listener = spawn_listener(&other);
    let exe = std::env::current_exe().expect("current_exe");

    // Control: unsandboxed, both sockets are live and connectable. This is what makes a
    // sandboxed EPERM below attributable to Seatbelt and not to a dead socket.
    UnixStream::connect(&allowed).expect("control connect to the allowed socket");
    UnixStream::connect(&other).expect("control connect to the other socket");

    // 1. WITH the rule → the connect succeeds.
    let with = connect_under_profile(&profile_for(&exe, Some(&allowed)), &dir, &allowed);
    assert!(
        with.connected,
        "the resident-socket rule must permit the connect; got: {}",
        with.output
    );

    // 2. WITHOUT the rule → EPERM, the failure observed on M5 #4.
    let without = connect_under_profile(&profile_for(&exe, None), &dir, &allowed);
    assert!(
        !without.connected,
        "(deny network*) must deny an AF_UNIX connect: {}",
        without.output
    );
    assert_eq!(
        without.errno,
        Some(libc_eperm()),
        "expected EPERM (Operation not permitted), got: {}",
        without.output
    );

    // 3. WITH the rule, a DIFFERENT live socket → still EPERM. The allowance is exactly one
    //    path-literal; it does not reopen AF_UNIX generally.
    let wrong = connect_under_profile(&profile_for(&exe, Some(&allowed)), &dir, &other);
    assert!(
        !wrong.connected,
        "the rule must not permit a connect to another socket: {}",
        wrong.output
    );
    assert_eq!(
        wrong.errno,
        Some(libc_eperm()),
        "expected EPERM for the non-allowed socket, got: {}",
        wrong.output
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// `EPERM` (1). Spelled out rather than pulled from a `libc` dependency this crate does not
/// otherwise need.
fn libc_eperm() -> i32 {
    1
}
