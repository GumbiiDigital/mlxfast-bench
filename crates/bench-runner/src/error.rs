//! Std-only error type for the runner.
//!
//! No external error crates (`thiserror`/`anyhow`) — the offline build only has
//! `serde`, `serde_json`, and std. `RunnerError` is a hand-written enum implementing
//! `Display` + `std::error::Error`, with `From` conversions for the two foreign error
//! types that cross the transport boundary (`std::io::Error`, `serde_json::Error`).

use std::fmt;

/// Reserved private-audit key for full scrubbed engine diagnostics.
pub const PRIVATE_ENGINE_DIAGNOSTICS_KEY: &str = "engine_diagnostics";

/// Everything that can go wrong driving an engine session.
#[derive(Debug)]
pub enum RunnerError {
    /// Underlying transport I/O failure (pipe write, read, spawn).
    Io(std::io::Error),
    /// A line could not be (de)serialized as an Engine Protocol message.
    Json(serde_json::Error),
    /// Protocol-level violation that is not an engine-signalled error:
    /// malformed hello, unexpected EOF, response id mismatch, unparseable line at EOF, etc.
    Protocol(String),
    /// A protocol violation with separate benchd-written and engine-written parts.
    ProtocolEngineText { detail: String, engine_text: String },
    /// The response nonce did not equal the session nonce established at hello.
    NonceMismatch {
        expected: String,
        got: Option<String>,
    },
    /// The engine returned `ok:false`. `kind` is the request kind that failed;
    /// `message` is the engine's `error` string (or a placeholder if absent).
    Engine { kind: String, message: String },
    /// Phase-close barrier failure (architecture §3): the engine's reported
    /// `completed_work` did not equal the number of timed steps the runner issued.
    CompletedWorkMismatch { issued: i64, reported: Option<i64> },
    /// Allocator-drain failure (#54): the engine reported a non-zero `cache_memory`
    /// on `phase_diagnostics`, so the MLX free-buffer cache was not drained to zero at
    /// the phase boundary (Swift `resetRuntimeWorkerAllocatorForPhaseStart` fails closed
    /// unless `Memory.cacheMemory == 0`). `reported` is the non-zero byte count.
    AllocatorCacheNotDrained { reported: i64 },
    /// A timed-benchmark response token diverged from the golden benchmark oracle
    /// (Swift `BenchmarkTokenMismatchError`, thrown by `measureWorkerPrefillSecondsPerToken`
    /// / `measureWorkerDecode`). `label` names the phase (e.g. "benchmark decode token"),
    /// `step` is the 0-based step index, and `expected`/`actual` are the oracle vs engine
    /// tokens. A fast engine returning garbage on the timed path is rejected here rather
    /// than being credited with an inflated speedup.
    TokenMismatch {
        label: String,
        step: usize,
        expected: i64,
        actual: i64,
    },
    /// A request was attempted after a prior error discarded the session.
    SessionDiscarded,
    /// benchd tried to issue a capability-gated request (a v1.1 `free_decode_*` kind) to an
    /// engine that did not advertise the required `hello.capabilities` flag. An unadvertised
    /// capability is a hard protocol error, never a silent fallback (PROTOCOL-v1.1.md §2.1);
    /// the session is discarded fail-closed.
    CapabilityNotAdvertised { capability: String },
    /// A v1.1 free-run decode phase failed the §2.6 consistency TRIPLE or a §2.4 count
    /// invariant (`bench_core::free_run::verify_consistency`): a doctored acceptance histogram
    /// or a miscounted forward barrier. `detail` is the specific inconsistency. Fail-closed.
    FreeRunConsistency { detail: String },
    /// H3 (cycle-3) — a timed round-trip exceeded the benchd RunTimeout budget
    /// (PROTOCOL-v1.1 §2.2/§4: `N × band-ceiling × margin`). A hung / looping / stalling engine did
    /// not return within the wall-clock budget, so benchd aborts the read, discards the session
    /// (fail-closed), and the run fails — the transport has no watchdog otherwise, so this is what
    /// bounds a wedge inside the timed window. `phase` names the timed phase; `budget_seconds` is the
    /// armed budget. This never affects the score (a passing run finishes well inside the budget).
    RunTimeout { phase: String, budget_seconds: f64 },
    /// #108 (M2) — the §2.2 RunTimeout budget could NOT be computed for this leg (`N × band-ceiling
    /// × margin` was degenerate: a non-finite / non-positive ceiling or margin, or N == 0). The
    /// ceiling is CALIBRATION-DERIVED (`serial_mean × serial_band_high`), so this condition is
    /// reachable from a `BASELINE_CALIBRATION` file; benchd refuses to open a timed window it cannot
    /// bound rather than arming no deadline at all. `detail` is
    /// [`bench_core::score::run_timeout_budget`]'s diagnostic. The leg fails under its own reject
    /// class — never a silent unbounded run.
    RunTimeoutBudgetInvalid { detail: String },
    /// Spec-never-ignored (`docs/spec-config-design.md` §6): a `decode_begin` carried a `spec`
    /// but the engine's echoed `effective_spec` was ABSENT or DIVERGED from the request. benchd
    /// seals only the echo and never lets a leg silently run a different (or default) spec than the
    /// one requested, so a divergence discards the session fail-closed (mirrors the
    /// `CapabilityNotAdvertised` posture). `requested`/`effective` are the wire JSON for the audit.
    SpecEchoDivergence {
        requested: String,
        effective: Option<String>,
    },
    /// Medium (#105) — a `decode_begin` requested a NON-default speculative `mode` that the engine
    /// did NOT advertise as runnable on its `hello.spec_modes`. Enforced BEFORE THE TIMED SEED
    /// FORWARD is issued, so no forward work is ever charged to a mode the engine cannot run, and
    /// the session is discarded fail-closed (mirrors `CapabilityNotAdvertised`).
    ///
    /// Cycle-5 finding 6 — this was written "PRE-CLOCK", which is not true of the code: the wall
    /// clock starts at `measure_decode`'s `Instant::now()` (`timing.rs`) BEFORE `decode_begin_spec`
    /// is called, so the refusal happens with the clock already running. What it precedes is the
    /// timed seed forward — which is the property that matters (the discarded session is never
    /// scored) but is not the property "pre-clock" claims.
    ///
    /// `serial` (the default path) is always runnable and never gated. `advertised` is the engine's
    /// runnable-mode list.
    SpecModeNotRunnable {
        mode: String,
        advertised: Vec<String>,
    },
    /// v1.2 (COHORT) — BATCH-NEVER-IGNORED: a batched `free_decode_begin` carried an explicit
    /// `batch_size` but the engine's echoed `effective_batch_size` was ABSENT or DIVERGED from the
    /// request. The cohort width is a pinned identity, not a configuration hint (the same posture
    /// as `SpecEchoDivergence`): benchd never lets a leg silently run a narrower (or wider) cohort
    /// than the one requested, because that would seal a different physical arrangement of work
    /// under the requested B's series tag. The session is discarded fail-closed.
    BatchEchoDivergence {
        requested: u32,
        effective: Option<u32>,
    },
    /// v1.2 (COHORT) — the requested cohort width exceeds the `max_batch_size` the engine
    /// advertised on its hello. Refused PRE-GPU — before the cool gate and before the clock — so
    /// no timed window is ever opened against an engine that cannot serve the width. Deterministic
    /// for a given engine build (the second spawn reads the same hello), like
    /// `CapabilityNotAdvertised`.
    BatchWidthExceedsEngineMax { requested: u32, max_batch_size: u32 },
    /// A per-phase thermal / quiescence / clock-floor GATE rejected the leg (measure-job
    /// finding 5). A TYPED rejection so the measure-job classifier keys on the VARIANT, not
    /// on error prose (the old `is_gate_class` substring match on `RunnerError::Protocol` was
    /// brittle). `phase` names the timed phase ("prefill"/"decode"), `reason` is the human
    /// diagnostic. This is the one-gated-retry class; other errors fail closed.
    // UNVERIFIED(measure-job): the gate-class classification for the retry decision.
    GateRejected { phase: String, reason: String },
}

impl fmt::Display for RunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunnerError::Io(e) => write!(f, "transport io error: {e}"),
            RunnerError::Json(e) => write!(f, "protocol json error: {}", json_error_summary(e)),
            RunnerError::Protocol(msg) => write!(f, "protocol violation: {msg}"),
            RunnerError::ProtocolEngineText {
                detail,
                engine_text,
            } => {
                write!(f, "protocol violation: {detail} (")?;
                write_engine_text_seal(f, engine_text)?;
                write!(f, ")")
            }
            RunnerError::NonceMismatch { got, .. } => match got {
                Some(got) => {
                    write!(f, "nonce mismatch: ")?;
                    write_engine_text_seal(f, got)
                }
                None => write!(f, "nonce mismatch: got none"),
            },
            RunnerError::Engine { kind, message } => {
                write!(f, "engine reported failure on {kind:?}: ")?;
                write_engine_text_seal(f, message)
            }
            RunnerError::CompletedWorkMismatch { issued, reported } => match reported {
                Some(reported) => write!(
                    f,
                    "phase-close barrier: issued {issued} timed steps but engine reported {reported} completed"
                ),
                None => write!(
                    f,
                    "phase-close barrier: issued {issued} timed steps but engine reported no completed_work counter"
                ),
            },
            RunnerError::AllocatorCacheNotDrained { reported } => write!(
                f,
                "runtime worker failed to clear the MLX allocator cache at phase start (cache_memory={reported} bytes, expected 0)"
            ),
            // No token value is named: the golden's token is hidden material, and the engine's
            // token at that step would tell a reader where the golden goes next.
            RunnerError::TokenMismatch { label, step, .. } => write!(
                f,
                "{label} mismatch at step {step}: the engine returned a token that is not the token the golden expects at this step"
            ),
            RunnerError::SessionDiscarded => {
                write!(f, "session discarded by a prior error; no further requests permitted")
            }
            RunnerError::CapabilityNotAdvertised { capability } => write!(
                f,
                "engine did not advertise the {capability:?} capability; refusing the capability-gated request (fail-closed)"
            ),
            RunnerError::FreeRunConsistency { detail } => {
                write!(f, "free-run decode consistency failure: {detail}")
            }
            RunnerError::GateRejected { phase, reason } => {
                write!(f, "gate rejected ({phase}): {reason}")
            }
            RunnerError::SpecEchoDivergence {
                requested,
                effective,
            } => match effective {
                Some(effective) => {
                    write!(
                        f,
                        "spec echo divergence: requested {requested} but the engine echoed "
                    )?;
                    write_engine_text_seal(f, effective)?;
                    write!(
                        f,
                        " (spec-never-ignored, fail-closed — session discarded)"
                    )
                }
                None => write!(
                    f,
                    "spec echo divergence: requested {requested} but the engine echoed no \
                     effective_spec (spec-never-ignored, fail-closed — session discarded)"
                ),
            },
            RunnerError::BatchEchoDivergence {
                requested,
                effective,
            } => match effective {
                Some(effective) => write!(
                    f,
                    "batch echo divergence: requested batch_size {requested} but the engine echoed \
                     effective_batch_size {effective} (batch-never-ignored, fail-closed — session \
                     discarded)"
                ),
                None => write!(
                    f,
                    "batch echo divergence: requested batch_size {requested} but the engine echoed \
                     no effective_batch_size (batch-never-ignored, fail-closed — session discarded)"
                ),
            },
            RunnerError::BatchWidthExceedsEngineMax {
                requested,
                max_batch_size,
            } => write!(
                f,
                "requested cohort batch_size {requested} exceeds the engine's advertised \
                 max_batch_size {max_batch_size}; refusing pre-GPU (fail-closed)"
            ),
            RunnerError::SpecModeNotRunnable { mode, advertised } => {
                let advertised = serde_json::to_string(advertised).unwrap_or_default();
                write!(
                    f,
                    "engine did not advertise speculative mode {mode:?} as runnable \
                     (hello.spec_modes: "
                )?;
                write_engine_text_seal(f, &advertised)?;
                write!(
                    f,
                    "); refusing the spec'd decode before any timed work (fail-closed — session \
                     discarded)"
                )
            }
            RunnerError::RunTimeout {
                phase,
                budget_seconds,
            } => write!(
                f,
                "run timeout ({phase}): the timed round-trip exceeded the {budget_seconds}s benchd \
                 budget (N × band-ceiling × margin); the engine did not respond in time — session \
                 discarded (fail-closed)"
            ),
            RunnerError::RunTimeoutBudgetInvalid { detail } => write!(
                f,
                "run timeout budget could not be armed: {detail}; benchd refuses to open a timed \
                 window with no §2.2 wall-clock bound (fail-closed)"
            ),
        }
    }
}

fn write_engine_text_seal(f: &mut fmt::Formatter<'_>, text: &str) -> fmt::Result {
    f.write_str(&engine_text_seal(text))
}

/// A stable seal for engine-written text stored in artifacts.
pub fn engine_text_seal(text: &str) -> String {
    let scrubbed = crate::scrub::scrub_engine_text(text);
    format!(
        "engine-text-sha256={} engine-text-bytes={}",
        bench_core::hash::sha256_hex(scrubbed.as_bytes()),
        scrubbed.len()
    )
}

/// Keep full scrubbed engine text only in the organizer's private audit, when configured.
/// Existing private-audit JSON is preserved so the later score writer can add its full payload.
pub fn record_private_engine_text(class: &str, text: &str) -> std::io::Result<()> {
    record_private_engine_value(class, &crate::scrub::scrub_engine_text(text))
}

/// Keep an exact engine-supplied identity value only in the organizer's private audit.
pub fn record_private_engine_value(class: &str, value: &str) -> std::io::Result<()> {
    use std::io::Write;

    let Some(path) = std::env::var_os("BENCHD_PRIVATE_AUDIT_PATH") else {
        return Ok(());
    };
    let mut root = match std::fs::read(&path) {
        Ok(bytes) if !bytes.is_empty() => serde_json::from_slice::<serde_json::Value>(&bytes)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?,
        Ok(_) => serde_json::json!({}),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => serde_json::json!({}),
        Err(error) => return Err(error),
    };
    let object = root.as_object_mut().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "BENCHD_PRIVATE_AUDIT_PATH must contain a JSON object",
        )
    })?;
    let diagnostics = object
        .entry(PRIVATE_ENGINE_DIAGNOSTICS_KEY)
        .or_insert_with(|| serde_json::json!([]))
        .as_array_mut()
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "private audit engine_diagnostics must be an array",
            )
        })?;
    diagnostics.push(serde_json::json!({
        "class": class,
        "text": value,
    }));
    let bytes = serde_json::to_vec_pretty(&root).map_err(std::io::Error::other)?;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(&bytes)
}

/// Preserve a documented verification mode and seal all other engine text.
pub fn seal_verification_mode(mode: &str) -> String {
    if bench_protocol::VERIFICATION_MODES.contains(&mode) {
        mode.to_string()
    } else {
        engine_text_seal(mode)
    }
}

/// Preserve a documented spec mode and seal all other engine text.
pub fn seal_spec_mode(mode: &str) -> String {
    if bench_protocol::SPEC_MODES.contains(&mode) {
        mode.to_string()
    } else {
        engine_text_seal(mode)
    }
}

/// Preserve documented logit provenance and seal all other engine text.
pub fn seal_logit_provenance(provenance: &str) -> String {
    if provenance == bench_protocol::LOGIT_PROVENANCE_POST_SOFTCAP {
        provenance.to_string()
    } else {
        engine_text_seal(provenance)
    }
}

/// Seal unsafe engine-written histogram keys before they enter an audit record.
pub fn seal_depth_clamp_reasons(
    reasons: &std::collections::BTreeMap<String, u32>,
) -> std::collections::BTreeMap<String, u64> {
    let mut sealed = std::collections::BTreeMap::new();
    for (key, count) in reasons {
        let key = if depth_clamp_reason_key_is_safe(key) {
            key.clone()
        } else {
            engine_text_seal(key)
        };
        let count = u64::from(*count);
        // The scrubber can map different unsafe keys to one seal. Audit counts use u64 so the
        // full sum of normal u32 wire counts is retained.
        sealed
            .entry(key)
            .and_modify(|total: &mut u64| *total += count)
            .or_insert(count);
    }
    sealed
}

fn depth_clamp_reason_key_is_safe(key: &str) -> bool {
    let bytes = key.as_bytes();
    (1..=48).contains(&bytes.len())
        && bytes[0].is_ascii_lowercase()
        && bytes[1..]
            .iter()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'_')
}

pub(crate) fn json_error_summary(error: &serde_json::Error) -> String {
    format!(
        "category={:?} line={} column={}",
        error.classify(),
        error.line(),
        error.column()
    )
}

impl RunnerError {
    /// Stable public class for an engine-originated diagnostic.
    pub fn engine_diagnostic_class(&self) -> &'static str {
        match self {
            RunnerError::Json(_) => "protocol-json",
            RunnerError::ProtocolEngineText { .. } => "protocol-engine-text",
            RunnerError::NonceMismatch { .. } => "nonce-mismatch",
            RunnerError::Engine { .. } => "engine-reported-failure",
            RunnerError::SpecEchoDivergence { .. } => "spec-echo-divergence",
            RunnerError::SpecModeNotRunnable { .. } => "spec-mode-not-runnable",
            _ => "runner-error",
        }
    }

    /// Full scrubbed text for the private audit and tests. Never print this to a public log.
    pub fn diagnostic(&self) -> String {
        match self {
            RunnerError::Json(e) => format!(
                "protocol json error: {}",
                crate::scrub::scrub_engine_text(&e.to_string())
            ),
            RunnerError::ProtocolEngineText {
                detail,
                engine_text,
            } => format!(
                "protocol violation: {detail} ({})",
                crate::scrub::scrub_engine_text(engine_text)
            ),
            RunnerError::NonceMismatch { expected, got } => {
                let expected = crate::scrub::scrub_engine_text(expected);
                match got {
                    Some(got) => format!(
                        "nonce mismatch: expected {expected:?}, got {:?}",
                        crate::scrub::scrub_engine_text(got)
                    ),
                    None => format!("nonce mismatch: expected {expected:?}, got none"),
                }
            }
            RunnerError::Engine { kind, message } => format!(
                "engine reported failure on {kind:?}: {}",
                crate::scrub::scrub_engine_text(message)
            ),
            RunnerError::SpecEchoDivergence {
                requested,
                effective,
            } => match effective {
                Some(effective) => format!(
                    "spec echo divergence: requested {requested} but the engine echoed {} \
                     (spec-never-ignored, fail-closed — session discarded)",
                    crate::scrub::scrub_engine_text(effective)
                ),
                None => format!(
                    "spec echo divergence: requested {requested} but the engine echoed no \
                     effective_spec (spec-never-ignored, fail-closed — session discarded)"
                ),
            },
            RunnerError::SpecModeNotRunnable { mode, advertised } => {
                let advertised: Vec<String> = advertised
                    .iter()
                    .map(|text| crate::scrub::scrub_engine_text(text))
                    .collect();
                format!(
                    "engine did not advertise speculative mode {mode:?} as runnable \
                     (hello.spec_modes = {advertised:?}); refusing the spec'd decode before any \
                     timed work (fail-closed — session discarded)"
                )
            }
            _ => self.to_string(),
        }
    }
}

impl std::error::Error for RunnerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RunnerError::Io(e) => Some(e),
            RunnerError::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RunnerError {
    fn from(e: std::io::Error) -> Self {
        RunnerError::Io(e)
    }
}

impl From<serde_json::Error> for RunnerError {
    fn from(e: serde_json::Error) -> Self {
        RunnerError::Json(e)
    }
}

/// Convenience alias used throughout the runner.
pub type Result<T> = std::result::Result<T, RunnerError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_text_variants_seal_display_and_keep_diagnostic_text() {
        const MARKER: &str = "PLANTED-PROMPT-TEXT 4242";
        const EXPECTED_NONCE: &str = "/Users/operator/private/nonce.txt";
        let errors = [
            RunnerError::Engine {
                kind: "decode_step".to_string(),
                message: MARKER.to_string(),
            },
            RunnerError::NonceMismatch {
                expected: EXPECTED_NONCE.to_string(),
                got: Some(MARKER.to_string()),
            },
            RunnerError::SpecEchoDivergence {
                requested: r#"{"mode":"mtp"}"#.to_string(),
                effective: Some(MARKER.to_string()),
            },
            RunnerError::SpecModeNotRunnable {
                mode: "mtp".to_string(),
                advertised: vec![MARKER.to_string()],
            },
            RunnerError::ProtocolEngineText {
                detail: "engine supplied a detail".to_string(),
                engine_text: MARKER.to_string(),
            },
        ];

        for error in errors {
            let sealed = error.to_string();
            assert!(sealed.contains("engine-text-sha256="), "{sealed}");
            assert!(!sealed.contains("PLANTED"), "{sealed}");
            assert!(!sealed.contains("4242"), "{sealed}");
            let diagnostic = error.diagnostic();
            assert!(diagnostic.contains(MARKER), "{diagnostic}");
            if matches!(error, RunnerError::NonceMismatch { .. }) {
                assert!(!diagnostic.contains(EXPECTED_NONCE), "{diagnostic}");
                assert!(
                    diagnostic.contains(&crate::scrub::scrub_engine_text(EXPECTED_NONCE)),
                    "{diagnostic}"
                );
            }
        }
    }

    #[test]
    fn json_display_never_includes_the_input_text() {
        const MARKER: &str = "PLANTED-PROMPT-TEXT 4242";
        let marker_error = serde_json::from_str::<bench_protocol::WorkerResponse>(
            r#"{"id":0,"ok":true,"PLANTED-PROMPT-TEXT 4242":1}"#,
        )
        .expect_err("the marker is an unknown protocol field");
        let custom = RunnerError::Json(marker_error);
        assert!(!custom.to_string().contains(MARKER));
        assert!(custom.diagnostic().contains(MARKER));

        let error = serde_json::from_str::<serde_json::Value>(r#"{"x": "PLANTED""#)
            .expect_err("the planted input is invalid JSON");
        let sealed = RunnerError::Json(error).to_string();
        assert!(sealed.contains("protocol json error"), "{sealed}");
        assert!(!sealed.contains("PLANTED"), "{sealed}");
    }

    /// A timed token mismatch names the phase and the step, never the golden's token or the
    /// engine's: this Display is what reaches `metrics.error`, `rejected_pairs[].reason` and
    /// stderr on every timed path.
    #[test]
    fn a_token_mismatch_names_no_token() {
        let text = RunnerError::TokenMismatch {
            label: "benchmark free-run decode token".to_string(),
            step: 3,
            expected: 123_457,
            actual: 234_568,
        }
        .to_string();
        assert!(
            text.starts_with("benchmark free-run decode token mismatch at step 3"),
            "{text}"
        );
        assert!(
            !text.contains("123457") && !text.contains("234568"),
            "{text}"
        );
    }
}
