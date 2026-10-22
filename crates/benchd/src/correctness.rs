//! `benchd correctness` — the standalone correctness gate (Swift `mlxfast-swift
//! correctness`, main.swift:154-167 → `QwenRuntime.runCorrectness`,
//! QwenRuntimeCorrectness.swift:54-90).
//!
//! Swift's contract, mirrored here:
//! - **Exit code is the verdict**: `return report.passed ? 0 : 1` (main.swift:167).
//!   `benchd correctness` exits `0` on a passing gate, `1` on a failing one, `2` on a
//!   usage error, and `1` on an engine/IO fault (a failed report is still a `1`).
//! - **Full correctness set**: `runCorrectness` calls `runLayeredCorrectness` with the
//!   default `checkGates: true` — base teacher-forced `cases[]` THEN the golden's
//!   anchor / free-run / behavior gates, `caseCount = totalCorrectnessCaseCount`
//!   (QwenRuntimeCorrectness.swift:186-191). So benchd runs `CorrectnessScope::Full`,
//!   the same scope B-2's official path uses — NOT the base-cases-only local default.
//! - **No benchmark oracle required**: `runCorrectness` preflights with
//!   `checkCorrectnessArtifacts` (`requiresBenchmarkOracle: false`,
//!   BenchmarkSupport.swift:96-107), so a golden that carries NO `benchmark` block is a
//!   VALID correctness input — unlike `benchmark`/`official`/`validate-golden`, which
//!   reject it. This module therefore loads the golden WITHOUT the oracle requirement
//!   (the read-first correction: correctness is oracle-optional).
//!
//! READ-FIRST decision on `--gates-only`: Swift's `correctness` command has NO sub-flag —
//! it unconditionally runs the full set (`checkGates: true`). There is no base-only
//! `correctness` variant to gate, so no `--gates-only` flag is added here; that would be a
//! non-Swift surface. (The base-only vs. superset choice lives on `iterate`'s `--strict`,
//! which keys off the local checked-timing modes.)
//!
//! Behavior/GPQA caveat (same as official §8 / MINOR-1): bench-core `CorrectnessScope::Full`
//! evaluates base + anchors + free_run but NOT the behavior/GPQA-TTFT gates (the report
//! carries no `behavior` vector). A behavior-carrying golden is therefore under-evaluated
//! here exactly as on the official path; behavior execution is B-3. Not a full-correctness
//! gate for a behavior-bearing golden.

use std::path::Path;

use bench_core::conformance::{run_conformance, CorrectnessScope};
use bench_core::golden::GoldenFixture;
use bench_core::runner_manifest::{check_hello_against_manifest, parse_manifest, HelloFacts};
use bench_runner::{Hello, LineTransport, Session};

use crate::iterate::{first_conformance_failure, SessionEngine};

/// The verdict + failure branding of a correctness run, shaped for the small JSON summary
/// and the 0/1 exit contract.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrectnessOutcome {
    pub passed: bool,
    /// Swift `CorrectnessReport.caseCount` — the golden's TOTAL declared correctness case
    /// count (checkGates == true ⇒ `totalCorrectnessCaseCount`).
    pub case_count: i64,
    pub first_failing_case: Option<String>,
    pub first_failing_step: Option<i64>,
    pub error: String,
}

impl CorrectnessOutcome {
    /// A concise JSON summary emitted to stdout (the exit code is the authoritative
    /// verdict; full `CorrectnessReport` field-parity — expert_* stats etc. — is a B-3
    /// refinement that needs the real engine, which reports those; the stub reports zero).
    pub fn to_json(&self) -> String {
        let case = self
            .first_failing_case
            .as_deref()
            .map(|c| format!("{c:?}"))
            .unwrap_or_else(|| "null".to_string());
        let step = self
            .first_failing_step
            .map(|s| s.to_string())
            .unwrap_or_else(|| "null".to_string());
        format!(
            "{{\"passed\":{},\"case_count\":{},\"first_failing_case\":{},\"first_failing_step\":{},\"error\":{:?}}}",
            self.passed, self.case_count, case, step, self.error
        )
    }
}

/// The manifest gate (runner contract §13): check the engine's `hello` against the
/// runner manifest at `path` before the correctness set runs.
///
/// `trusted` is the kit's `--trusted` posture — only a trusted build derives the
/// `cohort_reference_replay` capability (§6.1). Returns the NAMED failures, one line each, or
/// `Ok(())` when every check holds. A manifest that cannot be read or parsed is an `Err(String)`
/// too; the caller maps a bad input to the usage exit code.
pub fn check_manifest(path: &Path, hello: &Hello, trusted: bool) -> Result<(), ManifestGateError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ManifestGateError::Input(format!("cannot read {}: {e}", path.display())))?;
    let manifest = parse_manifest(&text)
        .map_err(|e| ManifestGateError::Input(format!("{}: {e}", path.display())))?;
    let facts = HelloFacts {
        backend: hello.backend.as_deref(),
        capabilities: &hello.capabilities,
        spec_modes: &hello.spec_modes,
        max_batch_size: hello.max_batch_size,
        runner: hello.runner.as_ref(),
    };
    let report = check_hello_against_manifest(&manifest, &facts, trusted);
    if report.passed() {
        return Ok(());
    }
    Err(ManifestGateError::Failed(
        report
            .failures
            .iter()
            .map(|failure| {
                let class = failure.name();
                let text = failure.to_string();
                if bench_runner::error::record_private_engine_text(class, &text).is_err() {
                    eprintln!(
                        "benchd: private engine diagnostic audit write failed: class={class}"
                    );
                }
                format!("{class}: {}", bench_runner::error::engine_text_seal(&text))
            })
            .collect(),
    ))
}

/// Why the manifest gate did not pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestGateError {
    /// The manifest file is unreadable or malformed (a usage fault, not a gate verdict).
    Input(String),
    /// The `hello` diverges from the manifest: one named failure per line.
    Failed(Vec<String>),
}

/// Run the FULL correctness set over `session` (Swift `runLayeredCorrectness`, checkGates
/// true). Pure over the transport so tests drive it with an in-process `MockEngine`.
///
/// Mirrors the `iterate_core` bracketing: run_conformance opens/closes a barrier sub-phase
/// per sequence; the LAST sub-phase is closed here via `close_phase` (which also enforces
/// the #54 allocator-drain and the completed-work barrier). A conformance protocol error or
/// a barrier failure is a FAILED outcome (exit 1), never a panic.
///
/// `identity` is the run's resolved per-track model identity: its vocabulary bound is what the
/// conformance path judges worker top-logit distributions against, so a track's own bound —
/// never a compile-time global — decides which distributions are trustworthy.
pub fn correctness_core<T: LineTransport>(
    session: &mut Session<T>,
    golden: &GoldenFixture,
    identity: &bench_core::constants::TrackModelIdentity,
) -> CorrectnessOutcome {
    let case_count = golden.total_correctness_case_count() as i64;

    let report = {
        let mut adapter = SessionEngine {
            session: &mut *session,
            drained_once: false,
        };
        match run_conformance(
            &mut adapter,
            golden,
            bench_core::constants::CORRECTNESS_STEPS,
            CorrectnessScope::Full,
            identity.vocab_size,
        ) {
            Ok(r) => r,
            Err(e) => {
                // A conformance-Err leaves a tainted, discarded session (cannot be closed).
                return CorrectnessOutcome {
                    passed: false,
                    case_count,
                    first_failing_case: None,
                    first_failing_step: None,
                    error: format!("{e}"),
                };
            }
        }
    };

    // Close the final correctness sub-phase (barrier owner) before reporting, on BOTH the
    // pass and the fail exit — a barrier / drain failure fails the run.
    if let Err(e) = session.close_phase() {
        return CorrectnessOutcome {
            passed: false,
            case_count,
            first_failing_case: None,
            first_failing_step: None,
            error: format!("{e}"),
        };
    }

    if report.passed {
        return CorrectnessOutcome {
            passed: true,
            case_count,
            first_failing_case: None,
            first_failing_step: None,
            error: String::new(),
        };
    }

    // Brand the failure with the Swift per-gate message (base → "teacher-forced token
    // mismatch", anchor → "anchor token mismatch", else free-run), in Swift's layered order.
    if let Some(f) = first_conformance_failure(&report) {
        let error = if f.is_base_case {
            "teacher-forced token mismatch"
        } else if report.anchors.iter().any(|a| a.name == f.case && !a.passed) {
            "anchor token mismatch"
        } else {
            "free-run token mismatch"
        };
        CorrectnessOutcome {
            passed: false,
            case_count,
            first_failing_case: Some(f.case),
            first_failing_step: f.step,
            error: error.to_string(),
        }
    } else {
        CorrectnessOutcome {
            passed: false,
            case_count,
            first_failing_case: None,
            first_failing_step: None,
            error: "correctness gate failed".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bench_core::constants::CORRECTNESS_PROMPT_TOKENS;
    use bench_core::golden::load_golden_fixture;
    use bench_runner::mock::MockEngine;
    use serde_json::json;

    /// A golden whose primary case is teacher-forced conformant to [2; 64], with optional
    /// gates spliced in and no benchmark oracle (correctness is oracle-optional).
    fn correctness_golden(gates: Option<serde_json::Value>) -> GoldenFixture {
        let mut doc = json!({
            "version": 1,
            "model_type": "qwen4_exp_text",
            "cases": [
                { "name": "case-a", "prompt_tokens": vec![1i64; CORRECTNESS_PROMPT_TOKENS], "expected_tokens": vec![2i64; 64] }
            ]
        });
        if let Some(g) = gates {
            doc["correctness_gates"] = g;
        }
        let bytes = serde_json::to_vec(&doc).unwrap();
        load_golden_fixture(
            &bytes,
            64,
            CORRECTNESS_PROMPT_TOKENS,
            &identity_125b(),
            Some("qwen4_exp_text"),
            None,
            None,
        )
        .unwrap()
    }

    /// The Qwen 3.8 125B-A6B (MLX) row — the identity these fixtures are written to.
    fn identity_125b() -> bench_core::constants::TrackModelIdentity {
        bench_core::constants::TrackModelIdentity::new("qwen4_exp_text", 248_320, 48, 1_024)
    }

    fn conformant_engine() -> MockEngine {
        MockEngine::new().teacher_forced_tokens(vec![2i64; 64])
    }

    fn run(golden: &GoldenFixture, engine: MockEngine) -> CorrectnessOutcome {
        let (mut session, _hello) = Session::connect(engine).unwrap();
        correctness_core(&mut session, golden, &identity_125b())
    }

    #[test]
    fn conformant_base_case_passes_no_oracle_needed() {
        // A benchmark-LESS golden is a valid correctness input (checkCorrectnessArtifacts,
        // requiresBenchmarkOracle:false); a conformant engine passes → exit-0 verdict.
        let golden = correctness_golden(None);
        assert!(golden.benchmark.is_none());
        let outcome = run(&golden, conformant_engine());
        assert!(outcome.passed, "error={}", outcome.error);
        assert_eq!(
            outcome.case_count,
            golden.total_correctness_case_count() as i64
        );
        assert!(outcome.error.is_empty());
    }

    #[test]
    fn full_scope_evaluates_anchor_gate() {
        // Full scope (Swift checkGates:true) evaluates the anchor gate: a corrupted anchor
        // (engine argmax ≠ 999) FAILS correctness → exit-1 verdict, branded.
        let golden = correctness_golden(Some(json!({
            "anchors": [
                { "name": "bad-anchor", "context_tokens": vec![1i64; 8], "expected_token": 999, "accepted_tokens": [999] }
            ]
        })));
        let outcome = run(&golden, conformant_engine());
        assert!(!outcome.passed);
        assert_eq!(outcome.error, "anchor token mismatch");
        assert_eq!(outcome.first_failing_case.as_deref(), Some("bad-anchor"));
    }

    #[test]
    fn mixed_cases_and_anchors_expressible_per_sequence() {
        // #55: a golden with BOTH a primary teacher-forced case AND anchors. A CONFORMANT
        // engine is now expressible with ONE token list per sequence — [base case tokens],
        // [anc-1 token], [anc-2 token] — WITHOUT hand-concatenating a single flat stream in
        // exact evaluation order. The base case is conformant to [2; 64]; the anchors' argmax
        // must equal their expected_token (7 then 9), which the per-sequence oracle sets as
        // the top-1 logit for each anchor sequence.
        let golden = correctness_golden(Some(json!({
            "anchors": [
                { "name": "anc-1", "context_tokens": vec![1i64; 8], "expected_token": 7, "accepted_tokens": [7] },
                { "name": "anc-2", "context_tokens": vec![1i64; 8], "expected_token": 9, "accepted_tokens": [9] }
            ]
        })));
        let conformant =
            MockEngine::new().teacher_forced_sequences(vec![vec![2i64; 64], vec![7], vec![9]]);
        let outcome = run(&golden, conformant);
        assert!(
            outcome.passed,
            "a conformant engine on a mixed cases+anchors golden must pass without \
             hand-concatenation; error={}",
            outcome.error
        );

        // A NON-conformant engine that diverges ONLY at the SECOND anchor (anc-2 returns 8,
        // golden expects 9) fails at exactly that sequence — the per-sequence failure the
        // flat global-index oracle could not target cleanly. The base case and anc-1 stay
        // conformant, proving the divergence is isolated to the intended sequence.
        let divergent =
            MockEngine::new().teacher_forced_sequences(vec![vec![2i64; 64], vec![7], vec![8]]);
        let outcome = run(&golden, divergent);
        assert!(!outcome.passed);
        assert_eq!(outcome.error, "anchor token mismatch");
        assert_eq!(outcome.first_failing_case.as_deref(), Some("anc-2"));
    }

    #[test]
    fn corrupted_base_case_fails_teacher_forced() {
        // A base-case divergence (engine emits 3 where the golden expects 2) fails with the
        // Swift base-gate message.
        let golden = correctness_golden(None);
        let engine = MockEngine::new().teacher_forced_tokens(vec![3i64; 64]);
        let outcome = run(&golden, engine);
        assert!(!outcome.passed);
        assert_eq!(outcome.error, "teacher-forced token mismatch");
        assert_eq!(outcome.first_failing_case.as_deref(), Some("case-a"));
        assert_eq!(outcome.first_failing_step, Some(0));
    }

    #[test]
    fn nonzero_cache_memory_fails_the_gate() {
        // #54 interplay: an undrained allocator cache trips the close_phase drain assertion,
        // failing the correctness run even though the tokens matched.
        let golden = correctness_golden(None);
        let engine = MockEngine::new()
            .teacher_forced_tokens(vec![2i64; 64])
            .cache_memory(4096);
        let outcome = run(&golden, engine);
        assert!(!outcome.passed);
        assert!(
            outcome.error.contains("clear the MLX allocator cache"),
            "got: {}",
            outcome.error
        );
    }

    #[test]
    fn json_summary_shape() {
        let outcome = CorrectnessOutcome {
            passed: false,
            case_count: 3,
            first_failing_case: Some("bad-anchor".to_string()),
            first_failing_step: None,
            error: "anchor token mismatch".to_string(),
        };
        let v: serde_json::Value = serde_json::from_str(&outcome.to_json()).unwrap();
        assert_eq!(v["passed"], json!(false));
        assert_eq!(v["case_count"], json!(3));
        assert_eq!(v["first_failing_case"], json!("bad-anchor"));
        assert_eq!(v["first_failing_step"], json!(null));
        assert_eq!(v["error"], json!("anchor token mismatch"));
    }
}
