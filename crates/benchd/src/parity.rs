//! §T — the parity verdict tool: `benchd parity-diff <a.json> <b.json>`.
//!
//! A Rust port of `scripts/parity-diff.py` that shares the ACTUAL `ScoreMetrics` type, so
//! the bucket roster is checked against the serde field names by a `cargo test`
//! (`roster_covers_score_metrics_exactly`): a new `ScoreMetrics` field that nobody buckets
//! fails the build, not a future live window (§T1 exhaustiveness). Same buckets, same 1e-9
//! float rule, same `peak_ram` tolerance, same failing-run + error-semantics comparison, same
//! PASS/FAIL verdict and exit codes as the Python differ (which becomes a shim, §T4).
//!
//! §F3 (failing-pair MODE): the pair's mode is decided ONCE from both sides' strict `passed`
//! bool. On a failing/superset pair the failing side zeroes/nulls its timing, so a Timed field
//! is waived ONLY when a side is actually zeroed/nulled — a genuine both-numeric divergence
//! still hard-fails (the mode is never a blanket ranking-surface amnesty). A `passed` that is
//! present-but-not-a-bool is itself a hard fail (never coerced), so a non-bool `passed` can't
//! silently un-gate ranking. The deterministic + failing-run surface always gates.

use std::path::Path;
use std::process::ExitCode;

use serde_json::Value;

/// Which comparison rule a field obeys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bucket {
    /// Ranking timing field — must agree within the 10% band, else FAIL (only when both runs
    /// passed; on a failing/superset pair the zeroed timing side is expected).
    Timed,
    /// Deterministic — exact for bool/int/str, 1e-9 numeric tolerance for floats.
    Det,
    /// Deterministic with a wider numeric tolerance (`peak_ram_gb`, David 2026-08-17).
    DetTol,
    /// Failing-run verification surface (null/0/"" on a passing run); exact/tol compare.
    Failing,
    /// Error text — compared by failure CLASS (semantics), not exact string (David 2026-08-17).
    Error,
    /// Environmental — expected to differ between honest producers (informational, waived).
    Env,
}

use Bucket::*;

/// Every top-level `score`/`passed` and every `metrics.<field>` in exactly one bucket. Mirrors
/// `scripts/parity-diff.py`; pinned to the real schema by the exhaustiveness test.
const ROSTER: &[(&str, Bucket)] = &[
    ("score", Timed),
    ("passed", Det),
    // 2a RANKING-TIMED
    ("metrics.decode_seconds_per_token", Timed),
    ("metrics.prefill_seconds_per_token", Timed),
    ("metrics.decode_speedup", Timed),
    ("metrics.prefill_speedup", Timed),
    // 2b/2c DETERMINISTIC
    ("metrics.passed_correctness", Det),
    ("metrics.decode_speedup_floor", Det),
    ("metrics.prefill_speedup_floor", Det),
    ("metrics.passed_decode_speedup_floor", Det),
    ("metrics.passed_prefill_speedup_floor", Det),
    ("metrics.baseline_decode_seconds_per_token", Det),
    ("metrics.baseline_prefill_seconds_per_token", Det),
    ("metrics.golden_hash", Det),
    ("metrics.case_count", Det),
    ("metrics.checked_steps", Det),
    ("metrics.bandwidth_gb_per_token", Det),
    ("metrics.bandwidth_source", Det),
    ("metrics.num_layers", Det),
    ("metrics.partial_result", Det),
    ("metrics.gpqa_ttft_passed", Det),
    ("metrics.gpqa_ttft_pass_count", Det),
    ("metrics.gpqa_ttft_case_count", Det),
    ("metrics.gpqa_ttft_seconds", Det),
    ("metrics.gpqa_ttft_p50_seconds", Det),
    ("metrics.gpqa_ttft_max_seconds", Det),
    ("metrics.gpqa_ttft_source", Det),
    ("metrics.semantic_gpqa_passed", Det),
    ("metrics.semantic_gpqa_pass_count", Det),
    ("metrics.semantic_gpqa_case_count", Det),
    ("metrics.semantic_gpqa_model", Det),
    ("metrics.expert_cache_hits", Det),
    ("metrics.expert_cache_misses", Det),
    ("metrics.expert_cache_evictions", Det),
    ("metrics.expert_bytes_read", Det),
    ("metrics.expert_read_seconds", Det),
    ("metrics.expert_peak_cached_tensors", Det),
    ("metrics.expert_hit_rate", Det),
    // 2f WEIGHTS (P2 digest port → deterministic)
    ("metrics.weights_hash", Det),
    ("metrics.weights_byte_count", Det),
    ("metrics.weights_file_count", Det),
    // 2g DET-with-tolerance
    ("metrics.peak_ram_gb", DetTol),
    // 2d FAILING-RUN surface
    ("metrics.first_failing_layer", Failing),
    ("metrics.first_failing_case", Failing),
    ("metrics.first_failing_step", Failing),
    ("metrics.expected_token", Failing),
    ("metrics.actual_token", Failing),
    ("metrics.max_abs_diff", Failing),
    // ERROR (semantics)
    ("metrics.error", Error),
    // 2e ENVIRONMENTAL (waived)
    ("metrics.runtime", Env),
    ("metrics.commit", Env),
    ("metrics.timestamp", Env),
    ("metrics.harness_hash", Env),
    ("metrics.process_resident_memory_gb", Env),
    ("metrics.benchmark_wall_seconds", Env),
    ("metrics.timed_benchmark_seconds", Env),
    ("metrics.preflight_seconds", Env),
    ("metrics.correctness_seconds", Env),
];

/// Keys the differ WAIVES outright: present or absent, on either side, they are never compared
/// and never drift-fail.
///
/// `metrics.gates` is the run's gate log (David 2026-09-17). It is environmental BY CONSTRUCTION —
/// how long a box took to go idle and to cool, and the load, utilization and temperature it
/// reached, are facts about that box on that day, so two honest producers never agree on them. It
/// cannot be ROSTERED as `Env`, because the roster's missing-key gate hard-fails any rostered key
/// that is absent on a side, and the reference never emits this one; and it cannot be left
/// UNKNOWN, because the unknown-key gate hard-fails any key that is present and unrostered. So it
/// is named here, waived on both counts, and the value comparison below (which walks the ROSTER,
/// not the keys) never reaches it.
const WAIVED_KEYS: &[&str] = &["metrics.gates"];

/// Whether the differ waives this flattened key entirely ([`WAIVED_KEYS`]).
pub fn is_waived(key: &str) -> bool {
    WAIVED_KEYS.contains(&key)
}

const DET_FLOAT_TOL: f64 = 1e-9;
const PEAK_RAM_REL_TOL: f64 = 0.05;
const TIMED_BAND: f64 = 0.10;

/// The bucket for a flattened key, or `None` if the key is not in the roster (→ hard fail).
pub fn bucket_of(key: &str) -> Option<Bucket> {
    ROSTER.iter().find(|(k, _)| *k == key).map(|(_, b)| *b)
}

/// Flatten a score `Value` to `{ "metrics.foo": v, "score": v, "passed": v }`.
fn flatten(v: &Value) -> Vec<(String, Value)> {
    let mut out = Vec::new();
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            if let Some(inner) = val.as_object() {
                for (ik, iv) in inner {
                    out.push((format!("{k}.{ik}"), iv.clone()));
                }
            } else {
                out.push((k.clone(), val.clone()));
            }
        }
    }
    out
}

fn as_f64(v: &Value) -> Option<f64> {
    // Exclude booleans (serde_json bools are not numbers, but guard anyway).
    if v.is_boolean() {
        return None;
    }
    v.as_f64()
}

/// Numeric closeness with a relative tolerance (NaN==NaN).
fn num_close(x: f64, y: f64, rel: f64) -> bool {
    if x.is_nan() || y.is_nan() {
        return x.is_nan() && y.is_nan();
    }
    if x == y {
        return true;
    }
    let denom = x.abs().max(y.abs()).max(1e-12);
    (x - y).abs() / denom <= rel
}

/// Coarse failure-class key for `error` (semantics compare, not exact string): null/empty stay
/// empty; a string maps to its leading clause before the first ':' lowercased (benchd/Swift
/// wording after the ':' differs; the class does not). Refined further as fixtures land.
///
/// A non-null NON-STRING `error` is off-schema; it must NOT collapse to "" (which would make
/// `error: 5` vs `error: 7` compare equal and PASS — Fable checkpoint Finding 3b). Give it a
/// class that encodes the value, so two differing off-schema shapes get distinct classes and
/// hard-fail like any other schema divergence.
fn error_class(s: &Value) -> String {
    match s {
        Value::Null => String::new(),
        Value::String(txt) => {
            let txt = txt.trim();
            if txt.is_empty() {
                return String::new();
            }
            let head = txt.split(':').next().unwrap_or("").trim().to_lowercase();
            if head.is_empty() {
                "<nonempty>".to_string()
            } else {
                head
            }
        }
        other => format!("<non-string-error:{other}>"),
    }
}

/// One field-level mismatch.
#[derive(Debug, Clone, PartialEq)]
pub struct Mismatch {
    pub tag: &'static str,
    pub key: String,
    pub a: Value,
    pub b: Value,
}

/// The verdict: `hard_fail` empty ⇒ PARITY: PASS.
#[derive(Debug, Default)]
pub struct Verdict {
    pub hard_fail: Vec<Mismatch>,
    pub timed_note: Vec<Mismatch>,
    /// Deterministic fields that differ but stay within tolerance (1-ULP float / peak_ram 5%) —
    /// not a fail, but surfaced so silent creep toward the tolerance edge stays visible.
    pub within_tol: Vec<Mismatch>,
    pub info: Vec<Mismatch>,
}

impl Verdict {
    pub fn passed(&self) -> bool {
        self.hard_fail.is_empty()
    }
}

/// Which comparison MODE a pair is in — decided ONCE from both sides' strict `passed` bool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Both runs passed: the ranking surface is real on both sides and fully band-gated.
    BothPassed,
    /// At least one run failed: the failing side zeroes/nulls its timing, so a timed field is
    /// waived ONLY when a side is zeroed/nulled. A both-numeric divergence still hard-fails —
    /// the failing-run mode never becomes a blanket ranking-surface amnesty (#66 must-fix 1).
    FailingPair,
}

/// A timing value the failing side is expected to blank: JSON null, or numeric 0.0.
fn is_zeroed_or_null(v: &Value) -> bool {
    v.is_null() || as_f64(v) == Some(0.0)
}

/// The 10%-band gate for a Timed field known to diverge — used for a both-passed pair AND for a
/// genuine both-numeric divergence on a failing pair: equal→ok, within band→note, else HARD FAIL.
fn timed_band_gate(v: &mut Verdict, rk: &str, va: &Value, vb: &Value) {
    match (as_f64(va), as_f64(vb)) {
        (Some(x), Some(y)) if num_close(x, y, 0.0) => {}
        (Some(x), Some(y)) if num_close(x, y, TIMED_BAND) => v.timed_note.push(mm("~", rk, va, vb)),
        _ => v.hard_fail.push(mm("TIMED", rk, va, vb)),
    }
}

/// Compare two score payloads field-by-field. `a` = benchd, `b` = swift.
pub fn diff(a: &Value, b: &Value) -> Verdict {
    let fa: std::collections::BTreeMap<String, Value> = flatten(a).into_iter().collect();
    let fb: std::collections::BTreeMap<String, Value> = flatten(b).into_iter().collect();
    let mut v = Verdict::default();

    let hard =
        |v: &mut Verdict, tag, k: &str, a: &Value, b: &Value| v.hard_fail.push(mm(tag, k, a, b));

    // §F3 / mode: decide the comparison MODE ONCE from both sides' STRICT `passed`. A `passed`
    // that is present-but-not-a-bool (e.g. `1`) is a HARD FAIL, never coerced to false — otherwise
    // a non-bool `passed` on both sides would silently un-gate the ranking surface (#66 must-fix 2).
    // outer None = key absent (SCHEMA-DRIFT-MISSING fires below); inner None = present, not a bool.
    let strict = |x: Option<&Value>| -> Option<Option<bool>> {
        match x {
            None => None,
            Some(Value::Bool(b)) => Some(Some(*b)),
            Some(_) => Some(None),
        }
    };
    let (pa, pb) = (strict(fa.get("passed")), strict(fb.get("passed")));
    if matches!(pa, Some(None)) || matches!(pb, Some(None)) {
        hard(
            &mut v,
            "PASSED-NOT-BOOL",
            "passed",
            fa.get("passed").unwrap_or(&Value::Null),
            fb.get("passed").unwrap_or(&Value::Null),
        );
    }
    // Per-side pass state (a non-bool/missing `passed` already hard-failed above; treat it as
    // not-passed for the mode/waiver logic). Used by the failing-pair timed waiver below.
    let passed_a = matches!(pa, Some(Some(true)));
    let passed_b = matches!(pb, Some(Some(true)));
    let mode = if passed_a && passed_b {
        Mode::BothPassed
    } else {
        Mode::FailingPair
    };

    // Drift gates: unknown key (present but unrostered) and missing roster key.
    let mut keys: Vec<&String> = fa.keys().chain(fb.keys()).collect();
    keys.sort();
    keys.dedup();
    for k in &keys {
        if bucket_of(k).is_none() && !is_waived(k) {
            hard(
                &mut v,
                "UNKNOWN-FIELD",
                k,
                fa.get(*k).unwrap_or(&Value::Null),
                fb.get(*k).unwrap_or(&Value::Null),
            );
        }
    }
    for (rk, _) in ROSTER {
        if !fa.contains_key(*rk) || !fb.contains_key(*rk) {
            hard(
                &mut v,
                "SCHEMA-DRIFT-MISSING",
                rk,
                fa.get(*rk).unwrap_or(&Value::Null),
                fb.get(*rk).unwrap_or(&Value::Null),
            );
        }
    }

    for (rk, bucket) in ROSTER {
        let (Some(va), Some(vb)) = (fa.get(*rk), fb.get(*rk)) else {
            continue; // missing already hard-failed above
        };
        if va == vb {
            continue;
        }
        match bucket {
            Env => v.info.push(mm("info", rk, va, vb)),
            Error => {
                if error_class(va) != error_class(vb) {
                    hard(&mut v, "ERROR-CLASS", rk, va, vb);
                } else {
                    v.info.push(mm("info", rk, va, vb)); // same class, different wording
                }
            }
            Timed => match mode {
                // Both passed → the ranking surface is real; band-gate it.
                Mode::BothPassed => timed_band_gate(&mut v, rk, va, vb),
                // Failing pair → waive a timed field ONLY when the divergence is fully explained
                // by a FAILED side blanking it: for EACH side, failed ⇒ zeroed/null (a passed side
                // keeps its real value). If a FAILED side reports a real non-zeroed timing (or both
                // failed and one blanks while the other keeps a residual), that is a genuine
                // producer divergence on the failure surface and is band-gated — never waived
                // (Fable checkpoint Finding 1; the deterministic + failing-run surface always
                // gates). A passing side's real value paired with a failed side's zero/null is the
                // legitimate F3 artifact; null `score` on a failed side is waived here too.
                Mode::FailingPair => {
                    let side_ok = |passed: bool, x: &Value| passed || is_zeroed_or_null(x);
                    if side_ok(passed_a, va) && side_ok(passed_b, vb) {
                        v.info.push(mm("info(F3)", rk, va, vb));
                    } else {
                        timed_band_gate(&mut v, rk, va, vb);
                    }
                }
            },
            DetTol => match (as_f64(va), as_f64(vb)) {
                (Some(x), Some(y)) if num_close(x, y, PEAK_RAM_REL_TOL) => {
                    v.within_tol.push(mm("~tol", rk, va, vb)) // within 5% but not equal — creep watch
                }
                _ => hard(&mut v, "DET-TOL", rk, va, vb),
            },
            Det | Failing => {
                let tag = if *bucket == Det { "DET" } else { "FAILING" };
                match (as_f64(va), as_f64(vb)) {
                    (Some(x), Some(y)) if num_close(x, y, DET_FLOAT_TOL) => {
                        v.within_tol.push(mm("~ulp", rk, va, vb)) // 1-ULP float drift — creep watch
                    }
                    _ => hard(&mut v, tag, rk, va, vb),
                }
            }
        }
    }
    v
}

fn mm(tag: &'static str, k: &str, a: &Value, b: &Value) -> Mismatch {
    Mismatch {
        tag,
        key: k.to_string(),
        a: a.clone(),
        b: b.clone(),
    }
}

const PARITY_USAGE: &str = "\
benchd parity-diff — field-by-field parity verdict on two score payloads

USAGE:
    benchd parity-diff <benchd.json> <swift.json>
    benchd parity-diff --version        print the differ version (for milestone pinning, #70)
    benchd parity-diff --emit-sample     print one complete, schema-current ScorePayload

EXIT: 0 = PARITY PASS · 1 = PARITY FAIL · 2 = usage · 3 = IO/parse error
The differ's self-test is `cargo test -p benchd parity` (roster↔ScoreMetrics
exhaustiveness + the mutate-every-field property test).
";

/// Behavioral version of the differ, bumped on a logic change. v2 = failing-pair MODE + strict
/// `passed` + Env-ledger row binding + off-schema-error classing (the #66/Fable-checkpoint era).
const PARITY_DIFF_VERSION: &str = "2";

/// A stable identity string for the differ, recorded in every declaration (T5/#70). Combines the
/// manual behavioral version with a fingerprint of the differ's behavioral surface — the roster
/// (each key+bucket, sorted) and the three tolerances — so a bucket/field/tolerance change
/// auto-bumps the fingerprint even if `PARITY_DIFF_VERSION` is not bumped. Hash is FNV-1a (stable,
/// no external dep), 32-bit hex.
pub fn version_string() -> String {
    let mut items: Vec<String> = ROSTER.iter().map(|(k, b)| format!("{k}:{b:?}")).collect();
    items.sort();
    let canonical = format!(
        "{}|det={DET_FLOAT_TOL:e}|ram={PEAK_RAM_REL_TOL:e}|band={TIMED_BAND:e}",
        items.join(",")
    );
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in canonical.bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!(
        "parity-diff v{PARITY_DIFF_VERSION} roster{}/{:08x}",
        ROSTER.len(),
        h & 0xffff_ffff
    )
}

/// `benchd parity-diff <a.json> <b.json>`: print the report and exit 0 (PASS) / 1 (FAIL) /
/// 2 (usage) / 3 (IO). Matches `scripts/parity-diff.py`'s verdict + native benchd exit codes.
pub fn run(args: &[String]) -> ExitCode {
    if args.len() == 1 {
        match args[0].as_str() {
            "-h" | "--help" => {
                print!("{PARITY_USAGE}");
                return ExitCode::SUCCESS;
            }
            "--version" => {
                println!("{}", version_string());
                return ExitCode::SUCCESS;
            }
            "--emit-sample" => {
                // A complete, schema-CURRENT ScorePayload (default metrics), for the driver's
                // GPU-free differ self-tests (T3). Being ScorePayload::default() serialized, it
                // never drifts from the real schema — a new field appears here automatically.
                let sample = crate::score::ScorePayload {
                    score: Some(2.88),
                    passed: true,
                    metrics: crate::score::ScoreMetrics::default(),
                };
                return match serde_json::to_string(&sample) {
                    Ok(s) => {
                        println!("{s}");
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("benchd parity-diff --emit-sample: {e}");
                        ExitCode::from(3)
                    }
                };
            }
            _ => {}
        }
    }
    if args.len() != 2 {
        eprint!("{PARITY_USAGE}");
        return ExitCode::from(2);
    }
    let load = |p: &str| -> Result<Value, ExitCode> {
        let bytes = std::fs::read(Path::new(p)).map_err(|e| {
            eprintln!("benchd parity-diff: IO error reading {p}: {e}");
            ExitCode::from(3)
        })?;
        serde_json::from_slice(&bytes).map_err(|e| {
            eprintln!("benchd parity-diff: {p} is not valid JSON: {e}");
            ExitCode::from(3)
        })
    };
    let a = match load(&args[0]) {
        Ok(v) => v,
        Err(c) => return c,
    };
    let b = match load(&args[1]) {
        Ok(v) => v,
        Err(c) => return c,
    };
    let v = diff(&a, &b);
    println!("=== HARD FAIL (deterministic / ranking / drift) ===");
    for m in &v.hard_fail {
        println!(
            "  FAIL [{}] {}: benchd={} vs swift={}",
            m.tag, m.key, m.a, m.b
        );
    }
    if v.hard_fail.is_empty() {
        println!("  (none)");
    }
    println!("\n=== TIMED within 10% (load noise) ===");
    for m in &v.timed_note {
        println!("  ~ {}: benchd={} vs swift={}", m.key, m.a, m.b);
    }
    println!("\n=== WITHIN TOLERANCE (creep watch: 1-ULP float / peak_ram 5%) ===");
    for m in &v.within_tol {
        println!("  {} {}: benchd={} vs swift={}", m.tag, m.key, m.a, m.b);
    }
    if v.within_tol.is_empty() {
        println!("  (none)");
    }
    println!("\n=== INFORMATIONAL (environmental / same-class error / F3-zeroed) ===");
    for m in &v.info {
        println!("  info {}: benchd={} vs swift={}", m.key, m.a, m.b);
    }
    println!(
        "\nPARITY: {}",
        if v.passed() {
            "PASS (no deterministic/ranking mismatch)"
        } else {
            "FAIL"
        }
    );
    if v.passed() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::ScoreMetrics;

    #[test]
    fn roster_covers_score_metrics_exactly() {
        // §T1 exhaustiveness: the roster's `metrics.*` keys equal ScoreMetrics's serde field
        // names, both directions — a new field that isn't bucketed fails HERE, not a live window.
        let v = serde_json::to_value(ScoreMetrics::default()).unwrap();
        let serde_keys: std::collections::BTreeSet<String> =
            v.as_object().unwrap().keys().cloned().collect();
        let roster_metrics: std::collections::BTreeSet<String> = ROSTER
            .iter()
            .filter_map(|(k, _)| k.strip_prefix("metrics.").map(str::to_string))
            .collect();
        let missing: Vec<_> = serde_keys.difference(&roster_metrics).collect();
        let phantom: Vec<_> = roster_metrics.difference(&serde_keys).collect();
        assert!(
            missing.is_empty(),
            "ScoreMetrics fields not bucketed (add to ROSTER): {missing:?}"
        );
        assert!(
            phantom.is_empty(),
            "ROSTER keys that are not ScoreMetrics fields (remove): {phantom:?}"
        );
        // Top-level keys are rostered too.
        assert_eq!(bucket_of("score"), Some(Timed));
        assert_eq!(bucket_of("passed"), Some(Det));

        // §T1 top-level pin (#66 must-fix 6): ScorePayload's top-level serde keys must be exactly
        // `metrics` (the nested container) + the rostered top-level keys. A 3rd top-level field
        // would otherwise pass this test and detonate as UNKNOWN-FIELD in a live window.
        let payload = serde_json::to_value(crate::score::ScorePayload {
            score: Some(0.0),
            passed: true,
            metrics: ScoreMetrics::default(),
        })
        .unwrap();
        let top_keys: std::collections::BTreeSet<String> =
            payload.as_object().unwrap().keys().cloned().collect();
        let roster_top: std::collections::BTreeSet<String> = ROSTER
            .iter()
            .filter(|(k, _)| !k.contains('.'))
            .map(|(k, _)| k.to_string())
            .collect();
        let non_metrics_top: std::collections::BTreeSet<String> = top_keys
            .iter()
            .filter(|k| *k != "metrics")
            .cloned()
            .collect();
        assert_eq!(
            non_metrics_top, roster_top,
            "top-level ScorePayload keys drifted from the roster's top-level keys"
        );
        assert!(
            top_keys.contains("metrics"),
            "ScorePayload must carry a `metrics` container"
        );
    }

    /// The ALLOWLIST of `metrics.*` keys that are deliberately UNROSTERED: `per_prompt`, the
    /// speculative-decode seal, the engine identity (backend/device/protocol version, the
    /// loaded-head digest, the runner identity, and the resident-process identity), and the
    /// PAIRED-BASELINE seal (which box, which calibration bytes, which reference commit, and the
    /// two legs' measured pairs), and the timed token tolerance with its counts. All are benchd-only additions the SWIFT
    /// reference never emits, and all are omitted-when-unset — rostering a key that is absent on
    /// BOTH sides would hard-fail every pair as SCHEMA-DRIFT-MISSING, which is why they are
    /// unrostered rather than bucketed.
    const UNROSTERED_ADDITIVE_METRICS: &[&str] = &[
        "acceptance_lengths",
        "baseline_band_passed",
        "baseline_box",
        "baseline_calibration_sha256",
        "baseline_golden_sha256",
        "baseline_leg_decode_seconds_per_token",
        "baseline_leg_decode_window_seconds_per_token",
        "baseline_leg_prefill_seconds_per_token",
        "baseline_leg_seed_prefill_window_seconds_per_token",
        "baseline_reference_commit",
        "baseline_source",
        "candidate_leg_decode_seconds_per_token",
        "candidate_leg_decode_window_seconds_per_token",
        "candidate_leg_prefill_seconds_per_token",
        "candidate_leg_seed_prefill_window_seconds_per_token",
        "contract_sha256",
        "contract_sources",
        "effective_spec_depth",
        "effective_spec_mode",
        "engine_backend",
        "engine_device",
        "engine_protocol_version",
        "gates",
        "head_provenance_sha256",
        "live_prompt",
        "local_phases",
        "paired_legs",
        "per_prompt",
        "resident_load_epoch",
        "resident_pid",
        "runner_build",
        "runner_id",
        "runner_manifest_sha256",
        "runner_model_type",
        "spec_acceptance_rate",
        "spec_accepted_total",
        "spec_drafted_total",
        "spec_rectangular_verification_rounds",
        "spec_rounds",
        "spec_serial_verification_rounds",
        "spec_verification_mode",
        "spec_verify_replay_disagreements",
        "timed_token_near_tie_relative_gap",
        "timed_token_tolerance_per_thousand",
        "token_mismatch_count",
        "token_mismatch_first_step",
        "token_mismatch_near_tie_count",
        "token_mismatch_second_choice_count",
        "token_mismatch_second_choice_max_relative_gap",
    ];

    /// §T1 EXHAUSTIVENESS, second half. `roster_covers_score_metrics_exactly` compares the roster
    /// against `ScoreMetrics::default()`, which OMITS any `skip_serializing_if` field — so an
    /// omitted-when-empty field can enter the schema unbucketed. This test pins that the ones which
    /// do are EXACTLY [`UNROSTERED_ADDITIVE_METRICS`]: a twentieth unrostered field fails here.
    #[test]
    fn only_the_sanctioned_additive_keys_are_unrostered() {
        let populated = ScoreMetrics {
            local_phases: Some(crate::score::LocalPhases::default()),
            per_prompt: vec![crate::score::ScorePerPrompt::default()],
            effective_spec_mode: Some("mtp".to_string()),
            effective_spec_depth: Some(1),
            spec_rounds: Some(1),
            spec_drafted_total: Some(1),
            spec_accepted_total: Some(1),
            spec_acceptance_rate: Some(1.0),
            spec_verify_replay_disagreements: Some(1),
            spec_verification_mode: Some("rectangular".to_string()),
            spec_rectangular_verification_rounds: Some(1),
            spec_serial_verification_rounds: Some(0),
            acceptance_lengths: crate::score::AcceptanceLengthsSummary::of(&[1]),
            engine_backend: Some("b".to_string()),
            engine_device: Some("d".to_string()),
            engine_protocol_version: Some(1),
            head_provenance_sha256: Some("h".to_string()),
            runner_id: Some("layr/qwen4exp-125b-a6b".to_string()),
            runner_model_type: Some("qwen4_exp".to_string()),
            runner_manifest_sha256: Some("0".repeat(64)),
            runner_build: Some("c4089870".to_string()),
            resident_pid: Some(4242),
            resident_load_epoch: Some(1_756_944_000),
            baseline_source: Some("serial-control-leg".to_string()),
            baseline_box: Some("m5-max-128gb-4-qwen38-125b-a6b-mlx".to_string()),
            contract_sha256: Some("a".repeat(64)),
            contract_sources: Some(bench_core::contract::ContractSources::ALL_TABLE),
            baseline_calibration_sha256: Some("c".repeat(64)),
            baseline_golden_sha256: Some("e".repeat(64)),
            live_prompt: Some("botany".to_string()),
            baseline_reference_commit: Some("d".repeat(40)),
            baseline_band_passed: Some(true),
            baseline_leg_prefill_seconds_per_token: Some(0.0006),
            baseline_leg_decode_seconds_per_token: Some(0.032),
            candidate_leg_prefill_seconds_per_token: Some(0.0006),
            candidate_leg_decode_seconds_per_token: Some(0.016),
            baseline_leg_decode_window_seconds_per_token: Some(0.031),
            candidate_leg_decode_window_seconds_per_token: Some(0.0113),
            baseline_leg_seed_prefill_window_seconds_per_token: Some(0.0006),
            candidate_leg_seed_prefill_window_seconds_per_token: Some(0.0004),
            gates: vec![crate::quiescegate::GateRecord::default()],
            paired_legs: vec![crate::score::PairedLegRecord {
                prompt_sha256: "f".repeat(64),
                pair: 1,
                control_prefill_seconds_per_token: 0.0006,
                control_decode_seconds_per_token: 0.032,
                candidate_prefill_seconds_per_token: 0.0006,
                candidate_decode_seconds_per_token: 0.016,
                control_seed_prefill_window_seconds_per_token: None,
                control_decode_window_seconds_per_token: None,
                candidate_seed_prefill_window_seconds_per_token: None,
                candidate_decode_window_seconds_per_token: None,
                token_mismatch_count: Some(0),
                token_mismatch_first_step: None,
                token_mismatch_near_tie_count: Some(0),
                token_mismatch_second_choice_count: Some(0),
                token_mismatch_second_choice_max_relative_gap: None,
            }],
            timed_token_tolerance_per_thousand: Some(100),
            timed_token_near_tie_relative_gap: Some(0.05),
            token_mismatch_count: Some(3),
            token_mismatch_first_step: Some(17),
            token_mismatch_near_tie_count: Some(2),
            token_mismatch_second_choice_count: Some(2),
            token_mismatch_second_choice_max_relative_gap: Some(0.0125),
            ..Default::default()
        };
        let v = serde_json::to_value(&populated).unwrap();
        let serde_keys: std::collections::BTreeSet<String> =
            v.as_object().unwrap().keys().cloned().collect();
        let roster_metrics: std::collections::BTreeSet<String> = ROSTER
            .iter()
            .filter_map(|(k, _)| k.strip_prefix("metrics.").map(str::to_string))
            .collect();
        let unrostered: Vec<_> = serde_keys.difference(&roster_metrics).collect();
        assert_eq!(
            unrostered, UNROSTERED_ADDITIVE_METRICS,
            "a ScoreMetrics field is neither bucketed nor on the sanctioned additive allowlist"
        );
    }

    #[test]
    fn roster_keys_are_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for (k, _) in ROSTER {
            assert!(seen.insert(*k), "duplicate roster key: {k}");
        }
    }

    fn base() -> Value {
        serde_json::to_value(crate::score::ScorePayload {
            score: Some(2.88),
            passed: true,
            metrics: ScoreMetrics {
                decode_seconds_per_token: 0.0686,
                prefill_seconds_per_token: 0.00113,
                baseline_decode_seconds_per_token: 0.2,
                baseline_prefill_seconds_per_token: 0.01,
                passed_correctness: true,
                golden_hash: "abc".into(),
                bandwidth_source: "ram_resident_model".into(),
                num_layers: 64,
                case_count: 1,
                checked_steps: 18,
                weights_hash: "wh".into(),
                weights_byte_count: 100,
                weights_file_count: 3,
                peak_ram_gb: 20.0,
                runtime: "rust-local-iterate".into(),
                ..Default::default()
            },
        })
        .unwrap()
    }

    #[test]
    fn identical_pair_passes() {
        assert!(diff(&base(), &base()).passed());
    }

    /// A value guaranteed different from `base_val` AND far enough to trip every gate
    /// (out of any band/tolerance; flips bools; a distinct non-`:`-leading string).
    fn mutated_value(base_val: &Value) -> Value {
        match base_val {
            Value::Null => serde_json::json!(123_456),
            Value::Bool(x) => serde_json::json!(!x),
            Value::Number(n) => {
                let x = n.as_f64().unwrap_or(0.0);
                serde_json::json!(x + x.abs().max(1.0) + 1.0)
            }
            Value::String(s) => serde_json::json!(format!("MUT_{s}")),
            _ => serde_json::json!("MUT"),
        }
    }

    fn set_flat(v: &mut Value, key: &str, new: Value) {
        match key.split_once('.') {
            Some((parent, child)) => v[parent][child] = new,
            None => v[key] = new,
        }
    }

    #[test]
    fn mutate_every_field_flips_verdict_unless_waived() {
        // §T2 killer property: from a passing pair, mutating EACH field in turn must flip the
        // verdict to FAIL — UNLESS the field is waived (Env). "No silently ignorable field",
        // proven by construction over the whole ScoreMetrics surface, not by review.
        let base = base();
        let flat: std::collections::BTreeMap<String, Value> = flatten(&base).into_iter().collect();
        for (key, bucket) in ROSTER {
            let base_val = flat.get(*key).cloned().unwrap_or(Value::Null);
            let mut b = base.clone();
            set_flat(&mut b, key, mutated_value(&base_val));
            let flipped = !diff(&base, &b).passed();
            if matches!(bucket, Env) {
                assert!(
                    !flipped,
                    "waived (Env) field {key} must NOT flip the verdict when mutated"
                );
            } else {
                assert!(
                    flipped,
                    "field {key} ({bucket:?}) MUST flip the verdict when mutated — \
                     no silently ignorable field"
                );
            }
        }
    }

    #[test]
    fn env_and_timed_band_and_peak_tol_pass() {
        let mut b = base();
        b["metrics"]["runtime"] = "swift-local-iterate".into();
        b["metrics"]["commit"] = "".into();
        b["metrics"]["decode_seconds_per_token"] = serde_json::json!(0.0686 * 1.05); // in band
        b["metrics"]["peak_ram_gb"] = serde_json::json!(20.5); // within 5%
        b["metrics"]["baseline_decode_seconds_per_token"] = serde_json::json!(0.2 + 1e-12); // 1-ULP
        assert!(diff(&base(), &b).passed());
    }

    #[test]
    fn parity_diff_tolerates_empty_harness_hash() {
        // HARD CONSTRAINT 1 guard — the David-signed EXPECT_DIFFER waiver (§13 ledger, ROSTER →
        // `Env`) makes an empty-vs-real harness_hash `info`, never a hard fail. The overlay's
        // gates/seam-1 harness_hash refusal lives on a DISTINCT code path
        // (`overlay::validate_gates`) and must NOT bleed into this comparison.
        //
        // F1 UPDATE — the premise this test was written under is GONE: benchd no longer emits an
        // empty harness_hash (it computes the real one, `iterate::HarnessIdentity`), and over the
        // same workspace it computes the SAME digest the Swift harness does. The waiver is
        // deliberately left in place regardless: it only ever DOWNGRADES a difference to `info`, so
        // it cannot mask a regression this differ would otherwise catch, and re-bucketing an
        // operator-signed ledger row is a separate, signed act — not a side effect of F1. The
        // fixture below keeps exercising the waiver's tolerance directly rather than through a
        // producer that no longer produces the empty value.
        assert_eq!(
            bucket_of("metrics.harness_hash"),
            Some(Env),
            "harness_hash stays waived (Env)"
        );
        let mut a = base(); // benchd side — emits an empty harness identity
        a["metrics"]["harness_hash"] = "".into();
        let mut b = base(); // swift side — carries a real 64-hex harness identity
        b["metrics"]["harness_hash"] = "a".repeat(64).into();
        assert!(
            diff(&a, &b).passed(),
            "empty-vs-real harness_hash must be waived (EXPECT_DIFFER), not a parity FAIL"
        );
    }

    #[test]
    fn det_mismatch_and_unknown_and_missing_hard_fail() {
        // deterministic mismatch
        let mut b = base();
        b["metrics"]["case_count"] = serde_json::json!(2);
        assert!(!diff(&base(), &b).passed());
        // weights_hash is deterministic (P2)
        let mut b = base();
        b["metrics"]["weights_hash"] = "OTHER".into();
        assert!(!diff(&base(), &b).passed());
        // unknown field
        let mut b = base();
        b["metrics"]["brand_new_field"] = serde_json::json!(1);
        assert!(!diff(&base(), &b).passed());
        // missing field
        let mut b = base();
        b["metrics"].as_object_mut().unwrap().remove("num_layers");
        assert!(!diff(&base(), &b).passed());
    }

    #[test]
    fn timed_out_of_band_hard_fails_when_both_passed() {
        let mut b = base();
        b["metrics"]["decode_seconds_per_token"] = serde_json::json!(0.0686 * 1.5);
        assert!(!diff(&base(), &b).passed());
    }

    #[test]
    fn f3_failing_pair_does_not_hard_fail_on_zeroed_timing() {
        // benchd FAILED (passed=false, timing zeroed); swift PASSED. §F3: the timing
        // divergence is expected — `passed` (DET) is the real mismatch and hard-fails.
        let mut a = base();
        a["passed"] = serde_json::json!(false);
        a["metrics"]["decode_seconds_per_token"] = serde_json::json!(0.0);
        a["metrics"]["prefill_seconds_per_token"] = serde_json::json!(0.0);
        a["metrics"]["decode_speedup"] = serde_json::json!(0.0);
        a["metrics"]["prefill_speedup"] = serde_json::json!(0.0);
        let v = diff(&a, &base());
        // The verdict FAILs — but only on `passed` (DET), not the zeroed timing fields.
        assert!(!v.passed());
        assert!(v.hard_fail.iter().any(|m| m.key == "passed"));
        assert!(
            !v.hard_fail.iter().any(|m| m.tag == "TIMED"),
            "zeroed timing on a failing side must not hard-fail (F3)"
        );
    }

    #[test]
    fn error_semantics_same_class_passes_diff_class_fails() {
        let mut a = base();
        let mut b = base();
        a["metrics"]["error"] = "correctness gate failed: case p1 step 3".into();
        b["metrics"]["error"] = "correctness gate failed: local-iterate step 4".into();
        // same class before ':' → not a hard fail
        assert!(!diff(&a, &b)
            .hard_fail
            .iter()
            .any(|m| m.key == "metrics.error"));
        b["metrics"]["error"] = "timing barrier: completed_work mismatch".into();
        assert!(diff(&a, &b)
            .hard_fail
            .iter()
            .any(|m| m.tag == "ERROR-CLASS"));
    }

    #[test]
    fn null_score_on_failing_pair_is_waived_not_timed_failed() {
        // #66 must-fix 3: benchd FAILED with `score = null` (the one nullable timed field);
        // swift PASSED with a real score. The null score is the failing-side zeroing artifact and
        // must be WAIVED (info F3) — the verdict fails only on `passed`, never on a null-vs-number
        // timed hard fail. (The pre-fix code let null score fall into the TIMED catch-all.)
        let mut a = base();
        a["passed"] = serde_json::json!(false);
        a["score"] = Value::Null;
        for f in [
            "decode_seconds_per_token",
            "prefill_seconds_per_token",
            "decode_speedup",
            "prefill_speedup",
        ] {
            a["metrics"][f] = serde_json::json!(0.0);
        }
        let v = diff(&a, &base());
        assert!(!v.passed());
        assert!(v.hard_fail.iter().any(|m| m.key == "passed"));
        assert!(
            !v.hard_fail
                .iter()
                .any(|m| m.key == "score" || m.tag == "TIMED"),
            "null score / zeroed timing on the failing side must be waived, not a TIMED hard fail"
        );
    }

    #[test]
    fn nonbool_passed_hard_fails_even_when_both_equal() {
        // #66 must-fix 2: `passed: 1` on BOTH sides is EQUAL, so the DET arm sees no diff — but a
        // non-bool `passed` must not be coerced to false and silently un-gate the ranking surface.
        let mut a = base();
        let mut b = base();
        a["passed"] = serde_json::json!(1);
        b["passed"] = serde_json::json!(1);
        let v = diff(&a, &b);
        assert!(!v.passed());
        assert!(v.hard_fail.iter().any(|m| m.tag == "PASSED-NOT-BOOL"));
    }

    #[test]
    fn both_numeric_timed_divergence_on_failing_pair_hard_fails() {
        // #66 must-fix 1: the hole the review caught. benchd FAILED but still reported a REAL,
        // non-zero decode timing that diverges from swift's real timing (0.0686 vs 0.5). This is
        // NOT the zeroing artifact (both sides numeric & non-zero), so the failing-pair mode must
        // STILL hard-fail it — the mode is not a blanket ranking-surface amnesty.
        let mut a = base();
        a["passed"] = serde_json::json!(false); // failing pair
        a["metrics"]["decode_seconds_per_token"] = serde_json::json!(0.0686);
        let mut b = base();
        b["metrics"]["decode_seconds_per_token"] = serde_json::json!(0.5);
        let v = diff(&a, &b);
        assert!(!v.passed());
        assert!(
            v.hard_fail
                .iter()
                .any(|m| m.tag == "TIMED" && m.key == "metrics.decode_seconds_per_token"),
            "a both-numeric timed divergence on a failing pair must hard-fail, not be F3-waived"
        );
    }

    #[test]
    fn within_tolerance_drift_is_reported_not_silent() {
        // fix-or-file (#66): restore creep visibility — a 1-ULP deterministic drift and a
        // within-5% peak_ram drift PASS the verdict but must be surfaced (within_tol), so slow
        // creep toward a tolerance edge stays visible instead of vanishing.
        let mut b = base();
        b["metrics"]["baseline_decode_seconds_per_token"] = serde_json::json!(0.2 + 1e-12);
        b["metrics"]["peak_ram_gb"] = serde_json::json!(20.5); // within 5%
        let v = diff(&base(), &b);
        assert!(v.passed());
        assert!(v
            .within_tol
            .iter()
            .any(|m| m.key == "metrics.baseline_decode_seconds_per_token"));
        assert!(v.within_tol.iter().any(|m| m.key == "metrics.peak_ram_gb"));
    }

    /// The set of fields carrying `signed_off: true` in the structured waiver ledger.
    /// Reads the STRUCTURED ledger fixture (not a Markdown doc): a malformed or missing file,
    /// or a row whose `signed_off` is absent/false, drops out of this set and turns the tests
    /// that depend on it RED.
    fn signed_waiver_fields() -> std::collections::BTreeSet<String> {
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/waiver-ledger.json"
        ))
        .expect("read tests/fixtures/waiver-ledger.json");
        let ledger: Value = serde_json::from_str(&raw).expect("waiver-ledger.json is valid JSON");
        ledger["waivers"]
            .as_array()
            .expect("waiver-ledger.json has a `waivers` array")
            .iter()
            .filter(|row| row["signed_off"].as_bool() == Some(true))
            .map(|row| {
                row["field"]
                    .as_str()
                    .expect("each waiver row has a string `field`")
                    .to_string()
            })
            .collect()
    }

    #[test]
    fn env_entries_are_signed_off_in_the_waiver_ledger() {
        // fix-or-file (#66): mis-bucketing a ranking field as Env satisfies every other test (Env
        // is waived). Bind Env membership to David's recorded ledger — every Env-bucketed field
        // must carry a signed row in the STRUCTURED waiver ledger
        // (tests/fixtures/waiver-ledger.json). Re-bucketing a field to Env now REQUIRES a visible,
        // reviewed ledger line; no silent escape hatch. (A test reading a Markdown doc for this
        // was the anti-pattern this migration removed — the data is structured now.)
        let signed = signed_waiver_fields();
        for (k, bucket) in ROSTER {
            if matches!(bucket, Env) {
                let field = k.strip_prefix("metrics.").unwrap_or(k);
                assert!(
                    signed.contains(field),
                    "Env field `{field}` has no `signed_off: true` row in the waiver ledger \
                     (tests/fixtures/waiver-ledger.json) — an Env bucket without a recorded waiver \
                     row is a silent ranking-surface escape hatch"
                );
            }
        }
    }

    #[test]
    fn structured_waiver_ledger_covers_exactly_the_env_roster() {
        // Revert-proof for the .md -> .json migration. The structured ledger the differ's Env
        // bucket depends on must load AND resolve to exactly the Env-bucketed ROSTER field set —
        // no missing waiver (would let an Env field lose its recorded sign-off) and no phantom
        // waiver (would sign off a field nothing waives). Goes RED if the fixture is
        // malformed/missing (load panics) or the reader/roster drifts (set mismatch).
        let signed = signed_waiver_fields();
        // A known key resolves: harness_hash is Env-bucketed and must stay signed.
        assert!(
            signed.contains("harness_hash"),
            "known waiver key `harness_hash` did not resolve from the structured ledger"
        );
        let env_roster: std::collections::BTreeSet<String> = ROSTER
            .iter()
            .filter(|(_, b)| matches!(b, Env))
            .map(|(k, _)| k.strip_prefix("metrics.").unwrap_or(k).to_string())
            .collect();
        assert_eq!(
            signed, env_roster,
            "structured waiver ledger must sign off exactly the Env-bucketed ROSTER fields \
             (left = ledger, right = ROSTER Env)"
        );
    }

    #[test]
    fn both_failed_one_blanks_one_keeps_real_timing_hard_fails() {
        // Fable checkpoint Finding 1: BOTH sides failed; benchd blanked decode to 0.0 but swift
        // left a real residual 0.4321. The old waiver (either-side-zeroed) demoted this to info →
        // false PASS. The failed side that did NOT blank (swift) breaks `failed ⇒ zeroed`, so this
        // producer divergence on the failure surface must HARD-FAIL.
        let mut a = base();
        a["passed"] = serde_json::json!(false);
        a["metrics"]["decode_seconds_per_token"] = serde_json::json!(0.0); // benchd blanked
        let mut b = base();
        b["passed"] = serde_json::json!(false);
        b["metrics"]["decode_seconds_per_token"] = serde_json::json!(0.4321); // swift kept a residual
        let v = diff(&a, &b);
        assert!(!v.passed());
        assert!(
            v.hard_fail
                .iter()
                .any(|m| m.tag == "TIMED" && m.key == "metrics.decode_seconds_per_token"),
            "a failed side that keeps a real (non-zeroed) timing must hard-fail, not be F3-waived"
        );
    }

    #[test]
    fn passed_side_real_vs_failed_side_zero_is_still_waived() {
        // The legitimate F3 artifact must still be waived: swift PASSED with a real decode, benchd
        // FAILED and blanked it. `failed ⇒ zeroed` holds for both sides → waive; fail only on passed.
        let mut a = base();
        a["passed"] = serde_json::json!(false);
        a["metrics"]["decode_seconds_per_token"] = serde_json::json!(0.0);
        let v = diff(&a, &base()); // base passed=true, decode=0.0686
        assert!(
            v.hard_fail.iter().all(|m| m.tag != "TIMED"),
            "legit F3 zeroing must stay waived"
        );
        assert!(v.hard_fail.iter().any(|m| m.key == "passed"));
    }

    #[test]
    fn version_string_fingerprints_the_roster_and_is_stable() {
        let v = version_string();
        assert!(
            v.starts_with("parity-diff v2 roster"),
            "unexpected version: {v}"
        );
        assert!(v.contains(&format!("roster{}/", ROSTER.len())));
        assert_eq!(v, version_string(), "version must be deterministic");
    }

    #[test]
    fn emit_sample_is_a_complete_payload_that_self_passes() {
        // The sample the driver's T3 self-tests use: ScorePayload::default() is schema-complete,
        // so diffing it against itself PASSES with zero drift (proving the differ + schema agree).
        let sample = serde_json::to_value(crate::score::ScorePayload {
            score: Some(2.88),
            passed: true,
            metrics: ScoreMetrics::default(),
        })
        .unwrap();
        assert!(
            diff(&sample, &sample).passed(),
            "a self-identical sample must PASS"
        );
    }

    /// THE GATE LOG NEVER BREAKS PARITY (David 2026-09-17). Its waits, loads, utilizations and
    /// temperatures are facts about one box on one day, so two honest producers never agree on
    /// them — and the reference emits no such key at all. The differ waives the key outright: a
    /// benchd payload that carries it, against a reference payload that does not, still PASSES,
    /// and two payloads carrying DIFFERENT gate logs still PASS. Nothing in the verdict names it.
    #[test]
    fn the_gate_log_is_waived_by_the_differ() {
        assert!(is_waived("metrics.gates"));
        assert_eq!(bucket_of("metrics.gates"), None, "waived, never bucketed");

        let gate_point = |waited: u64| {
            serde_json::json!([{
                "pair": 1,
                "leg": "control",
                "phase": "decode",
                "quiescence": {"state": "passed", "waited_seconds": waited, "load": 0.8, "gpu_util": 0.02},
                "cool": {"state": "passed", "waited_seconds": 0, "gpu_temp_c": 38.5},
            }])
        };

        // Present on the benchd side only — the reference has no such key.
        let mut a = base();
        let b = base();
        a["metrics"]["gates"] = gate_point(45);
        let v = diff(&a, &b);
        assert!(
            v.passed(),
            "gate log must not fail parity: {:?}",
            v.hard_fail
        );
        assert!(
            !v.hard_fail
                .iter()
                .chain(&v.info)
                .any(|m| m.key == "metrics.gates"),
            "the verdict must not name the waived key"
        );

        // Present on BOTH sides with different waits — still PASS.
        let mut a = base();
        let mut b = base();
        a["metrics"]["gates"] = gate_point(45);
        b["metrics"]["gates"] = gate_point(900);
        assert!(
            diff(&a, &b).passed(),
            "two honest gate logs differ by design"
        );
    }

    #[test]
    fn nonstring_error_values_do_not_collapse_to_pass() {
        // Fable checkpoint Finding 3b: off-schema numeric `error` values must NOT both map to class
        // "" (which compared equal → PASS). They get distinct classes and hard-fail.
        let mut a = base();
        let mut b = base();
        a["metrics"]["error"] = serde_json::json!(5);
        b["metrics"]["error"] = serde_json::json!(7);
        let v = diff(&a, &b);
        assert!(v
            .hard_fail
            .iter()
            .any(|m| m.tag == "ERROR-CLASS" && m.key == "metrics.error"));
    }
}
