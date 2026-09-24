//! Box QUIESCENCE gate — the gate that runs immediately BEFORE the cool gate
//! ([`crate::coolgate`]) on every timed measurement. The cool gate answers one question, "is the
//! GPU cold enough"; this gate answers the other one the original ranked workflow asked, "is the
//! box idle". Both must pass before a timer starts (David, 2026-09-17).
//!
//! The semantics are the original workflow step "Wait for quiescence before timing"
//! (`mlxfast-challenge-dev/.github/workflows/benchmark.yml`, the ranked benchmark workflow), which
//! the 2026-09-05 engine-workflow change dropped on the claim that the cool gate covered it. It
//! did not: the cool gate reads temperature only, so a box with a busy CPU or another process on
//! the GPU could still open a timed window. The step is restored here, inside benchd, so BOTH the
//! calibration passes and the ranked/server benchmark run behind it.
//!
//! Fixed constants, exactly like the cool gate's fixed temperature target:
//! - 1-minute load average below [`LOAD_MAX`] (2.0),
//! - GPU utilization below [`GPU_UTIL_MAX`] (0.10, a fraction),
//! - one sample every [`POLL_SECONDS`] (15 s),
//! - give up after [`MAX_WAIT_SECONDS`] (900 s) with the [`QUIESCENCE_TIMEOUT`] refusal.
//!
//! Readers, resolved natively per host OS so no env var is required:
//! - load: macOS `sysctl -n vm.loadavg` (the first number of `{ l1 l5 l15 }`), Linux
//!   `/proc/loadavg` (the first field);
//! - GPU utilization: macOS `macmon pipe -s1` → `gpu_usage[1]` (the SAME macmon binary the cool
//!   gate discovers), Linux `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits`
//!   (a percent, divided by 100).
//!
//! A DROPPED or unparsable sample counts as BUSY and the gate keeps waiting; it never counts as
//! idle. That is the original step's rule. A reader that does not resolve at all is a different
//! thing from a dropped sample: as with the cool gate, the gate is SKIPPED (warned, never failed),
//! and the ranked path turns that skip into a refusal of its own.
//!
//! The two gates share the one existing switch: `MLXFAST_LOCAL_COOL_GATE=0` and the local-mode
//! rule turn both off together. There is no second env knob.

use std::cell::RefCell;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use bench_core::constants::Platform;
use bench_runner::RunnerError;
use serde::{Deserialize, Serialize};

use crate::coolgate::{
    self, host_os, is_executable, macmon_bin, nvidia_smi_bin, CoolGateRecord, GateState, HostOs,
};

/// The 1-minute load average the box must be BELOW before a timed window opens.
pub const LOAD_MAX: f64 = 2.0;
/// The GPU utilization fraction the box must be BELOW before a timed window opens.
pub const GPU_UTIL_MAX: f64 = 0.10;
/// One sample every 15 s, the original step's `sleep 15`.
const POLL_SECONDS: u64 = 15;
/// The original step's 900 s ceiling.
const MAX_WAIT_SECONDS: u64 = 900;

/// The stable name of the give-up refusal, so an operator can grep a log for it.
pub const QUIESCENCE_TIMEOUT: &str = "QUIESCENCE-TIMEOUT";
/// The quiescence gate's own no-reader skip reason, sealed on a skipped record.
pub const QUIESCE_SKIP_NO_READER: &str = "no load or GPU utilization reader";

/// The sealed spelling of a gate that RAN and passed, whether or not it had to wait.
pub const GATE_STATE_PASSED: &str = "passed";
/// The sealed spelling of a gate that did not run.
pub const GATE_STATE_SKIPPED: &str = "skipped";

/// One observation of the box: the 1-minute load average and the GPU utilization fraction.
/// `None` on either side is a DROPPED sample — it counts as busy, never as idle.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct QuiesceSample {
    pub load: Option<f64>,
    pub gpu_util: Option<f64>,
}

impl QuiesceSample {
    /// The box is quiescent only when BOTH readings arrived and BOTH are under their targets.
    fn is_idle(self) -> bool {
        match (self.load, self.gpu_util) {
            (Some(load), Some(util)) => {
                load.is_finite() && util.is_finite() && load < LOAD_MAX && util < GPU_UTIL_MAX
            }
            _ => false,
        }
    }
}

/// A reading for a message: two decimals, or `?` when the sample was dropped.
fn show(value: Option<f64>) -> String {
    match value {
        Some(v) if v.is_finite() => format!("{v:.2}"),
        _ => "?".to_string(),
    }
}

/// Outcome of the poll loop (the pure core, independent of subprocess/sleep).
#[derive(Debug, PartialEq)]
pub enum QuiesceOutcome {
    /// The box was idle after `waited` seconds, at these readings.
    Passed {
        waited: u64,
        load: f64,
        gpu_util: f64,
    },
}

/// [`quiesce_gate_loop_with_progress`] without an observer (tests only; production reports
/// progress).
#[cfg(test)]
fn quiesce_gate_loop<R, S>(read: R, sleep: S) -> Result<QuiesceOutcome, String>
where
    R: FnMut() -> QuiesceSample,
    S: FnMut(u64),
{
    quiesce_gate_loop_with_progress(read, sleep, |_, _| {})
}

/// The pure poll loop: `read` yields one observation, `sleep` waits the given seconds. Returns
/// `Err` with the [`QUIESCENCE_TIMEOUT`] refusal once the box has stayed busy for
/// [`MAX_WAIT_SECONDS`]. Mirrors the original workflow step 1:1 — check, sleep 15, count, give up
/// at 900 — with a dropped sample treated as busy.
///
/// `progress` is an observation-only callback `(waited, sample)` called on every sample, including
/// the passing one, exactly as the cool gate's observer is.
fn quiesce_gate_loop_with_progress<R, S, P>(
    mut read: R,
    mut sleep: S,
    mut progress: P,
) -> Result<QuiesceOutcome, String>
where
    R: FnMut() -> QuiesceSample,
    S: FnMut(u64),
    P: FnMut(u64, QuiesceSample),
{
    let mut waited: u64 = 0;
    loop {
        let sample = read();
        progress(waited, sample);
        if sample.is_idle() {
            return Ok(QuiesceOutcome::Passed {
                waited,
                load: sample.load.unwrap_or(f64::NAN),
                gpu_util: sample.gpu_util.unwrap_or(f64::NAN),
            });
        }
        if waited >= MAX_WAIT_SECONDS {
            return Err(format!(
                "{QUIESCENCE_TIMEOUT}: the box did not reach quiescence within {MAX_WAIT_SECONDS}s \
                 (load={} gpu_util={}); reduce host load or stop other GPU work",
                show(sample.load),
                show(sample.gpu_util)
            ));
        }
        sleep(POLL_SECONDS);
        waited += POLL_SECONDS;
    }
}

/// How the gate reads the 1-minute load average.
enum LoadReader {
    /// macOS: `sysctl -n vm.loadavg` prints `{ l1 l5 l15 }`.
    Sysctl,
    /// Linux: `/proc/loadavg` starts with `l1 l5 l15`.
    ProcLoadavg,
}

/// The standard macOS `sysctl` location. It is addressed by absolute path because `/usr/sbin` is
/// not on every spawned `PATH`.
const SYSCTL_BIN: &str = "/usr/sbin/sysctl";

/// How the gate reads GPU utilization. Both readers are the cool gate's own discovered binaries.
enum UtilReader {
    /// macOS: `macmon pipe -s1` → `gpu_usage[1]`, already a fraction.
    Macmon(PathBuf),
    /// Linux: `nvidia-smi --query-gpu=utilization.gpu` → a percent.
    NvidiaSmi(PathBuf),
}

impl UtilReader {
    /// The stable provenance label of the reader actually resolved.
    fn source_label(&self) -> &'static str {
        match self {
            UtilReader::Macmon(_) => "macmon",
            UtilReader::NvidiaSmi(_) => "nvidia-smi",
        }
    }
}

/// Resolve the load reader for the host OS. `None` means this host has no load source the gate
/// knows, which SKIPS the gate exactly as a missing temperature reader skips the cool gate.
fn resolve_load_reader_for(os: HostOs) -> Option<LoadReader> {
    match os {
        HostOs::MacOs => Some(LoadReader::Sysctl),
        HostOs::Linux => Some(LoadReader::ProcLoadavg),
        HostOs::Other => None,
    }
}

/// Resolve the GPU utilization reader for the host OS: macOS → macmon, Linux → nvidia-smi, in
/// both cases the binary the cool gate's own discovery finds. `MLXFAST_MACMON_BIN` wins when set,
/// like the cool gate's. `MLXFAST_GPU_TEMP_CMD` is deliberately NOT consulted: it prints a
/// temperature and carries no utilization.
fn resolve_util_reader_for(os: HostOs) -> Option<UtilReader> {
    if let Ok(bin) = std::env::var("MLXFAST_MACMON_BIN") {
        if !bin.is_empty() {
            let p = PathBuf::from(&bin);
            if is_executable(&p) {
                return Some(UtilReader::Macmon(p));
            }
            eprintln!("benchd: MLXFAST_MACMON_BIN is set but not executable: {bin}");
            return None;
        }
    }
    match os {
        HostOs::MacOs => macmon_bin().map(UtilReader::Macmon),
        HostOs::Linux => nvidia_smi_bin().map(UtilReader::NvidiaSmi),
        HostOs::Other => macmon_bin()
            .map(UtilReader::Macmon)
            .or_else(|| nvidia_smi_bin().map(UtilReader::NvidiaSmi)),
    }
}

/// The load reader, resolved ONCE per process (the host and the env it reads are fixed at start).
fn resolve_load_reader() -> &'static Option<LoadReader> {
    static READER: OnceLock<Option<LoadReader>> = OnceLock::new();
    READER.get_or_init(|| resolve_load_reader_for(host_os()))
}

/// The GPU utilization reader, resolved ONCE per process.
fn resolve_util_reader() -> &'static Option<UtilReader> {
    static READER: OnceLock<Option<UtilReader>> = OnceLock::new();
    READER.get_or_init(|| resolve_util_reader_for(host_os()))
}

/// The first whitespace-separated token that parses as a number. `sysctl -n vm.loadavg` prints
/// `{ 1.68 1.72 1.66 }`, so the `{` is skipped and the 1-minute value is taken; `/proc/loadavg`
/// starts with that value directly.
fn first_number(text: &str) -> Option<f64> {
    text.split_whitespace().find_map(|t| t.parse::<f64>().ok())
}

/// The GPU utilization fraction carried by ONE macmon `pipe` JSON line, if it carries one.
/// `gpu_usage` is `[frequency, utilization]` and the utilization is already a fraction.
fn macmon_line_gpu_util(line: &str) -> Option<f64> {
    serde_json::from_str::<serde_json::Value>(line)
        .ok()?
        .get("gpu_usage")?
        .get(1)?
        .as_f64()
}

/// Read one 1-minute load average, or `None` on an unusable sample.
fn read_load(reader: &LoadReader) -> Option<f64> {
    match reader {
        LoadReader::Sysctl => {
            let out = Command::new(SYSCTL_BIN)
                .arg("-n")
                .arg("vm.loadavg")
                .output()
                .ok()?;
            first_number(&String::from_utf8_lossy(&out.stdout))
        }
        LoadReader::ProcLoadavg => first_number(&std::fs::read_to_string("/proc/loadavg").ok()?),
    }
}

/// Read one GPU utilization FRACTION, or `None` on an unusable sample.
fn read_gpu_util(reader: &UtilReader) -> Option<f64> {
    match reader {
        UtilReader::Macmon(bin) => {
            let out = Command::new(bin).arg("pipe").arg("-s1").output().ok()?;
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .find_map(macmon_line_gpu_util)
        }
        UtilReader::NvidiaSmi(bin) => {
            let out = Command::new(bin)
                .arg("--query-gpu=utilization.gpu")
                .arg("--format=csv,noheader,nounits")
                .output()
                .ok()?;
            // One line per GPU with `noheader,nounits`, so the first line is GPU 0's percent.
            let percent = String::from_utf8_lossy(&out.stdout)
                .lines()
                .next()?
                .trim()
                .parse::<f64>()
                .ok()?;
            Some(percent / 100.0)
        }
    }
}

/// The recorded state of the quiescence gate, the same three states the cool gate records:
/// `Fired` (the box was already idle, passed with no wait), `Waited` (the gate blocked until the
/// box went idle), `SkippedNoReader` (the gate was off or no reader resolved). A BUSY box that
/// never goes idle is never silently skipped — it refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuiesceState {
    Fired,
    Waited,
    #[default]
    SkippedNoReader,
}

impl QuiesceState {
    /// The sealed spelling of the state, as the per-pair record carries it.
    pub fn as_str(self) -> &'static str {
        match self {
            QuiesceState::Fired => "fired",
            QuiesceState::Waited => "waited",
            QuiesceState::SkippedNoReader => "skipped-no-reader",
        }
    }
}

/// What the gate sealed for one timed phase: its state, how long it waited, and the readings it
/// passed on. The readings are `None` on a skip, where nothing was observed, and `skip_reason`
/// says why instead.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct QuiesceRecord {
    pub state: QuiesceState,
    pub waited_seconds: u64,
    pub load: Option<f64>,
    pub gpu_util: Option<f64>,
    pub skip_reason: Option<String>,
}

impl QuiesceRecord {
    /// The record of a gate that did not run: no wait, no readings, and the reason it was skipped.
    pub fn skipped(reason: impl Into<String>) -> Self {
        Self {
            state: QuiesceState::SkippedNoReader,
            waited_seconds: 0,
            load: None,
            gpu_util: None,
            skip_reason: Some(reason.into()),
        }
    }
}

/// Run the quiescence gate before a timed `phase` and REPORT its resolved state. A give-up
/// returns the TYPED [`RunnerError::GateRejected`] (the one-gated-retry class), never `Ok`, so a
/// busy box can never be measured. The disabled / no-reader SKIP is the only path that returns
/// without enforcing, and it is recorded as `SkippedNoReader` rather than silently "passed".
pub fn quiesce_gate_report(phase: &str) -> Result<QuiesceRecord, RunnerError> {
    // The two gates share the ONE existing switch: no second env knob (David, 2026-09-17).
    if coolgate::gates_disabled() {
        eprintln!(
            "benchd: {phase} quiescence gate disabled (MLXFAST_LOCAL_COOL_GATE=0); timings taken on a loaded box are not comparable to gated runs"
        );
        return Ok(QuiesceRecord::skipped(coolgate::GATE_SKIP_DISABLED));
    }
    let (load_reader, util_reader) = match (resolve_load_reader(), resolve_util_reader()) {
        (Some(load), Some(util)) => (load, util),
        _ => {
            eprintln!(
                "benchd: skipping the {phase} box quiescence gate: no load or GPU utilization reader (install macmon, set MLXFAST_MACMON_BIN, or run on a box with nvidia-smi)"
            );
            return Ok(QuiesceRecord::skipped(QUIESCE_SKIP_NO_READER));
        }
    };
    eprintln!(
        "benchd: {phase} quiescence gate (reader {}): waiting for load < {LOAD_MAX:.2} and GPU util < {GPU_UTIL_MAX:.2} before timing...",
        util_reader.source_label()
    );
    let started = std::time::Instant::now();
    match quiesce_gate_loop_with_progress(
        || QuiesceSample {
            load: read_load(load_reader),
            gpu_util: read_gpu_util(util_reader),
        },
        |secs| std::thread::sleep(Duration::from_secs(secs)),
        |waited, sample| {
            eprintln!(
                "benchd: {phase} quiescing: load {}, gpu util {}, target load <{LOAD_MAX:.2} util <{GPU_UTIL_MAX:.2}, elapsed {:.0}s (gate wait {waited}s)",
                show(sample.load),
                show(sample.gpu_util),
                started.elapsed().as_secs_f64()
            );
        },
    ) {
        Ok(QuiesceOutcome::Passed {
            waited,
            load,
            gpu_util,
        }) => {
            eprintln!(
                "benchd: {phase} quiescence gate passed (waited {waited}s, load {load:.2}, gpu_util {gpu_util:.2})"
            );
            // waited==0 ⇒ already idle (fired without blocking); waited>0 ⇒ blocked to go idle.
            Ok(QuiesceRecord {
                state: if waited == 0 {
                    QuiesceState::Fired
                } else {
                    QuiesceState::Waited
                },
                waited_seconds: waited,
                load: Some(load),
                gpu_util: Some(gpu_util),
                skip_reason: None,
            })
        }
        Err(reason) => Err(RunnerError::GateRejected {
            phase: phase.to_string(),
            reason,
        }),
    }
}

/// The TWO gates every timed measurement runs behind, in the ONE order the ruling fixes:
/// quiescence first (the box is idle), then cool (the GPU is at or below the platform's gate
/// temperature). Both gates are passed IN, so the order is a property of this function and a test
/// can assert it with recording fakes instead of a live box. Every production call site goes
/// through here, so there is exactly one place the order is written down.
pub fn run_timed_phase_gates<Q, C, T>(
    quiesce: Q,
    cool: C,
) -> Result<(QuiesceRecord, T), RunnerError>
where
    Q: FnOnce() -> Result<QuiesceRecord, RunnerError>,
    C: FnOnce() -> Result<T, RunnerError>,
{
    let quiesced = quiesce()?;
    let cooled = cool()?;
    Ok((quiesced, cooled))
}

/// Both gates with the production readers, KEEPING both recorded states for the seal.
pub fn timed_phase_gates_report(
    phase: &str,
    platform: Platform,
) -> Result<(QuiesceRecord, CoolGateRecord), RunnerError> {
    run_timed_phase_gates(
        || quiesce_gate_report(phase),
        || coolgate::cool_gate_report(phase, platform),
    )
}

/// Both gates with the production readers, discarding the recorded states. This is the closure
/// benchd threads into the local timing path and the calibration passes.
pub fn timed_phase_gates(phase: &str, platform: Platform) -> Result<(), RunnerError> {
    timed_phase_gates_report(phase, platform).map(|_| ())
}

// ---------------------------------------------------------------------------------------------
// THE GATE LOG — one sealed record per gate POINT.
//
// David 2026-09-17: a run has MANY gate points, so the seal names each one separately instead of
// folding them into one figure. A paired official run of two pairs has eight points: two pairs x
// two legs x two phases, each with BOTH gates. A calibration run has two points per pass.
// ---------------------------------------------------------------------------------------------

/// What ONE gate read at one point. The readings are present only when the gate RAN; a skipped
/// gate carries `skip_reason` and no readings.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct QuiescenceReading {
    /// [`GATE_STATE_PASSED`] or [`GATE_STATE_SKIPPED`]. A gate that REFUSED never reaches a seal:
    /// the run stops instead (fail-closed).
    pub state: String,
    pub waited_seconds: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_util: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skip_reason: Option<String>,
}

/// The cool gate's half of a gate point, in the same shape as the quiescence half.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CoolReading {
    pub state: String,
    pub waited_seconds: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_temp_c: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skip_reason: Option<String>,
}

impl From<&QuiesceRecord> for QuiescenceReading {
    fn from(r: &QuiesceRecord) -> Self {
        let skipped = r.state == QuiesceState::SkippedNoReader;
        Self {
            state: if skipped {
                GATE_STATE_SKIPPED.to_string()
            } else {
                GATE_STATE_PASSED.to_string()
            },
            waited_seconds: r.waited_seconds,
            load: r.load,
            gpu_util: r.gpu_util,
            skip_reason: r.skip_reason.clone(),
        }
    }
}

impl From<&CoolGateRecord> for CoolReading {
    fn from(r: &CoolGateRecord) -> Self {
        let skipped = r.state == GateState::SkippedNoReader;
        Self {
            state: if skipped {
                GATE_STATE_SKIPPED.to_string()
            } else {
                GATE_STATE_PASSED.to_string()
            },
            waited_seconds: r.waited_seconds,
            gpu_temp_c: r.gpu_temp_c,
            skip_reason: r.skip_reason.clone(),
        }
    }
}

/// ONE gate point: WHERE the two gates applied, and what each of them read. `pair` and `leg` name
/// a point on the paired official path; `pass` names one on a calibration run, which has a single
/// leg. The two files keep the same field names for everything else.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct GateRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pair: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pass: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leg: Option<String>,
    /// `"prefill"` or `"decode"` — the timed phase this point guarded.
    pub phase: String,
    pub quiescence: QuiescenceReading,
    pub cool: CoolReading,
}

/// The leg name of the serial-control leg, as the seal spells it.
pub const LEG_CONTROL: &str = "control";
/// The leg name of the candidate leg, as the seal spells it.
pub const LEG_CANDIDATE: &str = "candidate";

/// WHERE the next gate points apply. The run sets it as it moves from leg to leg, or from
/// calibration pass to pass, so each recorded point names its own place in the run.
#[derive(Debug, Clone, Default, PartialEq)]
struct GateContext {
    pair: Option<i64>,
    pass: Option<i64>,
    leg: Option<String>,
}

/// The run's gate log: every gate point, in run order.
///
/// The gate closure and the run loop both reach it, so it keeps its state behind `RefCell` rather
/// than being threaded as `&mut`. benchd runs one measurement at a time, so there is no second
/// writer.
#[derive(Debug, Default)]
pub struct GateLog {
    context: RefCell<GateContext>,
    records: RefCell<Vec<GateRecord>>,
}

impl GateLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Name the PAIR and LEG the next gate points belong to (the paired official path).
    pub fn enter_leg(&self, pair: usize, leg: &str) {
        *self.context.borrow_mut() = GateContext {
            pair: Some(pair as i64),
            pass: None,
            leg: Some(leg.to_string()),
        };
    }

    /// Name the calibration PASS the next gate points belong to. A pass has one leg, so no leg
    /// name is recorded.
    pub fn enter_pass(&self, pass: u32) {
        *self.context.borrow_mut() = GateContext {
            pair: None,
            pass: Some(pass as i64),
            leg: None,
        };
    }

    /// Record ONE gate point: the phase it guarded and what both gates read.
    pub fn record(&self, phase: &str, quiescence: &QuiesceRecord, cool: &CoolGateRecord) {
        let context = self.context.borrow().clone();
        self.records.borrow_mut().push(GateRecord {
            pair: context.pair,
            pass: context.pass,
            leg: context.leg,
            phase: phase.to_string(),
            quiescence: quiescence.into(),
            cool: cool.into(),
        });
    }

    /// Every point recorded so far, in run order.
    pub fn records(&self) -> Vec<GateRecord> {
        self.records.borrow().clone()
    }
}

/// Both gates for a timed `phase`, RECORDED on `log` at the point they applied. This is what the
/// paired official path and the calibration passes call, so every gate point of a scored run
/// reaches the seal.
pub fn timed_phase_gates_logged(
    phase: &str,
    platform: Platform,
    log: &GateLog,
) -> Result<(), RunnerError> {
    let (quiescence, cool) = timed_phase_gates_report(phase, platform)?;
    log.record(phase, &quiescence, &cool);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// A sample reader driven by a fixed sequence; the last value repeats once exhausted.
    fn seq(values: Vec<QuiesceSample>) -> impl FnMut() -> QuiesceSample {
        let i = Rc::new(RefCell::new(0usize));
        move || {
            let mut idx = i.borrow_mut();
            let v = values.get(*idx).copied().unwrap_or(*values.last().unwrap());
            *idx += 1;
            v
        }
    }

    fn sample(load: f64, gpu_util: f64) -> QuiesceSample {
        QuiesceSample {
            load: Some(load),
            gpu_util: Some(gpu_util),
        }
    }

    #[test]
    fn passes_immediately_when_the_box_is_idle() {
        let r = quiesce_gate_loop(seq(vec![sample(0.4, 0.01)]), |_| {});
        assert_eq!(
            r,
            Ok(QuiesceOutcome::Passed {
                waited: 0,
                load: 0.4,
                gpu_util: 0.01
            })
        );
    }

    #[test]
    fn waits_then_passes_as_load_and_utilization_fall() {
        // busy load -> busy gpu -> both at the edge (not below) -> idle: 3 polls of 15 s.
        let slept = Rc::new(RefCell::new(0u64));
        let s = slept.clone();
        let r = quiesce_gate_loop(
            seq(vec![
                sample(6.0, 0.9),
                sample(1.0, 0.5),
                sample(LOAD_MAX, GPU_UTIL_MAX),
                sample(0.8, 0.02),
            ]),
            move |secs| *s.borrow_mut() += secs,
        );
        assert_eq!(
            r,
            Ok(QuiesceOutcome::Passed {
                waited: 45,
                load: 0.8,
                gpu_util: 0.02
            })
        );
        assert_eq!(*slept.borrow(), 45);
    }

    #[test]
    fn a_dropped_sample_counts_as_busy_and_keeps_waiting() {
        // An idle load with NO utilization reading must not pass; the gate waits for a full one.
        let r = quiesce_gate_loop(
            seq(vec![
                QuiesceSample {
                    load: Some(0.1),
                    gpu_util: None,
                },
                QuiesceSample {
                    load: None,
                    gpu_util: Some(0.01),
                },
                sample(0.1, 0.01),
            ]),
            |_| {},
        );
        assert_eq!(
            r,
            Ok(QuiesceOutcome::Passed {
                waited: 30,
                load: 0.1,
                gpu_util: 0.01
            })
        );
    }

    #[test]
    fn a_dropped_sample_alone_never_passes_the_gate() {
        let r = quiesce_gate_loop(seq(vec![QuiesceSample::default()]), |_| {});
        let err = r.unwrap_err();
        assert!(err.starts_with(QUIESCENCE_TIMEOUT), "{err}");
        assert!(err.contains("load=? gpu_util=?"), "{err}");
    }

    #[test]
    fn a_busy_box_refuses_with_the_stable_string() {
        let slept = Rc::new(RefCell::new(0u64));
        let s = slept.clone();
        let r = quiesce_gate_loop(seq(vec![sample(7.5, 0.83)]), move |secs| {
            *s.borrow_mut() += secs
        });
        assert_eq!(
            r,
            Err(
                "QUIESCENCE-TIMEOUT: the box did not reach quiescence within 900s (load=7.50 \
                 gpu_util=0.83); reduce host load or stop other GPU work"
                    .to_string()
            )
        );
        assert_eq!(*slept.borrow(), MAX_WAIT_SECONDS);
    }

    #[test]
    fn progress_reports_every_sample_without_changing_the_gate() {
        let mut seen = Vec::new();
        let samples = vec![sample(4.0, 0.7), sample(0.5, 0.01)];
        let result = quiesce_gate_loop_with_progress(
            seq(samples.clone()),
            |_| {},
            |waited, s| seen.push((waited, s)),
        );
        assert_eq!(result, quiesce_gate_loop(seq(samples.clone()), |_| {}));
        assert_eq!(seen, vec![(0, samples[0]), (15, samples[1])]);
    }

    #[test]
    fn the_edge_values_are_busy_because_the_targets_are_strict() {
        assert!(!sample(LOAD_MAX, 0.0).is_idle());
        assert!(!sample(0.0, GPU_UTIL_MAX).is_idle());
        assert!(sample(LOAD_MAX - 0.01, GPU_UTIL_MAX - 0.01).is_idle());
    }

    #[test]
    fn the_load_readings_are_parsed_from_both_platform_sources() {
        // macOS `sysctl -n vm.loadavg`, whose first token is a brace.
        assert_eq!(first_number("{ 1.68 1.72 1.66 }\n"), Some(1.68));
        // Linux `/proc/loadavg`.
        assert_eq!(first_number("0.52 0.58 0.59 1/1234 5678\n"), Some(0.52));
        assert_eq!(first_number(""), None);
        assert_eq!(first_number("no numbers here"), None);
    }

    #[test]
    fn the_macmon_utilization_is_the_second_gpu_usage_entry() {
        let line = r#"{"gpu_usage":[1398.0,0.0725],"temp":{"gpu_temp_avg":38.5}}"#;
        assert_eq!(macmon_line_gpu_util(line), Some(0.0725));
        assert_eq!(macmon_line_gpu_util("not json"), None);
        assert_eq!(macmon_line_gpu_util(r#"{"gpu_usage":[1398.0]}"#), None);
    }

    #[test]
    fn the_readers_resolve_natively_per_platform() {
        assert!(matches!(
            resolve_load_reader_for(HostOs::MacOs),
            Some(LoadReader::Sysctl)
        ));
        assert!(matches!(
            resolve_load_reader_for(HostOs::Linux),
            Some(LoadReader::ProcLoadavg)
        ));
        assert!(resolve_load_reader_for(HostOs::Other).is_none());
    }

    #[test]
    fn the_sealed_state_spellings_match_the_cool_gates() {
        assert_eq!(QuiesceState::Fired.as_str(), GateState::Fired.as_str());
        assert_eq!(QuiesceState::Waited.as_str(), GateState::Waited.as_str());
        assert_eq!(
            QuiesceState::SkippedNoReader.as_str(),
            GateState::SkippedNoReader.as_str()
        );
        assert_eq!(
            QuiesceRecord::skipped("no reader"),
            QuiesceRecord {
                state: QuiesceState::SkippedNoReader,
                waited_seconds: 0,
                load: None,
                gpu_util: None,
                skip_reason: Some("no reader".to_string()),
            }
        );
    }

    /// The SKIP path is the cool gate's: the one shared switch turns BOTH gates off, and the
    /// record says `skipped-no-reader` rather than pretending the box was measured idle.
    #[test]
    fn the_shared_switch_skips_both_gates_together() {
        assert_eq!(coolgate::GATE_SWITCH_ENV, "MLXFAST_LOCAL_COOL_GATE");
        assert!(coolgate::gates_disabled_by(Some("0")));
        assert!(!coolgate::gates_disabled_by(Some("1")));
        assert!(!coolgate::gates_disabled_by(None));
        let skipped = QuiesceRecord::skipped(coolgate::GATE_SKIP_DISABLED);
        assert_eq!(skipped.state, QuiesceState::SkippedNoReader);
        assert_eq!(skipped.load, None);
        assert_eq!(
            skipped.skip_reason.as_deref(),
            Some(coolgate::GATE_SKIP_DISABLED)
        );
        // The cool gate's skip carries the SAME reason from the SAME switch.
        assert_eq!(
            coolgate::CoolGateRecord::skipped(coolgate::GATE_SKIP_DISABLED).skip_reason,
            skipped.skip_reason
        );
    }

    /// THE GATE LOG names each point and keeps them in run order, and a SKIPPED gate seals its
    /// reason and NO readings — the seal never shows a load or a temperature nobody measured.
    #[test]
    fn the_gate_log_names_each_point_and_seals_a_skip_without_readings() {
        let log = GateLog::new();
        log.enter_leg(1, LEG_CONTROL);
        log.record(
            "prefill",
            &QuiesceRecord {
                state: QuiesceState::Waited,
                waited_seconds: 30,
                load: Some(0.80),
                gpu_util: Some(0.02),
                skip_reason: None,
            },
            &CoolGateRecord {
                state: GateState::Fired,
                waited_seconds: 0,
                gpu_temp_c: Some(38.5),
                skip_reason: None,
            },
        );
        log.enter_leg(1, LEG_CANDIDATE);
        log.record(
            "decode",
            &QuiesceRecord::skipped(coolgate::GATE_SKIP_DISABLED),
            &CoolGateRecord::skipped(coolgate::COOL_GATE_SKIP_NO_READER),
        );

        let records = log.records();
        assert_eq!(records.len(), 2);

        assert_eq!(records[0].pair, Some(1));
        assert_eq!(records[0].pass, None);
        assert_eq!(records[0].leg.as_deref(), Some("control"));
        assert_eq!(records[0].phase, "prefill");
        assert_eq!(records[0].quiescence.state, GATE_STATE_PASSED);
        assert_eq!(records[0].quiescence.waited_seconds, 30);
        assert_eq!(records[0].quiescence.load, Some(0.80));
        assert_eq!(records[0].quiescence.gpu_util, Some(0.02));
        assert_eq!(records[0].quiescence.skip_reason, None);
        assert_eq!(records[0].cool.state, GATE_STATE_PASSED);
        assert_eq!(records[0].cool.gpu_temp_c, Some(38.5));

        assert_eq!(records[1].leg.as_deref(), Some("candidate"));
        assert_eq!(records[1].quiescence.state, GATE_STATE_SKIPPED);
        assert_eq!(records[1].quiescence.load, None);
        assert_eq!(records[1].quiescence.gpu_util, None);
        assert_eq!(
            records[1].quiescence.skip_reason.as_deref(),
            Some(coolgate::GATE_SKIP_DISABLED)
        );
        assert_eq!(records[1].cool.state, GATE_STATE_SKIPPED);
        assert_eq!(records[1].cool.gpu_temp_c, None);
        assert_eq!(
            records[1].cool.skip_reason.as_deref(),
            Some(coolgate::COOL_GATE_SKIP_NO_READER)
        );

        // A SKIPPED record seals its reason and omits every reading it never took.
        let sealed = serde_json::to_value(&records[1]).expect("a gate record serializes");
        assert_eq!(sealed["quiescence"]["state"], "skipped");
        assert!(sealed["quiescence"].get("load").is_none());
        assert!(sealed["quiescence"].get("gpu_util").is_none());
        assert!(sealed["cool"].get("gpu_temp_c").is_none());
    }

    /// A CALIBRATION point names its PASS and carries no leg: a pass has one leg. Every other
    /// field is spelled exactly as the score's own gate records spell it.
    #[test]
    fn a_calibration_gate_point_names_its_pass_and_no_leg() {
        let log = GateLog::new();
        log.enter_pass(2);
        log.record(
            "decode",
            &QuiesceRecord {
                state: QuiesceState::Fired,
                waited_seconds: 0,
                load: Some(0.30),
                gpu_util: Some(0.01),
                skip_reason: None,
            },
            &CoolGateRecord {
                state: GateState::Waited,
                waited_seconds: 20,
                gpu_temp_c: Some(39.0),
                skip_reason: None,
            },
        );
        let record = log.records().remove(0);
        assert_eq!(record.pass, Some(2));
        assert_eq!(record.pair, None);
        assert_eq!(record.leg, None);
        assert_eq!(record.phase, "decode");
        assert_eq!(record.cool.waited_seconds, 20);
        let sealed = serde_json::to_value(&record).expect("a gate record serializes");
        assert!(sealed.get("pair").is_none());
        assert!(sealed.get("leg").is_none());
        assert_eq!(sealed["pass"], 2);
    }

    #[test]
    fn the_quiescence_gate_runs_before_the_cool_gate() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let q = calls.clone();
        let c = calls.clone();
        let (record, cooled) = run_timed_phase_gates(
            || {
                q.borrow_mut().push("quiesce");
                Ok(QuiesceRecord {
                    state: QuiesceState::Waited,
                    waited_seconds: 30,
                    load: Some(0.5),
                    gpu_util: Some(0.02),
                    skip_reason: None,
                })
            },
            || {
                c.borrow_mut().push("cool");
                Ok(GateState::Fired)
            },
        )
        .expect("both gates pass");
        assert_eq!(*calls.borrow(), vec!["quiesce", "cool"]);
        assert_eq!(record.state, QuiesceState::Waited);
        assert_eq!(record.waited_seconds, 30);
        assert_eq!(cooled, GateState::Fired);
    }

    #[test]
    fn a_refused_quiescence_gate_never_reaches_the_cool_gate() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let c = calls.clone();
        let err = run_timed_phase_gates(
            || {
                Err(RunnerError::GateRejected {
                    phase: "decode".to_string(),
                    reason: format!("{QUIESCENCE_TIMEOUT}: busy"),
                })
            },
            || {
                c.borrow_mut().push("cool");
                Ok(GateState::Fired)
            },
        )
        .expect_err("a busy box refuses");
        assert!(calls.borrow().is_empty(), "the cool gate must not have run");
        assert!(err.to_string().contains(QUIESCENCE_TIMEOUT), "{err}");
    }
}
