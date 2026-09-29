//! The PER-BOX baseline surface of the ranked paired path (David ruling 2026-09-08).
//!
//! A ranked run of a track whose fixture declares
//! [`bench_core::contract::scores_against_live_control_leg`] measures TWO legs on
//! ONE box in ONE job, on the prompt of each pair's golden:
//!
//! 1. a SERIAL-CONTROL leg on the organizer-staged REFERENCE tree, with no speculation;
//! 2. the CANDIDATE leg on the submission tree, at its declared draft depth.
//!
//! The score is the live ratio of the two. There is NO stored pair: not in the constants, not in
//! the fixture, not in the golden. This module owns everything around that ruling that is not a
//! measurement — the two runner inputs, the per-box calibration file, and every refusal by name:
//!
//! * [`resolve_workspace`] / [`load_calibration`] — the two required inputs, from the flags or the
//!   [`BASELINE_WORKSPACE_ENV`] / [`BASELINE_CALIBRATION_ENV`] environment variables;
//! * [`BaselineCalibration::check_identity`] — the file names THIS track and THIS box, and holds
//!   an entry for the prompt;
//! * [`BaselineCalibration::check_band`] — the control leg's measured seconds-per-token sit inside
//!   this box's HEALTH BAND for the prompt it measured. The band is a health gate on leg 1 and
//!   NEVER a denominator: no number in the file reaches the score;
//! * [`refuse_golden_with_stored_pair`] / [`refuse_stored_baseline_override`] — a golden carrying
//!   `benchmark.baseline_*_seconds_per_token`, and the `MLXFAST_PAIRED_BASELINE_*` env /
//!   `--baseline-*` flags, are refused on this path because each is a stored denominator.
//!
//! It also owns the AUTHORING half: [`calibration_from_passes`] turns the control legs
//! `benchd calibrate-baseline` measured on each prompt into the file, and refuses by name
//! ([`bench_core::constants::CALIBRATION_CV_EXCEEDED`]) when the box is too noisy for the mean to
//! describe it.

use bench_core::constants::{CALIBRATION_CV_EXCEEDED, CALIBRATION_MAX_CV_PERCENT};
use bench_core::golden::GoldenFixture;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The runner environment variable naming the built REFERENCE tree on this box.
pub const BASELINE_WORKSPACE_ENV: &str = "MLXFAST_BASELINE_WORKSPACE";
/// The runner environment variable naming this box's calibration file.
pub const BASELINE_CALIBRATION_ENV: &str = "MLXFAST_BASELINE_CALIBRATION";
/// The Actions variable naming the runner a job runs on. It is the authority for the calibration
/// file's `box` field whenever it is set; absent, the operator states the box with `--box`.
pub const RUNNER_NAME_ENV: &str = "RUNNER_NAME";

/// The schema version this benchd writes. It reads this version and [`CALIBRATION_VERSION_SINGLE_PROMPT`].
pub const CALIBRATION_VERSION: u32 = 2;
/// The earlier single-prompt form. This benchd still reads it, as a file with one prompt entry.
pub const CALIBRATION_VERSION_SINGLE_PROMPT: u32 = 1;

/// The value sealed as `metrics.baseline_source` on a paired run: the denominator was MEASURED by
/// the serial-control leg of this same job, not read from anywhere.
pub const BASELINE_SOURCE_SERIAL_CONTROL_LEG: &str = "serial-control-leg";

/// EXACT-MATCH refusal names. Each is one condition, so an operator greps for the one that
/// stopped the run.
pub const BASELINE_WORKSPACE_MISSING: &str = "BASELINE-WORKSPACE-MISSING";
/// The workspace exists but holds no engine at the candidate's own root-relative path.
pub const BASELINE_WORKSPACE_NO_ENGINE: &str = "BASELINE-WORKSPACE-NO-ENGINE";
/// The candidate engine is not addressable relative to the workspace root, or its re-rooted path
/// would leave the reference tree.
pub const BASELINE_ENGINE_NOT_ROOT_RELATIVE: &str = "BASELINE-ENGINE-NOT-ROOT-RELATIVE";
/// The candidate weights' re-rooted path would leave the reference tree.
pub const BASELINE_WEIGHTS_NOT_ROOT_RELATIVE: &str = "BASELINE-WEIGHTS-NOT-ROOT-RELATIVE";
/// The calibration file has no entry for a prompt this run measures.
pub const BASELINE_CALIBRATION_PROMPT_MISMATCH: &str = "BASELINE-CALIBRATION-PROMPT-MISMATCH";
/// No calibration file was named, or it could not be read.
pub const BASELINE_CALIBRATION_MISSING: &str = "BASELINE-CALIBRATION-MISSING";
/// The calibration file was read but is not a valid calibration.
pub const BASELINE_CALIBRATION_INVALID: &str = "BASELINE-CALIBRATION-INVALID";
/// The calibration file names another track.
pub const BASELINE_CALIBRATION_TRACK_MISMATCH: &str = "BASELINE-CALIBRATION-TRACK-MISMATCH";
/// The calibration file names another box.
pub const BASELINE_CALIBRATION_BOX_MISMATCH: &str = "BASELINE-CALIBRATION-BOX-MISMATCH";
/// The measured serial-control leg is outside this box's band on at least one axis.
pub const SERIAL_CONTROL_LEG_OUTSIDE_BAND: &str = "SERIAL-CONTROL-LEG-OUTSIDE-BAND";
/// The golden carries a stored baseline pair, which this path has no source for.
pub const GOLDEN_CARRIES_STORED_BASELINE: &str = "GOLDEN-CARRIES-STORED-BASELINE";
/// A stored-pair override (`MLXFAST_PAIRED_BASELINE_*` / `--baseline-*`) reached this path.
pub const STORED_BASELINE_OVERRIDE_REFUSED: &str = "STORED-BASELINE-OVERRIDE-REFUSED";
/// The reference tree holds no weights where the candidate's own root-relative path names them.
pub const BASELINE_WORKSPACE_NO_WEIGHTS: &str = "BASELINE-WORKSPACE-NO-WEIGHTS";
/// No box name is resolvable, so the calibration file's `box` field cannot be checked.
pub const BASELINE_BOX_UNRESOLVED: &str = "BASELINE-BOX-UNRESOLVED";
/// `--control-golden` names another prompt than `--golden`, so the two legs would measure two
/// different prompts and the ratio between them would mean nothing.
pub const CONTROL_GOLDEN_PROMPT_MISMATCH: &str = "CONTROL-GOLDEN-PROMPT-MISMATCH";
/// `--control-golden` was given on a run that measures no control leg, where it would be silently
/// ignored.
pub const CONTROL_GOLDEN_WITHOUT_PAIRED_PATH: &str = "CONTROL-GOLDEN-WITHOUT-PAIRED-PATH";
/// A golden flag was given a different number of times than `--golden`: a pin flag, a control
/// golden flag, or `--prompt`. Each one matches `--golden` by position, so the counts must agree.
pub const GOLDEN_COUNT_MISMATCH: &str = "GOLDEN-COUNT-MISMATCH";
/// More than one `--golden` was given on a run that measures no control leg. Only the paired path
/// measures several goldens, so every other path would ignore all but the first.
pub const MULTIPLE_GOLDENS_WITHOUT_PAIRED_PATH: &str = "MULTIPLE-GOLDENS-WITHOUT-PAIRED-PATH";
/// The track's pair count is not a multiple of the number of goldens, so the goldens would not
/// get the same number of pairs.
pub const OFFICIAL_PAIRS_NOT_A_MULTIPLE_OF_GOLDENS: &str =
    "OFFICIAL-PAIRS-NOT-A-MULTIPLE-OF-GOLDENS";

/// The band literals [`calibration_from_passes`] writes. benchd READS the band from the file — a
/// box that needs a different band re-calibrates, it does not edit a constant here.
pub const DEFAULT_PREFILL_BAND_LOW: f64 = 0.95;
/// See [`DEFAULT_PREFILL_BAND_LOW`].
pub const DEFAULT_PREFILL_BAND_HIGH: f64 = 1.05;
/// See [`DEFAULT_PREFILL_BAND_LOW`].
pub const DEFAULT_DECODE_BAND_LOW: f64 = 0.97;
/// See [`DEFAULT_PREFILL_BAND_LOW`].
pub const DEFAULT_DECODE_BAND_HIGH: f64 = 1.03;

/// One box's calibration file: VALUES ONLY, and every value is a HEALTH fact about the box.
///
/// The header names the track, the box, the reference tree and the benchd that measured. Each
/// entry of `prompts` holds the band of one prompt, because a control leg's cost is a property of
/// the prompt it measures.
///
/// `deny_unknown_fields` + no `serde(default)`: a file missing a field, or carrying one this
/// benchd does not know, is REFUSED rather than silently defaulted. A calibration is the thing
/// that decides whether a ranked leg is trustworthy, so it is read strictly or not at all.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineCalibration {
    pub version: u32,
    pub track_id: String,
    /// The runner name this file was captured on. `box` is a reserved word in Rust, so the field
    /// is spelled `box_name` and serialized under the contract's own key.
    #[serde(rename = "box")]
    pub box_name: String,
    /// The reference tree's engine commit at capture time (40 lowercase hex).
    pub reference_commit: String,
    pub captured_at: String,
    /// The benchd source commit that measured the legs (40 lowercase hex).
    pub benchd_source_commit: String,
    /// One entry per calibrated prompt, in the order they were measured. Prompt names are unique.
    pub prompts: Vec<PromptCalibration>,
    /// GATE LOG (David 2026-09-17): every GATE POINT the calibration passes ran behind, in run
    /// order, each naming its PASS and the phase it guarded, and what BOTH gates read. Same
    /// records and same field names as a score's `metrics.gates`, except that a pass has one leg,
    /// so a record carries `pass` in place of `pair` and no `leg`. Pass numbers run on across the
    /// prompts: the passes of entry 2 follow the passes of entry 1.
    ///
    /// OMITTED when empty, so a file captured with the gates off keeps the key set it had, and
    /// `default` lets this benchd read every calibration file written before the gates were
    /// sealed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gates: Vec<crate::quiescegate::GateRecord>,
}

/// The band of ONE prompt on one box.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptCalibration {
    /// The prompt name ([`golden_prompt_name`] of the golden that was measured).
    pub prompt: String,
    /// How many control legs the mean is over.
    pub passes: u32,
    pub prefill_seconds_per_token_mean: f64,
    pub decode_seconds_per_token_mean: f64,
    /// Sample coefficients of variation across the passes, as FRACTIONS (0.004 = 0.4%).
    pub prefill_cv: f64,
    pub decode_cv: f64,
    pub prefill_band_low: f64,
    pub prefill_band_high: f64,
    pub decode_band_low: f64,
    pub decode_band_high: f64,
}

/// The single-prompt file form ([`CALIBRATION_VERSION_SINGLE_PROMPT`]). This benchd reads it as a
/// file with one prompt entry, so a box that holds this form keeps working.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SinglePromptCalibration {
    #[serde(rename = "version")]
    _version: u32,
    track_id: String,
    #[serde(rename = "box")]
    box_name: String,
    reference_commit: String,
    prompt: String,
    passes: u32,
    prefill_seconds_per_token_mean: f64,
    decode_seconds_per_token_mean: f64,
    prefill_cv: f64,
    decode_cv: f64,
    prefill_band_low: f64,
    prefill_band_high: f64,
    decode_band_low: f64,
    decode_band_high: f64,
    captured_at: String,
    benchd_source_commit: String,
    #[serde(default)]
    gates: Vec<crate::quiescegate::GateRecord>,
}

impl From<SinglePromptCalibration> for BaselineCalibration {
    fn from(f: SinglePromptCalibration) -> Self {
        BaselineCalibration {
            // The in-memory value is the current form, so it serializes as one.
            version: CALIBRATION_VERSION,
            track_id: f.track_id,
            box_name: f.box_name,
            reference_commit: f.reference_commit,
            captured_at: f.captured_at,
            benchd_source_commit: f.benchd_source_commit,
            prompts: vec![PromptCalibration {
                prompt: f.prompt,
                passes: f.passes,
                prefill_seconds_per_token_mean: f.prefill_seconds_per_token_mean,
                decode_seconds_per_token_mean: f.decode_seconds_per_token_mean,
                prefill_cv: f.prefill_cv,
                decode_cv: f.decode_cv,
                prefill_band_low: f.prefill_band_low,
                prefill_band_high: f.prefill_band_high,
                decode_band_low: f.decode_band_low,
                decode_band_high: f.decode_band_high,
            }],
            gates: f.gates,
        }
    }
}

/// Only the `version` key, read first to choose the file form.
#[derive(Deserialize)]
struct CalibrationVersion {
    version: u32,
}

/// A calibration file together with the identity of the BYTES it was read from — the digest the
/// paired run seals as `metrics.baseline_calibration_sha256`.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedCalibration {
    pub calibration: BaselineCalibration,
    pub sha256: String,
    pub path: PathBuf,
}

/// A 40-character lowercase-hex commit id.
fn is_commit_sha40(s: &str) -> bool {
    s.len() == 40
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl BaselineCalibration {
    /// Parse and VALIDATE one calibration file's bytes, in either file form. Every refusal names
    /// [`BASELINE_CALIBRATION_INVALID`] and the field that failed.
    pub fn parse(bytes: &[u8]) -> Result<BaselineCalibration, String> {
        let invalid = |e: serde_json::Error| format!("{BASELINE_CALIBRATION_INVALID}: {e}");
        let CalibrationVersion { version } = serde_json::from_slice(bytes).map_err(invalid)?;
        let calibration: BaselineCalibration = match version {
            CALIBRATION_VERSION => serde_json::from_slice(bytes).map_err(invalid)?,
            CALIBRATION_VERSION_SINGLE_PROMPT => {
                serde_json::from_slice::<SinglePromptCalibration>(bytes)
                    .map_err(invalid)?
                    .into()
            }
            other => {
                return Err(format!(
                    "{BASELINE_CALIBRATION_INVALID}: version is {other}, and this benchd reads \
                     versions {CALIBRATION_VERSION_SINGLE_PROMPT} and {CALIBRATION_VERSION}"
                ))
            }
        };
        calibration.validate()?;
        Ok(calibration)
    }

    fn validate(&self) -> Result<(), String> {
        let bad = |what: &str| Err(format!("{BASELINE_CALIBRATION_INVALID}: {what}"));
        if self.track_id.trim().is_empty() {
            return bad("track_id is empty");
        }
        if self.box_name.trim().is_empty() {
            return bad("box is empty");
        }
        if !is_commit_sha40(&self.reference_commit) {
            return bad(&format!(
                "reference_commit {:?} is not a 40-character lowercase-hex commit sha",
                self.reference_commit
            ));
        }
        if !is_commit_sha40(&self.benchd_source_commit) {
            return bad(&format!(
                "benchd_source_commit {:?} is not a 40-character lowercase-hex commit sha",
                self.benchd_source_commit
            ));
        }
        if self.captured_at.trim().is_empty() {
            return bad("captured_at is empty");
        }
        if self.prompts.is_empty() {
            return bad("prompts is empty");
        }
        for (i, entry) in self.prompts.iter().enumerate() {
            if self.prompts[..i].iter().any(|e| e.prompt == entry.prompt) {
                return bad(&format!(
                    "prompt {:?} has more than one entry",
                    entry.prompt
                ));
            }
            entry.validate()?;
        }
        Ok(())
    }

    /// The entry of `prompt`, or a refusal by name that lists the prompts the file holds.
    pub fn entry(&self, prompt: &str) -> Result<&PromptCalibration, String> {
        self.prompts
            .iter()
            .find(|e| e.prompt == prompt)
            .ok_or_else(|| {
                let held: Vec<&str> = self.prompts.iter().map(|e| e.prompt.as_str()).collect();
                format!(
                    "{BASELINE_CALIBRATION_PROMPT_MISMATCH}: the calibration holds prompts \
                     {held:?}, and this run measures prompt {prompt:?}; a band describes the leg \
                     it was measured from, so it cannot gate a leg on another prompt"
                )
            })
    }

    /// The file must name THIS track and THIS box, and hold an entry for THIS prompt. Every
    /// refusal quotes both values.
    ///
    /// The PROMPT is checked for the same reason the box is: a band describes what a control leg
    /// costs, and a control leg's cost is a property of the prompt it measured. An entry captured
    /// on one prompt says nothing about a leg measured on another, so a run whose golden has no
    /// entry is refused rather than checked against a band that does not describe it.
    pub fn check_identity(
        &self,
        track_id: &str,
        box_name: &str,
        prompt: &str,
    ) -> Result<(), String> {
        if self.track_id != track_id {
            return Err(format!(
                "{BASELINE_CALIBRATION_TRACK_MISMATCH}: the calibration names track {:?}, and \
                 this run scores track {track_id:?}",
                self.track_id
            ));
        }
        if self.box_name != box_name {
            return Err(format!(
                "{BASELINE_CALIBRATION_BOX_MISMATCH}: the calibration was captured on box {:?}, \
                 and this run is on box {box_name:?}; each ranked box carries its own calibration",
                self.box_name
            ));
        }
        self.entry(prompt).map(|_| ())
    }

    /// The HEALTH GATE on the serial-control leg of `prompt`: `measured <= mean * high` on both
    /// axes, against that prompt's entry. It is regression detection only: a control leg SLOWER
    /// than the band says the box is not well (thermal, contention, a wrong tree) and the run
    /// seals no score. A leg FASTER than its calibration is a well box and passes; the candidate is
    /// scored against that same live leg, so a fast box hands the candidate nothing. `*_band_low`
    /// is recorded by the calibrator and never read here. Nothing here reaches the score — a leg
    /// inside the band is scored by its own measured value.
    pub fn check_band(
        &self,
        prompt: &str,
        prefill_spt: f64,
        decode_spt: f64,
    ) -> Result<(), String> {
        let entry = self.entry(prompt)?;
        for (axis, measured, mean, high) in [
            (
                "prefill",
                prefill_spt,
                entry.prefill_seconds_per_token_mean,
                entry.prefill_band_high,
            ),
            (
                "decode",
                decode_spt,
                entry.decode_seconds_per_token_mean,
                entry.decode_band_high,
            ),
        ] {
            if !(measured.is_finite() && measured > 0.0) {
                return Err(format!(
                    "{SERIAL_CONTROL_LEG_OUTSIDE_BAND}: serial-control leg outside this box's \
                     band: the {axis} leg measured {measured} seconds per token, which is not a \
                     finite positive number"
                ));
            }
            let hi = mean * high;
            if measured > hi {
                return Err(format!(
                    "{SERIAL_CONTROL_LEG_OUTSIDE_BAND}: serial-control leg outside this box's \
                     band: the {axis} leg measured {measured} seconds per token, and box {:?} is \
                     calibrated at {mean} on prompt {prompt:?} with a ceiling of {hi} ({high} of \
                     the mean); the box is slower than when it was calibrated; refusing to seal a \
                     score",
                    self.box_name
                ));
            }
        }
        Ok(())
    }
}

/// WHICH ENTRY gates the control leg of a pair that measures prompt `measured`.
///
/// Without rotation (`calibration_prompt` is `None`) it is `measured`'s own entry, as upstream
/// rules. On a rotating track (`calibration_prompt` is the contract's `calibration_prompt`) the box
/// is calibrated on that ONE prompt, and its band gates every pool prompt: a v1 file (one entry) or
/// a v2 file without an entry for the drawn prompt resolves to `calibration_prompt`. A v2 file that
/// DOES hold an entry for the drawn prompt gates it with that entry, since a band measured on the
/// prompt itself describes the leg more closely than one measured on another.
pub fn band_prompt<'a>(
    calibration: &BaselineCalibration,
    measured: &'a str,
    calibration_prompt: Option<&'a str>,
) -> &'a str {
    match calibration_prompt {
        Some(fallback) if calibration.entry(measured).is_err() => fallback,
        _ => measured,
    }
}

impl PromptCalibration {
    /// Every refusal names the prompt and the field that failed.
    fn validate(&self) -> Result<(), String> {
        let bad = |what: &str| {
            Err(format!(
                "{BASELINE_CALIBRATION_INVALID}: prompt {:?}: {what}",
                self.prompt
            ))
        };
        if self.prompt.trim().is_empty() {
            return bad("prompt is empty");
        }
        if self.passes < 2 {
            return bad(&format!(
                "passes is {}, and a mean with a coefficient of variation needs at least 2",
                self.passes
            ));
        }
        for (name, value) in [
            (
                "prefill_seconds_per_token_mean",
                self.prefill_seconds_per_token_mean,
            ),
            (
                "decode_seconds_per_token_mean",
                self.decode_seconds_per_token_mean,
            ),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return bad(&format!("{name} is {value}, not a finite positive number"));
            }
        }
        let max_cv = CALIBRATION_MAX_CV_PERCENT / 100.0;
        for (name, value) in [
            ("prefill_cv", self.prefill_cv),
            ("decode_cv", self.decode_cv),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                return bad(&format!(
                    "{name} is {value}, not a finite non-negative number"
                ));
            }
            if value > max_cv {
                return Err(format!(
                    "{CALIBRATION_CV_EXCEEDED}: prompt {:?}: {name} is {value} \
                     ({:.4}%), above the fixed maximum of {CALIBRATION_MAX_CV_PERCENT}%",
                    self.prompt,
                    value * 100.0
                ));
            }
        }
        for (low_name, low, high_name, high) in [
            (
                "prefill_band_low",
                self.prefill_band_low,
                "prefill_band_high",
                self.prefill_band_high,
            ),
            (
                "decode_band_low",
                self.decode_band_low,
                "decode_band_high",
                self.decode_band_high,
            ),
        ] {
            if !(low.is_finite() && low > 0.0) {
                return bad(&format!(
                    "{low_name} is {low}, not a finite positive number"
                ));
            }
            if !(high.is_finite() && high > 0.0) {
                return bad(&format!(
                    "{high_name} is {high}, not a finite positive number"
                ));
            }
            // A band that excludes its own mean is not a band: it would refuse a box that is
            // behaving exactly as calibrated.
            if !(low <= 1.0 && high >= 1.0) {
                return bad(&format!(
                    "the band [{low_name}={low}, {high_name}={high}] does not contain the mean \
                     (it must satisfy low <= 1 <= high)"
                ));
            }
        }
        Ok(())
    }
}

/// Refuse a run whose pair count is not a multiple of its golden count. Pair `k` (1-based)
/// measures golden `(k - 1) mod goldens`, so only a multiple gives every golden the same number
/// of pairs.
pub fn check_pairs_cover_goldens(pairs: usize, goldens: usize) -> Result<(), String> {
    if goldens == 0 || !pairs.is_multiple_of(goldens) {
        return Err(format!(
            "{OFFICIAL_PAIRS_NOT_A_MULTIPLE_OF_GOLDENS}: the track fixture declares \
             official_pairs: {pairs}, and this run was given {goldens} golden(s); each golden gets \
             the same number of pairs, so official_pairs must be a multiple of the golden count"
        ));
    }
    Ok(())
}

/// Resolve the REFERENCE WORKSPACE from the flag, else [`BASELINE_WORKSPACE_ENV`]. It must exist
/// and be a directory; every other state refuses by name.
pub fn resolve_workspace(flag: Option<&Path>, env: Option<&str>) -> Result<PathBuf, String> {
    let raw = match (flag, env.map(str::trim).filter(|s| !s.is_empty())) {
        (Some(p), _) => p.to_path_buf(),
        (None, Some(e)) => PathBuf::from(e),
        (None, None) => {
            return Err(format!(
                "{BASELINE_WORKSPACE_MISSING}: the ranked path measures its own denominator on \
                 the organizer-staged reference tree, so it needs that tree; pass \
                 --baseline-workspace <dir> or set {BASELINE_WORKSPACE_ENV}"
            ))
        }
    };
    if !raw.is_dir() {
        return Err(format!(
            "{BASELINE_WORKSPACE_MISSING}: the reference workspace {} is not a directory",
            raw.display()
        ));
    }
    Ok(raw)
}

/// Resolve, READ and validate the per-box calibration file from the flag, else
/// [`BASELINE_CALIBRATION_ENV`]. Returns the parsed file with the digest of its bytes.
pub fn load_calibration(
    flag: Option<&Path>,
    env: Option<&str>,
) -> Result<LoadedCalibration, String> {
    let path = match (flag, env.map(str::trim).filter(|s| !s.is_empty())) {
        (Some(p), _) => p.to_path_buf(),
        (None, Some(e)) => PathBuf::from(e),
        (None, None) => {
            return Err(format!(
                "{BASELINE_CALIBRATION_MISSING}: the ranked path checks its serial-control leg \
                 against this box's health band; pass --baseline-calibration <file> or set \
                 {BASELINE_CALIBRATION_ENV}"
            ))
        }
    };
    let bytes = std::fs::read(&path).map_err(|e| {
        format!(
            "{BASELINE_CALIBRATION_MISSING}: the calibration file {} could not be read: {e}",
            path.display()
        )
    })?;
    let calibration = BaselineCalibration::parse(&bytes)
        .map_err(|e| format!("{e} (calibration file {})", path.display()))?;
    Ok(LoadedCalibration {
        calibration,
        sha256: crate::score::sha256_hex(&bytes),
        path,
    })
}

/// The box a ranked run is on: `RUNNER_NAME` when the job sets it, else the operator's `--box`.
/// A run that can name neither refuses — the calibration's `box` field has nothing to be checked
/// against, and an unchecked calibration is another box's calibration.
pub fn resolve_box_name(
    flag: Option<&str>,
    runner_name_env: Option<&str>,
) -> Result<String, String> {
    let from_env = runner_name_env.map(str::trim).filter(|s| !s.is_empty());
    let from_flag = flag.map(str::trim).filter(|s| !s.is_empty());
    // RUNNER_NAME is the JOB's own statement of where it runs, so it wins over an operator flag.
    match from_env.or(from_flag) {
        Some(name) => Ok(name.to_string()),
        None => Err(format!(
            "{BASELINE_BOX_UNRESOLVED}: the calibration file names the box it was captured on, \
             and this run can name none; set {RUNNER_NAME_ENV} (Actions does) or pass --box"
        )),
    }
}

/// REFUSE a golden that carries a stored baseline pair on the ranked paired path. The pair has no
/// consumer here — the denominator is measured — so a golden that still declares one is either a
/// stale artifact or an attempt to supply a denominator, and both stop the run.
pub fn refuse_golden_with_stored_pair(golden: &GoldenFixture) -> Result<(), String> {
    let benchmark = match golden.benchmark.as_ref() {
        Some(b) => b,
        None => return Ok(()),
    };
    let carries = benchmark.baseline_prefill_seconds_per_token.is_some()
        || benchmark.baseline_decode_seconds_per_token.is_some();
    if carries {
        return Err(format!(
            "{GOLDEN_CARRIES_STORED_BASELINE}: the golden declares \
             benchmark.baseline_{{prefill,decode}}_seconds_per_token, and this track scores \
             against a serial-control leg measured on this box; re-author the golden without the \
             pair"
        ));
    }
    Ok(())
}

/// REFUSE a stored-pair override on the ranked paired path. `MLXFAST_PAIRED_BASELINE_*` and the
/// `--baseline-*` flags are both denominator sources, and this path has exactly one denominator:
/// the leg it measured.
pub fn refuse_stored_baseline_override(
    env_prefill: Option<&str>,
    env_decode: Option<&str>,
    flags_present: bool,
) -> Result<(), String> {
    let env_present = [env_prefill, env_decode]
        .into_iter()
        .flatten()
        .any(|v| !v.trim().is_empty());
    if env_present {
        return Err(format!(
            "{STORED_BASELINE_OVERRIDE_REFUSED}: MLXFAST_PAIRED_BASELINE_{{PREFILL,DECODE}}_\
             SECONDS_PER_TOKEN is set, and this track's denominator is the serial-control leg this \
             job measures; unset both"
        ));
    }
    if flags_present {
        return Err(format!(
            "{STORED_BASELINE_OVERRIDE_REFUSED}: --baseline-prefill-spt/--baseline-decode-spt were \
             given, and this track's denominator is the serial-control leg this job measures; drop \
             both flags"
        ));
    }
    Ok(())
}

/// How a candidate path relates to the run's own workspace root.
#[derive(Debug, Clone, PartialEq, Eq)]
enum RootRelative {
    /// Addressable from the root, and the relative form stays inside it.
    Inside(PathBuf),
    /// Not addressable from the root at all — an absolute path to something outside the
    /// submission tree.
    OutOfTree,
    /// Addressable, but the relative form walks OUT of the tree with `..`. Re-rooting it would
    /// resolve to something outside the reference workspace, so it is never a valid re-root and
    /// never an out-of-tree path either: it is a refusal.
    Escapes,
}

/// A candidate path expressed RELATIVE to the run's own workspace root.
///
/// A relative form carrying `..` is [`RootRelative::Escapes`], NOT a usable relative path: joining
/// it onto the reference workspace would land outside that workspace, which is exactly what
/// re-rooting exists to prevent. It is also not treated as out-of-tree, because falling through to
/// the shared-tree rule would hand the leg the candidate's own path — the thing the rule forbids.
fn root_relative(candidate: &Path, workspace_root: &Path) -> RootRelative {
    let relative = if candidate.is_relative() {
        candidate.to_path_buf()
    } else {
        match candidate.strip_prefix(workspace_root) {
            Ok(r) => r.to_path_buf(),
            Err(_) => return RootRelative::OutOfTree,
        }
    };
    if relative
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return RootRelative::Escapes;
    }
    RootRelative::Inside(relative)
}

/// Whether `joined` really resolves INSIDE `baseline_workspace`, after both are canonicalized.
///
/// The `..` check above is LEXICAL; this one is not. A symlink inside the reference tree that
/// points out of it resolves outside, and a leg that followed it would load something the
/// organizer did not stage. Both paths exist by the time this runs (the caller has already checked
/// the target is a file or a directory), so a canonicalization that fails is itself a refusal —
/// this returns `false` rather than assuming containment.
fn resolves_inside(joined: &Path, baseline_workspace: &Path) -> bool {
    match (joined.canonicalize(), baseline_workspace.canonicalize()) {
        (Ok(target), Ok(root)) => target.starts_with(root),
        _ => false,
    }
}

/// Re-root the candidate ENGINE into the REFERENCE workspace: the reference leg runs the SAME
/// root-relative path inside the organizer's tree that the candidate leg runs inside the
/// submission tree.
///
/// The candidate path is taken relative to `workspace_root` (the run's own workspace, which is the
/// process working directory). A candidate engine that is not addressable from that root cannot be
/// re-rooted, and the run refuses by name rather than guessing which file in the reference tree the
/// operator meant.
pub fn reference_engine_path(
    candidate_engine: &str,
    workspace_root: &Path,
    baseline_workspace: &Path,
) -> Result<PathBuf, String> {
    let relative = match root_relative(Path::new(candidate_engine), workspace_root) {
        RootRelative::Inside(relative) => relative,
        RootRelative::OutOfTree => {
            return Err(format!(
                "{BASELINE_ENGINE_NOT_ROOT_RELATIVE}: the candidate engine {candidate_engine} is \
                 not under the run's workspace root {}, so the same path cannot be resolved \
                 inside the reference workspace {}; invoke benchd with an engine path relative to \
                 the workspace root",
                workspace_root.display(),
                baseline_workspace.display()
            ))
        }
        RootRelative::Escapes => {
            return Err(format!(
                "{BASELINE_ENGINE_NOT_ROOT_RELATIVE}: the candidate engine {candidate_engine} \
                 walks out of the run's workspace root {} with `..`, so re-rooting it would land \
                 outside the reference workspace {}; the engine path must stay inside the \
                 workspace root",
                workspace_root.display(),
                baseline_workspace.display()
            ))
        }
    };
    let reference = baseline_workspace.join(&relative);
    if !reference.is_file() {
        return Err(format!(
            "{BASELINE_WORKSPACE_NO_ENGINE}: the reference workspace {} holds no engine at {}, \
             the candidate engine's own root-relative path",
            baseline_workspace.display(),
            relative.display()
        ));
    }
    if !resolves_inside(&reference, baseline_workspace) {
        return Err(format!(
            "{BASELINE_ENGINE_NOT_ROOT_RELATIVE}: {} resolves outside the reference workspace {}; \
             the control leg runs the organizer's engine and nothing else",
            reference.display(),
            baseline_workspace.display()
        ));
    }
    Ok(reference)
}

/// The PROMPT NAME of a golden, from its file name: `botany.golden.json` is `botany`.
///
/// It is the one name a calibration file and a ranked run can both state without either reading
/// the other: the calibrator records the golden it measured, and the ranked run names the golden
/// it is measuring. `None` for a path with no file name at all.
pub fn golden_prompt_name(golden: &Path) -> Option<String> {
    let name = golden.file_name()?.to_str()?;
    let stem = name.strip_suffix(".golden.json").unwrap_or(name);
    // A per-depth oracle is the SAME prompt with a depth-specific tape: `botany.mtp1.golden.json`
    // measures prompt `botany`. The prompt is the first `.`-segment of the stem; the depth suffix
    // names the tape, not the prompt, so a box calibrated on `botany` bands every depth of it.
    let prompt = stem.split('.').next().unwrap_or(stem);
    if prompt.is_empty() {
        return None;
    }
    Some(prompt.to_string())
}

/// The default weights directory inside a tree: the directory the tree's own transform writes to.
pub const TREE_WEIGHTS_DIR: &str = "weights";

/// Where the SERIAL-CONTROL leg's weights come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceWeights {
    /// The submission tree TRANSFORMS its own weights, so the reference leg loads the REFERENCE
    /// tree's transform output at the same root-relative path. This is the MLX shape, and it is
    /// the case the rule exists for: a transform is participant-editable, so the control leg must
    /// never load the candidate's output.
    ReferenceTree(PathBuf),
    /// The weights are not addressable from the submission tree at all — an organizer-staged
    /// snapshot outside every checkout, which no participant transform can touch (the CUDA GGUF
    /// target). Both legs load that one tree, and the reference tree has no transform output of
    /// its own to prefer.
    SharedOutOfTree(PathBuf),
}

impl ReferenceWeights {
    /// The directory to load.
    pub fn path(&self) -> &Path {
        match self {
            ReferenceWeights::ReferenceTree(p) | ReferenceWeights::SharedOutOfTree(p) => p,
        }
    }
}

/// THE SERIAL-CONTROL LEG'S WEIGHTS. The control leg must never load the CANDIDATE's transform
/// output: the transform is participant-editable, so a control leg that read it would price the
/// candidate against the candidate's own weights.
///
/// Three rules, in order:
///
/// 1. The candidate weights are addressable from the workspace root (the submission tree
///    transforms into its own checkout — the MLX shape) ⇒ the reference leg loads the SAME
///    root-relative path inside the reference tree. A reference tree with nothing there refuses by
///    name: the organizer staged a tree that has not been transformed.
/// 2. Otherwise, when the reference tree has a transform output of its own
///    ([`TREE_WEIGHTS_DIR`]) ⇒ load that. The reference tree's own output always wins over
///    anything the candidate named.
/// 3. Otherwise the weights are an organizer-staged tree outside every checkout (the CUDA GGUF
///    snapshot), which no participant transform can reach ⇒ both legs load that one tree.
pub fn reference_weights_path(
    candidate_weights: &Path,
    workspace_root: &Path,
    baseline_workspace: &Path,
) -> Result<ReferenceWeights, String> {
    match root_relative(candidate_weights, workspace_root) {
        RootRelative::Inside(relative) => {
            let reference = baseline_workspace.join(&relative);
            if !reference.is_dir() {
                return Err(format!(
                    "{BASELINE_WORKSPACE_NO_WEIGHTS}: the reference workspace {} holds no weights \
                     at {}, the candidate weights' own root-relative path; the control leg must \
                     load the reference tree's own transform output, never the candidate's",
                    baseline_workspace.display(),
                    relative.display()
                ));
            }
            if !resolves_inside(&reference, baseline_workspace) {
                return Err(format!(
                    "{BASELINE_WEIGHTS_NOT_ROOT_RELATIVE}: {} resolves outside the reference \
                     workspace {}; the control leg loads the organizer's transform output and \
                     nothing else",
                    reference.display(),
                    baseline_workspace.display()
                ));
            }
            return Ok(ReferenceWeights::ReferenceTree(reference));
        }
        RootRelative::Escapes => {
            return Err(format!(
                "{BASELINE_WEIGHTS_NOT_ROOT_RELATIVE}: the candidate weights {} walk out of the \
                 run's workspace root {} with `..`, so re-rooting them would land outside the \
                 reference workspace {}; they are neither the reference tree's own output nor an \
                 organizer-staged tree",
                candidate_weights.display(),
                workspace_root.display(),
                baseline_workspace.display()
            ))
        }
        RootRelative::OutOfTree => {}
    }
    let own = baseline_workspace.join(TREE_WEIGHTS_DIR);
    if own.is_dir() {
        return Ok(ReferenceWeights::ReferenceTree(own));
    }
    Ok(ReferenceWeights::SharedOutOfTree(
        candidate_weights.to_path_buf(),
    ))
}

/// WHAT a calibration is OF: the track, the box, the reference tree, the benchd that measured,
/// and when. Everything in the file header that is not a measurement.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationIdentity<'a> {
    pub track_id: &'a str,
    pub box_name: &'a str,
    pub reference_commit: &'a str,
    pub benchd_source_commit: &'a str,
    pub captured_at: &'a str,
}

/// The control legs `benchd calibrate-baseline` measured on ONE prompt.
#[derive(Debug, Clone, Copy)]
pub struct PromptPasses<'a> {
    pub prompt: &'a str,
    pub prefill_legs: &'a [f64],
    pub decode_legs: &'a [f64],
}

/// Author the calibration file from the control legs `benchd calibrate-baseline` measured, one
/// entry per prompt, in the order given.
///
/// The gate is the SAME fixed one the stored-pair capture used: a per-axis SAMPLE coefficient of
/// variation above [`CALIBRATION_MAX_CV_PERCENT`] refuses by name — a box whose legs spread that
/// wide has no mean that describes it, so it has no band either.
pub fn calibration_from_passes(
    identity: &CalibrationIdentity<'_>,
    prompts: &[PromptPasses<'_>],
    gates: Vec<crate::quiescegate::GateRecord>,
) -> Result<BaselineCalibration, String> {
    let mut entries = Vec::with_capacity(prompts.len());
    for passes in prompts {
        entries.push(prompt_calibration_from_passes(passes)?);
    }
    let calibration = BaselineCalibration {
        version: CALIBRATION_VERSION,
        track_id: identity.track_id.to_string(),
        box_name: identity.box_name.to_string(),
        reference_commit: identity.reference_commit.to_string(),
        captured_at: identity.captured_at.to_string(),
        benchd_source_commit: identity.benchd_source_commit.to_string(),
        prompts: entries,
        gates,
    };
    // The file this run writes must be one this same benchd would accept.
    calibration.validate()?;
    Ok(calibration)
}

/// One prompt's entry from its measured legs.
fn prompt_calibration_from_passes(passes: &PromptPasses<'_>) -> Result<PromptCalibration, String> {
    let PromptPasses {
        prompt,
        prefill_legs,
        decode_legs,
    } = *passes;
    if prefill_legs.len() != decode_legs.len() {
        return Err(format!(
            "{BASELINE_CALIBRATION_INVALID}: prompt {prompt:?}: {} prefill legs against {} decode \
             legs",
            prefill_legs.len(),
            decode_legs.len()
        ));
    }
    if prefill_legs.len() < 2 {
        return Err(format!(
            "{BASELINE_CALIBRATION_INVALID}: prompt {prompt:?}: {} pass(es); a mean with a \
             coefficient of variation needs at least 2",
            prefill_legs.len()
        ));
    }
    let mut cvs = Vec::with_capacity(2);
    let mut means = Vec::with_capacity(2);
    for (axis, legs) in [("prefill", prefill_legs), ("decode", decode_legs)] {
        let mean = crate::capture::mean(legs).ok_or_else(|| {
            format!(
                "{BASELINE_CALIBRATION_INVALID}: prompt {prompt:?}: the {axis} legs have no \
                 finite positive mean"
            )
        })?;
        let cv = crate::capture::sample_cv_percent(legs).ok_or_else(|| {
            format!(
                "{BASELINE_CALIBRATION_INVALID}: prompt {prompt:?}: the {axis} legs have no \
                 sample CV"
            )
        })?;
        if cv > CALIBRATION_MAX_CV_PERCENT {
            return Err(format!(
                "{CALIBRATION_CV_EXCEEDED}: prompt {prompt:?}: the {axis} legs vary by {cv:.4}%, \
                 above the fixed maximum of {CALIBRATION_MAX_CV_PERCENT}%; this box is not quiet \
                 enough for a mean to describe it"
            ));
        }
        means.push(mean);
        cvs.push(cv / 100.0);
    }
    Ok(PromptCalibration {
        prompt: prompt.to_string(),
        passes: prefill_legs.len() as u32,
        prefill_seconds_per_token_mean: means[0],
        decode_seconds_per_token_mean: means[1],
        prefill_cv: cvs[0],
        decode_cv: cvs[1],
        prefill_band_low: DEFAULT_PREFILL_BAND_LOW,
        prefill_band_high: DEFAULT_PREFILL_BAND_HIGH,
        decode_band_low: DEFAULT_DECODE_BAND_LOW,
        decode_band_high: DEFAULT_DECODE_BAND_HIGH,
    })
}

/// Write the calibration file ATOMICALLY (temp file + rename) and return the digest of the bytes
/// written, so the operator can pin what they published.
pub fn write_calibration(path: &Path, calibration: &BaselineCalibration) -> Result<String, String> {
    let json = serde_json::to_string_pretty(calibration)
        .map_err(|e| format!("calibration serialize failed: {e}"))?;
    let bytes = format!("{json}\n").into_bytes();
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes)
        .map_err(|e| format!("calibration write failed ({}): {e}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .map_err(|e| format!("calibration rename failed ({}): {e}", path.display()))?;
    Ok(crate::score::sha256_hex(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn valid_document() -> serde_json::Value {
        json!({
            "version": 1,
            "track_id": "qwen3.8-125b-a6b-mlx-v1",
            "box": "m5-max-128gb-4-qwen38-125b-a6b-mlx",
            "reference_commit": "a".repeat(40),
            "prompt": "botany",
            "passes": 4,
            "prefill_seconds_per_token_mean": 0.0006282488193359375,
            "decode_seconds_per_token_mean": 0.0329116748046875,
            "prefill_cv": 0.004,
            "decode_cv": 0.002,
            "prefill_band_low": 0.95,
            "prefill_band_high": 1.05,
            "decode_band_low": 0.97,
            "decode_band_high": 1.03,
            "captured_at": "2026-09-08T00:00:00Z",
            "benchd_source_commit": "b".repeat(40),
        })
    }

    fn parse(doc: &serde_json::Value) -> Result<BaselineCalibration, String> {
        BaselineCalibration::parse(serde_json::to_vec(doc).unwrap().as_slice())
    }

    /// THE SINGLE-PROMPT FILE FORM STILL LOADS, as a file with one prompt entry, so a box that
    /// holds it keeps working.
    #[test]
    fn a_valid_calibration_parses_with_every_field_carried() {
        let cal = parse(&valid_document()).unwrap();
        assert_eq!(cal.version, CALIBRATION_VERSION);
        assert_eq!(cal.track_id, "qwen3.8-125b-a6b-mlx-v1");
        assert_eq!(cal.box_name, "m5-max-128gb-4-qwen38-125b-a6b-mlx");
        assert_eq!(cal.prompts.len(), 1);
        let entry = &cal.prompts[0];
        assert_eq!(entry.prompt, "botany");
        assert_eq!(entry.passes, 4);
        assert_eq!(entry.prefill_seconds_per_token_mean, 0.0006282488193359375);
        assert_eq!(entry.decode_seconds_per_token_mean, 0.0329116748046875);
        assert_eq!(entry.prefill_band_low, 0.95);
        assert_eq!(entry.decode_band_low, 0.97);
        assert_eq!(entry.decode_band_high, 1.03);
        // Round-trip: what this benchd writes is what it reads.
        let round_tripped =
            BaselineCalibration::parse(serde_json::to_string(&cal).unwrap().as_bytes()).unwrap();
        assert_eq!(round_tripped, cal);
    }

    #[test]
    fn an_old_calibration_loads_and_keeps_its_stored_decode_band() {
        let mut doc = valid_document();
        doc["decode_band_low"] = json!(0.98);
        doc["decode_band_high"] = json!(1.02);
        let dir = std::env::temp_dir().join(format!(
            "benchd-old-calibration-test.{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("calibration.json");
        std::fs::write(&path, serde_json::to_vec(&doc).unwrap()).unwrap();

        let cal = load_calibration(Some(&path), None).unwrap().calibration;
        let entry = cal.entry("botany").unwrap();
        assert_eq!(entry.decode_band_low, 0.98);
        assert_eq!(entry.decode_band_high, 1.02);
        let (p, d) = (
            entry.prefill_seconds_per_token_mean,
            entry.decode_seconds_per_token_mean,
        );
        assert!(cal.check_band("botany", p, d * 1.02).is_ok());
        let err = cal.check_band("botany", p, d * 1.025).unwrap_err();
        assert!(err.contains(SERIAL_CONTROL_LEG_OUTSIDE_BAND), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// ROTATION: a rotated job's drawn prompt is gated by the contract's calibration_prompt entry
    /// when the file holds none of its own (every v1 file), and by its own entry when a v2 file
    /// holds one. Without rotation the measured prompt is always the key.
    #[test]
    fn a_rotated_prompt_is_gated_by_its_own_entry_else_the_calibration_prompt() {
        const TRACK: &str = "qwen3.8-125b-a6b-mlx-v1";
        const BOX: &str = "m5-max-128gb-4-qwen38-125b-a6b-mlx";
        // v1, captured on "botany"; the job drew "geology".
        let v1 = parse(&valid_document()).unwrap();
        let band = band_prompt(&v1, "geology", Some("botany"));
        assert_eq!(band, "botany");
        assert!(v1.check_identity(TRACK, BOX, band).is_ok());
        let e = v1.entry(band).unwrap();
        let d = e.decode_seconds_per_token_mean;
        assert!(v1
            .check_band(band, e.prefill_seconds_per_token_mean, d * 1.03)
            .is_ok());
        // Without rotation the unknown prompt is refused by name, as before.
        let band = band_prompt(&v1, "geology", None);
        assert!(v1
            .check_identity(TRACK, BOX, band)
            .unwrap_err()
            .contains(BASELINE_CALIBRATION_PROMPT_MISMATCH));

        // v2 holding an entry for "geology": that entry gates it, not "botany"'s.
        let mut doc = two_prompt_document();
        doc["track_id"] = json!(TRACK);
        doc["box"] = json!(BOX);
        let v2 = parse(&doc).unwrap();
        let other = v2.prompts[1].prompt.clone();
        assert_ne!(other, "botany");
        assert_eq!(band_prompt(&v2, &other, Some("botany")), other);
        assert_eq!(band_prompt(&v2, "not-held", Some("botany")), "botany");
    }

    #[test]
    fn the_calibration_must_name_this_track_this_box_and_this_prompt() {
        const TRACK: &str = "qwen3.8-125b-a6b-mlx-v1";
        const BOX: &str = "m5-max-128gb-4-qwen38-125b-a6b-mlx";
        let cal = parse(&valid_document()).unwrap();
        assert!(cal.check_identity(TRACK, BOX, "botany").is_ok());

        let err = cal
            .check_identity("qwen3.8-125b-a6b-cuda-v1", BOX, "botany")
            .unwrap_err();
        assert!(err.contains(BASELINE_CALIBRATION_TRACK_MISMATCH), "{err}");
        assert!(err.contains("qwen3.8-125b-a6b-cuda-v1"), "{err}");
        assert!(err.contains(TRACK), "{err}");

        let err = cal
            .check_identity(TRACK, "spark-4-qwen38-125b-a6b-cuda", "botany")
            .unwrap_err();
        assert!(err.contains(BASELINE_CALIBRATION_BOX_MISMATCH), "{err}");
        assert!(err.contains("spark-4-qwen38-125b-a6b-cuda"), "{err}");
        assert!(err.contains(BOX), "{err}");

        // THE PROMPT, both directions. A band describes the leg it was measured from, so a run on
        // another prompt is refused rather than gated against a band that does not describe it.
        let err = cal.check_identity(TRACK, BOX, "kelp").unwrap_err();
        assert!(err.contains(BASELINE_CALIBRATION_PROMPT_MISMATCH), "{err}");
        assert!(err.contains("kelp"), "{err}");
        assert!(err.contains("botany"), "{err}");
        // …and the calibrated prompt still passes, so the check is not refusing everything.
        assert!(cal.check_identity(TRACK, BOX, "botany").is_ok());

        // The name both sides state comes from the GOLDEN's file name, one rule for both halves.
        assert_eq!(
            golden_prompt_name(Path::new("/goldens/botany.golden.json")).as_deref(),
            Some("botany")
        );
        assert_eq!(
            golden_prompt_name(Path::new("botany.json")).as_deref(),
            Some("botany")
        );
        assert_eq!(
            golden_prompt_name(Path::new("kelp.golden.json")).as_deref(),
            Some("kelp")
        );
        assert_eq!(golden_prompt_name(Path::new("/")), None);
        assert_eq!(golden_prompt_name(Path::new(".golden.json")), None);
        // A per-depth oracle is the same prompt with a depth-specific tape.
        assert_eq!(
            golden_prompt_name(Path::new("/goldens/botany.mtp1.golden.json")).as_deref(),
            Some("botany")
        );
        assert_eq!(
            golden_prompt_name(Path::new("botany.mtp6.golden.json")).as_deref(),
            Some("botany")
        );
    }

    /// The current form: the shared header, then one entry per prompt.
    fn two_prompt_document() -> serde_json::Value {
        let entry = |prompt: &str, decode_mean: f64| {
            json!({
                "prompt": prompt,
                "passes": 4,
                "prefill_seconds_per_token_mean": 0.0006,
                "decode_seconds_per_token_mean": decode_mean,
                "prefill_cv": 0.004,
                "decode_cv": 0.002,
                "prefill_band_low": 0.95,
                "prefill_band_high": 1.05,
                "decode_band_low": 0.98,
                "decode_band_high": 1.02,
            })
        };
        json!({
            "version": 2,
            "track_id": "track-a",
            "box": "box-a",
            "reference_commit": "a".repeat(40),
            "captured_at": "2026-09-27T00:00:00Z",
            "benchd_source_commit": "b".repeat(40),
            "prompts": [entry("botany", 0.030), entry("kelp", 0.060)],
        })
    }

    /// ONE FILE, ONE BAND PER PROMPT. The identity check passes for each prompt the file holds and
    /// refuses, by name, a prompt it does not hold. The band check reads the entry of the prompt
    /// the leg measured: the same decode time is inside one prompt's band and outside the other's.
    #[test]
    fn a_calibration_holds_one_band_per_prompt() {
        let cal = parse(&two_prompt_document()).unwrap();
        assert_eq!(cal.prompts.len(), 2);
        for prompt in ["botany", "kelp"] {
            assert!(cal.check_identity("track-a", "box-a", prompt).is_ok());
        }
        let err = cal.check_identity("track-a", "box-a", "fern").unwrap_err();
        assert!(err.contains(BASELINE_CALIBRATION_PROMPT_MISMATCH), "{err}");
        assert!(err.contains("fern") && err.contains("kelp"), "{err}");

        assert!(cal.check_band("kelp", 0.0006, 0.060).is_ok());
        let err = cal.check_band("botany", 0.0006, 0.060).unwrap_err();
        assert!(err.contains(SERIAL_CONTROL_LEG_OUTSIDE_BAND), "{err}");
        assert!(err.contains("botany"), "{err}");
        let err = cal.check_band("fern", 0.0006, 0.030).unwrap_err();
        assert!(err.contains(BASELINE_CALIBRATION_PROMPT_MISMATCH), "{err}");

        // Two entries for one prompt would make the lookup ambiguous.
        let mut doc = two_prompt_document();
        doc["prompts"][1]["prompt"] = json!("botany");
        let err = parse(&doc).unwrap_err();
        assert!(err.contains(BASELINE_CALIBRATION_INVALID), "{err}");
        assert!(err.contains("more than one entry"), "{err}");
    }

    /// Pair `k` measures golden `(k - 1) mod N`, so the pair count must be a multiple of N.
    #[test]
    fn the_pair_count_must_be_a_multiple_of_the_golden_count() {
        assert!(check_pairs_cover_goldens(2, 1).is_ok());
        assert!(check_pairs_cover_goldens(6, 3).is_ok());
        let err = check_pairs_cover_goldens(4, 3).unwrap_err();
        assert!(
            err.contains(OFFICIAL_PAIRS_NOT_A_MULTIPLE_OF_GOLDENS),
            "{err}"
        );
        assert!(
            err.contains("official_pairs: 4") && err.contains('3'),
            "{err}"
        );
    }

    #[test]
    fn a_missing_field_refuses_by_name() {
        for field in [
            "version",
            "track_id",
            "box",
            "reference_commit",
            "prompt",
            "passes",
            "prefill_seconds_per_token_mean",
            "decode_seconds_per_token_mean",
            "prefill_cv",
            "decode_cv",
            "prefill_band_low",
            "prefill_band_high",
            "decode_band_low",
            "decode_band_high",
            "captured_at",
            "benchd_source_commit",
        ] {
            let mut doc = valid_document();
            doc.as_object_mut().unwrap().remove(field);
            let err = parse(&doc).unwrap_err();
            assert!(
                err.contains(BASELINE_CALIBRATION_INVALID) && err.contains(field),
                "dropping {field} must refuse by name: {err}"
            );
        }
        // An UNKNOWN field is refused too: a calibration is read strictly or not at all.
        let mut doc = valid_document();
        doc["baseline_prefill_seconds_per_token"] = json!(0.0006);
        let err = parse(&doc).unwrap_err();
        assert!(err.contains(BASELINE_CALIBRATION_INVALID), "{err}");
    }

    #[test]
    fn a_wrong_version_a_short_commit_and_a_noisy_capture_refuse_by_name() {
        let mut doc = valid_document();
        doc["version"] = json!(3);
        let err = parse(&doc).unwrap_err();
        assert!(
            err.contains(BASELINE_CALIBRATION_INVALID) && err.contains("version"),
            "{err}"
        );

        let mut doc = valid_document();
        doc["reference_commit"] = json!("deadbeef");
        let err = parse(&doc).unwrap_err();
        assert!(err.contains("reference_commit"), "{err}");

        let mut doc = valid_document();
        doc["passes"] = json!(1);
        let err = parse(&doc).unwrap_err();
        assert!(err.contains("passes"), "{err}");

        // A file whose recorded CV is above the fixed maximum is refused at READ time too, not
        // only when it is written: the file is the only evidence a reader has.
        let mut doc = valid_document();
        doc["decode_cv"] = json!(0.02);
        let err = parse(&doc).unwrap_err();
        assert!(err.contains(CALIBRATION_CV_EXCEEDED), "{err}");
    }

    #[test]
    fn a_band_that_excludes_its_own_mean_is_refused() {
        for (field, value) in [
            ("prefill_band_low", json!(1.01)),
            ("prefill_band_high", json!(0.99)),
            ("decode_band_low", json!(0.0)),
            ("decode_band_high", json!(-1.0)),
            ("prefill_band_low", json!("wide")),
        ] {
            let mut doc = valid_document();
            doc[field] = value.clone();
            let err = parse(&doc).unwrap_err();
            assert!(
                err.contains(BASELINE_CALIBRATION_INVALID),
                "{field}={value} must refuse: {err}"
            );
        }
    }

    #[test]
    fn the_band_check_refuses_only_a_slower_leg_on_either_axis() {
        let cal = parse(&valid_document()).unwrap();
        let (p, d) = (
            cal.prompts[0].prefill_seconds_per_token_mean,
            cal.prompts[0].decode_seconds_per_token_mean,
        );
        // Dead centre and the ceiling of each band are INSIDE.
        assert!(cal.check_band("botany", p, d).is_ok());
        assert!(cal.check_band("botany", p * 1.05, d * 1.03).is_ok());
        // A +2.5% control leg fits the new default ceiling.
        assert!(cal.check_band("botany", p, d * 1.025).is_ok());
        // A FASTER leg is a well box: below `*_band_low`, and far below it, both pass. The
        // low bound is recorded, never read.
        assert!(cal.check_band("botany", p * 0.95, d * 0.97).is_ok());
        assert!(cal.check_band("botany", p * 0.9, d).is_ok());
        assert!(cal.check_band("botany", p, d * 0.9).is_ok());
        assert!(cal.check_band("botany", p * 0.5, d * 0.5).is_ok());

        // A SLOWER leg, on either axis, refuses by name.
        for (label, prefill, decode) in [
            ("prefill high", p * 1.1, d),
            ("decode high", p, d * 1.1),
            ("decode just over", p, d * 1.0301),
            ("decode +3.5%", p, d * 1.035),
        ] {
            let err = cal.check_band("botany", prefill, decode).unwrap_err();
            assert!(
                err.contains(SERIAL_CONTROL_LEG_OUTSIDE_BAND)
                    && err.contains("serial-control leg outside this box's band"),
                "{label} must refuse by name: {err}"
            );
            assert!(err.contains(&cal.box_name), "{label}: {err}");
        }
        // A non-finite or non-positive measurement is outside every band.
        assert!(cal.check_band("botany", f64::NAN, d).is_err());
        assert!(cal.check_band("botany", p, 0.0).is_err());
    }

    #[test]
    fn the_box_name_comes_from_runner_name_first_then_the_flag() {
        assert_eq!(
            resolve_box_name(Some("flag-box"), Some("runner-box")).unwrap(),
            "runner-box"
        );
        assert_eq!(
            resolve_box_name(Some("flag-box"), None).unwrap(),
            "flag-box"
        );
        assert_eq!(
            resolve_box_name(Some("flag-box"), Some("  ")).unwrap(),
            "flag-box"
        );
        let err = resolve_box_name(None, None).unwrap_err();
        assert!(err.contains(BASELINE_BOX_UNRESOLVED), "{err}");
        assert!(err.contains(RUNNER_NAME_ENV), "{err}");
    }

    #[test]
    fn a_stored_pair_override_is_refused_from_either_door() {
        assert!(refuse_stored_baseline_override(None, None, false).is_ok());
        assert!(refuse_stored_baseline_override(Some(""), Some("   "), false).is_ok());

        let err = refuse_stored_baseline_override(Some("0.0006"), None, false).unwrap_err();
        assert!(err.contains(STORED_BASELINE_OVERRIDE_REFUSED), "{err}");
        assert!(err.contains("MLXFAST_PAIRED_BASELINE"), "{err}");

        let err = refuse_stored_baseline_override(None, Some("0.03"), false).unwrap_err();
        assert!(err.contains(STORED_BASELINE_OVERRIDE_REFUSED), "{err}");

        let err = refuse_stored_baseline_override(None, None, true).unwrap_err();
        assert!(err.contains(STORED_BASELINE_OVERRIDE_REFUSED), "{err}");
        assert!(err.contains("--baseline-prefill-spt"), "{err}");
    }

    /// THE CONTROL LEG NEVER LOADS THE CANDIDATE'S TRANSFORM OUTPUT. All three rules, each with
    /// the negative control that proves the rule discriminates.
    #[test]
    fn the_control_leg_loads_the_reference_trees_own_weights() {
        let root = std::env::temp_dir().join(format!("benchd-refw.{}", std::process::id()));
        let candidate_root = root.join("candidate");
        let reference_root = root.join("reference");
        std::fs::create_dir_all(candidate_root.join("weights")).unwrap();
        std::fs::create_dir_all(reference_root.join("weights")).unwrap();

        // RULE 1 — the submission transforms into its own checkout: the reference leg loads the
        // SAME root-relative path inside the reference tree, which is a DIFFERENT directory.
        for candidate in [PathBuf::from("weights"), candidate_root.join("weights")] {
            let got = reference_weights_path(&candidate, &candidate_root, &reference_root).unwrap();
            assert_eq!(
                got,
                ReferenceWeights::ReferenceTree(reference_root.join("weights"))
            );
            assert_ne!(
                got.path(),
                candidate_root.join("weights"),
                "the control leg must never load the candidate's transform output"
            );
        }

        // RULE 1, negative: a reference tree that was never transformed refuses BY NAME rather
        // than falling back to the candidate's output.
        let bare = root.join("bare-reference");
        std::fs::create_dir_all(&bare).unwrap();
        let err = reference_weights_path(Path::new("weights"), &candidate_root, &bare).unwrap_err();
        assert!(err.contains(BASELINE_WORKSPACE_NO_WEIGHTS), "{err}");
        assert!(err.contains("weights"), "{err}");

        // RULE 2 — an out-of-tree candidate path, and the reference tree HAS its own transform
        // output: the reference tree's own output wins.
        let outside = root.join("organizer-snapshot");
        std::fs::create_dir_all(&outside).unwrap();
        assert_eq!(
            reference_weights_path(&outside, &candidate_root, &reference_root).unwrap(),
            ReferenceWeights::ReferenceTree(reference_root.join("weights"))
        );

        // RULE 3 — an out-of-tree candidate path and a reference tree with no transform output:
        // an organizer-staged snapshot no participant transform can reach, so both legs load it.
        assert_eq!(
            reference_weights_path(&outside, &candidate_root, &bare).unwrap(),
            ReferenceWeights::SharedOutOfTree(outside.clone())
        );

        // ESCAPE — a candidate path that walks OUT of the workspace root with `..`. Re-rooting it
        // would land outside the reference workspace, and treating it as out-of-tree would hand
        // the control leg the candidate's own directory. Both are refused, BY NAME. The escaping
        // path is a REAL directory, so the refusal is the containment rule and not a missing file.
        let escape_target = candidate_root.join("weights");
        assert!(
            escape_target.is_dir(),
            "the escape target must really exist"
        );
        for candidate in [
            PathBuf::from("../candidate/weights"),
            candidate_root.join("../candidate/weights"),
        ] {
            let err =
                reference_weights_path(&candidate, &candidate_root, &reference_root).unwrap_err();
            assert!(
                err.contains(BASELINE_WEIGHTS_NOT_ROOT_RELATIVE),
                "{candidate:?}: {err}"
            );
            assert!(
                !err.contains(BASELINE_WORKSPACE_NO_WEIGHTS),
                "{candidate:?}: an escaping path is not a missing-weights refusal: {err}"
            );
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    /// The SAME containment rule on the ENGINE path: a candidate engine that walks out of the
    /// workspace root with `..` is refused BY NAME, even when the escaping path names a REAL
    /// executable — otherwise the control leg would run the candidate's binary.
    #[test]
    fn an_escaping_engine_path_is_refused_by_name() {
        let root = std::env::temp_dir().join(format!("benchd-refeng.{}", std::process::id()));
        let candidate_root = root.join("candidate");
        let reference_root = root.join("reference");
        std::fs::create_dir_all(candidate_root.join(".build/release")).unwrap();
        std::fs::create_dir_all(reference_root.join(".build/release")).unwrap();
        let candidate_engine = candidate_root.join(".build/release/bench-worker");
        std::fs::write(&candidate_engine, b"#!/bin/sh\n").unwrap();
        std::fs::write(
            reference_root.join(".build/release/bench-worker"),
            b"#!/bin/sh\n",
        )
        .unwrap();

        // The honest case still resolves, and to the REFERENCE tree's binary.
        assert_eq!(
            reference_engine_path(
                ".build/release/bench-worker",
                &candidate_root,
                &reference_root
            )
            .unwrap(),
            reference_root.join(".build/release/bench-worker")
        );

        // The escape, both spellings, against a REAL file.
        assert!(
            candidate_engine.is_file(),
            "the escape target must really exist"
        );
        for candidate in [
            "../candidate/.build/release/bench-worker".to_string(),
            candidate_root
                .join("../candidate/.build/release/bench-worker")
                .to_string_lossy()
                .to_string(),
        ] {
            let err =
                reference_engine_path(&candidate, &candidate_root, &reference_root).unwrap_err();
            assert!(
                err.contains(BASELINE_ENGINE_NOT_ROOT_RELATIVE),
                "{candidate}: {err}"
            );
            assert!(
                !err.contains(BASELINE_WORKSPACE_NO_ENGINE),
                "{candidate}: an escaping path is not a missing-engine refusal: {err}"
            );
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn calibration_authoring_computes_the_means_and_refuses_a_noisy_box() {
        let reference = "a".repeat(40);
        let benchd = "b".repeat(40);
        let identity = CalibrationIdentity {
            track_id: "qwen3.8-125b-a6b-mlx-v1",
            box_name: "m5-max-128gb-4-qwen38-125b-a6b-mlx",
            reference_commit: &reference,
            benchd_source_commit: &benchd,
            captured_at: "2026-09-08T00:00:00Z",
        };
        let passes = |prefill_legs: &'static [f64], decode_legs: &'static [f64]| PromptPasses {
            prompt: "botany",
            prefill_legs,
            decode_legs,
        };
        let cal = calibration_from_passes(
            &identity,
            &[passes(
                &[0.001, 0.001, 0.001, 0.001],
                &[0.030, 0.030, 0.030, 0.030],
            )],
            Vec::new(),
        )
        .unwrap();
        let entry = &cal.prompts[0];
        assert_eq!(entry.passes, 4);
        assert_eq!(entry.prefill_seconds_per_token_mean, 0.001);
        assert_eq!(entry.decode_seconds_per_token_mean, 0.030);
        assert_eq!(entry.prefill_cv, 0.0);
        assert_eq!(entry.prefill_band_low, DEFAULT_PREFILL_BAND_LOW);
        assert_eq!(entry.decode_band_low, DEFAULT_DECODE_BAND_LOW);
        assert_eq!(entry.decode_band_high, DEFAULT_DECODE_BAND_HIGH);

        // A decode axis that varies by ~4.7% is well past the fixed 1% maximum.
        let err = calibration_from_passes(
            &identity,
            &[passes(
                &[0.001, 0.001, 0.001, 0.001],
                &[0.030, 0.032, 0.029, 0.031],
            )],
            Vec::new(),
        )
        .unwrap_err();
        assert!(err.contains(CALIBRATION_CV_EXCEEDED), "{err}");
        assert!(err.contains("decode"), "{err}");

        // WRITE + READ BACK: the file this verb writes is one this same benchd accepts, and its
        // digest identifies the bytes that were written.
        let dir =
            std::env::temp_dir().join(format!("benchd-calibration-test.{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("baseline-calibration.json");
        let sha = write_calibration(&out, &cal).unwrap();
        assert_eq!(sha.len(), 64, "{sha}");
        let loaded = load_calibration(Some(&out), None).unwrap();
        assert_eq!(loaded.calibration, cal);
        assert_eq!(loaded.sha256, sha);
        assert_eq!(
            loaded.sha256,
            crate::score::sha256_hex(&std::fs::read(&out).unwrap()),
            "the sealed digest must be the digest of the file on disk"
        );
        assert!(loaded
            .calibration
            .check_identity(&cal.track_id, &cal.box_name, "botany")
            .is_ok());
        // The temp file the atomic write used does not survive.
        assert!(!out.with_extension("json.tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);

        // One pass has no coefficient of variation at all.
        let err = calibration_from_passes(&identity, &[passes(&[0.001], &[0.030])], Vec::new())
            .unwrap_err();
        assert!(err.contains(BASELINE_CALIBRATION_INVALID), "{err}");
    }

    /// `benchd calibrate-baseline` over two goldens writes ONE file with one entry per prompt, in
    /// the order given, and that file loads back with both entries.
    #[test]
    fn calibration_authoring_writes_one_entry_per_prompt() {
        let identity = CalibrationIdentity {
            track_id: "track-a",
            box_name: "box-a",
            reference_commit: &"a".repeat(40),
            benchd_source_commit: &"b".repeat(40),
            captured_at: "2026-09-27T00:00:00Z",
        };
        let cal = calibration_from_passes(
            &identity,
            &[
                PromptPasses {
                    prompt: "botany",
                    prefill_legs: &[0.001, 0.001],
                    decode_legs: &[0.030, 0.030],
                },
                PromptPasses {
                    prompt: "kelp",
                    prefill_legs: &[0.002, 0.002],
                    decode_legs: &[0.060, 0.060],
                },
            ],
            Vec::new(),
        )
        .unwrap();
        let prompts: Vec<&str> = cal.prompts.iter().map(|e| e.prompt.as_str()).collect();
        assert_eq!(prompts, ["botany", "kelp"]);
        assert_eq!(cal.prompts[1].decode_seconds_per_token_mean, 0.060);

        let dir = std::env::temp_dir().join(format!(
            "benchd-calibration-prompts-test.{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("baseline-calibration.json");
        write_calibration(&out, &cal).unwrap();
        let sealed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
        assert_eq!(sealed["version"], CALIBRATION_VERSION);
        assert_eq!(sealed["prompts"][1]["prompt"], "kelp");
        assert_eq!(load_calibration(Some(&out), None).unwrap().calibration, cal);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// THE CALIBRATION FILE CARRIES EVERY GATE POINT, PER PASS (David 2026-09-17), in the same
    /// record shape a score's `metrics.gates` uses — `pass` in place of `pair`, and no leg. The
    /// file survives the write/read round trip with the records intact, and a file with NO gate
    /// points keeps the key set it always had.
    #[test]
    fn the_calibration_file_seals_the_gate_points_of_every_pass() {
        use crate::quiescegate::{CoolReading, GateRecord, QuiescenceReading};
        let identity = CalibrationIdentity {
            track_id: "mlx-qwen38",
            box_name: "box-3",
            reference_commit: &"a".repeat(40),
            benchd_source_commit: &"b".repeat(40),
            captured_at: "2026-09-17T00:00:00Z",
        };
        let prompts = [PromptPasses {
            prompt: "p",
            prefill_legs: &[0.001, 0.001],
            decode_legs: &[0.030, 0.030],
        }];
        let point = |pass: i64, phase: &str, waited: u64| GateRecord {
            pair: None,
            pass: Some(pass),
            leg: None,
            phase: phase.to_string(),
            quiescence: QuiescenceReading {
                state: "passed".to_string(),
                waited_seconds: waited,
                load: Some(0.40),
                gpu_util: Some(0.01),
                skip_reason: None,
            },
            cool: CoolReading {
                state: "passed".to_string(),
                waited_seconds: 0,
                gpu_temp_c: Some(38.0),
                skip_reason: None,
            },
        };
        let gates = vec![
            point(1, "prefill", 0),
            point(1, "decode", 15),
            point(2, "prefill", 30),
            point(2, "decode", 45),
        ];
        let cal = calibration_from_passes(&identity, &prompts, gates.clone()).unwrap();
        assert_eq!(cal.gates, gates);

        let dir = std::env::temp_dir().join(format!(
            "benchd-calibration-gates-test.{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("baseline-calibration.json");
        write_calibration(&out, &cal).unwrap();
        let loaded = load_calibration(Some(&out), None).unwrap();
        assert_eq!(
            loaded.calibration.gates, gates,
            "the file keeps every point"
        );
        let sealed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
        assert_eq!(sealed["gates"][2]["pass"], 2);
        assert_eq!(sealed["gates"][2]["phase"], "prefill");
        assert_eq!(sealed["gates"][2]["quiescence"]["waited_seconds"], 30);
        assert!(sealed["gates"][2].get("pair").is_none());
        assert!(sealed["gates"][2].get("leg").is_none());

        // NO gate points (the gates were off): the key is omitted, so the file's key set is the
        // one every calibration file before this change had.
        let ungated = calibration_from_passes(&identity, &prompts, Vec::new()).unwrap();
        let out = dir.join("ungated.json");
        write_calibration(&out, &ungated).unwrap();
        let sealed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
        assert!(sealed.get("gates").is_none());
        // ...and this benchd still reads it.
        assert!(load_calibration(Some(&out), None)
            .unwrap()
            .calibration
            .gates
            .is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
