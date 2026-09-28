//! THE SCORED-PROMPT ROTATION: which prompt of a track's hidden pool a ranked job scores, and the
//! pins benchd resolves for it.
//!
//! A track fixture may declare
//!
//! ```json
//! "live_golden": "botany",
//! "live_golden_rotation": {"mode": "per_job_random", "pool": ["botany", ...]},
//! "calibration_prompt": "botany",
//! "speculative_oracles": {"botany": {"mtp1": {"r2_path": ..., "sha256": ..., "bytes": ...}, ...}}
//! ```
//!
//! The trusted workflow draws one pool name per job (from `/dev/urandom`, never from anything a
//! participant sees) and hands it to benchd as `--live-prompt`. benchd then resolves that prompt's
//! SERIAL pin (its `timed_prompt_pool[]` entry, `<name>.golden.json`) and its per-depth ORACLE pin
//! (`speculative_oracles.<name>.mtp<d>`) from the fixture ITSELF, so the pins a run is verified
//! against never come from the command line alone.
//!
//! FAIL-CLOSED OVER THE WHOLE POOL. Any pool name may be drawn, so a fixture whose pool holds a
//! prompt without its serial pin, or without an oracle for EVERY permitted draft depth, is refused
//! at load — not in the one job that happens to draw it.
//!
//! A fixture with no `live_golden_rotation` key has no rotation: [`rotation_from_contract`] returns
//! `Ok(None)` and nothing here applies.

use crate::golden::GoldenIntegrityPin;
use serde_json::Value;
use std::collections::BTreeMap;

/// The only rotation mode defined.
pub const ROTATION_MODE_PER_JOB_RANDOM: &str = "per_job_random";

/// The fixture's validated rotation: every pool prompt fully pinned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LivePromptRotation {
    /// `live_golden`, the default prompt. Always a pool member.
    pub live_golden: String,
    /// `live_golden_rotation.pool`, in fixture order.
    pub pool: Vec<String>,
    /// `calibration_prompt` (default `live_golden`): the prompt every box's calibration band is
    /// captured on, whichever pool prompt the job drew.
    pub calibration_prompt: String,
    serial: BTreeMap<String, GoldenIntegrityPin>,
    oracles: BTreeMap<String, BTreeMap<u32, GoldenIntegrityPin>>,
}

/// The pins one run resolves for its drawn prompt at its declared depth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedLivePrompt {
    /// The drawn prompt's name — public (one of the pool) and sealed as `metrics.live_prompt`.
    pub name: String,
    /// The prompt's serial golden: the serial-control leg's tape.
    pub serial: GoldenIntegrityPin,
    /// The candidate leg's timed oracle: the serial golden at depth 0, the per-depth oracle
    /// otherwise.
    pub oracle: GoldenIntegrityPin,
    /// The fixture's calibration prompt (see [`LivePromptRotation::calibration_prompt`]).
    pub calibration_prompt: String,
}

fn pin_of(entry: &Value, label: &str) -> Result<GoldenIntegrityPin, String> {
    let sha256 = entry.get("sha256").and_then(Value::as_str).unwrap_or("");
    let bytes = entry.get("bytes").and_then(Value::as_u64).unwrap_or(0);
    let hex = sha256.len() == 64
        && sha256
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    if !hex || bytes == 0 {
        return Err(format!(
            "{label} is unpinned or malformed (sha256 {sha256:?}, bytes {bytes}); a rotation pool \
             prompt must be fully pinned before it can be drawn"
        ));
    }
    Ok(GoldenIntegrityPin {
        sha256: sha256.to_string(),
        bytes,
    })
}

fn r2_basename(entry: &Value) -> Option<&str> {
    entry
        .get("r2_path")
        .and_then(Value::as_str)
        .and_then(|p| p.rsplit('/').next())
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

/// Read and validate the fixture's rotation from the contract BYTES. `Ok(None)` when the fixture
/// declares no `live_golden_rotation`; every malformed or under-pinned rotation is an error that
/// names the prompt and the missing pin.
pub fn rotation_from_contract(bytes: &[u8]) -> Result<Option<LivePromptRotation>, String> {
    let root: Value = serde_json::from_slice(bytes)
        .map_err(|e| format!("contract could not be decoded for its live_golden_rotation: {e}"))?;
    let Some(rotation) = root.get("live_golden_rotation") else {
        return Ok(None);
    };
    let mode = rotation.get("mode").and_then(Value::as_str).unwrap_or("");
    if mode != ROTATION_MODE_PER_JOB_RANDOM {
        return Err(format!(
            "contract live_golden_rotation.mode is {mode:?}; only {ROTATION_MODE_PER_JOB_RANDOM:?} \
             is defined"
        ));
    }
    let pool: Vec<String> = match rotation.get("pool").and_then(Value::as_array) {
        Some(items) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .filter(|n| valid_name(n))
                    .map(str::to_string)
                    .ok_or_else(|| {
                        format!(
                            "contract live_golden_rotation.pool entry {v} is not a [a-z0-9_-]+ name"
                        )
                    })
            })
            .collect::<Result<_, _>>()?,
        None => Vec::new(),
    };
    if pool.is_empty() {
        return Err(
            "contract live_golden_rotation.pool is empty or absent; there is nothing to draw"
                .to_string(),
        );
    }
    for (i, name) in pool.iter().enumerate() {
        if pool[..i].contains(name) {
            return Err(format!(
                "contract live_golden_rotation.pool names {name:?} twice"
            ));
        }
    }
    let live_golden = root
        .get("live_golden")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if !pool.contains(&live_golden) {
        return Err(format!(
            "contract live_golden {live_golden:?} (the default prompt) is not in \
             live_golden_rotation.pool {pool:?}"
        ));
    }
    let depths: Vec<u32> = root
        .pointer("/mtp_head/permitted_draft_depths")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_u64)
                .filter_map(|d| u32::try_from(d).ok())
                .collect()
        })
        .unwrap_or_default();
    if depths.is_empty() || depths.contains(&0) {
        return Err(
            "contract declares live_golden_rotation but no positive mtp_head.permitted_draft_depths; \
             the per-depth oracles every pool prompt needs cannot be enumerated"
                .to_string(),
        );
    }
    let timed_pool = root
        .get("timed_prompt_pool")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let serial_of = |name: &str| -> Result<GoldenIntegrityPin, String> {
        let file = format!("{name}.golden.json");
        let mut matches = timed_pool
            .iter()
            .filter(|e| r2_basename(e) == Some(file.as_str()));
        let entry = matches.next().ok_or_else(|| {
            format!(
                "rotation prompt {name:?} has no timed_prompt_pool entry at {file}; its serial \
                 golden carries no pin"
            )
        })?;
        if matches.next().is_some() {
            return Err(format!(
                "rotation prompt {name:?} matches more than one timed_prompt_pool entry at {file}"
            ));
        }
        pin_of(entry, &format!("timed_prompt_pool[{file}]"))
    };

    let mut serial = BTreeMap::new();
    let mut oracles = BTreeMap::new();
    for name in &pool {
        serial.insert(name.clone(), serial_of(name)?);
        let mut per_depth = BTreeMap::new();
        for &d in &depths {
            let label = format!("speculative_oracles.{name}.mtp{d}");
            let entry = root
                .get("speculative_oracles")
                .and_then(|o| o.get(name))
                .and_then(|o| o.get(format!("mtp{d}")))
                .ok_or_else(|| {
                    format!(
                        "rotation prompt {name:?} has no per-depth oracle {label}; every pool \
                         prompt needs one timed oracle per permitted draft depth"
                    )
                })?;
            let want = format!("{name}.mtp{d}.golden.json");
            if r2_basename(entry) != Some(want.as_str()) {
                return Err(format!(
                    "{label} names r2_path {:?}, not .../{want}",
                    entry.get("r2_path")
                ));
            }
            per_depth.insert(d, pin_of(entry, &label)?);
        }
        oracles.insert(name.clone(), per_depth);
    }
    let calibration_prompt = match root.get("calibration_prompt") {
        None => live_golden.clone(),
        Some(v) => v
            .as_str()
            .filter(|n| valid_name(n))
            .ok_or_else(|| format!("contract calibration_prompt {v} is not a prompt name"))?
            .to_string(),
    };
    serial_of(&calibration_prompt).map_err(|e| format!("contract calibration_prompt: {e}"))?;
    Ok(Some(LivePromptRotation {
        live_golden,
        pool,
        calibration_prompt,
        serial,
        oracles,
    }))
}

impl LivePromptRotation {
    /// The pins for `name` at draft `depth` (0 = serial). Refuses a name outside the pool and a
    /// depth the fixture does not permit.
    pub fn resolve(&self, name: &str, depth: u32) -> Result<ResolvedLivePrompt, String> {
        let serial = self.serial.get(name).ok_or_else(|| {
            format!(
                "--live-prompt {name:?} is not in the contract's live_golden_rotation.pool {:?}",
                self.pool
            )
        })?;
        let oracle = if depth == 0 {
            serial.clone()
        } else {
            self.oracles
                .get(name)
                .and_then(|m| m.get(&depth))
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "--live-prompt {name:?} has no timed oracle at draft depth {depth}; the \
                         contract permits no such depth"
                    )
                })?
        };
        Ok(ResolvedLivePrompt {
            name: name.to_string(),
            serial: serial.clone(),
            oracle,
            calibration_prompt: self.calibration_prompt.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pin(c: char, bytes: u64) -> Value {
        json!({"sha256": c.to_string().repeat(64), "bytes": bytes})
    }

    fn fixture(pool: &[&str], pinned: &[&str]) -> Value {
        let mut timed = Vec::new();
        let mut oracles = serde_json::Map::new();
        for (i, name) in ["botany", "beagle", "travel"].iter().enumerate() {
            let mut e = pin('a', 100 + i as u64);
            e["r2_path"] = json!(format!("p/{name}.golden.json"));
            timed.push(e);
            if pinned.contains(name) {
                let mut per = serde_json::Map::new();
                for d in 1..=3u64 {
                    let mut e = pin(char::from(b'0' + d as u8), 1000 * (i as u64 + 1) + d);
                    e["r2_path"] = json!(format!("p/{name}.mtp{d}.golden.json"));
                    per.insert(format!("mtp{d}"), e);
                }
                oracles.insert(name.to_string(), Value::Object(per));
            }
        }
        json!({
            "live_golden": "botany",
            "live_golden_rotation": {"mode": "per_job_random", "pool": pool},
            "timed_prompt_pool": timed,
            "speculative_oracles": oracles,
            "mtp_head": {"permitted_draft_depths": [1, 2, 3]},
        })
    }

    fn load(v: &Value) -> Result<Option<LivePromptRotation>, String> {
        rotation_from_contract(&serde_json::to_vec(v).unwrap())
    }

    #[test]
    fn no_rotation_key_is_no_rotation() {
        let mut v = fixture(&["botany"], &["botany"]);
        v.as_object_mut().unwrap().remove("live_golden_rotation");
        assert_eq!(load(&v).unwrap(), None);
    }

    #[test]
    fn resolves_each_pool_prompt_per_depth() {
        let r = load(&fixture(&["botany", "beagle"], &["botany", "beagle"]))
            .unwrap()
            .unwrap();
        assert_eq!(r.calibration_prompt, "botany");
        let serial = r.resolve("beagle", 0).unwrap();
        assert_eq!(serial.name, "beagle");
        assert_eq!(serial.serial.bytes, 101);
        assert_eq!(serial.oracle, serial.serial);
        let mtp2 = r.resolve("beagle", 2).unwrap();
        assert_eq!(mtp2.serial.bytes, 101);
        assert_eq!(mtp2.oracle.bytes, 2002);
        assert_eq!(mtp2.oracle.sha256, "2".repeat(64));
        assert_eq!(r.resolve("botany", 3).unwrap().oracle.bytes, 1003);
    }

    #[test]
    fn refuses_a_name_outside_the_pool_and_an_unpermitted_depth() {
        let r = load(&fixture(&["botany"], &["botany", "beagle"]))
            .unwrap()
            .unwrap();
        let e = r.resolve("beagle", 1).unwrap_err();
        assert!(
            e.contains("not in the contract's live_golden_rotation.pool"),
            "{e}"
        );
        let e = r.resolve("botany", 4).unwrap_err();
        assert!(e.contains("no timed oracle at draft depth 4"), "{e}");
    }

    #[test]
    fn refuses_a_pool_prompt_without_its_oracles() {
        let e = load(&fixture(&["botany", "beagle"], &["botany"])).unwrap_err();
        assert!(
            e.contains("\"beagle\" has no per-depth oracle speculative_oracles.beagle.mtp1"),
            "{e}"
        );
    }

    #[test]
    fn refuses_a_pool_prompt_missing_one_depth() {
        let mut v = fixture(&["botany", "beagle"], &["botany", "beagle"]);
        v["speculative_oracles"]["beagle"]
            .as_object_mut()
            .unwrap()
            .remove("mtp3");
        let e = load(&v).unwrap_err();
        assert!(e.contains("speculative_oracles.beagle.mtp3"), "{e}");
    }

    #[test]
    fn refuses_an_unpinned_oracle_or_serial() {
        let mut v = fixture(&["botany", "beagle"], &["botany", "beagle"]);
        v["speculative_oracles"]["beagle"]["mtp2"]["sha256"] = json!("");
        assert!(load(&v).unwrap_err().contains("unpinned or malformed"));
        let mut v = fixture(&["botany", "beagle"], &["botany", "beagle"]);
        v["timed_prompt_pool"][1]["bytes"] = json!(0);
        assert!(load(&v)
            .unwrap_err()
            .contains("timed_prompt_pool[beagle.golden.json]"));
    }

    #[test]
    fn refuses_a_pool_prompt_without_a_serial_golden() {
        let e = load(&fixture(&["botany", "nosuch"], &["botany"])).unwrap_err();
        assert!(
            e.contains("\"nosuch\" has no timed_prompt_pool entry"),
            "{e}"
        );
    }

    #[test]
    fn refuses_an_oracle_borrowed_from_another_prompt() {
        let mut v = fixture(&["botany", "beagle"], &["botany", "beagle"]);
        v["speculative_oracles"]["beagle"] = v["speculative_oracles"]["botany"].clone();
        assert!(load(&v)
            .unwrap_err()
            .contains("not .../beagle.mtp1.golden.json"));
    }

    #[test]
    fn refuses_malformed_rotation_shapes() {
        let mut v = fixture(&["botany"], &["botany"]);
        v["live_golden_rotation"]["mode"] = json!("round_robin");
        assert!(load(&v).unwrap_err().contains("only \"per_job_random\""));
        let v = fixture(&[], &["botany"]);
        assert!(load(&v).unwrap_err().contains("pool is empty"));
        let v = fixture(&["botany", "botany"], &["botany"]);
        assert!(load(&v).unwrap_err().contains("twice"));
        let v = fixture(&["beagle"], &["beagle"]);
        assert!(load(&v)
            .unwrap_err()
            .contains("is not in live_golden_rotation.pool"));
        let mut v = fixture(&["botany"], &["botany"]);
        v["calibration_prompt"] = json!("nosuch");
        assert!(load(&v).unwrap_err().contains("calibration_prompt"));
    }
}
