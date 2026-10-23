# Paired-run anti-cheat guards

These track-fixture keys are optional. If they are absent, the prior paired-run behavior and
output stay unchanged.

- `decode_speedup_ceiling` and `prefill_speedup_ceiling`: each must be finite and greater than
  `1.0`. A pair above a declared ceiling refuses with `SPEEDUP-ABOVE-CEILING`; equality passes.
- `seed_window_max_ratio`: must be finite and greater than `1.0`. Each control and candidate leg
  must report a seed-prefill window whose per-token time, divided by timed prefill per-token time,
  is finite and no greater than the limit. Refusals are `SEED-WINDOW-NOT-OBSERVED` and
  `SEED-WINDOW-ABOVE-LIMIT`. Six honest M5 pairs measured `1.13` to `1.22` on both legs.
- `decode_window_includes_seed: true`: scores and calibrates decode as the seed-prefill window plus
  the decode window, divided by the decode token count. Both legs use benchd's clock. A missing
  split refuses with `SEED-WINDOW-NOT-OBSERVED`. Absent or `false` keeps decode-window scoring.
- `require_private_dir: true`: `MLXFAST_PRIVATE_DIR` must be an existing directory, every candidate
  and control golden, plus the warm-up golden when declared, must resolve inside it, and sandbox
  bypass/profile overrides are refused. The directory and golden paths must be valid UTF-8; one
  validated canonical directory string is carried into every sandbox plan.
  Refusals are `PRIVATE-DIR-REQUIRED`, `GOLDEN-OUTSIDE-PRIVATE-DIR`, and
  `SANDBOX-OVERRIDE-REFUSED`.
- `warmup_golden: {"sha256":"...","bytes":N}`: pins one extra golden used only for the
  unmeasured full warm-up pass and untimed prefill warm-ups. `benchd iterate` and
  `calibrate-baseline` require `--warmup-golden` when this key is present and refuse the flag when
  it is absent. The file must match the fixture pin. `bytes` is optional.
- `decode_golden_offset`: must be at least `1` and less than the live golden count. Pair `g`
  prefills golden `g` and decodes golden `(g + offset) mod n`, for both control and candidate legs.
  Calibration uses the same composition while keeping the band key at prompt `g`.

When the new warm-up/composition keys are active, benchd checks all token sequences before the
first leg boots. It refuses with `PROMPT-REUSE-REFUSED` if the warm-up prefill or seed equals any
live prefill or seed, or if a composed pair's timed prefill equals its timed decode seed. This makes
each scored prompt or seed new to that engine process when its timer starts, so a process cannot
turn the warm-up into a replay cache for the scored window.
The resolved `official_prefill_timed_runs` must be exactly `1` whenever `warmup_golden` or
`decode_golden_offset` is declared; repeated timed prefill refuses with `PROMPT-REUSE-REFUSED`.

Leg teardown is always checked, even when none of these keys is present. After the stop script
succeeds, benchd waits up to two seconds for the resident socket to stop accepting connections. A
failed normal teardown refuses with `LEG-TEARDOWN-FAILED`; stop stderr is withheld. Cleanup on
an already-failed path remains best effort so it does not replace the first failure.

The runtime-worker Seatbelt profile also denies POSIX shared memory, System V IPC, POSIX named
semaphores, preference writes, and Mach lookups. It allows only the five Mach services required by
MLX: `com.apple.MTLCompilerService`, the two OpenDirectory services, `com.apple.logd`, and
`com.apple.diagnosticd`. These rules stop one worker from leaving state for a later worker.
It also denies signals to other processes and scheduling changes.

Public engine identity digest keys keep their existing names, but their values are now
the fixed `engine-value-withheld` marker. This applies to `head_provenance_sha256` and
`runner_manifest_sha256`, including the per-prompt head mirror. An engine-supplied 64-character
hex value is not treated as trusted proof and cannot affect the public marker. The raw engine value
is written only to the private audit when `BENCHD_PRIVATE_AUDIT_PATH` is configured.

The worker deny set includes the private audit file, its parent directory when that parent is not
the worker's own workspace, and the whole `MLXFAST_RESIDENT_LOG_ROOT`. An official candidate run
refuses when its private audit is inside the candidate-readable workspace. CLI-selected baseline,
calibration, candidate/control golden, and warm-up golden paths are denied in addition to the
environment-selected paths. Only an explicit control leg whose executable resolves inside the
trusted reference tree may read that reference tree.
