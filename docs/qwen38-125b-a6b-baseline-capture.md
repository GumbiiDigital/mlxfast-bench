# Qwen 3.8 125B-A6B per-box calibration

This page tells the box agent how to calibrate ONE ranked box for the Qwen 3.8
125B-A6B tracks. One bench tree serves two engines. The track id selects the
platform: `qwen3.8-125b-a6b-mlx-v1` or `qwen3.8-125b-a6b-cuda-v1`.

Do the procedure once for each ranked box. Do it again when the organizer moves
the reference tree. Do not run it on a laptop. Run it on the ranked box only.

## 1. What a ranked run measures

These tracks store NO baseline pair. There is no pair in the track fixture and
no pair in a golden. A ranked run measures its own denominator (David ruling,
2026-09-08).

A ranked run measures the number of PAIRS the track fixture declares in
`official_pairs` — 2 on both platforms (David ruling, 2026-09-09) — on ONE box
in ONE job, on the one fixed prompt the fixture's live golden carries. Every
pair is the same two legs in the same order:

1. The SERIAL-CONTROL leg, on the organizer-staged reference tree. No
   speculation.
2. The CANDIDATE leg, on the submission tree, at its declared draft depth.

The two engines are strictly sequential and each leg loads the model once.
benchd sums each role's per-token times over the pairs, and the aggregate is
what the score, the floors and the bands read. Every control leg is band-checked
on its own. The score is the live ratio of those aggregates:

```
(ref_prefill_spt / cand_prefill_spt)^0.25 * (ref_decode_spt / cand_decode_spt)^0.75
```

The speedup floors and the acceptance bands do not change. The band shape is
prefill +5 % up, decode +2 % up. The down bands are DISABLED on both axes: a
control leg faster than its calibration is a well box and passes. What changed
is the reference the shape is applied to. It is a live measurement, not a
stored pair.

Leg 1 runs the reference tree's OWN engine and the reference tree's OWN weights.
benchd finds both by re-rooting the candidate's own root-relative path into the
reference tree, so the two legs differ by their tree and by nothing else. The
transform that writes the weights is participant-editable, so the control leg
never loads the candidate's transform output. When the weights are an
organizer-staged tree outside every checkout, and the reference tree holds no
`weights/` of its own, both legs load that one tree; no participant transform
can reach it.

## 2. What this page calibrates

The calibration file is a HEALTH BAND for leg 1 only. It says what a control leg
costs on this box when the box is well. A ranked run compares its measured
control leg against that band, and refuses the run when the leg is SLOWER than
the band's ceiling (`mean * band_high` on either axis). This is regression
detection only: a slow control leg means the box is not well. A control leg
faster than the calibration passes; the candidate is scored against that same
live leg, so a fast box hands the candidate nothing. The `*_band_low` values are
recorded by the calibrator and never read. No number in the file is ever a
denominator.

Each ranked box carries its own file. A file captured on another box is refused
by name.

## 3. Before you start

1. Stage the REFERENCE tree on the box. It is the track's promoted baseline
   engine commit, built. Record its path. This page calls it
   `$REFERENCE_WORKSPACE`.
2. Build the reference tree's own worker or adapter, at the same path inside the
   tree that a submission builds its own at:
   - MLX: `$REFERENCE_WORKSPACE/.build/release/bench-worker`, staged by that
     tree's `tools/stage-bench-worker.sh`.
   - CUDA: the adapter that tree's `tools/stage-cuda-engine.sh` stages under
     `$REFERENCE_WORKSPACE/.build/release/`.
3. Build benchd from the merged bench branch. Record its commit SHA.
4. Hold the box GPU lock for the whole procedure.
5. Export the track id, the box name and the benchd source commit:

```sh
export MLXFAST_QWEN_MTP_TRACK_ID=qwen3.8-125b-a6b-mlx-v1   # or ...-cuda-v1
export RUNNER_NAME=m5-max-128gb-4-qwen38-125b-a6b-mlx      # this box's runner name
export MLXFAST_BENCHD_SOURCE_COMMIT=<40-hex benchd commit>
```

## 4. Run the calibration

Run this ONCE on the box:

```sh
benchd calibrate-baseline \
  --baseline-workspace "$REFERENCE_WORKSPACE" \
  --engine .build/release/bench-worker \
  --golden "$LIVE_GOLDEN" \
  --passes 4 \
  --out "$REFERENCE_WORKSPACE/baseline-calibration.json"
```

`--weights` defaults to `$REFERENCE_WORKSPACE/weights`, the reference tree's own
transform output. Name another directory only for a track whose weights are an
organizer-staged tree outside every checkout.

`--engine` is a path RELATIVE to the reference workspace root. A ranked run
resolves the reference leg's engine by re-rooting the candidate's own relative
engine path into the reference tree, so the two paths must be the same.

The verb runs the ranked path's own serial-control leg `--passes` times, under
the full official methodology: the quiescence gate and the cool gate before every timed phase, the
unmeasured warm-up leg, the per-platform prefill warm-up count, one resident
worker per pass, and the live golden's own oracle. On CUDA it boots and tears
down the reference tree's resident engine once per pass (see section 7).

It writes the file and refuses by name when the passes are too noisy:

```
CALIBRATION-CV-EXCEEDED
```

A box that refuses is not quiet enough for a mean to describe it. Find out why
before you calibrate again. Do not widen the gate. The maximum is fixed at 1 %
per axis and no flag can relax it.

## 5. The file

`baseline-calibration.json` holds values only:

```json
{
  "version": 1,
  "track_id": "qwen3.8-125b-a6b-mlx-v1",
  "box": "m5-max-128gb-4-qwen38-125b-a6b-mlx",
  "reference_commit": "<40 hex>",
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
  "benchd_source_commit": "<40 hex>"
}
```

`box` must equal the runner name the ranked job runs under, and `prompt` must
equal the name of the golden the ranked run measures (the file name minus
`.golden.json`). A band describes the leg it was measured from, so a file
captured on another box or another prompt is refused. benchd reads the band from
the file; the values above are the defaults the calibrator writes. Previously
captured files retain their stored bands, including the old 0.98/1.02 decode
band. The CV fields are fractions: `0.004` is 0.4 %.

The re-rooting that finds the reference tree's engine and weights refuses a
candidate path that walks out of the workspace root with `..`, and refuses a
re-rooted path that resolves outside the reference workspace through a symlink.
The control leg runs the organizer's tree and nothing else.

## 6. Wire the box

A ranked job needs both of these:

| variable | meaning |
|---|---|
| `MLXFAST_BASELINE_WORKSPACE` | the built reference tree on this box |
| `MLXFAST_BASELINE_CALIBRATION` | this box's calibration file |

Both are required on the ranked path. benchd refuses by name when either is
absent or does not match this track and this box. The `--baseline-workspace`,
`--baseline-calibration` and `--box` flags name the same values on the command
line.

Leg 1 is serial, so it verifies its decode tokens against the SERIAL tape. On a
track that ships one oracle tape per draft depth, give leg 1 its own golden with
`--control-golden <PATH>`: pass the serial live golden there and keep the
depth-N golden on `--golden`. Pin it with `--control-golden-sha256` and
`--control-golden-bytes`, which work exactly like the `--golden` pin flags: give
both or neither. Without the flag, leg 1 verifies against `--golden`, which is
correct only when the depth-N tape is byte-identical to the serial tape. The
control golden must name the SAME prompt as `--golden`, and a run that measures
no control leg refuses the flag by name. The digest of the golden leg 1 used is
sealed as `metrics.baseline_golden_sha256`.

## 7. The per-leg resident engine (both platforms)

On EITHER platform the model is owned by a RESIDENT process, and that process
belongs to ONE leg's tree:

* CUDA: the worker is a thin adapter over a resident `ds4-resident`.
* MLX: the worker attaches to a resident `bench-worker` that holds the whole
  checkpoint, because an in-process load per phase is unaffordable.

So each leg needs its own resident from its own tree. benchd boots and tears one
down per leg through a fixed convention. The argv contract is IDENTICAL on both
platforms; only the script name differs. Both commands run with the working
directory set to that leg's workspace:

```
<workspace>/<script> --boot --spec <serial|mtp> --draft-len <N> --socket-out <FILE>
<workspace>/<script> --stop --socket <PATH>
```

| platform | script | socket injected as |
|---|---|---|
| MLX | `tools/resident-up.sh` | `BENCH_WORKER_RESIDENT_SOCKET` |
| CUDA | `tools/serve-up.sh` | `DS4_RESIDENT_SOCKET` and `BENCH_WORKER_RESIDENT_SOCKET` |

`--boot` boots exactly one resident, waits until it answers a healthy hello,
writes the resident's Unix-socket path as the first line of `<FILE>`, and exits
0 with the resident still running. `--stop` tears it down and is idempotent: a
resident that is already gone is not an error. Leg 1 is always booted
`--spec serial --draft-len 0`, whatever the submission declares. benchd runs
`--stop` when the leg ends, on success and on failure alike, so the two legs
never hold GPU memory at the same time.

On macOS the Seatbelt profile carries ONE `network-outbound` allowance, naming ONE socket path
literal, because Seatbelt counts an AF_UNIX connect as network. That profile is therefore built
PER SPAWN from THAT leg's socket: the reference leg may reach the reference resident and nothing
else, and the candidate leg may reach the candidate resident and nothing else. The rest of the
profile is unchanged, and its `(allow default)` base means the reference leg reads the reference
tree's own weights and metallib with no extra allowance — the only denied reads are the private
golden and the private dir.

Do NOT wrap benchd in the resident wrapper on the paired path. benchd runs it
itself, twice. A resident socket inherited from benchd's own environment is
refused by name (`LEG-SERVE-INHERITED-SOCKET`): one resident for both legs
prices the candidate against itself on CUDA, and on MLX it does not even reach a
number — the reference leg's worker refuses, because the resident holds the
candidate tree's weights and the leg asked for the reference tree's.

The refusal is scoped to the PAIRED path. The single-resident shapes are
untouched: a LOCAL UNSCORED run has one tree and one leg, so a measure script
that wraps benchd in `tools/resident-up.sh` for it is exactly right, and the
inherited socket is used as before.

## 8. Refusals, by name

| name | meaning |
|---|---|
| `BASELINE-WORKSPACE-MISSING` | no reference tree was named, or it is not a directory |
| `BASELINE-WORKSPACE-NO-ENGINE` | the tree holds no engine at the candidate's own relative path |
| `BASELINE-WORKSPACE-NO-WEIGHTS` | the tree holds no weights at the candidate's own relative path |
| `BASELINE-ENGINE-NOT-ROOT-RELATIVE` | the candidate engine is not addressable from the workspace root, or its re-rooted path leaves the reference tree |
| `BASELINE-WEIGHTS-NOT-ROOT-RELATIVE` | the candidate weights' re-rooted path leaves the reference tree |
| `BASELINE-CALIBRATION-MISSING` | no calibration file was named, or it could not be read |
| `BASELINE-CALIBRATION-INVALID` | the file is not a valid version-1 calibration |
| `BASELINE-CALIBRATION-TRACK-MISMATCH` | the file names another track |
| `BASELINE-CALIBRATION-BOX-MISMATCH` | the file was captured on another box |
| `BASELINE-CALIBRATION-PROMPT-MISMATCH` | the file was captured on another prompt than the golden this run measures |
| `BASELINE-BOX-UNRESOLVED` | neither `RUNNER_NAME` nor `--box` names this box |
| `CONTROL-GOLDEN-PROMPT-MISMATCH` | `--control-golden` names another prompt than `--golden` |
| `CONTROL-GOLDEN-WITHOUT-PAIRED-PATH` | `--control-golden` was given on a run that measures no control leg |
| `SERIAL-CONTROL-LEG-FAILED` | the control leg did not complete |
| `SERIAL-CONTROL-LEG-OUTSIDE-BAND` | the control leg is slower than this box's band ceiling |
| `GOLDEN-CARRIES-STORED-BASELINE` | the golden still declares a baseline pair |
| `STORED-BASELINE-OVERRIDE-REFUSED` | `MLXFAST_PAIRED_BASELINE_*` or `--baseline-*` reached the ranked path |
| `CALIBRATION-CV-EXCEEDED` | the calibration passes vary by more than 1 % |
| `LEG-SERVE-SCRIPT-MISSING` | a leg's tree holds no `tools/serve-up.sh` |
| `LEG-SERVE-BOOT-FAILED` | a leg's resident did not boot |
| `LEG-SERVE-INHERITED-SOCKET` | a resident socket was inherited instead of booted per leg |

## 9. Local runs

A participant iterates locally without a reference tree. `--mode local-iterate`
and `--mode local-submit` therefore keep working on these tracks:

* With NO reference tree named, the run measures the CANDIDATE LEG ONLY and
  seals NO score. It writes the real timings, the real correctness verdict,
  `score: null`, an empty `metrics.error`, and
  `baseline_source: "none (local mode: unscored)"`. The printed summary reports
  the candidate leg in tokens per second. The run is not a failure; it simply
  has no denominator, and a score with no denominator is not a score.
* With BOTH `MLXFAST_BASELINE_WORKSPACE` and `MLXFAST_BASELINE_CALIBRATION`
  present, the local modes run the FULL PAIRED PATH, exactly as ranked: the
  control leg, the band check, the candidate leg, and the live ratio.

Half the inputs is not the paired path. One leg cannot be checked against a band
that is not there, so a run with only one of the two takes the unscored branch.

## 10. What the run seals

A ranked paired run seals these fields in `score.json` `metrics`:

- `baseline_source` — `serial-control-leg`.
- `baseline_box`, `baseline_calibration_sha256`, `baseline_reference_commit`.
- `baseline_band_passed`.
- `baseline_leg_prefill_seconds_per_token`,
  `baseline_leg_decode_seconds_per_token` — the scored pair's serial-control
  leg, per token.
- `candidate_leg_prefill_seconds_per_token`,
  `candidate_leg_decode_seconds_per_token` — the scored pair's candidate leg,
  per token. The scored pair is the one whose composite is the lower median
  over the pairs. Pairs are never averaged.
- `paired_legs` — one row per pair, in order: `pair`,
  `control_prefill_seconds_per_token`, `control_decode_seconds_per_token`,
  `candidate_prefill_seconds_per_token`, `candidate_decode_seconds_per_token`,
  as measured. The track fixture's `official_pairs` sets the row count (2 on
  both platforms, David ruling 2026-09-09). A run that stops early keeps the
  pairs it measured in this list and seals no score.

- `decode_speedup_floor`, `prefill_speedup_floor` — the two floors this run
  enforced, from the track fixture's `decode_speedup_floor` and
  `prefill_speedup_floor` (0.95 and 0.95, David ruling 2026-09-09). The
  fixture is the only source: a fixture that declares neither refuses the run.
- `passed_decode_speedup_floor`, `passed_prefill_speedup_floor` — whether each
  live speedup reached its own floor. The floor beside each flag is the floor
  the flag was decided against.

`baseline_prefill_seconds_per_token` and `baseline_decode_seconds_per_token`
carry the serial-control mean values, so the board reads them unchanged.
`prefill_speedup`, `decode_speedup` and the score are the live ratios. A run
that misses either floor seals no score.
