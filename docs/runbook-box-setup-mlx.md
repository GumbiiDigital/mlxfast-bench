# Runbook: stand up a Mac (MLX) ranked box

**Class:** runbook. Follow it as written. Overview and the Spark counterpart:
[`box-setup-runbook.md`](box-setup-runbook.md), [`runbook-box-setup-cuda.md`](runbook-box-setup-cuda.md).

A Mac joins the ranked network by becoming a self-hosted GitHub Actions
runner of the MLX engine repository. Yukon dispatches that repository's
`benchmark.yml` for every submission; the workflow selects a runner by label.
The engine is the mlxfast Swift runtime; the model is the published MLX 4-bit
checkpoint, which the engine's `setup.sh` downloads and verifies in the job.

## 1. Base box

- An Apple Silicon Mac with enough unified memory for the target model and its
  working set (the checkpoint is 113 GB across 32 pinned files) and at least
  260 GiB of free disk before the first download.
- macOS 14 or later; Swift 6 (`swift-tools-version: 6.3`).
- Full `Xcode.app` with the Metal toolchain. Full Xcode cannot be installed
  unattended (an Apple ID is needed), so an operator stages it. Then
  `xcode-select -p` must point at its Developer directory (the Command Line
  Tools instance cannot build Metal), the license must be accepted
  (`sudo xcodebuild -license accept`), and `xcrun -sdk macosx metal -v` must
  run. The `m5-machine-scripts` `xcode-toolchain` converge unit checks these
  four facts and fails closed when `Xcode.app` is absent.
- CMake and Git (`setup.sh` installs CMake through Homebrew when missing).
- `macmon` at `/opt/homebrew/bin/macmon`: the workflow reads GPU utilization
  and temperature from it for quiescence and the cool gate, and refuses to
  time when the reader stops returning samples.
- No Rust: the pair arrives prebuilt.

## 2. Accounts and the GPU lock

Same posture as the Spark: an operator account owns the staged assets, a
`bench` group reads them, and `/tmp/mtplx-gpu-exclusive.lock` is a regular
file taken with `flock` by every GPU user, including calibration and local
runs.

## 3. Weights

Nothing to stage by hand. The engine's `./setup.sh` fetches the public MLX
4-bit checkpoint anonymously from Hugging Face (no token by default;
`MLXFAST_REFERENCE_AUTH_HEADER` is an optional fallback for a private mirror or
rate limiting), verifies every file by sha256 and bytes, and the engine's
`transform` command writes the `weights/` tree the worker loads. The MTP head
ships inside the checkpoint; there is no separate head file.

## 4. The benchmarker pair

The ranked job holds no credential and the bench repository is private, so
the job cannot download the pair. Stage the pair on the box before the first
job:

1. On a machine with access to the bench repository, check out the track's
   dist channel (`qwen3.8-125b-a6b-v1`) and copy `dist/benchd` and
   `dist/benchd.manifest.json` to the box, into the directory that
   `BENCHD_BIN_DIR` names. Set the mode of `benchd` to 755.
2. The job runs `./tools/fetch-benchd.sh`. When the pair is present, the tool
   verifies the binary against the manifest beside it and uses it. It does not
   contact the channel.

To move the box to a new channel tip, replace both files. A binary that does
not match its manifest refuses to run.

## 5. Goldens

The goldens are organizer material. They are in R2, under the object keys that
`r2_path` names in the track fixture. They are never in git, and the ranked job
holds no credential, so the box stages them one time, out of band.

Stage them with the `goldens` converge unit in `m5-machine-scripts`. Run
`converge --check --unit goldens` to see what is missing. Then run
`converge --apply --unit goldens` to place the pinned files. The unit writes
each file read-only, refuses a file that does not match its pin, and deletes
nothing.

Set `MLXFAST_QWEN38_GOLDEN_DIR` to that directory in the runner service
environment (section 6). The workflow refuses to start when the variable is
missing. Before every ranked run, `tools/ranked-box-preflight.sh` compares the
byte count and the sha256 of each staged file with the fixture pin. It also
refuses the directory when the directory holds one more `*.json`, because the
job passes every `*.json` there as a golden.

## 6. Runner registration and service

- Register the runner to the MLX engine repository with the labels
  `self-hosted, macOS, <track id>` (today `qwen3.8-125b-a6b-mlx-v1`).
- Run it under the root LaunchDaemon supervisor from
  `m5-machine-scripts/runner-isolation`: single-use registration through the
  GitHub App, one job as the unprivileged `benchrunner`, reset between jobs.
- The runner environment file (`.env` in the runner directory, one
  `NAME=value` per line; restart the service after a change) must carry:
  - `BENCHD_BIN_DIR`: the directory with the staged pair (section 4).
  - `MLXFAST_QWEN38_GOLDEN_DIR`: the directory with the staged goldens
    (section 5).
  - `MLXFAST_REFERENCE_DIR`: the reference checkpoint directory itself. The
    first job verifies the full checkpoint hash once and writes a trusted
    stamp in that directory; later jobs skip the hash while the stamp is
    present.
  - `MLXFAST_BASELINE_WORKSPACE`: the organizer-staged reference tree, built
    (section 7).
  - `MLXFAST_BASELINE_CALIBRATION`: this box's `baseline-calibration.json`
    (section 7).
  - `MLXFAST_MACMON_BIN`: the `macmon` binary when it is not at
    `/opt/homebrew/bin/macmon`.
  - `MLXFAST_FORK_MIRROR`: a bare mirror of the engine fork repository,
    staged out of band (`git clone --mirror`). The job resolves the
    `Vendor/mlx-swift-lm` submodule from it without a credential. A mirror
    that does not carry the pinned commit fails the job; re-stage it.
  - `MLXFAST_METALLIB_STAGE`, on a box without full Xcode: a directory with
    `mlx.metallib` and `mlx.metallib.fingerprint` built from the engine's
    vendored tree. Dispatch such a box with the workflow input
    `prestaged_metallib: true`; the job accepts the library only when the
    fingerprint matches the checkout.
- The runner service PATH must carry `swift`, `git`, `jq`, `python3`, `bc`
  and `shasum`. The Command Line Tools are sufficient when the Metal library
  is pre-staged.
- The job keeps the worker build between jobs in a content-keyed cache under
  the runner user's `~/.cache/mlxfast-engine-build`. A box with a staged
  engine checkout can seed the cache with `tools/build-cache.sh save`.

## 7. The reference tree and this box's health band

The track pins no serial pair. A ranked run measures the pairs the track fixture
declares in `official_pairs` — 2 on both platforms — and every pair times a
serial-control leg on the reference tree and then the candidate leg. The score
is the live ratio.

So the box carries the organizer-staged reference tree, built, and its own
calibration file. Write the file once on the box, and again whenever the
organizer moves the reference tree:

```bash
flock /tmp/mtplx-gpu-exclusive.lock -c '
  benchd calibrate-baseline \
    --baseline-workspace "$REFERENCE_WORKSPACE" \
    --engine .build/release/bench-worker \
    --golden "$LIVE_GOLDEN" \
    --passes 4 \
    --out "$REFERENCE_WORKSPACE/baseline-calibration.json"'
```

The file is a health band for the control leg, never a denominator. Its decode
mean is the decode window per token (decode-run time / N, no seed prefill). The
verb refuses by name (`CALIBRATION-CV-EXCEEDED`) when the box is too noisy for a
mean to describe it. benchd refuses a file older than version 3 by name
(`BASELINE-CALIBRATION-WHOLE-WINDOW-DECODE`): it recorded the whole window as
decode. Recalibrate. The full procedure is in
[`qwen38-125b-a6b-baseline-capture.md`](qwen38-125b-a6b-baseline-capture.md).

## 8. The measurement topology

The engine's `bench-worker` runs as a resident that every phase attaches to, so
the model loads once. Three concurrent fresh workers would need about 190 GiB,
which is what the resident prevents. WHO starts it depends on the path:

- **A PAIRED run** measures two legs from two trees, so benchd boots ONE
  resident PER LEG from that leg's own tree — `<workspace>/tools/resident-up.sh
  --boot …` before the leg, `--stop …` after it — and injects that leg's socket
  into that leg's worker spawns only. The measure script must NOT wrap benchd on
  this path: a single resident serves the candidate tree's weights, and the
  reference leg's worker refuses them by name. An inherited
  `BENCH_WORKER_RESIDENT_SOCKET` is refused (`LEG-SERVE-INHERITED-SOCKET`).
- **A LOCAL UNSCORED run** has one tree and one leg, so the measure script's
  own `tools/resident-up.sh` wrap is exactly right and is untouched.

The GPU lock is held by the outermost process for the whole window either way.

## 9. Local checks

```bash
MLXFAST_ENGINE_BIN=.build/release/mlxfast-runtime-worker \
MLXFAST_CORRECTNESS_GOLDEN_PATH=correctness_prompts/public_longcopy_gate_english_1024_256.json \
  ./benchmark.sh --local-iterate
```

The local test has no default golden; the variable must be set. Do not pass
`--golden`, `--weights` or `--score-path`; the script rejects them.

## 10. Dispatch and receipt

```bash
gh workflow run benchmark.yml --ref <engine branch>
```

The box job runs the preflight, fetches the pair, runs `setup.sh` (toolchain
check, Swift build, `mlx.metallib`, checkpoint download and verification),
waits for quiescence, and measures. The run is paired: it measures the pairs the
track fixture declares, and each pair times the reference tree and then the
submission tree. A serial declaration puts two serial legs against each other,
so the readiness receipt is a passing run that scores near 1.00.

## 11. Readiness checklist

- [ ] macOS 14+, Swift 6, full `Xcode.app` selected, license accepted, `xcrun -sdk macosx metal -v` runs
- [ ] CMake, Git, and `macmon` at `/opt/homebrew/bin/macmon`
- [ ] at least 260 GiB free before the first `setup.sh`
- [ ] operator account and `bench` group; GPU lock present and group-readable
- [ ] benchd pair installed by `tools/fetch-benchd.sh` for darwin, manifest beside it
- [ ] pool goldens staged in `MLXFAST_QWEN38_GOLDEN_DIR` and passing `tools/ranked-box-preflight.sh`
- [ ] runner registered with labels `self-hosted, macOS, <track id>` and online, under the LaunchDaemon supervisor
- [ ] iogpu wired limit pinned by the boot daemon (`sysctl iogpu.wired_limit_mb`)
- [ ] `./setup.sh` completes: toolchain, Metal kernels, checkpoint verified, `weights/` transformed
- [ ] local `benchmark.sh --local-iterate` passes correctness on the public golden
- [ ] reference tree built at `baseline_reference_commit` and calibrated on this box with CV at or under 2 %; `MLXFAST_BASELINE_WORKSPACE` and `MLXFAST_BASELINE_CALIBRATION` in the runner `.env`
- [ ] one `workflow_dispatch` of `benchmark.yml` passes end to end
- [ ] sealed serial score near 1.00 on the paired run

## Reading local failures

Local `iterate` results include `metrics.local_phases` alongside the legacy score
fields. Inspect it with:

```sh
jq '{passed, score, phases: .metrics.local_phases, error: .metrics.error}' score.local-iterate.json
```

`correctness` describes the untimed conformance gate; `timing` describes completion
of the measured prefill/decode pair. Each is `not_run`, `passed`, or `failed`.
`correctness_checked_steps` counts the conformance steps actually checked; it is
`null` if the gate started but a protocol/transport error prevented a report.
A timing result is `not_run` when no timing cool gate cleared, which includes a
timing failure raised before the first gate (a parameter or worker-spawn error:
read `metrics.error`), or `failed` when a started measurement did not complete
(including a later decode cool-gate abort). Old artifacts and official paths do not carry this local-only field.

For example, a prefill thermal abort after conformance can correctly report
`correctness: "passed"` and `timing: "not_run"`. The legacy `checked_steps` remains
zero on that failure path for Swift checked-timing parity; it is not evidence that
the separate conformance gate was skipped. `passed_correctness` remains false if
a failed gate is followed by a phase-close error, or a timed-only pass fails without
running the gate. A successful local measurement without a baseline still has
`score: null`; consult `passed`, phase outcomes, and `error` together.

While cooling, benchd prints the phase, current and minimum GPU temperature,
target, wall-clock elapsed time, and logical gate wait on each usable sample.
The MLX threshold is 50 C, with the existing polling, stall-abort, and maximum
wait rules. Progress output does not relax or bypass the gate.
