# Box setup runbook: DGX Spark (CUDA) and Mac (MLX)

**Class:** runbook. Follow it as written.

This document is the overview: what both box types share, the box
calibration, the differences at a glance, and the faults seen in practice. The step-by-step procedure per platform, each with a readiness
checklist, is in its own runbook:

- [`runbook-box-setup-cuda.md`](runbook-box-setup-cuda.md) for a DGX Spark
- [`runbook-box-setup-mlx.md`](runbook-box-setup-mlx.md) for a Mac

The software chain from a runner (in the fork, with the track's editable copy in the engine repo) to a scored Yukon row, done once per
track rather than once per box, is [`runbook-new-engine.md`](runbook-new-engine.md).

## 1. What both box types need

| item | where it comes from | notes |
|---|---|---|
| the benchmarker pair (`benchd`, `record-correctness-golden`) | this repository's dist channel, one directory per platform, resolved by the engine's `tools/fetch-benchd.sh` | the pair is verified against its own `benchd.manifest.json` (sha256, bytes, `target_triple`); while the track is internal the fetch needs `GITHUB_TOKEN` or `BENCHD_DIST_TOKEN` |
| the GPU lock | `/tmp/mtplx-gpu-exclusive.lock`, a regular file taken with `flock` | every GPU user on the box takes it, including calibration and local runs; a box release never waives it |
| the engine checkout | the ranked job checks it out fresh from the submission; an operator checkout is separate | never measure with an operator checkout that has local edits |
| the runner | a self-hosted GitHub Actions runner whose labels include the track id | the label set is what the workflow's `runs-on` selects |
| the box calibration | this box's `baseline-calibration.json`, written by one calibration on the box | see section 2 |
| `git` at `/usr/bin/git` | the platform image: the Command Line Tools on a Mac, the stock Ubuntu image on a DGX Spark | benchd's two fork-point gates spawn that absolute path, so the runner service `PATH` cannot move which git judges a submission; a box without it fails both gates closed |

## 2. Calibration: this box's health band

Both Qwen 3.8 125B-A6B tracks measure their own denominator. A ranked run
measures the pairs the track fixture declares in `official_pairs` — 2 on both
platforms — on this box, in the same job. Every pair is a SERIAL-CONTROL leg on
the organizer-staged reference tree and then a CANDIDATE leg on the submission
tree. Each pair scores on its own control leg, and the run scores the pair whose
composite is the lower median over the pairs. Pairs are never averaged. No
denominator is pinned anywhere.

What the box needs is its own HEALTH BAND for that control leg. Write it once
per box with:

```sh
benchd calibrate-baseline \
  --baseline-workspace "$REFERENCE_WORKSPACE" \
  --engine .build/release/bench-worker \
  --golden "$LIVE_GOLDEN" \
  --passes 4 \
  --out "$REFERENCE_WORKSPACE/baseline-calibration.json"
```

`--weights` defaults to the reference tree's own transform output
(`$REFERENCE_WORKSPACE/weights`): the control leg must never load the
candidate's, because the transform is participant-editable.

The verb runs the ranked path's own control leg four times, under the full
official methodology, and refuses by name (`CALIBRATION-CV-EXCEEDED`) when the
box is too noisy for the mean to describe it. Repeat it whenever the organizer
moves the reference tree.

The ranked job then names both inputs: `MLXFAST_BASELINE_WORKSPACE` and
`MLXFAST_BASELINE_CALIBRATION`.

See [`qwen38-125b-a6b-baseline-capture.md`](qwen38-125b-a6b-baseline-capture.md)
for the full procedure, and
[`official-baseline-capture.md`](official-baseline-capture.md) for the
stored-pair capture the earlier tracks still use.

## 3. Differences at a glance

| | DGX Spark (CUDA) | Mac (MLX) |
|---|---|---|
| engine | ds4, built in the job (nvcc + cargo) | mlxfast Swift, built by `setup.sh` (Xcode + Metal) |
| model on the box | organizer-staged GGUF snapshot, verified, never fetched by the job | downloaded and verified by `setup.sh` from the organizer mirror |
| MTP head | separate `mtp-*.gguf` beside the shards | embedded in the checkpoint |
| who holds the weights in a window | `ds4-resident`, one connection at a time | `bench-worker resident`, one connection at a time |
| resident boot/stop script | `tools/serve-up.sh --boot` / `--stop` | `tools/resident-up.sh --boot` / `--stop` |
| worker lifecycle (paired ranked run) | benchd boots one resident per leg from that leg's tree; each phase spawns a thin adapter that attaches to it | benchd boots one resident per leg from that leg's tree; each phase spawns a sandboxed `bench-worker` that attaches to it |
| worker lifecycle (local unscored run) | the measure script's own `tools/serve-up.sh` wrap; one attached worker per window | the measure script's own `tools/resident-up.sh` wrap |
| goldens | organizer material in R2; staged on the box by pin | organizer material in R2; staged on the box by pin |
| cool gate | 50 C | 40 C |
| quiescence gate | load < 2.0, GPU util < 0.10 | load < 2.0, GPU util < 0.10 |
| runner supervisor | systemd | LaunchDaemon |
| pair platform | `linux-aarch64` | darwin |

## 4. Faults seen in practice

| symptom | cause | fix |
|---|---|---|
| ranked job fails at "runner environment" with `cargo`/`nvcc` not on PATH | listener started without the toolchain paths | add `PATH` to the runner `.env`, restart through the supervisor |
| runner shows offline, log says a session already exists | a second listener was started by hand | stop the extra listener; let the supervisor's one reconnect |
| preflight refuses an unpinned `*.json` in the pool directory | a non-pool golden placed beside the pool | move it out of `correctness_prompts/<track>/` |
| calibration or a window stalls with the GPU idle and two workers | a second connection waiting on the one-connection resident | one attached worker per phase; on the paired path never wrap benchd in the resident script (benchd boots each leg's resident itself and refuses an inherited socket, `LEG-SERVE-INHERITED-SOCKET`) |
| speculative candidate misses the prefill band while serial passes | the engine re-uploaded the draft head on first use inside the timed prefill | fixed in the engine (ds4 pin 5f36517); keep the head resident from load |
