# Official baseline capture

Class: runbook. **Legacy path:** stored-pair tracks only (`qwen3.8-27b-mtp-v1`,
`gemma4-26b-a4b-mlx-v1`). A live-control-leg track (both Qwen 3.8 125B-A6B tracks) stores no
pair, refuses this mode by name (`CAPTURE-RETIRED-FOR-LIVE-CONTROL-LEG`), and calibrates its
box instead: [`qwen38-125b-a6b-baseline-capture.md`](qwen38-125b-a6b-baseline-capture.md).

This page tells you how to capture a STORED-PAIR track's official baseline pair. The pair is the
serial prefill and decode seconds-per-token that such a track scores against.

The Qwen 3.8 125B-A6B tracks store no pair. They measure their denominator live, on the box, in
the same job, and they refuse `--capture-baseline` by name
(`CAPTURE-RETIRED-FOR-LIVE-CONTROL-LEG`). What their boxes calibrate instead is a health band —
see [`qwen38-125b-a6b-baseline-capture.md`](qwen38-125b-a6b-baseline-capture.md).

Read [`track-release-branches.md`](track-release-branches.md) first. It gives the rules for the
declared baseline pair. This page gives the procedure.

## 1. When you use this procedure

Use it when a track is `OFFICIAL_BASELINE_PENDING`. A pending track is one whose `--contract`
track fixture declares no `official_baseline_prefill_seconds_per_token` /
`official_baseline_decode_seconds_per_token` pair.

Do not use it on a track whose fixture declares the pair. A captured track refuses this mode by
name.

## 2. Why the mode exists

The local checked-timing legs resolve the track's official baseline before they start the
engine. A pending track therefore refuses the run before it measures anything. The measurement
that ends the pending state could not run.

`benchd iterate --capture-baseline` is the one safe way out of that loop. The mode has three
properties. Together they make sure that it cannot become a way to score without a baseline.

1. The mode runs only while the track is pending. A captured track refuses the mode by name.
2. The mode runs the checked-timing legs with an inert baseline pair. It resolves no official
   baseline, and it computes no score.
3. The mode writes one file: the capture record you name. It writes no `score.json`, no
   `.sha256` sidecar, and no integrity sidecar. It stops before those writers.

## 3. Before you start

1. Run this procedure on the track's own benchmark hardware. Do not run it on a laptop.
2. Merge the engine change first. Then verify it independently. Record the merged head SHA.
   A capture from an unmerged engine or an unverified engine is void.
3. Build benchd from `main`. Record its commit SHA.
4. Confirm that the track's fixture declares no baseline pair.
5. Stage the correctness golden and the transformed weights on the box.

## 4. Capture the pair

Run four passes. Use one fresh invocation for each pass. The mode starts its own worker for each
pass. Obey the box cool-down rule between the passes.

Give every pass the same record path. The mode merges each pass into that one record.

```sh
benchd iterate --mode local-iterate \
  --engine "$STOCK_WORKER" \
  --weights "$WEIGHTS_DIR" \
  --golden "$CORRECTNESS_GOLDEN" \
  --contract "$TRACK_CONTRACT" \
  --capture-baseline "$CAPTURE_DIR/official-baseline.json"
```

`--contract` is required on every mode. The fixture declares the measurement window and the
model shape the golden loads under, and neither has a fallback left.

Use the stock tree. Do not modify the engine for the capture.

`--capture-baseline` is a local-iterate mode. `--mode local-submit` and `--mode official` refuse
the flag when you parse the command line.

Know the regime you capture in, and keep it the same for all four passes. The mode runs
`local-iterate`, whose timed-phase gates are OFF by default. Add `--cool-gate` to make them ON:
the one flag controls BOTH the quiescence gate and the cool gate. The box cool-down rule tells you
which of the two to use. Use the same one for every pass, and name it in
the pull request.

One pair serves BOTH local legs. `local-submit` scores against the same captured pair, but it
runs its gates ON by default, and it uses the decode window the fixture declares in
`local_submit_benchmark_decode_steps` in place of `benchmark_decode_steps`. Ruling #127 makes
both local legs score against the declared pair and nothing else, and the fixture declares one
pair for each track. Record the regime you captured in, so that a later reader knows what the
number means.

## 4a. The window loads the weights once

Each pass runs one timed window. A window has an unmeasured warm-up leg, a timed prefill phase
and a timed decode phase. All three run on ONE worker. The worker loads the weights one time for
the whole window.

A shared worker does not share cache state. The engine resets the KV cache and the allocator at
the start of each phase. benchd makes sure of this at each phase boundary: `phase_diagnostics`
must report `cache_memory` of 0. If the value is not 0, benchd refuses the run and discards the
session. The rule is in
[`../crates/bench-protocol/PROTOCOL.md`](../crates/bench-protocol/PROTOCOL.md).

A baseline pair is only comparable with a candidate that the same window shape measured. A pair
that was captured under the earlier fresh-process-per-phase shape was measured with one worker
for each phase. Capture such a pair again on the official path before you use it for a ranked
run.

## 5. What the record holds

The record is a JSON object. It holds values only.

| field | what it holds |
|---|---|
| `track_id` | the track this tree serves |
| `mode` | the mode the passes ran in |
| `decode_steps` | the decode window the passes measured over |
| `engine_sha256` | the sha256 of the engine binary |
| `weights_sha256` | the sha256 of the transformed weights directory |
| `golden_sha256` | the sha256 of the golden bytes |
| `benchd_sha256` | the sha256 of the benchd binary that did the timing |
| `run_count` | the number of passes in the record |
| `runs` | each pass's prefill and decode seconds-per-token |
| `prefill_cv_percent`, `decode_cv_percent` | the per-axis CV of the passes |

The seven identity fields must agree across the passes. A pass with a different value in any one
of them refuses to merge. The refusal names the field. Start a new record file for a new engine,
a new checkpoint, a new golden, a new window, or a new benchd.

Each identity field is there because the pair depends on it. Different weights measure
differently. A different decode window has a different cost for each token. benchd does the
timing, so the harness that measured is part of the measurement. Two passes over different inputs
must not average into one mean that describes neither of them.

`CV` means the SAMPLE coefficient of variation, as a percent. It is the standard deviation with
the N−1 denominator, divided by the mean. It is the sample form, not the population form,
because the passes are a sample of the box's run-to-run distribution and N is small. The CV is
absent from the record until the record holds two passes. Two passes are the minimum for the
statistic.

The mode writes the record with a temporary file and a rename. An interrupted pass therefore
cannot leave a part-written record.

## 6. If a pass refuses

The mode refuses a pass that did not measure. It prints the cause. Correct the cause, then run
the pass again.

| refusal | cause |
|---|---|
| `--capture-baseline refused: track ... is not OFFICIAL-BASELINE-PENDING-CAPTURE` | the track has a captured pair; there is nothing to capture |
| `--capture-baseline refused: the engine ... did not resolve` | benchd could not read the engine binary, so it cannot name the engine |
| `--capture-baseline refused: the correctness gate failed` | the run is not healthy; do not capture from it |
| `capture refused: timed phase failed: ...` | the timed phase did not produce a measurement; the text after the colon is the cause |

A refused pass does not change the record.

## 7. What you do with the record

The record is the INPUT to a separate pull request. That pull request replaces the track's
pending state with a value: it adds the track's pair to the track fixture, in the engine
repository, as `official_baseline_prefill_seconds_per_token` and
`official_baseline_decode_seconds_per_token`.

The record is not itself a baseline. No code reads it back. A person reads it, and a reviewer
checks the pull request against it.

Put this in the pull request body:

* the whole capture record, with every identity field: `track_id`, `mode`, `decode_steps`,
  `engine_sha256`, `weights_sha256`, `golden_sha256`, `benchd_sha256`;
* the merged engine head SHA and the benchd commit SHA;
* the `--cool-gate` setting the passes used. It turns the quiescence gate and the cool gate on or
  off together. With the gates on, the calibration file itself carries a `gates` array: one record
  per gate point, naming its pass and phase and what both gates read.

The record is OPERATOR-ATTESTED. It is not machine-attested. benchd merges each pass into
whatever record parses at the path you give it, and no code reads the record again after that. A
person can thus write a record by hand, or change one. The reviewer of the sentinel-to-value pull
request must therefore use `engine_sha256`, `weights_sha256` and `benchd_sha256` as the
load-bearing evidence. Check each of the three against the merged engine, the staged weights, and
the built benchd. Do not read them as labels.

The band tolerances that go with the pair are track policy, not part of this mechanism. This
page does not give the band arithmetic. Ask the track's ruling for it.
