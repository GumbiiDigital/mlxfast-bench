# How a track binds to this repo

benchd is developed and published from `main`. There is no per-track development branch and no
per-track bench fork (David ruling, 2026-09-07). A track's engine repo pins ONE benchd commit, and
it reads the bytes of that commit from a published channel.

> **Superseded in part, 2026-09-07 (David: "we can go with main now").** benchd is built and
> published from `main`; there is no per-track bench release branch for a new track. The
> Qwen 3.8 125B-A6B engine repos resolve benchd by one bench COMMIT named in their
> `tools/fetch-benchd.sh` (advancing benchd is one line in the engine repo); the older tracks
> still resolve the channel tip. `main` publishes its own `dist/` for all three platforms since
> 2026-09-14. The `qwen3.8-125b-a6b-v1` branch is `main` plus merge and republish commits, and
> it stays the branch the 125B pair is fetched from until each engine repository re-pins. The sections "Two names, two jobs", "Name construction"
> and "Branch lifecycle" below describe the model the older tracks were cut under and stay as
> the record of it. "How an engine repo resolves this repo", "Official baseline", "Model
> identity", "Scored regime" and "Goldens" are current.

## Two names, two jobs

A track uses two names. Do not treat them as one name.

| name | what it identifies | where it is used |
|---|---|---|
| **`track_id`** | the PLATFORM — one model generation on one hardware platform | the sealed `track_id` field; the R2 key prefix; the runner label; the ranked-board row |
| **channel branch** | the PROJECT — the branch that carries the published `dist/` for one model generation | the `branch` field of `benchd.manifest.json`; the engine's `BENCHD_DIST_BRANCH` |

One channel branch serves every platform of one model generation. The Qwen 3.8 125B-A6B project
publishes on `qwen3.8-125b-a6b-v1`, and both the MLX and the CUDA engine read it. The branch holds
merges of `main` plus the republished `dist/` directories and nothing else, so the bytes it
publishes always name a commit on `main` in `source_commit`.

Platforms that share a channel do not share a `track_id`. Each platform keeps its own `track_id`,
its own R2 key prefix and its own goldens.

## Name construction

Build a `track_id` this way:

```
{model}{ver}-{params}-{platform}-v{N}
```

Names are **decoder-neutral**. Never put a spec-decoder kind (`mtp`, `dflash`, `dspark`) in a track
name. A version bump (`-v{N+1}`) is a new track generation: a new R2 key namespace and re-authored
goldens.

| Track | `track_id` | channel branch |
|---|---|---|
| qwen 3.8 125B-A6B MLX | `qwen3.8-125b-a6b-mlx-v1` | `qwen3.8-125b-a6b-v1` |
| qwen 3.8 125B-A6B CUDA | `qwen3.8-125b-a6b-cuda-v1` | `qwen3.8-125b-a6b-v1` |
| qwen 3.8 27B MLX | `qwen3.8-27b-mtp-v1` | `main` |
| gemma 4 26B A4B MLX | `gemma4-26b-a4b-mlx-v1` | `gemma4-26b-a4b-mlx-v1` |

Each engine repo is named `{mlxfast|cudafast}-{model}{ver}-{params}-engine`.

`qwen3.8-27b-mtp-v1` predates the decoder-neutral rule, which is why its name still carries `mtp`.
It is also the value of `TRACK_ID` in `crates/bench-core/src/constants.rs`.

The steps that stand up a new track are in
[`runbook-new-engine.md`](runbook-new-engine.md).

## How an engine repo resolves this repo

An engine repo gets benchd through ONE of two channels. Both channels resolve to exact bytes.
Neither channel lets the engine repo choose a different benchd at run time.

**Channel 1 — SHA-pinned submodule.** The engine repo carries this repo as a submodule. The
gitlink IS the pin. Advance the pin in a deliberate two-verdict PR on the engine side. Verify
containment with a remote-tracking branch:

```
git -C benchd branch -r --contains $(git rev-parse HEAD:benchd)
```

**Channel 2 — dist channel.** A public engine repo cannot resolve a gitlink into a private repo.
It reads a published benchd instead. The engine repo names the channel in the
`BENCHD_DIST_BRANCH` environment variable, and resolves that channel through a manifest. Since
2026-09-05 the 125B engine repos name a bench COMMIT as well and fetch the pair by that commit
(the manifest's `branch` must still match); the older tracks fetch the branch tip. Channel 2 is
the live channel for every track today; no engine repo carries the submodule of channel 1 any
more.

The manifest carries four fields. Two fields verify the bytes. Two fields record where the bytes
came from:

| field | what it carries | kind |
|---|---|---|
| `sha256` | the digest of the published bytes | integrity |
| `bytes` | the byte count of the published bytes | integrity |
| `branch` | which channel branch published this benchd | provenance |
| `source_commit` | which commit the build came from | provenance |

`sha256` and `bytes` together are the integrity pin. Verify both before use. Refuse a manifest
whose `sha256` or `bytes` does not agree with the published bytes. Do not fall back to an
unverified benchd.

`branch` and `source_commit` are provenance. They are not checks, and they cannot be verified
against the bytes. They record which channel published the benchd, and which commit the build came
from, so a reader can find the source. Keep them correct: a wrong value here does not stop a run,
but it sends the next reader to the wrong source.

The channel publishes more than one binary from that one `source_commit`. The four fields above
describe `benchd`. A fifth field, `binaries`, gives the same integrity pair — `sha256` and
`bytes` — for every published binary, one line for each: `benchd`, and
`record-correctness-golden`, which an engine repo runs to author its track goldens. A consumer
that reads only the four fields above still resolves `benchd` correctly. A consumer that needs
another binary reads its entry and verifies it the same way. See the repository README,
"Publishing `dist/`", for the shape and the reader.

The dist channel is CONSUMER-side machinery. This repo is the SOURCE the channel publishes, which
is what `branch` and `source_commit` name. No code in this repo reads the channel or resolves
`BENCHD_DIST_BRANCH`. Do not look for the resolver here.

## Engine-repo counterpart

The engine side of a new track is a **new, fresh-seeded org repo** (single signed commit,
tree-identical to the source engine's `main`, source commit named in the seed message — the org
ruleset's verified-signature requirement rejects history-carrying pushes). Its procedure doc
lives in the engine repo as `docs/new-track-repo-procedure.md`.

## Official baseline

A track scores against a baseline in ONE of two ways. The two never mix.

| kind | where the denominator comes from | tracks |
|---|---|---|
| **live control leg** | the run MEASURES it: a serial-control leg on the reference tree, on the ranked box, in the same job | `qwen3.8-125b-a6b-mlx-v1`, `qwen3.8-125b-a6b-cuda-v1` |
| **stored pair** | a captured pair declared in the track fixture | `qwen3.8-27b-mtp-v1`, `gemma4-26b-a4b-mlx-v1` |

### Live-control-leg tracks

Ruled by David on 2026-09-08. Each ranked box has its own baseline, and the
baseline is a MEASUREMENT, not a pin. A ranked run measures the number of PAIRS
the track fixture declares in `official_pairs` — 2 on both platforms — on ONE
box in ONE job. The run is given N goldens, and pair k (1-based) measures golden
(k - 1) mod N. Every pair is the same two legs in the same order, on the same
prompt:

1. the SERIAL-CONTROL leg on the organizer-staged reference tree
   (`MLXFAST_BASELINE_WORKSPACE`), with no speculation;
2. the CANDIDATE leg on the submission tree, at its declared draft depth.

The two engines are strictly sequential and each leg loads the model once. benchd
sums each role's per-token times over the pairs, and the floors and the bands
apply to that aggregate. Every control leg is band-checked on its own. A fixture
that declares no `official_pairs` refuses the ranked run; benchd never guesses
the count.

Leg 1 runs the reference tree's OWN engine and the reference tree's OWN weights.
Both are found by re-rooting the candidate's own root-relative path into the
reference tree, so the two legs differ by their tree and by nothing else. The
transform that produces the weights is participant-editable, so the control leg
must never load the candidate's transform output.

The score is the live ratio of the aggregates,
`(ref_prefill / cand_prefill)^0.25 * (ref_decode / cand_decode)^0.75`, at batch
size 1 on one stream.

These tracks store NO pair. There is none in the fixture and none in a golden,
and every door a stored pair could come through is shut BY NAME on this path:

* a golden that declares `benchmark.baseline_{prefill,decode}_seconds_per_token`
  is refused (`GOLDEN-CARRIES-STORED-BASELINE`);
* the `MLXFAST_PAIRED_BASELINE_*` environment pair and the `--baseline-*` flags
  are refused (`STORED-BASELINE-OVERRIDE-REFUSED`);
* the LOCAL modes measure the CANDIDATE LEG ONLY and seal NO score. A
  participant iterating on a laptop has no reference tree, and David requires
  the local benchmark to keep working, so the run is UNSCORED, not refused: it
  seals the real timings, the real correctness verdict, `score: null`, an empty
  `metrics.error`, and `baseline_source: "none (local mode: unscored)"`. When
  BOTH runner inputs are present locally, the local modes run the full paired
  path instead, exactly as ranked.

The track fixture decides. `bench_core::contract::scores_against_live_control_leg`
reads the declaration. A fixture that declares no
`scores_against_live_control_leg` is refused: a track either measures its own
denominator or scores against a stored pair, and benchd picks neither for it.

Each ranked box carries a CALIBRATION FILE
(`MLXFAST_BASELINE_CALIBRATION`). It is a HEALTH BAND for leg 1 and never a
denominator: benchd checks the measured control leg against the band and refuses
the run when the leg falls outside it
(`SERIAL-CONTROL-LEG-OUTSIDE-BAND`). `benchd calibrate-baseline` writes the
file, once per box, on that box. The file stays there: each device tracks its own
reference benchmark, and benchd does not upload it and does not save it to a
centralized repository. The procedure is
[`qwen38-125b-a6b-baseline-capture.md`](qwen38-125b-a6b-baseline-capture.md).

### Stored-pair tracks

A stored-pair track has ONE official baseline pair: the serial prefill and
decode seconds-per-token that the track scores against, together with its
acceptance bands. The pair belongs to the `track_id`. It does not belong to the
branch, and it does not belong to the repo.

The pair is declared in the track fixture, as
`official_baseline_prefill_seconds_per_token` and
`official_baseline_decode_seconds_per_token`. A fixture carries the pair only
after you capture it on that track's own benchmark hardware. Before the capture,
the fixture declares neither half. That state is `OFFICIAL_BASELINE_PENDING`.

Read the pair with `bench_core::contract::official_baseline(contract, track_id)`.
It is the only accessor. It refuses a fixture that declares neither half, and the
refusal names the `track_id` and the `OFFICIAL-BASELINE-PENDING-CAPTURE` sentinel.
A fixture that declares one half is refused when it is parsed: one half is not a
pair.

### Where each path gets its pair

The paths do not agree, and that is deliberate. Read the row for the path you
are on.

| path | sources, in order | if no source gives a pair |
|---|---|---|
| official, timed, LIVE-CONTROL-LEG track | the SERIAL-CONTROL LEG this run measures, and nothing else | the run refuses by name: the reference workspace, the calibration file, the box and the band each refuse for themselves |
| official, timed, stored-pair track | the `MLXFAST_PAIRED_BASELINE_{PREFILL,DECODE}_SECONDS_PER_TOKEN` pair, else both `--baseline-*` flags, else the golden's `benchmark.baseline_{prefill,decode}_seconds_per_token` pair | the run PREFLIGHT-FAILS. The fixture pair is not a fallback here. |
| official, gates-only (`MLXFAST_BENCHMARK_SKIP_TIMED=1`), LIVE-CONTROL-LEG track | none: a gates-only run measures no leg, so it seals the zero placeholders | not applicable |
| official, gates-only, stored-pair track | the `MLXFAST_PAIRED_BASELINE_*` pair, else the golden's pair, else the fixture's captured pair | the run refuses and names the track, the sentinel, and the sources it tried |
| local (`--local-iterate`, `--local-submit`), LIVE-CONTROL-LEG track | the SERIAL-CONTROL LEG, when both runner inputs are present; otherwise NONE — the run is UNSCORED | not applicable: an unscored run seals no score and is not a failure |
| local (`--local-iterate`, `--local-submit`), stored-pair track | the fixture's captured pair only. #127 makes these legs score against the declaration and nothing else: no environment pair, no flag, no golden pair | the track is pending: the run refuses and names the track and the sentinel |

### The environment override

One function reads the `MLXFAST_PAIRED_BASELINE_{PREFILL,DECODE}_SECONDS_PER_TOKEN` pair,
`benchd::official::paired_baseline_override`, and every path that consults it has the same
posture.

| the pair in the environment | what the run does |
|---|---|
| both values set, each a finite positive number | the override WINS over every later source in the table above |
| one value set and the other absent | the run REFUSES, and the message names both keys and says they `must be provided together` |
| a value that is not a finite positive number | the run REFUSES, and the message names that key and says it `must be a finite positive seconds-per-token value` |
| neither value set | the run falls through to the next source in the table above |

A malformed override never degrades to "no override". A run that scored against a denominator
the operator did not ask for is the fault this refusal exists to stop.

A LIVE-CONTROL-LEG track is a separate case: it refuses the pair itself, set or malformed,
with `STORED-BASELINE-OVERRIDE-REFUSED`.

The official timed path of a STORED-PAIR track still reads the declared pair in
one more place. Its PREFLIGHT-FAILURE record carries a baseline pair, and that
pair comes from the fixture. A pending track therefore stops the run before the
record is written, instead of writing a record with another track's numbers in
it.

Do not give a new track another track's numbers. A track that has no captured
pair, and does not measure its own, cannot be scored on any path.

### Ending the pending state

This applies to STORED-PAIR tracks only. A live-control-leg track has no pending
state, because it stores nothing; it refuses `--capture-baseline` by name
(`CAPTURE-RETIRED-FOR-LIVE-CONTROL-LEG`) and calibrates its box instead.

A pending stored-pair track cannot reach a timed phase on any scored path, so it
cannot measure its own pair there. `benchd iterate --capture-baseline` is the one
mode that can. It runs only while the track is pending, it resolves no official
baseline, and it writes only its capture record. A captured track refuses it by
name.

The procedure is [`official-baseline-capture.md`](official-baseline-capture.md).
The record it writes is the input to a separate, reviewed pull request that adds
the track's pair to the track fixture, in the engine repository.

### The track_id fence

The workflow-declared track id must be ONE value: `constant ≡ contract ≡ env`.

* `measure-job` resolves the track id from the `MLXFAST_QWEN_MTP_TRACK_ID` environment variable or
  from the `--contract` track fixture. The two must agree with each other, AND the result must
  equal `TRACK_ID`.
* `overlay-timing` refuses a `results.json` whose sealed `track_id` is not `TRACK_ID`, even when
  the operator names that same foreign track.
* Both refusals name BOTH values: the track this tree serves, and the track the run declares.

The fence exists because the two other legs cannot see this error. A tree built for one track,
driven by another track's contract, seals THIS track's baseline pair under THAT track's
`track_id`. The commit and the weights-hash cross-checks still agree, because both halves really
do come from one run. Only the constant can tell.

`benchd iterate` takes `--contract` on every mode, and it reads the track id from the
`MLXFAST_QWEN_MTP_TRACK_ID` environment variable. The variable is not optional: the run
refuses when it is not set, because the track id keys the platform and the model identity.
The fixture is not optional either: every scored value the run resolves comes from it.

## Model identity

Each track ALSO declares its MODEL IDENTITY. The identity has four facts:

| fact | what it controls |
|---|---|
| `golden_model_type` | the `model_type` value the track's goldens declare |
| `vocab_size` | the token-id range (`0` to `vocab_size`) that goldens, tapes and worker logits must stay inside |
| `num_hidden_layers` | the sealed `metrics.num_layers` audit field |
| `seed_tokens` | the length of the correctness prompt, the prefill prompt and the decode seed |

The identity belongs to the `track_id`. It does not belong to the branch. The four facts are
declared in the track fixture, together or not at all. Read them with
`bench_core::contract::model_identity(contract, track_id)`. It is the only accessor. It refuses
a fixture that declares no model shape, and the refusal names the `track_id` and the
`MODEL-IDENTITY-UNDECLARED-FOR-TRACK` sentinel. A fixture that declares part of the shape is
refused when it is parsed: a half-declared shape describes no checkpoint.

benchd resolves the identity ONCE for each run. It resolves it from the same track id that gives
the platform, and it resolves it BEFORE it reads a golden. The golden loader, the timed-prompt
tape loader, the correctness gates and the sealed audit metrics all use that one identity.

The result is that ONE benchd loads a golden of ANY declared track. A golden of a different track
is refused. The refusal names the track, the `model_type` the track needs and the `model_type` the
golden declares.

These commands need a track id, because each one reads or writes a golden:

* `benchd iterate` and `benchd correctness` read the `MLXFAST_QWEN_MTP_TRACK_ID` environment
  variable.
* `benchd validate-golden` takes a `--track` option. It falls back to the same environment
  variable.
* `benchd measure-job` uses the track id it already resolves for the seal.
* `record-correctness-golden` takes a `--track` option, because it AUTHORS a golden under that
  track's identity.

## Scored regime

Each track that scores a stored pair ALSO declares what it scores: the batch size of its scored
point, and the two exponents that make the composite. The three values are declared in the same
track fixture, together or not at all, and are read with
`bench_core::contract::scored_regime(contract, track_id)`. A fixture that declares none of them
is `SCORED_REGIME_PENDING` on a path that needs a regime.

The batch size selects the measured path. A width of 1 is the single-stream path. A width above
1 is the batched cohort path, and only a width with a certified series tag can run it; today
that width is 8.

A live-control-leg track declares no regime. Each device measures its own
denominator and `iterate --mode official` computes the live ratio, so there is no
stored regime for it to state. Its sealed `metrics.contract_sources` reads
`"device"` for that group.

The declaration decides whether benchd enforces anything on the prefill half of the timed window.
See [`scored-regime-and-prefill-window.md`](scored-regime-and-prefill-window.md).

A new stored-pair track therefore declares THREE things before it can score: its model shape,
its official baseline pair (captured by the procedure above), and its scored regime. None of
the three has a fallback. A live-control-leg track declares its model shape, its
`official_pairs` count and its two speedup floors instead.

## Deploy order

A benchd built from `main` refuses a fixture that does not carry the keys of a group it needs.
Publish the engine fixture first, and the public cut of the engine repository with it. Publish
the benchd that requires those keys second. The reverse order stops every ranked run on the
track until the fixture catches up.

## Goldens

Track goldens are authored on the track's designated benchmark hardware, A≡B double-generated
(byte-identical asserted before pinning), pinned by sha256 + bytes, and uploaded append-only to
R2 under `{track_id}/{sha256}.json` once the track is stable and ready for testing — with
GET + sha + bytes round-trip verification after upload.

The R2 key prefix is the **`track_id`**, not a branch name. Two platforms of one model generation
therefore keep two separate golden namespaces. Each platform authors its goldens on its own
hardware, so the two namespaces must not be merged.
