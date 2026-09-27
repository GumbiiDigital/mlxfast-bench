# Runbook: stand up a DGX Spark (CUDA) ranked box

**Class:** runbook. Do the steps in the order given. For the overview and the
Mac procedure, see [`box-setup-runbook.md`](box-setup-runbook.md) and
[`runbook-box-setup-mlx.md`](runbook-box-setup-mlx.md).

A Spark joins the ranked network as a self-hosted GitHub Actions runner of
the CUDA engine repository. Yukon dispatches that repository's
`benchmark.yml` for each submission. The workflow selects a runner by label.
The job builds the ds4 engine. The model is the pinned unsloth GGUF
snapshot, staged once on the box.

**The box runs one container.** The container is the runner. It carries the
CUDA toolchain, `cargo`, `python3` and `jq`. It runs as root in its own
namespace. The box needs no group, no lock file rule, no user Rust
installation, no runner service unit and no runner environment file. The
only host change is the docker daemon.

**The container is the only GPU user on the box.** Ranked jobs run in it.
Calibration runs in it. The local checks run in it. Nothing on the host
touches the GPU. The GPU lock keeps its path
(`/tmp/mtplx-gpu-exclusive.lock`) and its `flock` discipline in the
container's own `/tmp`.

## Terms and conventions

- `<op>`: the operator account on the box.
- `<box>`: the name of the box. It is also the runner name.
- `<root>`: the staging directory. On the fleet it is `/home/<op>/qwen38-125b-bringup`.
- `<track>`: the track identifier. Today it is `qwen3.8-125b-a6b-cuda-v1`.
- `<peer>`: a fleet Spark that already has a verified snapshot.
- Token commands read the token from stdin. Do not put a token on a
  command line. Do not write a token to a file on the box.
- The link from a laptop to a Spark can be slow (50 KB/s on the fleet's
  tailnet). A 15 MB copy can take minutes. `scp` can report success for a
  truncated file. Verify each file that you copy from the laptop by byte
  count or sha256 before you use it. The boxes reach GitHub and Hugging
  Face at 58 MB/s.
- Time budget: one fresh Spark took 51 minutes on 2026-09-04, 33 of them
  the weights download. Seven fleet boxes took 25 to 36 minutes each on
  2026-09-05 with the weights copied from a peer.

## 1. Base box

A fresh Ubuntu 24.04 image on the GB10 ships `docker-ce`, the NVIDIA
container toolkit, `git`, `jq`, `python3`, `curl`, `sha256sum` and
`nvidia-smi`. Do not install packages with apt.

The stock CUDA toolkit at `/usr/local/cuda` stays on the host, unused. The
container carries its own toolchain. Do not install Rust on the host.

`docker.service` is installed but disabled on a fleet box. Section 4 enables
it.

## 2. Weights

Stage the five pinned files flat in one directory: the four
`Qwen3.8-Flash-Next-UD-Q4_K_XL-0000N-of-00004.gguf` shards and the draft
head `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`. The pins (bytes and sha256
per file) are in `fixtures/qwen3_8_125b_a6b_track.json` in the engine
repository. The engine's `./setup.sh` verifies bytes first, then sha256. It
fails closed on a miss. It never downloads.

Budget 114 GB of disk plus a 20 GiB margin. The 26.8 GiB n-gram table
inside the snapshot stays on the SSD. The job reads the directory from
`MLXFAST_TARGET_SNAPSHOT_DIR`. Shard `00001` is 10,946,624 bytes. That is
its pinned size, not a truncated download.

Source: Hugging Face repository `unsloth/Qwen3.8-Flash-Next-GGUF` at
revision `38bb39ee97821de2c9009abb7e93950eec396e66`. The files are not flat
there. The shards are under `UD-Q4_K_XL/`. The head is under `MTP/`. The
tree API reports each file's `lfs.oid`. That value equals the fixture's
sha256. Check the listing against the pins before the transfer:

```bash
curl -s "https://huggingface.co/api/models/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66?recursive=true" \
  | jq -r '.[] | select(.type=="file") | "\(.size)\t\(.lfs.oid // "-")\t\(.path)"' | grep -E 'UD-Q4_K_XL/|MTP/'
```

If a peer holds a verified snapshot, go to section 2a. If not, download from
Hugging Face with resume and retries, four files at a time:

```bash
BASE=https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/38bb39ee97821de2c9009abb7e93950eec396e66
mkdir -p <root>/weights/gguf-unsloth-q4 && cd <root>/weights/gguf-unsloth-q4
printf '%s\n' \
  UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
  UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf \
  UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf \
  MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  | xargs -P 4 -I{} bash -c 'curl -sS -L -C - --retry 5 --retry-delay 10 -o "$(basename {})" "'"$BASE"'/{}"'
```

Measured: 58 MB/s from Hugging Face on the fleet uplink, 33 minutes.

### 2a. Copy the weights from a fleet peer

The fleet Sparks share a 200 GbE link. A copy from a peer runs at 1.2 to
2.3 GB/s and takes one to two minutes. Do not add any file to the snapshot
directory on the peer: the weights digest covers the whole directory.

On the peer, serve the snapshot directory for the duration of the fleet
rollout:

```bash
cd <root>/weights/gguf-unsloth-q4
python3 -m http.server 8765 --bind <peer LAN address> >/tmp/wmirror.log 2>&1 &
echo $! >/tmp/wmirror.pid
```

On the new box, make sure that the peer answers. Then fetch the five names.
The server does not honor byte ranges. Do not resume with `-C -`. Delete a
failed file and fetch it again from the start.

```bash
curl -sI http://<peer LAN address>:8765/ | head -1     # HTTP/1.0 200 OK
mkdir -p <root>/weights/gguf-unsloth-q4 && cd <root>/weights/gguf-unsloth-q4
for n in Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1,2,3,4}-of-00004.gguf mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf; do
  curl -sS -f --retry 3 -o "$n.part" "http://<peer LAN address>:8765/$n" && mv "$n.part" "$n" || rm -f "$n.part"
done
```

Several boxes can copy from one peer at the same time. Stop the server once,
after the last box has copied: `kill "$(cat /tmp/wmirror.pid)"`. Section 6
verifies the copy against the fixture pins. Do not trust a peer copy
without that verification.

## 3. The engine checkout

The engine repository is internal. The box holds no GitHub credential. The
ranked job does not need one: Actions supplies its own token to the
checkout. The operator checkout comes from a bundle. Make the bundle on a
machine that has access, at the tip to be dispatched (the engine branch):

```bash
# on the machine with access
git -C <engine clone> fetch origin
git -C <engine clone> bundle create engine-<sha>.bundle <engine branch>
shasum -a 256 engine-<sha>.bundle
scp engine-<sha>.bundle <box>:~/
# on the box
sha256sum ~/engine-<sha>.bundle
git clone ~/engine-<sha>.bundle -b <engine branch> <root>/engine
```

Make sure that the two sha256 values are equal before the clone. A truncated
bundle clones with `fatal: early EOF` or `index-pack died`. Copy it again
(`rsync --partial`, or in chunks) until the sha256 matches. `git bundle
verify` does not work outside a repository.

Keep the bundle in place. It is the checkout's `origin`. If you remove it,
`git fetch` from that checkout fails. To move the checkout to a newer tip,
make a new bundle, copy it, and run `git fetch <bundle> <branch>`.

This checkout is the operator's. The container mounts it at `/engine`, so
every command of sections 6 to 11 runs against it. The ranked job does not
use it. Actions checks the dispatched commit out under `/runner/_work`.

## 4. The container

`tools/spark-box/converge.sh` in the engine checkout is the whole box-side
procedure. It asserts what the box already ships, enables `docker.service`,
builds the image from `tools/spark-box/`, creates the runner volume and
starts the container. It is idempotent. It never installs a package. It
never holds a token.

Report first. `--check` changes nothing:

```bash
ssh <box> 'cd <root>/engine && sudo tools/spark-box/converge.sh --check'
```

Then converge:

```bash
ssh <box> 'cd <root>/engine && sudo tools/spark-box/converge.sh'
```

The image tag is `spark-box:<engine commit sha>`. The build takes 35 seconds
with the CUDA base cached, and about 2 minutes without it. The image is
7.7 GB. Nothing is pushed to a registry.

The container is named `spark-box`. Its restart policy is
`unless-stopped`, so it comes back after a reboot. The runner volume is
`runner-qwen38-cuda`. The volume holds the registration and `_work`, so a
rebuild for a new engine commit keeps both.

The container reports `not registered` and restarts until you do section 5.
This is expected.

## 5. Register the runner

The container runs the listener only after this step, and `docker exec`
needs a running container, so register before you stage the rest.

Register the box to the CUDA engine repository. The label is the track
identifier. The runner adds `self-hosted`, `Linux` and `ARM64` itself. A
second Spark with the same label adds capacity. If a box ever serves two
tracks, register both labels on the one runner. Never start a second
container.

The registration token is single-use and short-lived. Mint it on the machine
with access and pass it on stdin:

```bash
gh api -X POST repos/Layr-Labs/cudafast-qwen38-125b-a6b-engine-dev/actions/runners/registration-token --jq .token \
  | ssh <box> "cd <root>/engine && sudo env SPARK_BOX_RUNNER_NAME=<box> tools/spark-box/converge.sh register"
```

Use `sudo env`, not `sudo VAR=value`. The default `sudoers` resets the
environment, and a variable set that way does not reach the command. The
runner name defaults to the short hostname when you leave the variable out.

The registration lands in the volume. It survives every later converge.

Make sure that the runner shows online:

```bash
gh api repos/Layr-Labs/cudafast-qwen38-125b-a6b-engine-dev/actions/runners --jq '.runners[] | "\(.name) \(.status)"'
```

Do not dispatch a ranked job until the checks of section 11 pass.

## 6. Work inside the container

Every command from here runs inside the container. Open a shell:

```bash
ssh <box> -t 'sudo docker exec -it spark-box bash'
cd /engine
```

The five box paths are already in the environment. Do not set them by hand.

| name | container path |
|---|---|
| `MLXFAST_TARGET_SNAPSHOT_DIR` | `/weights` |
| `BENCHD_BIN_DIR` | `/opt/benchd-bin` |
| `MLXFAST_QWEN38_GOLDEN_DIR` | `/goldens` |
| `MLXFAST_BASELINE_WORKSPACE` | `/opt/baseline/workspace` |
| `MLXFAST_BASELINE_CALIBRATION` | `/opt/baseline/baseline-calibration.json` |

Build the engine and verify the snapshot:

```bash
cd /engine && ./setup.sh
```

This builds ds4 and the adapter (two to four minutes from cold). It then
verifies the five staged files by bytes and sha256 against the fixture. It
exits non-zero on any miss.

## 7. The benchmarker pair

The benchmarker pair (`benchd`, `record-correctness-golden`, target
`aarch64-unknown-linux-gnu`) comes from the track's dist channel. The
engine's `tools/fetch-benchd.sh` downloads it and verifies it against its
own `benchd.manifest.json`. While the bench repository is private, the fetch
needs a token for that one call. Pass it on stdin:

```bash
gh auth token | ssh <box> 'sudo docker exec -i spark-box \
  bash -c "cd /engine && GITHUB_TOKEN=\$(cat) ./tools/fetch-benchd.sh"'
```

The pair lands in `/opt/benchd-bin`, which is `<root>/benchd-bin` on the
host. The mount is read-write. `fetch-benchd.sh` sets the mode of the
verified pair, and a read-only mount makes the local checks fail.

An installed pair stays in place. After a dist republish, refresh it:

```bash
gh auth token | ssh <box> 'sudo docker exec -i spark-box \
  bash -c "cd /engine && BENCHD_REFRESH=1 GITHUB_TOKEN=\$(cat) ./tools/fetch-benchd.sh"'
```

## 8. Goldens

The timed-pool goldens and the per-depth oracles are organizer material.
They are in R2, under the object keys that `r2_path` names in the track
fixture. They are never in git, and the ranked job holds no credential, so
the operator stages them one time, out of band.

Download each pinned object with the organizer's credentialed download. Put
the files in `<root>/goldens/<track>` on the host, one file per pin, named
by the last part of its `r2_path`. Put nothing else there. The container
sees them at `/goldens`, read-only.

Before every ranked run, the preflight compares the byte count and the
sha256 of each staged file with the fixture pin. It also refuses the
directory when the directory holds one more `*.json`, because the job passes
every `*.json` there as a golden. Section 11 runs the preflight, after the
reference tree and the band exist: the preflight checks those too, and it
refuses while either is missing.

## 9. The reference tree and this box's health band

The CUDA track pins no serial pair. A ranked run measures the pairs the
track fixture declares in `official_pairs`, and every pair times a
serial-control leg on the reference tree and then the candidate leg. The
score is the live ratio.

So the box carries two things instead of a pinned pair.

| name | what it holds |
|---|---|
| `MLXFAST_BASELINE_WORKSPACE` | the organizer-staged reference tree, built |
| `MLXFAST_BASELINE_CALIBRATION` | this box's `baseline-calibration.json` |

Both live under `<root>/baseline` on the host, which the container sees at
`/opt/baseline`. The reference tree is a subdirectory of it. The stager
refuses a destination that already exists, so do not create the tree by
hand.

Stage the reference tree from a local bundle or mirror, inside the
container. The stager clones it detached at the fixture's
`baseline_reference_commit`, then builds it:

```bash
cd /engine
tools/stage-baseline-workspace.sh --dry-run --source <bundle or mirror> "$MLXFAST_BASELINE_WORKSPACE"
tools/stage-baseline-workspace.sh --source <bundle or mirror> "$MLXFAST_BASELINE_WORKSPACE"
```

`--dry-run` prints the plan and writes nothing. Copy the bundle to the box
the way section 3 copies the engine bundle. The stager contacts no network
host and uses no credential.

Write the calibration file once on the box, and again whenever the organizer
moves the reference tree:

```bash
cd /engine && tools/calibrate-box.sh <box> "$MLXFAST_BASELINE_CALIBRATION"
```

The driver takes the GPU lock itself and holds it for the whole
calibration. Do not wrap it in `flock`. It exits 3 when another run holds
the lock.

The verb runs the ranked path's own control leg four times. It refuses by
name (`CALIBRATION-CV-EXCEEDED`) when the box is too noisy for a mean to
describe it. The file is a health band for the control leg, never a
denominator. The full procedure, the refusal names and the file format are
in
[`qwen38-125b-a6b-baseline-capture.md`](qwen38-125b-a6b-baseline-capture.md).

## 10. The measurement topology

A ranked run is paired. benchd boots one `ds4-resident` per leg from that
leg's own tree, through that tree's `tools/serve-up.sh`. Leg 1 is always
booted serial. The weights load once per leg. Each phase attaches a thin
worker over that leg's Unix socket. The resident serves one connection at a
time. `tools/serve-up.sh` does not take the GPU lock. Its caller does.

Do not wrap the score check in `tools/serve-up.sh`. benchd boots each leg's
resident itself, and it refuses an inherited `DS4_RESIDENT_SOCKET` by name.
The wrapper form stays correct for a single-leg run, which is what the
correctness check of section 11a is.

The local checks have a pre-timing gate. The gate waits for the GPU to idle
and to cool to 60 C. The ranked path has no local gate.

## 11. Local checks

Run two checks from `/engine` inside the container, one after the other,
each under the lock. The two serve values belong to the correctness check
alone. Set them in that command. Do not export them in the shell: the score
check refuses when they are preset, because on the paired path they would
reach the serial-control leg's serve.

```bash
cd /engine
export MLXFAST_WEIGHTS_PATH="$MLXFAST_TARGET_SNAPSHOT_DIR"
export RUNNER_NAME=<box>
./tools/spec-declaration.sh describe
```

`MLXFAST_WEIGHTS_PATH` is the directory that `benchmark.sh` digests.
`SERVE_UP_WEIGHTS_DIR` only feeds the resident. `RUNNER_NAME` is the name
the calibration file was captured on. The ranked job gets that name from
Actions. Without it the score check refuses with `BASELINE-BOX-UNRESOLVED`.

The last command prints `serial` or the declared depth. The engine branch
prints `mtp1` since the depth-1 promotion of 2026-09-04.

Run the preflight before the two checks. It is the ranked job's own gate: it checks the
goldens against their pins, the reference tree, this box's band, and the
environment. It refuses by name and it measures nothing.

```bash
cd /engine && tools/ranked-box-preflight.sh
```

### 11a. The correctness check

```bash
flock /tmp/mtplx-gpu-exclusive.lock -c '
  SERVE_UP_SPECULATIVE="$(./tools/spec-declaration.sh speculative)" \
  SERVE_UP_SPEC_DRAFT_LEN="$(./tools/spec-declaration.sh draft-len)" \
  MLXFAST_ENGINE_BIN=.build/release/mlxfast-runtime-worker \
  MLXFAST_CORRECTNESS_GOLDEN_PATH=correctness_prompts/public-longcopy-gate-english-1024.golden.json \
  SERVE_UP_WEIGHTS_DIR="$MLXFAST_WEIGHTS_PATH" \
  tools/serve-up.sh ./benchmark.sh --local-iterate'
jq '.metrics.passed_correctness, .passed, .score' score.local-iterate.json
```

The receipt is exit code 0 and `passed_correctness` true in
`score.local-iterate.json`. The flag is under `.metrics`. Read
`passed_correctness`, not the score. To run this check the way a participant
does, without the box calibration, unset `MLXFAST_BASELINE_WORKSPACE` and
`MLXFAST_BASELINE_CALIBRATION` for that command. It then measures the
candidate leg only and seals `score: null`.

Time: three to six minutes. The weights digest of 114 GB takes 85 seconds.
The gate then waits for the GPU to cool.

Some boxes idle warm. On two fleet boxes the loaded resident held the die
at 51 to 52 C with nothing else on the GPU. The gate then rejects the run:
`gate rejected (prefill): GPU is hot and not cooling down`. On such a box,
run the correctness check with `MLXFAST_LOCAL_COOL_GATE=0` inside the
`flock -c` environment. `passed_correctness` stays valid. The timings are
then hot-start and not comparable. The gate is local-only. The ranked path
is not affected.

### 11b. The score check

This is the ranked job's own measurement. Call the measure script directly,
not through `tools/serve-up.sh`.

```bash
flock /tmp/mtplx-gpu-exclusive.lock -c 'tools/qwen38-125b-a6b-measure-and-score.sh'
jq '.passed, .score, .metrics.effective_spec_depth' score.json
```

The receipt is `passed: true` in `score.json`. The run is paired, so the
score is the live ratio of the reference legs to the candidate legs: a
serial declaration puts two serial legs against each other and scores near
1.00, and a depth declaration scores above it. Fleet band on `mtp1`, 2026-09-05:
1.07 to 1.10. Time: two to three minutes once the engine is built.

Both files are gitignored. The tree stays clean.

## 12. Dispatch and receipt

```bash
gh workflow run benchmark.yml --ref <engine branch>
```

The box job checks the runner environment, runs the preflight, verifies the
pair, builds the engine, and measures under the lock. The build takes 90
seconds from cold. A content-keyed cache skips it when the tree is
unchanged. A passing run of section 11b is the readiness receipt. One
dispatch per fleet is sufficient. GitHub assigns the job to any idle runner
with the label.

## 13. Refresh a box

To move the operator checkout to a newer engine tip, fetch the new bundle
into `<root>/engine` (section 3), then converge again:

```bash
ssh <box> 'cd <root>/engine && sudo tools/spark-box/converge.sh'
```

The image tag carries the engine commit, so a new tip builds a new image and
replaces the container. The volume keeps the registration and `_work`. A
converge that finds the tag already running replaces nothing.

To refresh the benchmarker pair, use `BENCHD_REFRESH=1` (section 7).

## 14. Readiness checklist

- [ ] `docker.service` active; NVIDIA container toolkit configured
- [ ] five pinned snapshot files staged flat; `./setup.sh` verified them by bytes and sha256
- [ ] operator engine checkout at the engine branch tip, from a bundle with a matching sha256, clean tree
- [ ] container `spark-box` running the `spark-box:<engine sha>` image
- [ ] runner registered with label `<track>` (the runner adds `self-hosted, Linux, ARM64`); runner online
- [ ] benchd pair installed by `tools/fetch-benchd.sh` for `aarch64-unknown-linux-gnu`, manifest beside it
- [ ] goldens staged in `<root>/goldens/<track>`, one file per pin and nothing else
- [ ] reference tree staged and built under `<root>/baseline/workspace`; `<root>/baseline/baseline-calibration.json` written for this box
- [ ] `tools/ranked-box-preflight.sh` passes inside the container
- [ ] correctness check under the lock: exit 0, `.metrics.passed_correctness` true
- [ ] score check under the lock: `passed: true`; a serial declaration scores near 1.00, a depth declaration above it
- [ ] one `workflow_dispatch` of `benchmark.yml` passes end to end (one per fleet)

## 15. Faults seen in practice

| symptom | cause | fix |
|---|---|---|
| the container restarts in a loop and the log says `not registered` | the box is converged but not registered | register it (section 5) |
| `docker exec` fails with "container is not running" | the same | register it first; `docker exec` needs a running container |
| converge refuses: `docker is not installed` or `nvidia-container-toolkit is not installed` | the image is not a stock DGX OS | fix the base image; converge never installs a package |
| converge refuses: no CDI spec and no `nvidia` runtime | the NVIDIA container toolkit is not configured | run `nvidia-ctk` on the box first |
| converge refuses: no target snapshot | the weights are not staged | do section 2 first |
| a local check fails on `chmod`: `Read-only file system` | `<root>/benchd-bin` was mounted read-only | the mount is read-write; converge again |
| job fails at "runner environment": a path is missing | a mount source is empty or absent on the host | stage it, then converge again |
| `benchd iterate: weights digest failed (weights): No such file or directory` | `MLXFAST_WEIGHTS_PATH` not set for `benchmark.sh --local-iterate` | export it (section 11) |
| the score check refuses with `BASELINE-BOX-UNRESOLVED` | the operator shell has no `RUNNER_NAME` | export `RUNNER_NAME=<box>` (section 11) |
| the score check refuses a preset `SERVE_UP_SPECULATIVE` | the two serve values were exported in the shell | set them on the correctness check only (section 11a) |
| the score check refuses an inherited `DS4_RESIDENT_SOCKET` | the measure script was wrapped in `tools/serve-up.sh` | call it directly (section 11b) |
| a local check fails with `spec mode "mtp" is not runnable on this engine` | the resident was booted serial against a tree that declares a depth | derive the two values from `tools/spec-declaration.sh` (section 11) |
| correctness check: `gate rejected (prefill): GPU is hot and not cooling down` with nothing else on the GPU | the loaded resident holds the die above the 60 C local gate on this box | `MLXFAST_LOCAL_COOL_GATE=0` for that local check only (section 11a); never in the container environment |
| `git clone` of the bundle: `fatal: early EOF` or `index-pack died` | the bundle was truncated in transit; `scp` over a relayed link can return 0 on a partial file | compare sha256 on both ends; copy again with `rsync --partial` or in chunks |
| `git ls-remote` of the engine repository fails on the box | the box has no GitHub credential | expected; the operator checkout comes from a bundle; the ranked job uses the Actions token |
| `fetch-benchd.sh`: "manifest sha256 is not 64 lowercase hex characters" | a hand-made `benchd.manifest.json` with a `binaries` entry split over lines | one entry per line, the channel's layout |
| `benchd calibrate-baseline` refuses with `CALIBRATION-CV-EXCEEDED` | the four passes differed by more than 1% on one axis | a result, not a fault; find out why the box is noisy, then calibrate again |
| runner offline; log says a session already exists | a second listener | never start a second container; `docker logs spark-box` shows the one listener |
| preflight refuses an unpinned `*.json` in the pool directory | a non-pool golden placed beside the pool | move it out of `<root>/goldens/<track>` |
| a window stalls, GPU idle, two worker processes | a second connection waiting on the one-connection resident | one attached worker per window (benchd does this when `DS4_RESIDENT_SOCKET` is set) |
| speculative candidate misses the prefill band while serial passes | the engine re-uploaded the draft head on first use inside the timed prefill | fixed at ds4 pin 5f36517; the head stays resident from load |
| `nvidia-smi --query-gpu=memory.total` prints `[N/A]` | GB10 unified memory | read `/proc/meminfo`; `serve-up.sh` does (107 GiB available, 91 GiB required) |
