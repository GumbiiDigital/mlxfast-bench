# benchd

What benchd is, what it owns, how one scored run works, and how an engine repository gets it. Read this page first. The other pages go deeper on one part each.

Every fact on this page is read from `main`. The page cites a file, not a line: a line moves, a file does not.

## 1. What benchd is

benchd is the trusted benchmarker. It drives one inference engine over a frozen wire protocol, holds the clock, checks the engine's tokens against the oracle, computes the score, and seals the artifact. It never links model code.

| benchd owns | benchd does not own |
|---|---|
| Engine Protocol v1: the wire types, the JSON Schema, the verb semantics (`crates/bench-protocol/src/lib.rs`) | The model, its weights, its drafters, its kernels |
| The parent-side wall clock over every round trip. No engine-reported time is scored | Tokenization. Only token ids cross the wire |
| Phase barriers, allocator-drain checks, the completed-work counter (`crates/bench-runner/src/session.rs`) | The engine binary and how it is built |
| Oracle checks: exact match of free-run tokens, or the timed token tolerance of the paired path, teacher-forced tolerance gates, cohort reference replay on the trusted build | The submission boundary. Yukon owns it, through the engine repository's `benchmark.json` |
| The score formula, floors, bands, and the sealed `score.json` with its sha256 | Box tooling: fan control, preflight scripts, runner registration. The thermal threshold itself is benchd's |
| The organizer's pinned scoring inputs: baseline pairs, reference checkpoint identity, scored regime, golden integrity pins, resolved from the track fixture first (`crates/bench-core/src/contract.rs`) | Serving. benchd measures one engine process, not a fleet |

## 2. The workspace

| Crate | Role |
|---|---|
| `bench-protocol` | Engine Protocol v1 wire types and JSON Schema. Normative. The envelope is closed: an unknown field is refused |
| `bench-core` | Golden schema, score formula, floors, bands, sealing, conformance kit, per-track constants |
| `bench-runner` | Engine session: spawn, sandbox, hello handshake, nonce and id checks, phase barriers, parent-side timing, environment allowlist |
| `benchd` | The CLI. Every flow runs through it (`crates/benchd/src/main.rs`) |

Prebuilt binaries ship in `dist/` per platform: `benchd` and `record-correctness-golden`, each with a `benchd.manifest.json` that carries the branch, the source commit, the sha256, and the byte count (`scripts/build-dist.sh`, `scripts/dist-lib.sh`).

## 3. One scored run

`benchd iterate --mode official` is the whole run. The engine repository's driver script resolves the binary and forwards the track id, the fixture path, and the golden pins. It configures nothing else.

1. Resolve the track. The `MLXFAST_QWEN_MTP_TRACK_ID` value and the `--contract` fixture must agree. The platform comes from the platform token in the track id, never from a branch name (`crates/bench-core/src/constants.rs`). benchd reads the fixture one time and keeps the sha256 of those bytes. It then certifies each value the fixture declares: a speedup floor must be more than zero, a pair count must be one or more, and a declared batch size must be one stream or more. A bad value stops the run at this step (`crates/bench-core/src/contract.rs`).
2. Resolve the live golden. Verify its integrity pin, sha256 and bytes, from the fixture. A pending track refuses by name.
3. Quiescence gate, then cool gate. Wait for the box to go idle, a 1-minute load average below 2.0 and GPU utilization below 0.10 (`crates/benchd/src/quiescegate.rs`), then wait for the GPU to reach the fixed thermal contract (`crates/benchd/src/coolgate.rs`). Both gates run before every timed phase, in that order. Preflight the window (`docs/window-preflight.md`).
4. Spawn the engine in the sandbox with an environment built from empty and an allowlist (`crates/bench-runner/src/transport.rs`; `crates/bench-runner/src/sandbox.rs`). The child environment is byte-identical across the unscored and the scored pass. On a paired run benchd boots the resident engine of each leg from that leg's own workspace, and the spawned worker attaches to it over a Unix socket (`crates/benchd/src/legserve.rs`).
5. Hello. Check `protocol_version`. Record backend, device, capabilities, spec modes, max batch size, head provenance (`crates/bench-runner/src/session.rs`). Refuse every verb the engine did not advertise, before the clock starts.
6. Correctness gates. Teacher-forced anchor cases and the free-run prefix, inside the golden's tolerance bands, with the allocator drained at the start of every sequence.
7. Two timed windows on one clock. The prefill window opens when benchd sends `free_decode_begin` with the seed tokens and the requested `spec`. It closes when the returned seed token validates against the golden. The decode window opens at that instant and closes when `free_decode_run` returns. There is no untimed gap. The engine emits no timing (`docs/single-stream-prefill-window.md`; `crates/bench-runner/src/session.rs`).
8. Phase close. `phase_diagnostics` must report `completed_work` equal to the issued forwards and `cache_memory` of zero (`crates/bench-runner/src/session.rs`). The echoed `effective_spec` and `effective_batch_size` must equal the request. A divergence discards the leg. The whole window runs over one attached worker, so the weights load one time (`run_timed_window`, `crates/benchd/src/official.rs`). The drain assertion, not a fresh process, is what isolates one phase from the next.
9. Oracle. By default every committed token is exact-matched against the golden continuation, and one different token fails the run. A track fixture can declare `timed_token_tolerance_per_thousand` (N) instead. Then the candidate leg of a paired run does not stop at a different token. After the last pair, the reference tree's engine replays the candidate's own tokens teacher-forced, and a pair fails when more than N per thousand of its tokens differ from the reference engine's choice for the same prefix (`TIMED-DIVERGENCE-OVER-TOLERANCE`). The serial-control leg stays exact. A candidate whose tokens equal the oracle is not replayed. The cost of the rule: with a tolerance, a ranked run does not prove that the candidate's output is the reference output. It proves that at most N per thousand of its tokens differ from what the reference engine chooses for the same prefix. A fixture can also declare `timed_token_near_tie_relative_gap` (G) beside the tolerance. That selects the near-tie rule: a different token is tolerated only when it is the reference engine's second choice and the relative gap between the reference's first and second choice is at most G. Any other different token fails the run (`TIMED-DIVERGENCE-NOT-A-NEAR-TIE`), whatever the count, and near ties still count against N. The cost of the near-tie rule: it forgives a change of winner between two tokens that the reference engine scores almost the same, and nothing else. Audit counters are cross-checked for internal consistency and never scored.
10. Score. Seal `score.json` as `{score, metrics}` with its sha256 and the benchd identity (`crates/benchd/src/iterate.rs`). The seal records the sha256 of the `--contract` bytes as `metrics.contract_sha256`, beside the golden's own sha256, and records in `metrics.contract_sources` which of the eight resolver groups the fixture declared.

When a track scores prefill, the split is certified: a leg with no observable prefill window refuses with `PREFILL-WINDOW-NOT-OBSERVED`, and a split that disagrees with the whole-window figure refuses with `PREFILL-WINDOW-DISAGREES` (`crates/bench-core/src/prefill_window.rs`).

Two invariants carry the trust model. The engine is never trusted for time: the clock is on this side of the wire. The engine is never trusted for correctness: the tokens are checked against material the engine never saw.

## 4. Engine Protocol v1

NDJSON over stdio. A request is `{id, kind, ...}`. A response is `{id, nonce, ok, ...}`. `hello` is the unsolicited `id = 0` response the engine emits once, after in-engine weight validation. The full definition is `crates/bench-protocol/PROTOCOL.md` and the schema beside it.

| Kind | In | Out | Timed |
|---|---|---|---|
| `prefill` | prompt_tokens[] | token | no |
| `decode_begin` | seed_tokens[], spec? | seed_token, effective_spec | yes |
| `decode_step` | token | token | yes |
| `correctness` | prompt_tokens[], steps | tokens[], peak_ram_gb | no |
| `correctness_begin`, `correctness_step` | prompt_tokens[] or token | token, top_logits[8] | yes |
| `free_decode_begin` | seed_tokens[] or seed_tokens_by_stream[][], batch_size?, spec? | seed_token(s), effective_spec, effective_batch_size | prefill window |
| `free_decode_run` | count, batch_size? | tokens[] or tokens_by_stream[][], acceptance audit | decode window |
| `cohort_reference_replay` | replay_seeds_by_stream, committed_by_stream, replay_width | per-stream reference argmax | no; trusted build only |
| `phase_diagnostics` | none | completed_work, cache_memory, memory watermarks | barrier |

The timed-step set for the phase barrier is exactly `decode_begin`, `decode_step`, `correctness_begin`, `correctness_step` (`crates/bench-protocol/src/lib.rs`). A free-run phase counts `R + 1` forwards instead.

Capability flags on `hello`: `free_run_decode`, `batched_free_run_decode`, `per_stream_timing`, `cohort_reference_replay` (`crates/bench-protocol/src/lib.rs`). Speculative configuration rides as `spec`, a tagged union over `serial`, `mtp`, `dflash`, `dspark` (`crates/bench-protocol/src/lib.rs`; `docs/spec-config-design.md`). The engine's echoed `effective_spec` is the only sealed value. Every extension is additive and gated by a flag. The envelope stays closed.

## 5. How an engine repository gets benchd

The engine repository runs its `tools/fetch-benchd.sh` from its `setupCommand`. The script downloads the channel's `benchd.manifest.json` first, then the binary from the same directory, verifies sha256 and bytes, and installs only what verified. An installed pair that agrees with its manifest is used offline. A pair that disagrees refuses.

| Resolution | What selects the binary | Where it applies |
|---|---|---|
| Channel tip | The bench branch name. The tip is what runs. A benchd fix reaches the box with no engine commit | Gemma 4 and the 27B track |
| Channel commit | A bench commit hash in the fetch script. The URL names the commit. The manifest must name the expected branch. sha256 and bytes verify. To advance benchd, change one line in the engine repository | Qwen 3.8 125B-A6B, on both platforms |

In both cases provenance is recorded, not trusted. The resolved branch, source commit and sha256 are logged on every resolve. The manifest is installed beside the binary. benchd writes `benchd_sha256` into the calibration and capture identity (`crates/benchd/src/calibrate.rs`, `crates/benchd/src/capture.rs`). The fetch script and the channel constants are outside the engine repository's editable paths, and its submission guard forbids the `benchd.pin` and `benchd-bin` spellings.

## 6. Where the scoring inputs live

Every scored value comes from the `--contract` track fixture (`crates/bench-core/src/contract.rs`). benchd holds no per-track table. A value the fixture does not declare is a refusal, by name, before any engine starts.

`--contract` is required on every path. `measure-job`, `overlay-timing`, `validate-golden` and `calibrate-baseline` refuse without it by name, and `iterate --mode official` refuses when the command line is parsed. The local modes of `iterate` refuse as soon as they resolve the window shape or the model shape, because neither has a fallback left. There is no local default.

One schema reads the fixture, `bench_core::contract::Contract`. Eight groups resolve from it: the scored regime, the live-control-leg selector, the paired-flow selector, the official baseline pair, the acceptance bands, the scoring weights, the window shape, and the model shape. Each group is all-or-nothing: a half-declared group is refused. `docs/architecture.md` §5.1 lists every field of every group.

`metrics.contract_sources` seals one entry for each group. It reads `"contract"` for a group the fixture declares and `"device"` for a group the device measures for itself. The 125B tracks declare no scored regime and no official baseline: each device tracks its own reference benchmark, measures it on that box in the same job, and scores the live ratio against it. benchd does not upload that reference and does not save it to a centralized repository. The 27B track declares both groups. `"table"` is a historical spelling that no run seals any more; it stays readable so an artifact sealed before the tables were deleted still reads as what it was.

benchd certifies a declared value at parse time, not at use. A floor must be more than zero. A pair count must be one or more. A declared batch size must be one stream or more. The refusal lands on the file, before any box time is spent.

The batch size is part of the configuration. The fixture declares `scored_batch_size`, and that width selects the measured path: 1 is the single-stream path, and a width above 1 is the batched cohort path. Only a width with a certified series tag runs the cohort path; today that width is 8.

A benchd built from this `main` refuses a fixture that does not carry the keys of a group it needs, so an engine repository publishes its fixture keys before the benchd that requires them.

`crates/benchd/tests/fixtures/contract-full/` holds one full fixture for each track this tree carries a reference copy of. A test proves that each one resolves, group by group, to the expected values checked in beside it. Gemma does not migrate: it runs its own channel benchd, and this tree carries none of its values.

`targets/README.md` describes a further replacement: one signed `target.toml` per model and platform. Nothing is checked in yet.

## 7. Who talks to benchd

| Party | How |
|---|---|
| Engine repository | Its driver script calls `benchd` and forwards pins. It configures the invocation, never the measurement |
| Yukon | Indirectly. It reads `score.json` where `benchmark.json` says it is. It never sees benchd |
| Operators | `benchd iterate --capture-baseline` to capture a pending track's official pair (`docs/official-baseline-capture.md`); `calibrate`; `validate-golden`; `weights-digest` |
| First-party runners | The generic Swift `bench-worker` in the mlx-swift-lm fork, one binary for every CBv2 model, behind the runner contract (`docs/runner-contract.md`). The Mac engine repository builds its own copy as `track-bench-worker` |
| Third-party runners | Any binary that speaks the wire and ships a manifest. Direction: an SDK crate beside `bench-protocol`, with the conformance kit (`crates/bench-core/src/conformance.rs`) as the onboarding gate. The ds4 `cuda-engine` adapter is the first runner of this shape |

## 8. What a run records

- Score. Per track. The 125B tracks seal `prefill_seconds_per_token` and `decode_seconds_per_token` from the two windows, derive `prefill_speedup` and `decode_speedup` against the pinned baseline pair, and seal `score = decode_speedup^0.75 * prefill_speedup^0.25` with a floor flag on each speedup (`crates/bench-core/src/score.rs`; `crates/benchd/src/iterate.rs`). The 27B track is decode-only through the paired flow (`crates/bench-core/src/score.rs`). `composite_score` reads the declared regime and has no production call site.
- Timing. Parent-clock windows, prefill and decode, contiguous, per leg. `elapsed_seconds` is their sum (`crates/benchd/src/iterate.rs`). Engine-reported per-stream nanoseconds are sealed as report-only diagnostics.
- Spec. The echoed `effective_spec` and, for a cohort, `effective_batch_size`.
- Audit. `spec_decoder`, `drafted_total`, `accepted_total`, `acceptance_lengths`, `natural_accepted_by_stream`, `depth_clamp_reasons`, `verify_replay_disagreements`. Cross-checked, never scored.
- Identity. Track id, platform, golden pin, harness hash, head provenance, `benchd_sha256`, hardware digest, `contract_sha256` and `contract_sources`.

## 9. Direction

These items are agreed direction, not shipped behavior. Each one changes this page when it lands.

| Item | Change in benchd | Status |
|---|---|---|
| Runner identity on the wire | Additive `hello.runner` block: id, model type, manifest digest, build. Sealed like `head_provenance` | pending a two-verdict PR |
| Third-party SDK | An SDK crate beside `bench-protocol`: the NDJSON loop, the hello builder, nonce and barrier handling over an `Engine` trait. The earlier draft crate was deleted as dead code | not started |
| Platform as data | The per-track tables are deleted and the track fixture is the only source | done |
| One scored flow per regime | Free-run at the batch the fixture declares, paired or single-leg by the same declaration | not started |
| Conformance kit as vendor gate | Accepts a manifest, checks `hello` against it, runs per registered runner | not started |
| benchd on `main` | `main` publishes its own `dist/` for all three platforms. The 125B engine repositories still pin the project channel branch; each one moves with a one-line change | partly landed |

## 10. Standing rules

1. Measurement lives in benchd. An engine reports raw counters and never a time.
2. A baseline is measured by the scored path itself, on the box, once, and pinned.
3. A report-only surface becomes a scored input only by an atomic change with explicit refusals.
4. A reference that moves toward the candidate needs a negative control.
5. Our engine, our benchmark: no anti-gaming hardening beyond what the wire and the oracle give.
6. Calibration and acceptance run once, at the end, on the secured tip.
