# Per-module speculative configuration (MTP / DFlash / DSpark)

> **STATUS: NORMATIVE — signed off and IMPLEMENTED.** This document is the definition of the
> `spec` / `effective_spec` wire surface. It is cited as the contract by the frozen Engine
> Protocol v1 JSON Schema (`crates/bench-protocol/schema/engine-protocol-v1.schema.json`) and
> by 14 sites across `bench-protocol` and `bench-runner`; a change here is a wire change.
> Ranked-side operational claims remain UNVERIFIED(B-4). Repo facts cited at
> `mlxfast-bench@3fbca3a`.
>
> **Ruling (David, 2026-08-19): the depth setting is coded into each module.** There is no
> global depth. Each speculative path owns its configuration block — schema, defaults,
> bounds — and the benchmarker treats the block as opaque module input, sealed only as
> echoed back by the engine.

## 1. Why

- Review finding **R1**: `measure-job` seals `serial_depth: 0, candidate_depth: N` while
  neither leg ever configures a depth — provenance describes a run that never happened
  (`crates/benchd/src/measure_job.rs:48,369`). The point-fix (plumb a `--depth` flag)
  would harden a design error: depth is not a benchmarker concept; it is a property of a
  specific drafter module.
- David direction (2026-08-19): "better configuration for dspark, dflash, and mtp
  working with this" — one surface that configures all three paths across contract,
  CLI, wire, and both engines.

## 2. The config object — a tagged union, one block per module

```jsonc
// speculative config ("spec"): exactly one mode, config nested UNDER the mode key
{ "mode": "serial" }

{ "mode": "mtp",    "mtp":    { "depth": 2 } }

{ "mode": "dflash", "dflash": { "depth": 4,
                                "draft": { "artifact": "…", "sha256": "…" } } }

{ "mode": "dspark", "dspark": { /* RESERVED — schema pending cudafast#26 ruling
                                   (port-to-Qwen vs DeepSeek-only) */ } }
```

Rules:

- **Depth is a module field.** `mtp.depth` and `dflash.depth` are separate fields with
  separate defaults and bounds, owned by their modules. `serial` has no depth. `dspark`
  defines its own if/when #26 rules it in.
- **Module owns validation.** The engine-side module for each mode parses its own block:
  unknown fields → error (deny_unknown_fields, matching bench-protocol posture); missing
  fields → the module's coded default (MTP: the envelope's own resolution/clamp;
  DFlash: the module's block-size-derived default). The benchmarker never interprets
  block contents — it forwards bytes. Benchd itself supplies NO depth default (David
  ruling 2026-08-27): an omitted `--mtp-depth` sends `{"mode":"mtp","mtp":{}}` and the
  echo reports the module-resolved depth. (The pre-ruling text here said "MTP: depth 2
  per the Option A naive baseline" — that was benchd's retired injected default.)
- **Cross-module keys are rejected.** `{"mode":"mtp","dflash":{…}}` is an error, not
  ignored — fail-closed against config drift.
- **Artifacts are pinned.** Any module block that names an external artifact (DFlash
  draft gguf) carries a mandatory `sha256`; the engine verifies before load (same
  posture as golden pins).

## 3. Where the block lives, seam by seam

| Surface | Change | Notes |
|---|---|---|
| **Contract / track fixture** | `speculative: { candidate: <spec>, baseline: <spec> }` | `baseline` defaults to `{"mode":"serial"}`. Ranked shape UNVERIFIED(B-4). |
| **measure-job CLI** | reads from `--contract`; `--candidate-spec / --baseline-spec <json>` as explicit overrides (recorded as `spec_source: "cli-override"` in provenance) | replaces the R1 point-fix; no `--depth` flag anywhere |
| **Wire (Engine Protocol v1, additive)** | `decode_begin` request gains optional `spec` object; the **response echoes `effective_spec`** — the module-parsed, default-filled block the engine will actually run. `hello` gains optional `spec_modes: ["serial","mtp",…]` capability list. | Additive like `cache_memory`; absent `spec` = engine default (v1 engines unchanged/valid). Unsupported mode → error + session discard (fail-closed). |
| **results.json provenance** | keeps the echoed effective mode; omits engine-derived numeric fields | requested config pins stay exact; private audit can retain the full echo |
| **score.json `metrics.per_prompt`** | the official path and the local paths seal one record for each timed prompt | Score timings stay exact. Engine-derived counters and means are omitted. |
| **`benchd iterate` CLI** | `--mtp-depth <N>` builds `{"mode":"mtp","mtp":{"depth":N}}` for the TIMED free-run decode window; `--candidate-spec <json>` is the explicit override. The two are mutually exclusive, and the depth obeys the same cap as measure-job | ABSENT = no `spec` on the wire = the engine's default (serial). The teacher-forced correctness gate never carries a spec |
| **score.json spec + identity seal** | `effective_spec_mode` and text identity fields | Numeric spec audit fields are omitted from the public seal. |
| **v1.1 free-run mode** | orthogonal; unchanged | scoring any of the three still requires v1.1 (mlxfast-bench#100) |

### 3.1 The `score.json` speculative-decode and engine-identity keys

The sealed `score.json` records the effective mode as text. It omits `effective_spec_depth`,
spec counters, acceptance rates, verification-round counts, acceptance arrays, resident process
numbers, and per-prompt draft-length means. Benchd still checks these exact values in memory.

The identity keys come from the timed worker's `hello`. Free-text values use
`engine-text-sha256` seals. `head_provenance_sha256` and `runner_manifest_sha256` stay verbatim
only as 64 lowercase hex characters. Other digest text uses an `engine-text-sha256` seal.
The in-window resident identity check is unchanged. None of these fields changes a score or gate.

### 3.2 Sealed numbers

Benchd applies this rule only during JSON serialization. Checks, gates, decisions, and scores use
exact in-memory values. The public seal removes each JSON number and numeric array unless its path
is in this allow-list. The seal also removes an object if number removal leaves it empty.

- Score and score inputs stay exact because readers must recompute the published score:
  `score`, `metrics.{decode,prefill,baseline_decode,baseline_prefill}_seconds_per_token`,
  `metrics.{decode,prefill}_speedup`, `metrics.{decode,prefill}_speedup_floor`,
  `metrics.{baseline,candidate}_leg_{prefill,decode}_seconds_per_token`,
  `metrics.{baseline,candidate}_leg_decode_window_seconds_per_token`,
  `metrics.per_prompt[].mtp_seconds_per_token_mean`, and
  `metrics.paired_legs[].{control,candidate}_{prefill,decode}_seconds_per_token`.
  The allow-list also includes
  `metrics.paired_legs[].{control,candidate}_decode_window_seconds_per_token`,
  `decode_speedup_ceiling`,
  `cohort.{prefill_gain,decode_gain,composite_score}`,
  `per_cohort[].composite.{prefill_gain,decode_gain,composite_score}`,
  `pairs[].{serial,mtp}_seconds_per_token`, `pairs[].raw_ratio`,
  `pairs[].cohort_phase_windows.{serial,candidate}_{prefill,decode}_window_seconds`,
  `aggregate.{baseline_serial,candidate_mtp}_seconds_per_token_mean`,
  `aggregate.{raw_decode_speedup_median,mtp_decode_speedup_median,mtp_decode_speedup_min}`,
  `per_prompt[].{serial,mtp}_seconds_per_token_mean`, and
  `per_prompt[].{raw_ratio_of_means,noop_reference_decode_speedup}`,
  `per_cohort[].{serial,candidate}_seconds_per_token_mean`,
  `per_cohort[].raw_ratio_of_means`, and
  `per_cohort[].composite_scored_exponents.{prefill_gain_exponent,decode_gain_exponent}`.
- Fixture and config pins stay exact because they define the run:
  `metrics.{gpqa_ttft,semantic_gpqa}_case_count`, `metrics.num_layers`,
  `metrics.case_count`, `metrics.weights_{byte,file}_count`, `metrics.engine_protocol_version`,
  `metrics.{timed_token_tolerance_per_thousand,timed_token_near_tie_relative_gap}`, and
  `decode_tokens`, `{min_pairs,min_pairs_per_prompt,pairs_per_prompt,prompt_count}`,
  `{serial_control_depth,mtp_depth,scored_batch_size,pairs_per_cohort,min_pairs_per_cohort}`,
  `{candidate_spec,baseline_spec}.{mtp,dflash}.depth`, `pairs[].prompt_index`,
  `per_prompt[].prompt_index`, `per_cohort[].cohort_index`,
  `per_cohort[].members[].{slot_index,bytes}`, and `per_cohort[].batch_size`.
- Benchd-selected indices and official counts stay exact:
  `metrics.paired_legs[].pair`,
  `{accepted_pair_count,per_prompt[].accepted_pair_count,per_cohort[].accepted_pair_count}`,
  and `cohort.member_count`.
- Token-window pins stay exact:
  `pairs[].phase_windows.{prefill,decode}_token_total`,
  `pairs[].cohort_phase_windows.{prefill,decode}_token_total`, and
  `per_cohort[].{prefill,decode}_token_total`.
- Limits stay exact because they define acceptance:
  `aggregate.{mtp_max_draft_depth,decode_speedup_floor,published_speedup_ceiling}`,
  `cohort.{composite_speedup_floor,prefill_gain_exponent,decode_gain_exponent}`, and
  `per_cohort[].composite.composite_speedup_floor`.
- Gate pins stay exact:
  `provenance.thermal.{cool_gate_c,clock_floor_mhz,loaded_util}`,
  `provenance.baseline_calibration.{band_low,band_high,decode_tokens}`, and
  `provenance.serial_band_outcome.{band_low,band_high,calibration_decode_tokens,run_decode_tokens}`.

This list does not include engine audit counters, rates, non-scoring means, non-scoring medians,
throttle counts, truncation counts, per-stream ratios, clamp counts, `checked_steps`, or
`first_failing_step`.

In `error`, `reason`, `detail`, `*_reason`, and `*_refused` text, each digit run becomes `#`.
The seal does not change digits inside `engine-text-sha256=<hex>` or `engine-text-bytes=<value>`.

If `BENCHD_PRIVATE_AUDIT_PATH` is set, benchd writes the full JSON from before this pass to that
path. On Unix, benchd creates or resets the file with mode `0600`. If the variable is unset,
benchd writes no private copy. Benchd never writes the private JSON to standard output or standard
error.

## 4. Engine-side module mapping

- **mlxfast-engine (Metal):** the MTP driver module consumes `mtp.depth` (Swift-side
  clamp `depth_max.min(ctx − pos − 2)` stays module-internal). `dflash`/`dspark` modes
  → capability-absent → wire error (correct: Metal has no such modules today).
- **cudafast-engine (adapter → Pulsar):** the adapter translates block → module
  activation: `mtp` → nextn path with the block's depth; `dflash` → `PULSAR_DFLASH` +
  sha-verified draft gguf (enablement: cudafast#25); `dspark` → reserved pending
  cudafast#26. Translation lives in the adapter (`harness/protocol-adapter/`), NOT in
  vendored `engine/` — the fork stays clean.

## 4.5 Stub modules + a drafting-model slot for each path (David, 2026-08-19)

Every engine ships **one module per mode — implemented or STUB** — so the structure
exists everywhere and enablement is fill-in, never restructure:

- **Module table is total.** Both engines register `serial`, `mtp`, `dflash`, `dspark`
  modules. A stub module still **parses and validates its full config block** (so schema
  enforcement is uniform across engines) and then fails closed with a distinct
  `mode not implemented on this engine` wire error. `hello.spec_modes` lists only
  *runnable* modes — stubs are visible in code, never in capability claims.
- **Drafting-model slot per module.** Each module — stub included — declares its
  drafting-model manifest entry: `mtp` → the native head (Metal: fork weights; CUDA:
  the reference bf16 MTP head, 15 tensors under `mtp.`); `dflash` → draft gguf
  (artifact + sha256); `dspark` → reserved slot, schema TBD by cudafast#26. The slot
  defines where the artifact mounts, what pins it, and what the module loads — filling
  the slot + deleting the stub error IS the enablement task (cudafast#25 for DFlash).
- **Stubs are contestant-visible structure.** On the CUDA side the module table sits in
  the adapter (`harness/`, pinned), while the modules' engine hooks live in `engine/`
  (editable) — a contestant improving a path edits the module body, never the table.
- **The submission picks which module to import (David, 2026-08-19).** The candidate
  workspace declares its chosen path in a manifest field —
  `speculative: { "mode": "dflash", "dflash": { … } }` — and measure-job takes that as
  the candidate-leg spec: the contestant selects the module (and supplies its drafting
  model, BYO per the declared-head policy precedent); the track contract's role shrinks
  to (a) the **allowed-modes list** for the track and (b) the baseline spec (serial).
  A submission declaring a mode outside the track's allowed list, or one the engine
  reports as stub/unsupported, REJECTS before any timed work. Provenance seals the
  submission-declared spec, the track constraint it satisfied, and the engine-echoed
  `effective_spec` — three fields, so a mismatch anywhere is visible in the artifact.

## 5. What this deliberately does not do

- No global depth, no benchmarker-side interpretation of module blocks.
- No scoring change: mode selection never alters the regime; comparability rules
  (v1.1 judge-less series) unchanged.
- No ranked-wrapper claim: whether the organizer contract carries `speculative` blocks
  is UNVERIFIED(B-4); until then this is the local + Spark surface, and measure-job's
  contract parser treats an absent `speculative` block as serial-vs-engine-default with
  provenance saying exactly that.

## 6. Acceptance

1. Unit: per-module parse/default/bounds tests; cross-module-key rejection; unknown-mode
   fail-closed; artifact-pin mismatch fail-closed.
2. Conformance kit: `decode_begin.spec` echo semantics + capability gating, positive and
   negative controls, mock engine.
3. measure-job: sealed `effective_spec` matches the engine echo byte-for-byte on both
   legs; a leg whose echo differs from request records `spec_source` divergence and the
   run REJECTS (no silent fallback).
