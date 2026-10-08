# Bonsai 9B benchd channel (`gumbii/bonsai9b-own`)

This branch is the **benchd dist channel** for the Bonsai 9B CUDA track (`bonsai-9b-cuda-v1`). It forks from `gumbii/nemotron35-own` at the same scoring, gates, sandbox, and sealing logic; only harness identity and published binaries differ.

## What changed vs `gumbii/nemotron35-own`

| Area | Nemotron channel | Bonsai 9B channel |
|------|------------------|-------------------|
| Dist manifest `branch` | `gumbii/nemotron35-own` | `gumbii/bonsai9b-own` |
| `HARNESS_HASH_ROOTS` | 11 roots (MLX nine + `harness/` + `vllm/`) | 12 roots: same roster **plus `llama.cpp/`** |
| Prebuilt Linux binaries | Built from prior `nemotron35-own` commits | Rebuilt from `77ee5b0` (roster change only) |

### Why `llama.cpp` was added

The Nemotron CUDA engine vendors **vLLM** at `vllm/`; the Bonsai engine swaps that for a pinned **PrismML llama.cpp** fork at `llama.cpp/` (see `bonsai-9b-challenge` design §2–3). Harness hash must cover the participant-editable engine tree so benchd’s trusted-side `harness_hash` matches what the engine computes over the same workspace.

David’s missing-root rule still applies: roots absent on disk are logged and skipped, so **MLX-shaped trees and Nemotron CUDA trees without `llama.cpp/` keep the same digest** they had on `nemotron35-own`. A Bonsai workspace with files under `llama.cpp/` gets a different (correct) identity.

### What did **not** change

- Scoring, gate bands, seal logic, overlay, editable-surface gates, and roster-of-eight trusted scope.
- No engine-repo or track-repo code on this branch (benchd-only step **1d**).

## Commits on this branch (relative to `gumbii/nemotron35-own`)

1. `77ee5b0` — `benchd: add llama.cpp to harness hash roster for Bonsai CUDA`
2. `7591948` — `dist: linux-aarch64 + x86_64 benchd from source 77ee5b0 (bonsai9b channel)`

## Dist layout

- `dist/linux-aarch64/` — Spark boxes (`aarch64-unknown-linux-gnu`), built on `spark6` under `~/bonsai-dev/1d/`
- `dist/linux-x86_64/` — developer workstations (`x86_64-unknown-linux-gnu`), built locally with `scripts/build-dist.sh`

Manifests name `source_commit` `77ee5b08606a5a1213aa543c24314666baee100d` and `branch` `gumbii/bonsai9b-own`.

## Consumption

Engine/track repos point `tools/fetch-benchd.sh` (or equivalent) at this branch instead of `gumbii/nemotron35-own` once the lead reviews and pushes. **This branch is not pushed by the 1d worker** per challenge brief.
