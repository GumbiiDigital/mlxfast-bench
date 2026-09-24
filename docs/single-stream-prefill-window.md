# Single-stream prefill window (wire contract)

Ruling (David, 2026-08-27): score the Qwen 3.8 125B-A6B tracks (mlx and cuda)
single-stream, paired serial vs the built-in mtp, on
`prefill_gain ^ 0.25 * decode_gain ^ 0.75`. This page is the wire contract for the
prefill half of that score on the single-stream series. The engine side and the
bench side implement it from this page.

## 1. Summary

- No new message. No new field on the wire. The single-stream free-run verbs
  stay `free_decode_begin` and `free_decode_run`.
- benchd splits its own parent clock at the boundary between the two verbs.
- The engine does not emit a timing. The engine must do all seed-prefill work
  inside `free_decode_begin`.
- The track fixture declares `score_prefill_weight: 0.25` and
  `score_decode_weight: 0.75`. Those are the exponents benchd raises the two
  gains to. These tracks declare no `scored_batch_size`: they measure their own
  denominator on the live control leg, at one stream.

## 2. The two windows

benchd runs one worker for each leg, and that worker runs the whole leg. For
each leg it opens two contiguous windows on its own clock
(`std::time::Instant`):

| Window | Opens | Closes | Token count |
|---|---|---|---|
| `prefill_elapsed_seconds` | immediately before benchd sends `free_decode_begin` (with the golden's `decode_seed_tokens`, 1024 tokens, and the requested `spec`) | immediately after benchd validates the response `seed_token` against the golden's `expected_decode_seed_token` | `prefill_token_total` = `decode_seed_tokens.len()` |
| `decode_elapsed_seconds` | the instant the prefill window closes | when `free_decode_run(N)` returns | `decode_token_total` = `N` (128) |

Rules:

- There is no untimed gap between the two windows.
- `elapsed_seconds` = `prefill_elapsed_seconds + decode_elapsed_seconds`, by
  construction. benchd never measures the whole window a second time.
- `seconds_per_token` = `elapsed_seconds / N`. This is unchanged. It stays
  the enforced whole-window figure for the serial band, the run timeout and
  the paired decode-only median.
- The seed oracle check is charged to the prefill window. The run oracle check
  is outside both windows.
- The RunTimeout deadline is armed when the prefill window opens and covers
  both windows, as today.

## 3. Engine obligations

1. `free_decode_begin` must run the full seed prefill (all `decode_seed_tokens`)
   and must reply only after the seed forward is complete. The reply carries
   `seed_token` (the argmax after the full seed) and the echoed `effective_spec`.
2. `free_decode_run` must not prefill. It must decode from the state that
   `free_decode_begin` left. It must not re-run any part of the seed.
3. The engine must not do any prefill work before `free_decode_begin` arrives.
   One worker runs the whole window, so the engine must reset its KV cache and
   its allocator at the start of each phase. benchd makes sure of the reset at
   each phase boundary: the `phase_diagnostics` reply must report a
   `cache_memory` of 0, or benchd refuses the run.
4. The hello must advertise `free_run_decode`. The single-stream series does
   not use `batched_free_run_decode` or `max_batch_size`. benchd does not read
   them on this series.
5. Units on the wire stay as they are. benchd's clock is in seconds (f64).

If the engine moves prefill work into `free_decode_run`, the prefill window
gets smaller and the decode window gets larger by the same amount. The whole
window does not change. The composite then moves against the engine, because
the decode exponent (0.75) is larger than the prefill exponent (0.25).

## 4. What benchd seals

The ranked path of these tracks is `benchd iterate --mode official`, and it seals
`score.json`. Per role it seals the per-token times of the legs
(`baseline_prefill_seconds_per_token`, `baseline_decode_seconds_per_token`,
`prefill_seconds_per_token`, `decode_seconds_per_token`), the two gains
(`prefill_speedup`, `decode_speedup`) and the composite as the run's `score`.
`metrics.paired_legs` carries one row for each measured pair.

benchd measures the pairs the fixture declares in `official_pairs` and scores
one of them. Each pair has its own composite, from its own control leg:

```
prefill_gain = control prefill s/tok / candidate prefill s/tok
decode_gain  = control decode s/tok  / candidate decode s/tok
composite    = prefill_gain ^ 0.25 * decode_gain ^ 0.75
```

The run scores the pair whose composite is the lower median over the pairs:
the middle pair on an odd count, the lower of the two central pairs on an even
count. Pairs are never averaged. Every enforced figure in `score.json` is that
one pair's, and the other pairs stay in `metrics.paired_legs` as measured.

The window split of section 2 is what makes the two gains separable. Without it
there is one whole-window number and no prefill half to score.

## 5. What does not change

- `free_decode_begin` / `free_decode_run` request and response fields.
- The captured engine-wire fixture (`ENGINE_WIRE_V1_SHA256`). No re-pin.
- The whole window and its `seconds_per_token`. The split cannot move them.
