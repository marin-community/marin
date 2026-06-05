---
date: 2026-06-04
system: levanter
severity: degraded
resolution: mitigated
pr: https://github.com/marin-community/marin/pull/6176
issue: https://github.com/marin-community/marin/issues/6184
---

# Qwen3 Mixed-Prompt TPU Inference Progress

## TL;DR

- Levanter `mixed_b32_i512_o512_n1` on v6e-8 generated only 4096 of the expected
  16384 completion tokens, then stopped after two zero-progress decode
  iterations.
- The issue was caused by passing 32 prompts of 512 tokens into one engine
  `generate()` call while `max_prefill_size` was 4096. Only the first 8 prompts
  fit in the single prefill admission; the remaining 24 requests were never
  admitted.
- PR #6176 mitigates correctness by splitting OpenAI serving batches by
  aggregate prompt-token budget, so long-prompt workloads complete instead of
  silently dropping requests.
- This mitigation serializes the original 32-request workload into four
  8-request engine calls. Issue #6184 tracks the next performance fix:
  engine-level multi-prefill admission or overlap for one logical service batch.

## Goal

Goal: isolate why Levanter `mixed_b32_i512_o512_n1` on v6e-8 generated about 4096 completion tokens instead of 16384, then hit two zero-progress decode iterations.

## Initial Status

The dense Qwen3 expansion harness is on branch `codex/qwen3-dense-parity-matrix`. The prior v6e-8 mixed run used TP=8, `--max-pages 512`, `--warmup-rounds 2`, `--max-rounds 512`, and `mixed_b32_i512_o512_n1`. vLLM completed the expected 32 by 512 decode workload. Levanter reported about 1445-1463 decode tok/s, generated about 4096 completion tokens per measured round, then logged two zero-progress decode iterations and stopped.

## Hypothesis 1

The failure may depend on prompt length, output length, batch size, or decode round packing rather than on broad mixed workloads in general.

## Changes To Make

Add exact isolation cases to `dense_qwen3_8b_v2` without changing the default matrix:

- `mixed_b32_i512_o128_n1` checks whether nontrivial prompt with shorter output completes the expected 4096 completion tokens.
- `mixed_b32_i128_o512_n1` checks whether a shorter prompt with 512 output completes the expected 16384 completion tokens.
- `mixed_b8_i512_o512_n1` checks whether the original prompt/output shape succeeds at lower active-sequence pressure.

## Results

`/dlwh/qwen3-mixed-triage-v6e8-i512-o128-n1` succeeded at the Iris job level.

- Case: `mixed_b32_i512_o128_n1`, greedy/no-logprobs, TP=8, `--max-pages 512`, `--warmup-rounds 2`, `--max-rounds 512`, backend `both`.
- Expected completion tokens: 32 active sequences by 128 output tokens = 4096.
- vLLM canonical row: 4188.58 decode tok/s, 20942.91 total tok/s, 0.978 steady seconds.
- Levanter measured rows: 1140.17, 1144.64, 1142.01 decode tok/s; total tok/s 19382.88, 19458.86, 19414.19; HBM bytes 1207959552; shape buckets 2.
- Levanter generated 1024 completion tokens per measured round, not 4096.
- Each Levanter round logged one decode iteration with 1016 new tokens followed by two 0-new decode iterations and `No progress in decoding for 2 consecutive iterations; breaking to avoid hang.`

This shows the issue is not only the original 512-output target; a 512-token prompt with 128 requested output also stops after one quarter of the requested completion budget.

## Hypothesis 2

The engine admits only one prefill batch per `generate()` call. With `max_prefill_size` defaulting to 4096, `mixed_b32_i512_*` can only prefill 8 requests at a time because 8 by 512 = 4096. The remaining 24 requests are never admitted, while the host-side loop waits for all 32 expected children. That explains:

- `mixed_b32_i512_o128_n1` returning 1024 tokens = 8 admitted requests by 128 output tokens.
- The earlier `mixed_b32_i512_o512_n1` returning 4096 tokens = 8 admitted requests by 512 output tokens.
- The two zero-progress iterations after those 8 admitted requests finish.

`/dlwh/qwen3-mixed-triage-v6e8-i128-o512-n1` succeeded at the Iris job level.

- Case: `mixed_b32_i128_o512_n1`, greedy/no-logprobs, TP=8, `--max-pages 512`, `--warmup-rounds 2`, `--max-rounds 512`, backend `both`.
- Expected completion tokens: 32 active sequences by 512 output tokens = 16384.
- vLLM canonical row: 4983.27 decode tok/s, 6229.09 total tok/s, 3.288 steady seconds.
- Levanter measured rows generated all 16384 completion tokens with one decode iteration per round and no no-progress warnings.
- Levanter rows: 4335.58, 4315.63, 4312.87 decode tok/s; total tok/s 5419.48, 5394.54, 5391.08; HBM bytes 1207959552; shape buckets 2; decode ratios 0.870, 0.866, 0.865.

This supports the prefill-admission hypothesis: 32 by 128 prompt tokens exactly fits the default 4096-token prefill buffer, while 32 by 512 does not.

## Hypothesis 3

Increasing `--max-prefill-size` enough to fit all prompts in one prefill should confirm the diagnosis, but only if the larger prefill buffer is compatible with the engine's sequence-position buffers.

## Results

`/dlwh/qwen3-mixed-triage-v6e8-i512-o512-prefill16384` failed at the Iris job level.

- Case: original `mixed_b32_i512_o512_n1`, with `--max-prefill-size 16384`.
- vLLM completed normally; canonical row was 4902.84 decode tok/s, 9805.69 total tok/s.
- Levanter failed on the completion endpoint with `Axis position has different sizes ... 4096 != 16384`.

This does not disprove the prefill-admission hypothesis. It shows that simply making `max_prefill_size` larger than `max_seq_len` is invalid with the current prefill-work arrays. The smaller code fix is to prevent the OpenAI service batcher from passing a batch whose aggregate prompt tokens exceed the configured prefill buffer. That should split `mixed_b32_i512_*` into four 8-request engine calls instead of passing all 32 requests to one `generate()` call.

## Hypothesis 4

After splitting OpenAI service batches by aggregate prompt-token budget, the original `mixed_b32_i512_o512_n1` shape should complete all 16384 completion tokens without no-progress warnings. Throughput may drop because the service now processes four 8-request engine batches sequentially.

## Results

`/dlwh/qwen3-mixed-triage-v6e8-i512-o512-patched` succeeded at the Iris job level.

- Case: original `mixed_b32_i512_o512_n1`, default prefill size, patched OpenAI batcher.
- Expected completion tokens: 32 active sequences by 512 output tokens = 16384.
- vLLM canonical row: 4739.37 decode tok/s, 9478.74 total tok/s, 3.457 steady seconds.
- Levanter completed all 16384 tokens per measured round as four 4096-token engine batches. There were no no-progress warnings.
- Levanter rows: 1544.84, 1543.93, 1539.65 decode tok/s; total tok/s 3089.67, 3087.85, 3079.30; HBM bytes 1207959552; shape buckets 2; decode ratios about 0.326.

The progress bug is isolated: service batching must respect prefill prompt-token capacity. The implemented fix restores correctness and avoids the no-progress bail-out. The remaining performance gap is expected from serializing the 32-request long-prompt workload into four engine calls; the next performance fix would need the engine to admit multiple prefill chunks for one logical service batch or otherwise overlap those chunks without dropping requests.
