# Debugging log for Qwen3 mixed progress

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

Pending v6e-8 sequential Iris runs.
