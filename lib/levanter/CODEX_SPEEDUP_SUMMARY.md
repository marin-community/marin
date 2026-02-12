# CODEX Speedup Summary (M5 -> M8)

Date: 2026-02-07

## Scope and Metric

- Primary metric: `round_total_s` (lower is better).
- Primary workload for speed tracking: `10 prompts x 2048 max_new_tokens x 1 round` on `v5p-16`.
- M5 did not run a dedicated 10x2048 perf milestone, so the **M5 baseline** here is the first M6 baseline run that keeps M5-safe semantics (no M6+ perf tuning yet):
  - `M5 baseline round_total_s = 505.266s`
  - Source: `/tmp/levanter_run_m6_baseline_10prompts_2048_round1_m61.log`

## Stage-by-Stage Timeline

| Stage | Workload | round_total_s | Speedup vs M5 baseline | Faster vs M5 baseline | Delta vs prior stage |
|---|---|---:|---:|---:|---:|
| M5 baseline (carried into M6.1) | 10x2048 | 505.266 | 1.000x | 0.00% | n/a |
| M6.2 | 10x2048 | 505.000 | 1.001x | 0.05% | 0.05% |
| M6.3 | 10x2048 | 164.534 | 3.071x | 67.44% | 67.42% |
| M6.4 | 10x2048 | 161.859 | 3.122x | 67.97% | 1.63% |
| M6.5 | 10x2048 | 161.337 | 3.132x | 68.07% | 0.32% |
| M6.6 (M6 final) | 10x2048 | 160.436 | 3.149x | 68.25% | 0.56% |
| M7 | 1x2048 (OOM-fix track) | n/a | n/a | n/a | n/a |
| M8 B0 (kernel-off ref) | 10x2048 | 161.431 | 3.130x | 68.05% | -0.62% |
| M8 K2 best (`q32`,`kv16`) | 10x2048 | 74.738 | 6.760x | 85.21% | 53.70% (vs M8 B0) |
| M8 final validation (`m8_final`) | 10x2048 | 75.324 | 6.708x | 85.09% | -0.78% (run variance vs K2) |

## M7 Interpretation

- M7 was a correctness/enabler milestone, not a 10x2048 throughput milestone.
- It fixed TPU kernel VMEM OOM behavior by wiring explicit ragged kernel block-size parameters.
- Practical speed implication: M7 unlocked safe kernel-on tuning in M8, which then delivered the large throughput jump.

## Net Results

- M5 baseline -> M6 final: `505.266s -> 160.436s` (`3.149x`, `68.25%` faster).
- M6 final -> M8 best: `160.436s -> 74.738s` (`2.147x`, `53.42%` faster).
- M5 baseline -> M8 best: `505.266s -> 74.738s` (`6.760x`, `85.21%` faster).

## Source Pointers

- Milestone rollup and M6/M8 metrics: `lib/levanter/CODEX_REFACTOR_KV.md`
- M7 details (OOM root cause + fix): `lib/levanter/CODEX_INFERENCE_M7.md`
- M8 experiment table + logs: `lib/levanter/CODEX_INFERENCE_M8.md`
