## TL;DR

Replicated result on the overlapping exact-cap-style `current` EP=8 H100x8 path: the #3821-style expert-padded `w13` lowering is still a large `xla` win, but it is a small consistent regression on top of the #4297 Triton path. Across seeds `0,1,2`, `triton + w13 padded` was `-3.28%` tokens/s slower than plain `triton`, so this thread found no additional lift on top of #4297 for this surface.

## Scope

- Base branch: `research/pr-4297-followup`
- Fixed comparison: `4297 triton current` vs `4297 triton current + 3821-style w13 expert padded`
- Hardware: CoreWeave `iris-ci` / `h100-8x`
- Public logbook path: `https://github.com/marin-community/marin/tree/research/pr-4297-plus-w13-lift/.agents/logbooks/pr-4297-plus-w13-lift.md`

## Description

This experiment is a direct follow-up to two earlier threads:

- #4297 added the Triton `ragged_dot` kernel for GPU MoE grouped matmul and produced an approximately `+8.8%` end-to-end gain on the fixed Grug MoE H100x8 validation harness versus the XLA baseline.
- #3821 later found a much larger win on a JAX exact-cap H100x8 path by applying an expert-padded `w13` FC1 lowering that reduced the local expert compute cost.

The open question is whether a similar `w13` expert-padded lowering still produces a meaningful additional gain once the #4297 Triton kernel is already present.

Important scope note:
- the PR 4297 branch does not carry the full DeepEP module tree used by the original #3821 transport benchmark stack
- this thread therefore benchmarks the overlapping exact-cap-style `current` EP=8 kernel path instead of replaying the full historical DeepEP ladder
- this keeps the measurement on the shared routed FC1 grouped-matmul surface that both optimizations touch

This thread keeps the measurement apples-to-apples:

1. same `RAGGED_DOT_IMPL=triton` baseline
2. same exact-cap-style `current` EP=8 kernel path across all variants
3. only one tested axis inside the `triton` comparison: enable or disable the `#3821`-style `w13` expert-padded path

## Hypothesis or Goal

Primary goal:
- quantify the incremental throughput lift from `4297 triton current` to `4297 triton current + 3821-style w13 expert padded`

Initial hypotheses:
- H1: the `w13` expert-padded path still helps on top of Triton because it changes how the routed FC1 grouped compute is packed and lowered, not just the backend used inside `ragged_dot`
- H2: the incremental gain will be smaller than the original `#3821` exact-cap uplift because the Triton `ragged_dot` path already removes part of the old local compute bottleneck

Stop criteria:
- collect at least one clean paired measurement and extend to replication only if the first delta is material and stable enough to justify more pod time

## Decision Log

- 2026-04-04, agent: create a new research thread rather than reuse #3821 or the earlier PR 4297 validation comment, because the scope is a new interaction question across two prior optimizations.
- 2026-04-04, agent: benchmark the overlapping exact-cap-style `current` EP=8 kernel path instead of the historical DeepEP ladder, because the PR 4297 branch does not contain the full `levanter.kernels.deepep` stack.
- 2026-04-04, agent: stop after a 3-seed H100x8 replication because the key `triton` vs `triton + w13 padded` delta was directionally stable in all seeds and pointed negative.

## Negative Results Index

- 2026-04-04: the first direct H100 Iris submission failed before measurement because the default task environment did not include `levanter` dependencies (`numpy` missing). The successful path explicitly used `uv run --python 3.11 --package levanter --extra gpu ...`.
- 2026-04-04: on the shared exact-cap `current` EP=8 surface, `triton + w13 padded` was consistently slower than plain `triton` across seeds `0,1,2`.

## Current Baseline

- Prior PR 4297 follow-up training result on H100x8:
  - `xla`: `463640.70` tokens/s
  - `triton`: `504399.46` tokens/s
  - delta: `+8.8%`

- Fixed benchmark cell for this thread:
  - `tokens=262144`
  - `hidden=2048`
  - `mlp_dim=768`
  - `experts=128`
  - `topk=2`
  - `shared_expert_dim=0`
  - `distribution=random`
  - `ep_size=8`

## Results

Confidence: `replicated` for the exact-cap microbenchmark surface tested here.

- Fixed benchmark cell:
  - `tokens=262144`
  - `hidden=2048`
  - `mlp_dim=768`
  - `experts=128`
  - `topk=2`
  - `shared_expert_dim=0`
  - `distribution=random`
  - `ep_size=8`
  - `warmup=5`
  - `iters=20`
  - seeds: `0,1,2`
- Aggregate means on CoreWeave `iris-ci` / `h100-8x`:
  - `xla`: `9982858.40` tokens/s, `0.026271 s`
  - `xla + w13 padded`: `17092122.30` tokens/s, `0.015337 s`, delta `+71.21%`
  - `triton`: `26065183.53` tokens/s, `0.010057 s`, delta `+161.10%` vs `xla`
  - `triton + w13 padded`: `25209636.97` tokens/s, `0.010399 s`, delta `-3.28%` vs `triton`
- Key paired deltas for the user’s question:
  - seed `0`: `-3.13%`
  - seed `1`: `-3.35%`
  - seed `2`: `-3.37%`

## Conclusion

For the overlapping exact-cap-style `current` EP=8 path, following the #3821-style `w13` expert-padded approach on top of #4297 does not provide additional performance lift. The incremental effect on top of `triton` is a small stable regression of about `3.3%` tokens/s in this benchmark, even though the same padded lowering remains a large `xla` win.
