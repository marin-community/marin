## TL;DR

Measure the incremental H100x8 gain from applying the #3821-style expert-padded `w13` FC1 lowering on top of the #4297 Triton `ragged_dot` baseline, using the overlapping exact-cap-style `current` EP=8 kernel path.

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

## Negative Results Index

- None yet.

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

Pending. This issue will track only milestone updates and stable conclusions; detailed command history will live in the research logbook.
