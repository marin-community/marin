## Description

This experiment starts from three sealed GPU MoE threads that currently point in different directions:

- `#3641`: torch-side DeepEP / Hybrid-EP looked promising on H100x8
- `#3665`: the first JAX custom-call DeepEP path was runnable but did not improve the sealed `#3633` benchmark
- `#3666`: Megatron-LM's full `MoELayer` benchmark changed the ranking again and favored `deepep` or `hybridep` depending on scale

The new question is not "is DeepEP good?" in the abstract. The question is:

> what are the concrete root causes of the performance gap between the positive torch/Megatron results and the negative JAX/Marin result?

This thread is specifically about separating:

1. benchmark methodology differences
2. kernel coverage differences
3. transport / communication backend differences
4. JAX-specific overheads or missing fusions

The goal is to produce a falsifiable explanation rather than another isolated timing table.

## Hypothesis or Goal

- Hypothesis: `#3665` stayed tied with `ragged_a2a` because it only replaced JAX-side dispatch-layout metadata production, while the positive results in `#3641` / `#3666` came from exercising materially different transport and full-layer kernels.
- Hypothesis: part of the apparent gap is methodology, not just implementation. `#3665` reused the fixed-shape `#3633` dispatch benchmark, while `#3666` measured a full Megatron `MoELayer` with different timing hygiene and compute/communication balance.
- Goal: build a rigorous root-cause matrix that explains which differences matter, how much they matter, and which ones are sufficient to reproduce the torch/Megatron gains.
- Goal: include a direct head-to-head between JAX/Marin and Megatron-LM on the same CoreWeave H100x8 cluster.

### Links

* Prior fixed-shape GPU issue (`#3633`): https://github.com/marin-community/marin/issues/3633
* Prior torch-side DeepEP / Hybrid-EP issue (`#3641`): https://github.com/marin-community/marin/issues/3641
* Prior JAX custom-call issue (`#3665`): https://github.com/marin-community/marin/issues/3665
* Prior Megatron scaling issue (`#3666`): https://github.com/marin-community/marin/issues/3666
* Prior torch-side seal tag: https://github.com/marin-community/marin/tree/moe-deepep-hybrid-ep-seal-20260314
* Prior JAX custom-call seal tag: https://github.com/marin-community/marin/tree/moe-deepep-jax-layout-ffi-h100-matrix-20260314
* Prior Megatron seal tag: https://github.com/marin-community/marin/tree/moe-megatron-qwen-scale-h100-matrix-20260314
* Research logbook: `.agents/logbooks/moe-jax-megatron-root-cause.md`

## Results

In progress as of 2026-03-14.

Current state:
- New research branch/worktree created from the sealed Megatron snapshot so the starting point includes both the prior JAX custom-call code and the Megatron harness.
- Root-cause categories being audited first:
  - `benchmark scope`: dispatch-only vs full `MoELayer`
  - `kernel scope`: layout-only custom call vs real dispatch/combine transport
  - `backend scope`: JAX `ragged_all_to_all` / current ring EP vs DeepEP / Hybrid-EP
  - `measurement scope`: warmup window, dummy GEMM, router setup, fixed inputs, and fp32 router dtype
- The first experiment matrix is intentionally matched rather than broad:
  1. re-establish the sealed JAX/Marin fixed-shape baseline on H100x8
  2. run a same-shape torch-side DeepEP dispatch/combine comparison, with optional JAX->Torch bridge cost made explicit
  3. run a direct JAX/Marin vs Megatron-LM head-to-head on the same H100x8 cluster with the closest practical shared synthetic shape
  4. attribute the observed gap to the smallest validated set of causes

## Decision Log

- 2026-03-14: start a new experiment rather than reopening `#3665` or `#3666`, because the new question is explanatory and cross-methodological rather than just another benchmark variant.
- 2026-03-14: branch from the sealed Megatron snapshot so the new thread starts from the previously working JAX custom-call and Megatron harness state.
- 2026-03-14: make the first milestone a root-cause matrix, not an optimization sprint.

## Negative Results

- `#3665` is already a sealed negative result for the narrow claim that replacing only the JAX dispatch-layout metadata producer improves the `#3633` fixed-shape H100x8 benchmark.

## Conclusion

Not complete yet. The immediate milestone is to turn the prior sealed results into a matched root-cause matrix and a direct JAX/Marin vs Megatron-LM comparison on CoreWeave H100x8.
