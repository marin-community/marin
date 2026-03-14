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

Current state as of 2026-03-14:
- The same H100x8 shape now has three matched views:
  1. JAX/Marin fixed-shape forward-only control on the sealed `#3633` harness
  2. Megatron-LM full `MoELayer` timing on a same-shape `marin_3633_topk_{2,8}` case
  3. direct DeepEP dispatch/combine isolation with explicit JAX -> Torch and Torch -> JAX bridge timing

Matched JAX/Marin result:
- Shape: `tokens=32768 hidden=2048 mlp_dim=768 experts=128 shared_expert_dim=0`
- Timing hygiene: `warmup=5 iters=20`
- `current` still wins every distributed point (`EP > 1`)
- `current / ragged_a2a` ranges from `1.73x` to `2.48x`
- `deepep_layout_ragged_a2a` stays effectively tied with `ragged_a2a`, usually within `~1%`

Matched Megatron result on the same shape:
- `topk=2`
  - `alltoall`: `forward 14.62 ms`, `backward 19.57 ms`
  - `deepep`: `forward 7.74 ms`, `backward 6.00 ms`
  - `hybridep`: `forward 6.31 ms`, `backward 8.35 ms`
- `topk=8`
  - `alltoall`: `forward 11.26 ms`, `backward 20.86 ms`
  - `deepep`: `forward 4.84 ms`, `backward 6.15 ms`
  - `hybridep`: `forward 6.19 ms`, `backward 10.43 ms`

Direct DeepEP dispatch/combine isolation:
- Raw torch-side DeepEP transport on the sealed shape is strong:
  - `random, topk=2`: `67.44M tokens/s`
  - `random, topk=8`: `30.85M tokens/s`
  - `runs, topk=2`: `35.90M tokens/s`
  - `runs, topk=8`: `25.25M tokens/s`
- Once tensors are already in Torch, JAX-originating tensors see similar steady-state transport cost:
  - JAX/Torch `dispatch_combine_full_s` ratio ranges from `0.92x` to `1.36x`
- But the bridge itself is expensive:
  - `bridge_to_torch_s`: about `85 ms` to `105 ms`
  - `bridge_to_jax_s`: about `2 ms` to `12 ms`

Root-cause summary:
1. `#3665` never exercised the winning DeepEP / Hybrid-EP transport kernels. It only replaced dispatch-layout metadata and then still used JAX `ragged_all_to_all`.
2. That kernel-coverage difference is sufficient to explain the main discrepancy: on the same shape, Megatron using real DeepEP / Hybrid-EP is positive while JAX `deepep_layout_ragged_a2a` remains tied with plain ragged.
3. A naive JAX -> Torch bridge each training step would erase the gain. The direct bridge cost is tens of milliseconds, far larger than the sub-millisecond to low-millisecond transport kernel time.
4. JAX likely also has a separate absolute-performance issue in grouped expert GEMMs on this shape; the JAX run emitted repeated XLA slow-kernel warnings on `topk=8`. That helps explain absolute JAX throughput, but it is not needed to explain why `deepep_layout_ragged_a2a` failed to beat `ragged_a2a`.

## Decision Log

- 2026-03-14: start a new experiment rather than reopening `#3665` or `#3666`, because the new question is explanatory and cross-methodological rather than just another benchmark variant.
- 2026-03-14: branch from the sealed Megatron snapshot so the new thread starts from the previously working JAX custom-call and Megatron harness state.
- 2026-03-14: make the first milestone a root-cause matrix, not an optimization sprint.

## Negative Results

- `#3665` is already a sealed negative result for the narrow claim that replacing only the JAX dispatch-layout metadata producer improves the `#3633` fixed-shape H100x8 benchmark.
- The first direct dispatch-isolation attempt failed because the harness misused the DeepEP cached-handle API; fixing that did not change the final conclusion.
- The second dispatch-isolation attempt failed because the PyTorch benchmark image did not have JAX installed; adding `jax[cuda12]==0.8.0` resolved that setup issue.

## Conclusion

The root cause is now specific:

- The positive H100x8 gains from `#3641` / `#3666` do exist on the sealed `#3633` shape, but only when the benchmark actually uses DeepEP / Hybrid-EP dispatch-combine transport.
- The negative JAX result in `#3665` was not a faithful test of that path. It swapped in DeepEP layout metadata, but still paid JAX `ragged_all_to_all` transport costs, so it stayed tied with plain `ragged_a2a`.
- Direct Torch interop is not an acceptable substitute for a real JAX integration because the JAX -> Torch bridge cost is too large to pay per step.

The next meaningful follow-up is therefore not “retune `deepep_layout_ragged_a2a`”. The next meaningful follow-up is to wire a real JAX custom call for dispatch/combine transport, or to explicitly decide not to pursue that integration.
