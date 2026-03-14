## Description

This follow-up experiment starts from the sealed torch-side DeepEP / Hybrid-EP benchmark in `moe-deepep-hybrid-ep-seal-20260314` and moves to the next engineering question: can we wire a real JAX custom-call path for DeepEP-backed MoE dispatch/combine and compare it directly against the sealed `#3633` GPU baseline?

This is intentionally narrower than “integrate DeepEP into all of Marin.” The scope is:

1. build a repo-local JAX custom-call benchmark path that executes DeepEP-backed GPU work without routing through a Torch benchmarking harness
2. compare that path head-to-head against the fixed-shape `#3633` H100x8 `ragged_a2a` / `current` benchmark regime

The new thread should not redefine the benchmark. It should reuse the sealed `#3633` comparison shape and report deltas against that same path wherever practical.

## Hypothesis or Goal

- Hypothesis: the positive torch-side DeepEP / Hybrid-EP results from `#3641` will not be meaningful for Marin unless a JAX-native custom-call path can be made runnable on the same H100x8 regime used in `#3633`.
- Goal: implement a JAX custom-call benchmark path for a DeepEP-backed dispatch/combine kernel and run it directly against the `#3633` fixed-shape GPU harness.
- Goal: keep the comparison scoped and reproducible enough that the answer is “faster/slower/on par than `ragged_a2a` on the sealed H100x8 shape,” not “maybe promising in a different benchmark.”

### Links

* Prior benchmark issue (`#3633`): https://github.com/marin-community/marin/issues/3633
* Prior DeepEP torch-side issue (`#3641`): https://github.com/marin-community/marin/issues/3641
* Prior torch-side seal tag: https://github.com/marin-community/marin/tree/moe-deepep-hybrid-ep-seal-20260314
* Prior `#3633` seal tag: https://github.com/marin-community/marin/tree/moe-gpu-ragged-all-all-h100-seal-20260313
* Research logbook: `.agents/logbooks/moe-deepep-jax-custom-call.md`

## Results

Completed fixed-shape H100x8 sweep as of 2026-03-14.

Current state:
- The JAX custom-call path is runnable on CoreWeave H100x8 and passed a GPU correctness smoke before the main benchmark.
- The completed implementation milestone is still narrow: `deepep_layout_ragged_a2a` replaces only the dispatch-layout metadata producer with a DeepEP CUDA custom call and then feeds that metadata into the existing JAX `ragged_a2a` path.
- The full head-to-head matrix against the sealed `#3633` shape is now complete for `current`, `ragged_a2a`, and `deepep_layout_ragged_a2a`.

Completed implementation checkpoint:
- Added `lib/levanter/src/levanter/kernels/deepep/layout_ffi.py` and `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_layout_ffi.cu`.
- Added `deepep_layout_ragged_a2a` to `lib/levanter/scripts/bench/bench_moe_hillclimb.py`.
- Added raw CoreWeave launcher `.agents/scripts/deepep_jax_krt_bench.py`.
- The successful matrix ran from repo tag `moe-deepep-jax-layout-ffi-smokefix-20260314` as task `deepep-jax-krt-bench-20260314-114210`.

Completed benchmark target:
- Fixed-shape H100x8 regime from `#3633`:
  - `tokens=32768`
  - `hidden=2048`
  - `mlp_dim=768`
  - `experts=128`
  - `shared_expert_dim=2048`
  - `topk in {2, 8}`
  - `distribution in {random, runs}`
  - `EP in {1, 2, 4, 8}`
  - `bench_pass=forward_backward`

Primary result:
- `current` won at every distributed point (`EP > 1`).
- `deepep_layout_ragged_a2a` did not close the gap to `current`; `current` was about `1.48x` to `1.81x` faster across the completed distributed matrix.
- `deepep_layout_ragged_a2a` stayed effectively tied with vanilla `ragged_a2a` on the distributed cells, so replacing only the JAX-side layout metadata producer does not move the GPU result.

Completed matrix (`tokens/s`):

| distribution | topk | kernel | EP1 | EP2 | EP4 | EP8 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| random | 2 | current | 610333 | 892081 | 1724413 | 2966491 |
| random | 2 | ragged_a2a | 620045 | 553780 | 1058353 | 2001628 |
| random | 2 | deepep_layout_ragged_a2a | 617592 | 538626 | 1057965 | 2002620 |
| random | 8 | current | 151577 | 237428 | 482195 | 956226 |
| random | 8 | ragged_a2a | 150450 | 138256 | 279590 | 546790 |
| random | 8 | deepep_layout_ragged_a2a | 151781 | 132724 | 276921 | 539028 |
| runs | 2 | current | 616690 | 886567 | 1728405 | 2963386 |
| runs | 2 | ragged_a2a | 614976 | 524339 | 970674 | 1668893 |
| runs | 2 | deepep_layout_ragged_a2a | 619300 | 527523 | 974219 | 1665751 |
| runs | 8 | current | 152001 | 235254 | 489235 | 958629 |
| runs | 8 | ragged_a2a | 151848 | 135279 | 274323 | 526984 |
| runs | 8 | deepep_layout_ragged_a2a | 130124 | 138148 | 273787 | 530715 |

Confidence:
- `exploratory` for the general DeepEP-on-JAX question.
- `replicated` for the narrower claim that this layout-only custom-call insertion does not improve the sealed `#3633` H100x8 benchmark.

## Decision Log

- 2026-03-14: start the JAX custom-call thread from the sealed torch-side DeepEP / Hybrid-EP experiment, not from an unsealed branch tip.
- 2026-03-14: reuse the sealed `#3633` fixed-shape GPU benchmark regime as the direct comparison target.
- 2026-03-14: treat Torch-only benchmarks as prior evidence, not as the final comparison harness for this thread.
- 2026-03-14: use DeepEP `get_dispatch_layout` as the first JAX custom-call milestone because it is the narrowest raw CUDA kernel entry point without Torch process-group dependencies.
- 2026-03-14: stop the layout-only custom-call path here as a negative result; further JAX-native DeepEP work should target dispatch/combine transport rather than more layout-only tuning.

## Negative Results

- The first draft of the benchmark integration incorrectly tried to use DeepEP token-per-rank counts as `ragged_a2a` assignment send sizes. That mapping is wrong for repeated top-k assignments and would have made the comparison invalid.
- The first draft of the FFI wrapper also requested `num_tokens_per_rdma_rank`, which is invalid on the single-node H100x8 regime and would have triggered a DeepEP device assertion.
- The raw CoreWeave launch path exposed several non-benchmark blockers before the successful run: mounted `/app` cleanup, missing `git` in the image, NVSHMEM-disabled DeepEP compilation, CUDA FFI handler symbol registration, and JAX 64-bit operand expectations.
- The completed benchmark itself is a negative result for the targeted hypothesis: the JAX layout-only custom call never beat `ragged_a2a` in a meaningful way and never approached `current` on the distributed cells.

## Conclusion

Conclusion: the repo-local JAX custom-call path is runnable and benchmarkable on the sealed `#3633` H100x8 regime, but this first layout-only DeepEP insertion does not improve the GPU result. The direct head-to-head comparison shows that `deepep_layout_ragged_a2a` is effectively just another face of `ragged_a2a` on this workload, while `current` remains decisively faster for every distributed `EP`.
