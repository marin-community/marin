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

In progress as of 2026-03-14.

Current state:
- The prior torch-side experiment is sealed; this new thread starts from that snapshot.
- CoreWeave H100x8 access is healthy and the submission credentials are present in a fresh shell.
- A repo-local JAX custom-call path is now wired into `bench_moe_hillclimb.py` as `deepep_layout_ragged_a2a`.
- The current implementation milestone wraps DeepEP `deep_ep::layout::get_dispatch_layout` via JAX FFI and feeds its per-expert metadata into the existing `ragged_a2a` dispatch contract.
- A raw CoreWeave launcher exists at `.agents/scripts/deepep_jax_krt_bench.py` to run:
  - a GPU correctness smoke for the new custom call
  - the fixed-shape `#3633` matrix for `current`, `ragged_a2a`, and `deepep_layout_ragged_a2a`

Current implementation checkpoint:
- Added `lib/levanter/src/levanter/kernels/deepep/layout_ffi.py` and `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_layout_ffi.cu`.
- Added a new benchmark kernel path in `lib/levanter/scripts/bench/bench_moe_hillclimb.py`.
- Local validation passed:
  - `python -m py_compile` on the edited Python files
  - `uv run ruff check` on the edited Python and launcher files
- Two integration bugs were found and fixed before the first cluster run:
  - `num_tokens_per_rdma_rank` must be passed as `nullptr` on the single-node H100x8 regime; requesting it would trip DeepEP's multi-node assertion
  - DeepEP token-per-rank counts do not match this benchmark's repeated-assignment send sizes, so shard counts must still be derived from per-expert assignment counts

Planned benchmark target:
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

Planned implementation target:
- repo-local JAX custom-call path, not a Torch DLPack bridge benchmark
- direct comparison against the sealed `bench_moe_hillclimb.py` results from `#3633`

## Decision Log

- 2026-03-14: start the JAX custom-call thread from the sealed torch-side DeepEP / Hybrid-EP experiment, not from an unsealed branch tip.
- 2026-03-14: reuse the sealed `#3633` fixed-shape GPU benchmark regime as the direct comparison target.
- 2026-03-14: treat Torch-only benchmarks as prior evidence, not as the final comparison harness for this thread.
- 2026-03-14: use DeepEP `get_dispatch_layout` as the first JAX custom-call milestone because it is the narrowest raw CUDA kernel entry point without Torch process-group dependencies.

## Negative Results

- The first draft of the benchmark integration incorrectly tried to use DeepEP token-per-rank counts as `ragged_a2a` assignment send sizes. That mapping is wrong for repeated top-k assignments and would have made the comparison invalid.
- The first draft of the FFI wrapper also requested `num_tokens_per_rdma_rank`, which is invalid on the single-node H100x8 regime and would have triggered a DeepEP device assertion.

## Conclusion

Not complete yet. The goal of this thread is to determine whether a real JAX custom-call path can be benchmarked directly against the sealed `#3633` H100x8 GPU baseline.
