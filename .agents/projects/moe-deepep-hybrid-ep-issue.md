## Description

This follow-up experiment starts from the sealed GPU ragged-all-to-all snapshot in `moe-gpu-ragged-all-all-h100-seal-20260313` and evaluates a different implementation strategy: calling DeepEP's torch-targeted MoE dispatch/combine kernels directly, including the newer Hybrid-EP path.

This is a dual experiment:

1. prove out a repo-local benchmark path that can call DeepEP / Hybrid-EP kernels from Marin research code without modifying production JAX kernels yet
2. measure how those kernels perform on the same CoreWeave H100x8 cluster used in `#3633`

The first goal is execution and benchmarking, not production integration. A true JAX-to-Torch or JAX custom-call bridge is a separate engineering step and should not be conflated with the initial kernel evaluation.

## Hypothesis or Goal

- Hypothesis: DeepEP's torch kernels, especially Hybrid-EP, may outperform the JAX `ragged_a2a` path on GPU because they use a more mature, GPU-specialized dispatch/combine implementation.
- Goal: establish a reproducible H100x8 benchmark for:
  - DeepEP `dispatch + combine`
  - Hybrid-EP `dispatch + combine`
  - Hybrid-EP `dispatch_with_permute + combine_with_unpermute`
- Goal: run a shape regime that is as comparable as practical to `#3633`, while also allowing one stock DeepEP-style smoke case if needed for bring-up.

### Links

* Prior experiment issue: https://github.com/marin-community/marin/issues/3633
* Prior sealed tag: https://github.com/marin-community/marin/tree/moe-gpu-ragged-all-all-h100-seal-20260313
* Research logbook: `.agents/logbooks/moe-deepep-hybrid-ep.md`
* DeepEP mainline: https://github.com/deepseek-ai/DeepEP
* DeepEP Hybrid-EP docs: https://github.com/deepseek-ai/DeepEP/blob/hybrid-ep/docs/README_Hybrid-EP.md

## Results

In progress as of 2026-03-13.

Initial plan:
- Add a repo-local torch benchmark that can import `deep_ep` and drive DeepEP / Hybrid-EP kernels with a controlled routing distribution.
- Start with intranode H100x8 on CoreWeave because it is available now and does not require multi-node RDMA bring-up.
- Match the prior issue's total expert count when practical:
  - global experts: `128`
  - local experts per rank: `16` on `8` GPUs
  - `topk in {2, 8}`
  - routing distributions: `random`, `runs`
- Record both bring-up failures and steady-state timings.

## Decision Log

- 2026-03-13: Treat this as a torch-kernel benchmark experiment first, not a production JAX integration project.
- 2026-03-13: Start with intranode H100x8 DeepEP / Hybrid-EP measurements before any multi-node or JAX-bridge work.

## Negative Results

- None yet.

## Conclusion

Pending.
