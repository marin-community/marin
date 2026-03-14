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

In progress as of 2026-03-14.

Current state:
- The direct `KubernetesRuntime` path on CoreWeave H100x8 now works end-to-end for environment bring-up:
  - pod image: `pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel`
  - `nvcc` is present and `CUDA_HOME=/usr/local/cuda` is valid
  - the repo-local benchmark script can be staged into the pod reliably
  - `pip install --no-build-isolation /tmp/DeepEP` succeeds
  - `import deep_ep` succeeds and `HybridEPBuffer` is available
- The repo-local harness now has a narrow intranode fallback for `deep_ep.Buffer`: when `world_size <= 8` and the build was compiled without NVSHMEM, it skips the RDMA size hint instead of treating that assertion as fatal.
- That change produces the first steady-state DeepEP timings on H100x8 (`exploratory` confidence):
  - `deep_ep`, `distribution=random`, `topk=2`: `time_s=0.000463`, `tokens_per_s=70,742,659.76`
  - `deep_ep`, `distribution=runs`, `topk=8`: `time_s=0.001047`, `tokens_per_s=31,285,893.37`
- Hybrid-EP is still blocked before any timing loop runs:
  - baseline H100 / CUDA 12.8 runtime JIT fails with `cuda::ptx::cp_async_bulk` overload mismatches in `hybrid_ep_backend.cuh`
  - `DISABLE_AGGRESSIVE_PTX_INSTRS=1` does not change the failure
  - `DISABLE_SM90_FEATURES=1` does not change the failure
- Upstream DeepEP docs still claim Hopper support on CUDA `12.3+`, so the current Hybrid-EP result still looks like a narrower toolchain/kernel compatibility issue rather than a generic unsupported-environment case.

## Decision Log

- 2026-03-13: Treat this as a torch-kernel benchmark experiment first, not a production JAX integration project.
- 2026-03-13: Start with intranode H100x8 DeepEP / Hybrid-EP measurements before any multi-node or JAX-bridge work.
- 2026-03-14: Accept a harness-side intranode RDMA hint skip for exploratory DeepEP bring-up on `world_size <= 8`; keep Hybrid-EP blocked until the runtime JIT compiles on Hopper.

## Negative Results

- 2026-03-13: the default Iris task image (`ghcr.io/marin-community/iris-task:latest`) cannot build DeepEP because it has no CUDA toolkit (`CUDA_HOME` unset, `nvcc` missing).
- 2026-03-14: `deep_ep.Buffer` initially asserted on `get_rdma_buffer_size_hint(...)` with `NVSHMEM is disable during compilation`; this turned out to be a harness-level intranode bring-up bug rather than a hard stop for the NVLink path.
- 2026-03-14: `HybridEPBuffer` runtime JIT on H100 / CUDA 12.8 fails with `cuda::ptx::cp_async_bulk` overload mismatches before any timing loop runs.
- 2026-03-14: `DISABLE_AGGRESSIVE_PTX_INSTRS=1` and `DISABLE_SM90_FEATURES=1` do not change the Hybrid-EP JIT failure on H100 / CUDA 12.8.

## Conclusion

Partial success. The torch-side benchmark path is now validated for DeepEP on CoreWeave H100x8 and yields real dispatch/combine timings, but Hybrid-EP remains blocked by an upstream/runtime JIT compile incompatibility on Hopper that is not resolved by the documented compatibility flags.
