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
- There are still no steady-state timings because the failures have moved into the kernel paths:
  - `deep_ep.Buffer` still asserts on `get_rdma_buffer_size_hint(...)` when the build is intranode-only with NVSHMEM disabled
  - `HybridEPBuffer` reaches its runtime JIT, but the JIT compile fails on H100 / CUDA 12.8 with `cuda::ptx::cp_async_bulk` overload mismatches in `hybrid_ep_backend.cuh`
- Upstream DeepEP docs still claim Hopper support on CUDA `12.3+`, so the current Hybrid-EP failure looks like a narrower toolchain/kernel compatibility issue rather than a generic unsupported-environment case.

## Decision Log

- 2026-03-13: Treat this as a torch-kernel benchmark experiment first, not a production JAX integration project.
- 2026-03-13: Start with intranode H100x8 DeepEP / Hybrid-EP measurements before any multi-node or JAX-bridge work.

## Negative Results

- 2026-03-13: the default Iris task image (`ghcr.io/marin-community/iris-task:latest`) cannot build DeepEP because it has no CUDA toolkit (`CUDA_HOME` unset, `nvcc` missing).
- 2026-03-14: the direct CUDA-devel pod path fixes the environment blocker, but `deep_ep.Buffer` still asserts when NVSHMEM is disabled for the intranode-only build.
- 2026-03-14: `HybridEPBuffer` runtime JIT on H100 / CUDA 12.8 fails with `cuda::ptx::cp_async_bulk` overload mismatches before any timing loop runs.

## Conclusion

Pending. The experiment has advanced from task-image failure to kernel-specific failure, but there is still no valid timing table yet.
