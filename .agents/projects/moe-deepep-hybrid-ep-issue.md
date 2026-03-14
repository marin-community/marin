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
- That change now produces a complete fixed-shape DeepEP mini-matrix on H100x8 (`exploratory` confidence):

  | distribution | topk | time_s   | tokens_per_s |
  | --- | ---: | ---: | ---: |
  | random | 2 | `0.000463` | `70,742,659.76` |
  | runs   | 2 | `0.000728` | `45,030,617.74` |
  | random | 8 | `0.001004` | `32,637,536.88` |
  | runs   | 8 | `0.001047` | `31,285,893.37` |

- In this small table, `topk=8` is the dominant slowdown and `random` is modestly faster than `runs` once `topk` is fixed.
- Unmodified Hybrid-EP is blocked on H100 / CUDA 12.8:
  - baseline runtime JIT fails with `cuda::ptx::cp_async_bulk` overload mismatches in `hybrid_ep_backend.cuh`
  - `DISABLE_AGGRESSIVE_PTX_INSTRS=1` does not change the failure
  - `DISABLE_SM90_FEATURES=1` does not change the failure
  - `TORCH_CUDA_ARCH_LIST=9.0a` does not change the failure
- A local launcher-applied workaround now exists:
  - before install, rewrite the 11 failing `cp_async_bulk(cuda::ptx::space_shared, ... , mbarrier)` callsites in the extracted `csrc/hybrid_ep/backend/hybrid_ep_backend.cuh` to use `cuda::ptx::space_cluster`
  - with that workaround, Hybrid-EP compiles and runs on H100x8
- Exploratory fixed-shape comparison table (`deep_ep` vs patched `hybrid_ep`):

  | distribution | topk | deep_ep tokens_per_s | patched hybrid_ep tokens_per_s | hybrid / deep |
  | --- | ---: | ---: | ---: | ---: |
  | random | 2 | `70,742,659.76` | `81,669,478.85` | `1.15x` |
  | runs   | 2 | `45,030,617.74` | `75,096,827.43` | `1.67x` |
  | random | 8 | `32,637,536.88` | `38,288,395.31` | `1.17x` |
  | runs   | 8 | `31,285,893.37` | `37,512,463.91` | `1.20x` |

- The same workaround also unblocks `patched hybrid_ep_permute`, which exercises `dispatch_with_permute(...)` / `combine_with_unpermute(...)` rather than plain dispatch/combine.
- Short-window (`warmup=1`, `iters=3`) `hybrid_ep_permute` results on H100x8 (`exploratory`):

  | distribution | topk | patched hybrid_ep_permute tokens_per_s |
  | --- | ---: | ---: |
  | random | 2 | `80,927,091.96` |
  | runs   | 2 | `29,307,925.82` then rerun `80,190,884.10` |
  | random | 8 | `37,527,342.70` |
  | runs   | 8 | `25,826,479.39` then rerun `18,091,692.99` |

- Because the short-window `runs` cells were noisy for both Hybrid-EP paths, a longer-window confirmation run now exists on the same fixed-shape `runs` slice (`warmup=5`, `iters=20`, `kernel=all` in the same pod), moving those points to `replicated` confidence:

  | topk | deep_ep tokens_per_s | patched hybrid_ep tokens_per_s | patched hybrid_ep_permute tokens_per_s | hybrid / deep | permute / deep |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 2 | `46,681,271.18` | `85,139,043.38` | `87,689,861.36` | `1.82x` | `1.88x` |
  | 8 | `31,639,255.33` | `42,557,723.33` | `39,033,396.91` | `1.35x` | `1.23x` |

- Interpretation of the confirmed `runs` slice:
  - DeepEP stays close to the original exploratory table once measured over a longer window.
  - Both patched Hybrid-EP paths remain ahead of DeepEP.
  - `hybrid_ep_permute` is competitive with plain `hybrid_ep`: slightly faster at `topk=2`, slightly slower at `topk=8`.

## Decision Log

- 2026-03-13: Treat this as a torch-kernel benchmark experiment first, not a production JAX integration project.
- 2026-03-13: Start with intranode H100x8 DeepEP / Hybrid-EP measurements before any multi-node or JAX-bridge work.
- 2026-03-14: Accept a harness-side intranode RDMA hint skip for exploratory DeepEP bring-up on `world_size <= 8`; keep Hybrid-EP blocked until the runtime JIT compiles on Hopper.
- 2026-03-14: Use a launcher-applied `space_cluster` rewrite as the current experimental workaround for Hybrid-EP on H100 / CUDA 12.8.
- 2026-03-14: Treat short-window `runs` cells as noisy and use longer `warmup=5`, `iters=20`, same-pod `kernel=all` runs before drawing conclusions on the patched Hybrid-EP paths.

## Negative Results

- 2026-03-13: the default Iris task image (`ghcr.io/marin-community/iris-task:latest`) cannot build DeepEP because it has no CUDA toolkit (`CUDA_HOME` unset, `nvcc` missing).
- 2026-03-14: `deep_ep.Buffer` initially asserted on `get_rdma_buffer_size_hint(...)` with `NVSHMEM is disable during compilation`; this turned out to be a harness-level intranode bring-up bug rather than a hard stop for the NVLink path.
- 2026-03-14: `HybridEPBuffer` runtime JIT on H100 / CUDA 12.8 fails with `cuda::ptx::cp_async_bulk` overload mismatches before any timing loop runs.
- 2026-03-14: `DISABLE_AGGRESSIVE_PTX_INSTRS=1` and `DISABLE_SM90_FEATURES=1` do not change the Hybrid-EP JIT failure on H100 / CUDA 12.8.
- 2026-03-14: `TORCH_CUDA_ARCH_LIST=9.0a` changes the JIT target to `sm_90a`, but does not change the Hybrid-EP overload failure.
- 2026-03-14: Short-window (`warmup=1`, `iters=3`) `runs` measurements for the patched Hybrid-EP paths are noisy enough to produce misleading outliers; use the longer-window confirmation table for conclusions on that slice.

## Conclusion

Partial success with a local workaround. Unmodified Hybrid-EP is still broken on H100 / CUDA 12.8, but a minimal launcher-applied `space_cluster` rewrite gets both `hybrid_ep` and `hybrid_ep_permute` compiling and running on the CoreWeave H100 machine. On the replicated fixed-shape `runs` slice, patched `hybrid_ep` beats DeepEP by `1.35x` to `1.82x`, and patched `hybrid_ep_permute` beats DeepEP by `1.23x` to `1.88x`. Confidence outside that longer-window `runs` slice remains exploratory, and all positive results still depend on the local launcher-applied rewrite rather than an upstream fix.
