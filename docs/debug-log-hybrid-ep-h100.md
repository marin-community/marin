# Debugging log for Hybrid-EP H100

Get DeepEP Hybrid-EP to compile and run on the CoreWeave H100x8 path used by the torch-side benchmark harness.

## Initial status

Hybrid-EP installs and imports on `pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel`, but the runtime JIT fails on H100 with:

- `error: no instance of overloaded function "cuda::ptx::cp_async_bulk" matches the argument list`

The failure reproduces for the baseline build, `DISABLE_AGGRESSIVE_PTX_INSTRS=1`, and `DISABLE_SM90_FEATURES=1`.

## Hypothesis 1

The failure is caused by the runtime JIT targeting `sm_90` instead of a Hopper-specific arch variant such as `sm_90a`.

## Changes to make

- Run the existing launcher with `TORCH_CUDA_ARCH_LIST=9.0a` so the installed extension sets `SM_ARCH=9.0a` and the runtime JIT emits `compute_90a` / `sm_90a`.

## Future Work

- [ ] Verify whether the CUDA 12.8 `cp_async_bulk` header overload set differs from newer CCCL docs.
- [ ] Reduce the launcher workaround to the minimal required callsite changes and decide whether to upstream it.
- [ ] Test whether `hybrid_ep_permute` needs the same rewrite or a different fix.
- [ ] Recheck the noisy `runs, topk=2` point with more repeats if we want stronger confidence.

## Results

- `TORCH_CUDA_ARCH_LIST=9.0a` changes the runtime JIT compile command to `-gencode=arch=compute_90a,code=sm_90a`.
- The compile still fails with the same `cuda::ptx::cp_async_bulk(space_shared, space_global, ..., mbarrier)` overload mismatch.
- Conclusion: arch targeting is not the primary blocker.

## Hypothesis 2

The failing Hybrid-EP source is using the wrong PTX API overload for the six-argument global-to-shared bulk copy. On CUDA 12.8, the compiler candidates only accept `space_cluster` for that mbarrier form, not `space_shared`.

## Changes to make

- Patch the extracted `/tmp/DeepEP/csrc/hybrid_ep/backend/hybrid_ep_backend.cuh` in the launcher before install.
- Replace `cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,` with `cuda::ptx::cp_async_bulk(cuda::ptx::space_cluster,` for the Hybrid-EP backend callsites, then rerun the H100 smoke benchmark.

## Results

- A launcher-applied source rewrite that changes `cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,` to `cuda::ptx::cp_async_bulk(cuda::ptx::space_cluster,` at 11 callsites in the extracted `hybrid_ep_backend.cuh` gets Hybrid-EP compiling on H100 / CUDA 12.8.
- First successful H100 point:
  - `random, topk=2`: `time_s=0.000401`, `tokens_per_s=81,669,478.85`
- Additional successful points with the same workaround:
  - `random, topk=8`: `time_s=0.000856`, `tokens_per_s=38,288,395.31`
  - `runs, topk=8`: `time_s=0.000874`, `tokens_per_s=37,512,463.91`
  - `runs, topk=2`: first run `time_s=0.002073`, rerun `time_s=0.000436`, `tokens_per_s=75,096,827.43`
- Conclusion:
  - The H100 blocker is broken. The remaining question is not “can Hybrid-EP run?” but “how much of the apparent speedup is stable once the outlier is resolved and the workaround is reduced to a principled patch?”
