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
- [x] Test whether `hybrid_ep_permute` needs the same rewrite or a different fix.
- [x] Recheck the noisy `runs, topk=2` point with more repeats if we want stronger confidence.
- [x] Extend the longer-window confirmation pass to the `random` slice if we want a replicated full mini-matrix.
- [ ] Understand why `hybrid_ep_permute` trails plain `hybrid_ep` at `runs, topk=8` while slightly leading at `runs, topk=2`.
- [ ] Understand why plain `hybrid_ep` is repeat-sensitive at `random, topk=8` while `deep_ep` and `hybrid_ep_permute` stay stable on the same shape.

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

## Hypothesis 3

The large swings on the `runs` cells are mostly caused by the very short timing window (`warmup=1`, `iters=3`), not by an unstable kernel implementation. If so, a longer-window same-pod run should keep DeepEP near its earlier numbers and collapse the spread on `hybrid_ep` / `hybrid_ep_permute`.

## Changes to make

- Run `bench_deepep_torch.py` with `--kernel all --warmup 5 --iters 20` so `deep_ep`, `hybrid_ep`, and `hybrid_ep_permute` are measured sequentially in the same pod.
- Limit the confirmation pass to the noisy `distribution=runs`, `topk in {2, 8}` slice.

## Results

- The same launcher-side `space_cluster` rewrite also unblocks `hybrid_ep_permute`.
- Short-window `hybrid_ep_permute` points were noisy on `runs`:
  - `runs, topk=2`: first run `29,307,925.82`, immediate rerun `80,190,884.10`
  - `runs, topk=8`: first run `25,826,479.39`, immediate rerun `18,091,692.99`
- Longer-window same-pod confirmation (`warmup=5`, `iters=20`) stabilizes the `runs` slice:

  | topk | deep_ep tokens_per_s | patched hybrid_ep tokens_per_s | patched hybrid_ep_permute tokens_per_s |
  | --- | ---: | ---: | ---: |
  | 2 | 46,681,271.18 | 85,139,043.38 | 87,689,861.36 |
  | 8 | 31,639,255.33 | 42,557,723.33 | 39,033,396.91 |

- Interpretation:
  - The earlier outliers were mostly a short-window measurement problem.
  - DeepEP remains close to its original exploratory `runs` values, while both patched Hybrid-EP paths remain materially faster.
  - The original “can Hybrid-EP run on our H100 machine?” question is now answered yes for both plain and permute paths under the current launcher-applied rewrite.

## Hypothesis 4

The remaining ambiguity is now performance stability, not functionality. If the longer-window method is applied to the `random` slice, we should either confirm the earlier `random` wins or isolate a cell where one patched Hybrid-EP path is genuinely repeat-sensitive.

## Changes to make

- Run the same `--kernel all --warmup 5 --iters 20` confirmation method on `distribution=random`.
- Repeat `random, topk=8` if the first longer-window pass disagrees with the earlier exploratory result.

## Results

- `random, topk=2` longer-window confirmation stays clean:
  - `deep_ep`: `73,257,870.37`
  - patched `hybrid_ep`: `90,777,677.24`
  - patched `hybrid_ep_permute`: `91,250,614.89`
- `random, topk=8` longer-window repeats:

  | run | deep_ep tokens_per_s | patched hybrid_ep tokens_per_s | patched hybrid_ep_permute tokens_per_s |
  | --- | ---: | ---: | ---: |
  | 1 | 34,442,870.69 | 25,409,362.29 | 38,541,941.24 |
  | 2 | 34,379,952.19 | 42,567,832.25 | 39,170,413.95 |
  | 3 | 34,383,801.42 | 41,951,223.51 | 39,252,724.81 |

- Interpretation:
  - DeepEP and `hybrid_ep_permute` are stable on `random, topk=8`.
  - Plain `hybrid_ep` is repeat-sensitive on that cell: one slow outlier, then two faster clustered repeats.
  - So the current performance story is nuanced:
    - `runs` slice: replicated wins for both patched Hybrid-EP paths
    - `random, topk=2`: confirmed win for both patched Hybrid-EP paths
    - `random, topk=8`: `hybrid_ep_permute` looks stably ahead, while plain `hybrid_ep` is ahead by median but not yet stable
