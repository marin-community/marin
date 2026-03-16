# JAX DeepEP Autodiff And Scaling: Research Logbook

## Scope
- Goal: add JAX autodiff support to the DeepEP transport path used by the exact-cap reintegrated MoE benchmark, then rerun the decisive fixed-shape `forward_backward` comparison and the requested token sweep.
- Primary metric(s): `step_s` and `tokens/s` on H100x8 for the reintegrated JAX full-layer MoE benchmark, with matched Torch full-layer DeepEP numbers where applicable.
- Constraints:
  - zero Torch in the JAX step path
  - preserve the sealed `#3711` forward exact-cap behavior as the baseline
  - use the isolated CoreWeave lane for H100x8 runs
- GitHub issue: https://github.com/marin-community/marin/issues/3717

## Baseline
- Date: 2026-03-16
- Prior sealed issue: `#3711`
- Code refs:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py`
  - `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu`
- Baseline numbers:
  - fixed-shape JAX exact-cap forward (`tokens=32768`, `EP=8`, H100x8):
    - `random, topk=2`: `current=0.003876`, `deepep_transport_capped_prewarmed=0.003325`, `1.17x`
    - `random, topk=8`: `current=0.011067`, `deepep_transport_capped_prewarmed=0.009854`, `1.12x`
    - `runs, topk=2`: `current=0.003867`, `deepep_transport_capped_prewarmed=0.003785`, `1.02x`
    - `runs, topk=8`: `current=0.011085`, `deepep_transport_capped_prewarmed=0.010062`, `1.10x`
  - larger-token JAX exact-cap forward wins from `#3711`:
    - `tokens=262144`, `topk=2`: `1.23x` (`random`), `1.18x` (`runs`)
    - `tokens=524288`, `topk=2`: `1.24x` (`random`), `1.21x` (`runs`)
    - `tokens=1048576`, `topk=2`: `1.40x` (`random`), `1.26x` (`runs`)
    - `tokens=262144`, `topk=8`: `1.20x` (`random`), `1.24x` (`runs`)
    - `tokens=524288`, `topk=8`: `1.23x` (`random`), `1.21x` (`runs`)
  - fixed-shape Torch full-layer DeepEP baseline (`tokens=32768`, H100x8):
    - `topk=2`: `alltoall=34.160 ms`, `deepep=11.586 ms`, `2.95x`
    - `topk=8`: `alltoall=34.838 ms`, `deepep=11.025 ms`, `3.16x`
  - current blocker for JAX `forward_backward`:
    - `ValueError: The FFI call to levanter_deepep_dispatch_intranode cannot be differentiated.`

## Experiment Log
### 2026-03-16 18:00 - Kickoff
- Hypothesis:
  - the remaining missing JAX `forward_backward` benchmark column is blocked by missing AD definitions around the DeepEP dispatch/combine custom calls, not by another transport or benchmark integration bug.
- Command:
  - research thread kickoff; no benchmark command yet
- Config:
  - new branch/worktree from sealed `#3711` snapshot `8a1b629e0dc8ab904fa0eca47b4369d59d34c77d`
  - branch: `research/moe-jax-deepep-autodiff-scaling`
  - worktree: `/Users/romain/marin-wt/moe-jax-deepep-autodiff-scaling`
- Result:
  - initialized the new research logbook for the AD/scaling follow-up thread
- Interpretation:
  - the new thread starts from a narrow, concrete blocker rather than a broad benchmark mystery
- Next action:
  - create the experiment issue and then inspect the current transport FFI path to decide where to add JAX AD support

### 2026-03-16 18:35 - First custom-VJP pass
- Hypothesis:
  - DeepEP transport backward can be expressed from the same upstream identities used on the Torch side: `dispatch_backward == combine`, `combine_backward == cached dispatch(handle=...)`.
- Command:
  - source inspection only, then local code changes on `research/moe-jax-deepep-autodiff-scaling`
- Config:
  - upstream DeepEP ref inspected: `7febc6e25660af0f54d95dd781ecdcd62265ecca`
  - key local files changed:
    - `lib/levanter/src/levanter/kernels/deepep/transport_ffi.py`
    - `lib/levanter/src/levanter/kernels/deepep/csrc/deepep_transport_ffi.cu`
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
    - `lib/levanter/scripts/bench/bench_deepep_dispatch_jax.py`
  - commit: `e9b1bbfd07de5973b19a3d14649f7293c089d1bb`
- Result:
  - added a cached-dispatch FFI target and first `custom_vjp` plumbing for `deepep_dispatch_intranode(...)` and `deepep_combine_intranode(...)`
  - first H100x8 fixed-shape `forward_backward` probe moved past the old nondifferentiable-FFI error
  - the new failure was local to the wrapper:

```text
AttributeError: 'NoneType' object has no attribute 'shape'
```

- Interpretation:
  - the AD implementation was being reached, but the forward rule used the wrong `custom_vjp(..., nondiff_argnums=...)` calling convention, so JAX was handing the wrapper the wrong arguments
- Next action:
  - fix the forward-rule signature, rerun the same decisive `forward_backward` cell, and see what runtime frontier remains once the wrapper arguments are correct

### 2026-03-16 19:10 - AD wrapper fixed; runtime frontier moved to live transport
- Hypothesis:
  - once the `custom_vjp` forward-rule signature matches JAX’s actual calling convention, the first fixed-shape `forward_backward` cell will either run or fail in the transport runtime rather than at the differentiation boundary.
- Command:
  - local fix + reruns on the isolated CoreWeave lane:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --task-id deepep-jax-krt-bench-20260316-ad-fb-probe-v2 \
  --kernels current,deepep_transport_capped_prewarmed \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 1 \
  --per-bench-timeout-seconds 180 \
  --per-bench-kill-after-seconds 10 \
  --build-with-torch-extension \
  --load-as-python-module
```

  - then, after clearing stale peer-access status, reran with task id `deepep-jax-krt-bench-20260316-ad-fb-probe-v3`
- Config:
  - fixed-shape probe:
    - `tokens=32768`
    - `hidden=2048`
    - `mlp_dim=768`
    - `experts=128`
    - `shared_expert_dim=2048`
    - `EP=8`
    - `distribution=random`
    - `topk=2`
    - `bench_pass=forward_backward`
  - commits:
    - signature fix: `899ef966e200b71cb0a3c3a4f76e38bbfedb6255`
    - peer-access stale-error cleanup: `26bd8d62e0341b94667149afe4cbc10b586ffa14`
- Result:
  - the old JAX autodiff blocker is gone; the run now reaches live DeepEP transport under `forward_backward`
  - second probe (`v2`) failed before module lookup with:

```text
JaxRuntimeError: INTERNAL: [0] There was an error before calling cuModuleGetFunction (704): cudaErrorPeerAccessAlreadyEnabled : peer access is already enabled
```

  - clearing the stale CUDA last-error state after `cudaErrorPeerAccessAlreadyEnabled` removed that failure mode
  - third probe (`v3`) progressed further, then failed inside live transport progress with repeated receiver stalls and:

```text
DeepEP timeout for dispatch receivers, rank <varies>, responsible_channel = <varies>, tokens remained: <varies>
JaxRuntimeError: INTERNAL: cudaMemcpyAsync(read num_recv_tokens): unspecified launch failure
```

  - the baseline `current` kernel still completed on the same run:

```text
RESULT kernel=current ep=8 pass=forward_backward time_s=0.011138 tokens_per_s=2941944.89
```

- Interpretation:
  - this thread has now crossed two concrete blockers:
    - `levanter_deepep_dispatch_intranode` is no longer rejected as nondifferentiable
    - stale peer-access status is no longer killing the run before module lookup
  - the remaining blocker is deeper and matches the same transport-progress family seen during the earlier pure-JAX bring-up: dispatch receiver timeout / `read num_recv_tokens` launch failure during live transport
- Next action:
  - isolate whether the failing backward leg is `dispatch_backward == combine`, `combine_backward == cached dispatch`, or both, using the smallest transport-only gradient probes possible before rerunning the full-layer benchmark

### 2026-03-16 19:58 - Combine backward probe succeeded on the full fixed-shape transport input
- Hypothesis:
  - if the cached-dispatch backward path in `deepep_combine_intranode(...)` uses the original forward `channel_prefix_matrix` rather than the recv-side prefix matrix, the isolated combine-backward transport probe should stop reproducing the dispatch-receiver timeout family.
- Command:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_transport_krt.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --worktree /Users/romain/marin-wt/moe-jax-deepep-autodiff-scaling \
  --repo-ref 95e2e589ca16ec2008cbb36efb47bad4269bc9e8 \
  --task-id deepep-jax-transport-grad-combine-20260316-4 \
  --tokens 32768 \
  --hidden 2048 \
  --experts 128 \
  --topk-list 2 \
  --distributions random \
  --execution-model shard_map \
  --seed 0 \
  --warmup 0 \
  --iters 1 \
  --build-with-torch-extension \
  --load-as-python-module \
  --grad-probe combine
```

- Config:
  - `tokens=32768`
  - `hidden=2048`
  - `experts=128`
  - `topk=2`
  - `distribution=random`
  - `execution_model=shard_map`
  - `commit=95e2e589ca16ec2008cbb36efb47bad4269bc9e8`
- Result:
  - the isolated combine-backward gradient probe completed and emitted finite per-rank losses and gradient norms:

```text
GRAD_PROBE mode=combine losses=[-6517.24658203125, 6363.537109375, 6998.06640625, -3070.1884765625, -3576.71435546875, 7493.3134765625, 9713.0625, -7272.54052734375] grad_norms=[4000.1279296875, 3982.682373046875, 3952.744873046875, 3990.64501953125, 3944.96533203125, 3968.51611328125, 3947.30078125, 3984.995849609375]
```

- Interpretation:
  - both isolated backward legs now run on the same full fixed-shape transport input:
    - `dispatch_backward == combine` had already passed in the earlier dispatch grad probe
    - `combine_backward == cached dispatch` now passes too
  - the transport-only AD surface is no longer the active blocker for the first full-layer `forward_backward` cell
- Next action:
  - rerun the decisive fixed-shape `forward_backward` full-layer benchmark cell immediately, using the exact-cap DeepEP path on the same H100x8 regime

### 2026-03-16 20:01 - Exact-cap DeepEP cleared the decisive fixed-shape `forward_backward` full-layer cell
- Hypothesis:
  - once both isolated transport backward legs work, the reintegrated exact-cap DeepEP full-layer path should clear the first authoritative `forward_backward` benchmark cell rather than failing in live transport progress.
- Command:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --repo-ref 95e2e589ca16ec2008cbb36efb47bad4269bc9e8 \
  --task-id deepep-jax-krt-bench-20260316-ad-fb-probe-v4 \
  --kernels current,deepep_transport_capped_prewarmed \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 1 \
  --per-bench-timeout-seconds 180 \
  --per-bench-kill-after-seconds 10 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke
```

- Config:
  - `tokens=32768`
  - `hidden=2048`
  - `mlp_dim=768`
  - `experts=128`
  - `shared_expert_dim=2048`
  - `EP=8`
  - `distribution=random`
  - `topk=2`
  - `bench_pass=forward_backward`
  - `commit=95e2e589ca16ec2008cbb36efb47bad4269bc9e8`
- Result:
  - baseline `current` completed at:

```text
RESULT kernel=current ep=8 pass=forward_backward time_s=0.011118 tokens_per_s=2947338.80
```

  - exact-cap DeepEP completed at:

```text
DEEPEP_EXACT_CAPS max_recv_tokens=7808 max_local_assignments=8320 recv_factor=4.196721 assign_factor=7.876923
RESULT kernel=deepep_transport_capped_prewarmed ep=8 pass=forward_backward time_s=0.009942 tokens_per_s=3295880.18
```

  - ratio on the decisive cell:
    - `0.011118 / 0.009942 = 1.1189x`
  - after the benchmark result line, the pod still emitted CUDA teardown noise, including `DeepEP timeout check failed ...` and XLA stream/event destruction errors ending in `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure`
  - despite that teardown noise, the benchmark shell returned:

```text
bench_status=0
BENCH_END kernel=deepep_transport_capped_prewarmed distribution=random topk=2
EXIT_CODE=0
```

- Interpretation:
  - the reintegrated exact-cap JAX DeepEP path now clears the first authoritative full-layer fixed-shape `forward_backward` benchmark cell
  - the exact-cap path is positive on that cell, beating `current` by about `1.12x`
  - the remaining caveat has shifted again: the active issue is now post-result teardown noise rather than the old nondifferentiable-FFI or live transport progress failures
- Next action:
  - record the milestone on `#3717`, then start the requested `forward_backward` token sweep at `32768`, `65536`, `131072`, and `262144` global tokens

### 2026-03-16 20:17 - Completed the first `forward_backward` downward sweep (`random`, `topk=2`)
- Hypothesis:
  - once the fixed-shape `forward_backward` cell is live, the exact-cap DeepEP path should stay positive across the requested first scaling spine at `32768`, `65536`, `131072`, and `262144` global tokens.
- Commands:
  - completed fixed-shape base point from the previous milestone:

```text
tokens=32768
```

  - then launched three additional H100x8 runs with the same regime and filtered the launcher output down to benchmark result lines:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --repo-ref 95e2e589ca16ec2008cbb36efb47bad4269bc9e8 \
  --task-id deepep-jax-krt-bench-20260316-ad-sweep-65536-random-topk2 \
  --kernels current,deepep_transport_capped_prewarmed \
  --tokens 65536 \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 1 \
  --per-bench-timeout-seconds 180 \
  --per-bench-kill-after-seconds 10 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke
```

  - repeated the same command shape for:
    - `tokens=131072`
    - `tokens=262144`
- Config:
  - shared across the sweep:
    - `hidden=2048`
    - `mlp_dim=768`
    - `experts=128`
    - `shared_expert_dim=2048`
    - `EP=8`
    - `distribution=random`
    - `topk=2`
    - `bench_pass=forward_backward`
    - `commit=95e2e589ca16ec2008cbb36efb47bad4269bc9e8`
- Result:
  - full sweep:

```text
tokens=32768:
  current: 0.011118 s
  deepep_transport_capped_prewarmed: 0.009942 s
  speedup: 1.12x

tokens=65536:
  current: 0.020879 s
  deepep_transport_capped_prewarmed: 0.017481 s
  speedup: 1.19x

tokens=131072:
  current: 0.040377 s
  deepep_transport_capped_prewarmed: 0.033944 s
  speedup: 1.19x

tokens=262144:
  current: 0.081226 s
  deepep_transport_capped_prewarmed: 0.067991 s
  speedup: 1.19x
```

  - exact caps reported by the DeepEP path on the larger three cells:

```text
tokens=65536:  max_recv_tokens=15616  max_local_assignments=16640  recv_factor=4.196721  assign_factor=7.876923
tokens=131072: max_recv_tokens=30976  max_local_assignments=33024  recv_factor=4.231405  assign_factor=7.937984
tokens=262144: max_recv_tokens=61952  max_local_assignments=65920  recv_factor=4.231405  assign_factor=7.953398
```

  - all three added runs ended cleanly from the launcher’s point of view with `EXIT_CODE=0`
  - as on the decisive `32768` cell, post-result CUDA teardown noise was still present after the successful DeepEP result lines
- Interpretation:
  - the corrected exact-cap JAX DeepEP path stays positive across the first requested `forward_backward` scaling spine
  - the speedup is already positive at the smallest point and then stabilizes around `1.19x` on the larger three points
  - the teardown noise remains a caveat, but it did not prevent successful benchmark completion on any of the four sweep points
- Next action:
  - publish the sweep milestone on `#3717`
  - then decide between:
    - extending the JAX coverage to `runs` / `topk=8`
    - or bringing up the matched Torch full-layer `forward_backward` baseline

### 2026-03-16 20:31 - Completed the fixed-shape `forward_backward` JAX quadrant
- Hypothesis:
  - after the successful `random, topk=2` cell and first scaling spine, the exact-cap JAX DeepEP path should remain live on the other fixed-shape `forward_backward` cells at `tokens=32768`, especially `topk=8` and `runs`.
- Commands:
  - fixed-shape `random, topk=8`:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --repo-ref 95e2e589ca16ec2008cbb36efb47bad4269bc9e8 \
  --task-id deepep-jax-krt-bench-20260316-ad-fixed-random-topk8 \
  --kernels current,deepep_transport_capped_prewarmed \
  --tokens 32768 \
  --topk-list 8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 1 \
  --per-bench-timeout-seconds 180 \
  --per-bench-kill-after-seconds 10 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke
```

  - fixed-shape `runs, topk=2`:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --repo-ref 95e2e589ca16ec2008cbb36efb47bad4269bc9e8 \
  --task-id deepep-jax-krt-bench-20260316-ad-fixed-runs-topk2 \
  --kernels current,deepep_transport_capped_prewarmed \
  --tokens 32768 \
  --topk-list 2 \
  --distributions runs \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 1 \
  --per-bench-timeout-seconds 180 \
  --per-bench-kill-after-seconds 10 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke
```

  - fixed-shape `runs, topk=8`:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --repo-ref 95e2e589ca16ec2008cbb36efb47bad4269bc9e8 \
  --task-id deepep-jax-krt-bench-20260316-ad-fixed-runs-topk8 \
  --kernels current,deepep_transport_capped_prewarmed \
  --tokens 32768 \
  --topk-list 8 \
  --distributions runs \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 1 \
  --per-bench-timeout-seconds 180 \
  --per-bench-kill-after-seconds 10 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke
```

- Config:
  - shared fixed-shape regime:
    - `tokens=32768`
    - `hidden=2048`
    - `mlp_dim=768`
    - `experts=128`
    - `shared_expert_dim=2048`
    - `EP=8`
    - `bench_pass=forward_backward`
    - `commit=95e2e589ca16ec2008cbb36efb47bad4269bc9e8`
- Result:
  - complete fixed-shape quadrant:

```text
random, topk=2:
  current: 0.011118 s
  deepep_transport_capped_prewarmed: 0.009942 s
  speedup: 1.12x

random, topk=8:
  current: 0.033823 s
  deepep_transport_capped_prewarmed: 0.029962 s
  speedup: 1.13x

runs, topk=2:
  current: 0.011068 s
  deepep_transport_capped_prewarmed: 0.010662 s
  speedup: 1.04x

runs, topk=8:
  current: 0.034288 s
  deepep_transport_capped_prewarmed: 0.030217 s
  speedup: 1.13x
```

  - exact-cap metadata on the added three cells:

```text
random, topk=8: max_recv_tokens=22144  max_local_assignments=33152  recv_factor=1.479769  assign_factor=7.907336
runs, topk=2:   max_recv_tokens=8064   max_local_assignments=8448   recv_factor=4.063492  assign_factor=7.757576
runs, topk=8:   max_recv_tokens=22144  max_local_assignments=33152  recv_factor=1.479769  assign_factor=7.907336
```

  - all three added runs exited cleanly with `EXIT_CODE=0`
  - the same post-result CUDA teardown noise remained present on the DeepEP path
- Interpretation:
  - the exact-cap JAX DeepEP path is now positive on all four fixed-shape `forward_backward` cells
  - the gain is materially smaller on `runs, topk=2` (`1.04x`) than on the other three cells (about `1.12x` to `1.13x`)
  - the backward-enabled path is now benchmark-complete for the fixed-shape JAX quadrant even though teardown noise remains
- Next action:
  - publish the fixed-shape quadrant milestone on `#3717`
  - then decide whether the next highest-value extension is:
    - the matched Torch full-layer `forward_backward` baseline
    - or a broader JAX scaling sweep beyond the completed `random, topk=2` spine

### 2026-03-16 20:56 - Captured the matched Torch full-layer scaling baselines on the fixed shape
- Context:
  - The JAX side now has:
    - a complete fixed-shape `forward_backward` quadrant
    - a completed `random, topk=2` scaling spine
  - The remaining major comparison gap was the matched Torch full-layer `forward_backward` baseline on the same fixed shape and token scales.
  - I reused the existing Megatron harness rather than adding a new Torch benchmark path:
    - `.agents/scripts/megatron_qwen_krt_bench.py`
    - `.agents/scripts/megatron_qwen_moe_perf.py`
  - I added fixed-shape Megatron cases for `micro_batch_size in {2, 4, 8}` so the harness now covers:
    - global tokens `32768`, `65536`, `131072`, `262144`
    - both `topk=2` and `topk=8`
- Commands:
  - topk=2 sweep:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/megatron_qwen_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --worktree /Users/romain/marin-wt/moe-jax-deepep-autodiff-scaling \
  --task-id megatron-qwen-krt-bench-20260316-ad-topk2-sweep \
  --cases marin_3633_topk_2,marin_3633_topk_2_mb2,marin_3633_topk_2_mb4,marin_3633_topk_2_mb8 \
  --dispatchers alltoall,deepep \
  --warmup-iters 5 \
  --measure-iters 20
```

  - topk=8 sweep:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/megatron_qwen_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --worktree /Users/romain/marin-wt/moe-jax-deepep-autodiff-scaling \
  --task-id megatron-qwen-krt-bench-20260316-ad-topk8-sweep \
  --cases marin_3633_topk_8,marin_3633_topk_8_mb2,marin_3633_topk_8_mb4,marin_3633_topk_8_mb8 \
  --dispatchers alltoall,deepep \
  --warmup-iters 5 \
  --measure-iters 20
```

- Result:
  - topk=2 totals (`forward_ms + backward_ms`):

```text
global tokens=32768  (mb1):
  alltoall: 31.633 ms
  deepep:   13.861 ms
  speedup:  2.28x

global tokens=65536  (mb2):
  alltoall: 34.658 ms
  deepep:    8.804 ms
  speedup:  3.94x

global tokens=131072 (mb4):
  alltoall: 32.281 ms
  deepep:   12.275 ms
  speedup:  2.63x

global tokens=262144 (mb8):
  alltoall: 33.144 ms
  deepep:   14.640 ms
  speedup:  2.26x
```

  - topk=8 totals (`forward_ms + backward_ms`):

```text
global tokens=32768  (mb1):
  alltoall: 34.167 ms
  deepep:   11.674 ms
  speedup:  2.93x

global tokens=65536  (mb2):
  alltoall: 33.873 ms
  deepep:   12.664 ms
  speedup:  2.67x

global tokens=131072 (mb4):
  alltoall: 38.009 ms
  deepep:   17.690 ms
  speedup:  2.15x

global tokens=262144 (mb8):
  alltoall: 36.856 ms
  deepep:   33.949 ms
  speedup:  1.09x
```

  - both Megatron sweeps completed cleanly with:

```text
BENCH_DONE
EXIT_CODE=0
```

- Interpretation:
  - the matched Torch full-layer baseline is now available on the same fixed shape across the same first token scaling spine
  - for `topk=2`, Torch DeepEP remains strongly positive across all four points, but the speedup is not monotonic with scale
  - for `topk=8`, Torch DeepEP remains clearly positive through `131072` global tokens, but the gain shrinks materially by `262144`
  - this remains a same-shape comparison, not a perfectly distribution-controlled comparison:
    - the JAX hillclimb harness explicitly controls `distribution in {random, runs}`
    - the Megatron harness uses its own router behavior
- Next action:
  - publish the matched Torch scaling milestone on `#3717`
  - then compare the new Torch scaling table against the completed JAX `forward_backward` results to decide whether to deepen JAX coverage or seal the experiment

### 2026-03-16 21:16 - JAX `random, topk=8` scaling spine is positive through `131072`, and DeepEP expands feasibility at `262144`
- Context:
  - After the matched Torch scaling sweeps, the most obvious missing JAX coverage was the `random, topk=8` scaling spine.
  - The fixed-shape `32768` point was already complete from the fixed-shape quadrant.
- Commands:
  - `65536`:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --repo-ref 95e2e589ca16ec2008cbb36efb47bad4269bc9e8 \
  --task-id deepep-jax-krt-bench-20260316-ad-sweep-65536-random-topk8 \
  --kernels current,deepep_transport_capped_prewarmed \
  --tokens 65536 \
  --topk-list 8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 1 \
  --per-bench-timeout-seconds 180 \
  --per-bench-kill-after-seconds 10 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke
```

  - `131072`:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --repo-ref 95e2e589ca16ec2008cbb36efb47bad4269bc9e8 \
  --task-id deepep-jax-krt-bench-20260316-ad-sweep-131072-random-topk8 \
  --kernels current,deepep_transport_capped_prewarmed \
  --tokens 131072 \
  --topk-list 8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 1 \
  --per-bench-timeout-seconds 180 \
  --per-bench-kill-after-seconds 10 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke
```

  - `262144` baseline debug:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --repo-ref 95e2e589ca16ec2008cbb36efb47bad4269bc9e8 \
  --task-id deepep-jax-krt-bench-20260316-ad-current-262144-random-topk8-debug \
  --kernels current \
  --tokens 262144 \
  --topk-list 8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 1 \
  --per-bench-timeout-seconds 180 \
  --per-bench-kill-after-seconds 10 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup
```

  - `262144` DeepEP first attempt and timeout-at-exit:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --repo-ref 95e2e589ca16ec2008cbb36efb47bad4269bc9e8 \
  --task-id deepep-jax-krt-bench-20260316-ad-deepep-262144-random-topk8-debug \
  --kernels deepep_transport_capped_prewarmed \
  --tokens 262144 \
  --topk-list 8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 1 \
  --per-bench-timeout-seconds 180 \
  --per-bench-kill-after-seconds 10 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup
```

  - `262144` DeepEP rerun with a looser bench timeout:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --repo-ref 95e2e589ca16ec2008cbb36efb47bad4269bc9e8 \
  --task-id deepep-jax-krt-bench-20260316-ad-deepep-262144-random-topk8-timeout420 \
  --kernels deepep_transport_capped_prewarmed \
  --tokens 262144 \
  --topk-list 8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 1 \
  --per-bench-timeout-seconds 420 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke
```

- Result:
  - completed `random, topk=8` JAX rows:

```text
tokens=32768:
  current: 0.033823 s
  deepep_transport_capped_prewarmed: 0.029962 s
  speedup: 1.13x

tokens=65536:
  current: 0.068674 s
  deepep_transport_capped_prewarmed: 0.059088 s
  speedup: 1.16x

tokens=131072:
  current: 0.136179 s
  deepep_transport_capped_prewarmed: 0.122988 s
  speedup: 1.11x
```

  - exact-cap metadata on the added completed rows:

```text
tokens=65536:  max_recv_tokens=44032   max_local_assignments=65792   recv_factor=1.488372  assign_factor=7.968872
tokens=131072: max_recv_tokens=87808   max_local_assignments=131712  recv_factor=1.492711  assign_factor=7.961127
tokens=262144: max_recv_tokens=175360  max_local_assignments=262912  recv_factor=1.494891  assign_factor=7.976631
```

  - `current` at `tokens=262144` failed with an explicit OOM:

```text
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 66573344024 bytes.
```

  - DeepEP at the same `262144` cell produced a real result line on the first attempt:

```text
RESULT kernel=deepep_transport_capped_prewarmed ep=8 pass=forward_backward time_s=0.273059 tokens_per_s=960025.78
EXIT_CODE=124
```

  - rerunning the same DeepEP cell with `--per-bench-timeout-seconds 420 --per-bench-kill-after-seconds 20` allowed it to complete cleanly:

```text
RESULT kernel=deepep_transport_capped_prewarmed ep=8 pass=forward_backward time_s=0.283450 tokens_per_s=924833.76
BENCH_END kernel=deepep_transport_capped_prewarmed distribution=random topk=8
EXIT_CODE=0
```

- Interpretation:
  - the JAX `random, topk=8` scaling spine is positive through `131072` global tokens
  - at `262144`, the baseline `current` path is outside the tested memory envelope on this H100x8 regime, while the exact-cap DeepEP path remains runnable
  - on that largest `topk=8` point, the main caveat shifts from raw feasibility to lifecycle behavior:
    - the exact-cap path needs a looser per-bench timeout to get through post-result teardown and report cleanly
- Next action:
  - publish this feasibility milestone on `#3717`
  - then decide whether the experiment has enough evidence to seal or whether `runs` scaling coverage is still worth adding

### 2026-03-16 21:33 - `runs, topk=2` scaling becomes a steadier win as tokens grow
- Context:
  - The fixed-shape `runs, topk=2` JAX `forward_backward` cell was only a small win (`1.04x`), which left open whether the exact-cap DeepEP gain was especially sensitive to the `runs` distribution.
  - I extended that controlled JAX spine across the same token points already completed for `random, topk=2`.
- Commands:
  - `65536`:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --repo-ref 95e2e589ca16ec2008cbb36efb47bad4269bc9e8 \
  --task-id deepep-jax-krt-bench-20260316-ad-sweep-65536-runs-topk2 \
  --kernels current,deepep_transport_capped_prewarmed \
  --tokens 65536 \
  --topk-list 2 \
  --distributions runs \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 1 \
  --per-bench-timeout-seconds 180 \
  --per-bench-kill-after-seconds 10 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke
```

  - `131072` and `262144` used the same command shape with the corresponding `--tokens` values.
- Result:
  - complete `runs, topk=2` JAX spine:

```text
tokens=32768:
  current: 0.011068 s
  deepep_transport_capped_prewarmed: 0.010662 s
  speedup: 1.04x

tokens=65536:
  current: 0.020899 s
  deepep_transport_capped_prewarmed: 0.018808 s
  speedup: 1.11x

tokens=131072:
  current: 0.040382 s
  deepep_transport_capped_prewarmed: 0.036104 s
  speedup: 1.12x

tokens=262144:
  current: 0.080684 s
  deepep_transport_capped_prewarmed: 0.070273 s
  speedup: 1.15x
```

  - exact-cap metadata on the added three rows:

```text
tokens=65536:  max_recv_tokens=15744  max_local_assignments=16768  recv_factor=4.162602  assign_factor=7.816794
tokens=131072: max_recv_tokens=31232  max_local_assignments=33152  recv_factor=4.196721  assign_factor=7.907336
tokens=262144: max_recv_tokens=61952  max_local_assignments=65792  recv_factor=4.231405  assign_factor=7.968872
```

  - all three added runs completed with `EXIT_CODE=0`
- Interpretation:
  - the small fixed-shape `runs, topk=2` win was not a scaling dead-end
  - as tokens increase, the exact-cap DeepEP path widens from `1.04x` to about `1.11x`, `1.12x`, and `1.15x`
  - for `topk=2`, the current experiment now has controlled JAX scaling spines for both:
    - `distribution=random`
    - `distribution=runs`
- Next action:
  - decide whether to spend more cluster time on `runs, topk=8` or synthesize/seal the experiment around the now-complete controlled `topk=2` picture plus the `topk=8` feasibility result
