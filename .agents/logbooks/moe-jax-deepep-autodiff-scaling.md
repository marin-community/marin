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
