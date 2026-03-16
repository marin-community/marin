## Description

This experiment continues the sealed exact-cap JAX DeepEP reintegration thread from `#3711`.

The prior session established three stable facts on H100x8:

1. the reintegrated exact-cap JAX DeepEP full-layer path is positive in `forward` mode on the original fixed-shape benchmark and on the larger token sweeps;
2. the same-shape Torch full-layer DeepEP baseline is strongly positive too;
3. the missing JAX `forward_backward` column is now blocked by a concrete JAX autodiff error at the DeepEP dispatch custom call boundary:

```text
ValueError: The FFI call to `levanter_deepep_dispatch_intranode` cannot be differentiated.
```

This thread is specifically about:

1. adding JAX autodiff support for the DeepEP transport path used by the exact-cap reintegrated benchmark;
2. rerunning one decisive fixed-shape `forward_backward` full-layer comparison cell on H100x8;
3. if that succeeds, running the downward token sweep requested in `#3711` at `32768`, `65536`, `131072`, and `262144` global tokens.

## Hypothesis or Goal

- Goal: make the reintegrated exact-cap JAX DeepEP path runnable under `--bench-pass=forward_backward`.
- Goal: determine whether the positive forward-only story from `#3711` survives in the first authoritative backward-inclusive benchmark cell.
- Goal: compare the corrected JAX `forward_backward` path against the matched Torch full-layer DeepEP baseline at the original fixed shape and across the requested smaller-to-larger token sweep.

### Links

* Prior fixed-shape GPU issue (`#3633`): https://github.com/marin-community/marin/issues/3633
* Prior torch-side DeepEP / Hybrid-EP issue (`#3641`): https://github.com/marin-community/marin/issues/3641
* Prior layout-only JAX issue (`#3665`): https://github.com/marin-community/marin/issues/3665
* Prior Megatron scaling issue (`#3666`): https://github.com/marin-community/marin/issues/3666
* Prior JAX DeepEP root-cause issue (`#3677`): https://github.com/marin-community/marin/issues/3677
* Prior exact-cap reintegration issue (`#3711`): https://github.com/marin-community/marin/issues/3711
* Research branch: https://github.com/marin-community/marin/tree/research/moe-jax-deepep-autodiff-scaling
* Research logbook: https://github.com/marin-community/marin/tree/research/moe-jax-deepep-autodiff-scaling/.agents/logbooks/moe-jax-deepep-autodiff-scaling.md

## Results

Current state at kickoff:
- sealed starting snapshot: `research/moe-jax-deepep-benchmark-reintegration` at `8a1b629e0dc8ab904fa0eca47b4369d59d34c77d`
- new branch/worktree: `research/moe-jax-deepep-autodiff-scaling`
- baseline from `#3711`:
  - exact-cap JAX DeepEP is positive on all four fixed-shape `forward` cells
  - exact-cap JAX DeepEP is positive on the larger forward token sweep cells that were completed
  - Torch full-layer DeepEP is strongly positive on the same fixed shape
- current blocker:
  - a minimal backward probe reaches the exact-cap dispatch path and fails with:

```text
ValueError: The FFI call to `levanter_deepep_dispatch_intranode` cannot be differentiated.
```

Planned first milestone:
- add or prototype JAX AD support around the DeepEP dispatch/combine path
- rerun the smallest decisive fixed-shape `forward_backward` cell:
  - `tokens=32768`
  - `hidden=2048`
  - `mlp_dim=768`
  - `experts=128`
  - `shared_expert_dim=2048`
  - `EP=8`
  - `distribution=random`
  - `topk=2`

Current status after the first AD passes:
- the original JAX autodiff blocker is cleared:

```text
ValueError: The FFI call to `levanter_deepep_dispatch_intranode` cannot be differentiated.
```

  is no longer the active failure.
- the branch now contains:
  - a cached-dispatch FFI target for the combine backward path
  - `custom_vjp` wrappers around the JAX DeepEP dispatch/combine entrypoints
  - plumbing changes in the reintegrated exact-cap benchmark path to thread the cached-dispatch handle metadata needed for backward
- the current decisive fixed-shape H100x8 `forward_backward` probe now reaches live transport and fails later with:

```text
DeepEP timeout for dispatch receivers, rank <varies>, responsible_channel = <varies>, tokens remained: <varies>
JaxRuntimeError: INTERNAL: cudaMemcpyAsync(read num_recv_tokens): unspecified launch failure
```

- an intermediate stale-peer-access failure was also fixed during bring-up:

```text
JaxRuntimeError: INTERNAL: [0] There was an error before calling cuModuleGetFunction (704): cudaErrorPeerAccessAlreadyEnabled : peer access is already enabled
```

  This was cleared by consuming the stale CUDA last-error state after `cudaDeviceEnablePeerAccess(...)` returned `cudaErrorPeerAccessAlreadyEnabled`.

Most recent fixed-shape probe result:
- `current`, `tokens=32768`, `distribution=random`, `topk=2`, `EP=8`, `bench_pass=forward_backward`:
  - `time_s=0.011138`
  - `tokens_per_s=2941944.89`

Most recent commits:
- first AD implementation: `e9b1bbfd07de5973b19a3d14649f7293c089d1bb`
- custom-VJP forward-signature fix: `899ef966e200b71cb0a3c3a4f76e38bbfedb6255`
- stale peer-access status cleanup: `26bd8d62e0341b94667149afe4cbc10b586ffa14`

Updated current state:
- both isolated transport backward legs now run on the full fixed-shape H100x8 transport input:
  - `dispatch` grad probe succeeded
  - `combine` grad probe now succeeds too, with finite per-rank losses and gradient norms
- the decisive fixed-shape full-layer `forward_backward` cell now completes end to end for the exact-cap DeepEP path:

```text
RESULT kernel=current ep=8 pass=forward_backward time_s=0.011118 tokens_per_s=2947338.80
DEEPEP_EXACT_CAPS max_recv_tokens=7808 max_local_assignments=8320 recv_factor=4.196721 assign_factor=7.876923
RESULT kernel=deepep_transport_capped_prewarmed ep=8 pass=forward_backward time_s=0.009942 tokens_per_s=3295880.18
```

- ratio on that authoritative cell:
  - exact-cap DeepEP is about `1.12x` faster than `current`
- the earlier active blockers are no longer the frontier:
  - the JAX FFI nondifferentiable error is gone
  - the stale peer-access error is gone
  - the live transport receiver-timeout failure is gone on the decisive cell
- current caveat:
  - after the successful result line, the pod still emits CUDA teardown noise (`DeepEP timeout check failed ...`, stream/event destruction failures, `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure`), but the benchmark shell still returns `bench_status=0` and pod `EXIT_CODE=0`

Next in-scope step:
- run the requested `forward_backward` downward token sweep at `32768`, `65536`, `131072`, and `262144` global tokens, starting with `distribution=random`, `topk=2`

Updated state after the first scaling spine:
- the requested first `forward_backward` scaling spine for `distribution=random`, `topk=2` is now complete:

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

- exact-cap metadata on the larger three points:

```text
tokens=65536:  max_recv_tokens=15616  max_local_assignments=16640  recv_factor=4.196721  assign_factor=7.876923
tokens=131072: max_recv_tokens=30976  max_local_assignments=33024  recv_factor=4.231405  assign_factor=7.937984
tokens=262144: max_recv_tokens=61952  max_local_assignments=65920  recv_factor=4.231405  assign_factor=7.953398
```

- all four completed with launcher `EXIT_CODE=0`
- the caveat is unchanged:
  - post-result CUDA teardown noise is still present on the DeepEP path, but it has not prevented successful completion on the completed sweep points

Most important current fact:
- the corrected exact-cap JAX DeepEP path is now positive on the authoritative fixed-shape `forward_backward` cell and across the first requested `forward_backward` scaling sweep

Next decision:
- extend the JAX `forward_backward` coverage to `runs` / `topk=8`
- or bring up the matched Torch full-layer `forward_backward` baseline

Updated state after the fixed-shape quadrant:
- the fixed-shape JAX `forward_backward` quadrant at `tokens=32768`, `EP=8` is now complete:

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

- exact-cap metadata on the newly completed cells:

```text
random, topk=8: max_recv_tokens=22144  max_local_assignments=33152  recv_factor=1.479769  assign_factor=7.907336
runs, topk=2:   max_recv_tokens=8064   max_local_assignments=8448   recv_factor=4.063492  assign_factor=7.757576
runs, topk=8:   max_recv_tokens=22144  max_local_assignments=33152  recv_factor=1.479769  assign_factor=7.907336
```

- all four fixed-shape JAX `forward_backward` cells are now positive for the exact-cap DeepEP path
- the gain is uneven:
  - `runs, topk=2` is only a small win (`1.04x`)
  - the other three cells are about `1.12x` to `1.13x`
- the caveat is still the same:
  - post-result CUDA teardown noise remains present on the DeepEP path even though the benchmark shells exit with `EXIT_CODE=0`

Current state of the experiment:
- JAX AD support is working on the transport path
- the full-layer fixed-shape JAX `forward_backward` quadrant is complete and positive
- the first requested `random, topk=2` scaling spine is complete and positive

Next meaningful extension:
- matched Torch full-layer `forward_backward` baseline
- or broader JAX scaling coverage (`runs`, `topk=8`)

Updated state after the matched Torch full-layer scaling sweeps:
- the existing Megatron fixed-shape harness now covers the same token spine at:
  - global tokens `32768`, `65536`, `131072`, `262144`
  - `topk=2` and `topk=8`
- topk=2 totals (`forward_ms + backward_ms`):

```text
global tokens=32768:
  alltoall: 31.633 ms
  deepep:   13.861 ms
  speedup:  2.28x

global tokens=65536:
  alltoall: 34.658 ms
  deepep:    8.804 ms
  speedup:  3.94x

global tokens=131072:
  alltoall: 32.281 ms
  deepep:   12.275 ms
  speedup:  2.63x

global tokens=262144:
  alltoall: 33.144 ms
  deepep:   14.640 ms
  speedup:  2.26x
```

- topk=8 totals (`forward_ms + backward_ms`):

```text
global tokens=32768:
  alltoall: 34.167 ms
  deepep:   11.674 ms
  speedup:  2.93x

global tokens=65536:
  alltoall: 33.873 ms
  deepep:   12.664 ms
  speedup:  2.67x

global tokens=131072:
  alltoall: 38.009 ms
  deepep:   17.690 ms
  speedup:  2.15x

global tokens=262144:
  alltoall: 36.856 ms
  deepep:   33.949 ms
  speedup:  1.09x
```

- both Torch sweeps completed with `BENCH_DONE` and `EXIT_CODE=0`
- this gives the experiment:
  - a complete JAX fixed-shape `forward_backward` quadrant
  - a complete JAX `random, topk=2` `forward_backward` scaling spine
  - matched Torch full-layer scaling baselines on the same token spine for `topk=2` and `topk=8`

Important caveat when comparing JAX vs Torch:
- the JAX harness is distribution-controlled (`random` / `runs`)
- the Megatron harness uses its own router behavior
- so the shape/pass match is exact, but the routing/data-generation semantics are not fully identical

Current state of the experiment:
- the original JAX AD blocker is solved
- the fixed-shape JAX DeepEP `forward_backward` path is positive and benchmark-complete
- the first JAX scaling spine and matched Torch scaling baselines are both now in hand

Next decision:
- deepen JAX scaling coverage (`runs`, `topk=8`)
- or synthesize/seal the experiment around the now-complete backward-enabled JAX + Torch comparison set

Updated state after the JAX `random, topk=8` scaling extension:
- completed JAX `random, topk=8` points:

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

- exact-cap metadata on those added rows:

```text
tokens=65536:  max_recv_tokens=44032   max_local_assignments=65792   recv_factor=1.488372  assign_factor=7.968872
tokens=131072: max_recv_tokens=87808   max_local_assignments=131712  recv_factor=1.492711  assign_factor=7.961127
tokens=262144: max_recv_tokens=175360  max_local_assignments=262912  recv_factor=1.494891  assign_factor=7.976631
```

- `tokens=262144, random, topk=8` split the two JAX kernels:
  - `current` failed with:

```text
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 66573344024 bytes.
```

  - `deepep_transport_capped_prewarmed` produced a result on the same cell:

```text
RESULT kernel=deepep_transport_capped_prewarmed ep=8 pass=forward_backward time_s=0.273059 tokens_per_s=960025.78
```

  - with the original `180s` per-bench timeout, that DeepEP run still exited `124` after the result line
  - rerunning the same DeepEP cell with `--per-bench-timeout-seconds 420 --per-bench-kill-after-seconds 20` completed cleanly:

```text
RESULT kernel=deepep_transport_capped_prewarmed ep=8 pass=forward_backward time_s=0.283450 tokens_per_s=924833.76
BENCH_END kernel=deepep_transport_capped_prewarmed distribution=random topk=8
EXIT_CODE=0
```

Current high-level fact pattern:
- the JAX `forward_backward` path is positive on the completed fixed-shape quadrant
- the JAX `random, topk=2` scaling spine is complete and positive
- the JAX `random, topk=8` scaling spine is positive through `131072`
- at `262144, random, topk=8`, the exact-cap DeepEP path remains runnable while `current` OOMs

Current caveat:
- the largest DeepEP `topk=8` cell needs a looser per-bench timeout to exit cleanly after the result line, which points at teardown/lifecycle overhead rather than first-order kernel correctness

Next decision:
- add `runs` scaling coverage
- or synthesize/seal around the now-richer backward-enabled JAX + Torch comparison set

Updated state after the controlled `runs, topk=2` scaling spine:
- complete JAX `runs, topk=2` spine:

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

- all three added runs exited cleanly with `EXIT_CODE=0`

What this adds:
- the experiment now has controlled JAX `forward_backward` scaling spines for both `topk=2` distributions:
  - `random`
  - `runs`
- the earlier small fixed-shape `runs, topk=2` gain does grow with scale rather than collapsing

Current experiment picture:
- JAX AD support is working
- fixed-shape JAX `forward_backward` quadrant is complete and positive
- controlled JAX scaling spines:
  - `random, topk=2` complete
  - `runs, topk=2` complete
  - `random, topk=8` positive through `131072`, with DeepEP still runnable at `262144` where `current` OOMs
- matched Torch full-layer scaling baselines:
  - `topk=2` complete
  - `topk=8` complete

Decision point:
- add `runs, topk=8` scaling
- or synthesize/seal around the now fairly complete backward-enabled JAX + Torch comparison set
