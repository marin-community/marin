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
