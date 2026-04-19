# Debugging Log For v6e euwest

Investigate why `v6e-4` inference in `europe-west4` fails during vLLM TPU mesh initialization while the same checkpoint and code path succeed in `us-east5`.

## Initial Status

The failing regional job is `/ahmed/bloom-eval-dpo-europe-west4-v6e4`. Its child inference step dies in `tpu_inference` during mesh bring-up with an `AttributeError` on `x.coords`, followed by `RuntimeError: Engine core initialization failed`. The same checkpoint `beta0.01_lr7.5e-7_seed0` has already succeeded on `us-east5` `v6e-4`.

## Hypothesis 1

The raw JAX device objects exposed on the eu-west4 TPU worker differ from the working us-east5 worker, and `tpu_inference` is assuming `.coords` exists when it does not.

## Changes To Make

- Add a small inspection script that dumps JAX device attributes, package versions, and TPU-related environment variables from inside an Iris task.
- Launch Iris CPU-only control jobs first with the same dependency extras.
- Launch matching Iris TPU jobs in `eu-west4` and `us-east5`.

## Future Work

- [ ] Compare raw JAX device attributes between eu-west4 and us-east5
- [ ] Check whether the difference is node-specific or region-wide
- [ ] Decide whether the right fix is retry, dependency pinning, or a code fallback around missing `.coords`

## Results

Added `experiments/posttrain/inspect_jax_runtime.py` and ran it on the Iris
cluster in both a working region (`us-east5`) and the failing region
(`europe-west4`).

Confirmed:
- both regions resolve the same Python package set inside Iris tasks:
  - `jax==0.8.0`
  - `jaxlib==0.8.0`
  - `vllm-tpu==0.13.2.post6`
  - `libtpu==0.0.24`
- on `us-east5` `v6e-4`, `jax.devices()` succeeds and returns
  `TpuDevice(..., coords=(...), core_on_chip=0)` objects
- on the original failing eu-west4 `v6e-4` worker
  `marin-tpu-v6e-4-europe-west4-a-20260403-0048-1f9f1d88`, `jax.devices()`
  fails before returning any devices with:
  - `TPU initialization failed: TPU_RET_CHECK failure`
  - `GetChip(i)->location().index_on_host() == i (0 vs. 1)`

This shows the `.coords` crash seen in vLLM is downstream. The underlying
problem is a lower-level TPU/JAX runtime bring-up failure on that eu-west4
worker, not a package mismatch and not a pure application bug.

The original eu-west4 `v6e-4` TPU VM was deleted and replaced with:

- `marin-tpu-v6e-4-europe-west4-a-20260403-2324-cc3cb2ab`

The same minimal TPU diagnostic failed on the replacement with the same raw JAX
error:

- `TPU initialization failed: TPU_RET_CHECK failure`
- `GetChip(i)->location().index_on_host() == i (0 vs. 1)`

That rules out a single bad worker and points to a broader issue with eu-west4
`v6e-4` TPU bring-up. A follow-up `v6e-8` eu-west4 control job is pending so we
can tell whether this is specific to the eu-west4 `v6e-4` pool or affects other
eu-west4 TPU shapes too.

The `v6e-8` eu-west4 control job then succeeded:

- job: `/ahmed/debug-v6e8-tpu-euw4-r1`
- worker: `marin-tpu-v6e-8-europe-west4-a-20260403-2329-df171a1a`
- `jax.devices()` returned 8 `TpuDevice(..., coords=...)` devices normally

So the issue is not "eu-west4 TPU bring-up is broken." The evidence now points
specifically at the eu-west4 `v6e-4` pool or topology path:

- `us-east5 v6e-4`: works
- `eu-west4 v6e-8`: works
- `eu-west4 v6e-4`: fails on two separate workers with the same raw JAX TPU
  initialization error
