# Training V2 (TPU) Design

## Overview

This document proposes how TPU training in `lib/marin/src/marin/training/training.py` should move from `fray.cluster` to `fray.v2` while preserving behavior and enabling reliable testing. The goal is to make training submission backend-agnostic (Ray or Iris) and align with the fray-lite interfaces described in `fray-lite-design.md`.

## Goals

1. Use `fray.v2` APIs (`current_client`, `JobRequest`, `JobHandle`) instead of `fray.cluster`.
2. Preserve TPU-specific behavior: replicas, environment variables, and region checks.
3. Provide a clean testing seam to mock the client and avoid TPU or Ray/Iris dependencies.
4. Keep training entrypoints unchanged for now (`train_lm.main`).

## Non-Goals

1. Changing Levanter training semantics or configs.
2. Redesigning TPU resource selection logic.
3. Adding compatibility shims between v1 and v2.

## Current State

`run_levanter_train_lm()` in `lib/marin/src/marin/training/training.py`:

1. Builds and mutates `TrainLmOnPodConfig`.
2. Calls `current_cluster()` from `fray.cluster`.
3. Submits a `JobRequest` and waits via `cluster.wait(job_id)`.

This is a pure job-submission path (no actors). It is the simplest fray-v2 migration target.

## Proposed Design

### API Swap

Replace v1 imports:

- `from fray.cluster import ... current_cluster` -> `from fray.v2 import ... current_client`
- `cluster.launch(job_request)` -> `client.submit(job_request)`
- `cluster.wait(job_id, ...)` -> `job.wait(...)`

### TPU Replicas

Fray v2 moves `replicas` from `ResourceConfig` to `JobRequest`. The current TPU config path should be updated to set:

- `JobRequest.replicas = resources.device.topology_replicas` (or the equivalent field used today by `ResourceConfig.with_tpu`).

If the v1 helper hides this, we should add a small helper in `training.py` to derive replicas from the TPU config and set it explicitly on `JobRequest`.

### ResourceConfig

Continue to build `ResourceConfig` as today (including `TpuConfig` or `CpuConfig`). This config stays on the job request unchanged. The backend will map the TPU config to Ray/Iris.

### Environment Variables

Keep the current environment and run-id logic. `EnvironmentConfig.create(env_vars=env)` remains unchanged. No new TPU env logic is introduced here.

### Client Resolution

Use `current_client()` for auto-detection. This will resolve to:

- Iris when running in an Iris job (via `iris_ctx` auto-detect)
- Ray when `ray.is_initialized()`
- LocalClient otherwise

This mirrors the fray-lite design.

## Mocking and Testing Strategy

### Unit Tests (No Ray/Iris/TPU)

Goal: Validate that we construct the right `JobRequest` without executing a training job.

1. Provide a test-only fake client:
   - `FakeClient.submit(request)` records the request and returns `FakeJobHandle`.
   - `FakeJobHandle.wait()` returns a succeeded `JobStatus` without executing anything.
2. Patch `marin.training.training.current_client` to return the fake client.
3. Assert:
   - `request.resources.device.kind` matches the input config
   - `request.replicas` matches expected TPU replicas
   - `request.environment.env_vars` includes computed env (RUN_ID, WANDB, etc.)

This keeps unit tests fast and deterministic.

### Integration Tests (LocalClient)

Use `fray.v2.local.LocalClient` (or `current_client` default) and a no-op entrypoint:

1. Patch `train_lm.main` to a small function that writes a marker file or returns quickly.
2. Run `run_levanter_train_lm()` and ensure it completes.

This validates the `JobRequest` flow without Ray/Iris.

### TPU-Specific Assertions

We do not attempt to run TPU code in tests. Instead, validate that the TPU `JobRequest` fields are set correctly:

- `resources.device` is `TpuConfig` with expected `variant`
- `replicas` matches TPU topology

## Implementation Notes

1. Introduce a small helper in `training.py` for client creation to simplify mocking:

```python

def _get_client() -> Client:
    return current_client()
```

Tests can patch `marin.training.training._get_client` instead of patching `current_client` directly.

2. Keep `_suppress_ray_config` intact for now. It is still relevant for Ray, and harmless for Iris/Local.

3. Avoid adding compatibility shims. All call sites should be updated to v2 directly.

## Migration Steps

1. Update `training.py` imports and submission logic to use `fray.v2`.
2. Update tests under `tests/test_training.py` to use the new v2 types.
3. Add a unit test for `replicas` on `JobRequest` once the helper is added.
4. Run `uv run --package marin pytest tests/test_training.py`.

## Spiral Implementation Plan (Iris Guidance)

### Step 1: Local CPU smoke (fast, deterministic)

Goal: Validate the v2 call path without Ray/Iris/TPU.

1. Patch `marin.training.training._get_client` to return `LocalClient`.
2. Patch `train_lm.main` to a no-op that returns quickly.
3. Run `run_levanter_train_lm()` and assert completion.

### Step 2: Iris smoke-test job (fast sanity)

Goal: Validate `current_client()` auto-detect + Iris submission on the smallest
possible job, before moving to Levanter training.

Use the existing Iris smoke-test callable (fast, deterministic), then move on.

### Step 3: Iris CPU Levanter job (tiny tutorial run)

Goal: Validate Iris + fray v2 + Levanter end-to-end on CPU.

Use the tutorial job:

- `experiments/tutorials/train_tiny_model_cpu.py`

This runs a tiny Llama nano model on TinyStories with CPU resources and
`num_train_steps=100`. It is the smallest end-to-end Levanter training job
in the repo.

### Step 4: Iris TPU Levanter job (tiny tutorial run)

Goal: Validate TPU scheduling and Levanter training path on Iris.

Use the tutorial job:

- `experiments/tutorials/train_tiny_model_tpu.py`

This runs a tiny 30M model on TinyStories with `ResourceConfig.with_tpu("v4-8")`.
It is the smallest TPU training job in the tutorials.

### Step 5: Iris TPU follow-up (optional, more coverage)

If we need more coverage (e.g., JAX device checks), add a separate TPU sanity
step, but keep training as the primary validation path. Prefer extending
the tutorial training configs over generic smoke-callables.

## Open Questions (Resolved)

1. **Where is the canonical TPU replicas field in v2?**
   Resolved: `JobRequest.replicas` is canonical, but it defaults from
   `ResourceConfig.replicas` in `JobRequest.__post_init__`. In v2,
   `ResourceConfig.with_tpu(..., slice_count=N)` sets `replicas=N`. Therefore
   `run_levanter_train_lm()` can rely on the existing `ResourceConfig.with_tpu`
   call site and does **not** need to compute replicas manually.

   Recommendation: keep using `ResourceConfig.with_tpu(..., slice_count=...)`
   and leave `JobRequest.replicas=None` so the defaulting logic applies. Only
   set `JobRequest.replicas` explicitly if we need to override the slice count
   independently of the resource spec.

2. **Should `_suppress_ray_config` stay in a backend-agnostic file?**
   Resolved: It is still acceptable to keep it in `training.py` because it is
   a no-op outside Ray and prevents accidental Ray auto-start when the backend
   is Ray. Moving it behind backend detection adds complexity without benefit
   today. Revisit if Iris/Local semantics diverge or if we want to avoid Ray
   references in the training module entirely.
