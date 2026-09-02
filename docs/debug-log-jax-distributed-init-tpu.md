# Debug Log: Iris TPU JAX Distributed Init

## Context

Two 520M Iris training parents failed after reaching Levanter trainer setup:

- `/calvinxu/dm-qsplit240-520m-chinchilla-pilot-20260414-174654`
- `/calvinxu/dm-stratified-520m-10p4b-20260414-174726`

The child logs raised:

```text
ValueError: Please initialize the distributed system via `jax.distributed.initialize()` at the start of your program.
```

The stack was in JAX async checkpoint manager creation during `Trainer(...)`.

## Root Cause

Levanter detected the Iris job context and called `iris.runtime.jax_init.initialize_jax()`, but Iris returned early on TPU:

```text
TPU detected; skipping Iris JAX distributed init (TPU runtime handles it)
```

That assumption is no longer valid for the current JAX checkpoint path. The TPU runtime can make devices available, but JAX async checkpointing still requires `jax.distributed.initialize()` to have run before the checkpoint manager is created.

## Fix

`iris.runtime.jax_init.initialize_jax()` now uses the Iris endpoint registry for TPU jobs too:

- Task 0 registers `jax_coordinator`.
- Non-zero tasks poll for the coordinator endpoint.
- All tasks call `jax.distributed.initialize(...)`.
- Single-task Iris jobs initialize as a one-process distributed runtime.

## Validation

Ran:

```bash
uv run --with pytest --with pytest-xdist --with pytest-timeout --with pytest-asyncio --with pytest-flakefinder python -m pytest lib/iris/tests/test_jax_init.py lib/iris/tests/test_jax_init_integration.py
```

Result: 18 passed.

## Follow-up: Concurrent Child Endpoint Collision

The first fixed 520M pilot relaunch:

```text
/calvinxu/dm-qsplit240-520m-chinchilla-pilot-20260414-224306
```

got past the missing-initialization failure, but concurrent training children resolved each other's JAX
coordinator endpoints. For example, task 0 of `train_lm_run_00090-0b8a0d` advertised `10.202.0.132`,
while its siblings connected to `10.202.0.82`, which belonged to another child running under the same
root Iris job namespace.

Root cause: Iris endpoint namespaces are root-job scoped, but `initialize_jax()` used the fixed endpoint
name `jax_coordinator` for every child job. Concurrent Levanter trainings in one parent therefore raced
on the same endpoint name.

Fix: `initialize_jax()` now scopes the endpoint name by the full child job id using
`JobName.to_safe_token()`, so siblings of one child still rendezvous together while parallel children
cannot cross-resolve.

Validation after the follow-up fix:

```bash
uv run --with pytest --with pytest-xdist --with pytest-timeout --with pytest-asyncio --with pytest-flakefinder python -m pytest lib/iris/tests/test_jax_init.py lib/iris/tests/test_jax_init_integration.py
```

Result: 19 passed.
