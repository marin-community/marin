# Simple Fray Migration Template

This document provides a step-by-step template for migrating simple `@ray.remote` decorated functions to Fray. For background, see [fray-migration.md](fray-migration.md).

## When to Use This Template

This template applies when:
1. A function is decorated with `@ray.remote` (with or without options)
2. The function is called via `ray.get(fn.remote(...))` or `ray.get(fn.options(...).remote(...))`
3. The function is a simple task (not an actor class)
4. The function doesn't use Ray Data or actor pools

## Migration Pattern

### Step 1: Update Imports

**Remove:**
```python
import ray
from fray.cluster.ray import as_remote_kwargs
from fray.cluster.ray.deps import build_runtime_env_for_packages
```

**Add:**
```python
from fray.cluster import (
    Entrypoint,
    JobRequest,
    ResourceConfig,
    create_environment,
    current_cluster,
)
```

**Keep if used by Levanter internally:**
```python
from levanter.distributed import RayConfig  # Still needed by Levanter
```

### Step 2: Remove @ray.remote Decorator

**Before:**
```python
@ray.remote(
    memory=64 * 1024 * 1024 * 1024,
    max_calls=1,
    runtime_env={"env_vars": {"FOO": "bar"}},
)
@remove_tpu_lockfile_on_exit  # if present
def do_work(config: Config) -> None:
    ...
```

**After:**
```python
def do_work(config: Config) -> None:
    ...
```

Notes:
- `memory` and `max_calls` are Ray-specific, drop them
- `runtime_env` env vars move to `create_environment(env_vars=...)`
- `@remove_tpu_lockfile_on_exit` decorator moves to a wrapper (see Step 3)

### Step 3: Update the Caller

**Before:**
```python
def run_work(config: WorkConfig) -> None:
    ray.get(
        do_work.options(
            resources={
                "TPU": config.resource_config.chip_count(),
                f"TPU-{config.resource_config.device.type}-head": 1,
            }
        ).remote(inner_config)
    )
```

**After:**
```python
def run_work(config: WorkConfig) -> None:
    def _run():
        with remove_tpu_lockfile_on_exit():  # if TPU task
            do_work(inner_config)

    job_request = JobRequest(
        name="descriptive-job-name",
        entrypoint=Entrypoint.from_callable(_run),
        resources=config.resource_config,
        environment=create_environment(env_vars={"FOO": "bar"}),
    )

    cluster = current_cluster()
    job_id = cluster.launch(job_request)
    cluster.wait(job_id)
```

### Step 4: Handle as_remote_kwargs Pattern

If the code uses `as_remote_kwargs`:

**Before:**
```python
from fray.cluster.ray import as_remote_kwargs

@ray.remote(**as_remote_kwargs(resource_config, env_vars=env), max_calls=1)
def task():
    ...
```

**After:**
```python
job_request = JobRequest(
    name="task-name",
    entrypoint=Entrypoint.from_callable(task),
    resources=resource_config,
    environment=create_environment(env_vars=env),
)
```

### Step 5: Handle CPU-only Tasks

For tasks that only need CPU (no TPU):

```python
job_request = JobRequest(
    name="cpu-task",
    entrypoint=Entrypoint.from_callable(task),
    resources=ResourceConfig.with_cpu(),  # or just ResourceConfig()
    environment=create_environment(env_vars={"JAX_PLATFORMS": "cpu"}),
)
```

### Step 6: Clean Up Unused Imports

After migration, remove:
- `from fray.cluster.base import TpuConfig` (if only used for assertions)
- Any other Ray-specific imports no longer needed

## Checklist

- [ ] Updated imports (removed ray, added fray.cluster imports)
- [ ] Removed `@ray.remote` decorator from function
- [ ] Moved `@remove_tpu_lockfile_on_exit` to wrapper if present
- [ ] Created `JobRequest` with appropriate name, entrypoint, resources, environment
- [ ] Replaced `ray.get(fn.remote(...))` with `cluster.launch()` + `cluster.wait()`
- [ ] Removed unused imports
- [ ] Ran `ruff check --fix` and `ruff format`
- [ ] Verified code compiles

## Examples

### Example 1: visualize.py (TPU task)

See commit `cffbf3fb3` for a complete example of migrating a TPU visualization task.

### Example 2: CPU-only task with env vars

```python
# Before
@ray.remote(runtime_env={"env_vars": {"JAX_PLATFORMS": "cpu", "PJRT_DEVICE": "cpu"}})
def run_dedup(config):
    ...

def caller():
    ray.get(run_dedup.remote(config))

# After
def run_dedup(config):
    ...

def caller():
    job_request = JobRequest(
        name="run-dedup",
        entrypoint=Entrypoint.from_callable(run_dedup, args=[config]),
        resources=ResourceConfig.with_cpu(),
        environment=create_environment(env_vars={"JAX_PLATFORMS": "cpu", "PJRT_DEVICE": "cpu"}),
    )
    cluster = current_cluster()
    cluster.wait(cluster.launch(job_request))
```
