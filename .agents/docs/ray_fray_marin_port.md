# Ray → Fray Migration Guide for Marin

## Overview

This document provides step-by-step porting patterns for migrating Marin's codebase from direct Ray usage to the Fray abstraction layer. Fray provides a backend-agnostic API that currently supports Ray, with the ability to add additional backends (like local execution for testing).

**Goals:**
- Improved testability (use `LocalJobContext` for tests)
- Cleaner architecture (abstract distributed execution)
- Preparation for multi-backend support
- No direct Ray imports in application code

**Scope:** ~90 files across Marin codebase:
- Core execution framework (`execution/`)
- Data processing (`processing/`, `generation/`)
- Training (`classifiers/`, `training/`)
- Transform scripts (`transform/`)

**Timeline:** 8-10 weeks with one engineer

---

## Required Additions to Fray

Fray already supports most features needed for migration:
- ✅ Task creation with resource specifications
- ✅ Actor creation with lifetime management (`Lifetime.DETACHED`)
- ✅ Task `max_calls` parameter for TPU cleanup
- ✅ Runtime environments (pip packages, env vars)
- ✅ `get()`, `wait()`, `put()` operations

**Missing features:**

### 1. Head Node Scheduling (ALREADY SUPPORTED)

**Solution:** Fray uses resource-based scheduling. To schedule an actor on the head node, use the `"head_node"` resource:

```python
from fray import ActorOptions, Lifetime

options = ActorOptions(
    name="status_actor",
    get_if_exists=True,
    lifetime=Lifetime.DETACHED,
    resources={"CPU": 0, "head_node": 0.0001}  # Small value claims the resource
)
actor = ctx.create_actor(StatusActor, options=options)
```

**How it works:**
- Ray clusters are initialized with `resources={"head_node": 1}` on the head node
- Actors request this resource to be scheduled there
- This is simpler than NodeAffinitySchedulingStrategy and works across backends

**No changes needed to Fray** - already supported!

### 2. PlacementGroups (OPTIONAL - DEFER)

**Problem:** vLLM inference uses PlacementGroups for tensor parallelism across TPU chips.

**Decision:** Keep Ray Data code as-is for vLLM inference, document as known exception. This is a complex specialized feature used in only one place (`generation/inference.py`).

**Effort:** 0 hours (deferred)

### 3. Ray Data Pipelines (OPTIONAL - DEFER)

**Problem:** vLLM inference uses Ray Data for scalable batch processing.

**Decision:** Ray Data is a high-level library that sits on top of Ray. Abstracting it is out of scope for Fray's core mission. Document as known exception.

**Effort:** 0 hours (deferred)

---

## Porting Patterns

### Pattern 1: Converting `@ray.remote` Tasks

**Description:** Replace decorated functions with plain functions, move configuration to call site.

**Before:**
```python
import ray

@ray.remote(max_calls=1, resources={"TPU": 4})
def process_file(input_path: str, output_path: str):
    # Processing logic
    return result

# Call site
ref = process_file.remote(input_path, output_path)
result = ray.get(ref)
```

**After:**
```python
from fray import get_job_context, TaskOptions

def process_file(input_path: str, output_path: str):
    # Processing logic (unchanged)
    return result

# Call site
ctx = get_job_context()
options = TaskOptions(
    max_calls=1,
    resources={"TPU": 4}
)
ref = ctx.create_task(process_file, (input_path, output_path), options=options)
result = ctx.get(ref)
```

**Key Changes:**
1. Remove `@ray.remote` decorator from function
2. Remove `import ray` if no longer needed
3. Add `from fray import get_job_context, TaskOptions` at call site
4. Create `TaskOptions` with resource specifications
5. Use `ctx.create_task()` instead of `.remote()`
6. Use `ctx.get()` instead of `ray.get()`

**Locations:**
- `processing/classification/inference.py` - `process_file_ray()`, `_process_dir()`, `run_inference()`
- `processing/tokenize/tokenize.py` - Multiple tokenization functions
- `classifiers/hf/launch_ray_training.py` - `train_classifier_distributed()`
- `generation/inference.py` - `run_inference()` (uses Ray Data - see Pattern 6)
- All `transform/` scripts (~80 files)

**Potential Issues:**
- Functions that return `ray.ObjectRef` types need signature updates
- Nested task creation requires context propagation (Fray handles this automatically)
- Decorators like `@remove_tpu_lockfile_on_exit` still work (applied before Fray wrapping)

---

### Pattern 2: Converting `@ray.remote` Actors

**Description:** Remove decorator, create actors via `ctx.create_actor()` with options.

**Before:**
```python
import ray

@ray.remote(num_cpus=0)
class StatusActor:
    def __init__(self, cache_size: int = 10_000):
        self.data = {}

    def get_value(self, key: str):
        return self.data.get(key)

# Create actor
actor = StatusActor.options(
    name="status_actor",
    get_if_exists=True,
    lifetime="detached",
    scheduling_strategy=schedule_on_head_node_strategy()
).remote()

# Call method
result = ray.get(actor.get_value.remote("key"))
```

**After:**
```python
from fray import get_job_context, ActorOptions, Lifetime

class StatusActor:  # No decorator!
    def __init__(self, cache_size: int = 10_000):
        self.data = {}

    def get_value(self, key: str):
        return self.data.get(key)

# Create actor
ctx = get_job_context()
options = ActorOptions(
    name="status_actor",
    get_if_exists=True,
    lifetime=Lifetime.DETACHED,
    resources={"CPU": 0, "head_node": 0.0001}  # Resource-based scheduling
)
actor = ctx.create_actor(StatusActor, options=options)

# Call method
result = ctx.get(actor.get_value("key"))
```

**Key Changes:**
1. Remove `@ray.remote` decorator from class
2. Remove `import ray`
3. Use `ActorOptions` instead of `.options()`
4. Use `Lifetime.DETACHED` enum instead of string
5. Use `resources={"head_node": 0.0001}` instead of `scheduling_strategy`
6. No `.remote()` on actor methods (Fray handles this)
7. Use `ctx.get()` instead of `ray.get()`

**Locations:**
- `execution/status_actor.py` - `StatusActor` class
- `execution/executor.py` - StatusActor creation (line ~548)

**Potential Issues:**
- Need to ensure Ray cluster is initialized with `resources={"head_node": 1}` on head node
- Actor handle types change from `ray.ActorHandle` to `RayActorHandle` (Fray wrapper)

---

### Pattern 3: Converting `ray.get()` and `ray.wait()`

**Description:** Replace Ray functions with context methods.

**Before:**
```python
import ray

refs = [task.remote(x) for x in items]

# Blocking wait for all
results = ray.get(refs)

# Wait for first N to complete
done_refs, pending_refs = ray.wait(refs, num_returns=3, timeout=10.0)
done_results = ray.get(done_refs)
```

**After:**
```python
from fray import get_job_context

ctx = get_job_context()
refs = [ctx.create_task(task, (x,)) for x in items]

# Blocking wait for all
results = ctx.get(refs)

# Wait for first N to complete
done_refs, pending_refs = ctx.wait(refs, num_returns=3, timeout=10.0)
done_results = ctx.get(done_refs)
```

**Key Changes:**
1. Replace `ray.get()` with `ctx.get()`
2. Replace `ray.wait()` with `ctx.wait()`
3. Signatures are identical

**Locations:**
- `execution/executor.py` - Lines 601, 638, 661, 1017, 1032
- All files that currently call `ray.get()` or `ray.wait()`

**Potential Issues:**
- None - drop-in replacement

---

### Pattern 4: Converting Resource Specifications

**Description:** Move resource specs from decorator to `TaskOptions` or `ActorOptions`.

**Before:**
```python
@ray.remote(
    num_cpus=4,
    num_gpus=1,
    memory=8*1024**3,
    resources={"TPU": 8, "TPU-v4-32-head": 1}
)
def big_task(data):
    return process(data)
```

**After:**
```python
from fray import TaskOptions

def big_task(data):
    return process(data)

# At call site
options = TaskOptions(
    resources={
        "CPU": 4,
        "GPU": 1,
        "memory": 8*1024**3,
        "TPU": 8,
        "TPU-v4-32-head": 1
    }
)
ref = ctx.create_task(big_task, (data,), options=options)
```

**Key Changes:**
1. Use `"CPU"` and `"GPU"` keys (instead of `num_cpus`, `num_gpus`)
2. Custom resources (like `"TPU"`) go in same dict
3. Memory specification uses `"memory"` key

**Locations:**
- `classifiers/hf/launch_ray_training.py` - Line 42-44
- `execution/executor.py` - Dynamic resource specs (line 705-714)
- All transform scripts with resource specifications

**Potential Issues:**
- Need to convert `num_cpus=X` to `"CPU": X`
- Need to convert `num_gpus=X` to `"GPU": X`

---

### Pattern 5: Converting Runtime Environments

**Description:** Convert Ray's `runtime_env` dict to Fray's `RuntimeEnv` object.

**Before:**
```python
@ray.remote(
    runtime_env={
        "pip": ["transformers>=4.0", "torch"],
        "env_vars": {"PJRT_DEVICE": "TPU"}
    }
)
def train_model(config):
    # training logic
    pass

# Or at call site:
ref = task.options(
    runtime_env={
        "pip": ["numpy"],
        "env_vars": {"FOO": "bar"}
    }
).remote(args)
```

**After:**
```python
from fray import TaskOptions, RuntimeEnv

def train_model(config):
    # training logic (unchanged)
    pass

# At call site
runtime_env = RuntimeEnv(
    package_requirements=["transformers>=4.0", "torch"],
    env={"PJRT_DEVICE": "TPU"}
)
options = TaskOptions(runtime_env=runtime_env)
ref = ctx.create_task(train_model, (config,), options=options)
```

**Key Changes:**
1. Create `RuntimeEnv` object
2. Use `package_requirements` instead of `"pip"`
3. Use `env` instead of `"env_vars"`
4. Pass as `runtime_env` field in `TaskOptions`

**Locations:**
- `classifiers/hf/launch_ray_training.py` - Line 44
- `execution/executor.py` - Lines 694-698 (per-step runtime_env)
- Various transform scripts with pip dependencies

**Potential Issues:**
- Need to handle conditional runtime_env (local vs cluster)
- Per-task runtime_env supported but may have overhead

---

### Pattern 6: Handling Ray Data (EXCEPTION)

**Description:** Ray Data usage is kept as-is, documented as known exception.

**Current Code (generation/inference.py):**
```python
import ray
import ray.data

@ray.remote(max_calls=1)
@remove_tpu_lockfile_on_exit
def run_inference(config: TextGenerationInferenceConfig):
    ds = ray.data.read_json(config.input_path)
    ds = ds.map_batches(
        vLLMTextGeneration,
        concurrency=config.num_instances,
        batch_size=config.batch_size,
        # ... PlacementGroup scheduling
    )
    ds.write_json(config.output_path)
```

**Approach:**
```python
"""
vLLM inference pipeline using Ray Data.

MIGRATION NOTE:
This module uses Ray Data directly due to:
1. PlacementGroup support for tensor parallelism (not in Fray core scope)
2. Ray Data's specialized batch processing pipeline
3. vLLM-specific Ray Data integration

This is a documented exception to the Fray migration.
"""
import ray
import ray.data

# Keep existing code unchanged
```

**Key Changes:**
1. Add module-level docstring documenting exception
2. No code changes
3. File remains Ray-dependent

**Locations:**
- `generation/inference.py` - `run_inference()` function
- `generation/ray_utils.py` - PlacementGroup scheduling functions

**Potential Issues:**
- None - intentionally left unchanged

---

### Pattern 7: Converting Executor Framework

**Description:** Update Executor to use Fray context for task scheduling and actor management.

**Before (executor.py:543-554):**
```python
import ray

if not is_local_ray_cluster():
    strategy = schedule_on_head_node_strategy()
else:
    strategy = None

self.status_actor: StatusActor = StatusActor.options(
    name="status_actor",
    get_if_exists=True,
    lifetime="detached",
    scheduling_strategy=strategy,
).remote()
```

**After:**
```python
from fray import get_job_context, set_job_context, ActorOptions, Lifetime
from fray.backend.ray.ray_job import RayJobContext

# Initialize Fray context
import ray
if not ray.is_initialized():
    ray.init(
        namespace="marin",
        ignore_reinit_error=True,
        resources={"head_node": 1} if is_local_ray_cluster() else None,
    )

ctx = RayJobContext()
set_job_context(ctx)

# Create status actor
options = ActorOptions(
    name="status_actor",
    get_if_exists=True,
    lifetime=Lifetime.DETACHED,
    resources={"CPU": 0, "head_node": 0.0001}  # Schedule on head node
)
self.status_actor = ctx.create_actor(StatusActor, options=options)
self.ctx = ctx  # Store for task launching
```

**Task Launching (executor.py:705-714):**

**Before:**
```python
if isinstance(step.fn, ray.remote_function.RemoteFunction):
    ref = step.fn.options(
        name=f"{get_fn_name(step.fn, short=True)}:{step.name}",
        runtime_env=runtime_env
    ).remote(config)
else:
    remote_fn = ray.remote(step.fn)
    ref = remote_fn.options(
        name=f"{get_fn_name(step.fn, short=True)}:{step.name}",
        runtime_env=runtime_env,
    ).remote(config)
```

**After:**
```python
from fray import TaskOptions, RuntimeEnv

# Extract function from RemoteFunction if needed
if isinstance(step.fn, ray.remote_function.RemoteFunction):
    fn = step.fn._function
else:
    fn = step.fn

# Build runtime_env
fray_runtime_env = RuntimeEnv(
    package_requirements=pip_packages,  # From build_runtime_env_for_packages
    env=runtime_env.get("env_vars", {})
)

options = TaskOptions(
    name=f"{get_fn_name(step.fn, short=True)}:{step.name}",
    runtime_env=fray_runtime_env
)
ref = self.ctx.create_task(fn, (config,), options=options)
```

**Key Changes:**
1. Initialize Fray context at startup
2. Create StatusActor via context
3. Extract function from `RemoteFunction` wrapper
4. Convert runtime_env dict to `RuntimeEnv` object
5. Use `ctx.create_task()` for step execution

**Locations:**
- `execution/executor.py` - `__init__()` method (line ~520-555)
- `execution/executor.py` - `_launch_step()` method (line ~664-718)
- `execution/executor.py` - `_run_steps()` method (line ~603-662)

**Potential Issues:**
- Need to handle `ExecutorFunction` type (can be `Callable | ray.remote_function.RemoteFunction | None`)
- Runtime environment building logic needs conversion
- Task naming might need adjustment

---

### Pattern 8: Initializing Fray Context

**Description:** Initialize Fray context at application entry points.

**Before:**
```python
import ray

ray.init(
    namespace="marin",
    ignore_reinit_error=True,
    resources={"head_node": 1} if is_local_ray_cluster() else None,
)

# Use Ray directly
ref = some_task.remote(args)
result = ray.get(ref)
```

**After:**
```python
import ray
from fray import set_job_context
from fray.backend.ray.ray_job import RayJobContext

# Still need to initialize Ray for the backend
ray.init(
    namespace="marin",
    ignore_reinit_error=True,
    resources={"head_node": 1} if is_local_ray_cluster() else None,
)

# Create and set Fray context
ctx = RayJobContext()
set_job_context(ctx)

# Now use Fray
from fray import get_job_context
ctx = get_job_context()
ref = ctx.create_task(some_task, (args,))
result = ctx.get(ref)
```

**Or for tests:**
```python
from fray import LocalJobContext, set_job_context

# No Ray needed!
ctx = LocalJobContext()
set_job_context(ctx)

# Tasks run in-memory
ctx = get_job_context()
ref = ctx.create_task(some_task, (args,))
result = ctx.get(ref)  # Executes immediately
```

**Key Changes:**
1. Ray still needs to be initialized (Fray uses it as backend)
2. Create `RayJobContext` and set via `set_job_context()`
3. Use `LocalJobContext` for tests (no Ray required)

**Locations:**
- `execution/executor.py` - `executor_main()` function (line ~1108-1112)
- Test files - Replace Ray initialization with `LocalJobContext`

**Potential Issues:**
- Context must be set before any Fray operations
- Need to ensure context propagates to nested tasks (Fray handles this automatically)

---

## Migration Sequence

Recommended order to minimize disruption:

1. **Add missing features to Fray** (1 week)
   - Implement `schedule_on_head` in `ActorOptions`
   - Add tests for new features

2. **Migrate StatusActor** (2 days)
   - Remove `@ray.remote` decorator
   - Update Executor to use Fray for actor creation
   - Update all StatusActor method calls

3. **Migrate Executor framework** (1 week)
   - Update initialization to use Fray context
   - Convert task launching to use `ctx.create_task()`
   - Update `ray.get()` and `ray.wait()` calls

4. **Migrate processing pipelines** (1 week)
   - `processing/classification/inference.py`
   - `processing/tokenize/tokenize.py`
   - Other processing scripts

5. **Migrate training code** (1 week)
   - `classifiers/hf/launch_ray_training.py`
   - `training/training.py`

6. **Document Ray Data exception** (2 days)
   - Add docstrings to `generation/inference.py`
   - Document PlacementGroup usage in `generation/ray_utils.py`

7. **Migrate transform scripts** (2 weeks)
   - Create helper function to reduce boilerplate
   - Migrate in batches by domain
   - ~80 files × 30 min avg = 40 hours

8. **Update tests** (1 week)
   - Replace Ray initialization with `LocalJobContext`
   - Add Fray-specific tests
   - Validate all functionality

---

## Testing Strategy

### Unit Tests
Use `LocalJobContext` for fast, deterministic tests:

```python
from fray import LocalJobContext, set_job_context

def test_my_function():
    ctx = LocalJobContext()
    set_job_context(ctx)

    ref = ctx.create_task(my_function, (args,))
    result = ctx.get(ref)

    assert result == expected
```

### Integration Tests
Use `RayJobContext` for distributed testing:

```python
import ray
from fray import RayJobContext, set_job_context

def test_distributed_workflow():
    ray.init(address="local")
    ctx = RayJobContext()
    set_job_context(ctx)

    # Test actual distributed execution
    ...
```

### Validation
- All existing tests must pass
- No performance regression (< 5% overhead)
- Ray cluster features work identically

---

## Common Pitfalls

1. **Forgetting to set context:** Always call `set_job_context()` before using `get_job_context()`

2. **Mixing Ray and Fray:** Don't use `ray.get()` on Fray refs or vice versa - pick one abstraction

3. **Resource specification confusion:** Use `"CPU"` not `num_cpus`, `"GPU"` not `num_gpus`

4. **Actor method calls:** Fray actors don't need `.remote()` - just call methods directly

5. **Runtime environment conversion:** Remember `"pip"` → `package_requirements`, `"env_vars"` → `env`

---

## Success Criteria

- ✅ Zero `@ray.remote` decorators in migrated code (except Ray Data exceptions)
- ✅ No direct `import ray` except for initialization and Ray Data files
- ✅ StatusActor uses Fray with `Lifetime.DETACHED` and `schedule_on_head=True`
- ✅ Executor launches tasks via `ctx.create_task()`
- ✅ All tests pass with both `LocalJobContext` and `RayJobContext`
- ✅ Performance within 5% of direct Ray implementation
- ✅ Documented exceptions for Ray Data and PlacementGroups

---

## Document Metadata

**Version:** 3.0
**Last Updated:** 2025-10-15
**Status:** Ready for implementation
**Effort Estimate:** 8-10 weeks (1 engineer)

**Changelog:**
- v3.0: Restructured as porting patterns guide
- v2.0: Complete rewrite with detailed Fray requirements
- v1.0: Initial draft
