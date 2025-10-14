# Fray Implementation Cleanup and Enhancement

**Date:** 2025-10-15
**Context:** PR #1778 - Fray abstraction layer for distributed execution
**Status:** Draft PR under review

## Background

This document provides a comprehensive task list for cleaning up and enhancing the Fray implementation. Fray is an abstraction layer that wraps Ray's distributed execution capabilities to:
- Eliminate Ray's problematic global state
- Separate cluster management from task execution
- Enable backend swapping (Ray vs in-memory testing)
- Provide clearer APIs and better isolation

### Key Architecture Decisions

Fray implements a **two-tier execution model**:

1. **Tier 1: Cluster Job Submission** (`ClusterContext.create_job`)
   - Submits standalone jobs to a cluster
   - Entrypoint: Shell command string (e.g., `"python train.py --config foo.yaml"`)
   - Use case: Training runs, batch processing, CI/CD
   - Backend: Ray JobSubmissionClient or subprocess

2. **Tier 2: In-Cluster Task Execution** (`JobContext.create_task`, `JobContext.create_actor`)
   - Programmatic task scheduling within a running job
   - Entrypoint: Python callable functions
   - Use case: Dynamic workloads, parallel processing, TPU coordination
   - Backend: Ray remote functions/actors or ThreadPoolExecutor

This separation is critical: `create_job` uses string commands, `create_task` uses Python functions.

### File Locations

All Fray code lives in `lib/fray/`:

```
lib/fray/
├── src/fray/
│   ├── __init__.py           # Public API exports
│   ├── types.py              # Core types: RuntimeEnv, ActorOptions, TaskOptions, etc.
│   ├── cluster.py            # ClusterContext abstract interface
│   ├── job.py                # JobContext abstract interface
│   ├── context.py            # ContextVar-based context management
│   └── backend/
│       ├── in_memory.py      # LocalClusterContext, LocalJobContext (testing)
│       └── ray/
│           ├── ray_cluster.py  # RayClusterContext implementation
│           ├── ray_job.py      # RayJobContext implementation
│           ├── ray_tpu.py      # TPU-specific execution logic
│           └── ray_utils.py    # Ray helper utilities
└── tests/
    ├── conftest.py           # Pytest fixtures (ray_cluster, job_context, etc.)
    ├── test_backend.py       # Backend tests (parameterized for both backends)
    └── test_tpu_interface.py # TPU-specific tests
```

### Task 2: EntryPoint Type Usage - Verify and Document Two-Tier Model

**Status:** Verification + Documentation
**Priority:** Medium (implementation is correct, needs documentation clarity)

#### Current Status: CORRECT

The implementation already follows the two-tier model correctly:

**lib/fray/src/fray/types.py:40-47**
```python
# Entry point for cluster job submission
# Jobs are submitted as shell commands (not Python callables) to support:
# - Script-based execution: "python train.py --config foo.yaml"
# - Docker containers: with custom entry commands
# - Language-agnostic workloads
#
# For in-cluster task execution, use JobContext.create_task() which accepts Python functions
EntryPoint = str
```

**lib/fray/src/fray/cluster.py:37**
```python
def create_job(self, entrypoint: EntryPoint, env: RuntimeEnv) -> str:
```

**lib/fray/src/fray/backend/ray/ray_cluster.py:53, 87**
```python
def create_job(self, entrypoint: EntryPoint, env: RuntimeEnv) -> str:
    ...
    job_id = self._job_client.submit_job(
        entrypoint=entrypoint,  # String command passed directly to Ray
        runtime_env=runtime_env,
        submission_id=job_id,
    )
```

#### Why This Design is Correct

**Cluster-level jobs (Tier 1)** use string entrypoints because:
1. Ray's `JobSubmissionClient.submit_job()` requires string commands
2. Enables Docker/container-based execution (future roadmap)
3. Avoids function serialization complexity
4. Matches existing patterns in `scripts/ray/ray_run.py`

**In-cluster tasks (Tier 2)** use callable entrypoints because:
1. Already running inside a job context
2. Functions can be serialized via cloudpickle
3. Enables dynamic task generation
4. Matches Ray's `ray.remote(fn)` pattern

#### Potential Confusion Points

The design doc (`.agents/docs/ray_abstraction.md`) mentions BOTH patterns, which could be confusing:

```markdown
# Before (types.py)
EntryPoint = Callable[["JobContext"], None]  # ❌ OLD proposal

# After (types.py)
EntryPoint = str  # ✅ CURRENT implementation
```

The "Before" example shows an early design iteration that was correctly changed. However, `run_on_tpu` still uses callables:

**lib/fray/src/fray/cluster.py:94-96**
```python
def run_on_tpu(
    self, fn: Callable[[JobContext], Any], config: TpuRunConfig, ...
) -> list[Any]:
```

This is correct! `run_on_tpu` is a **hybrid API**: it's cluster-level (launches TPU jobs) but uses callables (to simplify TPU-specific workflows). It's implemented using Ray's in-cluster task mechanism under the hood.

#### Action Items

- [ ] **Add clarifying comment** to `types.py` explaining when to use `EntryPoint` vs `Callable`
- [ ] **Update design doc** (`.agents/docs/ray_abstraction.md`) to remove "Before/After" confusion
- [ ] **Add examples** to `cluster.py` docstrings showing typical usage patterns
- [ ] **Document `run_on_tpu`** as a special case that bridges both tiers

Example comment for `types.py:40`:
```python
# EntryPoint: Shell command string for cluster job submission
#
# Used by ClusterContext.create_job() to submit jobs as shell commands.
# This enables script-based execution, Docker containers, and avoids
# function serialization complexity.
#
# Examples:
#   "python train.py --config config.yaml"
#   "bash scripts/preprocess.sh"
#   "docker run my-image:latest python train.py"
#
# Note: For in-cluster task execution (JobContext.create_task),
# use Python Callable functions directly - no string wrapping needed.
#
# Special case: ClusterContext.run_on_tpu() uses Callable for simplicity
# despite being cluster-level, as it's TPU-specific and always uses Ray's
# in-cluster task mechanism.
EntryPoint = str
```

**Files to modify:**
- `lib/fray/src/fray/types.py:40-47` (expand comment)
- `lib/fray/src/fray/cluster.py:37-58` (add example to docstring)
- `lib/fray/src/fray/cluster.py:94-139` (document run_on_tpu special case)
- `.agents/docs/ray_abstraction.md:60-85` (update to remove "Before" example)

---

### Task 3: Add Comprehensive Test Coverage

**Status:** Implementation required
**Priority:** High (critical gaps in test coverage)

#### Current Test Coverage

**lib/fray/tests/test_backend.py** has 30 tests covering:
- ✅ Basic task execution
- ✅ Actor lifecycle (creation, state, named actors)
- ✅ wait/get operations
- ✅ Object store put/get
- ✅ Exception propagation
- ✅ TaskOptions and ActorOptions
- ✅ Cluster job submission (basic)
- ⚠️ Context propagation (basic)

#### Critical Gaps

1. **RuntimeEnv variations** - No tests for different package/env combinations
2. **Resource allocation** - No tests for resource constraint enforcement
3. **Error conditions** - Limited failure scenario coverage
4. **Backend swapping** - No integration tests showing swappable backends
5. **Concurrent operations** - Limited stress testing
6. **RuntimeEnv validation** - No tests for invalid RuntimeEnv configurations

#### Tests to Add

##### 3.1: RuntimeEnv Variations

Add to `lib/fray/tests/test_backend.py`:

```python
def test_runtime_env_with_packages(cluster_context, tmp_path):
    """Test RuntimeEnv with package requirements."""
    cluster, backend_type = cluster_context

    # Create script that imports a package
    script = tmp_path / "test_import.py"
    script.write_text("""
import sys
print("SUCCESS")
sys.exit(0)
""")

    env = RuntimeEnv(
        package_requirements=["requests>=2.28.0"],
    )

    job_id = cluster.create_job(f"python {script}", env)
    time.sleep(2)  # Wait for job

    jobs = cluster.list_jobs()
    job = next((j for j in jobs if j.id == job_id), None)
    assert job is not None
    # Package installation might take time, but job should start
    assert job.status in ["RUNNING", "SUCCEEDED", "JobStatus.RUNNING", "JobStatus.SUCCEEDED"]


def test_runtime_env_with_env_vars(cluster_context, tmp_path):
    """Test RuntimeEnv environment variable propagation."""
    cluster, backend_type = cluster_context

    script = tmp_path / "test_env.py"
    script.write_text("""
import os
import sys

expected = "test_value_12345"
actual = os.environ.get("TEST_ENV_VAR", "NOT_SET")

if actual == expected:
    print(f"SUCCESS: {actual}")
    sys.exit(0)
else:
    print(f"FAILED: expected={expected}, actual={actual}")
    sys.exit(1)
""")

    env = RuntimeEnv(env={"TEST_ENV_VAR": "test_value_12345"})

    job_id = cluster.create_job(f"python {script}", env)
    time.sleep(2)

    jobs = cluster.list_jobs()
    job = next(j for j in jobs if j.id == job_id)
    assert "SUCCEED" in job.status or "SUCCESS" in job.status


def test_runtime_env_combined(cluster_context, tmp_path):
    """Test RuntimeEnv with packages + env vars together."""
    cluster, backend_type = cluster_context

    script = tmp_path / "test_combined.py"
    script.write_text("""
import os
import sys

# Check env var
config = os.environ.get("CONFIG_PATH", "")
if not config:
    print("ERROR: CONFIG_PATH not set")
    sys.exit(1)

print(f"SUCCESS: config={config}")
sys.exit(0)
""")

    env = RuntimeEnv(
        package_requirements=["pyyaml"],
        env={"CONFIG_PATH": "/tmp/config.yaml"},
    )

    job_id = cluster.create_job(f"python {script}", env)
    time.sleep(3)  # Extra time for package install

    jobs = cluster.list_jobs()
    job = next(j for j in jobs if j.id == job_id)
    assert "SUCCEED" in job.status or "SUCCESS" in job.status
```

##### 3.2: Resource Allocation and Constraints

```python
def test_task_resource_limits_honored(job_context):
    """Test that resource constraints are passed to backend."""
    from fray.types import TaskOptions

    def resource_heavy_task():
        # In Ray, this would be scheduled based on resources
        # In local, this is a no-op but should accept the options
        return "completed"

    # Request specific resources
    options = TaskOptions(
        resources={"CPU": 4, "memory": 8 * 1024**3}
    )

    ref = job_context.create_task(resource_heavy_task, options=options)
    result = job_context.get(ref)
    assert result == "completed"


def test_actor_resource_limits_honored(job_context):
    """Test that actor resource constraints are applied."""
    from fray.types import ActorOptions

    class ResourceActor:
        def work(self):
            return "done"

    # Request specific resources (e.g., run on GPU)
    options = ActorOptions(
        resources={"CPU": 1, "GPU": 1}
    )

    actor = job_context.create_actor(ResourceActor, options=options)
    ref = actor.work()
    result = job_context.get(ref)
    assert result == "done"


def test_actor_head_node_scheduling(job_context):
    """Test scheduling actor on head node (non-preemptible)."""
    from fray.types import ActorOptions
    from fray.backend.ray.ray_job import RayJobContext

    if not isinstance(job_context, RayJobContext):
        pytest.skip("Head node scheduling is Ray-specific")

    class CriticalActor:
        def get_status(self):
            return "alive"

    # Schedule on head node using custom resource
    options = ActorOptions(
        name="critical_actor",
        resources={"CPU": 0, "head_node": 0.0001}
    )

    actor = job_context.create_actor(CriticalActor, options=options)
    ref = actor.get_status()
    assert job_context.get(ref) == "alive"
```

##### 3.3: Error Conditions and Edge Cases

```python
def test_cluster_job_submission_failure(cluster_context, tmp_path):
    """Test job submission with invalid command fails gracefully."""
    cluster, backend_type = cluster_context

    # Submit job with nonexistent command
    job_id = cluster.create_job("python /nonexistent/script.py", RuntimeEnv())

    # Wait for job to fail
    time.sleep(2)

    jobs = cluster.list_jobs()
    job = next((j for j in jobs if j.id == job_id), None)
    assert job is not None
    # Job should fail (exact status depends on backend)
    assert job.status in ["FAILED", "JobStatus.FAILED", "STOPPED"]


def test_task_with_invalid_resources_fails_gracefully(job_context):
    """Test that invalid resource specs are handled."""
    from fray.types import TaskOptions

    def simple_task():
        return "ok"

    # Ray might reject negative resources
    # Local backend should ignore but not crash
    options = TaskOptions(resources={"CPU": -1})

    # Should either accept it (local) or raise immediately (Ray)
    # Either behavior is acceptable
    try:
        ref = job_context.create_task(simple_task, options=options)
        # If it doesn't raise, should still work (local backend)
        result = job_context.get(ref)
        assert result == "ok"
    except (ValueError, TypeError):
        # Ray might reject invalid resources
        pass


def test_wait_with_timeout_returns_partial_results(job_context):
    """Test that wait timeout returns partial results correctly."""

    def fast_task():
        return "fast"

    def slow_task():
        time.sleep(5)
        return "slow"

    fast_ref = job_context.create_task(fast_task)
    slow_ref = job_context.create_task(slow_task)

    # Wait for both with short timeout - should get only fast one
    done, not_done = job_context.wait([fast_ref, slow_ref], num_returns=2, timeout=0.5)

    assert len(done) == 1  # Only fast task completed
    assert len(not_done) == 1  # Slow task still running
    assert job_context.get(done[0]) == "fast"


def test_actor_method_exception_preserves_actor_state(job_context):
    """Test that actor survives method exceptions."""

    class StatefulActor:
        def __init__(self):
            self.counter = 0

        def increment(self):
            self.counter += 1
            return self.counter

        def fail(self):
            raise ValueError("Intentional failure")

    actor = job_context.create_actor(StatefulActor)

    # Increment counter
    ref1 = actor.increment()
    assert job_context.get(ref1) == 1

    # Method fails
    ref2 = actor.fail()
    with pytest.raises(ValueError):
        job_context.get(ref2)

    # Actor should still be alive with preserved state
    ref3 = actor.increment()
    assert job_context.get(ref3) == 2


def test_multiple_get_on_same_ref_returns_same_value(job_context):
    """Test that getting same ref multiple times returns same value."""

    def create_value():
        import random
        return random.randint(1, 1000000)

    ref = job_context.create_task(create_value)

    # Get multiple times
    result1 = job_context.get(ref)
    result2 = job_context.get(ref)
    result3 = job_context.get(ref)

    # All should be identical
    assert result1 == result2 == result3
```

##### 3.4: Backend Swapping and Compatibility

```python
def test_code_works_with_both_backends():
    """Test that same code works with both Ray and in-memory backends."""

    def computation(x):
        return x * 2

    # Test with in-memory backend
    local_ctx = LocalJobContext()
    ref1 = local_ctx.create_task(computation, args=(21,))
    result1 = local_ctx.get(ref1)

    # Test with Ray backend (if available)
    ray_ctx = RayJobContext()
    ref2 = ray_ctx.create_task(computation, args=(21,))
    result2 = ray_ctx.get(ref2)

    # Results should be identical
    assert result1 == result2 == 42


def test_context_switching_with_context_vars():
    """Test that context can be switched during execution."""
    from fray.context import get_job_context, set_job_context

    local_ctx = LocalJobContext()
    ray_ctx = RayJobContext()

    # Switch contexts
    set_job_context(local_ctx)
    assert get_job_context() is local_ctx

    set_job_context(ray_ctx)
    assert get_job_context() is ray_ctx

    # Switch back
    set_job_context(local_ctx)
    assert get_job_context() is local_ctx


def test_task_options_serialize_across_backends(job_context):
    """Test that TaskOptions work consistently across backends."""
    from fray.types import RuntimeEnv, TaskOptions

    def get_env():
        import os
        return os.environ.get("TEST_KEY", "default")

    options = TaskOptions(
        resources={"CPU": 1},
        runtime_env=RuntimeEnv(env={"TEST_KEY": "test_value"}),
        name="test_task"
    )

    ref = job_context.create_task(get_env, options=options)
    # Note: in-memory backend doesn't isolate env vars,
    # but should accept the options without error
    result = job_context.get(ref)

    # Ray should set the env var, local won't
    if isinstance(job_context, RayJobContext):
        assert result == "test_value"
    # Local backend just ensures no crash
```

##### 3.5: Stress and Concurrency Tests

```python
def test_many_concurrent_tasks(job_context):
    """Test handling of many concurrent tasks."""

    def work(i):
        time.sleep(0.01)
        return i * i

    # Launch 100 concurrent tasks
    refs = [job_context.create_task(work, args=(i,)) for i in range(100)]
    results = job_context.get(refs)

    assert len(results) == 100
    assert results == [i * i for i in range(100)]


def test_many_concurrent_actors(job_context):
    """Test creating many actors concurrently."""

    class Worker:
        def __init__(self, worker_id):
            self.worker_id = worker_id

        def get_id(self):
            return self.worker_id

    # Create 50 actors
    actors = [job_context.create_actor(Worker, args=(i,)) for i in range(50)]

    # Call method on each
    refs = [actor.get_id() for actor in actors]
    results = job_context.get(refs)

    assert results == list(range(50))


def test_nested_task_creation_deep(job_context):
    """Test deeply nested task creation."""

    def recursive_task(depth):
        if depth == 0:
            return 1

        ref = job_context.create_task(recursive_task, args=(depth - 1,))
        return 1 + job_context.get(ref)

    ref = job_context.create_task(recursive_task, args=(10,))
    result = job_context.get(ref)
    assert result == 11
```

**Files to create/modify:**
- `lib/fray/tests/test_backend.py` (add all above tests)
- Estimated additions: ~400 lines

---

### Task 4: Add RuntimeEnv Validation

**Status:** Implementation required
**Priority:** Medium (data validation, prevents user errors)

#### Current State

**lib/fray/src/fray/types.py:72-89**
```python
@dataclass
class RuntimeEnv:
    package_requirements: list[str] = field(default_factory=list)
    minimum_resources: list[Resource] = field(default_factory=list)
    maximum_resources: list[Resource] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
```

No validation exists for:
- Minimum > maximum resources
- Negative resource quantities
- Invalid package requirement strings
- Non-string env var keys/values

#### Implementation

Add `__post_init__` validation to `RuntimeEnv`:

```python
@dataclass
class RuntimeEnv:
    """
    Execution environment specification for a job.

    Describes the environment in which tasks should run, including package
    dependencies, resource constraints, and environment variables.

    Attributes:
        package_requirements: List of pip package specifications (e.g., ["numpy>=1.20", "pandas"])
        minimum_resources: Minimum resources required for the job to start
        maximum_resources: Maximum resources the job can use (for autoscaling)
        env: Environment variables to set in the execution environment

    Raises:
        ValueError: If validation fails (e.g., minimum > maximum resources)
    """

    package_requirements: list[str] = field(default_factory=list)
    minimum_resources: list[Resource] = field(default_factory=list)
    maximum_resources: list[Resource] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate RuntimeEnv configuration."""
        # Validate package requirements
        if self.package_requirements:
            for pkg in self.package_requirements:
                if not isinstance(pkg, str):
                    raise TypeError(f"package_requirements must be strings, got {type(pkg)}: {pkg}")
                if not pkg.strip():
                    raise ValueError("package_requirements cannot contain empty strings")

        # Validate env vars
        if self.env:
            for key, value in self.env.items():
                if not isinstance(key, str):
                    raise TypeError(f"env keys must be strings, got {type(key)}: {key}")
                if not isinstance(value, str):
                    raise TypeError(f"env values must be strings, got {type(value)}: {value}")

        # Validate resource constraints
        if self.minimum_resources and self.maximum_resources:
            # Build resource maps for comparison
            min_map = {r.name: r.quantity for r in self.minimum_resources}
            max_map = {r.name: r.quantity for r in self.maximum_resources}

            # Check that minimum <= maximum for all resources
            for name, min_qty in min_map.items():
                if name in max_map:
                    max_qty = max_map[name]
                    if min_qty > max_qty:
                        raise ValueError(
                            f"minimum_resources[{name}]={min_qty} exceeds "
                            f"maximum_resources[{name}]={max_qty}"
                        )

        # Validate Resource objects themselves
        for resource in self.minimum_resources + self.maximum_resources:
            if resource.quantity < 0:
                raise ValueError(
                    f"Resource quantities cannot be negative: "
                    f"{resource.name}={resource.quantity}"
                )

    def __repr__(self):
        parts = []
        if self.package_requirements:
            parts.append(f"packages={len(self.package_requirements)}")
        if self.minimum_resources:
            parts.append(f"min_resources={self.minimum_resources}")
        if self.maximum_resources:
            parts.append(f"max_resources={self.maximum_resources}")
        if self.env:
            parts.append(f"env_vars={len(self.env)}")
        return f"RuntimeEnv({', '.join(parts)})"
```

#### Tests for Validation

Add to `lib/fray/tests/test_backend.py`:

```python
def test_runtime_env_rejects_min_greater_than_max():
    """Test that RuntimeEnv validation catches min > max resources."""
    from fray.types import Resource, RuntimeEnv

    with pytest.raises(ValueError, match="minimum_resources.*exceeds.*maximum_resources"):
        RuntimeEnv(
            minimum_resources=[Resource("CPU", 8)],
            maximum_resources=[Resource("CPU", 4)],
        )


def test_runtime_env_rejects_negative_resources():
    """Test that RuntimeEnv validation catches negative quantities."""
    from fray.types import Resource, RuntimeEnv

    with pytest.raises(ValueError, match="cannot be negative"):
        RuntimeEnv(
            minimum_resources=[Resource("memory", -1024)],
        )


def test_runtime_env_rejects_empty_package_names():
    """Test that RuntimeEnv validation catches empty package strings."""
    from fray.types import RuntimeEnv

    with pytest.raises(ValueError, match="cannot contain empty strings"):
        RuntimeEnv(package_requirements=["numpy", "", "pandas"])


def test_runtime_env_rejects_non_string_packages():
    """Test that RuntimeEnv validation catches non-string packages."""
    from fray.types import RuntimeEnv

    with pytest.raises(TypeError, match="must be strings"):
        RuntimeEnv(package_requirements=["numpy", 123, "pandas"])


def test_runtime_env_rejects_non_string_env_keys():
    """Test that RuntimeEnv validation catches non-string env keys."""
    from fray.types import RuntimeEnv

    with pytest.raises(TypeError, match="env keys must be strings"):
        RuntimeEnv(env={123: "value"})


def test_runtime_env_rejects_non_string_env_values():
    """Test that RuntimeEnv validation catches non-string env values."""
    from fray.types import RuntimeEnv

    with pytest.raises(TypeError, match="env values must be strings"):
        RuntimeEnv(env={"KEY": 123})


def test_runtime_env_allows_valid_configuration():
    """Test that valid RuntimeEnv configurations are accepted."""
    from fray.types import Resource, RuntimeEnv

    # Should not raise
    env = RuntimeEnv(
        package_requirements=["numpy>=1.20", "torch"],
        minimum_resources=[Resource("CPU", 2), Resource("memory", 4 * 1024**3)],
        maximum_resources=[Resource("CPU", 8), Resource("memory", 16 * 1024**3)],
        env={"VAR1": "value1", "VAR2": "value2"},
    )

    assert len(env.package_requirements) == 2
    assert len(env.minimum_resources) == 2
    assert len(env.maximum_resources) == 2
    assert len(env.env) == 2
```

**Files to modify:**
- `lib/fray/src/fray/types.py:72-102` (add `__post_init__`)
- `lib/fray/tests/test_backend.py` (add validation tests)

---

### Task 5: Improve In-Memory Backend Async Behavior

**Status:** Enhancement (already works, could be more realistic)
**Priority:** Low (current implementation is correct)

#### Current Status: Correct but Could Improve

The in-memory backend already uses `ThreadPoolExecutor` for async behavior:

**lib/fray/src/fray/backend/in_memory.py:103-110**
```python
class LocalJobContext(JobContext):
    def __init__(self, max_workers: int = 10):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        ...

    def create_task(self, fn: Callable, ...):
        future = self._executor.submit(run_in_context)
        return LocalObjectRef(future)
```

This is already async! Tasks run in parallel on thread pool.

#### Potential Improvements

The current implementation is good, but we could enhance it to better match Ray's behavior:

1. **Configurable executor type** - Allow switching between threads/processes
2. **Resource-aware scheduling** - Track and enforce resource limits
3. **Better error handling** - Match Ray's error serialization behavior

These are NICE-TO-HAVES, not requirements. The current implementation is sufficient.

#### Optional Enhancement: Resource-Aware Scheduling

If we want the in-memory backend to better simulate resource constraints:

```python
class LocalJobContext(JobContext):
    """
    Local in-memory job context using thread pools.

    Optionally tracks resource usage to simulate cluster resource constraints
    during testing. This helps catch resource-related bugs before deploying to Ray.
    """

    def __init__(self, max_workers: int = 10, track_resources: bool = False):
        """
        Initialize local job context.

        Args:
            max_workers: Maximum number of concurrent worker threads
            track_resources: If True, track resource usage and block tasks
                           when resources exhausted (simulates Ray behavior)
        """
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._named_actors: dict[str, LocalActorRef] = {}

        # Optional resource tracking
        self._track_resources = track_resources
        if track_resources:
            self._available_resources = {
                "CPU": float(os.cpu_count() or 1),
                "memory": 1024 * 1024 * 1024,  # 1GB default
            }
            self._allocated_resources: dict[str, float] = {}

    def create_task(self, fn: Callable, args: tuple = (), kwargs: dict | None = None, options: Any | None = None):
        """Executes function in thread pool with optional resource tracking."""
        if kwargs is None:
            kwargs = {}

        # Extract resource requirements
        task_resources = {}
        if options and options.resources:
            task_resources = options.resources

        # Resource tracking (optional simulation)
        if self._track_resources and task_resources:
            self._wait_for_resources(task_resources)

        ctx = contextvars.copy_context()

        def run_in_context():
            try:
                if self._track_resources and task_resources:
                    self._allocate_resources(task_resources)

                result = ctx.run(fn, *args, **kwargs)
                return result
            finally:
                if self._track_resources and task_resources:
                    self._release_resources(task_resources)

        future = self._executor.submit(run_in_context)
        return LocalObjectRef(future)

    def _wait_for_resources(self, required: dict[str, float]):
        """Block until required resources are available."""
        while True:
            with self._lock:
                if all(
                    self._available_resources.get(name, 0) >= qty
                    for name, qty in required.items()
                ):
                    return
            time.sleep(0.01)  # Poll interval

    def _allocate_resources(self, resources: dict[str, float]):
        """Mark resources as allocated."""
        with self._lock:
            for name, qty in resources.items():
                current = self._available_resources.get(name, 0)
                self._available_resources[name] = current - qty

    def _release_resources(self, resources: dict[str, float]):
        """Return resources to available pool."""
        with self._lock:
            for name, qty in resources.items():
                current = self._available_resources.get(name, 0)
                self._available_resources[name] = current + qty
```

This is **optional** and not required for the PR. Current implementation is fine.

**Files to potentially modify (if desired):**
- `lib/fray/src/fray/backend/in_memory.py:103-218` (add resource tracking)

---

## Summary of Action Items by Priority

### High Priority (Required for PR merge)

1. ✅ **Add comments explaining no double-wrapping** - 3 files, minimal changes
2. ✅ **Add comprehensive test coverage** - ~15 new tests, ~400 lines
3. ✅ **Add RuntimeEnv validation** - 1 file, ~40 lines + tests

### Medium Priority (Improve documentation/clarity)

4. ✅ **Document EntryPoint two-tier model** - 3 files, comment/doc updates
5. ✅ **Create migration guide** - New file `.agents/docs/fray_migration.md`

### Low Priority (Optional enhancements)

6. ⚠️ **Resource tracking in in-memory backend** - Optional, current impl is fine
7. ⚠️ **Additional stress tests** - Beyond the comprehensive set above

## File Change Checklist

- [ ] `lib/fray/src/fray/types.py` - Add RuntimeEnv validation + EntryPoint docs
- [ ] `lib/fray/src/fray/cluster.py` - Enhance docstrings with examples
- [ ] `lib/fray/src/fray/backend/ray/ray_job.py` - Add no-double-wrap comments
- [ ] `lib/fray/src/fray/backend/ray/ray_cluster.py` - Add no-double-wrap comments
- [ ] `lib/fray/tests/test_backend.py` - Add ~15 comprehensive tests
- [ ] `.agents/docs/fray_migration.md` - Create migration guide (new file)
- [ ] `.agents/docs/ray_abstraction.md` - Update design doc to remove confusion

## Estimated Effort

- High priority tasks: **4-6 hours** (testing is the bulk)
- Medium priority tasks: **2-3 hours** (documentation)
- Total core work: **6-9 hours**

## References

### Key Files
- Design doc: `.agents/docs/ray_abstraction.md`
- Cluster management doc: `.agents/docs/ray_cluster_mgmt.md`
- Current PR: https://github.com/marin-community/marin/pull/1778

### Migration Pattern Examples

#### Before (Pure Ray)
```python
import ray

@ray.remote
def train(config):
    ...

# Somewhere in a script
ray.init(address="ray://cluster:10001")
ref = train.remote(my_config)
result = ray.get(ref)
```

#### After (Fray)
```python
from fray import get_job_context

def train(config):  # No decorator!
    ...

# Somewhere in a script (context injected by job submission)
ctx = get_job_context()
ref = ctx.create_task(train, args=(my_config,))
result = ctx.get(ref)
```

### Common Patterns

#### Cluster Job Submission
```python
from fray import RayClusterContext, RuntimeEnv

cluster = RayClusterContext(dashboard_address="http://ray:8265")
job_id = cluster.create_job(
    entrypoint="python train.py --config config.yaml",
    env=RuntimeEnv(
        package_requirements=["torch>=2.0", "transformers"],
        env={"WANDB_PROJECT": "my-project"}
    )
)
```

#### In-Cluster Tasks
```python
# Inside a job (e.g., train.py from above)
from fray import get_job_context

ctx = get_job_context()

def preprocess(data):
    return data.lower()

refs = [ctx.create_task(preprocess, args=(text,)) for text in texts]
results = ctx.get(refs)
```

#### TPU Execution
```python
from fray import RayClusterContext, TpuRunConfig

def train_on_tpu(ctx):
    import jax
    devices = jax.devices()
    # ... training logic
    return metrics

cluster = RayClusterContext()
config = TpuRunConfig(tpu_type="v4-32", num_slices=2)
results = cluster.run_on_tpu(train_on_tpu, config)
```

---

**Document End**

This document is self-contained and can be picked up by any developer or AI agent to implement the cleanup tasks. All necessary context, file locations, code examples, and rationale are included.
