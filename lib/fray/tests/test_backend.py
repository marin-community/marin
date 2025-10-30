# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for backend implementations (in-memory, Ray, and Monarch)."""

import time

import pytest
from fray.backend.in_memory import LocalClusterContext, LocalJobContext
from fray.backend.ray.ray_cluster import RayClusterContext
from fray.backend.ray.ray_job import RayJobContext
from fray.types import RuntimeEnv

# Conditionally import Monarch backend if available
try:
    from fray.backend.monarch import MONARCH_AVAILABLE, MonarchClusterContext, MonarchJobContext
except ImportError:
    MONARCH_AVAILABLE = False
    MonarchClusterContext = None
    MonarchJobContext = None

# Build list of backends to test based on availability
AVAILABLE_BACKENDS = ["in_memory", "ray"]
if MONARCH_AVAILABLE:
    AVAILABLE_BACKENDS.append("monarch")


@pytest.fixture(params=AVAILABLE_BACKENDS)
def job_context(request, ray_cluster):
    """Fixture that provides all available backend types."""
    backend_type = request.param

    if backend_type == "in_memory":
        return LocalJobContext()
    elif backend_type == "ray":
        # Ray cluster is already initialized by the session fixture
        return RayJobContext()
    elif backend_type == "monarch":
        # Monarch context with minimal resources for testing
        return MonarchJobContext(resource_config={"num_procs": 2})
    else:
        raise ValueError(f"Unknown backend: {backend_type}")


@pytest.fixture(params=AVAILABLE_BACKENDS)
def cluster_context(request, ray_cluster):
    """Fixture that provides cluster context for all available backend types."""
    backend_type = request.param

    if backend_type == "in_memory":
        return LocalClusterContext(), backend_type
    elif backend_type == "ray":
        # Ray cluster is already initialized by the session fixture
        # Get the dashboard URL from the Ray cluster
        dashboard_url = f"http://{ray_cluster.dashboard_url}"
        return RayClusterContext(dashboard_address=dashboard_url), backend_type
    elif backend_type == "monarch":
        return MonarchClusterContext(), backend_type
    else:
        raise ValueError(f"Unknown backend: {backend_type}")


def test_remote_task_execution(job_context):
    """Test basic task execution with create_task."""

    def add(a, b):
        return a + b

    ref = job_context.create_task(add, args=(2, 3))
    result = job_context.get(ref)

    assert result == 5


def test_multiple_tasks_parallel(job_context):
    """Test parallel execution of multiple tasks."""

    def square(x):
        time.sleep(0.01)  # Simulate work
        return x * x

    refs = [job_context.create_task(square, args=(i,)) for i in range(10)]
    results = job_context.get(refs)

    assert results == [i * i for i in range(10)]


def test_wait_functionality(job_context):
    """Test wait with num_returns."""

    def slow_task():
        time.sleep(0.1)
        return "done"

    refs = [job_context.create_task(slow_task) for _ in range(3)]
    done, not_done = job_context.wait(refs, num_returns=1, timeout=0.2)

    assert len(done) >= 1
    # Verify we can get results from done refs
    for ref in done:
        assert job_context.get(ref) == "done"


def test_wait_timeout(job_context):
    """Test wait with timeout that expires."""

    def very_slow_task():
        time.sleep(10)
        return "done"

    refs = [job_context.create_task(very_slow_task) for _ in range(2)]
    done, not_done = job_context.wait(refs, num_returns=2, timeout=0.1)

    # Should timeout before any tasks complete
    assert len(done) < 2
    assert len(not_done) > 0


def test_object_store_put_get(job_context):
    """Test put/get for object storage."""
    data = {"key": "value", "numbers": [1, 2, 3]}
    ref = job_context.put(data)
    retrieved = job_context.get(ref)

    assert retrieved == data
    # Note: Ray may return the same object, in-memory returns a copy
    # So we don't test object identity here


def test_actor_with_complex_state(job_context):
    """Test actor with more complex internal state."""

    class DataStore:
        def __init__(self):
            self.data = {}

        def put(self, key, value):
            self.data[key] = value
            return True

        def get(self, key):
            return self.data.get(key)

        def size(self):
            return len(self.data)

    actor = job_context.create_actor(DataStore)

    # Store multiple items
    job_context.get(actor.put("a", 1))
    job_context.get(actor.put("b", 2))
    job_context.get(actor.put("c", 3))

    # Retrieve
    assert job_context.get(actor.get("a")) == 1
    assert job_context.get(actor.get("b")) == 2
    assert job_context.get(actor.size()) == 3


def test_cluster_multiple_jobs(cluster_context, tmp_path):
    """Test cluster can manage multiple jobs."""
    cluster, backend_type = cluster_context

    # Create a simple test script
    script_path = tmp_path / "simple_job.py"
    script_path.write_text(
        """
import sys
print("Job executed successfully")
sys.exit(0)
"""
    )

    job_id1 = cluster.create_job(f"python {script_path}", RuntimeEnv())
    job_id2 = cluster.create_job(f"python {script_path}", RuntimeEnv())

    # Wait for jobs to complete
    time.sleep(2)

    jobs = cluster.list_jobs()
    assert len(jobs) == 2
    assert {job.id for job in jobs} == {job_id1, job_id2}

    # Both jobs should have completed successfully
    for job in jobs:
        assert "SUCCEED" in job.status or "completed" in job.status


def test_cluster_delete_job(cluster_context, tmp_path):
    """Test cluster can delete jobs."""
    cluster, backend_type = cluster_context

    # Create a long-running test script
    script_path = tmp_path / "long_job.py"
    script_path.write_text(
        """
import time
time.sleep(30)
"""
    )

    job_id = cluster.create_job(f"python {script_path}", RuntimeEnv())

    # Wait for job to start
    time.sleep(1)

    jobs = cluster.list_jobs()
    # Find our specific job
    our_job = next((j for j in jobs if j.id == job_id), None)
    assert our_job is not None, f"Job {job_id} not found in {[j.id for j in jobs]}"
    assert our_job.status in ["RUNNING", "PENDING"]

    # Delete should stop it
    cluster.delete_job(job_id)

    # Wait for stop to propagate (Ray is async)
    time.sleep(2)

    # Verify deleted or stopped
    jobs = cluster.list_jobs()
    our_job = next((j for j in jobs if j.id == job_id), None)

    if backend_type == "in_memory":
        # In-memory backend removes the job entirely
        assert our_job is None, f"Job {job_id} should be deleted but found: {our_job}"
    else:
        # Ray backend may mark it as STOPPED/FAILED or remove it
        assert our_job is None or our_job.status in ["STOPPED", "FAILED"]


def test_context_var_integration(job_context):
    """Test that context variables work correctly across threads."""
    from fray.context import get_job_context, set_job_context

    set_job_context(job_context)

    retrieved = get_job_context()
    assert retrieved is job_context

    # Test it works in remote tasks
    def task_using_context():
        task_ctx = get_job_context()
        # Task should see the same context
        assert task_ctx is job_context
        return "success"

    ref = job_context.create_task(task_using_context)
    result = job_context.get(ref)
    assert result == "success"


def test_nested_remote_calls(job_context):
    """Test that tasks can launch other tasks."""

    def outer_task():
        def inner_task(x):
            return x * 2

        ref = job_context.create_task(inner_task, args=(21,))
        return job_context.get(ref)

    ref = job_context.create_task(outer_task)
    result = job_context.get(ref)
    assert result == 42


def test_exception_handling(job_context):
    """Test that exceptions in tasks are properly propagated."""

    def failing_task():
        raise ValueError("Task failed!")

    ref = job_context.create_task(failing_task)

    with pytest.raises(ValueError, match="Task failed!"):
        job_context.get(ref)


def test_actor_exception_handling(job_context):
    """Test that exceptions in actor methods are properly propagated."""

    class FailingActor:
        def fail(self):
            raise RuntimeError("Actor method failed!")

    actor = job_context.create_actor(FailingActor)
    ref = actor.fail()

    with pytest.raises(RuntimeError, match="Actor method failed!"):
        job_context.get(ref)


def test_empty_wait(job_context):
    """Test wait with empty ref list."""
    done, not_done = job_context.wait([], num_returns=1)

    assert len(done) == 0
    assert len(not_done) == 0


def test_get_single_value(job_context):
    """Test get with a non-ref value passes through."""
    # Getting a plain value should just return it for in-memory backend
    # Ray backend will raise an error for non-ObjectRef values
    # So we skip this assertion for specific values
    if isinstance(job_context, LocalJobContext):
        result = job_context.get(42)
        assert result == 42

        result = job_context.get("hello")
        assert result == "hello"


def test_named_actor_creation(job_context):
    """Test creating a named actor."""
    from fray.types import ActorOptions

    class Counter:
        def __init__(self):
            self.count = 0

        def increment(self):
            self.count += 1
            return self.count

    actor = job_context.create_actor(Counter, options=ActorOptions(name="my_counter"))

    ref = actor.increment()
    assert job_context.get(ref) == 1


def test_named_actor_get_if_exists_returns_same_instance(job_context):
    """Test that get_if_exists returns the same actor instance."""
    from fray.types import ActorOptions

    class Counter:
        def __init__(self):
            self.count = 0

        def increment(self):
            self.count += 1
            return self.count

    # Create first actor
    actor1 = job_context.create_actor(Counter, options=ActorOptions(name="shared_counter", get_if_exists=True))
    ref1 = actor1.increment()
    assert job_context.get(ref1) == 1

    # Get same actor again
    actor2 = job_context.create_actor(Counter, options=ActorOptions(name="shared_counter", get_if_exists=True))
    ref2 = actor2.increment()
    assert job_context.get(ref2) == 2  # State persisted from actor1


def test_named_actor_state_persists(job_context):
    """Test that named actor state persists across get_if_exists calls."""
    from fray.types import ActorOptions

    class DataStore:
        def __init__(self):
            self.data = {}

        def put(self, key, value):
            self.data[key] = value

        def get(self, key):
            return self.data.get(key)

    # Create actor and store data
    actor1 = job_context.create_actor(DataStore, options=ActorOptions(name="store", get_if_exists=True))
    job_context.get(actor1.put("key1", "value1"))

    # Retrieve same actor
    actor2 = job_context.create_actor(DataStore, options=ActorOptions(name="store", get_if_exists=True))
    result = job_context.get(actor2.get("key1"))
    assert result == "value1"


def test_unnamed_actors_independent(job_context):
    """Test that multiple unnamed actors are independent."""

    class Counter:
        def __init__(self):
            self.count = 0

        def increment(self):
            self.count += 1
            return self.count

    # Create two unnamed actors
    actor1 = job_context.create_actor(Counter)
    actor2 = job_context.create_actor(Counter)

    # They should have independent state
    ref1 = actor1.increment()
    ref2 = actor1.increment()
    ref3 = actor2.increment()

    assert job_context.get(ref1) == 1
    assert job_context.get(ref2) == 2
    assert job_context.get(ref3) == 1  # actor2 starts at 0


def test_actor_options_with_resources(job_context):
    """Test that ActorOptions with resources are accepted."""
    from fray.types import ActorOptions

    class SimpleActor:
        def work(self):
            return "done"

    # Resources are accepted (Ray honors them, local ignores them)
    actor = job_context.create_actor(
        SimpleActor,
        options=ActorOptions(resources={"CPU": 0, "GPU": 1}),
    )

    ref = actor.work()
    assert job_context.get(ref) == "done"


def test_actor_with_args_and_kwargs(job_context):
    """Test actor creation with both positional and keyword arguments."""

    class ComplexActor:
        def __init__(self, a, b, c=10, d=20):
            self.sum = a + b + c + d

        def get_sum(self):
            return self.sum

    actor = job_context.create_actor(ComplexActor, args=(1, 2), kwargs={"c": 3, "d": 4})

    ref = actor.get_sum()
    assert job_context.get(ref) == 10  # 1 + 2 + 3 + 4


def test_task_with_kwargs(job_context):
    """Test task with keyword arguments."""

    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}!"

    ref = job_context.create_task(greet, args=("World",), kwargs={"greeting": "Hi"})
    assert job_context.get(ref) == "Hi, World!"


def test_task_with_resource_options(job_context):
    """Test that TaskOptions with resources are accepted."""
    from fray.types import TaskOptions

    def compute(x):
        return x * 2

    # Resources are accepted (Ray honors them, local ignores them)
    options = TaskOptions(resources={"CPU": 2, "memory": 1024 * 1024 * 1024})
    ref = job_context.create_task(compute, args=(21,), options=options)
    assert job_context.get(ref) == 42


def test_task_with_runtime_env_options(job_context):
    """Test TaskOptions with runtime environment."""
    from fray.types import RuntimeEnv, TaskOptions

    def get_env_var():
        import os

        return os.environ.get("TEST_VAR", "not_set")

    options = TaskOptions(runtime_env=RuntimeEnv(env={"TEST_VAR": "test_value"}))

    ref = job_context.create_task(get_env_var, options=options)
    result = job_context.get(ref)

    # In-memory backend doesn't support runtime_env isolation
    # Ray backend will set the env var
    if isinstance(job_context, RayJobContext):
        assert result == "test_value"
    # LocalJobContext ignores runtime_env


def test_task_with_combined_options(job_context):
    """Test TaskOptions with both resources and runtime_env."""
    from fray.types import RuntimeEnv, TaskOptions

    def compute():
        return "done"

    options = TaskOptions(
        resources={"CPU": 1, "GPU": 0},
        runtime_env=RuntimeEnv(package_requirements=["numpy"], env={"WORKER_TYPE": "compute"}),
        name="test_task",
    )

    ref = job_context.create_task(compute, options=options)
    assert job_context.get(ref) == "done"


def test_task_options_none_is_default(job_context):
    """Test that omitting options works (backward compatibility)."""

    def add(a, b):
        return a + b

    # Explicitly passing None should work
    ref = job_context.create_task(add, args=(2, 3), options=None)
    assert job_context.get(ref) == 5

    # Omitting options should work
    ref = job_context.create_task(add, args=(5, 7))
    assert job_context.get(ref) == 12


def test_task_with_max_calls(job_context):
    """Test TaskOptions with max_calls parameter."""
    from fray.types import TaskOptions

    call_count = [0]  # Use list to allow mutation in closure

    def counting_task():
        call_count[0] += 1
        return call_count[0]

    # max_calls=1 forces process restart after each call
    options = TaskOptions(max_calls=1)
    refs = [job_context.create_task(counting_task, options=options) for _ in range(3)]
    results = job_context.get(refs)

    # With max_calls=1, each task runs in fresh process, so all return 1
    # (For in-memory backend, this doesn't apply, but API should accept it)
    if isinstance(job_context, RayJobContext):
        assert all(r == 1 for r in results)


def test_actor_with_detached_lifetime(job_context):
    """Test ActorOptions with DETACHED lifetime."""
    from fray import Lifetime
    from fray.types import ActorOptions

    class PersistentActor:
        def __init__(self):
            self.value = 42

        def get_value(self):
            return self.value

    # Create actor with detached lifetime
    actor = job_context.create_actor(
        PersistentActor,
        options=ActorOptions(
            name="persistent",
            lifetime=Lifetime.DETACHED,
        ),
    )

    ref = actor.get_value()
    assert job_context.get(ref) == 42


def test_actor_default_ephemeral_lifetime(job_context):
    """Test that actors default to EPHEMERAL lifetime."""
    from fray import Lifetime
    from fray.types import ActorOptions

    class SimpleActor:
        def work(self):
            return "done"

    # Default should be EPHEMERAL
    options = ActorOptions()
    assert options.lifetime == Lifetime.EPHEMERAL

    actor = job_context.create_actor(SimpleActor, options=options)
    ref = actor.work()
    assert job_context.get(ref) == "done"


# ============================================================================
# RuntimeEnv Variation Tests
# ============================================================================


def test_runtime_env_with_packages(cluster_context, tmp_path):
    """Test RuntimeEnv with package requirements."""
    cluster, backend_type = cluster_context

    # Create script that imports a package
    script = tmp_path / "test_import.py"
    script.write_text(
        """
import sys
print("SUCCESS")
sys.exit(0)
"""
    )

    env = RuntimeEnv(
        package_requirements=["requests>=2.28.0"],
    )

    job_id = cluster.create_job(f"python {script}", env)
    time.sleep(3)

    jobs = cluster.list_jobs()
    job = next((j for j in jobs if j.id == job_id), None)
    assert job is not None
    # Package installation might take time, accept PENDING/RUNNING/SUCCEEDED
    assert job.status in [
        "PENDING",
        "RUNNING",
        "SUCCEEDED",
        "JobStatus.PENDING",
        "JobStatus.RUNNING",
        "JobStatus.SUCCEEDED",
    ]


def test_runtime_env_with_env_vars(cluster_context, tmp_path):
    """Test RuntimeEnv environment variable propagation."""
    cluster, backend_type = cluster_context

    script = tmp_path / "test_env.py"
    script.write_text(
        """
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
"""
    )

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
    script.write_text(
        """
import os
import sys

# Check env var
config = os.environ.get("CONFIG_PATH", "")
if not config:
    print("ERROR: CONFIG_PATH not set")
    sys.exit(1)

print(f"SUCCESS: config={config}")
sys.exit(0)
"""
    )

    env = RuntimeEnv(
        package_requirements=["pyyaml"],
        env={"CONFIG_PATH": "/tmp/config.yaml"},
    )

    job_id = cluster.create_job(f"python {script}", env)
    time.sleep(4)  # Extra time for package install

    jobs = cluster.list_jobs()
    job = next(j for j in jobs if j.id == job_id)
    # Accept PENDING/RUNNING/SUCCEEDED since package installation takes time
    assert any(status in job.status for status in ["PENDING", "RUNNING", "SUCCEED", "SUCCESS"])


# ============================================================================
# Resource Allocation Tests
# ============================================================================


def test_task_resource_limits_honored(job_context):
    """Test that resource constraints are passed to backend."""
    from fray.types import TaskOptions

    def resource_heavy_task():
        # In Ray, this would be scheduled based on resources
        # In local, this is a no-op but should accept the options
        return "completed"

    # Request specific resources
    options = TaskOptions(resources={"CPU": 4, "memory": 8 * 1024**3})

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
    options = ActorOptions(resources={"CPU": 1, "GPU": 1})

    actor = job_context.create_actor(ResourceActor, options=options)
    ref = actor.work()
    result = job_context.get(ref)
    assert result == "done"


def test_actor_head_node_scheduling(job_context):
    """Test scheduling actor on head node (non-preemptible)."""
    from fray.types import ActorOptions

    if not isinstance(job_context, RayJobContext):
        pytest.skip("Head node scheduling is Ray-specific")

    class CriticalActor:
        def get_status(self):
            return "alive"

    # Schedule on head node using custom resource
    options = ActorOptions(name="critical_actor", resources={"CPU": 0, "head_node": 0.0001})

    actor = job_context.create_actor(CriticalActor, options=options)
    ref = actor.get_status()
    assert job_context.get(ref) == "alive"


# ============================================================================
# Error Condition Tests
# ============================================================================


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

    assert len(done) == 1
    assert len(not_done) == 1
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


# ============================================================================
# Backend Swapping Tests
# ============================================================================


def test_code_works_with_both_backends():
    """Test that same code works with both Ray and in-memory backends."""

    def computation(x):
        return x * 2

    # Test with in-memory backend
    local_ctx = LocalJobContext()
    ref1 = local_ctx.create_task(computation, args=(21,))
    result1 = local_ctx.get(ref1)

    # Test with Ray backend
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
    from fray.types import TaskOptions

    def get_env():
        import os

        return os.environ.get("TEST_KEY", "default")

    options = TaskOptions(resources={"CPU": 1}, runtime_env=RuntimeEnv(env={"TEST_KEY": "test_value"}), name="test_task")

    ref = job_context.create_task(get_env, options=options)
    # Note: in-memory backend doesn't isolate env vars,
    # but should accept the options without error
    result = job_context.get(ref)

    # Ray should set the env var, local won't
    if isinstance(job_context, RayJobContext):
        assert result == "test_value"
    # Local backend just ensures no crash


# ============================================================================
# Stress and Concurrency Tests
# ============================================================================


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
    from fray.context import get_job_context

    def recursive_task(depth):
        if depth == 0:
            return 1

        # Use get_job_context() inside the task to avoid capturing outer scope
        ctx = get_job_context()
        ref = ctx.create_task(recursive_task, args=(depth - 1,))
        return 1 + ctx.get(ref)

    # Set context for nested tasks
    from fray.context import set_job_context

    set_job_context(job_context)

    ref = job_context.create_task(recursive_task, args=(5,))  # Reduce depth to avoid timeout
    result = job_context.get(ref)
    assert result == 6


# ============================================================================
# RuntimeEnv Validation Tests
# ============================================================================


def test_runtime_env_rejects_min_greater_than_max():
    """Test that RuntimeEnv validation catches min > max resources."""
    from fray.types import Resource

    with pytest.raises(ValueError, match="minimum_resources.*exceeds.*maximum_resources"):
        RuntimeEnv(
            minimum_resources=[Resource("CPU", 8)],
            maximum_resources=[Resource("CPU", 4)],
        )


def test_runtime_env_rejects_negative_resources():
    """Test that RuntimeEnv validation catches negative quantities."""
    from fray.types import Resource

    with pytest.raises(ValueError, match="cannot be negative"):
        RuntimeEnv(
            minimum_resources=[Resource("memory", -1024)],
        )


def test_runtime_env_rejects_empty_package_names():
    """Test that RuntimeEnv validation catches empty package strings."""
    with pytest.raises(ValueError, match="cannot contain empty strings"):
        RuntimeEnv(package_requirements=["numpy", "", "pandas"])


def test_runtime_env_rejects_non_string_packages():
    """Test that RuntimeEnv validation catches non-string packages."""
    with pytest.raises(TypeError, match="must be strings"):
        RuntimeEnv(package_requirements=["numpy", 123, "pandas"])


def test_runtime_env_rejects_non_string_env_keys():
    """Test that RuntimeEnv validation catches non-string env keys."""
    with pytest.raises(TypeError, match="env keys must be strings"):
        RuntimeEnv(env={123: "value"})


def test_runtime_env_rejects_non_string_env_values():
    """Test that RuntimeEnv validation catches non-string env values."""
    with pytest.raises(TypeError, match="env values must be strings"):
        RuntimeEnv(env={"KEY": 123})


def test_runtime_env_allows_valid_configuration():
    """Test that valid RuntimeEnv configurations are accepted."""
    from fray.types import Resource

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
