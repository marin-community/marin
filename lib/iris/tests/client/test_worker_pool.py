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

"""Tests for WorkerPool behavior.

These tests exercise the WorkerPool through its public interface, testing
observable behavior rather than implementation details. The test harness
uses real ActorServers with TaskExecutorActor instances, bypassing only
the job-launching infrastructure (ClusterClient).
"""

import time
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue

import cloudpickle
import pytest
from connectrpc.errors import ConnectError

from iris.actor import ActorServer
from iris.actor.resolver import FixedResolver
from iris.client.worker_pool import (
    PendingTask,
    TaskExecutorActor,
    WorkerDispatcher,
    WorkerPool,
    WorkerPoolConfig,
    WorkerState,
    WorkerStatus,
)
from iris.cluster.types import Entrypoint, JobId, create_resource_spec
from iris.rpc import cluster_pb2

# =============================================================================
# Unit tests for TaskExecutorActor
# =============================================================================


def test_execute_basic():
    """Test basic function execution."""
    executor = TaskExecutorActor()

    fn_bytes = cloudpickle.dumps(lambda x, y: x + y)
    args_bytes = cloudpickle.dumps((1, 2))
    kwargs_bytes = cloudpickle.dumps({})

    result = executor.execute(fn_bytes, args_bytes, kwargs_bytes)
    assert result == 3


def test_execute_with_kwargs():
    """Test execution with keyword arguments."""
    executor = TaskExecutorActor()

    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}!"

    fn_bytes = cloudpickle.dumps(greet)
    args_bytes = cloudpickle.dumps(("World",))
    kwargs_bytes = cloudpickle.dumps({"greeting": "Hi"})

    result = executor.execute(fn_bytes, args_bytes, kwargs_bytes)
    assert result == "Hi, World!"


def test_execute_returns_complex_object():
    """Test that complex objects can be returned."""
    executor = TaskExecutorActor()

    def create_dict():
        return {"a": [1, 2, 3], "b": {"nested": True}}

    fn_bytes = cloudpickle.dumps(create_dict)
    args_bytes = cloudpickle.dumps(())
    kwargs_bytes = cloudpickle.dumps({})

    result = executor.execute(fn_bytes, args_bytes, kwargs_bytes)
    assert result == {"a": [1, 2, 3], "b": {"nested": True}}


def test_execute_propagates_exception():
    """Test that exceptions are propagated."""
    executor = TaskExecutorActor()

    def raise_error():
        raise ValueError("test error")

    fn_bytes = cloudpickle.dumps(raise_error)
    args_bytes = cloudpickle.dumps(())
    kwargs_bytes = cloudpickle.dumps({})

    with pytest.raises(ValueError, match="test error"):
        executor.execute(fn_bytes, args_bytes, kwargs_bytes)


def test_execute_with_closure():
    """Test execution of closures that capture variables."""
    executor = TaskExecutorActor()

    multiplier = 10

    def multiply(x):
        return x * multiplier

    fn_bytes = cloudpickle.dumps(multiply)
    args_bytes = cloudpickle.dumps((5,))
    kwargs_bytes = cloudpickle.dumps({})

    result = executor.execute(fn_bytes, args_bytes, kwargs_bytes)
    assert result == 50


# =============================================================================
# WorkerDispatcher behavioral tests
# =============================================================================


@pytest.fixture
def worker_server():
    """Start a real ActorServer with TaskExecutorActor for testing."""
    server = ActorServer(host="127.0.0.1", port=0)
    actor_name = "_test_worker"
    server.register(actor_name, TaskExecutorActor())
    port = server.serve_background()
    yield f"http://127.0.0.1:{port}", actor_name


def test_dispatch_discovers_worker_endpoint(worker_server):
    """Dispatcher transitions worker from PENDING to IDLE when endpoint is discovered."""
    url, actor_name = worker_server

    worker_state = WorkerState(
        worker_id="w-0",
        worker_name=actor_name,
        status=WorkerStatus.PENDING,
    )

    resolver = FixedResolver({actor_name: url})
    task_queue: Queue[PendingTask] = Queue()

    dispatcher = WorkerDispatcher(
        state=worker_state,
        task_queue=task_queue,
        resolver=resolver,
        timeout=5.0,
    )
    dispatcher.start()

    # Wait for discovery
    deadline = time.time() + 2.0
    while worker_state.status == WorkerStatus.PENDING and time.time() < deadline:
        time.sleep(0.05)

    dispatcher.stop()
    dispatcher.join(timeout=1.0)

    assert worker_state.status == WorkerStatus.IDLE
    assert worker_state.endpoint_url == url


def test_dispatch_executes_task_on_worker(worker_server):
    """Dispatcher executes tasks and sets results on futures."""
    url, actor_name = worker_server

    worker_state = WorkerState(
        worker_id="w-0",
        worker_name=actor_name,
        endpoint_url=url,
        status=WorkerStatus.IDLE,
    )

    resolver = FixedResolver({actor_name: url})
    task_queue: Queue[PendingTask] = Queue()

    dispatcher = WorkerDispatcher(
        state=worker_state,
        task_queue=task_queue,
        resolver=resolver,
        timeout=5.0,
    )
    dispatcher.start()

    # Submit a task
    future: Future = Future()
    task = PendingTask(
        task_id="task-1",
        serialized_fn=cloudpickle.dumps(lambda x: x * 2),
        serialized_args=cloudpickle.dumps((21,)),
        serialized_kwargs=cloudpickle.dumps({}),
        future=future,
        fn_name="multiply",
        submitted_at=time.monotonic(),
        retries_remaining=0,
    )
    task_queue.put(task)

    # Wait for result
    result = future.result(timeout=5.0)
    assert result == 42

    dispatcher.stop()
    dispatcher.join(timeout=1.0)

    assert worker_state.tasks_completed == 1


def test_dispatch_propagates_user_exceptions(worker_server):
    """User exceptions are propagated without retry."""
    url, actor_name = worker_server

    worker_state = WorkerState(
        worker_id="w-0",
        worker_name=actor_name,
        endpoint_url=url,
        status=WorkerStatus.IDLE,
    )

    resolver = FixedResolver({actor_name: url})
    task_queue: Queue[PendingTask] = Queue()

    dispatcher = WorkerDispatcher(
        state=worker_state,
        task_queue=task_queue,
        resolver=resolver,
        timeout=5.0,
    )
    dispatcher.start()

    # Submit a task that raises
    def raise_error():
        raise ValueError("test error")

    future: Future = Future()
    task = PendingTask(
        task_id="task-err",
        serialized_fn=cloudpickle.dumps(raise_error),
        serialized_args=cloudpickle.dumps(()),
        serialized_kwargs=cloudpickle.dumps({}),
        future=future,
        fn_name="raise_error",
        submitted_at=time.monotonic(),
        retries_remaining=3,  # Should not retry user exceptions
    )
    task_queue.put(task)

    # Should get the ValueError, not be retried
    with pytest.raises(ValueError, match="test error"):
        future.result(timeout=5.0)

    dispatcher.stop()
    dispatcher.join(timeout=1.0)

    assert worker_state.tasks_failed == 1
    # Worker should still be IDLE (not FAILED) since it was a user exception
    assert worker_state.status == WorkerStatus.IDLE


def test_dispatch_retries_on_infrastructure_failure():
    """Infrastructure failures cause re-queue if retries remain."""
    # Start a real server for the "good" worker
    server = ActorServer(host="127.0.0.1", port=0)
    actor_name_good = "_test_worker_good"
    server.register(actor_name_good, TaskExecutorActor())
    port = server.serve_background()
    good_url = f"http://127.0.0.1:{port}"

    # Worker 1 points to a non-existent endpoint (will fail)
    actor_name_bad = "_test_worker_bad"
    worker_state_1 = WorkerState(
        worker_id="w-0",
        worker_name=actor_name_bad,
        endpoint_url="http://127.0.0.1:9999",
        status=WorkerStatus.IDLE,
    )

    # Worker 2 points to the real server
    worker_state_2 = WorkerState(
        worker_id="w-1",
        worker_name=actor_name_good,
        endpoint_url=good_url,
        status=WorkerStatus.IDLE,
    )

    resolver = FixedResolver(
        {
            actor_name_bad: "http://127.0.0.1:9999",
            actor_name_good: good_url,
        }
    )
    task_queue: Queue[PendingTask] = Queue()

    # Start both dispatchers
    dispatcher_1 = WorkerDispatcher(
        state=worker_state_1,
        task_queue=task_queue,
        resolver=resolver,
        timeout=1.0,  # Short timeout for faster failure
    )
    dispatcher_2 = WorkerDispatcher(
        state=worker_state_2,
        task_queue=task_queue,
        resolver=resolver,
        timeout=5.0,
    )

    dispatcher_1.start()
    dispatcher_2.start()

    # Submit a task with retries
    future: Future = Future()
    task = PendingTask(
        task_id="task-retry",
        serialized_fn=cloudpickle.dumps(lambda: "success"),
        serialized_args=cloudpickle.dumps(()),
        serialized_kwargs=cloudpickle.dumps({}),
        future=future,
        fn_name="success_fn",
        submitted_at=time.monotonic(),
        retries_remaining=2,
    )
    task_queue.put(task)

    # Worker 1 fails, task re-queued, worker 2 succeeds
    result = future.result(timeout=10.0)
    assert result == "success"

    dispatcher_1.stop()
    dispatcher_2.stop()
    dispatcher_1.join(timeout=1.0)
    dispatcher_2.join(timeout=1.0)

    # Worker 1 should be FAILED after the connection error
    assert worker_state_1.status == WorkerStatus.FAILED
    # Worker 2 completed the task
    assert worker_state_2.tasks_completed == 1


# =============================================================================
# E2E tests for WorkerPool
# =============================================================================


@dataclass
class MockJob:
    """Tracks a job submission."""

    job_id: JobId
    name: str
    entrypoint: Entrypoint
    resources: cluster_pb2.ResourceSpec


class MockClusterClient:
    """Mock ClusterClient that tracks submissions without launching jobs.

    For E2E testing, we bypass the actual job infrastructure. Workers are
    started directly as ActorServers and discovered via FixedResolver.
    """

    def __init__(self):
        self._jobs: dict[JobId, MockJob] = {}
        self._job_counter = 0

    def resolver(self):
        # Return a FixedResolver that will be replaced by test harness
        return FixedResolver({})

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: cluster_pb2.ResourceSpec,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
    ) -> JobId:
        job_id = JobId(f"mock-job-{self._job_counter}")
        self._job_counter += 1
        self._jobs[job_id] = MockJob(
            job_id=job_id,
            name=name,
            entrypoint=entrypoint,
            resources=resources,
        )
        return job_id

    def status(self, job_id: JobId) -> cluster_pb2.JobStatus:
        return cluster_pb2.JobStatus(
            job_id=str(job_id),
            name=self._jobs[job_id].name if job_id in self._jobs else "",
            state=cluster_pb2.JOB_STATE_RUNNING,
        )

    def wait(
        self,
        job_id: JobId,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
    ) -> cluster_pb2.JobStatus:
        return self.status(job_id)

    def terminate(self, job_id: JobId) -> None:
        pass

    @property
    def submitted_jobs(self) -> list[MockJob]:
        return list(self._jobs.values())


@dataclass
class WorkerPoolTestHarness:
    """Test harness that manages worker servers and provides a configured pool.

    Starts real ActorServers with TaskExecutorActor instances and configures
    a FixedResolver to discover them. The WorkerPool uses these servers instead
    of launching jobs via ClusterClient.
    """

    pool: WorkerPool
    client: MockClusterClient
    servers: list[ActorServer]
    endpoints: dict[str, str]


@pytest.fixture
def worker_pool_harness():
    """Create a WorkerPool with 2 real worker servers."""
    num_workers = 2

    # Start real ActorServers
    servers = []
    endpoints = {}
    for _ in range(num_workers):
        server = ActorServer(host="127.0.0.1", port=0)
        # Worker names are generated by WorkerPool as _workerpool_{pool_id}:worker-{i}
        # We'll register with a placeholder name and update the resolver after pool creation
        servers.append(server)

    # Create the pool with mock client
    client = MockClusterClient()
    config = WorkerPoolConfig(
        num_workers=num_workers,
        resources=create_resource_spec(cpu=1, memory="512m"),
        max_retries=1,
    )
    pool = WorkerPool(client, config, timeout=5.0)

    # Get the pool_id and set up workers with correct names
    pool_id = pool.pool_id
    for i, server in enumerate(servers):
        worker_name = f"_workerpool_{pool_id}:worker-{i}"
        server.register(worker_name, TaskExecutorActor())
        port = server.serve_background()
        endpoints[worker_name] = f"http://127.0.0.1:{port}"

    # Create resolver and inject it
    resolver = FixedResolver(endpoints)
    pool._resolver = resolver

    yield WorkerPoolTestHarness(
        pool=pool,
        client=client,
        servers=servers,
        endpoints=endpoints,
    )

    # Cleanup
    pool.shutdown(wait=False)


class TestWorkerPoolE2E:
    """End-to-end tests for WorkerPool through its public interface."""

    def test_pool_discovers_workers(self, worker_pool_harness):
        """Workers transition from PENDING to IDLE when discovered."""
        harness = worker_pool_harness
        pool = harness.pool

        # Manually trigger worker launch (normally done by __enter__)
        pool._launch_workers()

        # Wait for workers to be discovered
        pool.wait_for_workers(min_workers=2, timeout=5.0)

        assert pool.size == 2
        assert pool.idle_count == 2

    def test_submit_executes_task(self, worker_pool_harness):
        """submit() dispatches a task and returns correct result."""
        harness = worker_pool_harness
        pool = harness.pool

        pool._launch_workers()
        pool.wait_for_workers(min_workers=1, timeout=5.0)

        def add(a, b):
            return a + b

        future = pool.submit(add, 10, 20)
        result = future.result(timeout=5.0)

        assert result == 30

    def test_submit_with_kwargs(self, worker_pool_harness):
        """submit() passes keyword arguments correctly."""
        harness = worker_pool_harness
        pool = harness.pool

        pool._launch_workers()
        pool.wait_for_workers(min_workers=1, timeout=5.0)

        def greet(name, prefix="Hello"):
            return f"{prefix}, {name}!"

        future = pool.submit(greet, "World", prefix="Hi")
        result = future.result(timeout=5.0)

        assert result == "Hi, World!"

    def test_map_executes_in_parallel(self, worker_pool_harness):
        """map() distributes work across workers."""
        harness = worker_pool_harness
        pool = harness.pool

        pool._launch_workers()
        pool.wait_for_workers(min_workers=2, timeout=5.0)

        def square(x):
            return x * x

        futures = pool.map(square, [1, 2, 3, 4, 5])
        results = [f.result(timeout=5.0) for f in futures]

        assert results == [1, 4, 9, 16, 25]

    def test_exception_propagates_to_caller(self, worker_pool_harness):
        """Exceptions raised by user code propagate to the caller."""
        harness = worker_pool_harness
        pool = harness.pool

        pool._launch_workers()
        pool.wait_for_workers(min_workers=1, timeout=5.0)

        def fail():
            raise ValueError("intentional error")

        future = pool.submit(fail)

        with pytest.raises(ValueError, match="intentional error"):
            future.result(timeout=5.0)

    def test_complex_return_values(self, worker_pool_harness):
        """Complex objects are properly serialized and returned."""
        harness = worker_pool_harness
        pool = harness.pool

        pool._launch_workers()
        pool.wait_for_workers(min_workers=1, timeout=5.0)

        def create_complex():
            return {
                "numbers": [1, 2, 3],
                "nested": {"a": 1, "b": 2},
                "tuple": (1, "two", 3.0),
            }

        future = pool.submit(create_complex)
        result = future.result(timeout=5.0)

        assert result["numbers"] == [1, 2, 3]
        assert result["nested"]["b"] == 2
        assert result["tuple"] == (1, "two", 3.0)

    def test_closures_work(self, worker_pool_harness):
        """Functions that capture variables work correctly."""
        harness = worker_pool_harness
        pool = harness.pool

        pool._launch_workers()
        pool.wait_for_workers(min_workers=1, timeout=5.0)

        multiplier = 7

        def multiply(x):
            return x * multiplier

        future = pool.submit(multiply, 6)
        result = future.result(timeout=5.0)

        assert result == 42

    def test_status_reflects_pool_state(self, worker_pool_harness):
        """status() returns accurate information about pool state."""
        harness = worker_pool_harness
        pool = harness.pool

        pool._launch_workers()
        pool.wait_for_workers(min_workers=2, timeout=5.0)

        status = pool.status()

        assert status.pool_id == pool.pool_id
        assert status.num_workers == 2
        assert status.workers_idle == 2
        assert status.workers_pending == 0
        assert status.tasks_queued == 0

    def test_future_done_and_exception(self, worker_pool_harness):
        """WorkerFuture.done() and exception() work correctly."""
        harness = worker_pool_harness
        pool = harness.pool

        pool._launch_workers()
        pool.wait_for_workers(min_workers=1, timeout=5.0)

        # Test successful completion
        future_success = pool.submit(lambda: 42)
        result = future_success.result(timeout=5.0)
        assert result == 42
        assert future_success.done()
        assert future_success.exception() is None

        # Test exception case
        def fail():
            raise RuntimeError("expected")

        future_fail = pool.submit(fail)
        with pytest.raises(RuntimeError):
            future_fail.result(timeout=5.0)

        assert future_fail.done()
        assert isinstance(future_fail.exception(), RuntimeError)

    def test_context_manager_waits_for_at_least_one_worker(self):
        """__enter__ waits for at least one worker before returning."""
        # Set up a dedicated server for this test
        server = ActorServer(host="127.0.0.1", port=0)
        worker_name = "_workerpool_ctxtest:worker-0"
        server.register(worker_name, TaskExecutorActor())
        port = server.serve_background()

        endpoints = {worker_name: f"http://127.0.0.1:{port}"}

        client = MockClusterClient()
        config = WorkerPoolConfig(
            num_workers=1,
            resources=cluster_pb2.ResourceSpec(cpu=1),
        )
        pool = WorkerPool(client, config, timeout=5.0, resolver=FixedResolver(endpoints))
        pool._pool_id = "ctxtest"

        with pool:
            # By the time __enter__ returns, we should have at least 1 worker
            assert pool.size >= 1

    def test_shutdown_prevents_new_submissions(self, worker_pool_harness):
        """After shutdown, submit() raises RuntimeError."""
        harness = worker_pool_harness
        pool = harness.pool

        pool._launch_workers()
        pool.wait_for_workers(min_workers=1, timeout=5.0)

        pool.shutdown(wait=False)

        with pytest.raises(RuntimeError, match="shutdown"):
            pool.submit(lambda: 42)


class TestWorkerPoolRetry:
    """Tests for retry behavior on infrastructure failures."""

    def test_task_retries_on_worker_failure(self):
        """When a worker fails, the task is re-queued and picked up by another worker."""
        # Set up one bad worker and one good worker
        good_server = ActorServer(host="127.0.0.1", port=0)
        good_server.register("_workerpool_test:worker-1", TaskExecutorActor())
        good_port = good_server.serve_background()

        # Endpoints: worker-0 points to non-existent server, worker-1 is real
        endpoints = {
            "_workerpool_test:worker-0": "http://127.0.0.1:9999",  # Will fail
            "_workerpool_test:worker-1": f"http://127.0.0.1:{good_port}",
        }

        client = MockClusterClient()
        config = WorkerPoolConfig(
            num_workers=2,
            resources=cluster_pb2.ResourceSpec(cpu=1),
            max_retries=2,
        )

        pool = WorkerPool(
            client,
            config,
            timeout=2.0,
            resolver=FixedResolver(endpoints),
        )

        # Override pool_id to match our endpoints
        pool._pool_id = "test"

        pool._launch_workers()
        pool.wait_for_workers(min_workers=1, timeout=5.0)

        # Submit task - may hit failed worker first but should succeed after retry
        future = pool.submit(lambda: "success")
        result = future.result(timeout=10.0)

        assert result == "success"

        pool.shutdown(wait=False)

    def test_task_fails_when_retries_exhausted(self):
        """When all retries are exhausted, the error propagates to caller."""
        # All workers point to non-existent servers
        endpoints = {
            "_workerpool_noretry:worker-0": "http://127.0.0.1:9999",
        }

        client = MockClusterClient()
        config = WorkerPoolConfig(
            num_workers=1,
            resources=cluster_pb2.ResourceSpec(cpu=1),
            max_retries=0,  # No retries
        )

        pool = WorkerPool(
            client,
            config,
            timeout=1.0,
            resolver=FixedResolver(endpoints),
        )

        pool._pool_id = "noretry"
        pool._launch_workers()
        pool.wait_for_workers(min_workers=1, timeout=5.0)

        future = pool.submit(lambda: "should fail")

        # Should fail with connection error from failed RPC
        with pytest.raises(ConnectError):
            future.result(timeout=5.0)

        pool.shutdown(wait=False)
