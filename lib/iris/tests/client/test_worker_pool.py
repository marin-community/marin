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

Dispatcher tests use real ActorServers but bypass job infrastructure for fast testing.
E2E tests use LocalClusterClient for full job submission flow.
"""

import time
from concurrent.futures import Future
from queue import Queue

import cloudpickle
import pytest

from iris.actor import ActorServer
from iris.actor.resolver import FixedResolver
from iris.client import IrisClient
from iris.client.worker_pool import (
    PendingTask,
    TaskExecutorActor,
    WorkerDispatcher,
    WorkerPool,
    WorkerPoolConfig,
    WorkerState,
    WorkerStatus,
)
from iris.cluster.types import ResourceSpec

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
    server.shutdown()


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
    server = ActorServer(host="127.0.0.1", port=0)
    actor_name_good = "_test_worker_good"
    server.register(actor_name_good, TaskExecutorActor())
    port = server.serve_background()
    good_url = f"http://127.0.0.1:{port}"

    try:
        actor_name_bad = "_test_worker_bad"
        worker_state_1 = WorkerState(
            worker_id="w-0",
            worker_name=actor_name_bad,
            endpoint_url="http://127.0.0.1:9999",
            status=WorkerStatus.IDLE,
        )

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

        dispatcher_1 = WorkerDispatcher(
            state=worker_state_1,
            task_queue=task_queue,
            resolver=resolver,
            timeout=1.0,
        )
        dispatcher_2 = WorkerDispatcher(
            state=worker_state_2,
            task_queue=task_queue,
            resolver=resolver,
            timeout=5.0,
        )

        dispatcher_1.start()
        dispatcher_2.start()

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

        result = future.result(timeout=10.0)
        assert result == "success"

        dispatcher_1.stop()
        dispatcher_2.stop()
        dispatcher_1.join(timeout=1.0)
        dispatcher_2.join(timeout=1.0)

        assert worker_state_1.status == WorkerStatus.FAILED
        assert worker_state_2.tasks_completed == 1
    finally:
        server.shutdown()


# =============================================================================
# E2E tests for WorkerPool using LocalClusterClient
# =============================================================================


@pytest.fixture(scope="module")
def local_client():
    """Create a LocalClusterClient-backed IrisClient for true E2E testing.

    This fixture starts a real Controller and Worker with in-process execution,
    ensuring WorkerPool tests go through the full job submission infrastructure.
    """
    client = IrisClient.local()
    yield client
    client.shutdown(wait=False)


class TestWorkerPoolE2E:
    """True end-to-end tests for WorkerPool using LocalClusterClient.

    These tests exercise the full job submission flow:
    WorkerPool -> IrisClient -> LocalClusterClient -> Controller -> Worker -> task execution.
    """

    def test_submit_executes_task(self, local_client):
        """submit() dispatches a task through real job infrastructure and returns correct result."""
        config = WorkerPoolConfig(
            num_workers=1,
            resources=ResourceSpec(cpu=1, memory="512m"),
        )

        with WorkerPool(local_client, config, timeout=30.0) as pool:

            def add(a, b):
                return a + b

            future = pool.submit(add, 10, 20)
            result = future.result(timeout=60.0)

            assert result == 30

    def test_map_executes_tasks(self, local_client):
        """map() distributes work through real job infrastructure."""
        config = WorkerPoolConfig(
            num_workers=2,
            resources=ResourceSpec(cpu=1, memory="512m"),
        )

        with WorkerPool(local_client, config, timeout=30.0) as pool:

            def square(x):
                return x * x

            futures = pool.map(square, [1, 2, 3, 4, 5])
            results = [f.result(timeout=60.0) for f in futures]

            assert results == [1, 4, 9, 16, 25]

    def test_exception_propagates_to_caller(self, local_client):
        """Exceptions raised by user code propagate through job infrastructure to caller."""
        config = WorkerPoolConfig(
            num_workers=1,
            resources=ResourceSpec(cpu=1, memory="512m"),
        )

        with WorkerPool(local_client, config, timeout=30.0) as pool:

            def fail():
                raise ValueError("intentional error")

            future = pool.submit(fail)

            with pytest.raises(ValueError, match="intentional error"):
                future.result(timeout=60.0)

    def test_shutdown_prevents_new_submissions(self, local_client):
        """After shutdown, submit() raises RuntimeError."""
        config = WorkerPoolConfig(
            num_workers=1,
            resources=ResourceSpec(cpu=1, memory="512m"),
        )

        pool = WorkerPool(local_client, config, timeout=30.0)
        pool.__enter__()

        pool.shutdown(wait=False)

        with pytest.raises(RuntimeError, match="shutdown"):
            pool.submit(lambda: 42)

    def test_multiple_sequential_tasks(self, local_client):
        """Multiple tasks can be submitted sequentially to the same pool."""
        config = WorkerPoolConfig(
            num_workers=1,
            resources=ResourceSpec(cpu=1, memory="512m"),
        )

        with WorkerPool(local_client, config, timeout=30.0) as pool:
            results = []
            for i in range(3):
                future = pool.submit(lambda x: x * 2, i)
                results.append(future.result(timeout=60.0))

            assert results == [0, 2, 4]
