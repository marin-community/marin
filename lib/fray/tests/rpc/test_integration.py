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

"""End-to-end integration tests for Fray RPC system."""

import threading
import time

import pytest

from fray.job.rpc.context import FrayContext
from fray.job.rpc.controller import FrayControllerServer
from fray.job.rpc.worker import FrayWorker


@pytest.fixture
def controller():
    """Start a controller server on a random port."""
    server = FrayControllerServer(port=0)
    port = server.start()
    yield server, port
    server.stop()
    time.sleep(0.1)  # Brief pause for cleanup


@pytest.fixture
def worker_factory(controller):
    """Factory for creating workers connected to the controller."""
    _, port = controller
    workers = []
    threads = []

    def create_worker():
        worker = FrayWorker(f"http://localhost:{port}", port=0)
        workers.append(worker)

        # Start worker in background thread
        thread = threading.Thread(target=worker.run, daemon=True)
        thread.start()
        threads.append(thread)

        # Give worker time to register and start
        time.sleep(0.2)
        return worker

    yield create_worker

    # Cleanup all workers
    for worker in workers:
        worker.stop()

    # Wait for threads to finish
    for thread in threads:
        thread.join(timeout=2.0)


def test_end_to_end_simple_task(controller, worker_factory):
    """Test complete workflow: submit task → worker executes → get result."""
    _, port = controller
    _worker = worker_factory()

    # Create context and submit task
    ctx = FrayContext(f"http://localhost:{port}")
    future = ctx.run(lambda x: x * 2, 5)

    # Get result
    result = ctx.get(future)
    assert result == 10


def test_end_to_end_with_string_task(controller, worker_factory):
    """Test task that processes strings."""
    _, port = controller
    _worker = worker_factory()

    ctx = FrayContext(f"http://localhost:{port}")
    future = ctx.run(lambda s: s.upper(), "hello world")

    result = ctx.get(future)
    assert result == "HELLO WORLD"


def test_multiple_tasks_sequential(controller, worker_factory):
    """Test submitting and completing multiple tasks sequentially."""
    _, port = controller
    _worker = worker_factory()

    ctx = FrayContext(f"http://localhost:{port}")

    # Submit multiple tasks
    futures = []
    for i in range(5):
        future = ctx.run(lambda x: x * 2, i)
        futures.append(future)

    # Verify all results
    results = [ctx.get(f) for f in futures]
    assert results == [0, 2, 4, 6, 8]


def test_multiple_tasks_concurrent(controller, worker_factory):
    """Test multiple tasks executing concurrently with multiple workers."""
    _, port = controller
    _worker1 = worker_factory()
    _worker2 = worker_factory()

    ctx = FrayContext(f"http://localhost:{port}")

    # Submit multiple tasks
    def slow_task(x):
        time.sleep(0.1)
        return x * 2

    futures = [ctx.run(slow_task, i) for i in range(10)]

    # All tasks should complete successfully
    results = [ctx.get(f) for f in futures]
    assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]


def test_task_with_exception(controller, worker_factory):
    """Test that task exceptions are properly reported back to context."""
    _, port = controller
    _worker = worker_factory()

    ctx = FrayContext(f"http://localhost:{port}")

    def failing_task():
        raise ValueError("Task failed intentionally")

    future = ctx.run(failing_task)

    # Getting the result should re-raise the original exception
    with pytest.raises(ValueError, match="Task failed intentionally"):
        ctx.get(future)


def test_task_with_complex_return_value(controller, worker_factory):
    """Test task returning complex Python objects."""
    _, port = controller
    _worker = worker_factory()

    ctx = FrayContext(f"http://localhost:{port}")

    def complex_task():
        return {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tuple": (4, 5, 6),
        }

    future = ctx.run(complex_task)
    result = ctx.get(future)

    assert result == {
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
        "tuple": (4, 5, 6),
    }


def test_wait_for_futures(controller, worker_factory):
    """Test wait() functionality with multiple futures."""
    _, port = controller
    _worker = worker_factory()

    ctx = FrayContext(f"http://localhost:{port}")

    # Submit multiple tasks - use a function to avoid lambda capture issues
    def double(x):
        return x * 2

    futures = [ctx.run(double, i) for i in range(5)]

    # Wait for all to complete
    ready, pending = ctx.wait(futures, num_returns=5)

    assert len(ready) == 5
    assert len(pending) == 0

    # Verify all are marked as done
    for f in ready:
        assert f.done()


def test_worker_handles_multiple_sequential_tasks(controller, worker_factory):
    """Test single worker processing multiple tasks in sequence."""
    _, port = controller
    _worker = worker_factory()

    ctx = FrayContext(f"http://localhost:{port}")

    # Submit tasks one at a time, ensuring each completes
    for i in range(10):
        future = ctx.run(lambda x: x + 1, i)
        result = ctx.get(future)
        assert result == i + 1


def test_task_with_multiple_arguments(controller, worker_factory):
    """Test task execution with multiple function arguments."""
    _, port = controller
    _worker = worker_factory()

    ctx = FrayContext(f"http://localhost:{port}")

    def add_three(a, b, c):
        return a + b + c

    future = ctx.run(add_three, 1, 2, 3)
    result = ctx.get(future)
    assert result == 6


def test_no_workers_available_timeout(controller):
    """Test that tasks remain pending when no workers are available."""
    _, port = controller

    ctx = FrayContext(f"http://localhost:{port}")
    future = ctx.run(lambda x: x * 2, 5)

    # Without workers, task should remain pending
    # We don't wait indefinitely - just check status isn't completed
    assert not future.done()

    # Try to get result with short timeout - should timeout
    with pytest.raises(TimeoutError):
        future.result(timeout=0.5)
