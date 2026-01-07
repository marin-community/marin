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

"""Tests for FrayWorker RPC implementation."""

import threading
import time

import cloudpickle
import httpx
import pytest

from fray.job.rpc.controller import FrayControllerServer
from fray.job.rpc.proto import fray_pb2
from fray.job.rpc.proto.fray_connect import FrayControllerClientSync, FrayWorkerClientSync
from fray.job.rpc.worker import FrayWorker


@pytest.fixture
def controller():
    """Start a controller server on a random port."""
    server = FrayControllerServer(port=0)
    port = server.start()
    yield server, port
    server.stop()
    time.sleep(0.1)


@pytest.fixture
def worker(controller):
    """Create a worker (not started)."""
    _, port = controller
    worker = FrayWorker(f"http://localhost:{port}", port=0)
    yield worker
    worker.stop()


def test_worker_initialization():
    """Test worker initializes with correct configuration."""
    worker = FrayWorker("localhost:8080", worker_id="test-worker", port=9999)

    assert worker.worker_id == "test-worker"
    assert worker._port == 9999
    assert worker.controller_address == "http://localhost:8080"
    assert not worker._running


def test_worker_auto_generates_id():
    """Test worker auto-generates ID when not provided."""
    worker = FrayWorker("localhost:8080")

    assert worker.worker_id is not None
    assert len(worker.worker_id) > 0


def test_worker_registration(controller):
    """Test worker successfully registers with controller."""
    server, port = controller

    worker = FrayWorker(f"http://localhost:{port}", worker_id="test-worker", port=0)
    worker.register()

    # Verify worker in controller's registry
    assert "test-worker" in server.servicer._workers
    registered_worker = server.servicer._workers["test-worker"]
    assert registered_worker.worker_id == "test-worker"


def test_worker_address_formatting():
    """Test worker address is formatted correctly."""
    worker = FrayWorker("localhost:8080", port=9999)
    assert worker.address == "localhost:9999"


def test_worker_starts_health_check_server(controller):
    """Test worker starts its own Connect RPC server for health checks."""
    _, port = controller
    worker = FrayWorker(f"http://localhost:{port}", port=0)

    # Start worker in background thread
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()

    # Wait for server to start
    time.sleep(0.3)

    try:
        # Try to connect to worker's health check endpoint
        with httpx.Client() as http_client:
            worker_client = FrayWorkerClientSync(f"http://{worker.address}", session=http_client)
            status = worker_client.health_check(fray_pb2.Empty())

            assert status.worker_id == worker.worker_id
            assert status.healthy is True
            assert status.uptime_ms > 0

    finally:
        worker.stop()
        thread.join(timeout=2.0)


def test_worker_executes_task(controller):
    """Test worker fetches and executes a task."""
    _, port = controller
    _server, _ = controller

    # Submit a task to controller
    client = FrayControllerClientSync(f"http://localhost:{port}")
    task_data = {"fn": lambda x: x * 2, "args": (5,)}
    serialized_fn = cloudpickle.dumps(task_data)

    task_spec = fray_pb2.TaskSpec(serialized_fn=serialized_fn)
    handle = client.submit_task(task_spec)

    # Start worker
    worker = FrayWorker(f"http://localhost:{port}", port=0)
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()

    # Wait for task to complete
    time.sleep(0.5)

    try:
        # Check task status
        status = client.get_task_status(handle)
        assert status.status == fray_pb2.TASK_STATUS_COMPLETED

        # Verify result
        result = client.get_task_result(handle)
        actual_result = cloudpickle.loads(result.serialized_result)
        assert actual_result == 10

    finally:
        worker.stop()
        thread.join(timeout=2.0)


def test_worker_reports_task_failure(controller):
    """Test worker properly reports task failures to controller."""
    _, port = controller
    _server, _ = controller

    # Submit a failing task
    client = FrayControllerClientSync(f"http://localhost:{port}")

    def failing_task():
        raise ValueError("Intentional failure")

    task_data = {"fn": failing_task, "args": ()}
    serialized_fn = cloudpickle.dumps(task_data)

    task_spec = fray_pb2.TaskSpec(serialized_fn=serialized_fn)
    handle = client.submit_task(task_spec)

    # Start worker
    worker = FrayWorker(f"http://localhost:{port}", port=0)
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()

    # Wait for task to fail
    time.sleep(0.5)

    try:
        # Check task status
        status = client.get_task_status(handle)
        assert status.status == fray_pb2.TASK_STATUS_FAILED
        assert "ValueError" in status.error
        assert "Intentional failure" in status.error

    finally:
        worker.stop()
        thread.join(timeout=2.0)


def test_worker_handles_multiple_tasks(controller):
    """Test worker processes multiple tasks in sequence."""
    _, port = controller
    _server, _ = controller

    client = FrayControllerClientSync(f"http://localhost:{port}")

    # Submit multiple tasks
    handles = []
    for i in range(5):
        task_data = {"fn": lambda x: x * 2, "args": (i,)}
        serialized_fn = cloudpickle.dumps(task_data)
        task_spec = fray_pb2.TaskSpec(serialized_fn=serialized_fn)
        handle = client.submit_task(task_spec)
        handles.append(handle)

    # Start worker
    worker = FrayWorker(f"http://localhost:{port}", port=0)
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()

    # Wait for all tasks to complete
    time.sleep(1.0)

    try:
        # Verify all tasks completed
        for handle in handles:
            status = client.get_task_status(handle)
            assert status.status == fray_pb2.TASK_STATUS_COMPLETED

    finally:
        worker.stop()
        thread.join(timeout=2.0)


def test_worker_graceful_shutdown(controller):
    """Test worker shuts down gracefully and unregisters."""
    server, port = controller

    worker = FrayWorker(f"http://localhost:{port}", worker_id="shutdown-test", port=0)
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()

    # Wait for worker to register
    time.sleep(0.3)
    assert "shutdown-test" in server.servicer._workers

    # Stop worker
    worker.stop()
    thread.join(timeout=2.0)

    # Verify worker unregistered
    assert "shutdown-test" not in server.servicer._workers


def test_worker_tracks_current_tasks(controller):
    """Test worker's health check reports current tasks."""
    _, port = controller
    _server, _ = controller

    client = FrayControllerClientSync(f"http://localhost:{port}")

    # Submit a slow task
    def slow_task():
        time.sleep(2.0)
        return "done"

    task_data = {"fn": slow_task, "args": ()}
    serialized_fn = cloudpickle.dumps(task_data)
    task_spec = fray_pb2.TaskSpec(serialized_fn=serialized_fn)
    handle = client.submit_task(task_spec)

    # Start worker
    worker = FrayWorker(f"http://localhost:{port}", port=0)
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()

    # Wait for task to start
    time.sleep(0.3)

    try:
        # Check worker status shows running task
        with httpx.Client() as http_client:
            worker_client = FrayWorkerClientSync(f"http://{worker.address}", session=http_client)
            status = worker_client.health_check(fray_pb2.Empty())

            assert len(status.current_tasks) == 1
            assert status.current_tasks[0].task_id == handle.task_id
            assert status.current_tasks[0].status == fray_pb2.TASK_STATUS_RUNNING

    finally:
        worker.stop()
        thread.join(timeout=3.0)


def test_worker_list_tasks_endpoint(controller):
    """Test worker's list_tasks endpoint returns task information."""
    _, port = controller
    worker = FrayWorker(f"http://localhost:{port}", port=0)

    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()

    # Wait for server to start
    time.sleep(0.3)

    try:
        with httpx.Client() as http_client:
            worker_client = FrayWorkerClientSync(f"http://{worker.address}", session=http_client)
            status = worker_client.list_tasks(fray_pb2.Empty())

            assert status.worker_id == worker.worker_id
            assert status.healthy is True
            # current_tasks is a protobuf repeated field, check it's list-like
            assert len(status.current_tasks) == 0

    finally:
        worker.stop()
        thread.join(timeout=2.0)


def test_worker_uptime_tracking(controller):
    """Test worker accurately tracks uptime."""
    _, port = controller
    worker = FrayWorker(f"http://localhost:{port}", port=0)

    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()

    # Wait a bit
    time.sleep(0.5)

    try:
        with httpx.Client() as http_client:
            worker_client = FrayWorkerClientSync(f"http://{worker.address}", session=http_client)
            status = worker_client.health_check(fray_pb2.Empty())

            # Uptime should be at least 500ms
            assert status.uptime_ms >= 500

    finally:
        worker.stop()
        thread.join(timeout=2.0)


def test_worker_handles_controller_unavailable():
    """Test worker handles controller being unavailable during startup."""
    # Try to connect to non-existent controller
    worker = FrayWorker("http://localhost:59999", port=0)

    # Registration should fail (httpx.ConnectError or ConnectError)
    with pytest.raises((httpx.ConnectError, Exception)):
        worker.register()


def test_worker_servicer_status_methods():
    """Test FrayWorkerServicer status generation methods directly."""
    from fray.job.rpc.worker import FrayWorkerServicer

    servicer = FrayWorkerServicer("test-worker")

    # Test _get_status with no tasks
    status = servicer._get_status()
    assert status.worker_id == "test-worker"
    assert status.healthy is True
    assert len(status.current_tasks) == 0
    assert status.uptime_ms >= 0

    # Add a task
    servicer.add_task("task-1", fray_pb2.TASK_STATUS_RUNNING)
    status = servicer._get_status()
    assert len(status.current_tasks) == 1
    assert status.current_tasks[0].task_id == "task-1"
    assert status.current_tasks[0].status == fray_pb2.TASK_STATUS_RUNNING

    # Update task status
    servicer.update_task_status("task-1", fray_pb2.TASK_STATUS_COMPLETED)
    status = servicer._get_status()
    assert status.current_tasks[0].status == fray_pb2.TASK_STATUS_COMPLETED

    # Remove task
    servicer.remove_task("task-1")
    status = servicer._get_status()
    assert len(status.current_tasks) == 0


@pytest.mark.asyncio
async def test_worker_servicer_health_check_method():
    """Test FrayWorkerServicer health_check method."""
    from fray.job.rpc.worker import FrayWorkerServicer

    servicer = FrayWorkerServicer("test-worker")

    # Mock RequestContext
    class MockContext:
        pass

    status = await servicer.health_check(fray_pb2.Empty(), MockContext())
    assert status.worker_id == "test-worker"
    assert status.healthy is True


@pytest.mark.asyncio
async def test_worker_servicer_list_tasks_method():
    """Test FrayWorkerServicer list_tasks method."""
    from fray.job.rpc.worker import FrayWorkerServicer

    servicer = FrayWorkerServicer("test-worker")
    servicer.add_task("task-1", fray_pb2.TASK_STATUS_RUNNING)

    # Mock RequestContext
    class MockContext:
        pass

    status = await servicer.list_tasks(fray_pb2.Empty(), MockContext())
    assert status.worker_id == "test-worker"
    assert len(status.current_tasks) == 1
