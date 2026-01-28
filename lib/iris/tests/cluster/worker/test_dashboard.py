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

"""Tests for WorkerDashboard HTTP/RPC endpoints and WorkerService implementation."""

import asyncio
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from iris.cluster.types import Entrypoint
from iris.cluster.worker.builder import BuildResult, ImageCache
from iris.cluster.worker.bundle_cache import BundleCache
from iris.cluster.worker.dashboard import WorkerDashboard
from iris.cluster.worker.docker import ContainerStats, ContainerStatus, DockerRuntime
from iris.cluster.worker.service import WorkerServiceImpl
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import WorkerServiceClient
from starlette.testclient import TestClient

# ============================================================================
# Shared fixtures
# ============================================================================


@pytest.fixture
def mock_bundle_cache():
    """Create mock BundleCache."""
    cache = Mock(spec=BundleCache)
    cache.get_bundle = Mock(return_value=Path("/tmp/bundle"))
    return cache


@pytest.fixture
def mock_image_cache():
    """Create mock ImageCache."""
    cache = Mock(spec=ImageCache)
    cache.build = Mock(
        return_value=BuildResult(
            image_tag="test-image:latest",
            build_time_ms=1000,
            from_cache=False,
        )
    )
    return cache


@pytest.fixture
def mock_runtime():
    """Create mock DockerRuntime for sync model.

    The sync model uses create_container/start_container/inspect pattern
    instead of the blocking run() method.
    """
    runtime = Mock(spec=DockerRuntime)

    # Container lifecycle methods
    runtime.create_container = Mock(return_value="container123")
    runtime.start_container = Mock()

    # Inspect returns not-running so tasks complete immediately
    runtime.inspect = Mock(return_value=ContainerStatus(running=False, exit_code=0))

    runtime.kill = Mock()
    runtime.remove = Mock()
    runtime.get_stats = Mock(
        return_value=ContainerStats(
            memory_mb=100,
            cpu_percent=50,
            process_count=1,
            available=True,
        )
    )
    runtime.get_logs = Mock(return_value=[])
    runtime.list_iris_containers = Mock(return_value=[])
    runtime.remove_all_iris_containers = Mock(return_value=0)
    return runtime


@pytest.fixture
def worker(mock_bundle_cache, mock_image_cache, mock_runtime):
    """Create Worker with mocked dependencies."""
    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
    )
    return Worker(
        config,
        bundle_provider=mock_bundle_cache,
        image_provider=mock_image_cache,
        container_runtime=mock_runtime,
    )


def create_test_entrypoint():
    """Create a simple test entrypoint."""
    from dataclasses import dataclass

    @dataclass
    class Entrypoint:
        callable: object
        args: tuple = ()
        kwargs: dict | None = None

        def __post_init__(self):
            if self.kwargs is None:
                self.kwargs = {}

    def test_fn():
        print("Hello from test")

    return Entrypoint(callable=test_fn)


def create_run_task_request(
    task_id: str = "test-task-1",
    job_id: str = "test-job-1",
    task_index: int = 0,
    num_tasks: int = 1,
    ports: list[str] | None = None,
):
    """Create a RunTaskRequest for testing."""

    def test_fn():
        print("Hello from test")

    entrypoint_proto = Entrypoint.from_callable(test_fn).to_proto()

    env_config = cluster_pb2.EnvironmentConfig(
        env_vars={
            "TEST_VAR": "value",
            "TASK_VAR": "task_value",
        },
        extras=["dev"],
    )

    resources = cluster_pb2.ResourceSpecProto(cpu=2, memory_bytes=4 * 1024**3)

    return cluster_pb2.Worker.RunTaskRequest(
        task_id=task_id,
        job_id=job_id,
        task_index=task_index,
        num_tasks=num_tasks,
        entrypoint=entrypoint_proto,
        environment=env_config,
        bundle_gcs_path="gs://bucket/bundle.zip",
        resources=resources,
        timeout_seconds=300,
        ports=ports or [],
    )


@pytest.fixture
def service(worker):
    """Create WorkerServiceImpl."""
    return WorkerServiceImpl(provider=worker)


@pytest.fixture
def server(service):
    """Create WorkerDashboard."""
    return WorkerDashboard(service=service, host="127.0.0.1", port=0)


@pytest.fixture
def client(server):
    """Create test client for HTTP requests."""
    return TestClient(server._app)


@pytest.fixture
def request_context():
    """Create a mock RequestContext for RPC calls."""
    return Mock(spec=RequestContext)


def test_list_tasks_with_data(client, service):
    """Test /api/tasks returns all tasks."""
    for i in range(3):
        request = create_run_task_request(task_id=f"task-{i}", job_id=f"job-{i}")
        service.run_task(request, Mock())

    response = client.get("/api/tasks")
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) == 3

    task_ids = {t["task_id"] for t in tasks}
    assert task_ids == {"task-0", "task-1", "task-2"}


def test_get_task_not_found(client):
    """Test /api/tasks/{task_id} with nonexistent task."""
    response = client.get("/api/tasks/nonexistent")
    assert response.status_code == 404


def test_get_task_success(client, service):
    """Test /api/tasks/{task_id} returns task details."""
    request = create_run_task_request(task_id="task-details", job_id="job-details", ports=["http", "grpc"])
    service.run_task(request, Mock())

    # Wait for task to complete (mock runtime returns running=False immediately)
    task = service._provider.get_task("task-details")
    task.thread.join(timeout=5.0)

    response = client.get("/api/tasks/task-details")
    assert response.status_code == 200
    data = response.json()

    assert data["task_id"] == "task-details"
    assert data["job_id"] == "job-details"
    assert data["status"] == "TASK_STATE_SUCCEEDED"  # Task completes immediately with mock runtime
    assert data["exit_code"] == 0
    assert "http" in data["ports"]
    assert "grpc" in data["ports"]


def test_get_logs_with_tail_parameter(client, service):
    """Test /api/tasks/{task_id}/logs?tail=N returns last N lines."""
    request = create_run_task_request(task_id="task-tail", job_id="job-tail")
    service.run_task(request, Mock())

    # wait for task to transition out of building state
    import time

    while service._provider.get_task("task-tail").status == cluster_pb2.TASK_STATE_BUILDING:
        time.sleep(0.1)

    # inject logs
    task = service._provider.get_task("task-tail")
    for i in range(100):
        task.logs.add("stdout", f"Log line {i}")

    response = client.get("/api/tasks/task-tail/logs?tail=5")
    assert response.status_code == 200
    logs = response.json()

    assert len(logs) == 5
    assert logs[0]["data"] == "Log line 95", logs
    assert logs[4]["data"] == "Log line 99", logs


def test_get_logs_with_source_filter(client, service):
    """Test /api/tasks/{task_id}/logs?source=stdout filters by source."""
    import time

    request = create_run_task_request(task_id="task-source-filter", job_id="job-source-filter")
    service.run_task(request, Mock())

    # Stop the task thread so it doesn't add more logs
    task = service._provider.get_task("task-source-filter")
    time.sleep(0.05)
    task.should_stop = True
    if task.thread:
        task.thread.join(timeout=1.0)

    # Clear any existing logs and add test logs
    task.logs.lines.clear()
    task.logs.add("stdout", "stdout line 1")
    task.logs.add("stdout", "stdout line 2")
    task.logs.add("stderr", "stderr line 1")
    task.logs.add("stderr", "stderr line 2")

    # Test stdout filter
    response = client.get("/api/tasks/task-source-filter/logs?source=stdout")
    assert response.status_code == 200
    logs = response.json()
    assert len(logs) == 2
    assert all(log["source"] == "stdout" for log in logs)

    # Test stderr filter
    response = client.get("/api/tasks/task-source-filter/logs?source=stderr")
    assert response.status_code == 200
    logs = response.json()
    assert len(logs) == 2
    assert all(log["source"] == "stderr" for log in logs)

    # Test without filter - should get all logs
    response = client.get("/api/tasks/task-source-filter/logs")
    assert response.status_code == 200
    logs = response.json()
    assert len(logs) == 4  # 2 stdout + 2 stderr


def test_fetch_task_logs_tail_with_negative_start_line(service, request_context):
    """Test fetch_task_logs with negative start_line for tailing."""
    request = create_run_task_request(task_id="task-logs-tail", job_id="job-logs-tail")
    service.run_task(request, request_context)

    # Add logs directly to task.logs
    task = service._provider.get_task("task-logs-tail")
    for i in range(10):
        task.logs.add("stdout", f"Log line {i}")

    log_filter = cluster_pb2.Worker.FetchLogsFilter(start_line=-3)
    logs_request = cluster_pb2.Worker.FetchTaskLogsRequest(task_id="task-logs-tail", filter=log_filter)
    response = service.fetch_task_logs(logs_request, request_context)

    assert len(response.logs) == 3
    assert response.logs[0].data == "Log line 7"
    assert response.logs[1].data == "Log line 8"
    assert response.logs[2].data == "Log line 9"


def test_fetch_task_logs_with_regex_filter(service, request_context):
    """Test fetch_task_logs with regex content filter."""
    request = create_run_task_request(task_id="task-logs-regex", job_id="job-logs-regex")
    service.run_task(request, request_context)

    # Add logs with different patterns
    task = service._provider.get_task("task-logs-regex")
    task.logs.add("stdout", "ERROR: something bad")
    task.logs.add("stdout", "INFO: normal log")
    task.logs.add("stdout", "ERROR: another error")
    task.logs.add("stdout", "DEBUG: details")

    log_filter = cluster_pb2.Worker.FetchLogsFilter(regex="ERROR")
    logs_request = cluster_pb2.Worker.FetchTaskLogsRequest(task_id="task-logs-regex", filter=log_filter)
    response = service.fetch_task_logs(logs_request, request_context)

    assert len(response.logs) == 2
    assert "ERROR" in response.logs[0].data
    assert "ERROR" in response.logs[1].data


def test_fetch_task_logs_combined_filters(service, request_context):
    """Test fetch_task_logs with multiple filters combined."""
    request = create_run_task_request(task_id="task-logs-combined", job_id="job-logs-combined")
    service.run_task(request, request_context)

    # Add logs
    task = service._provider.get_task("task-logs-combined")
    task.logs.add("stdout", "ERROR: first error")
    task.logs.add("stdout", "INFO: normal")
    task.logs.add("stdout", "ERROR: second error")
    task.logs.add("stdout", "ERROR: third error")
    task.logs.add("stdout", "ERROR: fourth error")
    task.logs.add("stdout", "ERROR: fifth error")

    # Use regex to filter ERRORs, then limit to 2
    log_filter = cluster_pb2.Worker.FetchLogsFilter(regex="ERROR", max_lines=2)
    logs_request = cluster_pb2.Worker.FetchTaskLogsRequest(task_id="task-logs-combined", filter=log_filter)
    response = service.fetch_task_logs(logs_request, request_context)

    assert len(response.logs) == 2
    assert "ERROR" in response.logs[0].data
    assert "ERROR" in response.logs[1].data


def test_task_detail_page_loads(client):
    """Test /task/{task_id} page loads successfully."""
    response = client.get("/task/test-task-123")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"


# ============================================================================
# RPC Service tests (WorkerServiceImpl)
# ============================================================================


def test_run_task_with_ports(service, request_context):
    """Test run_task allocates ports correctly."""
    request = create_run_task_request(task_id="task-with-ports", job_id="job-with-ports", ports=["http", "grpc"])
    service.run_task(request, request_context)

    # Verify ports were allocated
    task = service._provider.get_task("task-with-ports")
    assert len(task.ports) == 2
    assert "http" in task.ports
    assert "grpc" in task.ports


def test_get_task_status_not_found(service, request_context):
    """Test get_task_status raises NOT_FOUND for nonexistent task."""
    status_request = cluster_pb2.Worker.GetTaskStatusRequest(task_id="nonexistent")

    with pytest.raises(ConnectError) as exc_info:
        service.get_task_status(status_request, request_context)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in str(exc_info.value)


def test_get_task_status_completed_task(service, request_context):
    """Test get_task_status for completed task includes timing info."""
    request = create_run_task_request(task_id="task-completed", job_id="job-completed")
    service.run_task(request, request_context)

    # Wait for task to complete
    task = service._provider.get_task("task-completed")
    task.thread.join(timeout=5.0)

    status_request = cluster_pb2.Worker.GetTaskStatusRequest(task_id="task-completed")
    status = service.get_task_status(status_request, request_context)

    assert status.task_id == "task-completed"
    assert status.job_id == "job-completed"
    assert status.state == cluster_pb2.TASK_STATE_SUCCEEDED
    assert status.exit_code == 0
    assert status.started_at_ms > 0
    assert status.finished_at_ms > 0


def test_kill_task_not_found(service, request_context):
    """Test kill_task raises NOT_FOUND for nonexistent task."""
    kill_request = cluster_pb2.Worker.KillTaskRequest(task_id="nonexistent")

    with pytest.raises(ConnectError) as exc_info:
        service.kill_task(kill_request, request_context)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in str(exc_info.value)


def test_kill_task_already_completed(service, request_context):
    """Test kill_task fails for already completed task."""
    request = create_run_task_request(task_id="task-completed", job_id="job-completed")
    service.run_task(request, request_context)

    # Wait for task to complete
    task = service._provider.get_task("task-completed")
    task.thread.join(timeout=5.0)

    # Try to kill completed task
    kill_request = cluster_pb2.Worker.KillTaskRequest(task_id="task-completed")

    with pytest.raises(ConnectError) as exc_info:
        service.kill_task(kill_request, request_context)

    assert exc_info.value.code == Code.FAILED_PRECONDITION
    assert "already completed" in str(exc_info.value)


def test_kill_task_with_custom_timeout(service, request_context):
    """Test kill_task accepts custom term_timeout_ms and attempts termination.

    Note: With mocks, the task thread completes immediately. This test verifies
    the API works and runtime.kill is called, not the actual kill behavior.
    """
    request = create_run_task_request(task_id="task-kill", job_id="job-kill")
    service.run_task(request, request_context)

    # Wait for task thread to finish (mock makes it complete immediately)
    task = service._provider.get_task("task-kill")
    if task.thread:
        task.thread.join(timeout=5.0)

    # Manually set task to RUNNING to simulate mid-execution state
    task.status = cluster_pb2.TASK_STATE_RUNNING
    task.container_id = "container123"

    kill_request = cluster_pb2.Worker.KillTaskRequest(task_id="task-kill", term_timeout_ms=100)
    service.kill_task(kill_request, request_context)

    # Verify that should_stop was set
    assert task.should_stop is True
    # The runtime.kill should have been called (may be called twice: SIGTERM then SIGKILL)
    assert service._provider._runtime.kill.called


# ============================================================================
# Connect RPC integration tests
# ============================================================================


@pytest.mark.asyncio
async def test_rpc_run_task_via_connect_client(service):
    """Test calling run_task via Connect RPC client."""
    # Create server on ephemeral port
    server = WorkerDashboard(service=service, host="127.0.0.1", port=0)

    # Run server in background
    async def run_server():
        import uvicorn

        config = uvicorn.Config(server._app, host="127.0.0.1", port=18080)
        server_obj = uvicorn.Server(config)
        await server_obj.serve()

    server_task = asyncio.create_task(run_server())

    try:
        # Give server time to start
        await asyncio.sleep(0.5)

        # Create Connect client (async client talks to WSGI server via WSGIMiddleware)
        async with httpx.AsyncClient() as http_client:
            client = WorkerServiceClient(address="http://127.0.0.1:18080", session=http_client)

            # Submit task via RPC
            request = create_run_task_request(task_id="rpc-test-task", job_id="rpc-test-job")
            response = await client.run_task(request)

            assert response.task_id == "rpc-test-task"
            # Task may have already transitioned from PENDING since threads start immediately
            assert response.state in (
                cluster_pb2.TASK_STATE_PENDING,
                cluster_pb2.TASK_STATE_BUILDING,
                cluster_pb2.TASK_STATE_RUNNING,
                cluster_pb2.TASK_STATE_SUCCEEDED,
            )

    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
