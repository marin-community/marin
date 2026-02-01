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

import socket
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from iris.cluster.types import Entrypoint
from iris.time_utils import Duration
from iris.cluster.worker.builder import BuildResult, ImageCache
from iris.cluster.worker.bundle_cache import BundleCache
from iris.cluster.worker.dashboard import WorkerDashboard
from iris.cluster.worker.docker import ContainerStats, ContainerStatus, DockerRuntime
from iris.cluster.worker.service import WorkerServiceImpl
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.rpc import cluster_pb2
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

    request = cluster_pb2.Worker.RunTaskRequest(
        task_id=task_id,
        job_id=job_id,
        task_index=task_index,
        num_tasks=num_tasks,
        entrypoint=entrypoint_proto,
        environment=env_config,
        bundle_gcs_path="gs://bucket/bundle.zip",
        resources=resources,
        ports=ports or [],
    )
    request.timeout.CopyFrom(Duration.from_seconds(300).to_proto())
    return request


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


def rpc_post(client, method, body=None):
    """Call a Connect RPC method via the test client."""
    return client.post(
        f"/iris.cluster.WorkerService/{method}",
        json=body or {},
        headers={"Content-Type": "application/json"},
    )


def test_list_tasks_with_data(client, worker):
    """Test ListTasks RPC returns all tasks."""
    for i in range(3):
        request = create_run_task_request(task_id=f"task-{i}", job_id=f"job-{i}")
        worker.submit_task(request)

    response = rpc_post(client, "ListTasks")
    assert response.status_code == 200
    data = response.json()
    tasks = data.get("tasks", [])
    assert len(tasks) == 3

    task_ids = {t["taskId"] for t in tasks}
    assert task_ids == {"task-0", "task-1", "task-2"}


def test_get_task_not_found(client):
    """Test GetTaskStatus RPC with nonexistent task returns error."""
    response = rpc_post(client, "GetTaskStatus", {"taskId": "nonexistent"})
    assert response.status_code != 200


def test_get_task_success(client, worker):
    """Test GetTaskStatus RPC returns task details."""
    request = create_run_task_request(task_id="task-details", job_id="job-details", ports=["http", "grpc"])
    worker.submit_task(request)

    # Wait for task to complete (mock runtime returns running=False immediately)
    task = worker.get_task("task-details")
    task.thread.join(timeout=5.0)

    response = rpc_post(client, "GetTaskStatus", {"taskId": "task-details"})
    assert response.status_code == 200
    data = response.json()

    assert data["taskId"] == "task-details"
    assert data["jobId"] == "job-details"
    assert data["state"] == "TASK_STATE_SUCCEEDED"
    assert data["exitCode"] == 0
    assert "http" in data["ports"]
    assert "grpc" in data["ports"]


def test_get_logs_with_tail_parameter(client, worker):
    """Test FetchTaskLogs RPC with negative start_line for tailing."""
    request = create_run_task_request(task_id="task-tail", job_id="job-tail")
    worker.submit_task(request)

    # wait for task to transition out of building state
    import time

    while worker.get_task("task-tail").status == cluster_pb2.TASK_STATE_BUILDING:
        time.sleep(0.1)

    # inject logs
    task = worker.get_task("task-tail")
    for i in range(100):
        task.logs.add("stdout", f"Log line {i}")

    response = rpc_post(
        client,
        "FetchTaskLogs",
        {
            "taskId": "task-tail",
            "filter": {"startLine": -5},
        },
    )
    assert response.status_code == 200
    data = response.json()
    logs = data.get("logs", [])

    assert len(logs) == 5
    assert logs[0]["data"] == "Log line 95", logs
    assert logs[4]["data"] == "Log line 99", logs


def test_get_logs_with_source_filter(client, worker):
    """Test FetchTaskLogs RPC returns logs that can be filtered client-side."""
    import time

    request = create_run_task_request(task_id="task-source-filter", job_id="job-source-filter")
    worker.submit_task(request)

    # Stop the task thread so it doesn't add more logs
    task = worker.get_task("task-source-filter")
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

    response = rpc_post(client, "FetchTaskLogs", {"taskId": "task-source-filter"})
    assert response.status_code == 200
    data = response.json()
    logs = data.get("logs", [])
    assert len(logs) == 4

    stdout_logs = [entry for entry in logs if entry["source"] == "stdout"]
    stderr_logs = [entry for entry in logs if entry["source"] == "stderr"]
    assert len(stdout_logs) == 2
    assert len(stderr_logs) == 2


def test_fetch_task_logs_tail_with_negative_start_line(service, worker, request_context):
    """Test fetch_task_logs with negative start_line for tailing."""
    request = create_run_task_request(task_id="task-logs-tail", job_id="job-logs-tail")
    worker.submit_task(request)

    # Add logs directly to task.logs
    task = worker.get_task("task-logs-tail")
    for i in range(10):
        task.logs.add("stdout", f"Log line {i}")

    log_filter = cluster_pb2.Worker.FetchLogsFilter(start_line=-3)
    logs_request = cluster_pb2.Worker.FetchTaskLogsRequest(task_id="task-logs-tail", filter=log_filter)
    response = service.fetch_task_logs(logs_request, request_context)

    assert len(response.logs) == 3
    assert response.logs[0].data == "Log line 7"
    assert response.logs[1].data == "Log line 8"
    assert response.logs[2].data == "Log line 9"


def test_fetch_task_logs_with_regex_filter(service, worker, request_context):
    """Test fetch_task_logs with regex content filter."""
    request = create_run_task_request(task_id="task-logs-regex", job_id="job-logs-regex")
    worker.submit_task(request)

    # Add logs with different patterns
    task = worker.get_task("task-logs-regex")
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


def test_fetch_task_logs_combined_filters(service, worker, request_context):
    """Test fetch_task_logs with multiple filters combined."""
    request = create_run_task_request(task_id="task-logs-combined", job_id="job-logs-combined")
    worker.submit_task(request)

    # Add logs
    task = worker.get_task("task-logs-combined")
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


def test_run_task_with_ports(worker):
    """Test run_task allocates ports correctly."""
    request = create_run_task_request(task_id="task-with-ports", job_id="job-with-ports", ports=["http", "grpc"])
    worker.submit_task(request)

    # Verify ports were allocated
    task = worker.get_task("task-with-ports")
    assert len(task.ports) == 2
    assert "http" in task.ports
    assert "grpc" in task.ports


def test_get_task_status_not_found(service, worker, request_context):
    """Test get_task_status raises NOT_FOUND for nonexistent task."""
    status_request = cluster_pb2.Worker.GetTaskStatusRequest(task_id="nonexistent")

    with pytest.raises(ConnectError) as exc_info:
        service.get_task_status(status_request, request_context)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in str(exc_info.value)


def test_get_task_status_completed_task(service, worker, request_context):
    """Test get_task_status for completed task includes timing info."""
    request = create_run_task_request(task_id="task-completed", job_id="job-completed")
    worker.submit_task(request)

    # Wait for task to complete
    task = worker.get_task("task-completed")
    task.thread.join(timeout=5.0)

    status_request = cluster_pb2.Worker.GetTaskStatusRequest(task_id="task-completed")
    status = service.get_task_status(status_request, request_context)

    assert status.task_id == "task-completed"
    assert status.job_id == "job-completed"
    assert status.state == cluster_pb2.TASK_STATE_SUCCEEDED
    assert status.exit_code == 0
    assert status.started_at.epoch_ms > 0
    assert status.finished_at.epoch_ms > 0


# ============================================================================
# Connect RPC integration tests
# ============================================================================


@pytest.mark.skip(reason="Flaky test - timing issues with server startup")
def test_rpc_heartbeat_via_connect_client(service):
    """Test calling heartbeat via Connect RPC client."""
    import threading
    import time

    import uvicorn
    from iris.rpc.cluster_connect import WorkerServiceClientSync

    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]

    # Create server
    server = WorkerDashboard(service=service, host="127.0.0.1", port=port)

    # Run server in background thread (sync version)
    config = uvicorn.Config(server._app, host="127.0.0.1", port=port, log_level="error")
    uvicorn_server = uvicorn.Server(config)

    def run_server():
        uvicorn_server.run()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    try:
        # Wait for server to start with retry logic
        max_retries = 10
        for i in range(max_retries):
            time.sleep(0.2)

            try:
                with httpx.Client(timeout=2.0) as test_client:
                    test_client.get(f"http://127.0.0.1:{port}/")
                break
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if i == max_retries - 1:
                    raise TimeoutError(f"Could not connect to server on port {port} after {max_retries} retries") from e
                continue

        # Call heartbeat via sync Connect client
        client = WorkerServiceClientSync(address=f"http://127.0.0.1:{port}", timeout_ms=5000)

        run_req = create_run_task_request(task_id="rpc-test-task", job_id="rpc-test-job")
        heartbeat_req = cluster_pb2.HeartbeatRequest(tasks_to_run=[run_req])
        response = client.heartbeat(heartbeat_req)

        # Heartbeat response should have the task in running or completed
        all_task_ids = {t.task_id for t in response.running_tasks} | {t.task_id for t in response.completed_tasks}
        assert "rpc-test-task" in all_task_ids

    finally:
        uvicorn_server.should_exit = True
        server_thread.join(timeout=2.0)
