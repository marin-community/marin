# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for WorkerDashboard HTTP/RPC endpoints and WorkerService implementation."""

import socket
from unittest.mock import Mock

import httpx
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from iris.cluster.types import Entrypoint, JobName
from iris.time_utils import Duration
from iris.cluster.worker.bundle_cache import BundleCache
from iris.cluster.worker.dashboard import WorkerDashboard
from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.runtime.types import ContainerStats, ContainerStatus
from iris.cluster.worker.service import WorkerServiceImpl
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.rpc import cluster_pb2
from starlette.testclient import TestClient

# ============================================================================
# Shared fixtures
# ============================================================================


@pytest.fixture
def mock_bundle_cache(tmp_path):
    """Create mock BundleCache with a real temp directory."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "test_file.py").write_text("print('hello')")

    cache = Mock(spec=BundleCache)
    cache.get_bundle = Mock(return_value=bundle_dir)
    return cache


def create_mock_container_handle():
    """Create a mock ContainerHandle that completes immediately.

    Returns a handle where status() returns not-running right away,
    so tasks complete immediately in tests.
    """
    handle = Mock()
    handle.container_id = "container123"
    handle.build = Mock(return_value=[])
    handle.run = Mock()
    handle.status = Mock(return_value=ContainerStatus(running=False, exit_code=0))
    handle.stop = Mock()
    handle.logs = Mock(return_value=[])
    handle.stats = Mock(return_value=ContainerStats(memory_mb=100, cpu_percent=50, process_count=1, available=True))
    handle.cleanup = Mock()
    return handle


@pytest.fixture
def mock_runtime():
    """Create mock DockerRuntime that returns ContainerHandle objects."""
    runtime = Mock(spec=DockerRuntime)

    # create_container returns a ContainerHandle mock
    runtime.create_container = Mock(side_effect=lambda config: create_mock_container_handle())

    runtime.list_iris_containers = Mock(return_value=[])
    runtime.remove_all_iris_containers = Mock(return_value=0)
    return runtime


@pytest.fixture
def worker(mock_bundle_cache, mock_runtime, tmp_path):
    """Create Worker with mocked dependencies."""
    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
        cache_dir=tmp_path / "cache",
    )
    return Worker(
        config,
        bundle_provider=mock_bundle_cache,
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
    task_id: str = JobName.root("test-job-1").task(0).to_wire(),
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
        dockerfile="FROM python:3.11-slim\nRUN echo test",
    )

    resources = cluster_pb2.ResourceSpecProto(cpu=2, memory_bytes=4 * 1024**3)

    request = cluster_pb2.Worker.RunTaskRequest(
        task_id=task_id,
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
        request = create_run_task_request(task_id=JobName.root(f"job-{i}").task(0).to_wire())
        worker.submit_task(request)

    response = rpc_post(client, "ListTasks")
    assert response.status_code == 200
    data = response.json()
    tasks = data.get("tasks", [])
    assert len(tasks) == 3

    task_ids = {t["taskId"] for t in tasks}
    assert task_ids == {
        JobName.root("job-0").task(0).to_wire(),
        JobName.root("job-1").task(0).to_wire(),
        JobName.root("job-2").task(0).to_wire(),
    }


def test_get_task_not_found(client):
    """Test GetTaskStatus RPC with nonexistent task returns error."""
    response = rpc_post(
        client,
        "GetTaskStatus",
        {"taskId": JobName.root("nonexistent").task(0).to_wire()},
    )
    assert response.status_code != 200


def test_get_task_success(client, worker):
    """Test GetTaskStatus RPC returns task details."""
    task_id = JobName.root("job-details").task(0).to_wire()
    request = create_run_task_request(task_id=task_id, ports=["http", "grpc"])
    worker.submit_task(request)

    # Wait for task to complete (mock runtime returns running=False immediately)
    task = worker.get_task(task_id)
    task.thread.join(timeout=5.0)

    response = rpc_post(client, "GetTaskStatus", {"taskId": task_id})
    assert response.status_code == 200
    data = response.json()

    assert data["taskId"] == task_id
    assert JobName.from_wire(data["taskId"]).require_task()[0].to_wire() == JobName.root("job-details").to_wire()
    assert data["state"] == "TASK_STATE_SUCCEEDED"
    assert data["exitCode"] == 0
    assert "http" in data["ports"]
    assert "grpc" in data["ports"]


def test_get_logs_with_tail_parameter(client, worker):
    """Test FetchTaskLogs RPC with negative start_line for tailing."""
    import time

    task_id = JobName.root("job-tail").task(0).to_wire()
    request = create_run_task_request(task_id=task_id)
    worker.submit_task(request)

    # Wait for task to transition out of building state (should be fast with mocked runtime)
    deadline = time.time() + 2.0
    while time.time() < deadline and worker.get_task(task_id).status == cluster_pb2.TASK_STATE_BUILDING:
        time.sleep(0.01)

    # Inject logs
    task = worker.get_task(task_id)
    for i in range(100):
        task.logs.add("stdout", f"Log line {i}")

    response = rpc_post(
        client,
        "FetchTaskLogs",
        {
            "taskId": task_id,
            "filter": {"startLine": -5},
        },
    )
    assert response.status_code == 200
    data = response.json()
    logs = data.get("logs", [])

    # Filter to only stdout logs (build logs may be interleaved when timestamps collide)
    stdout_logs = [log for log in logs if log["source"] == "stdout"]
    assert len(stdout_logs) >= 4, stdout_logs
    assert stdout_logs[-1]["data"] == "Log line 99", stdout_logs


def test_get_logs_with_source_filter(client, worker):
    """Test FetchTaskLogs RPC returns logs that can be filtered client-side."""
    import time

    task_id = JobName.root("job-source-filter").task(0).to_wire()
    request = create_run_task_request(task_id=task_id)
    worker.submit_task(request)

    # Stop the task thread so it doesn't add more logs
    task = worker.get_task(task_id)
    # Give the thread a moment to start
    time.sleep(0.02)
    task.should_stop = True
    if task.thread:
        task.thread.join(timeout=1.0)

    # Clear any existing logs and add test logs
    task.logs.lines.clear()
    task.logs.add("stdout", "stdout line 1")
    task.logs.add("stdout", "stdout line 2")
    task.logs.add("stderr", "stderr line 1")
    task.logs.add("stderr", "stderr line 2")

    response = rpc_post(
        client,
        "FetchTaskLogs",
        {"taskId": task_id},
    )
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
    task_id = JobName.root("job-logs-tail").task(0).to_wire()
    request = create_run_task_request(task_id=task_id)
    worker.submit_task(request)

    # Add logs directly to task.logs
    task = worker.get_task(task_id)
    for i in range(10):
        task.logs.add("stdout", f"Log line {i}")

    log_filter = cluster_pb2.Worker.FetchLogsFilter(start_line=-3)
    logs_request = cluster_pb2.Worker.FetchTaskLogsRequest(task_id=task_id, filter=log_filter)
    response = service.fetch_task_logs(logs_request, request_context)

    assert len(response.logs) == 3
    assert response.logs[0].data == "Log line 7"
    assert response.logs[1].data == "Log line 8"
    assert response.logs[2].data == "Log line 9"


def test_fetch_task_logs_with_regex_filter(service, worker, request_context):
    """Test fetch_task_logs with regex content filter."""
    task_id = JobName.root("job-logs-regex").task(0).to_wire()
    request = create_run_task_request(task_id=task_id)
    worker.submit_task(request)

    # Add logs with different patterns
    task = worker.get_task(task_id)
    task.logs.add("stdout", "ERROR: something bad")
    task.logs.add("stdout", "INFO: normal log")
    task.logs.add("stdout", "ERROR: another error")
    task.logs.add("stdout", "DEBUG: details")

    log_filter = cluster_pb2.Worker.FetchLogsFilter(regex="ERROR")
    logs_request = cluster_pb2.Worker.FetchTaskLogsRequest(task_id=task_id, filter=log_filter)
    response = service.fetch_task_logs(logs_request, request_context)

    assert len(response.logs) == 2
    assert "ERROR" in response.logs[0].data
    assert "ERROR" in response.logs[1].data


def test_fetch_task_logs_combined_filters(service, worker, request_context):
    """Test fetch_task_logs with multiple filters combined."""
    task_id = JobName.root("job-logs-combined").task(0).to_wire()
    request = create_run_task_request(task_id=task_id)
    worker.submit_task(request)

    # Add logs
    task = worker.get_task(task_id)
    task.logs.add("stdout", "ERROR: first error")
    task.logs.add("stdout", "INFO: normal")
    task.logs.add("stdout", "ERROR: second error")
    task.logs.add("stdout", "ERROR: third error")
    task.logs.add("stdout", "ERROR: fourth error")
    task.logs.add("stdout", "ERROR: fifth error")

    # Use regex to filter ERRORs, then limit to 2
    log_filter = cluster_pb2.Worker.FetchLogsFilter(regex="ERROR", max_lines=2)
    logs_request = cluster_pb2.Worker.FetchTaskLogsRequest(task_id=task_id, filter=log_filter)
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
    task_id = JobName.root("job-with-ports").task(0).to_wire()
    request = create_run_task_request(task_id=task_id, ports=["http", "grpc"])
    worker.submit_task(request)

    # Verify ports were allocated
    task = worker.get_task(task_id)
    assert len(task.ports) == 2
    assert "http" in task.ports
    assert "grpc" in task.ports


def test_get_task_status_not_found(service, worker, request_context):
    """Test get_task_status raises NOT_FOUND for nonexistent task."""
    status_request = cluster_pb2.Worker.GetTaskStatusRequest(task_id=JobName.root("nonexistent").task(0).to_wire())

    with pytest.raises(ConnectError) as exc_info:
        service.get_task_status(status_request, request_context)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in str(exc_info.value)


def test_get_task_status_completed_task(service, worker, request_context):
    """Test get_task_status for completed task includes timing info."""
    task_id = JobName.root("job-completed").task(0).to_wire()
    request = create_run_task_request(task_id=task_id)
    worker.submit_task(request)

    # Wait for task to complete
    task = worker.get_task(task_id)
    task.thread.join(timeout=5.0)

    status_request = cluster_pb2.Worker.GetTaskStatusRequest(task_id=task_id)
    status = service.get_task_status(status_request, request_context)

    assert status.task_id == task_id
    assert JobName.from_wire(status.task_id).require_task()[0].to_wire() == JobName.root("job-completed").to_wire()
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

        run_req = create_run_task_request(task_id=JobName.root("rpc-test-job").task(0).to_wire())
        heartbeat_req = cluster_pb2.HeartbeatRequest(tasks_to_run=[run_req])
        response = client.heartbeat(heartbeat_req)

        # Heartbeat response should have the task in running or completed
        all_task_ids = {t.task_id for t in response.running_tasks} | {t.task_id for t in response.completed_tasks}
        assert "rpc-test-task" in all_task_ids

    finally:
        uvicorn_server.should_exit = True
        server_thread.join(timeout=2.0)
