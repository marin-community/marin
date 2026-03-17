# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for WorkerDashboard HTTP/RPC endpoints and WorkerService implementation."""

from unittest.mock import Mock

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from iris.cluster.types import Entrypoint, JobName
from iris.time_utils import Duration
from iris.cluster.bundle import BundleStore
from iris.cluster.worker.dashboard import WorkerDashboard
from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.runtime.types import ContainerPhase, ContainerStats, ContainerStatus
from iris.cluster.worker.service import WorkerServiceImpl
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.rpc import cluster_pb2
from starlette.testclient import TestClient
from tests.test_utils import wait_for_condition

# ============================================================================
# Shared fixtures
# ============================================================================


@pytest.fixture
def mock_bundle_store(tmp_path):
    """Create mock BundleStore with a real temp directory."""
    cache = Mock(spec=BundleStore)
    cache.extract_bundle_to = Mock()
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
    handle.status = Mock(return_value=ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=0))
    handle.stop = Mock()
    handle.logs = Mock(return_value=[])
    handle.stats = Mock(return_value=ContainerStats(memory_mb=100, cpu_percent=50, process_count=1, available=True))
    handle.disk_usage_mb = Mock(return_value=0)
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
def worker(mock_bundle_store, mock_runtime, tmp_path):
    """Create Worker with mocked dependencies."""
    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
        cache_dir=tmp_path / "cache",
        default_task_image="mock-image",
    )
    return Worker(
        config,
        bundle_store=mock_bundle_store,
        container_runtime=mock_runtime,
    )


def create_run_task_request(
    task_id: str = JobName.root("test-user", "test-job-1").task(0).to_wire(),
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

    resources = cluster_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=4 * 1024**3)

    request = cluster_pb2.Worker.RunTaskRequest(
        task_id=task_id,
        num_tasks=num_tasks,
        entrypoint=entrypoint_proto,
        environment=env_config,
        bundle_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
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
        request = create_run_task_request(task_id=JobName.root("test-user", f"job-{i}").task(0).to_wire())
        worker.submit_task(request)

    response = rpc_post(client, "ListTasks")
    assert response.status_code == 200
    data = response.json()
    tasks = data.get("tasks", [])
    assert len(tasks) == 3

    task_ids = {t["taskId"] for t in tasks}
    assert task_ids == {
        JobName.root("test-user", "job-0").task(0).to_wire(),
        JobName.root("test-user", "job-1").task(0).to_wire(),
        JobName.root("test-user", "job-2").task(0).to_wire(),
    }


def test_get_task_not_found(client):
    """Test GetTaskStatus RPC with nonexistent task returns error."""
    response = rpc_post(
        client,
        "GetTaskStatus",
        {"taskId": JobName.root("test-user", "nonexistent").task(0).to_wire()},
    )
    assert response.status_code != 200


def test_get_task_success(client, worker):
    """Test GetTaskStatus RPC returns task details."""
    task_id = JobName.root("test-user", "job-details").task(0).to_wire()
    request = create_run_task_request(task_id=task_id, ports=["http", "grpc"])
    worker.submit_task(request)

    # Wait for task to complete (mock runtime returns running=False immediately)
    task = worker.get_task(task_id)
    task.thread.join(timeout=5.0)

    response = rpc_post(client, "GetTaskStatus", {"taskId": task_id})
    assert response.status_code == 200
    data = response.json()

    assert data["taskId"] == task_id
    assert (
        JobName.from_wire(data["taskId"]).require_task()[0].to_wire()
        == JobName.root("test-user", "job-details").to_wire()
    )
    assert data["state"] == "TASK_STATE_SUCCEEDED"
    assert data["exitCode"] == 0
    assert "http" in data["ports"]
    assert "grpc" in data["ports"]


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
    task_id = JobName.root("test-user", "job-with-ports").task(0).to_wire()
    request = create_run_task_request(task_id=task_id, ports=["http", "grpc"])
    worker.submit_task(request)

    # Ports are allocated in the task thread during setup, so wait for the
    # task to move past PENDING before checking.
    task = worker.get_task(task_id)
    wait_for_condition(lambda: task.status != cluster_pb2.TASK_STATE_PENDING)

    assert len(task.ports) == 2
    assert "http" in task.ports
    assert "grpc" in task.ports


def test_get_task_status_not_found(service, worker, request_context):
    """Test get_task_status raises NOT_FOUND for nonexistent task."""
    status_request = cluster_pb2.Worker.GetTaskStatusRequest(
        task_id=JobName.root("test-user", "nonexistent").task(0).to_wire()
    )

    with pytest.raises(ConnectError) as exc_info:
        service.get_task_status(status_request, request_context)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in str(exc_info.value)


def test_get_task_status_completed_task(service, worker, request_context):
    """Test get_task_status for completed task includes timing info."""
    task_id = JobName.root("test-user", "job-completed").task(0).to_wire()
    request = create_run_task_request(task_id=task_id)
    worker.submit_task(request)

    # Wait for task to complete
    task = worker.get_task(task_id)
    task.thread.join(timeout=5.0)

    status_request = cluster_pb2.Worker.GetTaskStatusRequest(task_id=task_id)
    status = service.get_task_status(status_request, request_context)

    assert status.task_id == task_id
    assert (
        JobName.from_wire(status.task_id).require_task()[0].to_wire()
        == JobName.root("test-user", "job-completed").to_wire()
    )
    assert status.state == cluster_pb2.TASK_STATE_SUCCEEDED
    assert status.exit_code == 0
    assert status.started_at.epoch_ms > 0
    assert status.finished_at.epoch_ms > 0
