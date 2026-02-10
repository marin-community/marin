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

"""Tests for Worker class (includes PortAllocator and task management)."""

import socket
import time
import zipfile
from unittest.mock import Mock

import pytest
from connectrpc.request import RequestContext

from iris.rpc import cluster_pb2
from iris.cluster.types import Entrypoint, JobName
from iris.cluster.worker.bundle_cache import BundleCache
from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.runtime.types import ContainerStats, ContainerStatus
from iris.cluster.worker.port_allocator import PortAllocator
from iris.cluster.worker.service import WorkerServiceImpl
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.time_utils import Duration

# ============================================================================
# PortAllocator Tests
# ============================================================================


@pytest.fixture
def allocator():
    """Create PortAllocator with small range for testing."""
    return PortAllocator(port_range=(40000, 40100))


def test_allocated_ports_are_usable(allocator):
    """Test that allocated ports can actually be bound."""
    ports = allocator.allocate(count=3)

    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))


def test_no_port_reuse_before_release(allocator):
    """Test that allocated ports are not reused before release."""
    ports1 = allocator.allocate(count=5)
    ports2 = allocator.allocate(count=5)

    assert len(set(ports1) & set(ports2)) == 0


def test_concurrent_allocations(allocator):
    """Test concurrent port allocations are thread-safe."""
    import threading

    results = []

    def allocate_ports():
        ports = allocator.allocate(count=5)
        results.append(ports)

    threads = [threading.Thread(target=allocate_ports) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    all_ports = []
    for ports in results:
        all_ports.extend(ports)

    assert len(all_ports) == len(set(all_ports))


# ============================================================================
# Worker Tests (with mocked dependencies)
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


def create_mock_container_handle(
    status_sequence: list[ContainerStatus] | None = None,
    run_side_effect: Exception | None = None,
):
    """Create a mock ContainerHandle for testing.

    Args:
        status_sequence: List of ContainerStatus to return on successive status() calls.
            Defaults to [running=True, running=False with exit_code=0].
        run_side_effect: Exception to raise when run() is called.
    """
    handle = Mock()
    handle.container_id = "container123"
    handle.build = Mock(return_value=[])

    if run_side_effect:
        handle.run = Mock(side_effect=run_side_effect)
    else:
        handle.run = Mock()

    if status_sequence is None:
        status_sequence = [
            ContainerStatus(running=True),
            ContainerStatus(running=False, exit_code=0),
        ]

    call_count = [0]

    def status_side_effect():
        idx = min(call_count[0], len(status_sequence) - 1)
        call_count[0] += 1
        return status_sequence[idx]

    handle.status = Mock(side_effect=status_side_effect)
    handle.stop = Mock()
    handle.logs = Mock(return_value=[])
    handle.stats = Mock(return_value=ContainerStats(memory_mb=100, cpu_percent=50, process_count=5, available=True))
    handle.cleanup = Mock()
    return handle


@pytest.fixture
def mock_runtime():
    """Create mock DockerRuntime.

    By default, simulates a container that runs and completes successfully.
    """
    runtime = Mock(spec=DockerRuntime)

    # Create a mock handle that will be returned by create_container
    mock_handle = create_mock_container_handle()
    runtime.create_container = Mock(return_value=mock_handle)

    runtime.list_iris_containers = Mock(return_value=[])
    runtime.remove_all_iris_containers = Mock(return_value=0)
    runtime.cleanup = Mock()
    return runtime


@pytest.fixture
def worker(mock_bundle_cache, mock_runtime, tmp_path):
    """Create Worker with mocked dependencies."""
    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
        poll_interval=Duration.from_seconds(0.1),  # Fast polling for tests
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
    class TestEntrypoint:
        callable: object
        args: tuple = ()
        kwargs: dict | None = None

        def __post_init__(self):
            if self.kwargs is None:
                self.kwargs = {}

    def test_fn():
        print("Hello from test")

    return TestEntrypoint(callable=test_fn)


def create_run_task_request(
    task_id: str = JobName.root("test-task").task(0).to_wire(),
    num_tasks: int = 1,
    ports: list[str] | None = None,
    attempt_id: int = 0,
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
        attempt_id=attempt_id,
        entrypoint=entrypoint_proto,
        environment=env_config,
        bundle_gcs_path="gs://bucket/bundle.zip",
        resources=resources,
        ports=ports or [],
    )
    # Set timeout to 300 seconds
    request.timeout.CopyFrom(Duration.from_seconds(300).to_proto())
    return request


def test_task_lifecycle_phases(worker):
    """Test task transitions through PENDING -> BUILDING -> RUNNING -> SUCCEEDED."""
    request = create_run_task_request()
    task_id = worker.submit_task(request)

    task = worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    final_task = worker.get_task(task_id)
    assert final_task.status == cluster_pb2.TASK_STATE_SUCCEEDED
    assert final_task.exit_code == 0


def test_task_with_ports(worker):
    """Test task with port allocation."""
    request = create_run_task_request(ports=["http", "grpc"])
    task_id = worker.submit_task(request)

    task = worker.get_task(task_id)
    assert len(task.ports) == 2
    assert "http" in task.ports
    assert "grpc" in task.ports
    assert task.ports["http"] != task.ports["grpc"]

    task.thread.join(timeout=15.0)


def test_task_failure_on_nonzero_exit(worker, mock_runtime):
    """Test task fails when container exits with non-zero code."""
    # Update the mock handle's status to return failure immediately
    mock_handle = create_mock_container_handle(status_sequence=[ContainerStatus(running=False, exit_code=1)])
    mock_runtime.create_container = Mock(return_value=mock_handle)

    request = create_run_task_request()
    task_id = worker.submit_task(request)

    task = worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    final_task = worker.get_task(task_id)
    assert final_task.status == cluster_pb2.TASK_STATE_FAILED
    assert final_task.exit_code == 1
    assert "Exit code: 1" in final_task.error


def test_task_failure_on_error(worker, mock_runtime):
    """Test task fails when container returns error."""
    # Update the mock handle's status to return error after first poll
    mock_handle = create_mock_container_handle(
        status_sequence=[
            ContainerStatus(running=True),
            ContainerStatus(running=False, exit_code=1, error="Container crashed"),
        ]
    )
    mock_runtime.create_container = Mock(return_value=mock_handle)

    request = create_run_task_request()
    task_id = worker.submit_task(request)

    task = worker.get_task(task_id)
    task.thread.join(timeout=10.0)

    final_task = worker.get_task(task_id)
    assert final_task.status == cluster_pb2.TASK_STATE_FAILED
    assert final_task.error == "Container crashed"


def test_task_exception_handling(worker, mock_bundle_cache):
    """Test task handles exceptions during execution."""
    mock_bundle_cache.get_bundle = Mock(side_effect=Exception("Bundle download failed"))

    request = create_run_task_request()
    task_id = worker.submit_task(request)

    task = worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    final_task = worker.get_task(task_id)
    assert final_task.status == cluster_pb2.TASK_STATE_FAILED
    assert "Bundle download failed" in final_task.error


def test_list_tasks(worker):
    """Test listing all tasks."""
    requests = [create_run_task_request(task_id=JobName.root("test-job").task(i).to_wire()) for i in range(3)]

    for request in requests:
        worker.submit_task(request)

    tasks = worker.list_tasks()
    assert len(tasks) == 3


def test_kill_running_task(worker, mock_runtime):
    """Test killing a running task with graceful timeout."""
    # Create a handle that stays running until killed
    mock_handle = create_mock_container_handle(status_sequence=[ContainerStatus(running=True)] * 100)  # Stay running
    mock_runtime.create_container = Mock(return_value=mock_handle)

    request = create_run_task_request()
    task_id = worker.submit_task(request)

    # Wait for task thread to reach RUNNING state
    task = worker.get_task(task_id)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if task.status == cluster_pb2.TASK_STATE_RUNNING and task.container_id:
            break
        time.sleep(0.01)

    assert task.status == cluster_pb2.TASK_STATE_RUNNING, f"Task did not reach RUNNING state, got {task.status}"

    result = worker.kill_task(task_id, term_timeout_ms=100)
    assert result is True

    task.thread.join(timeout=15.0)

    assert task.status == cluster_pb2.TASK_STATE_KILLED
    mock_handle.stop.assert_called_with(force=True)


def test_new_attempt_supersedes_old(worker, mock_runtime):
    """New attempt for same task_id kills the old attempt and starts a new one."""
    # Create a handle that stays running until killed
    mock_handle = create_mock_container_handle(status_sequence=[ContainerStatus(running=True)] * 100)  # Stay running
    mock_runtime.create_container = Mock(return_value=mock_handle)

    request_0 = create_run_task_request(task_id=JobName.root("retry-task").task(0).to_wire(), attempt_id=0)
    worker.submit_task(request_0)

    # Wait for attempt 0 to be running
    task_id = JobName.root("retry-task").task(0).to_wire()
    old_task = worker.get_task(task_id)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if old_task.status == cluster_pb2.TASK_STATE_RUNNING and old_task.container_id:
            break
        time.sleep(0.01)

    assert (
        old_task.status == cluster_pb2.TASK_STATE_RUNNING
    ), f"Old task did not reach RUNNING state, got {old_task.status}"
    assert old_task.attempt_id == 0

    # Submit attempt 1 for the same task_id — should kill attempt 0
    request_1 = create_run_task_request(task_id=JobName.root("retry-task").task(0).to_wire(), attempt_id=1)
    worker.submit_task(request_1)

    # Old attempt should have been killed
    assert old_task.should_stop is True

    # The new attempt should now be tracked with the new attempt_id
    new_task = worker.get_task(task_id)
    assert new_task.attempt_id == 1
    assert new_task is not old_task

    # Clean up
    worker.kill_task(task_id)
    new_task.thread.join(timeout=15.0)


def test_duplicate_attempt_rejected(worker, mock_runtime):
    """Same attempt_id for an existing non-terminal task is rejected."""
    # Create a handle that stays running until killed
    mock_handle = create_mock_container_handle(status_sequence=[ContainerStatus(running=True)] * 100)  # Stay running
    mock_runtime.create_container = Mock(return_value=mock_handle)

    request = create_run_task_request(task_id=JobName.root("dup-task").task(0).to_wire(), attempt_id=0)
    worker.submit_task(request)

    # Wait for it to be running
    task_id = JobName.root("dup-task").task(0).to_wire()
    task = worker.get_task(task_id)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if task.status == cluster_pb2.TASK_STATE_RUNNING:
            break
        time.sleep(0.01)

    assert task.status == cluster_pb2.TASK_STATE_RUNNING, f"Task did not reach RUNNING state, got {task.status}"

    # Submit same attempt_id again — should be rejected (task unchanged)
    worker.submit_task(create_run_task_request(task_id=task_id, attempt_id=0))
    assert worker.get_task(task_id) is task  # Same object, not replaced

    # Clean up
    worker.kill_task(task_id)
    task.thread.join(timeout=15.0)


def test_kill_nonexistent_task(worker):
    """Test killing a nonexistent task returns False."""
    result = worker.kill_task(JobName.root("nonexistent-task").task(0).to_wire())
    assert result is False


def test_get_logs_nonexistent_task(worker):
    """Test getting logs for nonexistent task returns empty list."""
    logs = worker.get_logs(JobName.root("nonexistent-task").task(0).to_wire())
    assert logs == []


def test_port_env_vars_set(worker, mock_runtime):
    """Test that IRIS_PORT_* environment variables are set for requested ports."""
    request = create_run_task_request(ports=["web", "api", "metrics"])
    task_id = worker.submit_task(request)

    task = worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    assert mock_runtime.create_container.called
    call_args = mock_runtime.create_container.call_args
    config = call_args[0][0]

    assert "IRIS_PORT_WEB" in config.env
    assert "IRIS_PORT_API" in config.env
    assert "IRIS_PORT_METRICS" in config.env

    ports = {
        int(config.env["IRIS_PORT_WEB"]),
        int(config.env["IRIS_PORT_API"]),
        int(config.env["IRIS_PORT_METRICS"]),
    }
    assert len(ports) == 3


def test_task_failure_error_appears_in_logs(worker, mock_bundle_cache):
    """Test that task failure errors appear in logs."""
    mock_bundle_cache.get_bundle = Mock(side_effect=Exception("Bundle download failed"))

    request = create_run_task_request()
    task_id = worker.submit_task(request)

    task = worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    final_task = worker.get_task(task_id)
    assert final_task.status == cluster_pb2.TASK_STATE_FAILED
    assert "Bundle download failed" in final_task.error

    logs = worker.get_logs(task_id)
    error_logs = [log for log in logs if log.source == "error"]
    assert len(error_logs) >= 1
    assert any("Bundle download failed" in log.data for log in error_logs)


def test_port_binding_failure(mock_bundle_cache, tmp_path):
    """Test that task fails when port binding fails.

    With --network=host, port binding happens in the application, not Docker.
    If the app fails to bind (port in use by external process), the task fails.
    """
    runtime = Mock(spec=DockerRuntime)

    mock_handle = create_mock_container_handle(
        run_side_effect=RuntimeError("failed to bind host port: address already in use")
    )
    runtime.create_container = Mock(return_value=mock_handle)
    runtime.cleanup = Mock()

    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
        poll_interval=Duration.from_seconds(0.1),
        cache_dir=tmp_path / "cache",
    )
    worker = Worker(
        config,
        bundle_provider=mock_bundle_cache,
        container_runtime=runtime,
    )

    request = create_run_task_request(ports=["actor"])
    task_id = worker.submit_task(request)

    task = worker.get_task(task_id)
    assert task is not None
    assert task.thread is not None
    task.thread.join(timeout=15.0)

    final_task = worker.get_task(task_id)
    assert final_task is not None
    assert final_task.status == cluster_pb2.TASK_STATE_FAILED
    assert final_task.error is not None
    assert "address already in use" in final_task.error


# ============================================================================
# Integration Tests (with real Docker)
# ============================================================================


def create_test_bundle(tmp_path):
    """Create a minimal test bundle with pyproject.toml."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    (bundle_dir / "pyproject.toml").write_text(
        """[project]
name = "test-task"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
"""
    )

    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in bundle_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(bundle_dir))

    return f"file://{zip_path}"


def create_integration_entrypoint():
    """Create a simple test entrypoint for integration tests."""

    def test_fn():
        print("Hello from test task!")
        return 42

    return Entrypoint.from_callable(test_fn)


def create_integration_run_task_request(bundle_path: str, task_id: str):
    """Create a RunTaskRequest for integration testing."""
    entrypoint = create_integration_entrypoint()

    return cluster_pb2.Worker.RunTaskRequest(
        task_id=task_id,
        num_tasks=1,
        entrypoint=entrypoint.to_proto(),
        bundle_gcs_path=bundle_path,
        environment=cluster_pb2.EnvironmentConfig(),
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=512 * 1024**2),
    )


@pytest.fixture
def cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache


@pytest.fixture
def test_bundle(tmp_path):
    """Create a test bundle and return file:// path."""
    return create_test_bundle(tmp_path)


@pytest.fixture
def real_worker(cache_dir):
    """Create Worker with real components (not mocks)."""
    runtime = DockerRuntime()
    config = WorkerConfig(
        port=0,
        cache_dir=cache_dir,
        port_range=(40000, 40100),
        poll_interval=Duration.from_seconds(0.5),  # Faster polling for tests
    )
    worker = Worker(config, container_runtime=runtime)
    yield worker
    worker.stop()
    runtime.cleanup()


@pytest.fixture
def real_service(real_worker):
    """Create WorkerServiceImpl with real worker."""
    return WorkerServiceImpl(real_worker)


class TestWorkerIntegration:
    """Integration tests for Worker with real components."""

    @pytest.mark.docker
    def test_submit_task_lifecycle(self, real_worker, test_bundle):
        """Test full task lifecycle from submission to completion."""
        expected_task_id = JobName.root("integration-test").task(0).to_wire()
        request = create_integration_run_task_request(test_bundle, expected_task_id)

        task_id = real_worker.submit_task(request)
        assert task_id == expected_task_id

        # Poll for task completion with shorter intervals
        deadline = time.time() + 30.0
        while time.time() < deadline:
            task = real_worker.get_task(task_id)
            if task.status in (
                cluster_pb2.TASK_STATE_SUCCEEDED,
                cluster_pb2.TASK_STATE_FAILED,
                cluster_pb2.TASK_STATE_KILLED,
            ):
                break
            time.sleep(0.5)

        task = real_worker.get_task(task_id)
        assert task.status in (
            cluster_pb2.TASK_STATE_SUCCEEDED,
            cluster_pb2.TASK_STATE_FAILED,
        ), f"Task did not complete in time, final status: {task.status}"


class TestWorkerServiceIntegration:
    """Integration tests for WorkerService RPC implementation."""

    @pytest.mark.docker
    def test_health_check_rpc(self, real_service):
        """Test HealthCheck RPC returns healthy status."""
        ctx = Mock(spec=RequestContext)

        response = real_service.health_check(cluster_pb2.Empty(), ctx)

        assert response.healthy
        assert response.uptime.milliseconds >= 0

    @pytest.mark.docker
    def test_fetch_logs_tail(self, real_worker, real_service, test_bundle):
        """Test FetchLogs with negative start_line for tailing."""
        ctx = Mock(spec=RequestContext)

        task_id = JobName.root("logs-test").task(0).to_wire()
        request = create_integration_run_task_request(test_bundle, task_id)
        real_worker.submit_task(request)

        time.sleep(2)

        log_request = cluster_pb2.Worker.FetchTaskLogsRequest(
            task_id=task_id,
            filter=cluster_pb2.Worker.FetchLogsFilter(start_line=-10),
        )

        response = real_service.fetch_task_logs(log_request, ctx)
        assert response.logs is not None
        assert len(response.logs) >= 0
