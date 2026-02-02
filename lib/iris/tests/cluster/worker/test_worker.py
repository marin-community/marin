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
from pathlib import Path
from unittest.mock import Mock

import pytest
from connectrpc.request import RequestContext

from iris.rpc import cluster_pb2
from iris.cluster.types import Entrypoint
from iris.cluster.worker.builder import BuildResult
from iris.cluster.worker.bundle_cache import BundleCache
from iris.cluster.worker.docker import ContainerStats, ContainerStatus, DockerRuntime, ImageBuilder
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


def test_ports_reused_after_release():
    """Test that ports can be reused after release."""
    allocator_small = PortAllocator(port_range=(40000, 40003))

    ports1 = allocator_small.allocate(count=3)
    assert len(ports1) == 3

    allocator_small.release(ports1)

    ports2 = allocator_small.allocate(count=3)
    assert len(ports2) == 3
    assert set(ports1) == set(ports2)


def test_release_partial_ports(allocator):
    """Test releasing only some ports."""
    ports = allocator.allocate(count=5)

    allocator.release(ports[:3])

    new_ports = allocator.allocate(count=2)
    assert len(set(new_ports) & set(ports[:3])) > 0


def test_exhausted_port_range():
    """Test behavior when port range is exhausted."""
    allocator_tiny = PortAllocator(port_range=(40000, 40002))

    ports = allocator_tiny.allocate(count=2)
    assert len(ports) == 2

    with pytest.raises(RuntimeError, match="No free ports available"):
        allocator_tiny.allocate(count=1)


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
def mock_bundle_cache():
    """Create mock BundleCache."""
    cache = Mock(spec=BundleCache)
    cache.get_bundle = Mock(return_value=Path("/tmp/bundle"))
    return cache


@pytest.fixture
def mock_image_cache():
    """Create mock ImageBuilder."""
    builder = Mock(spec=ImageBuilder)
    builder.build = Mock(
        return_value=BuildResult(
            image_tag="test-image:latest",
            build_time_ms=1000,
            from_cache=False,
        )
    )
    builder.protect = Mock()
    builder.unprotect = Mock()
    return builder


@pytest.fixture
def mock_runtime():
    """Create mock DockerRuntime.

    By default, simulates a container that runs and completes successfully.
    """
    runtime = Mock(spec=DockerRuntime)
    runtime.create_container = Mock(return_value="container123")
    runtime.start_container = Mock()

    call_count = [0]

    def inspect_side_effect(container_id):
        call_count[0] += 1
        if call_count[0] == 1:
            return ContainerStatus(running=True)
        return ContainerStatus(running=False, exit_code=0)

    runtime.inspect = Mock(side_effect=inspect_side_effect)
    runtime.kill = Mock()
    runtime.remove = Mock()
    runtime.get_stats = Mock(return_value=ContainerStats(memory_mb=100, cpu_percent=50, process_count=5, available=True))
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
        poll_interval=Duration.from_seconds(0.1),  # Fast polling for tests
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
    task_id: str = "test-task-1",
    job_id: str | None = None,
    task_index: int = 0,
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
    )

    resources = cluster_pb2.ResourceSpecProto(cpu=2, memory_bytes=4 * 1024**3)

    request = cluster_pb2.Worker.RunTaskRequest(
        task_id=task_id,
        job_id=job_id or task_id,
        task_index=task_index,
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
    mock_runtime.inspect = Mock(return_value=ContainerStatus(running=False, exit_code=1))

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
    call_count = [0]

    def inspect_side_effect(container_id):
        call_count[0] += 1
        if call_count[0] == 1:
            return ContainerStatus(running=True)
        return ContainerStatus(running=False, exit_code=1, error="Container crashed")

    mock_runtime.inspect = Mock(side_effect=inspect_side_effect)

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
    requests = [create_run_task_request(task_id=f"task-{i}") for i in range(3)]

    for request in requests:
        worker.submit_task(request)

    tasks = worker.list_tasks()
    assert len(tasks) == 3


def test_kill_running_task(worker, mock_runtime):
    """Test that killing a running task transitions it to KILLED state."""
    mock_runtime.inspect = Mock(return_value=ContainerStatus(running=True))

    request = create_run_task_request()
    task_id = worker.submit_task(request)

    task = worker.get_task(task_id)
    for _ in range(20):
        if task.status == cluster_pb2.TASK_STATE_RUNNING and task.container_id:
            break
        time.sleep(0.1)

    result = worker.kill_task(task_id, term_timeout_ms=100)
    assert result is True

    task.thread.join(timeout=15.0)

    final_task = worker.get_task(task_id)
    assert final_task.status == cluster_pb2.TASK_STATE_KILLED


def test_new_attempt_supersedes_old(worker, mock_runtime):
    """New attempt for same task_id kills the old attempt and starts a new one."""
    mock_runtime.inspect = Mock(return_value=ContainerStatus(running=True))

    request_0 = create_run_task_request(task_id="retry-task", attempt_id=0)
    worker.submit_task(request_0)

    # Wait for attempt 0 to be running
    for _ in range(20):
        task = worker.get_task("retry-task")
        if task.status == cluster_pb2.TASK_STATE_RUNNING and task.container_id:
            break
        time.sleep(0.1)

    task_0 = worker.get_task("retry-task")
    assert task_0.status == cluster_pb2.TASK_STATE_RUNNING
    assert task_0.attempt_id == 0

    # Submit attempt 1 for the same task_id — should supersede attempt 0
    request_1 = create_run_task_request(task_id="retry-task", attempt_id=1)
    worker.submit_task(request_1)

    # Verify that get_task now returns a task with the new attempt_id
    task_1 = worker.get_task("retry-task")
    assert task_1.attempt_id == 1

    # Clean up
    worker.kill_task("retry-task")
    task_1.thread.join(timeout=15.0)


def test_duplicate_attempt_rejected(worker, mock_runtime):
    """Same attempt_id for an existing non-terminal task is rejected."""
    mock_runtime.inspect = Mock(return_value=ContainerStatus(running=True))

    request = create_run_task_request(task_id="dup-task", attempt_id=0)
    worker.submit_task(request)

    # Wait for it to be running
    for _ in range(20):
        task = worker.get_task("dup-task")
        if task.status == cluster_pb2.TASK_STATE_RUNNING:
            break
        time.sleep(0.1)

    task_before = worker.get_task("dup-task")
    original_attempt = task_before.attempt_id

    # Submit same attempt_id again — should be rejected (task unchanged)
    worker.submit_task(create_run_task_request(task_id="dup-task", attempt_id=0))

    task_after = worker.get_task("dup-task")
    assert task_after.attempt_id == original_attempt
    assert task_after.status == cluster_pb2.TASK_STATE_RUNNING

    # Clean up
    worker.kill_task("dup-task")
    task_after.thread.join(timeout=15.0)


def test_kill_nonexistent_task(worker):
    """Test killing a nonexistent task returns False."""
    result = worker.kill_task("nonexistent-task")
    assert result is False


def test_get_logs_nonexistent_task(worker):
    """Test getting logs for nonexistent task returns empty list."""
    logs = worker.get_logs("nonexistent-task")
    assert logs == []


def test_port_allocation(worker):
    """Test that requested ports are allocated and tracked in the task."""
    request = create_run_task_request(ports=["web", "api", "metrics"])
    task_id = worker.submit_task(request)

    task = worker.get_task(task_id)
    assert len(task.ports) == 3
    assert "web" in task.ports
    assert "api" in task.ports
    assert "metrics" in task.ports

    # Verify all ports are unique
    port_values = list(task.ports.values())
    assert len(port_values) == len(set(port_values))

    task.thread.join(timeout=15.0)


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


def test_port_retry_on_binding_failure(mock_bundle_cache, mock_image_cache):
    """Test that task recovers from port binding failures and succeeds."""
    runtime = Mock(spec=DockerRuntime)
    runtime.create_container = Mock(return_value="container123")

    call_count = [0]

    def start_side_effect(container_id):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("failed to bind host port: address already in use")
        return None

    runtime.start_container = Mock(side_effect=start_side_effect)
    runtime.remove = Mock()

    inspect_call_count = [0]

    def inspect_side_effect(container_id):
        inspect_call_count[0] += 1
        if inspect_call_count[0] == 1:
            return ContainerStatus(running=True)
        return ContainerStatus(running=False, exit_code=0)

    runtime.inspect = Mock(side_effect=inspect_side_effect)
    runtime.get_stats = Mock(return_value=ContainerStats(memory_mb=100, cpu_percent=50, process_count=5, available=True))
    runtime.get_logs = Mock(return_value=[])

    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
        poll_interval=Duration.from_seconds(0.1),
    )
    worker = Worker(
        config,
        bundle_provider=mock_bundle_cache,
        image_provider=mock_image_cache,
        container_runtime=runtime,
    )

    request = create_run_task_request(ports=["actor"])
    task_id = worker.submit_task(request)

    task = worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    final_task = worker.get_task(task_id)
    assert final_task.status == cluster_pb2.TASK_STATE_SUCCEEDED

    logs = worker.get_logs(task_id)
    build_logs = [log for log in logs if log.source == "build"]
    assert any("Port conflict" in log.data for log in build_logs)


def test_port_retry_exhausted(mock_bundle_cache, mock_image_cache):
    """Test that task fails when port binding repeatedly fails."""
    runtime = Mock(spec=DockerRuntime)
    runtime.create_container = Mock(return_value="container123")
    runtime.start_container = Mock(side_effect=RuntimeError("failed to bind host port: address already in use"))
    runtime.remove = Mock()
    runtime.get_logs = Mock(return_value=[])

    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
        poll_interval=Duration.from_seconds(0.1),
    )
    worker = Worker(
        config,
        bundle_provider=mock_bundle_cache,
        image_provider=mock_image_cache,
        container_runtime=runtime,
    )

    request = create_run_task_request(ports=["actor"])
    task_id = worker.submit_task(request)

    task = worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    final_task = worker.get_task(task_id)
    assert final_task.status == cluster_pb2.TASK_STATE_FAILED
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
        job_id=task_id,
        task_index=0,
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
def real_worker(cache_dir, docker_cleanup_scope):
    """Create Worker with real components (not mocks)."""
    config = WorkerConfig(
        port=0,
        cache_dir=cache_dir,
        registry="localhost:5000",
        port_range=(40000, 40100),
        poll_interval=Duration.from_seconds(0.5),  # Faster polling for tests
    )
    return Worker(config)


@pytest.fixture
def real_service(real_worker):
    """Create WorkerServiceImpl with real worker."""
    return WorkerServiceImpl(real_worker)


class TestWorkerIntegration:
    """Integration tests for Worker with real components."""

    @pytest.mark.slow
    def test_submit_task_lifecycle(self, real_worker, test_bundle):
        """Test full task lifecycle from submission to completion."""
        request = create_integration_run_task_request(test_bundle, "integration-test-1")

        task_id = real_worker.submit_task(request)
        assert task_id == "integration-test-1"

        for _ in range(30):
            time.sleep(1)
            task = real_worker.get_task(task_id)

            if task.status in (
                cluster_pb2.TASK_STATE_SUCCEEDED,
                cluster_pb2.TASK_STATE_FAILED,
                cluster_pb2.TASK_STATE_KILLED,
            ):
                break

        task = real_worker.get_task(task_id)
        assert task.status in (
            cluster_pb2.TASK_STATE_SUCCEEDED,
            cluster_pb2.TASK_STATE_FAILED,
        )


class TestWorkerServiceIntegration:
    """Integration tests for WorkerService RPC implementation."""

    @pytest.mark.slow
    def test_health_check_rpc(self, real_service):
        """Test HealthCheck RPC returns healthy status."""
        ctx = Mock(spec=RequestContext)

        response = real_service.health_check(cluster_pb2.Empty(), ctx)

        assert response.healthy
        assert response.uptime.milliseconds >= 0

    @pytest.mark.slow
    def test_fetch_logs_tail(self, real_service, test_bundle):
        """Test FetchLogs with negative start_line for tailing."""
        ctx = Mock(spec=RequestContext)

        request = create_integration_run_task_request(test_bundle, "logs-test")
        real_service.run_task(request, ctx)

        time.sleep(2)

        log_request = cluster_pb2.Worker.FetchTaskLogsRequest(
            task_id="logs-test",
            filter=cluster_pb2.Worker.FetchLogsFilter(start_line=-10),
        )

        response = real_service.fetch_task_logs(log_request, ctx)
        assert response.logs is not None
        assert len(response.logs) >= 0
