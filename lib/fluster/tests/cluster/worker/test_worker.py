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

"""Tests for Worker class (includes PortAllocator and job management)."""

import socket
import subprocess
import time
import zipfile
from pathlib import Path
from unittest.mock import Mock

import cloudpickle
import pytest
from connectrpc.request import RequestContext

from fluster.rpc import cluster_pb2
from fluster.cluster.types import Entrypoint
from fluster.cluster.worker.builder import BuildResult, VenvCache
from fluster.cluster.worker.worker_types import Job as WorkerJob
from fluster.cluster.worker.bundle import BundleCache
from fluster.cluster.worker.docker import ContainerConfig, ContainerStats, ContainerStatus, DockerRuntime, ImageBuilder
from fluster.cluster.worker.service import WorkerServiceImpl
from fluster.cluster.worker.worker import PortAllocator, Worker, WorkerConfig

# ============================================================================
# PortAllocator Tests
# ============================================================================


@pytest.fixture
def allocator():
    """Create PortAllocator with small range for testing."""
    return PortAllocator(port_range=(40000, 40100))


def test_allocate_single_port(allocator):
    """Test allocating a single port."""
    ports = allocator.allocate(count=1)
    assert len(ports) == 1
    assert 40000 <= ports[0] < 40100


def test_allocate_multiple_ports(allocator):
    """Test allocating multiple ports at once."""
    ports = allocator.allocate(count=5)
    assert len(ports) == 5
    assert len(set(ports)) == 5  # All unique
    for port in ports:
        assert 40000 <= port < 40100


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


def test_release_nonexistent_port(allocator):
    """Test that releasing a non-allocated port doesn't cause errors."""
    allocator.release([99999])


def test_default_port_range():
    """Test default port range is 30000-40000."""
    allocator = PortAllocator()
    ports = allocator.allocate(count=5)

    for port in ports:
        assert 30000 <= port < 40000


# ============================================================================
# Worker Job (worker_types) Tests
# ============================================================================


def test_worker_job_to_proto_roundtrip():
    """Test WorkerJob.to_proto() preserves all fields including attempt_id."""
    request = cluster_pb2.Worker.RunJobRequest(
        job_id="test-job",
        serialized_entrypoint=b"test",
    )

    job = WorkerJob(
        job_id="test-job",
        attempt_id=3,
        request=request,
        status=cluster_pb2.JOB_STATE_RUNNING,
        exit_code=None,
        error=None,
        started_at_ms=1000,
        finished_at_ms=None,
        ports={"http": 8080, "grpc": 9090},
        status_message="Building image",
        current_memory_mb=512,
        peak_memory_mb=1024,
        current_cpu_percent=50,
        process_count=5,
        disk_mb=100,
        build_started_ms=500,
        build_finished_ms=900,
        build_from_cache=True,
        image_tag="test:v1",
    )

    proto = job.to_proto()

    # Core fields
    assert proto.job_id == "test-job"
    assert proto.current_attempt_id == 3
    assert proto.state == cluster_pb2.JOB_STATE_RUNNING
    assert proto.started_at_ms == 1000
    assert proto.finished_at_ms == 0
    assert proto.status_message == "Building image"
    assert dict(proto.ports) == {"http": 8080, "grpc": 9090}

    # Resource usage
    assert proto.resource_usage.memory_mb == 512
    assert proto.resource_usage.memory_peak_mb == 1024
    assert proto.resource_usage.cpu_percent == 50
    assert proto.resource_usage.process_count == 5
    assert proto.resource_usage.disk_mb == 100

    # Build metrics
    assert proto.build_metrics.build_started_ms == 500
    assert proto.build_metrics.build_finished_ms == 900
    assert proto.build_metrics.from_cache is True
    assert proto.build_metrics.image_tag == "test:v1"


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
def mock_venv_cache():
    """Create mock VenvCache."""
    cache = Mock(spec=VenvCache)
    cache.compute_deps_hash = Mock(return_value="abc123")
    return cache


@pytest.fixture
def mock_image_cache():
    """Create mock ImageBuilder."""
    builder = Mock(spec=ImageBuilder)
    builder.build = Mock(
        return_value=BuildResult(
            image_tag="test-image:latest",
            deps_hash="abc123",
            build_time_ms=1000,
            from_cache=False,
        )
    )
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
    return runtime


@pytest.fixture
def worker(mock_bundle_cache, mock_venv_cache, mock_image_cache, mock_runtime):
    """Create Worker with mocked dependencies."""
    config = WorkerConfig(
        port=0,
        max_concurrent_jobs=5,
        port_range=(50000, 50100),
        poll_interval_seconds=0.1,  # Fast polling for tests
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


def create_run_job_request(job_id: str = "test-job-1", ports: list[str] | None = None):
    """Create a RunJobRequest for testing."""
    entrypoint = create_test_entrypoint()
    serialized_entrypoint = cloudpickle.dumps(entrypoint)

    env_config = cluster_pb2.EnvironmentConfig(
        workspace="/workspace",
        env_vars={
            "TEST_VAR": "value",
            "JOB_VAR": "job_value",
        },
        extras=["dev"],
    )

    resources = cluster_pb2.ResourceSpec(
        cpu=2,
        memory="4g",
    )

    return cluster_pb2.Worker.RunJobRequest(
        job_id=job_id,
        serialized_entrypoint=serialized_entrypoint,
        environment=env_config,
        bundle_gcs_path="gs://bucket/bundle.zip",
        resources=resources,
        timeout_seconds=300,
        ports=ports or [],
    )


def test_submit_job_returns_job_id(worker):
    """Test that submit_job returns job_id immediately."""
    request = create_run_job_request()
    job_id = worker.submit_job(request)

    assert job_id == "test-job-1"

    job = worker.get_job(job_id)
    assert job is not None
    assert job.job_id == job_id


def test_job_lifecycle_phases(worker):
    """Test job transitions through PENDING → BUILDING → RUNNING → SUCCEEDED."""
    request = create_run_job_request()
    job_id = worker.submit_job(request)

    job = worker.get_job(job_id)
    job.thread.join(timeout=15.0)

    final_job = worker.get_job(job_id)
    assert final_job.status == cluster_pb2.JOB_STATE_SUCCEEDED
    assert final_job.exit_code == 0


def test_job_with_ports(worker):
    """Test job with port allocation."""
    request = create_run_job_request(ports=["http", "grpc"])
    job_id = worker.submit_job(request)

    job = worker.get_job(job_id)
    assert len(job.ports) == 2
    assert "http" in job.ports
    assert "grpc" in job.ports
    assert job.ports["http"] != job.ports["grpc"]

    job.thread.join(timeout=15.0)


def test_job_failure_on_nonzero_exit(worker, mock_runtime):
    """Test job fails when container exits with non-zero code."""
    mock_runtime.inspect = Mock(return_value=ContainerStatus(running=False, exit_code=1))

    request = create_run_job_request()
    job_id = worker.submit_job(request)

    job = worker.get_job(job_id)
    job.thread.join(timeout=15.0)

    final_job = worker.get_job(job_id)
    assert final_job.status == cluster_pb2.JOB_STATE_FAILED
    assert final_job.exit_code == 1
    assert "Exit code: 1" in final_job.error


def test_job_failure_on_error(worker, mock_runtime):
    """Test job fails when container returns error."""
    call_count = [0]

    def inspect_side_effect(container_id):
        call_count[0] += 1
        if call_count[0] == 1:
            return ContainerStatus(running=True)
        return ContainerStatus(running=False, exit_code=1, error="Container crashed")

    mock_runtime.inspect = Mock(side_effect=inspect_side_effect)

    request = create_run_job_request()
    job_id = worker.submit_job(request)

    job = worker.get_job(job_id)
    job.thread.join(timeout=10.0)

    final_job = worker.get_job(job_id)
    assert final_job.status == cluster_pb2.JOB_STATE_FAILED
    assert final_job.error == "Container crashed"


def test_job_exception_handling(worker, mock_bundle_cache):
    """Test job handles exceptions during execution."""
    mock_bundle_cache.get_bundle = Mock(side_effect=Exception("Bundle download failed"))

    request = create_run_job_request()
    job_id = worker.submit_job(request)

    job = worker.get_job(job_id)
    job.thread.join(timeout=15.0)

    final_job = worker.get_job(job_id)
    assert final_job.status == cluster_pb2.JOB_STATE_FAILED
    assert "Bundle download failed" in final_job.error


def test_list_jobs(worker):
    """Test listing all jobs."""
    requests = [create_run_job_request(job_id=f"job-{i}") for i in range(3)]

    for request in requests:
        worker.submit_job(request)

    jobs = worker.list_jobs()
    assert len(jobs) == 3
    assert {job.job_id for job in jobs} == {"job-0", "job-1", "job-2"}


def test_kill_running_job(worker, mock_runtime):
    """Test killing a running job with graceful timeout."""
    mock_runtime.inspect = Mock(return_value=ContainerStatus(running=True))

    request = create_run_job_request()
    job_id = worker.submit_job(request)

    job = worker.get_job(job_id)
    for _ in range(20):
        if job.status == cluster_pb2.JOB_STATE_RUNNING and job.container_id:
            break
        time.sleep(0.1)

    result = worker.kill_job(job_id, term_timeout_ms=100)
    assert result is True

    job.thread.join(timeout=15.0)

    assert job.status == cluster_pb2.JOB_STATE_KILLED
    mock_runtime.kill.assert_any_call("container123", force=False)


def test_kill_nonexistent_job(worker):
    """Test killing a nonexistent job returns False."""
    result = worker.kill_job("nonexistent-job")
    assert result is False


def test_get_logs_empty(worker):
    """Test getting logs for job immediately after submission."""
    request = create_run_job_request()
    job_id = worker.submit_job(request)

    logs = worker.get_logs(job_id)
    assert isinstance(logs, list)


def test_get_logs_nonexistent_job(worker):
    """Test getting logs for nonexistent job returns empty list."""
    logs = worker.get_logs("nonexistent-job")
    assert logs == []


def test_build_command_with_entrypoint(worker):
    """Test _build_command creates correct cloudpickle command."""
    entrypoint = create_test_entrypoint()
    command = worker._build_command(entrypoint, ports={})

    assert command[0] == "python"
    assert command[1] == "-c"
    assert "cloudpickle" in command[2]
    assert "base64" in command[2]


def test_fray_port_mapping_env_var(worker, mock_runtime):
    """Test that FRAY_PORT_MAPPING environment variable is set with port mappings."""
    request = create_run_job_request(ports=["web", "api", "metrics"])
    job_id = worker.submit_job(request)

    job = worker.get_job(job_id)
    job.thread.join(timeout=15.0)

    assert mock_runtime.create_container.called
    call_args = mock_runtime.create_container.call_args
    config = call_args[0][0]

    assert "FRAY_PORT_MAPPING" in config.env

    port_mapping = config.env["FRAY_PORT_MAPPING"]
    mappings = {}
    for pair in port_mapping.split(","):
        name, port = pair.split(":")
        mappings[name] = int(port)

    assert set(mappings.keys()) == {"web", "api", "metrics"}
    assert len(set(mappings.values())) == 3

    assert "FLUSTER_PORT_WEB" in config.env
    assert "FLUSTER_PORT_API" in config.env
    assert "FLUSTER_PORT_METRICS" in config.env


def test_fray_port_mapping_not_set_when_no_ports(worker, mock_runtime):
    """Test that FRAY_PORT_MAPPING is not set when no ports are requested."""
    request = create_run_job_request(ports=[])
    job_id = worker.submit_job(request)

    job = worker.get_job(job_id)
    job.thread.join(timeout=15.0)

    assert mock_runtime.create_container.called
    call_args = mock_runtime.create_container.call_args
    config = call_args[0][0]

    assert "FRAY_PORT_MAPPING" not in config.env


def test_job_failure_error_appears_in_logs(worker, mock_bundle_cache):
    """Test that job failure errors appear in logs."""
    mock_bundle_cache.get_bundle = Mock(side_effect=Exception("Bundle download failed"))

    request = create_run_job_request()
    job_id = worker.submit_job(request)

    job = worker.get_job(job_id)
    job.thread.join(timeout=15.0)

    final_job = worker.get_job(job_id)
    assert final_job.status == cluster_pb2.JOB_STATE_FAILED
    assert "Bundle download failed" in final_job.error

    logs = worker.get_logs(job_id)
    error_logs = [log for log in logs if log.source == "error"]
    assert len(error_logs) >= 1
    assert any("Bundle download failed" in log.data for log in error_logs)


def test_port_retry_on_binding_failure(mock_bundle_cache, mock_venv_cache, mock_image_cache):
    """Test that job retries with new ports when port binding fails."""
    del mock_venv_cache  # unused
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
        max_concurrent_jobs=5,
        port_range=(50000, 50100),
        poll_interval_seconds=0.1,
    )
    worker = Worker(
        config,
        bundle_provider=mock_bundle_cache,
        image_provider=mock_image_cache,
        container_runtime=runtime,
    )

    request = create_run_job_request(ports=["actor"])
    job_id = worker.submit_job(request)

    job = worker.get_job(job_id)
    assert job is not None
    assert job.thread is not None
    job.thread.join(timeout=15.0)

    final_job = worker.get_job(job_id)
    assert final_job is not None
    assert final_job.status == cluster_pb2.JOB_STATE_SUCCEEDED

    assert runtime.start_container.call_count == 2
    assert runtime.remove.call_count == 1

    logs = worker.get_logs(job_id)
    build_logs = [log for log in logs if log.source == "build"]
    assert any("Port conflict" in log.data for log in build_logs)


def test_port_retry_exhausted(mock_bundle_cache, mock_venv_cache, mock_image_cache):
    """Test that job fails after max port retries are exhausted."""
    del mock_venv_cache  # unused
    runtime = Mock(spec=DockerRuntime)
    runtime.create_container = Mock(return_value="container123")
    runtime.start_container = Mock(side_effect=RuntimeError("failed to bind host port: address already in use"))
    runtime.remove = Mock()
    runtime.get_logs = Mock(return_value=[])

    config = WorkerConfig(
        port=0,
        max_concurrent_jobs=5,
        port_range=(50000, 50100),
        poll_interval_seconds=0.1,
    )
    worker = Worker(
        config,
        bundle_provider=mock_bundle_cache,
        image_provider=mock_image_cache,
        container_runtime=runtime,
    )

    request = create_run_job_request(ports=["actor"])
    job_id = worker.submit_job(request)

    job = worker.get_job(job_id)
    assert job is not None
    assert job.thread is not None
    job.thread.join(timeout=15.0)

    final_job = worker.get_job(job_id)
    assert final_job is not None
    assert final_job.status == cluster_pb2.JOB_STATE_FAILED
    assert final_job.error is not None
    assert "address already in use" in final_job.error

    assert runtime.start_container.call_count == 3


# ============================================================================
# Integration Tests (with real Docker)
# ============================================================================


def check_docker_available():
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def create_test_bundle(tmp_path):
    """Create a minimal test bundle with pyproject.toml."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    (bundle_dir / "pyproject.toml").write_text(
        """[project]
name = "test-job"
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
        print("Hello from test job!")
        return 42

    return Entrypoint(callable=test_fn, args=(), kwargs={})


def create_integration_run_job_request(bundle_path: str, job_id: str):
    """Create a RunJobRequest for integration testing."""
    entrypoint = create_integration_entrypoint()

    return cluster_pb2.Worker.RunJobRequest(
        job_id=job_id,
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=bundle_path,
        environment=cluster_pb2.EnvironmentConfig(
            workspace="/app",
        ),
        resources=cluster_pb2.ResourceSpec(
            cpu=1,
            memory="512m",
        ),
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
    config = WorkerConfig(
        port=0,
        cache_dir=cache_dir,
        registry="localhost:5000",
        max_concurrent_jobs=2,
        port_range=(40000, 40100),
        poll_interval_seconds=0.5,  # Faster polling for tests
    )
    return Worker(config)


@pytest.fixture
def real_service(real_worker):
    """Create WorkerServiceImpl with real worker."""
    return WorkerServiceImpl(real_worker)


@pytest.fixture
def runtime():
    """Create DockerRuntime instance."""
    return DockerRuntime()


class TestDockerRuntimeIntegration:
    """Integration tests for DockerRuntime with real containers."""

    @pytest.mark.slow
    def test_create_and_start_container(self, runtime):
        """Create and start a simple container and verify it runs."""
        if not check_docker_available():
            pytest.skip("Docker not available")

        config = ContainerConfig(
            image="alpine:latest",
            command=["echo", "hello"],
            env={},
        )

        container_id = runtime.create_container(config)
        assert container_id is not None

        runtime.start_container(container_id)

        time.sleep(1)

        status = runtime.inspect(container_id)
        assert not status.running
        assert status.exit_code == 0

        runtime.remove(container_id)

    @pytest.mark.slow
    def test_kill_container(self, runtime):
        """Test killing a running container."""
        if not check_docker_available():
            pytest.skip("Docker not available")

        config = ContainerConfig(
            image="alpine:latest",
            command=["sleep", "60"],
            env={},
        )

        container_id = runtime.create_container(config)
        runtime.start_container(container_id)

        time.sleep(1)

        runtime.kill(container_id, force=True)

        status = runtime.inspect(container_id)
        assert not status.running

        runtime.remove(container_id)


class TestWorkerIntegration:
    """Integration tests for Worker with real components."""

    @pytest.mark.slow
    def test_submit_job_lifecycle(self, real_worker, test_bundle):
        """Test full job lifecycle from submission to completion."""
        if not check_docker_available():
            pytest.skip("Docker not available")

        request = create_integration_run_job_request(test_bundle, "integration-test-1")

        job_id = real_worker.submit_job(request)
        assert job_id == "integration-test-1"

        for _ in range(30):
            time.sleep(1)
            job = real_worker.get_job(job_id)

            if job.status in (
                cluster_pb2.JOB_STATE_SUCCEEDED,
                cluster_pb2.JOB_STATE_FAILED,
                cluster_pb2.JOB_STATE_KILLED,
            ):
                break

        job = real_worker.get_job(job_id)
        assert job.status in (
            cluster_pb2.JOB_STATE_SUCCEEDED,
            cluster_pb2.JOB_STATE_FAILED,
        )

    @pytest.mark.slow
    def test_concurrent_job_limit(self, real_worker, test_bundle):
        """Test that max_concurrent_jobs is enforced."""
        if not check_docker_available():
            pytest.skip("Docker not available")

        requests = [create_integration_run_job_request(test_bundle, f"concurrent-{i}") for i in range(4)]

        _job_ids = [real_worker.submit_job(r) for r in requests]

        time.sleep(1)

        jobs = real_worker.list_jobs()
        running = sum(1 for j in jobs if j.status == cluster_pb2.JOB_STATE_RUNNING)

        assert running <= 2


class TestWorkerServiceIntegration:
    """Integration tests for WorkerService RPC implementation."""

    @pytest.mark.slow
    def test_health_check_rpc(self, real_service):
        """Test HealthCheck RPC returns healthy status."""
        ctx = Mock(spec=RequestContext)

        response = real_service.health_check(cluster_pb2.Empty(), ctx)

        assert response.healthy
        assert response.uptime_ms >= 0

    @pytest.mark.slow
    def test_fetch_logs_tail(self, real_service, test_bundle):
        """Test FetchLogs with negative start_line for tailing."""
        if not check_docker_available():
            pytest.skip("Docker not available")

        ctx = Mock(spec=RequestContext)

        request = create_integration_run_job_request(test_bundle, "logs-test")
        real_service.run_job(request, ctx)

        time.sleep(2)

        log_request = cluster_pb2.Worker.FetchLogsRequest(
            job_id="logs-test",
            filter=cluster_pb2.Worker.FetchLogsFilter(start_line=-10),
        )

        response = real_service.fetch_logs(log_request, ctx)
        assert response.logs is not None
        assert len(response.logs) >= 0
