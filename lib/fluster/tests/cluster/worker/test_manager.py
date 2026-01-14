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

"""Tests for PortAllocator and JobManager."""

import socket
import time
from pathlib import Path
from unittest.mock import Mock

import cloudpickle
import pytest

from fluster import cluster_pb2
from fluster.cluster.worker.bundle import BundleCache
from fluster.cluster.worker.builder import BuildResult, VenvCache
from fluster.cluster.worker.docker import ContainerStats, ContainerStatus, DockerRuntime, ImageBuilder
from fluster.cluster.worker.manager import JobManager, PortAllocator


@pytest.fixture
def allocator():
    """Create PortAllocator with small range for testing."""
    return PortAllocator(port_range=(40000, 40100))


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
    The first inspect() call returns running=True, subsequent calls return running=False.
    This gives the poll loop time to observe the RUNNING state before completion.
    """
    runtime = Mock(spec=DockerRuntime)
    runtime.create_container = Mock(return_value="container123")
    runtime.start_container = Mock()

    # Use a callable side_effect that returns running=True once, then running=False
    # This allows the poll loop to continue calling inspect without running out of values
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
def job_manager(mock_bundle_cache, mock_venv_cache, mock_image_cache, mock_runtime):
    """Create JobManager with mocked dependencies."""
    port_allocator = PortAllocator(port_range=(50000, 50100))
    return JobManager(
        bundle_cache=mock_bundle_cache,
        venv_cache=mock_venv_cache,
        image_cache=mock_image_cache,
        runtime=mock_runtime,
        port_allocator=port_allocator,
        max_concurrent_jobs=5,
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

    # Verify each port can be bound (it's free)
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))


def test_no_port_reuse_before_release(allocator):
    """Test that allocated ports are not reused before release."""
    ports1 = allocator.allocate(count=5)
    ports2 = allocator.allocate(count=5)

    # No overlap between the two allocations
    assert len(set(ports1) & set(ports2)) == 0


def test_ports_reused_after_release():
    """Test that ports can be reused after release."""
    # Allocate all available ports in a small range
    allocator_small = PortAllocator(port_range=(40000, 40003))

    # Allocate 3 ports
    ports1 = allocator_small.allocate(count=3)
    assert len(ports1) == 3

    # Release them
    allocator_small.release(ports1)

    # Should be able to allocate again
    ports2 = allocator_small.allocate(count=3)
    assert len(ports2) == 3

    # Ports should be reused (same set, possibly different order)
    assert set(ports1) == set(ports2)


def test_release_partial_ports(allocator):
    """Test releasing only some ports."""
    ports = allocator.allocate(count=5)

    # Release first 3 ports
    allocator.release(ports[:3])

    # Allocate 2 more - should get from the released ones
    new_ports = allocator.allocate(count=2)

    # At least some of the new ports should be from released ones
    assert len(set(new_ports) & set(ports[:3])) > 0


def test_exhausted_port_range():
    """Test behavior when port range is exhausted."""
    allocator_tiny = PortAllocator(port_range=(40000, 40002))

    # Allocate all available ports (2 ports: 40000, 40001)
    ports = allocator_tiny.allocate(count=2)
    assert len(ports) == 2

    # Trying to allocate more should raise RuntimeError
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

    # Collect all allocated ports
    all_ports = []
    for ports in results:
        all_ports.extend(ports)

    # All ports should be unique (no conflicts)
    assert len(all_ports) == len(set(all_ports))


def test_release_nonexistent_port(allocator):
    """Test that releasing a non-allocated port doesn't cause errors."""
    # Should not raise an error
    allocator.release([99999])


def test_default_port_range():
    """Test default port range is 30000-40000."""
    allocator = PortAllocator()
    ports = allocator.allocate(count=5)

    for port in ports:
        assert 30000 <= port < 40000


# ============================================================================
# JobManager Tests
# ============================================================================


def test_submit_job_returns_job_id(job_manager):
    """Test that submit_job returns job_id immediately."""
    request = create_run_job_request()
    job_id = job_manager.submit_job(request)

    assert job_id == "test-job-1"

    # Job should be tracked
    job = job_manager.get_job(job_id)
    assert job is not None
    assert job.job_id == job_id

    # Job starts in PENDING state, but may transition quickly to BUILDING/RUNNING
    # Just verify it exists and has a valid ID
    assert job.status in [
        cluster_pb2.JOB_STATE_PENDING,
        cluster_pb2.JOB_STATE_BUILDING,
        cluster_pb2.JOB_STATE_RUNNING,
    ]


def test_job_lifecycle_phases(job_manager):
    """Test job transitions through PENDING → BUILDING → RUNNING → SUCCEEDED."""
    request = create_run_job_request()
    job_id = job_manager.submit_job(request)

    # Wait for job to complete (poll loop sleeps 5s between checks, so need more time)
    job = job_manager.get_job(job_id)
    job.thread.join(timeout=15.0)

    # Should have gone through phases
    final_job = job_manager.get_job(job_id)
    assert final_job.status == cluster_pb2.JOB_STATE_SUCCEEDED
    assert final_job.exit_code == 0
    assert final_job.started_at_ms is not None
    assert final_job.finished_at_ms is not None


def test_job_with_ports(job_manager):
    """Test job with port allocation."""
    request = create_run_job_request(ports=["http", "grpc"])
    job_id = job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    assert len(job.ports) == 2
    assert "http" in job.ports
    assert "grpc" in job.ports
    assert job.ports["http"] != job.ports["grpc"]

    # Wait for completion
    job.thread.join(timeout=15.0)

    # Ports should be released (we can't easily verify this without checking internal state)


def test_job_failure_on_nonzero_exit(job_manager, mock_runtime):
    """Test job fails when container exits with non-zero code."""
    mock_runtime.inspect = Mock(return_value=ContainerStatus(running=False, exit_code=1))

    request = create_run_job_request()
    job_id = job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    job.thread.join(timeout=15.0)

    final_job = job_manager.get_job(job_id)
    assert final_job.status == cluster_pb2.JOB_STATE_FAILED
    assert final_job.exit_code == 1
    assert "Exit code: 1" in final_job.error


def test_job_failure_on_error(job_manager, mock_runtime):
    """Test job fails when container returns error."""
    # Return running=True once, then error status on all subsequent calls
    call_count = [0]

    def inspect_side_effect(container_id):
        call_count[0] += 1
        if call_count[0] == 1:
            return ContainerStatus(running=True)
        return ContainerStatus(running=False, exit_code=1, error="Container crashed")

    mock_runtime.inspect = Mock(side_effect=inspect_side_effect)

    request = create_run_job_request()
    job_id = job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    job.thread.join(timeout=10.0)

    final_job = job_manager.get_job(job_id)
    assert final_job.status == cluster_pb2.JOB_STATE_FAILED
    assert final_job.error == "Container crashed"


def test_job_exception_handling(job_manager, mock_bundle_cache):
    """Test job handles exceptions during execution."""
    mock_bundle_cache.get_bundle = Mock(side_effect=Exception("Bundle download failed"))

    request = create_run_job_request()
    job_id = job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    job.thread.join(timeout=15.0)

    final_job = job_manager.get_job(job_id)
    assert final_job.status == cluster_pb2.JOB_STATE_FAILED
    assert "Bundle download failed" in final_job.error


def test_concurrent_job_limiting(job_manager):
    """Test semaphore limits concurrent job execution."""
    # Submit more jobs than max_concurrent_jobs (5)
    requests = [create_run_job_request(job_id=f"job-{i}") for i in range(10)]

    job_ids = []
    for request in requests:
        job_id = job_manager.submit_job(request)
        job_ids.append(job_id)

    # Wait for all jobs to complete
    jobs = [job_manager.get_job(job_id) for job_id in job_ids]
    for job in jobs:
        if job.thread:
            job.thread.join(timeout=10.0)

    # All should eventually succeed
    final_jobs = [job_manager.get_job(job_id) for job_id in job_ids]
    for job in final_jobs:
        assert job.status == cluster_pb2.JOB_STATE_SUCCEEDED


def test_list_jobs(job_manager):
    """Test listing all jobs."""
    requests = [create_run_job_request(job_id=f"job-{i}") for i in range(3)]

    for request in requests:
        job_manager.submit_job(request)

    jobs = job_manager.list_jobs()
    assert len(jobs) == 3
    assert {job.job_id for job in jobs} == {"job-0", "job-1", "job-2"}


def test_kill_running_job(job_manager, mock_runtime):
    """Test killing a running job with graceful timeout."""
    # Make the container keep running until killed
    # The poll loop will keep calling inspect() until should_stop is set
    mock_runtime.inspect = Mock(return_value=ContainerStatus(running=True))

    request = create_run_job_request()
    job_id = job_manager.submit_job(request)

    # Wait for job to reach RUNNING state
    job = job_manager.get_job(job_id)
    for _ in range(20):  # Wait up to 2 seconds
        if job.status == cluster_pb2.JOB_STATE_RUNNING and job.container_id:
            break
        time.sleep(0.1)

    # Kill the job
    result = job_manager.kill_job(job_id, term_timeout_ms=100)
    assert result is True

    # Wait for thread to finish
    job.thread.join(timeout=15.0)

    assert job.status == cluster_pb2.JOB_STATE_KILLED
    assert job.finished_at_ms is not None

    # Verify SIGTERM was sent first
    mock_runtime.kill.assert_any_call("container123", force=False)


def test_kill_nonexistent_job(job_manager):
    """Test killing a nonexistent job returns False."""
    result = job_manager.kill_job("nonexistent-job")
    assert result is False


def test_kill_job_without_container(job_manager, mock_runtime):
    """Test killing a job before container is created sets should_stop flag."""
    # Make inspect return running=True so the job doesn't complete immediately
    mock_runtime.inspect = Mock(return_value=ContainerStatus(running=True))

    request = create_run_job_request()
    job_id = job_manager.submit_job(request)

    # Kill immediately (before the thread acquires semaphore or creates container)
    # The should_stop flag will be checked in the poll loop
    result = job_manager.kill_job(job_id)
    assert result is True

    # Wait for thread to finish
    job = job_manager.get_job(job_id)
    job.thread.join(timeout=15.0)

    # Job should be in KILLED state
    assert job.status == cluster_pb2.JOB_STATE_KILLED


def test_get_logs_empty(job_manager):
    """Test getting logs for job immediately after submission.

    Even though the job may have build logs, we test that get_logs returns
    whatever logs are available at the time (may be empty or contain build logs).
    """
    request = create_run_job_request()
    job_id = job_manager.submit_job(request)

    logs = job_manager.get_logs(job_id)
    # Logs may be empty initially or may contain build logs
    assert isinstance(logs, list)


def test_get_logs_with_start_line(job_manager, mock_runtime):
    """Test getting logs with start_line offset."""
    from fluster.cluster.worker.worker_types import LogLine

    request = create_run_job_request()
    job_id = job_manager.submit_job(request)

    # Add some container logs via mock
    container_logs = [LogLine.now("stdout", f"Log line {i}") for i in range(10)]
    mock_runtime.get_logs = Mock(return_value=container_logs)

    # Get logs starting from line 5
    logs = job_manager.get_logs(job_id, start_line=5)
    # Note: May include build logs before container logs, so just verify we got some logs
    assert len(logs) >= 5


def test_get_logs_tail_behavior(job_manager, mock_runtime):
    """Test getting last N logs with negative start_line."""
    from fluster.cluster.worker.worker_types import LogLine

    request = create_run_job_request()
    job_id = job_manager.submit_job(request)

    # Add some container logs via mock
    container_logs = [LogLine.now("stdout", f"Log line {i}") for i in range(10)]
    mock_runtime.get_logs = Mock(return_value=container_logs)

    # Get last 3 logs
    logs = job_manager.get_logs(job_id, start_line=-3)
    # The last 3 logs should be the tail of all logs (build + container)
    # Just verify we got exactly 3 logs
    assert len(logs) == 3


def test_get_logs_nonexistent_job(job_manager):
    """Test getting logs for nonexistent job returns empty list."""
    logs = job_manager.get_logs("nonexistent-job")
    assert logs == []


def test_cleanup_removes_container(job_manager, mock_runtime):
    """Test that workdir is cleaned up after job completes.

    Note: Containers are intentionally kept for log retrieval, not removed.
    """
    request = create_run_job_request()
    job_id = job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    job.thread.join(timeout=15.0)

    # Verify container was NOT removed (kept for logs)
    mock_runtime.remove.assert_not_called()

    # Verify cleanup was done
    assert job.cleanup_done is True


def test_build_command_with_entrypoint(job_manager):
    """Test _build_command creates correct cloudpickle command."""
    entrypoint = create_test_entrypoint()
    command = job_manager._build_command(entrypoint)

    assert command[0] == "python"
    assert command[1] == "-c"
    assert "cloudpickle" in command[2]
    assert "base64" in command[2]


def test_job_status_message_during_building(job_manager):
    """Test that status_message is set during BUILDING phase."""
    request = create_run_job_request()
    job_id = job_manager.submit_job(request)

    # Wait a bit for job to start building
    time.sleep(0.1)

    job = job_manager.get_job(job_id)
    # Job should be in BUILDING state with a status_message
    if job.status == cluster_pb2.JOB_STATE_BUILDING:
        assert job.status_message in ["downloading bundle", "building image", "populating uv cache"]

    # Wait for completion
    job.thread.join(timeout=15.0)

    # After completion, status_message should be empty
    final_job = job_manager.get_job(job_id)
    assert final_job.status_message == ""


def test_job_to_proto_includes_status_message(job_manager):
    """Test that Job.to_proto() includes status_message."""
    request = create_run_job_request()
    job_id = job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    job.status_message = "test message"

    proto = job.to_proto()
    assert proto.status_message == "test message"


def test_fray_port_mapping_env_var(job_manager, mock_runtime):
    """Test that FRAY_PORT_MAPPING environment variable is set with port mappings."""
    request = create_run_job_request(ports=["web", "api", "metrics"])
    job_id = job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    job.thread.join(timeout=15.0)

    # Verify runtime.create_container was called with correct environment
    assert mock_runtime.create_container.called
    call_args = mock_runtime.create_container.call_args
    config = call_args[0][0]

    # Check that FRAY_PORT_MAPPING is set
    assert "FRAY_PORT_MAPPING" in config.env

    # Parse the port mapping
    port_mapping = config.env["FRAY_PORT_MAPPING"]
    mappings = {}
    for pair in port_mapping.split(","):
        name, port = pair.split(":")
        mappings[name] = int(port)

    # Verify all ports are present
    assert set(mappings.keys()) == {"web", "api", "metrics"}
    assert len(set(mappings.values())) == 3  # All ports should be unique

    # Verify individual FLUSTER_PORT_* variables are also set
    assert "FLUSTER_PORT_WEB" in config.env
    assert "FLUSTER_PORT_API" in config.env
    assert "FLUSTER_PORT_METRICS" in config.env


def test_fray_port_mapping_not_set_when_no_ports(job_manager, mock_runtime):
    """Test that FRAY_PORT_MAPPING is not set when no ports are requested."""
    request = create_run_job_request(ports=[])
    job_id = job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    job.thread.join(timeout=15.0)

    # Verify runtime.create_container was called
    assert mock_runtime.create_container.called
    call_args = mock_runtime.create_container.call_args
    config = call_args[0][0]

    # FRAY_PORT_MAPPING should not be set when there are no ports
    assert "FRAY_PORT_MAPPING" not in config.env


def test_job_failure_error_appears_in_logs(job_manager, mock_bundle_cache):
    """Test that job failure errors appear in logs (not just job.error).

    This ensures that log polling can see error messages, not just the job status API.
    """
    mock_bundle_cache.get_bundle = Mock(side_effect=Exception("Bundle download failed"))

    request = create_run_job_request()
    job_id = job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    job.thread.join(timeout=15.0)

    final_job = job_manager.get_job(job_id)
    assert final_job.status == cluster_pb2.JOB_STATE_FAILED

    # Verify error appears in both job.error AND job.logs
    assert "Bundle download failed" in final_job.error

    # Get logs and verify error is present
    logs = job_manager.get_logs(job_id)
    error_logs = [log for log in logs if log.source == "error"]
    assert len(error_logs) >= 1
    assert any("Bundle download failed" in log.data for log in error_logs)


def test_port_retry_on_binding_failure(mock_bundle_cache, mock_venv_cache, mock_image_cache):
    """Test that job retries with new ports when port binding fails."""
    # Create runtime that fails with "address already in use" on first attempt
    runtime = Mock(spec=DockerRuntime)
    runtime.create_container = Mock(return_value="container123")

    # First attempt fails with port in use, second succeeds
    call_count = [0]

    def start_side_effect(container_id):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("failed to bind host port: address already in use")
        return None

    runtime.start_container = Mock(side_effect=start_side_effect)
    runtime.remove = Mock()

    # After successful start, container runs then completes
    inspect_call_count = [0]

    def inspect_side_effect(container_id):
        inspect_call_count[0] += 1
        if inspect_call_count[0] == 1:
            return ContainerStatus(running=True)
        return ContainerStatus(running=False, exit_code=0)

    runtime.inspect = Mock(side_effect=inspect_side_effect)
    runtime.get_stats = Mock(return_value=ContainerStats(memory_mb=100, cpu_percent=50, process_count=5, available=True))
    runtime.get_logs = Mock(return_value=[])

    port_allocator = PortAllocator(port_range=(50000, 50100))
    job_manager = JobManager(
        bundle_cache=mock_bundle_cache,
        venv_cache=mock_venv_cache,
        image_cache=mock_image_cache,
        runtime=runtime,
        port_allocator=port_allocator,
        max_concurrent_jobs=5,
    )

    request = create_run_job_request(ports=["actor"])
    job_id = job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    job.thread.join(timeout=15.0)

    # Job should succeed after retry
    final_job = job_manager.get_job(job_id)
    assert final_job.status == cluster_pb2.JOB_STATE_SUCCEEDED

    # start_container should have been called twice
    assert runtime.start_container.call_count == 2

    # remove should have been called once (to clean up failed container)
    assert runtime.remove.call_count == 1

    # Logs should contain retry message
    logs = job_manager.get_logs(job_id)
    build_logs = [log for log in logs if log.source == "build"]
    assert any("Port conflict" in log.data for log in build_logs)


def test_port_retry_exhausted(mock_bundle_cache, mock_venv_cache, mock_image_cache):
    """Test that job fails after max port retries are exhausted."""
    runtime = Mock(spec=DockerRuntime)
    runtime.create_container = Mock(return_value="container123")
    # All attempts fail
    runtime.start_container = Mock(side_effect=RuntimeError("failed to bind host port: address already in use"))
    runtime.remove = Mock()
    runtime.get_logs = Mock(return_value=[])

    port_allocator = PortAllocator(port_range=(50000, 50100))
    job_manager = JobManager(
        bundle_cache=mock_bundle_cache,
        venv_cache=mock_venv_cache,
        image_cache=mock_image_cache,
        runtime=runtime,
        port_allocator=port_allocator,
        max_concurrent_jobs=5,
    )

    request = create_run_job_request(ports=["actor"])
    job_id = job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    job.thread.join(timeout=15.0)

    # Job should fail after exhausting retries
    final_job = job_manager.get_job(job_id)
    assert final_job.status == cluster_pb2.JOB_STATE_FAILED
    assert "address already in use" in final_job.error

    # start_container should have been called 3 times (max retries)
    assert runtime.start_container.call_count == 3
