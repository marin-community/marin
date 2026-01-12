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

import asyncio
import socket
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import cloudpickle
import pytest

from fluster import cluster_pb2
from fluster.cluster.worker.bundle import BundleCache
from fluster.cluster.worker.builder import BuildResult, ImageBuilder, VenvCache
from fluster.cluster.worker.manager import JobManager, PortAllocator
from fluster.cluster.worker.runtime import ContainerResult, DockerRuntime


@pytest.fixture
def allocator():
    """Create PortAllocator with small range for testing."""
    return PortAllocator(port_range=(40000, 40100))


@pytest.mark.asyncio
async def test_allocate_single_port(allocator):
    """Test allocating a single port."""
    ports = await allocator.allocate(count=1)
    assert len(ports) == 1
    assert 40000 <= ports[0] < 40100


@pytest.mark.asyncio
async def test_allocate_multiple_ports(allocator):
    """Test allocating multiple ports at once."""
    ports = await allocator.allocate(count=5)
    assert len(ports) == 5
    assert len(set(ports)) == 5  # All unique
    for port in ports:
        assert 40000 <= port < 40100


@pytest.mark.asyncio
async def test_allocated_ports_are_usable(allocator):
    """Test that allocated ports can actually be bound."""
    ports = await allocator.allocate(count=3)

    # Verify each port can be bound (it's free)
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))


@pytest.mark.asyncio
async def test_no_port_reuse_before_release(allocator):
    """Test that allocated ports are not reused before release."""
    ports1 = await allocator.allocate(count=5)
    ports2 = await allocator.allocate(count=5)

    # No overlap between the two allocations
    assert len(set(ports1) & set(ports2)) == 0


@pytest.mark.asyncio
async def test_ports_reused_after_release():
    """Test that ports can be reused after release."""
    # Allocate all available ports in a small range
    allocator_small = PortAllocator(port_range=(40000, 40003))

    # Allocate 3 ports
    ports1 = await allocator_small.allocate(count=3)
    assert len(ports1) == 3

    # Release them
    await allocator_small.release(ports1)

    # Should be able to allocate again
    ports2 = await allocator_small.allocate(count=3)
    assert len(ports2) == 3

    # Ports should be reused (same set, possibly different order)
    assert set(ports1) == set(ports2)


@pytest.mark.asyncio
async def test_release_partial_ports(allocator):
    """Test releasing only some ports."""
    ports = await allocator.allocate(count=5)

    # Release first 3 ports
    await allocator.release(ports[:3])

    # Allocate 2 more - should get from the released ones
    new_ports = await allocator.allocate(count=2)

    # At least some of the new ports should be from released ones
    assert len(set(new_ports) & set(ports[:3])) > 0


@pytest.mark.asyncio
async def test_exhausted_port_range():
    """Test behavior when port range is exhausted."""
    allocator_tiny = PortAllocator(port_range=(40000, 40002))

    # Allocate all available ports (2 ports: 40000, 40001)
    ports = await allocator_tiny.allocate(count=2)
    assert len(ports) == 2

    # Trying to allocate more should raise RuntimeError
    with pytest.raises(RuntimeError, match="No free ports available"):
        await allocator_tiny.allocate(count=1)


@pytest.mark.asyncio
async def test_concurrent_allocations(allocator):
    """Test concurrent port allocations are thread-safe."""

    async def allocate_ports():
        return await allocator.allocate(count=5)

    # Run multiple concurrent allocations
    results = await asyncio.gather(
        allocate_ports(),
        allocate_ports(),
        allocate_ports(),
    )

    # Collect all allocated ports
    all_ports = []
    for ports in results:
        all_ports.extend(ports)

    # All ports should be unique (no conflicts)
    assert len(all_ports) == len(set(all_ports))


@pytest.mark.asyncio
async def test_release_nonexistent_port(allocator):
    """Test that releasing a non-allocated port doesn't cause errors."""
    # Should not raise an error
    await allocator.release([99999])


@pytest.mark.asyncio
async def test_default_port_range():
    """Test default port range is 30000-40000."""
    allocator = PortAllocator()
    ports = await allocator.allocate(count=5)

    for port in ports:
        assert 30000 <= port < 40000


# ============================================================================
# JobManager Tests
# ============================================================================


@pytest.fixture
def mock_bundle_cache():
    """Create mock BundleCache."""
    cache = AsyncMock(spec=BundleCache)
    cache.get_bundle = AsyncMock(return_value=Path("/tmp/bundle"))
    return cache


@pytest.fixture
def mock_venv_cache():
    """Create mock VenvCache."""
    cache = Mock(spec=VenvCache)
    cache.compute_deps_hash = Mock(return_value="abc123")
    return cache


@pytest.fixture
def mock_image_builder():
    """Create mock ImageBuilder."""
    builder = AsyncMock(spec=ImageBuilder)
    builder.build = AsyncMock(
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
    """Create mock DockerRuntime."""
    runtime = AsyncMock(spec=DockerRuntime)
    runtime.run = AsyncMock(
        return_value=ContainerResult(
            container_id="container123",
            exit_code=0,
            started_at=0.0,
            finished_at=1.0,
        )
    )
    runtime.kill = AsyncMock()
    runtime.remove = AsyncMock()
    return runtime


@pytest.fixture
def job_manager(mock_bundle_cache, mock_venv_cache, mock_image_builder, mock_runtime):
    """Create JobManager with mocked dependencies."""
    port_allocator = PortAllocator(port_range=(50000, 50100))
    return JobManager(
        bundle_cache=mock_bundle_cache,
        venv_cache=mock_venv_cache,
        image_builder=mock_image_builder,
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
        env_vars={"TEST_VAR": "value"},
        extras=["dev"],
    )

    resources = cluster_pb2.ResourceSpec(
        cpu=2,
        memory="4g",
    )

    return cluster_pb2.RunJobRequest(
        job_id=job_id,
        serialized_entrypoint=serialized_entrypoint,
        environment=env_config,
        bundle_gcs_path="gs://bucket/bundle.zip",
        resources=resources,
        env_vars={"JOB_VAR": "job_value"},
        timeout_seconds=300,
        ports=ports or [],
    )


@pytest.mark.asyncio
async def test_submit_job_returns_job_id(job_manager):
    """Test that submit_job returns job_id immediately."""
    request = create_run_job_request()
    job_id = await job_manager.submit_job(request)

    assert job_id == "test-job-1"

    # Job should be tracked
    job = job_manager.get_job(job_id)
    assert job is not None
    assert job.job_id == job_id
    assert job.status == cluster_pb2.JOB_STATE_PENDING


@pytest.mark.asyncio
async def test_job_lifecycle_phases(job_manager):
    """Test job transitions through PENDING → BUILDING → RUNNING → SUCCEEDED."""
    request = create_run_job_request()
    job_id = await job_manager.submit_job(request)

    # Wait for job to complete
    job = job_manager.get_job(job_id)
    await asyncio.wait_for(job.task, timeout=5.0)

    # Should have gone through phases
    final_job = job_manager.get_job(job_id)
    assert final_job.status == cluster_pb2.JOB_STATE_SUCCEEDED
    assert final_job.exit_code == 0
    assert final_job.started_at_ms is not None
    assert final_job.finished_at_ms is not None


@pytest.mark.asyncio
async def test_job_with_ports(job_manager):
    """Test job with port allocation."""
    request = create_run_job_request(ports=["http", "grpc"])
    job_id = await job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    assert len(job.ports) == 2
    assert "http" in job.ports
    assert "grpc" in job.ports
    assert job.ports["http"] != job.ports["grpc"]

    # Wait for completion
    await asyncio.wait_for(job.task, timeout=5.0)

    # Ports should be released (we can't easily verify this without checking internal state)


@pytest.mark.asyncio
async def test_job_failure_on_nonzero_exit(job_manager, mock_runtime):
    """Test job fails when container exits with non-zero code."""
    mock_runtime.run = AsyncMock(
        return_value=ContainerResult(
            container_id="container123",
            exit_code=1,
            started_at=0.0,
            finished_at=1.0,
        )
    )

    request = create_run_job_request()
    job_id = await job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    await asyncio.wait_for(job.task, timeout=5.0)

    final_job = job_manager.get_job(job_id)
    assert final_job.status == cluster_pb2.JOB_STATE_FAILED
    assert final_job.exit_code == 1
    assert "Exit code: 1" in final_job.error


@pytest.mark.asyncio
async def test_job_failure_on_error(job_manager, mock_runtime):
    """Test job fails when container returns error."""
    mock_runtime.run = AsyncMock(
        return_value=ContainerResult(
            container_id="container123",
            exit_code=-1,
            started_at=0.0,
            finished_at=1.0,
            error="Container crashed",
        )
    )

    request = create_run_job_request()
    job_id = await job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    await asyncio.wait_for(job.task, timeout=5.0)

    final_job = job_manager.get_job(job_id)
    assert final_job.status == cluster_pb2.JOB_STATE_FAILED
    assert final_job.error == "Container crashed"


@pytest.mark.asyncio
async def test_job_exception_handling(job_manager, mock_bundle_cache):
    """Test job handles exceptions during execution."""
    mock_bundle_cache.get_bundle = AsyncMock(side_effect=Exception("Bundle download failed"))

    request = create_run_job_request()
    job_id = await job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    await asyncio.wait_for(job.task, timeout=5.0)

    final_job = job_manager.get_job(job_id)
    assert final_job.status == cluster_pb2.JOB_STATE_FAILED
    assert "Bundle download failed" in final_job.error


@pytest.mark.asyncio
async def test_concurrent_job_limiting(job_manager):
    """Test semaphore limits concurrent job execution."""
    # Submit more jobs than max_concurrent_jobs (5)
    requests = [create_run_job_request(job_id=f"job-{i}") for i in range(10)]

    job_ids = []
    for request in requests:
        job_id = await job_manager.submit_job(request)
        job_ids.append(job_id)

    # All jobs should be submitted
    assert len(job_ids) == 10

    # Wait for all to complete
    jobs = [job_manager.get_job(job_id) for job_id in job_ids]
    await asyncio.gather(*[job.task for job in jobs])

    # All should eventually succeed
    final_jobs = [job_manager.get_job(job_id) for job_id in job_ids]
    for job in final_jobs:
        assert job.status == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.asyncio
async def test_list_jobs(job_manager):
    """Test listing all jobs."""
    requests = [create_run_job_request(job_id=f"job-{i}") for i in range(3)]

    for request in requests:
        await job_manager.submit_job(request)

    jobs = job_manager.list_jobs()
    assert len(jobs) == 3
    assert {job.job_id for job in jobs} == {"job-0", "job-1", "job-2"}


@pytest.mark.asyncio
async def test_kill_running_job(job_manager, mock_runtime):
    """Test killing a running job with graceful timeout."""
    # Create a job and manually set it to RUNNING state with a container_id
    # to simulate a job that's mid-execution
    request = create_run_job_request()
    job_id = await job_manager.submit_job(request)

    # Wait a bit for the job to start executing
    await asyncio.sleep(0.1)

    # Manually set up the job to simulate it being in RUNNING state
    job = job_manager.get_job(job_id)
    job.status = cluster_pb2.JOB_STATE_RUNNING
    job.container_id = "container123"

    # Kill the job
    result = await job_manager.kill_job(job_id, term_timeout_ms=100)
    assert result is True

    assert job.status == cluster_pb2.JOB_STATE_KILLED
    assert job.finished_at_ms is not None

    # Verify SIGTERM was sent first
    mock_runtime.kill.assert_any_call("container123", force=False)


@pytest.mark.asyncio
async def test_kill_nonexistent_job(job_manager):
    """Test killing a nonexistent job returns False."""
    result = await job_manager.kill_job("nonexistent-job")
    assert result is False


@pytest.mark.asyncio
async def test_kill_job_without_container(job_manager):
    """Test killing a job before container is created cancels the task."""
    request = create_run_job_request()
    job_id = await job_manager.submit_job(request)

    # Try to kill immediately (before container is created)
    result = await job_manager.kill_job(job_id)
    assert result is True

    # Job should be in KILLED state
    job = job_manager.get_job(job_id)
    assert job.status == cluster_pb2.JOB_STATE_KILLED


@pytest.mark.asyncio
async def test_get_logs_empty(job_manager):
    """Test getting logs for job with no logs."""
    request = create_run_job_request()
    job_id = await job_manager.submit_job(request)

    logs = await job_manager.get_logs(job_id)
    assert logs == []


@pytest.mark.asyncio
async def test_get_logs_with_start_line(job_manager):
    """Test getting logs with start_line offset."""
    request = create_run_job_request()
    job_id = await job_manager.submit_job(request)

    # Write some logs to the STDOUT file
    job = job_manager.get_job(job_id)
    stdout_file = job.workdir / "STDOUT"
    stdout_file.write_text("\n".join(f"Log line {i}" for i in range(10)))

    # Get logs starting from line 5
    logs = await job_manager.get_logs(job_id, start_line=5)
    assert len(logs) == 5
    assert logs[0].data == "Log line 5"


@pytest.mark.asyncio
async def test_get_logs_tail_behavior(job_manager):
    """Test getting last N logs with negative start_line."""
    request = create_run_job_request()
    job_id = await job_manager.submit_job(request)

    # Write some logs to the STDOUT file
    job = job_manager.get_job(job_id)
    stdout_file = job.workdir / "STDOUT"
    stdout_file.write_text("\n".join(f"Log line {i}" for i in range(10)))

    # Get last 3 logs
    logs = await job_manager.get_logs(job_id, start_line=-3)
    assert len(logs) == 3
    assert logs[0].data == "Log line 7"
    assert logs[1].data == "Log line 8"
    assert logs[2].data == "Log line 9"


@pytest.mark.asyncio
async def test_get_logs_nonexistent_job(job_manager):
    """Test getting logs for nonexistent job returns empty list."""
    logs = await job_manager.get_logs("nonexistent-job")
    assert logs == []


@pytest.mark.asyncio
async def test_cleanup_removes_container(job_manager, mock_runtime):
    """Test that container is removed after job completes."""
    request = create_run_job_request()
    job_id = await job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    await asyncio.wait_for(job.task, timeout=5.0)

    # Verify container was removed
    mock_runtime.remove.assert_called_once_with("container123")


@pytest.mark.asyncio
async def test_build_command_with_entrypoint(job_manager):
    """Test _build_command creates correct cloudpickle command."""
    entrypoint = create_test_entrypoint()
    command = job_manager._build_command(entrypoint)

    assert command[0] == "python"
    assert command[1] == "-c"
    assert "cloudpickle" in command[2]
    assert "base64" in command[2]


@pytest.mark.asyncio
async def test_job_status_message_during_building(job_manager):
    """Test that status_message is set during BUILDING phase."""
    request = create_run_job_request()
    job_id = await job_manager.submit_job(request)

    # Wait a bit for job to start building
    await asyncio.sleep(0.1)

    job = job_manager.get_job(job_id)
    # Job should be in BUILDING state with a status_message
    if job.status == cluster_pb2.JOB_STATE_BUILDING:
        assert job.status_message in ["downloading bundle", "building image", "populating uv cache"]

    # Wait for completion
    await asyncio.wait_for(job.task, timeout=5.0)

    # After completion, status_message should be empty
    final_job = job_manager.get_job(job_id)
    assert final_job.status_message == ""


@pytest.mark.asyncio
async def test_job_to_proto_includes_status_message(job_manager):
    """Test that Job.to_proto() includes status_message."""
    request = create_run_job_request()
    job_id = await job_manager.submit_job(request)

    job = job_manager.get_job(job_id)
    job.status_message = "test message"

    proto = job.to_proto()
    assert proto.status_message == "test message"
