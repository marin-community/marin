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

"""Tests for WorkerService RPC implementation."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import cloudpickle
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from fluster import cluster_pb2
from fluster.cluster.worker.bundle import BundleCache
from fluster.cluster.worker.builder import BuildResult, ImageBuilder, VenvCache
from fluster.cluster.worker.manager import JobManager, PortAllocator
from fluster.cluster.worker.runtime import ContainerResult, DockerRuntime
from fluster.cluster.worker.service import WorkerServiceImpl


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


@pytest.fixture
def service(job_manager):
    """Create WorkerServiceImpl with JobManager."""
    return WorkerServiceImpl(manager=job_manager)


@pytest.fixture
def request_context():
    """Create a mock RequestContext for RPC calls."""
    return Mock(spec=RequestContext)


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


# ============================================================================
# run_job tests
# ============================================================================


@pytest.mark.asyncio
async def test_run_job_success(service, request_context):
    """Test run_job returns job_id and initial state."""
    request = create_run_job_request(job_id="job-123")
    response = await service.run_job(request, request_context)

    assert response.job_id == "job-123"
    assert response.state == cluster_pb2.JOB_STATE_PENDING


@pytest.mark.asyncio
async def test_run_job_generates_job_id_if_missing(service, request_context):
    """Test run_job generates job_id when not provided."""
    request = create_run_job_request(job_id="")
    response = await service.run_job(request, request_context)

    assert response.job_id  # Should have a generated ID
    assert len(response.job_id) > 0
    assert response.state == cluster_pb2.JOB_STATE_PENDING


@pytest.mark.asyncio
async def test_run_job_with_ports(service, request_context):
    """Test run_job with port allocations."""
    request = create_run_job_request(job_id="job-with-ports", ports=["http", "grpc"])
    response = await service.run_job(request, request_context)

    assert response.job_id == "job-with-ports"

    # Verify ports were allocated
    job = service._manager.get_job("job-with-ports")
    assert len(job.ports) == 2
    assert "http" in job.ports
    assert "grpc" in job.ports


# ============================================================================
# get_job_status tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_job_status_success(service, request_context):
    """Test get_job_status returns correct status."""
    request = create_run_job_request(job_id="job-456")
    await service.run_job(request, request_context)

    status_request = cluster_pb2.GetStatusRequest(job_id="job-456")
    status = await service.get_job_status(status_request, request_context)

    assert status.job_id == "job-456"
    assert status.state in [
        cluster_pb2.JOB_STATE_PENDING,
        cluster_pb2.JOB_STATE_BUILDING,
        cluster_pb2.JOB_STATE_RUNNING,
        cluster_pb2.JOB_STATE_SUCCEEDED,
    ]


@pytest.mark.asyncio
async def test_get_job_status_not_found(service, request_context):
    """Test get_job_status raises NOT_FOUND for nonexistent job."""
    status_request = cluster_pb2.GetStatusRequest(job_id="nonexistent")

    with pytest.raises(ConnectError) as exc_info:
        await service.get_job_status(status_request, request_context)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_job_status_completed_job(service, request_context):
    """Test get_job_status for completed job."""
    request = create_run_job_request(job_id="job-completed")
    await service.run_job(request, request_context)

    # Wait for job to complete
    job = service._manager.get_job("job-completed")
    await asyncio.wait_for(job.task, timeout=5.0)

    status_request = cluster_pb2.GetStatusRequest(job_id="job-completed")
    status = await service.get_job_status(status_request, request_context)

    assert status.job_id == "job-completed"
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
    assert status.exit_code == 0
    assert status.started_at_ms > 0
    assert status.finished_at_ms > 0


# ============================================================================
# list_jobs tests
# ============================================================================


@pytest.mark.asyncio
async def test_list_jobs_empty(service, request_context):
    """Test list_jobs returns empty list when no jobs."""
    list_request = cluster_pb2.ListJobsRequest()
    response = await service.list_jobs(list_request, request_context)

    assert len(response.jobs) == 0


@pytest.mark.asyncio
async def test_list_jobs_multiple(service, request_context):
    """Test list_jobs returns all jobs."""
    # Submit multiple jobs
    for i in range(3):
        request = create_run_job_request(job_id=f"job-{i}")
        await service.run_job(request, request_context)

    list_request = cluster_pb2.ListJobsRequest()
    response = await service.list_jobs(list_request, request_context)

    assert len(response.jobs) == 3
    job_ids = {job.job_id for job in response.jobs}
    assert job_ids == {"job-0", "job-1", "job-2"}


@pytest.mark.asyncio
async def test_list_jobs_with_namespace_filter(service, request_context):
    """Test list_jobs with namespace (currently ignored)."""
    # Submit jobs
    for i in range(2):
        request = create_run_job_request(job_id=f"job-{i}")
        await service.run_job(request, request_context)

    # Namespace filter is currently not implemented, so all jobs are returned
    list_request = cluster_pb2.ListJobsRequest(namespace="test-namespace")
    response = await service.list_jobs(list_request, request_context)

    assert len(response.jobs) == 2


# ============================================================================
# fetch_logs tests
# ============================================================================


@pytest.mark.asyncio
async def test_fetch_logs_empty(service, request_context):
    """Test fetch_logs for job with no logs."""
    request = create_run_job_request(job_id="job-logs-empty")
    await service.run_job(request, request_context)

    log_filter = cluster_pb2.FetchLogsFilter()
    logs_request = cluster_pb2.FetchLogsRequest(job_id="job-logs-empty", filter=log_filter)
    response = await service.fetch_logs(logs_request, request_context)

    assert len(response.logs) == 0


@pytest.mark.asyncio
async def test_fetch_logs_with_data(service, request_context):
    """Test fetch_logs returns log entries."""
    request = create_run_job_request(job_id="job-logs-data")
    await service.run_job(request, request_context)

    # Write some logs
    job = service._manager.get_job("job-logs-data")
    stdout_file = job.workdir / "STDOUT"
    stdout_file.write_text("\n".join(f"Log line {i}" for i in range(10)))

    log_filter = cluster_pb2.FetchLogsFilter()
    logs_request = cluster_pb2.FetchLogsRequest(job_id="job-logs-data", filter=log_filter)
    response = await service.fetch_logs(logs_request, request_context)

    assert len(response.logs) == 10
    assert response.logs[0].data == "Log line 0"
    assert response.logs[0].source == "stdout"


@pytest.mark.asyncio
async def test_fetch_logs_with_start_line(service, request_context):
    """Test fetch_logs with start_line filter."""
    request = create_run_job_request(job_id="job-logs-start")
    await service.run_job(request, request_context)

    # Write logs
    job = service._manager.get_job("job-logs-start")
    stdout_file = job.workdir / "STDOUT"
    stdout_file.write_text("\n".join(f"Log line {i}" for i in range(10)))

    log_filter = cluster_pb2.FetchLogsFilter(start_line=5)
    logs_request = cluster_pb2.FetchLogsRequest(job_id="job-logs-start", filter=log_filter)
    response = await service.fetch_logs(logs_request, request_context)

    assert len(response.logs) == 5
    assert response.logs[0].data == "Log line 5"


@pytest.mark.asyncio
async def test_fetch_logs_tail_with_negative_start_line(service, request_context):
    """Test fetch_logs with negative start_line for tailing."""
    request = create_run_job_request(job_id="job-logs-tail")
    await service.run_job(request, request_context)

    # Write logs
    job = service._manager.get_job("job-logs-tail")
    stdout_file = job.workdir / "STDOUT"
    stdout_file.write_text("\n".join(f"Log line {i}" for i in range(10)))

    log_filter = cluster_pb2.FetchLogsFilter(start_line=-3)
    logs_request = cluster_pb2.FetchLogsRequest(job_id="job-logs-tail", filter=log_filter)
    response = await service.fetch_logs(logs_request, request_context)

    assert len(response.logs) == 3
    assert response.logs[0].data == "Log line 7"
    assert response.logs[1].data == "Log line 8"
    assert response.logs[2].data == "Log line 9"


@pytest.mark.asyncio
async def test_fetch_logs_with_max_lines(service, request_context):
    """Test fetch_logs with max_lines limit."""
    request = create_run_job_request(job_id="job-logs-max")
    await service.run_job(request, request_context)

    # Write logs
    job = service._manager.get_job("job-logs-max")
    stdout_file = job.workdir / "STDOUT"
    stdout_file.write_text("\n".join(f"Log line {i}" for i in range(100)))

    log_filter = cluster_pb2.FetchLogsFilter(max_lines=5)
    logs_request = cluster_pb2.FetchLogsRequest(job_id="job-logs-max", filter=log_filter)
    response = await service.fetch_logs(logs_request, request_context)

    assert len(response.logs) == 5


@pytest.mark.asyncio
async def test_fetch_logs_with_regex_filter(service, request_context):
    """Test fetch_logs with regex content filter."""
    request = create_run_job_request(job_id="job-logs-regex")
    await service.run_job(request, request_context)

    # Write logs with different patterns
    job = service._manager.get_job("job-logs-regex")
    stdout_file = job.workdir / "STDOUT"
    stdout_file.write_text("ERROR: something bad\nINFO: normal log\nERROR: another error\nDEBUG: details")

    log_filter = cluster_pb2.FetchLogsFilter(regex="ERROR")
    logs_request = cluster_pb2.FetchLogsRequest(job_id="job-logs-regex", filter=log_filter)
    response = await service.fetch_logs(logs_request, request_context)

    assert len(response.logs) == 2
    assert "ERROR" in response.logs[0].data
    assert "ERROR" in response.logs[1].data


@pytest.mark.asyncio
async def test_fetch_logs_with_time_range(service, request_context):
    """Test fetch_logs with time range filter."""
    request = create_run_job_request(job_id="job-logs-time")
    await service.run_job(request, request_context)

    # Write logs
    job = service._manager.get_job("job-logs-time")
    stdout_file = job.workdir / "STDOUT"
    stdout_file.write_text("\n".join(f"Log line {i}" for i in range(10)))

    # Set time range that won't match anything (far in the future)
    now_ms = int(time.time() * 1000)
    future_ms = now_ms + 1000000
    log_filter = cluster_pb2.FetchLogsFilter(start_ms=future_ms)
    logs_request = cluster_pb2.FetchLogsRequest(job_id="job-logs-time", filter=log_filter)
    response = await service.fetch_logs(logs_request, request_context)

    # Should filter out all logs (timestamps are current time)
    assert len(response.logs) == 0


@pytest.mark.asyncio
async def test_fetch_logs_combined_filters(service, request_context):
    """Test fetch_logs with multiple filters combined."""
    request = create_run_job_request(job_id="job-logs-combined")
    await service.run_job(request, request_context)

    # Write logs
    job = service._manager.get_job("job-logs-combined")
    stdout_file = job.workdir / "STDOUT"
    logs = [
        "ERROR: first error",
        "INFO: normal",
        "ERROR: second error",
        "ERROR: third error",
        "ERROR: fourth error",
        "ERROR: fifth error",
    ]
    stdout_file.write_text("\n".join(logs))

    # Use regex to filter ERRORs, then limit to 2
    log_filter = cluster_pb2.FetchLogsFilter(regex="ERROR", max_lines=2)
    logs_request = cluster_pb2.FetchLogsRequest(job_id="job-logs-combined", filter=log_filter)
    response = await service.fetch_logs(logs_request, request_context)

    assert len(response.logs) == 2
    assert "ERROR" in response.logs[0].data
    assert "ERROR" in response.logs[1].data


# ============================================================================
# kill_job tests
# ============================================================================


@pytest.mark.asyncio
async def test_kill_job_success(service, request_context):
    """Test kill_job terminates running job."""
    request = create_run_job_request(job_id="job-kill")
    await service.run_job(request, request_context)

    # Wait a bit for job to start
    await asyncio.sleep(0.1)

    # Manually set job to RUNNING to simulate mid-execution
    job = service._manager.get_job("job-kill")
    job.status = cluster_pb2.JOB_STATE_RUNNING
    job.container_id = "container123"

    kill_request = cluster_pb2.KillJobRequest(job_id="job-kill", term_timeout_ms=100)
    response = await service.kill_job(kill_request, request_context)

    assert isinstance(response, cluster_pb2.Empty)
    assert job.status == cluster_pb2.JOB_STATE_KILLED


@pytest.mark.asyncio
async def test_kill_job_not_found(service, request_context):
    """Test kill_job raises NOT_FOUND for nonexistent job."""
    kill_request = cluster_pb2.KillJobRequest(job_id="nonexistent")

    with pytest.raises(ConnectError) as exc_info:
        await service.kill_job(kill_request, request_context)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in str(exc_info.value)


@pytest.mark.asyncio
async def test_kill_job_already_completed(service, request_context):
    """Test kill_job fails for already completed job."""
    request = create_run_job_request(job_id="job-completed")
    await service.run_job(request, request_context)

    # Wait for job to complete
    job = service._manager.get_job("job-completed")
    await asyncio.wait_for(job.task, timeout=5.0)

    # Try to kill completed job
    kill_request = cluster_pb2.KillJobRequest(job_id="job-completed")

    with pytest.raises(ConnectError) as exc_info:
        await service.kill_job(kill_request, request_context)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "not running" in str(exc_info.value)


@pytest.mark.asyncio
async def test_kill_job_default_timeout(service, request_context):
    """Test kill_job uses default timeout when not specified."""
    request = create_run_job_request(job_id="job-kill-default")
    await service.run_job(request, request_context)

    await asyncio.sleep(0.1)

    job = service._manager.get_job("job-kill-default")
    job.status = cluster_pb2.JOB_STATE_RUNNING
    job.container_id = "container123"

    # Don't specify timeout (should use default 5000ms)
    kill_request = cluster_pb2.KillJobRequest(job_id="job-kill-default")
    response = await service.kill_job(kill_request, request_context)

    assert isinstance(response, cluster_pb2.Empty)


# ============================================================================
# health_check tests
# ============================================================================


@pytest.mark.asyncio
async def test_health_check_no_jobs(service, request_context):
    """Test health_check with no running jobs."""
    # Wait a bit to ensure non-zero uptime
    await asyncio.sleep(0.01)

    empty_request = cluster_pb2.Empty()
    response = await service.health_check(empty_request, request_context)

    assert response.healthy is True
    assert response.uptime_ms >= 0
    assert response.running_jobs == 0


@pytest.mark.asyncio
async def test_health_check_with_running_jobs(service, request_context):
    """Test health_check reports running jobs count."""
    # Submit jobs and set them to RUNNING state
    for i in range(3):
        request = create_run_job_request(job_id=f"job-{i}")
        await service.run_job(request, request_context)

    # Manually set jobs to RUNNING
    for i in range(3):
        job = service._manager.get_job(f"job-{i}")
        job.status = cluster_pb2.JOB_STATE_RUNNING

    empty_request = cluster_pb2.Empty()
    response = await service.health_check(empty_request, request_context)

    assert response.healthy is True
    assert response.uptime_ms > 0
    assert response.running_jobs == 3


@pytest.mark.asyncio
async def test_health_check_uptime_increases(service, request_context):
    """Test health_check uptime increases over time."""
    empty_request = cluster_pb2.Empty()

    response1 = await service.health_check(empty_request, request_context)
    uptime1 = response1.uptime_ms

    # Wait a bit
    await asyncio.sleep(0.1)

    response2 = await service.health_check(empty_request, request_context)
    uptime2 = response2.uptime_ms

    assert uptime2 > uptime1


@pytest.mark.asyncio
async def test_health_check_ignores_completed_jobs(service, request_context):
    """Test health_check only counts RUNNING jobs."""
    # Submit jobs
    for i in range(3):
        request = create_run_job_request(job_id=f"job-{i}")
        await service.run_job(request, request_context)

    # Set one to RUNNING, others to different states
    job0 = service._manager.get_job("job-0")
    job0.status = cluster_pb2.JOB_STATE_RUNNING

    job1 = service._manager.get_job("job-1")
    job1.status = cluster_pb2.JOB_STATE_SUCCEEDED

    job2 = service._manager.get_job("job-2")
    job2.status = cluster_pb2.JOB_STATE_FAILED

    empty_request = cluster_pb2.Empty()
    response = await service.health_check(empty_request, request_context)

    assert response.running_jobs == 1  # Only job-0 is running
