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
from unittest.mock import AsyncMock, Mock

import cloudpickle
import httpx
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from starlette.testclient import TestClient

from fluster import cluster_pb2
from fluster.cluster.worker.bundle import BundleCache
from fluster.cluster.worker.builder import BuildResult, ImageBuilder, VenvCache
from fluster.cluster.worker.dashboard import WorkerDashboard
from fluster.cluster.worker.manager import JobManager, PortAllocator
from fluster.cluster.worker.runtime import ContainerResult, DockerRuntime
from fluster.cluster.worker.service import WorkerServiceImpl
from fluster.cluster_connect import WorkerServiceClient

# ============================================================================
# Shared fixtures
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


@pytest.fixture
def service(job_manager):
    """Create WorkerServiceImpl."""
    return WorkerServiceImpl(manager=job_manager)


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


# ============================================================================
# Dashboard tests
# ============================================================================


def test_dashboard_loads(client):
    """Test dashboard HTML loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert "Fluster Worker Dashboard" in response.text


# ============================================================================
# Stats API tests
# ============================================================================


@pytest.mark.asyncio
async def test_stats_empty(client, service):
    """Test /api/stats with no jobs."""
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert data == {"running": 0, "pending": 0, "building": 0, "completed": 0}


@pytest.mark.asyncio
async def test_stats_with_jobs(client, service):
    """Test /api/stats with various job states."""
    # Submit jobs
    for i in range(5):
        request = create_run_job_request(job_id=f"job-{i}")
        await service.run_job(request, Mock())

    # Set jobs to different states
    jobs = service._manager.list_jobs()
    jobs[0].status = cluster_pb2.JOB_STATE_RUNNING
    jobs[1].status = cluster_pb2.JOB_STATE_PENDING
    jobs[2].status = cluster_pb2.JOB_STATE_BUILDING
    jobs[3].status = cluster_pb2.JOB_STATE_SUCCEEDED
    jobs[4].status = cluster_pb2.JOB_STATE_FAILED

    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["running"] == 1
    assert data["pending"] == 1
    assert data["building"] == 1
    assert data["completed"] == 2  # succeeded + failed


# ============================================================================
# List jobs API tests
# ============================================================================


def test_list_jobs_empty(client):
    """Test /api/jobs with no jobs."""
    response = client.get("/api/jobs")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_list_jobs_with_data(client, service):
    """Test /api/jobs returns all jobs."""
    for i in range(3):
        request = create_run_job_request(job_id=f"job-{i}")
        await service.run_job(request, Mock())

    response = client.get("/api/jobs")
    assert response.status_code == 200
    jobs = response.json()
    assert len(jobs) == 3

    job_ids = {j["job_id"] for j in jobs}
    assert job_ids == {"job-0", "job-1", "job-2"}


# ============================================================================
# Get job API tests
# ============================================================================


def test_get_job_not_found(client):
    """Test /api/jobs/{job_id} with nonexistent job."""
    response = client.get("/api/jobs/nonexistent")
    assert response.status_code == 404
    assert response.json() == {"error": "Not found"}


@pytest.mark.asyncio
async def test_get_job_success(client, service):
    """Test /api/jobs/{job_id} returns job details."""
    request = create_run_job_request(job_id="job-details", ports=["http", "grpc"])
    await service.run_job(request, Mock())

    # Set some details
    job = service._manager.get_job("job-details")
    job.status = cluster_pb2.JOB_STATE_RUNNING
    job.started_at_ms = 1000
    job.exit_code = 0

    response = client.get("/api/jobs/job-details")
    assert response.status_code == 200
    data = response.json()

    assert data["job_id"] == "job-details"
    assert data["status"] == "running"
    assert data["started_at"] == 1000
    assert data["exit_code"] == 0
    assert "http" in data["ports"]
    assert "grpc" in data["ports"]


# ============================================================================
# Get logs API tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_logs_with_tail_parameter(client, service):
    """Test /api/jobs/{job_id}/logs?tail=N returns last N lines."""
    request = create_run_job_request(job_id="job-tail")
    await service.run_job(request, Mock())

    # Write logs
    job = service._manager.get_job("job-tail")
    stdout_file = job.workdir / "STDOUT"
    stdout_file.write_text("\n".join(f"Log line {i}" for i in range(100)))

    response = client.get("/api/jobs/job-tail/logs?tail=5")
    assert response.status_code == 200
    logs = response.json()

    assert len(logs) == 5
    assert logs[0]["data"] == "Log line 95"
    assert logs[4]["data"] == "Log line 99"


# ============================================================================
# RPC Service tests (WorkerServiceImpl)
# ============================================================================


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
    """Test run_job allocates ports correctly."""
    request = create_run_job_request(job_id="job-with-ports", ports=["http", "grpc"])
    response = await service.run_job(request, request_context)

    assert response.job_id == "job-with-ports"

    # Verify ports were allocated
    job = service._manager.get_job("job-with-ports")
    assert len(job.ports) == 2
    assert "http" in job.ports
    assert "grpc" in job.ports


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
    """Test get_job_status for completed job includes timing info."""
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

    assert exc_info.value.code == Code.FAILED_PRECONDITION
    assert "already completed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_kill_job_with_custom_timeout(service, request_context):
    """Test kill_job respects custom term_timeout_ms."""
    request = create_run_job_request(job_id="job-kill")
    await service.run_job(request, request_context)

    await asyncio.sleep(0.1)

    # Manually set job to RUNNING to simulate mid-execution
    job = service._manager.get_job("job-kill")
    job.status = cluster_pb2.JOB_STATE_RUNNING
    job.container_id = "container123"

    kill_request = cluster_pb2.KillJobRequest(job_id="job-kill", term_timeout_ms=100)
    response = await service.kill_job(kill_request, request_context)

    assert isinstance(response, cluster_pb2.Empty)
    assert job.status == cluster_pb2.JOB_STATE_KILLED


# ============================================================================
# Connect RPC integration tests
# ============================================================================


@pytest.mark.asyncio
async def test_rpc_endpoint_mounted_correctly(server):
    """Test Connect RPC is mounted at correct path."""
    # Check that the RPC path is included in routes
    route_paths = [route.path for route in server._app.routes]
    assert "/fluster.cluster.WorkerService" in route_paths


@pytest.mark.asyncio
async def test_rpc_run_job_via_connect_client(service):
    """Test calling run_job via Connect RPC client."""
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

        # Create Connect client
        async with httpx.AsyncClient() as http_client:
            client = WorkerServiceClient(address="http://127.0.0.1:18080", session=http_client)

            # Submit job via RPC
            request = create_run_job_request(job_id="rpc-test-job")
            response = await client.run_job(request)

            assert response.job_id == "rpc-test-job"
            assert response.state == cluster_pb2.JOB_STATE_PENDING

    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


# ============================================================================
# Server properties tests
# ============================================================================


def test_server_port_property(service):
    """Test server port property returns configured port."""
    server = WorkerDashboard(service=service, host="127.0.0.1", port=9999)
    assert server.port == 9999


def test_server_default_host_and_port(service):
    """Test server uses default host and port."""
    server = WorkerDashboard(service=service)
    assert server._host == "0.0.0.0"
    assert server._port == 8080
