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

"""Tests for WorkerServer HTTP and RPC endpoints."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import cloudpickle
import httpx
import pytest
from starlette.testclient import TestClient

from fluster import cluster_pb2
from fluster.cluster.worker.builder import BuildResult, ImageBuilder, VenvCache
from fluster.cluster.worker.bundle import BundleCache
from fluster.cluster.worker.manager import JobManager, PortAllocator
from fluster.cluster.worker.runtime import ContainerResult, DockerRuntime
from fluster.cluster.worker.server import WorkerServer
from fluster.cluster.worker.service import WorkerServiceImpl
from fluster.cluster_connect import WorkerServiceClient


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
    """Create WorkerServiceImpl."""
    return WorkerServiceImpl(manager=job_manager)


@pytest.fixture
def server(service):
    """Create WorkerServer."""
    return WorkerServer(service=service, host="127.0.0.1", port=0)


@pytest.fixture
def client(server):
    """Create test client for HTTP requests."""
    return TestClient(server._app)


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
# Dashboard tests
# ============================================================================


def test_dashboard_loads(client):
    """Test dashboard HTML loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert "Fluster Worker Dashboard" in response.text
    assert "async function refresh()" in response.text


def test_dashboard_contains_stats_div(client):
    """Test dashboard contains stats div for live updates."""
    response = client.get("/")
    assert '<div id="stats"></div>' in response.text


def test_dashboard_contains_jobs_table(client):
    """Test dashboard contains jobs table."""
    response = client.get("/")
    assert '<table id="jobs">' in response.text
    assert "<th>ID</th>" in response.text
    assert "<th>Status</th>" in response.text


def test_dashboard_has_auto_refresh(client):
    """Test dashboard has auto-refresh JavaScript."""
    response = client.get("/")
    assert "setInterval(refresh, 5000)" in response.text


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


@pytest.mark.asyncio
async def test_stats_counts_killed_as_completed(client, service):
    """Test /api/stats counts killed jobs as completed."""
    request = create_run_job_request(job_id="job-killed")
    await service.run_job(request, Mock())

    job = service._manager.get_job("job-killed")
    job.status = cluster_pb2.JOB_STATE_KILLED

    response = client.get("/api/stats")
    data = response.json()
    assert data["completed"] == 1


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

    # Check structure
    for job in jobs:
        assert "job_id" in job
        assert "status" in job
        assert "started_at" in job
        assert "finished_at" in job
        assert "error" in job


@pytest.mark.asyncio
async def test_list_jobs_shows_status_names(client, service):
    """Test /api/jobs returns status as string names."""
    request = create_run_job_request(job_id="job-test")
    await service.run_job(request, Mock())

    job = service._manager.get_job("job-test")
    job.status = cluster_pb2.JOB_STATE_RUNNING

    response = client.get("/api/jobs")
    jobs = response.json()
    assert jobs[0]["status"] == "running"


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


@pytest.mark.asyncio
async def test_get_job_with_error(client, service):
    """Test /api/jobs/{job_id} includes error message."""
    request = create_run_job_request(job_id="job-error")
    await service.run_job(request, Mock())

    job = service._manager.get_job("job-error")
    job.status = cluster_pb2.JOB_STATE_FAILED
    job.error = "Something went wrong"

    response = client.get("/api/jobs/job-error")
    data = response.json()
    assert data["error"] == "Something went wrong"


# ============================================================================
# Get logs API tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_logs_empty(client, service):
    """Test /api/jobs/{job_id}/logs with no logs."""
    request = create_run_job_request(job_id="job-no-logs")
    await service.run_job(request, Mock())

    response = client.get("/api/jobs/job-no-logs/logs")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_logs_with_data(client, service):
    """Test /api/jobs/{job_id}/logs returns log entries."""
    request = create_run_job_request(job_id="job-logs")
    await service.run_job(request, Mock())

    # Write some logs
    job = service._manager.get_job("job-logs")
    stdout_file = job.workdir / "STDOUT"
    stdout_file.write_text("\n".join(f"Log line {i}" for i in range(10)))

    response = client.get("/api/jobs/job-logs/logs")
    assert response.status_code == 200
    logs = response.json()

    assert len(logs) == 10
    assert logs[0]["data"] == "Log line 0"
    assert logs[0]["source"] == "stdout"
    assert "timestamp" in logs[0]


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


@pytest.mark.asyncio
async def test_get_logs_without_tail_returns_all(client, service):
    """Test /api/jobs/{job_id}/logs without tail returns all logs."""
    request = create_run_job_request(job_id="job-all-logs")
    await service.run_job(request, Mock())

    job = service._manager.get_job("job-all-logs")
    stdout_file = job.workdir / "STDOUT"
    stdout_file.write_text("\n".join(f"Log line {i}" for i in range(50)))

    response = client.get("/api/jobs/job-all-logs/logs")
    logs = response.json()
    assert len(logs) == 50


# ============================================================================
# Connect RPC tests
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
    server = WorkerServer(service=service, host="127.0.0.1", port=0)

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


@pytest.mark.asyncio
async def test_rpc_health_check_via_connect_client(service):
    """Test calling health_check via Connect RPC client."""
    server = WorkerServer(service=service, host="127.0.0.1", port=0)

    async def run_server():
        import uvicorn

        config = uvicorn.Config(server._app, host="127.0.0.1", port=18081)
        server_obj = uvicorn.Server(config)
        await server_obj.serve()

    server_task = asyncio.create_task(run_server())

    try:
        await asyncio.sleep(0.5)

        async with httpx.AsyncClient() as http_client:
            client = WorkerServiceClient(address="http://127.0.0.1:18081", session=http_client)

            response = await client.health_check(cluster_pb2.Empty())

            assert response.healthy is True
            assert response.uptime_ms >= 0
            assert response.running_jobs == 0

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
    server = WorkerServer(service=service, host="127.0.0.1", port=9999)
    assert server.port == 9999


def test_server_default_host_and_port(service):
    """Test server uses default host and port."""
    server = WorkerServer(service=service)
    assert server._host == "0.0.0.0"
    assert server._port == 8080
