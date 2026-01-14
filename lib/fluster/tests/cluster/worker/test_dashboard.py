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
from unittest.mock import Mock

import cloudpickle
import httpx
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from starlette.testclient import TestClient

from fluster import cluster_pb2
from fluster.cluster.worker.builder import BuildResult, ImageCache, VenvCache
from fluster.cluster.worker.bundle import BundleCache
from fluster.cluster.worker.dashboard import WorkerDashboard
from fluster.cluster.worker.docker import ContainerStats, ContainerStatus, DockerRuntime
from fluster.cluster.worker.manager import JobManager, PortAllocator
from fluster.cluster.worker.service import WorkerServiceImpl
from fluster.cluster_connect import WorkerServiceClient

# ============================================================================
# Shared fixtures
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
    """Create mock ImageCache."""
    cache = Mock(spec=ImageCache)
    cache.build = Mock(
        return_value=BuildResult(
            image_tag="test-image:latest",
            deps_hash="abc123",
            build_time_ms=1000,
            from_cache=False,
        )
    )
    return cache


@pytest.fixture
def mock_runtime():
    """Create mock DockerRuntime for sync model.

    The sync model uses create_container/start_container/inspect pattern
    instead of the blocking run() method.
    """
    runtime = Mock(spec=DockerRuntime)

    # Container lifecycle methods
    runtime.create_container = Mock(return_value="container123")
    runtime.start_container = Mock()

    # Inspect returns not-running so jobs complete immediately
    runtime.inspect = Mock(return_value=ContainerStatus(running=False, exit_code=0))

    runtime.kill = Mock()
    runtime.remove = Mock()
    runtime.get_stats = Mock(
        return_value=ContainerStats(
            memory_mb=100,
            cpu_percent=50,
            process_count=1,
            available=True,
        )
    )
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


def test_stats_empty(client, service):
    """Test /api/stats with no jobs."""
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert data == {"running": 0, "pending": 0, "building": 0, "completed": 0}


def test_stats_with_jobs(client, service):
    """Test /api/stats with various job states."""
    # Submit jobs and let them complete
    for i in range(5):
        request = create_run_job_request(job_id=f"job-{i}")
        service.run_job(request, Mock())

    # Wait for all job threads to complete (mock makes them finish immediately)
    jobs = service._manager.list_jobs()
    for job in jobs:
        if job.thread:
            job.thread.join(timeout=5.0)

    # Now manually set states (threads are done, we control the state)
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


def test_list_jobs_with_data(client, service):
    """Test /api/jobs returns all jobs."""
    for i in range(3):
        request = create_run_job_request(job_id=f"job-{i}")
        service.run_job(request, Mock())

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


def test_get_job_success(client, service):
    """Test /api/jobs/{job_id} returns job details."""
    request = create_run_job_request(job_id="job-details", ports=["http", "grpc"])
    service.run_job(request, Mock())

    # Wait for job to complete (mock runtime returns running=False immediately)
    job = service._manager.get_job("job-details")
    job.thread.join(timeout=5.0)

    response = client.get("/api/jobs/job-details")
    assert response.status_code == 200
    data = response.json()

    assert data["job_id"] == "job-details"
    assert data["status"] == "succeeded"  # Job completes immediately with mock runtime
    assert data["exit_code"] == 0
    assert "http" in data["ports"]
    assert "grpc" in data["ports"]


# ============================================================================
# Get logs API tests
# ============================================================================


def test_get_logs_with_tail_parameter(client, service):
    """Test /api/jobs/{job_id}/logs?tail=N returns last N lines."""
    request = create_run_job_request(job_id="job-tail")
    service.run_job(request, Mock())

    # Add logs directly to job.logs (since we no longer use file-based logging)
    job = service._manager.get_job("job-tail")
    for i in range(100):
        job.logs.add("stdout", f"Log line {i}")

    response = client.get("/api/jobs/job-tail/logs?tail=5")
    assert response.status_code == 200
    logs = response.json()

    assert len(logs) == 5
    assert logs[0]["data"] == "Log line 95"
    assert logs[4]["data"] == "Log line 99"


def test_get_logs_with_source_filter(client, service):
    """Test /api/jobs/{job_id}/logs?source=stdout filters by source."""
    import time

    request = create_run_job_request(job_id="job-source-filter")
    service.run_job(request, Mock())

    # Stop the job thread so it doesn't add more logs
    job = service._manager.get_job("job-source-filter")
    time.sleep(0.05)
    job.should_stop = True
    if job.thread:
        job.thread.join(timeout=1.0)

    # Clear any existing logs and add test logs
    job.logs.lines.clear()
    job.logs.add("stdout", "stdout line 1")
    job.logs.add("stdout", "stdout line 2")
    job.logs.add("stderr", "stderr line 1")
    job.logs.add("stderr", "stderr line 2")

    # Test stdout filter
    response = client.get("/api/jobs/job-source-filter/logs?source=stdout")
    assert response.status_code == 200
    logs = response.json()
    assert len(logs) == 2
    assert all(log["source"] == "stdout" for log in logs)

    # Test stderr filter
    response = client.get("/api/jobs/job-source-filter/logs?source=stderr")
    assert response.status_code == 200
    logs = response.json()
    assert len(logs) == 2
    assert all(log["source"] == "stderr" for log in logs)

    # Test without filter - should get all logs
    response = client.get("/api/jobs/job-source-filter/logs")
    assert response.status_code == 200
    logs = response.json()
    assert len(logs) == 4  # 2 stdout + 2 stderr


def test_list_jobs_includes_resource_and_build_metrics(client, service):
    """Test /api/jobs includes resource and build metrics."""
    request = create_run_job_request(job_id="job-with-metrics")
    service.run_job(request, Mock())

    # Set some resource and build metrics
    job = service._manager.get_job("job-with-metrics")
    job.current_memory_mb = 256
    job.peak_memory_mb = 512
    job.current_cpu_percent = 45
    job.process_count = 3
    job.disk_mb = 1024
    job.build_from_cache = True
    job.image_tag = "test-image:v1.0"

    response = client.get("/api/jobs")
    assert response.status_code == 200
    jobs = response.json()

    job_data = next(j for j in jobs if j["job_id"] == "job-with-metrics")
    assert job_data["memory_mb"] == 256
    assert job_data["memory_peak_mb"] == 512
    assert job_data["cpu_percent"] == 45
    assert job_data["process_count"] == 3
    assert job_data["disk_mb"] == 1024
    assert job_data["build_from_cache"] is True
    assert job_data["image_tag"] == "test-image:v1.0"


def test_get_job_includes_nested_resources_and_build(client, service):
    """Test /api/jobs/{job_id} includes nested resources and build objects."""
    request = create_run_job_request(job_id="job-nested-metrics")
    service.run_job(request, Mock())

    # Set some resource and build metrics
    job = service._manager.get_job("job-nested-metrics")
    job.current_memory_mb = 128
    job.peak_memory_mb = 256
    job.current_cpu_percent = 30
    job.process_count = 2
    job.disk_mb = 512
    job.build_started_ms = 1000
    job.build_finished_ms = 2500
    job.build_from_cache = False
    job.image_tag = "test-image:v2.0"

    response = client.get("/api/jobs/job-nested-metrics")
    assert response.status_code == 200
    data = response.json()

    # Check resources nested object
    assert "resources" in data
    resources = data["resources"]
    assert resources["memory_mb"] == 128
    assert resources["memory_peak_mb"] == 256
    assert resources["cpu_percent"] == 30
    assert resources["process_count"] == 2
    assert resources["disk_mb"] == 512

    # Check build nested object
    assert "build" in data
    build = data["build"]
    assert build["started_ms"] == 1000
    assert build["finished_ms"] == 2500
    assert build["duration_ms"] == 1500
    assert build["from_cache"] is False
    assert build["image_tag"] == "test-image:v2.0"


def test_job_detail_page_loads(client):
    """Test /job/{job_id} page loads successfully."""
    response = client.get("/job/test-job-123")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert "Job: <code>test-job-123</code>" in response.text
    assert "Back to Dashboard" in response.text


# ============================================================================
# RPC Service tests (WorkerServiceImpl)
# ============================================================================


def test_run_job_generates_job_id_if_missing(service, request_context):
    """Test run_job generates job_id when not provided."""
    request = create_run_job_request(job_id="")
    response = service.run_job(request, request_context)

    assert response.job_id  # Should have a generated ID
    assert len(response.job_id) > 0
    # Job may have already transitioned from PENDING since threads start immediately
    assert response.state in (
        cluster_pb2.JOB_STATE_PENDING,
        cluster_pb2.JOB_STATE_BUILDING,
        cluster_pb2.JOB_STATE_RUNNING,
        cluster_pb2.JOB_STATE_SUCCEEDED,
    )


def test_run_job_with_ports(service, request_context):
    """Test run_job allocates ports correctly."""
    request = create_run_job_request(job_id="job-with-ports", ports=["http", "grpc"])
    response = service.run_job(request, request_context)

    assert response.job_id == "job-with-ports"

    # Verify ports were allocated
    job = service._manager.get_job("job-with-ports")
    assert len(job.ports) == 2
    assert "http" in job.ports
    assert "grpc" in job.ports


def test_get_job_status_not_found(service, request_context):
    """Test get_job_status raises NOT_FOUND for nonexistent job."""
    status_request = cluster_pb2.Worker.GetJobStatusRequest(job_id="nonexistent")

    with pytest.raises(ConnectError) as exc_info:
        service.get_job_status(status_request, request_context)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in str(exc_info.value)


def test_get_job_status_completed_job(service, request_context):
    """Test get_job_status for completed job includes timing info."""
    request = create_run_job_request(job_id="job-completed")
    service.run_job(request, request_context)

    # Wait for job to complete
    job = service._manager.get_job("job-completed")
    job.thread.join(timeout=5.0)

    status_request = cluster_pb2.Worker.GetJobStatusRequest(job_id="job-completed")
    status = service.get_job_status(status_request, request_context)

    assert status.job_id == "job-completed"
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
    assert status.exit_code == 0
    assert status.started_at_ms > 0
    assert status.finished_at_ms > 0


def test_fetch_logs_tail_with_negative_start_line(service, request_context):
    """Test fetch_logs with negative start_line for tailing."""
    request = create_run_job_request(job_id="job-logs-tail")
    service.run_job(request, request_context)

    # Add logs directly to job.logs
    job = service._manager.get_job("job-logs-tail")
    for i in range(10):
        job.logs.add("stdout", f"Log line {i}")

    log_filter = cluster_pb2.Worker.FetchLogsFilter(start_line=-3)
    logs_request = cluster_pb2.Worker.FetchLogsRequest(job_id="job-logs-tail", filter=log_filter)
    response = service.fetch_logs(logs_request, request_context)

    assert len(response.logs) == 3
    assert response.logs[0].data == "Log line 7"
    assert response.logs[1].data == "Log line 8"
    assert response.logs[2].data == "Log line 9"


def test_fetch_logs_with_regex_filter(service, request_context):
    """Test fetch_logs with regex content filter."""
    request = create_run_job_request(job_id="job-logs-regex")
    service.run_job(request, request_context)

    # Add logs with different patterns
    job = service._manager.get_job("job-logs-regex")
    job.logs.add("stdout", "ERROR: something bad")
    job.logs.add("stdout", "INFO: normal log")
    job.logs.add("stdout", "ERROR: another error")
    job.logs.add("stdout", "DEBUG: details")

    log_filter = cluster_pb2.Worker.FetchLogsFilter(regex="ERROR")
    logs_request = cluster_pb2.Worker.FetchLogsRequest(job_id="job-logs-regex", filter=log_filter)
    response = service.fetch_logs(logs_request, request_context)

    assert len(response.logs) == 2
    assert "ERROR" in response.logs[0].data
    assert "ERROR" in response.logs[1].data


def test_fetch_logs_combined_filters(service, request_context):
    """Test fetch_logs with multiple filters combined."""
    request = create_run_job_request(job_id="job-logs-combined")
    service.run_job(request, request_context)

    # Add logs
    job = service._manager.get_job("job-logs-combined")
    job.logs.add("stdout", "ERROR: first error")
    job.logs.add("stdout", "INFO: normal")
    job.logs.add("stdout", "ERROR: second error")
    job.logs.add("stdout", "ERROR: third error")
    job.logs.add("stdout", "ERROR: fourth error")
    job.logs.add("stdout", "ERROR: fifth error")

    # Use regex to filter ERRORs, then limit to 2
    log_filter = cluster_pb2.Worker.FetchLogsFilter(regex="ERROR", max_lines=2)
    logs_request = cluster_pb2.Worker.FetchLogsRequest(job_id="job-logs-combined", filter=log_filter)
    response = service.fetch_logs(logs_request, request_context)

    assert len(response.logs) == 2
    assert "ERROR" in response.logs[0].data
    assert "ERROR" in response.logs[1].data


def test_kill_job_not_found(service, request_context):
    """Test kill_job raises NOT_FOUND for nonexistent job."""
    kill_request = cluster_pb2.Worker.KillJobRequest(job_id="nonexistent")

    with pytest.raises(ConnectError) as exc_info:
        service.kill_job(kill_request, request_context)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in str(exc_info.value)


def test_kill_job_already_completed(service, request_context):
    """Test kill_job fails for already completed job."""
    request = create_run_job_request(job_id="job-completed")
    service.run_job(request, request_context)

    # Wait for job to complete
    job = service._manager.get_job("job-completed")
    job.thread.join(timeout=5.0)

    # Try to kill completed job
    kill_request = cluster_pb2.Worker.KillJobRequest(job_id="job-completed")

    with pytest.raises(ConnectError) as exc_info:
        service.kill_job(kill_request, request_context)

    assert exc_info.value.code == Code.FAILED_PRECONDITION
    assert "already completed" in str(exc_info.value)


def test_kill_job_with_custom_timeout(service, request_context):
    """Test kill_job accepts custom term_timeout_ms and attempts termination.

    Note: With mocks, the job thread completes immediately. This test verifies
    the API works and runtime.kill is called, not the actual kill behavior.
    """
    request = create_run_job_request(job_id="job-kill")
    service.run_job(request, request_context)

    # Wait for job thread to finish (mock makes it complete immediately)
    job = service._manager.get_job("job-kill")
    if job.thread:
        job.thread.join(timeout=5.0)

    # Manually set job to RUNNING to simulate mid-execution state
    job.status = cluster_pb2.JOB_STATE_RUNNING
    job.container_id = "container123"

    kill_request = cluster_pb2.Worker.KillJobRequest(job_id="job-kill", term_timeout_ms=100)
    response = service.kill_job(kill_request, request_context)

    # Verify API response and that should_stop was set
    assert isinstance(response, cluster_pb2.Empty)
    assert job.should_stop is True
    # The runtime.kill should have been called (may be called twice: SIGTERM then SIGKILL)
    assert service._manager._runtime.kill.called


# ============================================================================
# Connect RPC integration tests
# ============================================================================


def test_rpc_endpoint_mounted_correctly(server):
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

        # Create Connect client (async client talks to WSGI server via WSGIMiddleware)
        async with httpx.AsyncClient() as http_client:
            client = WorkerServiceClient(address="http://127.0.0.1:18080", session=http_client)

            # Submit job via RPC
            request = create_run_job_request(job_id="rpc-test-job")
            response = await client.run_job(request)

            assert response.job_id == "rpc-test-job"
            # Job may have already transitioned from PENDING since threads start immediately
            assert response.state in (
                cluster_pb2.JOB_STATE_PENDING,
                cluster_pb2.JOB_STATE_BUILDING,
                cluster_pb2.JOB_STATE_RUNNING,
                cluster_pb2.JOB_STATE_SUCCEEDED,
            )

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
