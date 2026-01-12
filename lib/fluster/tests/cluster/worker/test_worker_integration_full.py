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

"""Comprehensive integration tests for Fluster worker.

Tests the complete worker system end-to-end with real Docker containers,
Connect RPC, and all job lifecycle features.
"""

import asyncio
import socket
import subprocess
import time
import zipfile

import cloudpickle
import httpx
import pytest
import pytest_asyncio
from fluster import cluster_pb2
from fluster.cluster.types import Entrypoint
from fluster.cluster.worker.builder import ImageBuilder, VenvCache
from fluster.cluster.worker.bundle import BundleCache
from fluster.cluster.worker.dashboard import WorkerDashboard
from fluster.cluster.worker.manager import JobManager, PortAllocator
from fluster.cluster.worker.runtime import DockerRuntime
from fluster.cluster.worker.service import WorkerServiceImpl
from fluster.cluster_connect import WorkerServiceClient


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


@pytest.fixture
def check_docker():
    """Skip test if Docker is not available."""
    if not check_docker_available():
        pytest.skip("Docker not available")


@pytest.fixture
def workspace_bundle(tmp_path):
    """Create a test workspace bundle with dependencies.

    This creates a realistic workspace with:
    - pyproject.toml with requests and pydantic dependencies
    - Simple Python module to import
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create pyproject.toml with real dependencies
    (workspace / "pyproject.toml").write_text(
        """[project]
name = "test-workspace"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "requests",
    "pydantic",
]
"""
    )

    # Create a simple module
    (workspace / "test_module.py").write_text(
        """def hello():
    return "Hello from test module"

def test_deps():
    import requests
    import pydantic
    return "Dependencies available"
"""
    )

    # Create bundle zip
    bundle_path = tmp_path / "workspace.zip"
    with zipfile.ZipFile(bundle_path, "w") as zf:
        for f in workspace.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(workspace))

    return f"file://{bundle_path}"


@pytest_asyncio.fixture
async def worker_server(tmp_path, check_docker):
    """Start a real worker server on ephemeral port.

    Returns tuple of (dashboard, client, port).
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    uv_cache_dir = tmp_path / "uv_cache"
    uv_cache_dir.mkdir()

    # Initialize components
    bundle_cache = BundleCache(cache_dir, max_bundles=10)
    venv_cache = VenvCache(uv_cache_dir)
    image_builder = ImageBuilder(cache_dir, registry="localhost:5000", max_images=10)
    runtime = DockerRuntime()
    port_allocator = PortAllocator((40000, 40100))

    manager = JobManager(
        bundle_cache=bundle_cache,
        venv_cache=venv_cache,
        image_builder=image_builder,
        runtime=runtime,
        port_allocator=port_allocator,
        max_concurrent_jobs=3,
    )

    service = WorkerServiceImpl(manager)

    # Find a free port
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()

    dashboard = WorkerDashboard(service, host="127.0.0.1", port=port)

    # Run server in background
    server_task = asyncio.create_task(dashboard.run_async())

    # Wait for server to start
    await asyncio.sleep(0.5)

    # Create client
    http_client = httpx.AsyncClient()
    client = WorkerServiceClient(address=f"http://127.0.0.1:{port}", session=http_client)

    try:
        yield dashboard, client, port
    finally:
        await http_client.aclose()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


def create_entrypoint_simple():
    """Create simple test entrypoint that prints and exits."""

    def run():
        print("Job started")
        print("Job completed")
        return 42

    return Entrypoint(callable=run, args=(), kwargs={})


def create_entrypoint_with_deps():
    """Create entrypoint that uses workspace dependencies."""

    def run():
        import pydantic
        import requests

        print("Dependencies loaded successfully")
        print(f"Requests version: {requests.__version__}")
        print(f"Pydantic version: {pydantic.__version__}")
        return "success"

    return Entrypoint(callable=run, args=(), kwargs={})


def create_entrypoint_with_ports():
    """Create entrypoint that reads port environment variables."""

    def run():
        import os

        http_port = os.environ.get("FLUSTER_PORT_HTTP")
        grpc_port = os.environ.get("FLUSTER_PORT_GRPC")
        port_mapping = os.environ.get("FRAY_PORT_MAPPING")

        print(f"HTTP port: {http_port}")
        print(f"GRPC port: {grpc_port}")
        print(f"Port mapping: {port_mapping}")

        assert http_port is not None, "FLUSTER_PORT_HTTP not set"
        assert grpc_port is not None, "FLUSTER_PORT_GRPC not set"
        assert port_mapping == f"http:{http_port},grpc:{grpc_port}"

        return "ports_ok"

    return Entrypoint(callable=run, args=(), kwargs={})


def create_entrypoint_long_running():
    """Create entrypoint that runs for a while (for timeout/kill tests)."""

    def run():
        import time

        print("Starting long job")
        for i in range(60):
            print(f"Tick {i}")
            time.sleep(1)
        print("Job finished")

    return Entrypoint(callable=run, args=(), kwargs={})


def create_entrypoint_failing():
    """Create entrypoint that fails."""

    def run():
        print("About to fail")
        raise RuntimeError("Intentional failure")

    return Entrypoint(callable=run, args=(), kwargs={})


async def wait_for_job_state(client, job_id: str, target_states: set, timeout: float = 60.0):
    """Poll job until it reaches one of the target states.

    Returns the final JobStatus.
    """
    start = time.time()
    while time.time() - start < timeout:
        status = await client.get_job_status(cluster_pb2.GetStatusRequest(job_id=job_id))
        if status.state in target_states:
            return status
        await asyncio.sleep(0.5)

    raise TimeoutError(f"Job {job_id} did not reach target states {target_states} in {timeout}s")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_basic_job_lifecycle(worker_server, workspace_bundle):
    """Test complete job lifecycle from submission to completion."""
    _dashboard, client, _port = worker_server

    # Create and submit job
    entrypoint = create_entrypoint_simple()
    request = cluster_pb2.RunJobRequest(
        job_id="test-lifecycle",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
        timeout_seconds=60,
    )

    response = await client.run_job(request)
    assert response.job_id == "test-lifecycle"
    assert response.state == cluster_pb2.JOB_STATE_PENDING

    # Wait for job to complete
    final_status = await wait_for_job_state(
        client, "test-lifecycle", {cluster_pb2.JOB_STATE_SUCCEEDED, cluster_pb2.JOB_STATE_FAILED}
    )

    # Verify success
    assert final_status.state == cluster_pb2.JOB_STATE_SUCCEEDED
    assert final_status.exit_code == 0
    assert final_status.started_at_ms > 0
    assert final_status.finished_at_ms > final_status.started_at_ms


@pytest.mark.asyncio
@pytest.mark.slow
async def test_job_state_transitions(worker_server, workspace_bundle):
    """Test that job transitions through expected states."""
    _dashboard, client, _port = worker_server

    entrypoint = create_entrypoint_simple()
    request = cluster_pb2.RunJobRequest(
        job_id="test-states",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
    )

    await client.run_job(request)

    # Track state transitions
    seen_states = set()
    for _ in range(120):  # 60 second timeout
        status = await client.get_job_status(cluster_pb2.GetStatusRequest(job_id="test-states"))
        seen_states.add(status.state)

        if status.state in (cluster_pb2.JOB_STATE_SUCCEEDED, cluster_pb2.JOB_STATE_FAILED):
            break

        await asyncio.sleep(0.5)

    # Should see: PENDING → BUILDING → RUNNING → SUCCEEDED
    assert cluster_pb2.JOB_STATE_PENDING in seen_states
    assert cluster_pb2.JOB_STATE_BUILDING in seen_states
    assert cluster_pb2.JOB_STATE_RUNNING in seen_states
    assert cluster_pb2.JOB_STATE_SUCCEEDED in seen_states


@pytest.mark.asyncio
@pytest.mark.slow
async def test_port_allocation_and_environment(worker_server, workspace_bundle):
    """Test port allocation and FLUSTER_PORT_* environment injection."""
    _dashboard, client, _port = worker_server

    entrypoint = create_entrypoint_with_ports()
    request = cluster_pb2.RunJobRequest(
        job_id="test-ports",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
        ports=["http", "grpc"],
    )

    _response = await client.run_job(request)

    # Get status to see allocated ports
    status = await client.get_job_status(cluster_pb2.GetStatusRequest(job_id="test-ports"))

    # Verify ports were allocated
    assert "http" in status.ports
    assert "grpc" in status.ports
    assert 40000 <= status.ports["http"] < 40100
    assert 40000 <= status.ports["grpc"] < 40100
    assert status.ports["http"] != status.ports["grpc"]

    # Wait for job to complete (it validates env vars internally)
    final_status = await wait_for_job_state(
        client, "test-ports", {cluster_pb2.JOB_STATE_SUCCEEDED, cluster_pb2.JOB_STATE_FAILED}
    )

    assert final_status.state == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.slow
async def test_cloudpickle_deserialization(worker_server, workspace_bundle):
    """Test that cloudpickle-serialized functions execute correctly."""
    _dashboard, client, _port = worker_server

    # Create entrypoint with closure
    captured_value = "test_value_123"

    def entrypoint_with_closure():
        print(f"Captured: {captured_value}")
        assert captured_value == "test_value_123"
        return captured_value

    entrypoint = Entrypoint(callable=entrypoint_with_closure, args=(), kwargs={})

    request = cluster_pb2.RunJobRequest(
        job_id="test-pickle",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
    )

    await client.run_job(request)

    final_status = await wait_for_job_state(
        client, "test-pickle", {cluster_pb2.JOB_STATE_SUCCEEDED, cluster_pb2.JOB_STATE_FAILED}
    )

    assert final_status.state == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.slow
async def test_dependency_caching(worker_server, workspace_bundle):
    """Test that venv/image caching works (second job faster than first)."""
    _dashboard, client, _port = worker_server

    entrypoint = create_entrypoint_with_deps()

    # Submit first job and measure time
    request1 = cluster_pb2.RunJobRequest(
        job_id="test-cache-1",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
    )

    start1 = time.time()
    await client.run_job(request1)
    await wait_for_job_state(client, "test-cache-1", {cluster_pb2.JOB_STATE_SUCCEEDED, cluster_pb2.JOB_STATE_FAILED})
    duration1 = time.time() - start1

    # Submit second identical job
    request2 = cluster_pb2.RunJobRequest(
        job_id="test-cache-2",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
    )

    start2 = time.time()
    await client.run_job(request2)
    final_status = await wait_for_job_state(
        client, "test-cache-2", {cluster_pb2.JOB_STATE_SUCCEEDED, cluster_pb2.JOB_STATE_FAILED}
    )
    duration2 = time.time() - start2

    assert final_status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # Second job should be significantly faster (cached image)
    # Allow some margin but expect at least 30% speedup
    assert duration2 < duration1 * 0.7, f"Second job ({duration2}s) not faster than first ({duration1}s)"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_job_timeout(worker_server, workspace_bundle):
    """Test that job timeout works correctly."""
    _dashboard, client, _port = worker_server

    entrypoint = create_entrypoint_long_running()
    request = cluster_pb2.RunJobRequest(
        job_id="test-timeout",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
        timeout_seconds=3,  # Short timeout
    )

    await client.run_job(request)

    final_status = await wait_for_job_state(
        client, "test-timeout", {cluster_pb2.JOB_STATE_FAILED, cluster_pb2.JOB_STATE_KILLED}
    )

    # Should fail due to timeout
    assert final_status.state == cluster_pb2.JOB_STATE_FAILED
    assert "timeout" in final_status.error.lower()


@pytest.mark.asyncio
@pytest.mark.slow
async def test_job_killing_graceful(worker_server, workspace_bundle):
    """Test graceful job termination (SIGTERM)."""
    _dashboard, client, _port = worker_server

    entrypoint = create_entrypoint_long_running()
    request = cluster_pb2.RunJobRequest(
        job_id="test-kill-graceful",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
    )

    await client.run_job(request)

    # Wait for job to start running
    await wait_for_job_state(client, "test-kill-graceful", {cluster_pb2.JOB_STATE_RUNNING})

    # Kill the job with graceful timeout
    await client.kill_job(
        cluster_pb2.KillJobRequest(
            job_id="test-kill-graceful",
            term_timeout_ms=5000,
        )
    )

    # Verify killed state
    status = await client.get_job_status(cluster_pb2.GetStatusRequest(job_id="test-kill-graceful"))
    assert status.state == cluster_pb2.JOB_STATE_KILLED


@pytest.mark.asyncio
@pytest.mark.slow
async def test_job_killing_forced(worker_server, workspace_bundle):
    """Test forced job termination (SIGKILL after timeout)."""
    _dashboard, client, _port = worker_server

    entrypoint = create_entrypoint_long_running()
    request = cluster_pb2.RunJobRequest(
        job_id="test-kill-forced",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
    )

    await client.run_job(request)
    await wait_for_job_state(client, "test-kill-forced", {cluster_pb2.JOB_STATE_RUNNING})

    # Kill with very short graceful timeout to force SIGKILL
    start = time.time()
    await client.kill_job(
        cluster_pb2.KillJobRequest(
            job_id="test-kill-forced",
            term_timeout_ms=100,  # Very short
        )
    )
    duration = time.time() - start

    status = await client.get_job_status(cluster_pb2.GetStatusRequest(job_id="test-kill-forced"))
    assert status.state == cluster_pb2.JOB_STATE_KILLED

    # Should complete quickly (forced kill)
    assert duration < 2.0


@pytest.mark.asyncio
@pytest.mark.slow
async def test_log_retrieval(worker_server, workspace_bundle):
    """Test log retrieval with filtering."""
    _dashboard, client, _port = worker_server

    # Create entrypoint that generates logs
    def log_generator():
        for i in range(20):
            if i % 2 == 0:
                print(f"INFO: Line {i}")
            else:
                print(f"ERROR: Line {i}")

    entrypoint = Entrypoint(callable=log_generator, args=(), kwargs={})
    request = cluster_pb2.RunJobRequest(
        job_id="test-logs",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
    )

    await client.run_job(request)
    await wait_for_job_state(client, "test-logs", {cluster_pb2.JOB_STATE_SUCCEEDED, cluster_pb2.JOB_STATE_FAILED})

    # Test 1: Get all logs
    all_logs = await client.fetch_logs(
        cluster_pb2.FetchLogsRequest(
            job_id="test-logs",
            filter=cluster_pb2.FetchLogsFilter(),
        )
    )
    assert len(all_logs.logs) == 20

    # Test 2: Filter by regex (ERROR lines only)
    error_logs = await client.fetch_logs(
        cluster_pb2.FetchLogsRequest(
            job_id="test-logs",
            filter=cluster_pb2.FetchLogsFilter(regex="ERROR"),
        )
    )
    assert len(error_logs.logs) == 10
    assert all("ERROR" in log.data for log in error_logs.logs)

    # Test 3: Tail last 5 lines
    tail_logs = await client.fetch_logs(
        cluster_pb2.FetchLogsRequest(
            job_id="test-logs",
            filter=cluster_pb2.FetchLogsFilter(start_line=-5),
        )
    )
    assert len(tail_logs.logs) == 5
    assert "Line 19" in tail_logs.logs[-1].data

    # Test 4: Max lines limit
    limited_logs = await client.fetch_logs(
        cluster_pb2.FetchLogsRequest(
            job_id="test-logs",
            filter=cluster_pb2.FetchLogsFilter(max_lines=3),
        )
    )
    assert len(limited_logs.logs) == 3


@pytest.mark.asyncio
@pytest.mark.slow
async def test_concurrent_jobs(worker_server, workspace_bundle):
    """Test concurrent job execution respects max_concurrent_jobs."""
    _dashboard, client, _port = worker_server

    # Server fixture has max_concurrent_jobs=3
    entrypoint = create_entrypoint_simple()

    # Submit 6 jobs
    job_ids = []
    for i in range(6):
        request = cluster_pb2.RunJobRequest(
            job_id=f"test-concurrent-{i}",
            serialized_entrypoint=cloudpickle.dumps(entrypoint),
            bundle_gcs_path=workspace_bundle,
            environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
        )
        await client.run_job(request)
        job_ids.append(f"test-concurrent-{i}")

    # Give jobs time to start
    await asyncio.sleep(2)

    # Check how many are running
    running_count = 0
    for job_id in job_ids:
        status = await client.get_job_status(cluster_pb2.GetStatusRequest(job_id=job_id))
        if status.state == cluster_pb2.JOB_STATE_RUNNING:
            running_count += 1

    # Should have at most 3 running (max_concurrent_jobs)
    assert running_count <= 3


@pytest.mark.asyncio
@pytest.mark.slow
async def test_failing_job(worker_server, workspace_bundle):
    """Test that failing jobs are marked as FAILED."""
    _dashboard, client, _port = worker_server

    entrypoint = create_entrypoint_failing()
    request = cluster_pb2.RunJobRequest(
        job_id="test-failure",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
    )

    await client.run_job(request)

    final_status = await wait_for_job_state(client, "test-failure", {cluster_pb2.JOB_STATE_FAILED})

    assert final_status.state == cluster_pb2.JOB_STATE_FAILED
    assert final_status.exit_code != 0
    assert len(final_status.error) > 0


@pytest.mark.asyncio
@pytest.mark.slow
async def test_health_check(worker_server):
    """Test health check endpoint."""
    _dashboard, client, _port = worker_server

    health = await client.health_check(cluster_pb2.Empty())

    assert health.healthy
    assert health.uptime_ms > 0
    assert health.running_jobs >= 0


@pytest.mark.asyncio
@pytest.mark.slow
async def test_list_jobs(worker_server, workspace_bundle):
    """Test listing all jobs."""
    _dashboard, client, _port = worker_server

    # Submit several jobs
    for i in range(3):
        entrypoint = create_entrypoint_simple()
        request = cluster_pb2.RunJobRequest(
            job_id=f"test-list-{i}",
            serialized_entrypoint=cloudpickle.dumps(entrypoint),
            bundle_gcs_path=workspace_bundle,
            environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
        )
        await client.run_job(request)

    # List jobs
    response = await client.list_jobs(cluster_pb2.ListJobsRequest())

    assert len(response.jobs) >= 3
    job_ids = {job.job_id for job in response.jobs}
    assert "test-list-0" in job_ids
    assert "test-list-1" in job_ids
    assert "test-list-2" in job_ids


def create_entrypoint_memory_allocator():
    """Create entrypoint that allocates memory and runs for 10+ seconds."""

    def run():
        import time

        print("Starting memory allocation test")
        # Allocate approximately 50MB of memory
        data = []
        for i in range(50):
            # Each allocation is ~1MB
            chunk = "x" * (1024 * 1024)
            data.append(chunk)
            if i % 10 == 0:
                print(f"Allocated {i + 1} MB")

        print("Memory allocated, sleeping for 12 seconds")
        time.sleep(12)
        print("Job completed")
        return len(data)

    return Entrypoint(callable=run, args=(), kwargs={})


def create_entrypoint_with_timed_output():
    """Create entrypoint that prints messages with sleep between them."""

    def run():
        import time

        print("First message")
        time.sleep(2)
        print("Second message")
        return "done"

    return Entrypoint(callable=run, args=(), kwargs={})


@pytest.mark.asyncio
@pytest.mark.slow
async def test_resource_monitoring(worker_server, workspace_bundle):
    """Test that resource metrics are collected during job execution.

    This test verifies that the stats collection loop (5s interval) properly
    collects memory, CPU, and process count metrics during job execution.
    """
    _dashboard, client, _port = worker_server

    entrypoint = create_entrypoint_memory_allocator()
    request = cluster_pb2.RunJobRequest(
        job_id="test-resource-monitoring",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
        timeout_seconds=60,
    )

    await client.run_job(request)

    # Wait for job to start running
    await wait_for_job_state(client, "test-resource-monitoring", {cluster_pb2.JOB_STATE_RUNNING})

    # Wait 6 seconds for at least one stats collection (poll interval is 5s)
    await asyncio.sleep(6)

    # Get status and verify resource metrics are populated
    status = await client.get_job_status(cluster_pb2.GetStatusRequest(job_id="test-resource-monitoring"))

    # Verify resource usage is being tracked
    assert status.resource_usage is not None
    assert status.resource_usage.memory_mb > 0, "Memory usage should be greater than 0"
    assert status.resource_usage.process_count > 0, "Process count should be greater than 0"
    assert status.resource_usage.memory_peak_mb >= status.resource_usage.memory_mb, "Peak memory should be >= current"

    # Wait for job to complete
    final_status = await wait_for_job_state(
        client, "test-resource-monitoring", {cluster_pb2.JOB_STATE_SUCCEEDED, cluster_pb2.JOB_STATE_FAILED}
    )

    assert final_status.state == cluster_pb2.JOB_STATE_SUCCEEDED
    # Peak memory should reflect the allocated memory
    assert final_status.resource_usage.memory_peak_mb > 0


@pytest.mark.asyncio
@pytest.mark.slow
async def test_build_log_capture(worker_server, workspace_bundle):
    """Test that build logs are captured and retrievable.

    This test verifies that logs with source="build" are properly collected
    during the image build phase and can be fetched via FetchLogs.
    """
    _dashboard, client, _port = worker_server

    entrypoint = create_entrypoint_simple()
    request = cluster_pb2.RunJobRequest(
        job_id="test-build-logs",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
    )

    await client.run_job(request)

    # Wait for job to complete
    await wait_for_job_state(client, "test-build-logs", {cluster_pb2.JOB_STATE_SUCCEEDED, cluster_pb2.JOB_STATE_FAILED})

    # Fetch all logs
    all_logs = await client.fetch_logs(
        cluster_pb2.FetchLogsRequest(
            job_id="test-build-logs",
            filter=cluster_pb2.FetchLogsFilter(),
        )
    )

    # Filter for build logs (logs should have source field)
    build_logs = [log for log in all_logs.logs if log.source == "build"]

    # Assert build logs exist
    assert len(build_logs) > 0, "Build logs should be captured"

    # Check that build logs contain expected keywords
    build_log_text = " ".join(log.data.lower() for log in build_logs)
    # Build logs should mention docker operations
    build_keywords = ["image", "docker", "cache", "build"]
    found_keywords = [kw for kw in build_keywords if kw in build_log_text]
    assert len(found_keywords) > 0, f"Build logs should contain build-related keywords, got: {build_log_text[:200]}"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_log_timestamps_accurate(worker_server, workspace_bundle):
    """Test that log timestamps reflect actual write times.

    This test verifies that LogEntry timestamps are accurate and properly
    reflect when logs were written, not when they were collected.
    """
    _dashboard, client, _port = worker_server

    entrypoint = create_entrypoint_with_timed_output()
    request = cluster_pb2.RunJobRequest(
        job_id="test-log-timestamps",
        serialized_entrypoint=cloudpickle.dumps(entrypoint),
        bundle_gcs_path=workspace_bundle,
        environment=cluster_pb2.EnvironmentConfig(workspace="/app"),
    )

    await client.run_job(request)

    # Wait for job to complete
    await wait_for_job_state(
        client, "test-log-timestamps", {cluster_pb2.JOB_STATE_SUCCEEDED, cluster_pb2.JOB_STATE_FAILED}
    )

    # Fetch logs that contain our messages
    logs = await client.fetch_logs(
        cluster_pb2.FetchLogsRequest(
            job_id="test-log-timestamps",
            filter=cluster_pb2.FetchLogsFilter(regex="(First message|Second message)"),
        )
    )

    # Find the two messages
    first_message = None
    second_message = None

    for log in logs.logs:
        if "First message" in log.data:
            first_message = log
        elif "Second message" in log.data:
            second_message = log

    assert first_message is not None, "First message log not found"
    assert second_message is not None, "Second message log not found"

    # Calculate time difference in milliseconds
    time_diff_ms = second_message.timestamp_ms - first_message.timestamp_ms

    # Should be approximately 2000ms (2 seconds), allow ±500ms tolerance
    assert 1500 <= time_diff_ms <= 2500, (
        f"Timestamp difference should be ~2000ms (±500ms), got {time_diff_ms}ms. "
        f"First: {first_message.timestamp_ms}, Second: {second_message.timestamp_ms}"
    )
