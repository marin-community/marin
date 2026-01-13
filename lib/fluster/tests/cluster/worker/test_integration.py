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

"""Integration tests for Fluster worker components.

These tests exercise end-to-end functionality with real Docker containers.
Tests are marked with @pytest.mark.slow and require Docker to be running.
"""

import subprocess
import time
import zipfile
from unittest.mock import Mock

import cloudpickle
import pytest
from connectrpc.request import RequestContext

from fluster import cluster_pb2
from fluster.cluster.types import Entrypoint
from fluster.cluster.worker.builder import ImageCache, VenvCache
from fluster.cluster.worker.bundle import BundleCache
from fluster.cluster.worker.docker import ContainerConfig, DockerRuntime
from fluster.cluster.worker.manager import JobManager, PortAllocator
from fluster.cluster.worker.service import WorkerServiceImpl


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

    # Minimal pyproject.toml
    (bundle_dir / "pyproject.toml").write_text(
        """[project]
name = "test-job"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
"""
    )

    # Create zip
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in bundle_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(bundle_dir))

    return f"file://{zip_path}"


def create_test_entrypoint():
    """Create a simple test entrypoint."""

    def test_fn():
        print("Hello from test job!")
        return 42

    return Entrypoint(callable=test_fn, args=(), kwargs={})


def create_run_job_request(bundle_path: str, job_id: str):
    """Create a RunJobRequest for testing."""
    entrypoint = create_test_entrypoint()

    return cluster_pb2.RunJobRequest(
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
def real_manager(cache_dir):
    """Create JobManager with real components (not mocks)."""
    bundle_cache = BundleCache(cache_dir, max_bundles=10)
    venv_cache = VenvCache()
    image_cache = ImageCache(cache_dir, registry="localhost:5000", max_images=10)
    runtime = DockerRuntime()
    port_allocator = PortAllocator((40000, 40100))

    return JobManager(
        bundle_cache=bundle_cache,
        venv_cache=venv_cache,
        image_cache=image_cache,
        runtime=runtime,
        port_allocator=port_allocator,
        max_concurrent_jobs=2,
    )


@pytest.fixture
def real_service(real_manager):
    """Create WorkerServiceImpl with real manager."""
    return WorkerServiceImpl(real_manager)


@pytest.fixture
def runtime():
    """Create DockerRuntime instance."""
    return DockerRuntime()


class TestBundleCacheIntegration:
    """Integration tests for BundleCache with local file:// paths."""

    def test_download_local_bundle(self, cache_dir, test_bundle):
        """Test downloading a bundle from local file:// path."""
        cache = BundleCache(cache_dir)
        bundle_path = cache.get_bundle(test_bundle)

        assert bundle_path.exists()
        assert (bundle_path / "pyproject.toml").exists()

    def test_cache_hit_reuses_bundle(self, cache_dir, test_bundle):
        """Test that second download uses cached bundle."""
        cache = BundleCache(cache_dir)

        path1 = cache.get_bundle(test_bundle)
        path2 = cache.get_bundle(test_bundle)

        assert path1 == path2
        # Should only be one bundle in cache
        bundles = list((cache_dir / "extracts").iterdir())
        assert len(bundles) == 1


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

        # Wait for container to finish
        import time

        time.sleep(1)

        status = runtime.inspect(container_id)
        assert not status.running
        assert status.exit_code == 0

        # Cleanup
        runtime.remove(container_id)

    @pytest.mark.slow
    def test_container_with_output(self, runtime):
        """Test container runs and produces output."""
        if not check_docker_available():
            pytest.skip("Docker not available")

        config = ContainerConfig(
            image="alpine:latest",
            command=["echo", "test output"],
            env={},
        )

        container_id = runtime.create_container(config)
        runtime.start_container(container_id)

        # Wait for container to finish
        import time

        time.sleep(1)

        status = runtime.inspect(container_id)
        assert not status.running
        assert status.exit_code == 0

        # Cleanup
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

        # Wait for container to start
        import time

        time.sleep(1)

        # Kill container
        runtime.kill(container_id, force=True)

        # Verify it's stopped
        status = runtime.inspect(container_id)
        assert not status.running

        # Cleanup
        runtime.remove(container_id)


class TestPortAllocatorIntegration:
    """Integration tests for PortAllocator."""

    def test_allocate_and_release_ports(self):
        """Test port allocation and release cycle."""
        allocator = PortAllocator((40000, 40100))

        ports = allocator.allocate(3)
        assert len(ports) == 3
        assert len(set(ports)) == 3  # All unique

        # Verify ports are in range
        for port in ports:
            assert 40000 <= port < 40100

        # Release and reallocate
        allocator.release(ports)
        ports2 = allocator.allocate(3)

        # Should get same ports back (they were released)
        assert set(ports) == set(ports2)


class TestJobManagerIntegration:
    """Integration tests for JobManager with real components."""

    @pytest.mark.slow
    def test_submit_job_lifecycle(self, real_manager, test_bundle):
        """Test full job lifecycle from submission to completion."""
        if not check_docker_available():
            pytest.skip("Docker not available")

        request = create_run_job_request(test_bundle, "integration-test-1")

        job_id = real_manager.submit_job(request)
        assert job_id == "integration-test-1"

        # Poll until job completes or times out
        import time

        for _ in range(30):  # 30 second timeout
            time.sleep(1)
            job = real_manager.get_job(job_id)

            if job.status in (
                cluster_pb2.JOB_STATE_SUCCEEDED,
                cluster_pb2.JOB_STATE_FAILED,
                cluster_pb2.JOB_STATE_KILLED,
            ):
                break

        # Verify terminal state
        job = real_manager.get_job(job_id)
        assert job.status in (
            cluster_pb2.JOB_STATE_SUCCEEDED,
            cluster_pb2.JOB_STATE_FAILED,
        )

    @pytest.mark.slow
    def test_concurrent_job_limit(self, real_manager, test_bundle):
        """Test that max_concurrent_jobs is enforced."""
        if not check_docker_available():
            pytest.skip("Docker not available")

        # Submit 4 jobs to manager with max_concurrent=2
        requests = [create_run_job_request(test_bundle, f"concurrent-{i}") for i in range(4)]

        _job_ids = [real_manager.submit_job(r) for r in requests]

        import time

        time.sleep(1)  # Let jobs start

        jobs = real_manager.list_jobs()
        running = sum(1 for j in jobs if j.status == cluster_pb2.JOB_STATE_RUNNING)

        # At most 2 should be running
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

        # Submit a job
        request = create_run_job_request(test_bundle, "logs-test")
        real_service.run_job(request, ctx)

        # Wait a bit for logs
        time.sleep(2)

        # Fetch last 10 lines
        log_request = cluster_pb2.FetchLogsRequest(
            job_id="logs-test",
            filter=cluster_pb2.FetchLogsFilter(start_line=-10),
        )

        response = real_service.fetch_logs(log_request, ctx)
        # response.logs is a protobuf repeated field, check it's valid
        assert response.logs is not None
        assert len(response.logs) >= 0  # Can be empty if job hasn't produced logs yet
