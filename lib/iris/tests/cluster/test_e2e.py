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

"""E2E integration tests for cluster controller and worker.

Tests the full job lifecycle through real Controller<->Worker RPC.
Jobs execute in-process (no Docker) for fast, reliable CI.

Uses the same local providers as LocalClusterClient for consistency.
"""

import socket
import tempfile
import time
import uuid
from pathlib import Path

import pytest

from iris.client import IrisClient
from iris.cluster.client.local_client import (
    LocalEnvironmentProvider,
    _LocalBundleProvider,
    _LocalContainerRuntime,
    _LocalImageProvider,
)
from iris.cluster.controller.controller import Controller, ControllerConfig, DefaultWorkerStubFactory
from iris.cluster.types import Entrypoint, create_resource_spec
from iris.cluster.worker.builder import ImageCache
from iris.cluster.worker.bundle_cache import BundleCache
from iris.cluster.worker.docker import DockerRuntime
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync


def find_free_port() -> int:
    """Find an available port."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# =============================================================================
# Test Cluster
# =============================================================================


class E2ECluster:
    """Synchronous context manager running a controller + worker cluster.

    Uses in-process execution (no Docker) by default for fast testing.
    Set use_docker=True to use real Docker containers.
    """

    def __init__(self, num_workers: int = 1, use_docker: bool = False):
        self._controller_port = find_free_port()
        self._num_workers = num_workers
        self._use_docker = use_docker

        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._controller: Controller | None = None
        self._workers: list[Worker] = []
        self._worker_ids: list[str] = []
        self._worker_ports: list[int] = []
        self._controller_client: ControllerServiceClientSync | None = None
        self._rpc_client: IrisClient | None = None

    def __enter__(self):
        self._temp_dir = tempfile.TemporaryDirectory(prefix="test_cluster_")
        temp_path = Path(self._temp_dir.name)
        bundle_dir = temp_path / "bundles"
        bundle_dir.mkdir()
        cache_path = temp_path / "cache"
        cache_path.mkdir()

        # Create fake bundle with minimal structure
        fake_bundle = temp_path / "fake_bundle"
        fake_bundle.mkdir()
        (fake_bundle / "pyproject.toml").write_text("[project]\nname = 'test'\n")

        # Start Controller first (workers need to connect to it)
        controller_config = ControllerConfig(
            host="127.0.0.1",
            port=self._controller_port,
            bundle_dir=bundle_dir,
        )
        self._controller = Controller(
            config=controller_config,
            worker_stub_factory=DefaultWorkerStubFactory(),
        )
        self._controller.start()

        # Create RPC client
        self._controller_client = ControllerServiceClientSync(
            address=f"http://127.0.0.1:{self._controller_port}",
            timeout_ms=30000,
        )

        # Select providers based on use_docker flag
        if self._use_docker:
            bundle_provider = BundleCache(cache_path, max_bundles=10)
            image_provider = ImageCache(cache_path, registry="", max_images=10)
            container_runtime = DockerRuntime()
            environment_provider = None  # Use default (probe real system)
        else:
            bundle_provider = _LocalBundleProvider(fake_bundle)
            image_provider = _LocalImageProvider()
            container_runtime = _LocalContainerRuntime()
            # 4 CPUs to match test expectations for resource scheduling tests
            environment_provider = LocalEnvironmentProvider(cpu=4, memory_gb=8)

        # Start Workers
        for i in range(self._num_workers):
            worker_id = f"worker-{i}-{uuid.uuid4().hex[:8]}"
            worker_port = find_free_port()
            worker_config = WorkerConfig(
                host="127.0.0.1",
                port=worker_port,
                cache_dir=cache_path,
                controller_address=f"http://127.0.0.1:{self._controller_port}",
                worker_id=worker_id,
                poll_interval_seconds=0.1,  # Fast polling for tests
            )
            worker = Worker(
                worker_config,
                cache_dir=cache_path,
                bundle_provider=bundle_provider,
                image_provider=image_provider,
                container_runtime=container_runtime,
                environment_provider=environment_provider,
            )
            worker.start()
            self._workers.append(worker)
            self._worker_ids.append(worker_id)
            self._worker_ports.append(worker_port)

        # Wait for workers to register with controller
        time.sleep(2.0)

        return self

    def __exit__(self, *args):
        if self._rpc_client:
            # RpcClusterClient doesn't have close method, just drop reference
            self._rpc_client = None
        if self._controller_client:
            self._controller_client.close()
        for worker in self._workers:
            worker.stop()
        if self._controller:
            self._controller.stop()
        if self._temp_dir:
            self._temp_dir.cleanup()

    def submit(
        self,
        fn,
        *args,
        name: str | None = None,
        cpu: int = 1,
        memory: str = "1g",
        ports: list[str] | None = None,
        scheduling_timeout_seconds: int = 0,
        **kwargs,
    ) -> str:
        entrypoint = Entrypoint.from_callable(fn, *args, **kwargs)
        environment = cluster_pb2.EnvironmentConfig(workspace="/app", env_vars={})
        resources = create_resource_spec(cpu=cpu, memory=memory)
        return self.get_client().submit(
            entrypoint=entrypoint,
            name=name or fn.__name__,
            resources=resources,
            environment=environment,
            ports=ports,
            scheduling_timeout_seconds=scheduling_timeout_seconds,
        )

    def status(self, job_id: str) -> dict:
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        response = self._controller_client.get_job_status(request)
        return {
            "jobId": response.job.job_id,
            "state": cluster_pb2.JobState.Name(response.job.state),
            "exitCode": response.job.exit_code,
            "error": response.job.error,
            "workerId": response.job.worker_id,
        }

    def wait(self, job_id: str, timeout: float = 60.0, poll_interval: float = 0.1) -> dict:
        start = time.time()
        terminal_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_KILLED",
            "JOB_STATE_UNSCHEDULABLE",
        }
        while time.time() - start < timeout:
            status = self.status(job_id)
            if status["state"] in terminal_states:
                return status
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")

    def kill(self, job_id: str) -> None:
        request = cluster_pb2.Controller.TerminateJobRequest(job_id=job_id)
        self._controller_client.terminate_job(request)

    def get_client(self) -> IrisClient:
        if self._rpc_client is None:
            self._rpc_client = IrisClient.remote(
                f"http://127.0.0.1:{self._controller_port}",
                workspace=Path(__file__).parent.parent.parent,  # lib/iris
            )
        return self._rpc_client


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_cluster(use_docker):
    """Provide a running test cluster for E2E tests (session-scoped)."""
    with E2ECluster(use_docker=use_docker) as cluster:
        yield cluster


@pytest.fixture(scope="session")
def multi_worker_cluster(use_docker):
    """Provide a cluster with multiple workers (session-scoped)."""
    with E2ECluster(num_workers=3, use_docker=use_docker) as cluster:
        yield cluster


# =============================================================================
# Tests: Job Lifecycle
# =============================================================================


class TestJobLifecycle:
    """Test basic job submission and completion."""

    def test_submit_and_complete(self, test_cluster):
        """Job completes successfully."""

        def hello():
            return 42

        job_id = test_cluster.submit(hello, name="test-job")
        status = test_cluster.wait(job_id, timeout=30)
        assert status["state"] == "JOB_STATE_SUCCEEDED"

    def test_job_with_args(self, test_cluster):
        """Job receives arguments correctly."""

        def add(a, b):
            return a + b

        job_id = test_cluster.submit(add, 10, 32, name="add-job")
        status = test_cluster.wait(job_id, timeout=30)
        assert status["state"] == "JOB_STATE_SUCCEEDED"

    def test_concurrent_jobs(self, test_cluster):
        """Multiple jobs run concurrently."""

        def fast_job(n):
            return n * 2

        job_ids = [test_cluster.submit(fast_job, i, name=f"job-{i}") for i in range(5)]
        for job_id in job_ids:
            status = test_cluster.wait(job_id, timeout=30)
            assert status["state"] == "JOB_STATE_SUCCEEDED"

    def test_kill_running_job(self, test_cluster):
        """Running job can be killed."""

        def long_job():
            import time

            time.sleep(60)

        job_id = test_cluster.submit(long_job, name="long-job")

        # Wait for job to start running
        for _ in range(50):
            status = test_cluster.status(job_id)
            if status["state"] == "JOB_STATE_RUNNING":
                break
            time.sleep(0.1)

        test_cluster.kill(job_id)
        status = test_cluster.wait(job_id, timeout=10)
        assert status["state"] == "JOB_STATE_KILLED"

    def test_job_failure_propagates(self, test_cluster):
        """Job that raises exception is marked FAILED."""

        def failing_job():
            raise ValueError("intentional failure")

        job_id = test_cluster.submit(failing_job, name="fail-job")
        status = test_cluster.wait(job_id, timeout=30)
        assert status["state"] == "JOB_STATE_FAILED"


# =============================================================================
# Tests: Resource Scheduling
# =============================================================================


class TestResourceScheduling:
    """Test scheduler resource management."""

    def test_small_job_skips_oversized_job(self, test_cluster):
        """Small job scheduled even when large job is waiting."""
        # Submit job requiring 100 CPUs (won't fit on 4-CPU worker)
        big_job_id = test_cluster.submit(lambda: None, name="big-job", cpu=100)

        # Submit small job
        small_job_id = test_cluster.submit(lambda: "done", name="small-job", cpu=1)

        # Small job should complete even though big job is first
        status = test_cluster.wait(small_job_id, timeout=10)
        assert status["state"] == "JOB_STATE_SUCCEEDED"

        # Big job should still be pending (can't fit)
        big_status = test_cluster.status(big_job_id)
        assert big_status["state"] == "JOB_STATE_PENDING"

    def test_scheduling_timeout(self, test_cluster):
        """Job that can't be scheduled times out."""
        # Submit job requiring 100 CPUs with 1 second timeout
        job_id = test_cluster.submit(
            lambda: None,
            name="impossible-job",
            cpu=100,
            scheduling_timeout_seconds=1,
        )

        # Should become UNSCHEDULABLE
        status = test_cluster.wait(job_id, timeout=10)
        assert status["state"] == "JOB_STATE_UNSCHEDULABLE"


# =============================================================================
# Tests: Multi-Worker
# =============================================================================


class TestMultiWorker:
    """Tests requiring multiple workers."""

    def test_multi_worker_execution(self, multi_worker_cluster):
        """Jobs distributed across multiple workers."""
        job_ids = [multi_worker_cluster.submit(lambda n=n: n * 2, name=f"job-{n}", cpu=2) for n in range(6)]

        # Wait for all to complete
        for job_id in job_ids:
            status = multi_worker_cluster.wait(job_id, timeout=30)
            assert status["state"] == "JOB_STATE_SUCCEEDED"

        # Verify jobs ran on different workers
        workers_used = set()
        for job_id in job_ids:
            status = multi_worker_cluster.status(job_id)
            if status["workerId"]:
                workers_used.add(status["workerId"])

        assert len(workers_used) > 1, "Jobs should run on multiple workers"


# =============================================================================
# Tests: Ports
# =============================================================================


class TestPorts:
    """Test port allocation and forwarding."""

    def test_port_allocation(self, test_cluster):
        """Ports are allocated and passed via JobInfo."""

        def port_job():
            from iris.cluster.client import get_job_info

            info = get_job_info()
            if info is None:
                raise ValueError("JobInfo not available")
            # Verify ports are set
            if "http" not in info.ports or "grpc" not in info.ports:
                raise ValueError(f"Ports not set: {info.ports}")
            # Verify they're valid port numbers
            assert info.ports["http"] > 0
            assert info.ports["grpc"] > 0

        job_id = test_cluster.submit(port_job, name="port-job", ports=["http", "grpc"])
        status = test_cluster.wait(job_id, timeout=30)
        # Job succeeds only if ports were correctly passed
        assert status["state"] == "JOB_STATE_SUCCEEDED", f"Job failed: {status}"


# =============================================================================
# Tests: JobInfo Context
# =============================================================================


class TestJobInfo:
    """Test JobInfo contextvar is available in jobs."""

    def test_job_info_contextvar(self, test_cluster):
        """JobInfo is available via contextvar with correct job_id and worker_id."""

        def test_fn():
            from iris.cluster.client import get_job_info

            info = get_job_info()
            assert info is not None
            assert info.job_id == "test-job-info"
            assert info.worker_id is not None
            assert "actor" in info.ports
            return info.job_id

        job_id = test_cluster.submit(test_fn, name="test-job-info", ports=["actor"])
        status = test_cluster.wait(job_id, timeout=30)
        assert status["state"] == "JOB_STATE_SUCCEEDED"

    def test_job_info_with_multiple_ports(self, test_cluster):
        """JobInfo contains all allocated ports and they are unique."""

        def test_fn():
            from iris.cluster.client import get_job_info

            info = get_job_info()
            assert info is not None
            assert "actor" in info.ports
            assert "metrics" in info.ports
            assert "custom" in info.ports
            # Ports should be unique
            port_values = list(info.ports.values())
            assert len(port_values) == len(set(port_values))
            return info.ports

        job_id = test_cluster.submit(test_fn, name="test-job-ports", ports=["actor", "metrics", "custom"])
        status = test_cluster.wait(job_id, timeout=30)
        assert status["state"] == "JOB_STATE_SUCCEEDED"


# =============================================================================
# Tests: Endpoints
# =============================================================================


class TestEndpoints:
    """Test endpoint registration from within jobs."""

    def test_endpoint_registration_from_job(self, test_cluster):
        """Job can register endpoints that are visible externally."""

        def register_endpoint_job():
            import time

            from iris.cluster.client import get_job_info
            from iris.rpc import cluster_pb2
            from iris.rpc.cluster_connect import ControllerServiceClientSync

            info = get_job_info()
            if info is None:
                raise ValueError("JobInfo not available")
            if not info.controller_address:
                raise ValueError("controller_address not set in JobInfo")

            client = ControllerServiceClientSync(address=info.controller_address, timeout_ms=5000)
            try:
                # Register an endpoint
                request = cluster_pb2.Controller.RegisterEndpointRequest(
                    name="test/actor1",
                    address="localhost:5000",
                    job_id=info.job_id,
                    metadata={"type": "actor"},
                )
                response = client.register_endpoint(request)
                assert response.endpoint_id

                # List endpoints to verify
                list_request = cluster_pb2.Controller.ListEndpointsRequest(prefix="test/")
                list_response = client.list_endpoints(list_request)
                assert len(list_response.endpoints) == 1
                assert list_response.endpoints[0].name == "test/actor1"
                assert list_response.endpoints[0].metadata["type"] == "actor"

                # Keep job alive briefly so endpoint stays registered
                time.sleep(0.5)
            finally:
                client.close()

        job_id = test_cluster.submit(register_endpoint_job, name="endpoint-job")
        status = test_cluster.wait(job_id, timeout=30)
        assert status["state"] == "JOB_STATE_SUCCEEDED", f"Job failed: {status}"

    def test_endpoint_prefix_matching(self, test_cluster):
        """Endpoint prefix matching works correctly."""

        def register_multiple_endpoints():
            import time

            from iris.cluster.client import get_job_info
            from iris.rpc import cluster_pb2
            from iris.rpc.cluster_connect import ControllerServiceClientSync

            info = get_job_info()
            if info is None:
                raise ValueError("JobInfo not available")
            if not info.controller_address:
                raise ValueError("controller_address not set in JobInfo")
            client = ControllerServiceClientSync(address=info.controller_address, timeout_ms=5000)
            try:
                # Register endpoints with various prefixes
                for name, addr in [
                    ("ns1/actor1", "host1:5000"),
                    ("ns1/actor2", "host2:5001"),
                    ("ns1/service/actor3", "host3:5002"),
                    ("ns2/actor1", "host4:5003"),
                ]:
                    request = cluster_pb2.Controller.RegisterEndpointRequest(
                        name=name,
                        address=addr,
                        job_id=info.job_id,
                    )
                    client.register_endpoint(request)

                # Test prefix matching
                ns1_all = client.list_endpoints(cluster_pb2.Controller.ListEndpointsRequest(prefix="ns1/"))
                assert len(ns1_all.endpoints) == 3

                ns1_service = client.list_endpoints(cluster_pb2.Controller.ListEndpointsRequest(prefix="ns1/service/"))
                assert len(ns1_service.endpoints) == 1
                assert ns1_service.endpoints[0].name == "ns1/service/actor3"

                ns2 = client.list_endpoints(cluster_pb2.Controller.ListEndpointsRequest(prefix="ns2/"))
                assert len(ns2.endpoints) == 1

                time.sleep(0.5)
            finally:
                client.close()

        job_id = test_cluster.submit(register_multiple_endpoints, name="prefix-job")
        status = test_cluster.wait(job_id, timeout=30)
        assert status["state"] == "JOB_STATE_SUCCEEDED", f"Job failed: {status}"
