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
from iris.cluster.client import get_job_info
from iris.cluster.client.local_client import (
    LocalEnvironmentProvider,
    _LocalBundleProvider,
    _LocalContainerRuntime,
    _LocalImageProvider,
)
from iris.cluster.controller.controller import Controller, ControllerConfig, RpcWorkerStubFactory
from iris.cluster.types import EnvironmentSpec, Entrypoint, ResourceSpec
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


def unique_name(prefix: str) -> str:
    """Generate a unique job name with the given prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


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
            bundle_prefix=f"file://{bundle_dir}",
        )
        self._controller = Controller(
            config=controller_config,
            worker_stub_factory=RpcWorkerStubFactory(),
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
        self._wait_for_workers(timeout=10.0)

        return self

    def _wait_for_workers(self, timeout: float = 10.0) -> None:
        """Wait for all workers to register with the controller."""
        start = time.time()
        while time.time() - start < timeout:
            request = cluster_pb2.Controller.ListWorkersRequest()
            assert self._controller_client is not None
            response = self._controller_client.list_workers(request)
            healthy_workers = [w for w in response.workers if w.healthy]
            if len(healthy_workers) >= self._num_workers:
                return
            time.sleep(0.1)
        raise TimeoutError(f"Workers failed to register within {timeout}s")

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
    ):
        """Submit a job and return a Job handle."""
        entrypoint = Entrypoint.from_callable(fn, *args, **kwargs)
        environment = EnvironmentSpec(workspace="/app")
        resources = ResourceSpec(cpu=cpu, memory=memory)
        return self.get_client().submit(
            entrypoint=entrypoint,
            name=name or fn.__name__,
            resources=resources,
            environment=environment,
            ports=ports,
            scheduling_timeout_seconds=scheduling_timeout_seconds,
        )

    def _to_job_id_str(self, job_or_id) -> str:
        """Convert Job object or string to job_id string."""
        if isinstance(job_or_id, str):
            return job_or_id
        # Assume it's a Job object
        return str(job_or_id.job_id)

    def status(self, job_or_id) -> dict:
        job_id = self._to_job_id_str(job_or_id)
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        assert self._controller_client is not None
        response = self._controller_client.get_job_status(request)
        return {
            "jobId": response.job.job_id,
            "state": cluster_pb2.JobState.Name(response.job.state),
            "exitCode": response.job.exit_code,
            "error": response.job.error,
        }

    def task_status(self, job_or_id, task_index: int = 0) -> dict:
        """Get status of a specific task within a job."""
        job_id = self._to_job_id_str(job_or_id)
        request = cluster_pb2.Controller.GetTaskStatusRequest(job_id=job_id, task_index=task_index)
        assert self._controller_client is not None
        response = self._controller_client.get_task_status(request)
        return {
            "taskId": response.task.task_id,
            "jobId": response.task.job_id,
            "taskIndex": response.task.task_index,
            "state": cluster_pb2.TaskState.Name(response.task.state),
            "workerId": response.task.worker_id,
            "workerAddress": response.task.worker_address,
            "exitCode": response.task.exit_code,
            "error": response.task.error,
        }

    def wait(self, job_or_id, timeout: float = 60.0, poll_interval: float = 0.1) -> dict:
        job_id = self._to_job_id_str(job_or_id)
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

    def kill(self, job_or_id) -> None:
        job_id = self._to_job_id_str(job_or_id)
        request = cluster_pb2.Controller.TerminateJobRequest(job_id=job_id)
        assert self._controller_client is not None
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
def test_cluster(use_docker, docker_cleanup_session):
    """Provide a running test cluster for E2E tests (session-scoped)."""
    with E2ECluster(use_docker=use_docker) as cluster:
        yield cluster


@pytest.fixture(scope="session")
def multi_worker_cluster(use_docker, docker_cleanup_session):
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

        job_id = test_cluster.submit(hello, name=unique_name("test-job"))
        status = test_cluster.wait(job_id, timeout=30)
        assert status["state"] == "JOB_STATE_SUCCEEDED"

    def test_job_with_args(self, test_cluster):
        """Job receives arguments correctly."""

        def add(a, b):
            return a + b

        job_id = test_cluster.submit(add, 10, 32, name=unique_name("add-job"))
        status = test_cluster.wait(job_id, timeout=30)
        assert status["state"] == "JOB_STATE_SUCCEEDED"

    def test_multiple_jobs_complete(self, test_cluster):
        """Multiple jobs complete successfully."""
        run_id = uuid.uuid4().hex[:8]

        def fast_job(n):
            return n * 2

        job_ids = [test_cluster.submit(fast_job, i, name=f"job-{run_id}-{i}") for i in range(5)]
        for job_id in job_ids:
            status = test_cluster.wait(job_id, timeout=30)
            assert status["state"] == "JOB_STATE_SUCCEEDED"

    def test_kill_running_job(self, test_cluster):
        """Running job can be killed."""

        def long_job():
            time.sleep(60)

        job_id = test_cluster.submit(long_job, name=unique_name("long-job"))

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

        job_id = test_cluster.submit(failing_job, name=unique_name("fail-job"))
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
        big_job_id = test_cluster.submit(lambda: None, name=unique_name("big-job"), cpu=100)

        # Submit small job
        small_job_id = test_cluster.submit(lambda: "done", name=unique_name("small-job"), cpu=1)

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
            name=unique_name("impossible-job"),
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
        run_id = uuid.uuid4().hex[:8]
        job_ids = [multi_worker_cluster.submit(lambda n=n: n * 2, name=f"mw-job-{run_id}-{n}", cpu=2) for n in range(6)]

        # Wait for all to complete
        for job_id in job_ids:
            status = multi_worker_cluster.wait(job_id, timeout=30)
            assert status["state"] == "JOB_STATE_SUCCEEDED"

        # Verify jobs ran on different workers (via task-level worker info)
        workers_used = set()
        for job_id in job_ids:
            task_status = multi_worker_cluster.task_status(job_id, task_index=0)
            if task_status["workerId"]:
                workers_used.add(task_status["workerId"])

        assert len(workers_used) > 1, "Jobs should run on multiple workers"


# =============================================================================
# Tests: Ports
# =============================================================================


class TestPorts:
    """Test port allocation and forwarding."""

    def test_port_allocation(self, test_cluster):
        """Ports are allocated and passed via JobInfo."""

        def port_job():
            info = get_job_info()
            if info is None:
                raise ValueError("JobInfo not available")
            # Verify ports are set
            if "http" not in info.ports or "grpc" not in info.ports:
                raise ValueError(f"Ports not set: {info.ports}")
            # Verify they're valid port numbers
            assert info.ports["http"] > 0
            assert info.ports["grpc"] > 0

        job_id = test_cluster.submit(port_job, name=unique_name("port-job"), ports=["http", "grpc"])
        status = test_cluster.wait(job_id, timeout=30)
        # Job succeeds only if ports were correctly passed
        assert status["state"] == "JOB_STATE_SUCCEEDED", f"Job failed: {status}"


# =============================================================================
# Tests: JobInfo Context
# =============================================================================


class TestJobInfo:
    """Test JobInfo contextvar is available in jobs and provides runtime context."""

    def test_job_info_provides_context(self, test_cluster):
        """JobInfo is available and provides job_id, worker_id, and ports during execution."""
        job_name = unique_name("test-job-info")

        def test_fn(expected_job_id):
            info = get_job_info()
            # Verify JobInfo is available and provides expected context
            if info is None:
                raise ValueError("JobInfo not available")
            if info.job_id != expected_job_id:
                raise ValueError(f"JobInfo has wrong job_id: {info.job_id}")
            if info.worker_id is None:
                raise ValueError("JobInfo missing worker_id")
            if "actor" not in info.ports:
                raise ValueError("JobInfo missing expected port 'actor'")
            return "success"

        job_id = test_cluster.submit(test_fn, job_name, name=job_name, ports=["actor"])
        status = test_cluster.wait(job_id, timeout=30)
        assert status["state"] == "JOB_STATE_SUCCEEDED"

    def test_job_info_port_allocation(self, test_cluster):
        """JobInfo provides all requested ports and they are unique."""

        def test_fn():
            info = get_job_info()
            if info is None:
                raise ValueError("JobInfo not available")

            # Verify all ports are present
            required_ports = {"actor", "metrics", "custom"}
            if not required_ports.issubset(info.ports.keys()):
                raise ValueError(f"Missing ports. Expected {required_ports}, got {info.ports.keys()}")

            # Verify ports are unique
            port_values = list(info.ports.values())
            if len(port_values) != len(set(port_values)):
                raise ValueError(f"Ports are not unique: {port_values}")

            return "success"

        job_id = test_cluster.submit(test_fn, name=unique_name("test-job-ports"), ports=["actor", "metrics", "custom"])
        status = test_cluster.wait(job_id, timeout=30)
        assert status["state"] == "JOB_STATE_SUCCEEDED"

    def test_job_info_task_context(self, test_cluster):
        """JobInfo provides task-specific context (task_id, task_index, num_tasks)."""
        job_name = unique_name("test-task-fields")

        def test_fn(expected_job_name):
            info = get_job_info()
            if info is None:
                raise ValueError("JobInfo not available")

            # Verify task context is correct
            expected_task_id = f"{expected_job_name}/task-0"
            if info.task_id != expected_task_id:
                raise ValueError(f"Expected task_id {expected_task_id}, got {info.task_id}")
            if info.task_index != 0:
                raise ValueError(f"Expected task_index 0, got {info.task_index}")
            if info.num_tasks != 1:
                raise ValueError(f"Expected num_tasks 1, got {info.num_tasks}")

            return "success"

        job_id = test_cluster.submit(test_fn, job_name, name=job_name)
        status = test_cluster.wait(job_id, timeout=30)
        assert status["state"] == "JOB_STATE_SUCCEEDED"


# =============================================================================
# Tests: Endpoints
# =============================================================================


class TestEndpoints:
    """Test endpoint registration from within jobs."""

    def test_endpoint_registration_from_job(self, test_cluster):
        """Job can register endpoints that are visible externally."""
        endpoint_prefix = f"test-{uuid.uuid4().hex[:8]}"

        def register_endpoint_job(prefix):
            info = get_job_info()
            if info is None:
                raise ValueError("JobInfo not available")
            if not info.controller_address:
                raise ValueError("controller_address not set in JobInfo")

            client = ControllerServiceClientSync(address=info.controller_address, timeout_ms=5000)
            try:
                endpoint_name = f"{prefix}/actor1"
                # Register an endpoint
                request = cluster_pb2.Controller.RegisterEndpointRequest(
                    name=endpoint_name,
                    address="localhost:5000",
                    job_id=info.job_id,
                    metadata={"type": "actor"},
                )
                response = client.register_endpoint(request)
                assert response.endpoint_id

                # List endpoints to verify
                list_request = cluster_pb2.Controller.ListEndpointsRequest(prefix=f"{prefix}/")
                list_response = client.list_endpoints(list_request)
                assert len(list_response.endpoints) == 1
                assert list_response.endpoints[0].name == endpoint_name
                assert list_response.endpoints[0].metadata["type"] == "actor"

                # Keep job alive briefly so endpoint stays registered
                time.sleep(0.5)
            finally:
                client.close()

        job_id = test_cluster.submit(register_endpoint_job, endpoint_prefix, name=unique_name("endpoint-job"))
        status = test_cluster.wait(job_id, timeout=30)
        assert status["state"] == "JOB_STATE_SUCCEEDED", f"Job failed: {status}"

    def test_endpoint_prefix_matching(self, test_cluster):
        """Endpoint prefix matching works correctly."""
        run_id = uuid.uuid4().hex[:8]
        ns1 = f"ns1-{run_id}"
        ns2 = f"ns2-{run_id}"

        def register_multiple_endpoints(ns1_prefix, ns2_prefix):
            info = get_job_info()
            if info is None:
                raise ValueError("JobInfo not available")
            if not info.controller_address:
                raise ValueError("controller_address not set in JobInfo")
            client = ControllerServiceClientSync(address=info.controller_address, timeout_ms=5000)
            try:
                # Register endpoints with various prefixes
                for name, addr in [
                    (f"{ns1_prefix}/actor1", "host1:5000"),
                    (f"{ns1_prefix}/actor2", "host2:5001"),
                    (f"{ns1_prefix}/service/actor3", "host3:5002"),
                    (f"{ns2_prefix}/actor1", "host4:5003"),
                ]:
                    request = cluster_pb2.Controller.RegisterEndpointRequest(
                        name=name,
                        address=addr,
                        job_id=info.job_id,
                    )
                    client.register_endpoint(request)

                # Test prefix matching
                ns1_all = client.list_endpoints(cluster_pb2.Controller.ListEndpointsRequest(prefix=f"{ns1_prefix}/"))
                assert len(ns1_all.endpoints) == 3

                ns1_service = client.list_endpoints(
                    cluster_pb2.Controller.ListEndpointsRequest(prefix=f"{ns1_prefix}/service/")
                )
                assert len(ns1_service.endpoints) == 1
                assert ns1_service.endpoints[0].name == f"{ns1_prefix}/service/actor3"

                ns2_all = client.list_endpoints(cluster_pb2.Controller.ListEndpointsRequest(prefix=f"{ns2_prefix}/"))
                assert len(ns2_all.endpoints) == 1

                time.sleep(0.5)
            finally:
                client.close()

        job_id = test_cluster.submit(register_multiple_endpoints, ns1, ns2, name=unique_name("prefix-job"))
        status = test_cluster.wait(job_id, timeout=30)
        assert status["state"] == "JOB_STATE_SUCCEEDED", f"Job failed: {status}"
