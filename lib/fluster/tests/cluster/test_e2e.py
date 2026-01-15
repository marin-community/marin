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

This file is self-contained - all test infrastructure is defined here.
"""

import base64
import re
import socket
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cloudpickle
import pytest

from fluster.cluster.client import RpcClusterClient
from fluster.cluster.controller.controller import Controller, ControllerConfig, DefaultWorkerStubFactory
from fluster.cluster.types import Entrypoint
from fluster.cluster.worker.builder import BuildResult
from fluster.cluster.worker.docker import ContainerConfig, ContainerStats, ContainerStatus
from fluster.cluster.worker.worker import Worker, WorkerConfig
from fluster.cluster.worker.worker_types import LogLine
from fluster.rpc import cluster_pb2
from fluster.rpc.cluster_connect import ControllerServiceClientSync

# =============================================================================
# Mock Infrastructure
# =============================================================================


@dataclass
class MockContainer:
    """Simulates a container executing a job function in-process."""

    config: ContainerConfig
    _thread: threading.Thread | None = field(default=None, repr=False)
    _running: bool = False
    _exit_code: int | None = None
    _error: str | None = None
    _logs: list[LogLine] = field(default_factory=list)
    _killed: threading.Event = field(default_factory=threading.Event)

    def start(self):
        """Execute the job function in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._execute, daemon=True)
        self._thread.start()

    def _execute(self):
        import os

        # Small delay to let controller update job state with worker_id
        # before we complete and report state back.
        time.sleep(0.2)

        original_env = {}
        try:
            # Set environment variables from container config
            for key, value in self.config.env.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value

            # Extract encoded data from command (command is ['python', '-c', script])
            script = self.config.command[2]
            fn, args, kwargs = self._extract_entrypoint(script)

            # Check if killed before executing
            if self._killed.is_set():
                self._exit_code = 137
                return

            # Execute the function
            fn(*args, **kwargs)
            self._exit_code = 0

        except Exception as e:
            self._error = str(e)
            self._exit_code = 1
        finally:
            self._running = False
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    def _extract_entrypoint(self, script: str):
        """Extract pickled (fn, args, kwargs) from the thunk script."""
        match = re.search(r"base64\.b64decode\('([^']+)'\)\)", script)
        if match:
            encoded = match.group(1)
            return cloudpickle.loads(base64.b64decode(encoded))
        raise ValueError("Could not extract entrypoint from command")

    def kill(self):
        self._killed.set()
        # Give thread a moment to notice
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self._running:
            self._running = False
            self._exit_code = 137


class InProcessContainerRuntime:
    """Container runtime that executes jobs in-process without Docker.

    Implements the ContainerRuntime protocol for testing.
    """

    def __init__(self):
        self._containers: dict[str, MockContainer] = {}

    def create_container(self, config: ContainerConfig) -> str:
        container_id = f"mock-{uuid.uuid4().hex[:8]}"
        self._containers[container_id] = MockContainer(config=config)
        return container_id

    def start_container(self, container_id: str) -> None:
        self._containers[container_id].start()

    def inspect(self, container_id: str) -> ContainerStatus:
        c = self._containers.get(container_id)
        if not c:
            return ContainerStatus(running=False, exit_code=1, error="container not found")
        return ContainerStatus(
            running=c._running,
            exit_code=c._exit_code,
            error=c._error,
        )

    def kill(self, container_id: str, force: bool = False) -> None:
        if container_id in self._containers:
            self._containers[container_id].kill()

    def remove(self, container_id: str) -> None:
        self._containers.pop(container_id, None)

    def get_logs(self, container_id: str) -> list[LogLine]:
        c = self._containers.get(container_id)
        return c._logs if c else []

    def get_stats(self, container_id: str) -> ContainerStats:
        return ContainerStats(memory_mb=100, cpu_percent=10, process_count=1, available=True)


class MockBundleProvider:
    """Returns a fake bundle path without downloading."""

    def __init__(self, bundle_path: Path):
        self._bundle_path = bundle_path

    def get_bundle(self, gcs_path: str, expected_hash: str | None = None) -> Path:
        return self._bundle_path


class MockImageProvider:
    """Skips image building, returns a fake result."""

    def build(
        self,
        bundle_path: Path,
        base_image: str,
        extras: list[str],
        job_id: str,
        deps_hash: str,
        job_logs=None,
    ) -> BuildResult:
        return BuildResult(
            image_tag="mock:latest",
            deps_hash=deps_hash,
            build_time_ms=0,
            from_cache=True,
        )


def find_free_port() -> int:
    """Find an available port."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# =============================================================================
# Test Cluster
# =============================================================================


class TestCluster:
    """Synchronous context manager running a controller + worker cluster.

    Uses in-process execution (no Docker) for fast testing.
    """

    def __init__(self, max_concurrent_jobs: int = 3, num_workers: int = 1):
        self._controller_port = find_free_port()
        self._max_concurrent_jobs = max_concurrent_jobs
        self._num_workers = num_workers

        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._controller: Controller | None = None
        self._workers: list[Worker] = []
        self._worker_ids: list[str] = []
        self._worker_ports: list[int] = []
        self._controller_client: ControllerServiceClientSync | None = None
        self._rpc_client: RpcClusterClient | None = None

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

        # Start Workers with mocked dependencies
        for i in range(self._num_workers):
            worker_id = f"worker-{i}-{uuid.uuid4().hex[:8]}"
            worker_port = find_free_port()
            worker_config = WorkerConfig(
                host="127.0.0.1",
                port=worker_port,
                cache_dir=cache_path,
                max_concurrent_jobs=self._max_concurrent_jobs,
                controller_address=f"http://127.0.0.1:{self._controller_port}",
                worker_id=worker_id,
            )
            worker = Worker(
                worker_config,
                cache_dir=cache_path,
                bundle_provider=MockBundleProvider(fake_bundle),
                image_provider=MockImageProvider(),
                container_runtime=InProcessContainerRuntime(),
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
        resources = cluster_pb2.ResourceSpec(cpu=cpu, memory=memory)
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

    def get_client(self) -> RpcClusterClient:
        if self._rpc_client is None:
            self._rpc_client = RpcClusterClient(
                f"http://127.0.0.1:{self._controller_port}",
                workspace=Path(__file__).parent.parent.parent,  # lib/fluster
            )
        return self._rpc_client


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def test_cluster():
    """Provide a running test cluster for E2E tests."""
    with TestCluster(max_concurrent_jobs=3) as cluster:
        yield cluster


@pytest.fixture
def multi_worker_cluster():
    """Provide a cluster with multiple workers."""
    with TestCluster(max_concurrent_jobs=3, num_workers=3) as cluster:
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
        """Ports are allocated and passed via environment."""
        received_ports = {}

        def port_job():
            import os

            received_ports["http"] = os.environ.get("FLUSTER_PORT_HTTP")
            received_ports["grpc"] = os.environ.get("FLUSTER_PORT_GRPC")

        job_id = test_cluster.submit(port_job, name="port-job", ports=["http", "grpc"])
        status = test_cluster.wait(job_id, timeout=30)
        assert status["state"] == "JOB_STATE_SUCCEEDED"

        # Ports should have been allocated (non-None)
        assert received_ports.get("http") is not None
        assert received_ports.get("grpc") is not None
