# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2ECluster: context manager for running Controller + Worker clusters in tests.

Supports both in-process (LocalController) and Docker (real containers) modes.
Docker mode manually wires up Controller + Workers with DockerRuntime, which is
needed for tests that exercise container-specific behavior (OOM, JAX env vars).
"""

import socket
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from iris.client import IrisClient
from iris.cluster.controller.controller import Controller, ControllerConfig, RpcWorkerStubFactory
from iris.cluster.controller.local import LocalController
from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec
from iris.cluster.worker.bundle_cache import BundleCache
from iris.cluster.worker.env_probe import EnvironmentProvider
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.rpc import cluster_pb2, config_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.time_utils import Duration

# Factory type for creating per-worker environment providers.
# Signature: (worker_id, num_workers) -> EnvironmentProvider
EnvProviderFactory = Callable[[int, int], EnvironmentProvider]


def find_free_port() -> int:
    """Find an available port."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def unique_name(prefix: str) -> str:
    """Generate a unique job name with the given prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _make_e2e_config(num_workers: int) -> config_pb2.IrisClusterConfig:
    """Build a fully-configured IrisClusterConfig for E2E tests with num_workers.

    Sets up controller.local, bundle_prefix, scale groups with local vm_type,
    and fast autoscaler evaluation for tests.
    """
    config = config_pb2.IrisClusterConfig()

    config.controller.local.port = 0
    config.controller.bundle_prefix = ""
    config.platform.local.SetInParent()

    sg = config_pb2.ScaleGroupConfig(
        name="local-cpu",
        min_slices=num_workers,
        max_slices=num_workers,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
        num_vms=1,
        resources=config_pb2.ScaleGroupResources(
            cpu=8,
            memory_bytes=16 * 1024**3,
            disk_bytes=50 * 1024**3,
            gpu_count=0,
            tpu_count=0,
        ),
    )
    config.scale_groups["local-cpu"].CopyFrom(sg)

    config.defaults.autoscaler.evaluation_interval.CopyFrom(Duration.from_seconds(0.5).to_proto())
    config.defaults.autoscaler.scale_up_delay.CopyFrom(Duration.from_seconds(1).to_proto())
    config.defaults.autoscaler.scale_down_delay.CopyFrom(Duration.from_minutes(5).to_proto())

    return config


class E2ECluster:
    """Synchronous context manager running a controller + worker cluster.

    Uses in-process execution (no Docker) by default for fast testing.
    Set use_docker=True to use real Docker containers.

    Args:
        num_workers: Number of workers to create.
        use_docker: If True, use Docker containers instead of subprocesses.
        cache_dir: Shared cache directory for Docker tests (uv, cargo, bundles).
            When None, a fresh temp directory is created per cluster.
        env_provider_factory: Optional factory for creating per-worker
            EnvironmentProviders. Signature: (worker_id, num_workers) -> provider.
            Use TPUSimEnvironmentProvider for TPU simulation tests.
    """

    def __init__(
        self,
        num_workers: int = 1,
        use_docker: bool = False,
        cache_dir: Path | None = None,
        env_provider_factory: EnvProviderFactory | None = None,
    ):
        self._num_workers = num_workers
        self._use_docker = use_docker
        self._cache_dir = cache_dir
        self._env_provider_factory = env_provider_factory
        self._controller: LocalController | Controller | None = None
        self._controller_port: int | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._container_runtime: DockerRuntime | None = None
        self._workers: list[Worker] = []
        self._worker_ids: list[str] = []
        self._worker_ports: list[int] = []
        self._controller_client: ControllerServiceClientSync | None = None
        self._rpc_client: IrisClient | None = None

    def __enter__(self):
        if not self._use_docker:
            config = _make_e2e_config(self._num_workers)
            self._controller = LocalController(config)
            address = self._controller.start()
            self._controller_port = int(address.rsplit(":", 1)[1])
            self._controller_client = ControllerServiceClientSync(
                address=address,
                timeout_ms=30000,
            )
            self._wait_for_workers(timeout=10.0)
            return self

        # Docker path: manual Controller + Worker setup
        self._controller_port = find_free_port()
        self._temp_dir = tempfile.TemporaryDirectory(prefix="test_cluster_")
        temp_path = Path(self._temp_dir.name)
        bundle_dir = temp_path / "bundles"
        bundle_dir.mkdir()
        cache_path = self._cache_dir if self._cache_dir else temp_path / "cache"
        cache_path.mkdir(exist_ok=True)

        fake_bundle = temp_path / "fake_bundle"
        fake_bundle.mkdir()
        (fake_bundle / "pyproject.toml").write_text("[project]\nname = 'test'\n")

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

        self._controller_client = ControllerServiceClientSync(
            address=f"http://127.0.0.1:{self._controller_port}",
            timeout_ms=30000,
        )

        bundle_provider = BundleCache(cache_path, max_bundles=10)
        self._container_runtime = DockerRuntime()
        container_runtime = self._container_runtime

        for i in range(self._num_workers):
            worker_id = f"worker-{i}-{uuid.uuid4().hex[:8]}"
            worker_port = find_free_port()
            worker_config = WorkerConfig(
                host="127.0.0.1",
                port=worker_port,
                cache_dir=cache_path,
                controller_address=f"http://127.0.0.1:{self._controller_port}",
                worker_id=worker_id,
                poll_interval=Duration.from_seconds(0.1),
            )
            env_provider = None
            if self._env_provider_factory:
                env_provider = self._env_provider_factory(i, self._num_workers)
            worker = Worker(
                worker_config,
                bundle_provider=bundle_provider,
                container_runtime=container_runtime,
                environment_provider=env_provider,
            )
            worker.start()
            self._workers.append(worker)
            self._worker_ids.append(worker_id)
            self._worker_ports.append(worker_port)

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
            self._rpc_client = None
        if self._controller_client:
            self._controller_client.close()
        if not self._use_docker:
            if self._controller:
                self._controller.stop()
        else:
            for worker in self._workers:
                worker.stop()
            if self._container_runtime:
                self._container_runtime.cleanup()
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
        scheduling_timeout: Duration | None = None,
        replicas: int = 1,
        **kwargs,
    ):
        """Submit a job and return a Job handle."""
        entrypoint = Entrypoint.from_callable(fn, *args, **kwargs)
        environment = EnvironmentSpec()
        resources = ResourceSpec(cpu=cpu, memory=memory)
        return self.get_client().submit(
            entrypoint=entrypoint,
            name=name or fn.__name__,
            resources=resources,
            environment=environment,
            ports=ports,
            scheduling_timeout=scheduling_timeout,
            replicas=replicas,
        )

    def _to_job_id_str(self, job_or_id) -> str:
        """Convert Job object or string to job_id string."""
        if isinstance(job_or_id, str):
            return (
                JobName.from_string(job_or_id).to_wire()
                if job_or_id.startswith("/")
                else JobName.root(job_or_id).to_wire()
            )
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
        task_id = JobName.from_wire(job_id).task(task_index).to_wire()
        request = cluster_pb2.Controller.GetTaskStatusRequest(task_id=task_id)
        assert self._controller_client is not None
        response = self._controller_client.get_task_status(request)
        return {
            "taskId": response.task.task_id,
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

    def get_task_logs(self, job_or_id, task_index: int = 0) -> list[str]:
        """Fetch container logs for a task."""
        job_id = self._to_job_id_str(job_or_id)
        task_id = JobName.from_wire(job_id).task(task_index).to_wire()
        request = cluster_pb2.Controller.GetTaskLogsRequest(id=task_id)
        assert self._controller_client is not None
        response = self._controller_client.get_task_logs(request)
        lines = []
        for batch in response.task_logs:
            for entry in batch.logs:
                lines.append(f"{entry.source}: {entry.data}")
        return lines

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

    @property
    def wait_timeout(self) -> float:
        """Default timeout for wait() calls. Longer for Docker due to uv sync overhead."""
        return 120.0 if self._use_docker else 30.0
