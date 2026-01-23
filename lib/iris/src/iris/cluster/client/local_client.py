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

"""Local cluster client using real Controller/Worker with in-process execution.

Spins up a real Controller and Worker but executes jobs in-process using threads
instead of Docker containers, ensuring local execution follows the same code path
as production cluster execution.
"""

import io
import socket
import tempfile
import threading
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Self

from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.controller.controller import Controller, ControllerConfig, RpcWorkerStubFactory
from iris.cluster.types import Entrypoint
from iris.cluster.worker.builder import BuildResult
from iris.cluster.worker.docker import ContainerConfig, ContainerRuntime, ContainerStats, ContainerStatus
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.cluster.worker.worker_types import LogLine
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# =============================================================================
# Local Providers (private implementation details)
# =============================================================================


class LocalEnvironmentProvider:
    def __init__(
        self,
        cpu: int = 1000,
        memory_gb: int = 1000,
        attributes: dict[str, str | int | float] | None = None,
    ):
        self._cpu = cpu
        self._memory_gb = memory_gb
        self._attributes = attributes or {}

    def probe(self) -> cluster_pb2.WorkerMetadata:
        device = cluster_pb2.DeviceConfig()
        device.cpu.CopyFrom(cluster_pb2.CpuDevice(variant="cpu"))

        proto_attrs = {}
        for key, value in self._attributes.items():
            if isinstance(value, str):
                proto_attrs[key] = cluster_pb2.AttributeValue(string_value=value)
            elif isinstance(value, int):
                proto_attrs[key] = cluster_pb2.AttributeValue(int_value=value)
            elif isinstance(value, float):
                proto_attrs[key] = cluster_pb2.AttributeValue(float_value=value)

        return cluster_pb2.WorkerMetadata(
            hostname="local",
            ip_address="127.0.0.1",
            cpu_count=self._cpu,
            memory_bytes=self._memory_gb * 1024**3,
            disk_bytes=100 * 1024**3,  # Default 100GB for local
            device=device,
            attributes=proto_attrs,
        )


@dataclass
class _LocalContainer:
    config: ContainerConfig
    _thread: threading.Thread | None = field(default=None, repr=False)
    _running: bool = False
    _exit_code: int | None = None
    _error: str | None = None
    _logs: list[LogLine] = field(default_factory=list)
    _killed: threading.Event = field(default_factory=threading.Event)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._execute, daemon=True)
        self._thread.start()

    def _execute(self):
        from iris.cluster.client.job_info import JobInfo, _parse_ports_from_env, set_job_info

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Build JobInfo from container config env vars
            env = self.config.env
            job_info = JobInfo(
                job_id=env.get("IRIS_JOB_ID", ""),
                task_id=env.get("IRIS_TASK_ID"),
                task_index=int(env.get("IRIS_TASK_INDEX", "0")),
                num_tasks=int(env.get("IRIS_NUM_TASKS", "1")),
                attempt_id=int(env.get("IRIS_ATTEMPT_ID", "0")),
                worker_id=env.get("IRIS_WORKER_ID"),
                controller_address=env.get("IRIS_CONTROLLER_ADDRESS"),
                ports=_parse_ports_from_env(env),
            )
            set_job_info(job_info)

            # Use entrypoint directly - no command parsing needed
            fn, args, kwargs = self.config.entrypoint

            # Check if killed before executing
            if self._killed.is_set():
                self._exit_code = 137
                return

            # Execute the function with captured output
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                fn(*args, **kwargs)
            self._exit_code = 0

        except Exception as e:
            self._error = str(e)
            self._exit_code = 1
        finally:
            self._running = False
            self._capture_output(stdout_capture, "stdout")
            self._capture_output(stderr_capture, "stderr")

    def _capture_output(self, capture: io.StringIO, source: str) -> None:
        capture.seek(0)
        for line in capture:
            line = line.rstrip("\n")
            if line:
                self._logs.append(
                    LogLine(
                        timestamp=datetime.now(timezone.utc),
                        source=source,
                        data=line,
                    )
                )

    def kill(self):
        self._killed.set()
        # Give thread a moment to notice
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self._running:
            self._running = False
            self._exit_code = 137


class _LocalContainerRuntime(ContainerRuntime):
    def __init__(self):
        self._containers: dict[str, _LocalContainer] = {}

    def create_container(self, config: ContainerConfig) -> str:
        container_id = f"local-{uuid.uuid4().hex[:8]}"
        self._containers[container_id] = _LocalContainer(config=config)
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
        del force  # Local containers don't distinguish force vs graceful
        if container_id in self._containers:
            self._containers[container_id].kill()

    def remove(self, container_id: str) -> None:
        self._containers.pop(container_id, None)

    def get_logs(self, container_id: str) -> list[LogLine]:
        c = self._containers.get(container_id)
        return c._logs if c else []

    def get_stats(self, container_id: str) -> ContainerStats:
        del container_id
        return ContainerStats(memory_mb=100, cpu_percent=10, process_count=1, available=True)


class _LocalBundleProvider:
    def __init__(self, bundle_path: Path):
        self._bundle_path = bundle_path

    def get_bundle(self, gcs_path: str, expected_hash: str | None = None) -> Path:
        del gcs_path, expected_hash
        return self._bundle_path


class _LocalImageProvider:
    def build(
        self,
        bundle_path: Path,
        base_image: str,
        extras: list[str],
        job_id: str,
        deps_hash: str,
        task_logs=None,
    ) -> BuildResult:
        del bundle_path, base_image, extras, job_id, task_logs
        return BuildResult(
            image_tag="local:latest",
            deps_hash=deps_hash,
            build_time_ms=0,
            from_cache=True,
        )

    def protect(self, tag: str) -> None:
        """No-op for local provider (no eviction)."""
        del tag

    def unprotect(self, tag: str) -> None:
        """No-op for local provider (no eviction)."""
        del tag


class LocalClusterClient:
    """Local cluster client using real Controller/Worker with in-process execution.

    Provides the same execution path as production clusters while running
    entirely in-process without Docker or network dependencies.

    Use the create() classmethod to instantiate:
        client = LocalClusterClient.create()
        # ... use client ...
        client.shutdown()
    """

    def __init__(
        self,
        temp_dir: tempfile.TemporaryDirectory,
        controller: Controller,
        worker: Worker,
        remote_client: RemoteClusterClient,
    ):
        self._temp_dir = temp_dir
        self._controller = controller
        self._worker = worker
        self._remote_client = remote_client

    @classmethod
    def create(
        cls,
        max_workers: int = 4,
        port_range: tuple[int, int] = (50000, 60000),
    ) -> Self:
        """Create and start a local cluster client.

        Args:
            max_workers: Maximum concurrent job threads
            port_range: Port range for actor servers (inclusive start, exclusive end)

        Returns:
            A fully initialized LocalClusterClient ready for use
        """
        temp_dir = tempfile.TemporaryDirectory(prefix="iris_local_")
        temp_path = Path(temp_dir.name)
        bundle_dir = temp_path / "bundles"
        bundle_dir.mkdir()
        cache_path = temp_path / "cache"
        cache_path.mkdir()

        # Create fake bundle with minimal structure
        fake_bundle = temp_path / "fake_bundle"
        fake_bundle.mkdir()
        (fake_bundle / "pyproject.toml").write_text("[project]\nname = 'local'\n")

        # Start Controller
        controller_port = _find_free_port()
        controller_config = ControllerConfig(
            host="127.0.0.1",
            port=controller_port,
            bundle_dir=bundle_dir,
        )
        controller = Controller(
            config=controller_config,
            worker_stub_factory=RpcWorkerStubFactory(),
        )
        controller.start()

        controller_address = f"http://127.0.0.1:{controller_port}"

        # Start Worker with local providers
        worker_port = _find_free_port()
        worker_config = WorkerConfig(
            host="127.0.0.1",
            port=worker_port,
            cache_dir=cache_path,
            controller_address=controller_address,
            worker_id=f"local-worker-{uuid.uuid4().hex[:8]}",
            poll_interval_seconds=0.1,  # Fast polling for local
            port_range=port_range,
        )
        worker = Worker(
            worker_config,
            cache_dir=cache_path,
            bundle_provider=_LocalBundleProvider(fake_bundle),
            image_provider=_LocalImageProvider(),
            container_runtime=_LocalContainerRuntime(),
            environment_provider=LocalEnvironmentProvider(cpu=1000, memory_gb=1000),
        )
        worker.start()

        # Wait for worker registration using a temporary RPC client
        cls._wait_for_worker_registration(controller_address)

        # Create RemoteClusterClient for all subsequent operations
        remote_client = RemoteClusterClient(
            controller_address=controller_address,
            timeout_ms=30000,
        )

        return cls(temp_dir, controller, worker, remote_client)

    @staticmethod
    def _wait_for_worker_registration(controller_address: str, timeout: float = 5.0) -> None:
        temp_client = ControllerServiceClientSync(
            address=controller_address,
            timeout_ms=30000,
        )
        try:
            start = time.monotonic()
            while time.monotonic() - start < timeout:
                response = temp_client.list_workers(cluster_pb2.Controller.ListWorkersRequest())
                if response.workers:
                    return
                time.sleep(0.1)
            raise TimeoutError("Worker failed to register with controller")
        finally:
            temp_client.close()

    def shutdown(self, wait: bool = True) -> None:
        del wait
        self._remote_client.shutdown()
        self._worker.stop()
        self._controller.stop()
        self._temp_dir.cleanup()

    def submit_job(
        self,
        job_id: str,
        entrypoint: Entrypoint,
        resources: cluster_pb2.ResourceSpecProto,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        ports: list[str] | None = None,
        scheduling_timeout_seconds: int = 0,
        constraints: list[cluster_pb2.Constraint] | None = None,
        coscheduling: cluster_pb2.CoschedulingConfig | None = None,
    ) -> None:
        self._remote_client.submit_job(
            job_id=job_id,
            entrypoint=entrypoint,
            resources=resources,
            environment=environment,
            ports=ports,
            scheduling_timeout_seconds=scheduling_timeout_seconds,
            constraints=constraints,
            coscheduling=coscheduling,
        )

    def get_job_status(self, job_id: str) -> cluster_pb2.JobStatus:
        return self._remote_client.get_job_status(job_id)

    def wait_for_job(
        self,
        job_id: str,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> cluster_pb2.JobStatus:
        return self._remote_client.wait_for_job(job_id, timeout=timeout, poll_interval=poll_interval)

    def terminate_job(self, job_id: str) -> None:
        self._remote_client.terminate_job(job_id)

    def register_endpoint(
        self,
        name: str,
        address: str,
        job_id: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        return self._remote_client.register_endpoint(name=name, address=address, job_id=job_id, metadata=metadata)

    def unregister_endpoint(self, endpoint_id: str) -> None:
        self._remote_client.unregister_endpoint(endpoint_id)

    def list_endpoints(self, prefix: str) -> list[cluster_pb2.Controller.Endpoint]:
        return self._remote_client.list_endpoints(prefix)

    def list_jobs(self) -> list[cluster_pb2.JobStatus]:
        return self._remote_client.list_jobs()

    def get_task_status(self, job_id: str, task_index: int) -> cluster_pb2.TaskStatus:
        return self._remote_client.get_task_status(job_id, task_index)

    def list_tasks(self, job_id: str) -> list[cluster_pb2.TaskStatus]:
        return self._remote_client.list_tasks(job_id)

    def fetch_task_logs(
        self,
        task_id: str,
        start_ms: int = 0,
        max_lines: int = 0,
    ) -> list[cluster_pb2.Worker.LogEntry]:
        return self._remote_client.fetch_task_logs(task_id, start_ms, max_lines)
