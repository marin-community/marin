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

"""Local platform: in-process VmManager for testing without GCP.

Provides LocalVmManager (VmManagerProtocol) and LocalVmGroup (VmGroupProtocol)
that create real Worker instances running in-process with subprocess-based execution
instead of Docker containers.

Also provides the local provider implementations used by workers:
- LocalEnvironmentProvider: probes local system resources
- _LocalBundleProvider: serves pre-built bundles from local filesystem
- _LocalImageProvider: no-op image provider (uses local:latest)
- _LocalContainerRuntime: executes containers as subprocesses
- _LocalContainer: subprocess-based container execution (enables hard kill)
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.util
import logging
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iris.cluster.types import get_tpu_topology, tpu_device
from iris.cluster.vm.managed_vm import ManagedVm, VmRegistry
from iris.cluster.vm.vm_platform import VmGroupProtocol, VmGroupStatus, VmSnapshot
from iris.cluster.worker.builder import BuildResult
from iris.cluster.worker.docker import ContainerConfig, ContainerRuntime, ContainerStats, ContainerStatus
from iris.cluster.worker.worker import PortAllocator, Worker, WorkerConfig
from iris.cluster.worker.worker_types import LogLine, TaskLogs
from iris.managed_thread import ManagedThread, ThreadContainer, get_thread_container
from iris.rpc import cluster_pb2, config_pb2, vm_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)


# =============================================================================
# Process management utilities
# =============================================================================


def _set_pdeathsig_preexec():
    """Use prctl(PR_SET_PDEATHSIG, SIGKILL) to kill subprocess if parent dies.

    This is a Linux-specific feature that ensures container processes are
    automatically killed if the worker process dies unexpectedly. On other
    platforms, this is a no-op.
    """
    if sys.platform == "linux":
        PR_SET_PDEATHSIG = 1
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
                errno = ctypes.get_errno()
                logger.warning(f"Failed to set parent death signal: errno {errno}")
        except Exception as e:
            logger.debug(f"Could not set parent death signal: {e}")


# =============================================================================
# Local Providers (in-process implementations for testing)
# =============================================================================


@dataclass
class _LocalContainer:
    """Container execution via subprocess (not thread).

    Uses subprocess.Popen to run both callable and command entrypoints,
    enabling hard termination and proper log capture. Mirrors the Docker
    runtime's thunk pattern for callable entrypoints.
    """

    config: ContainerConfig
    _process: subprocess.Popen | None = field(default=None, repr=False)
    _log_thread: ManagedThread | None = field(default=None, repr=False)
    _running: bool = False
    _exit_code: int | None = None
    _error: str | None = None
    _logs: list[LogLine] = field(default_factory=list)

    def start(self):
        """Start container as subprocess and begin streaming logs."""
        self._running = True
        cmd = self._build_command()

        try:
            env = dict(self.config.env)
            iris_root = Path(__file__).resolve().parents[4]
            extra_paths = [str(iris_root / "src"), str(iris_root)]
            existing = env.get("PYTHONPATH", "")
            prefix = os.pathsep.join(p for p in extra_paths if p not in existing.split(os.pathsep))
            env["PYTHONPATH"] = f"{prefix}{os.pathsep}{existing}" if existing else prefix

            # Use process groups on Unix for clean termination
            # Set PR_SET_PDEATHSIG on Linux for automatic cleanup if parent dies
            popen_kwargs: dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "env": env,
                "bufsize": 1,  # Line buffered
            }

            if sys.platform != "win32":
                # Create new process group for clean termination
                popen_kwargs["start_new_session"] = True
                # Set up automatic termination if parent dies (Linux only)
                popen_kwargs["preexec_fn"] = _set_pdeathsig_preexec

            self._process = subprocess.Popen(cmd, **popen_kwargs)

            # Spawn thread to stream logs asynchronously
            name_suffix = self.config.task_id or self.config.job_id or "unnamed"
            self._log_thread = get_thread_container().spawn(
                target=self._stream_logs,
                name=f"logs-{name_suffix}",
            )
        except Exception as e:
            self._error = str(e)
            self._exit_code = 1
            self._running = False
            logger.exception("Failed to start container")

    def _build_command(self) -> list[str]:
        """Build command for both callable and command entrypoints.

        Uses the same thunk pattern as DockerContainerRuntime to ensure
        consistent behavior between local and Docker execution.
        """
        if self.config.entrypoint.is_command:
            assert self.config.entrypoint.command is not None
            return self.config.entrypoint.command

        # Callable entrypoint: build Python thunk using cloudpickle bytes
        # This mirrors DockerContainerRuntime._build_command()
        assert self.config.entrypoint.callable_bytes is not None
        encoded = base64.b64encode(self.config.entrypoint.callable_bytes).decode()

        # Build job_info setup from environment variables
        env = self.config.env
        task_id_str = env.get("IRIS_JOB_ID")
        job_info_setup = f"""
num_tasks={env.get("IRIS_NUM_TASKS", "1")},
attempt_id={env.get("IRIS_ATTEMPT_ID", "0")},
worker_id={env.get("IRIS_WORKER_ID")!r},
controller_address={env.get("IRIS_CONTROLLER_ADDRESS")!r},
bundle_gcs_path={env.get("IRIS_BUNDLE_GCS_PATH")!r},
"""

        thunk = f"""
import cloudpickle
import base64
import sys
import traceback

# Set up job_info context (same as in-thread execution)
try:
    from iris.cluster.client.job_info import JobInfo, set_job_info, _parse_ports_from_env
    from iris.cluster.types import JobName
    task_id_str = {task_id_str!r}
    if task_id_str:
        job_info = JobInfo(
            task_id=JobName.from_wire(task_id_str),
            {job_info_setup}
            ports=_parse_ports_from_env({env!r}),
        )
        set_job_info(job_info)
except Exception:
    pass  # Best-effort job_info setup; job execution should still proceed

# Execute cloudpickled function
try:
    fn, args, kwargs = cloudpickle.loads(base64.b64decode('{encoded}'))
    fn(*args, **kwargs)
except Exception:
    traceback.print_exc()
    sys.exit(1)
"""
        return [sys.executable, "-u", "-c", thunk]

    def _stream_logs(self, stop_event: threading.Event):
        """Stream stdout/stderr from subprocess to log buffer.

        Runs in a separate thread to avoid blocking. Uses select() for
        non-blocking reads with timeout to respect stop_event.
        """
        if not self._process:
            return

        try:
            while self._process.poll() is None:
                if stop_event.is_set():
                    break

                # Non-blocking read with timeout
                ready, _, _ = select.select([self._process.stdout, self._process.stderr], [], [], 0.1)

                for stream in ready:
                    line = stream.readline()
                    if line:
                        source = "stdout" if stream == self._process.stdout else "stderr"
                        self._logs.append(
                            LogLine(
                                timestamp=datetime.now(timezone.utc),
                                source=source,
                                data=line.rstrip(),
                            )
                        )

            # Process exited - drain remaining output
            if self._process.stdout:
                for line in self._process.stdout:
                    self._logs.append(
                        LogLine(
                            timestamp=datetime.now(timezone.utc),
                            source="stdout",
                            data=line.rstrip(),
                        )
                    )
            if self._process.stderr:
                for line in self._process.stderr:
                    self._logs.append(
                        LogLine(
                            timestamp=datetime.now(timezone.utc),
                            source="stderr",
                            data=line.rstrip(),
                        )
                    )

            self._exit_code = self._process.returncode
            self._running = False

        except Exception as e:
            logger.exception("Error streaming logs from container")
            self._error = str(e)
            self._exit_code = 1
            self._running = False

    def kill(self):
        """Hard kill the subprocess immediately via SIGKILL.

        On Unix: kills the entire process group to ensure all children are terminated.
        On Windows: kills just the process.
        """
        if self._process and self._process.poll() is None:
            logger.debug("Killing container process %s", self._process.pid)
            try:
                if sys.platform == "win32":
                    self._process.kill()
                else:
                    # Kill entire process group
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            except ProcessLookupError:
                # Process already terminated
                pass
            except Exception as e:
                logger.warning("Failed to kill process %s: %s", self._process.pid, e)
                # Fall back to just killing the process itself
                self._process.kill()

            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                logger.warning("Process did not terminate after SIGKILL")

        self._running = False
        if self._exit_code is None:
            self._exit_code = 137  # 128 + SIGKILL


class _LocalContainerRuntime(ContainerRuntime):
    def __init__(self):
        self._containers: dict[str, _LocalContainer] = {}

    def create_container(self, config: ContainerConfig) -> str:
        container_id = f"local-{uuid.uuid4().hex[:8]}"
        self._containers[container_id] = _LocalContainer(
            config=config,
        )
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

    def list_iris_containers(self, all_states: bool = True) -> list[str]:
        del all_states
        return list(self._containers.keys())

    def remove_all_iris_containers(self) -> int:
        count = len(self._containers)
        self._containers.clear()
        return count


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
        dockerfile: str,
        job_id: str,
        task_logs: TaskLogs | None = None,
    ) -> BuildResult:
        del bundle_path, dockerfile, job_id, task_logs
        return BuildResult(
            image_tag="local:latest",
            build_time_ms=0,
            from_cache=True,
        )

    def protect(self, tag: str) -> None:
        """No-op for local provider (no eviction)."""
        del tag

    def unprotect(self, tag: str) -> None:
        """No-op for local provider (no eviction)."""
        del tag


class LocalEnvironmentProvider:
    def __init__(
        self,
        cpu: int = 1000,
        memory_gb: int = 1000,
        attributes: dict[str, str | int | float] | None = None,
        device: cluster_pb2.DeviceConfig | None = None,
    ):
        self._cpu = cpu
        self._memory_gb = memory_gb
        self._attributes = attributes or {}
        self._device = device

    def probe(self) -> cluster_pb2.WorkerMetadata:
        if self._device is not None:
            device = self._device
        else:
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


def find_free_port() -> int:
    """Find an available port."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class _StubManagedVm(ManagedVm):
    """Minimal ManagedVm stub that holds VmInfo without lifecycle management.

    Used by LocalVmGroup to satisfy the VmGroupProtocol interface without
    running actual bootstrap threads.
    """

    def __init__(self, info: vm_pb2.VmInfo):
        # Don't call super().__init__ - just set the minimal attributes
        self.info = info
        self._log_lines: list[str] = []

    def start(self) -> None:
        pass

    def stop(self, timeout: Duration = Duration.from_seconds(10.0)) -> None:
        pass

    def init_log(self, tail: int | None = None) -> str:
        return ""

    def check_health(self) -> bool:
        return True


class LocalVmGroup(VmGroupProtocol):
    """In-process VM group that wraps Worker instances.

    For the demo, each VM group represents a "slice" that contains one or more
    workers. Workers become ready immediately (no bootstrap delay).
    """

    def __init__(
        self,
        group_id: str,
        scale_group: str,
        workers: list[Worker],
        worker_ids: list[str],
        worker_ports: list[int],
        vm_registry: VmRegistry,
    ):
        self._group_id = group_id
        self._scale_group = scale_group
        self._workers = workers
        self._worker_ids = worker_ids
        self._created_at = Timestamp.now()
        self._vm_registry = vm_registry
        self._terminated = False

        # Create mock ManagedVm instances for each worker (for autoscaler compatibility)
        self._managed_vms: list[ManagedVm] = []
        for i, (worker_id, port) in enumerate(zip(worker_ids, worker_ports, strict=True)):
            # Create a minimal ManagedVm that's immediately ready
            # We don't use ManagedVm's lifecycle thread since workers are in-process
            vm_info = vm_pb2.VmInfo(
                vm_id=f"{group_id}-vm-{i}",
                slice_id=group_id,
                scale_group=scale_group,
                state=vm_pb2.VM_STATE_READY,
                address=f"127.0.0.1:{port}",
                zone="local",
                worker_id=worker_id,
                created_at=self._created_at.to_proto(),
                state_changed_at=self._created_at.to_proto(),
            )
            # Create a stub ManagedVm that just holds the info
            managed_vm = _StubManagedVm(vm_info)
            self._managed_vms.append(managed_vm)
            vm_registry.register(managed_vm)

    @property
    def group_id(self) -> str:
        return self._group_id

    @property
    def slice_id(self) -> str:
        return self._group_id

    @property
    def scale_group(self) -> str:
        return self._scale_group

    @property
    def created_at_ms(self) -> int:
        """Timestamp when this VM group was created (milliseconds since epoch)."""
        return self._created_at.epoch_ms()

    def status(self) -> VmGroupStatus:
        if self._terminated:
            return VmGroupStatus(
                vms=[
                    VmSnapshot(
                        vm_id=vm.info.vm_id,
                        state=vm_pb2.VM_STATE_TERMINATED,
                        address="",
                        init_phase="",
                        init_error="",
                    )
                    for vm in self._managed_vms
                ]
            )
        return VmGroupStatus(
            vms=[
                VmSnapshot(
                    vm_id=vm.info.vm_id,
                    state=vm_pb2.VM_STATE_READY,
                    address=vm.info.address,
                    init_phase="ready",
                    init_error="",
                )
                for vm in self._managed_vms
            ]
        )

    def vms(self) -> list[ManagedVm]:
        return self._managed_vms

    def terminate(self) -> None:
        if self._terminated:
            return
        self._terminated = True
        for worker in self._workers:
            worker.stop()
        for vm in self._managed_vms:
            self._vm_registry.unregister(vm.info.vm_id)

    def to_proto(self) -> vm_pb2.SliceInfo:
        return vm_pb2.SliceInfo(
            slice_id=self._group_id,
            scale_group=self._scale_group,
            created_at=self._created_at.to_proto(),
            vms=[vm.info for vm in self._managed_vms],
        )


class LocalVmManager:
    """VmManager for in-process demo workers.

    Creates LocalVmGroup instances containing in-process Worker instances.
    Workers are created with appropriate attributes based on the scale group
    configuration (TPU topology, etc.).
    """

    def __init__(
        self,
        scale_group_config: config_pb2.ScaleGroupConfig,
        controller_address: str,
        cache_path: Path,
        fake_bundle: Path,
        vm_registry: VmRegistry,
        port_allocator: PortAllocator,
        threads: ThreadContainer | None = None,
    ):
        self._config = scale_group_config
        self._controller_address = controller_address
        self._cache_path = cache_path
        self._fake_bundle = fake_bundle
        self._vm_registry = vm_registry
        self._port_allocator = port_allocator
        self._slice_counter = 0
        self._threads = threads if threads is not None else get_thread_container()

    def create_vm_group(self, tags: dict[str, str] | None = None) -> VmGroupProtocol:
        """Create a new VM group with workers."""
        slice_id = f"{self._config.name}-slice-{self._slice_counter}"
        self._slice_counter += 1

        # Determine worker count based on accelerator type
        if self._config.accelerator_type != config_pb2.ACCELERATOR_TYPE_CPU:
            try:
                topo = get_tpu_topology(self._config.accelerator_variant)
                worker_count = topo.vm_count
            except ValueError:
                worker_count = 1
        else:
            worker_count = 1

        # Create workers
        workers: list[Worker] = []
        worker_ids: list[str] = []
        worker_ports: list[int] = []

        for tpu_worker_id in range(worker_count):
            # Each worker needs its own runtime to avoid container dict conflicts
            bundle_provider = _LocalBundleProvider(self._fake_bundle)
            image_provider = _LocalImageProvider()
            container_runtime = _LocalContainerRuntime()
            worker_id = f"worker-{slice_id}-{tpu_worker_id}-{uuid.uuid4().hex[:8]}"
            worker_port = find_free_port()

            # Set up worker attributes
            attributes: dict[str, str | int | float] = {}
            if self._config.accelerator_type != config_pb2.ACCELERATOR_TYPE_CPU:
                attributes["tpu-name"] = slice_id
                attributes["tpu-worker-id"] = tpu_worker_id
                attributes["tpu-topology"] = self._config.accelerator_variant

            # Create device config if accelerator is specified
            device = None
            if self._config.accelerator_type != config_pb2.ACCELERATOR_TYPE_CPU:
                topo = get_tpu_topology(self._config.accelerator_variant)
                device = tpu_device(self._config.accelerator_variant, count=topo.chips_per_vm)

            environment_provider = LocalEnvironmentProvider(
                cpu=1000,
                memory_gb=1000,
                attributes=attributes,
                device=device,
            )

            worker_config = WorkerConfig(
                host="127.0.0.1",
                port=worker_port,
                cache_dir=self._cache_path,
                controller_address=self._controller_address,
                worker_id=worker_id,
                poll_interval=Duration.from_seconds(0.1),
            )
            worker_threads = self._threads.create_child(f"worker-{worker_id}")
            worker = Worker(
                worker_config,
                cache_dir=self._cache_path,
                bundle_provider=bundle_provider,
                image_provider=image_provider,
                container_runtime=container_runtime,
                environment_provider=environment_provider,
                port_allocator=self._port_allocator,
                threads=worker_threads,
            )
            worker.start()
            workers.append(worker)
            worker_ids.append(worker_id)
            worker_ports.append(worker_port)

        logger.info(
            "LocalVmManager created VM group %s with %d workers for scale group %s",
            slice_id,
            len(workers),
            self._config.name,
        )

        return LocalVmGroup(
            group_id=slice_id,
            scale_group=self._config.name,
            workers=workers,
            worker_ids=worker_ids,
            worker_ports=worker_ports,
            vm_registry=self._vm_registry,
        )

    def discover_vm_groups(self) -> list[VmGroupProtocol]:
        """Return empty list - no recovery for local demo."""
        return []

    def stop(self) -> None:
        """Stop all container threads managed by this VM manager."""
        self._threads.stop(timeout=Duration.from_seconds(5.0))
