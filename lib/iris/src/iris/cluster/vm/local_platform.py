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
that create real Worker instances running in-process with thread-based execution
instead of Docker containers.

Also provides the local provider implementations used by workers:
- LocalEnvironmentProvider: probes local system resources
- _LocalBundleProvider: serves pre-built bundles from local filesystem
- _LocalImageProvider: no-op image provider (uses local:latest)
- _LocalContainerRuntime: executes containers as threads
- _LocalContainer: thread-based container execution model
"""

from __future__ import annotations

import logging
import socket
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from iris.cluster.types import get_tpu_topology, tpu_device
from iris.cluster.vm.autoscaler import Autoscaler
from iris.cluster.vm.managed_vm import ManagedVm, VmRegistry
from iris.cluster.vm.scaling_group import ScalingGroup
from iris.cluster.vm.vm_platform import VmGroupProtocol, VmGroupStatus, VmSnapshot
from iris.cluster.worker.builder import BuildResult
from iris.cluster.worker.worker_types import TaskLogs
from iris.cluster.worker.docker import ContainerConfig, ContainerRuntime, ContainerStats, ContainerStatus
from iris.cluster.worker.worker import PortAllocator, Worker, WorkerConfig
from iris.cluster.worker.worker_types import LogLine
from iris.managed_thread import ManagedThread, ThreadContainer, get_thread_registry
from iris.rpc import cluster_pb2, config_pb2, vm_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)


# =============================================================================
# Local Providers (in-process implementations for testing)
# =============================================================================


@dataclass
class _LocalContainer:
    config: ContainerConfig
    _thread_container: ThreadContainer
    _managed_thread: ManagedThread | None = field(default=None, repr=False)
    _running: bool = False
    _exit_code: int | None = None
    _error: str | None = None
    _logs: list[LogLine] = field(default_factory=list)
    _killed: threading.Event = field(default_factory=threading.Event)

    def start(self):
        self._running = True
        # Use task_id or job_id as name, or fall back to unnamed
        name_suffix = self.config.task_id or self.config.job_id or "unnamed"
        self._managed_thread = self._thread_container.spawn(
            target=self._execute_managed,
            name=f"container-{name_suffix}",
        )

    def _execute_managed(self, stop_event: threading.Event):
        """Managed thread wrapper that can be stopped via stop_event."""
        # Just run the container execution directly
        # The stop_event allows ThreadContainer to signal us to stop,
        # but we don't need to check it during normal execution
        self._execute()

    def _execute(self):
        from iris.cluster.client.job_info import JobInfo, _parse_ports_from_env, set_job_info

        container_name = self.config.task_id or self.config.job_id or "unnamed"
        container_logger = logging.getLogger(f"iris.container.{container_name}")

        try:
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

            entrypoint = self.config.entrypoint

            if self._killed.is_set():
                self._exit_code = 137
                return

            if entrypoint.is_callable:
                # Run callable directly in-thread; output goes to parent process.
                # Use a named logger so output can be attributed to this container.
                container_logger.info("Starting callable entrypoint")
                fn, args, kwargs = entrypoint.resolve()
                fn(*args, **kwargs)
            else:
                # Command entrypoint: run as subprocess, capture and log output.
                assert entrypoint.command is not None
                container_logger.info("Starting command: %s", entrypoint.command)
                result = subprocess.run(
                    entrypoint.command,
                    capture_output=True,
                    text=True,
                    check=False,
                    env=self.config.env,
                )
                self._append_subprocess_logs(result.stdout, "stdout")
                self._append_subprocess_logs(result.stderr, "stderr")
                if result.returncode != 0:
                    raise RuntimeError(f"Command failed with exit code {result.returncode}")

            self._exit_code = 0

        except Exception as e:
            self._error = str(e)
            self._exit_code = 1
        finally:
            self._running = False

    def _append_subprocess_logs(self, output: str, source: str) -> None:
        """Append subprocess output lines to the container log."""
        for line in output.splitlines():
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
        if self._managed_thread and self._managed_thread.is_alive:
            self._managed_thread.stop()
            self._managed_thread.join(timeout=0.5)
        if self._running:
            self._running = False
            self._exit_code = 137


class _LocalContainerRuntime(ContainerRuntime):
    def __init__(self, thread_container: ThreadContainer):
        self._containers: dict[str, _LocalContainer] = {}
        self._thread_container = thread_container

    def create_container(self, config: ContainerConfig) -> str:
        container_id = f"local-{uuid.uuid4().hex[:8]}"
        self._containers[container_id] = _LocalContainer(
            config=config,
            _thread_container=self._thread_container,
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

    def stop(self, timeout: float = 10.0) -> None:
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
        thread_container: ThreadContainer,
    ):
        self._group_id = group_id
        self._scale_group = scale_group
        self._workers = workers
        self._worker_ids = worker_ids
        self._created_at = Timestamp.now()
        self._vm_registry = vm_registry
        self._terminated = False
        self._thread_container = thread_container

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
        # Stop all container threads
        self._thread_container.stop(timeout=5.0)

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
        self._threads = threads if threads is not None else get_thread_registry().container

    def create_vm_group(self, tags: dict[str, str] | None = None) -> VmGroupProtocol:
        """Create a new VM group with workers."""
        slice_id = f"{self._config.name}-slice-{self._slice_counter}"
        self._slice_counter += 1

        # Create a child container for this VM group's threads
        group_threads = self._threads.create_child(f"group-{slice_id}")

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
            container_runtime = _LocalContainerRuntime(thread_container=group_threads)
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
            worker = Worker(
                worker_config,
                cache_dir=self._cache_path,
                bundle_provider=bundle_provider,
                image_provider=image_provider,
                container_runtime=container_runtime,
                environment_provider=environment_provider,
                port_allocator=self._port_allocator,
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
            thread_container=group_threads,
        )

    def discover_vm_groups(self) -> list[VmGroupProtocol]:
        """Return empty list - no recovery for local demo."""
        return []

    def stop(self) -> None:
        """Stop all container threads managed by this VM manager."""
        self._threads.stop(timeout=5.0)


def _create_local_autoscaler(
    config: config_pb2.IrisClusterConfig,
    controller_address: str,
    cache_path: Path,
    fake_bundle: Path,
) -> Autoscaler:
    """Create Autoscaler with LocalVmManagers for all scale groups.

    Parallels create_autoscaler_from_config() but uses LocalVmManagers.
    Each scale group in the config gets a LocalVmManager that creates
    in-process workers instead of cloud VMs.
    """
    vm_registry = VmRegistry()
    shared_port_allocator = PortAllocator(port_range=(30000, 40000))

    scale_groups: dict[str, ScalingGroup] = {}
    for name, sg_config in config.scale_groups.items():
        manager = LocalVmManager(
            scale_group_config=sg_config,
            controller_address=controller_address,
            cache_path=cache_path,
            fake_bundle=fake_bundle,
            vm_registry=vm_registry,
            port_allocator=shared_port_allocator,
        )
        scale_groups[name] = ScalingGroup(
            config=sg_config,
            vm_manager=manager,
            scale_up_cooldown=Duration.from_ms(1000),
            scale_down_cooldown=Duration.from_ms(300_000),
        )

    return Autoscaler(
        scale_groups=scale_groups,
        vm_registry=vm_registry,
        config=config.autoscaler,
    )
