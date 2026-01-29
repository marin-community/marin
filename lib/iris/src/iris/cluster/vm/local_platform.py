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

Extracted from demo_cluster.py. Provides LocalVmManager (VmManagerProtocol)
and LocalVmGroup (VmGroupProtocol) that create real Worker instances running
in-process with thread-based execution instead of Docker containers.

Worker dependencies (bundle, image, container, environment providers) come
from iris.cluster.client.local_client.
"""

from __future__ import annotations

import logging
import socket
import uuid
from pathlib import Path

from iris.cluster.client.local_client import (
    LocalEnvironmentProvider,
    _LocalBundleProvider,
    _LocalContainerRuntime,
    _LocalImageProvider,
)
from iris.cluster.types import get_tpu_topology, tpu_device
from iris.cluster.vm.autoscaler import Autoscaler
from iris.cluster.vm.managed_vm import ManagedVm, VmRegistry
from iris.cluster.vm.scaling_group import ScalingGroup
from iris.cluster.vm.vm_platform import VmGroupProtocol, VmGroupStatus, VmSnapshot
from iris.cluster.worker.worker import PortAllocator, Worker, WorkerConfig
from iris.rpc import config_pb2, vm_pb2
from iris.time_utils import now_ms

logger = logging.getLogger(__name__)


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

    def stop(self) -> None:
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
        self._created_at_ms = now_ms()
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
                created_at_ms=self._created_at_ms,
                state_changed_at_ms=self._created_at_ms,
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
        return self._created_at_ms

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
            created_at_ms=self._created_at_ms,
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
    ):
        self._config = scale_group_config
        self._controller_address = controller_address
        self._cache_path = cache_path
        self._fake_bundle = fake_bundle
        self._vm_registry = vm_registry
        self._port_allocator = port_allocator
        self._slice_counter = 0

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

        bundle_provider = _LocalBundleProvider(self._fake_bundle)
        image_provider = _LocalImageProvider()
        container_runtime = _LocalContainerRuntime()

        for tpu_worker_id in range(worker_count):
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
                poll_interval_seconds=0.1,  # Fast polling for demos
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
        )

    def discover_vm_groups(self) -> list[VmGroupProtocol]:
        """Return empty list - no recovery for local demo."""
        return []


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
            scale_up_cooldown_ms=1000,
            scale_down_cooldown_ms=300_000,
        )

    return Autoscaler(
        scale_groups=scale_groups,
        vm_registry=vm_registry,
        config=config_pb2.AutoscalerConfig(evaluation_interval_seconds=2.0),
    )
