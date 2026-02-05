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

"""Controller-owned slice lifecycle helpers and protocols."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, cast

from iris.cluster.controller.worker_vm import TrackedVmFactory, WorkerVm
from iris.cluster.platform import Platform
from iris.cluster.platform.base import SliceHandle
from iris.rpc import config_pb2, vm_pb2
from iris.time_utils import Timestamp

# Maximum workers for parallel host checks during discovery
MAX_RECONCILE_WORKERS = 8


@dataclass
class VmSnapshot:
    """Point-in-time snapshot of a VM's state."""

    vm_id: str
    state: vm_pb2.VmState
    address: str
    init_phase: str
    init_error: str


@dataclass
class VmGroupStatus:
    """Slice status computed from VM states."""

    vms: list[VmSnapshot]

    @property
    def all_ready(self) -> bool:
        """True if all VMs in the group are in READY state."""
        return all(v.state == vm_pb2.VM_STATE_READY for v in self.vms)

    @property
    def any_failed(self) -> bool:
        """True if any VM has failed or been preempted."""
        return any(v.state in (vm_pb2.VM_STATE_FAILED, vm_pb2.VM_STATE_PREEMPTED) for v in self.vms)

    @property
    def is_terminal(self) -> bool:
        """True if all VMs are in a terminal state (no further transitions expected)."""
        terminal = {
            vm_pb2.VM_STATE_READY,
            vm_pb2.VM_STATE_FAILED,
            vm_pb2.VM_STATE_TERMINATED,
            vm_pb2.VM_STATE_PREEMPTED,
        }
        return all(v.state in terminal for v in self.vms)

    @property
    def vm_count(self) -> int:
        """Total number of VMs in the group."""
        return len(self.vms)

    @property
    def ready_count(self) -> int:
        """Number of VMs in READY state."""
        return sum(1 for v in self.vms if v.state == vm_pb2.VM_STATE_READY)

    @property
    def error_messages(self) -> list[str]:
        """Collect non-empty error messages from VMs."""
        return [v.init_error for v in self.vms if v.init_error]


class SliceGroupProtocol(Protocol):
    """Controller-owned slice with worker lifecycle management."""

    @property
    def slice_id(self) -> str: ...

    @property
    def scale_group(self) -> str: ...

    @property
    def created_at_ms(self) -> int: ...

    def status(self) -> VmGroupStatus: ...

    def vms(self) -> list[WorkerVm]: ...

    def terminate(self) -> None: ...

    def to_proto(self) -> vm_pb2.SliceInfo: ...


class SliceFactoryProtocol(Protocol):
    """Factory for creating and discovering slices in a scale group."""

    def create_slice(self, tags: dict[str, str] | None = None) -> SliceGroupProtocol: ...

    def discover_slices(self) -> list[SliceGroupProtocol]: ...

    def stop(self) -> None: ...


class WorkerSlice:
    """Controller-owned slice that wraps a platform slice handle + WorkerVms."""

    def __init__(
        self,
        handle: SliceHandle,
        bootstrap_config: config_pb2.BootstrapConfig,
        timeouts: config_pb2.TimeoutConfig,
        vm_factory: TrackedVmFactory,
        created_at: Timestamp | None = None,
    ):
        self._handle = handle
        self._bootstrap = bootstrap_config
        self._timeouts = timeouts
        self._vm_factory = vm_factory
        self._created_at = created_at if created_at is not None else Timestamp.now()
        self._vms = self._create_vms()

    @property
    def slice_id(self) -> str:
        return self._handle.slice_id

    @property
    def scale_group(self) -> str:
        return self._handle.scale_group

    @property
    def created_at_ms(self) -> int:
        return self._created_at.epoch_ms()

    def vms(self) -> list[WorkerVm]:
        return list(self._vms)

    def status(self) -> VmGroupStatus:
        snapshots = [
            VmSnapshot(
                vm_id=vm.info.vm_id,
                state=vm.info.state,
                address=vm.info.address,
                init_phase=vm.info.init_phase,
                init_error=vm.info.init_error,
            )
            for vm in self._vms
        ]
        return VmGroupStatus(vms=snapshots)

    def terminate(self) -> None:
        for vm in self._vms:
            vm.shutdown(graceful=True)
            vm.stop()
            self._vm_factory.registry.unregister(vm.info.vm_id)
        self._handle.terminate()

    def to_proto(self) -> vm_pb2.SliceInfo:
        return vm_pb2.SliceInfo(
            slice_id=self.slice_id,
            scale_group=self.scale_group,
            created_at=self._created_at.to_proto(),
            vms=[vm.info for vm in self._vms],
        )

    def _create_vms(self) -> list[WorkerVm]:
        vms: list[WorkerVm] = []
        for target in self._handle.vm_targets():
            vm = self._vm_factory.create_vm(
                vm_id=target.vm_id,
                slice_id=self.slice_id,
                scale_group=self.scale_group,
                zone=target.zone,
                conn=target.conn,
                bootstrap_config=self._bootstrap,
                timeouts=self._timeouts,
                labels=dict(self._handle.labels),
                address=target.address,
                discovery_preamble=self._handle.discovery_preamble,
            )
            vms.append(vm)
        return vms


class PlatformSliceFactory:
    """Slice factory that composes platform slice handles + controller workers."""

    def __init__(
        self,
        platform: Platform,
        group_config: config_pb2.ScaleGroupConfig,
        bootstrap_config: config_pb2.BootstrapConfig,
        timeouts: config_pb2.TimeoutConfig,
        vm_factory: TrackedVmFactory,
        *,
        dry_run: bool = False,
    ):
        self._platform = platform
        self._group_config = group_config
        self._bootstrap = bootstrap_config
        self._timeouts = timeouts
        self._vm_factory = vm_factory
        self._dry_run = dry_run

    def create_slice(self, tags: dict[str, str] | None = None) -> SliceGroupProtocol:
        if self._dry_run:
            handle = DryRunSliceHandle(
                slice_id=f"dry-run-{self._group_config.name}-{Timestamp.now().epoch_ms()}",
                scale_group=self._group_config.name,
                labels=tags or {},
            )
        else:
            handle = self._platform.create_slice(self._group_config, tags=tags)
        return cast(SliceGroupProtocol, WorkerSlice(handle, self._bootstrap, self._timeouts, self._vm_factory))

    def discover_slices(self) -> list[SliceGroupProtocol]:
        if self._dry_run:
            handles = []
        else:
            handles = self._platform.discover_slices(self._group_config)
        return [
            cast(SliceGroupProtocol, WorkerSlice(handle, self._bootstrap, self._timeouts, self._vm_factory))
            for handle in handles
        ]

    def stop(self) -> None:
        return None


class DryRunSliceHandle:
    """No-op slice handle for dry-run operations."""

    def __init__(self, slice_id: str, scale_group: str, labels: dict[str, str]):
        self._slice_id = slice_id
        self._scale_group = scale_group
        self._labels = labels

    @property
    def slice_id(self) -> str:
        return self._slice_id

    @property
    def scale_group(self) -> str:
        return self._scale_group

    @property
    def labels(self) -> dict[str, str]:
        return self._labels

    @property
    def discovery_preamble(self) -> str:
        return ""

    def vm_targets(self) -> list:
        return []

    def describe(self) -> vm_pb2.SliceInfo:
        return vm_pb2.SliceInfo(slice_id=self._slice_id, scale_group=self._scale_group)

    def terminate(self) -> None:
        return None


def slice_all_ready(slice_info: vm_pb2.SliceInfo) -> bool:
    """Compute all_ready from vms[] in proto."""
    return all(vm.state == vm_pb2.VM_STATE_READY for vm in slice_info.vms)


def slice_any_failed(slice_info: vm_pb2.SliceInfo) -> bool:
    """Compute any_failed from vms[] in proto."""
    return any(vm.state in (vm_pb2.VM_STATE_FAILED, vm_pb2.VM_STATE_PREEMPTED) for vm in slice_info.vms)


def slice_is_terminal(slice_info: vm_pb2.SliceInfo) -> bool:
    """Compute is_terminal from vms[] in proto."""
    terminal = {
        vm_pb2.VM_STATE_READY,
        vm_pb2.VM_STATE_FAILED,
        vm_pb2.VM_STATE_TERMINATED,
        vm_pb2.VM_STATE_PREEMPTED,
    }
    return all(vm.state in terminal for vm in slice_info.vms)


def compute_slice_state_counts(slices: Iterable[vm_pb2.SliceInfo]) -> dict[str, int]:
    """Compute slice state counts from a list of SliceInfo protos."""
    counts = {"booting": 0, "initializing": 0, "ready": 0, "failed": 0}
    for s in slices:
        if slice_any_failed(s):
            counts["failed"] += 1
        elif slice_all_ready(s):
            counts["ready"] += 1
        elif any(vm.state == vm_pb2.VM_STATE_INITIALIZING for vm in s.vms):
            counts["initializing"] += 1
        else:
            counts["booting"] += 1
    return counts
