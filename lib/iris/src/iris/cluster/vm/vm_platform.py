# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Platform-agnostic protocols and data classes for VM management.

This module provides:
- VmManagerProtocol: Factory protocol for creating VM groups
- VmGroupProtocol: Protocol for VM group lifecycle management
- VmSnapshot: Point-in-time snapshot of a VM's state
- VmGroupStatus: Aggregate status computed from VM states

Platform-specific implementations live in separate modules:
- gcp_tpu_platform.py: TpuVmManager, TpuVmGroup
- manual_platform.py: ManualVmManager, ManualVmGroup
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from iris.cluster.vm.managed_vm import ManagedVm
from iris.rpc import vm_pb2

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
    """VM group status computed from VM states.

    Holds raw VM snapshots and computes aggregate properties on demand,
    rather than precomputing booleans at construction time.
    """

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


class VmGroupProtocol(Protocol):
    """A running VM group with lifecycle management.

    A VM group is the atomic unit of scaling - for TPUs, this corresponds to
    a complete TPU pod which may span multiple hosts.
    """

    @property
    def group_id(self) -> str:
        """Unique identifier for this VM group."""
        ...

    @property
    def slice_id(self) -> str:
        """Alias for group_id - the primary ID used in SliceInfo protos."""
        ...

    @property
    def scale_group(self) -> str:
        """Name of the scale group this VM group belongs to."""
        ...

    @property
    def created_at_ms(self) -> int:
        """Timestamp when this VM group was created (milliseconds since epoch)."""
        ...

    def status(self) -> VmGroupStatus:
        """Current status computed from VM states."""
        ...

    def vms(self) -> list[ManagedVm]:
        """Individual VM instances in this group."""
        ...

    def terminate(self) -> None:
        """Terminate this VM group and all its VMs.

        This stops VM lifecycle threads, unregisters VMs from the registry,
        and deletes the underlying cloud resource.
        """
        ...

    def to_proto(self) -> vm_pb2.SliceInfo:
        """Convert to proto for RPC APIs."""
        ...


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


class VmManagerProtocol(Protocol):
    """Factory for creating VM groups. One instance per scale group.

    The manager knows its scale group configuration and creates appropriately
    configured VM groups. It does NOT track groups - that responsibility belongs
    to the ScalingGroup.
    """

    def create_vm_group(self, tags: dict[str, str] | None = None) -> VmGroupProtocol:
        """Create a new VM group. Returns a ready-to-use VmGroup object.

        Args:
            tags: Optional labels/tags for the VM group and its VMs

        Returns:
            A VmGroup object with lifecycle management
        """
        ...

    def discover_vm_groups(self) -> list[VmGroupProtocol]:
        """Find and adopt existing VM groups from cloud.

        Called once at startup to recover state from a previous controller.
        Returns ready-to-use VmGroup objects with their VMs already started.

        Returns:
            List of discovered VmGroup objects
        """
        ...
