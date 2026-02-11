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

"""Domain types and free functions for computing slice/VM status.

Provides VmGroupStatus and VmSnapshot as domain types for aggregate VM state,
plus functions that operate on SliceHandle to compute status and produce protos.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from iris.cluster.platform.base import CloudSliceState, CloudVmState, SliceHandle
from iris.rpc import time_pb2, vm_pb2

logger = logging.getLogger(__name__)


# ============================================================================
# VM status types
# ============================================================================


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
        return all(v.state == vm_pb2.VM_STATE_READY for v in self.vms)

    @property
    def any_failed(self) -> bool:
        return any(v.state in (vm_pb2.VM_STATE_FAILED, vm_pb2.VM_STATE_PREEMPTED) for v in self.vms)

    @property
    def is_terminal(self) -> bool:
        terminal = {
            vm_pb2.VM_STATE_READY,
            vm_pb2.VM_STATE_FAILED,
            vm_pb2.VM_STATE_TERMINATED,
            vm_pb2.VM_STATE_PREEMPTED,
        }
        return all(v.state in terminal for v in self.vms)

    @property
    def vm_count(self) -> int:
        return len(self.vms)

    @property
    def ready_count(self) -> int:
        return sum(1 for v in self.vms if v.state == vm_pb2.VM_STATE_READY)

    @property
    def error_messages(self) -> list[str]:
        return [v.init_error for v in self.vms if v.init_error]


def slice_all_ready(slice_info: vm_pb2.SliceInfo) -> bool:
    """Compute all_ready from vms[] in proto."""
    return all(vm.state == vm_pb2.VM_STATE_READY for vm in slice_info.vms)


def slice_any_failed(slice_info: vm_pb2.SliceInfo) -> bool:
    """Compute any_failed from vms[] in proto."""
    return any(vm.state in (vm_pb2.VM_STATE_FAILED, vm_pb2.VM_STATE_PREEMPTED) for vm in slice_info.vms)


def compute_slice_state_counts(slices: Iterable[vm_pb2.SliceInfo]) -> dict[str, int]:
    """Compute slice state counts from a list of SliceInfo protos."""
    counts = {
        "requesting": 0,
        "booting": 0,
        "initializing": 0,
        "ready": 0,
        "failed": 0,
    }
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


def _cloud_vm_state_to_iris(state: CloudVmState) -> vm_pb2.VmState:
    """Map cloud-level VM state to Iris lifecycle state."""
    if state == CloudVmState.RUNNING:
        return vm_pb2.VM_STATE_READY
    if state == CloudVmState.STOPPED:
        return vm_pb2.VM_STATE_FAILED
    if state == CloudVmState.TERMINATED:
        return vm_pb2.VM_STATE_TERMINATED
    return vm_pb2.VM_STATE_BOOTING


def _cloud_slice_state_to_vm_states(slice_state: CloudSliceState, vm_count: int) -> list[vm_pb2.VmState]:
    """Derive VM states from a cloud slice state when individual VM status is unavailable."""
    if slice_state == CloudSliceState.READY:
        return [vm_pb2.VM_STATE_READY] * vm_count
    if slice_state in (CloudSliceState.CREATING, CloudSliceState.REPAIRING):
        return [vm_pb2.VM_STATE_BOOTING] * vm_count
    if slice_state == CloudSliceState.DELETING:
        return [vm_pb2.VM_STATE_TERMINATED] * vm_count
    return [vm_pb2.VM_STATE_BOOTING] * vm_count


def slice_handle_status(handle: SliceHandle) -> VmGroupStatus:
    """Compute VmGroupStatus by querying a SliceHandle."""
    slice_status = handle.status()

    try:
        vm_handles = handle.list_vms()
    except Exception:
        logger.warning(
            "Failed to list VMs for slice %s; falling back to slice-level state", handle.slice_id, exc_info=True
        )
        vm_states = _cloud_slice_state_to_vm_states(slice_status.state, max(slice_status.vm_count, 1))
        snapshots = [
            VmSnapshot(
                vm_id=f"{handle.slice_id}-vm-{i}",
                state=state,
                address="",
                init_phase="",
                init_error="",
            )
            for i, state in enumerate(vm_states)
        ]
        return VmGroupStatus(vms=snapshots)

    # When the slice itself is still being set up (CREATING/REPAIRING), individual
    # VMs may report RUNNING at the cloud level before bootstrap has completed.
    # Cap VM states at BOOTING to avoid falsely reporting readiness.
    slice_not_ready = slice_status.state in (CloudSliceState.CREATING, CloudSliceState.REPAIRING)

    snapshots = []
    for vm_handle in vm_handles:
        vm_status = vm_handle.status()
        iris_state = _cloud_vm_state_to_iris(vm_status.state)
        if slice_not_ready and iris_state == vm_pb2.VM_STATE_READY:
            iris_state = vm_pb2.VM_STATE_BOOTING
        snapshots.append(
            VmSnapshot(
                vm_id=vm_handle.vm_id,
                state=iris_state,
                address=vm_handle.internal_address,
                init_phase="",
                init_error="",
            )
        )

    if not snapshots:
        vm_states = _cloud_slice_state_to_vm_states(slice_status.state, max(slice_status.vm_count, 1))
        snapshots = [
            VmSnapshot(
                vm_id=f"{handle.slice_id}-vm-{i}",
                state=state,
                address="",
                init_phase="",
                init_error="",
            )
            for i, state in enumerate(vm_states)
        ]

    return VmGroupStatus(vms=snapshots)


def slice_handle_to_proto(handle: SliceHandle) -> vm_pb2.SliceInfo:
    """Convert a SliceHandle to a SliceInfo proto for RPC APIs."""
    created_at = handle.created_at
    status = slice_handle_status(handle)
    return vm_pb2.SliceInfo(
        slice_id=handle.slice_id,
        scale_group=handle.scale_group,
        created_at=time_pb2.Timestamp(epoch_ms=created_at.epoch_ms()),
        vms=[
            vm_pb2.VmInfo(
                vm_id=s.vm_id,
                state=s.state,
                address=s.address,
                init_phase=s.init_phase,
                init_error=s.init_error,
                created_at=time_pb2.Timestamp(epoch_ms=created_at.epoch_ms()),
                state_changed_at=time_pb2.Timestamp(epoch_ms=created_at.epoch_ms()),
            )
            for s in status.vms
        ],
    )
