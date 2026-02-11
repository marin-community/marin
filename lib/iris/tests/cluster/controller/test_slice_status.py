# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

"""Tests for slice_handle_status() readiness gating."""

from unittest.mock import MagicMock

import pytest

from iris.cluster.controller.slice_status import slice_handle_status
from iris.cluster.platform.base import CloudSliceState, CloudVmState, SliceStatus, VmStatus
from iris.rpc import vm_pb2


def _make_vm_handle(vm_id: str, cloud_state: CloudVmState, address: str = "10.0.0.1") -> MagicMock:
    handle = MagicMock()
    handle.vm_id = vm_id
    handle.internal_address = address
    handle.status.return_value = VmStatus(state=cloud_state)
    return handle


def _make_slice_handle(
    slice_id: str,
    slice_state: CloudSliceState,
    vm_handles: list[MagicMock],
) -> MagicMock:
    handle = MagicMock()
    handle.slice_id = slice_id
    handle.status.return_value = SliceStatus(state=slice_state, vm_count=len(vm_handles))
    handle.list_vms.return_value = vm_handles
    return handle


class TestSliceHandleStatusReadinessGating:
    """VMs in a CREATING or REPAIRING slice must not appear READY."""

    @pytest.mark.parametrize("slice_state", [CloudSliceState.CREATING, CloudSliceState.REPAIRING])
    def test_running_vms_clamped_to_booting_when_slice_not_ready(self, slice_state: CloudSliceState):
        """A VM reporting RUNNING should show as BOOTING when the slice is still being set up."""
        vm = _make_vm_handle("vm-0", CloudVmState.RUNNING)
        handle = _make_slice_handle("slice-1", slice_state, [vm])

        status = slice_handle_status(handle)

        assert len(status.vms) == 1
        assert status.vms[0].state == vm_pb2.VM_STATE_BOOTING
        assert not status.all_ready

    def test_running_vms_report_ready_when_slice_ready(self):
        """A VM reporting RUNNING should show as READY when the slice itself is READY."""
        vm = _make_vm_handle("vm-0", CloudVmState.RUNNING)
        handle = _make_slice_handle("slice-1", CloudSliceState.READY, [vm])

        status = slice_handle_status(handle)

        assert len(status.vms) == 1
        assert status.vms[0].state == vm_pb2.VM_STATE_READY
        assert status.all_ready

    def test_non_running_vm_states_unaffected_by_clamping(self):
        """STOPPED/TERMINATED VMs keep their mapped state regardless of slice state."""
        vms = [
            _make_vm_handle("vm-0", CloudVmState.STOPPED),
            _make_vm_handle("vm-1", CloudVmState.TERMINATED, address="10.0.0.2"),
        ]
        handle = _make_slice_handle("slice-1", CloudSliceState.CREATING, vms)

        status = slice_handle_status(handle)

        assert status.vms[0].state == vm_pb2.VM_STATE_FAILED
        assert status.vms[1].state == vm_pb2.VM_STATE_TERMINATED

    def test_mixed_vms_in_creating_slice(self):
        """In a CREATING slice, only RUNNING VMs get clamped; others keep their state."""
        vms = [
            _make_vm_handle("vm-0", CloudVmState.RUNNING, address="10.0.0.1"),
            _make_vm_handle("vm-1", CloudVmState.UNKNOWN, address="10.0.0.2"),
        ]
        handle = _make_slice_handle("slice-1", CloudSliceState.CREATING, vms)

        status = slice_handle_status(handle)

        assert status.vms[0].state == vm_pb2.VM_STATE_BOOTING  # clamped
        assert status.vms[1].state == vm_pb2.VM_STATE_BOOTING  # already BOOTING from UNKNOWN
