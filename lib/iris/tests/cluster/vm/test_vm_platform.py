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

"""Tests for slice lifecycle helpers."""

from __future__ import annotations

from iris.cluster.controller.slice_lifecycle import (
    VmGroupStatus,
    VmSnapshot,
    compute_slice_state_counts,
    slice_all_ready,
    slice_any_failed,
    slice_is_terminal,
)
from iris.rpc import vm_pb2


def make_snapshot(
    vm_id: str = "vm-0",
    state: vm_pb2.VmState = vm_pb2.VM_STATE_READY,
    address: str = "10.0.0.1",
    init_phase: str = "",
    init_error: str = "",
) -> VmSnapshot:
    return VmSnapshot(
        vm_id=vm_id,
        state=state,
        address=address,
        init_phase=init_phase,
        init_error=init_error,
    )


def test_vm_group_status_aggregate_properties():
    vms = [
        make_snapshot(state=vm_pb2.VM_STATE_READY),
        make_snapshot(vm_id="vm-1", state=vm_pb2.VM_STATE_READY),
    ]
    status = VmGroupStatus(vms=vms)
    assert status.all_ready is True
    assert status.any_failed is False
    assert status.is_terminal is True
    assert status.vm_count == 2
    assert status.ready_count == 2


def test_vm_group_status_error_messages():
    vms = [
        make_snapshot(state=vm_pb2.VM_STATE_FAILED, init_error="boom"),
        make_snapshot(state=vm_pb2.VM_STATE_READY, init_error=""),
    ]
    status = VmGroupStatus(vms=vms)
    assert status.any_failed is True
    assert status.error_messages == ["boom"]


def test_slice_state_helpers():
    slices = [
        vm_pb2.SliceInfo(
            slice_id="s1",
            vms=[vm_pb2.VmInfo(state=vm_pb2.VM_STATE_READY)],
        ),
        vm_pb2.SliceInfo(
            slice_id="s2",
            vms=[vm_pb2.VmInfo(state=vm_pb2.VM_STATE_FAILED)],
        ),
        vm_pb2.SliceInfo(
            slice_id="s3",
            vms=[vm_pb2.VmInfo(state=vm_pb2.VM_STATE_INITIALIZING)],
        ),
    ]

    assert slice_all_ready(slices[0]) is True
    assert slice_any_failed(slices[1]) is True
    assert slice_is_terminal(slices[0]) is True
    assert slice_is_terminal(slices[2]) is False

    counts = compute_slice_state_counts(slices)
    assert counts["ready"] == 1
    assert counts["failed"] == 1
    assert counts["initializing"] == 1
