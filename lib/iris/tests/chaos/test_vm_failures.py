# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Autoscaler/VM failure chaos tests.

Tests that validate VM lifecycle failure modes:
- Test 18: VM creation fails with quota exceeded, retry after clearing
- Test 19: VM boots but never initializes (stuck in INITIALIZING)
- Test 20: VM preempted (terminated)

These tests use FakePlatform directly to simulate VM failures at the
infrastructure layer, not requiring the full cluster fixture.
"""

import pytest

from iris.cluster.controller.scaling_group import slice_handle_status
from iris.cluster.platform.base import QuotaExhaustedError
from iris.rpc import config_pb2, vm_pb2
from iris.time_utils import Timestamp
from tests.cluster.platform.fakes import FailureMode, FakePlatform, FakePlatformConfig


def _make_scale_group_config(name: str, **kwargs) -> config_pb2.ScaleGroupConfig:
    """Create a ScaleGroupConfig with GCP zones in slice_template."""
    return config_pb2.ScaleGroupConfig(
        name=name,
        accelerator_variant="v4-8",
        min_slices=0,
        max_slices=3,
        slice_template=config_pb2.SliceConfig(
            gcp=config_pb2.GcpSliceConfig(zones=["us-central1-a"]),
        ),
        **kwargs,
    )


def _make_slice_config(name: str = "test") -> config_pb2.SliceConfig:
    return config_pb2.SliceConfig(name_prefix=name, slice_size=1, accelerator_variant="v4-8")


@pytest.mark.chaos
def test_quota_exceeded_retry():
    """Test 18: VM creation fails with quota exceeded, retry after clearing."""
    config = _make_scale_group_config("test")
    platform = FakePlatform(FakePlatformConfig(config=config, failure_mode=FailureMode.QUOTA_EXCEEDED))

    # First attempt should fail with quota error
    with pytest.raises(QuotaExhaustedError):
        platform.create_slice(_make_slice_config())

    # Clear failure mode and retry
    platform.set_failure_mode(FailureMode.NONE)
    handle = platform.create_slice(_make_slice_config())

    # Tick to advance VM state transitions
    platform.tick(Timestamp.now().epoch_ms())

    # Verify VMs reach READY state
    status = slice_handle_status(handle)
    assert any(
        vm.state == vm_pb2.VM_STATE_READY for vm in status.vms
    ), f"Expected at least one VM in READY state, got states: {[vm.state for vm in status.vms]}"


@pytest.mark.chaos
def test_vm_init_stuck():
    """Test 19: VM boots but worker never initializes (stuck in INITIALIZING)."""
    config = _make_scale_group_config("stuck")
    # Set init_delay_ms to a huge value so VMs never complete initialization
    platform = FakePlatform(FakePlatformConfig(config=config, init_delay_ms=999_999_999))
    handle = platform.create_slice(_make_slice_config("stuck"))

    # Tick to complete boot phase
    platform.tick(Timestamp.now().epoch_ms())

    # VMs should not reach READY state (stuck waiting for init_delay to pass).
    # The cloud state UNKNOWN maps to VM_STATE_BOOTING via _cloud_vm_state_to_iris.
    status = slice_handle_status(handle)
    vm_states = [vm.state for vm in status.vms]

    assert all(
        vm.state != vm_pb2.VM_STATE_READY for vm in status.vms
    ), f"Expected no VMs in READY state, got states: {vm_states}"


@pytest.mark.chaos
def test_vm_preempted():
    """Test 20: VM preempted (terminated)."""
    config = _make_scale_group_config("preempt")
    platform = FakePlatform(FakePlatformConfig(config=config))
    handle = platform.create_slice(_make_slice_config("preempt"))

    # Tick to advance VMs to READY state
    platform.tick(Timestamp.now().epoch_ms())

    # Verify at least one VM is READY
    status = slice_handle_status(handle)
    assert any(
        vm.state == vm_pb2.VM_STATE_READY for vm in status.vms
    ), f"Expected at least one VM in READY state before termination, got: {[vm.state for vm in status.vms]}"

    # Simulate preemption by terminating the slice
    handle.terminate()

    # Verify all VMs are now TERMINATED
    status = slice_handle_status(handle)
    assert all(
        vm.state == vm_pb2.VM_STATE_TERMINATED for vm in status.vms
    ), f"Expected all VMs in TERMINATED state after preemption, got: {[vm.state for vm in status.vms]}"
