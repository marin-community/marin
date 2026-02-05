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

"""Autoscaler/VM failure chaos tests.

Tests that validate VM lifecycle failure modes:
- Test 18: VM creation fails with quota exceeded, retry after clearing
- Test 19: VM boots but never initializes (stuck in INITIALIZING)
- Test 20: VM preempted (terminated)

These tests use FakeVmManager directly to simulate VM failures at the
infrastructure layer, not requiring the full cluster fixture.
"""

import pytest

from iris.rpc import config_pb2, vm_pb2
from iris.time_utils import Timestamp
from tests.cluster.vm.fakes import FailureMode, FakeVmManager, FakeVmManagerConfig


@pytest.mark.chaos
def test_quota_exceeded_retry():
    """Test 18: VM creation fails with quota exceeded, retry after clearing."""
    config = config_pb2.ScaleGroupConfig(
        name="test",
        accelerator_variant="v4-8",
        min_slices=0,
        max_slices=3,
        zones=["us-central1-a"],
    )
    manager = FakeVmManager(FakeVmManagerConfig(config=config, failure_mode=FailureMode.QUOTA_EXCEEDED))

    # First attempt should fail with quota error
    try:
        manager.create_slice()
        pytest.fail("should have raised QuotaExceededError")
    except Exception as e:
        # Verify it's a quota error
        assert "Quota exceeded" in str(e) or "quota" in str(e).lower()

    # Clear failure mode and retry
    manager.set_failure_mode(FailureMode.NONE)
    group = manager.create_slice()

    # Tick to advance VM state transitions
    manager.tick(Timestamp.now().epoch_ms())

    # Verify VMs reach READY state
    status = group.status()
    assert any(
        vm.state == vm_pb2.VM_STATE_READY for vm in status.vms
    ), f"Expected at least one VM in READY state, got states: {[vm.state for vm in status.vms]}"


@pytest.mark.chaos
def test_vm_init_stuck():
    """Test 19: VM boots but worker never initializes (stuck in INITIALIZING)."""
    config = config_pb2.ScaleGroupConfig(
        name="stuck",
        accelerator_variant="v4-8",
        min_slices=0,
        max_slices=3,
        zones=["us-central1-a"],
    )
    # Set init_delay_ms to a huge value so VMs never complete initialization
    manager = FakeVmManager(FakeVmManagerConfig(config=config, init_delay_ms=999_999_999))
    group = manager.create_slice()

    # Tick to complete boot phase
    manager.tick(Timestamp.now().epoch_ms())

    # VMs should transition from BOOTING -> INITIALIZING but not to READY
    status = group.status()
    vm_states = [vm.state for vm in status.vms]

    # Verify no VMs reach READY state (they should be stuck in INITIALIZING or earlier)
    assert all(
        vm.state != vm_pb2.VM_STATE_READY for vm in status.vms
    ), f"Expected no VMs in READY state, got states: {vm_states}"

    # Most should be in INITIALIZING (after boot completes)
    assert any(
        vm.state == vm_pb2.VM_STATE_INITIALIZING for vm in status.vms
    ), f"Expected at least one VM in INITIALIZING state, got states: {vm_states}"


@pytest.mark.chaos
def test_vm_preempted():
    """Test 20: VM preempted (terminated)."""
    config = config_pb2.ScaleGroupConfig(
        name="preempt",
        accelerator_variant="v4-8",
        min_slices=0,
        max_slices=3,
        zones=["us-central1-a"],
    )
    manager = FakeVmManager(FakeVmManagerConfig(config=config))
    group = manager.create_slice()

    # Tick to advance VMs to READY state
    manager.tick(Timestamp.now().epoch_ms())

    # Verify at least one VM is READY
    status = group.status()
    assert any(
        vm.state == vm_pb2.VM_STATE_READY for vm in status.vms
    ), f"Expected at least one VM in READY state before termination, got: {[vm.state for vm in status.vms]}"

    # Simulate preemption by terminating the VM group
    group.terminate()

    # Verify all VMs are now TERMINATED
    status = group.status()
    assert all(
        vm.state == vm_pb2.VM_STATE_TERMINATED for vm in status.vms
    ), f"Expected all VMs in TERMINATED state after preemption, got: {[vm.state for vm in status.vms]}"
