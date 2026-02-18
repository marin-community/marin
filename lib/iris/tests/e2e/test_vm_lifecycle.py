# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""VM lifecycle tests: quota exhaustion, stuck init, preemption.

These tests use FakePlatform directly and don't need a full cluster fixture.
Migrated from tests/chaos/test_vm_failures.py.
"""

import pytest
from iris.cluster.platform.base import CloudWorkerState, QuotaExhaustedError
from iris.rpc import config_pb2
from iris.time_utils import Timestamp
from tests.cluster.platform.fakes import FailureMode, FakePlatform, FakePlatformConfig

pytestmark = pytest.mark.e2e


def _make_scale_group_config(name: str, **kwargs) -> config_pb2.ScaleGroupConfig:
    """Create a ScaleGroupConfig with GCP zones in slice_template."""
    return config_pb2.ScaleGroupConfig(
        name=name,
        accelerator_variant="v4-8",
        min_slices=0,
        max_slices=3,
        slice_template=config_pb2.SliceConfig(
            gcp=config_pb2.GcpSliceConfig(zone="us-central1-a"),
        ),
        **kwargs,
    )


def _make_slice_config(name: str = "test") -> config_pb2.SliceConfig:
    return config_pb2.SliceConfig(name_prefix=name, num_vms=1, accelerator_variant="v4-8")


def test_quota_exceeded_retry():
    """VM creation fails with quota exceeded, retries successfully after clearing."""
    config = _make_scale_group_config("test")
    platform = FakePlatform(FakePlatformConfig(config=config, failure_mode=FailureMode.QUOTA_EXCEEDED))

    with pytest.raises(QuotaExhaustedError):
        platform.create_slice(_make_slice_config())

    platform.set_failure_mode(FailureMode.NONE)
    handle = platform.create_slice(_make_slice_config())

    platform.tick(Timestamp.now().epoch_ms())

    status = handle.describe()
    assert any(
        vm.status().state == CloudWorkerState.RUNNING for vm in status.workers
    ), f"Expected at least one VM in RUNNING state, got states: {[vm.status().state for vm in status.workers]}"


def test_vm_init_stuck():
    """VM boots but worker never initializes (stuck in INITIALIZING)."""
    config = _make_scale_group_config("stuck")
    platform = FakePlatform(FakePlatformConfig(config=config, init_delay_ms=999_999_999))
    handle = platform.create_slice(_make_slice_config("stuck"))

    platform.tick(Timestamp.now().epoch_ms())

    status = handle.describe()
    vm_states = [vm.status().state for vm in status.workers]
    assert all(
        vm.status().state != CloudWorkerState.RUNNING for vm in status.workers
    ), f"Expected no VMs in RUNNING state, got states: {vm_states}"


def test_vm_preempted():
    """VM reaches READY, then preemption terminates all VMs."""
    config = _make_scale_group_config("preempt")
    platform = FakePlatform(FakePlatformConfig(config=config))
    handle = platform.create_slice(_make_slice_config("preempt"))

    platform.tick(Timestamp.now().epoch_ms())

    status = handle.describe()
    assert any(vm.status().state == CloudWorkerState.RUNNING for vm in status.workers), (
        f"Expected at least one VM in RUNNING state before termination,"
        f" got: {[vm.status().state for vm in status.workers]}"
    )

    handle.terminate()

    status = handle.describe()
    assert all(
        vm.status().state == CloudWorkerState.TERMINATED for vm in status.workers
    ), f"Expected all VMs in TERMINATED state after preemption, got: {[vm.status().state for vm in status.workers]}"
