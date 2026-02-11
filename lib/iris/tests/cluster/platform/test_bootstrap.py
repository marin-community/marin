# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for WorkerBootstrap, focusing on address validation during bootstrap."""

from __future__ import annotations

import pytest

from iris.cluster.platform.base import PlatformError
from iris.cluster.platform.bootstrap import WorkerBootstrap
from iris.rpc import config_pb2
from iris.time_utils import Timestamp
from tests.cluster.platform.fakes import FakeSliceHandle, FakeVmHandle


def _make_cluster_config() -> config_pb2.IrisClusterConfig:
    return config_pb2.IrisClusterConfig(
        defaults=config_pb2.DefaultsConfig(
            bootstrap=config_pb2.BootstrapConfig(
                docker_image="gcr.io/test/iris-worker:latest",
                worker_port=10001,
                cache_dir="/var/cache/iris",
            ),
        ),
    )


def _make_slice(addresses: list[str]) -> FakeSliceHandle:
    """Build a FakeSliceHandle with VMs at the given addresses."""
    vms = [
        FakeVmHandle(
            vm_id=f"slice-1-vm-{i}",
            address=addr,
            created_at_ms=Timestamp.now().epoch_ms(),
        )
        for i, addr in enumerate(addresses)
    ]
    return FakeSliceHandle(
        slice_id="slice-1",
        scale_group="group-1",
        zone="us-central1-a",
        vms=vms,
    )


def test_bootstrap_slice_raises_on_empty_address():
    """bootstrap_slice() should raise PlatformError when a VM has no internal address."""
    config = _make_cluster_config()
    bootstrap = WorkerBootstrap(config)
    handle = _make_slice(["10.0.0.1", "", "10.0.0.3"])

    with pytest.raises(PlatformError, match="has no internal address"):
        bootstrap.bootstrap_slice(handle)


def test_bootstrap_slice_succeeds_with_valid_addresses():
    """bootstrap_slice() should call bootstrap on each VM and return logs."""
    config = _make_cluster_config()
    bootstrap = WorkerBootstrap(config)
    handle = _make_slice(["10.0.0.1", "10.0.0.2"])

    logs = bootstrap.bootstrap_slice(handle)

    for vm in handle.list_vms():
        assert vm._bootstrap_count == 1
    assert len(logs) == 2
    for vm in handle.list_vms():
        assert vm.vm_id in logs


def test_bootstrap_slice_raises_on_connection_timeout():
    """bootstrap_slice() should raise PlatformError when wait_for_connection times out."""
    config = _make_cluster_config()
    bootstrap = WorkerBootstrap(config)

    # Create a handle with one VM that times out on wait_for_connection
    vm = FakeVmHandle(
        vm_id="slice-1-vm-0",
        address="10.0.0.1",
        created_at_ms=Timestamp.now().epoch_ms(),
    )
    vm._wait_for_connection_succeeds = False  # Simulate timeout
    handle = FakeSliceHandle(
        slice_id="slice-1",
        scale_group="group-1",
        zone="us-central1-a",
        vms=[vm],
        # Add label so VM count check passes and we proceed to wait_for_connection
        labels={"iris-accelerator-variant": "v4-8"},  # v4-8 has 1 VM
    )

    with pytest.raises(PlatformError, match="failed to become reachable"):
        bootstrap.bootstrap_slice(handle)


def test_bootstrap_slice_waits_for_all_vms():
    """bootstrap_slice() should wait for all expected VMs to appear before bootstrapping."""
    config = _make_cluster_config()
    bootstrap = WorkerBootstrap(config)

    # Create a slice that will gradually reveal VMs
    all_vms = [
        FakeVmHandle(
            vm_id=f"slice-1-vm-{i}",
            address=f"10.0.0.{i+1}",
            created_at_ms=Timestamp.now().epoch_ms(),
        )
        for i in range(4)  # v4-16 has 2 VMs, but we'll test with a custom scenario
    ]

    # Create handle with labels indicating v4-16 topology (2 VMs expected)
    handle = FakeSliceHandle(
        slice_id="slice-1",
        scale_group="group-1",
        zone="us-central1-a",
        vms=all_vms[:1],  # Start with only 1 VM visible
        labels={"iris-accelerator-variant": "v4-16"},  # Expect 2 VMs
    )

    # Store a reference to track VM list updates
    vm_list_calls = []

    def mock_list_vms():
        vm_list_calls.append(len(handle._vms))
        # Simulate gradual VM appearance: first call returns 1, second returns 2
        if len(vm_list_calls) == 1:
            return all_vms[:1]
        else:
            # Update internal state to show both VMs
            handle._vms = all_vms[:2]
            return all_vms[:2]

    handle.list_vms = mock_list_vms

    # Bootstrap should succeed after waiting
    logs = bootstrap.bootstrap_slice(handle)

    # Should have called list_vms multiple times and bootstrapped all VMs
    assert len(vm_list_calls) >= 2
    assert len(logs) == 2


def test_bootstrap_slice_raises_on_partial_vm_count_timeout():
    """bootstrap_slice() should raise PlatformError if not all expected VMs appear."""
    import unittest.mock

    config = _make_cluster_config()
    bootstrap = WorkerBootstrap(config)

    # Create a slice that never reaches expected VM count
    vms = [
        FakeVmHandle(
            vm_id="slice-1-vm-0",
            address="10.0.0.1",
            created_at_ms=Timestamp.now().epoch_ms(),
        )
    ]

    handle = FakeSliceHandle(
        slice_id="slice-1",
        scale_group="group-1",
        zone="us-central1-a",
        vms=vms,
        labels={"iris-accelerator-variant": "v4-16"},  # Expect 2 VMs, but only 1 present
    )

    # Mock time.time to advance quickly and avoid waiting 60 seconds
    with unittest.mock.patch("iris.cluster.platform.bootstrap.time") as mock_time:
        # Make time.time() advance by 65 seconds on each call (exceeds 60s timeout)
        mock_time.time.side_effect = [0.0, 65.0]
        mock_time.sleep.return_value = None  # No-op sleep

        with pytest.raises(PlatformError, match="has only 1/2 VMs ready"):
            bootstrap.bootstrap_slice(handle)
