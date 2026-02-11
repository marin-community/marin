# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for bootstrap_slice_vms() — parallel VM discovery and bootstrap."""

from __future__ import annotations

import unittest.mock

import pytest

from iris.cluster.controller.autoscaler import bootstrap_slice_vms
from iris.cluster.platform.base import CloudSliceState, PlatformError, SliceStatus
from iris.cluster.platform.bootstrap import WorkerBootstrap
from iris.managed_thread import ThreadContainer
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


def _make_vm(vm_id: str, address: str = "10.0.0.1") -> FakeVmHandle:
    return FakeVmHandle(
        vm_id=vm_id,
        address=address,
        created_at_ms=Timestamp.now().epoch_ms(),
    )


def test_bootstrap_slice_vms_all_present():
    """All VMs present immediately — bootstraps all in parallel and returns logs."""
    vms = [_make_vm(f"vm-{i}", f"10.0.0.{i+1}") for i in range(3)]
    handle = FakeSliceHandle(
        slice_id="slice-1",
        scale_group="group-1",
        zone="us-central1-a",
        vms=vms,
    )
    bootstrap = WorkerBootstrap(_make_cluster_config())
    threads = ThreadContainer(name="test")

    try:
        logs = bootstrap_slice_vms(handle, bootstrap, threads)

        assert len(logs) == 3
        for vm in vms:
            assert vm.vm_id in logs
            assert vm._bootstrap_count == 1
    finally:
        threads.stop()


def test_bootstrap_slice_vms_streams_as_discovered():
    """VMs appear incrementally; early VMs start bootstrap before all are discovered."""
    vm0 = _make_vm("vm-0", "10.0.0.1")
    vm1 = _make_vm("vm-1", "10.0.0.2")

    # Start with only vm0 visible. On second describe() call, reveal vm1.
    handle = FakeSliceHandle(
        slice_id="slice-1",
        scale_group="group-1",
        zone="us-central1-a",
        vms=[vm0],
        labels={"iris-accelerator-variant": "v4-16"},  # 2 VMs expected
    )

    describe_count = 0
    original_describe = handle.describe

    def _gradual_describe():
        nonlocal describe_count
        describe_count += 1
        if describe_count == 1:
            return SliceStatus(state=CloudSliceState.CREATING, vm_count=1, vms=[vm0])
        handle._vms = [vm0, vm1]
        return original_describe()

    handle.describe = _gradual_describe

    bootstrap = WorkerBootstrap(_make_cluster_config())
    threads = ThreadContainer(name="test")

    try:
        logs = bootstrap_slice_vms(handle, bootstrap, threads, poll_interval=0.01)

        assert len(logs) == 2
        assert vm0._bootstrap_count == 1
        assert vm1._bootstrap_count == 1
    finally:
        threads.stop()


def test_bootstrap_slice_vms_no_label_bootstraps_immediately():
    """No accelerator variant label → bootstrap whatever VMs exist without waiting."""
    vms = [_make_vm("vm-0", "10.0.0.1")]
    handle = FakeSliceHandle(
        slice_id="slice-1",
        scale_group="group-1",
        zone="us-central1-a",
        vms=vms,
        labels={},  # No iris-accelerator-variant
    )
    bootstrap = WorkerBootstrap(_make_cluster_config())
    threads = ThreadContainer(name="test")

    try:
        logs = bootstrap_slice_vms(handle, bootstrap, threads)

        assert len(logs) == 1
        assert vms[0]._bootstrap_count == 1
    finally:
        threads.stop()


def test_bootstrap_slice_vms_timeout_raises():
    """Expected VMs never appear — raises PlatformError after timeout."""
    vms = [_make_vm("vm-0", "10.0.0.1")]
    handle = FakeSliceHandle(
        slice_id="slice-1",
        scale_group="group-1",
        zone="us-central1-a",
        vms=vms,
        labels={"iris-accelerator-variant": "v4-16"},  # 2 VMs expected, only 1 present
    )
    bootstrap = WorkerBootstrap(_make_cluster_config())
    threads = ThreadContainer(name="test")

    try:
        with unittest.mock.patch("iris.cluster.controller.autoscaler.time") as mock_time:
            mock_time.time.side_effect = [0.0, 601.0]
            mock_time.sleep.return_value = None

            with pytest.raises(PlatformError, match="only 1/2 VMs appeared"):
                bootstrap_slice_vms(handle, bootstrap, threads, timeout=600.0)
    finally:
        threads.stop()


def test_bootstrap_slice_vms_single_vm_failure_raises():
    """One VM fails bootstrap — error propagates after all threads complete."""
    vm_ok = _make_vm("vm-ok", "10.0.0.1")
    vm_bad = _make_vm("vm-bad", "10.0.0.2")
    vm_bad._wait_for_connection_succeeds = False  # This VM will fail

    handle = FakeSliceHandle(
        slice_id="slice-1",
        scale_group="group-1",
        zone="us-central1-a",
        vms=[vm_ok, vm_bad],
    )
    bootstrap = WorkerBootstrap(_make_cluster_config())
    threads = ThreadContainer(name="test")

    try:
        with pytest.raises(PlatformError, match="Bootstrap failed for 1 VMs"):
            bootstrap_slice_vms(handle, bootstrap, threads)

        # The successful VM should still have been bootstrapped
        assert vm_ok._bootstrap_count == 1
    finally:
        threads.stop()
