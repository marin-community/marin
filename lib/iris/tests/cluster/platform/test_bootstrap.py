# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for WorkerBootstrap.bootstrap_vm(), focusing on per-VM bootstrap behavior."""

from __future__ import annotations

import pytest

from iris.cluster.platform.base import PlatformError
from iris.cluster.platform.bootstrap import WorkerBootstrap
from iris.rpc import config_pb2
from iris.time_utils import Timestamp
from tests.cluster.platform.fakes import FakeVmHandle


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


def _make_vm(address: str = "10.0.0.1", vm_id: str = "vm-0") -> FakeVmHandle:
    return FakeVmHandle(
        vm_id=vm_id,
        address=address,
        created_at_ms=Timestamp.now().epoch_ms(),
    )


def test_bootstrap_vm_succeeds():
    """bootstrap_vm() should call bootstrap on the VM and return its log."""
    config = _make_cluster_config()
    bootstrap = WorkerBootstrap(config)
    vm = _make_vm("10.0.0.1")

    log = bootstrap.bootstrap_vm(vm)

    assert vm._bootstrap_count == 1
    assert log == vm.bootstrap_log


def test_bootstrap_vm_raises_on_empty_address():
    """bootstrap_vm() should raise PlatformError when a VM has no internal address."""
    config = _make_cluster_config()
    bootstrap = WorkerBootstrap(config)
    vm = _make_vm(address="")

    with pytest.raises(PlatformError, match="has no internal address"):
        bootstrap.bootstrap_vm(vm)


def test_bootstrap_vm_raises_on_connection_timeout():
    """bootstrap_vm() should raise PlatformError when wait_for_connection times out."""
    config = _make_cluster_config()
    bootstrap = WorkerBootstrap(config)
    vm = _make_vm("10.0.0.1")
    vm._wait_for_connection_succeeds = False

    with pytest.raises(PlatformError, match="failed to become reachable"):
        bootstrap.bootstrap_vm(vm)
