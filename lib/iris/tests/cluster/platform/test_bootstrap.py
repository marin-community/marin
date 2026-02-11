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
