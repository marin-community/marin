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

"""Tests for LocalPlatform implementation.

Tests verify in-memory VM/slice management, label filtering, and that
shutdown() stops the ThreadContainer.
"""

from __future__ import annotations

import threading

import pytest

from iris.cluster.platform.base import CloudSliceState, CloudVmState
from iris.cluster.platform.local import (
    LocalPlatform,
    LocalSliceHandle,
    _LocalStandaloneVmHandle,
)
from iris.managed_thread import ThreadContainer
from iris.rpc import config_pb2
from iris.time_utils import Duration

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def local_platform() -> LocalPlatform:
    p = LocalPlatform("test")
    yield p
    p.shutdown()


# =============================================================================
# create_vm Tests
# =============================================================================


def test_create_vm_returns_standalone_handle(local_platform: LocalPlatform):
    """create_vm() returns a _LocalStandaloneVmHandle with correct properties."""
    cfg = config_pb2.VmConfig(name="controller")
    cfg.labels["role"] = "controller"
    cfg.metadata["addr"] = "http://localhost:10000"

    handle = local_platform.create_vm(cfg)

    assert isinstance(handle, _LocalStandaloneVmHandle)
    assert handle.vm_id == "controller"
    assert handle.internal_address == "localhost"
    assert handle.external_address is None
    assert handle.labels["role"] == "controller"
    assert handle.metadata["addr"] == "http://localhost:10000"


def test_create_vm_wait_for_connection_is_immediate(local_platform: LocalPlatform):
    """wait_for_connection() returns True immediately for local VMs."""
    cfg = config_pb2.VmConfig(name="controller")
    handle = local_platform.create_vm(cfg)

    assert handle.wait_for_connection(Duration.from_seconds(1))


def test_create_vm_run_command_executes_locally(local_platform: LocalPlatform):
    """run_command() executes bash commands locally."""
    cfg = config_pb2.VmConfig(name="controller")
    handle = local_platform.create_vm(cfg)

    result = handle.run_command("echo hello")
    assert result.returncode == 0
    assert "hello" in result.stdout


def test_create_vm_terminate_marks_terminated(local_platform: LocalPlatform):
    """terminate() marks the VM as terminated."""
    cfg = config_pb2.VmConfig(name="controller")
    handle = local_platform.create_vm(cfg)

    assert handle.status().state == CloudVmState.RUNNING
    handle.terminate()
    assert handle.status().state == CloudVmState.TERMINATED


def test_create_vm_set_labels_updates_in_memory(local_platform: LocalPlatform):
    """set_labels() updates in-memory labels."""
    cfg = config_pb2.VmConfig(name="controller")
    handle = local_platform.create_vm(cfg)

    handle.set_labels({"iris-controller": "true"})
    assert handle.labels["iris-controller"] == "true"


def test_create_vm_set_metadata_updates_in_memory(local_platform: LocalPlatform):
    """set_metadata() updates in-memory metadata."""
    cfg = config_pb2.VmConfig(name="controller")
    handle = local_platform.create_vm(cfg)

    handle.set_metadata({"addr": "http://localhost:10000"})
    assert handle.metadata["addr"] == "http://localhost:10000"


# =============================================================================
# create_slice Tests
# =============================================================================


def test_create_slice_returns_local_slice_handle(local_platform: LocalPlatform):
    """create_slice() returns a LocalSliceHandle with correct properties."""
    cfg = config_pb2.SliceConfig(name_prefix="test-group", slice_size=2)
    cfg.labels["test-scale-group"] = "my-group"

    handle = local_platform.create_slice(cfg)

    assert isinstance(handle, LocalSliceHandle)
    assert handle.slice_id.startswith("test-group-")
    assert handle.zone == "local"
    assert handle.scale_group == "my-group"


def test_create_slice_creates_correct_number_of_vms(local_platform: LocalPlatform):
    """create_slice() creates the requested number of worker VMs."""
    cfg = config_pb2.SliceConfig(name_prefix="test-group", slice_size=3)
    cfg.labels["test-scale-group"] = "my-group"

    handle = local_platform.create_slice(cfg)
    vms = handle.list_vms()

    assert len(vms) == 3
    for i, vm in enumerate(vms):
        assert f"worker-{i}" in vm.vm_id


def test_create_slice_default_size_is_one(local_platform: LocalPlatform):
    """create_slice() defaults to 1 VM when slice_size is 0."""
    cfg = config_pb2.SliceConfig(name_prefix="test-group")
    handle = local_platform.create_slice(cfg)

    assert len(handle.list_vms()) == 1


def test_slice_terminate_changes_status(local_platform: LocalPlatform):
    """Terminating a slice changes its status to DELETING."""
    cfg = config_pb2.SliceConfig(name_prefix="test-group", slice_size=1)
    handle = local_platform.create_slice(cfg)

    assert handle.status().state == CloudSliceState.READY
    handle.terminate()
    assert handle.status().state == CloudSliceState.DELETING


# =============================================================================
# list_slices / list_vms Tests
# =============================================================================


def test_list_slices_returns_created_slices(local_platform: LocalPlatform):
    """list_slices() returns slices created by this platform."""
    cfg = config_pb2.SliceConfig(name_prefix="test-group", slice_size=1)
    cfg.labels["iris-scale-group"] = "group-a"

    local_platform.create_slice(cfg)

    slices = local_platform.list_slices(zones=["local"])
    assert len(slices) == 1


def test_list_slices_filters_by_labels(local_platform: LocalPlatform):
    """list_slices() filters by exact label match."""
    cfg1 = config_pb2.SliceConfig(name_prefix="test-a", slice_size=1)
    cfg1.labels["test-scale-group"] = "group-a"

    cfg2 = config_pb2.SliceConfig(name_prefix="test-b", slice_size=1)
    cfg2.labels["test-scale-group"] = "group-b"

    local_platform.create_slice(cfg1)
    local_platform.create_slice(cfg2)

    slices = local_platform.list_slices(zones=["local"], labels={"test-scale-group": "group-a"})
    assert len(slices) == 1
    assert slices[0].scale_group == "group-a"


def test_list_vms_returns_standalone_vms(local_platform: LocalPlatform):
    """list_vms() returns VMs created via create_vm()."""
    cfg = config_pb2.VmConfig(name="controller")
    cfg.labels["iris-controller"] = "true"

    local_platform.create_vm(cfg)

    vms = local_platform.list_vms(zones=["local"])
    assert len(vms) == 1
    assert vms[0].vm_id == "controller"


def test_list_slices_ignores_zones_parameter(local_platform: LocalPlatform):
    """list_slices() returns slices regardless of the zones parameter value.

    LocalPlatform ignores zones — all resources live in zone="local".
    """
    cfg = config_pb2.SliceConfig(name_prefix="test-group", slice_size=1)
    cfg.labels["test-scale-group"] = "group-a"

    local_platform.create_slice(cfg)

    assert len(local_platform.list_slices(zones=["us-central1-a"])) == 1
    assert len(local_platform.list_slices(zones=["nonexistent"])) == 1
    assert len(local_platform.list_slices(zones=[])) == 1


def test_list_vms_ignores_zones_parameter(local_platform: LocalPlatform):
    """list_vms() returns VMs regardless of the zones parameter value."""
    cfg = config_pb2.VmConfig(name="controller")
    local_platform.create_vm(cfg)

    assert len(local_platform.list_vms(zones=["us-central1-a"])) == 1
    assert len(local_platform.list_vms(zones=[])) == 1


def test_list_vms_filters_by_labels(local_platform: LocalPlatform):
    """list_vms() filters by exact label match."""
    cfg1 = config_pb2.VmConfig(name="ctrl")
    cfg1.labels["role"] = "controller"

    cfg2 = config_pb2.VmConfig(name="other")
    cfg2.labels["role"] = "worker"

    local_platform.create_vm(cfg1)
    local_platform.create_vm(cfg2)

    vms = local_platform.list_vms(zones=["local"], labels={"role": "controller"})
    assert len(vms) == 1
    assert vms[0].vm_id == "ctrl"


# =============================================================================
# Tunnel Tests
# =============================================================================


def test_tunnel_returns_address_directly(local_platform: LocalPlatform):
    """tunnel() returns the address as-is (no tunneling for local)."""
    with local_platform.tunnel("http://localhost:10000") as url:
        assert url == "http://localhost:10000"


# =============================================================================
# shutdown() Tests
# =============================================================================


def test_shutdown_stops_thread_container():
    """shutdown() stops the ThreadContainer and clears internal state."""
    threads = ThreadContainer(name="test-threads")
    platform = LocalPlatform("test", threads=threads)

    # Spawn a thread that runs until stopped
    started = threading.Event()

    def worker(stop_event: threading.Event) -> None:
        started.set()
        stop_event.wait()

    threads.spawn(worker, name="test-worker")
    started.wait(timeout=5.0)
    assert threads.is_alive

    # Create some state
    cfg_vm = config_pb2.VmConfig(name="controller")
    platform.create_vm(cfg_vm)

    cfg_slice = config_pb2.SliceConfig(name_prefix="test", slice_size=1)
    platform.create_slice(cfg_slice)

    assert len(platform.list_vms(zones=["local"])) == 1
    assert len(platform.list_slices(zones=["local"])) == 1

    # shutdown() should stop threads and clear state
    platform.shutdown()

    assert not threads.is_alive
    assert len(platform.list_vms(zones=["local"])) == 0
    assert len(platform.list_slices(zones=["local"])) == 0


def test_shutdown_clears_slices_and_vms():
    """shutdown() clears the internal slice and VM registries."""
    platform = LocalPlatform("test")

    cfg_vm = config_pb2.VmConfig(name="ctrl")
    platform.create_vm(cfg_vm)

    cfg_slice = config_pb2.SliceConfig(name_prefix="group", slice_size=1)
    platform.create_slice(cfg_slice)

    platform.shutdown()

    assert platform.list_vms(zones=[]) == []
    assert platform.list_slices(zones=[]) == []
