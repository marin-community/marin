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

"""Tests for ManualPlatform implementation.

Tests verify host pool management, slice/VM lifecycle, and label filtering
without requiring real SSH infrastructure.
"""

from __future__ import annotations

import pytest

from iris.cluster.platform.base import CloudSliceState
from iris.cluster.platform.manual import (
    ManualPlatform,
    ManualSliceHandle,
    ManualStandaloneVmHandle,
)
from iris.rpc import config_pb2

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def manual_platform() -> ManualPlatform:
    return ManualPlatform(
        label_prefix="iris",
        hosts=["10.0.0.1", "10.0.0.2", "10.0.0.3"],
    )


# =============================================================================
# create_vm Tests
# =============================================================================


def test_create_vm_allocates_specific_host(manual_platform: ManualPlatform):
    """create_vm() with a specific host removes it from the pool."""
    cfg = config_pb2.VmConfig(name="controller")
    cfg.manual.host = "10.0.0.1"
    cfg.labels["iris-controller"] = "true"

    handle = manual_platform.create_vm(cfg)

    assert isinstance(handle, ManualStandaloneVmHandle)
    assert handle.vm_id == "controller"
    assert handle.internal_address == "10.0.0.1"
    assert manual_platform.available_host_count == 2


def test_create_vm_pops_from_pool_when_no_host_specified(manual_platform: ManualPlatform):
    """create_vm() without specific host draws from the pool."""
    cfg = config_pb2.VmConfig(name="controller")
    cfg.manual.CopyFrom(config_pb2.ManualVmConfig())

    handle = manual_platform.create_vm(cfg)

    assert handle.internal_address in {"10.0.0.1", "10.0.0.2", "10.0.0.3"}
    assert manual_platform.available_host_count == 2


def test_create_vm_raises_when_pool_empty():
    """create_vm() raises when no hosts are available."""
    platform = ManualPlatform(label_prefix="iris", hosts=[])
    cfg = config_pb2.VmConfig(name="controller")
    cfg.manual.CopyFrom(config_pb2.ManualVmConfig())

    with pytest.raises(RuntimeError, match="No hosts available"):
        platform.create_vm(cfg)


def test_create_vm_terminate_returns_host_to_pool(manual_platform: ManualPlatform):
    """Terminating a standalone VM returns its host to the pool."""
    cfg = config_pb2.VmConfig(name="controller")
    cfg.manual.host = "10.0.0.1"

    handle = manual_platform.create_vm(cfg)
    assert manual_platform.available_host_count == 2

    handle.terminate()
    assert manual_platform.available_host_count == 3


# =============================================================================
# create_slice Tests
# =============================================================================


def test_create_slice_with_explicit_hosts(manual_platform: ManualPlatform):
    """create_slice() with explicit hosts allocates those specific hosts."""
    cfg = config_pb2.SliceConfig(name_prefix="iris-group")
    cfg.manual.hosts.extend(["10.0.0.1", "10.0.0.2"])
    cfg.labels["iris-scale-group"] = "my-group"

    handle = manual_platform.create_slice(cfg)

    assert isinstance(handle, ManualSliceHandle)
    assert handle.slice_id.startswith("iris-group-")
    assert handle.zone == "manual"
    assert handle.scale_group == "my-group"
    assert manual_platform.available_host_count == 1

    vms = handle.list_vms()
    assert len(vms) == 2
    addresses = {vm.internal_address for vm in vms}
    assert addresses == {"10.0.0.1", "10.0.0.2"}


def test_create_slice_draws_from_pool(manual_platform: ManualPlatform):
    """create_slice() without explicit hosts draws from the pool."""
    cfg = config_pb2.SliceConfig(name_prefix="iris-group", slice_size=2)
    cfg.manual.CopyFrom(config_pb2.ManualSliceConfig())
    cfg.labels["iris-scale-group"] = "my-group"

    handle = manual_platform.create_slice(cfg)

    assert manual_platform.available_host_count == 1
    vms = handle.list_vms()
    assert len(vms) == 2


def test_create_slice_raises_when_insufficient_hosts():
    """create_slice() raises when not enough hosts are available."""
    platform = ManualPlatform(label_prefix="iris", hosts=["10.0.0.1"])
    cfg = config_pb2.SliceConfig(name_prefix="iris-group", slice_size=3)
    cfg.manual.CopyFrom(config_pb2.ManualSliceConfig())

    with pytest.raises(RuntimeError, match="Need 3 hosts but only 1 available"):
        platform.create_slice(cfg)


def test_create_slice_terminate_returns_hosts(manual_platform: ManualPlatform):
    """Terminating a slice returns its hosts to the pool."""
    cfg = config_pb2.SliceConfig(name_prefix="iris-group", slice_size=2)
    cfg.manual.CopyFrom(config_pb2.ManualSliceConfig())

    handle = manual_platform.create_slice(cfg)
    assert manual_platform.available_host_count == 1

    handle.terminate()
    assert manual_platform.available_host_count == 3


def test_slice_status_reflects_terminated_state(manual_platform: ManualPlatform):
    """Slice status changes after termination."""
    cfg = config_pb2.SliceConfig(name_prefix="iris-group", slice_size=1)
    cfg.manual.CopyFrom(config_pb2.ManualSliceConfig())

    handle = manual_platform.create_slice(cfg)
    assert handle.status().state == CloudSliceState.READY

    handle.terminate()
    assert handle.status().state == CloudSliceState.DELETING


# =============================================================================
# list_slices / list_vms Tests
# =============================================================================


def test_list_slices_returns_created_slices(manual_platform: ManualPlatform):
    """list_slices() returns slices created by this platform."""
    cfg = config_pb2.SliceConfig(name_prefix="iris-group", slice_size=1)
    cfg.manual.CopyFrom(config_pb2.ManualSliceConfig())
    cfg.labels["iris-scale-group"] = "group-a"

    manual_platform.create_slice(cfg)

    slices = manual_platform.list_slices(zones=["manual"])
    assert len(slices) == 1
    assert slices[0].scale_group == "group-a"


def test_list_slices_filters_by_labels(manual_platform: ManualPlatform):
    """list_slices() filters by label match."""
    cfg1 = config_pb2.SliceConfig(name_prefix="iris-a", slice_size=1)
    cfg1.manual.CopyFrom(config_pb2.ManualSliceConfig())
    cfg1.labels["iris-scale-group"] = "group-a"

    cfg2 = config_pb2.SliceConfig(name_prefix="iris-b", slice_size=1)
    cfg2.manual.CopyFrom(config_pb2.ManualSliceConfig())
    cfg2.labels["iris-scale-group"] = "group-b"

    manual_platform.create_slice(cfg1)
    manual_platform.create_slice(cfg2)

    slices = manual_platform.list_slices(zones=["manual"], labels={"iris-scale-group": "group-a"})
    assert len(slices) == 1
    assert slices[0].scale_group == "group-a"


def test_list_vms_returns_standalone_vms(manual_platform: ManualPlatform):
    """list_vms() returns VMs created via create_vm()."""
    cfg = config_pb2.VmConfig(name="controller")
    cfg.manual.host = "10.0.0.1"
    cfg.labels["iris-controller"] = "true"

    manual_platform.create_vm(cfg)

    vms = manual_platform.list_vms(zones=["manual"])
    assert len(vms) == 1
    assert vms[0].vm_id == "controller"


def test_list_slices_ignores_zones_parameter(manual_platform: ManualPlatform):
    """list_slices() returns slices regardless of the zones parameter value.

    ManualPlatform ignores zones — all slices live in zone="manual", and callers
    may pass arbitrary zone values without affecting results.
    """
    cfg = config_pb2.SliceConfig(name_prefix="iris-a", slice_size=1)
    cfg.manual.CopyFrom(config_pb2.ManualSliceConfig())
    cfg.labels["iris-scale-group"] = "group-a"

    manual_platform.create_slice(cfg)

    assert len(manual_platform.list_slices(zones=["us-central1-a"])) == 1
    assert len(manual_platform.list_slices(zones=["nonexistent-zone"])) == 1
    assert len(manual_platform.list_slices(zones=[])) == 1


def test_list_vms_ignores_zones_parameter(manual_platform: ManualPlatform):
    """list_vms() returns VMs regardless of the zones parameter value."""
    cfg = config_pb2.VmConfig(name="ctrl")
    cfg.manual.host = "10.0.0.1"

    manual_platform.create_vm(cfg)

    assert len(manual_platform.list_vms(zones=["us-central1-a"])) == 1
    assert len(manual_platform.list_vms(zones=[])) == 1


def test_list_vms_filters_by_labels(manual_platform: ManualPlatform):
    """list_vms() filters by label match."""
    cfg1 = config_pb2.VmConfig(name="ctrl-1")
    cfg1.manual.host = "10.0.0.1"
    cfg1.labels["role"] = "controller"

    cfg2 = config_pb2.VmConfig(name="other")
    cfg2.manual.host = "10.0.0.2"
    cfg2.labels["role"] = "worker"

    manual_platform.create_vm(cfg1)
    manual_platform.create_vm(cfg2)

    vms = manual_platform.list_vms(zones=["manual"], labels={"role": "controller"})
    assert len(vms) == 1
    assert vms[0].vm_id == "ctrl-1"


# =============================================================================
# StandaloneVmHandle label/metadata Tests
# =============================================================================


def test_standalone_handle_set_labels_updates_in_memory(manual_platform: ManualPlatform):
    """set_labels() updates the in-memory label store."""
    cfg = config_pb2.VmConfig(name="controller")
    cfg.manual.host = "10.0.0.1"

    handle = manual_platform.create_vm(cfg)
    handle.set_labels({"iris-controller": "true"})

    assert handle.labels["iris-controller"] == "true"

    # Labels should be filterable via list_vms
    vms = manual_platform.list_vms(zones=["manual"], labels={"iris-controller": "true"})
    assert len(vms) == 1


def test_standalone_handle_set_metadata(manual_platform: ManualPlatform):
    """set_metadata() updates the in-memory metadata store."""
    cfg = config_pb2.VmConfig(name="controller")
    cfg.manual.host = "10.0.0.1"

    handle = manual_platform.create_vm(cfg)
    handle.set_metadata({"iris-addr": "http://10.0.0.1:10000"})

    assert handle.metadata["iris-addr"] == "http://10.0.0.1:10000"


# =============================================================================
# Tunnel and discovery Tests
# =============================================================================


def test_tunnel_returns_address_directly(manual_platform: ManualPlatform):
    """tunnel() returns the address as-is (no SSH tunnel needed)."""
    with manual_platform.tunnel("http://10.0.0.1:10000") as url:
        assert url == "http://10.0.0.1:10000"


def test_shutdown_is_noop(manual_platform: ManualPlatform):
    """shutdown() is a no-op for ManualPlatform."""
    manual_platform.shutdown()


# =============================================================================
# Host exclusivity Tests
# =============================================================================


def test_create_vm_rejects_already_allocated_host(manual_platform: ManualPlatform):
    """create_vm() raises when the requested host is already allocated."""
    cfg1 = config_pb2.VmConfig(name="ctrl-1")
    cfg1.manual.host = "10.0.0.1"
    manual_platform.create_vm(cfg1)

    cfg2 = config_pb2.VmConfig(name="ctrl-2")
    cfg2.manual.host = "10.0.0.1"
    with pytest.raises(RuntimeError, match="already allocated"):
        manual_platform.create_vm(cfg2)


def test_create_slice_rejects_already_allocated_host(manual_platform: ManualPlatform):
    """create_slice() raises when an explicit host is already allocated to a VM."""
    vm_cfg = config_pb2.VmConfig(name="ctrl")
    vm_cfg.manual.host = "10.0.0.1"
    manual_platform.create_vm(vm_cfg)

    slice_cfg = config_pb2.SliceConfig(name_prefix="group")
    slice_cfg.manual.hosts.extend(["10.0.0.1", "10.0.0.2"])
    with pytest.raises(RuntimeError, match="already allocated"):
        manual_platform.create_slice(slice_cfg)


def test_create_slice_rejects_host_allocated_by_another_slice(manual_platform: ManualPlatform):
    """create_slice() raises when an explicit host is already in another slice."""
    cfg1 = config_pb2.SliceConfig(name_prefix="group-a")
    cfg1.manual.hosts.extend(["10.0.0.1"])
    manual_platform.create_slice(cfg1)

    cfg2 = config_pb2.SliceConfig(name_prefix="group-b")
    cfg2.manual.hosts.extend(["10.0.0.1"])
    with pytest.raises(RuntimeError, match="already allocated"):
        manual_platform.create_slice(cfg2)


def test_terminated_host_can_be_reallocated(manual_platform: ManualPlatform):
    """After termination, a host can be allocated again."""
    cfg = config_pb2.VmConfig(name="ctrl")
    cfg.manual.host = "10.0.0.1"
    handle = manual_platform.create_vm(cfg)
    handle.terminate()

    cfg2 = config_pb2.VmConfig(name="ctrl-2")
    cfg2.manual.host = "10.0.0.1"
    handle2 = manual_platform.create_vm(cfg2)
    assert handle2.internal_address == "10.0.0.1"
