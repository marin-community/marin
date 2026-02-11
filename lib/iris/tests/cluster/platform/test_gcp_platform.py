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

"""Tests for GcpPlatform implementation.

Tests use mocked subprocess.run calls to verify gcloud CLI interactions
without requiring real GCP infrastructure.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from iris.cluster.platform.base import (
    CloudSliceState,
    CloudVmState,
    PlatformError,
    QuotaExhaustedError,
)
from iris.cluster.controller.scaling_group import ScalingGroup
from iris.cluster.platform.gcp import (
    GcpPlatform,
    GcpSliceHandle,
    GcpStandaloneVmHandle,
    GcpVmHandle,
    _build_label_filter,
    _classify_gcloud_error,
    _extract_node_name,
    _format_labels,
    _validate_slice_config,
)
from iris.rpc import config_pb2
from tests.cluster.platform.fake_gcp import FakeGcloud

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def gcp_config() -> config_pb2.GcpPlatformConfig:
    return config_pb2.GcpPlatformConfig(project_id="test-project")


@pytest.fixture
def platform(gcp_config: config_pb2.GcpPlatformConfig) -> GcpPlatform:
    return GcpPlatform(gcp_config, label_prefix="iris")


@pytest.fixture
def vm_config() -> config_pb2.VmConfig:
    cfg = config_pb2.VmConfig(name="test-controller")
    cfg.gcp.zone = "us-central1-a"
    cfg.gcp.machine_type = "n2-standard-4"
    cfg.gcp.boot_disk_size_gb = 50
    cfg.labels["iris-managed"] = "true"
    cfg.metadata["iris-controller-iris"] = "true"
    return cfg


@pytest.fixture
def slice_config() -> config_pb2.SliceConfig:
    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu-group",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-16",
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"
    cfg.preemptible = True
    cfg.labels["iris-managed"] = "true"
    cfg.labels["iris-scale-group"] = "tpu-group"
    return cfg


# =============================================================================
# Helper Function Tests
# =============================================================================


def test_format_labels():
    labels = {"key1": "val1", "key2": "val2"}
    result = _format_labels(labels)
    assert "key1=val1" in result
    assert "key2=val2" in result
    assert "," in result


def test_build_label_filter():
    labels = {"iris-managed": "true", "iris-scale-group": "tpu"}
    result = _build_label_filter(labels)
    assert "labels.iris-managed=true" in result
    assert "labels.iris-scale-group=tpu" in result
    assert " AND " in result


def test_extract_node_name_from_full_path():
    assert _extract_node_name("projects/p/locations/z/nodes/my-tpu") == "my-tpu"


def test_extract_node_name_from_simple_name():
    assert _extract_node_name("my-tpu") == "my-tpu"


def test_classify_gcloud_error_quota():
    err = _classify_gcloud_error("RESOURCE_EXHAUSTED: quota limit reached")
    assert isinstance(err, QuotaExhaustedError)


def test_classify_gcloud_error_generic():
    err = _classify_gcloud_error("Some other error")
    assert isinstance(err, PlatformError)
    assert not isinstance(err, QuotaExhaustedError)


# =============================================================================
# GcpPlatform.create_vm Tests
# =============================================================================


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_create_vm_calls_gcloud_correctly(mock_run: MagicMock, platform: GcpPlatform, vm_config: config_pb2.VmConfig):
    """create_vm() issues the correct gcloud compute instances create command."""
    # First call: create, second call: describe for IP
    create_output = json.dumps(
        [{"networkInterfaces": [{"networkIP": "10.128.0.5", "accessConfigs": [{"natIP": "34.1.2.3"}]}]}]
    )
    mock_run.return_value = MagicMock(returncode=0, stdout=create_output, stderr="")

    # Mock the describe call for _get_vm_ips
    describe_output = json.dumps(
        {"networkInterfaces": [{"networkIP": "10.128.0.5", "accessConfigs": [{"natIP": "34.1.2.3"}]}]}
    )

    def run_side_effect(cmd, **kwargs):
        if "describe" in cmd:
            return MagicMock(returncode=0, stdout=describe_output, stderr="")
        return MagicMock(returncode=0, stdout=create_output, stderr="")

    mock_run.side_effect = run_side_effect

    handle = platform.create_vm(vm_config)

    assert isinstance(handle, GcpStandaloneVmHandle)
    assert handle.vm_id == "test-controller"
    assert handle.internal_address == "10.128.0.5"
    assert handle.external_address == "34.1.2.3"

    # Verify gcloud create was called
    create_call = mock_run.call_args_list[0]
    cmd = create_call[0][0]
    assert "gcloud" in cmd
    assert "instances" in cmd
    assert "create" in cmd
    assert "test-controller" in cmd
    assert "--project=test-project" in cmd
    assert "--zone=us-central1-a" in cmd


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_create_vm_handles_already_exists(mock_run: MagicMock, platform: GcpPlatform, vm_config: config_pb2.VmConfig):
    """create_vm() recovers when the VM already exists."""
    describe_output = json.dumps({"networkInterfaces": [{"networkIP": "10.0.0.1"}]})

    def run_side_effect(cmd, **kwargs):
        if "create" in cmd:
            return MagicMock(returncode=1, stdout="", stderr="The resource 'test-controller' already exists")
        return MagicMock(returncode=0, stdout=describe_output, stderr="")

    mock_run.side_effect = run_side_effect

    handle = platform.create_vm(vm_config)
    assert handle.internal_address == "10.0.0.1"


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_create_vm_raises_on_quota_error(mock_run: MagicMock, platform: GcpPlatform, vm_config: config_pb2.VmConfig):
    """create_vm() raises QuotaExhaustedError on quota failures."""
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="RESOURCE_EXHAUSTED: Quota limit reached")

    with pytest.raises(QuotaExhaustedError):
        platform.create_vm(vm_config)


# =============================================================================
# GcpPlatform.create_slice Tests
# =============================================================================


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_create_slice_calls_gcloud_tpu_create(
    mock_run: MagicMock, platform: GcpPlatform, slice_config: config_pb2.SliceConfig
):
    """create_slice() issues the correct gcloud compute tpus tpu-vm create command."""
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    handle = platform.create_slice(slice_config)

    assert isinstance(handle, GcpSliceHandle)
    assert handle.slice_id.startswith("iris-tpu-group-")
    assert handle.zone == "us-central2-b"
    assert handle.scale_group == "tpu-group"

    cmd = mock_run.call_args[0][0]
    assert "tpus" in cmd
    assert "create" in cmd
    assert "--accelerator-type=v5litepod-16" in cmd
    assert "--version=tpu-ubuntu2204-base" in cmd
    assert "--preemptible" in cmd
    assert "--project=test-project" in cmd
    assert "--zone=us-central2-b" in cmd


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_create_slice_includes_labels(mock_run: MagicMock, platform: GcpPlatform, slice_config: config_pb2.SliceConfig):
    """create_slice() passes labels to gcloud."""
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    platform.create_slice(slice_config)

    cmd = mock_run.call_args[0][0]
    labels_idx = cmd.index("--labels")
    labels_val = cmd[labels_idx + 1]
    assert "iris-managed=true" in labels_val
    assert "iris-scale-group=tpu-group" in labels_val


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_create_slice_raises_on_failure(
    mock_run: MagicMock, platform: GcpPlatform, slice_config: config_pb2.SliceConfig
):
    """create_slice() raises PlatformError on gcloud failure."""
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="TPU creation failed")

    with pytest.raises(PlatformError, match="TPU creation failed"):
        platform.create_slice(slice_config)


# =============================================================================
# GcpPlatform.list_slices Tests
# =============================================================================


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_list_slices_queries_per_zone(mock_run: MagicMock, platform: GcpPlatform):
    """list_slices() queries each zone independently."""
    tpu_data_z1 = [
        {"name": "tpu-1", "state": "READY", "labels": {"iris-managed": "true"}, "acceleratorType": "v5litepod-16"}
    ]
    tpu_data_z2 = [
        {"name": "tpu-2", "state": "READY", "labels": {"iris-managed": "true"}, "acceleratorType": "v5litepod-16"}
    ]

    def run_side_effect(cmd, **kwargs):
        for arg in cmd:
            if arg.startswith("--zone="):
                zone = arg.split("=")[1]
                if zone == "zone-a":
                    return MagicMock(returncode=0, stdout=json.dumps(tpu_data_z1), stderr="")
                elif zone == "zone-b":
                    return MagicMock(returncode=0, stdout=json.dumps(tpu_data_z2), stderr="")
        return MagicMock(returncode=0, stdout="[]", stderr="")

    mock_run.side_effect = run_side_effect

    slices = platform.list_slices(zones=["zone-a", "zone-b"], labels={"iris-managed": "true"})

    assert len(slices) == 2
    slice_ids = {s.slice_id for s in slices}
    assert "tpu-1" in slice_ids
    assert "tpu-2" in slice_ids


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_list_slices_skips_deleting_tpus(mock_run: MagicMock, platform: GcpPlatform):
    """list_slices() skips TPUs in DELETING or other non-adoptable states."""
    tpu_data = [
        {"name": "tpu-ready", "state": "READY", "labels": {}, "acceleratorType": "v5litepod-16"},
        {"name": "tpu-creating", "state": "CREATING", "labels": {}, "acceleratorType": "v5litepod-16"},
        {"name": "tpu-deleting", "state": "DELETING", "labels": {}, "acceleratorType": "v5litepod-16"},
    ]
    mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(tpu_data), stderr="")

    slices = platform.list_slices(zones=["zone-a"])

    assert len(slices) == 2
    ids = {s.slice_id for s in slices}
    assert "tpu-ready" in ids
    assert "tpu-creating" in ids
    assert "tpu-deleting" not in ids


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_list_slices_with_label_filter(mock_run: MagicMock, platform: GcpPlatform):
    """list_slices() passes label filter to gcloud."""
    mock_run.return_value = MagicMock(returncode=0, stdout="[]", stderr="")

    platform.list_slices(zones=["zone-a"], labels={"iris-scale-group": "my-group"})

    cmd = mock_run.call_args[0][0]
    filter_args = [a for a in cmd if a.startswith("--filter=")]
    assert len(filter_args) == 1
    assert "labels.iris-scale-group=my-group" in filter_args[0]


# =============================================================================
# GcpPlatform.list_vms Tests
# =============================================================================


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_list_vms_returns_standalone_handles(mock_run: MagicMock, platform: GcpPlatform):
    """list_vms() returns GcpStandaloneVmHandle instances with correct IPs."""
    instances = [
        {
            "name": "iris-controller-iris",
            "networkInterfaces": [{"networkIP": "10.0.0.1", "accessConfigs": [{"natIP": "34.1.2.3"}]}],
        }
    ]
    mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(instances), stderr="")

    vms = platform.list_vms(zones=["zone-a"], labels={"iris-controller-iris": "true"})

    assert len(vms) == 1
    assert isinstance(vms[0], GcpStandaloneVmHandle)
    assert vms[0].vm_id == "iris-controller-iris"
    assert vms[0].internal_address == "10.0.0.1"
    assert vms[0].external_address == "34.1.2.3"


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_list_vms_handles_empty_result(mock_run: MagicMock, platform: GcpPlatform):
    """list_vms() returns empty list when no VMs found."""
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    vms = platform.list_vms(zones=["zone-a"])
    assert vms == []


# =============================================================================
# GcpStandaloneVmHandle Tests
# =============================================================================


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_standalone_handle_terminate_calls_gcloud_delete(mock_run: MagicMock):
    """terminate() calls gcloud compute instances delete."""
    ssh = MagicMock()
    handle = GcpStandaloneVmHandle(
        _vm_id="my-vm",
        _internal_address="10.0.0.1",
        _external_address=None,
        _zone="zone-a",
        _project_id="proj",
        _ssh=ssh,
    )
    mock_run.return_value = MagicMock(returncode=0)

    handle.terminate()

    cmd = mock_run.call_args[0][0]
    assert "delete" in cmd
    assert "my-vm" in cmd
    assert "--zone=zone-a" in cmd


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_standalone_handle_set_labels(mock_run: MagicMock):
    """set_labels() calls gcloud compute instances update --update-labels."""
    ssh = MagicMock()
    handle = GcpStandaloneVmHandle(
        _vm_id="my-vm",
        _internal_address="10.0.0.1",
        _external_address=None,
        _zone="zone-a",
        _project_id="proj",
        _ssh=ssh,
    )
    mock_run.return_value = MagicMock(returncode=0)

    handle.set_labels({"iris-controller": "true"})

    cmd = mock_run.call_args[0][0]
    assert "update" in cmd
    assert "--update-labels=iris-controller=true" in cmd


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_standalone_handle_set_metadata(mock_run: MagicMock):
    """set_metadata() calls gcloud compute instances add-metadata."""
    ssh = MagicMock()
    handle = GcpStandaloneVmHandle(
        _vm_id="my-vm",
        _internal_address="10.0.0.1",
        _external_address=None,
        _zone="zone-a",
        _project_id="proj",
        _ssh=ssh,
    )
    mock_run.return_value = MagicMock(returncode=0)

    handle.set_metadata({"iris-addr": "http://10.0.0.1:10000"})

    cmd = mock_run.call_args[0][0]
    assert "add-metadata" in cmd
    assert "--metadata=iris-addr=http://10.0.0.1:10000" in cmd


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_standalone_handle_status(mock_run: MagicMock):
    """status() queries gcloud and returns the correct VmStatus."""
    ssh = MagicMock()
    handle = GcpStandaloneVmHandle(
        _vm_id="my-vm",
        _internal_address="10.0.0.1",
        _external_address=None,
        _zone="zone-a",
        _project_id="proj",
        _ssh=ssh,
    )
    mock_run.return_value = MagicMock(returncode=0, stdout="RUNNING\n", stderr="")

    status = handle.status()
    assert status.state == CloudVmState.RUNNING


# =============================================================================
# GcpSliceHandle Tests
# =============================================================================


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_slice_handle_list_vms_returns_vm_handles(mock_run: MagicMock):
    """list_vms() queries gcloud tpu describe and returns GcpVmHandle instances.

    Returns vm_count handles based on topology. v5litepod-16 has 4 VMs.
    """
    tpu_data = {
        "networkEndpoints": [
            {"ipAddress": "10.0.0.1"},
            {"ipAddress": "10.0.0.2"},
            {"ipAddress": "10.0.0.3"},
            {"ipAddress": "10.0.0.4"},
        ]
    }
    mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(tpu_data), stderr="")

    handle = GcpSliceHandle(
        _slice_id="my-tpu",
        _zone="zone-a",
        _project_id="proj",
        _labels={"iris-scale-group": "g1"},
        _created_at=MagicMock(epoch_ms=lambda: 123),
        _label_prefix="iris",
        _accelerator_variant="v5litepod-16",
    )

    vms = handle.list_vms()

    assert len(vms) == 4
    assert all(isinstance(v, GcpVmHandle) for v in vms)
    assert vms[0].internal_address == "10.0.0.1"
    assert vms[1].internal_address == "10.0.0.2"
    assert vms[2].internal_address == "10.0.0.3"
    assert vms[3].internal_address == "10.0.0.4"
    assert vms[0].vm_id == "my-tpu-worker-0"
    assert vms[1].vm_id == "my-tpu-worker-1"
    assert vms[2].vm_id == "my-tpu-worker-2"
    assert vms[3].vm_id == "my-tpu-worker-3"


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_slice_handle_list_vms_partial_endpoints(mock_run: MagicMock):
    """list_vms() returns all VMs even when endpoints are incomplete.

    This is the Task 11 bug fix: if a v5litepod-16 (4 VMs) has only 3 endpoints
    provisioned, we must return 4 handles (with the 4th having empty address).
    Otherwise bootstrap will skip the 4th VM.
    """
    tpu_data = {
        "networkEndpoints": [
            {"ipAddress": "10.0.0.1"},
            {"ipAddress": "10.0.0.2"},
            {"ipAddress": "10.0.0.3"},
            # Missing 4th endpoint - VM still provisioning
        ]
    }
    mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(tpu_data), stderr="")

    handle = GcpSliceHandle(
        _slice_id="my-tpu",
        _zone="zone-a",
        _project_id="proj",
        _labels={"iris-scale-group": "g1"},
        _created_at=MagicMock(epoch_ms=lambda: 123),
        _label_prefix="iris",
        _accelerator_variant="v5litepod-16",
    )

    vms = handle.list_vms()

    # Must return 4 VMs (topology count), not 3 (endpoint count)
    assert len(vms) == 4
    assert all(isinstance(v, GcpVmHandle) for v in vms)
    assert vms[0].internal_address == "10.0.0.1"
    assert vms[1].internal_address == "10.0.0.2"
    assert vms[2].internal_address == "10.0.0.3"
    assert vms[3].internal_address == ""  # Empty address - not yet provisioned
    assert vms[3].vm_id == "my-tpu-worker-3"


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_slice_handle_list_vms_no_endpoints(mock_run: MagicMock):
    """list_vms() returns all VMs even when no endpoints are provisioned yet."""
    tpu_data = {"networkEndpoints": []}
    mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(tpu_data), stderr="")

    handle = GcpSliceHandle(
        _slice_id="my-tpu",
        _zone="zone-a",
        _project_id="proj",
        _labels={"iris-scale-group": "g1"},
        _created_at=MagicMock(epoch_ms=lambda: 123),
        _label_prefix="iris",
        _accelerator_variant="v5litepod-16",
    )

    vms = handle.list_vms()

    # Must return 4 VMs (topology count), all with empty addresses
    assert len(vms) == 4
    assert all(isinstance(v, GcpVmHandle) for v in vms)
    assert all(v.internal_address == "" for v in vms)


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_slice_handle_list_vms_unknown_topology(mock_run: MagicMock):
    """list_vms() falls back to endpoint count for unknown topologies."""
    tpu_data = {
        "networkEndpoints": [
            {"ipAddress": "10.0.0.1"},
            {"ipAddress": "10.0.0.2"},
        ]
    }
    mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(tpu_data), stderr="")

    handle = GcpSliceHandle(
        _slice_id="my-tpu",
        _zone="zone-a",
        _project_id="proj",
        _labels={"iris-scale-group": "g1"},
        _created_at=MagicMock(epoch_ms=lambda: 123),
        _label_prefix="iris",
        _accelerator_variant="unknown-tpu-type",
    )

    vms = handle.list_vms()

    # Unknown topology: fall back to endpoint count
    assert len(vms) == 2
    assert vms[0].internal_address == "10.0.0.1"
    assert vms[1].internal_address == "10.0.0.2"


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_slice_handle_terminate_calls_tpu_delete(mock_run: MagicMock):
    """terminate() calls gcloud compute tpus tpu-vm delete."""
    handle = GcpSliceHandle(
        _slice_id="my-tpu",
        _zone="zone-a",
        _project_id="proj",
        _labels={},
        _created_at=MagicMock(epoch_ms=lambda: 123),
        _label_prefix="iris",
        _accelerator_variant="v5litepod-16",
    )
    mock_run.return_value = MagicMock(returncode=0)

    handle.terminate()

    cmd = mock_run.call_args[0][0]
    assert "tpus" in cmd
    assert "delete" in cmd
    assert "my-tpu" in cmd


@patch("iris.cluster.platform.gcp.subprocess.run")
def test_slice_handle_status_returns_cloud_state(mock_run: MagicMock):
    """status() queries gcloud tpu describe and returns SliceStatus."""
    tpu_data = {"state": "READY", "networkEndpoints": [{"ipAddress": "10.0.0.1"}]}
    mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(tpu_data), stderr="")

    handle = GcpSliceHandle(
        _slice_id="my-tpu",
        _zone="zone-a",
        _project_id="proj",
        _labels={},
        _created_at=MagicMock(epoch_ms=lambda: 123),
        _label_prefix="iris",
        _accelerator_variant="v5litepod-16",
    )

    status = handle.status()
    assert status.state == CloudSliceState.READY
    assert status.vm_count == 1


# =============================================================================
# GcpPlatform.shutdown Tests
# =============================================================================


def test_shutdown_is_noop(platform: GcpPlatform):
    """shutdown() is a no-op for GcpPlatform."""
    platform.shutdown()  # Should not raise


# =============================================================================
# Tests using FakeGcloud
# =============================================================================


def test_create_slice_with_fake_gcloud(
    fake_gcloud: FakeGcloud, platform: GcpPlatform, slice_config: config_pb2.SliceConfig
):
    """create_slice() creates a TPU tracked in the fake's state."""
    handle = platform.create_slice(slice_config)

    assert isinstance(handle, GcpSliceHandle)
    assert handle.slice_id.startswith("iris-tpu-group-")
    assert handle.zone == "us-central2-b"
    assert handle.scale_group == "tpu-group"

    # Verify the TPU was stored in the fake
    assert len(fake_gcloud._tpus) == 1
    key = next(iter(fake_gcloud._tpus))
    tpu = fake_gcloud._tpus[key]
    assert tpu["acceleratorType"] == "v5litepod-16"
    assert tpu["labels"]["iris-managed"] == "true"


def test_create_slice_empty_accelerator_variant_fails(
    fake_gcloud: FakeGcloud,
    platform: GcpPlatform,
):
    """create_slice() with empty accelerator_variant is rejected by validation.

    This catches the Task 22 bug where accelerator_variant was empty, producing
    --accelerator-type= with no value.
    """
    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu-group",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="",  # Bug: empty string
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"

    with pytest.raises(ValueError, match="accelerator_variant"):
        platform.create_slice(cfg)


def test_create_vm_with_fake_gcloud(fake_gcloud: FakeGcloud, platform: GcpPlatform, vm_config: config_pb2.VmConfig):
    """create_vm() creates a VM tracked in the fake's state."""
    handle = platform.create_vm(vm_config)

    assert isinstance(handle, GcpStandaloneVmHandle)
    assert handle.vm_id == "test-controller"
    assert handle.internal_address.startswith("10.128.0.")
    assert handle.external_address is not None

    # Verify the VM was stored in the fake
    assert len(fake_gcloud._vms) == 1


def test_list_slices_with_fake_gcloud(
    fake_gcloud: FakeGcloud, platform: GcpPlatform, slice_config: config_pb2.SliceConfig
):
    """list_slices() returns slices previously created via create_slice()."""
    platform.create_slice(slice_config)

    slices = platform.list_slices(
        zones=["us-central2-b"],
        labels={"iris-managed": "true"},
    )
    assert len(slices) == 1
    assert slices[0].zone == "us-central2-b"


def test_fake_gcloud_failure_injection(
    fake_gcloud: FakeGcloud, platform: GcpPlatform, slice_config: config_pb2.SliceConfig
):
    """set_failure() makes the next operation fail with the injected error."""
    fake_gcloud.set_failure("tpu_create", "RESOURCE_EXHAUSTED: no capacity")

    with pytest.raises(QuotaExhaustedError):
        platform.create_slice(slice_config)


# =============================================================================
# ScalingGroup integration: accelerator_variant propagation (Task 22)
# =============================================================================


def test_scale_up_propagates_accelerator_variant(
    fake_gcloud: FakeGcloud,
    gcp_config: config_pb2.GcpPlatformConfig,
):
    """scale_up() propagates accelerator_variant from ScaleGroupConfig to the TPU create command.

    Real config files set accelerator_variant on the ScaleGroupConfig, not on the
    slice_template. Before the fix, this resulted in an empty --accelerator-type=
    flag which GCP rejects.
    """
    sg_config = config_pb2.ScaleGroupConfig(
        name="tpu-group",
        min_slices=0,
        max_slices=4,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-16",
        slice_size=4,
    )
    sg_config.slice_template.gcp.zone = "us-central2-b"
    sg_config.slice_template.gcp.runtime_version = "tpu-ubuntu2204-base"

    platform = GcpPlatform(gcp_config, label_prefix="iris")
    group = ScalingGroup(config=sg_config, platform=platform, label_prefix="iris")

    handle = group.scale_up()

    assert handle.slice_id.startswith("iris-tpu-group-")

    assert len(fake_gcloud._tpus) == 1
    tpu = next(iter(fake_gcloud._tpus.values()))
    assert tpu["acceleratorType"] == "v5litepod-16"


# =============================================================================
# Config validation tests
# =============================================================================


def test_create_slice_validates_config(
    fake_gcloud: FakeGcloud,
    platform: GcpPlatform,
):
    """create_slice() raises ValueError when accelerator_variant is empty."""
    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu-group",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="",
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"

    with pytest.raises(ValueError, match="accelerator_variant"):
        platform.create_slice(cfg)


def test_create_vm_validates_config(
    fake_gcloud: FakeGcloud,
    platform: GcpPlatform,
):
    """create_vm() raises ValueError when zone is empty."""
    cfg = config_pb2.VmConfig(name="test-vm")
    # gcp.zone is empty by default

    with pytest.raises(ValueError, match="zone"):
        platform.create_vm(cfg)


def test_validate_slice_config_lists_all_missing_fields():
    """_validate_slice_config reports all missing fields, not just the first."""
    cfg = config_pb2.SliceConfig(name_prefix="test")
    # All three required fields are empty

    with pytest.raises(ValueError, match="accelerator_variant") as exc_info:
        _validate_slice_config(cfg)
    assert "zone" in str(exc_info.value)
    assert "runtime_version" in str(exc_info.value)
