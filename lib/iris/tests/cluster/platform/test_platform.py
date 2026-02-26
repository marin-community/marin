# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidated Platform protocol tests.

Tests exercise the Platform protocol contract through parameterized fixtures
(GCP via FakeGcloud, Manual, Local). Platform-specific behavioral tests
that cannot be expressed through the protocol are in dedicated sections.
"""

from __future__ import annotations

import threading
import unittest.mock
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

import pytest

from iris.cluster.platform.base import (
    CloudSliceState,
    Labels,
    PlatformError,
    QuotaExhaustedError,
)
from iris.cluster.platform.gcp import (
    GcpPlatform,
    _build_vm_slice_id,
    _validate_slice_config,
)
from iris.cluster.platform.local import LocalPlatform
from iris.cluster.platform.manual import ManualPlatform
from iris.managed_thread import ThreadContainer
from iris.rpc import config_pb2
from tests.cluster.platform.fakes import FakeGcloud

# =============================================================================
# Fixture infrastructure
# =============================================================================


@dataclass
class PlatformEnv:
    """Test environment for a single Platform implementation."""

    platform: GcpPlatform | ManualPlatform | LocalPlatform
    zone: str
    name: str
    label_prefix: str


def _make_slice_config(env: PlatformEnv, group_name: str) -> config_pb2.SliceConfig:
    """Build a SliceConfig appropriate for the platform under test."""
    labels = Labels(env.label_prefix)

    if env.name == "gcp":
        cfg = config_pb2.SliceConfig(
            name_prefix=f"iris-{group_name}",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5litepod-16",
        )
        cfg.gcp.zone = env.zone
        cfg.gcp.runtime_version = "tpu-ubuntu2204-base"
        cfg.labels[labels.iris_managed] = "true"
        cfg.labels[labels.iris_scale_group] = group_name
        return cfg
    elif env.name == "manual":
        cfg = config_pb2.SliceConfig(name_prefix=f"iris-{group_name}", num_vms=1)
        cfg.manual.CopyFrom(config_pb2.ManualSliceConfig())
        cfg.labels[labels.iris_scale_group] = group_name
        return cfg
    else:
        cfg = config_pb2.SliceConfig(name_prefix=f"test-{group_name}", num_vms=1)
        cfg.labels[labels.iris_scale_group] = group_name
        return cfg


def _make_vm_config(env: PlatformEnv, name: str = "test-controller") -> config_pb2.VmConfig:
    """Build a VmConfig appropriate for the platform under test."""
    cfg = config_pb2.VmConfig(name=name)
    if env.name == "gcp":
        cfg.gcp.zone = env.zone
        cfg.gcp.machine_type = "n2-standard-4"
    elif env.name == "manual":
        cfg.manual.CopyFrom(config_pb2.ManualVmConfig())
    return cfg


@pytest.fixture(params=["gcp", "manual", "local"])
def platform_env(request) -> Iterator[PlatformEnv]:
    """Yield a PlatformEnv for each platform implementation.

    GCP is backed by FakeGcloud patching subprocess.run.
    """
    name = request.param

    if name == "gcp":
        fake = FakeGcloud()
        gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
        platform = GcpPlatform(gcp_config, label_prefix="iris")
        with unittest.mock.patch("iris.cluster.platform.gcp.subprocess.run", side_effect=fake):
            yield PlatformEnv(platform=platform, zone="us-central2-b", name="gcp", label_prefix="iris")
    elif name == "manual":
        platform = ManualPlatform(
            label_prefix="iris",
            hosts=["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4", "10.0.0.5"],
        )
        yield PlatformEnv(platform=platform, zone="manual", name="manual", label_prefix="iris")
    else:
        platform = LocalPlatform("test")
        yield PlatformEnv(platform=platform, zone="local", name="local", label_prefix="test")
        platform.shutdown()


# =============================================================================
# Section 1: Parameterized Protocol Tests
#
# These test the Platform protocol contract that all implementations satisfy.
# =============================================================================


def test_create_slice_returns_handle_with_vms_and_scale_group(platform_env: PlatformEnv):
    """create_slice returns a handle with a non-empty id, correct scale_group, and at least one VM."""
    cfg = _make_slice_config(platform_env, "my-group")
    handle = platform_env.platform.create_slice(cfg)

    assert handle.slice_id
    assert handle.scale_group == "my-group"
    assert handle.zone
    assert len(handle.describe().workers) >= 1


def test_created_slice_appears_in_list_slices(platform_env: PlatformEnv):
    """A slice returned by create_slice is discoverable via list_slices."""
    cfg = _make_slice_config(platform_env, "listed-group")
    handle = platform_env.platform.create_slice(cfg)

    slices = platform_env.platform.list_slices(zones=[platform_env.zone])
    slice_ids = {s.slice_id for s in slices}
    assert handle.slice_id in slice_ids


def test_list_slices_filters_by_labels(platform_env: PlatformEnv):
    """list_slices with label filter returns only matching slices."""
    labels = Labels(platform_env.label_prefix)
    cfg_a = _make_slice_config(platform_env, "group-a")
    cfg_b = _make_slice_config(platform_env, "group-b")

    platform_env.platform.create_slice(cfg_a)
    platform_env.platform.create_slice(cfg_b)

    slices = platform_env.platform.list_slices(zones=[platform_env.zone], labels={labels.iris_scale_group: "group-a"})
    assert len(slices) == 1
    assert slices[0].scale_group == "group-a"


def test_create_vm_returns_handle_with_address(platform_env: PlatformEnv):
    """create_vm returns a handle with non-empty id and internal address."""
    cfg = _make_vm_config(platform_env)
    handle = platform_env.platform.create_vm(cfg)

    assert handle.vm_id
    assert handle.internal_address


def test_created_vm_appears_in_list_vms(platform_env: PlatformEnv):
    """A VM returned by create_vm is discoverable via list_vms."""
    cfg = _make_vm_config(platform_env)
    platform_env.platform.create_vm(cfg)

    vms = platform_env.platform.list_vms(zones=[platform_env.zone])
    assert len(vms) >= 1


def test_shutdown_completes_without_error(platform_env: PlatformEnv):
    """shutdown() does not raise."""
    platform_env.platform.shutdown()


# Tunnel and terminate have different semantics on GCP (SSH tunnel, gcloud delete)
# so we test them on in-memory platforms only.


@pytest.fixture(params=["manual", "local"])
def in_memory_env(request) -> Iterator[PlatformEnv]:
    """Platforms with fully in-memory state (no subprocess calls)."""
    name = request.param
    if name == "manual":
        platform = ManualPlatform(
            label_prefix="iris",
            hosts=["10.0.0.1", "10.0.0.2", "10.0.0.3"],
        )
        yield PlatformEnv(platform=platform, zone="manual", name="manual", label_prefix="iris")
    else:
        platform = LocalPlatform("test")
        yield PlatformEnv(platform=platform, zone="local", name="local", label_prefix="test")
        platform.shutdown()


def test_terminate_then_status_is_deleting(in_memory_env: PlatformEnv):
    """After terminate(), slice status reports DELETING."""
    cfg = _make_slice_config(in_memory_env, "term-group")
    handle = in_memory_env.platform.create_slice(cfg)

    handle.terminate()
    assert handle.describe().state == CloudSliceState.DELETING


def test_tunnel_returns_address_directly(in_memory_env: PlatformEnv):
    """tunnel() is a passthrough on in-memory platforms (no SSH)."""
    addr = "http://10.0.0.1:10000"
    with in_memory_env.platform.tunnel(addr) as tunneled:
        assert tunneled == addr


# =============================================================================
# Section 2: GCP-Specific Tests
#
# Behaviors that only apply to GcpPlatform and need FakeGcloud.
# =============================================================================


def test_gcp_quota_error_raises_quota_exhausted():
    """GcpPlatform raises QuotaExhaustedError when gcloud reports RESOURCE_EXHAUSTED."""
    fake = FakeGcloud()
    fake.set_failure("tpu_create", "RESOURCE_EXHAUSTED: no capacity")

    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpPlatform(gcp_config, label_prefix="iris")

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu-group",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-16",
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"

    with unittest.mock.patch("iris.cluster.platform.gcp.subprocess.run", side_effect=fake):
        with pytest.raises(QuotaExhaustedError):
            platform.create_slice(cfg)


def test_gcp_validate_slice_config_reports_all_missing_fields():
    """_validate_slice_config reports all missing fields at once."""
    cfg = config_pb2.SliceConfig(name_prefix="test")

    with pytest.raises(ValueError, match="accelerator_variant") as exc_info:
        _validate_slice_config(cfg)
    assert "zone" in str(exc_info.value)
    assert "runtime_version" in str(exc_info.value)


def test_gcp_validate_vm_slice_config_requires_machine_type():
    """VM slice mode requires gcp.machine_type."""
    cfg = config_pb2.SliceConfig(
        name_prefix="test",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM

    with pytest.raises(ValueError, match=r"gcp\.machine_type"):
        _validate_slice_config(cfg)


def test_gcp_validate_vm_slice_config_rejects_preemptible():
    """VM slice mode rejects preemptible instances."""
    cfg = config_pb2.SliceConfig(
        name_prefix="test",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
        preemptible=True,
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
    cfg.gcp.machine_type = "n2-standard-4"

    with pytest.raises(ValueError, match="does not support preemptible"):
        _validate_slice_config(cfg)


def test_gcp_validate_vm_slice_config_rejects_num_vms_not_one():
    """VM slice mode requires exactly one VM."""
    cfg = config_pb2.SliceConfig(
        name_prefix="test",
        num_vms=2,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
    cfg.gcp.machine_type = "n2-standard-4"

    with pytest.raises(ValueError, match="num_vms=1"):
        _validate_slice_config(cfg)


def test_gcp_create_vm_slice_mode_produces_single_worker_slice():
    """VM slice mode creates a single-worker slice that is discoverable and terminable."""
    fake = FakeGcloud()
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpPlatform(gcp_config, label_prefix="iris")

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
    cfg.gcp.machine_type = "n2-standard-4"
    cfg.labels[Labels("iris").iris_managed] = "true"
    cfg.labels[Labels("iris").iris_scale_group] = "cpu-vm"

    with unittest.mock.patch("iris.cluster.platform.gcp.subprocess.run", side_effect=fake):
        handle = platform.create_slice(cfg)
        status = handle.describe()
        assert status.worker_count == 1
        assert len(status.workers) == 1
        assert status.workers[0].internal_address
        assert handle.scale_group == "cpu-vm"

        listed = platform.list_all_slices(labels={Labels("iris").iris_managed: "true"})
        assert handle.slice_id in {s.slice_id for s in listed}

        handle.terminate()
        listed_after = platform.list_all_slices(labels={Labels("iris").iris_managed: "true"})
        assert handle.slice_id not in {s.slice_id for s in listed_after}


def test_gcp_build_vm_slice_id_bounds_and_normalizes():
    slice_id = _build_vm_slice_id(
        "smoke-cpu_vm_e2_standard_4_ondemand-europe-west4-b",
        1772123761944,
    )
    assert len(slice_id) <= 63
    assert "_" not in slice_id
    assert slice_id.endswith("-1772123761944")


def test_gcp_create_vm_slice_mode_with_long_prefix_uses_valid_slice_id():
    fake = FakeGcloud()
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpPlatform(gcp_config, label_prefix="iris")

    cfg = config_pb2.SliceConfig(
        name_prefix="smoke-cpu_vm_e2_standard_4_ondemand-europe-west4-b",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
    cfg.gcp.machine_type = "n2-standard-4"
    cfg.labels[Labels("iris").iris_managed] = "true"
    cfg.labels[Labels("iris").iris_scale_group] = "cpu-vm"

    with unittest.mock.patch("iris.cluster.platform.gcp.subprocess.run", side_effect=fake):
        handle = platform.create_slice(cfg)
        assert len(handle.slice_id) <= 63
        assert "_" not in handle.slice_id
        status = handle.describe()
        assert len(status.workers) == 1
        assert getattr(status.workers[0]._remote_exec, "ssh_user", None) == "iris"
        listed = platform.list_all_slices(labels={Labels("iris").iris_managed: "true"})
        assert handle.slice_id in {s.slice_id for s in listed}


def test_gcp_empty_accelerator_variant_rejected():
    """create_slice with empty accelerator_variant raises ValueError."""
    fake = FakeGcloud()
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpPlatform(gcp_config, label_prefix="iris")

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu-group",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="",
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"

    with unittest.mock.patch("iris.cluster.platform.gcp.subprocess.run", side_effect=fake):
        with pytest.raises(ValueError, match="accelerator_variant"):
            platform.create_slice(cfg)


def test_gcp_create_vm_validates_config():
    """create_vm with empty zone raises ValueError."""
    fake = FakeGcloud()
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpPlatform(gcp_config, label_prefix="iris")

    cfg = config_pb2.VmConfig(name="test-vm")

    with unittest.mock.patch("iris.cluster.platform.gcp.subprocess.run", side_effect=fake):
        with pytest.raises(ValueError, match="zone"):
            platform.create_vm(cfg)


def test_gcp_list_slices_skips_deleting_tpus():
    """list_slices omits TPUs in DELETING state."""
    fake = FakeGcloud()
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpPlatform(gcp_config, label_prefix="iris")

    with unittest.mock.patch("iris.cluster.platform.gcp.subprocess.run", side_effect=fake):
        cfg = config_pb2.SliceConfig(
            name_prefix="iris-tpu",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5litepod-16",
        )
        cfg.gcp.zone = "us-central2-b"
        cfg.gcp.runtime_version = "tpu-ubuntu2204-base"
        cfg.labels[Labels("iris").iris_managed] = "true"

        handle = platform.create_slice(cfg)

        # Mark the TPU as DELETING in the fake's internal state
        for _key, tpu_data in fake._tpus.items():
            if tpu_data["name"] == handle.slice_id:
                tpu_data["state"] = "DELETING"
                break

        slices = platform.list_slices(zones=["us-central2-b"])
        slice_ids = {s.slice_id for s in slices}
        assert handle.slice_id not in slice_ids


def test_gcp_create_slice_resolves_ghcr_image_in_bootstrap_config():
    """create_slice rewrites GHCR images in bootstrap_config via resolve_image."""
    fake = FakeGcloud()
    gcp_config = config_pb2.GcpPlatformConfig(project_id="my-proj")
    platform = GcpPlatform(gcp_config, label_prefix="iris")

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-16",
    )
    cfg.gcp.zone = "europe-west4-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"

    bc = config_pb2.BootstrapConfig(
        docker_image="ghcr.io/marin-community/iris-worker:latest",
        worker_port=10001,
        controller_address="controller:10000",
        cache_dir="/var/cache/iris",
    )

    with unittest.mock.patch("iris.cluster.platform.gcp.subprocess.run", side_effect=fake):
        platform.create_slice(cfg, bootstrap_config=bc)

    assert bc.docker_image == "europe-docker.pkg.dev/my-proj/ghcr-mirror/marin-community/iris-worker:latest"


def test_gcp_list_slices_skips_inactive_vm_instances():
    """list_slices omits VM-backed slices for instances in inactive states."""
    fake = FakeGcloud()
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpPlatform(gcp_config, label_prefix="iris")

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
    cfg.gcp.machine_type = "n2-standard-4"
    cfg.labels[Labels("iris").iris_managed] = "true"
    cfg.labels[Labels("iris").iris_scale_group] = "cpu-vm"

    with unittest.mock.patch("iris.cluster.platform.gcp.subprocess.run", side_effect=fake):
        handle = platform.create_slice(cfg)
        for _key, vm_data in fake._vms.items():
            if vm_data.get("labels", {}).get(Labels("iris").iris_slice_id) == handle.slice_id:
                vm_data["status"] = "TERMINATED"
                break

        listed = platform.list_all_slices(labels={Labels("iris").iris_managed: "true"})
        assert handle.slice_id not in {s.slice_id for s in listed}


def test_gcp_list_slices_preserves_vm_creation_timestamp():
    """VM-backed slices use cloud creation timestamp when rediscovered."""
    fake = FakeGcloud()
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpPlatform(gcp_config, label_prefix="iris")

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
    cfg.gcp.machine_type = "n2-standard-4"
    cfg.labels[Labels("iris").iris_managed] = "true"
    cfg.labels[Labels("iris").iris_scale_group] = "cpu-vm"

    vm_creation_ts = "2024-01-02T03:04:05.000Z"
    expected_epoch_ms = int(datetime.fromisoformat(vm_creation_ts.replace("Z", "+00:00")).timestamp() * 1000)

    with unittest.mock.patch("iris.cluster.platform.gcp.subprocess.run", side_effect=fake):
        handle = platform.create_slice(cfg)
        for _key, vm_data in fake._vms.items():
            if vm_data.get("labels", {}).get(Labels("iris").iris_slice_id) == handle.slice_id:
                vm_data["creationTimestamp"] = vm_creation_ts
                break

        listed = platform.list_all_slices(labels={Labels("iris").iris_managed: "true"})
        listed_by_id = {s.slice_id: s for s in listed}
        assert listed_by_id[handle.slice_id].created_at.epoch_ms() == expected_epoch_ms


def test_gcp_reload_uses_full_stop_then_start():
    """reload delegates to stop_all() before starting controller again."""
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpPlatform(gcp_config, label_prefix="iris")
    cfg = config_pb2.IrisClusterConfig()

    with (
        unittest.mock.patch.object(platform, "stop_all") as stop_all,
        unittest.mock.patch.object(platform, "start_controller", return_value="10.0.0.1:10000") as start_controller,
    ):
        address = platform.reload(cfg)

    stop_all.assert_called_once_with(cfg)
    start_controller.assert_called_once_with(cfg)
    assert address == "10.0.0.1:10000"


# =============================================================================
# Section 3: Manual-Specific Tests
#
# Host pool management, exclusivity, and reallocation.
# =============================================================================


def test_manual_host_pool_exhaustion_raises():
    """create_slice raises when not enough hosts are available."""
    platform = ManualPlatform(label_prefix="iris", hosts=["10.0.0.1"])
    cfg = config_pb2.SliceConfig(name_prefix="iris-group", num_vms=3)
    cfg.manual.CopyFrom(config_pb2.ManualSliceConfig())

    with pytest.raises(RuntimeError, match="Need 3 hosts but only 1 available"):
        platform.create_slice(cfg)


def test_manual_host_exclusivity():
    """A host allocated to one VM cannot be allocated to another."""
    platform = ManualPlatform(label_prefix="iris", hosts=["10.0.0.1", "10.0.0.2"])

    cfg1 = config_pb2.VmConfig(name="ctrl-1")
    cfg1.manual.host = "10.0.0.1"
    platform.create_vm(cfg1)

    cfg2 = config_pb2.VmConfig(name="ctrl-2")
    cfg2.manual.host = "10.0.0.1"
    with pytest.raises(RuntimeError, match="already allocated"):
        platform.create_vm(cfg2)


def test_manual_terminated_host_returns_to_pool():
    """After terminating a VM, its host can be reallocated."""
    platform = ManualPlatform(label_prefix="iris", hosts=["10.0.0.1"])

    cfg = config_pb2.VmConfig(name="ctrl")
    cfg.manual.host = "10.0.0.1"
    handle = platform.create_vm(cfg)
    assert platform.available_host_count == 0

    handle.terminate()
    assert platform.available_host_count == 1

    cfg2 = config_pb2.VmConfig(name="ctrl-2")
    cfg2.manual.host = "10.0.0.1"
    handle2 = platform.create_vm(cfg2)
    assert handle2.internal_address == "10.0.0.1"


def test_manual_slice_terminate_returns_hosts():
    """Terminating a slice returns all its hosts to the pool."""
    platform = ManualPlatform(label_prefix="iris", hosts=["10.0.0.1", "10.0.0.2", "10.0.0.3"])

    cfg = config_pb2.SliceConfig(name_prefix="iris-group", num_vms=2)
    cfg.manual.CopyFrom(config_pb2.ManualSliceConfig())
    handle = platform.create_slice(cfg)
    assert platform.available_host_count == 1

    handle.terminate()
    assert platform.available_host_count == 3


# =============================================================================
# Section 4: Local-Specific Tests
# =============================================================================


def test_local_shutdown_stops_thread_container():
    """shutdown() stops the ThreadContainer and clears internal state."""
    threads = ThreadContainer(name="test-threads")
    platform = LocalPlatform("test", threads=threads)

    started = threading.Event()

    def worker(stop_event: threading.Event) -> None:
        started.set()
        stop_event.wait()

    threads.spawn(worker, name="test-worker")
    started.wait(timeout=5.0)
    assert threads.is_alive

    cfg_slice = config_pb2.SliceConfig(name_prefix="test", num_vms=1)
    platform.create_slice(cfg_slice)
    assert len(platform.list_slices(zones=["local"])) == 1

    platform.shutdown()
    assert not threads.is_alive
    assert len(platform.list_slices(zones=["local"])) == 0


# =============================================================================
# Section 5: list_all_slices Tests
# =============================================================================


def test_list_all_slices_returns_created_slices(platform_env: PlatformEnv):
    """list_all_slices discovers slices without caller specifying zones."""
    cfg = _make_slice_config(platform_env, "group-a")
    handle = platform_env.platform.create_slice(cfg)
    all_slices = platform_env.platform.list_all_slices()
    assert handle.slice_id in {s.slice_id for s in all_slices}


def test_list_all_slices_filters_by_labels(platform_env: PlatformEnv):
    """list_all_slices respects label filter."""
    cfg_a = _make_slice_config(platform_env, "group-a")
    cfg_b = _make_slice_config(platform_env, "group-b")
    platform_env.platform.create_slice(cfg_a)
    platform_env.platform.create_slice(cfg_b)

    labels = Labels(platform_env.label_prefix)
    filtered = platform_env.platform.list_all_slices(labels={labels.iris_scale_group: "group-a"})
    assert all(s.scale_group == "group-a" for s in filtered)
    assert len(filtered) == 1


def test_gcp_list_all_slices_raises_without_zones():
    """GcpPlatform.list_all_slices raises when no zones configured."""
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpPlatform(gcp_config, label_prefix="iris")
    with pytest.raises(ValueError, match="no zones configured"):
        platform.list_all_slices()


def test_gcp_list_all_slices_multi_zone():
    """GcpPlatform.list_all_slices returns slices across multiple zones."""
    fake = FakeGcloud()
    gcp_config = config_pb2.GcpPlatformConfig(
        project_id="test-project",
        zones=["zone-a", "zone-b"],
    )
    platform = GcpPlatform(gcp_config, label_prefix="iris")

    iris_labels = Labels("iris")
    with unittest.mock.patch("iris.cluster.platform.gcp.subprocess.run", side_effect=fake):
        cfg_a = config_pb2.SliceConfig(
            name_prefix="iris-tpu",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5litepod-16",
        )
        cfg_a.gcp.zone = "zone-a"
        cfg_a.gcp.runtime_version = "tpu-ubuntu2204-base"
        cfg_a.labels[iris_labels.iris_managed] = "true"

        cfg_b = config_pb2.SliceConfig(
            name_prefix="iris-tpu",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5litepod-16",
        )
        cfg_b.gcp.zone = "zone-b"
        cfg_b.gcp.runtime_version = "tpu-ubuntu2204-base"
        cfg_b.labels[iris_labels.iris_managed] = "true"

        handle_a = platform.create_slice(cfg_a)
        handle_b = platform.create_slice(cfg_b)

        all_slices = platform.list_all_slices()
        slice_ids = {s.slice_id for s in all_slices}
        assert handle_a.slice_id in slice_ids
        assert handle_b.slice_id in slice_ids
        assert handle_a.slice_id in slice_ids
        assert handle_b.slice_id in slice_ids


# =============================================================================
# Section 6: GCE VM Slice Bootstrap via Startup-Script
#
# Tests for the startup-script metadata bootstrap path (no SSH for bootstrap).
# =============================================================================


def test_gcp_vm_slice_bootstrap_monitors_serial_port():
    """VM slice bootstrap waits for startup-script completion via serial port monitoring."""
    fake = FakeGcloud()
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpPlatform(gcp_config, label_prefix="iris")

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
    cfg.gcp.machine_type = "n2-standard-4"

    bc = config_pb2.BootstrapConfig(
        docker_image="test-image:latest",
        worker_port=10001,
        controller_address="controller:10000",
        cache_dir="/var/cache/iris",
    )

    with unittest.mock.patch("iris.cluster.platform.gcp.subprocess.run", side_effect=fake):
        handle = platform.create_slice(cfg, bootstrap_config=bc)

        # Simulate serial port output from the startup-script.
        fake.append_serial_output(
            handle._vm_name,
            "us-central2-b",
            "[iris-init] Starting Iris worker bootstrap\n"
            "[iris-init] Phase: prerequisites\n"
            "[iris-init] Docker installed\n"
            "[iris-init] Phase: docker_pull\n"
            "[iris-init] Worker container started\n"
            "[iris-init] Worker is healthy\n"
            "[iris-init] Bootstrap complete\n",
        )

        # Run bootstrap synchronously (the background thread does this).
        platform._run_vm_slice_bootstrap(handle, bc, poll_interval=0.01, cloud_ready_timeout=5.0)

        with handle._bootstrap_lock:
            assert handle._bootstrap_state == CloudSliceState.READY


def test_gcp_vm_slice_bootstrap_detects_startup_script_failure():
    """VM slice bootstrap raises on [iris-init] ERROR in serial output."""
    fake = FakeGcloud()
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpPlatform(gcp_config, label_prefix="iris")

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
    cfg.gcp.machine_type = "n2-standard-4"

    bc = config_pb2.BootstrapConfig(
        docker_image="test-image:latest",
        worker_port=10001,
        controller_address="controller:10000",
        cache_dir="/var/cache/iris",
    )

    with unittest.mock.patch("iris.cluster.platform.gcp.subprocess.run", side_effect=fake):
        handle = platform.create_slice(cfg, bootstrap_config=bc)

        # Simulate an error in the startup-script.
        fake.append_serial_output(
            handle._vm_name,
            "us-central2-b",
            "[iris-init] Starting Iris worker bootstrap\n" "[iris-init] ERROR: Worker container exited unexpectedly\n",
        )

        with pytest.raises(PlatformError, match="bootstrap failed"):
            platform._run_vm_slice_bootstrap(handle, bc, poll_interval=0.01, cloud_ready_timeout=5.0)
