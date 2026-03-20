# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidated Platform protocol tests.

Tests exercise the Platform protocol contract through parameterized fixtures
(GCP via GcpServiceImpl DRY_RUN, Manual). Platform-specific behavioral
tests that cannot be expressed through the protocol are in dedicated sections.
"""

from __future__ import annotations

import unittest.mock
from collections.abc import Iterator
from dataclasses import dataclass

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
from iris.cluster.platform.gcp_service_impl import GcpServiceImpl
from iris.cluster.platform.manual import ManualPlatform
from iris.cluster.platform.service_mode import ServiceMode
from iris.rpc import config_pb2

# =============================================================================
# Fixture infrastructure
# =============================================================================


@dataclass
class PlatformEnv:
    """Test environment for a single Platform implementation."""

    platform: GcpPlatform | ManualPlatform
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
            accelerator_variant="v5litepod-8",
        )
        cfg.gcp.zone = env.zone
        cfg.gcp.runtime_version = "tpu-ubuntu2204-base"
        cfg.labels[labels.iris_managed] = "true"
        cfg.labels[labels.iris_scale_group] = group_name
        return cfg
    else:
        cfg = config_pb2.SliceConfig(name_prefix=f"iris-{group_name}", num_vms=1)
        cfg.manual.CopyFrom(config_pb2.ManualSliceConfig())
        cfg.labels[labels.iris_managed] = "true"
        cfg.labels[labels.iris_scale_group] = group_name
        return cfg


def _make_vm_config(env: PlatformEnv, name: str = "test-controller") -> config_pb2.VmConfig:
    """Build a VmConfig appropriate for the platform under test."""
    cfg = config_pb2.VmConfig(name=name)
    if env.name == "gcp":
        cfg.gcp.zone = env.zone
        cfg.gcp.machine_type = "n2-standard-4"
    else:
        cfg.manual.CopyFrom(config_pb2.ManualVmConfig())
    return cfg


@pytest.fixture(params=["gcp", "manual"])
def platform_env(request) -> Iterator[PlatformEnv]:
    """Yield a PlatformEnv for each platform implementation.

    GCP is backed by GcpServiceImpl in DRY_RUN mode.
    """
    name = request.param

    if name == "gcp":
        gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")
        gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
        platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)
        yield PlatformEnv(platform=platform, zone="us-central2-b", name="gcp", label_prefix="iris")
    else:
        platform = ManualPlatform(
            label_prefix="iris",
            hosts=["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4", "10.0.0.5"],
        )
        yield PlatformEnv(platform=platform, zone="manual", name="manual", label_prefix="iris")


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
# so we test them on ManualPlatform which is fully in-memory.


def test_terminate_then_status_is_deleting():
    """After terminate(), slice status reports DELETING."""
    platform = ManualPlatform(label_prefix="iris", hosts=["10.0.0.1", "10.0.0.2", "10.0.0.3"])
    cfg = config_pb2.SliceConfig(name_prefix="iris-term-group", num_vms=2)
    cfg.manual.CopyFrom(config_pb2.ManualSliceConfig())
    cfg.labels[Labels("iris").iris_managed] = "true"
    cfg.labels[Labels("iris").iris_scale_group] = "term-group"
    handle = platform.create_slice(cfg)

    handle.terminate()
    assert handle.describe().state == CloudSliceState.DELETING


def test_tunnel_returns_address_directly():
    """tunnel() is a passthrough on ManualPlatform (no SSH)."""
    platform = ManualPlatform(label_prefix="iris", hosts=["10.0.0.1"])
    addr = "http://10.0.0.1:10000"
    with platform.tunnel(addr) as tunneled:
        assert tunneled == addr


# =============================================================================
# Section 2: GCP-Specific Tests
#
# Behaviors that only apply to GcpPlatform.
# =============================================================================


def test_gcp_quota_error_raises_quota_exhausted():
    """GcpPlatform raises QuotaExhaustedError when service reports quota exhaustion."""
    gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_service.inject_failure("tpu_create", QuotaExhaustedError("RESOURCE_EXHAUSTED: no capacity"))

    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu-group",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-8",
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"

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
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
    cfg.gcp.machine_type = "n2-standard-4"

    with pytest.raises(ValueError, match="num_vms=1"):
        _validate_slice_config(cfg)


def test_gcp_create_vm_slice_mode_produces_single_worker_slice():
    """VM slice mode creates a single-worker slice that is discoverable and terminable."""
    gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)

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

    handle = platform.create_slice(cfg)
    status = handle.describe()
    assert status.worker_count == 1
    assert len(status.workers) == 1
    assert status.workers[0].internal_address
    assert handle.scale_group == "cpu-vm"

    listed = platform.list_all_slices()
    assert handle.slice_id in {s.slice_id for s in listed}

    handle.terminate()
    listed_after = platform.list_all_slices()
    assert handle.slice_id not in {s.slice_id for s in listed_after}


def test_gcp_build_vm_slice_id_bounds_and_normalizes():
    suffix = "20260307-1755-a3b1c9d2"
    slice_id = _build_vm_slice_id(
        "smoke-cpu_vm_e2_standard_4_ondemand-europe-west4-b",
        suffix,
    )
    assert len(slice_id) <= 63
    assert "_" not in slice_id
    assert slice_id.endswith(f"-{suffix}")


def test_gcp_create_vm_slice_mode_with_long_prefix_uses_valid_slice_id():
    gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)

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

    handle = platform.create_slice(cfg)
    assert len(handle.slice_id) <= 63
    assert "_" not in handle.slice_id
    status = handle.describe()
    assert len(status.workers) == 1
    assert getattr(status.workers[0]._remote_exec, "ssh_user", None) == "iris"
    listed = platform.list_all_slices()
    assert handle.slice_id in {s.slice_id for s in listed}


def test_gcp_empty_accelerator_variant_rejected():
    """create_slice with empty accelerator_variant raises ValueError."""
    gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu-group",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"

    with pytest.raises(ValueError, match="accelerator_variant"):
        platform.create_slice(cfg)


def test_gcp_create_vm_validates_config():
    """create_vm with empty zone raises ValueError."""
    gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    cfg = config_pb2.VmConfig(name="test-vm")

    with pytest.raises(ValueError, match="zone"):
        platform.create_vm(cfg)


def test_gcp_list_slices_skips_deleting_tpus():
    """list_slices omits TPUs in DELETING state."""
    gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-8",
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"
    cfg.labels[Labels("iris").iris_managed] = "true"

    handle = platform.create_slice(cfg)

    # Mark the TPU as DELETING in the service's in-memory state
    for _key, tpu_info in gcp_service._tpus.items():
        if tpu_info.name == handle.slice_id:
            tpu_info.state = "DELETING"
            break

    slices = platform.list_slices(zones=["us-central2-b"])
    slice_ids = {s.slice_id for s in slices}
    assert handle.slice_id not in slice_ids


def test_gcp_create_slice_resolves_ghcr_image_in_worker_config():
    """create_slice rewrites GHCR images in worker_config via resolve_image."""
    gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="my-proj")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="my-proj")
    platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-8",
    )
    cfg.gcp.zone = "europe-west4-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"

    wc = config_pb2.WorkerConfig(
        docker_image="ghcr.io/marin-community/iris-worker:latest",
        port=10001,
        controller_address="controller:10000",
        cache_dir="/var/cache/iris",
    )

    with unittest.mock.patch("iris.cluster.platform.gcp.threading.Thread"):
        platform.create_slice(cfg, worker_config=wc)

    assert wc.docker_image == "europe-docker.pkg.dev/my-proj/ghcr-mirror/marin-community/iris-worker:latest"


def test_gcp_list_slices_skips_inactive_vm_instances():
    """list_slices omits VM-backed slices for instances in inactive states."""
    gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)

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

    handle = platform.create_slice(cfg)
    for _key, vm_info in gcp_service._vms.items():
        if vm_info.labels.get(Labels("iris").iris_slice_id) == handle.slice_id:
            vm_info.status = "TERMINATED"
            break

    listed = platform.list_all_slices()
    assert handle.slice_id not in {s.slice_id for s in listed}


def test_gcp_list_slices_preserves_vm_slice_discovery():
    """VM-backed slices are discoverable via list_all_slices after creation."""
    gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)

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

    handle = platform.create_slice(cfg)
    listed = platform.list_all_slices()
    listed_by_id = {s.slice_id: s for s in listed}
    assert handle.slice_id in listed_by_id
    assert listed_by_id[handle.slice_id].created_at.epoch_ms() > 0


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
# Section 5: list_all_slices Tests
# =============================================================================


def test_list_all_slices_returns_created_slices(platform_env: PlatformEnv):
    """list_all_slices discovers slices without caller specifying zones."""
    cfg = _make_slice_config(platform_env, "group-a")
    handle = platform_env.platform.create_slice(cfg)
    all_slices = platform_env.platform.list_all_slices()
    assert handle.slice_id in {s.slice_id for s in all_slices}


def test_list_all_slices_returns_all_managed(platform_env: PlatformEnv):
    """list_all_slices returns all managed slices regardless of scale group."""
    cfg_a = _make_slice_config(platform_env, "group-a")
    cfg_b = _make_slice_config(platform_env, "group-b")
    platform_env.platform.create_slice(cfg_a)
    platform_env.platform.create_slice(cfg_b)

    all_slices = platform_env.platform.list_all_slices()
    assert len(all_slices) == 2


def test_gcp_list_all_slices_multi_zone():
    """GcpPlatform.list_all_slices returns slices across multiple zones."""
    gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")
    # Add synthetic zones that aren't in KNOWN_GCP_ZONES
    gcp_service._valid_zones.update(["zone-a", "zone-b"])
    gcp_config = config_pb2.GcpPlatformConfig(
        project_id="test-project",
        zones=["zone-a", "zone-b"],
    )
    platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    iris_labels = Labels("iris")
    cfg_a = config_pb2.SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-8",
    )
    cfg_a.gcp.zone = "zone-a"
    cfg_a.gcp.runtime_version = "tpu-ubuntu2204-base"
    cfg_a.labels[iris_labels.iris_managed] = "true"

    cfg_b = config_pb2.SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-8",
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


# =============================================================================
# Section 6: GCE VM Slice Bootstrap via Startup-Script
#
# Tests for the startup-script metadata bootstrap path (no SSH for bootstrap).
# =============================================================================


def test_gcp_vm_slice_bootstrap_monitors_serial_port():
    """VM slice bootstrap waits for startup-script completion via serial port monitoring."""
    gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
    cfg.gcp.machine_type = "n2-standard-4"

    wc = config_pb2.WorkerConfig(
        docker_image="test-image:latest",
        port=10001,
        controller_address="controller:10000",
        cache_dir="/var/cache/iris",
    )

    handle = platform.create_slice(cfg, worker_config=wc)

    serial_output = (
        "[iris-init] Starting Iris worker bootstrap\n"
        "[iris-init] Phase: prerequisites\n"
        "[iris-init] Docker installed\n"
        "[iris-init] Phase: docker_pull\n"
        "[iris-init] Worker container started\n"
        "[iris-init] Worker is healthy\n"
        "[iris-init] Bootstrap complete\n"
    )

    with unittest.mock.patch.object(gcp_service, "vm_get_serial_port_output", return_value=serial_output):
        platform._run_vm_slice_bootstrap(handle, wc, poll_interval=0.01, cloud_ready_timeout=5.0)

    with handle._bootstrap_lock:
        assert handle._bootstrap_state == CloudSliceState.READY


def test_gcp_vm_slice_bootstrap_detects_startup_script_failure():
    """VM slice bootstrap raises on [iris-init] ERROR in serial output."""
    gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
    cfg.gcp.machine_type = "n2-standard-4"

    wc = config_pb2.WorkerConfig(
        docker_image="test-image:latest",
        port=10001,
        controller_address="controller:10000",
        cache_dir="/var/cache/iris",
    )

    handle = platform.create_slice(cfg, worker_config=wc)

    serial_output = (
        "[iris-init] Starting Iris worker bootstrap\n" "[iris-init] ERROR: Worker container exited unexpectedly\n"
    )

    with (
        unittest.mock.patch.object(gcp_service, "vm_get_serial_port_output", return_value=serial_output),
        pytest.raises(PlatformError, match="bootstrap failed"),
    ):
        platform._run_vm_slice_bootstrap(handle, wc, poll_interval=0.01, cloud_ready_timeout=5.0)


# =============================================================================
# Section 7: TPU Slice Bootstrap via Startup-Script + Health Polling
#
# Tests for the startup-script metadata bootstrap path for TPUs.
# =============================================================================


def test_gcp_tpu_slice_passes_startup_script_metadata():
    """_create_tpu_slice with worker_config embeds startup-script in TPU metadata."""
    gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-8",
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"

    wc = config_pb2.WorkerConfig(
        docker_image="test-image:latest",
        port=10001,
        controller_address="controller:10000",
        cache_dir="/var/cache/iris",
    )

    with unittest.mock.patch("iris.cluster.platform.gcp.threading.Thread"):
        platform.create_slice(cfg, worker_config=wc)

    # Verify the service's in-memory TPU has startup-script metadata with [iris-init] markers.
    tpu_entries = list(gcp_service._tpus.values())
    assert len(tpu_entries) == 1
    metadata = tpu_entries[0].metadata
    assert "startup-script" in metadata
    assert "[iris-init]" in metadata["startup-script"]
    assert "test-image:latest" in metadata["startup-script"]


def test_gcp_tpu_bootstrap_monitors_health_endpoints():
    """_run_tpu_bootstrap detects all workers healthy via health endpoint polling."""
    gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-8",
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"

    wc = config_pb2.WorkerConfig(
        docker_image="test-image:latest",
        port=10001,
        controller_address="controller:10000",
        cache_dir="/var/cache/iris",
    )

    handle = platform.create_slice(cfg)
    handle._bootstrap_state = None

    mock_resp = unittest.mock.MagicMock()
    mock_resp.status = 200
    with unittest.mock.patch("iris.cluster.platform.gcp.urllib.request.urlopen", return_value=mock_resp):
        platform._run_tpu_bootstrap(handle, wc, poll_interval=0.01, cloud_ready_timeout=5.0, bootstrap_timeout=5.0)

    with handle._bootstrap_lock:
        assert handle._bootstrap_state == CloudSliceState.READY


def test_gcp_tpu_bootstrap_timeout_fetches_cloud_logs():
    """_run_tpu_bootstrap on timeout raises PlatformError and fetches Cloud Logging."""
    gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-8",
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"

    wc = config_pb2.WorkerConfig(
        docker_image="test-image:latest",
        port=10001,
        controller_address="controller:10000",
        cache_dir="/var/cache/iris",
    )

    handle = platform.create_slice(cfg)
    handle._bootstrap_state = None

    with (
        unittest.mock.patch(
            "iris.cluster.platform.gcp.urllib.request.urlopen",
            side_effect=ConnectionRefusedError("Connection refused"),
        ),
        unittest.mock.patch.object(platform, "_fetch_bootstrap_logs"),
        pytest.raises(PlatformError, match=r"bootstrap timed out.*0/1 workers healthy"),
    ):
        platform._run_tpu_bootstrap(handle, wc, poll_interval=0.01, cloud_ready_timeout=5.0, bootstrap_timeout=0.05)


def test_gcp_tpu_bootstrap_partial_healthy():
    """Multi-VM TPU where only some workers become healthy reports correct count."""
    gcp_service = GcpServiceImpl(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpPlatform(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    # v4-32 has vm_count=4
    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v4-32",
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"

    wc = config_pb2.WorkerConfig(
        docker_image="test-image:latest",
        port=10001,
        controller_address="controller:10000",
        cache_dir="/var/cache/iris",
    )

    handle = platform.create_slice(cfg)
    handle._bootstrap_state = None

    # GcpServiceImpl creates 4 endpoints for v4-32 (10.0.0.0..10.0.0.3).
    # Only the first worker responds healthy; the rest refuse connections.
    first_ip = gcp_service._tpus[next(iter(gcp_service._tpus))].network_endpoints[0]

    def _selective_urlopen(url, timeout=None):
        if first_ip in url:
            mock_resp = unittest.mock.MagicMock()
            mock_resp.status = 200
            return mock_resp
        raise ConnectionRefusedError("Connection refused")

    with (
        unittest.mock.patch(
            "iris.cluster.platform.gcp.urllib.request.urlopen",
            side_effect=_selective_urlopen,
        ),
        unittest.mock.patch.object(platform, "_fetch_bootstrap_logs"),
        pytest.raises(PlatformError, match="1/4 workers healthy"),
    ):
        platform._run_tpu_bootstrap(handle, wc, poll_interval=0.01, cloud_ready_timeout=5.0, bootstrap_timeout=0.05)
