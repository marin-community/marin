# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidated Platform protocol tests.

Tests exercise the Platform protocol contract through parameterized fixtures
(GCP via InMemoryGcpService DRY_RUN, Manual). Platform-specific behavioral
tests that cannot be expressed through the protocol are in dedicated sections.
"""

import unittest.mock
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from iris.cluster.config import (
    GcpPlatformConfig,
    GcpSliceConfig,
    GcpVmConfig,
    ManualSliceConfig,
    ManualVmConfig,
    SliceConfig,
    SshConfig,
    VmConfig,
    WorkerConfig,
)
from iris.cluster.platforms.gcp.controller import GcpControllerProvider
from iris.cluster.platforms.gcp.fake import InMemoryGcpService
from iris.cluster.platforms.gcp.handles import (
    GcpSliceHandle,
    GcpVmSliceHandle,
    _build_gce_resource_name,
    _composite_slice_state,
)
from iris.cluster.platforms.gcp.service import OperationStatus, TpuCreateRequest, VmCreateRequest
from iris.cluster.platforms.gcp.workers import (
    GcpWorkerProvider,
    _run_tpu_bootstrap,
    _run_vm_slice_bootstrap,
    _spawn_bootstrap_thread,
    _validate_slice_config,
)
from iris.cluster.platforms.manual.controller import ManualControllerProvider
from iris.cluster.platforms.manual.workers import ManualWorkerProvider
from iris.cluster.platforms.remote_exec import GceRemoteExec, GcloudRemoteExec
from iris.cluster.platforms.types import (
    CloudSliceState,
    InfraError,
    Labels,
    QuotaExhaustedError,
)
from iris.cluster.service_mode import ServiceMode
from iris.cluster.tpu_topology import get_tpu_topology
from iris.cluster.types import AcceleratorType, CapacityType, GcpSliceMode
from rigging.timing import Timestamp

# =============================================================================
# Fixture infrastructure
# =============================================================================


@dataclass
class PlatformEnv:
    """Test environment for a single Platform implementation."""

    platform: GcpWorkerProvider | ManualWorkerProvider
    zone: str
    name: str
    label_prefix: str


def _make_slice_config(env: PlatformEnv, group_name: str) -> SliceConfig:
    """Build a SliceConfig appropriate for the platform under test."""
    labels = Labels(env.label_prefix)

    if env.name == "gcp":
        cfg = SliceConfig(
            name_prefix=f"iris-{group_name}",
            accelerator_type=AcceleratorType.TPU,
            accelerator_variant="v5litepod-8",
            gcp=GcpSliceConfig(zone=env.zone, runtime_version="tpu-ubuntu2204-base"),
        )
        cfg.labels[labels.iris_managed] = "true"
        cfg.labels[labels.iris_scale_group] = group_name
        return cfg
    else:
        cfg = SliceConfig(name_prefix=f"iris-{group_name}", num_vms=1, manual=ManualSliceConfig())
        cfg.labels[labels.iris_managed] = "true"
        cfg.labels[labels.iris_scale_group] = group_name
        return cfg


def _make_vm_config(env: PlatformEnv, name: str = "test-controller") -> VmConfig:
    """Build a VmConfig appropriate for the platform under test."""
    if env.name == "gcp":
        return VmConfig(name=name, gcp=GcpVmConfig(zone=env.zone, machine_type="n2-standard-4"))
    return VmConfig(name=name, manual=ManualVmConfig())


@pytest.fixture(params=["gcp", "manual"])
def platform_env(request) -> Iterator[PlatformEnv]:
    """Yield a PlatformEnv for each platform implementation.

    GCP is backed by InMemoryGcpService in DRY_RUN mode.
    """
    name = request.param

    if name == "gcp":
        gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
        gcp_config = GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
        platform = GcpWorkerProvider(gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)
        yield PlatformEnv(platform=platform, zone="us-central2-b", name="gcp", label_prefix="iris")
    else:
        platform = ManualWorkerProvider(
            label_prefix="iris",
            worker_port=10001,
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
# so we test them on ManualWorkerProvider which is fully in-memory.


def test_terminate_then_status_is_deleting():
    """After terminate(), slice status reports DELETING."""
    platform = ManualWorkerProvider(label_prefix="iris", worker_port=10001, hosts=["10.0.0.1", "10.0.0.2", "10.0.0.3"])
    cfg = SliceConfig(name_prefix="iris-term-group", num_vms=2, manual=ManualSliceConfig())
    cfg.labels[Labels("iris").iris_managed] = "true"
    cfg.labels[Labels("iris").iris_scale_group] = "term-group"
    handle = platform.create_slice(cfg)

    handle.terminate()
    assert handle.describe().state == CloudSliceState.DELETING


def test_tunnel_returns_address_directly():
    """tunnel() is a passthrough on ManualControllerProvider (no SSH)."""
    worker_provider = ManualWorkerProvider(label_prefix="iris", worker_port=10001, hosts=["10.0.0.1"])
    controller = ManualControllerProvider(worker_provider=worker_provider)
    addr = "http://10.0.0.1:10000"
    with controller.tunnel(addr) as tunneled:
        assert tunneled == addr


def _register_controller_vm(gcp_service: InMemoryGcpService, *, os_login: bool, zone: str = "us-central2-b") -> None:
    """Register a controller VM in the in-memory service for tunnel tests."""
    metadata = {"enable-oslogin": "TRUE", "block-project-ssh-keys": "TRUE"} if os_login else {}
    gcp_service.vm_create(
        VmCreateRequest(
            name="iris-controller-iris",
            zone=zone,
            machine_type="n2-standard-4",
            labels={Labels("iris").iris_controller: "true"},
            metadata=metadata,
        )
    )
    # InMemoryGcpService creates VMs in PROVISIONING; the tunnel filters for RUNNING.
    gcp_service._vms[("iris-controller-iris", zone)].status = "RUNNING"


def test_gcp_tunnel_prefers_ssh_impersonation_config():
    """Tunnel passes --impersonate-service-account through; gcloud picks user/key itself."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    _register_controller_vm(gcp_service, os_login=True)

    gcp_config = GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    ssh_config = SshConfig(
        impersonate_service_account="iris-controller@test-project.iam.gserviceaccount.com",
    )
    worker_provider = GcpWorkerProvider(
        gcp_config, label_prefix="iris", worker_port=10001, ssh_config=ssh_config, gcp_service=gcp_service
    )
    controller = GcpControllerProvider(
        worker_provider=worker_provider,
        controller_service_account="iris-worker@test-project.iam.gserviceaccount.com",
    )

    ssh_proc = unittest.mock.Mock()
    ssh_proc.poll.return_value = None
    ssh_proc.terminate.return_value = None
    ssh_proc.wait.return_value = 0

    with (
        unittest.mock.patch("iris.cluster.platforms.gcp.controller._check_gcloud_ssh_key"),
        unittest.mock.patch("iris.cluster.platforms.gcp.controller.find_free_port", return_value=10042),
        unittest.mock.patch("iris.cluster.platforms.gcp.controller.wait_for_port"),
        unittest.mock.patch(
            "iris.cluster.platforms.gcp.controller.subprocess.Popen", return_value=ssh_proc
        ) as popen_mock,
    ):
        with controller.tunnel("unused") as tunneled:
            assert tunneled == "http://127.0.0.1:10042"

    ssh_cmd = popen_mock.call_args.args[0]
    assert f"--impersonate-service-account={ssh_config.impersonate_service_account}" in ssh_cmd
    # No explicit user@vm prefix or --ssh-key-file: gcloud auto-detects.
    assert "iris-controller-iris" in ssh_cmd
    assert "@iris-controller-iris" not in " ".join(ssh_cmd)
    assert not any("--ssh-key-file" in arg for arg in ssh_cmd)


def test_gce_remote_exec_builds_optional_flags_inline():
    remote_exec = GceRemoteExec(
        project_id="test-project",
        zone="us-central1-a",
        vm_name="vm-1",
        ssh_key_file="/tmp/test-key",
        impersonate_service_account="svc@test-project.iam.gserviceaccount.com",
    )

    cmd = remote_exec._build_cmd("echo ok")

    assert cmd[:4] == ["gcloud", "compute", "ssh", "vm-1"]
    assert "--ssh-key-file=/tmp/test-key" in cmd
    assert "--impersonate-service-account=svc@test-project.iam.gserviceaccount.com" in cmd
    assert cmd[-2:] == ["--command", "echo ok"]


def test_gcloud_remote_exec_builds_optional_flags_inline():
    remote_exec = GcloudRemoteExec(
        project_id="test-project",
        _zone="us-west4-a",
        vm_id="slice-1",
        worker_index=0,
        ssh_key_file="/tmp/test-key",
        impersonate_service_account="svc@test-project.iam.gserviceaccount.com",
    )

    cmd = remote_exec._build_cmd("echo ok")

    assert cmd[:6] == ["gcloud", "compute", "tpus", "tpu-vm", "ssh", "slice-1"]
    assert "--ssh-key-file=/tmp/test-key" in cmd
    assert "--impersonate-service-account=svc@test-project.iam.gserviceaccount.com" in cmd
    assert cmd[-2:] == ["--command", "echo ok"]


# =============================================================================
# Section 2: GCP-Specific Tests
#
# Behaviors that only apply to GcpWorkerProvider.
# =============================================================================


def test_gcp_quota_error_raises_quota_exhausted():
    """GcpWorkerProvider raises QuotaExhaustedError when service reports quota exhaustion."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_service.inject_failure("tpu_create", QuotaExhaustedError("RESOURCE_EXHAUSTED: no capacity"))

    gcp_config = GcpPlatformConfig(project_id="test-project")
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)

    cfg = SliceConfig(
        name_prefix="iris-tpu-group",
        accelerator_type=AcceleratorType.TPU,
        accelerator_variant="v5litepod-8",
        gcp=GcpSliceConfig(zone="us-central2-b", runtime_version="tpu-ubuntu2204-base"),
    )

    with pytest.raises(QuotaExhaustedError):
        platform.create_slice(cfg)


def test_gcp_validate_slice_config_reports_all_missing_fields():
    """_validate_slice_config reports all missing fields at once."""
    cfg = SliceConfig(name_prefix="test", gcp=GcpSliceConfig())

    with pytest.raises(ValueError, match="accelerator_variant") as exc_info:
        _validate_slice_config(cfg)
    assert "zone" in str(exc_info.value)
    assert "runtime_version" in str(exc_info.value)


def test_gcp_validate_vm_slice_config_requires_machine_type():
    """VM slice mode requires gcp.machine_type."""
    cfg = SliceConfig(
        name_prefix="test",
        num_vms=1,
        gcp=GcpSliceConfig(zone="us-central2-b", mode=GcpSliceMode.VM),
    )

    with pytest.raises(ValueError, match=r"gcp\.machine_type"):
        _validate_slice_config(cfg)


def test_gcp_validate_vm_slice_config_rejects_non_on_demand():
    """VM slice mode rejects non-on-demand capacity types."""
    cfg = SliceConfig(
        name_prefix="test",
        num_vms=1,
        capacity_type=CapacityType.PREEMPTIBLE,
        gcp=GcpSliceConfig(zone="us-central2-b", mode=GcpSliceMode.VM, machine_type="n2-standard-4"),
    )

    with pytest.raises(ValueError, match="only supports capacity_type on-demand"):
        _validate_slice_config(cfg)


def test_gcp_validate_vm_slice_config_rejects_num_vms_not_one():
    """VM slice mode requires exactly one VM."""
    cfg = SliceConfig(
        name_prefix="test",
        num_vms=2,
        gcp=GcpSliceConfig(zone="us-central2-b", mode=GcpSliceMode.VM, machine_type="n2-standard-4"),
    )

    with pytest.raises(ValueError, match="num_vms=1"):
        _validate_slice_config(cfg)


def test_gcp_create_vm_slice_mode_produces_single_worker_slice():
    """VM slice mode creates a single-worker slice that is discoverable and terminable."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)

    cfg = SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=AcceleratorType.CPU,
        capacity_type=CapacityType.ON_DEMAND,
        gcp=GcpSliceConfig(zone="us-central2-b", mode=GcpSliceMode.VM, machine_type="n2-standard-4"),
    )
    cfg.labels[Labels("iris").iris_managed] = "true"
    cfg.labels[Labels("iris").iris_scale_group] = "cpu-vm"

    handle = platform.create_slice(cfg)
    status = handle.describe()
    assert status.worker_count == 1
    assert len(status.workers) == 1
    assert status.workers[0].internal_address
    assert handle.scale_group == "cpu-vm"

    listed = platform.list_all_slices()
    assert handle.slice_id in {s.handle.slice_id for s in listed}

    handle.terminate()
    listed_after = platform.list_all_slices()
    assert handle.slice_id not in {s.handle.slice_id for s in listed_after}


def test_gcp_build_gce_resource_name_bounds_and_normalizes():
    suffix = "20260307-1755-a3b1c9d2"
    slice_id = _build_gce_resource_name(
        "smoke-cpu_vm_e2_standard_4_ondemand-europe-west4-b",
        suffix,
    )
    assert len(slice_id) <= 63
    assert "_" not in slice_id
    assert slice_id.endswith(f"-{suffix}")


def test_gcp_create_vm_slice_mode_with_long_prefix_uses_valid_slice_id():
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)

    cfg = SliceConfig(
        name_prefix="smoke-cpu_vm_e2_standard_4_ondemand-europe-west4-b",
        num_vms=1,
        accelerator_type=AcceleratorType.CPU,
        capacity_type=CapacityType.ON_DEMAND,
        gcp=GcpSliceConfig(zone="us-central2-b", mode=GcpSliceMode.VM, machine_type="n2-standard-4"),
    )
    cfg.labels[Labels("iris").iris_managed] = "true"
    cfg.labels[Labels("iris").iris_scale_group] = "cpu-vm"

    handle = platform.create_slice(cfg)
    assert len(handle.slice_id) <= 63
    assert "_" not in handle.slice_id
    status = handle.describe()
    assert len(status.workers) == 1
    assert isinstance(status.workers[0]._remote_exec, GceRemoteExec)
    listed = platform.list_all_slices()
    assert handle.slice_id in {s.handle.slice_id for s in listed}


def test_gcp_vm_slice_sets_os_login_metadata_unconditionally():
    """Every VM slice gets enable-oslogin metadata; the GceRemoteExec carries no user/key."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)

    cfg = SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=AcceleratorType.CPU,
        capacity_type=CapacityType.ON_DEMAND,
        gcp=GcpSliceConfig(zone="us-central2-b", mode=GcpSliceMode.VM, machine_type="n2-standard-4"),
    )
    cfg.labels[Labels("iris").iris_managed] = "true"
    cfg.labels[Labels("iris").iris_scale_group] = "cpu-vm"

    handle = platform.create_slice(cfg)
    status = handle.describe()
    vm = next(iter(gcp_service._vms.values()))
    assert vm.metadata["enable-oslogin"] == "TRUE"
    assert vm.metadata["block-project-ssh-keys"] == "TRUE"
    remote_exec = status.workers[0]._remote_exec
    assert isinstance(remote_exec, GceRemoteExec)
    assert remote_exec.ssh_user is None
    assert remote_exec.ssh_key_file is None


def test_gcp_vm_slice_omits_impersonation_when_unset():
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)

    cfg = SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=AcceleratorType.CPU,
        capacity_type=CapacityType.ON_DEMAND,
        gcp=GcpSliceConfig(
            zone="us-central2-b",
            mode=GcpSliceMode.VM,
            machine_type="n2-standard-4",
            service_account="iris-worker@test-project.iam.gserviceaccount.com",
        ),
    )

    handle = platform.create_slice(cfg)
    status = handle.describe()
    assert status.workers[0]._remote_exec.impersonate_service_account is None


def test_gcp_vm_slice_propagates_explicit_ssh_impersonation_account():
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    ssh_config = SshConfig(
        impersonate_service_account="iris-controller@test-project.iam.gserviceaccount.com",
    )
    platform = GcpWorkerProvider(
        gcp_config, label_prefix="iris", worker_port=10001, ssh_config=ssh_config, gcp_service=gcp_service
    )

    cfg = SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=AcceleratorType.CPU,
        capacity_type=CapacityType.ON_DEMAND,
        gcp=GcpSliceConfig(
            zone="us-central2-b",
            mode=GcpSliceMode.VM,
            machine_type="n2-standard-4",
            service_account="iris-worker@test-project.iam.gserviceaccount.com",
        ),
    )

    handle = platform.create_slice(cfg)
    status = handle.describe()

    assert status.workers[0]._remote_exec.impersonate_service_account == ssh_config.impersonate_service_account


def test_gcp_empty_accelerator_variant_rejected():
    """create_slice with empty accelerator_variant raises ValueError."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = GcpPlatformConfig(project_id="test-project")
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)

    cfg = SliceConfig(
        name_prefix="iris-tpu-group",
        accelerator_type=AcceleratorType.TPU,
        gcp=GcpSliceConfig(zone="us-central2-b", runtime_version="tpu-ubuntu2204-base"),
    )

    with pytest.raises(ValueError, match="accelerator_variant"):
        platform.create_slice(cfg)


def test_gcp_create_vm_validates_config():
    """create_vm with empty zone raises ValueError."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = GcpPlatformConfig(project_id="test-project")
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)

    cfg = VmConfig(name="test-vm", gcp=GcpVmConfig())

    with pytest.raises(ValueError, match="zone"):
        platform.create_vm(cfg)


def test_gcp_list_slices_skips_deleting_tpus():
    """list_slices omits TPUs in DELETING state."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = GcpPlatformConfig(project_id="test-project")
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)

    cfg = SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=AcceleratorType.TPU,
        accelerator_variant="v5litepod-8",
        gcp=GcpSliceConfig(zone="us-central2-b", runtime_version="tpu-ubuntu2204-base"),
    )
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


def test_describe_resolves_topology_from_live_tpu_when_handle_variant_empty():
    """describe() sizes a slice from the live TPU's accelerator_type, not the handle's variant.

    Queued-resource (reserved TPU) handles adopted during boot recovery carry an empty
    accelerator_variant — the QR API reports no topology — and that handle is never refreshed
    once the backing TPU VM provisions. describe() must still resolve the worker count from
    the live tpu_info rather than raising on the stale empty variant.
    """
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    slice_id = "iris-tpu-v4-reserved-32-us-central2-b"
    zone = "us-central2-b"
    gcp_service.tpu_create(
        TpuCreateRequest(
            name=slice_id,
            zone=zone,
            accelerator_type="v4-32",
            runtime_version="tpu-ubuntu2204-base",
            capacity_type=CapacityType.RESERVED,
            labels={Labels("iris").iris_managed: "true"},
        )
    )

    handle = GcpSliceHandle(
        _slice_id=slice_id,
        _zone=zone,
        _project_id="test-project",
        _labels={Labels("iris").iris_managed: "true"},
        _created_at=Timestamp.from_ms(0),
        _label_prefix="iris",
        _worker_port=10001,
        _accelerator_variant="",  # adopted queued-resource handle has no variant
        _gcp_service=gcp_service,
        _is_queued_resource=True,
    )

    status = handle.describe()
    assert status.worker_count == get_tpu_topology("v4-32").vm_count
    assert len(status.workers) == status.worker_count


def test_gcp_create_slice_resolves_ghcr_image_in_worker_config():
    """create_slice rewrites mirrored images in worker_config via resolve_image."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="my-proj")
    gcp_config = GcpPlatformConfig(
        project_id="my-proj",
        registry_mirrors={"ghcr.io": {"europe": "europe-docker.pkg.dev/my-proj/ghcr-mirror"}},
    )
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)

    cfg = SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=AcceleratorType.TPU,
        accelerator_variant="v5litepod-8",
        gcp=GcpSliceConfig(zone="europe-west4-b", runtime_version="tpu-ubuntu2204-base"),
    )

    wc = WorkerConfig(
        docker_image="ghcr.io/marin-community/iris-worker:latest",
        port=10001,
        controller_address="controller:10000",
        cache_dir="/var/cache/iris",
    )

    with unittest.mock.patch("iris.cluster.platforms.gcp.workers.threading.Thread"):
        platform.create_slice(cfg, worker_config=wc)

    assert wc.docker_image == "europe-docker.pkg.dev/my-proj/ghcr-mirror/marin-community/iris-worker:latest"


def test_gcp_list_all_slices_includes_terminated_vm_instances():
    """list_all_slices surfaces VM-backed slices in non-live states so the boot
    reconciler can reclaim them. list_slices (live discovery) still filters."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)

    cfg = SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=AcceleratorType.CPU,
        capacity_type=CapacityType.ON_DEMAND,
        gcp=GcpSliceConfig(zone="us-central2-b", mode=GcpSliceMode.VM, machine_type="n2-standard-4"),
    )
    cfg.labels[Labels("iris").iris_managed] = "true"
    cfg.labels[Labels("iris").iris_scale_group] = "cpu-vm"

    handle = platform.create_slice(cfg)
    for _key, vm_info in gcp_service._vms.items():
        if vm_info.labels.get(Labels("iris").iris_slice_id) == handle.slice_id:
            vm_info.status = "TERMINATED"
            break

    listed = platform.list_all_slices()
    by_id = {s.handle.slice_id: s for s in listed}
    assert handle.slice_id in by_id
    assert by_id[handle.slice_id].state == CloudSliceState.DELETING

    live = platform.list_slices(zones=["us-central2-b"])
    assert handle.slice_id not in {s.slice_id for s in live}


def test_gcp_list_slices_preserves_vm_slice_discovery():
    """VM-backed slices are discoverable via list_all_slices after creation."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)

    cfg = SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=AcceleratorType.CPU,
        capacity_type=CapacityType.ON_DEMAND,
        gcp=GcpSliceConfig(zone="us-central2-b", mode=GcpSliceMode.VM, machine_type="n2-standard-4"),
    )
    cfg.labels[Labels("iris").iris_managed] = "true"
    cfg.labels[Labels("iris").iris_scale_group] = "cpu-vm"

    handle = platform.create_slice(cfg)
    listed = platform.list_all_slices()
    listed_by_id = {s.handle.slice_id: s.handle for s in listed}
    assert handle.slice_id in listed_by_id
    assert listed_by_id[handle.slice_id].created_at.epoch_ms() > 0


# =============================================================================
# Section 3: Manual-Specific Tests
#
# Host pool management, exclusivity, and reallocation.
# =============================================================================


def test_manual_host_pool_exhaustion_raises():
    """create_slice raises when not enough hosts are available."""
    platform = ManualWorkerProvider(label_prefix="iris", worker_port=10001, hosts=["10.0.0.1"])
    cfg = SliceConfig(name_prefix="iris-group", num_vms=3)
    cfg.manual = ManualSliceConfig()

    with pytest.raises(RuntimeError, match="Need 3 hosts but only 1 available"):
        platform.create_slice(cfg)


def test_manual_host_exclusivity():
    """A host allocated to one VM cannot be allocated to another."""
    platform = ManualWorkerProvider(label_prefix="iris", worker_port=10001, hosts=["10.0.0.1", "10.0.0.2"])

    cfg1 = VmConfig(name="ctrl-1", manual=ManualVmConfig(host="10.0.0.1"))
    platform.create_vm(cfg1)

    cfg2 = VmConfig(name="ctrl-2", manual=ManualVmConfig(host="10.0.0.1"))
    with pytest.raises(RuntimeError, match="already allocated"):
        platform.create_vm(cfg2)


def test_manual_terminated_host_returns_to_pool():
    """After terminating a VM, its host can be reallocated."""
    platform = ManualWorkerProvider(label_prefix="iris", worker_port=10001, hosts=["10.0.0.1"])

    cfg = VmConfig(name="ctrl", manual=ManualVmConfig(host="10.0.0.1"))
    handle = platform.create_vm(cfg)
    assert platform.available_host_count == 0

    handle.terminate()
    assert platform.available_host_count == 1

    cfg2 = VmConfig(name="ctrl-2", manual=ManualVmConfig(host="10.0.0.1"))
    handle2 = platform.create_vm(cfg2)
    assert handle2.internal_address == "10.0.0.1"


def test_manual_slice_terminate_returns_hosts():
    """Terminating a slice returns all its hosts to the pool."""
    platform = ManualWorkerProvider(label_prefix="iris", worker_port=10001, hosts=["10.0.0.1", "10.0.0.2", "10.0.0.3"])

    cfg = SliceConfig(name_prefix="iris-group", num_vms=2)
    cfg.manual = ManualSliceConfig()
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
    assert handle.slice_id in {s.handle.slice_id for s in all_slices}


def test_list_all_slices_returns_all_managed(platform_env: PlatformEnv):
    """list_all_slices returns all managed slices regardless of scale group."""
    cfg_a = _make_slice_config(platform_env, "group-a")
    cfg_b = _make_slice_config(platform_env, "group-b")
    platform_env.platform.create_slice(cfg_a)
    platform_env.platform.create_slice(cfg_b)

    all_slices = platform_env.platform.list_all_slices()
    assert len(all_slices) == 2


def test_list_all_slices_excludes_manual_slices(platform_env: PlatformEnv):
    """list_all_slices drops slices labeled iris_manual=true so the autoscaler ignores them."""
    labels = Labels(platform_env.label_prefix)

    cfg_auto = _make_slice_config(platform_env, "auto-group")
    handle_auto = platform_env.platform.create_slice(cfg_auto)

    cfg_manual = _make_slice_config(platform_env, "auto-group")
    cfg_manual.labels[labels.iris_manual] = "true"
    handle_manual = platform_env.platform.create_slice(cfg_manual)

    all_slices = platform_env.platform.list_all_slices()
    slice_ids = {s.handle.slice_id for s in all_slices}
    assert handle_auto.slice_id in slice_ids
    assert handle_manual.slice_id not in slice_ids

    # Manual slices are still discoverable when explicitly asked for (delete-slice path).
    manual_only = platform_env.platform.list_slices(zones=[platform_env.zone], labels={labels.iris_manual: "true"})
    assert {s.slice_id for s in manual_only} == {handle_manual.slice_id}


def test_gcp_list_all_slices_multi_zone():
    """GcpWorkerProvider.list_all_slices returns slices across multiple zones."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    # Add synthetic zones that aren't in KNOWN_GCP_ZONES
    gcp_service._valid_zones.update(["zone-a", "zone-b"])
    gcp_config = GcpPlatformConfig(
        project_id="test-project",
        zones=["zone-a", "zone-b"],
    )
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)

    iris_labels = Labels("iris")
    cfg_a = SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=AcceleratorType.TPU,
        accelerator_variant="v5litepod-8",
        gcp=GcpSliceConfig(zone="zone-a", runtime_version="tpu-ubuntu2204-base"),
    )
    cfg_a.labels[iris_labels.iris_managed] = "true"

    cfg_b = SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=AcceleratorType.TPU,
        accelerator_variant="v5litepod-8",
        gcp=GcpSliceConfig(zone="zone-b", runtime_version="tpu-ubuntu2204-base"),
    )
    cfg_b.labels[iris_labels.iris_managed] = "true"

    handle_a = platform.create_slice(cfg_a)
    handle_b = platform.create_slice(cfg_b)

    all_slices = platform.list_all_slices()
    slice_ids = {s.handle.slice_id for s in all_slices}
    assert handle_a.slice_id in slice_ids
    assert handle_b.slice_id in slice_ids


# Tests below were removed during refactor: they tested private methods
# (_run_vm_slice_bootstrap, _run_tpu_bootstrap) that no longer exist on
# GcpWorkerProvider. Bootstrap behavior is exercised by integration tests.


def test_gcp_tpu_slice_passes_startup_script_metadata():
    """_create_tpu_slice with worker_config embeds startup-script in TPU metadata."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = GcpPlatformConfig(project_id="test-project")
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)

    cfg = SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=AcceleratorType.TPU,
        accelerator_variant="v5litepod-8",
        gcp=GcpSliceConfig(zone="us-central2-b", runtime_version="tpu-ubuntu2204-base"),
    )

    wc = WorkerConfig(
        docker_image="test-image:latest",
        port=10001,
        controller_address="controller:10000",
        cache_dir="/var/cache/iris",
    )

    with unittest.mock.patch("iris.cluster.platforms.gcp.workers.threading.Thread"):
        platform.create_slice(cfg, worker_config=wc)

    # Verify the service's in-memory TPU has startup-script metadata with [iris-init] markers.
    tpu_entries = list(gcp_service._tpus.values())
    assert len(tpu_entries) == 1
    metadata = tpu_entries[0].metadata
    assert "startup-script" in metadata
    assert "[iris-init]" in metadata["startup-script"]
    assert "test-image:latest" in metadata["startup-script"]


def test_gcp_tpu_slice_sets_os_login_metadata_and_uses_gcloud_remote_exec():
    """TPU slices always set enable-oslogin metadata and build a GcloudRemoteExec."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = GcpPlatformConfig(project_id="test-project")
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)

    cfg = SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=AcceleratorType.TPU,
        accelerator_variant="v5litepod-8",
        gcp=GcpSliceConfig(zone="us-central2-b", runtime_version="tpu-ubuntu2204-base"),
    )

    handle = platform.create_slice(cfg)
    status = handle.describe()
    tpu = next(iter(gcp_service._tpus.values()))
    assert tpu.metadata["enable-oslogin"] == "TRUE"
    assert tpu.metadata["block-project-ssh-keys"] == "TRUE"
    remote_exec = status.workers[0]._remote_exec
    assert isinstance(remote_exec, GcloudRemoteExec)
    assert remote_exec.ssh_user is None
    assert remote_exec.ssh_key_file is None


def test_gcp_tpu_slice_propagates_explicit_ssh_impersonation_account():
    """SshConfig.impersonate_service_account is forwarded onto the GcloudRemoteExec."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = GcpPlatformConfig(project_id="test-project")
    ssh_config = SshConfig(
        impersonate_service_account="iris-controller@test-project.iam.gserviceaccount.com",
    )
    platform = GcpWorkerProvider(
        gcp_config, label_prefix="iris", worker_port=10001, ssh_config=ssh_config, gcp_service=gcp_service
    )

    cfg = SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=AcceleratorType.TPU,
        accelerator_variant="v5litepod-8",
        gcp=GcpSliceConfig(
            zone="us-central2-b",
            runtime_version="tpu-ubuntu2204-base",
            service_account="iris-worker@test-project.iam.gserviceaccount.com",
        ),
    )

    handle = platform.create_slice(cfg)
    status = handle.describe()

    remote_exec = status.workers[0]._remote_exec
    assert isinstance(remote_exec, GcloudRemoteExec)
    assert remote_exec.impersonate_service_account == ssh_config.impersonate_service_account


# =============================================================================
# Section 6: TPU/VM Bootstrap Tests
#
# Tests for bootstrap timeout sizing, diagnostics, and VM health probing.
# =============================================================================
def _make_vm_slice_for_bootstrap(
    gcp_service: InMemoryGcpService,
    zone: str = "us-central2-b",
) -> tuple[GcpVmSliceHandle, str]:
    """Create a VM in InMemoryGcpService and return a handle + vm_name for bootstrap testing."""
    vm_name = "test-bootstrap-vm"
    gcp_service.vm_create(
        VmCreateRequest(
            name=vm_name,
            zone=zone,
            machine_type="n2-standard-4",
            labels={Labels("iris").iris_slice_id: vm_name},
        )
    )
    handle = GcpVmSliceHandle(
        _slice_id=vm_name,
        _vm_name=vm_name,
        _zone=zone,
        _project_id="test-project",
        _gcp_service=gcp_service,
        _labels={Labels("iris").iris_slice_id: vm_name},
        _created_at=Timestamp.now(),
        _label_prefix="iris",
        _worker_port=10001,
        _bootstrapping=True,
    )
    return handle, vm_name


def test_vm_bootstrap_health_probe_succeeds_without_serial_port():
    """Bootstrap completes when health probe succeeds, even if serial port never shows 'Bootstrap complete'."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    handle, _vm_name = _make_vm_slice_for_bootstrap(gcp_service)

    with unittest.mock.patch(
        "iris.cluster.platforms.gcp.workers._probe_worker_health",
        return_value=True,
    ):
        _run_vm_slice_bootstrap(
            gcp_service,
            handle,
            poll_interval=0.01,
            cloud_ready_timeout=5.0,
            bootstrap_timeout=5.0,
        )

    assert handle._bootstrap_state == CloudSliceState.READY


def test_vm_bootstrap_serial_port_succeeds_without_health_probe():
    """Bootstrap completes via serial port 'Bootstrap complete' when health probe fails."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    handle, vm_name = _make_vm_slice_for_bootstrap(gcp_service)

    gcp_service.set_serial_port_output(
        vm_name,
        "us-central2-b",
        "[iris-init] Starting bootstrap\n[iris-init] Bootstrap complete\n",
    )

    with unittest.mock.patch(
        "iris.cluster.platforms.gcp.workers._probe_worker_health",
        return_value=False,
    ):
        _run_vm_slice_bootstrap(
            gcp_service,
            handle,
            poll_interval=0.01,
            cloud_ready_timeout=5.0,
            bootstrap_timeout=5.0,
        )

    assert handle._bootstrap_state == CloudSliceState.READY


def test_vm_bootstrap_serial_port_error_raises():
    """Bootstrap fails immediately when serial port shows '[iris-init] ERROR'."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    handle, vm_name = _make_vm_slice_for_bootstrap(gcp_service)

    gcp_service.set_serial_port_output(
        vm_name,
        "us-central2-b",
        "[iris-init] ERROR: Docker pull failed\n",
    )

    with unittest.mock.patch(
        "iris.cluster.platforms.gcp.workers._probe_worker_health",
        return_value=False,
    ):
        with pytest.raises(InfraError, match="bootstrap failed"):
            _run_vm_slice_bootstrap(
                gcp_service,
                handle,
                poll_interval=0.01,
                cloud_ready_timeout=5.0,
                bootstrap_timeout=5.0,
            )


def test_vm_bootstrap_phase2_has_independent_timeout():
    """Phase 2 uses its own timeout, not the remainder from phase 1."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    handle, _vm_name = _make_vm_slice_for_bootstrap(gcp_service)

    # Health probe never succeeds, serial port never shows complete.
    # With a very short bootstrap_timeout, this should fail with phase 2 message.
    with unittest.mock.patch(
        "iris.cluster.platforms.gcp.workers._probe_worker_health",
        return_value=False,
    ):
        with pytest.raises(InfraError, match=r"bootstrap did not complete within 0\.05s"):
            _run_vm_slice_bootstrap(
                gcp_service,
                handle,
                poll_interval=0.01,
                cloud_ready_timeout=600.0,
                bootstrap_timeout=0.05,
            )


def test_vm_bootstrap_cloud_not_ready_raises_phase1_timeout():
    """Phase 1 timeout triggers when VM never reaches READY."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")

    # Create a VM but set it to non-READY state
    vm_name = "test-stuck-vm"
    gcp_service.vm_create(
        VmCreateRequest(
            name=vm_name,
            zone="us-central2-b",
            machine_type="n2-standard-4",
            labels={Labels("iris").iris_slice_id: vm_name},
        )
    )
    # Set VM to STAGING so it never reaches READY
    gcp_service._vms[(vm_name, "us-central2-b")].status = "STAGING"

    handle = GcpVmSliceHandle(
        _slice_id=vm_name,
        _vm_name=vm_name,
        _zone="us-central2-b",
        _project_id="test-project",
        _gcp_service=gcp_service,
        _labels={Labels("iris").iris_slice_id: vm_name},
        _created_at=Timestamp.now(),
        _label_prefix="iris",
        _worker_port=10001,
        _bootstrapping=True,
    )

    with pytest.raises(InfraError, match=r"did not reach cloud READY within 0\.05s"):
        _run_vm_slice_bootstrap(
            gcp_service,
            handle,
            poll_interval=0.01,
            cloud_ready_timeout=0.05,
            bootstrap_timeout=300.0,
        )


def _make_tpu_slice_for_bootstrap(
    gcp_service: InMemoryGcpService,
    zone: str = "us-central2-b",
) -> GcpSliceHandle:
    """Create a TPU slice (bootstrap thread suppressed) for direct _run_tpu_bootstrap testing.

    The fake leaves the node at CREATING with synthetic worker IPs — the #6087
    scenario where workers are up while the cloud create-status still lags.
    """
    gcp_config = GcpPlatformConfig(project_id="test-project")
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)

    cfg = SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=AcceleratorType.TPU,
        accelerator_variant="v5litepod-8",
        gcp=GcpSliceConfig(zone=zone, runtime_version="tpu-ubuntu2204-base"),
    )
    wc = WorkerConfig(
        docker_image="test-image:latest",
        port=10001,
        controller_address="controller:10000",
        cache_dir="/var/cache/iris",
    )

    with unittest.mock.patch("iris.cluster.platforms.gcp.workers.threading.Thread"):
        handle = platform.create_slice(cfg, worker_config=wc)
    assert isinstance(handle, GcpSliceHandle)
    return handle


def test_tpu_bootstrap_marks_ready_while_cloud_stuck_creating():
    """Regression for #6087: a TPU whose cloud status is stuck CREATING but whose
    workers answer /health becomes READY and is never deleted."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    handle = _make_tpu_slice_for_bootstrap(gcp_service)
    assert gcp_service.tpu_describe(handle.slice_id, handle.zone).state == "CREATING"

    with unittest.mock.patch("iris.cluster.platforms.gcp.workers._probe_worker_health", return_value=True):
        _run_tpu_bootstrap(
            gcp_service,
            "test-project",
            handle,
            poll_interval=0.01,
            ip_wait_timeout=5.0,
            bootstrap_timeout=5.0,
        )

    assert handle._bootstrap_state == CloudSliceState.READY
    assert handle.describe().state == CloudSliceState.READY
    # Slice was kept despite the cloud status never reaching READY.
    surviving = gcp_service.tpu_describe(handle.slice_id, handle.zone)
    assert surviving is not None
    assert surviving.state == "CREATING"


@pytest.mark.parametrize(
    "error_code, expected_exc",
    [(8, QuotaExhaustedError), (13, InfraError)],
)
def test_tpu_bootstrap_fails_fast_on_create_operation_error(error_code, expected_exc):
    """A create LRO that completes with an error aborts bootstrap immediately,
    mapping RESOURCE_EXHAUSTED to QuotaExhaustedError and other codes to InfraError."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    handle = _make_tpu_slice_for_bootstrap(gcp_service)
    gcp_service.set_operation_status(
        handle._create_operation,
        OperationStatus(done=True, error_code=error_code, error_message="boom"),
    )

    with unittest.mock.patch("iris.cluster.platforms.gcp.workers._probe_worker_health", return_value=False):
        with pytest.raises(expected_exc):
            _run_tpu_bootstrap(
                gcp_service,
                "test-project",
                handle,
                poll_interval=0.01,
                ip_wait_timeout=5.0,
                bootstrap_timeout=5.0,
            )


def test_tpu_bootstrap_aborts_when_slice_enters_deleting():
    """A slice torn down (terminal cloud state) during bootstrap aborts rather than waiting out the timeout."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    handle = _make_tpu_slice_for_bootstrap(gcp_service)
    gcp_service.advance_tpu_state(handle.slice_id, handle.zone, "DELETING")

    with unittest.mock.patch("iris.cluster.platforms.gcp.workers._probe_worker_health", return_value=False):
        with pytest.raises(InfraError):
            _run_tpu_bootstrap(
                gcp_service,
                "test-project",
                handle,
                poll_interval=0.01,
                ip_wait_timeout=5.0,
                bootstrap_timeout=5.0,
            )


@pytest.mark.parametrize(
    "cloud_state, bootstrap_state, expected",
    [
        # #6087 fix: confirmed-healthy workers win over a laggy CREATING cloud state.
        (CloudSliceState.CREATING, CloudSliceState.READY, CloudSliceState.READY),
        # Bootstrap still in progress: reflect the raw cloud state.
        (CloudSliceState.CREATING, None, CloudSliceState.CREATING),
        (CloudSliceState.READY, None, CloudSliceState.BOOTSTRAPPING),
        (CloudSliceState.READY, CloudSliceState.READY, CloudSliceState.READY),
        # Gone/doomed cloud states override the bootstrap verdict so a vanished
        # node never lingers as READY on a stale sentinel.
        (CloudSliceState.DELETING, CloudSliceState.READY, CloudSliceState.DELETING),
        (CloudSliceState.FAILED, CloudSliceState.READY, CloudSliceState.FAILED),
        (CloudSliceState.UNKNOWN, CloudSliceState.READY, CloudSliceState.UNKNOWN),
        # Bootstrap failure surfaces as FAILED.
        (CloudSliceState.CREATING, CloudSliceState.FAILED, CloudSliceState.FAILED),
        # A definitive bootstrap failure wins over an UNKNOWN cloud state: a
        # stockout/quota create failure leaves no TPU to describe (UNKNOWN), but
        # the slice must be reaped now, not after the unresolvable timeout.
        (CloudSliceState.UNKNOWN, CloudSliceState.FAILED, CloudSliceState.FAILED),
    ],
)
def test_composite_slice_state(cloud_state, bootstrap_state, expected):
    assert _composite_slice_state(cloud_state, bootstrap_state) == expected


def _bootstrapping_tpu_handle(gcp_service, slice_id="slice-x"):
    return GcpSliceHandle(
        _slice_id=slice_id,
        _zone="us-east5-b",
        _project_id="test-project",
        _labels={Labels("iris").iris_managed: "true"},
        _created_at=Timestamp.from_ms(0),
        _label_prefix="iris",
        _worker_port=10001,
        _accelerator_variant="v5litepod-8",
        _gcp_service=gcp_service,
        _bootstrapping=True,
    )


def test_describe_surfaces_bootstrap_failure_reason():
    """A bootstrap failure's reason (e.g. the create-LRO stockout) is surfaced via
    describe().error_message so the autoscaler can classify the outcome."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    handle = _bootstrapping_tpu_handle(gcp_service)

    # In progress: no failure reason surfaced.
    assert handle.describe().error_message == ""

    with handle._bootstrap_lock:
        handle._bootstrap_state = CloudSliceState.FAILED
        handle._bootstrap_error = 'There is no more capacity in the zone "us-east5-b"'

    status = handle.describe()
    assert status.state == CloudSliceState.FAILED
    assert "no more capacity" in status.error_message


def test_bootstrap_thread_captures_failure_reason(monkeypatch):
    """The bootstrap watcher keeps the failure reason, not just the FAILED state,
    so a create-LRO stockout isn't reduced to a generic 'bootstrap failed'."""

    class _SyncThread:
        def __init__(self, target, name=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr("iris.cluster.platforms.gcp.workers.threading.Thread", _SyncThread)

    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    handle = _bootstrapping_tpu_handle(gcp_service)

    def boom():
        raise QuotaExhaustedError('There is no more capacity in the zone "us-east5-b"')

    _spawn_bootstrap_thread(handle, boom)

    assert handle._bootstrap_state == CloudSliceState.FAILED
    assert "no more capacity" in handle._bootstrap_error
