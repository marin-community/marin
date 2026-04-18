# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidated Platform protocol tests.

Tests exercise the Platform protocol contract through parameterized fixtures
(GCP via InMemoryGcpService DRY_RUN, Manual). Platform-specific behavioral
tests that cannot be expressed through the protocol are in dedicated sections.
"""

from __future__ import annotations

import unittest.mock
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from iris.cluster.providers.gcp.controller import GcpControllerProvider
from iris.cluster.providers.gcp.fake import InMemoryGcpService
from iris.cluster.providers.gcp.handles import GcpVmSliceHandle, _build_gce_resource_name
from iris.cluster.providers.remote_exec import DirectSshRemoteExec, GceRemoteExec, GcloudRemoteExec
from iris.cluster.providers.gcp.workers import (
    GcpWorkerProvider,
    _run_vm_slice_bootstrap,
    _validate_slice_config,
)
from iris.cluster.providers.manual.provider import ManualControllerProvider, ManualWorkerProvider
from iris.cluster.providers.types import (
    CloudSliceState,
    InfraError,
    Labels,
    QuotaExhaustedError,
)
from iris.cluster.service_mode import ServiceMode
from iris.rpc import config_pb2
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

    GCP is backed by InMemoryGcpService in DRY_RUN mode.
    """
    name = request.param

    if name == "gcp":
        gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
        gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
        platform = GcpWorkerProvider(gcp_config, label_prefix="iris", gcp_service=gcp_service)
        yield PlatformEnv(platform=platform, zone="us-central2-b", name="gcp", label_prefix="iris")
    else:
        platform = ManualWorkerProvider(
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
# so we test them on ManualWorkerProvider which is fully in-memory.


def test_terminate_then_status_is_deleting():
    """After terminate(), slice status reports DELETING."""
    platform = ManualWorkerProvider(label_prefix="iris", hosts=["10.0.0.1", "10.0.0.2", "10.0.0.3"])
    cfg = config_pb2.SliceConfig(name_prefix="iris-term-group", num_vms=2)
    cfg.manual.CopyFrom(config_pb2.ManualSliceConfig())
    cfg.labels[Labels("iris").iris_managed] = "true"
    cfg.labels[Labels("iris").iris_scale_group] = "term-group"
    handle = platform.create_slice(cfg)

    handle.terminate()
    assert handle.describe().state == CloudSliceState.DELETING


def test_tunnel_returns_address_directly():
    """tunnel() is a passthrough on ManualControllerProvider (no SSH)."""
    worker_provider = ManualWorkerProvider(label_prefix="iris", hosts=["10.0.0.1"])
    controller = ManualControllerProvider(worker_provider=worker_provider)
    addr = "http://10.0.0.1:10000"
    with controller.tunnel(addr) as tunneled:
        assert tunneled == addr


def test_gcp_tunnel_prefers_ssh_impersonation_config():
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    ssh_config = config_pb2.SshConfig(
        auth_mode=config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN,
        key_file="/tmp/iris-oslogin",
        impersonate_service_account="iris-controller@test-project.iam.gserviceaccount.com",
    )
    worker_provider = GcpWorkerProvider(gcp_config, label_prefix="iris", ssh_config=ssh_config, gcp_service=gcp_service)
    controller = GcpControllerProvider(
        worker_provider=worker_provider,
        controller_service_account="iris-worker@test-project.iam.gserviceaccount.com",
    )

    list_result = unittest.mock.Mock(returncode=0, stdout="iris-controller-iris us-central2-b\n", stderr="")
    ssh_proc = unittest.mock.Mock()
    ssh_proc.poll.return_value = None
    ssh_proc.terminate.return_value = None
    ssh_proc.wait.return_value = 0

    with (
        unittest.mock.patch("iris.cluster.providers.gcp.controller._check_gcloud_ssh_key"),
        unittest.mock.patch("iris.cluster.providers.gcp.controller.find_free_port", return_value=10042),
        unittest.mock.patch(
            "iris.cluster.providers.gcp.controller.resolve_current_os_login_user",
            return_value="svc-user",
        ) as resolve_user,
        unittest.mock.patch("iris.cluster.providers.gcp.controller.wait_for_port"),
        unittest.mock.patch(
            "iris.cluster.providers.gcp.controller.subprocess.run", return_value=list_result
        ) as run_mock,
        unittest.mock.patch(
            "iris.cluster.providers.gcp.controller.subprocess.Popen", return_value=ssh_proc
        ) as popen_mock,
    ):
        with controller.tunnel("unused") as tunneled:
            assert tunneled == "http://127.0.0.1:10042"

    resolve_user.assert_called_with(impersonate_service_account=ssh_config.impersonate_service_account)
    list_cmd = run_mock.call_args.args[0]
    ssh_cmd = popen_mock.call_args.args[0]
    assert f"--impersonate-service-account={ssh_config.impersonate_service_account}" in list_cmd
    assert f"--impersonate-service-account={ssh_config.impersonate_service_account}" in ssh_cmd


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

    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", gcp_service=gcp_service)

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


def test_gcp_validate_vm_slice_config_rejects_non_on_demand():
    """VM slice mode rejects non-on-demand capacity types."""
    cfg = config_pb2.SliceConfig(
        name_prefix="test",
        num_vms=1,
        capacity_type=config_pb2.CAPACITY_TYPE_PREEMPTIBLE,
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
    cfg.gcp.machine_type = "n2-standard-4"

    with pytest.raises(ValueError, match="only supports capacity_type on-demand"):
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
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
        capacity_type=config_pb2.CAPACITY_TYPE_ON_DEMAND,
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
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="smoke-cpu_vm_e2_standard_4_ondemand-europe-west4-b",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
        capacity_type=config_pb2.CAPACITY_TYPE_ON_DEMAND,
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
    assert status.workers[0]._remote_exec.ssh_user == "iris"
    listed = platform.list_all_slices()
    assert handle.slice_id in {s.slice_id for s in listed}


def test_gcp_vm_slice_os_login_sets_metadata_and_uses_gcloud_default_user():
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    ssh_config = config_pb2.SshConfig(
        auth_mode=config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN,
        key_file="/tmp/iris-oslogin",
    )
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", ssh_config=ssh_config, gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
        capacity_type=config_pb2.CAPACITY_TYPE_ON_DEMAND,
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
    cfg.gcp.machine_type = "n2-standard-4"
    cfg.labels[Labels("iris").iris_managed] = "true"
    cfg.labels[Labels("iris").iris_scale_group] = "cpu-vm"

    handle = platform.create_slice(cfg)
    status = handle.describe()
    vm = next(iter(gcp_service._vms.values()))
    assert vm.metadata["enable-oslogin"] == "TRUE"
    assert vm.metadata["block-project-ssh-keys"] == "TRUE"
    assert status.workers[0]._remote_exec.ssh_user is None
    assert status.workers[0]._remote_exec.ssh_key_file == "/tmp/iris-oslogin"


def test_gcp_vm_slice_os_login_uses_service_account_impersonation():
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    ssh_config = config_pb2.SshConfig(
        auth_mode=config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN,
        key_file="/tmp/iris-oslogin",
    )
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", ssh_config=ssh_config, gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
        capacity_type=config_pb2.CAPACITY_TYPE_ON_DEMAND,
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
    cfg.gcp.machine_type = "n2-standard-4"
    cfg.gcp.service_account = "iris-worker@test-project.iam.gserviceaccount.com"

    handle = platform.create_slice(cfg)
    status = handle.describe()
    assert status.workers[0]._remote_exec.impersonate_service_account is None


def test_gcp_vm_slice_os_login_prefers_explicit_ssh_impersonation_account():
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    ssh_config = config_pb2.SshConfig(
        auth_mode=config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN,
        key_file="/tmp/iris-oslogin",
        impersonate_service_account="iris-controller@test-project.iam.gserviceaccount.com",
    )
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", ssh_config=ssh_config, gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
        capacity_type=config_pb2.CAPACITY_TYPE_ON_DEMAND,
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.mode = config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM
    cfg.gcp.machine_type = "n2-standard-4"
    cfg.gcp.service_account = "iris-worker@test-project.iam.gserviceaccount.com"

    handle = platform.create_slice(cfg)
    status = handle.describe()

    assert status.workers[0]._remote_exec.impersonate_service_account == ssh_config.impersonate_service_account


def test_gcp_empty_accelerator_variant_rejected():
    """create_slice with empty accelerator_variant raises ValueError."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", gcp_service=gcp_service)

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
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    cfg = config_pb2.VmConfig(name="test-vm")

    with pytest.raises(ValueError, match="zone"):
        platform.create_vm(cfg)


def test_gcp_list_slices_skips_deleting_tpus():
    """list_slices omits TPUs in DELETING state."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", gcp_service=gcp_service)

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
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="my-proj")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="my-proj")
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", gcp_service=gcp_service)

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

    with unittest.mock.patch("iris.cluster.providers.gcp.workers.threading.Thread"):
        platform.create_slice(cfg, worker_config=wc)

    assert wc.docker_image == "europe-docker.pkg.dev/my-proj/ghcr-mirror/marin-community/iris-worker:latest"


def test_gcp_list_slices_skips_inactive_vm_instances():
    """list_slices omits VM-backed slices for instances in inactive states."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
        capacity_type=config_pb2.CAPACITY_TYPE_ON_DEMAND,
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
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central2-b"])
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-cpu-vm",
        num_vms=1,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
        capacity_type=config_pb2.CAPACITY_TYPE_ON_DEMAND,
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
    platform = ManualWorkerProvider(label_prefix="iris", hosts=["10.0.0.1"])
    cfg = config_pb2.SliceConfig(name_prefix="iris-group", num_vms=3)
    cfg.manual.CopyFrom(config_pb2.ManualSliceConfig())

    with pytest.raises(RuntimeError, match="Need 3 hosts but only 1 available"):
        platform.create_slice(cfg)


def test_manual_host_exclusivity():
    """A host allocated to one VM cannot be allocated to another."""
    platform = ManualWorkerProvider(label_prefix="iris", hosts=["10.0.0.1", "10.0.0.2"])

    cfg1 = config_pb2.VmConfig(name="ctrl-1")
    cfg1.manual.host = "10.0.0.1"
    platform.create_vm(cfg1)

    cfg2 = config_pb2.VmConfig(name="ctrl-2")
    cfg2.manual.host = "10.0.0.1"
    with pytest.raises(RuntimeError, match="already allocated"):
        platform.create_vm(cfg2)


def test_manual_terminated_host_returns_to_pool():
    """After terminating a VM, its host can be reallocated."""
    platform = ManualWorkerProvider(label_prefix="iris", hosts=["10.0.0.1"])

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
    platform = ManualWorkerProvider(label_prefix="iris", hosts=["10.0.0.1", "10.0.0.2", "10.0.0.3"])

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
    """GcpWorkerProvider.list_all_slices returns slices across multiple zones."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    # Add synthetic zones that aren't in KNOWN_GCP_ZONES
    gcp_service._valid_zones.update(["zone-a", "zone-b"])
    gcp_config = config_pb2.GcpPlatformConfig(
        project_id="test-project",
        zones=["zone-a", "zone-b"],
    )
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", gcp_service=gcp_service)

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


# Tests below were removed during refactor: they tested private methods
# (_run_vm_slice_bootstrap, _run_tpu_bootstrap) that no longer exist on
# GcpWorkerProvider. Bootstrap behavior is exercised by integration tests.


def test_gcp_tpu_slice_passes_startup_script_metadata():
    """_create_tpu_slice with worker_config embeds startup-script in TPU metadata."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", gcp_service=gcp_service)

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

    with unittest.mock.patch("iris.cluster.providers.gcp.workers.threading.Thread"):
        platform.create_slice(cfg, worker_config=wc)

    # Verify the service's in-memory TPU has startup-script metadata with [iris-init] markers.
    tpu_entries = list(gcp_service._tpus.values())
    assert len(tpu_entries) == 1
    metadata = tpu_entries[0].metadata
    assert "startup-script" in metadata
    assert "[iris-init]" in metadata["startup-script"]
    assert "test-image:latest" in metadata["startup-script"]


def test_gcp_tpu_slice_os_login_sets_metadata_and_uses_direct_ssh():
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    ssh_config = config_pb2.SshConfig(
        auth_mode=config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN,
        os_login_user="ci-user",
        key_file="/tmp/iris-oslogin",
    )
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", ssh_config=ssh_config, gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-8",
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"

    handle = platform.create_slice(cfg)
    status = handle.describe()
    tpu = next(iter(gcp_service._tpus.values()))
    assert tpu.metadata["enable-oslogin"] == "TRUE"
    assert tpu.metadata["block-project-ssh-keys"] == "TRUE"
    assert isinstance(status.workers[0]._remote_exec, DirectSshRemoteExec)
    assert status.workers[0]._remote_exec.user == "ci-user"
    assert status.workers[0]._remote_exec.key_file == "/tmp/iris-oslogin"
    assert status.workers[0].external_address is None
    assert status.workers[0]._remote_exec.host == status.workers[0].internal_address


def test_gcp_tpu_slice_os_login_resolves_user_from_service_account():
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    ssh_config = config_pb2.SshConfig(
        auth_mode=config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN,
        key_file="/tmp/iris-oslogin",
    )
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", ssh_config=ssh_config, gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-8",
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"
    cfg.gcp.service_account = "iris-worker@test-project.iam.gserviceaccount.com"

    with unittest.mock.patch(
        "iris.cluster.providers.gcp.handles.resolve_current_os_login_user",
        return_value="svc-user",
    ) as resolve_user:
        handle = platform.create_slice(cfg)
        status = handle.describe()

    resolve_user.assert_called_with(impersonate_service_account=None)
    assert status.workers[0]._remote_exec.user == "svc-user"


def test_gcp_tpu_slice_os_login_prefers_explicit_ssh_impersonation_account():
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    ssh_config = config_pb2.SshConfig(
        auth_mode=config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN,
        key_file="/tmp/iris-oslogin",
        impersonate_service_account="iris-controller@test-project.iam.gserviceaccount.com",
    )
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", ssh_config=ssh_config, gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-8",
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"
    cfg.gcp.service_account = "iris-worker@test-project.iam.gserviceaccount.com"

    with unittest.mock.patch(
        "iris.cluster.providers.gcp.handles.resolve_current_os_login_user",
        return_value="svc-user",
    ) as resolve_user:
        handle = platform.create_slice(cfg)
        status = handle.describe()

    resolve_user.assert_called_with(impersonate_service_account=ssh_config.impersonate_service_account)
    assert status.workers[0]._remote_exec.user == "svc-user"


def test_gcp_tpu_slice_os_login_prefers_external_ip_for_direct_ssh():
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project")
    ssh_config = config_pb2.SshConfig(
        auth_mode=config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN,
        os_login_user="svc-user",
        key_file="/tmp/iris-oslogin",
    )
    platform = GcpWorkerProvider(gcp_config, label_prefix="iris", ssh_config=ssh_config, gcp_service=gcp_service)

    cfg = config_pb2.SliceConfig(
        name_prefix="iris-tpu",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-8",
    )
    cfg.gcp.zone = "us-central2-b"
    cfg.gcp.runtime_version = "tpu-ubuntu2204-base"

    handle = platform.create_slice(cfg)
    tpu = next(iter(gcp_service._tpus.values()))
    tpu.state = "READY"
    tpu.external_network_endpoints = ["34.1.2.3"]

    status = handle.describe()

    assert status.workers[0].internal_address == "10.0.0.0"
    assert status.workers[0].external_address == "34.1.2.3"
    assert isinstance(status.workers[0]._remote_exec, DirectSshRemoteExec)
    assert status.workers[0]._remote_exec.host == "34.1.2.3"


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
    from iris.cluster.providers.gcp.service import VmCreateRequest

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
        _bootstrapping=True,
    )
    return handle, vm_name


def test_vm_bootstrap_health_probe_succeeds_without_serial_port():
    """Bootstrap completes when health probe succeeds, even if serial port never shows 'Bootstrap complete'."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    handle, _vm_name = _make_vm_slice_for_bootstrap(gcp_service)
    worker_config = config_pb2.WorkerConfig(port=10001)

    with unittest.mock.patch(
        "iris.cluster.providers.gcp.workers._probe_worker_health",
        return_value=True,
    ):
        _run_vm_slice_bootstrap(
            gcp_service,
            handle,
            worker_config,
            poll_interval=0.01,
            cloud_ready_timeout=5.0,
            bootstrap_timeout=5.0,
        )

    assert handle._bootstrap_state == CloudSliceState.READY


def test_vm_bootstrap_serial_port_succeeds_without_health_probe():
    """Bootstrap completes via serial port 'Bootstrap complete' when health probe fails."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    handle, vm_name = _make_vm_slice_for_bootstrap(gcp_service)
    worker_config = config_pb2.WorkerConfig(port=10001)

    gcp_service.set_serial_port_output(
        vm_name,
        "us-central2-b",
        "[iris-init] Starting bootstrap\n[iris-init] Bootstrap complete\n",
    )

    with unittest.mock.patch(
        "iris.cluster.providers.gcp.workers._probe_worker_health",
        return_value=False,
    ):
        _run_vm_slice_bootstrap(
            gcp_service,
            handle,
            worker_config,
            poll_interval=0.01,
            cloud_ready_timeout=5.0,
            bootstrap_timeout=5.0,
        )

    assert handle._bootstrap_state == CloudSliceState.READY


def test_vm_bootstrap_serial_port_error_raises():
    """Bootstrap fails immediately when serial port shows '[iris-init] ERROR'."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    handle, vm_name = _make_vm_slice_for_bootstrap(gcp_service)
    worker_config = config_pb2.WorkerConfig(port=10001)

    gcp_service.set_serial_port_output(
        vm_name,
        "us-central2-b",
        "[iris-init] ERROR: Docker pull failed\n",
    )

    with unittest.mock.patch(
        "iris.cluster.providers.gcp.workers._probe_worker_health",
        return_value=False,
    ):
        with pytest.raises(InfraError, match="bootstrap failed"):
            _run_vm_slice_bootstrap(
                gcp_service,
                handle,
                worker_config,
                poll_interval=0.01,
                cloud_ready_timeout=5.0,
                bootstrap_timeout=5.0,
            )


def test_vm_bootstrap_phase2_has_independent_timeout():
    """Phase 2 uses its own timeout, not the remainder from phase 1."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")
    handle, _vm_name = _make_vm_slice_for_bootstrap(gcp_service)
    worker_config = config_pb2.WorkerConfig(port=10001)

    # Health probe never succeeds, serial port never shows complete.
    # With a very short bootstrap_timeout, this should fail with phase 2 message.
    with unittest.mock.patch(
        "iris.cluster.providers.gcp.workers._probe_worker_health",
        return_value=False,
    ):
        with pytest.raises(InfraError, match=r"bootstrap did not complete within 0\.05s"):
            _run_vm_slice_bootstrap(
                gcp_service,
                handle,
                worker_config,
                poll_interval=0.01,
                cloud_ready_timeout=600.0,
                bootstrap_timeout=0.05,
            )


def test_vm_bootstrap_cloud_not_ready_raises_phase1_timeout():
    """Phase 1 timeout triggers when VM never reaches READY."""
    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project")

    # Create a VM but set it to non-READY state
    from iris.cluster.providers.gcp.service import VmCreateRequest

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
        _bootstrapping=True,
    )
    worker_config = config_pb2.WorkerConfig(port=10001)

    with pytest.raises(InfraError, match=r"did not reach cloud READY within 0\.05s"):
        _run_vm_slice_bootstrap(
            gcp_service,
            handle,
            worker_config,
            poll_interval=0.01,
            cloud_ready_timeout=0.05,
            bootstrap_timeout=300.0,
        )
