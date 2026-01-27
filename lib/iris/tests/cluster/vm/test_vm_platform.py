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

"""Tests for VM platform implementations.

Tests are organized around the protocols:
- VmGroupStatus: Parameterized matrix of state combinations
- VmGroupProtocol: Parameterized tests for TpuVmGroup and ManualVmGroup
- VmManagerProtocol: Parameterized tests for TpuVmManager and ManualVmManager
- Platform-specific behavior tests
- VmRegistry: Concurrency and state management
- TrackedVmFactory: Integration with registry
"""

import json
import threading
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from iris.cluster.vm.gcp_tpu_platform import TpuVmGroup, TpuVmManager
from iris.cluster.vm.managed_vm import PoolExhaustedError, TrackedVmFactory, VmRegistry
from iris.cluster.vm.manual_platform import ManualVmGroup, ManualVmManager
from iris.cluster.vm.vm_platform import VmGroupProtocol, VmGroupStatus, VmSnapshot
from iris.rpc import config_pb2, vm_pb2

# =============================================================================
# Test Helpers
# =============================================================================


def make_mock_vm(
    vm_id: str,
    slice_id: str = "slice-001",
    scale_group: str = "test-group",
    state: vm_pb2.VmState = vm_pb2.VM_STATE_READY,
    address: str = "10.0.0.1",
    init_phase: str = "",
    init_error: str = "",
) -> MagicMock:
    """Create a mock ManagedVm for testing."""
    vm = MagicMock()
    vm.info = vm_pb2.VmInfo(
        vm_id=vm_id,
        slice_id=slice_id,
        scale_group=scale_group,
        state=state,
        address=address,
        init_phase=init_phase,
        init_error=init_error,
    )
    return vm


def make_snapshot(
    vm_id: str = "vm-0",
    state: vm_pb2.VmState = vm_pb2.VM_STATE_READY,
    address: str = "10.0.0.1",
    init_phase: str = "",
    init_error: str = "",
) -> VmSnapshot:
    """Create a VmSnapshot for testing."""
    return VmSnapshot(
        vm_id=vm_id,
        state=state,
        address=address,
        init_phase=init_phase,
        init_error=init_error,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def registry() -> VmRegistry:
    """Create a fresh VmRegistry for testing."""
    return VmRegistry()


@pytest.fixture
def bootstrap_config() -> config_pb2.BootstrapConfig:
    """Standard bootstrap configuration for tests."""
    return config_pb2.BootstrapConfig(
        controller_address="10.0.0.1:10000",
        worker_id="test-worker",
        worker_port=10001,
        docker_image="gcr.io/test/iris-worker:latest",
        cache_dir="/var/cache/iris",
    )


@pytest.fixture
def timeout_config() -> config_pb2.TimeoutConfig:
    """Timeout configuration for tests."""
    return config_pb2.TimeoutConfig(
        boot_timeout_seconds=5,
        init_timeout_seconds=10,
        ssh_poll_interval_seconds=1,
    )


@pytest.fixture
def v5p8_scale_group() -> config_pb2.ScaleGroupConfig:
    """Single-host TPU scale group (v5p-8)."""
    return config_pb2.ScaleGroupConfig(
        name="tpu-v5p-8",
        min_slices=0,
        max_slices=10,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5p-8",
        runtime_version="v2-alpha-tpuv5",
        zones=["us-central1-a"],
    )


@pytest.fixture
def v5p16_scale_group() -> config_pb2.ScaleGroupConfig:
    """Multi-host TPU scale group (v5p-16, 2 VMs per slice)."""
    return config_pb2.ScaleGroupConfig(
        name="tpu-v5p-16",
        min_slices=0,
        max_slices=5,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5p-16",
        runtime_version="v2-alpha-tpuv5",
        zones=["us-central1-a"],
    )


@pytest.fixture
def manual_scale_group() -> config_pb2.ScaleGroupConfig:
    """Scale group config for manual hosts."""
    return config_pb2.ScaleGroupConfig(
        name="manual-hosts",
        min_slices=0,
        max_slices=3,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
        runtime_version="manual",
        zones=["manual"],
    )


@pytest.fixture(params=["tpu", "manual"])
def vm_group_factory(
    request: pytest.FixtureRequest, registry: VmRegistry
) -> Callable[[list[MagicMock]], VmGroupProtocol]:
    """Factory that creates either TpuVmGroup or ManualVmGroup.

    This fixture is parameterized to test both implementations against
    the VmGroupProtocol interface.
    """

    def create_tpu_group(vms: list[MagicMock]) -> VmGroupProtocol:
        return TpuVmGroup(
            group_id="test-slice-001",
            scale_group="test-group",
            zone="us-central1-a",
            project_id="test-project",
            vms=vms,
            vm_registry=registry,
            created_at_ms=1234567890,
        )

    def create_manual_group(vms: list[MagicMock]) -> VmGroupProtocol:
        return ManualVmGroup(
            group_id="test-slice-001",
            scale_group="test-group",
            vms=vms,
            vm_registry=registry,
            created_at_ms=1234567890,
        )

    return create_tpu_group if request.param == "tpu" else create_manual_group


# =============================================================================
# VmGroupStatus Tests - Parameterized Matrix
# =============================================================================


@pytest.mark.parametrize(
    "vm_states,expected_all_ready,expected_any_failed,expected_is_terminal",
    [
        ([vm_pb2.VM_STATE_READY, vm_pb2.VM_STATE_READY], True, False, True),
        ([vm_pb2.VM_STATE_READY, vm_pb2.VM_STATE_BOOTING], False, False, False),
        ([vm_pb2.VM_STATE_READY, vm_pb2.VM_STATE_FAILED], False, True, True),
        ([vm_pb2.VM_STATE_PREEMPTED], False, True, True),
        ([vm_pb2.VM_STATE_BOOTING], False, False, False),
        ([vm_pb2.VM_STATE_INITIALIZING], False, False, False),
        ([vm_pb2.VM_STATE_TERMINATED], False, False, True),
        ([], True, False, True),  # empty slice - vacuous truth
        ([vm_pb2.VM_STATE_READY, vm_pb2.VM_STATE_FAILED, vm_pb2.VM_STATE_PREEMPTED], False, True, True),
        ([vm_pb2.VM_STATE_UNHEALTHY], False, False, False),
        ([vm_pb2.VM_STATE_STOPPING], False, False, False),
    ],
)
def test_vm_group_status_aggregate_properties(
    vm_states: list[vm_pb2.VmState],
    expected_all_ready: bool,
    expected_any_failed: bool,
    expected_is_terminal: bool,
):
    """VmGroupStatus computes aggregate properties correctly for all state combinations."""
    vms = [make_snapshot(f"vm-{i}", state) for i, state in enumerate(vm_states)]
    status = VmGroupStatus(vms=vms)

    assert status.all_ready == expected_all_ready
    assert status.any_failed == expected_any_failed
    assert status.is_terminal == expected_is_terminal
    assert status.vm_count == len(vm_states)
    assert status.ready_count == sum(1 for s in vm_states if s == vm_pb2.VM_STATE_READY)


def test_vm_group_status_collects_error_messages():
    """VmGroupStatus collects non-empty error messages from failed VMs."""
    status = VmGroupStatus(
        vms=[
            make_snapshot("vm-0", vm_pb2.VM_STATE_FAILED, init_error="Boot timeout"),
            make_snapshot("vm-1", vm_pb2.VM_STATE_READY, init_error=""),
            make_snapshot("vm-2", vm_pb2.VM_STATE_FAILED, init_error="SSH failed"),
        ]
    )
    errors = status.error_messages
    assert len(errors) == 2
    assert "Boot timeout" in errors
    assert "SSH failed" in errors


# =============================================================================
# VmGroupProtocol Tests - Parameterized for Both Platforms
# =============================================================================


@patch("iris.cluster.vm.gcp_tpu_platform.subprocess.run")
def test_vm_group_status_reflects_vm_states(
    mock_run: MagicMock,
    vm_group_factory: Callable[[list[MagicMock]], VmGroupProtocol],
):
    """VmGroup.status() reflects the aggregate state of its VMs."""
    vms = [
        make_mock_vm("vm-0", state=vm_pb2.VM_STATE_READY),
        make_mock_vm("vm-1", state=vm_pb2.VM_STATE_BOOTING),
    ]
    vm_group = vm_group_factory(vms)

    status = vm_group.status()

    assert status.all_ready is False
    assert status.any_failed is False
    assert status.is_terminal is False
    assert status.vm_count == 2


@patch("iris.cluster.vm.gcp_tpu_platform.subprocess.run")
def test_vm_group_to_proto_includes_vms(
    mock_run: MagicMock,
    vm_group_factory: Callable[[list[MagicMock]], VmGroupProtocol],
):
    """VmGroup.to_proto() includes VM information."""
    vms = [
        make_mock_vm("vm-0", state=vm_pb2.VM_STATE_READY),
        make_mock_vm("vm-1", state=vm_pb2.VM_STATE_READY),
    ]
    vm_group = vm_group_factory(vms)

    proto = vm_group.to_proto()

    assert proto.slice_id == "test-slice-001"
    assert proto.scale_group == "test-group"
    assert len(proto.vms) == 2
    assert all(vm.state == vm_pb2.VM_STATE_READY for vm in proto.vms)


@patch("iris.cluster.vm.gcp_tpu_platform.subprocess.run")
def test_vm_group_terminate_stops_and_unregisters_vms(
    mock_run: MagicMock,
    vm_group_factory: Callable[[list[MagicMock]], VmGroupProtocol],
    registry: VmRegistry,
):
    """VmGroup.terminate() stops all VMs and unregisters them from the registry."""
    vms = [make_mock_vm("vm-0"), make_mock_vm("vm-1")]
    for vm in vms:
        registry.register(vm)

    vm_group = vm_group_factory(vms)
    assert registry.vm_count() == 2

    vm_group.terminate()

    for vm in vms:
        vm.stop.assert_called_once()
    assert registry.vm_count() == 0


# =============================================================================
# VmManagerProtocol Tests - Parameterized
# =============================================================================


@pytest.fixture(params=["tpu", "manual"])
def vm_manager_factory(
    request: pytest.FixtureRequest,
    mock_factory: MagicMock,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
    v5p8_scale_group: config_pb2.ScaleGroupConfig,
    manual_scale_group: config_pb2.ScaleGroupConfig,
) -> tuple[str, object]:
    """Factory that creates either TpuVmManager or ManualVmManager."""
    if request.param == "tpu":
        manager = TpuVmManager(
            project_id="test-project",
            config=v5p8_scale_group,
            bootstrap_config=bootstrap_config,
            timeouts=timeout_config,
            vm_factory=mock_factory,
        )
        return ("tpu", manager)
    else:
        manager = ManualVmManager(
            hosts=["10.0.0.1", "10.0.0.2"],
            config=manual_scale_group,
            bootstrap_config=bootstrap_config,
            timeouts=timeout_config,
            vm_factory=mock_factory,
        )
        return ("manual", manager)


@pytest.fixture
def mock_factory(registry: VmRegistry) -> MagicMock:
    """Create a mock VmFactory that tracks created VMs."""
    mock = MagicMock()
    mock.registry = registry

    def create_vm_side_effect(**kwargs):
        vm = MagicMock()
        vm.info = vm_pb2.VmInfo(
            vm_id=kwargs["vm_id"],
            slice_id=kwargs["slice_id"],
            scale_group=kwargs["scale_group"],
            zone=kwargs["zone"],
            address=kwargs.get("address", ""),
            state=vm_pb2.VM_STATE_BOOTING,
        )
        return vm

    mock.create_vm.side_effect = create_vm_side_effect
    return mock


@patch("iris.cluster.vm.gcp_tpu_platform.subprocess.run")
def test_vm_manager_create_returns_vm_group(
    mock_run: MagicMock,
    vm_manager_factory: tuple[str, object],
):
    """VmManager.create_vm_group() returns a VmGroup with VMs."""
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    platform_type, manager = vm_manager_factory

    vm_group = manager.create_vm_group()

    assert vm_group is not None
    assert len(vm_group.vms()) >= 1
    if platform_type == "tpu":
        assert isinstance(vm_group, TpuVmGroup)
    else:
        assert isinstance(vm_group, ManualVmGroup)


@patch("iris.cluster.vm.gcp_tpu_platform.subprocess.run")
def test_vm_manager_create_with_tags_propagates(
    mock_run: MagicMock,
    mock_factory: MagicMock,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
    manual_scale_group: config_pb2.ScaleGroupConfig,
):
    """VmManager.create_vm_group(tags=...) propagates tags to VMs."""
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    # Test with ManualVmManager since it's easier to verify label propagation
    manager = ManualVmManager(
        hosts=["10.0.0.1"],
        config=manual_scale_group,
        bootstrap_config=bootstrap_config,
        timeouts=timeout_config,
        vm_factory=mock_factory,
    )

    manager.create_vm_group(tags={"env": "test", "team": "ml"})

    create_call = mock_factory.create_vm.call_args
    labels = create_call.kwargs["labels"]
    assert labels["env"] == "test"
    assert labels["team"] == "ml"


# =============================================================================
# TpuVmManager Platform-Specific Tests
# =============================================================================


@patch("iris.cluster.vm.gcp_tpu_platform.subprocess.run")
def test_tpu_manager_create_raises_on_gcloud_failure(
    mock_run: MagicMock,
    mock_factory: MagicMock,
    v5p8_scale_group: config_pb2.ScaleGroupConfig,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
):
    """TpuVmManager.create_vm_group() raises RuntimeError when gcloud fails."""
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="TPU quota exceeded")

    manager = TpuVmManager(
        project_id="test-project",
        config=v5p8_scale_group,
        bootstrap_config=bootstrap_config,
        timeouts=timeout_config,
        vm_factory=mock_factory,
    )

    with pytest.raises(RuntimeError, match="Failed to create TPU"):
        manager.create_vm_group()


@pytest.mark.parametrize(
    "accelerator_variant,expected_vm_count",
    [
        ("v5p-8", 1),  # Single host
        ("v5p-16", 2),  # Multi-host (2 VMs)
    ],
)
@patch("iris.cluster.vm.gcp_tpu_platform.subprocess.run")
def test_tpu_manager_creates_correct_vm_count_for_topology(
    mock_run: MagicMock,
    accelerator_variant: str,
    expected_vm_count: int,
    mock_factory: MagicMock,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
):
    """TpuVmManager creates correct number of VMs based on accelerator topology."""
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    config = config_pb2.ScaleGroupConfig(
        name=f"tpu-{accelerator_variant}",
        min_slices=0,
        max_slices=10,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant=accelerator_variant,
        runtime_version="v2-alpha-tpuv5",
        zones=["us-central1-a"],
    )

    manager = TpuVmManager(
        project_id="test-project",
        config=config,
        bootstrap_config=bootstrap_config,
        timeouts=timeout_config,
        vm_factory=mock_factory,
    )

    vm_group = manager.create_vm_group()

    assert mock_factory.create_vm.call_count == expected_vm_count
    assert len(vm_group.vms()) == expected_vm_count


@patch("iris.cluster.vm.gcp_tpu_platform.subprocess.run")
def test_tpu_manager_vm_ids_include_worker_index_for_multi_host(
    mock_run: MagicMock,
    mock_factory: MagicMock,
    v5p16_scale_group: config_pb2.ScaleGroupConfig,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
):
    """TpuVmManager generates VM IDs with worker index suffix for multi-host TPUs."""
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    manager = TpuVmManager(
        project_id="test-project",
        config=v5p16_scale_group,
        bootstrap_config=bootstrap_config,
        timeouts=timeout_config,
        vm_factory=mock_factory,
    )

    manager.create_vm_group()

    calls = mock_factory.create_vm.call_args_list
    vm_ids = [c.kwargs["vm_id"] for c in calls]

    assert any("-worker-0" in vm_id for vm_id in vm_ids)
    assert any("-worker-1" in vm_id for vm_id in vm_ids)


@patch("iris.cluster.vm.gcp_tpu_platform.subprocess.run")
def test_tpu_manager_discover_creates_groups_for_found_tpus(
    mock_run: MagicMock,
    mock_factory: MagicMock,
    v5p8_scale_group: config_pb2.ScaleGroupConfig,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
):
    """TpuVmManager.discover_vm_groups() creates TpuVmGroup for each discovered TPU."""
    tpu_data = [
        {
            "name": "iris-tpu-v5p-8-12345",
            "state": "READY",
            "labels": {
                "iris-managed": "true",
                "iris-scale-group": v5p8_scale_group.name,
                "iris-slice-id": "iris-tpu-v5p-8-12345",
            },
            "networkEndpoints": [{"ipAddress": "10.0.0.1"}],
        },
        {
            "name": "iris-tpu-v5p-8-67890",
            "state": "READY",
            "labels": {
                "iris-managed": "true",
                "iris-scale-group": v5p8_scale_group.name,
                "iris-slice-id": "iris-tpu-v5p-8-67890",
            },
            "networkEndpoints": [{"ipAddress": "10.0.0.2"}],
        },
    ]
    mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(tpu_data), stderr="")

    manager = TpuVmManager(
        project_id="test-project",
        config=v5p8_scale_group,
        bootstrap_config=bootstrap_config,
        timeouts=timeout_config,
        vm_factory=mock_factory,
    )

    vm_groups = manager.discover_vm_groups()

    assert len(vm_groups) == 2
    slice_ids = {g.slice_id for g in vm_groups}
    assert "iris-tpu-v5p-8-12345" in slice_ids
    assert "iris-tpu-v5p-8-67890" in slice_ids


@patch("iris.cluster.vm.gcp_tpu_platform.subprocess.run")
def test_tpu_manager_discover_skips_tpus_in_deleting_state(
    mock_run: MagicMock,
    mock_factory: MagicMock,
    v5p8_scale_group: config_pb2.ScaleGroupConfig,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
):
    """TpuVmManager.discover_vm_groups() skips TPUs in DELETING or other non-adoptable states."""
    tpu_data = [
        {
            "name": "iris-tpu-ready",
            "state": "READY",
            "labels": {"iris-scale-group": v5p8_scale_group.name},
            "networkEndpoints": [{"ipAddress": "10.0.0.1"}],
        },
        {
            "name": "iris-tpu-creating",
            "state": "CREATING",
            "labels": {"iris-scale-group": v5p8_scale_group.name},
            "networkEndpoints": [{"ipAddress": "10.0.0.2"}],
        },
        {
            "name": "iris-tpu-deleting",
            "state": "DELETING",
            "labels": {"iris-scale-group": v5p8_scale_group.name},
            "networkEndpoints": [{"ipAddress": "10.0.0.3"}],
        },
        {
            "name": "iris-tpu-stopped",
            "state": "STOPPED",
            "labels": {"iris-scale-group": v5p8_scale_group.name},
            "networkEndpoints": [{"ipAddress": "10.0.0.4"}],
        },
    ]
    mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(tpu_data), stderr="")

    manager = TpuVmManager(
        project_id="test-project",
        config=v5p8_scale_group,
        bootstrap_config=bootstrap_config,
        timeouts=timeout_config,
        vm_factory=mock_factory,
    )

    vm_groups = manager.discover_vm_groups()

    # Only READY and CREATING TPUs should be adopted
    assert len(vm_groups) == 2
    slice_ids = {g.slice_id for g in vm_groups}
    assert "iris-tpu-ready" in slice_ids
    assert "iris-tpu-creating" in slice_ids
    assert "iris-tpu-deleting" not in slice_ids
    assert "iris-tpu-stopped" not in slice_ids


@patch("iris.cluster.vm.gcp_tpu_platform.subprocess.run")
def test_tpu_vm_group_terminate_deletes_tpu_resource(mock_run: MagicMock, registry: VmRegistry):
    """TpuVmGroup.terminate() invokes gcloud to delete the TPU resource."""
    vms = [make_mock_vm("vm-0")]
    tpu_slice = TpuVmGroup(
        group_id="my-tpu-slice",
        scale_group="test-group",
        zone="us-central1-a",
        project_id="my-project",
        vms=vms,
        vm_registry=registry,
    )

    tpu_slice.terminate()

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "gcloud" in cmd
    assert "delete" in cmd
    assert "my-tpu-slice" in cmd


# =============================================================================
# ManualVmManager Platform-Specific Tests
# =============================================================================


def test_manual_manager_raises_pool_exhausted_when_no_hosts(
    mock_factory: MagicMock,
    manual_scale_group: config_pb2.ScaleGroupConfig,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
):
    """ManualVmManager.create_vm_group() raises PoolExhaustedError when no hosts available."""
    manager = ManualVmManager(
        hosts=["10.0.0.1"],
        config=manual_scale_group,
        bootstrap_config=bootstrap_config,
        timeouts=timeout_config,
        vm_factory=mock_factory,
    )

    manager.create_vm_group()  # Use the only host

    with pytest.raises(PoolExhaustedError, match="No hosts available"):
        manager.create_vm_group()


def test_manual_manager_removes_host_from_pool_on_create(
    mock_factory: MagicMock,
    manual_scale_group: config_pb2.ScaleGroupConfig,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
):
    """ManualVmManager.create_vm_group() removes host from available pool."""
    hosts = ["10.0.0.1", "10.0.0.2"]
    manager = ManualVmManager(
        hosts=hosts,
        config=manual_scale_group,
        bootstrap_config=bootstrap_config,
        timeouts=timeout_config,
        vm_factory=mock_factory,
    )

    assert manager.available_host_count == 2

    manager.create_vm_group()

    assert manager.available_host_count == 1


def test_manual_vm_group_terminate_returns_host_to_pool(
    mock_factory: MagicMock,
    manual_scale_group: config_pb2.ScaleGroupConfig,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
):
    """ManualVmGroup.terminate() returns host to the manager's available pool."""
    hosts = ["10.0.0.1"]
    manager = ManualVmManager(
        hosts=hosts,
        config=manual_scale_group,
        bootstrap_config=bootstrap_config,
        timeouts=timeout_config,
        vm_factory=mock_factory,
    )

    vm_group = manager.create_vm_group()
    assert manager.available_host_count == 0

    vm_group.terminate()
    assert manager.available_host_count == 1


def test_manual_vm_group_terminate_calls_on_terminate_callback(registry: VmRegistry):
    """ManualVmGroup.terminate() calls the on_terminate callback with host addresses."""
    vm = MagicMock()
    vm.info = vm_pb2.VmInfo(
        vm_id="vm-1",
        slice_id="slice-1",
        scale_group="manual",
        state=vm_pb2.VM_STATE_READY,
        address="10.0.0.1",
    )

    callback_hosts: list[str] = []

    def on_terminate(hosts: list[str]) -> None:
        callback_hosts.extend(hosts)

    manual_vm_group = ManualVmGroup(
        group_id="slice-1",
        scale_group="manual",
        vms=[vm],
        vm_registry=registry,
        on_terminate=on_terminate,
    )

    manual_vm_group.terminate()

    assert callback_hosts == ["10.0.0.1"]


@patch("iris.cluster.vm.manual_platform.DirectSshConnection")
def test_manual_manager_discovers_hosts_with_running_workers(
    mock_ssh_conn_class: MagicMock,
    mock_factory: MagicMock,
    manual_scale_group: config_pb2.ScaleGroupConfig,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
):
    """ManualVmManager.discover_vm_groups() creates groups for hosts with running workers."""
    hosts = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    manager = ManualVmManager(
        hosts=hosts,
        config=manual_scale_group,
        bootstrap_config=bootstrap_config,
        timeouts=timeout_config,
        vm_factory=mock_factory,
    )

    # Track which host each connection is for and return appropriate health check result
    def make_mock_connection(host, **kwargs):
        mock_conn = MagicMock()
        mock_conn.host = host
        # Simulate healthy for 10.0.0.1 and 10.0.0.3, unhealthy for 10.0.0.2
        if host in ["10.0.0.1", "10.0.0.3"]:
            mock_conn.run.return_value = MagicMock(returncode=0)
        else:
            mock_conn.run.return_value = MagicMock(returncode=1)
        return mock_conn

    mock_ssh_conn_class.side_effect = make_mock_connection

    vm_groups = manager.discover_vm_groups()

    assert len(vm_groups) == 2
    assert manager.available_host_count == 1
    assert "10.0.0.2" in manager._available_hosts


# =============================================================================
# VmRegistry Tests
# =============================================================================


def test_registry_register_replaces_existing_vm(registry: VmRegistry):
    """Registering a VM with same ID replaces the existing one."""
    vm1 = MagicMock()
    vm1.info = vm_pb2.VmInfo(vm_id="vm-001", address="10.0.0.1")
    vm2 = MagicMock()
    vm2.info = vm_pb2.VmInfo(vm_id="vm-001", address="10.0.0.2")

    registry.register(vm1)
    registry.register(vm2)

    assert registry.vm_count() == 1
    retrieved = registry.get_vm("vm-001")
    assert retrieved is not None
    assert retrieved.info.address == "10.0.0.2"


def test_registry_unregister_removes_vm(registry: VmRegistry):
    """Unregistering a VM removes it from the registry."""
    vm = MagicMock()
    vm.info = vm_pb2.VmInfo(vm_id="test-vm-001")

    registry.register(vm)
    registry.unregister("test-vm-001")

    assert registry.vm_count() == 0
    assert registry.get_vm("test-vm-001") is None


def test_registry_concurrent_access_is_safe(registry: VmRegistry):
    """Registry handles concurrent register/unregister safely."""
    errors: list[Exception] = []
    num_operations = 100

    def worker(worker_id: int):
        try:
            for i in range(num_operations):
                vm = MagicMock()
                vm_id = f"vm-{worker_id}-{i}"
                vm.info = vm_pb2.VmInfo(vm_id=vm_id)

                registry.register(vm)
                registry.unregister(vm_id)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errors during concurrent access: {errors}"
    assert registry.vm_count() == 0


# =============================================================================
# TrackedVmFactory Tests
# =============================================================================


@patch("iris.cluster.vm.managed_vm.ManagedVm")
def test_factory_creates_registers_and_starts_vm(
    mock_managed_vm_class: MagicMock,
    registry: VmRegistry,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
):
    """TrackedVmFactory creates a VM, registers it, and starts its lifecycle."""
    factory = TrackedVmFactory(registry)

    mock_vm = MagicMock()
    mock_vm.info = vm_pb2.VmInfo(vm_id="test-vm-001")
    mock_managed_vm_class.return_value = mock_vm

    conn = MagicMock()
    conn.address = "10.0.0.1"
    conn.zone = ""

    factory.create_vm(
        vm_id="test-vm-001",
        slice_id="slice-001",
        scale_group="test-group",
        zone="us-central1-a",
        conn=conn,
        bootstrap_config=bootstrap_config,
        timeouts=timeout_config,
        labels={},
    )

    assert registry.vm_count() == 1
    mock_vm.start.assert_called_once()


@patch("iris.cluster.vm.managed_vm.ManagedVm")
def test_factory_and_registry_integration(
    mock_managed_vm_class: MagicMock,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
):
    """Full workflow: create VMs via factory, query via registry, unregister."""
    registry = VmRegistry()
    factory = TrackedVmFactory(registry)

    mock_vms = []
    for i in range(3):
        mock_vm = MagicMock()
        mock_vm.info = vm_pb2.VmInfo(
            vm_id=f"vm-{i}",
            slice_id="slice-001",
            scale_group="test-group",
        )
        mock_vm.init_log.return_value = f"log for vm-{i}"
        mock_vms.append(mock_vm)

    mock_managed_vm_class.side_effect = mock_vms

    for i in range(3):
        conn = MagicMock()
        conn.address = f"10.0.0.{i}"
        conn.zone = ""
        factory.create_vm(
            vm_id=f"vm-{i}",
            slice_id="slice-001",
            scale_group="test-group",
            zone="us-central1-a",
            conn=conn,
            bootstrap_config=bootstrap_config,
            timeouts=timeout_config,
            labels={},
        )

    assert registry.vm_count() == 3

    # Query by ID returns correct VM
    retrieved = registry.get_vm("vm-1")
    assert retrieved is not None
    assert retrieved.info.vm_id == "vm-1"

    # Get init log
    log = registry.get_init_log("vm-2")
    assert log == "log for vm-2"

    # Unregister
    registry.unregister("vm-1")
    assert registry.vm_count() == 2
    assert registry.get_vm("vm-1") is None


# =============================================================================
# Multi-Host Integration Tests
# =============================================================================


@patch("iris.cluster.vm.gcp_tpu_platform.subprocess.run")
def test_multi_host_tpu_slice_status_aggregates_all_workers(mock_run: MagicMock, registry: VmRegistry):
    """A multi-host TPU slice reports correct aggregate status from all workers."""
    vms = [make_mock_vm(f"vm-{i}", state=vm_pb2.VM_STATE_READY) for i in range(4)]
    tpu_slice = TpuVmGroup(
        group_id="v5p-128-slice",
        scale_group="tpu-v5p-128",
        zone="us-central1-a",
        project_id="test-project",
        vms=vms,
        vm_registry=registry,
    )

    status = tpu_slice.status()

    assert status.vm_count == 4
    assert status.ready_count == 4
    assert status.all_ready is True


@patch("iris.cluster.vm.gcp_tpu_platform.subprocess.run")
def test_multi_host_tpu_partial_failure_marks_slice_as_failed(mock_run: MagicMock, registry: VmRegistry):
    """When one host fails, the whole slice is marked as failed."""
    vms = [
        make_mock_vm("vm-0", state=vm_pb2.VM_STATE_READY),
        make_mock_vm("vm-1", state=vm_pb2.VM_STATE_READY),
        make_mock_vm("vm-2", state=vm_pb2.VM_STATE_FAILED, init_error="Network timeout"),
        make_mock_vm("vm-3", state=vm_pb2.VM_STATE_READY),
    ]
    tpu_slice = TpuVmGroup(
        group_id="v5p-128-slice",
        scale_group="tpu-v5p-128",
        zone="us-central1-a",
        project_id="test-project",
        vms=vms,
        vm_registry=registry,
    )

    status = tpu_slice.status()

    assert status.vm_count == 4
    assert status.ready_count == 3
    assert status.all_ready is False
    assert status.any_failed is True
    assert "Network timeout" in status.error_messages


@patch("iris.cluster.vm.gcp_tpu_platform.subprocess.run")
def test_multi_host_tpu_terminate_stops_all_workers(mock_run: MagicMock, registry: VmRegistry):
    """Terminating a multi-host slice stops all VMs."""
    vms = [make_mock_vm(f"vm-{i}") for i in range(4)]
    for vm in vms:
        registry.register(vm)

    tpu_slice = TpuVmGroup(
        group_id="v5p-128-slice",
        scale_group="tpu-v5p-128",
        zone="us-central1-a",
        project_id="test-project",
        vms=vms,
        vm_registry=registry,
    )

    tpu_slice.terminate()

    for vm in vms:
        vm.stop.assert_called_once()
    assert registry.vm_count() == 0
    mock_run.assert_called_once()  # Only one gcloud delete call for the TPU
