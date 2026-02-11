# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Autoscaler behavior.

These tests focus on observable behavior - scaling decisions based on demand,
execution of those decisions, and integration with ScalingGroup.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from iris.cluster.controller.autoscaler import (
    Autoscaler,
    DemandEntry,
    ScalingAction,
    ScalingDecision,
    route_demand,
)
from iris.cluster.controller.scaling_group import ScalingGroup
from iris.cluster.platform.base import CloudSliceState, CloudVmState, QuotaExhaustedError, SliceStatus, VmStatus
from iris.cluster.types import DeviceType, VmWorkerStatus
from iris.rpc import cluster_pb2, config_pb2, vm_pb2
from iris.time_utils import Duration, Timestamp
from tests.cluster.platform.fakes import FailureMode, FakePlatform, FakePlatformConfig

# --- Test fixtures and helpers ---


def make_demand_entries(
    count: int,
    *,
    device_type: DeviceType = DeviceType.TPU,
    device_variant: str | None = "v5p-8",
    preemptible: bool | None = None,
    task_prefix: str = "task",
) -> list[DemandEntry]:
    if count <= 0:
        return []
    resources = cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024)
    return [
        DemandEntry(
            task_ids=[f"{task_prefix}-{i}"],
            coschedule_group_id=None,
            device_type=device_type,
            device_variant=device_variant,
            constraints=[],
            resources=resources,
            preemptible=preemptible,
        )
        for i in range(count)
    ]


DEFAULT_RESOURCES = config_pb2.ScaleGroupResources(
    cpu=128,
    memory_bytes=128 * 1024**3,
    disk_bytes=100 * 1024**3,
    gpu_count=8,
    tpu_count=8,
)


def ensure_scale_group_resources(config: config_pb2.ScaleGroupConfig) -> config_pb2.ScaleGroupConfig:
    if not config.HasField("resources"):
        config.resources.CopyFrom(DEFAULT_RESOURCES)
    if not config.HasField("slice_size"):
        config.slice_size = 1
    return config


def make_scale_group_config(**kwargs: object) -> config_pb2.ScaleGroupConfig:
    if "accelerator_type" not in kwargs:
        kwargs["accelerator_type"] = config_pb2.ACCELERATOR_TYPE_TPU
    if "accelerator_variant" not in kwargs:
        kwargs["accelerator_variant"] = "v5p-8"
    # Extract fields that moved to slice_template
    runtime_version = kwargs.pop("runtime_version", None)
    zones = kwargs.pop("zones", None)
    preemptible = kwargs.pop("preemptible", None)
    config = ensure_scale_group_resources(config_pb2.ScaleGroupConfig(**kwargs))
    if preemptible is not None:
        config.slice_template.preemptible = preemptible
    if runtime_version or zones:
        gcp = config.slice_template.gcp
        if runtime_version:
            gcp.runtime_version = runtime_version
        if zones:
            gcp.zone = zones[0]
    return config


def _cloud_vm_state_from_iris(state: vm_pb2.VmState) -> CloudVmState:
    """Reverse map from Iris VM state to CloudVmState for test setup."""
    if state == vm_pb2.VM_STATE_READY:
        return CloudVmState.RUNNING
    if state == vm_pb2.VM_STATE_FAILED:
        return CloudVmState.STOPPED
    if state == vm_pb2.VM_STATE_TERMINATED:
        return CloudVmState.TERMINATED
    return CloudVmState.UNKNOWN


def make_mock_vm_handle(vm_id: str, address: str, state: vm_pb2.VmState) -> MagicMock:
    """Create a mock VmHandle for testing."""
    handle = MagicMock()
    handle.vm_id = vm_id
    handle.internal_address = address
    handle.external_address = None
    handle.status.return_value = VmStatus(state=_cloud_vm_state_from_iris(state))
    return handle


def make_mock_slice_handle(
    slice_id: str,
    scale_group: str = "test-group",
    all_ready: bool = False,
    any_failed: bool = False,
    vm_states: list[vm_pb2.VmState] | None = None,
    created_at_ms: int = 1000000,
) -> MagicMock:
    """Create a mock SliceHandle for testing."""
    handle = MagicMock()
    handle.slice_id = slice_id
    handle.scale_group = scale_group
    handle.zone = "us-central1-a"
    handle.labels = {"iris-scale-group": scale_group, "iris-managed": "true"}
    handle.created_at = Timestamp.from_ms(created_at_ms)

    if vm_states is None:
        if any_failed:
            vm_states = [vm_pb2.VM_STATE_FAILED]
        elif all_ready:
            vm_states = [vm_pb2.VM_STATE_READY]
        else:
            vm_states = [vm_pb2.VM_STATE_BOOTING]

    # Derive slice state from VM states
    if any(s == vm_pb2.VM_STATE_FAILED for s in vm_states):
        slice_state = CloudSliceState.READY
    elif all(s == vm_pb2.VM_STATE_READY for s in vm_states):
        slice_state = CloudSliceState.READY
    elif all(s == vm_pb2.VM_STATE_TERMINATED for s in vm_states):
        slice_state = CloudSliceState.DELETING
    else:
        slice_state = CloudSliceState.CREATING

    slice_hash = abs(hash(slice_id)) % 256
    vm_handles = []
    for i, state in enumerate(vm_states):
        vm_handle = make_mock_vm_handle(
            vm_id=f"{slice_id}-vm-{i}",
            address=f"10.0.{slice_hash}.{i}",
            state=state,
        )
        vm_handles.append(vm_handle)

    handle.describe.return_value = SliceStatus(state=slice_state, vm_count=len(vm_states), vms=vm_handles)

    return handle


def make_mock_platform(slices_to_discover: list[MagicMock] | None = None) -> MagicMock:
    """Create a mock Platform for testing."""
    platform = MagicMock()
    platform.list_slices.return_value = slices_to_discover or []

    create_count = [0]

    def create_slice_side_effect(config: config_pb2.SliceConfig) -> MagicMock:
        create_count[0] += 1
        slice_id = f"new-slice-{create_count[0]}"
        return make_mock_slice_handle(slice_id)

    platform.create_slice.side_effect = create_slice_side_effect
    return platform


@pytest.fixture
def scale_group_config() -> config_pb2.ScaleGroupConfig:
    """A standard scale group configuration."""
    return make_scale_group_config(
        name="test-group",
        min_slices=0,
        max_slices=5,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5p-8",
        runtime_version="v2-alpha-tpuv5",
        zones=["us-central1-a"],
    )


@pytest.fixture
def empty_autoscaler(scale_group_config):
    """Empty autoscaler ready for scale-up tests."""
    platform = make_mock_platform()
    group = ScalingGroup(scale_group_config, platform, scale_up_cooldown=Duration.from_ms(0))
    autoscaler = make_autoscaler({"test-group": group})
    yield autoscaler
    autoscaler.shutdown()


@pytest.fixture
def autoscaler_with_ready_slices(scale_group_config):
    """Autoscaler with 2 ready slices for scale-down tests."""
    discovered = [
        make_mock_slice_handle("slice-001", all_ready=True),
        make_mock_slice_handle("slice-002", all_ready=True),
    ]
    platform = make_mock_platform(slices_to_discover=discovered)
    group = ScalingGroup(
        scale_group_config, platform, scale_down_cooldown=Duration.from_ms(0), idle_threshold=Duration.from_ms(0)
    )
    group.reconcile()
    autoscaler = make_autoscaler({"test-group": group})
    yield autoscaler, group, platform
    autoscaler.shutdown()


def make_autoscaler(
    scale_groups: dict[str, ScalingGroup],
    config: config_pb2.AutoscalerConfig | None = None,
    platform: MagicMock | None = None,
    worker_bootstrap=None,
) -> Autoscaler:
    """Create an Autoscaler with the given groups."""
    mock_platform = platform or make_mock_platform()

    if config:
        return Autoscaler.from_config(
            scale_groups=scale_groups,
            config=config,
            platform=mock_platform,
            worker_bootstrap=worker_bootstrap,
        )
    else:
        return Autoscaler(
            scale_groups=scale_groups,
            evaluation_interval=Duration.from_seconds(0.1),
            platform=mock_platform,
            worker_bootstrap=worker_bootstrap,
        )


# --- Tests for scaling decisions ---


class TestAutoscalerScaleUp:
    """Tests for scale-up decisions."""

    def test_scales_up_when_demand_exceeds_capacity(self, empty_autoscaler: Autoscaler):
        """Evaluates scale-up when demand > capacity."""
        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8")
        decisions = empty_autoscaler.evaluate(demand)

        assert len(decisions) == 1
        assert decisions[0].action == ScalingAction.SCALE_UP
        assert decisions[0].scale_group == "test-group"
        assert "demand=2 > capacity=0" in decisions[0].reason

    @pytest.mark.parametrize(
        "discovered,demand_count,reason",
        [
            ([make_mock_slice_handle(f"slice-{i}") for i in range(5)], 10, "at_max_slices"),
            (
                [
                    make_mock_slice_handle("slice-001", all_ready=True),
                    make_mock_slice_handle("slice-002", all_ready=True),
                ],
                2,
                "capacity_meets_demand",
            ),
            (
                [
                    make_mock_slice_handle("slice-001", vm_states=[vm_pb2.VM_STATE_BOOTING]),
                    make_mock_slice_handle("slice-002", vm_states=[vm_pb2.VM_STATE_BOOTING]),
                ],
                2,
                "pending_slices_count",
            ),
        ],
        ids=["at_max_slices", "capacity_meets_demand", "pending_slices_count"],
    )
    def test_no_scale_up_when_condition_met(
        self, scale_group_config: config_pb2.ScaleGroupConfig, discovered: list, demand_count: int, reason: str
    ):
        """Does not scale up when various conditions are met."""
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform, scale_up_cooldown=Duration.from_ms(0))
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(demand_count, device_type=DeviceType.TPU, device_variant="v5p-8")
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 0

    def test_no_scale_up_during_backoff(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Does not scale up during backoff period."""
        platform = make_mock_platform()
        group = ScalingGroup(
            scale_group_config,
            platform,
            scale_up_cooldown=Duration.from_ms(0),
            backoff_initial=Duration.from_hours(1),
        )
        group.record_failure(timestamp=Timestamp.now())
        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8")
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 0

    def test_no_scale_up_during_cooldown(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Does not scale up during cooldown period."""
        platform = make_mock_platform()
        group = ScalingGroup(scale_group_config, platform, scale_up_cooldown=Duration.from_ms(3600_000))
        ts = Timestamp.now()
        group.begin_scale_up()
        handle = group.scale_up(timestamp=ts)
        group.complete_scale_up(handle, ts)
        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8")
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 0

    def test_scales_up_to_enforce_min_slices(self):
        """Scales up to enforce min_slices even with zero demand."""
        config = make_scale_group_config(
            name="test-group",
            min_slices=2,
            max_slices=5,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            runtime_version="v2-alpha-tpuv5",
            zones=["us-central1-a"],
        )
        platform = make_mock_platform()
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0))
        autoscaler = make_autoscaler({"test-group": group})

        decisions = autoscaler.evaluate([])

        assert len(decisions) == 1
        assert decisions[0].action == ScalingAction.SCALE_UP
        assert "below min_slices" in decisions[0].reason
        assert "0 < 2" in decisions[0].reason

    def test_no_scale_up_when_at_min_slices(self):
        """Does not scale up when already at min_slices and no demand."""
        config = make_scale_group_config(
            name="test-group",
            min_slices=2,
            max_slices=5,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            runtime_version="v2-alpha-tpuv5",
            zones=["us-central1-a"],
        )
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True),
            make_mock_slice_handle("slice-002", all_ready=True),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0))
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        decisions = autoscaler.evaluate([])

        assert len(decisions) == 0


class TestAutoscalerScaleDown:
    """Tests for scale-down behavior (delegated to ScalingGroup)."""

    def test_scales_down_idle_slice_via_run_once(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """run_once() scales down idle slices via ScalingGroup.scale_down_if_idle()."""
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True, created_at_ms=100000),
            make_mock_slice_handle("slice-002", all_ready=True, created_at_ms=200000),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(
            scale_group_config,
            platform,
            scale_down_cooldown=Duration.from_ms(0),
            idle_threshold=Duration.from_ms(1000),
        )
        group.reconcile()
        group.refresh_all_slices()
        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")

        # Get VM addresses from the adapter
        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        slice_001_addr = slice_001.describe().vms[0].internal_address
        slice_002_addr = slice_002.describe().vms[0].internal_address
        vm_status_map = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset()),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset()),
        }

        autoscaler.run_once(demand, vm_status_map, timestamp=Timestamp.from_ms(10_000))

        assert group.slice_count() == 1

    def test_no_scale_down_at_min_slices(self):
        """Does not scale down when at min_slices."""
        config = make_scale_group_config(
            name="test-group",
            min_slices=2,
            max_slices=5,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            runtime_version="v2-alpha-tpuv5",
            zones=["us-central2-b"],
        )
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True),
            make_mock_slice_handle("slice-002", all_ready=True),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(
            config, platform, scale_down_cooldown=Duration.from_ms(0), idle_threshold=Duration.from_ms(0)
        )
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(0, device_type=DeviceType.TPU, device_variant="v5p-8")
        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        slice_001_addr = slice_001.describe().vms[0].internal_address
        slice_002_addr = slice_002.describe().vms[0].internal_address
        vm_status_map = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset()),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset()),
        }

        autoscaler.run_once(demand, vm_status_map)

        assert group.slice_count() == 2

    def test_no_scale_down_until_idle_threshold(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Does not scale down until slice has been idle long enough."""
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True),
            make_mock_slice_handle("slice-002", all_ready=True),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(
            scale_group_config,
            platform,
            scale_down_cooldown=Duration.from_ms(0),
            idle_threshold=Duration.from_ms(300_000),
        )
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        slice_001_addr = slice_001.describe().vms[0].internal_address
        slice_002_addr = slice_002.describe().vms[0].internal_address

        vm_status_map_active = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset({"task-1"})),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset({"task-2"})),
        }
        group.update_slice_activity(vm_status_map_active, timestamp=Timestamp.from_ms(1000))

        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")
        vm_status_map_idle = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset()),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset()),
        }

        autoscaler.run_once(demand, vm_status_map_idle, timestamp=Timestamp.from_ms(100_000))

        assert group.slice_count() == 2

    def test_failed_slices_cleaned_up_before_evaluate(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Failed slices are cleaned up in run_once() before evaluate()."""
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True, created_at_ms=100000),
            make_mock_slice_handle("slice-002", all_ready=True, created_at_ms=150000),
            make_mock_slice_handle("slice-003", any_failed=True, created_at_ms=200000),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(
            scale_group_config,
            platform,
            scale_down_cooldown=Duration.from_ms(0),
            idle_threshold=Duration.from_ms(0),
        )
        group.reconcile()
        group.refresh_all_slices()
        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")
        worker_idle_map = {}

        autoscaler.run_once(demand, worker_idle_map)

        assert group.get_slice("slice-003") is None
        assert group.slice_count() == 2

    def test_no_scale_down_during_cooldown(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Does not scale down during cooldown period."""
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True),
            make_mock_slice_handle("slice-002", all_ready=True),
            make_mock_slice_handle("slice-003", all_ready=True),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(
            scale_group_config,
            platform,
            scale_down_cooldown=Duration.from_ms(3600_000),
            idle_threshold=Duration.from_ms(0),
        )
        group.reconcile()
        group.scale_down("slice-003", timestamp=Timestamp.now())
        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")
        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        slice_001_addr = slice_001.describe().vms[0].internal_address
        slice_002_addr = slice_002.describe().vms[0].internal_address
        vm_status_map = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset()),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset()),
        }

        autoscaler.run_once(demand, vm_status_map)

        assert group.slice_count() == 2


class TestAutoscalerExecution:
    """Tests for decision execution."""

    def test_execute_scale_up_creates_slice(self, empty_autoscaler: Autoscaler):
        """execute() creates a slice via ScalingGroup."""
        decision = ScalingDecision(
            scale_group="test-group",
            action=ScalingAction.SCALE_UP,
            reason="test",
        )

        empty_autoscaler.execute([decision], timestamp=Timestamp.from_ms(1000))
        empty_autoscaler._wait_for_inflight()

        group = empty_autoscaler.groups["test-group"]
        assert group.slice_count() == 1

    def test_run_once_cleans_up_failed_slice(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """run_once() terminates failed slices via ScalingGroup.cleanup_failed_slices()."""
        mock_handle = make_mock_slice_handle("slice-001", any_failed=True)
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform, scale_down_cooldown=Duration.from_ms(0))
        group.reconcile()
        group.refresh_all_slices()
        autoscaler = make_autoscaler({"test-group": group})

        autoscaler.run_once([], {}, timestamp=Timestamp.from_ms(1000))

        assert group.slice_count() == 0

    def test_execute_records_failure_on_scale_up_error(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """execute() records failure when scale-up fails."""
        platform = make_mock_platform()
        platform.create_slice.side_effect = RuntimeError("TPU unavailable")
        backoff = Duration.from_seconds(5.0)
        group = ScalingGroup(
            scale_group_config, platform, scale_up_cooldown=Duration.from_ms(0), backoff_initial=backoff
        )
        autoscaler = make_autoscaler({"test-group": group})

        decision = ScalingDecision(
            scale_group="test-group",
            action=ScalingAction.SCALE_UP,
            reason="test",
        )

        autoscaler.execute([decision], timestamp=Timestamp.from_ms(1000))
        autoscaler._wait_for_inflight()

        assert group.consecutive_failures == 1
        assert group._backoff_until is not None

    def test_run_once_evaluates_and_executes(self, empty_autoscaler: Autoscaler):
        """run_once() performs evaluate then execute."""
        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8")
        vm_status_map = {}
        decisions = empty_autoscaler.run_once(demand, vm_status_map)
        empty_autoscaler._wait_for_inflight()

        assert len(decisions) == 1
        assert decisions[0].action == ScalingAction.SCALE_UP
        group = empty_autoscaler.groups["test-group"]
        assert group.slice_count() == 1

    def test_execute_skips_unknown_scale_group(self):
        """execute() skips decisions for unknown scale groups."""
        config = make_scale_group_config(name="known-group", min_slices=0, max_slices=5)
        platform = make_mock_platform()
        group = ScalingGroup(config, platform)

        autoscaler = make_autoscaler({"known-group": group})

        decisions = [
            ScalingDecision(
                scale_group="unknown-group",
                action=ScalingAction.SCALE_UP,
                reason="test",
            )
        ]

        autoscaler.execute(decisions, timestamp=Timestamp.from_ms(1000))

        assert group.slice_count() == 0


class TestAutoscalerReconcile:
    """Tests for reconcile behavior."""

    def test_reconcile_discovers_slices_in_all_groups(self):
        """reconcile() calls reconcile on all groups."""
        config1 = make_scale_group_config(name="group-1", min_slices=0, max_slices=5, zones=["us-central1-a"])
        config2 = make_scale_group_config(name="group-2", min_slices=0, max_slices=5, zones=["us-central1-a"])

        platform1 = make_mock_platform(
            slices_to_discover=[make_mock_slice_handle("slice-1", scale_group="group-1", all_ready=True)]
        )
        platform2 = make_mock_platform(
            slices_to_discover=[make_mock_slice_handle("slice-2", scale_group="group-2", all_ready=True)]
        )

        group1 = ScalingGroup(config1, platform1)
        group2 = ScalingGroup(config2, platform2)

        autoscaler = make_autoscaler({"group-1": group1, "group-2": group2})
        autoscaler.reconcile()

        assert group1.slice_count() == 1
        assert group2.slice_count() == 1


class TestAutoscalerWorkerFailure:
    """Tests for worker failure handling."""

    def test_notify_worker_failed_terminates_slice(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """notify_worker_failed() terminates the slice containing the worker."""
        mock_handle = make_mock_slice_handle("slice-001")
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        group.refresh_all_slices()
        autoscaler = make_autoscaler({"test-group": group})
        autoscaler.reconcile()

        vm_address = f"10.0.{abs(hash('slice-001')) % 256}.0"
        group.refresh_all_slices()
        autoscaler.notify_worker_failed(vm_address)

        assert group.slice_count() == 0

    def test_notify_worker_failed_unknown_worker_is_noop(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """notify_worker_failed() does nothing for unknown workers."""
        mock_handle = make_mock_slice_handle("slice-001", all_ready=True)
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        autoscaler.notify_worker_failed("10.1.2.3")

        assert group.slice_count() == 1


class TestAutoscalerIdleVerification:
    """Tests for idle verification during scale-down."""

    def test_verifies_idle_with_worker_idle_map(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Scale-down via run_once verifies workers are idle using worker_idle_map."""
        mock_handle = make_mock_slice_handle("slice-001", all_ready=True)
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(
            scale_group_config,
            platform,
            scale_down_cooldown=Duration.from_ms(0),
            idle_threshold=Duration.from_ms(0),
        )
        group.reconcile()

        autoscaler = make_autoscaler(
            scale_groups={"test-group": group},
        )

        demand = make_demand_entries(0, device_type=DeviceType.TPU, device_variant="v5p-8")

        slice_001 = group.get_slice("slice-001")
        slice_001_addr = slice_001.describe().vms[0].internal_address
        vm_status_map = {
            slice_001_addr: VmWorkerStatus(
                vm_address=slice_001_addr,
                running_task_ids=frozenset({"task-1"}),
            )
        }

        autoscaler.run_once(demand, vm_status_map)

        assert group.slice_count() == 1
        mock_handle.terminate.assert_not_called()


class TestAutoscalerStatusReporting:
    """Tests for status reporting."""

    def test_get_status_includes_all_groups(self):
        """get_status() includes status for all groups."""
        config1 = make_scale_group_config(name="group-1", min_slices=0, max_slices=5)
        config2 = make_scale_group_config(name="group-2", min_slices=0, max_slices=5)

        platform1 = make_mock_platform()
        platform2 = make_mock_platform()

        group1 = ScalingGroup(config1, platform1)
        group2 = ScalingGroup(config2, platform2)

        group1.update_demand(5)
        group2.update_demand(3)

        autoscaler = make_autoscaler({"group-1": group1, "group-2": group2})

        status = autoscaler.get_status()

        assert len(status.groups) == 2
        group_names = {g.name for g in status.groups}
        assert "group-1" in group_names
        assert "group-2" in group_names

        assert status.current_demand["group-1"] == 5
        assert status.current_demand["group-2"] == 3

    def test_get_status_includes_last_routing_decision(self):
        """get_status() includes the last routing decision."""
        config = make_scale_group_config(name="test-group", min_slices=0, max_slices=5)
        group = ScalingGroup(config, make_mock_platform())
        autoscaler = make_autoscaler({"test-group": group})

        autoscaler.evaluate(make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8"))
        status = autoscaler.get_status()

        assert status.HasField("last_routing_decision")
        assert "test-group" in status.last_routing_decision.routed_entries


class TestAutoscalerFailedSliceCleanup:
    """Tests for automatic cleanup of failed slices via run_once()."""

    def test_cleans_up_failed_slice_via_run_once(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """run_once() cleans up failed slices before evaluating scale-up."""
        discovered = [make_mock_slice_handle("failed-slice", any_failed=True)]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform, scale_up_cooldown=Duration.from_ms(0))
        group.reconcile()
        group.refresh_all_slices()
        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8")
        autoscaler.run_once(demand, {})

        assert group.get_slice("failed-slice") is None

    def test_failed_slice_cleanup_triggers_backoff(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Cleaning up a failed slice triggers backoff to prevent thrashing."""
        mock_handle = make_mock_slice_handle("failed-slice", any_failed=True)
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(
            scale_group_config,
            platform,
            scale_up_cooldown=Duration.from_ms(0),
            backoff_initial=Duration.from_seconds(60.0),
        )
        group.reconcile()
        group.refresh_all_slices()
        autoscaler = make_autoscaler({"test-group": group})

        autoscaler.run_once([], {}, timestamp=Timestamp.from_ms(1000))

        assert group.consecutive_failures == 1
        assert group._backoff_until is not None

    def test_evaluate_returns_only_scale_up(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """evaluate() only returns SCALE_UP decisions; failed cleanup happens in run_once()."""
        discovered = [
            make_mock_slice_handle("ready-slice", all_ready=True),
            make_mock_slice_handle("failed-slice", any_failed=True),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform, scale_up_cooldown=Duration.from_ms(0))
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(5, device_type=DeviceType.TPU, device_variant="v5p-8")
        decisions = autoscaler.evaluate(demand)

        for d in decisions:
            assert d.action == ScalingAction.SCALE_UP

    def test_failed_slice_cleanup_skips_idle_verification(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Failed slice cleanup does not require idle verification."""
        mock_handle = make_mock_slice_handle("failed-slice", any_failed=True)
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        group.refresh_all_slices()

        autoscaler = make_autoscaler(
            scale_groups={"test-group": group},
        )

        autoscaler.run_once([], {}, timestamp=Timestamp.from_ms(1000))

        assert group.slice_count() == 0


class TestWaterfallRouting:
    """Tests for priority-based waterfall demand routing."""

    def test_routes_demand_to_highest_priority_group_first(self):
        """Demand routes to highest priority (lowest number) matching group."""
        config_high = make_scale_group_config(name="high-priority", max_slices=5, priority=10)
        config_low = make_scale_group_config(name="low-priority", max_slices=5, priority=20)

        group_high = ScalingGroup(config_high, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))
        group_low = ScalingGroup(config_low, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler({"high-priority": group_high, "low-priority": group_low})

        demand = make_demand_entries(3, device_type=DeviceType.TPU, device_variant="v5p-8")
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 1
        assert decisions[0].scale_group == "high-priority"

    def test_cpu_demand_routes_by_priority(self):
        """CPU demand matches all groups and routes by priority."""
        config_high = make_scale_group_config(name="high-priority", max_slices=5, priority=10)
        config_low = make_scale_group_config(
            name="low-priority",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_GPU,
            accelerator_variant="A100",
            max_slices=5,
            priority=20,
        )

        group_high = ScalingGroup(config_high, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))
        group_low = ScalingGroup(config_low, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler({"high-priority": group_high, "low-priority": group_low})

        demand = make_demand_entries(2, device_type=DeviceType.CPU, device_variant=None)
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 1
        assert decisions[0].scale_group == "high-priority"
        assert group_high.current_demand == 2
        assert group_low.current_demand == 0

    def test_demand_overflows_to_lower_priority_when_at_capacity(self):
        """When high-priority group is at capacity, demand overflows to lower priority."""
        config_high = make_scale_group_config(name="high-priority", max_slices=2, priority=10)
        config_low = make_scale_group_config(name="low-priority", max_slices=5, priority=20)

        discovered = [make_mock_slice_handle(f"slice-{i}", all_ready=True) for i in range(2)]
        group_high = ScalingGroup(config_high, make_mock_platform(slices_to_discover=discovered))
        group_high.reconcile()

        group_low = ScalingGroup(config_low, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler({"high-priority": group_high, "low-priority": group_low})

        demand = make_demand_entries(3, device_type=DeviceType.TPU, device_variant="v5p-8")
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 1
        assert decisions[0].scale_group == "low-priority"

    def test_routing_filters_by_accelerator_type(self):
        """Only groups matching accelerator_type receive demand."""
        config_v5p = make_scale_group_config(name="v5p-group", accelerator_variant="v5p-8", max_slices=5, priority=10)
        config_v5lite = make_scale_group_config(
            name="v5lite-group", accelerator_variant="v5litepod-4", max_slices=5, priority=10
        )

        group_v5p = ScalingGroup(config_v5p, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))
        group_v5lite = ScalingGroup(config_v5lite, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler({"v5p-group": group_v5p, "v5lite-group": group_v5lite})

        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5litepod-4")
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 1
        assert decisions[0].scale_group == "v5lite-group"

    def test_demand_with_no_matching_group_is_unmet(self):
        """Demand for unknown accelerator type results in unmet demand."""
        config = make_scale_group_config(name="test-group", max_slices=5, priority=10)

        group = ScalingGroup(config, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="unknown-type")
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 0

    def test_multiple_demand_entries_route_independently(self):
        """Multiple demand entries with different accelerator types route to appropriate groups."""
        config_v5p = make_scale_group_config(name="v5p-group", accelerator_variant="v5p-8", max_slices=5, priority=10)
        config_v5lite = make_scale_group_config(
            name="v5lite-group", accelerator_variant="v5litepod-4", max_slices=5, priority=10
        )

        group_v5p = ScalingGroup(config_v5p, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))
        group_v5lite = ScalingGroup(config_v5lite, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler({"v5p-group": group_v5p, "v5lite-group": group_v5lite})

        demand = [
            *make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8", task_prefix="v5p"),
            *make_demand_entries(3, device_type=DeviceType.TPU, device_variant="v5litepod-4", task_prefix="v5lite"),
        ]
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 2
        groups_in_decisions = {d.scale_group for d in decisions}
        assert "v5p-group" in groups_in_decisions
        assert "v5lite-group" in groups_in_decisions


class TestPreemptibleRouting:
    """Tests for preemptible demand routing."""

    def test_route_demand_filters_by_preemptible_true(self):
        """Demand with preemptible=True only routes to preemptible groups."""
        config_preemptible = make_scale_group_config(
            name="preemptible-group", max_slices=5, priority=10, preemptible=True
        )
        config_on_demand = make_scale_group_config(name="on-demand-group", max_slices=5, priority=10)

        group_preemptible = ScalingGroup(config_preemptible, make_mock_platform())
        group_on_demand = ScalingGroup(config_on_demand, make_mock_platform())

        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8", preemptible=True)
        result = route_demand([group_preemptible, group_on_demand], demand)

        assert len(result.routed_entries["preemptible-group"]) == 2
        assert result.routed_entries.get("on-demand-group") is None

    def test_route_demand_filters_by_preemptible_false(self):
        """Demand with preemptible=False only routes to non-preemptible groups."""
        config_preemptible = make_scale_group_config(
            name="preemptible-group", max_slices=5, priority=10, preemptible=True
        )
        config_on_demand = make_scale_group_config(name="on-demand-group", max_slices=5, priority=10)

        group_preemptible = ScalingGroup(config_preemptible, make_mock_platform())
        group_on_demand = ScalingGroup(config_on_demand, make_mock_platform())

        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8", preemptible=False)
        result = route_demand([group_preemptible, group_on_demand], demand)

        assert result.routed_entries.get("preemptible-group") is None
        assert len(result.routed_entries["on-demand-group"]) == 2

    def test_route_demand_no_preference_routes_to_any(self):
        """Demand with preemptible=None routes to any matching group."""
        config_preemptible = make_scale_group_config(
            name="preemptible-group", max_slices=5, priority=10, preemptible=True
        )
        config_on_demand = make_scale_group_config(name="on-demand-group", max_slices=5, priority=20)

        group_preemptible = ScalingGroup(config_preemptible, make_mock_platform())
        group_on_demand = ScalingGroup(config_on_demand, make_mock_platform())

        demand = make_demand_entries(3, device_type=DeviceType.TPU, device_variant="v5p-8", preemptible=None)
        result = route_demand([group_preemptible, group_on_demand], demand)

        assert len(result.routed_entries["preemptible-group"]) == 3
        assert result.unmet_entries == []


class TestAutoscalerWaterfallEndToEnd:
    """End-to-end tests for waterfall routing with FakePlatform."""

    def test_demand_cascades_through_priority_groups_on_quota(self):
        """Full cascade: quota on primary routes to secondary."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        config_primary = make_scale_group_config(name="primary", max_slices=5, priority=10, zones=["us-central1-a"])
        config_fallback = make_scale_group_config(name="fallback", max_slices=5, priority=20, zones=["us-central1-a"])

        platform_primary = FakePlatform(
            FakePlatformConfig(config=config_primary, failure_mode=FailureMode.QUOTA_EXCEEDED)
        )
        platform_fallback = FakePlatform(FakePlatformConfig(config=config_fallback))

        group_primary = ScalingGroup(config_primary, platform_primary, scale_up_cooldown=Duration.from_ms(0))
        group_fallback = ScalingGroup(config_fallback, platform_fallback, scale_up_cooldown=Duration.from_ms(0))

        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
            config=config,
        )

        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8")

        autoscaler.run_once(demand, {})
        time.sleep(0.1)

        assert group_primary.availability().status == GroupAvailability.QUOTA_EXCEEDED
        assert group_fallback.slice_count() == 0

        autoscaler.run_once(demand, {})
        time.sleep(0.1)
        assert group_fallback.slice_count() == 1

        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        assert group_fallback.slice_count() == 2

    def test_quota_recovery_restores_primary_routing(self):
        """After quota timeout expires, demand routes to primary again."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        config_primary = make_scale_group_config(name="primary", max_slices=5, priority=10, zones=["us-central1-a"])
        config_fallback = make_scale_group_config(name="fallback", max_slices=5, priority=20, zones=["us-central1-a"])

        platform_primary = FakePlatform(
            FakePlatformConfig(config=config_primary, failure_mode=FailureMode.QUOTA_EXCEEDED)
        )
        platform_fallback = FakePlatform(FakePlatformConfig(config=config_fallback))

        group_primary = ScalingGroup(
            config_primary, platform_primary, scale_up_cooldown=Duration.from_ms(0), quota_timeout=Duration.from_ms(1000)
        )
        group_fallback = ScalingGroup(config_fallback, platform_fallback, scale_up_cooldown=Duration.from_ms(0))

        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
            config=config,
        )

        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")

        autoscaler.run_once(demand, {})
        time.sleep(0.1)
        ts_after_fail = Timestamp.now()
        assert group_primary.availability(ts_after_fail).status == GroupAvailability.QUOTA_EXCEEDED
        assert group_fallback.slice_count() == 0

        autoscaler.run_once(demand, {})
        time.sleep(0.1)
        assert group_fallback.slice_count() == 1

        time.sleep(1.1)

        platform_primary.set_failure_mode(FailureMode.NONE)

        ts_now = Timestamp.now()
        assert group_primary.availability(ts_now).status == GroupAvailability.AVAILABLE

        demand_increased = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8")
        decisions = autoscaler.evaluate(demand_increased, timestamp=ts_now)
        assert len(decisions) == 1
        assert decisions[0].scale_group == "primary"

        autoscaler.shutdown()

    def test_full_group_cascades_to_fallback(self):
        """When primary group hits max_slices, demand cascades to fallback."""

        config_primary = make_scale_group_config(name="primary", max_slices=1, priority=10, zones=["us-central1-a"])
        config_fallback = make_scale_group_config(name="fallback", max_slices=5, priority=20, zones=["us-central1-a"])

        platform_primary = FakePlatform(FakePlatformConfig(config=config_primary))
        platform_fallback = FakePlatform(FakePlatformConfig(config=config_fallback))

        group_primary = ScalingGroup(config_primary, platform_primary, scale_up_cooldown=Duration.from_ms(0))
        group_fallback = ScalingGroup(config_fallback, platform_fallback, scale_up_cooldown=Duration.from_ms(0))

        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
            config=config,
        )

        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")
        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        assert group_primary.slice_count() == 1

        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8")
        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        assert group_primary.slice_count() == 1
        assert group_fallback.slice_count() == 1

    def test_multiple_accelerator_types_route_independently(self):
        """Different accelerator types route through their own group chains."""

        config_v5p = make_scale_group_config(
            name="v5p-group", accelerator_variant="v5p-8", max_slices=5, priority=10, zones=["us-central1-a"]
        )
        config_v5lite = make_scale_group_config(
            name="v5lite-group", accelerator_variant="v5litepod-4", max_slices=5, priority=10, zones=["us-central1-a"]
        )

        platform_v5p = FakePlatform(FakePlatformConfig(config=config_v5p))
        platform_v5lite = FakePlatform(FakePlatformConfig(config=config_v5lite))

        group_v5p = ScalingGroup(config_v5p, platform_v5p, scale_up_cooldown=Duration.from_ms(0))
        group_v5lite = ScalingGroup(config_v5lite, platform_v5lite, scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler(
            scale_groups={"v5p-group": group_v5p, "v5lite-group": group_v5lite},
        )

        demand = [
            *make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8", task_prefix="v5p"),
            *make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5litepod-4", task_prefix="v5lite"),
        ]

        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()

        assert group_v5p.slice_count() == 1
        assert group_v5lite.slice_count() == 1

    def test_capacity_overflow_cascades_to_lower_priority(self):
        """When high-priority group fills up, overflow goes to lower priority."""

        config_primary = make_scale_group_config(name="primary", max_slices=2, priority=10, zones=["us-central1-a"])
        config_fallback = make_scale_group_config(name="fallback", max_slices=5, priority=20, zones=["us-central1-a"])

        platform_primary = FakePlatform(FakePlatformConfig(config=config_primary))
        platform_fallback = FakePlatform(FakePlatformConfig(config=config_fallback))

        group_primary = ScalingGroup(config_primary, platform_primary, scale_up_cooldown=Duration.from_ms(0))
        group_fallback = ScalingGroup(config_fallback, platform_fallback, scale_up_cooldown=Duration.from_ms(0))

        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
            config=config,
        )

        demand = make_demand_entries(4, device_type=DeviceType.TPU, device_variant="v5p-8")

        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        group_primary.refresh_all_slices()
        group_fallback.refresh_all_slices()
        assert group_primary.slice_count() == 1
        assert group_fallback.slice_count() == 1

        ts = Timestamp.now().epoch_ms()
        platform_primary.tick(ts)
        platform_fallback.tick(ts)
        group_primary.refresh_all_slices()
        group_fallback.refresh_all_slices()

        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        group_primary.refresh_all_slices()
        group_fallback.refresh_all_slices()
        assert group_primary.slice_count() == 1
        assert group_fallback.slice_count() == 2

        platform_primary.tick(Timestamp.now().epoch_ms())
        platform_fallback.tick(Timestamp.now().epoch_ms())

        autoscaler.run_once(demand, {})
        time.sleep(0.1)
        assert group_fallback.slice_count() == 3

        for _ in range(2):
            platform_primary.tick(Timestamp.now().epoch_ms())
            platform_fallback.tick(Timestamp.now().epoch_ms())
            autoscaler.run_once(demand, {})
            time.sleep(0.1)

        autoscaler._wait_for_inflight()

        total = group_primary.slice_count() + group_fallback.slice_count()
        assert total >= 4


class TestAutoscalerQuotaHandling:
    """Tests for quota exceeded error handling."""

    def test_quota_exceeded_sets_group_unavailable(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """QuotaExhaustedError sets group to QUOTA_EXCEEDED state."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        platform = make_mock_platform()
        platform.create_slice.side_effect = QuotaExhaustedError("Quota exceeded")
        group = ScalingGroup(
            scale_group_config, platform, scale_up_cooldown=Duration.from_ms(0), quota_timeout=Duration.from_ms(60_000)
        )
        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler({"test-group": group}, config=config)

        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")
        autoscaler.run_once(demand, {}, timestamp=Timestamp.from_ms(1000))
        autoscaler._wait_for_inflight()

        state = group.availability(timestamp=Timestamp.from_ms(2000))
        assert state.status == GroupAvailability.QUOTA_EXCEEDED

    def test_quota_exceeded_routes_to_fallback_group(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """When primary group has quota exceeded, demand routes to fallback."""
        config_primary = make_scale_group_config(name="primary", max_slices=5, priority=10)
        config_fallback = make_scale_group_config(name="fallback", max_slices=5, priority=20)

        platform_primary = make_mock_platform()
        platform_primary.create_slice.side_effect = QuotaExhaustedError("Quota exceeded")
        platform_fallback = make_mock_platform()

        group_primary = ScalingGroup(
            config_primary,
            platform_primary,
            scale_up_cooldown=Duration.from_ms(0),
            quota_timeout=Duration.from_ms(60_000),
        )
        group_fallback = ScalingGroup(config_fallback, platform_fallback, scale_up_cooldown=Duration.from_ms(0))
        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler({"primary": group_primary, "fallback": group_fallback}, config=config)

        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")
        autoscaler.run_once(demand, {}, timestamp=Timestamp.from_ms(1000))
        autoscaler._wait_for_inflight()

        decisions = autoscaler.evaluate(demand, timestamp=Timestamp.from_ms(2000))

        assert len(decisions) == 1
        assert decisions[0].scale_group == "fallback"

    def test_quota_state_expires_after_timeout(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """QUOTA_EXCEEDED state expires after timeout."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        platform = make_mock_platform()
        platform.create_slice.side_effect = QuotaExhaustedError("Quota exceeded")
        group = ScalingGroup(
            scale_group_config, platform, scale_up_cooldown=Duration.from_ms(0), quota_timeout=Duration.from_ms(1000)
        )

        ts = Timestamp.from_ms(1000)
        group.begin_scale_up()
        with pytest.raises(QuotaExhaustedError):
            group.scale_up(timestamp=ts)
        group.cancel_scale_up()
        group.record_quota_exceeded("quota exceeded", ts)

        assert group.availability(Timestamp.from_ms(1100)).status == GroupAvailability.QUOTA_EXCEEDED

        assert group.availability(Timestamp.from_ms(2100)).status == GroupAvailability.AVAILABLE

    def test_generic_error_triggers_backoff_not_quota(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Non-quota errors trigger backoff, not quota exceeded state."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        platform = make_mock_platform()
        platform.create_slice.side_effect = RuntimeError("TPU unavailable")

        backoff = Duration.from_seconds(5.0)
        group = ScalingGroup(
            scale_group_config, platform, scale_up_cooldown=Duration.from_ms(0), backoff_initial=backoff
        )
        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler({"test-group": group}, config=config)

        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")
        autoscaler.run_once(demand, {}, timestamp=Timestamp.from_ms(1000))
        autoscaler._wait_for_inflight()

        state = group.availability(timestamp=Timestamp.from_ms(2000))
        assert state.status == GroupAvailability.BACKOFF
        assert group.consecutive_failures == 1


class TestAutoscalerActionLogging:
    """Tests for autoscaler action logging."""

    def test_action_log_records_scale_up(self, empty_autoscaler: Autoscaler):
        """Verify scale-up actions are logged."""
        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8")
        empty_autoscaler.run_once(demand, {})
        empty_autoscaler._wait_for_inflight()

        status = empty_autoscaler.get_status()
        assert len(status.recent_actions) == 1
        action = status.recent_actions[0]
        assert action.action_type == "scale_up"
        assert action.scale_group == "test-group"
        assert action.slice_id != ""
        assert "demand" in action.reason

    def test_action_log_records_failed_cleanup(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Verify failed slice cleanup actions are logged."""
        mock_handle = make_mock_slice_handle("slice-001", any_failed=True)
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform, scale_down_cooldown=Duration.from_ms(0))
        group.reconcile()
        group.refresh_all_slices()
        autoscaler = make_autoscaler({"test-group": group})

        autoscaler.run_once([], {}, timestamp=Timestamp.from_ms(1000))

        status = autoscaler.get_status()
        assert len(status.recent_actions) == 1
        action = status.recent_actions[0]
        assert action.action_type == "failed_cleanup"
        assert action.scale_group == "test-group"
        assert action.slice_id == "slice-001"
        assert "cleaning up failed slice" in action.reason

    def test_action_log_records_quota_exceeded(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Verify quota exceeded events are logged."""
        platform = make_mock_platform()
        platform.create_slice.side_effect = QuotaExhaustedError("Quota exceeded in zone")
        group = ScalingGroup(scale_group_config, platform, scale_up_cooldown=Duration.from_ms(0))
        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")
        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()

        status = autoscaler.get_status()
        assert len(status.recent_actions) == 1
        action = status.recent_actions[0]
        assert action.action_type == "quota_exceeded"
        assert action.scale_group == "test-group"
        assert "Quota exceeded" in action.reason

    def test_action_log_records_worker_failed(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Verify worker failure events are logged."""
        mock_handle = make_mock_slice_handle("slice-001")
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        group.refresh_all_slices()
        autoscaler = make_autoscaler({"test-group": group})
        autoscaler.reconcile()

        vm_address = f"10.0.{abs(hash('slice-001')) % 256}.0"
        group.refresh_all_slices()
        autoscaler.notify_worker_failed(vm_address)

        status = autoscaler.get_status()
        assert len(status.recent_actions) == 1
        action = status.recent_actions[0]
        assert action.action_type == "worker_failed"
        assert action.scale_group == "test-group"
        assert action.slice_id == "slice-001"
        assert vm_address in action.reason

    def test_action_log_bounded_to_100_entries(self, empty_autoscaler: Autoscaler):
        """Verify action log is bounded to 100 entries."""
        for i in range(150):
            empty_autoscaler._log_action("test_action", "test-group", reason=f"action {i}")

        status = empty_autoscaler.get_status()
        assert len(status.recent_actions) == 100
        assert status.recent_actions[0].reason == "action 50"
        assert status.recent_actions[99].reason == "action 149"

    def test_get_status_includes_actions(self, empty_autoscaler: Autoscaler):
        """Verify get_status returns recent actions."""
        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")
        empty_autoscaler.run_once(demand, {})
        empty_autoscaler._wait_for_inflight()

        status = empty_autoscaler.get_status()

        assert len(status.groups) == 1
        assert status.current_demand["test-group"] == 1
        assert len(status.recent_actions) >= 1
        assert status.recent_actions[0].action_type == "scale_up"

    def test_action_log_includes_timestamp(self, empty_autoscaler: Autoscaler):
        """Verify actions include valid timestamps."""
        before = Timestamp.now().epoch_ms()
        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")
        empty_autoscaler.run_once(demand, {})
        empty_autoscaler._wait_for_inflight()
        after = Timestamp.now().epoch_ms()

        status = empty_autoscaler.get_status()
        action = status.recent_actions[0]
        assert before <= action.timestamp.epoch_ms <= after


class TestScalingGroupRequestingState:
    """Tests for REQUESTING state via slice-level placeholders in ScalingGroup."""

    def test_begin_scale_up_sets_requesting_state(self):
        """begin_scale_up() causes availability() to return REQUESTING."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        config = make_scale_group_config(name="test-group", min_slices=0, max_slices=5)
        platform = make_mock_platform()
        group = ScalingGroup(config, platform)

        ts = Timestamp.now()
        group.begin_scale_up()

        availability = group.availability(ts)
        assert availability.status == GroupAvailability.REQUESTING

    def test_complete_scale_up_clears_requesting_state(self):
        """complete_scale_up() removes REQUESTING state."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        config = make_scale_group_config(name="test-group", min_slices=0, max_slices=5)
        platform = make_mock_platform()
        group = ScalingGroup(config, platform)

        ts = Timestamp.now()
        group.begin_scale_up()
        assert group.availability(ts).status == GroupAvailability.REQUESTING

        handle = make_mock_slice_handle("new-slice-1", all_ready=True)
        group.complete_scale_up(handle, ts)

        assert group.availability(ts).status == GroupAvailability.AVAILABLE
        assert group.slice_count() == 1
        assert group.get_slice("new-slice-1") is not None

    def test_cancel_scale_up_clears_requesting_state(self):
        """cancel_scale_up() removes REQUESTING state."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        config = make_scale_group_config(name="test-group", min_slices=0, max_slices=5)
        platform = make_mock_platform()
        group = ScalingGroup(config, platform)

        ts = Timestamp.now()
        group.begin_scale_up()
        assert group.availability(ts).status == GroupAvailability.REQUESTING

        group.cancel_scale_up()

        assert group.availability(ts).status == GroupAvailability.AVAILABLE
        assert group.slice_count() == 0

    def test_pending_scale_up_counts_toward_slice_count(self):
        """Pending scale-up counts toward slice_count and max_slices check."""
        config = make_scale_group_config(name="test-group", min_slices=0, max_slices=1)
        platform = make_mock_platform()
        group = ScalingGroup(config, platform)

        group.begin_scale_up()
        assert group.slice_count() == 1
        assert not group.can_scale_up()

    def test_demand_routing_prefers_requesting_groups(self):
        """route_demand() prefers pending/requesting groups."""
        config1 = make_scale_group_config(name="group-1", min_slices=0, max_slices=5, priority=10)
        config2 = make_scale_group_config(name="group-2", min_slices=0, max_slices=5, priority=20)

        platform1 = make_mock_platform()
        platform2 = make_mock_platform()
        group1 = ScalingGroup(config1, platform1)
        group2 = ScalingGroup(config2, platform2)

        ts = Timestamp.now()
        group1.begin_scale_up()

        demand_entries = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8")

        result = route_demand([group1, group2], demand_entries, ts)

        assert len(result.routed_entries["group-1"]) == 2
        assert result.routed_entries.get("group-2") is None
        assert result.unmet_entries == []


class TestAutoscalerAsyncScaleUp:
    """Tests for async scale-up behavior."""

    def test_execute_scale_up_returns_immediately(self):
        """_execute_scale_up returns immediately without blocking."""
        config = make_scale_group_config(name="test-group", min_slices=0, max_slices=5)

        platform = make_mock_platform()
        original_create = platform.create_slice.side_effect

        def slow_create(config):
            time.sleep(0.5)
            return original_create(config)

        platform.create_slice.side_effect = slow_create

        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0))
        autoscaler = make_autoscaler({"test-group": group})

        decision = ScalingDecision(
            scale_group="test-group",
            action=ScalingAction.SCALE_UP,
            reason="test async",
        )

        start = time.time()
        autoscaler.execute([decision], timestamp=Timestamp.now())
        elapsed = time.time() - start

        assert elapsed < 0.1

        autoscaler._wait_for_inflight()

    def test_group_marked_requesting_during_scale_up(self):
        """Group shows REQUESTING immediately after execute(), cleared when done."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        config = make_scale_group_config(name="test-group", min_slices=0, max_slices=5)
        platform = make_mock_platform()
        original_create = platform.create_slice.side_effect

        def slow_create(config):
            time.sleep(0.2)
            return original_create(config)

        platform.create_slice.side_effect = slow_create

        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0))
        autoscaler = make_autoscaler({"test-group": group})

        ts = Timestamp.now().epoch_ms()
        decision = ScalingDecision(
            scale_group="test-group",
            action=ScalingAction.SCALE_UP,
            reason="test requesting",
        )

        autoscaler.execute([decision], timestamp=Timestamp.from_ms(ts))

        # Pending counter is incremented, so availability shows REQUESTING
        availability = group.availability(Timestamp.from_ms(ts))
        assert availability.status == GroupAvailability.REQUESTING

        autoscaler._wait_for_inflight()

        # After completion, pending counter is decremented and slice is added
        availability = group.availability(Timestamp.from_ms(ts + 300))
        assert availability.status == GroupAvailability.AVAILABLE
        assert group.slice_count() == 1

    def test_autoscaler_shutdown_waits_for_scale_up(self):
        """shutdown() waits for in-flight scale-ups to complete."""
        config = make_scale_group_config(name="test-group", min_slices=0, max_slices=5)

        platform = make_mock_platform()
        original_create = platform.create_slice.side_effect
        create_completed = []

        def slow_create(config):
            time.sleep(0.2)
            result = original_create(config)
            create_completed.append(True)
            return result

        platform.create_slice.side_effect = slow_create

        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0))
        autoscaler = make_autoscaler({"test-group": group})

        decision = ScalingDecision(
            scale_group="test-group",
            action=ScalingAction.SCALE_UP,
            reason="test shutdown",
        )

        autoscaler.execute([decision], timestamp=Timestamp.now())

        autoscaler.shutdown()

        assert len(create_completed) == 1

    def test_autoscaler_shutdown_terminates_all_slices(self):
        """shutdown() terminates all slices."""
        config = make_scale_group_config(name="test-group", min_slices=0, max_slices=5)

        discovered_handle = make_mock_slice_handle("slice-001", all_ready=True)
        platform = make_mock_platform(slices_to_discover=[discovered_handle])

        group = ScalingGroup(config, platform)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        autoscaler.shutdown()

        # All slices should be terminated
        assert group.slice_count() == 0
        discovered_handle.terminate.assert_called_once()


# --- Bug reproduction tests ---


def test_pending_counter_prevents_double_scaleup():
    """Verify that the pending scale-up counter prevents double scale-up when
    create_slice takes longer than expected.

    The pending counter is included in slice_count(), so the evaluator sees
    total=1 and does not trigger another scale-up.
    """
    create_barrier = threading.Event()

    class SlowFakePlatform(FakePlatform):
        """FakePlatform where create_slice blocks until barrier is released."""

        def create_slice(self, config):
            create_barrier.wait(timeout=10)
            return super().create_slice(config)

    sg_config = make_scale_group_config(
        name="test-group",
        min_slices=1,
        max_slices=4,
        zones=["us-central1-a"],
    )
    platform = SlowFakePlatform(FakePlatformConfig(config=sg_config))
    group = ScalingGroup(
        sg_config,
        platform,
        scale_up_cooldown=Duration.from_ms(0),
    )
    autoscaler = Autoscaler(
        scale_groups={"test-group": group},
        evaluation_interval=Duration.from_ms(100),
        platform=platform,
    )

    demand = make_demand_entries(1)
    t0 = Timestamp.from_ms(1_000_000)

    # First run_once: demand=1, current=0, below min_slices -> scale up.
    # This spawns a thread that blocks on create_barrier.
    decisions1 = autoscaler.run_once(demand, {}, t0)
    assert len(decisions1) == 1
    assert decisions1[0].action == ScalingAction.SCALE_UP

    # Advance time arbitrarily far — the pending counter prevents double scale-up
    # regardless of elapsed time.
    t1 = Timestamp.from_ms(t0.epoch_ms() + 600)

    # Second run_once: pending counter makes slice_count()=1, so evaluator sees
    # total=1 which satisfies min_slices=1. No new scale-up decision.
    decisions2 = autoscaler.run_once(demand, {}, t1)
    assert len(decisions2) == 0, "Pending counter should prevent second scale-up"

    # Release the barrier so threads complete
    create_barrier.set()
    autoscaler._wait_for_inflight()

    # Only 1 slice was created
    assert group.slice_count() == 1

    autoscaler.shutdown()


def test_bootstrap_called_after_scaleup():
    """After scale_up creates a slice, bootstrap() is called on every VM
    when a WorkerBootstrap is provided.
    """
    from iris.cluster.platform.bootstrap import WorkerBootstrap

    sg_config = make_scale_group_config(
        name="test-group",
        min_slices=0,
        max_slices=4,
        zones=["us-central1-a"],
    )
    platform = FakePlatform(FakePlatformConfig(config=sg_config))
    group = ScalingGroup(
        sg_config,
        platform,
        scale_up_cooldown=Duration.from_ms(0),
    )

    cluster_config = config_pb2.IrisClusterConfig()
    cluster_config.defaults.bootstrap.docker_image = "test:latest"
    cluster_config.defaults.bootstrap.worker_port = 10001
    worker_bootstrap = WorkerBootstrap(cluster_config)

    autoscaler = Autoscaler(
        scale_groups={"test-group": group},
        evaluation_interval=Duration.from_ms(100),
        platform=platform,
        worker_bootstrap=worker_bootstrap,
    )

    demand = make_demand_entries(1)
    t0 = Timestamp.from_ms(1_000_000)

    decisions = autoscaler.run_once(demand, {}, t0)
    assert len(decisions) == 1

    autoscaler._wait_for_inflight()
    platform.tick()

    assert group.slice_count() == 1

    slice_handle = group.slice_handles()[0]
    for vm in slice_handle.describe().vms:
        assert vm._bootstrap_count == 1, "bootstrap() should be called once per VM after scale-up"

    autoscaler.shutdown()


def test_bootstrap_skipped_without_config():
    """Without worker_bootstrap, bootstrap() is not called (test/local mode)."""
    sg_config = make_scale_group_config(
        name="test-group",
        min_slices=0,
        max_slices=4,
        zones=["us-central1-a"],
    )
    platform = FakePlatform(FakePlatformConfig(config=sg_config))
    group = ScalingGroup(
        sg_config,
        platform,
        scale_up_cooldown=Duration.from_ms(0),
    )
    autoscaler = Autoscaler(
        scale_groups={"test-group": group},
        evaluation_interval=Duration.from_ms(100),
        platform=platform,
    )

    demand = make_demand_entries(1)
    t0 = Timestamp.from_ms(1_000_000)

    autoscaler.run_once(demand, {}, t0)
    autoscaler._wait_for_inflight()

    assert group.slice_count() == 1

    slice_handle = group.slice_handles()[0]
    for vm in slice_handle.describe().vms:
        assert vm._bootstrap_count == 0, "No bootstrap without config"

    autoscaler.shutdown()
