# Copyright The Marin Authors
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
    AdditiveReq,
    Autoscaler,
    DEFAULT_UNRESOLVABLE_TIMEOUT,
    DemandEntry,
    ScalingAction,
    ScalingDecision,
    compute_required_slices,
    first_fit_decreasing,
    route_demand,
)
from iris.cluster.controller.scaling_group import ScalingGroup
from iris.cluster.platform.base import (
    CloudSliceState,
    CloudWorkerState,
    Labels,
    QuotaExhaustedError,
    SliceStatus,
    WorkerStatus as CloudWorkerStatus,
)
from iris.cluster.constraints import (
    Constraint,
    ConstraintOp,
    DeviceType,
    PlacementRequirements,
    WellKnownAttribute,
    device_variant_constraint,
    preemptible_constraint,
    region_constraint,
    zone_constraint,
)
from iris.cluster.types import WorkerStatus
from iris.rpc import cluster_pb2, config_pb2, vm_pb2
from iris.time_utils import Duration, Timestamp
from tests.cluster.platform.fakes import FailureMode, FakePlatform, FakePlatformConfig

# --- Test fixtures and helpers ---


def make_demand_entries(
    count: int,
    *,
    device_type: DeviceType = DeviceType.TPU,
    device_variant: str | None = "v5p-8",
    device_variants: frozenset[str] | None = None,
    preemptible: bool | None = None,
    required_regions: frozenset[str] | None = None,
    required_zones: frozenset[str] | None = None,
    task_prefix: str = "task",
) -> list[DemandEntry]:
    if count <= 0:
        return []
    resources = cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024)
    if device_type == DeviceType.TPU:
        resources.device.tpu.variant = device_variant or ""
    elif device_type == DeviceType.GPU:
        resources.device.gpu.variant = device_variant or ""
    elif device_type == DeviceType.CPU:
        resources.device.cpu.variant = ""
    effective_variants = device_variants
    if effective_variants is None and device_variant is not None:
        effective_variants = frozenset({device_variant})
    normalized = PlacementRequirements(
        device_type=device_type,
        device_variants=effective_variants,
        preemptible=preemptible,
        required_regions=required_regions,
        required_zones=required_zones,
    )

    # Build proto constraints matching the PlacementRequirements
    constraint_list: list[Constraint] = []
    if device_type is not None:
        constraint_list.append(
            Constraint(key=WellKnownAttribute.DEVICE_TYPE, op=ConstraintOp.EQ, value=device_type.value)
        )
    if effective_variants:
        constraint_list.append(device_variant_constraint(sorted(effective_variants)))
    if preemptible is not None:
        constraint_list.append(preemptible_constraint(preemptible))
    if required_regions:
        constraint_list.append(region_constraint(sorted(required_regions)))
    if required_zones:
        for z in sorted(required_zones):
            constraint_list.append(zone_constraint(z))
    proto_constraints = [c.to_proto() for c in constraint_list]

    return [
        DemandEntry(
            task_ids=[f"{task_prefix}-{i}"],
            coschedule_group_id=None,
            normalized=normalized,
            constraints=proto_constraints,
            resources=resources,
        )
        for i in range(count)
    ]


DEFAULT_RESOURCES = config_pb2.ScaleGroupResources(
    cpu_millicores=128000,
    memory_bytes=128 * 1024**3,
    disk_bytes=100 * 1024**3,
    device_type=config_pb2.ACCELERATOR_TYPE_TPU,
    device_variant="v5p-8",
    device_count=8,
)


def ensure_scale_group_resources(config: config_pb2.ScaleGroupConfig) -> config_pb2.ScaleGroupConfig:
    if not config.HasField("resources"):
        config.resources.CopyFrom(DEFAULT_RESOURCES)
    if not config.HasField("num_vms"):
        config.num_vms = 1
    return config


def make_scale_group_config(**kwargs: object) -> config_pb2.ScaleGroupConfig:
    # Extract accelerator fields that now live on resources, not ScaleGroupConfig
    accelerator_type = kwargs.pop("accelerator_type", config_pb2.ACCELERATOR_TYPE_TPU)
    accelerator_variant = kwargs.pop("accelerator_variant", "v5p-8")
    # Extract fields that moved to slice_template
    runtime_version = kwargs.pop("runtime_version", None)
    zones = kwargs.pop("zones", None)
    preemptible = kwargs.pop("preemptible", None)
    config = ensure_scale_group_resources(config_pb2.ScaleGroupConfig(**kwargs))
    config.resources.device_type = accelerator_type
    if accelerator_variant:
        config.resources.device_variant = accelerator_variant
    if preemptible is not None:
        config.slice_template.preemptible = preemptible
        config.resources.preemptible = preemptible
    if runtime_version or zones:
        gcp = config.slice_template.gcp
        if runtime_version:
            gcp.runtime_version = runtime_version
        if zones:
            gcp.zone = zones[0]
    return config


def _cloud_worker_state_from_iris(state: vm_pb2.VmState) -> CloudWorkerState:
    """Reverse map from Iris VM state to CloudWorkerState for test setup."""
    if state == vm_pb2.VM_STATE_READY:
        return CloudWorkerState.RUNNING
    if state == vm_pb2.VM_STATE_FAILED:
        return CloudWorkerState.STOPPED
    if state == vm_pb2.VM_STATE_TERMINATED:
        return CloudWorkerState.TERMINATED
    return CloudWorkerState.UNKNOWN


def make_mock_worker_handle(vm_id: str, address: str, state: vm_pb2.VmState, bootstrap_log: str = "") -> MagicMock:
    """Create a mock RemoteWorkerHandle for testing."""
    handle = MagicMock()
    handle.vm_id = vm_id
    handle.worker_id = vm_id
    handle.internal_address = address
    handle.external_address = None
    handle.bootstrap_log = bootstrap_log
    handle.status.return_value = CloudWorkerStatus(state=_cloud_worker_state_from_iris(state))
    return handle


def make_mock_slice_handle(
    slice_id: str,
    scale_group: str = "test-group",
    all_ready: bool = False,
    any_failed: bool = False,
    vm_states: list[vm_pb2.VmState] | None = None,
    bootstrap_logs: list[str] | None = None,
    created_at_ms: int = 1000000,
) -> MagicMock:
    """Create a mock SliceHandle for testing."""
    handle = MagicMock()
    handle.slice_id = slice_id
    handle.scale_group = scale_group
    handle.zone = "us-central1-a"
    iris_labels = Labels("iris")
    handle.labels = {iris_labels.iris_scale_group: scale_group, iris_labels.iris_managed: "true"}
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
    worker_handles = []
    for i, state in enumerate(vm_states):
        bootstrap_log = bootstrap_logs[i] if bootstrap_logs and i < len(bootstrap_logs) else ""
        worker_handle = make_mock_worker_handle(
            vm_id=f"{slice_id}-vm-{i}",
            address=f"10.0.{slice_hash}.{i}",
            state=state,
            bootstrap_log=bootstrap_log,
        )
        worker_handles.append(worker_handle)

    handle.describe.return_value = SliceStatus(state=slice_state, worker_count=len(vm_states), workers=worker_handles)

    return handle


def make_mock_platform(
    slices_to_discover: list[MagicMock] | None = None,
) -> MagicMock:
    """Create a mock Platform for testing.

    Infrastructure methods (create_slice, list_slices, etc.) are mocked
    because they would hit cloud APIs.

    Args:
        slices_to_discover: Pre-existing slices returned by list_slices.
    """
    platform = MagicMock()
    platform.list_slices.return_value = slices_to_discover or []

    create_count = [0]

    def create_slice_side_effect(config: config_pb2.SliceConfig, worker_config=None) -> MagicMock:
        create_count[0] += 1
        slice_id = f"new-slice-{create_count[0]}"
        return make_mock_slice_handle(slice_id)

    platform.create_slice.side_effect = create_slice_side_effect
    return platform


def _mark_discovered_ready(group: ScalingGroup, handles: list[MagicMock], timestamp: Timestamp | None = None) -> None:
    """Mark discovered slices as READY with their worker IDs."""
    for handle in handles:
        worker_ids = [w.worker_id for w in handle.describe().workers]
        group.mark_slice_ready(handle.slice_id, worker_ids, timestamp=timestamp)


def _mark_discovered_failed(group: ScalingGroup, handles: list[MagicMock]) -> None:
    """Mark discovered slices as FAILED."""
    for handle in handles:
        group.mark_slice_failed(handle.slice_id)


def _mark_all_slices_ready(group: ScalingGroup) -> None:
    """Mark all tracked slices as READY with their worker IDs.

    Used after FakePlatform.tick() to simulate the bootstrap thread
    marking slices ready once VMs are running.
    """
    for handle in group.slice_handles():
        desc = handle.describe()
        if desc.state == CloudSliceState.READY:
            worker_ids = [w.worker_id for w in desc.workers]
            group.mark_slice_ready(handle.slice_id, worker_ids)


@pytest.fixture
def scale_group_config() -> config_pb2.ScaleGroupConfig:
    """A standard scale group configuration."""
    return make_scale_group_config(
        name="test-group",
        min_slices=0,
        max_slices=5,
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
        scale_group_config,
        platform,
        idle_threshold=Duration.from_ms(0),
    )
    group.reconcile()
    autoscaler = make_autoscaler({"test-group": group})
    yield autoscaler, group, platform
    autoscaler.shutdown()


def make_autoscaler(
    scale_groups: dict[str, ScalingGroup],
    config: config_pb2.AutoscalerConfig | None = None,
    platform: MagicMock | None = None,
    base_worker_config: config_pb2.WorkerConfig | None = None,
) -> Autoscaler:
    """Create an Autoscaler with the given groups."""
    mock_platform = platform or make_mock_platform()

    if config:
        return Autoscaler.from_config(
            scale_groups=scale_groups,
            config=config,
            platform=mock_platform,
            base_worker_config=base_worker_config,
        )
    else:
        return Autoscaler(
            scale_groups=scale_groups,
            evaluation_interval=Duration.from_seconds(0.1),
            platform=mock_platform,
            base_worker_config=base_worker_config,
        )


# --- Tests for scaling decisions ---


class TestAutoscalerScaleUp:
    """Tests for scale-up decisions."""

    def test_scales_up_when_demand_exceeds_capacity(self, empty_autoscaler: Autoscaler):
        """Evaluates scale-up when demand > capacity."""
        demand = make_demand_entries(2, device_type=DeviceType.CPU, device_variant=None)
        decisions = empty_autoscaler.evaluate(demand)

        assert len(decisions) == 1
        assert decisions[0].action == ScalingAction.SCALE_UP
        assert decisions[0].scale_group == "test-group"
        assert "required_slices=1 > pending=0" in decisions[0].reason

    @pytest.mark.parametrize(
        "discovered,demand_count,reason",
        [
            ([make_mock_slice_handle(f"slice-{i}") for i in range(5)], 10, "at_max_slices"),
            (
                [
                    make_mock_slice_handle("slice-001", vm_states=[vm_pb2.VM_STATE_BOOTING]),
                    make_mock_slice_handle("slice-002", vm_states=[vm_pb2.VM_STATE_BOOTING]),
                ],
                2,
                "pending_slices_count",
            ),
        ],
        ids=["at_max_slices", "pending_slices_count"],
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

    def test_scale_up_when_ready_workers_full(self):
        """Scales up when all ready workers are full and demand survives dry-run.

        With 5 ready slices and 0 pending, a single unplaceable demand entry (which
        survived the scheduler dry-run against all ready workers) must trigger scale-up.
        The old comparison (required > ready+pending) would see 1 > 5 and deadlock.
        The correct comparison (required > pending) sees 1 > 0 and scales up.
        """
        config = make_scale_group_config(name="test-group", max_slices=10)
        discovered = [make_mock_slice_handle(f"slice-{i}", all_ready=True) for i in range(5)]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0))
        group.reconcile()
        _mark_discovered_ready(group, discovered)
        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 1
        assert decisions[0].action == ScalingAction.SCALE_UP
        assert "required_slices=1 > pending=0" in decisions[0].reason


class TestAutoscalerScaleDown:
    """Tests for scale-down behavior (delegated to ScalingGroup)."""

    def test_scales_down_idle_slice_via_run_once(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """run_once() scales down idle slices via ScalingGroup.scale_down_if_idle()."""
        ready_ts = Timestamp.from_ms(1_000)
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True, created_at_ms=100000),
            make_mock_slice_handle("slice-002", all_ready=True, created_at_ms=200000),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(
            scale_group_config,
            platform,
            idle_threshold=Duration.from_ms(1000),
        )
        group.reconcile()
        _mark_discovered_ready(group, discovered, timestamp=ready_ts)
        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")

        # Get VM addresses from the adapter
        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        slice_001_wid = slice_001.describe().workers[0].worker_id
        slice_002_wid = slice_002.describe().workers[0].worker_id
        vm_status_map = {
            slice_001_wid: WorkerStatus(worker_id=slice_001_wid, running_task_ids=frozenset()),
            slice_002_wid: WorkerStatus(worker_id=slice_002_wid, running_task_ids=frozenset()),
        }

        # Timestamp must be past idle_threshold (1000ms) from when slices became ready
        autoscaler.run_once(demand, vm_status_map, timestamp=Timestamp.from_ms(10_000))

        assert group.slice_count() == 1

    def test_no_scale_down_at_min_slices(self):
        """Does not scale down when at min_slices."""
        config = make_scale_group_config(
            name="test-group",
            min_slices=2,
            max_slices=5,
            runtime_version="v2-alpha-tpuv5",
            zones=["us-central2-b"],
        )
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True),
            make_mock_slice_handle("slice-002", all_ready=True),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(
            config,
            platform,
            idle_threshold=Duration.from_ms(0),
        )
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(0, device_type=DeviceType.TPU, device_variant="v5p-8")
        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        slice_001_wid = slice_001.describe().workers[0].worker_id
        slice_002_wid = slice_002.describe().workers[0].worker_id
        vm_status_map = {
            slice_001_wid: WorkerStatus(worker_id=slice_001_wid, running_task_ids=frozenset()),
            slice_002_wid: WorkerStatus(worker_id=slice_002_wid, running_task_ids=frozenset()),
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
            idle_threshold=Duration.from_ms(300_000),
        )
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        # refresh() observes platform-maintained slice state and marks discovered slices READY
        autoscaler.refresh({})

        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        slice_001_wid = slice_001.describe().workers[0].worker_id
        slice_002_wid = slice_002.describe().workers[0].worker_id

        vm_status_map_active = {
            slice_001_wid: WorkerStatus(worker_id=slice_001_wid, running_task_ids=frozenset({"task-1"})),
            slice_002_wid: WorkerStatus(worker_id=slice_002_wid, running_task_ids=frozenset({"task-2"})),
        }
        group.update_slice_activity(vm_status_map_active, timestamp=Timestamp.from_ms(1000))

        # Use empty demand (simulating dry-run absorbed all tasks onto ready workers)
        # to test that scale-down is blocked by idle_threshold, not scale-up.
        vm_status_map_idle = {
            slice_001_wid: WorkerStatus(worker_id=slice_001_wid, running_task_ids=frozenset()),
            slice_002_wid: WorkerStatus(worker_id=slice_002_wid, running_task_ids=frozenset()),
        }

        autoscaler.run_once([], vm_status_map_idle, timestamp=Timestamp.from_ms(100_000))

        assert group.slice_count() == 2

    def test_scale_down_rate_limited_by_token_bucket(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Scale-down is rate-limited by the token bucket (only 1 per minute with rate_limit=1)."""
        ready_ts = Timestamp.from_ms(1_000)
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True, created_at_ms=100000),
            make_mock_slice_handle("slice-002", all_ready=True, created_at_ms=200000),
            make_mock_slice_handle("slice-003", all_ready=True, created_at_ms=300000),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(
            scale_group_config,
            platform,
            idle_threshold=Duration.from_ms(0),
            scale_down_rate_limit=1,
        )
        group.reconcile()
        _mark_discovered_ready(group, discovered, timestamp=ready_ts)
        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(0, device_type=DeviceType.TPU, device_variant="v5p-8")
        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        slice_003 = group.get_slice("slice-003")
        slice_001_wid = slice_001.describe().workers[0].worker_id
        slice_002_wid = slice_002.describe().workers[0].worker_id
        slice_003_wid = slice_003.describe().workers[0].worker_id
        vm_status_map = {
            slice_001_wid: WorkerStatus(worker_id=slice_001_wid, running_task_ids=frozenset()),
            slice_002_wid: WorkerStatus(worker_id=slice_002_wid, running_task_ids=frozenset()),
            slice_003_wid: WorkerStatus(worker_id=slice_003_wid, running_task_ids=frozenset()),
        }

        # With rate_limit=1, only 1 slice should be scaled down per cycle
        autoscaler.run_once(demand, vm_status_map, timestamp=Timestamp.from_ms(10_000))
        assert group.slice_count() == 2

    def test_scale_down_multiple_idle_slices_in_one_cycle(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """With enough rate-limit tokens, multiple idle slices are scaled down in one cycle."""
        ready_ts = Timestamp.from_ms(1_000)
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True, created_at_ms=100000),
            make_mock_slice_handle("slice-002", all_ready=True, created_at_ms=200000),
            make_mock_slice_handle("slice-003", all_ready=True, created_at_ms=300000),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(
            scale_group_config,
            platform,
            idle_threshold=Duration.from_ms(1000),
            scale_down_rate_limit=5,
        )
        group.reconcile()
        _mark_discovered_ready(group, discovered, timestamp=ready_ts)
        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(0, device_type=DeviceType.TPU, device_variant="v5p-8")
        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        slice_003 = group.get_slice("slice-003")
        slice_001_wid = slice_001.describe().workers[0].worker_id
        slice_002_wid = slice_002.describe().workers[0].worker_id
        slice_003_wid = slice_003.describe().workers[0].worker_id
        vm_status_map = {
            slice_001_wid: WorkerStatus(worker_id=slice_001_wid, running_task_ids=frozenset()),
            slice_002_wid: WorkerStatus(worker_id=slice_002_wid, running_task_ids=frozenset()),
            slice_003_wid: WorkerStatus(worker_id=slice_003_wid, running_task_ids=frozenset()),
        }

        # With rate_limit=5, all 3 idle slices should be scaled down in one cycle
        autoscaler.run_once(demand, vm_status_map, timestamp=Timestamp.from_ms(10_000))
        assert group.slice_count() == 0


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

    def test_execute_records_failure_on_scale_up_error(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """execute() records failure when scale-up fails."""
        platform = make_mock_platform()
        platform.create_slice.side_effect = RuntimeError("TPU unavailable")
        backoff = Duration.from_seconds(5.0)
        group = ScalingGroup(
            scale_group_config,
            platform,
            scale_up_cooldown=Duration.from_ms(0),
            backoff_initial=backoff,
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
        demand = make_demand_entries(2, device_type=DeviceType.CPU, device_variant=None)
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


class TestAutoscalerWorkerFailure:
    """Tests for worker failure handling."""

    def test_notify_worker_failed_terminates_slice(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """notify_worker_failed() terminates the slice containing the worker."""
        mock_handle = make_mock_slice_handle("slice-001", all_ready=True)
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})
        _mark_discovered_ready(group, [mock_handle])

        failed_worker_id = "slice-001-vm-0"
        autoscaler.notify_worker_failed(failed_worker_id)

        assert group.slice_count() == 0

    def test_notify_worker_failed_unknown_worker_is_noop(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """notify_worker_failed() does nothing for unknown workers."""
        mock_handle = make_mock_slice_handle("slice-001", all_ready=True)
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        autoscaler.notify_worker_failed("unknown-worker-99")

        assert group.slice_count() == 1

    def test_notify_worker_failed_returns_sibling_worker_ids(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """notify_worker_failed() returns sibling worker IDs for multi-VM slices."""
        # Create a slice with 4 VMs
        mock_handle = make_mock_slice_handle(
            "slice-001",
            all_ready=True,
            vm_states=[vm_pb2.VM_STATE_READY] * 4,
        )
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})
        _mark_discovered_ready(group, [mock_handle])

        # Fail the first worker — should return 3 sibling worker IDs
        failed_worker_id = "slice-001-vm-0"
        siblings = autoscaler.notify_worker_failed(failed_worker_id)

        expected_siblings = [f"slice-001-vm-{i}" for i in range(1, 4)]
        assert sorted(siblings) == sorted(expected_siblings)
        assert group.slice_count() == 0

    def test_notify_worker_failed_returns_empty_for_single_vm_slice(
        self, scale_group_config: config_pb2.ScaleGroupConfig
    ):
        """Single-VM slices return no siblings."""
        mock_handle = make_mock_slice_handle("slice-001", all_ready=True)
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})
        _mark_discovered_ready(group, [mock_handle])

        failed_worker_id = "slice-001-vm-0"
        siblings = autoscaler.notify_worker_failed(failed_worker_id)

        assert siblings == []

    def test_notify_worker_failed_cleans_up_even_if_terminate_fails(
        self, scale_group_config: config_pb2.ScaleGroupConfig
    ):
        """notify_worker_failed() removes the slice even if terminate() raises.

        Prevents ghost slices where a preempted/deleted cloud resource causes
        terminate() to fail, leaving the slice tracked forever.
        """
        mock_handle = make_mock_slice_handle("slice-001", all_ready=True)
        mock_handle.terminate.side_effect = RuntimeError("resource not found")
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})
        _mark_discovered_ready(group, [mock_handle])

        failed_worker_id = "slice-001-vm-0"
        siblings = autoscaler.notify_worker_failed(failed_worker_id)

        # Slice should be removed despite terminate() failure
        assert group.slice_count() == 0
        assert siblings == []


class TestAutoscalerIdleVerification:
    """Tests for idle verification during scale-down."""

    def test_verifies_idle_with_worker_idle_map(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Scale-down via run_once verifies workers are idle using worker_idle_map."""
        mock_handle = make_mock_slice_handle("slice-001", all_ready=True)
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(
            scale_group_config,
            platform,
            idle_threshold=Duration.from_ms(0),
        )
        group.reconcile()

        autoscaler = make_autoscaler(
            scale_groups={"test-group": group},
        )

        demand = make_demand_entries(0, device_type=DeviceType.TPU, device_variant="v5p-8")

        slice_001 = group.get_slice("slice-001")
        slice_001_wid = slice_001.describe().workers[0].worker_id
        vm_status_map = {
            slice_001_wid: WorkerStatus(
                worker_id=slice_001_wid,
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


class TestAutoscalerBootstrapLogs:
    """Tests for bootstrap log reporting."""

    def test_get_init_log_returns_bootstrap_output(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Worker bootstrap logs are captured in autoscaler worker tracking."""
        bootstrap_log = "line1\nline2\nline3"
        mock_handle = make_mock_slice_handle("slice-001", all_ready=True, bootstrap_logs=[bootstrap_log])
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform)
        autoscaler = make_autoscaler({"test-group": group})

        workers = mock_handle.describe().workers
        autoscaler._register_slice_workers(workers, mock_handle.slice_id, "test-group")

        vm_id = mock_handle.describe().workers[0].worker_id
        assert autoscaler.get_init_log(vm_id) == bootstrap_log
        assert autoscaler.get_init_log(vm_id, tail=2) == "line2\nline3"

    def test_get_vm_by_worker_id(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """get_vm() uses platform worker_id as the only lookup key."""
        mock_handle = make_mock_slice_handle("slice-001", all_ready=True)
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform)
        autoscaler = make_autoscaler({"test-group": group})

        workers = mock_handle.describe().workers
        autoscaler._register_slice_workers(workers, mock_handle.slice_id, "test-group")

        worker = workers[0]
        # Lookup by platform worker_id
        info = autoscaler.get_vm(worker.worker_id)
        assert info is not None
        assert info.scale_group == "test-group"

        # Unknown keys return None — no address fallback
        assert autoscaler.get_vm(worker.internal_address) is None
        assert autoscaler.get_vm("192.168.0.99") is None


class TestWaterfallRouting:
    """Tests for priority-based waterfall demand routing."""

    def test_routes_demand_to_highest_priority_group_first(self):
        """Demand routes to highest priority (lowest number) matching group."""
        config_high = make_scale_group_config(name="high-priority", max_slices=5, priority=10)
        config_low = make_scale_group_config(name="low-priority", max_slices=5, priority=20)

        group_high = ScalingGroup(config_high, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))
        group_low = ScalingGroup(config_low, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler({"high-priority": group_high, "low-priority": group_low})

        demand = make_demand_entries(3, device_type=DeviceType.CPU, device_variant=None)
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 1
        assert decisions[0].scale_group == "high-priority"

    def test_cpu_demand_routes_by_priority(self):
        """CPU demand matches all groups and routes by priority."""
        config_high = make_scale_group_config(name="high-priority", max_slices=5, priority=10)
        config_low = make_scale_group_config(
            name="low-priority",
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
        # current_demand is required_slices (2 tiny entries pack into 1 slice)
        assert group_high.current_demand == 1
        assert group_low.current_demand == 0

    def test_demand_overflows_to_lower_priority_when_at_max_slices(self):
        """When high-priority group is at max_slices, demand falls through to lower priority."""
        config_high = make_scale_group_config(name="high-priority", max_slices=2, priority=10)
        config_low = make_scale_group_config(name="low-priority", max_slices=5, priority=20)

        discovered = [make_mock_slice_handle(f"slice-{i}", all_ready=True) for i in range(2)]
        group_high = ScalingGroup(config_high, make_mock_platform(slices_to_discover=discovered))
        group_high.reconcile()

        group_low = ScalingGroup(config_low, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler({"high-priority": group_high, "low-priority": group_low})

        demand = make_demand_entries(3, device_type=DeviceType.CPU, device_variant=None)
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

        # 2 TPU entries are VM-exclusive → 2 slices, each a separate SCALE_UP decision
        assert len(decisions) == 2
        assert all(d.scale_group == "v5lite-group" for d in decisions)

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
            *make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8", task_prefix="v5p"),
            *make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5litepod-4", task_prefix="v5lite"),
        ]
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 2
        groups_in_decisions = {d.scale_group for d in decisions}
        assert "v5p-group" in groups_in_decisions
        assert "v5lite-group" in groups_in_decisions

    def test_flexible_variant_routes_to_matching_group(self):
        """Demand with multiple device_variants routes to any matching group."""
        config_v4 = make_scale_group_config(name="v4-group", accelerator_variant="v4-8", max_slices=5, priority=10)
        config_v5p = make_scale_group_config(name="v5p-group", accelerator_variant="v5p-8", max_slices=5, priority=20)

        group_v4 = ScalingGroup(config_v4, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))
        group_v5p = ScalingGroup(config_v5p, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler({"v4-group": group_v4, "v5p-group": group_v5p})

        # Demand accepts either v4-8 or v5p-8; should route to v4-group (higher priority)
        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variants=frozenset({"v4-8", "v5p-8"}))
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) >= 1
        group_names = {d.scale_group for d in decisions}
        assert "v4-group" in group_names

    def test_flexible_variant_no_match_for_missing_variant(self):
        """Flexible demand with no matching groups is unmet."""
        config_v4 = make_scale_group_config(name="v4-group", accelerator_variant="v4-8", max_slices=5, priority=10)

        group_v4 = ScalingGroup(config_v4, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler({"v4-group": group_v4})

        # Demand requires v5p-8 or v5litepod-4, but only v4-8 group exists
        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variants=frozenset({"v5p-8", "v5litepod-4"}))
        decisions = autoscaler.evaluate(demand)
        assert len(decisions) == 0

    def test_backoff_group_falls_through_to_fallback(self):
        """When primary group is in BACKOFF, demand falls through to fallback."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        config_primary = make_scale_group_config(name="primary", max_slices=5, priority=10)
        config_fallback = make_scale_group_config(name="fallback", max_slices=5, priority=20)

        group_primary = ScalingGroup(
            config_primary,
            make_mock_platform(),
            scale_up_cooldown=Duration.from_ms(0),
            backoff_initial=Duration.from_seconds(60),
        )
        group_fallback = ScalingGroup(
            config_fallback,
            make_mock_platform(),
            scale_up_cooldown=Duration.from_ms(0),
        )

        ts = Timestamp.from_ms(1000)
        group_primary.record_failure(timestamp=ts)
        assert group_primary.availability(ts).status == GroupAvailability.BACKOFF

        autoscaler = make_autoscaler({"primary": group_primary, "fallback": group_fallback})
        demand = make_demand_entries(2, device_type=DeviceType.CPU, device_variant=None)
        decisions = autoscaler.evaluate(demand, timestamp=ts)

        assert len(decisions) == 1
        assert decisions[0].scale_group == "fallback"
        status_by_group = {s.group: s for s in autoscaler._last_routing_decision.group_statuses}
        assert status_by_group["primary"].decision == "blocked"
        assert "consecutive failure" in status_by_group["primary"].reason
        assert status_by_group["fallback"].decision == "selected"

    def test_backoff_group_with_ready_slices_still_falls_through(self):
        """Even with ready slices, a BACKOFF group rejects demand so it falls through."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        discovered = [make_mock_slice_handle("slice-0", all_ready=True)]
        config_primary = make_scale_group_config(name="primary", max_slices=5, priority=10)
        config_fallback = make_scale_group_config(name="fallback", max_slices=5, priority=20)

        group_primary = ScalingGroup(
            config_primary,
            make_mock_platform(slices_to_discover=discovered),
            scale_up_cooldown=Duration.from_ms(0),
            backoff_initial=Duration.from_seconds(60),
        )
        group_primary.reconcile()
        group_fallback = ScalingGroup(
            config_fallback,
            make_mock_platform(),
            scale_up_cooldown=Duration.from_ms(0),
        )

        ts = Timestamp.from_ms(1000)
        group_primary.record_failure(timestamp=ts)
        assert group_primary.availability(ts).status == GroupAvailability.BACKOFF
        assert group_primary.slice_count() == 1

        autoscaler = make_autoscaler({"primary": group_primary, "fallback": group_fallback})
        demand = make_demand_entries(2, device_type=DeviceType.CPU, device_variant=None)
        decisions = autoscaler.evaluate(demand, timestamp=ts)

        assert len(decisions) == 1
        assert decisions[0].scale_group == "fallback"
        assert group_primary.current_demand == 0

    def test_cooldown_does_not_cause_fallthrough(self):
        """Groups in COOLDOWN still accept demand — demand does not fall through."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        config_a = make_scale_group_config(name="group-a", max_slices=5, priority=10)
        config_b = make_scale_group_config(name="group-b", max_slices=5, priority=20)

        group_a = ScalingGroup(
            config_a,
            make_mock_platform(),
            scale_up_cooldown=Duration.from_ms(60_000),
        )
        group_b = ScalingGroup(
            config_b,
            make_mock_platform(),
            scale_up_cooldown=Duration.from_ms(0),
        )

        ts = Timestamp.from_ms(1_000_000)
        group_a.begin_scale_up(timestamp=ts)
        handle = group_a.scale_up(timestamp=ts)
        group_a.complete_scale_up(handle, ts)

        eval_ts = Timestamp.from_ms(1_030_000)
        assert group_a.availability(eval_ts).status == GroupAvailability.COOLDOWN

        autoscaler = make_autoscaler({"group-a": group_a, "group-b": group_b})
        demand = make_demand_entries(2, device_type=DeviceType.CPU, device_variant=None)
        autoscaler.evaluate(demand, timestamp=eval_ts)

        # current_demand is required_slices; 2 tiny entries pack into 1 slice
        assert group_a.current_demand == 1
        assert group_b.current_demand == 0

    def test_at_max_slices_causes_fallthrough(self):
        """Groups at AT_MAX_SLICES reject demand, causing fallthrough to lower-priority groups."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        config_a = make_scale_group_config(name="group-a", max_slices=1, priority=10)
        config_b = make_scale_group_config(name="group-b", max_slices=5, priority=20)

        discovered = [make_mock_slice_handle("slice-0", all_ready=True)]
        group_a = ScalingGroup(config_a, make_mock_platform(slices_to_discover=discovered))
        group_a.reconcile()
        assert group_a.availability().status == GroupAvailability.AT_MAX_SLICES

        group_b = ScalingGroup(
            config_b,
            make_mock_platform(),
            scale_up_cooldown=Duration.from_ms(0),
        )

        autoscaler = make_autoscaler({"group-a": group_a, "group-b": group_b})
        demand = make_demand_entries(2, device_type=DeviceType.CPU, device_variant=None)
        decisions = autoscaler.evaluate(demand)

        assert group_a.current_demand == 0
        # 2 tiny CPU entries pack into 1 slice
        assert group_b.current_demand == 1
        assert len(decisions) == 1
        assert decisions[0].scale_group == "group-b"


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


class TestRegionRouting:
    def test_route_demand_filters_by_required_region(self):
        config_west = make_scale_group_config(name="west", max_slices=5, priority=10, zones=["us-west4-b"])
        config_west.worker.attributes[WellKnownAttribute.REGION] = "us-west4"

        config_eu = make_scale_group_config(name="eu", max_slices=5, priority=10, zones=["europe-west4-b"])
        config_eu.worker.attributes[WellKnownAttribute.REGION] = "europe-west4"

        west = ScalingGroup(config_west, make_mock_platform())
        eu = ScalingGroup(config_eu, make_mock_platform())

        demand = make_demand_entries(
            2,
            device_type=DeviceType.TPU,
            device_variant="v5p-8",
            required_regions=frozenset({"us-west4"}),
        )

        result = route_demand([west, eu], demand)

        assert len(result.routed_entries["west"]) == 2
        assert result.routed_entries.get("eu") is None
        assert result.unmet_entries == []

    def test_route_demand_unmet_when_no_group_matches_region(self):
        config_eu = make_scale_group_config(name="eu", max_slices=5, priority=10, zones=["europe-west4-b"])
        config_eu.worker.attributes[WellKnownAttribute.REGION] = "europe-west4"
        eu = ScalingGroup(config_eu, make_mock_platform())

        demand = make_demand_entries(
            1,
            device_type=DeviceType.TPU,
            device_variant="v5p-8",
            required_regions=frozenset({"us-west4"}),
        )

        result = route_demand([eu], demand)

        assert result.routed_entries.get("eu") is None
        assert len(result.unmet_entries) == 1
        assert "no groups in region" in result.unmet_entries[0].reason
        assert "us-west4" in result.unmet_entries[0].reason

    def test_route_demand_combined_region_and_preemptible(self):
        """Demand requiring both region=us-west4 and preemptible=True only routes to the matching group."""
        config_west_preemptible = make_scale_group_config(
            name="west-preemptible", max_slices=5, priority=10, zones=["us-west4-b"], preemptible=True
        )
        config_west_preemptible.worker.attributes[WellKnownAttribute.REGION] = "us-west4"

        config_west_ondemand = make_scale_group_config(
            name="west-ondemand", max_slices=5, priority=10, zones=["us-west4-b"], preemptible=False
        )
        config_west_ondemand.worker.attributes[WellKnownAttribute.REGION] = "us-west4"

        config_eu_preemptible = make_scale_group_config(
            name="eu-preemptible", max_slices=5, priority=10, zones=["europe-west4-b"], preemptible=True
        )
        config_eu_preemptible.worker.attributes[WellKnownAttribute.REGION] = "europe-west4"

        west_preemptible = ScalingGroup(config_west_preemptible, make_mock_platform())
        west_ondemand = ScalingGroup(config_west_ondemand, make_mock_platform())
        eu_preemptible = ScalingGroup(config_eu_preemptible, make_mock_platform())

        demand = make_demand_entries(
            2,
            device_type=DeviceType.TPU,
            device_variant="v5p-8",
            preemptible=True,
            required_regions=frozenset({"us-west4"}),
        )

        result = route_demand([west_preemptible, west_ondemand, eu_preemptible], demand)

        assert len(result.routed_entries["west-preemptible"]) == 2
        assert result.routed_entries.get("west-ondemand") is None
        assert result.routed_entries.get("eu-preemptible") is None
        assert result.unmet_entries == []


class TestZoneRouting:
    def test_route_demand_filters_by_required_zone(self):
        config_a = make_scale_group_config(name="zone-a", max_slices=5, priority=10, zones=["us-central2-a"])
        config_a.worker.attributes[WellKnownAttribute.REGION] = "us-central2"
        config_a.worker.attributes[WellKnownAttribute.ZONE] = "us-central2-a"

        config_b = make_scale_group_config(name="zone-b", max_slices=5, priority=10, zones=["us-central2-b"])
        config_b.worker.attributes[WellKnownAttribute.REGION] = "us-central2"
        config_b.worker.attributes[WellKnownAttribute.ZONE] = "us-central2-b"

        zone_a = ScalingGroup(config_a, make_mock_platform())
        zone_b = ScalingGroup(config_b, make_mock_platform())

        demand = make_demand_entries(
            2,
            device_type=DeviceType.TPU,
            device_variant="v5p-8",
            required_zones=frozenset({"us-central2-b"}),
        )

        result = route_demand([zone_a, zone_b], demand)

        assert len(result.routed_entries["zone-b"]) == 2
        assert result.routed_entries.get("zone-a") is None
        assert result.unmet_entries == []

    def test_route_demand_unmet_when_no_group_matches_zone(self):
        config_a = make_scale_group_config(name="zone-a", max_slices=5, priority=10, zones=["us-central2-a"])
        config_a.worker.attributes[WellKnownAttribute.REGION] = "us-central2"
        config_a.worker.attributes[WellKnownAttribute.ZONE] = "us-central2-a"
        zone_a = ScalingGroup(config_a, make_mock_platform())

        demand = make_demand_entries(
            1,
            device_type=DeviceType.TPU,
            device_variant="v5p-8",
            required_zones=frozenset({"us-central2-b"}),
        )

        result = route_demand([zone_a], demand)

        assert result.routed_entries.get("zone-a") is None
        assert len(result.unmet_entries) == 1
        assert "no groups in zone" in result.unmet_entries[0].reason
        assert "us-central2-b" in result.unmet_entries[0].reason

    def test_zone_typo_suggests_close_match(self):
        """A zone typo like 'europe-west4b' triggers a 'did you mean' suggestion."""
        config = make_scale_group_config(name="eu", max_slices=5, priority=10, zones=["europe-west4-b"])
        config.worker.attributes[WellKnownAttribute.REGION] = "europe-west4"
        config.worker.attributes[WellKnownAttribute.ZONE] = "europe-west4-b"
        eu = ScalingGroup(config, make_mock_platform())

        demand = make_demand_entries(
            1,
            device_type=DeviceType.TPU,
            device_variant="v5p-8",
            required_zones=frozenset({"europe-west4b"}),
        )

        result = route_demand([eu], demand)

        assert len(result.unmet_entries) == 1
        reason = result.unmet_entries[0].reason
        assert "did you mean" in reason
        assert "europe-west4-b" in reason

    def test_device_mismatch_shows_available(self):
        """When device doesn't match, the reason mentions the requested device."""
        config = make_scale_group_config(
            name="gpu-group",
            max_slices=5,
            priority=10,
            zones=["us-central1-a"],
            accelerator_type=config_pb2.ACCELERATOR_TYPE_GPU,
            accelerator_variant="H100",
        )
        gpu_group = ScalingGroup(config, make_mock_platform())

        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")

        result = route_demand([gpu_group], demand)

        assert len(result.unmet_entries) == 1
        reason = result.unmet_entries[0].reason
        assert "no groups with device" in reason
        assert "tpu" in reason

    def test_reason_string_is_concise(self):
        """The no_matching_group reason stays under 200 chars even with many groups."""
        groups = []
        for i in range(60):
            zone = f"us-east{i % 5 + 1}-{'abc'[i % 3]}"
            config = make_scale_group_config(
                name=f"tpu_v6e_4-{zone}",
                max_slices=2,
                priority=10,
                zones=[zone],
            )
            config.worker.attributes[WellKnownAttribute.ZONE] = zone
            groups.append(ScalingGroup(config, make_mock_platform()))

        demand = make_demand_entries(
            1,
            device_type=DeviceType.TPU,
            device_variant="v5p-8",
            required_zones=frozenset({"nonexistent-zone-z"}),
        )

        result = route_demand(groups, demand)

        assert len(result.unmet_entries) == 1
        reason = result.unmet_entries[0].reason
        assert len(reason) < 200, f"Reason too long ({len(reason)} chars): {reason}"


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

        # 2 TPU entries are VM-exclusive → 2 slices on fallback
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

        demand = make_demand_entries(1, device_type=DeviceType.CPU, device_variant=None)

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

        demand_increased = make_demand_entries(2, device_type=DeviceType.CPU, device_variant=None)
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

        demand = make_demand_entries(1, device_type=DeviceType.CPU, device_variant=None)
        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        assert group_primary.slice_count() == 1

        demand = make_demand_entries(2, device_type=DeviceType.CPU, device_variant=None)
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

        assert group_v5p.slice_count() == 2
        assert group_v5lite.slice_count() == 1

    def test_capacity_overflow_cascades_to_lower_priority(self):
        """When high-priority group fills up, overflow goes to lower priority.

        Uses large entries (128GiB memory each) so each entry requires its own VM.
        Primary has max_slices=1, so only 1 entry routes there. The rest cascade
        to fallback.
        """

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

        big_resources = cluster_pb2.ResourceSpecProto(cpu_millicores=128000, memory_bytes=128 * 1024**3)
        normalized = PlacementRequirements(
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
            preemptible=None,
            required_regions=None,
            required_zones=None,
        )
        demand = [
            DemandEntry(
                task_ids=[f"task-{i}"],
                coschedule_group_id=None,
                normalized=normalized,
                constraints=[],
                resources=big_resources,
            )
            for i in range(3)
        ]

        # First cycle: primary gets 1 entry (remaining_vms=1), fallback gets 2.
        # Multi-slice scale-up creates all needed slices in one cycle.
        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        assert group_primary.slice_count() == 1
        assert group_fallback.slice_count() == 2

        total = group_primary.slice_count() + group_fallback.slice_count()
        assert total == 3

    def test_demand_cascades_through_priority_groups_on_backoff(self):
        """E2E: primary create fails → BACKOFF, second run cascades to fallback."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        config_primary = make_scale_group_config(name="primary", max_slices=5, priority=10, zones=["us-central1-a"])
        config_fallback = make_scale_group_config(name="fallback", max_slices=5, priority=20, zones=["us-central1-a"])

        platform_primary = FakePlatform(FakePlatformConfig(config=config_primary, failure_mode=FailureMode.CREATE_FAILS))
        platform_fallback = FakePlatform(FakePlatformConfig(config=config_fallback))

        group_primary = ScalingGroup(
            config_primary,
            platform_primary,
            scale_up_cooldown=Duration.from_ms(0),
            backoff_initial=Duration.from_seconds(60),
        )
        group_fallback = ScalingGroup(
            config_fallback,
            platform_fallback,
            scale_up_cooldown=Duration.from_ms(0),
        )

        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
            config=config,
        )

        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8")

        # First run: primary attempts scale-up, fails → enters BACKOFF
        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        assert group_primary.availability().status == GroupAvailability.BACKOFF
        assert group_primary.slice_count() == 0

        # Second run: primary in BACKOFF → demand cascades to fallback
        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        assert group_fallback.slice_count() >= 1


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
        group.begin_scale_up(timestamp=ts)
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
            scale_group_config,
            platform,
            scale_up_cooldown=Duration.from_ms(0),
            backoff_initial=backoff,
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
        assert len(status.recent_actions) >= 1
        action = status.recent_actions[0]
        assert action.action_type == "scale_up"
        assert action.scale_group == "test-group"
        assert action.slice_id != ""
        assert "required_slices" in action.reason

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
        mock_handle = make_mock_slice_handle("slice-001", all_ready=True)
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})
        _mark_discovered_ready(group, [mock_handle])

        failed_worker_id = "slice-001-vm-0"
        autoscaler.notify_worker_failed(failed_worker_id)

        status = autoscaler.get_status()
        actions_by_type = {a.action_type: a for a in status.recent_actions}
        assert "worker_failed" in actions_by_type
        action = actions_by_type["worker_failed"]
        assert action.scale_group == "test-group"
        assert action.slice_id == "slice-001"
        assert failed_worker_id in action.reason

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
        assert status.last_evaluation.epoch_ms > 0
        assert status.groups[0].availability_status != ""

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
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0))

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
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0))

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

        demand_entries = make_demand_entries(2, device_type=DeviceType.CPU, device_variant=None)

        result = route_demand([group1, group2], demand_entries, ts)

        assert len(result.routed_entries["group-1"]) == 2
        assert result.routed_entries.get("group-2") is None
        assert result.unmet_entries == []
        status_by_group = {s.group: s for s in result.group_statuses}
        assert status_by_group["group-1"].decision == "selected"
        # 2 tiny entries pack into 1 slice; 1 inflight slice covers it
        assert status_by_group["group-1"].launch == 0
        assert status_by_group["group-2"].decision == "idle"


class TestCommittedBudgetRouting:
    """Tests for two-phase routing with committed budgets."""

    def test_committed_budget_retains_demand_for_requesting_group(self):
        """Demand sticks to a group with requesting slices even when a fresh group is available.

        v6e has 3 requesting slices, v5e has 0 slices. Both priority 10.
        All 5 entries should go to v6e (committed budget absorbs them).
        """
        config_v6e = make_scale_group_config(name="v6e", max_slices=30, priority=10, num_vms=4)
        config_v5e = make_scale_group_config(name="v5e", max_slices=30, priority=10, num_vms=4)

        platform_v6e = make_mock_platform()
        platform_v5e = make_mock_platform()
        group_v6e = ScalingGroup(config_v6e, platform_v6e, scale_up_cooldown=Duration.from_ms(0))
        group_v5e = ScalingGroup(config_v5e, platform_v5e, scale_up_cooldown=Duration.from_ms(0))

        # v6e has 3 requesting slices
        for _ in range(3):
            group_v6e.begin_scale_up()

        ts = Timestamp.now()
        demand = make_demand_entries(5, device_type=DeviceType.TPU, device_variant="v5p-8")
        result = route_demand([group_v6e, group_v5e], demand, ts)

        assert len(result.routed_entries.get("v6e", [])) == 5
        assert result.routed_entries.get("v5e") is None
        assert result.unmet_entries == []
        # 5 tiny entries pack into 1 slice; 3 requesting slices cover it
        assert result.group_to_launch.get("v6e", 0) == 0

    def test_committed_budget_overflow_falls_to_waterfall(self):
        """When committed budget is insufficient, overflow goes through the waterfall.

        v6e has 1 requesting slice (num_vms=1), max_slices=2 → full budget = 2 VMs.
        Each entry fills one VM (resources match VM capacity).
        3 demand entries: 1 goes via committed, 1 more via waterfall to v6e, 1 overflows to v5e.
        """
        # 1 entry per VM: entry cpu matches VM capacity
        small_resources = config_pb2.ScaleGroupResources(
            cpu_millicores=1000,
            memory_bytes=1024,
            disk_bytes=1024,
            device_count=8,
            device_type=config_pb2.ACCELERATOR_TYPE_TPU,
            device_variant="v5p-8",
        )
        config_v6e = make_scale_group_config(name="v6e", max_slices=2, priority=10, num_vms=1)
        config_v6e.resources.CopyFrom(small_resources)
        config_v5e = make_scale_group_config(name="v5e", max_slices=10, priority=10, num_vms=1)
        config_v5e.resources.CopyFrom(small_resources)

        group_v6e = ScalingGroup(config_v6e, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))
        group_v5e = ScalingGroup(config_v5e, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        # v6e has 1 requesting slice → committed budget = 1 VM, full budget = 2 VMs
        group_v6e.begin_scale_up()

        ts = Timestamp.now()
        demand = make_demand_entries(3, device_type=DeviceType.TPU, device_variant="v5p-8")
        result = route_demand([group_v6e, group_v5e], demand, ts)

        assert result.unmet_entries == []
        # v6e gets committed (1) + waterfall (1) = 2 entries
        assert len(result.routed_entries.get("v6e", [])) == 2
        # v5e gets the overflow
        assert len(result.routed_entries.get("v5e", [])) == 1

    def test_no_committed_budget_when_no_requesting(self):
        """Groups with 0 requesting slices get no committed budget — normal waterfall only."""
        config_a = make_scale_group_config(name="group-a", max_slices=5, priority=10)
        config_b = make_scale_group_config(name="group-b", max_slices=5, priority=20)

        group_a = ScalingGroup(config_a, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))
        group_b = ScalingGroup(config_b, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        ts = Timestamp.now()
        demand = make_demand_entries(3, device_type=DeviceType.TPU, device_variant="v5p-8")
        result = route_demand([group_a, group_b], demand, ts)

        # All entries go to group-a (higher priority = lower number)
        assert len(result.routed_entries.get("group-a", [])) == 3
        assert result.routed_entries.get("group-b") is None
        assert result.unmet_entries == []


class TestAutoscalerAsyncScaleUp:
    """Tests for async scale-up behavior."""

    def test_execute_scale_up_returns_immediately(self):
        """_execute_scale_up returns immediately without blocking."""
        config = make_scale_group_config(name="test-group", min_slices=0, max_slices=5)

        platform = make_mock_platform()
        original_create = platform.create_slice.side_effect

        def slow_create(config, worker_config=None):
            time.sleep(0.5)
            return original_create(config, worker_config)

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

        def slow_create(config, worker_config=None):
            time.sleep(0.2)
            return original_create(config, worker_config)

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

        def slow_create(config, worker_config=None):
            time.sleep(0.2)
            result = original_create(config, worker_config)
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

        def create_slice(self, config, worker_config=None):
            create_barrier.wait(timeout=10)
            return super().create_slice(config, worker_config)

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
    """After scale_up with worker_config, platform handles bootstrap internally
    and describe() reaches READY after tick().
    """
    sg_config = make_scale_group_config(
        name="test-group",
        min_slices=0,
        max_slices=4,
        zones=["us-central1-a"],
    )
    worker_config = config_pb2.WorkerConfig(
        docker_image="test:latest",
        port=10001,
        controller_address="controller:10000",
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
        base_worker_config=worker_config,
    )

    demand = make_demand_entries(1)
    t0 = Timestamp.from_ms(1_000_000)

    decisions = autoscaler.run_once(demand, {}, t0)
    assert len(decisions) == 1
    autoscaler._wait_for_inflight()

    # tick() drives VM state transitions and bootstrap
    platform.tick()

    autoscaler.refresh({})

    assert group.slice_count() == 1
    assert group.ready_slice_count() == 1

    slice_handle = group.slice_handles()[0]
    for vm in slice_handle.describe().workers:
        assert vm._bootstrap_count == 1

    autoscaler.shutdown()


def test_bootstrap_skipped_without_config():
    """Without worker_config, slices reach READY immediately after tick() (no bootstrap)."""
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

    platform.tick()
    autoscaler.refresh({})

    assert group.slice_count() == 1
    assert group.ready_slice_count() == 1

    slice_handle = group.slice_handles()[0]
    for vm in slice_handle.describe().workers:
        assert vm._bootstrap_count == 0, "No bootstrap without config"

    autoscaler.shutdown()


class TestPerGroupWorkerConfig:
    """Tests for _per_group_worker_config merging worker attributes into WorkerConfig."""

    def test_merges_worker_attributes(self):
        """Worker attributes, env, and scale group name are merged into WorkerConfig."""
        base_wc = config_pb2.WorkerConfig(
            docker_image="test:latest",
            port=10001,
            controller_address="controller:10000",
        )
        sg_config = make_scale_group_config(name="west-group", max_slices=5)
        sg_config.worker.attributes[WellKnownAttribute.REGION] = "us-west4"
        sg_config.worker.attributes[WellKnownAttribute.PREEMPTIBLE] = "true"
        sg_config.worker.env["IRIS_REGION"] = "us-west4"

        group = ScalingGroup(sg_config, make_mock_platform())
        autoscaler = make_autoscaler({"west-group": group}, base_worker_config=base_wc)

        wc = autoscaler._per_group_worker_config(group)

        assert wc is not None
        assert wc.docker_image == "test:latest"
        assert wc.worker_attributes[WellKnownAttribute.REGION] == "us-west4"
        assert wc.worker_attributes[WellKnownAttribute.PREEMPTIBLE] == "true"
        assert wc.default_task_env["IRIS_REGION"] == "us-west4"
        assert wc.worker_attributes["scale-group"] == "west-group"
        assert wc.accelerator_type == config_pb2.ACCELERATOR_TYPE_TPU
        assert wc.accelerator_variant == "v5p-8"
        assert wc.gpu_count == 0

    def test_injects_accelerator_config_without_worker_settings(self):
        """Groups without worker settings still inject accelerator config."""
        base_wc = config_pb2.WorkerConfig(
            docker_image="test:latest",
            port=10001,
            controller_address="controller:10000",
        )
        sg_config = make_scale_group_config(
            name="plain-group",
            max_slices=5,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_GPU,
            accelerator_variant="H100",
        )
        sg_config.resources.device_count = 8
        group = ScalingGroup(sg_config, make_mock_platform())
        autoscaler = make_autoscaler({"plain-group": group}, base_worker_config=base_wc)

        wc = autoscaler._per_group_worker_config(group)

        assert wc is not None
        assert wc is not base_wc
        assert wc.accelerator_type == config_pb2.ACCELERATOR_TYPE_GPU
        assert wc.accelerator_variant == "H100"
        assert wc.gpu_count == 8
        assert wc.worker_attributes["scale-group"] == "plain-group"

    def test_returns_none_without_base(self):
        """Without a base worker config, returns None."""
        sg_config = make_scale_group_config(name="test-group", max_slices=5)
        sg_config.worker.attributes[WellKnownAttribute.REGION] = "us-west4"
        group = ScalingGroup(sg_config, make_mock_platform())
        autoscaler = make_autoscaler({"test-group": group}, base_worker_config=None)

        wc = autoscaler._per_group_worker_config(group)

        assert wc is None

    def test_does_not_mutate_base_config(self):
        """Merging should not modify the original base worker config."""
        base_wc = config_pb2.WorkerConfig(
            docker_image="test:latest",
            port=10001,
            controller_address="controller:10000",
        )
        sg_config = make_scale_group_config(name="west-group", max_slices=5)
        sg_config.worker.attributes[WellKnownAttribute.REGION] = "us-west4"

        group = ScalingGroup(sg_config, make_mock_platform())
        autoscaler = make_autoscaler({"west-group": group}, base_worker_config=base_wc)

        autoscaler._per_group_worker_config(group)

        assert WellKnownAttribute.REGION not in base_wc.worker_attributes

    def test_worker_attributes_injected(self):
        """Worker attributes are injected into WorkerConfig."""
        base_wc = config_pb2.WorkerConfig(
            docker_image="ghcr.io/marin-community/iris-worker:latest",
            port=10001,
            controller_address="controller:10000",
        )
        sg_config = make_scale_group_config(name="eu-group", max_slices=5, zones=["europe-west4-b"])
        sg_config.worker.attributes[WellKnownAttribute.REGION] = "europe-west4"

        group = ScalingGroup(sg_config, make_mock_platform())
        autoscaler = make_autoscaler({"eu-group": group}, base_worker_config=base_wc)

        wc = autoscaler._per_group_worker_config(group)

        assert wc is not None
        assert wc.worker_attributes[WellKnownAttribute.REGION] == "europe-west4"


class TestGpuScaleGroupBugs:
    """Reproduction tests for GPU scale group bugs observed on CoreWeave.

    Production behavior: GPU job submitted -> h100-8x scales up -> slice becomes
    READY 20s later -> immediately scaled down. Worker shows CPU=128, Memory=2TB
    but no GPU info. All GPU jobs stuck in no_capacity.

    Root cause: mark_slice_ready() doesn't initialize last_active, so freshly-ready
    slices have last_active=epoch(0) and are immediately eligible for scaledown
    regardless of idle_threshold.
    """

    def test_freshly_ready_slice_has_nonzero_last_active(self):
        """When a slice transitions to READY, last_active should be initialized
        to at least the current time, not epoch(0).

        Reproduces: mark_slice_ready() doesn't touch last_active, so it stays
        at epoch(0). is_slice_eligible_for_scaledown() returns True for
        last_active=epoch(0), making fresh slices immediately eligible for scaledown.
        """
        config = make_scale_group_config(
            name="h100-8x",
            min_slices=0,
            max_slices=1,
        )
        platform = make_mock_platform()
        group = ScalingGroup(
            config,
            platform,
            scale_up_cooldown=Duration.from_ms(0),
            idle_threshold=Duration.from_ms(60_000),
        )

        ts = Timestamp.from_ms(1_000_000)

        # Scale up and complete
        group.begin_scale_up(timestamp=ts)
        handle = group.scale_up(timestamp=ts)
        group.complete_scale_up(handle, ts)

        # Mark the slice as READY (simulates bootstrap completion)
        worker_ids = [w.worker_id for w in handle.describe().workers]
        group.mark_slice_ready(handle.slice_id, worker_ids)

        # last_active should be initialized to at least the ready time
        with group._slices_lock:
            state = group._slices[handle.slice_id]

        assert state.last_active.epoch_ms() > 0, (
            "Freshly READY slice should have last_active set to at least the ready time, "
            f"not epoch(0). Got last_active={state.last_active.epoch_ms()}"
        )

        # Consequently, the slice should NOT be eligible for scaledown immediately
        assert not group.is_slice_eligible_for_scaledown(
            handle.slice_id, ts
        ), "Freshly READY slice should not be eligible for scaledown immediately"

    def test_idle_threshold_protects_freshly_ready_slice(self):
        """A freshly-ready slice should be protected by idle_threshold even when
        demand temporarily drops to 0 (e.g., between job resubmission).

        With idle_threshold=300s, a slice that became ready 1 second ago should
        NOT be eligible for scaledown. But with last_active=epoch(0), the idle
        duration is computed as (current_time - 0) = millions of ms, which always
        exceeds idle_threshold.
        """
        config = make_scale_group_config(
            name="h100-8x",
            min_slices=0,
            max_slices=2,
        )
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True),
            make_mock_slice_handle("slice-002", all_ready=True),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(
            config,
            platform,
            scale_up_cooldown=Duration.from_ms(0),
            idle_threshold=Duration.from_ms(300_000),  # 5 minutes
        )
        group.reconcile()

        # Mark both slices as READY (simulating bootstrap completion)
        _mark_discovered_ready(group, discovered)

        autoscaler = make_autoscaler({"h100-8x": group})

        # Slices just became ready; demand drops to 0 (transient gap)
        # target_capacity = max(0, 0) = 0, ready=2 > 0 -> scaledown check runs
        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        wid1 = slice_001.describe().workers[0].worker_id
        wid2 = slice_002.describe().workers[0].worker_id
        vm_status_map = {
            wid1: WorkerStatus(worker_id=wid1, running_task_ids=frozenset()),
            wid2: WorkerStatus(worker_id=wid2, running_task_ids=frozenset()),
        }

        # Run 1 second after ready — well within the 5-minute idle_threshold.
        # Slices should NOT be scaled down.
        ts = Timestamp.from_ms(10_000)
        autoscaler.run_once([], vm_status_map, ts)

        assert group.slice_count() == 2, (
            "Freshly-ready slices should be protected by idle_threshold (300s). "
            "With last_active=epoch(0), idle_duration is computed from epoch, "
            f"making all slices appear idle for >300s. Got slice_count={group.slice_count()}"
        )


# --- Packing tests ---


def _make_big_demand_entries(
    count: int,
    *,
    cpu_millicores: int = 32000,
    memory_bytes: int = 32 * 1024**3,
    disk_bytes: int = 0,
    device_type: DeviceType = DeviceType.CPU,
    device_variants: frozenset[str] | None = None,
    task_prefix: str = "task",
    coschedule_group_id: str | None = None,
) -> list[DemandEntry]:
    """Create demand entries with explicit resource sizes for packing tests."""
    resources = cluster_pb2.ResourceSpecProto(
        cpu_millicores=cpu_millicores,
        memory_bytes=memory_bytes,
        disk_bytes=disk_bytes,
    )
    normalized = PlacementRequirements(
        device_type=device_type,
        device_variants=device_variants,
        preemptible=None,
        required_regions=None,
        required_zones=None,
    )
    if coschedule_group_id:
        # Coscheduled entries use count as num tasks
        return [
            DemandEntry(
                task_ids=[f"{task_prefix}-{i}" for i in range(count)],
                coschedule_group_id=coschedule_group_id,
                normalized=normalized,
                constraints=[],
                resources=resources,
            )
        ]
    return [
        DemandEntry(
            task_ids=[f"{task_prefix}-{i}"],
            coschedule_group_id=None,
            normalized=normalized,
            constraints=[],
            resources=resources,
        )
        for i in range(count)
    ]


class TestFirstFitDecreasing:
    """Unit tests for the FFD bin packing helper."""

    def test_basic_packing(self):
        """4 requests of (50, 50) each into bins of (100, 100) → 2 VMs."""
        reqs = [AdditiveReq(cpu_millicores=50, memory_bytes=50, disk_bytes=0) for _ in range(4)]
        vm_cap = AdditiveReq(cpu_millicores=100, memory_bytes=100, disk_bytes=0)
        assert first_fit_decreasing(reqs, vm_cap) == 2

    def test_empty_reqs_returns_zero(self):
        vm_cap = AdditiveReq(cpu_millicores=100, memory_bytes=100, disk_bytes=0)
        assert first_fit_decreasing([], vm_cap) == 0

    def test_single_item_per_bin(self):
        """3 items that each fill a bin entirely → 3 VMs."""
        reqs = [AdditiveReq(cpu_millicores=100, memory_bytes=100, disk_bytes=0) for _ in range(3)]
        vm_cap = AdditiveReq(cpu_millicores=100, memory_bytes=100, disk_bytes=0)
        assert first_fit_decreasing(reqs, vm_cap) == 3

    def test_heterogeneous_sizes(self):
        """Mix of large and small items packs efficiently."""
        reqs = [
            AdditiveReq(cpu_millicores=70, memory_bytes=70, disk_bytes=0),
            AdditiveReq(cpu_millicores=30, memory_bytes=30, disk_bytes=0),
            AdditiveReq(cpu_millicores=30, memory_bytes=30, disk_bytes=0),
            AdditiveReq(cpu_millicores=70, memory_bytes=70, disk_bytes=0),
        ]
        vm_cap = AdditiveReq(cpu_millicores=100, memory_bytes=100, disk_bytes=0)
        # FFD sorts descending: [70,70,30,30]. 70+30 fits in 1 bin → 2 VMs
        assert first_fit_decreasing(reqs, vm_cap) == 2

    def test_disk_dimension(self):
        """Disk is respected as a packing dimension."""
        reqs = [
            AdditiveReq(cpu_millicores=10, memory_bytes=10, disk_bytes=60),
            AdditiveReq(cpu_millicores=10, memory_bytes=10, disk_bytes=60),
        ]
        vm_cap = AdditiveReq(cpu_millicores=100, memory_bytes=100, disk_bytes=100)
        # 60+60 > 100 disk, so these need 2 VMs
        assert first_fit_decreasing(reqs, vm_cap) == 2


class TestComputeRequiredSlices:
    """Tests for compute_required_slices with different group configurations."""

    def test_tiny_entries_pack_densely(self):
        """Many small CPU entries pack into a single VM and therefore a single slice."""
        config = make_scale_group_config(
            name="cpu-group",
            max_slices=5,
            num_vms=1,
        )
        group = ScalingGroup(config, make_mock_platform())

        # 16 entries at 1000m CPU, 1024 bytes mem → all fit in 1 VM (128 cores, 128GiB)
        entries = make_demand_entries(16, device_type=DeviceType.CPU)
        assert compute_required_slices(group, entries) == 1

    def test_accelerator_entries_not_packed(self):
        """Accelerator entries get 1 VM each — they are not bin-packed."""
        config = make_scale_group_config(
            name="tpu-group",
            max_slices=10,
            num_vms=1,
        )
        group = ScalingGroup(config, make_mock_platform())

        entries = make_demand_entries(4, device_type=DeviceType.TPU, device_variant="v5p-8")
        assert compute_required_slices(group, entries) == 4

    def test_full_vm_entries_need_one_slice_each(self):
        """Entries that fill an entire VM each need 1 slice per entry (num_vms=1)."""
        config = make_scale_group_config(
            name="cpu-group",
            max_slices=5,
            num_vms=1,
        )
        group = ScalingGroup(config, make_mock_platform())

        entries = _make_big_demand_entries(
            3,
            cpu_millicores=128000,
            memory_bytes=128 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
        )
        assert compute_required_slices(group, entries) == 3

    def test_multi_vm_slice_packs_across_vms(self):
        """With num_vms=4, entries that need 4 VMs fit in 1 slice."""
        config = make_scale_group_config(
            name="multi-vm",
            max_slices=5,
            num_vms=4,
        )
        group = ScalingGroup(config, make_mock_platform())

        # 4 entries, each 128GiB = 4 VMs → ceil(4/4) = 1 slice
        entries = _make_big_demand_entries(
            4,
            cpu_millicores=128000,
            memory_bytes=128 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
        )
        assert compute_required_slices(group, entries) == 1

    def test_multi_vm_slice_needs_multiple_slices(self):
        """With num_vms=4, 5 full-VM entries need ceil(5/4) = 2 slices."""
        config = make_scale_group_config(
            name="multi-vm",
            max_slices=5,
            num_vms=4,
        )
        group = ScalingGroup(config, make_mock_platform())

        entries = _make_big_demand_entries(
            5,
            cpu_millicores=128000,
            memory_bytes=128 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
        )
        assert compute_required_slices(group, entries) == 2

    def test_coscheduled_entries_use_full_slice(self):
        """A coscheduled entry always consumes exactly 1 slice."""
        config = make_scale_group_config(
            name="csc-group",
            max_slices=5,
            num_vms=4,
        )
        group = ScalingGroup(config, make_mock_platform())

        entries = _make_big_demand_entries(
            4,
            cpu_millicores=1000,
            memory_bytes=1024,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
            coschedule_group_id="job-1",
        )
        assert len(entries) == 1
        assert compute_required_slices(group, entries) == 1

    def test_mixed_coscheduled_and_packable(self):
        """Coscheduled entries add 1 slice each; non-coscheduled entries are packed."""
        config = make_scale_group_config(
            name="mixed-group",
            max_slices=10,
            num_vms=4,
        )
        group = ScalingGroup(config, make_mock_platform())

        coscheduled = _make_big_demand_entries(
            4,
            cpu_millicores=1000,
            memory_bytes=1024,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
            coschedule_group_id="job-1",
        )
        # 4 entries at 64GiB each → 2 VMs → ceil(2/4) = 1 slice
        non_coscheduled = _make_big_demand_entries(
            4,
            cpu_millicores=64000,
            memory_bytes=64 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
            task_prefix="noncsc",
        )
        entries = coscheduled + non_coscheduled
        # 1 coscheduled slice + 1 packed slice = 2
        assert compute_required_slices(group, entries) == 2

    def test_no_resources_configured_falls_back_to_entry_count(self):
        """Without per-VM resources, each entry = 1 slice (pre-packing behavior)."""
        config = config_pb2.ScaleGroupConfig(
            name="no-resources",
            max_slices=5,
        )
        # Explicitly don't set resources
        group = ScalingGroup(config, make_mock_platform())
        assert group.resources is None

        entries = make_demand_entries(3, device_type=DeviceType.TPU, device_variant="v5p-8")
        assert compute_required_slices(group, entries) == 3

    def test_empty_entries_returns_zero(self):
        config = make_scale_group_config(name="test", max_slices=5, num_vms=1)
        group = ScalingGroup(config, make_mock_platform())
        assert compute_required_slices(group, []) == 0


class TestPackingRouting:
    """Tests for packing-aware routing and scaling decisions."""

    def test_packing_allows_multiple_cpu_tasks_per_vm(self):
        """16 CPU tasks at 32GiB each pack into 4 VMs of 128GiB → 1 slice.

        Verifies that bin-packing computes required_slices=1 (not 16), and
        that route_demand correctly determines 1 slice is needed.
        """
        config = make_scale_group_config(
            name="cpu-group",
            max_slices=5,
            num_vms=4,
            priority=10,
        )
        group = ScalingGroup(
            config,
            make_mock_platform(),
            scale_up_cooldown=Duration.from_ms(0),
        )

        # 16 entries x 32GiB = 512GiB total. 4 VMs x 128GiB = 512GiB capacity.
        # Packing: 4 entries per VM → 4 VMs → ceil(4/4) = 1 slice needed.
        entries = _make_big_demand_entries(
            16,
            cpu_millicores=32000,
            memory_bytes=32 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
        )
        result = route_demand([group], entries)

        assert result.group_required_slices.get("cpu-group") == 1

    def test_packing_prevents_cpu_walkup(self):
        """CPU entries that pack within group A's budget should not spill to group B."""
        config_a = make_scale_group_config(
            name="group-a",
            max_slices=5,
            num_vms=4,
            priority=10,
        )
        config_b = make_scale_group_config(
            name="group-b",
            max_slices=5,
            num_vms=4,
            priority=20,
        )

        group_a = ScalingGroup(
            config_a,
            make_mock_platform(),
            scale_up_cooldown=Duration.from_ms(0),
        )
        group_b = ScalingGroup(
            config_b,
            make_mock_platform(),
            scale_up_cooldown=Duration.from_ms(0),
        )

        # 8 entries at 32GiB each → 2 VMs needed → ceil(2/4) = 1 slice.
        # All demand routes to group A (higher priority). Nothing spills to B.
        entries = _make_big_demand_entries(
            8,
            cpu_millicores=32000,
            memory_bytes=32 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
        )
        result = route_demand([group_a, group_b], entries)

        assert result.group_required_slices.get("group-a") == 1
        assert result.group_required_slices.get("group-b", 0) == 0

    def test_evaluate_uses_packed_capacity(self):
        """Scale-up triggers when packed demand exceeds existing capacity."""
        config = make_scale_group_config(
            name="test-group",
            max_slices=5,
            num_vms=4,
            priority=10,
        )
        discovered = [make_mock_slice_handle("slice-0", all_ready=True)]
        group = ScalingGroup(
            config,
            make_mock_platform(slices_to_discover=discovered),
            scale_up_cooldown=Duration.from_ms(0),
        )
        group.reconcile()
        _mark_discovered_ready(group, discovered)
        autoscaler = make_autoscaler({"test-group": group})

        # No demand → no scale up (all tasks absorbed by scheduler dry-run)
        decisions = autoscaler.evaluate([])
        assert len(decisions) == 0

        # 5 entries that survived dry-run → 5 VMs → ceil(5/4) = 2 slices needed.
        # Scale-up compares against pending only (ready slices already tested by dry-run).
        big_demand = _make_big_demand_entries(
            5,
            cpu_millicores=128000,
            memory_bytes=128 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
            task_prefix="big",
        )
        decisions = autoscaler.evaluate(big_demand)
        assert len(decisions) == 2
        assert all(d.action == ScalingAction.SCALE_UP for d in decisions)
        assert "required_slices=2 > pending=0" in decisions[0].reason

    def test_scale_down_target_uses_packed_demand(self):
        """Scale-down uses packed required_slices, not entry count."""
        ready_ts = Timestamp.from_ms(1_000)
        config = make_scale_group_config(
            name="test-group",
            max_slices=5,
            num_vms=4,
            priority=10,
        )
        discovered = [
            make_mock_slice_handle("slice-0", all_ready=True, created_at_ms=100),
            make_mock_slice_handle("slice-1", all_ready=True, created_at_ms=200),
        ]
        group = ScalingGroup(
            config,
            make_mock_platform(slices_to_discover=discovered),
            scale_up_cooldown=Duration.from_ms(0),
            idle_threshold=Duration.from_ms(1000),
        )
        group.reconcile()
        _mark_discovered_ready(group, discovered, timestamp=ready_ts)
        autoscaler = make_autoscaler({"test-group": group})

        # 4 entries at 32GiB each → 1 VM → ceil(1/4) = 1 slice. But we have 2 slices.
        entries = _make_big_demand_entries(
            4,
            cpu_millicores=32000,
            memory_bytes=32 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
        )

        # Set current_demand (required_slices=1)
        autoscaler.evaluate(entries, timestamp=Timestamp.from_ms(2_000))
        assert group.current_demand == 1

        # run_once with empty demand (simulating dry-run absorption) and all VMs idle.
        # target_capacity = max(1, 0) = 1 (from current_demand set above).
        slice_0 = group.get_slice("slice-0")
        slice_1 = group.get_slice("slice-1")
        wid_0 = slice_0.describe().workers[0].worker_id
        wid_1 = slice_1.describe().workers[0].worker_id
        vm_status_map = {
            wid_0: WorkerStatus(worker_id=wid_0, running_task_ids=frozenset()),
            wid_1: WorkerStatus(worker_id=wid_1, running_task_ids=frozenset()),
        }
        autoscaler.run_once([], vm_status_map, timestamp=Timestamp.from_ms(10_000))

        # One idle slice should be scaled down.
        assert group.slice_count() == 1

    def test_group_to_launch_uses_packing(self):
        """route_demand computes group_to_launch from packing, not entry count."""
        config = make_scale_group_config(
            name="test-group",
            max_slices=5,
            num_vms=4,
            priority=10,
        )
        group = ScalingGroup(
            config,
            make_mock_platform(),
            scale_up_cooldown=Duration.from_ms(0),
        )

        # 16 entries at 32GiB → 4 VMs → ceil(4/4) = 1 slice.
        # No existing capacity → group_to_launch = 1.
        entries = _make_big_demand_entries(
            16,
            cpu_millicores=32000,
            memory_bytes=32 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
        )
        result = route_demand([group], entries)
        assert result.group_to_launch.get("test-group") == 1

    def test_group_to_launch_with_existing_capacity(self):
        """group_to_launch subtracts existing ready + inflight slices."""
        config = make_scale_group_config(
            name="test-group",
            max_slices=5,
            num_vms=4,
            priority=10,
        )
        discovered = [make_mock_slice_handle("slice-0", all_ready=True)]
        group = ScalingGroup(
            config,
            make_mock_platform(slices_to_discover=discovered),
            scale_up_cooldown=Duration.from_ms(0),
        )
        group.reconcile()
        _mark_discovered_ready(group, discovered)

        # 16 entries → 1 slice needed, 1 exists → group_to_launch = 0
        entries = _make_big_demand_entries(
            16,
            cpu_millicores=32000,
            memory_bytes=32 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
        )
        result = route_demand([group], entries)
        assert result.group_to_launch.get("test-group", 0) == 0


class TestMultiSliceScaleUp:
    """Tests for multi-slice scale-up in a single evaluation cycle."""

    def test_multi_slice_scale_up(self):
        """Group with 0 existing slices scales up to meet full demand in one cycle."""
        config = make_scale_group_config(name="test-group", max_slices=5, num_vms=1, priority=10)
        group = ScalingGroup(
            config, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0), scale_up_rate_limit=1000
        )
        autoscaler = make_autoscaler({"test-group": group})

        # 5 big entries, each fills 1 VM, num_vms=1 → 5 slices needed
        demand = _make_big_demand_entries(
            5,
            cpu_millicores=128000,
            memory_bytes=128 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
        )
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 5
        assert all(d.action == ScalingAction.SCALE_UP for d in decisions)
        assert all(d.scale_group == "test-group" for d in decisions)

    def test_multi_slice_capped_by_max_slices(self):
        """Scale-up decisions are capped by max_slices."""
        config = make_scale_group_config(name="test-group", max_slices=3, num_vms=1, priority=10)
        group = ScalingGroup(
            config, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0), scale_up_rate_limit=1000
        )
        autoscaler = make_autoscaler({"test-group": group})

        # 5 big entries, each fills 1 VM → 5 slices needed, but max=3
        demand = _make_big_demand_entries(
            5,
            cpu_millicores=128000,
            memory_bytes=128 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
        )
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 3
        assert all(d.action == ScalingAction.SCALE_UP for d in decisions)

    def test_cooldown_group_accepts_demand_but_blocks_scale_up(self):
        """A group in COOLDOWN accepts demand routing but blocks scale-up until cooldown expires."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        config = make_scale_group_config(name="test-group", max_slices=5, num_vms=1, priority=10)
        platform = make_mock_platform()
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(3600_000))

        # Put group into COOLDOWN: scale up, then complete
        ts = Timestamp.now()
        group.begin_scale_up()
        handle = group.scale_up(timestamp=ts)
        group.complete_scale_up(handle, ts)
        assert group.availability(ts).status == GroupAvailability.COOLDOWN

        autoscaler = make_autoscaler({"test-group": group})

        # 3 big entries that need 3 slices, but only 1 exists and group is in cooldown
        demand = _make_big_demand_entries(
            3,
            cpu_millicores=128000,
            memory_bytes=128 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
        )
        decisions = autoscaler.evaluate(demand, timestamp=ts)

        # Demand is routed (current_demand > 0) but no scale-up during cooldown
        assert group.current_demand > 0
        assert len(decisions) == 0

    def test_available_group_pre_seeded(self):
        """A group in AVAILABLE state is pre-seeded and accepts demand without a second loop."""
        config = make_scale_group_config(name="test-group", max_slices=5, priority=10)
        group = ScalingGroup(config, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))
        autoscaler = make_autoscaler({"test-group": group})

        demand = make_demand_entries(3, device_type=DeviceType.TPU, device_variant="v5p-8")
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) >= 1
        assert decisions[0].scale_group == "test-group"
        assert group.current_demand > 0

    def test_small_entries_route_with_ready_vm_budget(self):
        """Entries route using budget from ready VMs, not just headroom.

        With max_slices=4, 1 ready (num_vms=1): remaining_vms = (1+0+3)*1 = 4.
        4 tiny CPU entries should all route (not just 3 from headroom alone).
        """
        config = make_scale_group_config(name="test-group", max_slices=4, num_vms=1, priority=10)
        discovered = [make_mock_slice_handle("slice-0", all_ready=True)]
        group = ScalingGroup(
            config,
            make_mock_platform(slices_to_discover=discovered),
            scale_up_cooldown=Duration.from_ms(0),
        )
        group.reconcile()
        _mark_discovered_ready(group, discovered)

        demand = make_demand_entries(4, device_type=DeviceType.TPU, device_variant="v5p-8")
        result = route_demand([group], demand)

        assigned = len(result.routed_entries.get("test-group", []))
        assert assigned == 4
        assert len(result.unmet_entries) == 0

    def test_incremental_demand_growth_triggers_scale_up(self):
        """Starting with small demand then adding more triggers appropriate multi-slice scale-up.

        Phase 1: 4 big entries → 4 VMs → 4 slices (num_vms=1). 0 existing → 4 decisions.
        Execute and mark ready.
        Phase 2: 12 total entries → 12 slices needed. 4 exist → 8 more, capped by max_slices=10 → 6 decisions.
        """
        config = make_scale_group_config(name="test-group", max_slices=10, num_vms=1, priority=10)
        platform = FakePlatform(FakePlatformConfig(config=config))
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0), scale_up_rate_limit=1000)

        as_config = config_pb2.AutoscalerConfig()
        as_config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler({"test-group": group}, config=as_config)

        # Phase 1: 4 big entries, each fills 1 VM
        demand_4 = _make_big_demand_entries(
            4,
            cpu_millicores=128000,
            memory_bytes=128 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
            task_prefix="phase1",
        )
        autoscaler.run_once(demand_4, {})
        autoscaler._wait_for_inflight()
        assert group.slice_count() == 4

        # Mark all slices ready
        platform.tick(Timestamp.now().epoch_ms())
        _mark_all_slices_ready(group)

        # Phase 2: add 8 more entries (12 total)
        demand_12 = demand_4 + _make_big_demand_entries(
            8,
            cpu_millicores=128000,
            memory_bytes=128 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
            task_prefix="phase2",
        )
        decisions = autoscaler.evaluate(demand_12)
        # 12 slices needed, 4 exist → 6 more (capped by max_slices=10)
        assert len(decisions) == 6
        assert all(d.action == ScalingAction.SCALE_UP for d in decisions)

        # Execute and verify
        autoscaler.execute(decisions, Timestamp.now())
        autoscaler._wait_for_inflight()
        assert group.slice_count() == 10

    def test_marin_style_lifecycle(self):
        """Full lifecycle with marin.yaml-style groups: 5 tiers of 2^N VMs per slice.

        Simulates the autoscaler being called repeatedly with growing demand.
        Slices transition through REQUESTING → BOOTING → READY via platform ticks.
        Each entry fills exactly 1 VM (128 CPU, 128GB RAM).

        Verifies:
        1. During cooldown, demand stays routed to the current group (no cascade)
        2. When cooldown expires, the group scales up to serve the routed demand
        3. Only after a group reaches max_slices does demand cascade to the next
        4. The lowest-priority group (tpu-16vm) never receives any load

        Groups (mimicking marin.yaml's TPU v5e tiers):
          tpu-1vm:  num_vms=1,  max_slices=4, priority=10  (capacity:  4 VMs)
          tpu-2vm:  num_vms=2,  max_slices=4, priority=20  (capacity:  8 VMs)
          tpu-4vm:  num_vms=4,  max_slices=4, priority=30  (capacity: 16 VMs)
          tpu-8vm:  num_vms=8,  max_slices=4, priority=40  (capacity: 32 VMs)
          tpu-16vm: num_vms=16, max_slices=4, priority=50  (capacity: 64 VMs) ← no load
        Total capacity of first 4: 4+8+16+32 = 60 VMs. Demand never exceeds 28.
        """
        COOLDOWN_MS = 1000

        platforms: dict[str, FakePlatform] = {}
        groups: dict[str, ScalingGroup] = {}

        for num_vms, priority in [(1, 10), (2, 20), (4, 30), (8, 40), (16, 50)]:
            name = f"tpu-{num_vms}vm"
            cfg = make_scale_group_config(name=name, max_slices=4, num_vms=num_vms, priority=priority)
            plat = FakePlatform(FakePlatformConfig(config=cfg))
            platforms[name] = plat
            groups[name] = ScalingGroup(cfg, plat, scale_up_cooldown=Duration.from_ms(COOLDOWN_MS))

        as_config = config_pb2.AutoscalerConfig()
        as_config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler(groups, config=as_config)

        def make_demand(count):
            return _make_big_demand_entries(
                count,
                cpu_millicores=128000,
                memory_bytes=128 * 1024**3,
                device_type=DeviceType.TPU,
                device_variants=frozenset({"v5p-8"}),
            )

        def advance(ts):
            for plat in platforms.values():
                plat.tick(ts.epoch_ms())
            for g in groups.values():
                _mark_all_slices_ready(g)

        def routed(group_name):
            return len(autoscaler._last_routing_decision.routed_entries.get(group_name, []))

        def assert_no_load_on_last():
            assert routed("tpu-16vm") == 0, "tpu-16vm should never receive load"
            assert groups["tpu-16vm"].slice_count() == 0

        # ── Phase 1: Fill tpu-1vm (highest priority) ──

        # Cycle 1 (t=0): demand=2 → tpu-1vm scales up 2 slices
        t = Timestamp.from_ms(10_000)
        autoscaler.run_once(make_demand(2), {}, timestamp=t)
        autoscaler._wait_for_inflight()
        assert groups["tpu-1vm"].slice_count() == 2
        assert_no_load_on_last()

        # Cycle 2 (t+500ms): demand grows to 4, cooldown blocks scale-up
        t = Timestamp.from_ms(10_500)
        advance(t)
        autoscaler.run_once(make_demand(4), {}, timestamp=t)
        autoscaler._wait_for_inflight()
        assert groups["tpu-1vm"].slice_count() == 2, "cooldown blocks scale-up"
        assert routed("tpu-1vm") == 4, "demand stays in tpu-1vm during cooldown"
        assert routed("tpu-2vm") == 0, "no cascade during cooldown"
        assert_no_load_on_last()

        # Cycle 3 (t+1100ms): cooldown expired → tpu-1vm scales to max
        t = Timestamp.from_ms(11_100)
        advance(t)
        autoscaler.run_once(make_demand(4), {}, timestamp=t)
        autoscaler._wait_for_inflight()
        assert groups["tpu-1vm"].slice_count() == 4, "tpu-1vm at max_slices"
        assert_no_load_on_last()

        # ── Phase 2: tpu-1vm at max → cascade fills tpu-2vm ──

        # Cycle 4 (t+2200ms): demand=8, all cascade past tpu-1vm
        t = Timestamp.from_ms(12_200)
        advance(t)
        autoscaler.run_once(make_demand(8), {}, timestamp=t)
        autoscaler._wait_for_inflight()
        assert groups["tpu-1vm"].slice_count() == 4, "tpu-1vm unchanged"
        assert routed("tpu-2vm") == 8, "all demand cascades to tpu-2vm"
        # 8 entries / 2 vms_per_slice = 4 slices (fills to max)
        assert groups["tpu-2vm"].slice_count() == 4, "tpu-2vm filled to max"
        assert_no_load_on_last()

        # ── Phase 3: tpu-2vm at max → cascade to tpu-4vm with cooldown ──

        # Cycle 5 (t+3300ms): demand=8, cascades to tpu-4vm
        # tpu-4vm: 8 entries / 4 vms_per_slice = 2 slices (partial fill)
        t = Timestamp.from_ms(13_300)
        advance(t)
        autoscaler.run_once(make_demand(8), {}, timestamp=t)
        autoscaler._wait_for_inflight()
        assert groups["tpu-4vm"].slice_count() == 2
        assert_no_load_on_last()

        # Cycle 6 (t+3800ms): demand=16, tpu-4vm in cooldown
        # remaining_vms=(2+0+2)*4=16, routes all 16, but cooldown blocks scale-up
        t = Timestamp.from_ms(13_800)
        advance(t)
        autoscaler.run_once(make_demand(16), {}, timestamp=t)
        autoscaler._wait_for_inflight()
        assert groups["tpu-4vm"].slice_count() == 2, "cooldown blocks scale-up"
        assert routed("tpu-4vm") == 16, "demand stays in tpu-4vm during cooldown"
        assert routed("tpu-8vm") == 0, "no cascade to tpu-8vm during cooldown"
        assert_no_load_on_last()

        # Cycle 7 (t+4400ms): cooldown expired → tpu-4vm scales to max
        # need ceil(16/4)=4 slices, have 2, create 2 more
        t = Timestamp.from_ms(14_400)
        advance(t)
        autoscaler.run_once(make_demand(16), {}, timestamp=t)
        autoscaler._wait_for_inflight()
        assert groups["tpu-4vm"].slice_count() == 4, "tpu-4vm at max_slices"
        assert_no_load_on_last()

        # ── Phase 4: cascade to tpu-8vm ──

        # Cycle 8 (t+5500ms): demand=28, cascades to tpu-8vm
        # tpu-8vm: 28 entries, remaining_vms=(0+0+4)*8=32, routes 28
        # ceil(28/8) = 4 slices → fills to max in one shot
        t = Timestamp.from_ms(15_500)
        advance(t)
        autoscaler.run_once(make_demand(28), {}, timestamp=t)
        autoscaler._wait_for_inflight()
        assert routed("tpu-8vm") == 28
        assert groups["tpu-8vm"].slice_count() == 4, "tpu-8vm at max_slices"
        assert_no_load_on_last()

        # ── Verify final state ──
        expected_slices = {
            "tpu-1vm": 4,
            "tpu-2vm": 4,
            "tpu-4vm": 4,
            "tpu-8vm": 4,
            "tpu-16vm": 0,
        }
        for name, expected in expected_slices.items():
            assert (
                groups[name].slice_count() == expected
            ), f"{name}: expected {expected} slices, got {groups[name].slice_count()}"


class TestRoutingBinPacking:
    """Tests for per-VM bin packing during routing.

    The routing budget packs entries into VM bins during route_demand(), preventing
    premature overflow to lower-priority groups when multiple small entries fit
    in a single VM.
    """

    def _make_group(
        self,
        name: str = "group-a",
        max_slices: int = 1,
        priority: int = 10,
        memory_bytes: int = 128 * 1024**3,
        **kwargs,
    ) -> ScalingGroup:
        resources = config_pb2.ScaleGroupResources(
            cpu_millicores=128000,
            memory_bytes=memory_bytes,
            disk_bytes=100 * 1024**3,
            device_count=8,
            device_type=config_pb2.ACCELERATOR_TYPE_TPU,
            device_variant="v5p-8",
        )
        config = config_pb2.ScaleGroupConfig(
            name=name,
            max_slices=max_slices,
            priority=priority,
            **kwargs,
        )
        config.resources.CopyFrom(resources)
        config.num_vms = kwargs.pop("num_vms", 1)
        config.slice_template.gcp.zone = "us-central1-a"
        return ScalingGroup(config, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

    def _make_entries(self, count: int, memory_bytes: int = 32 * 1024**3) -> list[DemandEntry]:
        resources = cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=memory_bytes)
        normalized = PlacementRequirements(
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
            preemptible=None,
            required_regions=None,
            required_zones=None,
        )
        return [
            DemandEntry(
                task_ids=[f"task-{i}"],
                coschedule_group_id=None,
                normalized=normalized,
                constraints=[],
                resources=resources,
            )
            for i in range(count)
        ]

    def test_routing_packs_small_entries_into_shared_vm(self):
        """4 entries x 32GiB on group with 128GiB VMs, max_slices=1. All 4 route."""
        group = self._make_group(max_slices=1, memory_bytes=128 * 1024**3)
        entries = self._make_entries(4, memory_bytes=32 * 1024**3)

        result = route_demand([group], entries)

        assert len(result.routed_entries.get("group-a", [])) == 4
        assert result.unmet_entries == []
        assert result.group_required_slices["group-a"] == 1

    def test_routing_overflow_when_vm_actually_full(self):
        """5 entries x 32GiB on group A (max_slices=1, 128GiB). 4 pack, 5th overflows to B."""
        group_a = self._make_group(name="group-a", max_slices=1, priority=10, memory_bytes=128 * 1024**3)
        group_b = self._make_group(name="group-b", max_slices=5, priority=20, memory_bytes=128 * 1024**3)
        entries = self._make_entries(5, memory_bytes=32 * 1024**3)

        result = route_demand([group_a, group_b], entries)

        assert len(result.routed_entries.get("group-a", [])) == 4
        assert len(result.routed_entries.get("group-b", [])) == 1
        assert result.unmet_entries == []

    def test_routing_no_resources_falls_back_to_one_per_vm(self):
        """When vm_capacity is None in RoutingBudget, 1 entry = 1 VM (no packing)."""
        from iris.cluster.controller.autoscaler import RoutingBudget

        group = self._make_group(name="group-a", max_slices=2, memory_bytes=128 * 1024**3)
        budget = RoutingBudget(
            group=group,
            vm_capacity=None,  # Force no-resource fallback
            max_vms=2,
            packable_bins=[],
            coscheduled_slices=0,
            assigned_entries=[],
        )

        entries = self._make_entries(3, memory_bytes=32 * 1024**3)
        results = [budget.try_assign(e) for e in entries]

        assert results == [True, True, False]
        assert len(budget.assigned_entries) == 2
        assert len(budget.packable_bins) == 2

    def test_routing_opens_new_bins_from_headroom(self):
        """Entries fill existing VM bins, then headroom allows new bins until max_slices exhausted."""
        group = self._make_group(max_slices=2, memory_bytes=64 * 1024**3)
        entries = self._make_entries(4, memory_bytes=32 * 1024**3)

        result = route_demand([group], entries)

        assert len(result.routed_entries.get("group-a", [])) == 4
        assert result.unmet_entries == []
        assert result.group_required_slices["group-a"] == 2

    def test_routing_coscheduled_still_consumes_full_slice(self):
        """Coscheduled entries consume num_vms from budget, not bin-packed."""
        config = config_pb2.ScaleGroupConfig(
            name="csc-group",
            max_slices=3,
            priority=10,
            num_vms=2,
        )
        config.resources.CopyFrom(DEFAULT_RESOURCES)
        config.slice_template.gcp.zone = "us-central1-a"
        group = ScalingGroup(config, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        resources = cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024)
        normalized = PlacementRequirements(
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
            preemptible=None,
            required_regions=None,
            required_zones=None,
        )
        entries = [
            DemandEntry(
                task_ids=["t0", "t1"],
                coschedule_group_id="job-1",
                normalized=normalized,
                constraints=[],
                resources=resources,
            ),
            DemandEntry(
                task_ids=["t2", "t3"],
                coschedule_group_id="job-2",
                normalized=normalized,
                constraints=[],
                resources=resources,
            ),
        ]

        result = route_demand([group], entries)

        assert len(result.routed_entries.get("csc-group", [])) == 2
        assert result.group_required_slices["csc-group"] == 2

    def test_routing_budget_required_slices_mixed(self):
        """Verify required_slices for mixed coscheduled + packable entries."""
        config = config_pb2.ScaleGroupConfig(
            name="mixed",
            max_slices=5,
            priority=10,
            num_vms=2,
        )
        config.resources.CopyFrom(DEFAULT_RESOURCES)
        config.slice_template.gcp.zone = "us-central1-a"
        group = ScalingGroup(config, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        resources = cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024)
        normalized = PlacementRequirements(
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
            preemptible=None,
            required_regions=None,
            required_zones=None,
        )
        entries = [
            # 1 coscheduled entry (needs 1 slice = 2 VMs)
            DemandEntry(
                task_ids=["t0", "t1"],
                coschedule_group_id="job-1",
                normalized=normalized,
                constraints=[],
                resources=resources,
            ),
            # 3 packable entries (all fit in 1 VM → ceil(1/2) = 1 slice)
            *[
                DemandEntry(
                    task_ids=[f"t-pack-{i}"],
                    coschedule_group_id=None,
                    normalized=normalized,
                    constraints=[],
                    resources=resources,
                )
                for i in range(3)
            ],
        ]

        result = route_demand([group], entries)

        assert len(result.routed_entries.get("mixed", [])) == 4
        # 1 coscheduled slice + ceil(1 bin / 2 vms_per_slice) = 1 packable slice = 2 total
        assert result.group_required_slices["mixed"] == 2

    @pytest.mark.parametrize(
        "device_type,device_variant,make_device",
        [
            (
                DeviceType.GPU,
                "h100",
                lambda: cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant="h100", count=1)),
            ),
            (
                DeviceType.TPU,
                "v5p-8",
                lambda: cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5p-8")),
            ),
        ],
        ids=["gpu", "tpu"],
    )
    def test_routing_accelerator_entries_not_binpacked(self, device_type, device_variant, make_device):
        """Accelerator entries (GPU/TPU) must each get their own VM, not share a bin."""
        group = self._make_group(max_slices=2, memory_bytes=128 * 1024**3)

        resources = cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=32 * 1024**3, device=make_device())
        normalized = PlacementRequirements(
            device_type=device_type,
            device_variants=frozenset({device_variant}),
            preemptible=None,
            required_regions=None,
            required_zones=None,
        )
        entries = [
            DemandEntry(
                task_ids=[f"task-{i}"],
                coschedule_group_id=None,
                normalized=normalized,
                constraints=[],
                resources=resources,
            )
            for i in range(2)
        ]

        result = route_demand([group], entries)

        assert len(result.routed_entries.get("group-a", [])) == 2
        # Each accelerator entry needs its own VM → 2 VMs → 2 slices (num_vms=1).
        assert result.group_required_slices["group-a"] == 2


class TestScaleUpRateLimiting:
    """Tests for per-group token bucket rate limiting of scale-up execution."""

    def test_rate_limited_scale_up_logs_action(self):
        """With rate_limit=1, 5 decisions produce 1 executed + 4 rate_limited actions."""
        config = make_scale_group_config(name="test-group", max_slices=10, num_vms=1, priority=10)
        platform = FakePlatform(FakePlatformConfig(config=config))
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0), scale_up_rate_limit=1)

        as_config = config_pb2.AutoscalerConfig()
        as_config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler({"test-group": group}, config=as_config)

        demand = _make_big_demand_entries(
            5,
            cpu_millicores=128000,
            memory_bytes=128 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
        )
        ts = Timestamp.from_ms(100_000)
        decisions = autoscaler.evaluate(demand, timestamp=ts)
        assert len(decisions) == 5

        autoscaler.execute(decisions, ts)
        autoscaler._wait_for_inflight()

        # Only 1 should have actually executed (rate_limit=1)
        assert group.slice_count() == 1

        # Check action log: 1 scale_up + 4 rate_limited
        actions = list(autoscaler._action_log)
        rate_limited = [a for a in actions if a.action_type == "rate_limited"]
        scale_ups = [a for a in actions if a.action_type == "scale_up"]
        assert len(rate_limited) == 4
        assert len(scale_ups) == 1

    def test_rate_limited_decisions_served_next_cycle(self):
        """Deferred decisions get served on subsequent evaluate+execute cycles as tokens refill."""
        config = make_scale_group_config(name="test-group", max_slices=10, num_vms=1, priority=10)
        platform = FakePlatform(FakePlatformConfig(config=config))
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0), scale_up_rate_limit=2)

        as_config = config_pb2.AutoscalerConfig()
        as_config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler({"test-group": group}, config=as_config)

        demand = _make_big_demand_entries(
            6,
            cpu_millicores=128000,
            memory_bytes=128 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
        )

        # Cycle 1: 6 decisions, only 2 pass rate limit
        ts = Timestamp.from_ms(100_000)
        decisions = autoscaler.evaluate(demand, timestamp=ts)
        assert len(decisions) == 6
        autoscaler.execute(decisions, ts)
        autoscaler._wait_for_inflight()
        assert group.slice_count() == 2

        # Advance time by 1 minute for full refill, mark slices ready
        platform.tick(ts.epoch_ms())
        _mark_all_slices_ready(group)
        ts2 = ts.add_ms(60_000)

        # Cycle 2: re-evaluate with same demand, 4 remaining needed
        decisions2 = autoscaler.evaluate(demand, timestamp=ts2)
        assert len(decisions2) == 4  # 6 needed - 2 existing = 4
        autoscaler.execute(decisions2, ts2)
        autoscaler._wait_for_inflight()
        # 2 more tokens available after 1 minute refill
        assert group.slice_count() == 4

    def test_high_rate_limit_allows_all_decisions(self):
        """With a high rate limit, all decisions execute in one cycle."""
        config = make_scale_group_config(name="test-group", max_slices=10, num_vms=1, priority=10)
        platform = FakePlatform(FakePlatformConfig(config=config))
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0), scale_up_rate_limit=1000)

        as_config = config_pb2.AutoscalerConfig()
        as_config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler({"test-group": group}, config=as_config)

        demand = _make_big_demand_entries(
            10,
            cpu_millicores=128000,
            memory_bytes=128 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
        )
        ts = Timestamp.from_ms(100_000)
        decisions = autoscaler.evaluate(demand, timestamp=ts)
        assert len(decisions) == 10
        autoscaler.execute(decisions, ts)
        autoscaler._wait_for_inflight()
        assert group.slice_count() == 10


class TestCheckCoschedulingFeasibility:
    """Tests for Autoscaler.check_coscheduling_feasibility()."""

    def _make_constraints(self):
        return make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")[0].constraints

    def test_feasible_exact_match(self):
        """Replicas == num_vms is feasible."""
        config = make_scale_group_config(name="group-4", max_slices=5, num_vms=4)
        autoscaler = make_autoscaler({"group-4": ScalingGroup(config, make_mock_platform())})
        assert autoscaler.check_coscheduling_feasibility(4, self._make_constraints()) is None

    def test_feasible_exact_multiple(self):
        """Replicas that are an exact multiple of num_vms are feasible (e.g. 8 replicas on 4-VM group)."""
        config = make_scale_group_config(name="group-4", max_slices=5, num_vms=4)
        autoscaler = make_autoscaler({"group-4": ScalingGroup(config, make_mock_platform())})
        assert autoscaler.check_coscheduling_feasibility(8, self._make_constraints()) is None

    def test_infeasible_not_a_multiple(self):
        """Replicas that aren't a multiple of any group's num_vms are rejected."""
        config = make_scale_group_config(name="group-3", max_slices=5, num_vms=3)
        autoscaler = make_autoscaler({"group-3": ScalingGroup(config, make_mock_platform())})
        result = autoscaler.check_coscheduling_feasibility(8, self._make_constraints())
        assert result is not None
        assert "8" in result

    def test_infeasible_no_group_matches_constraints(self):
        """Returns error when no group matches the device constraints."""
        config = make_scale_group_config(
            name="gpu-group", max_slices=5, num_vms=8, accelerator_type=config_pb2.ACCELERATOR_TYPE_GPU
        )
        autoscaler = make_autoscaler({"gpu-group": ScalingGroup(config, make_mock_platform())})
        result = autoscaler.check_coscheduling_feasibility(8, self._make_constraints())
        assert result is not None
        assert "no scaling group matches" in result

    def test_no_groups_returns_none(self):
        """Returns None when there are no groups (no validation possible)."""
        autoscaler = make_autoscaler({})
        assert autoscaler.check_coscheduling_feasibility(8, []) is None


class TestAutoscalerUnresolvableTimeout:
    """Tests for UNKNOWN slice → FAILED after timeout behavior."""

    def _make_group_with_unknown_slice(
        self, scale_group_config: config_pb2.ScaleGroupConfig, created_at_ms: int
    ) -> tuple[Autoscaler, ScalingGroup, MagicMock]:
        """Set up a group with one BOOTING slice that reports UNKNOWN state."""
        handle = make_mock_slice_handle("slice-001", created_at_ms=created_at_ms)
        handle.describe.return_value = SliceStatus(state=CloudSliceState.UNKNOWN, worker_count=0, workers=[])

        platform = make_mock_platform(slices_to_discover=[handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()

        short_timeout = Duration.from_minutes(15)
        autoscaler = Autoscaler(
            scale_groups={"test-group": group},
            evaluation_interval=Duration.from_seconds(0.1),
            platform=platform,
            unresolvable_timeout=short_timeout,
        )
        return autoscaler, group, handle

    def test_unknown_before_timeout_stays_booting(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """A slice in UNKNOWN state before the timeout remains tracked (BOOTING)."""
        created_at_ms = 0
        autoscaler, group, _ = self._make_group_with_unknown_slice(scale_group_config, created_at_ms)

        # Refresh at 5 min — well under 15 min timeout
        autoscaler.refresh({}, timestamp=Timestamp.from_ms(5 * 60 * 1000))

        assert group.slice_count() == 1
        assert group.ready_slice_count() == 0
        autoscaler.shutdown()

    def test_unknown_after_timeout_triggers_failure(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """A slice in UNKNOWN state past the timeout is failed and removed."""
        created_at_ms = 0
        autoscaler, group, _ = self._make_group_with_unknown_slice(scale_group_config, created_at_ms)

        # Refresh at 16 min — past the 15 min timeout
        autoscaler.refresh({}, timestamp=Timestamp.from_ms(16 * 60 * 1000))

        assert group.slice_count() == 0
        autoscaler.shutdown()

    def test_unknown_then_ready_before_timeout_recovers(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """A slice that was UNKNOWN but becomes READY before timeout is marked ready."""
        created_at_ms = 0
        handle = make_mock_slice_handle("slice-001", created_at_ms=created_at_ms)
        platform = make_mock_platform(slices_to_discover=[handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()

        autoscaler = Autoscaler(
            scale_groups={"test-group": group},
            evaluation_interval=Duration.from_seconds(0.1),
            platform=platform,
            unresolvable_timeout=DEFAULT_UNRESOLVABLE_TIMEOUT,
        )

        # First refresh: UNKNOWN at 5 min → should stay BOOTING
        handle.describe.return_value = SliceStatus(state=CloudSliceState.UNKNOWN, worker_count=0, workers=[])
        autoscaler.refresh({}, timestamp=Timestamp.from_ms(5 * 60 * 1000))
        assert group.slice_count() == 1

        # Second refresh: READY before timeout
        worker = make_mock_worker_handle("slice-001-vm-0", "10.0.1.0", vm_pb2.VM_STATE_READY)
        handle.describe.return_value = SliceStatus(state=CloudSliceState.READY, worker_count=1, workers=[worker])
        autoscaler.refresh({}, timestamp=Timestamp.from_ms(10 * 60 * 1000))

        assert group.ready_slice_count() == 1
        autoscaler.shutdown()
