# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ScalingGroup behavior.

These tests focus on observable behavior - scaling policy decisions,
VM group management, and state tracking - not on implementation details.
"""

from unittest.mock import MagicMock

import pytest
from iris.cluster.controller.scaling_group import (
    ScalingGroup,
    SliceLifecycleState,
    SliceState,
    _zones_from_config,
    slice_handle_status,
)
from iris.cluster.platform.base import CloudSliceState, CloudVmState, QuotaExhaustedError, SliceStatus, VmStatus
from iris.cluster.types import VmWorkerStatus
from iris.rpc import config_pb2, vm_pb2
from iris.time_utils import Duration, Timestamp

DEFAULT_RESOURCES = config_pb2.ScaleGroupResources(
    cpu=64,
    memory_bytes=64 * 1024**3,
    disk_bytes=100 * 1024**3,
    gpu_count=0,
    tpu_count=8,
)


def _with_resources(config: config_pb2.ScaleGroupConfig, *, slice_size: int = 1) -> config_pb2.ScaleGroupConfig:
    if not config.HasField("resources"):
        config.resources.CopyFrom(DEFAULT_RESOURCES)
    if not config.HasField("slice_size"):
        config.slice_size = slice_size
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
    all_ready: bool = True,
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
        slice_state = CloudSliceState.READY  # Slice is up, but VM is failed
    elif all(s == vm_pb2.VM_STATE_READY for s in vm_states):
        slice_state = CloudSliceState.READY
    elif all(s == vm_pb2.VM_STATE_TERMINATED for s in vm_states):
        slice_state = CloudSliceState.DELETING
    else:
        slice_state = CloudSliceState.CREATING

    # Generate unique addresses by hashing slice_id
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


def make_mock_platform(slice_handles_to_discover: list[MagicMock] | None = None) -> MagicMock:
    """Create a mock Platform for testing."""
    platform = MagicMock()
    platform.list_slices.return_value = slice_handles_to_discover or []

    create_count = [0]

    def create_slice_side_effect(config: config_pb2.SliceConfig) -> MagicMock:
        create_count[0] += 1
        slice_id = f"new-slice-{create_count[0]}"
        return make_mock_slice_handle(slice_id)

    platform.create_slice.side_effect = create_slice_side_effect
    return platform


def _get_vm_address(handle: MagicMock) -> str:
    """Get the first VM's internal_address from a SliceHandle."""
    return handle.describe().vms[0].internal_address


def _get_slice_state(group: ScalingGroup, handle: MagicMock) -> SliceState:
    """Get the SliceState for a handle from its group."""
    with group._slices_lock:
        return group._slices[handle.slice_id]


def _tracked_scale_up(group: ScalingGroup, timestamp: Timestamp | None = None, **kwargs) -> MagicMock:
    """Scale up with full lifecycle tracking: begin -> create -> complete.

    This replaces the old group.scale_up() pattern in tests, since scale_up()
    no longer tracks state internally.
    """
    timestamp = timestamp or Timestamp.from_ms(1000000)
    group.begin_scale_up()
    handle = group.scale_up(timestamp=timestamp, **kwargs)
    group.complete_scale_up(handle, timestamp)
    return handle


@pytest.fixture
def scale_group_config() -> config_pb2.ScaleGroupConfig:
    """A standard scale group configuration for tests."""
    config = config_pb2.ScaleGroupConfig(
        name="test-group",
        min_slices=1,
        max_slices=5,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5p-8",
    )
    config.slice_template.gcp.runtime_version = "v2-alpha-tpuv5"
    config.slice_template.gcp.zone = "us-central1-a"
    return _with_resources(config)


@pytest.fixture
def unbounded_config() -> config_pb2.ScaleGroupConfig:
    """A scale group with no min/max constraints."""
    config = config_pb2.ScaleGroupConfig(
        name="unbounded-group",
        min_slices=0,
        max_slices=100,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5p-8",
    )
    config.slice_template.gcp.runtime_version = "v2-alpha-tpuv5"
    config.slice_template.gcp.zone = "us-central1-a"
    return _with_resources(config)


class TestScalingGroupVmGroupOwnership:
    """Tests for VM group ownership and lifecycle."""

    def test_reconcile_adopts_discovered_vm_groups(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """reconcile() populates VM groups from the Platform."""
        discovered = [
            make_mock_slice_handle("slice-001"),
            make_mock_slice_handle("slice-002"),
        ]
        platform = make_mock_platform(slice_handles_to_discover=discovered)

        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()

        assert group.slice_count() == 2
        assert group.get_slice("slice-001") is not None
        assert group.get_slice("slice-002") is not None

    def test_scale_up_creates_and_tracks_vm_group(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Full lifecycle (begin + scale_up + complete) creates and tracks a slice."""
        platform = make_mock_platform()
        group = ScalingGroup(scale_group_config, platform)

        new_handle = _tracked_scale_up(group)

        platform.create_slice.assert_called_once()
        assert group.slice_count() == 1
        assert new_handle in group.slice_handles()

    def test_scale_up_passes_tags_as_labels(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """scale_up() passes tags as labels in the slice config."""
        platform = make_mock_platform()
        group = ScalingGroup(scale_group_config, platform)

        group.scale_up(tags={"env": "prod", "team": "ml"})

        platform.create_slice.assert_called_once()
        slice_config = platform.create_slice.call_args[0][0]
        assert slice_config.labels["env"] == "prod"
        assert slice_config.labels["team"] == "ml"

    def test_scale_down_terminates_and_removes_vm_group(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """scale_down() terminates the VM group and removes it from tracking."""
        mock_handle = make_mock_slice_handle("slice-001")
        platform = make_mock_platform(slice_handles_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()

        assert group.slice_count() == 1

        group.scale_down("slice-001")

        mock_handle.terminate.assert_called_once()
        assert group.slice_count() == 0
        assert group.get_slice("slice-001") is None

    def test_scale_down_nonexistent_vm_group_is_noop(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """scale_down() on a nonexistent VM group does nothing."""
        platform = make_mock_platform()
        group = ScalingGroup(scale_group_config, platform)

        # Should not raise
        group.scale_down("nonexistent-slice")

        assert group.slice_count() == 0

    def test_ready_slice_count(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """ready_slice_count() only counts VM groups where all VMs are ready."""
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True),
            make_mock_slice_handle("slice-002", all_ready=False, vm_states=[vm_pb2.VM_STATE_BOOTING]),
            make_mock_slice_handle("slice-003", all_ready=True),
        ]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        group.refresh_all_slices()

        assert group.slice_count() == 3
        assert group.ready_slice_count() == 2


class TestScalingGroupScalingPolicy:
    """Tests for scaling policy decisions (can_scale_up, can_scale_down)."""

    def test_can_scale_up_when_below_max(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """can_scale_up() returns True when below max_slices."""
        platform = make_mock_platform()
        group = ScalingGroup(scale_group_config, platform)

        assert group.can_scale_up()

    def test_cannot_scale_up_at_max_slices(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """can_scale_up() returns False when at max_slices."""
        discovered = [make_mock_slice_handle(f"slice-{i}") for i in range(5)]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()

        assert group.slice_count() == 5  # max_slices
        assert not group.can_scale_up()

    def test_cannot_scale_up_during_backoff(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """can_scale_up() returns False during backoff period."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)

        ts = Timestamp.from_ms(1000000)
        group.record_failure(timestamp=ts)

        # During backoff period
        assert not group.can_scale_up(timestamp=Timestamp.from_ms(1001000))
        # After backoff expires (default 5s = 5000ms)
        assert group.can_scale_up(timestamp=Timestamp.from_ms(1006000))

    def test_cannot_scale_up_during_cooldown(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """can_scale_up() returns False during cooldown period after scale-up."""
        platform = make_mock_platform()
        group = ScalingGroup(
            unbounded_config,
            platform,
            scale_up_cooldown=Duration.from_ms(10000),
        )

        ts = Timestamp.from_ms(1000000)
        _tracked_scale_up(group, timestamp=ts)

        # During cooldown
        assert not group.can_scale_up(timestamp=Timestamp.from_ms(1005000))
        # After cooldown expires
        assert group.can_scale_up(timestamp=Timestamp.from_ms(1015000))

    def test_can_scale_down_when_above_min(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """can_scale_down() returns True when above min_slices."""
        discovered = [make_mock_slice_handle("slice-001"), make_mock_slice_handle("slice-002")]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()

        assert group.slice_count() == 2
        assert scale_group_config.min_slices == 1
        assert group.can_scale_down()

    def test_cannot_scale_down_at_min_slices(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """can_scale_down() returns False when at min_slices."""
        discovered = [make_mock_slice_handle("slice-001")]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()

        assert group.slice_count() == 1  # min_slices
        assert not group.can_scale_down()

    def test_cannot_scale_down_during_cooldown(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """can_scale_down() returns False during cooldown period."""
        discovered = [make_mock_slice_handle("slice-001"), make_mock_slice_handle("slice-002")]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(
            unbounded_config,
            platform,
            scale_down_cooldown=Duration.from_ms(10000),
        )
        group.reconcile()

        ts = Timestamp.from_ms(1000000)
        group.scale_down("slice-001", timestamp=ts)

        # During cooldown
        assert not group.can_scale_down(timestamp=Timestamp.from_ms(1005000))
        # After cooldown expires
        assert group.can_scale_down(timestamp=Timestamp.from_ms(1015000))


class TestScalingGroupBackoff:
    """Tests for exponential backoff behavior."""

    def test_record_failure_applies_initial_backoff(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """First failure applies initial backoff duration."""
        platform = make_mock_platform()
        group = ScalingGroup(
            unbounded_config,
            platform,
            backoff_initial=Duration.from_seconds(5.0),
        )

        ts = Timestamp.from_ms(1000000)
        group.record_failure(timestamp=ts)

        assert group.consecutive_failures == 1
        assert group._backoff_until is not None
        assert group._backoff_until.as_timestamp().epoch_ms() == 1005000

    def test_record_failure_applies_exponential_backoff(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Consecutive failures double the backoff time."""
        platform = make_mock_platform()
        group = ScalingGroup(
            unbounded_config,
            platform,
            backoff_initial=Duration.from_seconds(5.0),
            backoff_factor=2.0,
        )

        ts = Timestamp.from_ms(1000000)
        group.record_failure(timestamp=ts)  # 5000ms
        group.record_failure(timestamp=ts)  # 10000ms
        group.record_failure(timestamp=ts)  # 20000ms

        assert group.consecutive_failures == 3
        assert group._backoff_until is not None
        assert group._backoff_until.as_timestamp().epoch_ms() == 1020000

    def test_backoff_capped_at_maximum(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Backoff duration is capped at max value."""
        platform = make_mock_platform()
        group = ScalingGroup(
            unbounded_config,
            platform,
            backoff_initial=Duration.from_seconds(5.0),
            backoff_max=Duration.from_seconds(15.0),
            backoff_factor=2.0,
        )

        ts = Timestamp.from_ms(1000000)
        for _ in range(10):  # Many failures
            group.record_failure(timestamp=ts)

        # Should be capped at max
        assert group._backoff_until is not None
        assert group._backoff_until.as_timestamp().epoch_ms() == 1015000

    def test_scale_up_resets_backoff(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Successful scale-up via complete_scale_up resets backoff state."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)

        ts = Timestamp.from_ms(1000000)
        group.record_failure(timestamp=ts)
        group.record_failure(timestamp=ts)
        assert group.consecutive_failures == 2

        _tracked_scale_up(group, timestamp=Timestamp.from_ms(1100000))

        assert group.consecutive_failures == 0
        assert group._backoff_until is None


class TestScalingGroupDemandTracking:
    """Tests for demand tracking."""

    def test_update_demand_tracks_peak(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """update_demand() tracks peak demand."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)

        group.update_demand(5)
        group.update_demand(10)
        group.update_demand(3)

        assert group.current_demand == 3
        assert group.peak_demand == 10


class TestScalingGroupIdleTracking:
    """Tests for per-slice idle tracking and scale-down eligibility."""

    def test_slice_eligible_when_never_active(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Fresh slice (no activity tracked) is eligible for scaledown."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(60_000))
        _tracked_scale_up(group)
        slice_id = next(iter(group.slice_handles())).slice_id

        # Never had activity tracked -> eligible
        assert group.is_slice_eligible_for_scaledown(slice_id, Timestamp.from_ms(1000))

    def test_slice_not_eligible_when_recently_active(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Recently active slice is not eligible for scaledown."""
        discovered = [make_mock_slice_handle("slice-001", all_ready=True)]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(60_000))
        group.reconcile()
        group.refresh_all_slices()

        # Get the VM address from the SliceHandle
        handle = group.get_slice("slice-001")
        vm_address = _get_vm_address(handle)

        # Mark slice as active at t=1000 via update_slice_activity
        vm_status_map = {
            vm_address: VmWorkerStatus(vm_address=vm_address, running_task_ids=frozenset({"task-1"})),
        }
        group.update_slice_activity(vm_status_map, Timestamp.from_ms(1000))

        # Not enough time passed (30s < 60s threshold)
        assert not group.is_slice_eligible_for_scaledown("slice-001", Timestamp.from_ms(30_000))

    def test_slice_eligible_after_idle_threshold(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Slice is eligible after idle_threshold of inactivity."""
        discovered = [make_mock_slice_handle("slice-001", all_ready=True)]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(60_000))
        group.reconcile()

        handle = group.get_slice("slice-001")
        vm_address = _get_vm_address(handle)

        # Mark slice as active at t=1000 via update_slice_activity
        vm_status_map = {
            vm_address: VmWorkerStatus(vm_address=vm_address, running_task_ids=frozenset({"task-1"})),
        }
        group.update_slice_activity(vm_status_map, Timestamp.from_ms(1000))

        # After threshold (61s > 60s) -> eligible
        assert group.is_slice_eligible_for_scaledown("slice-001", Timestamp.from_ms(61_001))

    def test_get_idle_slices_returns_longest_idle_first(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """get_idle_slices returns slices sorted by idle time (longest first)."""
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True),
            make_mock_slice_handle("slice-002", all_ready=True),
        ]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(1000))
        group.reconcile()
        group.refresh_all_slices()

        # Get VM addresses from the handles
        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        slice_001_addr = _get_vm_address(slice_001)
        slice_002_addr = _get_vm_address(slice_002)

        # Mark slice-001 as active at t=1000 (will be idle longer)
        vm_status_map_001 = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset({"task-1"})),
        }
        group.update_slice_activity(vm_status_map_001, Timestamp.from_ms(1000))

        # Mark slice-002 as active at t=5000 (more recently active)
        vm_status_map_002 = {
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset({"task-2"})),
        }
        group.update_slice_activity(vm_status_map_002, Timestamp.from_ms(5000))

        # At timestamp 10000, slice-001 has been idle longer (9s vs 5s)
        idle_slices = group.get_idle_slices(Timestamp.from_ms(10_000))
        assert len(idle_slices) == 2
        assert idle_slices[0].handle.slice_id == "slice-001"  # Longest idle first

    def test_update_slice_activity_tracks_active_slices(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """update_slice_activity updates timestamp only for slices with active workers."""
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True),
            make_mock_slice_handle("slice-002", all_ready=True),
        ]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform)
        group.reconcile()
        group.refresh_all_slices()

        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        slice_001_addr = _get_vm_address(slice_001)
        slice_002_addr = _get_vm_address(slice_002)

        # slice-001 has running tasks, slice-002 is idle
        vm_status_map = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset({"task-1"})),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset()),
        }

        group.update_slice_activity(vm_status_map, Timestamp.from_ms(5000))

        # Observable behavior: slice-001 should not be eligible for scaledown (recently active)
        # slice-002 should remain eligible (no activity tracked)
        assert not group.is_slice_eligible_for_scaledown("slice-001", Timestamp.from_ms(5000))
        assert group.is_slice_eligible_for_scaledown("slice-002", Timestamp.from_ms(5000))

    def test_scale_down_if_idle_terminates_eligible_slice(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """scale_down_if_idle terminates an eligible idle slice."""
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True),
            make_mock_slice_handle("slice-002", all_ready=True),
        ]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(
            unbounded_config, platform, idle_threshold=Duration.from_ms(1000), scale_down_cooldown=Duration.from_ms(0)
        )
        group.reconcile()
        group.refresh_all_slices()

        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        slice_001_addr = _get_vm_address(slice_001)
        slice_002_addr = _get_vm_address(slice_002)

        # Mark both slices as active at t=0 (they'll be idle for 10s, exceeding 1s threshold)
        vm_status_map_active = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset({"task-1"})),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset({"task-2"})),
        }
        group.update_slice_activity(vm_status_map_active, Timestamp.from_ms(0))

        # At t=10_000, workers are now idle
        vm_status_map_idle = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset()),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset()),
        }

        # Target capacity = 1, but we have 2 ready slices
        scaled_down = group.scale_down_if_idle(
            vm_status_map_idle, target_capacity=1, timestamp=Timestamp.from_ms(10_000)
        )

        assert scaled_down is not None
        assert group.slice_count() == 1  # One slice was terminated

    def test_scale_down_if_idle_respects_target_capacity(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """scale_down_if_idle does nothing when at or below target capacity."""
        discovered = [make_mock_slice_handle("slice-001", all_ready=True)]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(
            unbounded_config, platform, idle_threshold=Duration.from_ms(1000), scale_down_cooldown=Duration.from_ms(0)
        )
        group.reconcile()

        slice_001 = group.get_slice("slice-001")
        slice_001_addr = _get_vm_address(slice_001)
        vm_status_map = {slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset())}

        # Target = 1, ready = 1, should not scale down
        scaled_down = group.scale_down_if_idle(vm_status_map, target_capacity=1, timestamp=Timestamp.from_ms(10_000))

        assert scaled_down is None
        assert group.slice_count() == 1

    def test_scale_down_cleans_up_idle_tracking(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """scale_down removes the slice from idle tracking."""
        discovered = [make_mock_slice_handle("slice-001", all_ready=True)]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform)
        group.reconcile()

        handle = group.get_slice("slice-001")
        vm_address = _get_vm_address(handle)

        vm_status_map = {
            vm_address: VmWorkerStatus(vm_address=vm_address, running_task_ids=frozenset({"task-1"})),
        }
        group.update_slice_activity(vm_status_map, Timestamp.from_ms(1000))

        # Scale down
        group.scale_down("slice-001")

        assert group.get_slice("slice-001") is None


class TestScalingGroupVmGroupStateCounts:
    """Tests for slice_state_counts() aggregation."""

    @pytest.mark.parametrize(
        "vm_state,expected_state",
        [
            (vm_pb2.VM_STATE_READY, SliceLifecycleState.READY),
            (vm_pb2.VM_STATE_BOOTING, SliceLifecycleState.BOOTING),
            # INITIALIZING is an Iris lifecycle concept not present at the cloud level.
            # The Platform adapter maps unknown cloud states to BOOTING. INITIALIZING
            # will come from WorkerVm lifecycle tracking in a future task.
            (vm_pb2.VM_STATE_FAILED, SliceLifecycleState.FAILED),
        ],
    )
    def test_counts_vm_groups_by_state(
        self,
        scale_group_config: config_pb2.ScaleGroupConfig,
        vm_state: vm_pb2.VmState,
        expected_state: SliceLifecycleState,
    ):
        """VM groups are counted in the correct category based on VM state."""
        discovered = [make_mock_slice_handle("slice-001", vm_states=[vm_state])]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        group.refresh_all_slices()

        counts = group.slice_state_counts()

        assert counts[expected_state] == 1
        for state in SliceLifecycleState:
            if state != expected_state:
                assert counts[state] == 0

    def test_failed_takes_precedence(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """A VM group with any failed VM is counted as failed."""
        discovered = [
            make_mock_slice_handle(
                "slice-001",
                vm_states=[vm_pb2.VM_STATE_READY, vm_pb2.VM_STATE_FAILED],
            ),
        ]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        group.refresh_all_slices()

        counts = group.slice_state_counts()

        assert counts[SliceLifecycleState.FAILED] == 1
        assert counts[SliceLifecycleState.READY] == 0

    def test_skips_terminated_vm_groups(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Terminated VM groups are not counted."""
        discovered = [
            make_mock_slice_handle("slice-001", vm_states=[vm_pb2.VM_STATE_TERMINATED]),
        ]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        group.refresh_all_slices()

        counts = group.slice_state_counts()

        assert counts[SliceLifecycleState.READY] == 0
        assert counts[SliceLifecycleState.BOOTING] == 0
        assert counts[SliceLifecycleState.INITIALIZING] == 0
        assert counts[SliceLifecycleState.FAILED] == 0


class TestScalingGroupAvailability:
    """Tests for availability state computation and waterfall routing support."""

    def test_available_when_no_constraints(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Group is AVAILABLE when not in backoff, quota ok, and under capacity."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)

        state = group.availability()
        assert state.status == GroupAvailability.AVAILABLE

    def test_at_capacity_when_at_max_slices(self):
        """Group is AT_CAPACITY when at max_slices."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        config = _with_resources(
            config_pb2.ScaleGroupConfig(
                name="test-group",
                min_slices=0,
                max_slices=2,
                accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
                accelerator_variant="v5p-8",
            ),
        )
        discovered = [make_mock_slice_handle(f"slice-{i}") for i in range(2)]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(config, platform)
        group.reconcile()

        state = group.availability()
        assert state.status == GroupAvailability.AT_CAPACITY

    def test_backoff_when_in_backoff_period(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Group is in BACKOFF when backoff timer is active."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform, backoff_initial=Duration.from_seconds(60.0))
        ts = Timestamp.from_ms(1000000)
        group.record_failure(timestamp=ts)

        state = group.availability(Timestamp.from_ms(1001000))  # Still in backoff
        assert state.status == GroupAvailability.BACKOFF
        assert state.until is not None

    def test_can_accept_demand_true_when_available(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """can_accept_demand() returns True when AVAILABLE."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)

        assert group.can_accept_demand() is True

    def test_can_accept_demand_false_when_at_capacity(self):
        """can_accept_demand() returns False when AT_CAPACITY."""
        config = _with_resources(
            config_pb2.ScaleGroupConfig(
                name="test-group",
                min_slices=0,
                max_slices=1,
                accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
                accelerator_variant="v5p-8",
            ),
        )
        discovered = [make_mock_slice_handle("slice-0")]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(config, platform)
        group.reconcile()

        assert group.can_accept_demand() is False

    def test_quota_exceeded_blocks_demand_until_timeout(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Quota exceeded state auto-expires after timeout."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        platform = make_mock_platform()
        platform.create_slice.side_effect = QuotaExhaustedError("TPU quota exhausted")

        group = ScalingGroup(unbounded_config, platform, quota_timeout=Duration.from_ms(60_000))

        ts = Timestamp.from_ms(1000)
        group.begin_scale_up()
        with pytest.raises(QuotaExhaustedError):
            group.scale_up(timestamp=ts)
        group.cancel_scale_up()
        group.record_quota_exceeded("quota exceeded", ts)

        # Before timeout: QUOTA_EXCEEDED
        assert not group.can_accept_demand(timestamp=Timestamp.from_ms(30_000))
        state = group.availability(timestamp=Timestamp.from_ms(30_000))
        assert state.status == GroupAvailability.QUOTA_EXCEEDED

        # After timeout (1000 + 60_000 = 61_000)
        assert group.can_accept_demand(timestamp=Timestamp.from_ms(70_000))

    def test_successful_scale_up_clears_quota_state(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Successful scale-up via complete_scale_up clears any quota exceeded state."""

        platform = make_mock_platform()
        platform.create_slice.side_effect = [
            QuotaExhaustedError("TPU quota exhausted"),
            make_mock_slice_handle("slice-1"),
        ]

        group = ScalingGroup(unbounded_config, platform, quota_timeout=Duration.from_ms(300_000))

        ts1 = Timestamp.from_ms(1000)
        group.begin_scale_up()
        with pytest.raises(QuotaExhaustedError):
            group.scale_up(timestamp=ts1)
        group.cancel_scale_up()
        group.record_quota_exceeded("quota exceeded", ts1)
        assert not group.can_accept_demand(timestamp=Timestamp.from_ms(2000))

        # Second attempt succeeds via complete_scale_up, which clears quota state
        ts2 = Timestamp.from_ms(3000)
        group.begin_scale_up()
        handle = group.scale_up(timestamp=ts2)
        group.complete_scale_up(handle, ts2)
        assert group.can_accept_demand(timestamp=Timestamp.from_ms(4000))

    def test_quota_exceeded_takes_precedence_over_backoff(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Quota exceeded has higher precedence than backoff."""
        from iris.cluster.controller.scaling_group import GroupAvailability

        platform = make_mock_platform()
        platform.create_slice.side_effect = QuotaExhaustedError("Quota exhausted")

        group = ScalingGroup(unbounded_config, platform, quota_timeout=Duration.from_ms(60_000))

        ts = Timestamp.from_ms(1000)
        # Record a failure to trigger backoff
        group.record_failure(timestamp=ts)

        # Then trigger quota exceeded via failed scale-up
        group.begin_scale_up()
        with pytest.raises(QuotaExhaustedError):
            group.scale_up(timestamp=ts)
        group.cancel_scale_up()
        group.record_quota_exceeded("quota exceeded", ts)

        # Availability should report QUOTA_EXCEEDED, not BACKOFF
        state = group.availability(timestamp=Timestamp.from_ms(2000))
        assert state.status == GroupAvailability.QUOTA_EXCEEDED

    def test_matches_device_requirement_filters_by_type_and_variant(
        self, scale_group_config: config_pb2.ScaleGroupConfig
    ):
        """matches_device_requirement filters groups by device type and variant."""
        from iris.cluster.types import DeviceType

        platform = make_mock_platform()
        group = ScalingGroup(scale_group_config, platform)  # TPU with accelerator_variant="v5p-8"

        # CPU matches any group
        assert group.matches_device_requirement(DeviceType.CPU, None)

        # TPU with matching variant
        assert group.matches_device_requirement(DeviceType.TPU, "v5p-8")
        assert group.matches_device_requirement(DeviceType.TPU, None)  # None = any TPU
        assert not group.matches_device_requirement(DeviceType.TPU, "v5litepod-4")

        # GPU doesn't match TPU group
        assert not group.matches_device_requirement(DeviceType.GPU, None)


class TestScalingGroupFailedSliceCleanup:
    """Tests for cleanup_failed_slices() behavior."""

    def test_cleanup_failed_slices_terminates_all_failed(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """cleanup_failed_slices() terminates all failed slices."""
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True),
            make_mock_slice_handle("slice-002", any_failed=True),
            make_mock_slice_handle("slice-003", any_failed=True),
        ]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform)
        group.reconcile()
        group.refresh_all_slices()

        cleaned = group.cleanup_failed_slices(timestamp=Timestamp.from_ms(1000))

        assert len(cleaned) == 2
        assert group.slice_count() == 1
        assert group.get_slice("slice-001") is not None
        assert group.get_slice("slice-002") is None
        assert group.get_slice("slice-003") is None

    def test_cleanup_failed_slices_triggers_backoff_once(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """cleanup_failed_slices() triggers backoff exactly once regardless of slice count."""
        discovered = [
            make_mock_slice_handle("slice-001", any_failed=True),
            make_mock_slice_handle("slice-002", any_failed=True),
            make_mock_slice_handle("slice-003", any_failed=True),
        ]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, backoff_initial=Duration.from_seconds(5.0))
        group.reconcile()
        group.refresh_all_slices()

        group.cleanup_failed_slices(timestamp=Timestamp.from_ms(1000))

        # Only one failure recorded despite cleaning up 3 slices
        assert group.consecutive_failures == 1
        assert group._backoff_until is not None
        assert group._backoff_until.as_timestamp().epoch_ms() == 6000

    def test_cleanup_failed_slices_returns_empty_when_no_failures(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """cleanup_failed_slices() returns empty list when no slices are failed."""
        discovered = [
            make_mock_slice_handle("slice-001", all_ready=True),
            make_mock_slice_handle("slice-002", all_ready=True),
        ]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform)
        group.reconcile()

        cleaned = group.cleanup_failed_slices(timestamp=Timestamp.from_ms(1000))

        assert len(cleaned) == 0
        assert group.slice_count() == 2
        assert group.consecutive_failures == 0

    def test_cleanup_failed_slices_cleans_idle_tracking(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """cleanup_failed_slices() removes slices from idle tracking."""
        discovered = [make_mock_slice_handle("slice-001", any_failed=True)]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform)
        group.reconcile()
        group.refresh_all_slices()

        handle = group.get_slice("slice-001")
        vm_address = _get_vm_address(handle)

        vm_status_map = {
            vm_address: VmWorkerStatus(vm_address=vm_address, running_task_ids=frozenset({"task-1"})),
        }
        group.update_slice_activity(vm_status_map, Timestamp.from_ms(1000))

        group.cleanup_failed_slices(timestamp=Timestamp.from_ms(2000))

        assert group.get_slice("slice-001") is None

    def test_cleanup_failed_slices_ignores_cooldown(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """cleanup_failed_slices() does not respect scale_down_cooldown."""
        discovered = [make_mock_slice_handle("slice-001", any_failed=True)]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, scale_down_cooldown=Duration.from_ms(60_000))
        group.reconcile()
        group.refresh_all_slices()

        # Set last scale-down to trigger cooldown for normal scale_down
        group._last_scale_down = Timestamp.from_ms(1000)

        # Despite cooldown, cleanup_failed_slices() should still work
        cleaned = group.cleanup_failed_slices(timestamp=Timestamp.from_ms(2000))

        assert len(cleaned) == 1
        assert group.slice_count() == 0


class TestVerifySliceIdle:
    """Tests for _verify_slice_idle behavior with unknown workers."""

    def test_unknown_workers_do_not_count_as_idle(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """A slice with no workers in the status map is NOT idle (we don't know yet)."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)
        ts = Timestamp.from_ms(1000000)
        handle = _tracked_scale_up(group, timestamp=ts)
        state = _get_slice_state(group, handle)

        # Empty status map -- no workers known
        assert not group._verify_slice_idle(state, {})

    def test_known_idle_workers_are_idle(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """A slice where all known workers are idle IS idle."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)
        ts = Timestamp.from_ms(1000000)
        handle = _tracked_scale_up(group, timestamp=ts)
        group.refresh_all_slices()
        state = _get_slice_state(group, handle)

        # Get VM addresses from mock
        vm_address = _get_vm_address(handle)
        status_map = {vm_address: VmWorkerStatus(vm_address="", running_task_ids=frozenset())}
        assert group._verify_slice_idle(state, status_map)

    def test_known_busy_worker_blocks_idle(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """A slice with a known busy worker is NOT idle."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)
        ts = Timestamp.from_ms(1000000)
        handle = _tracked_scale_up(group, timestamp=ts)
        state = _get_slice_state(group, handle)

        vm_address = _get_vm_address(handle)
        status_map = {vm_address: VmWorkerStatus(vm_address="", running_task_ids=frozenset({"task-1"}))}
        assert not group._verify_slice_idle(state, status_map)


class TestSliceLivenessTimeout:
    """Tests for slice liveness tracking and reaping."""

    def test_new_slice_gets_startup_grace(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """complete_scale_up sets a liveness deadline at startup_grace from creation time."""
        platform = make_mock_platform()
        grace = Duration.from_seconds(10)
        group = ScalingGroup(unbounded_config, platform, startup_grace_period=grace)

        ts = Timestamp.from_ms(1000000)
        handle = _tracked_scale_up(group, timestamp=ts)

        state = group._slices.get(handle.slice_id)
        assert state is not None
        assert state.liveness_deadline is not None
        assert not state.liveness_deadline.expired(now=ts)
        assert not state.liveness_deadline.expired(now=Timestamp.from_ms(1009000))
        assert state.liveness_deadline.expired(now=Timestamp.from_ms(1011000))

    def test_liveness_extended_by_heartbeat(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """update_slice_liveness extends deadline when workers appear in status map."""
        platform = make_mock_platform()
        startup = Duration.from_seconds(10)
        heartbeat = Duration.from_seconds(5)
        group = ScalingGroup(
            unbounded_config,
            platform,
            startup_grace_period=startup,
            heartbeat_grace_period=heartbeat,
        )

        ts = Timestamp.from_ms(1000000)
        handle = _tracked_scale_up(group, timestamp=ts)
        group.refresh_all_slices()
        vm_address = _get_vm_address(handle)

        # At t=8s, worker shows up -- extends deadline by heartbeat_grace (5s from now)
        t_heartbeat = Timestamp.from_ms(1008000)
        group.update_slice_liveness(
            {vm_address: VmWorkerStatus(vm_address="", running_task_ids=frozenset())}, t_heartbeat
        )

        deadline = group._slices[handle.slice_id].liveness_deadline
        # Should expire at t_heartbeat + 5s = 1013000, not original 1010000
        assert not deadline.expired(now=Timestamp.from_ms(1012000))
        assert deadline.expired(now=Timestamp.from_ms(1014000))

    def test_dead_slice_reaped(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Slice that misses its liveness deadline is terminated."""
        platform = make_mock_platform()
        grace = Duration.from_seconds(10)
        group = ScalingGroup(unbounded_config, platform, startup_grace_period=grace)

        ts = Timestamp.from_ms(1000000)
        handle = _tracked_scale_up(group, timestamp=ts)
        assert group.slice_count() == 1

        # No heartbeats arrive. After grace period, cleanup should reap it.
        dead = group.cleanup_dead_slices(Timestamp.from_ms(1011000))
        assert len(dead) == 1
        assert dead[0].slice_id == handle.slice_id
        assert group.slice_count() == 0

    def test_alive_slice_not_reaped(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Slice within its grace period is not reaped."""
        platform = make_mock_platform()
        grace = Duration.from_seconds(10)
        group = ScalingGroup(unbounded_config, platform, startup_grace_period=grace)

        ts = Timestamp.from_ms(1000000)
        _tracked_scale_up(group, timestamp=ts)

        dead = group.cleanup_dead_slices(Timestamp.from_ms(1005000))
        assert len(dead) == 0
        assert group.slice_count() == 1

    def test_reconciled_slice_gets_liveness_deadline(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Slices adopted during reconcile() get a liveness deadline and can be reaped."""
        discovered = [make_mock_slice_handle("slice-001", all_ready=True)]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        heartbeat = Duration.from_seconds(5)
        group = ScalingGroup(unbounded_config, platform, heartbeat_grace_period=heartbeat)
        group.reconcile()

        state = group._slices["slice-001"]
        assert state.liveness_deadline is not None

    def test_reconciled_slice_reaped_when_no_heartbeat(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """A reconciled slice with no heartbeats is reaped after the heartbeat grace period."""
        discovered = [make_mock_slice_handle("slice-001", all_ready=True)]
        platform = make_mock_platform(slice_handles_to_discover=discovered)
        heartbeat = Duration.from_seconds(5)
        group = ScalingGroup(unbounded_config, platform, heartbeat_grace_period=heartbeat)
        group.reconcile()

        # Advance well past the heartbeat grace period
        far_future = Timestamp.from_ms(Timestamp.now().epoch_ms() + 60_000)
        dead = group.cleanup_dead_slices(far_future)
        assert len(dead) == 1
        assert dead[0].slice_id == "slice-001"
        assert group.slice_count() == 0


class TestZonesFromConfig:
    """Tests for _zones_from_config fail-fast behavior."""

    def test_gcp_with_zone_returns_list(self):
        config = config_pb2.ScaleGroupConfig(name="g")
        config.slice_template.gcp.zone = "us-central1-a"
        assert _zones_from_config(config) == ["us-central1-a"]

    def test_gcp_with_no_zone_raises(self):
        config = config_pb2.ScaleGroupConfig(name="g")
        config.slice_template.gcp.runtime_version = "v2-alpha-tpuv5"
        with pytest.raises(ValueError, match="no zone configured"):
            _zones_from_config(config)

    def test_non_gcp_returns_empty(self):
        config = config_pb2.ScaleGroupConfig(name="g")
        assert _zones_from_config(config) == []


class TestCanScaleUpQuotaExhausted:
    """can_scale_up must respect quota_exceeded state."""

    def test_cannot_scale_up_during_quota_exhaustion(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """can_scale_up() returns False while quota_exceeded deadline is active."""

        platform = MagicMock()
        platform.list_slices.return_value = []
        platform.create_slice.side_effect = QuotaExhaustedError("no quota")

        group = ScalingGroup(unbounded_config, platform, quota_timeout=Duration.from_ms(5000))

        ts = Timestamp.from_ms(1000000)
        group.begin_scale_up()
        with pytest.raises(QuotaExhaustedError):
            group.scale_up(timestamp=ts)
        group.cancel_scale_up()
        group.record_quota_exceeded("quota exceeded", ts)

        # During quota exhaustion window
        assert not group.can_scale_up(timestamp=Timestamp.from_ms(1003000))
        # After quota timeout expires
        assert group.can_scale_up(timestamp=Timestamp.from_ms(1006000))


class TestPrepareSliceConfigPreemptible:
    """prepare_slice_config propagates preemptible from the template."""

    def test_preemptible_set_on_slice_template_is_preserved(self):
        """preemptible=True on slice_template is preserved through prepare_slice_config."""
        from iris.cluster.controller.scaling_group import prepare_slice_config

        parent = config_pb2.ScaleGroupConfig(
            name="test-group",
            accelerator_variant="v5litepod-16",
        )
        parent.slice_template.preemptible = True
        parent.slice_template.gcp.zone = "us-central1-a"
        parent.slice_template.gcp.runtime_version = "v2-alpha-tpuv5"

        result = prepare_slice_config(parent.slice_template, parent, "iris")
        assert result.preemptible is True

    def test_preemptible_false_by_default(self):
        """preemptible defaults to False when not set on template."""
        from iris.cluster.controller.scaling_group import prepare_slice_config

        parent = config_pb2.ScaleGroupConfig(
            name="test-group",
            accelerator_variant="v5litepod-16",
        )
        parent.slice_template.gcp.zone = "us-central1-a"
        parent.slice_template.gcp.runtime_version = "v2-alpha-tpuv5"

        result = prepare_slice_config(parent.slice_template, parent, "iris")
        assert result.preemptible is False


def _make_vm_handle(vm_id: str, cloud_state: CloudVmState, address: str = "10.0.0.1") -> MagicMock:
    handle = MagicMock()
    handle.vm_id = vm_id
    handle.internal_address = address
    handle.status.return_value = VmStatus(state=cloud_state)
    return handle


def _make_slice_handle(
    slice_id: str,
    slice_state: CloudSliceState,
    vm_handles: list[MagicMock],
) -> MagicMock:
    handle = MagicMock()
    handle.slice_id = slice_id
    handle.describe.return_value = SliceStatus(state=slice_state, vm_count=len(vm_handles), vms=vm_handles)
    return handle


class TestSliceHandleStatusReadinessGating:
    """VMs in a CREATING or REPAIRING slice must not appear READY."""

    @pytest.mark.parametrize("slice_state", [CloudSliceState.CREATING, CloudSliceState.REPAIRING])
    def test_running_vms_clamped_to_booting_when_slice_not_ready(self, slice_state: CloudSliceState):
        """A VM reporting RUNNING should show as BOOTING when the slice is still being set up."""
        vm = _make_vm_handle("vm-0", CloudVmState.RUNNING)
        handle = _make_slice_handle("slice-1", slice_state, [vm])

        status = slice_handle_status(handle)

        assert len(status.vms) == 1
        assert status.vms[0].state == vm_pb2.VM_STATE_BOOTING
        assert not status.all_ready

    def test_running_vms_report_ready_when_slice_ready(self):
        """A VM reporting RUNNING should show as READY when the slice itself is READY."""
        vm = _make_vm_handle("vm-0", CloudVmState.RUNNING)
        handle = _make_slice_handle("slice-1", CloudSliceState.READY, [vm])

        status = slice_handle_status(handle)

        assert len(status.vms) == 1
        assert status.vms[0].state == vm_pb2.VM_STATE_READY
        assert status.all_ready

    def test_non_running_vm_states_unaffected_by_clamping(self):
        """STOPPED/TERMINATED VMs keep their mapped state regardless of slice state."""
        vms = [
            _make_vm_handle("vm-0", CloudVmState.STOPPED),
            _make_vm_handle("vm-1", CloudVmState.TERMINATED, address="10.0.0.2"),
        ]
        handle = _make_slice_handle("slice-1", CloudSliceState.CREATING, vms)

        status = slice_handle_status(handle)

        assert status.vms[0].state == vm_pb2.VM_STATE_FAILED
        assert status.vms[1].state == vm_pb2.VM_STATE_TERMINATED

    def test_mixed_vms_in_creating_slice(self):
        """In a CREATING slice, only RUNNING VMs get clamped; others keep their state."""
        vms = [
            _make_vm_handle("vm-0", CloudVmState.RUNNING, address="10.0.0.1"),
            _make_vm_handle("vm-1", CloudVmState.UNKNOWN, address="10.0.0.2"),
        ]
        handle = _make_slice_handle("slice-1", CloudSliceState.CREATING, vms)

        status = slice_handle_status(handle)

        assert status.vms[0].state == vm_pb2.VM_STATE_BOOTING  # clamped
        assert status.vms[1].state == vm_pb2.VM_STATE_BOOTING  # already BOOTING from UNKNOWN
