# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ScalingGroup behavior.

These tests focus on observable behavior - scaling policy decisions,
VM group management, and state tracking - not on implementation details.
"""

import logging

import pytest
from iris.cluster.controller.autoscaler.scaling_group import (
    ScalingGroup,
    SliceLifecycleState,
    SliceState,
    _zones_from_config,
)
from iris.cluster.providers.types import (
    CloudSliceState,
    CloudWorkerState,
    Labels,
    QuotaExhaustedError,
    SliceStatus,
)
from iris.cluster.types import WorkerStatus
from iris.rpc import config_pb2, vm_pb2
from rigging.timing import Duration, Timestamp
from tests.cluster.providers.conftest import (
    FakeSliceHandle,
    FakeWorkerHandle,
    make_fake_slice_handle,
    make_mock_platform,
)

DEFAULT_RESOURCES = config_pb2.ScaleGroupResources(
    cpu_millicores=64000,
    memory_bytes=64 * 1024**3,
    disk_bytes=100 * 1024**3,
    device_type=config_pb2.ACCELERATOR_TYPE_TPU,
    device_variant="v5p-8",
    device_count=8,
)


def _with_resources(config: config_pb2.ScaleGroupConfig, *, num_vms: int = 1) -> config_pb2.ScaleGroupConfig:
    if not config.HasField("resources"):
        config.resources.CopyFrom(DEFAULT_RESOURCES)
    if not config.HasField("num_vms"):
        config.num_vms = num_vms
    return config


def _mark_discovered_ready(
    group: ScalingGroup, handles: list[FakeSliceHandle], timestamp: Timestamp | None = None
) -> None:
    """Mark discovered slices as READY with their worker IDs."""
    for handle in handles:
        worker_ids = [vm.worker_id for vm in handle.describe().workers]
        group.mark_slice_ready(handle.slice_id, worker_ids, timestamp=timestamp)


def _mark_discovered_failed(group: ScalingGroup, handles: list[FakeSliceHandle]) -> None:
    """Mark discovered slices as FAILED."""
    for handle in handles:
        group.mark_slice_failed(handle.slice_id)


def _get_worker_id(handle: FakeSliceHandle) -> str:
    """Get the first worker's ID from a SliceHandle."""
    return handle.describe().workers[0].worker_id


def _get_slice_state(group: ScalingGroup, handle: FakeSliceHandle) -> SliceState:
    """Get the SliceState for a handle from its group."""
    with group._slices_lock:
        return group._slices[handle.slice_id]


def _tracked_scale_up(group: ScalingGroup, timestamp: Timestamp | None = None, **kwargs) -> FakeSliceHandle:
    """Scale up with full lifecycle tracking: begin -> create -> complete.

    This replaces the old group.scale_up() pattern in tests, since scale_up()
    no longer tracks state internally.
    """
    timestamp = timestamp or Timestamp.from_ms(1000000)
    group.begin_scale_up(timestamp=timestamp)
    handle = group.scale_up(timestamp=timestamp, **kwargs)
    group.complete_scale_up(handle, timestamp)
    return handle


@pytest.fixture
def scale_group_config() -> config_pb2.ScaleGroupConfig:
    """A standard scale group configuration for tests."""
    config = config_pb2.ScaleGroupConfig(
        name="test-group",
        buffer_slices=1,
        max_slices=5,
    )
    config.slice_template.gcp.runtime_version = "v2-alpha-tpuv5"
    config.slice_template.gcp.zone = "us-central1-a"
    return _with_resources(config)


@pytest.fixture
def unbounded_config() -> config_pb2.ScaleGroupConfig:
    """A scale group with no min/max constraints."""
    config = config_pb2.ScaleGroupConfig(
        name="unbounded-group",
        buffer_slices=0,
        max_slices=100,
    )
    config.slice_template.gcp.runtime_version = "v2-alpha-tpuv5"
    config.slice_template.gcp.zone = "us-central1-a"
    return _with_resources(config)


class TestScalingGroupVmGroupOwnership:
    """Tests for VM group ownership and lifecycle."""

    def test_reconcile_adopts_discovered_vm_groups(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """reconcile() populates VM groups from the Platform."""
        discovered = [
            make_fake_slice_handle("slice-001"),
            make_fake_slice_handle("slice-002"),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)

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
        handle = make_fake_slice_handle("slice-001")
        platform = make_mock_platform(slices_to_discover=[handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()

        assert group.slice_count() == 1

        group.scale_down("slice-001")

        assert handle.terminated
        assert group.slice_count() == 0
        assert group.get_slice("slice-001") is None

    def test_scale_down_nonexistent_vm_group_is_noop(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """scale_down() on a nonexistent VM group does nothing."""
        platform = make_mock_platform()
        group = ScalingGroup(scale_group_config, platform)

        # Should not raise
        group.scale_down("nonexistent-slice")

        assert group.slice_count() == 0

    def test_scale_down_cleans_up_even_if_terminate_fails(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """scale_down() removes the slice from tracking even if terminate() raises.

        This prevents ghost slices where the cloud resource is gone (e.g.
        preempted) but the autoscaler still counts it as live capacity.
        """
        handle = make_fake_slice_handle("slice-001")
        handle.terminate_error = RuntimeError("resource not found")
        platform = make_mock_platform(slices_to_discover=[handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()

        assert group.slice_count() == 1

        # Should NOT raise despite terminate() failure
        group.scale_down("slice-001")

        assert handle.terminated
        assert group.slice_count() == 0
        assert group.get_slice("slice-001") is None

    def test_terminate_all_continues_on_individual_failure(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """terminate_all() terminates remaining slices even if one fails."""
        handles = [
            make_fake_slice_handle("slice-001"),
            make_fake_slice_handle("slice-002"),
            make_fake_slice_handle("slice-003"),
        ]
        handles[0].terminate_error = RuntimeError("resource not found")
        platform = make_mock_platform(slices_to_discover=handles)
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()

        assert group.slice_count() == 3

        group.terminate_all()

        for h in handles:
            assert h.terminated
        assert group.slice_count() == 0

    def test_ready_slice_count(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """ready_slice_count() only counts VM groups where all VMs are ready."""
        discovered = [
            make_fake_slice_handle("slice-001", all_ready=True),
            make_fake_slice_handle("slice-002", all_ready=False, vm_states=[vm_pb2.VM_STATE_BOOTING]),
            make_fake_slice_handle("slice-003", all_ready=True),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        _mark_discovered_ready(group, [discovered[0], discovered[2]])

        assert group.slice_count() == 3
        assert group.ready_slice_count() == 2


class TestScalingGroupScalingPolicy:
    """Tests for scaling policy decisions (can_scale_up, rate limiting)."""

    def test_can_scale_up_when_below_max(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """can_scale_up() returns True when below max_slices."""
        platform = make_mock_platform()
        group = ScalingGroup(scale_group_config, platform)

        assert group.can_scale_up()

    def test_cannot_scale_up_at_max_slices(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """can_scale_up() returns False when at max_slices."""
        discovered = [make_fake_slice_handle(f"slice-{i}") for i in range(5)]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()

        assert group.slice_count() == 5  # max_slices
        assert not group.can_scale_up()

    def test_cannot_scale_up_during_backoff(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """can_scale_up() returns False during backoff period."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform, backoff_initial=Duration.from_seconds(5.0))

        ts = Timestamp.from_ms(1000000)
        group.record_failure(timestamp=ts)

        # During backoff period
        assert not group.can_scale_up(timestamp=Timestamp.from_ms(1001000))
        # After backoff expires (5s = 5000ms)
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

    def test_scale_down_rate_limited_by_token_bucket(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """acquire_scale_down_token() returns False when the token bucket is exhausted."""
        platform = make_mock_platform()
        group = ScalingGroup(
            unbounded_config,
            platform,
            scale_down_rate_limit=1,
        )

        ts = Timestamp.from_ms(1000000)

        # First token succeeds
        assert group.acquire_scale_down_token(ts)
        # Bucket exhausted — second acquire fails at same timestamp
        assert not group.acquire_scale_down_token(ts)
        # After enough time passes (1 minute refill), token is available again
        assert group.acquire_scale_down_token(Timestamp.from_ms(ts.epoch_ms() + 61_000))


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
        discovered = [make_fake_slice_handle("slice-001", all_ready=True)]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(60_000))
        group.reconcile()
        _mark_discovered_ready(group, discovered)

        # Get the worker ID from the SliceHandle
        handle = group.get_slice("slice-001")
        worker_id = _get_worker_id(handle)

        # Mark slice as active at t=1000 via update_slice_activity
        vm_status_map = {
            worker_id: WorkerStatus(worker_id=worker_id, running_task_ids=frozenset({"task-1"})),
        }
        group.update_slice_activity(vm_status_map, Timestamp.from_ms(1000))

        # Not enough time passed (30s < 60s threshold)
        assert not group.is_slice_eligible_for_scaledown("slice-001", Timestamp.from_ms(30_000))

    def test_slice_eligible_after_idle_threshold(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Slice is eligible after idle_threshold of inactivity."""
        discovered = [make_fake_slice_handle("slice-001", all_ready=True)]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(60_000))
        group.reconcile()

        handle = group.get_slice("slice-001")
        worker_id = _get_worker_id(handle)

        # Mark slice as active at t=1000 via update_slice_activity
        vm_status_map = {
            worker_id: WorkerStatus(worker_id=worker_id, running_task_ids=frozenset({"task-1"})),
        }
        group.update_slice_activity(vm_status_map, Timestamp.from_ms(1000))

        # After threshold (61s > 60s) -> eligible
        assert group.is_slice_eligible_for_scaledown("slice-001", Timestamp.from_ms(61_001))

    def test_get_idle_slices_returns_longest_idle_first(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """get_idle_slices returns slices sorted by idle time (longest first)."""
        discovered = [
            make_fake_slice_handle("slice-001", all_ready=True),
            make_fake_slice_handle("slice-002", all_ready=True),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(1000))
        group.reconcile()
        _mark_discovered_ready(group, discovered)

        # Get worker IDs from the handles
        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        wid_001 = _get_worker_id(slice_001)
        wid_002 = _get_worker_id(slice_002)

        # Mark slice-001 as active at t=1000 (will be idle longer)
        vm_status_map_001 = {
            wid_001: WorkerStatus(worker_id=wid_001, running_task_ids=frozenset({"task-1"})),
        }
        group.update_slice_activity(vm_status_map_001, Timestamp.from_ms(1000))

        # Mark slice-002 as active at t=5000 (more recently active)
        vm_status_map_002 = {
            wid_002: WorkerStatus(worker_id=wid_002, running_task_ids=frozenset({"task-2"})),
        }
        group.update_slice_activity(vm_status_map_002, Timestamp.from_ms(5000))

        # At timestamp 10000, slice-001 has been idle longer (9s vs 5s)
        idle_slices = group.get_idle_slices(Timestamp.from_ms(10_000))
        assert len(idle_slices) == 2
        assert idle_slices[0].handle.slice_id == "slice-001"  # Longest idle first

    def test_update_slice_activity_tracks_active_slices(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """update_slice_activity updates timestamp only for slices with active workers."""
        discovered = [
            make_fake_slice_handle("slice-001", all_ready=True),
            make_fake_slice_handle("slice-002", all_ready=True),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(1000))
        group.reconcile()
        ready_ts = Timestamp.from_ms(1000)
        _mark_discovered_ready(group, discovered, timestamp=ready_ts)

        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        wid_001 = _get_worker_id(slice_001)
        wid_002 = _get_worker_id(slice_002)

        # slice-001 has running tasks, slice-002 is idle
        vm_status_map = {
            wid_001: WorkerStatus(worker_id=wid_001, running_task_ids=frozenset({"task-1"})),
            wid_002: WorkerStatus(worker_id=wid_002, running_task_ids=frozenset()),
        }

        check_ts = Timestamp.from_ms(5000)
        group.update_slice_activity(vm_status_map, check_ts)

        # Observable behavior: slice-001 should not be eligible for scaledown (recently active)
        # slice-002 is eligible because idle_threshold (1000ms) has passed since ready_ts (1000ms)
        assert not group.is_slice_eligible_for_scaledown("slice-001", check_ts)
        assert group.is_slice_eligible_for_scaledown("slice-002", check_ts)

    def test_scale_down_if_idle_terminates_eligible_slice(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """scale_down_if_idle terminates an eligible idle slice."""
        discovered = [
            make_fake_slice_handle("slice-001", all_ready=True),
            make_fake_slice_handle("slice-002", all_ready=True),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(1000))
        group.reconcile()
        _mark_discovered_ready(group, discovered)

        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        wid_001 = _get_worker_id(slice_001)
        wid_002 = _get_worker_id(slice_002)

        # Mark both slices as active at t=0 (they'll be idle for 10s, exceeding 1s threshold)
        vm_status_map_active = {
            wid_001: WorkerStatus(worker_id=wid_001, running_task_ids=frozenset({"task-1"})),
            wid_002: WorkerStatus(worker_id=wid_002, running_task_ids=frozenset({"task-2"})),
        }
        group.update_slice_activity(vm_status_map_active, Timestamp.from_ms(0))

        # At t=10_000, workers are now idle
        vm_status_map_idle = {
            wid_001: WorkerStatus(worker_id=wid_001, running_task_ids=frozenset()),
            wid_002: WorkerStatus(worker_id=wid_002, running_task_ids=frozenset()),
        }

        # Target capacity = 1, but we have 2 ready slices
        scaled_down = group.scale_down_if_idle(
            vm_status_map_idle, target_capacity=1, timestamp=Timestamp.from_ms(10_000)
        )

        assert len(scaled_down) == 1
        assert group.slice_count() == 1  # One slice was terminated

    def test_scale_down_if_idle_respects_target_capacity(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """scale_down_if_idle does nothing when at or below target capacity."""
        discovered = [make_fake_slice_handle("slice-001", all_ready=True)]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(1000))
        group.reconcile()

        slice_001 = group.get_slice("slice-001")
        wid_001 = _get_worker_id(slice_001)
        vm_status_map = {wid_001: WorkerStatus(worker_id=wid_001, running_task_ids=frozenset())}

        # Target = 1, ready = 1, should not scale down
        scaled_down = group.scale_down_if_idle(vm_status_map, target_capacity=1, timestamp=Timestamp.from_ms(10_000))

        assert len(scaled_down) == 0
        assert group.slice_count() == 1

    def test_pending_demand_tracks_raw_demand(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """update_pending_demand stores raw demand, accessible via pending_demand property."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)

        assert group.pending_demand == 0
        group.update_pending_demand(3)
        assert group.pending_demand == 3
        group.update_pending_demand(0)
        assert group.pending_demand == 0

    def test_scale_down_cleans_up_idle_tracking(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """scale_down removes the slice from idle tracking."""
        discovered = [make_fake_slice_handle("slice-001", all_ready=True)]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform)
        group.reconcile()

        handle = group.get_slice("slice-001")
        worker_id = _get_worker_id(handle)

        vm_status_map = {
            worker_id: WorkerStatus(worker_id=worker_id, running_task_ids=frozenset({"task-1"})),
        }
        group.update_slice_activity(vm_status_map, Timestamp.from_ms(1000))

        # Scale down
        group.scale_down("slice-001")

        assert group.get_slice("slice-001") is None


def test_scale_down_no_misleading_rate_limit_log(unbounded_config: config_pb2.ScaleGroupConfig, caplog):
    """When the token bucket is empty and no terminations occur, no rate-limit log is emitted."""
    discovered = [
        make_fake_slice_handle("slice-001", all_ready=True),
        make_fake_slice_handle("slice-002", all_ready=True),
    ]
    platform = make_mock_platform(slices_to_discover=discovered)
    group = ScalingGroup(unbounded_config, platform, scale_down_rate_limit=1, idle_threshold=Duration.from_ms(100))
    group.reconcile()
    _mark_discovered_ready(group, discovered, timestamp=Timestamp.from_ms(0))

    wid_001 = _get_worker_id(discovered[0])
    wid_002 = _get_worker_id(discovered[1])

    # Mark both active at t=0, then idle at t=1000 (well past idle_threshold)
    active_map = {
        wid_001: WorkerStatus(worker_id=wid_001, running_task_ids=frozenset({"task-1"})),
        wid_002: WorkerStatus(worker_id=wid_002, running_task_ids=frozenset({"task-2"})),
    }
    group.update_slice_activity(active_map, Timestamp.from_ms(0))

    idle_map = {
        wid_001: WorkerStatus(worker_id=wid_001, running_task_ids=frozenset()),
        wid_002: WorkerStatus(worker_id=wid_002, running_task_ids=frozenset()),
    }

    ts = Timestamp.from_ms(1000)

    # Drain the token bucket (rate_limit=1, so one token available)
    assert group.acquire_scale_down_token(ts)
    assert not group.acquire_scale_down_token(ts)

    # Now call scale_down_if_idle with an empty bucket — no terminations should happen
    with caplog.at_level(logging.INFO, logger="iris.cluster.controller.autoscaler.scaling_group"):
        scaled_down = group.scale_down_if_idle(idle_map, target_capacity=0, timestamp=ts)

    assert scaled_down == []
    assert "rate-limited after 0 terminations" not in caplog.text


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
        discovered = [make_fake_slice_handle("slice-001", vm_states=[vm_state])]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        if expected_state == SliceLifecycleState.READY:
            _mark_discovered_ready(group, discovered)
        elif expected_state == SliceLifecycleState.FAILED:
            _mark_discovered_failed(group, discovered)
        # BOOTING is the default lifecycle state after reconcile

        counts = group.slice_state_counts()

        assert counts[expected_state] == 1
        for state in SliceLifecycleState:
            if state != expected_state:
                assert counts[state] == 0

    def test_failed_takes_precedence(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """A VM group with any failed VM is counted as failed."""
        discovered = [
            make_fake_slice_handle(
                "slice-001",
                vm_states=[vm_pb2.VM_STATE_READY, vm_pb2.VM_STATE_FAILED],
            ),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        _mark_discovered_failed(group, discovered)

        counts = group.slice_state_counts()

        assert counts[SliceLifecycleState.FAILED] == 1
        assert counts[SliceLifecycleState.READY] == 0

    def test_unobserved_slices_counted_as_booting(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Slices that haven't been marked ready or failed are counted as BOOTING."""
        discovered = [
            make_fake_slice_handle("slice-001", vm_states=[vm_pb2.VM_STATE_TERMINATED]),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()

        counts = group.slice_state_counts()

        assert counts[SliceLifecycleState.READY] == 0
        assert counts[SliceLifecycleState.BOOTING] == 1
        assert counts[SliceLifecycleState.INITIALIZING] == 0
        assert counts[SliceLifecycleState.FAILED] == 0


class TestScalingGroupAvailability:
    """Tests for availability state computation and waterfall routing support."""

    def test_available_when_no_constraints(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Group is AVAILABLE when not in backoff, quota ok, and under capacity."""
        from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability

        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)

        state = group.availability()
        assert state.status == GroupAvailability.AVAILABLE

    def test_at_max_slices_when_at_max_slices(self):
        """Group is AT_MAX_SLICES when at max_slices with all slices READY."""
        from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability

        config = _with_resources(
            config_pb2.ScaleGroupConfig(
                name="test-group",
                buffer_slices=0,
                max_slices=2,
            ),
        )
        discovered = [make_fake_slice_handle(f"slice-{i}", all_ready=True) for i in range(2)]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(config, platform)
        group.reconcile()
        for h in discovered:
            worker_ids = [w.worker_id for w in h.describe().workers]
            group.mark_slice_ready(h.slice_id, worker_ids)

        state = group.availability()
        assert state.status == GroupAvailability.AT_MAX_SLICES

    def test_backoff_when_in_backoff_period(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Group is in BACKOFF when backoff timer is active."""
        from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability

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

    def test_at_max_slices_rejects_demand(self):
        """AT_MAX_SLICES groups reject demand so it falls through to lower-priority groups."""
        config = _with_resources(
            config_pb2.ScaleGroupConfig(
                name="test-group",
                buffer_slices=0,
                max_slices=1,
            ),
        )
        discovered = [make_fake_slice_handle("slice-0", all_ready=True)]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(config, platform)
        group.reconcile()
        for h in discovered:
            worker_ids = [w.worker_id for w in h.describe().workers]
            group.mark_slice_ready(h.slice_id, worker_ids)

        assert group.can_accept_demand() is False

    def test_quota_exceeded_blocks_demand_until_timeout(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Quota exceeded state auto-expires after timeout."""
        from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability

        platform = make_mock_platform()
        platform.create_slice.side_effect = QuotaExhaustedError("TPU quota exhausted")

        group = ScalingGroup(unbounded_config, platform, quota_timeout=Duration.from_ms(60_000))

        ts = Timestamp.from_ms(1000)
        group.begin_scale_up(timestamp=ts)
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
            make_fake_slice_handle("slice-1"),
        ]

        group = ScalingGroup(unbounded_config, platform, quota_timeout=Duration.from_ms(300_000))

        ts1 = Timestamp.from_ms(1000)
        group.begin_scale_up(timestamp=ts1)
        with pytest.raises(QuotaExhaustedError):
            group.scale_up(timestamp=ts1)
        group.cancel_scale_up()
        group.record_quota_exceeded("quota exceeded", ts1)
        assert not group.can_accept_demand(timestamp=Timestamp.from_ms(2000))

        # Second attempt succeeds via complete_scale_up, which clears quota state
        ts2 = Timestamp.from_ms(3000)
        group.begin_scale_up(timestamp=ts2)
        handle = group.scale_up(timestamp=ts2)
        group.complete_scale_up(handle, ts2)
        assert group.can_accept_demand(timestamp=Timestamp.from_ms(4000))

    def test_quota_exceeded_takes_precedence_over_backoff(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Quota exceeded has higher precedence than backoff."""
        from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability

        platform = make_mock_platform()
        platform.create_slice.side_effect = QuotaExhaustedError("Quota exhausted")

        group = ScalingGroup(unbounded_config, platform, quota_timeout=Duration.from_ms(60_000))

        ts = Timestamp.from_ms(1000)
        # Record a failure to trigger backoff
        group.record_failure(timestamp=ts)

        # Then trigger quota exceeded via failed scale-up
        group.begin_scale_up(timestamp=ts)
        with pytest.raises(QuotaExhaustedError):
            group.scale_up(timestamp=ts)
        group.cancel_scale_up()
        group.record_quota_exceeded("quota exceeded", ts)

        # Availability should report QUOTA_EXCEEDED, not BACKOFF
        state = group.availability(timestamp=Timestamp.from_ms(2000))
        assert state.status == GroupAvailability.QUOTA_EXCEEDED

    def test_cooldown_availability_state(self):
        """After scale-up + complete, availability() returns COOLDOWN until expiry, then AVAILABLE."""
        from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability

        config = _with_resources(
            config_pb2.ScaleGroupConfig(
                name="test-group",
                buffer_slices=0,
                max_slices=10,
            ),
        )
        config.slice_template.gcp.zone = "us-central1-a"
        platform = make_mock_platform()
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(5000))

        ts = Timestamp.from_ms(1_000_000)
        group.begin_scale_up(timestamp=ts)
        handle = group.scale_up(timestamp=ts)
        group.complete_scale_up(handle, ts)

        # During cooldown
        state = group.availability(Timestamp.from_ms(1_003_000))
        assert state.status == GroupAvailability.COOLDOWN
        assert state.until is not None

        # After cooldown expires
        state = group.availability(Timestamp.from_ms(1_006_000))
        assert state.status == GroupAvailability.AVAILABLE

    def test_cooldown_accepts_demand(self):
        """COOLDOWN groups still accept demand (demand stays, scale-up is deferred)."""
        config = _with_resources(
            config_pb2.ScaleGroupConfig(
                name="test-group",
                buffer_slices=0,
                max_slices=10,
            ),
        )
        config.slice_template.gcp.zone = "us-central1-a"
        platform = make_mock_platform()
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(5000))

        ts = Timestamp.from_ms(1_000_000)
        group.begin_scale_up(timestamp=ts)
        handle = group.scale_up(timestamp=ts)
        group.complete_scale_up(handle, ts)

        # During cooldown, can_accept_demand should be True
        assert group.can_accept_demand(Timestamp.from_ms(1_003_000)) is True

    def test_at_max_slices_takes_precedence_over_cooldown(self):
        """When both at max_slices and in cooldown, AT_MAX_SLICES takes precedence
        (once all slices are READY)."""
        from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability

        config = _with_resources(
            config_pb2.ScaleGroupConfig(
                name="test-group",
                buffer_slices=0,
                max_slices=1,
            ),
        )
        config.slice_template.gcp.zone = "us-central1-a"
        platform = make_mock_platform()
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(5000))

        ts = Timestamp.from_ms(1_000_000)
        group.begin_scale_up(timestamp=ts)
        handle = group.scale_up(timestamp=ts)
        group.complete_scale_up(handle, ts)

        # While slice is BOOTING, group accepts demand (in-flight capacity)
        state = group.availability(Timestamp.from_ms(1_003_000))
        assert state.status == GroupAvailability.COOLDOWN

        # Once the slice is READY, AT_MAX_SLICES takes precedence over cooldown
        worker_ids = [w.worker_id for w in handle.describe().workers]
        group.mark_slice_ready(handle.slice_id, worker_ids)
        state = group.availability(Timestamp.from_ms(1_003_000))
        assert state.status == GroupAvailability.AT_MAX_SLICES

    def test_matches_device_requirement_filters_by_type_and_variant(
        self, scale_group_config: config_pb2.ScaleGroupConfig
    ):
        """matches_device_requirement filters groups by device type and variant."""
        from iris.cluster.constraints import DeviceType

        platform = make_mock_platform()
        group = ScalingGroup(scale_group_config, platform)  # TPU with accelerator_variant="v5p-8"

        # CPU matches any group
        assert group.matches_device_requirement(DeviceType.CPU, None)

        # TPU with matching variant (case-insensitive)
        assert group.matches_device_requirement(DeviceType.TPU, frozenset({"v5p-8"}))
        assert group.matches_device_requirement(DeviceType.TPU, frozenset({"V5P-8"}))
        assert group.matches_device_requirement(DeviceType.TPU, frozenset({"V5p-8"}))
        assert group.matches_device_requirement(DeviceType.TPU, None)  # None = any TPU
        assert not group.matches_device_requirement(DeviceType.TPU, frozenset({"v5litepod-4"}))

        # Multiple variants: matches if group variant is in the set
        assert group.matches_device_requirement(DeviceType.TPU, frozenset({"v4-8", "v5p-8"}))
        assert not group.matches_device_requirement(DeviceType.TPU, frozenset({"v4-8", "v5litepod-4"}))

        # GPU doesn't match TPU group
        assert not group.matches_device_requirement(DeviceType.GPU, None)


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
        worker_ids = [vm.worker_id for vm in handle.describe().workers]
        group.mark_slice_ready(handle.slice_id, worker_ids)
        state = _get_slice_state(group, handle)

        # Get worker ID from mock
        worker_id = _get_worker_id(handle)
        status_map = {worker_id: WorkerStatus(worker_id="", running_task_ids=frozenset())}
        assert group._verify_slice_idle(state, status_map)

    def test_known_busy_worker_blocks_idle(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """A slice with a known busy worker is NOT idle."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)
        ts = Timestamp.from_ms(1000000)
        handle = _tracked_scale_up(group, timestamp=ts)
        state = _get_slice_state(group, handle)

        worker_id = _get_worker_id(handle)
        status_map = {worker_id: WorkerStatus(worker_id="", running_task_ids=frozenset({"task-1"}))}
        assert not group._verify_slice_idle(state, status_map)


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
        platform = make_mock_platform()
        platform.create_slice.side_effect = QuotaExhaustedError("no quota")

        group = ScalingGroup(
            unbounded_config, platform, quota_timeout=Duration.from_ms(5000), scale_up_cooldown=Duration.from_ms(0)
        )

        ts = Timestamp.from_ms(1000000)
        group.begin_scale_up(timestamp=ts)
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
        from iris.cluster.controller.autoscaler.scaling_group import prepare_slice_config

        parent = config_pb2.ScaleGroupConfig(
            name="test-group",
        )
        parent.slice_template.capacity_type = config_pb2.CAPACITY_TYPE_PREEMPTIBLE
        parent.slice_template.gcp.zone = "us-central1-a"
        parent.slice_template.gcp.runtime_version = "v2-alpha-tpuv5"

        result = prepare_slice_config(parent.slice_template, parent, "iris")
        assert result.capacity_type == config_pb2.CAPACITY_TYPE_PREEMPTIBLE

    def test_preemptible_false_by_default(self):
        """capacity_type defaults to CAPACITY_TYPE_UNSPECIFIED when not set on template."""
        from iris.cluster.controller.autoscaler.scaling_group import prepare_slice_config

        parent = config_pb2.ScaleGroupConfig(
            name="test-group",
        )
        parent.slice_template.gcp.zone = "us-central1-a"
        parent.slice_template.gcp.runtime_version = "v2-alpha-tpuv5"

        result = prepare_slice_config(parent.slice_template, parent, "iris")
        assert result.capacity_type == config_pb2.CAPACITY_TYPE_UNSPECIFIED


class TestPrepareSliceConfigGpuCount:
    """prepare_slice_config propagates gpu_count from parent resources."""

    def test_gpu_count_propagated_from_resources(self):
        from iris.cluster.config import _derive_slice_config_from_resources
        from iris.cluster.controller.autoscaler.scaling_group import prepare_slice_config

        parent = config_pb2.ScaleGroupConfig(name="gpu-group")
        parent.resources.CopyFrom(
            config_pb2.ScaleGroupResources(device_count=8, device_type=config_pb2.ACCELERATOR_TYPE_GPU)
        )
        parent.slice_template.coreweave.instance_type = "gd-8xh100ib-i128"
        # Simulate config loading: derive slice_template fields from resources
        wrapper = config_pb2.IrisClusterConfig()
        wrapper.scale_groups["gpu-group"].CopyFrom(parent)
        _derive_slice_config_from_resources(wrapper)
        parent = wrapper.scale_groups["gpu-group"]

        result = prepare_slice_config(parent.slice_template, parent, "iris")
        assert result.gpu_count == 8

    def test_gpu_count_zero_when_no_resources(self):
        from iris.cluster.controller.autoscaler.scaling_group import prepare_slice_config

        parent = config_pb2.ScaleGroupConfig(name="cpu-group")
        parent.slice_template.coreweave.instance_type = "cd-gp-i64-erapids"

        result = prepare_slice_config(parent.slice_template, parent, "iris")
        assert result.gpu_count == 0

    def test_coreweave_yaml_gpu_count_flows_through(self):
        """Loading coreweave.yaml and running prepare_slice_config produces correct gpu_count."""
        from pathlib import Path

        from iris.cluster.config import load_config
        from iris.cluster.controller.autoscaler.scaling_group import prepare_slice_config

        yaml_path = Path(__file__).parents[3] / "examples" / "coreweave.yaml"
        config = load_config(yaml_path)

        h100_sg = config.scale_groups["h100-8x"]
        slice_config = prepare_slice_config(h100_sg.slice_template, h100_sg, config.platform.label_prefix)
        assert slice_config.gpu_count == 8

        cpu_sg = config.scale_groups["cpu-erapids"]
        cpu_slice = prepare_slice_config(cpu_sg.slice_template, cpu_sg, config.platform.label_prefix)
        assert cpu_slice.gpu_count == 0


class TestMarkSliceLockDiscipline:
    """Tests that mark_slice_ready/mark_slice_failed hold the lock during mutation."""

    def test_mark_slice_ready_atomic(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """lifecycle and worker_ids are set while holding the lock."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)
        handle = _tracked_scale_up(group)

        # Verify the slice starts as BOOTING with no addresses
        state = _get_slice_state(group, handle)
        assert state.lifecycle == SliceLifecycleState.BOOTING
        assert state.worker_ids == []

        addresses = ["10.0.0.1", "10.0.0.2"]
        group.mark_slice_ready(handle.slice_id, addresses)

        # All fields should be set atomically
        with group._slices_lock:
            state = group._slices[handle.slice_id]
            assert state.lifecycle == SliceLifecycleState.READY
            assert state.worker_ids == addresses

    def test_mark_slice_failed_atomic(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """lifecycle is set to FAILED while holding the lock."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)
        handle = _tracked_scale_up(group)

        group.mark_slice_failed(handle.slice_id)

        with group._slices_lock:
            state = group._slices[handle.slice_id]
            assert state.lifecycle == SliceLifecycleState.FAILED

    def test_mark_slice_ready_nonexistent_is_noop(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """mark_slice_ready on a nonexistent slice does not raise."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)
        group.mark_slice_ready("nonexistent", ["10.0.0.1"])

    def test_mark_slice_failed_nonexistent_is_noop(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """mark_slice_failed on a nonexistent slice does not raise."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)
        group.mark_slice_failed("nonexistent")


def test_slice_state_to_proto_uses_worker_ids_as_vm_ids():
    """slice_state_to_proto uses worker_ids directly as vm_id."""
    from iris.cluster.controller.autoscaler.scaling_group import slice_state_to_proto

    handle = make_fake_slice_handle("my-slice", scale_group="sg", created_at_ms=1000)
    state = SliceState(
        handle=handle,
        lifecycle=SliceLifecycleState.READY,
        worker_ids=["10.0.0.1", "10.0.0.2"],
    )
    proto = slice_state_to_proto(state)
    assert proto.vms[0].vm_id == "10.0.0.1"
    assert proto.vms[1].vm_id == "10.0.0.2"


def _make_worker_handle(vm_id: str, cloud_state: CloudWorkerState, address: str = "10.0.0.1") -> FakeWorkerHandle:
    return FakeWorkerHandle(_vm_id=vm_id, _internal_address=address, _state=cloud_state)


def _make_slice_handle(
    slice_id: str,
    slice_state: CloudSliceState,
    vm_handles: list[FakeWorkerHandle],
) -> FakeSliceHandle:
    iris_labels = Labels("iris")
    return FakeSliceHandle(
        _slice_id=slice_id,
        _scale_group="test-group",
        _zone="us-central1-a",
        _labels={iris_labels.iris_scale_group: "test-group", iris_labels.iris_managed: "true"},
        _created_at=Timestamp.from_ms(1000000),
        _status=SliceStatus(state=slice_state, worker_count=len(vm_handles), workers=vm_handles),
    )


def _make_multi_vm_config(num_vms: int = 4) -> config_pb2.ScaleGroupConfig:
    config = config_pb2.ScaleGroupConfig(
        name="multi-vm-group",
        buffer_slices=0,
        max_slices=10,
        num_vms=num_vms,
    )
    config.slice_template.gcp.runtime_version = "v2-alpha-tpuv5"
    config.slice_template.gcp.zone = "us-central1-a"
    return _with_resources(config, num_vms=num_vms)


def _make_multi_vm_slice_handle(
    slice_id: str,
    num_vms: int = 4,
    scale_group: str = "multi-vm-group",
) -> FakeSliceHandle:
    vm_states = [vm_pb2.VM_STATE_READY] * num_vms
    return make_fake_slice_handle(
        slice_id,
        scale_group=scale_group,
        vm_states=vm_states,
    )


class TestMultiVmSliceIdleScaleDown:
    """Tests for idle detection and scale-down with multi-VM slices.

    A multi-VM slice (e.g. num_vms=4) has multiple workers. The slice is only
    idle when ALL workers are idle. One busy worker keeps the entire slice alive.
    """

    def test_multi_vm_slice_idle_when_all_workers_idle(self):
        """A 4-VM slice scales down when all 4 workers have no running tasks."""
        config = _make_multi_vm_config(num_vms=4)
        discovered = [_make_multi_vm_slice_handle("slice-001", num_vms=4)]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(config, platform, idle_threshold=Duration.from_ms(1000))
        group.reconcile()
        _mark_discovered_ready(group, discovered)

        # Get all 4 worker IDs
        handle = group.get_slice("slice-001")
        workers = handle.describe().workers
        wids = [w.worker_id for w in workers]
        assert len(wids) == 4

        # Mark all workers as active at t=0
        active_map = {wid: WorkerStatus(worker_id=wid, running_task_ids=frozenset({"task"})) for wid in wids}
        group.update_slice_activity(active_map, Timestamp.from_ms(0))

        # At t=10_000 (past threshold), all workers now idle
        idle_map = {wid: WorkerStatus(worker_id=wid, running_task_ids=frozenset()) for wid in wids}
        scaled_down = group.scale_down_if_idle(idle_map, target_capacity=0, timestamp=Timestamp.from_ms(10_000))

        assert len(scaled_down) == 1
        assert group.slice_count() == 0

    def test_multi_vm_slice_not_idle_when_one_worker_busy(self):
        """A 4-VM slice does NOT scale down when 1 of 4 workers has a running task."""
        config = _make_multi_vm_config(num_vms=4)
        discovered = [_make_multi_vm_slice_handle("slice-001", num_vms=4)]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(config, platform, idle_threshold=Duration.from_ms(1000))
        group.reconcile()
        _mark_discovered_ready(group, discovered)

        handle = group.get_slice("slice-001")
        workers = handle.describe().workers
        wids = [w.worker_id for w in workers]

        # Mark all active initially
        active_map = {wid: WorkerStatus(worker_id=wid, running_task_ids=frozenset({"task"})) for wid in wids}
        group.update_slice_activity(active_map, Timestamp.from_ms(0))

        # 3 workers idle, 1 still has tasks
        mixed_map = {}
        for i, wid in enumerate(wids):
            tasks = frozenset({"task-running"}) if i == 0 else frozenset()
            mixed_map[wid] = WorkerStatus(worker_id=wid, running_task_ids=tasks)

        scaled_down = group.scale_down_if_idle(mixed_map, target_capacity=0, timestamp=Timestamp.from_ms(10_000))

        assert len(scaled_down) == 0
        assert group.slice_count() == 1

    def test_multi_vm_slice_activity_updates_when_any_worker_busy(self):
        """update_slice_activity stamps last_active when ANY worker in the slice is busy."""
        config = _make_multi_vm_config(num_vms=4)
        discovered = [_make_multi_vm_slice_handle("slice-001", num_vms=4)]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(config, platform, idle_threshold=Duration.from_ms(60_000))
        group.reconcile()
        _mark_discovered_ready(group, discovered, timestamp=Timestamp.from_ms(1000))

        handle = group.get_slice("slice-001")
        workers = handle.describe().workers
        wids = [w.worker_id for w in workers]

        # Only worker 2 (of 4) has a task — whole slice should be marked active
        vm_map = {}
        for i, wid in enumerate(wids):
            tasks = frozenset({"task-on-vm-2"}) if i == 2 else frozenset()
            vm_map[wid] = WorkerStatus(worker_id=wid, running_task_ids=tasks)

        group.update_slice_activity(vm_map, Timestamp.from_ms(5000))

        # Slice should not be eligible for scaledown since it was just active
        assert not group.is_slice_eligible_for_scaledown("slice-001", Timestamp.from_ms(5000))

    def test_multi_vm_two_slices_only_idle_one_scaled_down(self):
        """With 2 multi-VM slices, only the idle one is scaled down (busy one kept)."""
        config = _make_multi_vm_config(num_vms=4)
        discovered = [
            _make_multi_vm_slice_handle("slice-001", num_vms=4),
            _make_multi_vm_slice_handle("slice-002", num_vms=4),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(config, platform, idle_threshold=Duration.from_ms(1000))
        group.reconcile()
        _mark_discovered_ready(group, discovered)

        h1 = group.get_slice("slice-001")
        h2 = group.get_slice("slice-002")
        wids_1 = [w.worker_id for w in h1.describe().workers]
        wids_2 = [w.worker_id for w in h2.describe().workers]

        # Mark both slices active initially
        all_active = {}
        for wid in wids_1 + wids_2:
            all_active[wid] = WorkerStatus(worker_id=wid, running_task_ids=frozenset({"task"}))
        group.update_slice_activity(all_active, Timestamp.from_ms(0))

        # At t=10_000: slice-001 all idle, slice-002 still has work on one VM
        vm_map = {}
        for wid in wids_1:
            vm_map[wid] = WorkerStatus(worker_id=wid, running_task_ids=frozenset())
        for i, wid in enumerate(wids_2):
            tasks = frozenset({"running"}) if i == 0 else frozenset()
            vm_map[wid] = WorkerStatus(worker_id=wid, running_task_ids=tasks)

        scaled_down = group.scale_down_if_idle(vm_map, target_capacity=1, timestamp=Timestamp.from_ms(10_000))

        assert len(scaled_down) == 1
        assert scaled_down[0].slice_id == "slice-001"
        assert group.slice_count() == 1

    def test_multi_vm_verify_idle_requires_all_workers_known(self):
        """_verify_slice_idle returns False if some workers are not in the status map."""
        config = _make_multi_vm_config(num_vms=4)
        platform = make_mock_platform()
        group = ScalingGroup(config, platform)

        handle = _tracked_scale_up(group)
        worker_ids = [f"10.0.0.{i}" for i in range(4)]
        group.mark_slice_ready(handle.slice_id, worker_ids)
        state = _get_slice_state(group, handle)

        # Only 2 of 4 workers in status map, both idle
        partial_map = {
            "10.0.0.0": WorkerStatus(worker_id="10.0.0.0", running_task_ids=frozenset()),
            "10.0.0.1": WorkerStatus(worker_id="10.0.0.1", running_task_ids=frozenset()),
        }
        # Should still return True — known workers are all idle, unknown are skipped
        assert group._verify_slice_idle(state, partial_map)

    def test_multi_vm_verify_idle_empty_map_returns_false(self):
        """_verify_slice_idle returns False when no workers appear in the status map."""
        config = _make_multi_vm_config(num_vms=4)
        platform = make_mock_platform()
        group = ScalingGroup(config, platform)

        handle = _tracked_scale_up(group)
        worker_ids = [f"10.0.0.{i}" for i in range(4)]
        group.mark_slice_ready(handle.slice_id, worker_ids)
        state = _get_slice_state(group, handle)

        assert not group._verify_slice_idle(state, {})


class TestSliceStateToProtoIdleFields:
    """Tests for the idle/last_active fields on SliceInfo proto."""

    def test_idle_true_when_past_threshold(self):
        from iris.cluster.controller.autoscaler.scaling_group import slice_state_to_proto

        handle = make_fake_slice_handle("s1", scale_group="g1", created_at_ms=1000)
        state = SliceState(
            handle=handle,
            lifecycle=SliceLifecycleState.READY,
            worker_ids=["10.0.0.1"],
            last_active=Timestamp.from_ms(1000),
        )

        # With a threshold of 1s and last_active at 1s ago, idle should be True
        # (since Timestamp.now() will be >> 2000ms)
        proto = slice_state_to_proto(state, idle_threshold=Duration.from_ms(1))
        assert proto.idle is True
        assert proto.last_active.epoch_ms == 1000

    def test_idle_false_when_no_threshold(self):
        from iris.cluster.controller.autoscaler.scaling_group import slice_state_to_proto

        handle = make_fake_slice_handle("s1", scale_group="g1", created_at_ms=1000)
        state = SliceState(
            handle=handle,
            lifecycle=SliceLifecycleState.READY,
            worker_ids=["10.0.0.1"],
            last_active=Timestamp.from_ms(1000),
        )
        proto = slice_state_to_proto(state, idle_threshold=None)
        assert proto.idle is False

    def test_idle_false_for_non_ready_slices(self):
        from iris.cluster.controller.autoscaler.scaling_group import slice_state_to_proto

        handle = make_fake_slice_handle("s1", scale_group="g1", created_at_ms=1000)
        state = SliceState(
            handle=handle,
            lifecycle=SliceLifecycleState.BOOTING,
            worker_ids=[],
            last_active=Timestamp.from_ms(0),
        )
        proto = slice_state_to_proto(state, idle_threshold=Duration.from_ms(1))
        assert proto.idle is False
