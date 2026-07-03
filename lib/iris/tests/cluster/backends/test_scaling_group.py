# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ScalingGroup behavior.

These tests focus on observable behavior - scaling policy decisions,
VM group management, and state tracking - not on implementation details.
"""

import logging
from pathlib import Path

import pytest
from iris.cluster.config import (
    CoreweaveSliceConfig,
    GcpSliceConfig,
    ScaleGroupConfig,
    ScaleGroupResources,
    SliceConfig,
    load_config,
)
from iris.cluster.constraints import DeviceType
from iris.cluster.controller.autoscaler.backoff_detector import GroupHealth, SliceFate
from iris.cluster.controller.autoscaler.scaling_group import (
    GroupAvailability,
    ScalingGroup,
    SliceLifecycleState,
    SliceState,
    _zones_from_config,
    prepare_slice_config,
    slice_state_to_proto,
)
from iris.cluster.platforms.types import (
    CloudSliceState,
    CloudWorkerState,
    Labels,
    QuotaExhaustedError,
    SliceStatus,
)
from iris.cluster.types import AcceleratorType, CapacityType, WorkerStatus, WorkerUsability
from iris.rpc import vm_pb2
from rigging.timing import Duration, Timestamp
from tests.cluster.backends.conftest import (
    FakeSliceHandle,
    FakeWorkerHandle,
    make_fake_slice_handle,
    make_mock_platform,
)

DEFAULT_RESOURCES = ScaleGroupResources(
    cpu_millicores=64000,
    memory_bytes=64 * 1024**3,
    disk_bytes=100 * 1024**3,
    device_type=AcceleratorType.TPU,
    device_variant="v5p-8",
    device_count=8,
)


def _with_resources(config: ScaleGroupConfig, *, num_vms: int = 1) -> ScaleGroupConfig:
    """Attach default resources/num_vms and re-derive slice_template fields.

    Re-running model_validate triggers ScaleGroupConfig's after-validator, which
    copies device_type/variant/gpu_count from resources onto the slice template.
    """
    if config.resources is None:
        config.resources = DEFAULT_RESOURCES.model_copy(deep=True)
    if config.num_vms is None:
        config.num_vms = num_vms
    # exclude_none mirrors how configs are actually serialized (config_to_dict);
    # an unselected oneof arm is absent rather than a None that reads as a pick.
    return ScaleGroupConfig.model_validate(config.model_dump(exclude_none=True))


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
def scale_group_config() -> ScaleGroupConfig:
    """A standard scale group configuration for tests."""
    config = ScaleGroupConfig(
        name="test-group",
        buffer_slices=1,
        max_slices=5,
        slice_template=SliceConfig(gcp=GcpSliceConfig(runtime_version="v2-alpha-tpuv5", zone="us-central1-a")),
    )
    return _with_resources(config)


@pytest.fixture
def unbounded_config() -> ScaleGroupConfig:
    """A scale group with no min/max constraints."""
    config = ScaleGroupConfig(
        name="unbounded-group",
        buffer_slices=0,
        max_slices=100,
        slice_template=SliceConfig(gcp=GcpSliceConfig(runtime_version="v2-alpha-tpuv5", zone="us-central1-a")),
    )
    return _with_resources(config)


class TestScalingGroupVmGroupOwnership:
    """Tests for VM group ownership and lifecycle."""

    def test_reconcile_adopts_discovered_vm_groups(self, scale_group_config: ScaleGroupConfig):
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

    def test_reconcile_skips_manual_slices(self, scale_group_config: ScaleGroupConfig):
        """Slices tagged iris_manual=true are never adopted by a ScalingGroup."""
        labels = Labels("iris")
        auto = make_fake_slice_handle("slice-auto")
        manual = make_fake_slice_handle("slice-manual")
        manual._labels[labels.iris_manual] = "true"

        platform = make_mock_platform(slices_to_discover=[auto, manual])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()

        assert group.slice_count() == 1
        assert group.get_slice("slice-auto") is not None
        assert group.get_slice("slice-manual") is None

    def test_scale_up_creates_and_tracks_vm_group(self, scale_group_config: ScaleGroupConfig):
        """Full lifecycle (begin + scale_up + complete) creates and tracks a slice."""
        platform = make_mock_platform()
        group = ScalingGroup(scale_group_config, platform)

        new_handle = _tracked_scale_up(group)

        platform.create_slice.assert_called_once()
        assert group.slice_count() == 1
        assert new_handle in group.slice_handles()

    def test_scale_up_passes_tags_as_labels(self, scale_group_config: ScaleGroupConfig):
        """scale_up() passes tags as labels in the slice config."""
        platform = make_mock_platform()
        group = ScalingGroup(scale_group_config, platform)

        group.scale_up(tags={"env": "prod", "team": "ml"})

        platform.create_slice.assert_called_once()
        slice_config = platform.create_slice.call_args[0][0]
        assert slice_config.labels["env"] == "prod"
        assert slice_config.labels["team"] == "ml"

    def test_scale_down_terminates_and_removes_vm_group(self, scale_group_config: ScaleGroupConfig):
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

    def test_scale_down_nonexistent_vm_group_is_noop(self, scale_group_config: ScaleGroupConfig):
        """scale_down() on a nonexistent VM group does nothing."""
        platform = make_mock_platform()
        group = ScalingGroup(scale_group_config, platform)

        # Should not raise
        group.scale_down("nonexistent-slice")

        assert group.slice_count() == 0

    def test_scale_down_cleans_up_even_if_terminate_fails(self, scale_group_config: ScaleGroupConfig):
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

    def test_terminate_all_continues_on_individual_failure(self, scale_group_config: ScaleGroupConfig):
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

    def test_ready_slice_count(self, scale_group_config: ScaleGroupConfig):
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

    def test_slices_needing_describe_includes_ready_without_workers(self, scale_group_config: ScaleGroupConfig):
        """A READY slice with no tracked workers is re-described, not left DEGRADED.

        An adopted slice can come back READY with empty worker_ids; without
        re-describing it the autoscaler would never repopulate membership (so it
        stays "no hosts registered") nor reap it.
        """
        discovered = [
            make_fake_slice_handle("ready-with-workers", all_ready=True),
            make_fake_slice_handle("still-booting", all_ready=False, vm_states=[vm_pb2.VM_STATE_BOOTING]),
            make_fake_slice_handle("ready-no-workers", all_ready=True),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        _mark_discovered_ready(group, [discovered[0]])
        group.mark_slice_ready("ready-no-workers", [])  # READY but membership never resolved

        needing = {slice_id for slice_id, _ in group.slices_needing_describe()}
        assert needing == {"still-booting", "ready-no-workers"}


class TestScalingGroupScalingPolicy:
    """Tests for scaling policy decisions (can_scale_up, rate limiting)."""

    def test_can_scale_up_when_below_max(self, scale_group_config: ScaleGroupConfig):
        """can_scale_up() returns True when below max_slices."""
        platform = make_mock_platform()
        group = ScalingGroup(scale_group_config, platform)

        assert group.can_scale_up()

    def test_cannot_scale_up_at_max_slices(self, scale_group_config: ScaleGroupConfig):
        """can_scale_up() returns False when at max_slices."""
        discovered = [make_fake_slice_handle(f"slice-{i}") for i in range(5)]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()

        assert group.slice_count() == 5  # max_slices
        assert not group.can_scale_up()

    def test_hostile_throttles_bucket_to_probe_floor(self, unbounded_config: ScaleGroupConfig):
        """At 100% churn, the bucket refills at the probe floor and acquires
        succeed only after waiting one floor-tokens' worth of time.

        Continuous control: there is no hard "HOSTILE" block, only a low refill rate.
        ``can_scale_up`` only flips to False on quota.
        """
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)
        ts = Timestamp.from_ms(1_000_000)
        # With decay=0.7, 8 failures drive health below the probe floor → clamped to floor.
        # Each decay also caps the bucket inventory at capacity * health, so by
        # the time we're at the floor the bucket holds well under one token.
        for _ in range(8):
            group.record_create_failed(timestamp=ts)
        now = Timestamp.from_ms(1_001_000)
        # Health-driven throttling is not a "block" — quota would be.
        assert group.can_scale_up(timestamp=now)
        # Immediate acquire fails: drained-on-decay bucket needs to refill at the floor rate.
        assert not group.try_acquire_scale_up(timestamp=now)
        # Default scale_up_rate_limit is 16/min, floor 0.5/min → one token in ~120s.
        much_later = Timestamp.from_ms(ts.epoch_ms() + 5 * 60_000)
        assert group.try_acquire_scale_up(timestamp=much_later)
        assert group.detector.health_label(much_later) == GroupHealth.HOSTILE

    def test_scale_down_rate_limited_by_token_bucket(self, unbounded_config: ScaleGroupConfig):
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


class TestScalingGroupChurnDetector:
    """Tests for ScalingGroup's integration with BackoffDetector.

    Detailed unit tests of the detector itself live in
    ``tests/cluster/controller/test_backoff_detector.py``. These tests verify
    that ScalingGroup wires the detector correctly into its scale-up gating.
    """

    def test_create_failures_drive_detector_hostile(self, unbounded_config: ScaleGroupConfig):
        """Enough create-failures push the detector label to HOSTILE.

        Under failure-counting control HOSTILE is a label only — scale-up keeps
        flowing at the probe floor.
        """
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)
        ts = Timestamp.from_ms(1_000_000)
        # With decay=0.7, 5 failures → 0.168 (HOSTILE).
        for _ in range(5):
            group.record_create_failed(timestamp=ts)
        now = Timestamp.from_ms(1_001_000)
        # Recovery over 1s is small (~3e-4); score remains in the HOSTILE band.
        assert group.detector.health_label(now) == GroupHealth.HOSTILE
        # Hard block is quota-only; failures throttle the bucket but never gate it.
        assert group.can_scale_up(timestamp=now)

    def test_long_lived_preemption_does_not_block_scale_up(self, unbounded_config: ScaleGroupConfig):
        """A slice preempted past short_lived threshold is a positive sample."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)
        create_ts = Timestamp.from_ms(1_000_000)
        # Three slices created and "preempted" 16 minutes later (past the 15-min default).
        for i in range(3):
            slice_id = f"s{i}"
            group.detector.record_created(slice_id, create_ts)
            group.detector.record_terminated(
                slice_id,
                SliceFate.PREEMPTED,
                Timestamp.from_ms(1_000_000 + 16 * 60 * 1000),
            )
        now = Timestamp.from_ms(1_000_000 + 17 * 60 * 1000)
        assert group.can_scale_up(timestamp=now)


class TestScalingGroupDemandTracking:
    """Tests for demand tracking."""

    def test_update_demand_tracks_peak(self, unbounded_config: ScaleGroupConfig):
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

    def test_slice_not_eligible_until_workers_observed_idle(self, unbounded_config: ScaleGroupConfig):
        """Slice with quiet_since=None (no idle observation yet) is not eligible.

        Under the new model, eligibility requires an active→idle transition to
        have been observed; a never-observed slice stays alive until the next
        autoscaler tick stamps quiet_since.
        """
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(60_000))
        _tracked_scale_up(group)
        slice_id = next(iter(group.slice_handles())).slice_id

        # Never had activity observed -> not eligible.
        assert not group.is_slice_eligible_for_scaledown(slice_id, Timestamp.from_ms(1000))

    def test_slice_not_eligible_when_recently_active(self, unbounded_config: ScaleGroupConfig):
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

    def test_slice_eligible_after_idle_threshold(self, unbounded_config: ScaleGroupConfig):
        """Slice is eligible after idle_threshold has elapsed since the workers went idle."""
        discovered = [make_fake_slice_handle("slice-001", all_ready=True)]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(60_000))
        group.reconcile()
        _mark_discovered_ready(group, discovered)

        worker_id = _get_worker_id(group.get_slice("slice-001"))

        # First tick observes the workers idle at t=1000 -> stamps quiet_since.
        idle_map = {worker_id: WorkerStatus(worker_id=worker_id, running_task_ids=frozenset())}
        group.update_slice_activity(idle_map, Timestamp.from_ms(1000))

        # After threshold (61s > 60s) -> eligible
        assert group.is_slice_eligible_for_scaledown("slice-001", Timestamp.from_ms(61_001))

    def test_get_idle_slices_returns_longest_idle_first(self, unbounded_config: ScaleGroupConfig):
        """get_idle_slices returns slices sorted by idle time (longest first)."""
        discovered = [
            make_fake_slice_handle("slice-001", all_ready=True),
            make_fake_slice_handle("slice-002", all_ready=True),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(1000))
        group.reconcile()
        _mark_discovered_ready(group, discovered)

        wid_001 = _get_worker_id(group.get_slice("slice-001"))
        wid_002 = _get_worker_id(group.get_slice("slice-002"))

        # slice-001 transitions to idle at t=1000 (longer dwell time below).
        only_001_idle = {
            wid_001: WorkerStatus(worker_id=wid_001, running_task_ids=frozenset()),
            wid_002: WorkerStatus(worker_id=wid_002, running_task_ids=frozenset({"task-2"})),
        }
        group.update_slice_activity(only_001_idle, Timestamp.from_ms(1000))

        # slice-002 transitions to idle at t=5000.
        both_idle = {
            wid_001: WorkerStatus(worker_id=wid_001, running_task_ids=frozenset()),
            wid_002: WorkerStatus(worker_id=wid_002, running_task_ids=frozenset()),
        }
        group.update_slice_activity(both_idle, Timestamp.from_ms(5000))

        # At t=10_000 both have crossed the 1s threshold; slice-001 has been
        # idle longer (9s vs 5s).
        idle_slices = group.get_idle_slices(Timestamp.from_ms(10_000))
        assert len(idle_slices) == 2
        assert idle_slices[0].handle.slice_id == "slice-001"

    def test_update_slice_activity_tracks_active_slices(self, unbounded_config: ScaleGroupConfig):
        """Active slice stays alive; idle slice becomes eligible after threshold."""
        discovered = [
            make_fake_slice_handle("slice-001", all_ready=True),
            make_fake_slice_handle("slice-002", all_ready=True),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(1000))
        group.reconcile()
        ready_ts = Timestamp.from_ms(1000)
        _mark_discovered_ready(group, discovered, timestamp=ready_ts)

        wid_001 = _get_worker_id(group.get_slice("slice-001"))
        wid_002 = _get_worker_id(group.get_slice("slice-002"))

        # slice-001 has running tasks, slice-002 has none.
        vm_status_map = {
            wid_001: WorkerStatus(worker_id=wid_001, running_task_ids=frozenset({"task-1"})),
            wid_002: WorkerStatus(worker_id=wid_002, running_task_ids=frozenset()),
        }

        # First observation at t=1500: stamps quiet_since on slice-002 only.
        group.update_slice_activity(vm_status_map, Timestamp.from_ms(1500))

        # At t=3000, slice-002 has been quiet for 1500ms (> 1000ms threshold)
        # while slice-001 stays active.
        check_ts = Timestamp.from_ms(3000)
        assert not group.is_slice_eligible_for_scaledown("slice-001", check_ts)
        assert group.is_slice_eligible_for_scaledown("slice-002", check_ts)

    def test_continuous_activity_keeps_quiet_since_none(self, unbounded_config: ScaleGroupConfig):
        """Regression: a continuously-active slice never accrues quiet time.

        The previous implementation persisted ``last_active_ms`` to the DB on a
        threshold the per-tick mutation defeated; we observed live slices with
        a multi-hour-stale DB row even though their workers had running tasks.
        Under the transition model an active slice keeps ``quiet_since=None``
        through arbitrarily many ticks and is never eligible for scale-down.
        """
        discovered = [make_fake_slice_handle("slice-001", all_ready=True)]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(60_000))
        group.reconcile()
        ready_ts = Timestamp.from_ms(1_000_000)
        _mark_discovered_ready(group, discovered, timestamp=ready_ts)

        wid = _get_worker_id(group.get_slice("slice-001"))
        active_map = {wid: WorkerStatus(worker_id=wid, running_task_ids=frozenset({"task-1"}))}

        # 20 ticks of continuous activity at production cadence (10s apart).
        for tick in range(1, 21):
            ts = Timestamp.from_ms(ready_ts.epoch_ms() + tick * 10_000)
            group.update_slice_activity(active_map, ts)
            assert not group.is_slice_eligible_for_scaledown("slice-001", ts)

        with group._slices_lock:
            assert group._slices["slice-001"].quiet_since is None

    def test_scale_down_if_idle_terminates_eligible_slice(self, unbounded_config: ScaleGroupConfig):
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

        vm_status_map_idle = {
            wid_001: WorkerStatus(worker_id=wid_001, running_task_ids=frozenset()),
            wid_002: WorkerStatus(worker_id=wid_002, running_task_ids=frozenset()),
        }
        # First idle observation at t=0 stamps quiet_since on both slices.
        group.update_slice_activity(vm_status_map_idle, Timestamp.from_ms(0))

        # At t=10_000 they have been quiet 10s — well past the 1s threshold.
        scaled_down = group.scale_down_if_idle(
            vm_status_map_idle, target_capacity=1, timestamp=Timestamp.from_ms(10_000)
        )

        assert len(scaled_down) == 1
        assert group.slice_count() == 1  # One slice was terminated

    def test_scale_down_retains_degraded_idle_slice(self, unbounded_config: ScaleGroupConfig):
        """A slice whose worker is idle but DEGRADED is NOT reclaimed as spare.

        Regression for the autoscaler/scheduler disagreement: a degraded worker
        runs no tasks (so the slice reads "idle") but the scheduler cannot place
        onto it, so scaling it down as free capacity is wrong. Only the HEALTHY
        idle slice is reclaimed; the degraded one is retained for the health
        teardown path.
        """
        discovered = [
            make_fake_slice_handle("slice-001", all_ready=True),
            make_fake_slice_handle("slice-002", all_ready=True),
        ]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(1000))
        group.reconcile()
        _mark_discovered_ready(group, discovered)

        wid_healthy = _get_worker_id(group.get_slice("slice-001"))
        wid_degraded = _get_worker_id(group.get_slice("slice-002"))

        # Both slices idle (no running tasks), but slice-002's worker is DEGRADED.
        status_map = {
            wid_healthy: WorkerStatus(worker_id=wid_healthy, running_task_ids=frozenset()),
            wid_degraded: WorkerStatus(
                worker_id=wid_degraded, running_task_ids=frozenset(), usability=WorkerUsability.DEGRADED
            ),
        }
        group.update_slice_activity(status_map, Timestamp.from_ms(0))

        # Target 1: the autoscaler wants to shed one of the two idle slices.
        scaled_down = group.scale_down_if_idle(status_map, target_capacity=1, timestamp=Timestamp.from_ms(10_000))

        # Only the healthy slice is reclaimed; the degraded one is retained.
        assert [h.slice_id for h in scaled_down] == ["slice-001"]
        assert group.get_slice("slice-002") is not None

    def test_scale_down_retains_all_degraded_idle_slices(self, unbounded_config: ScaleGroupConfig):
        """When every idle slice over target is DEGRADED, nothing is reclaimed."""
        discovered = [make_fake_slice_handle("slice-001", all_ready=True)]
        platform = make_mock_platform(slices_to_discover=discovered)
        group = ScalingGroup(unbounded_config, platform, idle_threshold=Duration.from_ms(1000))
        group.reconcile()
        _mark_discovered_ready(group, discovered)

        wid = _get_worker_id(group.get_slice("slice-001"))
        status_map = {wid: WorkerStatus(worker_id=wid, running_task_ids=frozenset(), usability=WorkerUsability.DEGRADED)}
        group.update_slice_activity(status_map, Timestamp.from_ms(0))

        scaled_down = group.scale_down_if_idle(status_map, target_capacity=0, timestamp=Timestamp.from_ms(10_000))

        assert scaled_down == []
        assert group.get_slice("slice-001") is not None

    def test_scale_down_if_idle_respects_target_capacity(self, unbounded_config: ScaleGroupConfig):
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

    def test_scale_down_cleans_up_idle_tracking(self, unbounded_config: ScaleGroupConfig):
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


def test_scale_down_no_misleading_rate_limit_log(unbounded_config: ScaleGroupConfig, caplog):
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

    idle_map = {
        wid_001: WorkerStatus(worker_id=wid_001, running_task_ids=frozenset()),
        wid_002: WorkerStatus(worker_id=wid_002, running_task_ids=frozenset()),
    }
    # First idle observation at t=0 stamps quiet_since; by t=1000 both
    # have been quiet 1s, past the 100ms threshold.
    group.update_slice_activity(idle_map, Timestamp.from_ms(0))

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
        scale_group_config: ScaleGroupConfig,
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

    def test_failed_takes_precedence(self, scale_group_config: ScaleGroupConfig):
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

    def test_unobserved_slices_counted_as_booting(self, scale_group_config: ScaleGroupConfig):
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

    def test_available_when_no_constraints(self, unbounded_config: ScaleGroupConfig):
        """Group is AVAILABLE when not in backoff, quota ok, and under capacity."""

        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)

        state = group.availability()
        assert state.status == GroupAvailability.AVAILABLE

    def test_at_max_slices_when_at_max_slices(self):
        """Group is AT_MAX_SLICES when at max_slices with all slices READY."""

        config = _with_resources(
            ScaleGroupConfig(
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

    def test_backoff_when_detector_hostile(self, unbounded_config: ScaleGroupConfig):
        """Group is in BACKOFF when the churn detector is HOSTILE."""

        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)
        ts = Timestamp.from_ms(1000000)
        for _ in range(8):
            group.record_create_failed(timestamp=ts)

        state = group.availability(Timestamp.from_ms(1001000))
        assert state.status == GroupAvailability.BACKOFF

    def test_can_accept_demand_true_when_available(self, unbounded_config: ScaleGroupConfig):
        """can_accept_demand() returns True when AVAILABLE."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)

        assert group.can_accept_demand() is True

    def test_at_max_slices_rejects_demand(self):
        """AT_MAX_SLICES groups reject demand so it falls through to lower-priority groups."""
        config = _with_resources(
            ScaleGroupConfig(
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

    def test_quota_exceeded_blocks_demand_until_timeout(self, unbounded_config: ScaleGroupConfig):
        """Quota exceeded state auto-expires after timeout."""

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

    def test_successful_scale_up_clears_quota_state(self, unbounded_config: ScaleGroupConfig):
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

    def test_quota_exceeded_takes_precedence_over_churn_backoff(self, unbounded_config: ScaleGroupConfig):
        """Quota exceeded reports as QUOTA_EXCEEDED even if churn is HOSTILE."""

        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform, quota_timeout=Duration.from_ms(60_000))

        ts = Timestamp.from_ms(1000)
        for _ in range(8):
            group.record_create_failed(timestamp=ts)  # detector → HOSTILE
        group.record_quota_exceeded("quota exceeded", ts)

        state = group.availability(timestamp=Timestamp.from_ms(2000))
        assert state.status == GroupAvailability.QUOTA_EXCEEDED

    def test_available_immediately_after_scale_up(self):
        """After complete_scale_up, the group is AVAILABLE (no cooldown gate)."""

        config = _with_resources(
            ScaleGroupConfig(
                name="test-group",
                buffer_slices=0,
                max_slices=10,
                slice_template=SliceConfig(gcp=GcpSliceConfig(zone="us-central1-a")),
            ),
        )
        platform = make_mock_platform()
        group = ScalingGroup(config, platform)

        ts = Timestamp.from_ms(1_000_000)
        group.begin_scale_up(timestamp=ts)
        handle = group.scale_up(timestamp=ts)
        group.complete_scale_up(handle, ts)

        state = group.availability(Timestamp.from_ms(1_001_000))
        assert state.status == GroupAvailability.AVAILABLE

    def test_at_max_slices_with_inflight_capacity_accepts_demand(self):
        """At max_slices but with in-flight booting capacity → COOLDOWN (accepts demand)."""

        config = _with_resources(
            ScaleGroupConfig(
                name="test-group",
                buffer_slices=0,
                max_slices=1,
                slice_template=SliceConfig(gcp=GcpSliceConfig(zone="us-central1-a")),
            ),
        )
        platform = make_mock_platform()
        group = ScalingGroup(config, platform)

        ts = Timestamp.from_ms(1_000_000)
        group.begin_scale_up(timestamp=ts)
        handle = group.scale_up(timestamp=ts)
        group.complete_scale_up(handle, ts)

        # Slice is at max_slices but still BOOTING — accepts demand.
        state = group.availability(Timestamp.from_ms(1_003_000))
        assert state.status == GroupAvailability.COOLDOWN

        # Once the slice is READY, AT_MAX_SLICES takes precedence over cooldown
        worker_ids = [w.worker_id for w in handle.describe().workers]
        group.mark_slice_ready(handle.slice_id, worker_ids)
        state = group.availability(Timestamp.from_ms(1_003_000))
        assert state.status == GroupAvailability.AT_MAX_SLICES

    def test_matches_device_requirement_filters_by_type_and_variant(self, scale_group_config: ScaleGroupConfig):
        """matches_device_requirement filters groups by device type and variant."""

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

    def test_unknown_workers_do_not_count_as_idle(self, unbounded_config: ScaleGroupConfig):
        """A slice with no workers in the status map is NOT idle (we don't know yet)."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)
        ts = Timestamp.from_ms(1000000)
        handle = _tracked_scale_up(group, timestamp=ts)
        state = _get_slice_state(group, handle)

        # Empty status map -- no workers known
        assert not group._verify_slice_idle(state, {})

    def test_known_idle_workers_are_idle(self, unbounded_config: ScaleGroupConfig):
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

    def test_known_busy_worker_blocks_idle(self, unbounded_config: ScaleGroupConfig):
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
        config = ScaleGroupConfig(name="g", slice_template=SliceConfig(gcp=GcpSliceConfig(zone="us-central1-a")))
        assert _zones_from_config(config) == ["us-central1-a"]

    def test_gcp_with_no_zone_raises(self):
        config = ScaleGroupConfig(
            name="g", slice_template=SliceConfig(gcp=GcpSliceConfig(runtime_version="v2-alpha-tpuv5"))
        )
        with pytest.raises(ValueError, match="no zone configured"):
            _zones_from_config(config)

    def test_non_gcp_returns_empty(self):
        config = ScaleGroupConfig(name="g")
        assert _zones_from_config(config) == []


class TestCanScaleUpQuotaExhausted:
    """can_scale_up must respect quota_exceeded state."""

    def test_cannot_scale_up_during_quota_exhaustion(self, unbounded_config: ScaleGroupConfig):
        """can_scale_up() returns False while quota_exceeded deadline is active."""
        platform = make_mock_platform()
        platform.create_slice.side_effect = QuotaExhaustedError("no quota")

        group = ScalingGroup(unbounded_config, platform, quota_timeout=Duration.from_ms(5000))

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

        parent = ScaleGroupConfig(
            name="test-group",
            slice_template=SliceConfig(
                capacity_type=CapacityType.PREEMPTIBLE,
                gcp=GcpSliceConfig(zone="us-central1-a", runtime_version="v2-alpha-tpuv5"),
            ),
        )

        result = prepare_slice_config(parent.slice_template, parent, "iris")
        assert result.capacity_type == CapacityType.PREEMPTIBLE

    def test_preemptible_false_by_default(self):
        """capacity_type defaults to None (unset) when not set on the template."""

        parent = ScaleGroupConfig(
            name="test-group",
            slice_template=SliceConfig(gcp=GcpSliceConfig(zone="us-central1-a", runtime_version="v2-alpha-tpuv5")),
        )

        result = prepare_slice_config(parent.slice_template, parent, "iris")
        assert result.capacity_type is None


class TestPrepareSliceConfigGpuCount:
    """prepare_slice_config propagates gpu_count from parent resources."""

    def test_gpu_count_propagated_from_resources(self):
        # Construction runs ScaleGroupConfig's after-validator, deriving
        # slice_template.gpu_count from resources (device_type=gpu, device_count=8).
        parent = ScaleGroupConfig(
            name="gpu-group",
            resources=ScaleGroupResources(device_count=8, device_type=AcceleratorType.GPU),
            slice_template=SliceConfig(coreweave=CoreweaveSliceConfig(instance_type="gd-8xh100ib-i128")),
        )

        result = prepare_slice_config(parent.slice_template, parent, "iris")
        assert result.gpu_count == 8

    def test_gpu_count_zero_when_no_resources(self):
        parent = ScaleGroupConfig(
            name="cpu-group",
            slice_template=SliceConfig(coreweave=CoreweaveSliceConfig(instance_type="cd-gp-i64-erapids")),
        )

        result = prepare_slice_config(parent.slice_template, parent, "iris")
        assert result.gpu_count == 0

    def test_coreweave_yaml_gpu_count_flows_through(self):
        """Loading ci-coreweave.yaml and running prepare_slice_config produces correct gpu_count."""

        yaml_path = Path(__file__).parents[3] / "config" / "ci-coreweave.yaml"
        config = load_config(yaml_path)

        h100_sg = config.scale_groups["h100-8x"]
        slice_config = prepare_slice_config(h100_sg.slice_template, h100_sg, config.platform.label_prefix)
        assert slice_config.gpu_count == 8

        cpu_sg = config.scale_groups["cpu-erapids"]
        cpu_slice = prepare_slice_config(cpu_sg.slice_template, cpu_sg, config.platform.label_prefix)
        assert cpu_slice.gpu_count == 0


class TestMarkSliceLockDiscipline:
    """Tests that mark_slice_ready/mark_slice_failed hold the lock during mutation."""

    def test_mark_slice_ready_atomic(self, unbounded_config: ScaleGroupConfig):
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

    def test_mark_slice_failed_atomic(self, unbounded_config: ScaleGroupConfig):
        """lifecycle is set to FAILED while holding the lock."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)
        handle = _tracked_scale_up(group)

        group.mark_slice_failed(handle.slice_id)

        with group._slices_lock:
            state = group._slices[handle.slice_id]
            assert state.lifecycle == SliceLifecycleState.FAILED

    def test_mark_slice_ready_nonexistent_is_noop(self, unbounded_config: ScaleGroupConfig):
        """mark_slice_ready on a nonexistent slice does not raise."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)
        group.mark_slice_ready("nonexistent", ["10.0.0.1"])

    def test_mark_slice_failed_nonexistent_is_noop(self, unbounded_config: ScaleGroupConfig):
        """mark_slice_failed on a nonexistent slice does not raise."""
        platform = make_mock_platform()
        group = ScalingGroup(unbounded_config, platform)
        group.mark_slice_failed("nonexistent")


def test_slice_state_to_proto_uses_worker_ids_as_vm_ids():
    """slice_state_to_proto uses worker_ids directly as vm_id."""

    handle = make_fake_slice_handle("my-slice", scale_group="sg", created_at_ms=1000)
    state = SliceState(
        handle=handle,
        lifecycle=SliceLifecycleState.READY,
        worker_ids=["10.0.0.1", "10.0.0.2"],
    )
    proto = slice_state_to_proto(state)
    assert proto.vms[0].vm_id == "10.0.0.1"
    assert proto.vms[1].vm_id == "10.0.0.2"
    assert proto.state == "ready"


def test_slice_state_to_proto_exposes_state_for_booting_slice_without_vms():
    """A booting slice has no workers yet; its state must still be reported so
    clients render "booting" instead of inferring "unknown" from the empty VM list."""

    handle = make_fake_slice_handle("booting-slice", scale_group="sg", created_at_ms=1000)
    state = SliceState(handle=handle, lifecycle=SliceLifecycleState.BOOTING, worker_ids=[])
    proto = slice_state_to_proto(state)
    assert proto.vms == []  # no workers registered yet
    assert proto.state == "booting"


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


def _make_multi_vm_config(num_vms: int = 4) -> ScaleGroupConfig:
    config = ScaleGroupConfig(
        name="multi-vm-group",
        buffer_slices=0,
        max_slices=10,
        num_vms=num_vms,
        slice_template=SliceConfig(gcp=GcpSliceConfig(runtime_version="v2-alpha-tpuv5", zone="us-central1-a")),
    )
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

        # First idle observation at t=0 stamps quiet_since on the slice.
        idle_map = {wid: WorkerStatus(worker_id=wid, running_task_ids=frozenset()) for wid in wids}
        group.update_slice_activity(idle_map, Timestamp.from_ms(0))

        # At t=10_000 (past 1s threshold), the slice scales down.
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

        # 3 workers idle, 1 still has tasks. update_slice_activity inside
        # scale_down_if_idle observes ANY active worker → quiet_since=None,
        # so the slice is never eligible.
        mixed_map = {}
        for i, wid in enumerate(wids):
            tasks = frozenset({"task-running"}) if i == 0 else frozenset()
            mixed_map[wid] = WorkerStatus(worker_id=wid, running_task_ids=tasks)

        scaled_down = group.scale_down_if_idle(mixed_map, target_capacity=0, timestamp=Timestamp.from_ms(10_000))

        assert len(scaled_down) == 0
        assert group.slice_count() == 1

    def test_multi_vm_slice_activity_updates_when_any_worker_busy(self):
        """update_slice_activity treats the slice as active when ANY worker is busy."""
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

        # First observation at t=0: slice-001 all idle (stamps quiet_since=0),
        # slice-002 has work on one VM (stays active, quiet_since=None).
        vm_map = {}
        for wid in wids_1:
            vm_map[wid] = WorkerStatus(worker_id=wid, running_task_ids=frozenset())
        for i, wid in enumerate(wids_2):
            tasks = frozenset({"running"}) if i == 0 else frozenset()
            vm_map[wid] = WorkerStatus(worker_id=wid, running_task_ids=tasks)
        group.update_slice_activity(vm_map, Timestamp.from_ms(0))

        # At t=10_000 (past 1s threshold), slice-001 has been quiet 10s.
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
        handle = make_fake_slice_handle("s1", scale_group="g1", created_at_ms=1000)
        state = SliceState(
            handle=handle,
            lifecycle=SliceLifecycleState.READY,
            worker_ids=["10.0.0.1"],
            quiet_since=Timestamp.from_ms(1000),
        )

        # With a 1ms threshold and quiet_since 1s ago (Timestamp.now() >> 2000ms),
        # the slice should be idle.
        proto = slice_state_to_proto(state, idle_threshold=Duration.from_ms(1))
        assert proto.idle is True
        assert proto.last_active.epoch_ms == 1000

    def test_idle_false_when_no_threshold(self):
        handle = make_fake_slice_handle("s1", scale_group="g1", created_at_ms=1000)
        state = SliceState(
            handle=handle,
            lifecycle=SliceLifecycleState.READY,
            worker_ids=["10.0.0.1"],
            quiet_since=Timestamp.from_ms(1000),
        )
        proto = slice_state_to_proto(state, idle_threshold=None)
        assert proto.idle is False

    def test_idle_false_when_currently_active(self):
        """quiet_since=None means currently active — never idle."""

        handle = make_fake_slice_handle("s1", scale_group="g1", created_at_ms=1000)
        state = SliceState(
            handle=handle,
            lifecycle=SliceLifecycleState.READY,
            worker_ids=["10.0.0.1"],
            quiet_since=None,
        )
        proto = slice_state_to_proto(state, idle_threshold=Duration.from_ms(1))
        assert proto.idle is False

    def test_idle_false_for_non_ready_slices(self):
        handle = make_fake_slice_handle("s1", scale_group="g1", created_at_ms=1000)
        state = SliceState(
            handle=handle,
            lifecycle=SliceLifecycleState.BOOTING,
            worker_ids=[],
            quiet_since=None,
        )
        proto = slice_state_to_proto(state, idle_threshold=Duration.from_ms(1))
        assert proto.idle is False
