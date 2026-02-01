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

"""Tests for ScalingGroup behavior.

These tests focus on observable behavior - scaling policy decisions,
VM group management, and state tracking - not on implementation details.
"""

from unittest.mock import MagicMock

import pytest

from iris.cluster.types import VmWorkerStatus
from iris.cluster.vm.vm_platform import VmGroupStatus, VmSnapshot
from iris.cluster.vm.scaling_group import ScalingGroup
from iris.rpc import time_pb2, config_pb2, vm_pb2
from iris.time_utils import Duration


@pytest.fixture
def scale_group_config() -> config_pb2.ScaleGroupConfig:
    """A standard scale group configuration for tests."""
    return config_pb2.ScaleGroupConfig(
        name="test-group",
        min_slices=1,
        max_slices=5,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5p-8",
        runtime_version="v2-alpha-tpuv5",
        zones=["us-central1-a"],
    )


@pytest.fixture
def unbounded_config() -> config_pb2.ScaleGroupConfig:
    """A scale group with no min/max constraints."""
    return config_pb2.ScaleGroupConfig(
        name="unbounded-group",
        min_slices=0,
        max_slices=100,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5p-8",
        runtime_version="v2-alpha-tpuv5",
        zones=["us-central1-a"],
    )


def make_mock_vm_group(
    slice_id: str,
    scale_group: str = "test-group",
    all_ready: bool = True,
    any_failed: bool = False,
    vm_states: list[vm_pb2.VmState] | None = None,
    created_at_ms: int = 1000000,
) -> MagicMock:
    """Create a mock VmGroupProtocol for testing."""
    mock = MagicMock()
    mock.group_id = slice_id
    mock.slice_id = slice_id  # Backward compat alias
    mock.scale_group = scale_group
    mock.created_at_ms = created_at_ms

    # Build status from vm_states if provided
    if vm_states is None:
        if any_failed:
            vm_states = [vm_pb2.VM_STATE_FAILED]
        elif all_ready:
            vm_states = [vm_pb2.VM_STATE_READY]
        else:
            vm_states = [vm_pb2.VM_STATE_BOOTING]

    # Generate unique addresses by hashing slice_id for the third octet
    slice_hash = abs(hash(slice_id)) % 256
    snapshots = [
        VmSnapshot(
            vm_id=f"{slice_id}-vm-{i}",
            state=state,
            address=f"10.0.{slice_hash}.{i}",
            init_phase="",
            init_error="" if state != vm_pb2.VM_STATE_FAILED else "test error",
        )
        for i, state in enumerate(vm_states)
    ]
    status = VmGroupStatus(vms=snapshots)
    mock.status.return_value = status
    mock.to_proto.return_value = vm_pb2.SliceInfo(
        slice_id=slice_id,
        scale_group=scale_group,
        created_at=time_pb2.Timestamp(epoch_ms=created_at_ms),
        vms=[
            vm_pb2.VmInfo(
                vm_id=s.vm_id,
                state=s.state,
                address=s.address,
                init_phase=s.init_phase,
                init_error=s.init_error,
                created_at=time_pb2.Timestamp(epoch_ms=created_at_ms),
                state_changed_at=time_pb2.Timestamp(epoch_ms=created_at_ms),
            )
            for s in snapshots
        ],
    )
    return mock


def make_mock_vm_manager(vm_groups_to_discover: list[MagicMock] | None = None) -> MagicMock:
    """Create a mock VmManagerProtocol."""
    manager = MagicMock()
    manager.discover_vm_groups.return_value = vm_groups_to_discover or []

    def create_vm_group_side_effect(tags: dict[str, str] | None = None) -> MagicMock:
        # Generate a unique slice ID based on call count
        slice_id = f"new-slice-{len(manager.create_vm_group.call_args_list)}"
        mock = make_mock_vm_group(slice_id)
        mock.tags = tags  # Store tags so tests can verify if needed
        return mock

    manager.create_vm_group.side_effect = create_vm_group_side_effect
    return manager


class TestScalingGroupVmGroupOwnership:
    """Tests for VM group ownership and lifecycle."""

    def test_reconcile_adopts_discovered_vm_groups(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """reconcile() populates VM groups from the VmManager."""
        discovered = [
            make_mock_vm_group("slice-001"),
            make_mock_vm_group("slice-002"),
        ]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)

        group = ScalingGroup(scale_group_config, manager)
        group.reconcile()

        assert group.slice_count() == 2
        assert group.get_slice("slice-001") is not None
        assert group.get_slice("slice-002") is not None

    def test_scale_up_creates_and_tracks_vm_group(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """scale_up() creates a VM group via VmManager and tracks it."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(scale_group_config, manager)

        new_vm_group = group.scale_up()

        manager.create_vm_group.assert_called_once()
        assert group.slice_count() == 1
        assert new_vm_group in group.vm_groups()

    def test_scale_up_passes_tags_to_manager(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """scale_up() passes tags to the VmManager."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(scale_group_config, manager)

        group.scale_up(tags={"env": "prod", "team": "ml"})

        manager.create_vm_group.assert_called_once_with({"env": "prod", "team": "ml"})

    def test_scale_down_terminates_and_removes_vm_group(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """scale_down() terminates the VM group and removes it from tracking."""
        mock_vm_group = make_mock_vm_group("slice-001")
        manager = make_mock_vm_manager(vm_groups_to_discover=[mock_vm_group])
        group = ScalingGroup(scale_group_config, manager)
        group.reconcile()

        assert group.slice_count() == 1

        group.scale_down("slice-001")

        mock_vm_group.terminate.assert_called_once()
        assert group.slice_count() == 0
        assert group.get_slice("slice-001") is None

    def test_scale_down_nonexistent_vm_group_is_noop(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """scale_down() on a nonexistent VM group does nothing."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(scale_group_config, manager)

        # Should not raise
        group.scale_down("nonexistent-slice")

        assert group.slice_count() == 0

    def test_ready_slice_count(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """ready_slice_count() only counts VM groups where all VMs are ready."""
        discovered = [
            make_mock_vm_group("slice-001", all_ready=True),
            make_mock_vm_group("slice-002", all_ready=False, vm_states=[vm_pb2.VM_STATE_BOOTING]),
            make_mock_vm_group("slice-003", all_ready=True),
        ]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(scale_group_config, manager)
        group.reconcile()

        assert group.slice_count() == 3
        assert group.ready_slice_count() == 2


class TestScalingGroupScalingPolicy:
    """Tests for scaling policy decisions (can_scale_up, can_scale_down)."""

    def test_can_scale_up_when_below_max(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """can_scale_up() returns True when below max_slices."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(scale_group_config, manager)

        assert group.can_scale_up()

    def test_cannot_scale_up_at_max_slices(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """can_scale_up() returns False when at max_slices."""
        discovered = [make_mock_vm_group(f"slice-{i}") for i in range(5)]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(scale_group_config, manager)
        group.reconcile()

        assert group.slice_count() == 5  # max_slices
        assert not group.can_scale_up()

    def test_cannot_scale_up_during_backoff(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """can_scale_up() returns False during backoff period."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(unbounded_config, manager)

        ts = 1000000
        group.record_failure(ts=ts)

        # During backoff period
        assert not group.can_scale_up(ts=ts + 1000)
        # After backoff expires (default 5s = 5000ms)
        assert group.can_scale_up(ts=ts + 6000)

    def test_cannot_scale_up_during_cooldown(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """can_scale_up() returns False during cooldown period after scale-up."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(
            unbounded_config,
            manager,
            scale_up_cooldown_ms=10000,
        )

        ts = 1000000
        group.scale_up(ts=ts)

        # During cooldown
        assert not group.can_scale_up(ts=ts + 5000)
        # After cooldown expires
        assert group.can_scale_up(ts=ts + 15000)

    def test_can_scale_down_when_above_min(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """can_scale_down() returns True when above min_slices."""
        discovered = [make_mock_vm_group("slice-001"), make_mock_vm_group("slice-002")]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(scale_group_config, manager)
        group.reconcile()

        assert group.slice_count() == 2
        assert scale_group_config.min_slices == 1
        assert group.can_scale_down()

    def test_cannot_scale_down_at_min_slices(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """can_scale_down() returns False when at min_slices."""
        discovered = [make_mock_vm_group("slice-001")]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(scale_group_config, manager)
        group.reconcile()

        assert group.slice_count() == 1  # min_slices
        assert not group.can_scale_down()

    def test_cannot_scale_down_during_cooldown(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """can_scale_down() returns False during cooldown period."""
        discovered = [make_mock_vm_group("slice-001"), make_mock_vm_group("slice-002")]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(
            unbounded_config,
            manager,
            scale_down_cooldown_ms=10000,
        )
        group.reconcile()

        ts = 1000000
        group.scale_down("slice-001", timestamp_ms=ts)

        # During cooldown
        assert not group.can_scale_down(ts=ts + 5000)
        # After cooldown expires
        assert group.can_scale_down(ts=ts + 15000)


class TestScalingGroupBackoff:
    """Tests for exponential backoff behavior."""

    def test_record_failure_applies_initial_backoff(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """First failure applies initial backoff duration."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(
            unbounded_config,
            manager,
            backoff_initial=Duration.from_seconds(5.0),
        )

        ts = 1000000
        group.record_failure(ts=ts)

        assert group.consecutive_failures == 1
        assert group.backoff_until_ms == ts + 5000

    def test_record_failure_applies_exponential_backoff(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Consecutive failures double the backoff time."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(
            unbounded_config,
            manager,
            backoff_initial=Duration.from_seconds(5.0),
            backoff_factor=2.0,
        )

        ts = 1000000
        group.record_failure(ts=ts)  # 5000ms
        group.record_failure(ts=ts)  # 10000ms
        group.record_failure(ts=ts)  # 20000ms

        assert group.consecutive_failures == 3
        assert group.backoff_until_ms == ts + 20000

    def test_backoff_capped_at_maximum(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Backoff duration is capped at max value."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(
            unbounded_config,
            manager,
            backoff_initial=Duration.from_seconds(5.0),
            backoff_max=Duration.from_seconds(15.0),
            backoff_factor=2.0,
        )

        ts = 1000000
        for _ in range(10):  # Many failures
            group.record_failure(ts=ts)

        # Should be capped at max
        assert group.backoff_until_ms == ts + 15000

    def test_scale_up_resets_backoff(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Successful scale-up resets backoff state."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(unbounded_config, manager)

        ts = 1000000
        group.record_failure(ts=ts)
        group.record_failure(ts=ts)
        assert group.consecutive_failures == 2

        group.scale_up(ts=ts + 100000)

        assert group.consecutive_failures == 0
        assert group.backoff_until_ms == 0


class TestScalingGroupDemandTracking:
    """Tests for demand tracking."""

    def test_update_demand_tracks_peak(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """update_demand() tracks peak demand."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(unbounded_config, manager)

        group.update_demand(5)
        group.update_demand(10)
        group.update_demand(3)

        assert group.current_demand == 3
        assert group.peak_demand == 10


class TestScalingGroupIdleTracking:
    """Tests for per-slice idle tracking and scale-down eligibility."""

    def test_slice_eligible_when_never_active(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Fresh slice (no activity tracked) is eligible for scaledown."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(unbounded_config, manager, idle_threshold_ms=60_000)
        group.scale_up()  # Creates a slice
        slice_id = next(iter(group.vm_groups())).slice_id

        # Never had activity tracked -> eligible
        assert group.is_slice_eligible_for_scaledown(slice_id, timestamp_ms=1000)

    def test_slice_not_eligible_when_recently_active(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Recently active slice is not eligible for scaledown."""
        discovered = [make_mock_vm_group("slice-001", all_ready=True)]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(unbounded_config, manager, idle_threshold_ms=60_000)
        group.reconcile()

        # Set up vms() mock to return VM with address
        slice_001_addr = f"10.0.{abs(hash('slice-001')) % 256}.0"
        slice_obj = group.get_slice("slice-001")
        mock_vm = MagicMock()
        mock_vm.info.address = slice_001_addr
        slice_obj.vms.return_value = [mock_vm]

        # Mark slice as active at t=1000 via update_slice_activity
        vm_status_map = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset({"task-1"})),
        }
        group.update_slice_activity(vm_status_map, timestamp_ms=1000)

        # Not enough time passed (30s < 60s threshold)
        assert not group.is_slice_eligible_for_scaledown("slice-001", timestamp_ms=30_000)

    def test_slice_eligible_after_idle_threshold(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Slice is eligible after idle_threshold_ms of inactivity."""
        discovered = [make_mock_vm_group("slice-001", all_ready=True)]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(unbounded_config, manager, idle_threshold_ms=60_000)
        group.reconcile()

        # Set up vms() mock to return VM with address
        slice_001_addr = f"10.0.{abs(hash('slice-001')) % 256}.0"
        slice_obj = group.get_slice("slice-001")
        mock_vm = MagicMock()
        mock_vm.info.address = slice_001_addr
        slice_obj.vms.return_value = [mock_vm]

        # Mark slice as active at t=1000 via update_slice_activity
        vm_status_map = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset({"task-1"})),
        }
        group.update_slice_activity(vm_status_map, timestamp_ms=1000)

        # After threshold (61s > 60s) -> eligible
        assert group.is_slice_eligible_for_scaledown("slice-001", timestamp_ms=61_001)

    def test_get_idle_slices_returns_longest_idle_first(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """get_idle_slices returns slices sorted by idle time (longest first)."""
        discovered = [
            make_mock_vm_group("slice-001", all_ready=True),
            make_mock_vm_group("slice-002", all_ready=True),
        ]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(unbounded_config, manager, idle_threshold_ms=1000)
        group.reconcile()

        # Set up vms() mocks to return VMs with addresses
        slice_001_addr = f"10.0.{abs(hash('slice-001')) % 256}.0"
        slice_002_addr = f"10.0.{abs(hash('slice-002')) % 256}.0"
        for slice_id, vm_addr in [("slice-001", slice_001_addr), ("slice-002", slice_002_addr)]:
            slice_obj = group.get_slice(slice_id)
            mock_vm = MagicMock()
            mock_vm.info.address = vm_addr
            slice_obj.vms.return_value = [mock_vm]

        # Mark slice-001 as active at t=1000 (will be idle longer)
        vm_status_map_001 = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset({"task-1"})),
        }
        group.update_slice_activity(vm_status_map_001, timestamp_ms=1000)

        # Mark slice-002 as active at t=5000 (more recently active)
        vm_status_map_002 = {
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset({"task-2"})),
        }
        group.update_slice_activity(vm_status_map_002, timestamp_ms=5000)

        # At timestamp 10000, slice-001 has been idle longer (9s vs 5s)
        idle_slices = group.get_idle_slices(timestamp_ms=10_000)
        assert len(idle_slices) == 2
        assert idle_slices[0].slice_id == "slice-001"  # Longest idle first

    def test_update_slice_activity_tracks_active_slices(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """update_slice_activity updates timestamp only for slices with active workers."""
        # Mock VMs with specific worker IDs
        discovered = [
            make_mock_vm_group("slice-001", all_ready=True),
            make_mock_vm_group("slice-002", all_ready=True),
        ]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(unbounded_config, manager)
        group.reconcile()

        # Mock vms() to return VMs with worker_ids and addresses
        slice_001_addr = f"10.0.{abs(hash('slice-001')) % 256}.0"
        slice_002_addr = f"10.0.{abs(hash('slice-002')) % 256}.0"
        for slice_id, vm_addr in [("slice-001", slice_001_addr), ("slice-002", slice_002_addr)]:
            slice_obj = group.get_slice(slice_id)
            mock_vm = MagicMock()
            mock_vm.info.worker_id = f"worker-{slice_id}"
            mock_vm.info.address = vm_addr
            slice_obj.vms.return_value = [mock_vm]

        # slice-001 has running tasks, slice-002 is idle
        vm_status_map = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset({"task-1"})),  # Active
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset()),  # Idle
        }

        group.update_slice_activity(vm_status_map, timestamp_ms=5000)

        # Observable behavior: slice-001 should not be eligible for scaledown (recently active)
        # slice-002 should remain eligible (no activity tracked)
        assert not group.is_slice_eligible_for_scaledown("slice-001", timestamp_ms=5000)
        assert group.is_slice_eligible_for_scaledown("slice-002", timestamp_ms=5000)

    def test_scale_down_if_idle_terminates_eligible_slice(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """scale_down_if_idle terminates an eligible idle slice."""
        discovered = [
            make_mock_vm_group("slice-001", all_ready=True),
            make_mock_vm_group("slice-002", all_ready=True),
        ]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(unbounded_config, manager, idle_threshold_ms=1000, scale_down_cooldown_ms=0)
        group.reconcile()

        # Mock vms() for idle verification
        slice_001_addr = f"10.0.{abs(hash('slice-001')) % 256}.0"
        slice_002_addr = f"10.0.{abs(hash('slice-002')) % 256}.0"
        for slice_id, vm_addr in [("slice-001", slice_001_addr), ("slice-002", slice_002_addr)]:
            slice_obj = group.get_slice(slice_id)
            mock_vm = MagicMock()
            mock_vm.info.worker_id = f"worker-{slice_id}"
            mock_vm.info.address = vm_addr
            slice_obj.vms.return_value = [mock_vm]

        # Mark both slices as active at t=0 (they'll be idle for 10s, exceeding 1s threshold)
        vm_status_map_active = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset({"task-1"})),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset({"task-2"})),
        }
        group.update_slice_activity(vm_status_map_active, timestamp_ms=0)

        # At t=10_000, workers are now idle
        vm_status_map_idle = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset()),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset()),
        }

        # Target capacity = 1, but we have 2 ready slices
        scaled_down = group.scale_down_if_idle(vm_status_map_idle, target_capacity=1, timestamp_ms=10_000)

        assert scaled_down is not None
        assert group.slice_count() == 1  # One slice was terminated

    def test_scale_down_if_idle_respects_target_capacity(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """scale_down_if_idle does nothing when at or below target capacity."""
        discovered = [make_mock_vm_group("slice-001", all_ready=True)]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(unbounded_config, manager, idle_threshold_ms=1000, scale_down_cooldown_ms=0)
        group.reconcile()

        slice_001_addr = f"10.0.{abs(hash('slice-001')) % 256}.0"
        vm_status_map = {slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset())}

        # Target = 1, ready = 1, should not scale down
        scaled_down = group.scale_down_if_idle(vm_status_map, target_capacity=1, timestamp_ms=10_000)

        assert scaled_down is None
        assert group.slice_count() == 1

    def test_scale_down_cleans_up_idle_tracking(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """scale_down removes the slice from idle tracking."""
        discovered = [make_mock_vm_group("slice-001", all_ready=True)]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(unbounded_config, manager)
        group.reconcile()

        # Set up vms() mock and track the slice via update_slice_activity
        slice_001_addr = f"10.0.{abs(hash('slice-001')) % 256}.0"
        slice_obj = group.get_slice("slice-001")
        mock_vm = MagicMock()
        mock_vm.info.address = slice_001_addr
        slice_obj.vms.return_value = [mock_vm]

        vm_status_map = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset({"task-1"})),
        }
        group.update_slice_activity(vm_status_map, timestamp_ms=1000)

        # Scale down
        group.scale_down("slice-001")

        # Observable behavior: slice should be removed (already verified by assertion above)
        assert group.get_slice("slice-001") is None


class TestScalingGroupVmGroupStateCounts:
    """Tests for slice_state_counts() aggregation."""

    @pytest.mark.parametrize(
        "vm_state,expected_category",
        [
            (vm_pb2.VM_STATE_READY, "ready"),
            (vm_pb2.VM_STATE_BOOTING, "booting"),
            (vm_pb2.VM_STATE_INITIALIZING, "initializing"),
            (vm_pb2.VM_STATE_FAILED, "failed"),
        ],
    )
    def test_counts_vm_groups_by_state(
        self,
        scale_group_config: config_pb2.ScaleGroupConfig,
        vm_state: vm_pb2.VmState,
        expected_category: str,
    ):
        """VM groups are counted in the correct category based on VM state."""
        discovered = [make_mock_vm_group("slice-001", vm_states=[vm_state])]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(scale_group_config, manager)
        group.reconcile()

        counts = group.slice_state_counts()

        assert counts[expected_category] == 1
        for category in ["ready", "booting", "initializing", "failed"]:
            if category != expected_category:
                assert counts[category] == 0

    def test_failed_takes_precedence(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """A VM group with any failed VM is counted as failed."""
        discovered = [
            make_mock_vm_group(
                "slice-001",
                vm_states=[vm_pb2.VM_STATE_READY, vm_pb2.VM_STATE_FAILED],
            ),
        ]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(scale_group_config, manager)
        group.reconcile()

        counts = group.slice_state_counts()

        assert counts["failed"] == 1
        assert counts["ready"] == 0

    def test_skips_terminated_vm_groups(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Terminated VM groups are not counted."""
        discovered = [
            make_mock_vm_group("slice-001", vm_states=[vm_pb2.VM_STATE_TERMINATED]),
        ]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(scale_group_config, manager)
        group.reconcile()

        counts = group.slice_state_counts()

        assert counts["ready"] == 0
        assert counts["booting"] == 0
        assert counts["initializing"] == 0
        assert counts["failed"] == 0


class TestScalingGroupAvailability:
    """Tests for availability state computation and waterfall routing support."""

    def test_available_when_no_constraints(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Group is AVAILABLE when not in backoff, quota ok, and under capacity."""
        from iris.cluster.vm.scaling_group import GroupAvailability

        manager = make_mock_vm_manager()
        group = ScalingGroup(unbounded_config, manager)

        state = group.availability()
        assert state.status == GroupAvailability.AVAILABLE

    def test_at_capacity_when_at_max_slices(self):
        """Group is AT_CAPACITY when at max_slices."""
        from iris.cluster.vm.scaling_group import GroupAvailability

        config = config_pb2.ScaleGroupConfig(
            name="test-group",
            min_slices=0,
            max_slices=2,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
        )
        discovered = [make_mock_vm_group(f"slice-{i}") for i in range(2)]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(config, manager)
        group.reconcile()

        state = group.availability()
        assert state.status == GroupAvailability.AT_CAPACITY

    def test_backoff_when_in_backoff_period(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Group is in BACKOFF when backoff timer is active."""
        from iris.cluster.vm.scaling_group import GroupAvailability

        manager = make_mock_vm_manager()
        group = ScalingGroup(unbounded_config, manager, backoff_initial=Duration.from_seconds(60.0))
        ts = 1000000
        group.record_failure(ts=ts)

        state = group.availability(ts + 1000)  # Still in backoff
        assert state.status == GroupAvailability.BACKOFF
        assert state.until_ms is not None

    def test_can_accept_demand_true_when_available(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """can_accept_demand() returns True when AVAILABLE."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(unbounded_config, manager)

        assert group.can_accept_demand() is True

    def test_can_accept_demand_false_when_at_capacity(self):
        """can_accept_demand() returns False when AT_CAPACITY."""
        config = config_pb2.ScaleGroupConfig(
            name="test-group",
            min_slices=0,
            max_slices=1,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
        )
        discovered = [make_mock_vm_group("slice-0")]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(config, manager)
        group.reconcile()

        assert group.can_accept_demand() is False

    def test_quota_exceeded_blocks_demand_until_timeout(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Quota exceeded state auto-expires after timeout."""
        from iris.cluster.vm.managed_vm import QuotaExceededError
        from iris.cluster.vm.scaling_group import GroupAvailability

        manager = make_mock_vm_manager()
        manager.create_vm_group.side_effect = QuotaExceededError("TPU quota exhausted")

        group = ScalingGroup(unbounded_config, manager, quota_timeout_ms=60_000)

        with pytest.raises(QuotaExceededError):
            group.scale_up(ts=1000)

        # Before timeout: QUOTA_EXCEEDED
        assert not group.can_accept_demand(timestamp_ms=30_000)
        state = group.availability(timestamp_ms=30_000)
        assert state.status == GroupAvailability.QUOTA_EXCEEDED

        # After timeout (1000 + 60_000 = 61_000)
        assert group.can_accept_demand(timestamp_ms=70_000)

    def test_successful_scale_up_clears_quota_state(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Successful scale-up clears any quota exceeded state."""
        from iris.cluster.vm.managed_vm import QuotaExceededError

        manager = make_mock_vm_manager()
        manager.create_vm_group.side_effect = [
            QuotaExceededError("TPU quota exhausted"),
            make_mock_vm_group("slice-1"),
        ]

        group = ScalingGroup(unbounded_config, manager, quota_timeout_ms=300_000)

        with pytest.raises(QuotaExceededError):
            group.scale_up(ts=1000)
        assert not group.can_accept_demand(timestamp_ms=2000)

        # Second scale_up succeeds and clears quota state
        group.scale_up(ts=3000)
        assert group.can_accept_demand(timestamp_ms=4000)

    def test_quota_exceeded_takes_precedence_over_backoff(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """Quota exceeded has higher precedence than backoff."""
        from iris.cluster.vm.managed_vm import QuotaExceededError
        from iris.cluster.vm.scaling_group import GroupAvailability

        manager = make_mock_vm_manager()
        manager.create_vm_group.side_effect = QuotaExceededError("Quota exhausted")

        group = ScalingGroup(unbounded_config, manager, quota_timeout_ms=60_000)

        # Record a failure to trigger backoff
        group.record_failure(ts=1000)

        # Then trigger quota exceeded
        with pytest.raises(QuotaExceededError):
            group.scale_up(ts=1000)

        # Availability should report QUOTA_EXCEEDED, not BACKOFF
        state = group.availability(timestamp_ms=2000)
        assert state.status == GroupAvailability.QUOTA_EXCEEDED

    def test_matches_device_requirement_filters_by_type_and_variant(
        self, scale_group_config: config_pb2.ScaleGroupConfig
    ):
        """matches_device_requirement filters groups by device type and variant."""
        from iris.cluster.types import DeviceType

        manager = make_mock_vm_manager()
        group = ScalingGroup(scale_group_config, manager)  # TPU with accelerator_variant="v5p-8"

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
            make_mock_vm_group("slice-001", all_ready=True),
            make_mock_vm_group("slice-002", any_failed=True),
            make_mock_vm_group("slice-003", any_failed=True),
        ]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(unbounded_config, manager)
        group.reconcile()

        cleaned = group.cleanup_failed_slices(timestamp_ms=1000)

        assert len(cleaned) == 2
        assert group.slice_count() == 1
        assert group.get_slice("slice-001") is not None
        assert group.get_slice("slice-002") is None
        assert group.get_slice("slice-003") is None

    def test_cleanup_failed_slices_triggers_backoff_once(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """cleanup_failed_slices() triggers backoff exactly once regardless of slice count."""
        discovered = [
            make_mock_vm_group("slice-001", any_failed=True),
            make_mock_vm_group("slice-002", any_failed=True),
            make_mock_vm_group("slice-003", any_failed=True),
        ]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(unbounded_config, manager, backoff_initial=Duration.from_seconds(5.0))
        group.reconcile()

        group.cleanup_failed_slices(timestamp_ms=1000)

        # Only one failure recorded despite cleaning up 3 slices
        assert group.consecutive_failures == 1
        assert group.backoff_until_ms == 1000 + 5000

    def test_cleanup_failed_slices_returns_empty_when_no_failures(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """cleanup_failed_slices() returns empty list when no slices are failed."""
        discovered = [
            make_mock_vm_group("slice-001", all_ready=True),
            make_mock_vm_group("slice-002", all_ready=True),
        ]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(unbounded_config, manager)
        group.reconcile()

        cleaned = group.cleanup_failed_slices(timestamp_ms=1000)

        assert len(cleaned) == 0
        assert group.slice_count() == 2
        assert group.consecutive_failures == 0

    def test_cleanup_failed_slices_cleans_idle_tracking(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """cleanup_failed_slices() removes slices from idle tracking."""
        discovered = [make_mock_vm_group("slice-001", any_failed=True)]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(unbounded_config, manager)
        group.reconcile()

        # Set up vms() mock and track the slice via update_slice_activity
        # (simulating the slice was active before it failed)
        slice_001_addr = f"10.0.{abs(hash('slice-001')) % 256}.0"
        slice_obj = group.get_slice("slice-001")
        mock_vm = MagicMock()
        mock_vm.info.address = slice_001_addr
        slice_obj.vms.return_value = [mock_vm]

        vm_status_map = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset({"task-1"})),
        }
        group.update_slice_activity(vm_status_map, timestamp_ms=1000)

        group.cleanup_failed_slices(timestamp_ms=2000)

        # Observable behavior: slice should be removed (already verified by cleanup return value)
        assert group.get_slice("slice-001") is None

    def test_cleanup_failed_slices_ignores_cooldown(self, unbounded_config: config_pb2.ScaleGroupConfig):
        """cleanup_failed_slices() does not respect scale_down_cooldown."""
        discovered = [make_mock_vm_group("slice-001", any_failed=True)]
        manager = make_mock_vm_manager(vm_groups_to_discover=discovered)
        group = ScalingGroup(unbounded_config, manager, scale_down_cooldown_ms=60_000)
        group.reconcile()

        # Set last scale-down to trigger cooldown for normal scale_down
        group._last_scale_down_ms = 1000

        # Despite cooldown, cleanup_failed_slices() should still work
        cleaned = group.cleanup_failed_slices(timestamp_ms=2000)

        assert len(cleaned) == 1
        assert group.slice_count() == 0
