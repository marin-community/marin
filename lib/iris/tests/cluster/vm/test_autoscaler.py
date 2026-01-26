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

"""Tests for Autoscaler behavior.

These tests focus on observable behavior - scaling decisions based on demand,
execution of those decisions, and integration with ScalingGroup and VmRegistry.
"""

from unittest.mock import MagicMock

import pytest

from iris.cluster.types import VmWorkerStatus
from iris.cluster.vm.autoscaler import (
    Autoscaler,
    AutoscalerConfig,
    DemandEntry,
    ScalingAction,
    ScalingDecision,
)
from iris.cluster.vm.managed_vm import VmRegistry
from iris.cluster.vm.vm_platform import VmGroupStatus, VmSnapshot
from iris.cluster.vm.scaling_group import ScalingGroup
from iris.rpc import vm_pb2

# --- Test fixtures and helpers ---


def make_mock_slice(
    slice_id: str,
    scale_group: str = "test-group",
    all_ready: bool = False,
    any_failed: bool = False,
    vm_states: list[vm_pb2.VmState] | None = None,
    created_at_ms: int = 1000000,
    worker_ids: list[str] | None = None,
) -> MagicMock:
    """Create a mock VmGroupProtocol for testing.

    Note: all_ready and any_failed are mutually exclusive. Set only one to True.
    If neither is set, creates a booting slice.
    """
    mock = MagicMock()
    mock.group_id = slice_id
    mock.slice_id = slice_id
    mock.scale_group = scale_group
    mock.created_at_ms = created_at_ms

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

    # Mock vms() to return ManagedVm-like mocks
    worker_ids = worker_ids or [f"worker-{slice_id}-{i}" for i in range(len(vm_states))]
    mock_vms = []
    for i, (state, worker_id) in enumerate(zip(vm_states, worker_ids, strict=True)):
        vm_mock = MagicMock()
        vm_mock.info = vm_pb2.VmInfo(
            vm_id=f"{slice_id}-vm-{i}",
            state=state,
            address=f"10.0.{slice_hash}.{i}",
            worker_id=worker_id,
        )
        mock_vms.append(vm_mock)
    mock.vms.return_value = mock_vms

    mock.to_proto.return_value = vm_pb2.SliceInfo(
        slice_id=slice_id,
        scale_group=scale_group,
        created_at_ms=created_at_ms,
        vms=[vm.info for vm in mock_vms],
    )
    return mock


def make_mock_vm_manager(slices_to_discover: list[MagicMock] | None = None) -> MagicMock:
    """Create a mock VmManagerProtocol."""
    manager = MagicMock()
    manager.discover_vm_groups.return_value = slices_to_discover or []
    manager._create_count = 0

    def create_vm_group_side_effect(_tags: dict[str, str] | None = None) -> MagicMock:
        manager._create_count += 1
        slice_id = f"new-slice-{manager._create_count}"
        return make_mock_slice(slice_id)

    manager.create_vm_group.side_effect = create_vm_group_side_effect
    return manager


@pytest.fixture
def scale_group_config() -> vm_pb2.ScaleGroupConfig:
    """A standard scale group configuration."""
    return vm_pb2.ScaleGroupConfig(
        name="test-group",
        min_slices=0,
        max_slices=5,
        accelerator_type="v5p-8",
        runtime_version="v2-alpha-tpuv5",
        zones=["us-central1-a"],
    )


def make_autoscaler(
    scale_groups: dict[str, ScalingGroup],
    vm_registry: VmRegistry | None = None,
    config: AutoscalerConfig | None = None,
) -> Autoscaler:
    """Create an Autoscaler with the given groups."""
    return Autoscaler(
        scale_groups=scale_groups,
        vm_registry=vm_registry or VmRegistry(),
        config=config,
    )


# --- Tests for scaling decisions ---


class TestAutoscalerScaleUp:
    """Tests for scale-up decisions."""

    def test_scales_up_when_demand_exceeds_capacity(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Evaluates scale-up when demand > capacity."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0)
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(accelerator_type="v5p-8", count=2)]
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 1
        assert decisions[0].action == ScalingAction.SCALE_UP
        assert decisions[0].scale_group == "test-group"
        assert "demand=2 > capacity=0" in decisions[0].reason

    def test_no_scale_up_when_at_max_slices(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Does not scale up when already at max_slices."""
        discovered = [make_mock_slice(f"slice-{i}") for i in range(5)]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(accelerator_type="v5p-8", count=10)]
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 0

    def test_no_scale_up_when_capacity_meets_demand(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Does not scale up when capacity >= demand."""
        discovered = [
            make_mock_slice("slice-001", all_ready=True),
            make_mock_slice("slice-002", all_ready=True),
        ]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(accelerator_type="v5p-8", count=2)]
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 0

    def test_no_scale_up_during_backoff(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Does not scale up during backoff period."""
        from iris.time_utils import now_ms

        manager = make_mock_vm_manager()
        # Use large backoff so it's still active when we evaluate
        group = ScalingGroup(
            scale_group_config,
            manager,
            scale_up_cooldown_ms=0,
            backoff_initial_ms=3600_000,  # 1 hour backoff
        )
        group.record_failure(ts=now_ms())
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(accelerator_type="v5p-8", count=2)]
        decisions = autoscaler.evaluate(demand)

        # Decision blocked by backoff
        assert len(decisions) == 0

    def test_no_scale_up_during_cooldown(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Does not scale up during cooldown period."""
        from iris.time_utils import now_ms

        manager = make_mock_vm_manager()
        # Use large cooldown so it's still active when we evaluate
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=3600_000)  # 1 hour
        group.scale_up(ts=now_ms())
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(accelerator_type="v5p-8", count=2)]
        decisions = autoscaler.evaluate(demand)

        # Decision blocked by cooldown (only 1 slice at time of evaluate)
        # Since we just called scale_up, cooldown is active
        assert len(decisions) == 0

    def test_scales_up_to_enforce_min_slices(self):
        """Scales up to enforce min_slices even with zero demand."""
        config = vm_pb2.ScaleGroupConfig(
            name="test-group",
            min_slices=2,
            max_slices=5,
            accelerator_type="v5p-8",
            runtime_version="v2-alpha-tpuv5",
            zones=["us-central1-a"],
        )
        manager = make_mock_vm_manager()
        group = ScalingGroup(config, manager, scale_up_cooldown_ms=0)
        autoscaler = make_autoscaler({"test-group": group})

        # No demand, but min_slices=2 means we should scale up
        decisions = autoscaler.evaluate([])

        assert len(decisions) == 1
        assert decisions[0].action == ScalingAction.SCALE_UP
        assert "below min_slices" in decisions[0].reason
        assert "0 < 2" in decisions[0].reason

    def test_no_scale_up_when_at_min_slices(self):
        """Does not scale up when already at min_slices and no demand."""
        config = vm_pb2.ScaleGroupConfig(
            name="test-group",
            min_slices=2,
            max_slices=5,
            accelerator_type="v5p-8",
            runtime_version="v2-alpha-tpuv5",
            zones=["us-central1-a"],
        )
        discovered = [
            make_mock_slice("slice-001", all_ready=True),
            make_mock_slice("slice-002", all_ready=True),
        ]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(config, manager, scale_up_cooldown_ms=0)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        # No demand and already at min_slices=2
        decisions = autoscaler.evaluate([])

        assert len(decisions) == 0

    def test_pending_slices_count_toward_capacity(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Booting/initializing slices count toward capacity."""
        discovered = [
            make_mock_slice("slice-001", vm_states=[vm_pb2.VM_STATE_BOOTING]),
            make_mock_slice("slice-002", vm_states=[vm_pb2.VM_STATE_INITIALIZING]),
        ]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(accelerator_type="v5p-8", count=2)]
        decisions = autoscaler.evaluate(demand)

        # capacity = 2 (pending), demand = 2, so no scale up
        assert len(decisions) == 0


class TestAutoscalerScaleDown:
    """Tests for scale-down behavior (delegated to ScalingGroup)."""

    def test_scales_down_idle_slice_via_run_once(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """run_once() scales down idle slices via ScalingGroup.scale_down_if_idle()."""
        discovered = [
            make_mock_slice("slice-001", all_ready=True, created_at_ms=100000),
            make_mock_slice("slice-002", all_ready=True, created_at_ms=200000),
        ]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(
            scale_group_config,
            manager,
            scale_down_cooldown_ms=0,
            idle_threshold_ms=1000,  # 1 second threshold
        )
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        # Mark slices as idle (never had activity, so eligible for scaledown)
        # The idle_threshold_ms=1000 means slices need to be idle for 1s

        demand = [DemandEntry(accelerator_type="v5p-8", count=1)]

        # All workers are idle
        # VM addresses are generated as f"10.0.{hash(slice_id) % 256}.{i}"
        slice_001_addr = f"10.0.{abs(hash('slice-001')) % 256}.0"
        slice_002_addr = f"10.0.{abs(hash('slice-002')) % 256}.0"
        vm_status_map = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset()),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset()),
        }

        # run_once should scale down via ScalingGroup.scale_down_if_idle
        autoscaler.run_once(demand, vm_status_map, timestamp_ms=10_000)

        # One slice should be terminated (longest idle first, which is slice-001)
        assert group.slice_count() == 1

    def test_no_scale_down_at_min_slices(self):
        """Does not scale down when at min_slices."""
        config = vm_pb2.ScaleGroupConfig(
            name="test-group",
            min_slices=2,
            max_slices=5,
            accelerator_type="v5p-8",
            runtime_version="v2-alpha-tpuv5",
        )
        discovered = [
            make_mock_slice("slice-001", all_ready=True),
            make_mock_slice("slice-002", all_ready=True),
        ]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(config, manager, scale_down_cooldown_ms=0, idle_threshold_ms=0)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(accelerator_type="v5p-8", count=0)]
        slice_001_addr = f"10.0.{abs(hash('slice-001')) % 256}.0"
        slice_002_addr = f"10.0.{abs(hash('slice-002')) % 256}.0"
        vm_status_map = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset()),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset()),
        }

        # target_capacity = max(0, min_slices=2) = 2, ready=2, no scale down
        autoscaler.run_once(demand, vm_status_map)

        # At min_slices, cannot scale down
        assert group.slice_count() == 2

    def test_no_scale_down_until_idle_threshold(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Does not scale down until slice has been idle long enough."""
        discovered = [
            make_mock_slice("slice-001", all_ready=True),
            make_mock_slice("slice-002", all_ready=True),
        ]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(
            scale_group_config,
            manager,
            scale_down_cooldown_ms=0,
            idle_threshold_ms=300_000,  # 5 minute threshold
        )
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        # Mark slices as recently active
        group._slice_last_active["slice-001"] = 1000
        group._slice_last_active["slice-002"] = 1000

        demand = [DemandEntry(accelerator_type="v5p-8", count=1)]
        slice_001_addr = f"10.0.{abs(hash('slice-001')) % 256}.0"
        slice_002_addr = f"10.0.{abs(hash('slice-002')) % 256}.0"
        vm_status_map = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset()),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset()),
        }

        # At timestamp 100_000, only 99 seconds have passed (need 300 seconds)
        autoscaler.run_once(demand, vm_status_map, timestamp_ms=100_000)

        # Should not scale down yet
        assert group.slice_count() == 2

    def test_failed_slices_cleaned_up_before_evaluate(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Failed slices are cleaned up in run_once() before evaluate()."""
        discovered = [
            make_mock_slice("slice-001", all_ready=True, created_at_ms=100000),
            make_mock_slice("slice-002", all_ready=True, created_at_ms=150000),
            make_mock_slice("slice-003", any_failed=True, created_at_ms=200000),
        ]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(
            scale_group_config,
            manager,
            scale_down_cooldown_ms=0,
            idle_threshold_ms=0,
        )
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(accelerator_type="v5p-8", count=1)]
        worker_idle_map = {}

        # run_once() cleans up failed slices first
        autoscaler.run_once(demand, worker_idle_map)

        # Failed slice should be cleaned up
        assert group.get_slice("slice-003") is None
        assert group.slice_count() == 2

    def test_no_scale_down_during_cooldown(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Does not scale down during cooldown period."""
        from iris.time_utils import now_ms

        discovered = [
            make_mock_slice("slice-001", all_ready=True),
            make_mock_slice("slice-002", all_ready=True),
            make_mock_slice("slice-003", all_ready=True),
        ]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(
            scale_group_config,
            manager,
            scale_down_cooldown_ms=3600_000,  # 1 hour
            idle_threshold_ms=0,  # Immediate idle
        )
        group.reconcile()
        # Trigger a scale down to start cooldown
        group.scale_down("slice-003", timestamp_ms=now_ms())
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(accelerator_type="v5p-8", count=1)]
        slice_001_addr = f"10.0.{abs(hash('slice-001')) % 256}.0"
        slice_002_addr = f"10.0.{abs(hash('slice-002')) % 256}.0"
        vm_status_map = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset()),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset()),
        }

        autoscaler.run_once(demand, vm_status_map)

        # Cooldown still active, should not scale down
        assert group.slice_count() == 2  # Started with 3, removed 1 manually


class TestAutoscalerExecution:
    """Tests for decision execution."""

    def test_execute_scale_up_creates_slice(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """execute() creates a slice via ScalingGroup."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0)
        autoscaler = make_autoscaler({"test-group": group})

        decision = ScalingDecision(
            scale_group="test-group",
            action=ScalingAction.SCALE_UP,
            reason="test",
        )

        autoscaler.execute([decision], timestamp_ms=1000)

        assert group.slice_count() == 1

    def test_run_once_cleans_up_failed_slice(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """run_once() terminates failed slices via ScalingGroup.cleanup_failed_slices()."""
        mock_slice = make_mock_slice("slice-001", any_failed=True)
        manager = make_mock_vm_manager(slices_to_discover=[mock_slice])
        group = ScalingGroup(scale_group_config, manager, scale_down_cooldown_ms=0)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        autoscaler.run_once([], {}, timestamp_ms=1000)

        assert group.slice_count() == 0

    def test_execute_records_failure_on_scale_up_error(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """execute() records failure when scale-up fails."""
        manager = make_mock_vm_manager()
        manager.create_vm_group.side_effect = RuntimeError("TPU unavailable")
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0, backoff_initial_ms=5000)
        autoscaler = make_autoscaler({"test-group": group})

        decision = ScalingDecision(
            scale_group="test-group",
            action=ScalingAction.SCALE_UP,
            reason="test",
        )

        autoscaler.execute([decision], timestamp_ms=1000)

        assert group.consecutive_failures == 1
        assert group.backoff_until_ms > 0

    def test_run_once_evaluates_and_executes(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """run_once() performs evaluate then execute."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0)
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(accelerator_type="v5p-8", count=2)]
        vm_status_map = {}  # Empty - no workers yet
        decisions = autoscaler.run_once(demand, vm_status_map)

        assert len(decisions) == 1
        assert decisions[0].action == ScalingAction.SCALE_UP
        # Slice was created
        assert group.slice_count() == 1

    def test_execute_skips_unknown_scale_group(self):
        """execute() skips decisions for unknown scale groups."""
        config = vm_pb2.ScaleGroupConfig(name="known-group", min_slices=0, max_slices=5)
        manager = make_mock_vm_manager()
        group = ScalingGroup(config, manager)

        autoscaler = make_autoscaler({"known-group": group})

        decisions = [
            ScalingDecision(
                scale_group="unknown-group",
                action=ScalingAction.SCALE_UP,
                reason="test",
            )
        ]

        autoscaler.execute(decisions, timestamp_ms=1000)

        assert group.slice_count() == 0


class TestAutoscalerReconcile:
    """Tests for reconcile behavior."""

    def test_reconcile_discovers_slices_in_all_groups(self):
        """reconcile() calls reconcile on all groups."""
        config1 = vm_pb2.ScaleGroupConfig(name="group-1", min_slices=0, max_slices=5)
        config2 = vm_pb2.ScaleGroupConfig(name="group-2", min_slices=0, max_slices=5)

        manager1 = make_mock_vm_manager(
            slices_to_discover=[make_mock_slice("slice-1", scale_group="group-1", all_ready=True)]
        )
        manager2 = make_mock_vm_manager(
            slices_to_discover=[make_mock_slice("slice-2", scale_group="group-2", all_ready=True)]
        )

        group1 = ScalingGroup(config1, manager1)
        group2 = ScalingGroup(config2, manager2)

        autoscaler = make_autoscaler({"group-1": group1, "group-2": group2})
        autoscaler.reconcile()

        assert group1.slice_count() == 1
        assert group2.slice_count() == 1


class TestAutoscalerWorkerFailure:
    """Tests for worker failure handling."""

    def test_notify_worker_failed_terminates_slice(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """notify_worker_failed() terminates the slice containing the worker."""
        mock_slice = make_mock_slice(
            "slice-001",
            worker_ids=["worker-slice-001-0"],
        )
        manager = make_mock_vm_manager(slices_to_discover=[mock_slice])
        group = ScalingGroup(scale_group_config, manager)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})
        autoscaler.reconcile()

        # notify_worker_failed now takes vm_address instead of worker_id
        vm_address = f"10.0.{abs(hash('slice-001')) % 256}.0"
        autoscaler.notify_worker_failed(vm_address)

        assert group.slice_count() == 0

    def test_notify_worker_failed_unknown_worker_is_noop(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """notify_worker_failed() does nothing for unknown workers."""
        mock_slice = make_mock_slice("slice-001", all_ready=True)
        manager = make_mock_vm_manager(slices_to_discover=[mock_slice])
        group = ScalingGroup(scale_group_config, manager)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        # Use an unknown VM address
        autoscaler.notify_worker_failed("10.1.2.3")

        assert group.slice_count() == 1


class TestAutoscalerIdleVerification:
    """Tests for idle verification during scale-down."""

    def test_verifies_idle_with_worker_idle_map(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Scale-down via run_once verifies workers are idle using worker_idle_map."""
        mock_slice = make_mock_slice("slice-001", all_ready=True)
        manager = make_mock_vm_manager(slices_to_discover=[mock_slice])
        group = ScalingGroup(
            scale_group_config,
            manager,
            scale_down_cooldown_ms=0,
            idle_threshold_ms=0,  # Immediate idle eligibility
        )
        group.reconcile()

        autoscaler = Autoscaler(
            scale_groups={"test-group": group},
            vm_registry=VmRegistry(),
        )

        demand = [DemandEntry(accelerator_type="v5p-8", count=0)]

        # Create vm_status_map that reports a task running on the worker
        slice_001_addr = f"10.0.{abs(hash('slice-001')) % 256}.0"
        vm_status_map = {
            slice_001_addr: VmWorkerStatus(
                vm_address=slice_001_addr,
                running_task_ids=frozenset({"task-1"}),  # Not idle
            )
        }

        # run_once should not scale down because worker is not idle
        autoscaler.run_once(demand, vm_status_map)

        # Slice should NOT be terminated because worker has running tasks
        assert group.slice_count() == 1
        mock_slice.terminate.assert_not_called()


class TestAutoscalerStatusReporting:
    """Tests for status reporting via VmRegistry."""

    def test_get_vm_returns_info(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """get_vm() returns VM info from registry."""
        registry = VmRegistry()

        mock_vm = MagicMock()
        mock_vm.info = vm_pb2.VmInfo(
            vm_id="test-vm",
            state=vm_pb2.VM_STATE_READY,
            address="10.0.0.1",
        )
        registry.register(mock_vm)

        manager = make_mock_vm_manager()
        group = ScalingGroup(scale_group_config, manager)
        autoscaler = make_autoscaler({"test-group": group}, vm_registry=registry)

        info = autoscaler.get_vm("test-vm")

        assert info is not None
        assert info.vm_id == "test-vm"
        assert info.state == vm_pb2.VM_STATE_READY

    def test_get_status_includes_all_groups(self):
        """get_status() includes status for all groups."""
        config1 = vm_pb2.ScaleGroupConfig(name="group-1", min_slices=0, max_slices=5)
        config2 = vm_pb2.ScaleGroupConfig(name="group-2", min_slices=0, max_slices=5)

        manager1 = make_mock_vm_manager()
        manager2 = make_mock_vm_manager()

        group1 = ScalingGroup(config1, manager1)
        group2 = ScalingGroup(config2, manager2)

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


class TestAutoscalerFailedSliceCleanup:
    """Tests for automatic cleanup of failed slices via run_once()."""

    def test_cleans_up_failed_slice_via_run_once(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """run_once() cleans up failed slices before evaluating scale-up."""
        discovered = [make_mock_slice("failed-slice", any_failed=True)]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(accelerator_type="v5p-8", count=2)]
        autoscaler.run_once(demand, {})

        # Failed slice should be cleaned up
        assert group.get_slice("failed-slice") is None

    def test_failed_slice_cleanup_triggers_backoff(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Cleaning up a failed slice triggers backoff to prevent thrashing."""
        mock_slice = make_mock_slice("failed-slice", any_failed=True)
        manager = make_mock_vm_manager(slices_to_discover=[mock_slice])
        group = ScalingGroup(
            scale_group_config,
            manager,
            scale_up_cooldown_ms=0,
            backoff_initial_ms=60_000,
        )
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        autoscaler.run_once([], {}, timestamp_ms=1000)

        # Backoff should be active from cleanup
        assert group.consecutive_failures == 1
        assert group.backoff_until_ms > 0

    def test_evaluate_returns_only_scale_up(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """evaluate() only returns SCALE_UP decisions; failed cleanup happens in run_once()."""
        discovered = [
            make_mock_slice("ready-slice", all_ready=True),
            make_mock_slice("failed-slice", any_failed=True),
        ]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(accelerator_type="v5p-8", count=5)]
        decisions = autoscaler.evaluate(demand)

        # evaluate() only returns SCALE_UP now
        for d in decisions:
            assert d.action == ScalingAction.SCALE_UP

    def test_failed_slice_cleanup_skips_idle_verification(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Failed slice cleanup does not require idle verification."""
        mock_slice = make_mock_slice("failed-slice", any_failed=True)
        manager = make_mock_vm_manager(slices_to_discover=[mock_slice])
        group = ScalingGroup(scale_group_config, manager)
        group.reconcile()

        autoscaler = Autoscaler(
            scale_groups={"test-group": group},
            vm_registry=VmRegistry(),
        )

        # run_once() cleans up failed slices without idle verification
        autoscaler.run_once([], {}, timestamp_ms=1000)

        assert group.slice_count() == 0


class TestWaterfallRouting:
    """Tests for priority-based waterfall demand routing."""

    def test_routes_demand_to_highest_priority_group_first(self):
        """Demand routes to highest priority (lowest number) matching group."""
        config_high = vm_pb2.ScaleGroupConfig(name="high-priority", accelerator_type="v5p-8", max_slices=5, priority=10)
        config_low = vm_pb2.ScaleGroupConfig(name="low-priority", accelerator_type="v5p-8", max_slices=5, priority=20)

        group_high = ScalingGroup(config_high, make_mock_vm_manager(), scale_up_cooldown_ms=0)
        group_low = ScalingGroup(config_low, make_mock_vm_manager(), scale_up_cooldown_ms=0)

        autoscaler = make_autoscaler({"high-priority": group_high, "low-priority": group_low})

        demand = [DemandEntry(accelerator_type="v5p-8", count=3)]
        decisions = autoscaler.evaluate(demand)

        # All demand should go to high-priority group
        assert len(decisions) == 1
        assert decisions[0].scale_group == "high-priority"

    def test_demand_overflows_to_lower_priority_when_at_capacity(self):
        """When high-priority group is at capacity, demand overflows to lower priority."""
        config_high = vm_pb2.ScaleGroupConfig(name="high-priority", accelerator_type="v5p-8", max_slices=2, priority=10)
        config_low = vm_pb2.ScaleGroupConfig(name="low-priority", accelerator_type="v5p-8", max_slices=5, priority=20)

        # Pre-fill high-priority group to capacity
        discovered = [make_mock_slice(f"slice-{i}", all_ready=True) for i in range(2)]
        group_high = ScalingGroup(config_high, make_mock_vm_manager(slices_to_discover=discovered))
        group_high.reconcile()

        group_low = ScalingGroup(config_low, make_mock_vm_manager(), scale_up_cooldown_ms=0)

        autoscaler = make_autoscaler({"high-priority": group_high, "low-priority": group_low})

        demand = [DemandEntry(accelerator_type="v5p-8", count=3)]
        decisions = autoscaler.evaluate(demand)

        # Overflow should go to low-priority
        assert len(decisions) == 1
        assert decisions[0].scale_group == "low-priority"

    def test_routing_filters_by_accelerator_type(self):
        """Only groups matching accelerator_type receive demand."""
        config_v5p = vm_pb2.ScaleGroupConfig(name="v5p-group", accelerator_type="v5p-8", max_slices=5, priority=10)
        config_v5lite = vm_pb2.ScaleGroupConfig(
            name="v5lite-group", accelerator_type="v5litepod-4", max_slices=5, priority=10
        )

        group_v5p = ScalingGroup(config_v5p, make_mock_vm_manager(), scale_up_cooldown_ms=0)
        group_v5lite = ScalingGroup(config_v5lite, make_mock_vm_manager(), scale_up_cooldown_ms=0)

        autoscaler = make_autoscaler({"v5p-group": group_v5p, "v5lite-group": group_v5lite})

        demand = [DemandEntry(accelerator_type="v5litepod-4", count=2)]
        decisions = autoscaler.evaluate(demand)

        # Only v5lite-group should receive demand
        assert len(decisions) == 1
        assert decisions[0].scale_group == "v5lite-group"

    def test_demand_with_no_matching_group_is_unmet(self):
        """Demand for unknown accelerator type results in unmet demand."""
        config = vm_pb2.ScaleGroupConfig(name="test-group", accelerator_type="v5p-8", max_slices=5, priority=10)

        group = ScalingGroup(config, make_mock_vm_manager(), scale_up_cooldown_ms=0)

        autoscaler = make_autoscaler({"test-group": group})

        # Request an accelerator type that doesn't exist
        demand = [DemandEntry(accelerator_type="unknown-type", count=2)]
        decisions = autoscaler.evaluate(demand)

        # No decisions should be made (no matching group)
        assert len(decisions) == 0

    def test_multiple_demand_entries_route_independently(self):
        """Multiple demand entries with different accelerator types route to appropriate groups."""
        config_v5p = vm_pb2.ScaleGroupConfig(name="v5p-group", accelerator_type="v5p-8", max_slices=5, priority=10)
        config_v5lite = vm_pb2.ScaleGroupConfig(
            name="v5lite-group", accelerator_type="v5litepod-4", max_slices=5, priority=10
        )

        group_v5p = ScalingGroup(config_v5p, make_mock_vm_manager(), scale_up_cooldown_ms=0)
        group_v5lite = ScalingGroup(config_v5lite, make_mock_vm_manager(), scale_up_cooldown_ms=0)

        autoscaler = make_autoscaler({"v5p-group": group_v5p, "v5lite-group": group_v5lite})

        demand = [
            DemandEntry(accelerator_type="v5p-8", count=2),
            DemandEntry(accelerator_type="v5litepod-4", count=3),
        ]
        decisions = autoscaler.evaluate(demand)

        # Both groups should get scale-up decisions
        assert len(decisions) == 2
        groups_in_decisions = {d.scale_group for d in decisions}
        assert "v5p-group" in groups_in_decisions
        assert "v5lite-group" in groups_in_decisions


class TestAutoscalerWaterfallEndToEnd:
    """End-to-end tests for waterfall routing with FakeVmManager.

    These tests exercise the full cascade behavior using real FakeVmManager
    instances with failure mode injection, rather than mocks. This validates
    the integration between Autoscaler, ScalingGroup, and VmManager.
    """

    def test_demand_cascades_through_priority_groups_on_quota(self):
        """Full cascade: quota on primary routes to secondary.

        The routing happens in evaluate() before execute(), so when quota fails
        during execute(), the demand doesn't re-route in the same cycle. The
        next run_once() will see primary in QUOTA_EXCEEDED and route to fallback.
        """
        from tests.cluster.vm.fakes import FakeVmManager, FakeVmManagerConfig, FailureMode
        from iris.cluster.vm.scaling_group import GroupAvailability

        config_primary = vm_pb2.ScaleGroupConfig(name="primary", accelerator_type="v5p-8", max_slices=5, priority=10)
        config_fallback = vm_pb2.ScaleGroupConfig(name="fallback", accelerator_type="v5p-8", max_slices=5, priority=20)

        manager_primary = FakeVmManager(
            FakeVmManagerConfig(config=config_primary, failure_mode=FailureMode.QUOTA_EXCEEDED)
        )
        manager_fallback = FakeVmManager(FakeVmManagerConfig(config=config_fallback))

        group_primary = ScalingGroup(config_primary, manager_primary, scale_up_cooldown_ms=0)
        group_fallback = ScalingGroup(config_fallback, manager_fallback, scale_up_cooldown_ms=0)

        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
        )

        demand = [DemandEntry(accelerator_type="v5p-8", count=2)]

        # First run: routes to primary, but execute fails with quota
        # This sets primary to QUOTA_EXCEEDED but nothing is created yet
        autoscaler.run_once(demand, {})

        # Primary should be in quota exceeded state
        assert group_primary.availability().status == GroupAvailability.QUOTA_EXCEEDED
        # Fallback hasn't been tried yet (routing happened before quota failure)
        assert group_fallback.slice_count() == 0

        # Second run: primary is now unavailable, demand routes to fallback
        autoscaler.run_once(demand, {})
        assert group_fallback.slice_count() == 1

        # Third run: another slice on fallback
        autoscaler.run_once(demand, {})
        assert group_fallback.slice_count() == 2

    def test_quota_recovery_restores_primary_routing(self):
        """After quota timeout expires, demand routes to primary again.

        Note: execute() uses now_ms() internally, so quota_exceeded_until_ms is
        set relative to real time, not test timestamps. We use a very short
        quota_timeout_ms and wait for it to expire naturally.
        """
        import time
        from tests.cluster.vm.fakes import FakeVmManager, FakeVmManagerConfig, FailureMode
        from iris.cluster.vm.scaling_group import GroupAvailability
        from iris.time_utils import now_ms

        config_primary = vm_pb2.ScaleGroupConfig(name="primary", accelerator_type="v5p-8", max_slices=5, priority=10)
        config_fallback = vm_pb2.ScaleGroupConfig(name="fallback", accelerator_type="v5p-8", max_slices=5, priority=20)

        manager_primary = FakeVmManager(
            FakeVmManagerConfig(config=config_primary, failure_mode=FailureMode.QUOTA_EXCEEDED)
        )
        manager_fallback = FakeVmManager(FakeVmManagerConfig(config=config_fallback))

        # Very short quota timeout (10ms) for testing
        group_primary = ScalingGroup(config_primary, manager_primary, scale_up_cooldown_ms=0, quota_timeout_ms=10)
        group_fallback = ScalingGroup(config_fallback, manager_fallback, scale_up_cooldown_ms=0)

        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
        )

        demand = [DemandEntry(accelerator_type="v5p-8", count=1)]

        # First run: tries primary, fails with quota (nothing created yet)
        autoscaler.run_once(demand, {})
        ts_after_fail = now_ms()
        assert group_primary.availability(ts_after_fail).status == GroupAvailability.QUOTA_EXCEEDED
        assert group_fallback.slice_count() == 0

        # Second run: primary unavailable, routes to fallback
        autoscaler.run_once(demand, {})
        assert group_fallback.slice_count() == 1

        # Wait for quota timeout to expire (10ms + buffer)
        time.sleep(0.02)

        # Clear the failure mode so primary can succeed
        manager_primary.set_failure_mode(FailureMode.NONE)

        # After timeout: primary should be available again
        ts_now = now_ms()
        assert group_primary.availability(ts_now).status == GroupAvailability.AVAILABLE

        # Increase demand to verify routing goes to primary now
        demand_increased = [DemandEntry(accelerator_type="v5p-8", count=2)]
        decisions = autoscaler.evaluate(demand_increased, timestamp_ms=ts_now)
        assert len(decisions) == 1
        assert decisions[0].scale_group == "primary"

    def test_backoff_cascades_to_fallback(self):
        """Generic failure triggers backoff, which cascades to fallback."""
        from tests.cluster.vm.fakes import FakeVmManager, FakeVmManagerConfig, FailureMode
        from iris.cluster.vm.scaling_group import GroupAvailability

        config_primary = vm_pb2.ScaleGroupConfig(name="primary", accelerator_type="v5p-8", max_slices=5, priority=10)
        config_fallback = vm_pb2.ScaleGroupConfig(name="fallback", accelerator_type="v5p-8", max_slices=5, priority=20)

        # Primary will fail with generic error (triggers backoff)
        manager_primary = FakeVmManager(
            FakeVmManagerConfig(config=config_primary, failure_mode=FailureMode.CREATE_FAILS)
        )
        manager_fallback = FakeVmManager(FakeVmManagerConfig(config=config_fallback))

        group_primary = ScalingGroup(config_primary, manager_primary, scale_up_cooldown_ms=0, backoff_initial_ms=60000)
        group_fallback = ScalingGroup(config_fallback, manager_fallback, scale_up_cooldown_ms=0)

        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
        )

        demand = [DemandEntry(accelerator_type="v5p-8", count=1)]

        # First run: primary fails, triggers backoff
        autoscaler.run_once(demand, {})

        # Primary should be in backoff (not quota)
        assert group_primary.availability().status == GroupAvailability.BACKOFF
        assert group_primary.consecutive_failures == 1

        # Second run: should route to fallback
        autoscaler.run_once(demand, {})
        assert group_fallback.slice_count() == 1

    def test_multiple_accelerator_types_route_independently(self):
        """Different accelerator types route through their own group chains."""
        from tests.cluster.vm.fakes import FakeVmManager, FakeVmManagerConfig

        config_v5p = vm_pb2.ScaleGroupConfig(name="v5p-group", accelerator_type="v5p-8", max_slices=5, priority=10)
        config_v5lite = vm_pb2.ScaleGroupConfig(
            name="v5lite-group", accelerator_type="v5litepod-4", max_slices=5, priority=10
        )

        manager_v5p = FakeVmManager(FakeVmManagerConfig(config=config_v5p))
        manager_v5lite = FakeVmManager(FakeVmManagerConfig(config=config_v5lite))

        group_v5p = ScalingGroup(config_v5p, manager_v5p, scale_up_cooldown_ms=0)
        group_v5lite = ScalingGroup(config_v5lite, manager_v5lite, scale_up_cooldown_ms=0)

        autoscaler = make_autoscaler(
            scale_groups={"v5p-group": group_v5p, "v5lite-group": group_v5lite},
        )

        # Demand for both types
        demand = [
            DemandEntry(accelerator_type="v5p-8", count=2),
            DemandEntry(accelerator_type="v5litepod-4", count=1),
        ]

        autoscaler.run_once(demand, {})

        # Each group should have received its own demand
        assert group_v5p.slice_count() == 1  # Created 1 slice toward demand of 2
        assert group_v5lite.slice_count() == 1

    def test_capacity_overflow_cascades_to_lower_priority(self):
        """When high-priority group fills up, overflow goes to lower priority.

        Routing distributes demand based on headroom (max_slices - current).
        Booting/initializing slices count as capacity, so they're not double-
        counted. With demand=4 and primary max=2, routing initially splits
        the demand; as primary fills up, more goes to fallback.

        Since FakeVm starts in BOOTING state which counts as capacity, we
        need multiple runs to fill up both groups.
        """
        from tests.cluster.vm.fakes import FakeVmManager, FakeVmManagerConfig
        from iris.time_utils import now_ms

        config_primary = vm_pb2.ScaleGroupConfig(name="primary", accelerator_type="v5p-8", max_slices=2, priority=10)
        config_fallback = vm_pb2.ScaleGroupConfig(name="fallback", accelerator_type="v5p-8", max_slices=5, priority=20)

        manager_primary = FakeVmManager(FakeVmManagerConfig(config=config_primary))
        manager_fallback = FakeVmManager(FakeVmManagerConfig(config=config_fallback))

        group_primary = ScalingGroup(config_primary, manager_primary, scale_up_cooldown_ms=0)
        group_fallback = ScalingGroup(config_fallback, manager_fallback, scale_up_cooldown_ms=0)

        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
        )

        # Demand exceeds primary's max_slices (4 > 2)
        demand = [DemandEntry(accelerator_type="v5p-8", count=4)]

        # First run: routing allocates primary=2, fallback=2 (based on headroom)
        # Creates one slice in each group (both now have capacity=1 from booting)
        autoscaler.run_once(demand, {})
        assert group_primary.slice_count() == 1
        assert group_fallback.slice_count() == 1

        # Tick the managers to advance VM states (makes them ready)
        # This simulates time passing where VMs finish booting
        ts = now_ms()
        manager_primary.tick(ts)
        manager_fallback.tick(ts)

        # Now ready slices count as ready_capacity, and pending=0
        # The routing still uses headroom (max - current_count)
        # primary: max=2, current=1, headroom=1 -> allocates 1
        # fallback: max=5, current=1, headroom=4 -> allocates remaining 3
        # But evaluate checks demand > capacity where capacity=ready+pending
        # With 1 ready slice and demand=1 allocated, no scale up for primary
        # (unless demand > capacity, which requires demand > 1)

        # Second run: After tick, primary has 1 ready slice
        # routing allocates primary=1 (headroom=1), fallback=3
        # Since primary's capacity=1 and demand=1, no scale-up (demand <= capacity)
        # fallback's capacity=1 and demand=3, so scale-up happens
        autoscaler.run_once(demand, {})
        # Primary stays at 1 because demand=capacity
        assert group_primary.slice_count() == 1
        assert group_fallback.slice_count() == 2

        # Tick again
        manager_primary.tick(now_ms())
        manager_fallback.tick(now_ms())

        # Third run: primary still has headroom=1, but gets allocated demand<=capacity
        # Let's verify that when we have demand > total capacity, the system works
        autoscaler.run_once(demand, {})
        # Now primary should scale up because we need more capacity
        # total capacity = 1 + 2 = 3, demand = 4, so 1 more needed
        # primary gets demand=1 (headroom=1), capacity=1, no scale (1 <= 1)
        # fallback gets demand=3, capacity=2, needs to scale
        assert group_fallback.slice_count() == 3

        # After several more runs, demand should be met
        for _ in range(2):
            manager_primary.tick(now_ms())
            manager_fallback.tick(now_ms())
            autoscaler.run_once(demand, {})

        # Total slices should meet or exceed demand
        total = group_primary.slice_count() + group_fallback.slice_count()
        assert total >= 4


class TestAutoscalerQuotaHandling:
    """Tests for quota exceeded error handling."""

    def test_quota_exceeded_sets_group_unavailable(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """QuotaExceededError sets group to QUOTA_EXCEEDED state."""
        from iris.cluster.vm.managed_vm import QuotaExceededError
        from iris.cluster.vm.scaling_group import GroupAvailability

        manager = make_mock_vm_manager()
        manager.create_vm_group.side_effect = QuotaExceededError("Quota exceeded")
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0, quota_timeout_ms=60_000)
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(accelerator_type="v5p-8", count=1)]
        autoscaler.run_once(demand, {}, timestamp_ms=1000)

        # Group should be in QUOTA_EXCEEDED state
        state = group.availability(timestamp_ms=2000)
        assert state.status == GroupAvailability.QUOTA_EXCEEDED

    def test_quota_exceeded_routes_to_fallback_group(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """When primary group has quota exceeded, demand routes to fallback."""
        from iris.cluster.vm.managed_vm import QuotaExceededError

        config_primary = vm_pb2.ScaleGroupConfig(name="primary", accelerator_type="v5p-8", max_slices=5, priority=10)
        config_fallback = vm_pb2.ScaleGroupConfig(name="fallback", accelerator_type="v5p-8", max_slices=5, priority=20)

        manager_primary = make_mock_vm_manager()
        manager_primary.create_vm_group.side_effect = QuotaExceededError("Quota exceeded")
        manager_fallback = make_mock_vm_manager()

        group_primary = ScalingGroup(config_primary, manager_primary, scale_up_cooldown_ms=0, quota_timeout_ms=60_000)
        group_fallback = ScalingGroup(config_fallback, manager_fallback, scale_up_cooldown_ms=0)
        autoscaler = make_autoscaler({"primary": group_primary, "fallback": group_fallback})

        # First call: primary fails with quota, triggers quota state
        demand = [DemandEntry(accelerator_type="v5p-8", count=1)]
        autoscaler.run_once(demand, {}, timestamp_ms=1000)

        # Second call: primary is in QUOTA_EXCEEDED, should route to fallback
        decisions = autoscaler.evaluate(demand, timestamp_ms=2000)

        assert len(decisions) == 1
        assert decisions[0].scale_group == "fallback"

    def test_quota_state_expires_after_timeout(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """QUOTA_EXCEEDED state expires after timeout."""
        from iris.cluster.vm.managed_vm import QuotaExceededError
        from iris.cluster.vm.scaling_group import GroupAvailability

        manager = make_mock_vm_manager()
        manager.create_vm_group.side_effect = QuotaExceededError("Quota exceeded")
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0, quota_timeout_ms=1000)

        # Directly trigger quota exceeded through scale_up with controlled timestamp
        with pytest.raises(QuotaExceededError):
            group.scale_up(ts=1000)

        # Immediately after: QUOTA_EXCEEDED
        assert group.availability(1100).status == GroupAvailability.QUOTA_EXCEEDED

        # After timeout (1000 + 1000 = 2000): AVAILABLE
        assert group.availability(2100).status == GroupAvailability.AVAILABLE

    def test_generic_error_triggers_backoff_not_quota(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Non-quota errors trigger backoff, not quota exceeded state."""
        from iris.cluster.vm.scaling_group import GroupAvailability

        manager = make_mock_vm_manager()
        manager.create_vm_group.side_effect = RuntimeError("TPU unavailable")

        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0, backoff_initial_ms=5000)
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(accelerator_type="v5p-8", count=1)]
        autoscaler.run_once(demand, {}, timestamp_ms=1000)

        # Should be in BACKOFF, not QUOTA_EXCEEDED
        state = group.availability(timestamp_ms=2000)
        assert state.status == GroupAvailability.BACKOFF
        assert group.consecutive_failures == 1


class TestAutoscalerActionLogging:
    """Tests for autoscaler action logging."""

    def test_action_log_records_scale_up(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Verify scale-up actions are logged."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0)
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(accelerator_type="v5p-8", count=2)]
        autoscaler.run_once(demand, {})

        # Check that the action log contains a scale_up action
        status = autoscaler.get_status()
        assert len(status.recent_actions) == 1
        action = status.recent_actions[0]
        assert action.action_type == "scale_up"
        assert action.scale_group == "test-group"
        assert action.slice_id != ""  # Should have a slice ID
        assert "demand" in action.reason

    def test_action_log_records_failed_cleanup(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Verify failed slice cleanup actions are logged."""
        mock_slice = make_mock_slice("slice-001", any_failed=True)
        manager = make_mock_vm_manager(slices_to_discover=[mock_slice])
        group = ScalingGroup(scale_group_config, manager, scale_down_cooldown_ms=0)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        autoscaler.run_once([], {}, timestamp_ms=1000)

        status = autoscaler.get_status()
        assert len(status.recent_actions) == 1
        action = status.recent_actions[0]
        assert action.action_type == "failed_cleanup"
        assert action.scale_group == "test-group"
        assert action.slice_id == "slice-001"
        assert "cleaning up failed slice" in action.reason

    def test_action_log_records_quota_exceeded(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Verify quota exceeded events are logged."""
        from iris.cluster.vm.managed_vm import QuotaExceededError

        manager = make_mock_vm_manager()
        manager.create_vm_group.side_effect = QuotaExceededError("Quota exceeded in zone")
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0)
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(accelerator_type="v5p-8", count=1)]
        autoscaler.run_once(demand, {})

        status = autoscaler.get_status()
        assert len(status.recent_actions) == 1
        action = status.recent_actions[0]
        assert action.action_type == "quota_exceeded"
        assert action.scale_group == "test-group"
        assert "Quota exceeded" in action.reason

    def test_action_log_records_worker_failed(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Verify worker failure events are logged."""
        mock_slice = make_mock_slice(
            "slice-001",
            worker_ids=["worker-slice-001-0"],
        )
        manager = make_mock_vm_manager(slices_to_discover=[mock_slice])
        group = ScalingGroup(scale_group_config, manager)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})
        autoscaler.reconcile()

        # notify_worker_failed now takes vm_address instead of worker_id
        vm_address = f"10.0.{abs(hash('slice-001')) % 256}.0"
        autoscaler.notify_worker_failed(vm_address)

        status = autoscaler.get_status()
        assert len(status.recent_actions) == 1
        action = status.recent_actions[0]
        assert action.action_type == "worker_failed"
        assert action.scale_group == "test-group"
        assert action.slice_id == "slice-001"
        assert vm_address in action.reason

    def test_action_log_bounded_to_100_entries(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Verify action log is bounded to 100 entries."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0)
        autoscaler = make_autoscaler({"test-group": group})

        # Directly add 150 actions to the log (bypassing actual scaling)
        for i in range(150):
            autoscaler._log_action("test_action", "test-group", reason=f"action {i}")

        status = autoscaler.get_status()
        assert len(status.recent_actions) == 100
        # Oldest entries should be dropped (first 50)
        assert status.recent_actions[0].reason == "action 50"
        assert status.recent_actions[99].reason == "action 149"

    def test_get_status_includes_actions(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Verify get_status returns recent actions."""
        manager = make_mock_vm_manager()
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0)
        autoscaler = make_autoscaler({"test-group": group})

        # Run a scale-up to generate an action
        demand = [DemandEntry(accelerator_type="v5p-8", count=1)]
        autoscaler.run_once(demand, {})

        status = autoscaler.get_status()

        # Status should include groups, demand, and actions
        assert len(status.groups) == 1
        assert status.current_demand["test-group"] == 1
        assert len(status.recent_actions) >= 1
        assert status.recent_actions[0].action_type == "scale_up"

    def test_action_log_includes_timestamp(self, scale_group_config: vm_pb2.ScaleGroupConfig):
        """Verify actions include valid timestamps."""
        from iris.time_utils import now_ms

        manager = make_mock_vm_manager()
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown_ms=0)
        autoscaler = make_autoscaler({"test-group": group})

        before = now_ms()
        demand = [DemandEntry(accelerator_type="v5p-8", count=1)]
        autoscaler.run_once(demand, {})
        after = now_ms()

        status = autoscaler.get_status()
        action = status.recent_actions[0]
        assert before <= action.timestamp_ms <= after
