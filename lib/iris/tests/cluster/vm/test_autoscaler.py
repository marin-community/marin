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

import time
from unittest.mock import MagicMock

import pytest

from iris.cluster.types import DeviceType, VmWorkerStatus
from iris.cluster.vm.autoscaler import (
    Autoscaler,
    DemandEntry,
    ScalingAction,
    ScalingDecision,
    route_demand,
)
from iris.cluster.vm.managed_vm import VmRegistry
from iris.cluster.vm.vm_platform import VmGroupStatus, VmSnapshot
from iris.cluster.vm.scaling_group import ScalingGroup
from iris.rpc import config_pb2, vm_pb2
from iris.time_utils import Duration, Timestamp

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

    from iris.rpc import time_pb2

    mock.to_proto.return_value = vm_pb2.SliceInfo(
        slice_id=slice_id,
        scale_group=scale_group,
        created_at=time_pb2.Timestamp(epoch_ms=created_at_ms),
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
def scale_group_config() -> config_pb2.ScaleGroupConfig:
    """A standard scale group configuration."""
    return config_pb2.ScaleGroupConfig(
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
    manager = make_mock_vm_manager()
    group = ScalingGroup(scale_group_config, manager, scale_up_cooldown=Duration.from_ms(0))
    autoscaler = make_autoscaler({"test-group": group})
    yield autoscaler
    autoscaler.shutdown()


@pytest.fixture
def autoscaler_with_ready_slices(scale_group_config):
    """Autoscaler with 2 ready slices for scale-down tests."""
    discovered = [
        make_mock_slice("slice-001", all_ready=True),
        make_mock_slice("slice-002", all_ready=True),
    ]
    manager = make_mock_vm_manager(slices_to_discover=discovered)
    group = ScalingGroup(
        scale_group_config, manager, scale_down_cooldown=Duration.from_ms(0), idle_threshold=Duration.from_ms(0)
    )
    group.reconcile()
    autoscaler = make_autoscaler({"test-group": group})
    yield autoscaler, group, manager
    autoscaler.shutdown()


def make_autoscaler(
    scale_groups: dict[str, ScalingGroup],
    vm_registry: VmRegistry | None = None,
    config: config_pb2.AutoscalerConfig | None = None,
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

    def test_scales_up_when_demand_exceeds_capacity(self, empty_autoscaler: Autoscaler):
        """Evaluates scale-up when demand > capacity."""
        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=2)]
        decisions = empty_autoscaler.evaluate(demand)

        assert len(decisions) == 1
        assert decisions[0].action == ScalingAction.SCALE_UP
        assert decisions[0].scale_group == "test-group"
        assert "demand=2 > capacity=0" in decisions[0].reason

    @pytest.mark.parametrize(
        "discovered,demand_count,reason",
        [
            ([make_mock_slice(f"slice-{i}") for i in range(5)], 10, "at_max_slices"),
            (
                [make_mock_slice("slice-001", all_ready=True), make_mock_slice("slice-002", all_ready=True)],
                2,
                "capacity_meets_demand",
            ),
            (
                [
                    make_mock_slice("slice-001", vm_states=[vm_pb2.VM_STATE_BOOTING]),
                    make_mock_slice("slice-002", vm_states=[vm_pb2.VM_STATE_INITIALIZING]),
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
        """Does not scale up when various conditions are met (max slices, sufficient capacity, pending slices)."""
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown=Duration.from_ms(0))
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=demand_count)]
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 0

    def test_no_scale_up_during_backoff(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Does not scale up during backoff period."""
        from iris.time_utils import Timestamp

        manager = make_mock_vm_manager()
        # Use large backoff so it's still active when we evaluate
        group = ScalingGroup(
            scale_group_config,
            manager,
            scale_up_cooldown=Duration.from_ms(0),
            backoff_initial=Duration.from_hours(1),  # 1 hour backoff
        )
        group.record_failure(timestamp=Timestamp.now())
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=2)]
        decisions = autoscaler.evaluate(demand)

        # Decision blocked by backoff
        assert len(decisions) == 0

    def test_no_scale_up_during_cooldown(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Does not scale up during cooldown period."""
        from iris.time_utils import Timestamp

        manager = make_mock_vm_manager()
        # Use large cooldown so it's still active when we evaluate
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown=Duration.from_ms(3600_000))  # 1 hour
        group.scale_up(timestamp=Timestamp.now())
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=2)]
        decisions = autoscaler.evaluate(demand)

        # Decision blocked by cooldown (only 1 slice at time of evaluate)
        # Since we just called scale_up, cooldown is active
        assert len(decisions) == 0

    def test_scales_up_to_enforce_min_slices(self):
        """Scales up to enforce min_slices even with zero demand."""
        config = config_pb2.ScaleGroupConfig(
            name="test-group",
            min_slices=2,
            max_slices=5,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            runtime_version="v2-alpha-tpuv5",
            zones=["us-central1-a"],
        )
        manager = make_mock_vm_manager()
        group = ScalingGroup(config, manager, scale_up_cooldown=Duration.from_ms(0))
        autoscaler = make_autoscaler({"test-group": group})

        # No demand, but min_slices=2 means we should scale up
        decisions = autoscaler.evaluate([])

        assert len(decisions) == 1
        assert decisions[0].action == ScalingAction.SCALE_UP
        assert "below min_slices" in decisions[0].reason
        assert "0 < 2" in decisions[0].reason

    def test_no_scale_up_when_at_min_slices(self):
        """Does not scale up when already at min_slices and no demand."""
        config = config_pb2.ScaleGroupConfig(
            name="test-group",
            min_slices=2,
            max_slices=5,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            runtime_version="v2-alpha-tpuv5",
            zones=["us-central1-a"],
        )
        discovered = [
            make_mock_slice("slice-001", all_ready=True),
            make_mock_slice("slice-002", all_ready=True),
        ]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(config, manager, scale_up_cooldown=Duration.from_ms(0))
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        # No demand and already at min_slices=2
        decisions = autoscaler.evaluate([])

        assert len(decisions) == 0


class TestAutoscalerScaleDown:
    """Tests for scale-down behavior (delegated to ScalingGroup)."""

    def test_scales_down_idle_slice_via_run_once(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """run_once() scales down idle slices via ScalingGroup.scale_down_if_idle()."""
        discovered = [
            make_mock_slice("slice-001", all_ready=True, created_at_ms=100000),
            make_mock_slice("slice-002", all_ready=True, created_at_ms=200000),
        ]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(
            scale_group_config,
            manager,
            scale_down_cooldown=Duration.from_ms(0),
            idle_threshold=Duration.from_ms(1000),  # 1 second threshold
        )
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        # Mark slices as idle (never had activity, so eligible for scaledown)
        # The idle_threshold=Duration.from_ms(1000) means slices need to be idle for 1s

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=1)]

        # All workers are idle
        # VM addresses are generated as f"10.0.{hash(slice_id) % 256}.{i}"
        slice_001_addr = f"10.0.{abs(hash('slice-001')) % 256}.0"
        slice_002_addr = f"10.0.{abs(hash('slice-002')) % 256}.0"
        vm_status_map = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset()),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset()),
        }

        # run_once should scale down via ScalingGroup.scale_down_if_idle
        autoscaler.run_once(demand, vm_status_map, timestamp=Timestamp.from_ms(10_000))

        # One slice should be terminated (longest idle first, which is slice-001)
        assert group.slice_count() == 1

    def test_no_scale_down_at_min_slices(self):
        """Does not scale down when at min_slices."""
        config = config_pb2.ScaleGroupConfig(
            name="test-group",
            min_slices=2,
            max_slices=5,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            runtime_version="v2-alpha-tpuv5",
        )
        discovered = [
            make_mock_slice("slice-001", all_ready=True),
            make_mock_slice("slice-002", all_ready=True),
        ]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(
            config, manager, scale_down_cooldown=Duration.from_ms(0), idle_threshold=Duration.from_ms(0)
        )
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=0)]
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

    def test_no_scale_down_until_idle_threshold(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Does not scale down until slice has been idle long enough."""
        discovered = [
            make_mock_slice("slice-001", all_ready=True),
            make_mock_slice("slice-002", all_ready=True),
        ]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(
            scale_group_config,
            manager,
            scale_down_cooldown=Duration.from_ms(0),
            idle_threshold=Duration.from_ms(300_000),  # 5 minute threshold
        )
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        slice_001_addr = f"10.0.{abs(hash('slice-001')) % 256}.0"
        slice_002_addr = f"10.0.{abs(hash('slice-002')) % 256}.0"

        # Mark slices as active at time 1000 by calling update_slice_activity with running tasks
        vm_status_map_active = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset({"task-1"})),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset({"task-2"})),
        }
        group.update_slice_activity(vm_status_map_active, timestamp=Timestamp.from_ms(1000))

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=1)]
        vm_status_map_idle = {
            slice_001_addr: VmWorkerStatus(vm_address=slice_001_addr, running_task_ids=frozenset()),
            slice_002_addr: VmWorkerStatus(vm_address=slice_002_addr, running_task_ids=frozenset()),
        }

        # At timestamp 100_000, only 99 seconds have passed (need 300 seconds)
        autoscaler.run_once(demand, vm_status_map_idle, timestamp=Timestamp.from_ms(100_000))

        # Should not scale down yet
        assert group.slice_count() == 2

    def test_failed_slices_cleaned_up_before_evaluate(self, scale_group_config: config_pb2.ScaleGroupConfig):
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
            scale_down_cooldown=Duration.from_ms(0),
            idle_threshold=Duration.from_ms(0),
        )
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=1)]
        worker_idle_map = {}

        # run_once() cleans up failed slices first
        autoscaler.run_once(demand, worker_idle_map)

        # Failed slice should be cleaned up
        assert group.get_slice("slice-003") is None
        assert group.slice_count() == 2

    def test_no_scale_down_during_cooldown(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Does not scale down during cooldown period."""
        from iris.time_utils import Timestamp

        discovered = [
            make_mock_slice("slice-001", all_ready=True),
            make_mock_slice("slice-002", all_ready=True),
            make_mock_slice("slice-003", all_ready=True),
        ]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(
            scale_group_config,
            manager,
            scale_down_cooldown=Duration.from_ms(3600_000),  # 1 hour
            idle_threshold=Duration.from_ms(0),  # Immediate idle
        )
        group.reconcile()
        # Trigger a scale down to start cooldown
        group.scale_down("slice-003", timestamp=Timestamp.now())
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=1)]
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

    def test_execute_scale_up_creates_slice(self, empty_autoscaler: Autoscaler):
        """execute() creates a slice via ScalingGroup."""
        decision = ScalingDecision(
            scale_group="test-group",
            action=ScalingAction.SCALE_UP,
            reason="test",
        )

        empty_autoscaler.execute([decision], timestamp=Timestamp.from_ms(1000))
        empty_autoscaler._wait_for_inflight()  # Wait for async scale-up

        group = empty_autoscaler.groups["test-group"]
        assert group.slice_count() == 1

    def test_run_once_cleans_up_failed_slice(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """run_once() terminates failed slices via ScalingGroup.cleanup_failed_slices()."""
        mock_slice = make_mock_slice("slice-001", any_failed=True)
        manager = make_mock_vm_manager(slices_to_discover=[mock_slice])
        group = ScalingGroup(scale_group_config, manager, scale_down_cooldown=Duration.from_ms(0))
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        autoscaler.run_once([], {}, timestamp=Timestamp.from_ms(1000))

        assert group.slice_count() == 0

    def test_execute_records_failure_on_scale_up_error(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """execute() records failure when scale-up fails."""
        manager = make_mock_vm_manager()
        manager.create_vm_group.side_effect = RuntimeError("TPU unavailable")
        backoff = Duration.from_seconds(5.0)
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown=Duration.from_ms(0), backoff_initial=backoff)
        autoscaler = make_autoscaler({"test-group": group})

        decision = ScalingDecision(
            scale_group="test-group",
            action=ScalingAction.SCALE_UP,
            reason="test",
        )

        autoscaler.execute([decision], timestamp=Timestamp.from_ms(1000))
        autoscaler._wait_for_inflight()  # Wait for async scale-up

        assert group.consecutive_failures == 1
        assert group.backoff_until_ms > 0

    def test_run_once_evaluates_and_executes(self, empty_autoscaler: Autoscaler):
        """run_once() performs evaluate then execute."""
        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=2)]
        vm_status_map = {}  # Empty - no workers yet
        decisions = empty_autoscaler.run_once(demand, vm_status_map)
        empty_autoscaler._wait_for_inflight()  # Wait for async scale-up

        assert len(decisions) == 1
        assert decisions[0].action == ScalingAction.SCALE_UP
        # Slice was created
        group = empty_autoscaler.groups["test-group"]
        assert group.slice_count() == 1

    def test_execute_skips_unknown_scale_group(self):
        """execute() skips decisions for unknown scale groups."""
        config = config_pb2.ScaleGroupConfig(name="known-group", min_slices=0, max_slices=5)
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

        autoscaler.execute(decisions, timestamp=Timestamp.from_ms(1000))

        assert group.slice_count() == 0


class TestAutoscalerReconcile:
    """Tests for reconcile behavior."""

    def test_reconcile_discovers_slices_in_all_groups(self):
        """reconcile() calls reconcile on all groups."""
        config1 = config_pb2.ScaleGroupConfig(name="group-1", min_slices=0, max_slices=5)
        config2 = config_pb2.ScaleGroupConfig(name="group-2", min_slices=0, max_slices=5)

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

    def test_notify_worker_failed_terminates_slice(self, scale_group_config: config_pb2.ScaleGroupConfig):
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

    def test_notify_worker_failed_unknown_worker_is_noop(self, scale_group_config: config_pb2.ScaleGroupConfig):
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

    def test_verifies_idle_with_worker_idle_map(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Scale-down via run_once verifies workers are idle using worker_idle_map."""
        mock_slice = make_mock_slice("slice-001", all_ready=True)
        manager = make_mock_vm_manager(slices_to_discover=[mock_slice])
        group = ScalingGroup(
            scale_group_config,
            manager,
            scale_down_cooldown=Duration.from_ms(0),
            idle_threshold=Duration.from_ms(0),  # Immediate idle eligibility
        )
        group.reconcile()

        autoscaler = Autoscaler(
            scale_groups={"test-group": group},
            vm_registry=VmRegistry(),
        )

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=0)]

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

    def test_get_vm_retrieves_from_registry(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """get_vm() retrieves VM from registry when present."""
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

        # Verify retrieval succeeds
        assert info is not None

    def test_get_status_includes_all_groups(self):
        """get_status() includes status for all groups."""
        config1 = config_pb2.ScaleGroupConfig(name="group-1", min_slices=0, max_slices=5)
        config2 = config_pb2.ScaleGroupConfig(name="group-2", min_slices=0, max_slices=5)

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

    def test_cleans_up_failed_slice_via_run_once(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """run_once() cleans up failed slices before evaluating scale-up."""
        discovered = [make_mock_slice("failed-slice", any_failed=True)]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown=Duration.from_ms(0))
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=2)]
        autoscaler.run_once(demand, {})

        # Failed slice should be cleaned up
        assert group.get_slice("failed-slice") is None

    def test_failed_slice_cleanup_triggers_backoff(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Cleaning up a failed slice triggers backoff to prevent thrashing."""
        mock_slice = make_mock_slice("failed-slice", any_failed=True)
        manager = make_mock_vm_manager(slices_to_discover=[mock_slice])
        group = ScalingGroup(
            scale_group_config,
            manager,
            scale_up_cooldown=Duration.from_ms(0),
            backoff_initial=Duration.from_seconds(60.0),
        )
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        autoscaler.run_once([], {}, timestamp=Timestamp.from_ms(1000))

        # Backoff should be active from cleanup
        assert group.consecutive_failures == 1
        assert group.backoff_until_ms > 0

    def test_evaluate_returns_only_scale_up(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """evaluate() only returns SCALE_UP decisions; failed cleanup happens in run_once()."""
        discovered = [
            make_mock_slice("ready-slice", all_ready=True),
            make_mock_slice("failed-slice", any_failed=True),
        ]
        manager = make_mock_vm_manager(slices_to_discover=discovered)
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown=Duration.from_ms(0))
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=5)]
        decisions = autoscaler.evaluate(demand)

        # evaluate() only returns SCALE_UP now
        for d in decisions:
            assert d.action == ScalingAction.SCALE_UP

    def test_failed_slice_cleanup_skips_idle_verification(self, scale_group_config: config_pb2.ScaleGroupConfig):
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
        autoscaler.run_once([], {}, timestamp=Timestamp.from_ms(1000))

        assert group.slice_count() == 0


class TestWaterfallRouting:
    """Tests for priority-based waterfall demand routing."""

    def test_routes_demand_to_highest_priority_group_first(self):
        """Demand routes to highest priority (lowest number) matching group."""
        config_high = config_pb2.ScaleGroupConfig(
            name="high-priority",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=10,
        )
        config_low = config_pb2.ScaleGroupConfig(
            name="low-priority",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=20,
        )

        group_high = ScalingGroup(config_high, make_mock_vm_manager(), scale_up_cooldown=Duration.from_ms(0))
        group_low = ScalingGroup(config_low, make_mock_vm_manager(), scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler({"high-priority": group_high, "low-priority": group_low})

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=3)]
        decisions = autoscaler.evaluate(demand)

        # All demand should go to high-priority group
        assert len(decisions) == 1
        assert decisions[0].scale_group == "high-priority"

    def test_cpu_demand_routes_by_priority(self):
        """CPU demand matches all groups and routes by priority."""
        config_high = config_pb2.ScaleGroupConfig(
            name="high-priority",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=10,
        )
        config_low = config_pb2.ScaleGroupConfig(
            name="low-priority",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_GPU,
            accelerator_variant="A100",
            max_slices=5,
            priority=20,
        )

        group_high = ScalingGroup(config_high, make_mock_vm_manager(), scale_up_cooldown=Duration.from_ms(0))
        group_low = ScalingGroup(config_low, make_mock_vm_manager(), scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler({"high-priority": group_high, "low-priority": group_low})

        demand = [DemandEntry(device_type=DeviceType.CPU, device_variant=None, count=2)]
        decisions = autoscaler.evaluate(demand)

        assert len(decisions) == 1
        assert decisions[0].scale_group == "high-priority"
        assert group_high.current_demand == 2
        assert group_low.current_demand == 0

    def test_demand_overflows_to_lower_priority_when_at_capacity(self):
        """When high-priority group is at capacity, demand overflows to lower priority."""
        config_high = config_pb2.ScaleGroupConfig(
            name="high-priority",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=2,
            priority=10,
        )
        config_low = config_pb2.ScaleGroupConfig(
            name="low-priority",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=20,
        )

        # Pre-fill high-priority group to capacity
        discovered = [make_mock_slice(f"slice-{i}", all_ready=True) for i in range(2)]
        group_high = ScalingGroup(config_high, make_mock_vm_manager(slices_to_discover=discovered))
        group_high.reconcile()

        group_low = ScalingGroup(config_low, make_mock_vm_manager(), scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler({"high-priority": group_high, "low-priority": group_low})

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=3)]
        decisions = autoscaler.evaluate(demand)

        # Overflow should go to low-priority
        assert len(decisions) == 1
        assert decisions[0].scale_group == "low-priority"

    def test_routing_filters_by_accelerator_type(self):
        """Only groups matching accelerator_type receive demand."""
        config_v5p = config_pb2.ScaleGroupConfig(
            name="v5p-group",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=10,
        )
        config_v5lite = config_pb2.ScaleGroupConfig(
            name="v5lite-group",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5litepod-4",
            max_slices=5,
            priority=10,
        )

        group_v5p = ScalingGroup(config_v5p, make_mock_vm_manager(), scale_up_cooldown=Duration.from_ms(0))
        group_v5lite = ScalingGroup(config_v5lite, make_mock_vm_manager(), scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler({"v5p-group": group_v5p, "v5lite-group": group_v5lite})

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5litepod-4", count=2)]
        decisions = autoscaler.evaluate(demand)

        # Only v5lite-group should receive demand
        assert len(decisions) == 1
        assert decisions[0].scale_group == "v5lite-group"

    def test_demand_with_no_matching_group_is_unmet(self):
        """Demand for unknown accelerator type results in unmet demand."""
        config = config_pb2.ScaleGroupConfig(
            name="test-group",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=10,
        )

        group = ScalingGroup(config, make_mock_vm_manager(), scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler({"test-group": group})

        # Request an accelerator type that doesn't exist
        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="unknown-type", count=2)]
        decisions = autoscaler.evaluate(demand)

        # No decisions should be made (no matching group)
        assert len(decisions) == 0

    def test_multiple_demand_entries_route_independently(self):
        """Multiple demand entries with different accelerator types route to appropriate groups."""
        config_v5p = config_pb2.ScaleGroupConfig(
            name="v5p-group",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=10,
        )
        config_v5lite = config_pb2.ScaleGroupConfig(
            name="v5lite-group",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5litepod-4",
            max_slices=5,
            priority=10,
        )

        group_v5p = ScalingGroup(config_v5p, make_mock_vm_manager(), scale_up_cooldown=Duration.from_ms(0))
        group_v5lite = ScalingGroup(config_v5lite, make_mock_vm_manager(), scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler({"v5p-group": group_v5p, "v5lite-group": group_v5lite})

        demand = [
            DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=2),
            DemandEntry(device_type=DeviceType.TPU, device_variant="v5litepod-4", count=3),
        ]
        decisions = autoscaler.evaluate(demand)

        # Both groups should get scale-up decisions
        assert len(decisions) == 2
        groups_in_decisions = {d.scale_group for d in decisions}
        assert "v5p-group" in groups_in_decisions
        assert "v5lite-group" in groups_in_decisions


class TestPreemptibleRouting:
    """Tests for preemptible demand routing."""

    def test_route_demand_filters_by_preemptible_true(self):
        """Demand with preemptible=True only routes to preemptible groups."""
        config_preemptible = config_pb2.ScaleGroupConfig(
            name="preemptible-group",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=10,
            preemptible=True,
        )
        config_on_demand = config_pb2.ScaleGroupConfig(
            name="on-demand-group",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=10,
        )

        group_preemptible = ScalingGroup(config_preemptible, make_mock_vm_manager())
        group_on_demand = ScalingGroup(config_on_demand, make_mock_vm_manager())

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=2, preemptible=True)]
        result = route_demand([group_preemptible, group_on_demand], demand)

        assert result.allocations["preemptible-group"] == 2
        assert result.allocations["on-demand-group"] == 0

    def test_route_demand_filters_by_preemptible_false(self):
        """Demand with preemptible=False only routes to non-preemptible groups."""
        config_preemptible = config_pb2.ScaleGroupConfig(
            name="preemptible-group",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=10,
            preemptible=True,
        )
        config_on_demand = config_pb2.ScaleGroupConfig(
            name="on-demand-group",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=10,
        )

        group_preemptible = ScalingGroup(config_preemptible, make_mock_vm_manager())
        group_on_demand = ScalingGroup(config_on_demand, make_mock_vm_manager())

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=2, preemptible=False)]
        result = route_demand([group_preemptible, group_on_demand], demand)

        assert result.allocations["preemptible-group"] == 0
        assert result.allocations["on-demand-group"] == 2

    def test_route_demand_no_preference_routes_to_any(self):
        """Demand with preemptible=None routes to any matching group."""
        config_preemptible = config_pb2.ScaleGroupConfig(
            name="preemptible-group",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=10,
            preemptible=True,
        )
        config_on_demand = config_pb2.ScaleGroupConfig(
            name="on-demand-group",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=20,
        )

        group_preemptible = ScalingGroup(config_preemptible, make_mock_vm_manager())
        group_on_demand = ScalingGroup(config_on_demand, make_mock_vm_manager())

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=3, preemptible=None)]
        result = route_demand([group_preemptible, group_on_demand], demand)

        # Both groups are eligible; preemptible has higher priority (10 < 20)
        assert result.allocations["preemptible-group"] == 3
        assert result.unmet_demand == 0


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

        config_primary = config_pb2.ScaleGroupConfig(
            name="primary",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=10,
        )
        config_fallback = config_pb2.ScaleGroupConfig(
            name="fallback",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=20,
        )

        manager_primary = FakeVmManager(
            FakeVmManagerConfig(config=config_primary, failure_mode=FailureMode.QUOTA_EXCEEDED)
        )
        manager_fallback = FakeVmManager(FakeVmManagerConfig(config=config_fallback))

        group_primary = ScalingGroup(config_primary, manager_primary, scale_up_cooldown=Duration.from_ms(0))
        group_fallback = ScalingGroup(config_fallback, manager_fallback, scale_up_cooldown=Duration.from_ms(0))

        # Use short evaluation interval to allow rapid re-evaluation
        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
            config=config,
        )

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=2)]

        # First run: routes to primary, but execute fails with quota
        # This sets primary to QUOTA_EXCEEDED but nothing is created yet
        autoscaler.run_once(demand, {})

        # Wait for async scale-up to complete

        time.sleep(0.1)

        # Primary should be in quota exceeded state
        assert group_primary.availability().status == GroupAvailability.QUOTA_EXCEEDED
        # Fallback hasn't been tried yet (routing happened before quota failure)
        assert group_fallback.slice_count() == 0

        # Second run: primary is now unavailable, demand routes to fallback
        autoscaler.run_once(demand, {})
        time.sleep(0.1)
        assert group_fallback.slice_count() == 1

        # Third run: another slice on fallback
        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        assert group_fallback.slice_count() == 2

    def test_quota_recovery_restores_primary_routing(self):
        """After quota timeout expires, demand routes to primary again.

        Note: execute() uses Timestamp.now().epoch_ms() internally, so quota_exceeded_until_ms is
        set relative to real time, not test timestamps. We use a very short
        quota_timeout_ms and wait for it to expire naturally.
        """

        from tests.cluster.vm.fakes import FakeVmManager, FakeVmManagerConfig, FailureMode
        from iris.cluster.vm.scaling_group import GroupAvailability
        from iris.time_utils import Timestamp

        config_primary = config_pb2.ScaleGroupConfig(
            name="primary",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=10,
        )
        config_fallback = config_pb2.ScaleGroupConfig(
            name="fallback",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=20,
        )

        manager_primary = FakeVmManager(
            FakeVmManagerConfig(config=config_primary, failure_mode=FailureMode.QUOTA_EXCEEDED)
        )
        manager_fallback = FakeVmManager(FakeVmManagerConfig(config=config_fallback))

        # Short quota timeout (1000ms) for testing
        group_primary = ScalingGroup(
            config_primary, manager_primary, scale_up_cooldown=Duration.from_ms(0), quota_timeout=Duration.from_ms(1000)
        )
        group_fallback = ScalingGroup(config_fallback, manager_fallback, scale_up_cooldown=Duration.from_ms(0))

        # Use short evaluation interval to allow rapid re-evaluation
        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
            config=config,
        )

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=1)]

        # First run: tries primary, fails with quota (nothing created yet)
        autoscaler.run_once(demand, {})
        time.sleep(0.1)  # Wait for async scale-up to complete
        ts_after_fail = Timestamp.now()
        assert group_primary.availability(ts_after_fail).status == GroupAvailability.QUOTA_EXCEEDED
        assert group_fallback.slice_count() == 0

        # Second run: primary unavailable, routes to fallback
        autoscaler.run_once(demand, {})
        time.sleep(0.1)  # Wait for async scale-up to complete
        assert group_fallback.slice_count() == 1

        # Wait for quota timeout to expire (1000ms + buffer)
        time.sleep(1.1)

        # Clear the failure mode so primary can succeed
        manager_primary.set_failure_mode(FailureMode.NONE)

        # After timeout: primary should be available again
        ts_now = Timestamp.now()
        assert group_primary.availability(ts_now).status == GroupAvailability.AVAILABLE

        # Increase demand to verify routing goes to primary now
        demand_increased = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=2)]
        decisions = autoscaler.evaluate(demand_increased, timestamp=ts_now)
        assert len(decisions) == 1
        assert decisions[0].scale_group == "primary"

    def test_backoff_cascades_to_fallback(self):
        """Generic failure triggers backoff, which cascades to fallback."""

        from tests.cluster.vm.fakes import FakeVmManager, FakeVmManagerConfig, FailureMode
        from iris.cluster.vm.scaling_group import GroupAvailability

        config_primary = config_pb2.ScaleGroupConfig(
            name="primary",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=10,
        )
        config_fallback = config_pb2.ScaleGroupConfig(
            name="fallback",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=20,
        )

        # Primary will fail with generic error (triggers backoff)
        manager_primary = FakeVmManager(
            FakeVmManagerConfig(config=config_primary, failure_mode=FailureMode.CREATE_FAILS)
        )
        manager_fallback = FakeVmManager(FakeVmManagerConfig(config=config_fallback))

        backoff = Duration.from_seconds(60.0)
        group_primary = ScalingGroup(
            config_primary, manager_primary, scale_up_cooldown=Duration.from_ms(0), backoff_initial=backoff
        )
        group_fallback = ScalingGroup(config_fallback, manager_fallback, scale_up_cooldown=Duration.from_ms(0))

        # Use short evaluation interval to allow rapid re-evaluation
        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
            config=config,
        )

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=1)]

        # First run: primary fails, triggers backoff
        autoscaler.run_once(demand, {})
        time.sleep(0.1)  # Wait for async scale-up to complete

        # Primary should be in backoff (not quota)
        assert group_primary.availability().status == GroupAvailability.BACKOFF
        assert group_primary.consecutive_failures == 1

        # Second run: should route to fallback
        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        assert group_fallback.slice_count() == 1

    def test_multiple_accelerator_types_route_independently(self):
        """Different accelerator types route through their own group chains."""
        from tests.cluster.vm.fakes import FakeVmManager, FakeVmManagerConfig

        config_v5p = config_pb2.ScaleGroupConfig(
            name="v5p-group",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=10,
        )
        config_v5lite = config_pb2.ScaleGroupConfig(
            name="v5lite-group",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5litepod-4",
            max_slices=5,
            priority=10,
        )

        manager_v5p = FakeVmManager(FakeVmManagerConfig(config=config_v5p))
        manager_v5lite = FakeVmManager(FakeVmManagerConfig(config=config_v5lite))

        group_v5p = ScalingGroup(config_v5p, manager_v5p, scale_up_cooldown=Duration.from_ms(0))
        group_v5lite = ScalingGroup(config_v5lite, manager_v5lite, scale_up_cooldown=Duration.from_ms(0))

        autoscaler = make_autoscaler(
            scale_groups={"v5p-group": group_v5p, "v5lite-group": group_v5lite},
        )

        # Demand for both types
        demand = [
            DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=2),
            DemandEntry(device_type=DeviceType.TPU, device_variant="v5litepod-4", count=1),
        ]

        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()  # Wait for async scale-ups to complete

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
        from iris.time_utils import Timestamp

        config_primary = config_pb2.ScaleGroupConfig(
            name="primary",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=2,
            priority=10,
        )
        config_fallback = config_pb2.ScaleGroupConfig(
            name="fallback",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=20,
        )

        manager_primary = FakeVmManager(FakeVmManagerConfig(config=config_primary))
        manager_fallback = FakeVmManager(FakeVmManagerConfig(config=config_fallback))

        group_primary = ScalingGroup(config_primary, manager_primary, scale_up_cooldown=Duration.from_ms(0))
        group_fallback = ScalingGroup(config_fallback, manager_fallback, scale_up_cooldown=Duration.from_ms(0))

        # Use short evaluation interval to allow rapid re-evaluation

        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
            config=config,
        )

        # Demand exceeds primary's max_slices (4 > 2)
        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=4)]

        # First run: routing allocates primary=2, fallback=2 (based on headroom)
        # Creates one slice in each group (both now have capacity=1 from booting)
        autoscaler.run_once(demand, {})
        time.sleep(0.1)  # Wait for async scale-up to complete
        assert group_primary.slice_count() == 1
        assert group_fallback.slice_count() == 1

        # Tick the managers to advance VM states (makes them ready)
        # This simulates time passing where VMs finish booting
        ts = Timestamp.now().epoch_ms()
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
        time.sleep(0.1)
        # Primary stays at 1 because demand=capacity
        assert group_primary.slice_count() == 1
        assert group_fallback.slice_count() == 2

        # Tick again
        manager_primary.tick(Timestamp.now().epoch_ms())
        manager_fallback.tick(Timestamp.now().epoch_ms())

        # Third run: primary still has headroom=1, but gets allocated demand<=capacity
        # Let's verify that when we have demand > total capacity, the system works
        autoscaler.run_once(demand, {})
        time.sleep(0.1)
        # Now primary should scale up because we need more capacity
        # total capacity = 1 + 2 = 3, demand = 4, so 1 more needed
        # primary gets demand=1 (headroom=1), capacity=1, no scale (1 <= 1)
        # fallback gets demand=3, capacity=2, needs to scale
        assert group_fallback.slice_count() == 3

        # After several more runs, demand should be met
        for _ in range(2):
            manager_primary.tick(Timestamp.now().epoch_ms())
            manager_fallback.tick(Timestamp.now().epoch_ms())
            autoscaler.run_once(demand, {})
            time.sleep(0.1)

        autoscaler._wait_for_inflight()

        # Total slices should meet or exceed demand
        total = group_primary.slice_count() + group_fallback.slice_count()
        assert total >= 4


class TestAutoscalerQuotaHandling:
    """Tests for quota exceeded error handling."""

    def test_quota_exceeded_sets_group_unavailable(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """QuotaExceededError sets group to QUOTA_EXCEEDED state."""
        from iris.cluster.vm.managed_vm import QuotaExceededError
        from iris.cluster.vm.scaling_group import GroupAvailability

        manager = make_mock_vm_manager()
        manager.create_vm_group.side_effect = QuotaExceededError("Quota exceeded")
        group = ScalingGroup(
            scale_group_config, manager, scale_up_cooldown=Duration.from_ms(0), quota_timeout=Duration.from_ms(60_000)
        )
        # Use short evaluation interval to avoid rate-limiting
        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler({"test-group": group}, config=config)

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=1)]
        autoscaler.run_once(demand, {}, timestamp=Timestamp.from_ms(1000))
        autoscaler._wait_for_inflight()  # Wait for async scale-up to complete

        # Group should be in QUOTA_EXCEEDED state
        state = group.availability(timestamp=Timestamp.from_ms(2000))
        assert state.status == GroupAvailability.QUOTA_EXCEEDED

    def test_quota_exceeded_routes_to_fallback_group(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """When primary group has quota exceeded, demand routes to fallback."""
        from iris.cluster.vm.managed_vm import QuotaExceededError

        config_primary = config_pb2.ScaleGroupConfig(
            name="primary",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=10,
        )
        config_fallback = config_pb2.ScaleGroupConfig(
            name="fallback",
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            max_slices=5,
            priority=20,
        )

        manager_primary = make_mock_vm_manager()
        manager_primary.create_vm_group.side_effect = QuotaExceededError("Quota exceeded")
        manager_fallback = make_mock_vm_manager()

        group_primary = ScalingGroup(
            config_primary,
            manager_primary,
            scale_up_cooldown=Duration.from_ms(0),
            quota_timeout=Duration.from_ms(60_000),
        )
        group_fallback = ScalingGroup(config_fallback, manager_fallback, scale_up_cooldown=Duration.from_ms(0))
        # Use short evaluation interval to avoid rate-limiting
        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler({"primary": group_primary, "fallback": group_fallback}, config=config)

        # First call: primary fails with quota, triggers quota state
        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=1)]
        autoscaler.run_once(demand, {}, timestamp=Timestamp.from_ms(1000))
        autoscaler._wait_for_inflight()  # Wait for async scale-up to complete

        # Second call: primary is in QUOTA_EXCEEDED, should route to fallback
        decisions = autoscaler.evaluate(demand, timestamp=Timestamp.from_ms(2000))

        assert len(decisions) == 1
        assert decisions[0].scale_group == "fallback"

    def test_quota_state_expires_after_timeout(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """QUOTA_EXCEEDED state expires after timeout."""
        from iris.cluster.vm.managed_vm import QuotaExceededError
        from iris.cluster.vm.scaling_group import GroupAvailability

        manager = make_mock_vm_manager()
        manager.create_vm_group.side_effect = QuotaExceededError("Quota exceeded")
        group = ScalingGroup(
            scale_group_config, manager, scale_up_cooldown=Duration.from_ms(0), quota_timeout=Duration.from_ms(1000)
        )

        # Directly trigger quota exceeded through scale_up with controlled timestamp
        with pytest.raises(QuotaExceededError):
            group.scale_up(timestamp=Timestamp.from_ms(1000))

        # Immediately after: QUOTA_EXCEEDED
        assert group.availability(Timestamp.from_ms(1100)).status == GroupAvailability.QUOTA_EXCEEDED

        # After timeout (1000 + 1000 = 2000): AVAILABLE
        assert group.availability(Timestamp.from_ms(2100)).status == GroupAvailability.AVAILABLE

    def test_generic_error_triggers_backoff_not_quota(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Non-quota errors trigger backoff, not quota exceeded state."""
        from iris.cluster.vm.scaling_group import GroupAvailability

        manager = make_mock_vm_manager()
        manager.create_vm_group.side_effect = RuntimeError("TPU unavailable")

        backoff = Duration.from_seconds(5.0)
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown=Duration.from_ms(0), backoff_initial=backoff)
        # Use short evaluation interval to avoid rate-limiting
        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(Duration.from_seconds(0.001).to_proto())
        autoscaler = make_autoscaler({"test-group": group}, config=config)

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=1)]
        autoscaler.run_once(demand, {}, timestamp=Timestamp.from_ms(1000))
        autoscaler._wait_for_inflight()  # Wait for async scale-up to complete

        # Should be in BACKOFF, not QUOTA_EXCEEDED
        state = group.availability(timestamp=Timestamp.from_ms(2000))
        assert state.status == GroupAvailability.BACKOFF
        assert group.consecutive_failures == 1


class TestAutoscalerActionLogging:
    """Tests for autoscaler action logging."""

    def test_action_log_records_scale_up(self, empty_autoscaler: Autoscaler):
        """Verify scale-up actions are logged."""
        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=2)]
        empty_autoscaler.run_once(demand, {})
        empty_autoscaler._wait_for_inflight()  # Wait for async scale-up

        # Check that the action log contains a scale_up action
        status = empty_autoscaler.get_status()
        assert len(status.recent_actions) == 1
        action = status.recent_actions[0]
        assert action.action_type == "scale_up"
        assert action.scale_group == "test-group"
        assert action.slice_id != ""  # Should have a slice ID
        assert "demand" in action.reason

    def test_action_log_records_failed_cleanup(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Verify failed slice cleanup actions are logged."""
        mock_slice = make_mock_slice("slice-001", any_failed=True)
        manager = make_mock_vm_manager(slices_to_discover=[mock_slice])
        group = ScalingGroup(scale_group_config, manager, scale_down_cooldown=Duration.from_ms(0))
        group.reconcile()
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
        from iris.cluster.vm.managed_vm import QuotaExceededError

        manager = make_mock_vm_manager()
        manager.create_vm_group.side_effect = QuotaExceededError("Quota exceeded in zone")
        group = ScalingGroup(scale_group_config, manager, scale_up_cooldown=Duration.from_ms(0))
        autoscaler = make_autoscaler({"test-group": group})

        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=1)]
        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()  # Wait for async scale-up

        status = autoscaler.get_status()
        assert len(status.recent_actions) == 1
        action = status.recent_actions[0]
        assert action.action_type == "quota_exceeded"
        assert action.scale_group == "test-group"
        assert "Quota exceeded" in action.reason

    def test_action_log_records_worker_failed(self, scale_group_config: config_pb2.ScaleGroupConfig):
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

    def test_action_log_bounded_to_100_entries(self, empty_autoscaler: Autoscaler):
        """Verify action log is bounded to 100 entries."""
        # Directly add 150 actions to the log (bypassing actual scaling)
        for i in range(150):
            empty_autoscaler._log_action("test_action", "test-group", reason=f"action {i}")

        status = empty_autoscaler.get_status()
        assert len(status.recent_actions) == 100
        # Oldest entries should be dropped (first 50)
        assert status.recent_actions[0].reason == "action 50"
        assert status.recent_actions[99].reason == "action 149"

    def test_get_status_includes_actions(self, empty_autoscaler: Autoscaler):
        """Verify get_status returns recent actions."""
        # Run a scale-up to generate an action
        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=1)]
        empty_autoscaler.run_once(demand, {})
        empty_autoscaler._wait_for_inflight()  # Wait for async scale-up

        status = empty_autoscaler.get_status()

        # Status should include groups, demand, and actions
        assert len(status.groups) == 1
        assert status.current_demand["test-group"] == 1
        assert len(status.recent_actions) >= 1
        assert status.recent_actions[0].action_type == "scale_up"

    def test_action_log_includes_timestamp(self, empty_autoscaler: Autoscaler):
        """Verify actions include valid timestamps."""
        from iris.time_utils import Timestamp

        before = Timestamp.now().epoch_ms()
        demand = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=1)]
        empty_autoscaler.run_once(demand, {})
        empty_autoscaler._wait_for_inflight()  # Wait for async scale-up
        after = Timestamp.now().epoch_ms()

        status = empty_autoscaler.get_status()
        action = status.recent_actions[0]
        assert before <= action.timestamp.epoch_ms <= after


class TestScalingGroupRequestingState:
    """Tests for REQUESTING state in ScalingGroup."""

    def test_mark_requesting_sets_requesting_state(self):
        """mark_requesting() causes availability() to return REQUESTING."""
        from iris.cluster.vm.scaling_group import GroupAvailability
        from iris.time_utils import Duration, Timestamp

        config = config_pb2.ScaleGroupConfig(
            name="test-group",
            min_slices=0,
            max_slices=5,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        )
        manager = make_mock_vm_manager()
        group = ScalingGroup(config, manager)

        ts = Timestamp.now()
        timeout = Duration.from_ms(120_000)  # 2 minutes

        # Mark as requesting
        group.mark_requesting(ts, timeout)

        # Should be in REQUESTING state
        availability = group.availability(ts)
        assert availability.status == GroupAvailability.REQUESTING

    def test_requesting_state_expires_after_timeout(self):
        """REQUESTING state expires after timeout."""
        from iris.cluster.vm.scaling_group import GroupAvailability
        from iris.time_utils import Duration, Timestamp

        config = config_pb2.ScaleGroupConfig(
            name="test-group",
            min_slices=0,
            max_slices=5,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        )
        manager = make_mock_vm_manager()
        group = ScalingGroup(config, manager)

        ts = Timestamp.now()
        timeout = Duration.from_ms(120_000)

        group.mark_requesting(ts, timeout)

        # During timeout, should be REQUESTING
        availability = group.availability(Timestamp.from_ms(ts.epoch_ms() + 60_000))
        assert availability.status == GroupAvailability.REQUESTING

        # After timeout, should be AVAILABLE
        availability = group.availability(Timestamp.from_ms(ts.epoch_ms() + 120_000 + 1000))
        assert availability.status == GroupAvailability.AVAILABLE

    def test_clear_requesting_removes_state(self):
        """clear_requesting() removes REQUESTING state."""
        from iris.cluster.vm.scaling_group import GroupAvailability
        from iris.time_utils import Duration, Timestamp

        config = config_pb2.ScaleGroupConfig(
            name="test-group",
            min_slices=0,
            max_slices=5,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        )
        manager = make_mock_vm_manager()
        group = ScalingGroup(config, manager)

        ts = Timestamp.now()
        group.mark_requesting(ts, Duration.from_ms(120_000))

        # Clear requesting
        group.clear_requesting()

        # Should be AVAILABLE again
        availability = group.availability(ts)
        assert availability.status == GroupAvailability.AVAILABLE

    def test_demand_routing_skips_requesting_groups(self):
        """route_demand() skips groups in REQUESTING state."""
        from iris.time_utils import Duration, Timestamp

        config1 = config_pb2.ScaleGroupConfig(
            name="group-1",
            min_slices=0,
            max_slices=5,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            priority=10,
        )
        config2 = config_pb2.ScaleGroupConfig(
            name="group-2",
            min_slices=0,
            max_slices=5,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
            accelerator_variant="v5p-8",
            priority=20,
        )

        manager1 = make_mock_vm_manager()
        manager2 = make_mock_vm_manager()
        group1 = ScalingGroup(config1, manager1)
        group2 = ScalingGroup(config2, manager2)

        ts = Timestamp.now()
        # Mark group1 as requesting (higher priority)
        group1.mark_requesting(ts, Duration.from_ms(120_000))

        demand_entries = [DemandEntry(device_type=DeviceType.TPU, device_variant="v5p-8", count=2)]

        # Route demand - should skip group1 and use group2
        result = route_demand([group1, group2], demand_entries, ts)

        assert result.allocations["group-1"] == 0
        assert result.allocations["group-2"] == 2
        assert result.unmet_demand == 0


class TestAutoscalerAsyncScaleUp:
    """Tests for async scale-up behavior."""

    def test_execute_scale_up_returns_immediately(self):
        """_execute_scale_up returns immediately without blocking."""

        from iris.time_utils import Timestamp

        config = config_pb2.ScaleGroupConfig(
            name="test-group",
            min_slices=0,
            max_slices=5,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        )

        # Create a manager with slow scale-up
        manager = make_mock_vm_manager()
        original_create = manager.create_vm_group.side_effect

        def slow_create(tags=None):
            time.sleep(0.5)  # Simulate slow VM creation
            return original_create(tags)

        manager.create_vm_group.side_effect = slow_create

        group = ScalingGroup(config, manager, scale_up_cooldown=Duration.from_ms(0))
        autoscaler = make_autoscaler({"test-group": group})

        decision = ScalingDecision(
            scale_group="test-group",
            action=ScalingAction.SCALE_UP,
            reason="test async",
        )

        # Execute should return quickly
        start = time.time()
        autoscaler.execute([decision], timestamp=Timestamp.now())
        elapsed = time.time() - start

        # Should return in < 0.1s, not wait for 0.5s creation
        assert elapsed < 0.1

        autoscaler._wait_for_inflight()  # Wait for async scale-up to avoid logging on closed streams

    def test_group_marked_requesting_during_scale_up(self):
        """Group is marked REQUESTING immediately after execute() is called."""

        from iris.cluster.vm.scaling_group import GroupAvailability
        from iris.time_utils import Timestamp

        config = config_pb2.ScaleGroupConfig(
            name="test-group",
            min_slices=0,
            max_slices=5,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        )
        # Create a manager with slow scale-up
        manager = make_mock_vm_manager()
        original_create = manager.create_vm_group.side_effect

        def slow_create(tags=None):
            time.sleep(0.2)  # Simulate slow VM creation
            return original_create(tags)

        manager.create_vm_group.side_effect = slow_create

        group = ScalingGroup(config, manager, scale_up_cooldown=Duration.from_ms(0))
        autoscaler = make_autoscaler({"test-group": group})

        ts = Timestamp.now().epoch_ms()
        decision = ScalingDecision(
            scale_group="test-group",
            action=ScalingAction.SCALE_UP,
            reason="test requesting",
        )

        autoscaler.execute([decision], timestamp=Timestamp.from_ms(ts))

        # Group should be in REQUESTING state immediately (before async operation completes)
        availability = group.availability(Timestamp.from_ms(ts))
        assert availability.status == GroupAvailability.REQUESTING

        # Wait for async operation to complete
        autoscaler._wait_for_inflight()

        # After completion, should be AVAILABLE again
        availability = group.availability(Timestamp.from_ms(ts + 300))
        assert availability.status == GroupAvailability.AVAILABLE

    def test_autoscaler_shutdown_waits_for_scale_up(self):
        """shutdown() waits for in-flight scale-ups to complete."""

        from iris.time_utils import Timestamp

        config = config_pb2.ScaleGroupConfig(
            name="test-group",
            min_slices=0,
            max_slices=5,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        )

        # Create a manager with slow scale-up
        manager = make_mock_vm_manager()
        original_create = manager.create_vm_group.side_effect
        create_completed = []

        def slow_create(tags=None):
            time.sleep(0.2)
            result = original_create(tags)
            create_completed.append(True)
            return result

        manager.create_vm_group.side_effect = slow_create

        group = ScalingGroup(config, manager, scale_up_cooldown=Duration.from_ms(0))
        autoscaler = make_autoscaler({"test-group": group})

        decision = ScalingDecision(
            scale_group="test-group",
            action=ScalingAction.SCALE_UP,
            reason="test shutdown",
        )

        autoscaler.execute([decision], timestamp=Timestamp.now())

        # Shutdown should wait for scale-up to complete
        autoscaler.shutdown()

        # Scale-up should have completed
        assert len(create_completed) == 1

    def test_autoscaler_shutdown_stops_vm_threads(self):
        """shutdown() calls stop() on all VM threads before terminating."""

        config = config_pb2.ScaleGroupConfig(
            name="test-group",
            min_slices=0,
            max_slices=5,
            accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        )

        # Create mock VM groups with VMs that track stop() calls
        discovered_slice = make_mock_slice("slice-001", all_ready=True)
        manager = make_mock_vm_manager(slices_to_discover=[discovered_slice])

        group = ScalingGroup(config, manager)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        # Track stop() calls on VMs
        stop_called = []
        for vm_group in group.vm_groups():
            for vm in vm_group.vms():
                original_stop = vm.stop

                def make_stop_wrapper(vm_id):
                    def stop_wrapper():
                        stop_called.append(vm_id)
                        return original_stop()

                    return stop_wrapper

                vm.stop = make_stop_wrapper(vm.info.vm_id)

        # Shutdown should call stop() on all VMs
        autoscaler.shutdown()

        # Verify stop() was called on the VM
        assert len(stop_called) == 1
