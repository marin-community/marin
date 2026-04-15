# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Autoscaler behavior.

These tests focus on observable behavior -- scaling decisions based on demand,
execution of those decisions, and integration with ScalingGroup.

Pure routing/packing logic is tested in test_demand_routing.py.
Integration tests with real GcpWorkerProvider are in test_autoscaler_integration.py.
"""

import time

import pytest

from iris.cluster.controller.autoscaler import Autoscaler, DEFAULT_UNRESOLVABLE_TIMEOUT
from iris.cluster.controller.autoscaler.models import ScalingAction, ScalingDecision
from iris.cluster.controller.autoscaler.routing import route_demand
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.cluster.providers.types import (
    CloudSliceState,
    QuotaExhaustedError,
    SliceStatus,
)
from tests.cluster.providers.conftest import (
    FakeSliceHandle,
    make_mock_platform,
    make_mock_slice_handle,
    make_mock_worker_handle,
)
from iris.cluster.constraints import DeviceType, WellKnownAttribute
from iris.cluster.types import WorkerStatus
from iris.rpc import config_pb2, vm_pb2
from iris.time_proto import duration_to_proto
from rigging.timing import Duration, Timestamp

from .conftest import (
    make_autoscaler,
    make_demand_entries,
    make_big_demand_entries as _make_big_demand_entries,
    make_scale_group_config,
    mark_discovered_ready as _mark_discovered_ready,
)


@pytest.fixture
def scale_group_config() -> config_pb2.ScaleGroupConfig:
    """A standard scale group configuration."""
    return make_scale_group_config(
        name="test-group",
        buffer_slices=0,
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
        assert "demand=1" in decisions[0].reason

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

    def test_scales_up_to_fill_buffer(self):
        """Scales up to fill buffer_slices even with zero demand."""
        config = make_scale_group_config(
            name="test-group",
            buffer_slices=2,
            max_slices=5,
            runtime_version="v2-alpha-tpuv5",
            zones=["us-central1-a"],
        )
        platform = make_mock_platform()
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0))
        autoscaler = make_autoscaler({"test-group": group})

        decisions = autoscaler.evaluate([])

        assert len(decisions) == 2
        assert all(d.action == ScalingAction.SCALE_UP for d in decisions)
        assert "target=2" in decisions[0].reason
        assert "buffer=2" in decisions[0].reason

    def test_no_scale_up_when_buffer_satisfied(self):
        """Does not scale up when buffer_slices already satisfied and no demand."""
        config = make_scale_group_config(
            name="test-group",
            buffer_slices=2,
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
        """Scales up when all ready workers are full and demand survives dry-run."""
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
        assert "demand=1" in decisions[0].reason


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

    def test_no_scale_down_at_buffer_target(self):
        """Does not scale down when at buffer target."""
        config = make_scale_group_config(
            name="test-group",
            buffer_slices=2,
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
        config = make_scale_group_config(name="known-group", buffer_slices=0, max_slices=5)
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

    def test_terminate_slices_for_workers_terminates_slice(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """terminate_slices_for_workers() terminates the slice containing the worker."""
        mock_handle = make_mock_slice_handle("slice-001", all_ready=True)
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})
        _mark_discovered_ready(group, [mock_handle])

        failed_worker_id = "slice-001-vm-0"
        autoscaler.terminate_slices_for_workers([failed_worker_id])

        assert group.slice_count() == 0

    def test_terminate_slices_for_workers_unknown_worker_is_noop(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """terminate_slices_for_workers() does nothing for unknown workers."""
        mock_handle = make_mock_slice_handle("slice-001", all_ready=True)
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        autoscaler.terminate_slices_for_workers(["unknown-worker-99"])

        assert group.slice_count() == 1

    def test_terminate_slices_for_workers_returns_sibling_worker_ids(
        self, scale_group_config: config_pb2.ScaleGroupConfig
    ):
        """terminate_slices_for_workers() returns sibling worker IDs for multi-VM slices."""
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

        # Fail the first worker -- should return 3 sibling worker IDs
        failed_worker_id = "slice-001-vm-0"
        siblings = autoscaler.terminate_slices_for_workers([failed_worker_id])

        expected_siblings = [f"slice-001-vm-{i}" for i in range(1, 4)]
        assert sorted(siblings) == sorted(expected_siblings)
        assert group.slice_count() == 0

    def test_terminate_slices_for_workers_returns_empty_for_single_vm_slice(
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
        siblings = autoscaler.terminate_slices_for_workers([failed_worker_id])

        assert siblings == []

    def test_terminate_slices_for_workers_dedupes_by_slice(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """Workers from the same slice trigger one slice termination."""
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

        siblings = autoscaler.terminate_slices_for_workers(["slice-001-vm-0", "slice-001-vm-1"])

        assert siblings == ["slice-001-vm-2", "slice-001-vm-3"]
        assert group.slice_count() == 0

    def test_terminate_slices_for_workers_cleans_up_even_if_terminate_fails(
        self, scale_group_config: config_pb2.ScaleGroupConfig
    ):
        """terminate_slices_for_workers() removes the slice even if terminate() raises."""
        mock_handle = make_mock_slice_handle("slice-001", all_ready=True)
        mock_handle.terminate_error = RuntimeError("resource not found")
        platform = make_mock_platform(slices_to_discover=[mock_handle])
        group = ScalingGroup(scale_group_config, platform)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})
        _mark_discovered_ready(group, [mock_handle])

        failed_worker_id = "slice-001-vm-0"
        siblings = autoscaler.terminate_slices_for_workers([failed_worker_id])

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
        assert not mock_handle.terminated


class TestAutoscalerStatusReporting:
    """Tests for status reporting."""

    def test_get_status_includes_all_groups(self):
        """get_status() includes status for all groups."""
        config1 = make_scale_group_config(name="group-1", buffer_slices=0, max_slices=5)
        config2 = make_scale_group_config(name="group-2", buffer_slices=0, max_slices=5)

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
        config = make_scale_group_config(name="test-group", buffer_slices=0, max_slices=5)
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

        # Unknown keys return None -- no address fallback
        assert autoscaler.get_vm(worker.internal_address) is None
        assert autoscaler.get_vm("192.168.0.99") is None


class TestAutoscalerQuotaHandling:
    """Tests for quota exceeded error handling."""

    def test_quota_exceeded_sets_group_unavailable(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """QuotaExhaustedError sets group to QUOTA_EXCEEDED state."""
        from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability

        platform = make_mock_platform()
        platform.create_slice.side_effect = QuotaExhaustedError("Quota exceeded")
        group = ScalingGroup(
            scale_group_config, platform, scale_up_cooldown=Duration.from_ms(0), quota_timeout=Duration.from_ms(60_000)
        )
        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(duration_to_proto(Duration.from_seconds(0.001)))
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
        config.evaluation_interval.CopyFrom(duration_to_proto(Duration.from_seconds(0.001)))
        autoscaler = make_autoscaler({"primary": group_primary, "fallback": group_fallback}, config=config)

        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")
        autoscaler.run_once(demand, {}, timestamp=Timestamp.from_ms(1000))
        autoscaler._wait_for_inflight()

        decisions = autoscaler.evaluate(demand, timestamp=Timestamp.from_ms(2000))

        assert len(decisions) == 1
        assert decisions[0].scale_group == "fallback"

    def test_quota_state_expires_after_timeout(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """QUOTA_EXCEEDED state expires after timeout."""
        from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability

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
        from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability

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
        config.evaluation_interval.CopyFrom(duration_to_proto(Duration.from_seconds(0.001)))
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
        assert "demand=" in action.reason

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
        autoscaler.terminate_slices_for_workers([failed_worker_id])

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
        from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability

        config = make_scale_group_config(name="test-group", buffer_slices=0, max_slices=5)
        platform = make_mock_platform()
        group = ScalingGroup(config, platform)

        ts = Timestamp.now()
        group.begin_scale_up()

        availability = group.availability(ts)
        assert availability.status == GroupAvailability.REQUESTING

    def test_complete_scale_up_clears_requesting_state(self):
        """complete_scale_up() removes REQUESTING state."""
        from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability

        config = make_scale_group_config(name="test-group", buffer_slices=0, max_slices=5)
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
        from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability

        config = make_scale_group_config(name="test-group", buffer_slices=0, max_slices=5)
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
        config = make_scale_group_config(name="test-group", buffer_slices=0, max_slices=1)
        platform = make_mock_platform()
        group = ScalingGroup(config, platform)

        group.begin_scale_up()
        assert group.slice_count() == 1
        assert not group.can_scale_up()

    def test_demand_routing_prefers_requesting_groups(self):
        """route_demand() prefers pending/requesting groups."""
        config1 = make_scale_group_config(name="group-1", buffer_slices=0, max_slices=5, priority=10)
        config2 = make_scale_group_config(name="group-2", buffer_slices=0, max_slices=5, priority=20)

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


class TestAutoscalerAsyncScaleUp:
    """Tests for async scale-up behavior."""

    def test_execute_scale_up_returns_immediately(self):
        """_execute_scale_up returns immediately without blocking."""
        config = make_scale_group_config(name="test-group", buffer_slices=0, max_slices=5)

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
        from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability

        config = make_scale_group_config(name="test-group", buffer_slices=0, max_slices=5)
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
        config = make_scale_group_config(name="test-group", buffer_slices=0, max_slices=5)

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
        config = make_scale_group_config(name="test-group", buffer_slices=0, max_slices=5)

        discovered_handle = make_mock_slice_handle("slice-001", all_ready=True)
        platform = make_mock_platform(slices_to_discover=[discovered_handle])

        group = ScalingGroup(config, platform)
        group.reconcile()
        autoscaler = make_autoscaler({"test-group": group})

        autoscaler.shutdown()

        # All slices should be terminated
        assert group.slice_count() == 0
        assert discovered_handle.terminated


class TestPerGroupWorkerConfig:
    """Tests for _per_group_worker_config merging worker attributes into WorkerConfig."""

    def test_merges_worker_attributes(self):
        """Worker attributes and scale group name are merged into WorkerConfig."""
        base_wc = config_pb2.WorkerConfig(
            docker_image="test:latest",
            port=10001,
            controller_address="controller:10000",
        )
        base_wc.task_env["MARIN_PREFIX"] = "s3://bucket/marin"
        sg_config = make_scale_group_config(name="west-group", max_slices=5)
        sg_config.worker.attributes["custom-label"] = "west-value"

        group = ScalingGroup(sg_config, make_mock_platform())
        autoscaler = make_autoscaler({"west-group": group}, base_worker_config=base_wc)

        wc = autoscaler._per_group_worker_config(group)

        assert wc is not None
        assert wc.docker_image == "test:latest"
        assert wc.worker_attributes["custom-label"] == "west-value"
        assert wc.task_env["MARIN_PREFIX"] == "s3://bucket/marin"
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
        sg_config.worker.attributes["custom-label"] = "value"
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
        sg_config.worker.attributes["custom-label"] = "value"

        group = ScalingGroup(sg_config, make_mock_platform())
        autoscaler = make_autoscaler({"west-group": group}, base_worker_config=base_wc)

        autoscaler._per_group_worker_config(group)

        assert "custom-label" not in base_wc.worker_attributes

    def test_worker_attributes_injected(self):
        """Worker attributes are injected into WorkerConfig."""
        base_wc = config_pb2.WorkerConfig(
            docker_image="ghcr.io/marin-community/iris-worker:latest",
            port=10001,
            controller_address="controller:10000",
        )
        sg_config = make_scale_group_config(name="eu-group", max_slices=5, zones=["europe-west4-b"])
        sg_config.worker.attributes["team"] = "euw4"

        group = ScalingGroup(sg_config, make_mock_platform())
        autoscaler = make_autoscaler({"eu-group": group}, base_worker_config=base_wc)

        wc = autoscaler._per_group_worker_config(group)

        assert wc is not None
        assert wc.worker_attributes["team"] == "euw4"

    def test_derives_region_and_zone_from_scale_group_when_missing(self):
        """Derived region and zone are injected when worker attrs omit them."""
        base_wc = config_pb2.WorkerConfig(
            docker_image="ghcr.io/marin-community/iris-worker:latest",
            port=10001,
            controller_address="controller:10000",
        )
        sg_config = make_scale_group_config(name="east-group", max_slices=5, zones=["us-east5-a"])

        group = ScalingGroup(sg_config, make_mock_platform())
        autoscaler = make_autoscaler({"east-group": group}, base_worker_config=base_wc)

        wc = autoscaler._per_group_worker_config(group)

        assert wc is not None
        assert wc.worker_attributes[WellKnownAttribute.REGION] == "us-east5"
        assert wc.worker_attributes[WellKnownAttribute.ZONE] == "us-east5-a"


class TestGpuScaleGroupBugs:
    """Reproduction tests for GPU scale group bugs observed on CoreWeave."""

    def test_freshly_ready_slice_has_nonzero_last_active(self):
        """When a slice transitions to READY, last_active should be initialized."""
        config = make_scale_group_config(
            name="h100-8x",
            buffer_slices=0,
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
        demand temporarily drops to 0."""
        config = make_scale_group_config(
            name="h100-8x",
            buffer_slices=0,
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

        slice_001 = group.get_slice("slice-001")
        slice_002 = group.get_slice("slice-002")
        wid1 = slice_001.describe().workers[0].worker_id
        wid2 = slice_002.describe().workers[0].worker_id
        vm_status_map = {
            wid1: WorkerStatus(worker_id=wid1, running_task_ids=frozenset()),
            wid2: WorkerStatus(worker_id=wid2, running_task_ids=frozenset()),
        }

        # Run 1 second after ready -- well within the 5-minute idle_threshold.
        ts = Timestamp.from_ms(10_000)
        autoscaler.run_once([], vm_status_map, ts)

        assert group.slice_count() == 2, (
            "Freshly-ready slices should be protected by idle_threshold (300s). "
            f"Got slice_count={group.slice_count()}"
        )


# --- Multi-slice and packing tests that exercise autoscaler.evaluate ---


class TestMultiSliceScaleUp:
    """Tests for multi-slice scale-up in a single evaluation cycle."""

    def test_multi_slice_scale_up(self):
        """Group with 0 existing slices scales up to meet full demand in one cycle."""
        config = make_scale_group_config(name="test-group", max_slices=5, num_vms=1, priority=10)
        group = ScalingGroup(
            config, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0), scale_up_rate_limit=1000
        )
        autoscaler = make_autoscaler({"test-group": group})

        # 5 big entries, each fills 1 VM, num_vms=1 -> 5 slices needed
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

        # 5 big entries, each fills 1 VM -> 5 slices needed, but max=3
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
        from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability

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
        """Entries route using budget from ready VMs, not just headroom."""
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

        # No demand -> no scale up (all tasks absorbed by scheduler dry-run)
        decisions = autoscaler.evaluate([])
        assert len(decisions) == 0

        # 5 entries that survived dry-run -> 5 VMs -> ceil(5/4) = 2 slices needed.
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
        assert "demand=2" in decisions[0].reason

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

        # 4 entries at 32GiB each -> 1 VM -> ceil(1/4) = 1 slice. But we have 2 slices.
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


class TestAutoscalerUnresolvableTimeout:
    """Tests for UNKNOWN slice -> FAILED after timeout behavior."""

    def _make_group_with_unknown_slice(
        self, scale_group_config: config_pb2.ScaleGroupConfig, created_at_ms: int
    ) -> tuple[Autoscaler, ScalingGroup, FakeSliceHandle]:
        """Set up a group with one BOOTING slice that reports UNKNOWN state."""
        handle = make_mock_slice_handle("slice-001", created_at_ms=created_at_ms)
        handle._status = SliceStatus(state=CloudSliceState.UNKNOWN, worker_count=0, workers=[])

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

        # Refresh at 5 min -- well under 15 min timeout
        autoscaler.refresh({}, timestamp=Timestamp.from_ms(5 * 60 * 1000))

        assert group.slice_count() == 1
        assert group.ready_slice_count() == 0
        autoscaler.shutdown()

    def test_unknown_after_timeout_triggers_failure(self, scale_group_config: config_pb2.ScaleGroupConfig):
        """A slice in UNKNOWN state past the timeout is failed and removed."""
        created_at_ms = 0
        autoscaler, group, _ = self._make_group_with_unknown_slice(scale_group_config, created_at_ms)

        # Refresh at 16 min -- past the 15 min timeout
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

        # First refresh: UNKNOWN at 5 min -> should stay BOOTING
        handle._status = SliceStatus(state=CloudSliceState.UNKNOWN, worker_count=0, workers=[])
        autoscaler.refresh({}, timestamp=Timestamp.from_ms(5 * 60 * 1000))
        assert group.slice_count() == 1

        # Second refresh: READY before timeout
        worker = make_mock_worker_handle("slice-001-vm-0", "10.0.1.0", vm_pb2.VM_STATE_READY)
        handle._status = SliceStatus(state=CloudSliceState.READY, worker_count=1, workers=[worker])
        autoscaler.refresh({}, timestamp=Timestamp.from_ms(10 * 60 * 1000))

        assert group.ready_slice_count() == 1
        autoscaler.shutdown()
