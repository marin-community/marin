# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for demand routing, bin packing, and compute_required_slices.

These tests exercise pure scheduling/routing logic. They call route_demand(),
compute_required_slices(), and first_fit_decreasing() directly -- no platform
or provider is needed.
"""

import pytest

from iris.cluster.controller.autoscaler import (
    AdditiveReq,
    DemandEntry,
    RoutingBudget,
    compute_required_slices,
    first_fit_decreasing,
    route_demand,
)
from iris.cluster.controller.scaling_group import ScalingGroup
from iris.cluster.constraints import (
    DeviceType,
    PlacementRequirements,
    WellKnownAttribute,
)
from iris.rpc import cluster_pb2, config_pb2
from rigging.timing import Duration, Timestamp
from tests.cluster.providers.conftest import (
    make_mock_platform,
    make_mock_slice_handle,
)

from tests.cluster.controller.conftest import (
    DEFAULT_RESOURCES,
    make_big_demand_entries as _make_big_demand_entries,
    mark_discovered_ready as _mark_discovered_ready,
    make_demand_entries,
    make_scale_group_config,
)

# ---------------------------------------------------------------------------
# first_fit_decreasing
# ---------------------------------------------------------------------------


class TestFirstFitDecreasing:
    """Unit tests for the FFD bin packing helper."""

    def test_basic_packing(self):
        """4 requests of (50, 50) each into bins of (100, 100) -> 2 VMs."""
        reqs = [AdditiveReq(cpu_millicores=50, memory_bytes=50, disk_bytes=0) for _ in range(4)]
        vm_cap = AdditiveReq(cpu_millicores=100, memory_bytes=100, disk_bytes=0)
        assert first_fit_decreasing(reqs, vm_cap) == 2

    def test_empty_reqs_returns_zero(self):
        vm_cap = AdditiveReq(cpu_millicores=100, memory_bytes=100, disk_bytes=0)
        assert first_fit_decreasing([], vm_cap) == 0

    def test_single_item_per_bin(self):
        """3 items that each fill a bin entirely -> 3 VMs."""
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
        # FFD sorts descending: [70,70,30,30]. 70+30 fits in 1 bin -> 2 VMs
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


# ---------------------------------------------------------------------------
# compute_required_slices
# ---------------------------------------------------------------------------


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

        # 16 entries at 1000m CPU, 1024 bytes mem -> all fit in 1 VM (128 cores, 128GiB)
        entries = make_demand_entries(16, device_type=DeviceType.CPU)
        assert compute_required_slices(group, entries) == 1

    def test_accelerator_entries_not_packed(self):
        """Accelerator entries get 1 VM each -- they are not bin-packed."""
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

        # 4 entries, each 128GiB = 4 VMs -> ceil(4/4) = 1 slice
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
        # 4 entries at 64GiB each -> 2 VMs -> ceil(2/4) = 1 slice
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


# ---------------------------------------------------------------------------
# Waterfall routing (route_demand)
# ---------------------------------------------------------------------------


class TestWaterfallRouting:
    """Tests for priority-based waterfall demand routing."""

    def test_routes_demand_to_highest_priority_group_first(self):
        """Demand routes to highest priority (lowest number) matching group."""
        config_high = make_scale_group_config(name="high-priority", max_slices=5, priority=10)
        config_low = make_scale_group_config(name="low-priority", max_slices=5, priority=20)

        group_high = ScalingGroup(config_high, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))
        group_low = ScalingGroup(config_low, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        demand = make_demand_entries(3, device_type=DeviceType.CPU, device_variant=None)
        result = route_demand([group_high, group_low], demand)

        assert len(result.routed_entries.get("high-priority", [])) == 3
        assert result.routed_entries.get("low-priority") is None

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

        demand = make_demand_entries(2, device_type=DeviceType.CPU, device_variant=None)
        result = route_demand([group_high, group_low], demand)

        assert len(result.routed_entries.get("high-priority", [])) == 2
        assert result.routed_entries.get("low-priority") is None

    def test_demand_overflows_to_lower_priority_when_at_max_slices(self):
        """When high-priority group is at max_slices, demand falls through to lower priority."""
        config_high = make_scale_group_config(name="high-priority", max_slices=2, priority=10)
        config_low = make_scale_group_config(name="low-priority", max_slices=5, priority=20)

        discovered = [make_mock_slice_handle(f"slice-{i}", all_ready=True) for i in range(2)]
        group_high = ScalingGroup(config_high, make_mock_platform(slices_to_discover=discovered))
        group_high.reconcile()

        group_low = ScalingGroup(config_low, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        demand = make_demand_entries(3, device_type=DeviceType.CPU, device_variant=None)
        result = route_demand([group_high, group_low], demand)

        # All routed to low-priority since high is at max
        assert len(result.routed_entries.get("low-priority", [])) >= 1

    def test_routing_filters_by_accelerator_type(self):
        """Only groups matching accelerator_type receive demand."""
        config_v5p = make_scale_group_config(name="v5p-group", accelerator_variant="v5p-8", max_slices=5, priority=10)
        config_v5lite = make_scale_group_config(
            name="v5lite-group", accelerator_variant="v5litepod-4", max_slices=5, priority=10
        )

        group_v5p = ScalingGroup(config_v5p, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))
        group_v5lite = ScalingGroup(config_v5lite, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5litepod-4")
        result = route_demand([group_v5p, group_v5lite], demand)

        assert len(result.routed_entries.get("v5lite-group", [])) == 2
        assert result.routed_entries.get("v5p-group") is None

    def test_demand_with_no_matching_group_is_unmet(self):
        """Demand for unknown accelerator type results in unmet demand."""
        config = make_scale_group_config(name="test-group", max_slices=5, priority=10)

        group = ScalingGroup(config, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="unknown-type")
        result = route_demand([group], demand)

        assert len(result.unmet_entries) == 2

    def test_multiple_demand_entries_route_independently(self):
        """Multiple demand entries with different accelerator types route to appropriate groups."""
        config_v5p = make_scale_group_config(name="v5p-group", accelerator_variant="v5p-8", max_slices=5, priority=10)
        config_v5lite = make_scale_group_config(
            name="v5lite-group", accelerator_variant="v5litepod-4", max_slices=5, priority=10
        )

        group_v5p = ScalingGroup(config_v5p, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))
        group_v5lite = ScalingGroup(config_v5lite, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        demand = [
            *make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8", task_prefix="v5p"),
            *make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5litepod-4", task_prefix="v5lite"),
        ]
        result = route_demand([group_v5p, group_v5lite], demand)

        assert len(result.routed_entries.get("v5p-group", [])) == 1
        assert len(result.routed_entries.get("v5lite-group", [])) == 1

    def test_flexible_variant_routes_to_matching_group(self):
        """Demand with multiple device_variants routes to any matching group."""
        config_v4 = make_scale_group_config(name="v4-group", accelerator_variant="v4-8", max_slices=5, priority=10)
        config_v5p = make_scale_group_config(name="v5p-group", accelerator_variant="v5p-8", max_slices=5, priority=20)

        group_v4 = ScalingGroup(config_v4, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))
        group_v5p = ScalingGroup(config_v5p, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        # Demand accepts either v4-8 or v5p-8; should route to v4-group (higher priority)
        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variants=frozenset({"v4-8", "v5p-8"}))
        result = route_demand([group_v4, group_v5p], demand)

        assert len(result.routed_entries.get("v4-group", [])) >= 1

    def test_flexible_variant_no_match_for_missing_variant(self):
        """Flexible demand with no matching groups is unmet."""
        config_v4 = make_scale_group_config(name="v4-group", accelerator_variant="v4-8", max_slices=5, priority=10)

        group_v4 = ScalingGroup(config_v4, make_mock_platform(), scale_up_cooldown=Duration.from_ms(0))

        # Demand requires v5p-8 or v5litepod-4, but only v4-8 group exists
        demand = make_demand_entries(1, device_type=DeviceType.TPU, device_variants=frozenset({"v5p-8", "v5litepod-4"}))
        result = route_demand([group_v4], demand)
        assert len(result.unmet_entries) == 1

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

        demand = make_demand_entries(2, device_type=DeviceType.CPU, device_variant=None)
        result = route_demand([group_primary, group_fallback], demand, ts)

        assert len(result.routed_entries.get("fallback", [])) == 2
        status_by_group = {s.group: s for s in result.group_statuses}
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

        demand = make_demand_entries(2, device_type=DeviceType.CPU, device_variant=None)
        result = route_demand([group_primary, group_fallback], demand, ts)

        assert len(result.routed_entries.get("fallback", [])) == 2

    def test_cooldown_does_not_cause_fallthrough(self):
        """Groups in COOLDOWN still accept demand -- demand does not fall through."""
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

        demand = make_demand_entries(2, device_type=DeviceType.CPU, device_variant=None)
        result = route_demand([group_a, group_b], demand, eval_ts)

        assert len(result.routed_entries.get("group-a", [])) == 2
        assert result.routed_entries.get("group-b") is None

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

        demand = make_demand_entries(2, device_type=DeviceType.CPU, device_variant=None)
        result = route_demand([group_a, group_b], demand)

        # 2 tiny CPU entries pack into 1 slice
        assert len(result.routed_entries.get("group-b", [])) == 2


# ---------------------------------------------------------------------------
# Preemptible routing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Region routing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Zone routing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Committed budget routing
# ---------------------------------------------------------------------------


class TestCommittedBudgetRouting:
    """Tests for two-phase routing with committed budgets."""

    def test_committed_budget_retains_demand_for_requesting_group(self):
        """Demand sticks to a group with requesting slices even when a fresh group is available."""
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
        """When committed budget is insufficient, overflow goes through the waterfall."""
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

        # v6e has 1 requesting slice -> committed budget = 1 VM, full budget = 2 VMs
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
        """Groups with 0 requesting slices get no committed budget -- normal waterfall only."""
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


# ---------------------------------------------------------------------------
# Packing-aware routing
# ---------------------------------------------------------------------------


class TestPackingRouting:
    """Tests for packing-aware routing and scaling decisions."""

    def test_packing_allows_multiple_cpu_tasks_per_vm(self):
        """16 CPU tasks at 32GiB each pack into 4 VMs of 128GiB -> 1 slice."""
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

        # 8 entries at 32GiB each -> 2 VMs needed -> ceil(2/4) = 1 slice.
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

        # 16 entries at 32GiB -> 4 VMs -> ceil(4/4) = 1 slice.
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

        # 16 entries -> 1 slice needed, 1 exists -> group_to_launch = 0
        entries = _make_big_demand_entries(
            16,
            cpu_millicores=32000,
            memory_bytes=32 * 1024**3,
            device_type=DeviceType.TPU,
            device_variants=frozenset({"v5p-8"}),
        )
        result = route_demand([group], entries)
        assert result.group_to_launch.get("test-group", 0) == 0


# ---------------------------------------------------------------------------
# Routing bin packing
# ---------------------------------------------------------------------------


class TestRoutingBinPacking:
    """Tests for per-VM bin packing during routing."""

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
            # 3 packable entries (all fit in 1 VM -> ceil(1/2) = 1 slice)
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
        # Each accelerator entry needs its own VM -> 2 VMs -> 2 slices (num_vms=1).
        assert result.group_required_slices["group-a"] == 2


# ---------------------------------------------------------------------------
# Coscheduling feasibility
# ---------------------------------------------------------------------------


class TestCheckCoschedulingFeasibility:
    """Tests for Autoscaler.check_coscheduling_feasibility()."""

    def _make_constraints(self):
        return make_demand_entries(1, device_type=DeviceType.TPU, device_variant="v5p-8")[0].constraints

    def _make_autoscaler(self, groups):
        from iris.cluster.controller.autoscaler import Autoscaler

        return Autoscaler(
            scale_groups=groups,
            evaluation_interval=Duration.from_seconds(0.1),
            platform=make_mock_platform(),
        )

    def test_feasible_exact_match(self):
        """Replicas == num_vms is feasible."""
        config = make_scale_group_config(name="group-4", max_slices=5, num_vms=4)
        autoscaler = self._make_autoscaler({"group-4": ScalingGroup(config, make_mock_platform())})
        assert autoscaler.check_coscheduling_feasibility(4, self._make_constraints()) is None

    def test_feasible_exact_multiple(self):
        """Replicas that are an exact multiple of num_vms are feasible."""
        config = make_scale_group_config(name="group-4", max_slices=5, num_vms=4)
        autoscaler = self._make_autoscaler({"group-4": ScalingGroup(config, make_mock_platform())})
        assert autoscaler.check_coscheduling_feasibility(8, self._make_constraints()) is None

    def test_infeasible_not_a_multiple(self):
        """Replicas that aren't a multiple of any group's num_vms are rejected."""
        config = make_scale_group_config(name="group-3", max_slices=5, num_vms=3)
        autoscaler = self._make_autoscaler({"group-3": ScalingGroup(config, make_mock_platform())})
        result = autoscaler.check_coscheduling_feasibility(8, self._make_constraints())
        assert result is not None
        assert "8" in result

    def test_infeasible_no_group_matches_constraints(self):
        """Returns error when no group matches the device constraints."""
        config = make_scale_group_config(
            name="gpu-group", max_slices=5, num_vms=8, accelerator_type=config_pb2.ACCELERATOR_TYPE_GPU
        )
        autoscaler = self._make_autoscaler({"gpu-group": ScalingGroup(config, make_mock_platform())})
        result = autoscaler.check_coscheduling_feasibility(8, self._make_constraints())
        assert result is not None
        assert "no scaling group matches" in result

    def test_no_groups_returns_none(self):
        """Returns None when there are no groups (no validation possible)."""
        autoscaler = self._make_autoscaler({})
        assert autoscaler.check_coscheduling_feasibility(8, []) is None
