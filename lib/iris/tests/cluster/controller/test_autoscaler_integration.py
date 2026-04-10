# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Autoscaler with real GcpWorkerProvider(InMemoryGcpService).

These tests exercise the full autoscaler lifecycle -- scale-up, waterfall cascading,
quota handling, backoff, and multi-tier scaling -- using real provider implementations
backed by in-memory fakes rather than MagicMock.
"""

import threading

from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.models import DemandEntry, ScalingAction
from iris.cluster.controller.autoscaler.scaling_group import GroupAvailability, ScalingGroup
from iris.cluster.constraints import DeviceType
from iris.cluster.providers.gcp.fake import InMemoryGcpService
from iris.cluster.providers.gcp.workers import GcpWorkerProvider
from iris.cluster.providers.types import CloudSliceState
from iris.cluster.service_mode import ServiceMode
from iris.rpc import config_pb2
from iris.time_proto import duration_to_proto
from rigging.timing import Duration, Timestamp

from tests.cluster.controller.conftest import (
    advance_all_tpus,
    make_autoscaler,
    make_big_demand_entries as _make_big_demand_entries,
    make_demand_entries,
    make_gcp_provider,
    make_scale_group_config,
    mark_all_slices_ready as _mark_all_slices_ready,
)

# ---------------------------------------------------------------------------
# Waterfall end-to-end (real GcpWorkerProvider)
# ---------------------------------------------------------------------------


class TestAutoscalerWaterfallEndToEnd:
    """End-to-end tests for waterfall routing with GcpWorkerProvider(InMemoryGcpService)."""

    def test_demand_cascades_through_priority_groups_on_quota(self):
        """Full cascade: quota on primary routes to secondary."""
        config_primary = make_scale_group_config(name="primary", max_slices=5, priority=10, zones=["us-central1-a"])
        config_fallback = make_scale_group_config(name="fallback", max_slices=5, priority=20, zones=["us-central1-a"])

        platform_primary, service_primary = make_gcp_provider(config_primary)
        platform_fallback, _ = make_gcp_provider(config_fallback)
        service_primary.set_zone_quota("us-central1-a", 0)

        group_primary = ScalingGroup(config_primary, platform_primary, scale_up_cooldown=Duration.from_ms(0))
        group_fallback = ScalingGroup(config_fallback, platform_fallback, scale_up_cooldown=Duration.from_ms(0))

        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(duration_to_proto(Duration.from_seconds(0.001)))
        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
            config=config,
        )

        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8")

        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()

        assert group_primary.availability().status == GroupAvailability.QUOTA_EXCEEDED
        assert group_fallback.slice_count() == 0

        # 2 TPU entries are VM-exclusive -> 2 slices on fallback
        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        assert group_fallback.slice_count() == 2

    def test_quota_recovery_restores_primary_routing(self):
        """After quota timeout expires, demand routes to primary again."""
        config_primary = make_scale_group_config(name="primary", max_slices=5, priority=10, zones=["us-central1-a"])
        config_fallback = make_scale_group_config(name="fallback", max_slices=5, priority=20, zones=["us-central1-a"])

        platform_primary, service_primary = make_gcp_provider(config_primary)
        platform_fallback, _ = make_gcp_provider(config_fallback)
        service_primary.set_zone_quota("us-central1-a", 0)

        group_primary = ScalingGroup(
            config_primary, platform_primary, scale_up_cooldown=Duration.from_ms(0), quota_timeout=Duration.from_ms(1000)
        )
        group_fallback = ScalingGroup(config_fallback, platform_fallback, scale_up_cooldown=Duration.from_ms(0))

        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(duration_to_proto(Duration.from_seconds(0.001)))
        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
            config=config,
        )

        demand = make_demand_entries(1, device_type=DeviceType.CPU, device_variant=None)

        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        ts_after_fail = Timestamp.now()
        assert group_primary.availability(ts_after_fail).status == GroupAvailability.QUOTA_EXCEEDED
        assert group_fallback.slice_count() == 0

        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        assert group_fallback.slice_count() == 1

        # Restore quota so primary can create slices again
        service_primary.set_zone_quota("us-central1-a", 100)

        # Jump past the 1000ms quota timeout by using a synthetic timestamp
        # instead of sleeping for real wall-clock time.
        ts_after_quota_expires = ts_after_fail.add_ms(1100)
        assert group_primary.availability(ts_after_quota_expires).status == GroupAvailability.AVAILABLE

        demand_increased = make_demand_entries(2, device_type=DeviceType.CPU, device_variant=None)
        decisions = autoscaler.evaluate(demand_increased, timestamp=ts_after_quota_expires)
        assert len(decisions) == 1
        assert decisions[0].scale_group == "primary"

        autoscaler.shutdown()

    def test_full_group_cascades_to_fallback(self):
        """When primary group hits max_slices with all slices READY, demand cascades to fallback."""

        config_primary = make_scale_group_config(name="primary", max_slices=1, priority=10, zones=["us-central1-a"])
        config_fallback = make_scale_group_config(name="fallback", max_slices=5, priority=20, zones=["us-central1-a"])

        platform_primary, service_primary = make_gcp_provider(config_primary)
        platform_fallback, _ = make_gcp_provider(config_fallback)

        group_primary = ScalingGroup(config_primary, platform_primary, scale_up_cooldown=Duration.from_ms(0))
        group_fallback = ScalingGroup(config_fallback, platform_fallback, scale_up_cooldown=Duration.from_ms(0))

        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(duration_to_proto(Duration.from_seconds(0.001)))
        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
            config=config,
        )

        demand = make_demand_entries(1, device_type=DeviceType.CPU, device_variant=None)
        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        assert group_primary.slice_count() == 1

        # Mark the primary slice as READY so it enters AT_MAX_SLICES (rejecting)
        advance_all_tpus(service_primary, "READY")
        _mark_all_slices_ready(group_primary)

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

        platform_v5p, _ = make_gcp_provider(config_v5p)
        platform_v5lite, _ = make_gcp_provider(config_v5lite)

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
        """When high-priority group fills up, overflow goes to lower priority."""

        config_primary = make_scale_group_config(name="primary", max_slices=1, priority=10, zones=["us-central1-a"])
        config_fallback = make_scale_group_config(name="fallback", max_slices=5, priority=20, zones=["us-central1-a"])

        platform_primary, _ = make_gcp_provider(config_primary)
        platform_fallback, _ = make_gcp_provider(config_fallback)

        group_primary = ScalingGroup(config_primary, platform_primary, scale_up_cooldown=Duration.from_ms(0))
        group_fallback = ScalingGroup(config_fallback, platform_fallback, scale_up_cooldown=Duration.from_ms(0))

        config = config_pb2.AutoscalerConfig()
        config.evaluation_interval.CopyFrom(duration_to_proto(Duration.from_seconds(0.001)))
        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
            config=config,
        )

        from iris.cluster.constraints import PlacementRequirements
        from iris.rpc import job_pb2

        big_resources = job_pb2.ResourceSpecProto(cpu_millicores=128000, memory_bytes=128 * 1024**3)
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

        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        assert group_primary.slice_count() == 1
        assert group_fallback.slice_count() == 2

        total = group_primary.slice_count() + group_fallback.slice_count()
        assert total == 3

    def test_demand_cascades_through_priority_groups_on_backoff(self):
        """E2E: primary create fails -> BACKOFF, second run cascades to fallback."""
        config_primary = make_scale_group_config(name="primary", max_slices=5, priority=10, zones=["us-central1-a"])
        config_fallback = make_scale_group_config(name="fallback", max_slices=5, priority=20, zones=["us-central1-a"])

        platform_primary, service_primary = make_gcp_provider(config_primary)
        platform_fallback, _ = make_gcp_provider(config_fallback)
        service_primary.set_tpu_type_unavailable("v5p-8")

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
        config.evaluation_interval.CopyFrom(duration_to_proto(Duration.from_seconds(0.001)))
        autoscaler = make_autoscaler(
            scale_groups={"primary": group_primary, "fallback": group_fallback},
            config=config,
        )

        demand = make_demand_entries(2, device_type=DeviceType.TPU, device_variant="v5p-8")

        # First run: primary attempts scale-up, fails -> enters BACKOFF
        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        assert group_primary.availability().status == GroupAvailability.BACKOFF
        assert group_primary.slice_count() == 0

        # Second run: primary in BACKOFF -> demand cascades to fallback
        autoscaler.run_once(demand, {})
        autoscaler._wait_for_inflight()
        assert group_fallback.slice_count() >= 1


# ---------------------------------------------------------------------------
# Bootstrap state tests (real GcpWorkerProvider)
# ---------------------------------------------------------------------------


def test_bootstrap_state_with_worker_config():
    """With worker_config, the slice enters BOOTSTRAPPING state when cloud is READY."""
    sg_config = make_scale_group_config(
        name="test-group",
        buffer_slices=0,
        max_slices=4,
        zones=["us-central1-a"],
    )
    worker_config = config_pb2.WorkerConfig(
        docker_image="test:latest",
        port=10001,
        controller_address="controller:10000",
        cache_dir="/tmp/iris-test-cache",
    )
    platform, service = make_gcp_provider(sg_config)
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

    assert group.slice_count() == 1

    # TPU is still in CREATING -- advance to READY
    advance_all_tpus(service, "READY")

    # With worker_config, the slice is BOOTSTRAPPING (waiting for health polls)
    slice_handle = group.slice_handles()[0]
    status = slice_handle.describe()
    assert status.state == CloudSliceState.BOOTSTRAPPING

    autoscaler.shutdown()


def test_no_bootstrap_without_worker_config():
    """Without worker_config, slices go directly to READY when cloud is READY."""
    sg_config = make_scale_group_config(
        name="test-group",
        buffer_slices=0,
        max_slices=4,
        zones=["us-central1-a"],
    )
    platform, service = make_gcp_provider(sg_config)
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

    # Advance TPUs to READY and refresh
    advance_all_tpus(service, "READY")
    autoscaler.refresh({})

    assert group.slice_count() == 1
    assert group.ready_slice_count() == 1

    # Without worker_config, describe() returns READY directly (no bootstrap)
    slice_handle = group.slice_handles()[0]
    status = slice_handle.describe()
    assert status.state == CloudSliceState.READY

    autoscaler.shutdown()


# ---------------------------------------------------------------------------
# Pending counter prevents double scale-up (real GcpWorkerProvider with slow create)
# ---------------------------------------------------------------------------


def test_pending_counter_prevents_double_scaleup():
    """Verify that the pending scale-up counter prevents double scale-up when
    create_slice takes longer than expected."""
    create_barrier = threading.Event()

    class SlowGcpWorkerProvider(GcpWorkerProvider):
        """GcpWorkerProvider where create_slice blocks until barrier is released."""

        def create_slice(self, config, worker_config=None):
            create_barrier.wait(timeout=10)
            return super().create_slice(config, worker_config)

    sg_config = make_scale_group_config(
        name="test-group",
        buffer_slices=0,
        max_slices=4,
        zones=["us-central1-a"],
    )
    service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project", label_prefix="iris")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=["us-central1-a"])
    platform = SlowGcpWorkerProvider(gcp_config=gcp_config, label_prefix="iris", gcp_service=service)
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

    # First run_once: demand=1, current=0 -> scale up.
    decisions1 = autoscaler.run_once(demand, {}, t0)
    assert len(decisions1) == 1
    assert decisions1[0].action == ScalingAction.SCALE_UP

    # Advance time arbitrarily far
    t1 = Timestamp.from_ms(t0.epoch_ms() + 600)

    # Second run_once: pending counter prevents double scale-up
    decisions2 = autoscaler.run_once(demand, {}, t1)
    assert len(decisions2) == 0, "Pending counter should prevent second scale-up"

    # Release the barrier so threads complete
    create_barrier.set()
    autoscaler._wait_for_inflight()

    # Only 1 slice was created
    assert group.slice_count() == 1

    autoscaler.shutdown()


# ---------------------------------------------------------------------------
# Multi-slice scale-up: incremental demand growth (real GcpWorkerProvider)
# ---------------------------------------------------------------------------


def test_incremental_demand_growth_triggers_scale_up():
    """Starting with small demand then adding more triggers appropriate multi-slice scale-up."""
    config = make_scale_group_config(name="test-group", max_slices=10, num_vms=1, priority=10)
    platform, service = make_gcp_provider(config)
    group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0), scale_up_rate_limit=1000)

    as_config = config_pb2.AutoscalerConfig()
    as_config.evaluation_interval.CopyFrom(duration_to_proto(Duration.from_seconds(0.001)))
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
    advance_all_tpus(service, "READY")
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
    # 12 slices needed, 4 exist -> 6 more (capped by max_slices=10)
    assert len(decisions) == 6
    assert all(d.action == ScalingAction.SCALE_UP for d in decisions)

    # Execute and verify
    autoscaler.execute(decisions, Timestamp.now())
    autoscaler._wait_for_inflight()
    assert group.slice_count() == 10


# ---------------------------------------------------------------------------
# Marin-style lifecycle (5 tiers with real GcpWorkerProvider)
# ---------------------------------------------------------------------------


def test_marin_style_lifecycle():
    """Full lifecycle with marin.yaml-style groups: 5 tiers of 2^N VMs per slice."""
    COOLDOWN_MS = 1000

    services: dict[str, InMemoryGcpService] = {}
    groups: dict[str, ScalingGroup] = {}

    for num_vms, priority in [(1, 10), (2, 20), (4, 30), (8, 40), (16, 50)]:
        name = f"tpu-{num_vms}vm"
        cfg = make_scale_group_config(name=name, max_slices=4, num_vms=num_vms, priority=priority)
        plat, svc = make_gcp_provider(cfg)
        services[name] = svc
        groups[name] = ScalingGroup(cfg, plat, scale_up_cooldown=Duration.from_ms(COOLDOWN_MS))

    as_config = config_pb2.AutoscalerConfig()
    as_config.evaluation_interval.CopyFrom(duration_to_proto(Duration.from_seconds(0.001)))
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
        for svc in services.values():
            advance_all_tpus(svc, "READY")
        for g in groups.values():
            _mark_all_slices_ready(g)

    def routed(group_name):
        return len(autoscaler._last_scale_plan.routing_decision.routed_entries.get(group_name, []))

    def assert_no_load_on_last():
        assert routed("tpu-16vm") == 0, "tpu-16vm should never receive load"
        assert groups["tpu-16vm"].slice_count() == 0

    # -- Phase 1: Fill tpu-1vm (highest priority) --

    t = Timestamp.from_ms(10_000)
    autoscaler.run_once(make_demand(2), {}, timestamp=t)
    autoscaler._wait_for_inflight()
    assert groups["tpu-1vm"].slice_count() == 2
    assert_no_load_on_last()

    # Cycle 2: demand grows to 4, cooldown blocks scale-up
    t = Timestamp.from_ms(10_500)
    advance(t)
    autoscaler.run_once(make_demand(4), {}, timestamp=t)
    autoscaler._wait_for_inflight()
    assert groups["tpu-1vm"].slice_count() == 2, "cooldown blocks scale-up"
    assert routed("tpu-1vm") == 4, "demand stays in tpu-1vm during cooldown"
    assert routed("tpu-2vm") == 0, "no cascade during cooldown"
    assert_no_load_on_last()

    # Cycle 3: cooldown expired -> tpu-1vm scales to max
    t = Timestamp.from_ms(11_100)
    advance(t)
    autoscaler.run_once(make_demand(4), {}, timestamp=t)
    autoscaler._wait_for_inflight()
    assert groups["tpu-1vm"].slice_count() == 4, "tpu-1vm at max_slices"
    assert_no_load_on_last()

    # -- Phase 2: tpu-1vm at max -> cascade fills tpu-2vm --

    t = Timestamp.from_ms(12_200)
    advance(t)
    autoscaler.run_once(make_demand(8), {}, timestamp=t)
    autoscaler._wait_for_inflight()
    assert groups["tpu-1vm"].slice_count() == 4, "tpu-1vm unchanged"
    assert routed("tpu-2vm") == 8, "all demand cascades to tpu-2vm"
    assert groups["tpu-2vm"].slice_count() == 4, "tpu-2vm filled to max"
    assert_no_load_on_last()

    # -- Phase 3: tpu-2vm at max -> cascade to tpu-4vm with cooldown --

    t = Timestamp.from_ms(13_300)
    advance(t)
    autoscaler.run_once(make_demand(8), {}, timestamp=t)
    autoscaler._wait_for_inflight()
    assert groups["tpu-4vm"].slice_count() == 2
    assert_no_load_on_last()

    # Cycle 6: tpu-4vm in cooldown
    t = Timestamp.from_ms(13_800)
    advance(t)
    autoscaler.run_once(make_demand(16), {}, timestamp=t)
    autoscaler._wait_for_inflight()
    assert groups["tpu-4vm"].slice_count() == 2, "cooldown blocks scale-up"
    assert routed("tpu-4vm") == 16, "demand stays in tpu-4vm during cooldown"
    assert routed("tpu-8vm") == 0, "no cascade to tpu-8vm during cooldown"
    assert_no_load_on_last()

    # Cycle 7: cooldown expired -> tpu-4vm scales to max
    t = Timestamp.from_ms(14_400)
    advance(t)
    autoscaler.run_once(make_demand(16), {}, timestamp=t)
    autoscaler._wait_for_inflight()
    assert groups["tpu-4vm"].slice_count() == 4, "tpu-4vm at max_slices"
    assert_no_load_on_last()

    # -- Phase 4: cascade to tpu-8vm --

    t = Timestamp.from_ms(15_500)
    advance(t)
    autoscaler.run_once(make_demand(28), {}, timestamp=t)
    autoscaler._wait_for_inflight()
    assert routed("tpu-8vm") == 28
    assert groups["tpu-8vm"].slice_count() == 4, "tpu-8vm at max_slices"
    assert_no_load_on_last()

    # -- Verify final state --
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


# ---------------------------------------------------------------------------
# Scale-up rate limiting (real GcpWorkerProvider)
# ---------------------------------------------------------------------------


class TestScaleUpRateLimiting:
    """Tests for per-group token bucket rate limiting of scale-up execution."""

    def test_rate_limited_scale_up_logs_action(self):
        """With rate_limit=1, 5 decisions produce 1 executed + 4 rate_limited actions."""
        config = make_scale_group_config(name="test-group", max_slices=10, num_vms=1, priority=10)
        platform, _ = make_gcp_provider(config)
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0), scale_up_rate_limit=1)

        as_config = config_pb2.AutoscalerConfig()
        as_config.evaluation_interval.CopyFrom(duration_to_proto(Duration.from_seconds(0.001)))
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
        platform, _ = make_gcp_provider(config)
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0), scale_up_rate_limit=2)

        as_config = config_pb2.AutoscalerConfig()
        as_config.evaluation_interval.CopyFrom(duration_to_proto(Duration.from_seconds(0.001)))
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

        # Advance time by 1 minute so rate-limit tokens refill
        ts2 = ts.add_ms(60_000)

        # Cycle 2: re-evaluate with same demand. 6 required, 2 pending -> 4 new needed
        decisions2 = autoscaler.evaluate(demand, timestamp=ts2)
        assert len(decisions2) == 4
        autoscaler.execute(decisions2, ts2)
        autoscaler._wait_for_inflight()
        assert group.slice_count() == 4

    def test_high_rate_limit_allows_all_decisions(self):
        """With a high rate limit, all decisions execute in one cycle."""
        config = make_scale_group_config(name="test-group", max_slices=10, num_vms=1, priority=10)
        platform, _ = make_gcp_provider(config)
        group = ScalingGroup(config, platform, scale_up_cooldown=Duration.from_ms(0), scale_up_rate_limit=1000)

        as_config = config_pb2.AutoscalerConfig()
        as_config.evaluation_interval.CopyFrom(duration_to_proto(Duration.from_seconds(0.001)))
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
