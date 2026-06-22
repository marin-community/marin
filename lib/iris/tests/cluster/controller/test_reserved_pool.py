# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fungible reservation chip ledger and the launch cap it feeds."""

import pytest
from iris.cluster.constraints import PlacementRequirements
from iris.cluster.controller.autoscaler.models import UNRANKED_DEMAND_BAND, DemandEntry, RoutingDecision
from iris.cluster.controller.autoscaler.planning import (
    _admit_in_band_order,
    _PoolCandidate,
    build_group_scale_plan,
    build_scale_plan,
)
from iris.cluster.controller.autoscaler.reserved_pool import build_reservation_ledger
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.rpc import job_pb2, vm_pb2
from rigging.timing import Timestamp

from tests.cluster.backends.conftest import make_fake_slice_handle, make_mock_platform

from .conftest import make_scale_group_config

# Lower band_sort_key = higher priority.
PRODUCTION = job_pb2.PRIORITY_BAND_PRODUCTION
BATCH = job_pb2.PRIORITY_BAND_BATCH


def _ready_group(
    name: str,
    variant: str,
    *,
    quota_pool: str,
    reservation_chips: int,
    slice_ids: list[str],
    vms_per_slice: int = 1,
) -> ScalingGroup:
    """Build a fungible-reservation ScalingGroup with READY slices.

    Each slice is discovered via the mock platform and marked READY so its
    worker ids populate, matching how the autoscaler tracks live slices.
    """
    config = make_scale_group_config(name=name, accelerator_variant=variant, max_slices=64)
    config.quota_pool = quota_pool
    config.reservation_chips = reservation_chips
    handles = [
        make_fake_slice_handle(
            sid,
            scale_group=name,
            vm_states=[vm_pb2.VM_STATE_READY] * vms_per_slice,
        )
        for sid in slice_ids
    ]
    platform = make_mock_platform(slices_to_discover=handles)
    group = ScalingGroup(config, platform)
    group.reconcile()
    for handle in handles:
        worker_ids = [w.worker_id for w in handle.describe().workers]
        group.mark_slice_ready(handle.slice_id, worker_ids)
    return group


# -- Reservation ledger chip buckets --


def test_live_chips_sum_across_variants_in_a_pool():
    # Two v4-8 slices (4 chips each) + one v4-16 slice (8 chips) = 16 chips
    # live against a shared 64-chip budget.
    v8 = _ready_group("v4-8", "v4-8", quota_pool="pool-a", reservation_chips=64, slice_ids=["a1", "a2"])
    v16 = _ready_group("v4-16", "v4-16", quota_pool="pool-a", reservation_chips=64, slice_ids=["b1"], vms_per_slice=2)

    ledger = build_reservation_ledger([v8, v16])

    assert set(ledger.pools) == {"pool-a"}
    pool = ledger.pools["pool-a"]
    assert pool.reservation_chips == 64
    assert pool.live_chips == 4 + 4 + 8
    assert pool.inflight_chips == 0
    assert pool.draining_chips == 0
    assert pool.allocated_chips == 16
    assert pool.free_chips == 64 - 16
    assert pool.incoming_chips == 64 - 16
    assert pool.utilization == pytest.approx(16 / 64)


def test_inflight_slices_counted_as_inflight_chips():
    # A pending scale-up (begin_scale_up) is one in-flight slice; a booting
    # discovered slice (not marked READY) is another. Both consume chips but
    # are not live.
    config = make_scale_group_config(name="v4-8", accelerator_variant="v4-8", max_slices=64)
    config.quota_pool = "pool-a"
    config.reservation_chips = 64
    booting = make_fake_slice_handle("boot1", scale_group="v4-8", vm_states=[vm_pb2.VM_STATE_BOOTING])
    platform = make_mock_platform(slices_to_discover=[booting])
    group = ScalingGroup(config, platform)
    group.reconcile()  # booting slice tracked as BOOTING (never marked READY)
    group.begin_scale_up()  # one pending (REQUESTING) scale-up

    pool = build_reservation_ledger([group]).pools["pool-a"]

    # 2 in-flight slices (1 booting + 1 requesting) * 4 chips each.
    assert pool.live_chips == 0
    assert pool.inflight_chips == 8
    assert pool.draining_chips == 0
    assert pool.free_chips == 64 - 8


def test_draining_slice_counted_as_draining_not_free():
    # A drained slice stays counted until reaped: its chips are draining, not
    # free, but they are incoming (free + draining).
    group = _ready_group("v4-8", "v4-8", quota_pool="pool-a", reservation_chips=16, slice_ids=["a1", "a2"])
    group.drain_slice("a1")

    pool = build_reservation_ledger([group]).pools["pool-a"]

    assert pool.live_chips == 4  # only a2 remains live
    assert pool.draining_chips == 4  # a1 is draining
    assert pool.allocated_chips == 8  # both still allocated
    assert pool.free_chips == 16 - 8
    assert pool.incoming_chips == (16 - 8) + 4  # free + draining


def test_inflight_slices_by_variant():
    # Two distinct variants share a pool; each contributes its in-flight slice
    # count keyed by variant.
    config8 = make_scale_group_config(name="v4-8", accelerator_variant="v4-8", max_slices=64)
    config8.quota_pool = "pool-a"
    config8.reservation_chips = 64
    g8 = ScalingGroup(config8, make_mock_platform())
    g8.begin_scale_up()
    g8.begin_scale_up()

    config16 = make_scale_group_config(name="v4-16", accelerator_variant="v4-16", max_slices=64)
    config16.quota_pool = "pool-a"
    config16.reservation_chips = 64
    g16 = ScalingGroup(config16, make_mock_platform())
    g16.begin_scale_up()

    ledger = build_reservation_ledger([g8, g16])

    assert ledger.inflight_slices("pool-a", "v4-8") == 2
    assert ledger.inflight_slices("pool-a", "v4-16") == 1
    assert ledger.inflight_slices("pool-a", "v4-32") == 0
    assert ledger.inflight_slices("absent-pool", "v4-8") == 0


def test_distinct_pools_bucket_separately():
    a = _ready_group("v4-8", "v4-8", quota_pool="pool-a", reservation_chips=32, slice_ids=["a1"])
    b = _ready_group("v5p-8", "v5p-8", quota_pool="pool-b", reservation_chips=16, slice_ids=["b1"])

    ledger = build_reservation_ledger([a, b])

    assert ledger.pools["pool-a"].live_chips == 4
    assert ledger.pools["pool-b"].live_chips == 4
    assert ledger.free_chips("pool-a") == 28
    assert ledger.free_chips("pool-b") == 12


def test_non_fungible_groups_excluded():
    fungible = _ready_group("v4-8", "v4-8", quota_pool="pool-a", reservation_chips=32, slice_ids=["a1"])
    # reservation_chips defaults to 0 -> not part of any fungible pool.
    plain_config = make_scale_group_config(name="plain", accelerator_variant="v4-8", max_slices=8)
    plain_config.quota_pool = "pool-a"
    plain = ScalingGroup(plain_config, make_mock_platform())

    ledger = build_reservation_ledger([fungible, plain])

    assert set(ledger.pools) == {"pool-a"}
    # Only the fungible group's slice counts; the plain group is invisible.
    assert ledger.pools["pool-a"].live_chips == 4


def test_conflicting_budgets_in_one_pool_raise():
    a = _ready_group("v4-8", "v4-8", quota_pool="pool-a", reservation_chips=64, slice_ids=["a1"])
    b = _ready_group("v4-16", "v4-16", quota_pool="pool-a", reservation_chips=128, slice_ids=["b1"], vms_per_slice=2)

    with pytest.raises(ValueError, match="conflicting reservation_chips"):
        build_reservation_ledger([a, b])


def test_reservation_chips_without_quota_pool_raises():
    config = make_scale_group_config(name="v4-8", accelerator_variant="v4-8", max_slices=8)
    config.reservation_chips = 64
    # quota_pool intentionally left empty.
    group = ScalingGroup(config, make_mock_platform())

    with pytest.raises(ValueError, match="no quota_pool"):
        build_reservation_ledger([group])


# -- Reservation ledger lookup maps (worker/variant/chip) --


def test_maps_workers_variants_and_chips():
    v8 = _ready_group("v4-8", "v4-8", quota_pool="pool-a", reservation_chips=64, slice_ids=["a1"])
    v16 = _ready_group("v4-16", "v4-16", quota_pool="pool-a", reservation_chips=64, slice_ids=["b1"], vms_per_slice=2)

    ledger = build_reservation_ledger([v8, v16])

    assert ledger.variant_pool == {"v4-8": "pool-a", "v4-16": "pool-a"}
    assert ledger.chips_per_variant == {"v4-8": 4, "v4-16": 8}
    # Every worker in both groups maps back to the shared pool.
    assert ledger.worker_pool == {
        "a1-vm-0": "pool-a",
        "b1-vm-0": "pool-a",
        "b1-vm-1": "pool-a",
    }
    # ...and to its physical slice, so the preemption pass groups victims by
    # slice. The two VMs of the v4-16 slice share one slice id.
    assert ledger.worker_slice == {
        "a1-vm-0": "a1",
        "b1-vm-0": "b1",
        "b1-vm-1": "b1",
    }
    assert not ledger.is_empty()


def test_empty_when_no_fungible_groups():
    plain_config = make_scale_group_config(name="plain", accelerator_variant="v4-8", max_slices=8)
    plain = ScalingGroup(plain_config, make_mock_platform())

    ledger = build_reservation_ledger([plain])

    assert ledger.is_empty()
    assert ledger.variant_pool == {}
    assert ledger.pools == {}
    assert ledger.free_chips("anything") == 0
    assert ledger.incoming_chips("anything") == 0


def _demand_entry(band: int) -> DemandEntry:
    return DemandEntry(
        task_ids=("/u/t/0",),
        coschedule_group_id=None,
        normalized=PlacementRequirements(
            device_type=None, device_variants=None, preemptible=None, required_regions=None, required_zones=None
        ),
        constraints=[],
        resources=job_pb2.ResourceSpecProto(),
        band=band,
    )


def _routing(required: dict[str, int], routed_bands: dict[str, int]) -> RoutingDecision:
    return RoutingDecision(
        group_to_launch={},
        group_required_slices=required,
        routed_entries={name: [_demand_entry(band)] for name, band in routed_bands.items()},
        unmet_entries=[],
        group_reasons={},
        group_statuses=[],
    )


# -- Band-ordered, head-of-line chip admission under one pool's budget --


def test_highest_band_claims_chips_first():
    # PROD v4-16 (8 chips) and BATCH v4-8 (4 chips) each want 2 slices; only 16
    # chips free. PROD takes all 16; BATCH gets nothing.
    admitted = _admit_in_band_order(
        [_PoolCandidate("v4-8", BATCH, 4, 2), _PoolCandidate("v4-16", PRODUCTION, 8, 2)], free_chips=16
    )
    assert admitted == {"v4-16": 2, "v4-8": 0}


def test_head_of_line_holds_chips_for_unsatisfiable_high_band():
    # Only 4 chips free — too few for the PROD v4-16 slice (8). The 4 chips are
    # held for it, NOT handed to the BATCH v4-8 that would fit. This is the
    # re-grab the cap prevents while a preemptor accumulates chips over ticks.
    admitted = _admit_in_band_order(
        [_PoolCandidate("v4-16", PRODUCTION, 8, 1), _PoolCandidate("v4-8", BATCH, 4, 1)], free_chips=4
    )
    assert admitted == {"v4-16": 0, "v4-8": 0}


def test_same_band_groups_share_remaining():
    # Equal priority: no head-of-line between them; admitted greedily in name
    # order until chips run out (a takes 2 of the 3 affordable, b takes 1).
    admitted = _admit_in_band_order(
        [_PoolCandidate("a", PRODUCTION, 4, 2), _PoolCandidate("b", PRODUCTION, 4, 2)], free_chips=12
    )
    assert admitted == {"a": 2, "b": 1}


def test_unranked_band_yields_to_ranked():
    admitted = _admit_in_band_order(
        [_PoolCandidate("v4-8", UNRANKED_DEMAND_BAND, 4, 1), _PoolCandidate("v4-16", PRODUCTION, 8, 1)],
        free_chips=8,
    )
    assert admitted == {"v4-16": 1, "v4-8": 0}


def test_demand_trimmed_to_budget():
    assert _admit_in_band_order([_PoolCandidate("v4-8", BATCH, 4, 10)], free_chips=16) == {"v4-8": 4}


def test_over_committed_pool_admits_nothing():
    # Negative free chips (pool already over budget) launches nothing more.
    assert _admit_in_band_order([_PoolCandidate("v4-8", BATCH, 4, 2)], free_chips=-4) == {"v4-8": 0}


# -- build_scale_plan caps a fungible pool's launches to its reservation budget --


def test_high_band_slice_wins_reservation_over_low_band():
    # Empty 16-chip pool. PROD v4-16 wants 2 slices (16 chips), BATCH v4-8 wants
    # 2 (8 chips): 24 requested, 16 available. PROD claims the reservation.
    v16 = _ready_group("v4-16", "v4-16", quota_pool="pool-a", reservation_chips=16, slice_ids=[])
    v8 = _ready_group("v4-8", "v4-8", quota_pool="pool-a", reservation_chips=16, slice_ids=[])
    groups = {g.name: g for g in (v16, v8)}

    plan = build_scale_plan(
        groups, _routing({"v4-16": 2, "v4-8": 2}, {"v4-16": PRODUCTION, "v4-8": BATCH}), Timestamp.now()
    )

    assert plan.group_plans["v4-16"].slices_to_add == 2
    assert plan.group_plans["v4-8"].slices_to_add == 0


def test_drained_victim_cannot_regrab_chips_held_for_preemptor():
    # 16-chip pool, 12 consumed by three live v4-8 slices -> 4 free. A PROD v4-16
    # (needs 8) is waiting; the just-drained victim's re-queued BATCH v4-8 (needs
    # 4) would fit the 4 free chips. Head-of-line holds them for the preemptor:
    # neither launches this tick, so the chips accumulate rather than re-grab.
    v16 = _ready_group("v4-16", "v4-16", quota_pool="pool-a", reservation_chips=16, slice_ids=[])
    v8 = _ready_group("v4-8", "v4-8", quota_pool="pool-a", reservation_chips=16, slice_ids=["a1", "a2", "a3"])
    groups = {g.name: g for g in (v16, v8)}

    plan = build_scale_plan(
        groups, _routing({"v4-16": 1, "v4-8": 1}, {"v4-16": PRODUCTION, "v4-8": BATCH}), Timestamp.now()
    )

    assert plan.group_plans["v4-16"].slices_to_add == 0
    assert plan.group_plans["v4-8"].slices_to_add == 0


def test_launch_cap_reads_free_not_incoming_so_draining_chips_are_unavailable():
    # A draining slice's chips are NOT free for a new launch until reaped. Pool
    # of 16 chips: a1,a2 live (8) + a3 draining (4) -> 4 free, 8 incoming. The
    # cap must read free (4) and admit exactly one v4-8 (4 chips); reading
    # incoming (8) would wrongly admit two.
    v8 = _ready_group("v4-8", "v4-8", quota_pool="pool-a", reservation_chips=16, slice_ids=["a1", "a2", "a3"])
    v8.drain_slice("a3")
    groups = {v8.name: v8}

    plan = build_scale_plan(groups, _routing({"v4-8": 2}, {"v4-8": BATCH}), Timestamp.now())

    assert plan.group_plans["v4-8"].slices_to_add == 1


def test_non_fungible_group_untouched():
    # A plain group (reservation_chips=0) is not part of any pool, so the cap
    # leaves its planned launches exactly as build_group_scale_plan computed them.
    ts = Timestamp.now()
    plain_config = make_scale_group_config(name="plain", accelerator_variant="v4-8", max_slices=8)
    plain = ScalingGroup(plain_config, make_mock_platform())
    expected = build_group_scale_plan(plain, 3, ts).slices_to_add

    plan = build_scale_plan({"plain": plain}, _routing({"plain": 3}, {"plain": BATCH}), ts)

    assert plan.group_plans["plain"].slices_to_add == expected
    assert expected > 0
