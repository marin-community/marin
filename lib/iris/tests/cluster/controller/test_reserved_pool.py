# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fungible reservation pool: chip ledger, the launch cap it feeds, and cross-variant preemption."""

import pytest
from iris.cluster.constraints import PlacementRequirements
from iris.cluster.controller.autoscaler.models import UNRANKED_DEMAND_BAND, DemandEntry, RoutingDecision
from iris.cluster.controller.autoscaler.planning import (
    _admit_in_band_order,
    _PoolCandidate,
    build_group_scale_plan,
    build_scale_plan,
)
from iris.cluster.controller.autoscaler.reserved_pool import PoolLedger, ReservationLedger, build_reservation_ledger
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.cluster.controller.backend import _stamp_demand_bands
from iris.cluster.controller.scheduling.policy import PreemptionCandidate, run_reserved_pool_preemption
from iris.cluster.controller.scheduling.scheduler import JobRequirements, RunningTaskInfo
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2, vm_pb2
from rigging.timing import Timestamp

from tests.cluster.backends.conftest import make_fake_slice_handle, make_mock_platform

from .conftest import make_scale_group_config

POOL = "v4-res/zone"
# Lower band_sort_key = higher priority.
PRODUCTION = job_pb2.PRIORITY_BAND_PRODUCTION
INTERACTIVE = job_pb2.PRIORITY_BAND_INTERACTIVE
BATCH = job_pb2.PRIORITY_BAND_BATCH

# Chip footprint per variant for these tests.
CHIPS_PER_VARIANT = {"v4-8": 4, "v4-16": 8, "v4-32": 16}


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


# -- Cross-variant preemption on a full fungible pool --


def _requirements(variant: str, *, coscheduled: bool = False) -> JobRequirements:
    return JobRequirements(
        req_cpu_millicores=1000,
        req_memory_bytes=1024,
        req_gpu_count=0,
        req_tpu_count=CHIPS_PER_VARIANT[variant],
        device_variant=variant,
        constraints=[],
        is_coscheduled=coscheduled,
        coscheduling_group_by="tpu" if coscheduled else None,
    )


def _candidate(task_wire: str, variant: str, band: int, *, coscheduled: bool = False) -> PreemptionCandidate:
    return PreemptionCandidate(
        job_name=JobName.from_wire(task_wire),
        requirements=_requirements(variant, coscheduled=coscheduled),
        band=band,
    )


def _running(task_wire: str, worker: str, variant: str, band: int, *, coscheduled: bool = False) -> RunningTaskInfo:
    return RunningTaskInfo(
        task_id=JobName.from_wire(task_wire),
        worker_id=WorkerId(worker),
        band_sort_key=band,
        resource_value=CHIPS_PER_VARIANT[variant],
        is_coscheduled=coscheduled,
        cpu_millicores=1000,
        memory_bytes=1024,
        gpu_count=0,
        tpu_count=CHIPS_PER_VARIANT[variant],
        device_variant=variant,
    )


def _ledger(
    free_chips: int,
    worker_slice: dict[str, str],
    *,
    draining_chips: int = 0,
    inflight_slices: dict[str, int] | None = None,
) -> ReservationLedger:
    # worker_slice maps each victim worker to its physical slice id; coscheduled
    # siblings share a slice so the drain reaps them together. Every worker here
    # belongs to the single test pool. ``draining_chips`` are chips a drain is
    # already freeing (incoming = free + draining); ``inflight_slices`` marks a
    # replacement slice of a variant already booting.
    pool = PoolLedger(
        pool_id=POOL,
        reservation_chips=free_chips + draining_chips,
        live_chips=0,
        inflight_chips=0,
        draining_chips=draining_chips,
        inflight_slices_by_variant=dict(inflight_slices or {}),
    )
    return ReservationLedger(
        pools={POOL: pool},
        worker_pool={worker: POOL for worker in worker_slice},
        worker_slice=dict(worker_slice),
        variant_pool={v: POOL for v in CHIPS_PER_VARIANT},
        chips_per_variant=dict(CHIPS_PER_VARIANT),
    )


def test_production_v4_8_preempts_interactive_v4_16_when_pool_full():
    # The issue scenario: a full pool, a production v4-8 (needs 4 chips), and an
    # interactive v4-16 slice (8 chips) on two workers. The big slice is evicted.
    preemptor = _candidate("/alice/prod/0", "v4-8", PRODUCTION)
    victims = [
        _running("/bob/inter/0", "w0", "v4-16", INTERACTIVE, coscheduled=True),
        _running("/bob/inter/1", "w1", "v4-16", INTERACTIVE, coscheduled=True),
    ]
    ledger = _ledger(free_chips=0, worker_slice={"w0": "s-inter", "w1": "s-inter"})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, ledger)

    assert {v.to_wire() for _, v in pairs} == {"/bob/inter/0", "/bob/inter/1"}
    assert all(p.to_wire() == "/alice/prod/0" for p, _ in pairs)
    assert drain == {WorkerId("w0"), WorkerId("w1")}


def test_same_band_victim_not_evicted():
    preemptor = _candidate("/alice/prod/0", "v4-8", INTERACTIVE)
    victims = [_running("/bob/other/0", "w0", "v4-16", INTERACTIVE, coscheduled=True)]
    ledger = _ledger(free_chips=0, worker_slice={"w0": "s0"})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, ledger)

    assert pairs == []
    assert drain == set()


def test_higher_priority_victim_not_evicted():
    # Preemptor is interactive; the only victim is production (higher priority).
    preemptor = _candidate("/alice/inter/0", "v4-8", INTERACTIVE)
    victims = [_running("/bob/prod/0", "w0", "v4-8", PRODUCTION)]
    ledger = _ledger(free_chips=0, worker_slice={"w0": "s0"})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, ledger)

    assert pairs == []
    assert drain == set()


def test_no_preemption_when_pool_has_enough_free_chips():
    preemptor = _candidate("/alice/prod/0", "v4-8", PRODUCTION)
    victims = [_running("/bob/inter/0", "w0", "v4-8", INTERACTIVE)]
    # 4 free chips already satisfy the v4-8 (4-chip) preemptor.
    ledger = _ledger(free_chips=4, worker_slice={"w0": "s0"})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, ledger)

    assert pairs == []
    assert drain == set()


def test_minimal_eviction_picks_fewest_lowest_priority_slices():
    # Need 16 chips (v4-32). Candidates: two batch v4-8 (4 each), one interactive
    # v4-16 (8). The deficit is best covered by the single batch + interactive?
    # Sort is lowest-priority-then-smallest: batch slices first (band higher).
    # Two batch v4-8 = 8 chips, still short; the interactive v4-16 = 8 more -> 16.
    preemptor = _candidate("/alice/prod/0", "v4-32", PRODUCTION)
    victims = [
        _running("/bob/batch-a/0", "wa", "v4-8", BATCH),
        _running("/bob/batch-b/0", "wb", "v4-8", BATCH),
        _running("/bob/inter/0", "wc", "v4-16", INTERACTIVE, coscheduled=True),
        _running("/bob/inter/1", "wd", "v4-16", INTERACTIVE, coscheduled=True),
    ]
    ledger = _ledger(free_chips=0, worker_slice={"wa": "sa", "wb": "sb", "wc": "s-inter", "wd": "s-inter"})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, ledger)

    # 2 batch v4-8 (8) + 1 interactive v4-16 (8) = 16 chips, covering the deficit.
    assert drain == {WorkerId("wa"), WorkerId("wb"), WorkerId("wc"), WorkerId("wd")}
    assert {v.to_wire() for _, v in pairs} == {
        "/bob/batch-a/0",
        "/bob/batch-b/0",
        "/bob/inter/0",
        "/bob/inter/1",
    }


def test_no_extra_slices_evicted_beyond_deficit():
    # Need 4 chips (v4-8). The smallest evictable slice (a v4-8, 4 chips) covers
    # the deficit exactly; the larger v4-16 slice must NOT also be evicted.
    preemptor = _candidate("/alice/prod/0", "v4-8", PRODUCTION)
    victims = [
        _running("/bob/batch-small/0", "wsmall", "v4-8", BATCH),
        _running("/bob/batch-big/0", "wbig0", "v4-16", BATCH, coscheduled=True),
        _running("/bob/batch-big/1", "wbig1", "v4-16", BATCH, coscheduled=True),
    ]
    ledger = _ledger(free_chips=0, worker_slice={"wsmall": "s-small", "wbig0": "s-big", "wbig1": "s-big"})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, ledger)

    # Smallest-first: the v4-8 (4 chips) covers the deficit alone; the v4-16 is spared.
    assert drain == {WorkerId("wsmall")}
    assert {v.to_wire() for _, v in pairs} == {"/bob/batch-small/0"}


def test_nothing_evicted_when_total_evictable_below_deficit():
    # Need 16 chips, only one evictable v4-8 (4 chips). Never partial-evict.
    preemptor = _candidate("/alice/prod/0", "v4-32", PRODUCTION)
    victims = [_running("/bob/batch/0", "wa", "v4-8", BATCH)]
    ledger = _ledger(free_chips=0, worker_slice={"wa": "sa"})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, ledger)

    assert pairs == []
    assert drain == set()


def test_draining_chips_cover_deficit_so_no_re_preempt():
    # Window 1: the victim chosen on a prior tick is now DRAINING. Its chips show
    # up as draining (not free), so incoming = free + draining covers the v4-8
    # preemptor's 4-chip need. A second eviction would kill extra work for capacity
    # already in flight, so nothing is evicted.
    preemptor = _candidate("/alice/prod/0", "v4-8", PRODUCTION)
    # An evictable batch victim still exists, but the deficit is already covered.
    victims = [_running("/bob/batch/0", "w0", "v4-8", BATCH)]
    ledger = _ledger(free_chips=0, draining_chips=4, worker_slice={"w0": "s0"})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, ledger)

    assert pairs == []
    assert drain == set()


def test_replacement_slice_booting_skips_re_preempt():
    # Window 2: the prior victim was already reaped and a replacement slice of the
    # preemptor's own variant is booting (inflight_slices >= 1). Free+draining is 0,
    # but the booting replacement covers the need, so no further eviction.
    preemptor = _candidate("/alice/prod/0", "v4-8", PRODUCTION)
    victims = [_running("/bob/batch/0", "w0", "v4-8", BATCH)]
    ledger = _ledger(free_chips=0, worker_slice={"w0": "s0"}, inflight_slices={"v4-8": 1})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, ledger)

    assert pairs == []
    assert drain == set()


def test_replacement_slice_of_other_variant_does_not_skip():
    # An in-flight slice of a DIFFERENT variant does not cover a v4-8 preemptor:
    # only a replacement of the preemptor's own variant counts. With no free or
    # draining chips and no v4-8 replacement, the batch victim is evicted.
    preemptor = _candidate("/alice/prod/0", "v4-8", PRODUCTION)
    victims = [_running("/bob/batch/0", "w0", "v4-8", BATCH)]
    ledger = _ledger(free_chips=0, worker_slice={"w0": "s0"}, inflight_slices={"v4-16": 1})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, ledger)

    assert {v.to_wire() for _, v in pairs} == {"/bob/batch/0"}
    assert drain == {WorkerId("w0")}


def test_two_preemptors_one_replacement_slice_evicts_for_the_second():
    # A single booting replacement slice covers exactly one preemptor. Two v4-8
    # preemptors of distinct jobs: the first is satisfied by the booting
    # replacement; the second has nothing left and must evict a victim.
    preemptors = [
        _candidate("/alice/prod-a/0", "v4-8", PRODUCTION),
        _candidate("/alice/prod-b/0", "v4-8", PRODUCTION),
    ]
    victims = [_running("/bob/batch/0", "w0", "v4-8", BATCH)]
    ledger = _ledger(free_chips=0, worker_slice={"w0": "s0"}, inflight_slices={"v4-8": 1})

    pairs, drain = run_reserved_pool_preemption(preemptors, victims, ledger)

    # First preemptor consumes the replacement; second evicts the batch victim.
    assert {p.to_wire() for p, _ in pairs} == {"/alice/prod-b/0"}
    assert drain == {WorkerId("w0")}


def test_batch_preemptor_never_preempts():
    preemptor = _candidate("/alice/batch/0", "v4-8", BATCH)
    # A lower-priority victim does not exist below batch, but even a hypothetical
    # one must not be evicted because batch never preempts.
    victims = [_running("/bob/other/0", "w0", "v4-8", BATCH)]
    ledger = _ledger(free_chips=0, worker_slice={"w0": "s0"})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, ledger)

    assert pairs == []
    assert drain == set()


def test_two_preemptors_do_not_claim_the_same_victim():
    # Two production v4-8 preemptors, only one v4-16 victim slice (8 chips). The
    # first claims it; the second finds nothing left and evicts nothing more.
    preemptors = [
        _candidate("/alice/prod-a/0", "v4-8", PRODUCTION),
        _candidate("/alice/prod-b/0", "v4-8", PRODUCTION),
    ]
    victims = [
        _running("/bob/inter/0", "w0", "v4-16", INTERACTIVE, coscheduled=True),
        _running("/bob/inter/1", "w1", "v4-16", INTERACTIVE, coscheduled=True),
    ]
    ledger = _ledger(free_chips=0, worker_slice={"w0": "s-inter", "w1": "s-inter"})

    pairs, drain = run_reserved_pool_preemption(preemptors, victims, ledger)

    # The single v4-16 slice is evicted once (freeing 8 chips); after the first
    # preemptor takes 4, 4 remain — enough for the second, so it preempts nothing.
    preemptors_in_pairs = {p.to_wire() for p, _ in pairs}
    assert preemptors_in_pairs == {"/alice/prod-a/0"}
    assert drain == {WorkerId("w0"), WorkerId("w1")}


def test_coscheduled_preemptor_handled_once():
    # A coscheduled v4-16 preemptor has two task candidates but is one job needing
    # 8 chips total. It must evict exactly one victim slice, not two.
    preemptors = [
        _candidate("/alice/prod/0", "v4-16", PRODUCTION, coscheduled=True),
        _candidate("/alice/prod/1", "v4-16", PRODUCTION, coscheduled=True),
    ]
    victims = [
        _running("/bob/batch-a/0", "wa", "v4-16", BATCH, coscheduled=True),
        _running("/bob/batch-a/1", "wb", "v4-16", BATCH, coscheduled=True),
        _running("/bob/batch-c/0", "wc", "v4-16", BATCH, coscheduled=True),
        _running("/bob/batch-c/1", "wd", "v4-16", BATCH, coscheduled=True),
    ]
    ledger = _ledger(free_chips=0, worker_slice={"wa": "s-a", "wb": "s-a", "wc": "s-c", "wd": "s-c"})

    pairs, drain = run_reserved_pool_preemption(preemptors, victims, ledger)

    # Exactly one victim slice (8 chips) is evicted to satisfy the 8-chip job.
    assert drain == {WorkerId("wa"), WorkerId("wb")}
    assert {v.to_wire() for _, v in pairs} == {"/bob/batch-a/0", "/bob/batch-a/1"}


def test_slice_with_a_higher_band_task_is_not_evicted():
    # Two independent (non-coscheduled) tasks share one physical slice: a batch
    # task and an interactive task. The drain tears down the whole slice, so the
    # interactive task (equal priority to the preemptor) must protect it — the
    # pass groups by slice, gates on the highest-priority member, and evicts
    # nothing even though the batch task alone would be fair game.
    preemptor = _candidate("/alice/prod/0", "v4-8", PRODUCTION)
    victims = [
        _running("/bob/batch/0", "w0", "v4-8", BATCH),
        _running("/bob/inter/0", "w1", "v4-8", PRODUCTION),
    ]
    # Both workers belong to the same physical slice.
    ledger = _ledger(free_chips=0, worker_slice={"w0": "shared", "w1": "shared"})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, ledger)

    assert pairs == []
    assert drain == set()


def test_slice_chips_counted_once_for_colocated_tasks():
    # Two independent batch tasks on one physical slice free that slice's chips
    # once (not once per task). A v4-16 preemptor needs 8 chips; the shared v4-16
    # slice supplies exactly 8, and both colocated tasks are preempted together.
    preemptor = _candidate("/alice/prod/0", "v4-16", PRODUCTION, coscheduled=True)
    victims = [
        _running("/bob/batch-x/0", "w0", "v4-16", BATCH),
        _running("/bob/batch-y/0", "w1", "v4-16", BATCH),
    ]
    ledger = _ledger(free_chips=0, worker_slice={"w0": "shared", "w1": "shared"})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, ledger)

    assert drain == {WorkerId("w0"), WorkerId("w1")}
    assert {v.to_wire() for _, v in pairs} == {"/bob/batch-x/0", "/bob/batch-y/0"}


def _demand(*task_wires: str) -> DemandEntry:
    return DemandEntry(
        task_ids=task_wires,
        coschedule_group_id=None,
        normalized=PlacementRequirements(
            device_type=None, device_variants=None, preemptible=None, required_regions=None, required_zones=None
        ),
        constraints=[],
        resources=job_pb2.ResourceSpecProto(),
    )


def _band_map(**by_wire: int) -> dict[JobName, int]:
    return {JobName.from_wire(wire): band for wire, band in by_wire.items()}


# -- _stamp_demand_bands: the scheduler stamps each demand entry's effective band for the cap --


def test_stamp_demand_entry_takes_its_task_band():
    bands = _band_map(**{"/u/prod/0": PRODUCTION})

    [stamped] = _stamp_demand_bands([_demand("/u/prod/0")], bands)

    assert stamped.band == PRODUCTION


def test_stamp_demand_entry_without_resolved_band_is_unranked():
    [stamped] = _stamp_demand_bands([_demand("/u/mystery/0")], _band_map())

    assert stamped.band == UNRANKED_DEMAND_BAND


def test_stamp_demand_coscheduled_entry_takes_highest_priority_member_band():
    # A gang carries several tasks; the entry's band is the min (highest
    # priority) so the whole slice is admitted at its strongest member's band.
    bands = _band_map(**{"/u/g/0": BATCH, "/u/g/1": PRODUCTION})

    [stamped] = _stamp_demand_bands([_demand("/u/g/0", "/u/g/1")], bands)

    assert stamped.band == PRODUCTION
