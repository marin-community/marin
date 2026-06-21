# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for cross-variant preemption on a fungible reservation pool."""

from iris.cluster.constraints import PlacementRequirements
from iris.cluster.controller.autoscaler.models import DemandEntry
from iris.cluster.controller.autoscaler.reserved_pool import ReservedPoolView
from iris.cluster.controller.backend import _order_reserved_demand_by_band
from iris.cluster.controller.scheduling.policy import (
    PreemptionCandidate,
    run_reserved_pool_preemption,
)
from iris.cluster.controller.scheduling.scheduler import JobRequirements, RunningTaskInfo
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2

POOL = "v4-res/zone"
# Lower band_sort_key = higher priority.
PRODUCTION = job_pb2.PRIORITY_BAND_PRODUCTION
INTERACTIVE = job_pb2.PRIORITY_BAND_INTERACTIVE
BATCH = job_pb2.PRIORITY_BAND_BATCH

# Chip footprint per variant for these tests.
CHIPS_PER_VARIANT = {"v4-8": 4, "v4-16": 8, "v4-32": 16}


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


def _view(free_chips: int, worker_pool: dict[str, str], *, cooldown: frozenset[str] = frozenset()) -> ReservedPoolView:
    return ReservedPoolView(
        free_chips={POOL: free_chips},
        worker_pool=worker_pool,
        variant_pool={v: POOL for v in CHIPS_PER_VARIANT},
        chips_per_variant=dict(CHIPS_PER_VARIANT),
        pools_on_cooldown=cooldown,
    )


def test_production_v4_8_preempts_interactive_v4_16_when_pool_full():
    # The issue scenario: a full pool, a production v4-8 (needs 4 chips), and an
    # interactive v4-16 slice (8 chips) on two workers. The big slice is evicted.
    preemptor = _candidate("/alice/prod/0", "v4-8", PRODUCTION)
    victims = [
        _running("/bob/inter/0", "w0", "v4-16", INTERACTIVE, coscheduled=True),
        _running("/bob/inter/1", "w1", "v4-16", INTERACTIVE, coscheduled=True),
    ]
    view = _view(free_chips=0, worker_pool={"w0": POOL, "w1": POOL})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, view)

    assert {v.to_wire() for _, v in pairs} == {"/bob/inter/0", "/bob/inter/1"}
    assert all(p.to_wire() == "/alice/prod/0" for p, _ in pairs)
    assert drain == {WorkerId("w0"), WorkerId("w1")}


def test_same_band_victim_not_evicted():
    preemptor = _candidate("/alice/prod/0", "v4-8", INTERACTIVE)
    victims = [_running("/bob/other/0", "w0", "v4-16", INTERACTIVE, coscheduled=True)]
    view = _view(free_chips=0, worker_pool={"w0": POOL})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, view)

    assert pairs == []
    assert drain == set()


def test_higher_priority_victim_not_evicted():
    # Preemptor is interactive; the only victim is production (higher priority).
    preemptor = _candidate("/alice/inter/0", "v4-8", INTERACTIVE)
    victims = [_running("/bob/prod/0", "w0", "v4-8", PRODUCTION)]
    view = _view(free_chips=0, worker_pool={"w0": POOL})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, view)

    assert pairs == []
    assert drain == set()


def test_no_preemption_when_pool_has_enough_free_chips():
    preemptor = _candidate("/alice/prod/0", "v4-8", PRODUCTION)
    victims = [_running("/bob/inter/0", "w0", "v4-8", INTERACTIVE)]
    # 4 free chips already satisfy the v4-8 (4-chip) preemptor.
    view = _view(free_chips=4, worker_pool={"w0": POOL})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, view)

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
    view = _view(free_chips=0, worker_pool={"wa": POOL, "wb": POOL, "wc": POOL, "wd": POOL})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, view)

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
    view = _view(free_chips=0, worker_pool={"wsmall": POOL, "wbig0": POOL, "wbig1": POOL})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, view)

    # Smallest-first: the v4-8 (4 chips) covers the deficit alone; the v4-16 is spared.
    assert drain == {WorkerId("wsmall")}
    assert {v.to_wire() for _, v in pairs} == {"/bob/batch-small/0"}


def test_nothing_evicted_when_total_evictable_below_deficit():
    # Need 16 chips, only one evictable v4-8 (4 chips). Never partial-evict.
    preemptor = _candidate("/alice/prod/0", "v4-32", PRODUCTION)
    victims = [_running("/bob/batch/0", "wa", "v4-8", BATCH)]
    view = _view(free_chips=0, worker_pool={"wa": POOL})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, view)

    assert pairs == []
    assert drain == set()


def test_cooldown_suppresses_preemption_for_that_pool():
    preemptor = _candidate("/alice/prod/0", "v4-8", PRODUCTION)
    victims = [_running("/bob/inter/0", "w0", "v4-8", INTERACTIVE)]
    view = _view(free_chips=0, worker_pool={"w0": POOL}, cooldown=frozenset({POOL}))

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, view)

    assert pairs == []
    assert drain == set()


def test_batch_preemptor_never_preempts():
    preemptor = _candidate("/alice/batch/0", "v4-8", BATCH)
    # A lower-priority victim does not exist below batch, but even a hypothetical
    # one must not be evicted because batch never preempts.
    victims = [_running("/bob/other/0", "w0", "v4-8", BATCH)]
    view = _view(free_chips=0, worker_pool={"w0": POOL})

    pairs, drain = run_reserved_pool_preemption([preemptor], victims, view)

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
    view = _view(free_chips=0, worker_pool={"w0": POOL, "w1": POOL})

    pairs, drain = run_reserved_pool_preemption(preemptors, victims, view)

    # The single v4-16 slice is evicted once (freeing 8 chips); after the first
    # preemptor takes 4, 4 remain — enough for the second, so it preempts nothing.
    preemptors_in_pairs = {p.to_wire() for p, _ in pairs}
    assert preemptors_in_pairs == {"/alice/prod-a/0"}
    assert drain == {WorkerId("w0"), WorkerId("w1")}


def _demand(task_wire: str, variant: str | None) -> DemandEntry:
    variants = frozenset({variant}) if variant is not None else None
    return DemandEntry(
        task_ids=(task_wire,),
        coschedule_group_id=None,
        normalized=PlacementRequirements(
            device_type=None,
            device_variants=variants,
            preemptible=None,
            required_regions=None,
            required_zones=None,
        ),
        constraints=[],
        resources=job_pb2.ResourceSpecProto(),
    )


def _band_map(**by_wire: int) -> dict[JobName, int]:
    return {JobName.from_wire(wire): band for wire, band in by_wire.items()}


class TestOrderReservedDemandByBand:
    """The per-tick demand reorder that keeps a preemptor ahead of its victim."""

    def test_reserved_entries_reorder_by_band_within_their_slots(self):
        # Submission order puts the batch victim's demand before the production
        # preemptor's; a non-reserved (cpu) entry sits between them. After
        # ordering, the two reserved slots hold prod-then-batch; the cpu entry,
        # in a slot of its own, never moves.
        batch = _demand("/u/batch/0", "v4-8")
        cpu = _demand("/u/cpu/0", None)
        prod = _demand("/u/prod/0", "v4-16")
        view = _view(free_chips=0, worker_pool={})
        bands = _band_map(**{"/u/batch/0": BATCH, "/u/cpu/0": INTERACTIVE, "/u/prod/0": PRODUCTION})

        ordered = _order_reserved_demand_by_band([batch, cpu, prod], view, bands)

        assert ordered == [prod, cpu, batch]

    def test_non_reserved_demand_left_untouched(self):
        # Only one reserved entry -> nothing to reorder; list returned as-is.
        cpu = _demand("/u/cpu/0", None)
        prod = _demand("/u/prod/0", "v4-8")
        view = _view(free_chips=0, worker_pool={})
        bands = _band_map(**{"/u/cpu/0": INTERACTIVE, "/u/prod/0": PRODUCTION})

        ordered = _order_reserved_demand_by_band([cpu, prod], view, bands)

        assert ordered == [cpu, prod]

    def test_entry_absent_from_band_map_sorts_last(self):
        # A reserved entry whose task carries no resolved band trails ranked
        # reserved demand for the same pool.
        ranked = _demand("/u/prod/0", "v4-8")
        unranked = _demand("/u/mystery/0", "v4-16")
        view = _view(free_chips=0, worker_pool={})
        bands = _band_map(**{"/u/prod/0": PRODUCTION})

        ordered = _order_reserved_demand_by_band([unranked, ranked], view, bands)

        assert ordered == [ranked, unranked]


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
    view = _view(free_chips=0, worker_pool={"wa": POOL, "wb": POOL, "wc": POOL, "wd": POOL})

    pairs, drain = run_reserved_pool_preemption(preemptors, victims, view)

    # Exactly one victim slice (8 chips) is evicted to satisfy the 8-chip job.
    assert drain == {WorkerId("wa"), WorkerId("wb")}
    assert {v.to_wire() for _, v in pairs} == {"/bob/batch-a/0", "/bob/batch-a/1"}
