# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for cross-variant preemption on a full fungible reservation pool."""

from iris.cluster.controller.autoscaler.reserved_pool import PoolLedger, ReservationLedger
from iris.cluster.controller.scheduling.policy import PreemptionCandidate, run_reserved_pool_preemption
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


def test_coscheduled_preemptor_does_not_double_spend_free_chips():
    # Pool already has exactly enough free chips (8) for the ONE v4-16 slice the
    # coscheduled job needs. A separate evictable v4-16 BATCH victim slice also
    # exists. Each host candidate must not independently re-charge the pool for
    # the job's single slice need — that would deplete free chips once per host
    # and evict the victim slice even though nothing needed to be freed.
    preemptors = [
        _candidate("/alice/prod/0", "v4-16", PRODUCTION, coscheduled=True),
        _candidate("/alice/prod/1", "v4-16", PRODUCTION, coscheduled=True),
    ]
    victims = [
        _running("/bob/batch/0", "wa", "v4-16", BATCH, coscheduled=True),
        _running("/bob/batch/1", "wb", "v4-16", BATCH, coscheduled=True),
    ]
    ledger = _ledger(free_chips=8, worker_slice={"wa": "s-a", "wb": "s-a"})

    pairs, drain = run_reserved_pool_preemption(preemptors, victims, ledger)

    assert pairs == []
    assert drain == set()


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
