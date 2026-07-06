# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for WorkerHealthTracker.

Exercises the two independent termination paths:
- Time-based unreachability (continuously unreachable past the grace, guarded by
  the failure floor; reset by a healthy reconcile)
- Build failures (monotonic counter, independent of reconciles)
"""

from pathlib import Path

import pytest
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.reads import healthy_active_workers_with_attributes, list_active_healthy_workers
from iris.cluster.controller.schema import workers_table
from iris.cluster.controller.worker_health import (
    MIN_UNREACHABLE_FAILURES,
    WorkerHealthEvent,
    WorkerHealthEventKind,
    WorkerHealthTracker,
    WorkerLiveness,
)
from iris.cluster.types import WorkerId, WorkerUsability
from rigging.timing import Duration, Timestamp
from sqlalchemy import insert, select
from tests.cluster.controller._test_support import set_worker_consecutive_failures_for_test

from .conftest import make_worker_metadata, register_worker

# Grace used by the fixture tracker. Workers register at now_ms=0, so a failure
# at now_ms >= GRACE_MS (past the floor) is over the unreachable threshold.
GRACE_MS = 10_000


@pytest.fixture
def tracker() -> WorkerHealthTracker:
    return WorkerHealthTracker(unreachable_grace=Duration.from_ms(GRACE_MS), build_threshold=10)


def _unreachable(tracker: WorkerHealthTracker, wid: WorkerId, *, now_ms: int = 0) -> list[WorkerId]:
    return tracker.apply([WorkerHealthEvent(wid, WorkerHealthEventKind.UNREACHABLE)], now_ms=now_ms)


def _reached(tracker: WorkerHealthTracker, wid: WorkerId, *, now_ms: int = 0) -> list[WorkerId]:
    return tracker.apply([WorkerHealthEvent(wid, WorkerHealthEventKind.REACHED)], now_ms=now_ms)


def _build_failed(tracker: WorkerHealthTracker, wid: WorkerId, *, now_ms: int = 0) -> list[WorkerId]:
    return tracker.apply([WorkerHealthEvent(wid, WorkerHealthEventKind.BUILD_FAILED)], now_ms=now_ms)


def test_unreachable_grace_boundary(tracker: WorkerHealthTracker) -> None:
    """Failures within the grace window don't trip; once the worker has been
    unreachable for the full grace (past the floor) it trips; a REACHED resets
    the heartbeat clock."""
    wid = WorkerId("w-1")
    tracker.register(wid, now_ms=0)

    # Floor met, but the grace has not elapsed since the t=0 heartbeat.
    assert _unreachable(tracker, wid, now_ms=3_000) == []
    assert _unreachable(tracker, wid, now_ms=6_000) == []
    assert _unreachable(tracker, wid, now_ms=GRACE_MS - 1) == []
    # Grace elapsed since the last heartbeat, floor met -> trips.
    assert _unreachable(tracker, wid, now_ms=GRACE_MS) == [wid]

    # A REACHED resets both the failure count and the heartbeat clock; the grace
    # is measured anew from the REACHED, even though it is past t=0 + grace.
    tracker.forget(wid)
    tracker.register(wid, now_ms=0)
    _unreachable(tracker, wid, now_ms=8_000)
    _reached(tracker, wid, now_ms=8_500)
    assert _unreachable(tracker, wid, now_ms=12_000) == []  # f=1, age 3_500
    assert _unreachable(tracker, wid, now_ms=15_000) == []  # f=2, age 6_500
    assert _unreachable(tracker, wid, now_ms=8_500 + GRACE_MS) == [wid]  # f=3, age = grace


def test_min_failures_floor_guards_single_long_tick(tracker: WorkerHealthTracker) -> None:
    """A single failed reconcile long past the grace must NOT trip: the floor
    requires several real failures, so one anomalously long tick (or a controller
    stall that ages every heartbeat at once) cannot reap a worker."""
    wid = WorkerId("w-1")
    tracker.register(wid, now_ms=0)
    # Each failure is well past the grace, but the floor isn't met until the
    # MIN_UNREACHABLE_FAILURES-th one.
    for i in range(MIN_UNREACHABLE_FAILURES - 1):
        assert _unreachable(tracker, wid, now_ms=60_000 + i) == []
    assert _unreachable(tracker, wid, now_ms=60_000 + MIN_UNREACHABLE_FAILURES) == [wid]


def test_build_failure_threshold_boundary(tracker: WorkerHealthTracker) -> None:
    """9 build failures are not enough; 10th trips; a REACHED event does not reset the counter."""
    wid = WorkerId("w-1")
    tracker.register(wid, now_ms=0)
    over: list[WorkerId] = []
    for _ in range(9):
        over = _build_failed(tracker, wid)
    assert over == []
    assert _build_failed(tracker, wid) == [wid]

    # A REACHED event resets consecutive reconcile failures but NOT build failures.
    assert _reached(tracker, wid) == [wid]


def test_reconcile_and_build_failures_are_independent(tracker: WorkerHealthTracker) -> None:
    """A worker can trip via either path; tripping one does not affect the other counter."""
    wid = WorkerId("w-1")
    tracker.register(wid, now_ms=0)
    for _ in range(5):
        _unreachable(tracker, wid, now_ms=1_000)
    for _ in range(5):
        _build_failed(tracker, wid, now_ms=1_000)
    # Reconcile failures all within the grace window; build still at 5 (< 10).
    assert tracker.apply([], now_ms=1_000) == []
    assert _unreachable(tracker, wid, now_ms=2_000) == []


def test_forget_removes_worker(tracker: WorkerHealthTracker) -> None:
    wid = WorkerId("w-1")
    tracker.register(wid, now_ms=0)
    over: list[WorkerId] = []
    for i in range(MIN_UNREACHABLE_FAILURES):
        over = _unreachable(tracker, wid, now_ms=GRACE_MS + i)
    assert over == [wid]
    tracker.forget(wid)
    assert tracker.apply([], now_ms=GRACE_MS * 2) == []


def test_forget_many_drops_only_listed_workers(tracker: WorkerHealthTracker) -> None:
    a, b, c = WorkerId("a"), WorkerId("b"), WorkerId("c")
    for wid in (a, b, c):
        tracker.register(wid, now_ms=0)
        for i in range(MIN_UNREACHABLE_FAILURES):
            _unreachable(tracker, wid, now_ms=GRACE_MS + i)
    tracker.forget_many([a, c])
    assert tracker.apply([], now_ms=GRACE_MS * 2) == [b]


def test_per_worker_counters_are_independent(tracker: WorkerHealthTracker) -> None:
    a, b = WorkerId("a"), WorkerId("b")
    tracker.register(a, now_ms=0)
    tracker.register(b, now_ms=0)
    over: list[WorkerId] = []
    for i in range(MIN_UNREACHABLE_FAILURES):
        over = _unreachable(tracker, a, now_ms=GRACE_MS + i)
    assert over == [a]
    assert b not in over


def test_snapshot_reports_both_counters(tracker: WorkerHealthTracker) -> None:
    wid = WorkerId("w-1")
    tracker.register(wid, now_ms=0)
    for _ in range(3):
        _unreachable(tracker, wid)
    for _ in range(2):
        _build_failed(tracker, wid)
    assert tracker.snapshot() == {wid: (3, 2)}


def test_seeds_liveness_from_persisted_workers(tmp_path: Path) -> None:
    """Seeding liveness from persisted workers marks every DB worker healthy.

    Without this seed (regression target), a controller restart hides every
    pre-existing worker from ``healthy_active_workers_with_attributes`` until
    the next ping cycle — the scheduler then makes no assignments.

    The seeding logic is now a free function on Controller; this test
    exercises it directly via WorkerHealthTracker.heartbeat.
    """

    db = ControllerDB(db_dir=tmp_path)
    try:
        with db.transaction() as cur:
            cur.execute(insert(workers_table).values(worker_id="w-seed-1", address="10.0.0.1:8080"))
            cur.execute(insert(workers_table).values(worker_id="w-seed-2", address="10.0.0.2:8080"))

        health = WorkerHealthTracker()
        worker_attrs = WorkerAttrsProjection(db)

        # Replicate _seed_liveness_from_workers
        now_ms = Timestamp.now().epoch_ms()
        with db.read_snapshot() as tx:
            rows = tx.execute(select(workers_table.c.worker_id)).all()
        worker_ids = [row.worker_id for row in rows]
        health.heartbeat(worker_ids, now_ms)

        liveness_one = health.liveness(WorkerId("w-seed-1"))
        liveness_two = health.liveness(WorkerId("w-seed-2"))
        assert liveness_one.healthy and liveness_one.active
        assert liveness_two.healthy and liveness_two.active
        assert liveness_one.last_heartbeat_ms > 0
        assert liveness_two.last_heartbeat_ms > 0

        with db.read_snapshot() as tx:
            schedulable = healthy_active_workers_with_attributes(tx, health, worker_attrs)
        ids = {str(w.worker_id) for w in schedulable}
        assert ids == {"w-seed-1", "w-seed-2"}
    finally:
        db.close()


def test_failing_worker_excluded_from_scheduling_but_still_reconciled(state):
    """A worker accruing reconcile failures stops getting new placements but is
    still reconciled, so it can recover or cross the teardown threshold.

    Pins the two-filter split: scheduling placement
    (``healthy_active_workers_with_attributes``) drops a worker with
    ``consecutive_failures > 0``, while the reconcile target set
    (``list_active_healthy_workers``) keeps probing every active worker.
    """
    ok = register_worker(state, "w-ok", "w-ok:8080", make_worker_metadata())
    failing = register_worker(state, "w-failing", "w-failing:8080", make_worker_metadata())

    # Mid-failure: unreachable for one reconcile pass but not yet over threshold.
    set_worker_consecutive_failures_for_test(state, failing, 1)

    with state._db.read_snapshot() as tx:
        schedulable = {
            w.worker_id for w in healthy_active_workers_with_attributes(tx, state._health, state._worker_attrs)
        }
        reconcile_targets = set(list_active_healthy_workers(tx, state._health))

    assert ok in schedulable
    assert failing not in schedulable, "a failing worker must not receive new placements"
    assert {ok, failing} <= reconcile_targets, "a failing worker must still be reconciled/probed"


@pytest.mark.parametrize(
    "liveness, expected",
    [
        # Reached and clean -> schedulable.
        (WorkerLiveness(healthy=True, active=True, consecutive_failures=0), WorkerUsability.HEALTHY),
        # Mid-failure (below threshold) -> reconciled but not placeable.
        (WorkerLiveness(healthy=True, active=True, consecutive_failures=1), WorkerUsability.DEGRADED),
        # At/over the teardown threshold is still DEGRADED, NOT a distinct DEAD: the
        # reconcile pass must keep probing it until apply() reaps it. The threshold
        # lives in apply(), not in the classifier.
        (WorkerLiveness(healthy=True, active=True, consecutive_failures=99), WorkerUsability.DEGRADED),
        # build_failures do NOT affect usability: they drive teardown via apply(),
        # not placement/reconcile membership. A build-failing but reachable worker
        # stays schedulable (preserving pre-refactor behavior).
        (WorkerLiveness(healthy=True, active=True, consecutive_failures=0, build_failures=5), WorkerUsability.HEALTHY),
        # Not reached / inactive -> excluded everywhere.
        (WorkerLiveness(healthy=False, active=True, consecutive_failures=0), WorkerUsability.DEAD),
        (WorkerLiveness(healthy=True, active=False, consecutive_failures=0), WorkerUsability.DEAD),
        (WorkerLiveness(), WorkerUsability.DEAD),
    ],
)
def test_worker_liveness_usability_classification(liveness: WorkerLiveness, expected: WorkerUsability) -> None:
    """The single classifier maps liveness to the verdict every predicate projects from."""
    assert liveness.usability is expected
