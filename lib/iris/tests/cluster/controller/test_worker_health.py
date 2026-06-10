# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for WorkerHealthTracker.

Exercises the two independent termination paths:
- Consecutive ping failures (reset by healthy ping)
- Build failures (monotonic counter, independent of pings)
"""

from pathlib import Path

import pytest
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.reads import healthy_active_workers_with_attributes
from iris.cluster.controller.schema import workers_table
from iris.cluster.controller.worker_health import (
    WorkerHealthEvent,
    WorkerHealthEventKind,
    WorkerHealthTracker,
)
from iris.cluster.types import WorkerId
from rigging.timing import Timestamp
from sqlalchemy import insert, select


@pytest.fixture
def tracker() -> WorkerHealthTracker:
    return WorkerHealthTracker(reconcile_failure_threshold=10, build_threshold=10)


def _unreachable(tracker: WorkerHealthTracker, wid: WorkerId, *, now_ms: int = 0) -> list[WorkerId]:
    return tracker.apply([WorkerHealthEvent(wid, WorkerHealthEventKind.UNREACHABLE)], now_ms=now_ms)


def _reached(tracker: WorkerHealthTracker, wid: WorkerId, *, now_ms: int = 0) -> list[WorkerId]:
    return tracker.apply([WorkerHealthEvent(wid, WorkerHealthEventKind.REACHED)], now_ms=now_ms)


def _build_failed(tracker: WorkerHealthTracker, wid: WorkerId, *, now_ms: int = 0) -> list[WorkerId]:
    return tracker.apply([WorkerHealthEvent(wid, WorkerHealthEventKind.BUILD_FAILED)], now_ms=now_ms)


def test_ping_failure_threshold_boundary(tracker: WorkerHealthTracker) -> None:
    """9 consecutive failures are not enough; 10th trips; a healthy ping resets and requires 10 more."""
    wid = WorkerId("w-1")
    over: list[WorkerId] = []
    for _ in range(9):
        over = _unreachable(tracker, wid)
    assert over == []
    assert _unreachable(tracker, wid) == [wid]

    tracker.forget(wid)
    for _ in range(9):
        _unreachable(tracker, wid)
    _reached(tracker, wid)  # reset consecutive failures
    for _ in range(9):
        over = _unreachable(tracker, wid)
    assert over == []
    assert _unreachable(tracker, wid) == [wid]


def test_build_failure_threshold_boundary(tracker: WorkerHealthTracker) -> None:
    """9 build failures are not enough; 10th trips; a healthy ping does not reset the counter."""
    wid = WorkerId("w-1")
    over: list[WorkerId] = []
    for _ in range(9):
        over = _build_failed(tracker, wid)
    assert over == []
    assert _build_failed(tracker, wid) == [wid]

    # A REACHED event resets consecutive ping failures but NOT build failures.
    assert _reached(tracker, wid) == [wid]


def test_ping_and_build_failures_are_independent(tracker: WorkerHealthTracker) -> None:
    """A worker can trip via either path; tripping one does not affect the other counter."""
    wid = WorkerId("w-1")
    for _ in range(5):
        _unreachable(tracker, wid)
    for _ in range(5):
        _build_failed(tracker, wid)
    assert tracker.apply([], now_ms=0) == []
    # 6 ping failures, still < 10; build still at 5.
    assert _unreachable(tracker, wid) == []


def test_forget_removes_worker(tracker: WorkerHealthTracker) -> None:
    wid = WorkerId("w-1")
    over: list[WorkerId] = []
    for _ in range(10):
        over = _unreachable(tracker, wid)
    assert over == [wid]
    tracker.forget(wid)
    assert tracker.apply([], now_ms=0) == []


def test_forget_many_drops_only_listed_workers(tracker: WorkerHealthTracker) -> None:
    a, b, c = WorkerId("a"), WorkerId("b"), WorkerId("c")
    for wid in (a, b, c):
        for _ in range(10):
            _unreachable(tracker, wid)
    tracker.forget_many([a, c])
    assert tracker.apply([], now_ms=0) == [b]


def test_per_worker_counters_are_independent(tracker: WorkerHealthTracker) -> None:
    a, b = WorkerId("a"), WorkerId("b")
    over: list[WorkerId] = []
    for _ in range(10):
        over = _unreachable(tracker, a)
    assert over == [a]
    assert b not in over


def test_snapshot_reports_both_counters(tracker: WorkerHealthTracker) -> None:
    wid = WorkerId("w-1")
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
