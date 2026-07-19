# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the ``iris.task_state`` emitter: per-root-job counts,
wait-age anchors, the cluster rollup row, and federated-task exclusion."""

import pytest
from iris.cluster.controller import writes
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import task_attempts_table, tasks_table
from iris.cluster.controller.task_state_stats import CLUSTER_ROLLUP_ROOT_JOB, TaskStateCollector
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from iris.test_util import FakeStatsTable
from rigging.timing import Timestamp
from sqlalchemy import update

NOW_MS = 1_700_000_000_000
NOW = Timestamp.from_ms(NOW_MS)


@pytest.fixture
def db(tmp_path):
    db = ControllerDB(db_dir=tmp_path / "db")
    yield db
    db.close()


def seed_job(tx, job: JobName, *, submitted_at_ms: int) -> None:
    writes.insert_job(
        tx,
        job_id=job,
        user_id="u",
        submitting_user="u",
        parent_job_id=None,
        root_job_id=job.to_wire(),
        depth=0,
        state=job_pb2.JOB_STATE_RUNNING,
        submitted_at_ms=submitted_at_ms,
        root_submitted_at_ms=submitted_at_ms,
        started_at_ms=None,
        finished_at_ms=None,
        scheduling_deadline_epoch_ms=None,
        error=None,
        exit_code=None,
        num_tasks=1,
        name=job.to_wire(),
    )


def seed_task(tx, job: JobName, index: int, state: int, *, submitted_at_ms: int) -> JobName:
    task_id = job.task(index)
    writes.bulk_insert_tasks(
        tx,
        [
            writes.task_row(
                task_id=task_id,
                job_id=job,
                task_index=index,
                state=state,
                submitted_at_ms=submitted_at_ms,
                max_retries_failure=0,
                max_retries_preemption=0,
                priority_neg_depth=0,
                priority_root_submitted_ms=submitted_at_ms,
                priority_insertion=index,
                priority_band=2,
            )
        ],
    )
    return task_id


def seed_attempt(
    tx, task_id: JobName, *, created_at_ms: int, finished_at_ms: int | None = None, attempt_id: int = 0
) -> None:
    """Give a task a current attempt, optionally already finished (a requeue)."""
    writes.insert_attempt(
        tx,
        task_id=task_id,
        attempt_id=attempt_id,
        worker_id=None,
        state=job_pb2.TASK_STATE_ASSIGNED,
        created_at_ms=created_at_ms,
    )
    if finished_at_ms is not None:
        tx.execute(
            update(task_attempts_table)
            .where(task_attempts_table.c.task_id == task_id)
            .values(finished_at_ms=finished_at_ms)
        )
    tx.execute(update(tasks_table).where(tasks_table.c.task_id == task_id).values(current_attempt_id=attempt_id))


def collect(db: ControllerDB) -> list:
    table = FakeStatsTable()
    TaskStateCollector(db, table).collect_once(NOW)
    (rows,) = table.writes
    return rows


def rows_by_root(rows) -> dict:
    return {row.root_job_id: row for row in rows}


def test_counts_ages_and_rollup_per_root_job(db):
    job_a = JobName.from_wire("/u/job-a")
    job_b = JobName.from_wire("/u/job-b")
    with db.transaction() as tx:
        seed_job(tx, job_a, submitted_at_ms=NOW_MS - 500_000)
        seed_job(tx, job_b, submitted_at_ms=NOW_MS - 100_000)
        # job-a: two PENDING (oldest 500s), one BUILDING dispatched 200s ago, one SUCCEEDED.
        seed_task(tx, job_a, 0, job_pb2.TASK_STATE_PENDING, submitted_at_ms=NOW_MS - 500_000)
        seed_task(tx, job_a, 1, job_pb2.TASK_STATE_PENDING, submitted_at_ms=NOW_MS - 100_000)
        building = seed_task(tx, job_a, 2, job_pb2.TASK_STATE_BUILDING, submitted_at_ms=NOW_MS - 500_000)
        seed_attempt(tx, building, created_at_ms=NOW_MS - 200_000)
        seed_task(tx, job_a, 3, job_pb2.TASK_STATE_SUCCEEDED, submitted_at_ms=NOW_MS - 500_000)
        # job-b: one RUNNING.
        seed_task(tx, job_b, 0, job_pb2.TASK_STATE_RUNNING, submitted_at_ms=NOW_MS - 100_000)

    rows = rows_by_root(collect(db))
    assert set(rows) == {"/u/job-a", "/u/job-b", CLUSTER_ROLLUP_ROOT_JOB}

    row_a = rows["/u/job-a"]
    assert (row_a.pending, row_a.building, row_a.succeeded) == (2, 1, 1)
    assert row_a.oldest_pending_age_ms == 500_000
    assert row_a.oldest_building_age_ms == 200_000

    row_b = rows["/u/job-b"]
    assert (row_b.pending, row_b.running) == (0, 1)
    assert row_b.oldest_pending_age_ms == 0
    assert row_b.oldest_building_age_ms == 0

    rollup = rows[CLUSTER_ROLLUP_ROOT_JOB]
    assert (rollup.pending, rollup.building, rollup.running, rollup.succeeded) == (2, 1, 1, 1)
    assert rollup.oldest_pending_age_ms == 500_000
    assert rollup.oldest_building_age_ms == 200_000


def test_idle_cluster_emits_zero_rollup_row(db):
    rows = collect(db)
    assert len(rows) == 1
    rollup = rows[0]
    assert rollup.root_job_id == CLUSTER_ROLLUP_ROOT_JOB
    assert (rollup.pending, rollup.assigned, rollup.building, rollup.running) == (0, 0, 0, 0)
    assert rollup.oldest_pending_age_ms == 0
    assert rollup.oldest_building_age_ms == 0


def test_fully_finished_root_job_emits_no_row(db):
    job = JobName.from_wire("/u/done-job")
    with db.transaction() as tx:
        seed_job(tx, job, submitted_at_ms=NOW_MS - 500_000)
        seed_task(tx, job, 0, job_pb2.TASK_STATE_SUCCEEDED, submitted_at_ms=NOW_MS - 500_000)
        seed_task(tx, job, 1, job_pb2.TASK_STATE_FAILED, submitted_at_ms=NOW_MS - 500_000)

    rows = collect(db)
    assert [row.root_job_id for row in rows] == [CLUSTER_ROLLUP_ROOT_JOB]


def test_requeued_pending_task_ages_from_last_attempt_finish(db):
    """A PENDING task that already ran ages from its requeue, not its submission."""
    job = JobName.from_wire("/u/retry-job")
    with db.transaction() as tx:
        seed_job(tx, job, submitted_at_ms=NOW_MS - 900_000)
        task_id = seed_task(tx, job, 0, job_pb2.TASK_STATE_PENDING, submitted_at_ms=NOW_MS - 900_000)
        seed_attempt(tx, task_id, created_at_ms=NOW_MS - 800_000, finished_at_ms=NOW_MS - 60_000)

    rows = rows_by_root(collect(db))
    assert rows["/u/retry-job"].oldest_pending_age_ms == 60_000


def test_tasks_handed_off_to_peer_are_excluded(db):
    """Tasks mirrored for a federation peer are the peer's to report, not ours."""
    job = JobName.from_wire("/u/fed-job")
    with db.transaction() as tx:
        seed_job(tx, job, submitted_at_ms=NOW_MS - 500_000)
        seed_task(tx, job, 0, job_pb2.TASK_STATE_PENDING, submitted_at_ms=NOW_MS - 500_000)
        seed_task(tx, job, 1, job_pb2.TASK_STATE_PENDING, submitted_at_ms=NOW_MS - 400_000)
        tx.execute(update(tasks_table).where(tasks_table.c.task_id == job.task(1)).values(cluster="peer-1"))

    rows = rows_by_root(collect(db))
    assert rows["/u/fed-job"].pending == 1
    assert rows[CLUSTER_ROLLUP_ROOT_JOB].pending == 1
