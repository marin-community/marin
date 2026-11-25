# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The one-time controller-rows backfill against an in-process Finelog, with made-up rows."""

import re
from datetime import datetime, timedelta

import pyarrow as pa
import pytest
from finelog.client.log_client import FlushResult
from finelog.errors import StatsError
from iris.cluster.controller import observability_backfill, ops
from iris.cluster.controller.checkpoint import CHECKPOINT_EPOCH_META_KEY
from iris.cluster.controller.observability_backfill import (
    ATTEMPTS,
    JOB_CONFIGS,
    JOBS,
    TARGETS,
    TASKS,
    BackfillRefused,
    plan_backfill,
    write_backfill,
)
from iris.cluster.controller.observability_rows import ObservabilityRowTables
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.schema import meta_table
from iris.cluster.stats.tables import IrisJob
from iris.rpc import job_pb2
from iris.testing.controller import (
    make_controller_state,
    make_job_request,
    make_worker_metadata,
    register_worker,
    submit_job,
    transition_task,
)
from rigging.timing import Timestamp
from sqlalchemy.dialects.sqlite import insert

CLUSTER = "marin"


@pytest.fixture
def state():
    with make_controller_state() as controller_state:
        yield controller_state


@pytest.fixture
def tables(log_client):
    return {
        target: log_client.get_table(target.namespace, target.row_class, storage_policy=target.storage_policy)
        for target in TARGETS
    }


def _attach_live_rows(state, tables) -> None:
    state._db.attach_observability_rows(
        ObservabilityRowTables(
            cluster=CLUSTER,
            job_table=tables[JOBS],
            job_config_table=tables[JOB_CONFIGS],
            task_table=tables[TASKS],
            task_attempt_table=tables[ATTEMPTS],
        )
    )


def _flush(tables) -> None:
    for table in tables.values():
        assert table.flush(timeout=30) == FlushResult.SUCCEEDED


def _record_checkpoint(state, epoch_ms: int) -> None:
    with state._db.transaction() as cur:
        stmt = insert(meta_table).values(key=CHECKPOINT_EPOCH_META_KEY, value=epoch_ms)
        cur.execute(stmt.on_conflict_do_update(index_elements=["key"], set_={"value": epoch_ms}))


def _job_rows(log_client, job_id: str) -> list[dict]:
    sql = f"SELECT state, ts FROM \"{JOBS.namespace}\" WHERE job_id = '{job_id}' ORDER BY ts DESC, seq DESC"
    return log_client.query(sql).to_pylist()


def _plan(state, log_client, live_start_ms: int, *, rerun: bool = False):
    return plan_backfill(
        state._db,
        log_client,
        cluster=CLUSTER,
        live_start_ms=live_start_ms,
        rerun=rerun,
        now=Timestamp.now().as_naive_utc(),
        sleep=lambda _: None,
    )


def _no_sleep(_: float) -> None:
    pass


def test_backfill_writes_entities_without_live_rows_once_and_below_every_live_row(state, log_client, tables):
    now_ms = Timestamp.now().epoch_ms()
    live_start_ms = now_ms - 3_600_000
    # Before the live rows: one job that finished, one still pending.
    done = submit_job(state, "done", make_job_request("done"), timestamp_ms=now_ms - 2 * 86_400_000)
    worker = register_worker(state, "w1", "host1:8080", make_worker_metadata())
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=done[0].task_id, worker_id=worker)], health=state._health)
    transition_task(state, done[0].task_id, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)
    pending = submit_job(state, "pending", make_job_request("pending"), timestamp_ms=now_ms - 86_400_000)
    _attach_live_rows(state, tables)
    submit_job(state, "live", make_job_request("live"))
    _flush(tables)
    _record_checkpoint(state, now_ms)

    plans = _plan(state, log_client, live_start_ms)
    write_backfill(plans, tables, sleep=_no_sleep)

    to_write = {plan.target: {row.job_id for row in plan.to_write} for plan in plans}
    assert to_write[JOBS] == to_write[JOB_CONFIGS] == {"/test-user/done", "/test-user/pending"}
    assert {plan.target: len(plan.to_write) for plan in plans if plan.target in (TASKS, ATTEMPTS)} == {
        TASKS: 2,
        ATTEMPTS: 1,
    }
    live_start = Timestamp.from_ms(live_start_ms).as_naive_utc()
    assert all(row.ts < live_start for plan in plans for row in plan.to_write)
    assert [r["state"] for r in _job_rows(log_client, "/test-user/done")] == ["JOB_STATE_SUCCEEDED"]
    assert all(r["ts"] >= live_start for r in _job_rows(log_client, "/test-user/live"))

    # The pending job changes after the backfill: its live row sorts first.
    transition_task(state, pending[0].task_id, job_pb2.TASK_STATE_KILLED, error="made-up cancel")
    _flush(tables)
    assert _job_rows(log_client, "/test-user/pending")[0]["state"] == "JOB_STATE_KILLED"

    again = _plan(state, log_client, live_start_ms, rerun=True)
    assert [len(plan.to_write) for plan in again] == [0, 0, 0, 0]


def test_a_rerun_must_be_declared(state, log_client, tables):
    now_ms = Timestamp.now().epoch_ms()
    live_start_ms = now_ms - 3_600_000
    submit_job(state, "old", make_job_request("old"), timestamp_ms=now_ms - 86_400_000)
    _record_checkpoint(state, now_ms)
    write_backfill(_plan(state, log_client, live_start_ms), tables, sleep=_no_sleep)
    _flush(tables)

    with pytest.raises(BackfillRefused, match="G4"):
        _plan(state, log_client, live_start_ms)


@pytest.mark.parametrize(
    "live_start_offset, checkpoint_offset, guard",
    [
        (timedelta(minutes=-5), timedelta(0), "G1"),
        (timedelta(hours=-1), timedelta(hours=-2), "G2"),
    ],
    ids=["live-start-too-recent", "checkpoint-before-live-start"],
)
def test_live_start_is_checked_against_now_and_the_checkpoint(
    state, log_client, live_start_offset, checkpoint_offset, guard
):
    now_ms = Timestamp.now().epoch_ms()
    _record_checkpoint(state, now_ms + int(checkpoint_offset.total_seconds() * 1000))

    with pytest.raises(BackfillRefused, match=guard):
        _plan(state, log_client, now_ms + int(live_start_offset.total_seconds() * 1000))


def test_a_live_start_before_the_live_rows_began_is_refused(state, log_client):
    # Submitted after the stated live_start, yet with no live config row: live rows started later.
    now_ms = Timestamp.now().epoch_ms()
    submit_job(state, "unseen", make_job_request("unseen"), timestamp_ms=now_ms - 1_800_000)
    _record_checkpoint(state, now_ms)

    with pytest.raises(BackfillRefused, match="G3"):
        _plan(state, log_client, now_ms - 3_600_000)


class _FailingQuerier:
    """Fails any key query whose window is wider than ``widest``; records the windows tried."""

    def __init__(self, widest: timedelta | None):
        self.widest = widest
        self.windows: list[timedelta] = []

    def query(self, sql: str, *, max_rows: int):
        lo, hi = (datetime.fromisoformat(value) for value in re.findall(r"TIMESTAMP '([^']+)'", sql))
        self.windows.append(hi - lo)
        if self.widest is None or hi - lo > self.widest:
            raise StatsError("made-up deadline exceeded")
        return pa.table({c: [] for c in (*JOBS.key_columns, "first_ts")})

    def list_namespaces(self):
        return []


def test_a_key_scan_halves_a_failed_day_and_stops_at_one_hour():
    start = Timestamp.from_ms(0).as_naive_utc()
    end = start + timedelta(days=1)

    halving = _FailingQuerier(widest=timedelta(hours=6))
    assert observability_backfill.existing_keys(halving, JOBS, CLUSTER, start, end, sleep=_no_sleep) == {}
    assert halving.windows[:3] == [timedelta(days=1), timedelta(hours=12), timedelta(hours=6)]

    with pytest.raises(BackfillRefused, match="smallest window"):
        observability_backfill.existing_keys(_FailingQuerier(widest=None), JOBS, CLUSTER, start, end, sleep=_no_sleep)


class _DroppingTable:
    def __init__(self):
        self.written: list[list] = []

    def write(self, rows) -> None:
        self.written.append(list(rows))

    def flush(self, timeout: float) -> FlushResult:
        return FlushResult.DROPPED


def test_a_batch_that_does_not_land_stops_the_run():
    plan = observability_backfill.TablePlan(JOBS, db_rows=[], existing={})
    plan.to_write = [IrisJob.__new__(IrisJob) for _ in range(3)]
    table = _DroppingTable()

    with pytest.raises(BackfillRefused, match="batch 1/2 flush dropped after 0 of 3 rows"):
        write_backfill([plan], {JOBS: table}, batch_rows=2, sleep=_no_sleep)
    assert [len(batch) for batch in table.written] == [2]

    with pytest.raises(BackfillRefused, match="batch_rows 10000"):
        write_backfill([plan], {JOBS: table}, batch_rows=10_000, sleep=_no_sleep)
    assert len(table.written) == 1
