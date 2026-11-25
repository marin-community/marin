# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The observability row check against a real controller DB and a real embedded Finelog."""

from datetime import UTC, date, datetime, timedelta

import pytest
from finelog.client.log_client import FlushResult
from iris.cluster.controller import ops, writes
from iris.cluster.controller.federation_store import ControllerFederationStore
from iris.cluster.controller.observability_rows import ObservabilityRowTables
from iris.cluster.federation.store import HandoffAdmission, HandoffSpec
from iris.cluster.stats.observability_rows_check import DayWindow, RowTable, TableCoverage, check_days, day_coverage
from iris.cluster.stats.tables import (
    CONTROLLER_JOB_CONFIG_NAMESPACE,
    CONTROLLER_JOB_NAMESPACE,
    CONTROLLER_TASK_ATTEMPT_NAMESPACE,
    CONTROLLER_TASK_NAMESPACE,
    IrisJob,
    IrisJobConfig,
    IrisTask,
    IrisTaskAttempt,
)
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from iris.test_util import FakeStatsTable
from iris.testing.controller import (
    assign_task,
    make_controller_state,
    make_direct_job_request,
    make_job_request,
    make_worker_metadata,
    register_worker,
    submit_job,
    transition_task,
)
from sqlalchemy import text

CLUSTER = "marin"


@pytest.fixture
def state():
    with make_controller_state() as controller_state:
        yield controller_state


@pytest.fixture
def finelog_tables(log_client) -> ObservabilityRowTables:
    return ObservabilityRowTables(
        cluster=CLUSTER,
        job_table=log_client.get_table(CONTROLLER_JOB_NAMESPACE, IrisJob),
        job_config_table=log_client.get_table(CONTROLLER_JOB_CONFIG_NAMESPACE, IrisJobConfig),
        task_table=log_client.get_table(CONTROLLER_TASK_NAMESPACE, IrisTask),
        task_attempt_table=log_client.get_table(CONTROLLER_TASK_ATTEMPT_NAMESPACE, IrisTaskAttempt),
    )


def _dropping_tables() -> ObservabilityRowTables:
    """Tables that swallow every row, as when a transaction's batch is lost."""
    return ObservabilityRowTables(
        cluster=CLUSTER,
        job_table=FakeStatsTable(),
        job_config_table=FakeStatsTable(),
        task_table=FakeStatsTable(),
        task_attempt_table=FakeStatsTable(),
    )


def _run_to_success(state, name: str, worker_id) -> None:
    tasks = submit_job(state, name, make_job_request(name))
    assign_task(state, tasks[0], worker_id)
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_RUNNING)
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)


def _hand_off(state, name: str) -> None:
    """A job this controller hands to a federation peer: it stays in the DB but is not run here."""
    job_id = JobName.root("test-user", name)
    spec = HandoffSpec(
        local_job_id=job_id,
        peer_id="cw-rno2a",
        owner_principal="test-user",
        submitting_user="test-user",
        request=make_direct_job_request(name, replicas=1, priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE),
    )
    assert ControllerFederationStore(state._db).admit_and_persist_queued(spec) is HandoffAdmission.ADMITTED
    with state._db.transaction() as cur:
        writes.promote_queued_handoff(cur, job_id, "cw-rno2a")


def _coverage(
    state, log_client, tables: ObservabilityRowTables, day: date, *, settled_before: datetime | None = None
) -> dict[RowTable, TableCoverage]:
    for table in (tables.job_table, tables.job_config_table, tables.task_table, tables.task_attempt_table):
        assert table.flush(timeout=30) == FlushResult.SUCCEEDED

    def controller_query(sql: str):
        with state._db.read_snapshot() as snap:
            return [tuple(row) for row in snap.execute(text(sql))]

    def finelog_query(sql: str):
        result = log_client.query(sql)
        return list(zip(*(column.to_pylist() for column in result.columns), strict=True))

    now = datetime.now(UTC)
    window = DayWindow(day=day, settled_before=settled_before or now, until=now + timedelta(minutes=1))
    return {c.table: c for c in day_coverage(controller_query, finelog_query, CLUSTER, window)}


def _counts(coverage: dict[RowTable, TableCoverage]) -> list[tuple[int, int]]:
    return [(coverage[t].controller, coverage[t].finelog) for t in RowTable]


def test_day_coverage_counts_only_that_days_local_jobs(state, log_client, finelog_tables):
    today = datetime.now(UTC).date()
    state._db.attach_observability_rows(finelog_tables)
    worker = register_worker(state, "w1", "host1:8080", make_worker_metadata())

    _run_to_success(state, "train", worker)
    submit_job(state, "pending", make_job_request("pending"))
    _hand_off(state, "fed-job")

    coverage = _coverage(state, log_client, finelog_tables, today)
    assert all(c.ok for c in coverage.values())
    assert _counts(coverage) == [(2, 2), (2, 2), (2, 2), (1, 1)]
    assert coverage[RowTable.JOB].settled_terminal == 1
    assert coverage[RowTable.TASK].settled_terminal == 1
    assert coverage[RowTable.TASK_ATTEMPT].settled_terminal == 1
    for other_day in (today - timedelta(days=1), today + timedelta(days=1)):
        assert _counts(_coverage(state, log_client, finelog_tables, other_day)) == [(0, 0), (0, 0), (0, 0), (0, 0)]


def test_day_coverage_reports_a_job_whose_submit_batch_was_dropped(state, log_client, finelog_tables):
    today = datetime.now(UTC).date()
    worker = register_worker(state, "w1", "host1:8080", make_worker_metadata())
    state._db.attach_observability_rows(_dropping_tables())
    submit_job(state, "lost", make_job_request("lost"))
    state._db.attach_observability_rows(finelog_tables)
    _run_to_success(state, "kept", worker)

    coverage = _coverage(state, log_client, finelog_tables, today)

    for table in (RowTable.JOB, RowTable.JOB_CONFIG):
        assert not coverage[table].ok
        assert [key[2] for key in coverage[table].missing] == ["/test-user/lost"]
        assert coverage[table].extra == ()
    assert [key[0] for key in coverage[RowTable.TASK].missing] == ["/test-user/lost/0"]
    assert coverage[RowTable.TASK_ATTEMPT].ok


def test_day_coverage_reports_a_settled_finish_whose_batch_was_dropped(state, log_client, finelog_tables):
    """The keys reached Finelog at submit and start; only the finish rows are lost."""
    today = datetime.now(UTC).date()
    state._db.attach_observability_rows(finelog_tables)
    worker = register_worker(state, "w1", "host1:8080", make_worker_metadata())
    tasks = submit_job(state, "train", make_job_request("train"))
    assign_task(state, tasks[0], worker)
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_RUNNING)
    state._db.attach_observability_rows(_dropping_tables())
    before_finish = datetime.now(UTC)
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)

    settled = _coverage(state, log_client, finelog_tables, today)
    for table in (RowTable.JOB, RowTable.TASK, RowTable.TASK_ATTEMPT):
        assert settled[table].missing == ()
        assert len(settled[table].missing_terminal) == 1
        assert not settled[table].ok
    assert settled[RowTable.JOB_CONFIG].ok

    # A finish after the settle cutoff may still be in flight, so it is not counted yet.
    in_flight = _coverage(state, log_client, finelog_tables, today, settled_before=before_finish)
    assert all(c.ok and c.settled_terminal == 0 for c in in_flight.values())


def test_day_coverage_reports_a_removed_job_as_extra_without_failing(state, log_client, finelog_tables):
    """A finished job removed before the pruner reaches it, as when a new job takes its id."""
    today = datetime.now(UTC).date()
    state._db.attach_observability_rows(finelog_tables)
    worker = register_worker(state, "w1", "host1:8080", make_worker_metadata())
    _run_to_success(state, "train", worker)
    with state._db.transaction() as cur:
        assert ops.job.remove_finished(cur, JobName.root("test-user", "train"))

    coverage = _coverage(state, log_client, finelog_tables, today)

    assert all(c.ok for c in coverage.values())
    assert [key[2] for key in coverage[RowTable.JOB].extra] == ["/test-user/train"]
    assert len(coverage[RowTable.TASK].extra) == 1
    assert len(coverage[RowTable.TASK_ATTEMPT].extra) == 1
    assert _counts(coverage) == [(0, 1), (0, 1), (0, 1), (0, 1)]


def test_check_days_covers_settled_days_inside_retention_from_the_start_date():
    morning = datetime(2026, 11, 20, 6, tzinfo=UTC)

    assert check_days(morning, start=date(2026, 11, 18), days=5) == [date(2026, 11, 18), date(2026, 11, 19)]
    assert check_days(morning, start=date(2026, 11, 1), days=3) == [
        date(2026, 11, 17),
        date(2026, 11, 18),
        date(2026, 11, 19),
    ]
    assert check_days(morning, start=date(2026, 11, 20), days=5) == []
    # Within an hour after midnight the previous day has not settled, and the
    # oldest day must still start after the retention cutoff (2026-11-13 00:30).
    just_after_midnight = datetime(2026, 11, 20, 0, 30, tzinfo=UTC)
    assert check_days(just_after_midnight, start=date(2026, 11, 1), days=6) == [
        date(2026, 11, 14) + timedelta(days=i) for i in range(5)
    ]
