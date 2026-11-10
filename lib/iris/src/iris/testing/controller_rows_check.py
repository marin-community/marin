# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A test-time check that every committed job or attempt change has a controller row.

Controller rows are marked at the write sites that change ``jobs`` and
``task_attempts`` (see ``iris.cluster.controller.controller_rows``). A new write
path that forgets to mark its rows would drop them silently in production. This
check wraps a test DB's ``transaction()``: after each commit, every locally owned
job or attempt whose columns changed must have a row written by that
transaction.

Not checked: ``job_config`` rows (written once, at submit), deleted keys, and
tests attached to real Finelog tables.
"""

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import select

import iris
from iris.cluster.controller.controller_rows import ControllerRowTables
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.schema import jobs_table, task_attempts_table
from iris.cluster.types import LOCAL_CLUSTER
from iris.test_util import FakeStatsTable

# Every controller column is copied (``placement`` is ``jobs.cluster``), so the
# check compares whole rows. ``worker_id`` is compared separately: remove_worker
# clears it on finished attempts, and the rows keep the old value on purpose.
_JOB_COLUMNS = tuple(column.name for column in jobs_table.columns)
_ATTEMPT_COLUMNS = tuple(column.name for column in task_attempts_table.columns if column.name != "worker_id")
_PACKAGE_TESTS_DIR = str(Path(iris.__file__).resolve().parents[2] / "tests") + os.sep


def _local_state(cur: Tx) -> tuple[dict[str, tuple], dict[tuple[str, int], tuple]]:
    j = jobs_table.c
    jobs = {
        r.job_id.to_wire(): tuple(getattr(r, c) for c in _JOB_COLUMNS)
        for r in cur.execute(select(jobs_table).where(j.cluster == LOCAL_CLUSTER))
    }
    attempts = {
        (r.task_id.to_wire(), r.attempt_id): (tuple(getattr(r, c) for c in _ATTEMPT_COLUMNS), r.worker_id)
        for r in cur.execute(select(task_attempts_table))
        if r.task_id.parent.to_wire() in jobs
    }
    return jobs, attempts


def _opened_by_a_test_file() -> bool:
    """Whether the ``with db.transaction()`` was written in a test module.

    Tests seed and override state with raw SQL on purpose; only transactions that
    production code (reached through the test helpers) opens are checked.
    """
    frame = sys._getframe(2)
    while frame is not None and frame.f_code.co_filename.endswith("contextlib.py"):
        frame = frame.f_back
    module_file = frame.f_globals.get("__file__") if frame is not None else None
    return module_file is not None and str(Path(module_file).resolve()).startswith(_PACKAGE_TESTS_DIR)


def _written_keys(tables: ControllerRowTables, marks: dict[int, int]) -> tuple[set[str], set[tuple[str, int]]]:
    job_table = tables.job_table
    attempt_table = tables.task_attempt_table
    assert isinstance(job_table, FakeStatsTable) and isinstance(attempt_table, FakeStatsTable)
    jobs = {row.job_id for write in job_table.writes[marks.get(id(job_table), 0) :] for row in write}
    attempts = {
        (row.task_id, row.attempt_id)
        for write in attempt_table.writes[marks.get(id(attempt_table), 0) :]
        for row in write
    }
    return jobs, attempts


def _write_marks(tables: ControllerRowTables) -> dict[int, int]:
    return {
        id(table): len(table.writes)
        for table in (tables.job_table, tables.task_attempt_table)
        if isinstance(table, FakeStatsTable)
    }


def install(db: ControllerDB) -> None:
    """Attach fake row tables to ``db`` and check its rows after every committed transaction.

    A test that attaches its own fake tables keeps the check; a test that makes
    row writes fail on purpose disables it with :func:`uninstall`.
    """
    db.attach_controller_rows(
        ControllerRowTables(
            cluster="test",
            job_table=FakeStatsTable(),
            job_config_table=FakeStatsTable(),
            task_attempt_table=FakeStatsTable(),
        )
    )
    unchecked = db.transaction

    @contextmanager
    def checked() -> Iterator[Tx]:
        tables = db._controller_rows
        if tables is None or not isinstance(tables.job_table, FakeStatsTable) or _opened_by_a_test_file():
            with unchecked() as cur:
                yield cur
            return
        marks = _write_marks(tables)
        # Both snapshots are read inside the transaction, under the write lock, so a
        # concurrent transaction cannot show up as this one's change.
        after: list = []
        with unchecked() as cur:
            before_jobs, before_attempts = _local_state(cur)
            cur.register_pre_commit(lambda: after.append(_local_state(cur)))
            yield cur
        if not after:
            return
        after_jobs, after_attempts = after[0]
        written_jobs, written_attempts = _written_keys(tables, marks)
        missed_jobs = [k for k, v in after_jobs.items() if before_jobs.get(k) != v and k not in written_jobs]
        missed_attempts = [
            k
            for k, (v, worker_id) in after_attempts.items()
            # remove_worker clears worker_id on finished attempts too; that is not a change to check.
            if k not in before_attempts
            or before_attempts[k][0] != v
            or (worker_id and worker_id != before_attempts[k][1])
            if k not in written_attempts
        ]
        assert not missed_jobs and not missed_attempts, (
            "a committed transaction changed jobs or attempts without writing their controller rows; "
            f"mark them at the write site (controller_rows.record_*): jobs {missed_jobs}, attempts {missed_attempts}"
        )

    db._row_check_unchecked = unchecked  # type: ignore[attr-defined]
    db.transaction = checked  # type: ignore[method-assign]


def uninstall(db: ControllerDB) -> None:
    db.transaction = db._row_check_unchecked  # type: ignore[attr-defined,method-assign]
