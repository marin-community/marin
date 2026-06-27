# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0032_backend_id``.

Builds a pre-migration DB (jobs/tasks/task_attempts without ``backend_id`` and no
``backends`` table), seeds rows, and asserts the migration: adds the columns,
backfills every existing row to the implicit ``DEFAULT_BACKEND_ID``, inserts one
ACTIVE ``backends`` row, creates the indexes — and is idempotent on re-run while
leaving an already-stamped row untouched.
"""

import importlib.util
import sqlite3
from pathlib import Path

from iris.cluster.types import DEFAULT_BACKEND_ID, BackendStatus

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0032_backend_id.py"

_OLD_SCHEMA = """
CREATE TABLE jobs (job_id VARCHAR PRIMARY KEY, state INTEGER NOT NULL, name VARCHAR NOT NULL DEFAULT '');
CREATE TABLE tasks (task_id VARCHAR PRIMARY KEY, job_id VARCHAR NOT NULL, state INTEGER NOT NULL);
CREATE TABLE task_attempts (
    task_id VARCHAR NOT NULL,
    attempt_id INTEGER NOT NULL,
    state INTEGER NOT NULL,
    attempt_uid VARCHAR NOT NULL,
    PRIMARY KEY (task_id, attempt_id)
);
"""


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0032", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(conn: sqlite3.Connection) -> None:
    conn.executescript(_OLD_SCHEMA)
    conn.execute("INSERT INTO jobs (job_id, state) VALUES ('/u1/a', 1), ('/u1/b', 1)")
    conn.execute("INSERT INTO tasks (task_id, job_id, state) VALUES ('/u1/a/0', '/u1/a', 1), ('/u1/b/0', '/u1/b', 1)")
    conn.execute(
        "INSERT INTO task_attempts (task_id, attempt_id, state, attempt_uid) "
        "VALUES ('/u1/a/0', 0, 1, 'uidA'), ('/u1/b/0', 0, 1, 'uidB')"
    )
    conn.commit()


def _indexes(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}


def test_migration_0032_backfills_and_is_idempotent():
    conn = sqlite3.connect(":memory:")
    _seed(conn)

    migration = _load_migration()
    migration.migrate(conn)

    # Every pre-existing row is stamped with the implicit backend id.
    for table in ("jobs", "tasks", "task_attempts"):
        ids = {row[0] for row in conn.execute(f"SELECT DISTINCT backend_id FROM {table}")}
        assert ids == {DEFAULT_BACKEND_ID}, table

    # Exactly one backends row for the implicit backend, status ACTIVE.
    rows = conn.execute("SELECT backend_id, status FROM backends").fetchall()
    assert rows == [(DEFAULT_BACKEND_ID, int(BackendStatus.ACTIVE))]

    assert {"idx_tasks_backend_state", "idx_task_attempts_backend"} <= _indexes(conn)

    # A row stamped with a different backend after the migration must survive a
    # re-run: the backfill only touches rows still at the '' default.
    conn.execute("INSERT INTO jobs (job_id, state, backend_id) VALUES ('/u1/c', 1, 'other')")
    conn.commit()

    migration.migrate(conn)  # idempotent re-run

    assert conn.execute("SELECT backend_id FROM jobs WHERE job_id='/u1/c'").fetchone()[0] == "other"
    assert conn.execute("SELECT COUNT(*) FROM backends").fetchone()[0] == 1
    a_backend = conn.execute("SELECT backend_id FROM jobs WHERE job_id='/u1/a'").fetchone()[0]
    assert a_backend == DEFAULT_BACKEND_ID
    conn.close()
