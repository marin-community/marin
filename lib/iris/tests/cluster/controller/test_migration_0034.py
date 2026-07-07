# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0034_federation``.

Builds a pre-migration DB (jobs/tasks without ``cluster`` and no federation
sidecar tables), seeds rows, and asserts the migration: adds the ``cluster``
column to jobs/tasks (defaulting existing rows to ``'local'``), creates the
three sidecar tables and the two partial indexes, and is idempotent on re-run
while leaving an already-set ``cluster`` untouched.
"""

import importlib.util
import sqlite3
from pathlib import Path

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0034_federation.py"

# The pre-0034 tasks table carries the priority columns the partial index
# ``idx_tasks_pending_local`` is built over.
_OLD_SCHEMA = """
CREATE TABLE jobs (job_id VARCHAR PRIMARY KEY, state INTEGER NOT NULL);
CREATE TABLE tasks (
    task_id VARCHAR PRIMARY KEY,
    job_id VARCHAR NOT NULL,
    state INTEGER NOT NULL,
    priority_band INTEGER NOT NULL DEFAULT 2,
    priority_neg_depth INTEGER NOT NULL DEFAULT 0,
    priority_root_submitted_ms INTEGER NOT NULL DEFAULT 0,
    submitted_at_ms INTEGER NOT NULL DEFAULT 0,
    priority_insertion INTEGER NOT NULL DEFAULT 0
);
"""


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0034", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(conn: sqlite3.Connection) -> None:
    conn.executescript(_OLD_SCHEMA)
    conn.execute("INSERT INTO jobs (job_id, state) VALUES ('/u1/a', 1), ('/u1/b', 1)")
    conn.execute("INSERT INTO tasks (task_id, job_id, state) VALUES ('/u1/a/0', '/u1/a', 1), ('/u1/b/0', '/u1/b', 1)")
    conn.commit()


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _indexes(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}


def test_migration_0034_adds_column_tables_indexes_and_is_idempotent():
    conn = sqlite3.connect(":memory:")
    _seed(conn)

    migration = _load_migration()
    migration.migrate(conn)

    # Existing rows are local — the added column defaults to 'local'.
    for table in ("jobs", "tasks"):
        values = {row[0] for row in conn.execute(f"SELECT DISTINCT cluster FROM {table}")}
        assert values == {"local"}, table

    assert {"federated_jobs", "federation_sync_state", "federated_tasks"} <= _tables(conn)
    assert {"idx_tasks_pending_local", "idx_tasks_state_local"} <= _indexes(conn)

    # A row federated after the migration must survive a re-run: the column add
    # is guarded and no backfill rewrites it.
    conn.execute("UPDATE tasks SET cluster = 'peer-west' WHERE task_id = '/u1/b/0'")
    conn.commit()

    migration.migrate(conn)  # idempotent re-run

    assert conn.execute("SELECT cluster FROM tasks WHERE task_id='/u1/b/0'").fetchone()[0] == "peer-west"
    assert conn.execute("SELECT cluster FROM tasks WHERE task_id='/u1/a/0'").fetchone()[0] == "local"
    conn.close()
