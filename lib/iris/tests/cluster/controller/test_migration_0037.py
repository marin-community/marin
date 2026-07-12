# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0037_federation_fixup``.

Builds a DB in the pre-revision 0034/0035 shape — jobs/tasks carrying
``child_cluster`` (default ``''``) instead of ``cluster``, the two tasks partial
indexes filtering on ``child_cluster = ''``, and the extra ``federated_jobs`` /
``federation_sync_state`` bookkeeping columns — and asserts the migration renames
the coordinate to ``cluster`` (defaulting existing rows to ``'local'``), drops the
stale columns, rebuilds the partial indexes on ``cluster = 'local'``, and is both
idempotent and a no-op on a DB already in the current shape.
"""

import importlib.util
import sqlite3
from pathlib import Path

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0037_federation_fixup.py"

# Pre-revision (post-#6821) shape of the tables 0037 reconciles: child_cluster on
# jobs/tasks, the tasks partial indexes over child_cluster, and the extra
# federated_jobs / federation_sync_state columns #6884 removed.
_OLD_SCHEMA = """
CREATE TABLE jobs (
    job_id VARCHAR PRIMARY KEY,
    state INTEGER NOT NULL,
    child_cluster VARCHAR NOT NULL DEFAULT ''
);
CREATE TABLE tasks (
    task_id VARCHAR PRIMARY KEY,
    job_id VARCHAR NOT NULL,
    state INTEGER NOT NULL,
    priority_band INTEGER NOT NULL DEFAULT 2,
    priority_neg_depth INTEGER NOT NULL DEFAULT 0,
    priority_root_submitted_ms INTEGER NOT NULL DEFAULT 0,
    submitted_at_ms INTEGER NOT NULL DEFAULT 0,
    priority_insertion INTEGER NOT NULL DEFAULT 0,
    child_cluster VARCHAR NOT NULL DEFAULT ''
);
CREATE INDEX idx_tasks_pending_local ON tasks
    (state, priority_band, priority_neg_depth, priority_root_submitted_ms,
     submitted_at_ms, priority_insertion) WHERE child_cluster = '';
CREATE INDEX idx_tasks_state_local ON tasks (state) WHERE child_cluster = '';
CREATE TABLE federated_jobs (
    job_id VARCHAR PRIMARY KEY,
    direction INTEGER NOT NULL,
    peer_id VARCHAR NOT NULL,
    owner_principal VARCHAR NOT NULL DEFAULT '',
    remote_job_id VARCHAR,
    handoff_state INTEGER,
    cancel_intent_version INTEGER NOT NULL DEFAULT 0,
    last_sync_ms INTEGER,
    terminal_error VARCHAR
);
CREATE TABLE federation_sync_state (
    peer_id VARCHAR PRIMARY KEY,
    cursor VARCHAR NOT NULL DEFAULT '',
    last_full_resync_ms INTEGER
);
"""

# The current (post-#6884) shape, built the way a fresh baseline DB is: cluster
# instead of child_cluster, partial indexes over cluster, no stale columns.
_CURRENT_SCHEMA = """
CREATE TABLE jobs (
    job_id VARCHAR PRIMARY KEY,
    state INTEGER NOT NULL,
    cluster VARCHAR NOT NULL DEFAULT 'local'
);
CREATE TABLE tasks (
    task_id VARCHAR PRIMARY KEY,
    job_id VARCHAR NOT NULL,
    state INTEGER NOT NULL,
    priority_band INTEGER NOT NULL DEFAULT 2,
    priority_neg_depth INTEGER NOT NULL DEFAULT 0,
    priority_root_submitted_ms INTEGER NOT NULL DEFAULT 0,
    submitted_at_ms INTEGER NOT NULL DEFAULT 0,
    priority_insertion INTEGER NOT NULL DEFAULT 0,
    cluster VARCHAR NOT NULL DEFAULT 'local'
);
CREATE INDEX idx_tasks_pending_local ON tasks
    (state, priority_band, priority_neg_depth, priority_root_submitted_ms,
     submitted_at_ms, priority_insertion) WHERE cluster = 'local';
CREATE INDEX idx_tasks_state_local ON tasks (state) WHERE cluster = 'local';
CREATE TABLE federated_jobs (
    job_id VARCHAR PRIMARY KEY,
    direction INTEGER NOT NULL,
    peer_id VARCHAR NOT NULL,
    owner_principal VARCHAR NOT NULL DEFAULT '',
    handoff_state INTEGER,
    cancel_intent_version INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE federation_sync_state (
    peer_id VARCHAR PRIMARY KEY,
    cursor VARCHAR NOT NULL DEFAULT ''
);
"""


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0037", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _index_sql(conn: sqlite3.Connection, name: str) -> str:
    return conn.execute("SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (name,)).fetchone()[0]


def test_migration_0037_reconciles_child_cluster_and_drops_stale_columns():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_OLD_SCHEMA)
    conn.execute("INSERT INTO jobs (job_id, state) VALUES ('/u1/a', 1), ('/u1/b', 1)")
    conn.execute("INSERT INTO tasks (task_id, job_id, state) VALUES ('/u1/a/0', '/u1/a', 1), ('/u1/b/0', '/u1/b', 1)")
    conn.commit()

    _load_migration().migrate(conn)
    conn.commit()

    # child_cluster is gone; cluster is present and every existing (local) row is 'local'.
    for table in ("jobs", "tasks"):
        assert "child_cluster" not in _columns(conn, table), table
        assert "cluster" in _columns(conn, table), table
        assert {row[0] for row in conn.execute(f"SELECT DISTINCT cluster FROM {table}")} == {"local"}, table

    # Stale federation bookkeeping columns are dropped.
    assert {"remote_job_id", "last_sync_ms", "terminal_error"}.isdisjoint(_columns(conn, "federated_jobs"))
    assert "last_full_resync_ms" not in _columns(conn, "federation_sync_state")

    # The partial indexes now filter on cluster='local', not child_cluster.
    for name in ("idx_tasks_pending_local", "idx_tasks_state_local"):
        sql = _index_sql(conn, name)
        assert "cluster = 'local'" in sql, name
        assert "child_cluster" not in sql, name

    conn.close()


def test_migration_0037_is_idempotent_on_repeat():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_OLD_SCHEMA)
    conn.commit()

    migration = _load_migration()
    migration.migrate(conn)
    conn.commit()
    migration.migrate(conn)  # second run must not error
    conn.commit()

    assert "cluster" in _columns(conn, "tasks")
    assert "child_cluster" not in _columns(conn, "tasks")
    conn.close()


def test_migration_0037_is_noop_on_current_schema():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_CURRENT_SCHEMA)
    conn.commit()

    before = {
        row[0]: row[1] for row in conn.execute("SELECT name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")
    }

    _load_migration().migrate(conn)
    conn.commit()

    after = {
        row[0]: row[1] for row in conn.execute("SELECT name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")
    }
    assert before == after
    conn.close()
