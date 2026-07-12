# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0040_drop_users``.

Builds a DB in the pre-0040 shape — ``users`` table plus ``jobs`` / ``user_budgets``
carrying ``FOREIGN KEY(user_id) REFERENCES users`` — with FK enforcement ON (as in
production), and asserts the migration rebuilds ``jobs`` / ``user_budgets`` without
the ``users`` FK, drops ``users``, preserves all rows and the ``jobs`` self-FK / its
child ``tasks``, leaves no FK violations, and is idempotent + a no-op on a DB already
in the current (no-``users``) shape. This is the riskiest migration in the tree — it
rebuilds the ``jobs`` table — so it gets direct coverage.
"""

import importlib.util
import sqlite3
from pathlib import Path

import pytest

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0040_drop_users.py"

# Pre-0040 shape: users table + the two FKs to it. jobs carries the full column set
# (the migration's INSERT names every column) plus its parent_job_id self-FK; tasks
# is a child of jobs, present to prove the rebuild does not cascade-delete children.
_OLD_SCHEMA = """
CREATE TABLE users (
    user_id VARCHAR NOT NULL,
    role VARCHAR NOT NULL DEFAULT 'user',
    PRIMARY KEY (user_id)
);
CREATE TABLE jobs (
    job_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    parent_job_id VARCHAR,
    root_job_id VARCHAR NOT NULL,
    depth INTEGER NOT NULL,
    state INTEGER NOT NULL,
    submitted_at_ms INTEGER NOT NULL,
    root_submitted_at_ms INTEGER NOT NULL,
    started_at_ms INTEGER,
    finished_at_ms INTEGER,
    scheduling_deadline_epoch_ms INTEGER,
    error VARCHAR,
    exit_code INTEGER,
    num_tasks INTEGER NOT NULL,
    name VARCHAR DEFAULT '' NOT NULL,
    backend_id VARCHAR DEFAULT '' NOT NULL,
    cluster VARCHAR DEFAULT 'local' NOT NULL,
    PRIMARY KEY (job_id),
    FOREIGN KEY(user_id) REFERENCES users (user_id),
    FOREIGN KEY(parent_job_id) REFERENCES jobs (job_id) ON DELETE CASCADE
);
CREATE INDEX idx_jobs_parent ON jobs (parent_job_id);
CREATE INDEX idx_jobs_state ON jobs (state, submitted_at_ms DESC);
CREATE INDEX idx_jobs_depth_state ON jobs (depth, state, submitted_at_ms DESC);
CREATE INDEX idx_jobs_user_state ON jobs (user_id, state);
CREATE INDEX idx_jobs_root_depth ON jobs (root_job_id, depth);
CREATE INDEX idx_jobs_depth_submitted ON jobs (depth, submitted_at_ms DESC);
CREATE INDEX idx_jobs_name ON jobs (name);
CREATE TABLE tasks (
    task_id VARCHAR NOT NULL,
    job_id VARCHAR NOT NULL,
    PRIMARY KEY (task_id),
    FOREIGN KEY(job_id) REFERENCES jobs (job_id) ON DELETE CASCADE
);
CREATE TABLE user_budgets (
    user_id VARCHAR NOT NULL,
    budget_limit INTEGER DEFAULT '0' NOT NULL,
    max_band INTEGER DEFAULT '2' NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (user_id),
    FOREIGN KEY(user_id) REFERENCES users (user_id)
);
"""

# Current shape: no users table, no users FK (what a fresh baseline DB builds).
_CURRENT_SCHEMA = """
CREATE TABLE jobs (
    job_id VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    PRIMARY KEY (job_id)
);
CREATE TABLE user_budgets (
    user_id VARCHAR NOT NULL,
    budget_limit INTEGER DEFAULT '0' NOT NULL,
    PRIMARY KEY (user_id)
);
"""


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0040", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fk_targets(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[2]) for row in conn.execute(f"PRAGMA foreign_key_list({table})")}


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _seed(conn: sqlite3.Connection) -> None:
    conn.executescript(_OLD_SCHEMA)
    conn.execute("INSERT INTO users (user_id, role) VALUES ('alice', 'admin'), ('bob', 'user')")
    # A parent job + a child job (self-FK) + a task child, all owned by alice/bob.
    conn.execute(
        "INSERT INTO jobs (job_id, user_id, parent_job_id, root_job_id, depth, state, "
        "submitted_at_ms, root_submitted_at_ms, num_tasks) VALUES "
        "('/alice/a', 'alice', NULL, '/alice/a', 0, 1, 100, 100, 1), "
        "('/alice/a/child', 'bob', '/alice/a', '/alice/a', 1, 1, 101, 100, 1)"
    )
    conn.execute("INSERT INTO tasks (task_id, job_id) VALUES ('/alice/a/0', '/alice/a')")
    conn.execute("INSERT INTO user_budgets (user_id, budget_limit, max_band, updated_at_ms) VALUES ('alice', 5, 2, 100)")
    conn.commit()


def _connect() -> sqlite3.Connection:
    # Autocommit so the migration's own PRAGMA foreign_keys toggles and explicit
    # BEGIN/commit behave as they do on the controller's raw connection; FK
    # enforcement ON mirrors production (db.py sets PRAGMA foreign_keys=ON).
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def test_migration_0040_drops_users_and_its_fks_preserving_rows():
    conn = _connect()
    _seed(conn)

    _load_migration().migrate(conn)

    # users is gone; jobs / user_budgets no longer reference it.
    assert "users" not in _table_names(conn)
    assert _fk_targets(conn, "jobs") == {"jobs"}  # only the parent_job_id self-FK remains
    assert _fk_targets(conn, "user_budgets") == set()

    # All rows survived, including the child job (self-FK) and the task child.
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2
    assert conn.execute("SELECT user_id FROM jobs WHERE job_id='/alice/a'").fetchone()[0] == "alice"
    assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
    assert conn.execute("SELECT budget_limit FROM user_budgets WHERE user_id='alice'").fetchone()[0] == 5

    # No dangling references, and a job for a user that never had a users row now inserts.
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.execute(
        "INSERT INTO jobs (job_id, user_id, root_job_id, depth, state, submitted_at_ms, "
        "root_submitted_at_ms, num_tasks) VALUES ('/carol/x', 'carol', '/carol/x', 0, 1, 200, 200, 1)"
    )
    # The self-FK still enforces: a child pointing at a missing parent is rejected.
    conn.execute("PRAGMA foreign_keys=ON")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO jobs (job_id, user_id, parent_job_id, root_job_id, depth, state, "
            "submitted_at_ms, root_submitted_at_ms, num_tasks) VALUES "
            "('/carol/orphan', 'carol', '/nope', '/carol/orphan', 1, 1, 201, 200, 1)"
        )
    conn.close()


def test_migration_0040_is_idempotent():
    conn = _connect()
    _seed(conn)
    migration = _load_migration()
    migration.migrate(conn)
    migration.migrate(conn)  # second run must be a no-op, not an error
    assert "users" not in _table_names(conn)
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2
    conn.close()


def test_migration_0040_is_noop_on_current_schema():
    conn = _connect()
    conn.executescript(_CURRENT_SCHEMA)
    conn.commit()
    _load_migration().migrate(conn)
    assert "users" not in _table_names(conn)
    assert _fk_targets(conn, "jobs") == set()
    conn.close()
