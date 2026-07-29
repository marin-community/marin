# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0049_resolve_priority_band``."""

import importlib.util
import sqlite3
from pathlib import Path

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0049_resolve_priority_band.py"

_PRODUCTION = 1
_INTERACTIVE = 2
_BATCH = 3

_PENDING = 1
_RUNNING = 3

_SCHEMA = """
CREATE TABLE jobs (job_id VARCHAR PRIMARY KEY, parent_job_id VARCHAR);
CREATE TABLE job_config (job_id VARCHAR PRIMARY KEY, priority_band INTEGER NOT NULL DEFAULT 0);
CREATE TABLE tasks (
    task_id VARCHAR PRIMARY KEY,
    job_id VARCHAR NOT NULL,
    state INTEGER NOT NULL,
    priority_band INTEGER NOT NULL DEFAULT 2
);
"""


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0049", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _add_job(conn: sqlite3.Connection, job_id: str, parent: str | None, band: int) -> None:
    conn.execute("INSERT INTO jobs (job_id, parent_job_id) VALUES (?, ?)", (job_id, parent))
    conn.execute("INSERT INTO job_config (job_id, priority_band) VALUES (?, ?)", (job_id, band))


def _add_task(conn: sqlite3.Connection, task_id: str, job_id: str, state: int, band: int) -> None:
    conn.execute(
        "INSERT INTO tasks (task_id, job_id, state, priority_band) VALUES (?, ?, ?, ?)",
        (task_id, job_id, state, band),
    )


def _bands(conn: sqlite3.Connection) -> dict[str, int]:
    return dict(conn.execute("SELECT job_id, priority_band FROM job_config"))


def _task_bands(conn: sqlite3.Connection) -> dict[str, int]:
    return dict(conn.execute("SELECT task_id, priority_band FROM tasks"))


def _tree() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA)
    _add_job(conn, "/u/batch", None, _BATCH)
    _add_job(conn, "/u/batch/child", "/u/batch", 0)
    _add_job(conn, "/u/batch/child/grand", "/u/batch/child", 0)
    _add_job(conn, "/u/prod", None, _PRODUCTION)
    _add_job(conn, "/u/prod/child", "/u/prod", 0)
    _add_job(conn, "/u/plain", None, 0)
    conn.commit()
    return conn


def test_migration_0049_resolves_job_bands_up_the_parent_chain():
    conn = _tree()

    _load_migration().migrate(conn)
    conn.commit()

    assert _bands(conn) == {
        "/u/batch": _BATCH,
        "/u/batch/child": _BATCH,
        # Two levels up — the chain walk does not stop at the zero parent.
        "/u/batch/child/grand": _BATCH,
        "/u/prod": _PRODUCTION,
        "/u/prod/child": _PRODUCTION,
        "/u/plain": _INTERACTIVE,
    }
    conn.close()


def test_migration_0049_restamps_pending_tasks_but_not_assigned_ones():
    conn = _tree()
    # Pre-migration submit stamped a flat INTERACTIVE default on every child's tasks.
    _add_task(conn, "/u/batch/child/0", "/u/batch/child", _PENDING, _INTERACTIVE)
    # A running task's band was stamped by the scheduler (here, a budget downgrade to
    # BATCH under a PRODUCTION job); the migration must not overwrite it.
    _add_task(conn, "/u/prod/child/0", "/u/prod/child", _RUNNING, _BATCH)
    conn.commit()

    _load_migration().migrate(conn)
    conn.commit()

    assert _task_bands(conn) == {"/u/batch/child/0": _BATCH, "/u/prod/child/0": _BATCH}
    conn.close()


def test_migration_0049_is_idempotent_on_repeat():
    conn = _tree()
    _add_task(conn, "/u/batch/child/0", "/u/batch/child", _PENDING, _INTERACTIVE)
    conn.commit()

    migration = _load_migration()
    migration.migrate(conn)
    conn.commit()
    after_first = (_bands(conn), _task_bands(conn))
    migration.migrate(conn)
    conn.commit()

    assert (_bands(conn), _task_bands(conn)) == after_first
    conn.close()
