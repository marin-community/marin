# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0036_federation_drop_dead_columns``.

Builds a post-0035 DB (unified ``federated_jobs`` plus ``federation_sync_state``,
each carrying the dead columns), seeds a row in each, and asserts the migration
drops exactly ``last_sync_ms``/``terminal_error`` from ``federated_jobs`` and
``last_full_resync_ms`` from ``federation_sync_state`` — preserving every other
column, the surviving rows, and the ``federated_jobs`` index — and is idempotent.
"""

import importlib.util
import sqlite3
from pathlib import Path

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0036_federation_drop_dead_columns.py"

# The pre-0036 (0035-era) shape: the unified federated_jobs and the sync-state
# table, each still carrying the columns this migration removes.
_OLD_SCHEMA = """
CREATE TABLE jobs (job_id VARCHAR PRIMARY KEY, state INTEGER NOT NULL);
CREATE TABLE federated_jobs (
    job_id VARCHAR NOT NULL PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
    direction INTEGER NOT NULL,
    peer_id VARCHAR NOT NULL,
    owner_principal VARCHAR NOT NULL DEFAULT '',
    remote_job_id VARCHAR,
    handoff_state INTEGER,
    cancel_intent_version INTEGER NOT NULL DEFAULT 0,
    last_sync_ms INTEGER,
    terminal_error VARCHAR
);
CREATE INDEX idx_federated_jobs_direction_peer ON federated_jobs (direction, peer_id);
CREATE TABLE federation_sync_state (
    peer_id VARCHAR NOT NULL PRIMARY KEY,
    cursor VARCHAR NOT NULL DEFAULT '',
    last_full_resync_ms INTEGER
);
"""


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0036", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(conn: sqlite3.Connection) -> None:
    conn.executescript(_OLD_SCHEMA)
    conn.execute("INSERT INTO jobs (job_id, state) VALUES ('/alice/train', 1)")
    conn.execute(
        "INSERT INTO federated_jobs "
        "(job_id, direction, peer_id, owner_principal, remote_job_id, handoff_state, "
        "cancel_intent_version, last_sync_ms, terminal_error) "
        "VALUES ('/alice/train', 0, 'cw', 'alice', '/alice/sf~train', 1, 3, 12345, 'boom')"
    )
    conn.execute("INSERT INTO federation_sync_state (peer_id, cursor, last_full_resync_ms) VALUES ('cw', 'c-7', 99)")
    conn.commit()


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _indexes(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}


def test_migration_0036_drops_dead_columns_and_preserves_the_rest():
    conn = sqlite3.connect(":memory:")
    _seed(conn)

    _load_migration().migrate(conn)

    fed_cols = _columns(conn, "federated_jobs")
    assert {"last_sync_ms", "terminal_error"}.isdisjoint(fed_cols)
    assert {"job_id", "direction", "peer_id", "owner_principal", "remote_job_id", "handoff_state"} <= fed_cols
    assert "idx_federated_jobs_direction_peer" in _indexes(conn)
    assert conn.execute(
        "SELECT direction, peer_id, remote_job_id, handoff_state, cancel_intent_version "
        "FROM federated_jobs WHERE job_id='/alice/train'"
    ).fetchone() == (0, "cw", "/alice/sf~train", 1, 3)

    sync_cols = _columns(conn, "federation_sync_state")
    assert "last_full_resync_ms" not in sync_cols
    assert conn.execute("SELECT peer_id, cursor FROM federation_sync_state WHERE peer_id='cw'").fetchone() == (
        "cw",
        "c-7",
    )
    conn.close()


def test_migration_0036_is_idempotent():
    conn = sqlite3.connect(":memory:")
    _seed(conn)

    migration = _load_migration()
    migration.migrate(conn)
    # A second run finds the columns already gone and leaves the rows untouched.
    migration.migrate(conn)

    assert "last_sync_ms" not in _columns(conn, "federated_jobs")
    assert conn.execute("SELECT peer_id FROM federated_jobs WHERE job_id='/alice/train'").fetchone() == ("cw",)
    conn.close()
