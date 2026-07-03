# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for migration ``0035_federation_unify``.

Builds a pre-unification DB (the 0034 ``federated_jobs`` shape, SENT-only, with
``remote_job_id``/``handoff_state`` ``NOT NULL`` and no ``direction``), seeds an
outbound handoff row, and asserts the migration: rebuilds ``federated_jobs`` into
the unified shape (adds ``direction``, relaxes the SENT-only columns to nullable,
tags existing rows ``SENT``), creates the ``federation_changelog`` table with its
``requester_id`` index, and is idempotent on re-run.
"""

import importlib.util
import sqlite3
from pathlib import Path

_MIGRATION = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0035_federation_unify.py"

# The pre-0035 (0034-era) federation schema: federated_jobs is SENT-only with the
# handoff columns NOT NULL, and there is no changelog table yet.
_OLD_SCHEMA = """
CREATE TABLE jobs (job_id VARCHAR PRIMARY KEY, state INTEGER NOT NULL);
CREATE TABLE federated_jobs (
    job_id VARCHAR NOT NULL PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
    peer_id VARCHAR NOT NULL,
    remote_job_id VARCHAR NOT NULL,
    owner_principal VARCHAR NOT NULL,
    handoff_state INTEGER NOT NULL,
    spend_snapshot_micros INTEGER NOT NULL DEFAULT 0,
    cancel_intent_version INTEGER NOT NULL DEFAULT 0,
    last_sync_ms INTEGER,
    terminal_error VARCHAR
);
"""


def _load_migration():
    spec = importlib.util.spec_from_file_location("m0035", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(conn: sqlite3.Connection) -> None:
    conn.executescript(_OLD_SCHEMA)
    conn.execute("INSERT INTO jobs (job_id, state) VALUES ('/alice/train', 1)")
    conn.execute(
        "INSERT INTO federated_jobs "
        "(job_id, peer_id, remote_job_id, owner_principal, handoff_state) "
        "VALUES ('/alice/train', 'cw', '/alice/sf~train', 'alice', 2)"
    )
    conn.commit()


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _indexes(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}


def test_migration_0035_unifies_federated_jobs_and_adds_changelog():
    conn = sqlite3.connect(":memory:")
    _seed(conn)

    migration = _load_migration()
    migration.migrate(conn)

    # federated_jobs gains the direction discriminator and its index; the seeded
    # outbound handoff becomes a SENT (0) row with its columns preserved.
    assert "direction" in _columns(conn, "federated_jobs")
    assert "idx_federated_jobs_direction_peer" in _indexes(conn)
    row = conn.execute(
        "SELECT direction, peer_id, remote_job_id, handoff_state FROM federated_jobs WHERE job_id='/alice/train'"
    ).fetchone()
    assert row == (0, "cw", "/alice/sf~train", 2)

    # The SENT-only columns are now nullable — a RECEIVED (1) row carries neither a
    # remote_job_id nor a handoff_state.
    conn.execute("INSERT INTO jobs (job_id, state) VALUES ('/bob/eval', 1)")
    conn.execute("INSERT INTO federated_jobs (job_id, direction, peer_id) VALUES ('/bob/eval', 1, 'requester-cluster')")
    conn.commit()
    received = conn.execute(
        "SELECT remote_job_id, handoff_state FROM federated_jobs WHERE job_id='/bob/eval'"
    ).fetchone()
    assert received == (None, None)

    # The changelog table exists, keyed for per-requester paging.
    assert "requester_id" in _columns(conn, "federation_changelog")
    assert "idx_federation_changelog_requester" in _indexes(conn)


def test_migration_0035_is_idempotent():
    conn = sqlite3.connect(":memory:")
    _seed(conn)

    migration = _load_migration()
    migration.migrate(conn)
    # A second run must not rebuild the already-unified table or drop its rows.
    migration.migrate(conn)

    row = conn.execute("SELECT direction, peer_id FROM federated_jobs WHERE job_id='/alice/train'").fetchone()
    assert row == (0, "cw")
    assert "requester_id" in _columns(conn, "federation_changelog")
    conn.close()
