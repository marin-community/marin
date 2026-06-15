# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ControllerDB transaction, read snapshot, and replace_from behavior."""

from pathlib import Path

import pytest
from iris.cluster.controller.db import ControllerDB
from sqlalchemy import text


@pytest.fixture
def db(tmp_path: Path) -> ControllerDB:
    return ControllerDB(db_dir=tmp_path)


def _create_simple_table(db: ControllerDB) -> None:
    """Create a simple key/value table for testing mutation helpers."""
    with db.transaction() as cur:
        cur.execute(text("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)"))


def test_transaction_rollback_on_exception(db: ControllerDB) -> None:
    _create_simple_table(db)
    with pytest.raises(ValueError):
        with db.transaction() as cur:
            cur.execute(text("INSERT INTO kv (key, value) VALUES (:k, :v)"), {"k": "should_not_persist", "v": "v"})
            raise ValueError("abort")

    with db.read_snapshot() as q:
        rows = q.execute(text("SELECT key FROM kv")).all()
    assert len(rows) == 0


def test_register_hook_fires(db: ControllerDB) -> None:
    """register fires post-commit hooks after the surrounding commit."""
    _create_simple_table(db)
    calls: list[int] = []

    with db.transaction() as cur:
        cur.execute(text("INSERT INTO kv (key, value) VALUES (:k, :v)"), {"k": "a", "v": "1"})
        cur.register(lambda: calls.append(1))

    assert calls == [1]


def test_read_snapshot_returns_consistent_data(db: ControllerDB) -> None:
    """Changes committed after BEGIN in read_snapshot are not visible within that snapshot."""
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.execute(text("INSERT INTO kv (key, value) VALUES (:k, :v)"), {"k": "a", "v": "1"})

    with db.read_snapshot() as q:
        rows_start = q.execute(text("SELECT key FROM kv")).all()
        assert len(rows_start) == 1

        # Commit a new row from outside the snapshot.
        with db.transaction() as cur:
            cur.execute(text("INSERT INTO kv (key, value) VALUES (:k, :v)"), {"k": "b", "v": "2"})

        # The snapshot should still only see the original row.
        rows_after = q.execute(text("SELECT key FROM kv")).all()
        assert len(rows_after) == 1

    # Outside the snapshot, both rows are visible.
    with db.read_snapshot() as q:
        all_rows = q.execute(text("SELECT key FROM kv ORDER BY key")).all()
    assert len(all_rows) == 2


def test_control_read_snapshot_reads_committed_data_via_dedicated_pool(db: ControllerDB) -> None:
    """control_read_snapshot reads committed data from its own (non-shared) pool."""
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.execute(text("INSERT INTO kv (key, value) VALUES (:k, :v)"), {"k": "a", "v": "1"})

    # Backed by a distinct engine so the control loop never shares connections
    # with RPC-handler reads.
    assert db.sa_control_read_engine is not db.sa_read_engine

    with db.control_read_snapshot() as q:
        rows = q.execute(text("SELECT key FROM kv")).all()
    assert [r[0] for r in rows] == ["a"]


def test_control_read_snapshot_is_read_only(db: ControllerDB) -> None:
    """The control pool pins query_only, so writes through it are rejected."""
    _create_simple_table(db)
    with pytest.raises(Exception, match=r"readonly|read-only|query_only"):
        with db.control_read_snapshot() as q:
            q.execute(text("INSERT INTO kv (key, value) VALUES (:k, :v)"), {"k": "x", "v": "1"})


def test_replace_from_reattaches_auth_db(tmp_path: Path) -> None:
    """replace_from() must re-attach the auth DB so auth tables remain accessible."""
    db = ControllerDB(db_dir=tmp_path)

    # Write to an auth table
    with db.transaction() as cur:
        cur.execute(
            text(
                "INSERT INTO auth.api_keys (key_id, key_prefix, name, user_id, created_at_ms) "
                "VALUES (:key_id, :key_prefix, :name, :user_id, :created_at_ms)"
            ),
            {
                "key_id": "id1",
                "key_prefix": "pfx",
                "name": "test-key",
                "user_id": "user1",
                "created_at_ms": 1000,
            },
        )

    # Create a copy of the DB to replace from (replace_from expects a directory)
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    db.backup_to(backup_dir / "controller.sqlite3")

    db.replace_from(str(backup_dir))

    # Auth tables should still be accessible after replace_from via the write connection.
    with db.transaction() as q:
        rows = q.execute(text("SELECT name FROM auth.api_keys WHERE key_id = 'id1'")).all()
    assert len(rows) == 1
    assert rows[0][0] == "test-key"
    db.close()
