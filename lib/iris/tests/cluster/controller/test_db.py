# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ControllerDB transactions and read snapshots."""

import sqlite3
from functools import partial
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


def test_backup_with_concurrent_commits_preserves_snapshot(db: ControllerDB, tmp_path: Path, monkeypatch) -> None:
    _create_simple_table(db)
    with db.transaction() as tx:
        tx.execute(text("INSERT INTO kv VALUES ('version', 'original')"))
        # Span multiple backup batches so writes land while the copy is in progress.
        tx.execute(
            text("INSERT INTO kv VALUES (:key, zeroblob(65536))"),
            [{"key": str(i)} for i in range(64)],
        )

    commits = 0

    def write_between_batches(status: int, _remaining: int, _total: int) -> None:
        nonlocal commits
        if status == sqlite3.SQLITE_DONE:
            return
        assert commits < 100, "Backup failed to finish under continuous writes"
        with db.transaction() as tx:
            tx.execute(text("UPDATE kv SET value = :value WHERE key = 'version'"), {"value": str(commits)})
        commits += 1

    class WritingConnection(sqlite3.Connection):
        def backup(self, target, **kwargs):
            # Keep real SQLite I/O; inject a committed write between its copy steps.
            super().backup(target, **(kwargs | {"progress": write_between_batches}))

    monkeypatch.setattr(sqlite3, "connect", partial(sqlite3.connect, factory=WritingConnection))
    destination = tmp_path / "backup.sqlite3"
    db.backup_to(destination)

    assert commits > 0
    with db.read_snapshot() as tx:
        assert tx.execute(text("SELECT value FROM kv WHERE key = 'version'")).scalar_one() == str(commits - 1)
    with sqlite3.connect(destination) as restored:
        assert restored.execute("SELECT value FROM kv WHERE key = 'version'").fetchone() == ("original",)
        assert restored.execute("SELECT COUNT(*) FROM kv").fetchone() == (65,)
        assert restored.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert not destination.with_name(destination.name + "-wal").exists()
    assert not destination.with_name(destination.name + "-shm").exists()
