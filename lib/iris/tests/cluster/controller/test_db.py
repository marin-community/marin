# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TransactionCursor escape-hatch methods and read pool in db.py."""

import sqlite3
import threading
from pathlib import Path

import pytest
from iris.cluster.controller.db import (
    ControllerDB,
    Row,
    TransactionCursor,
)


@pytest.fixture
def db(tmp_path: Path) -> ControllerDB:
    return ControllerDB(db_dir=tmp_path)


def _create_simple_table(db: ControllerDB) -> None:
    """Create a simple key/value table for testing mutation helpers."""
    with db.transaction() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)")


def test_transaction_yields_transaction_cursor(db: ControllerDB) -> None:
    _create_simple_table(db)
    with db.transaction() as cur:
        assert isinstance(cur, TransactionCursor)


def test_execute_escape_hatch(db: ControllerDB) -> None:
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.execute("INSERT INTO kv (key, value) VALUES (?, ?)", ("raw_key", "raw_val"))

    rows = db.fetchall("SELECT key FROM kv")
    assert rows[0]["key"] == "raw_key"


def test_executemany_escape_hatch(db: ControllerDB) -> None:
    _create_simple_table(db)
    data = [("em1", "v1"), ("em2", "v2"), ("em3", "v3")]
    with db.transaction() as cur:
        cur.executemany("INSERT INTO kv (key, value) VALUES (?, ?)", data)

    rows = db.fetchall("SELECT key FROM kv ORDER BY key")
    assert [r["key"] for r in rows] == ["em1", "em2", "em3"]


def test_transaction_rollback_on_exception(db: ControllerDB) -> None:
    _create_simple_table(db)
    with pytest.raises(ValueError):
        with db.transaction() as cur:
            cur.execute("INSERT INTO kv (key, value) VALUES (?, ?)", ("should_not_persist", "v"))
            raise ValueError("abort")

    rows = db.fetchall("SELECT key FROM kv")
    assert len(rows) == 0


def test_lastrowid_property(db: ControllerDB) -> None:
    """lastrowid is forwarded from the underlying cursor."""
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.execute("INSERT INTO kv (key, value) VALUES (?, ?)", ("lri", "v"))
        assert cur.lastrowid is not None
        assert cur.lastrowid > 0


def test_raw_group_by_query(db: ControllerDB) -> None:
    """raw() executes arbitrary SQL and returns Row objects with attribute access."""
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.execute("INSERT INTO kv (key, value) VALUES (?, ?)", ("a", "x"))
        cur.execute("INSERT INTO kv (key, value) VALUES (?, ?)", ("b", "x"))
        cur.execute("INSERT INTO kv (key, value) VALUES (?, ?)", ("c", "y"))

    with db.snapshot() as snap:
        rows = snap.raw(
            "SELECT value, COUNT(*) AS cnt FROM kv GROUP BY value ORDER BY value",
        )

    assert len(rows) == 2
    assert all(isinstance(r, Row) for r in rows)
    assert rows[0].value == "x"
    assert rows[0].cnt == 2
    assert rows[1].value == "y"


def test_worker_scheduling_columns_exist_after_migrations(db: ControllerDB) -> None:
    columns = {row[1] for row in db._conn.execute("PRAGMA table_info(workers)").fetchall()}
    assert "total_cpu_millicores" in columns
    assert "total_memory_bytes" in columns
    assert "total_gpu_count" in columns
    assert "total_tpu_count" in columns
    assert "device_type" in columns
    assert "device_variant" in columns


def test_job_scheduling_columns_exist_after_migrations(db: ControllerDB) -> None:
    columns = {row[1] for row in db._conn.execute("PRAGMA table_info(job_config)").fetchall()}
    assert "res_cpu_millicores" in columns
    assert "res_memory_bytes" in columns
    assert "res_disk_bytes" in columns
    assert "res_device_json" in columns
    assert "constraints_json" in columns
    assert "has_coscheduling" in columns
    assert "coscheduling_group_by" in columns
    assert "scheduling_timeout_ms" in columns
    assert "max_task_failures" in columns


def test_task_assignment_columns_exist_after_migrations(db: ControllerDB) -> None:
    columns = {row[1] for row in db._conn.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "current_worker_id" in columns
    assert "current_worker_address" in columns


def test_raw_with_decoder(db: ControllerDB) -> None:
    """raw() applies per-column decoders to matching columns."""
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.execute("INSERT INTO kv (key, value) VALUES (?, ?)", ("k1", "hello"))

    with db.snapshot() as snap:
        rows = snap.raw(
            "SELECT key, value FROM kv",
            decoders={"value": str.upper},
        )

    assert len(rows) == 1
    assert rows[0].key == "k1"
    assert rows[0].value == "HELLO"


def test_raw_attribute_error_on_missing_column(db: ControllerDB) -> None:
    """Row raises AttributeError when accessing a non-existent column."""
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.execute("INSERT INTO kv (key, value) VALUES (?, ?)", ("k", "v"))

    with db.snapshot() as snap:
        rows = snap.raw("SELECT key FROM kv")

    assert len(rows) == 1
    with pytest.raises(AttributeError, match="no column"):
        _ = rows[0].nonexistent


def test_read_snapshot_does_not_block_write(db: ControllerDB) -> None:
    """read_snapshot() uses a separate connection, so a concurrent write transaction proceeds."""
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.execute("INSERT INTO kv (key, value) VALUES (?, ?)", ("init", "v"))

    results: dict[str, bool] = {}

    def writer() -> None:
        """Hold the write lock for a short time, recording success."""
        with db.transaction() as cur:
            cur.execute("INSERT INTO kv (key, value) VALUES (?, ?)", ("from_writer", "w"))
        results["writer_done"] = True

    # Hold a read_snapshot open while a writer thread runs.
    with db.read_snapshot() as q:
        rows_before = q.raw("SELECT key FROM kv")
        t = threading.Thread(target=writer)
        t.start()
        t.join(timeout=5)
        assert not t.is_alive(), "writer should not block on read_snapshot"
        results["reader_saw"] = len(rows_before)

    assert results["writer_done"] is True
    assert results["reader_saw"] == 1


def test_read_snapshot_returns_consistent_data(db: ControllerDB) -> None:
    """Changes committed after BEGIN in read_snapshot are not visible within that snapshot."""
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.execute("INSERT INTO kv (key, value) VALUES (?, ?)", ("a", "1"))

    with db.read_snapshot() as q:
        rows_start = q.raw("SELECT key FROM kv")
        assert len(rows_start) == 1

        # Commit a new row from outside the snapshot.
        with db.transaction() as cur:
            cur.execute("INSERT INTO kv (key, value) VALUES (?, ?)", ("b", "2"))

        # The snapshot should still only see the original row.
        rows_after = q.raw("SELECT key FROM kv")
        assert len(rows_after) == 1

    # Outside the snapshot, both rows are visible.
    all_rows = db.fetchall("SELECT key FROM kv ORDER BY key")
    assert len(all_rows) == 2


def test_read_snapshot_pool_returns_connections(db: ControllerDB) -> None:
    """Connections are returned to the pool after read_snapshot exits."""
    _create_simple_table(db)
    pool_size = db._READ_POOL_SIZE

    for _i in range(pool_size * 2):
        with db.read_snapshot() as q:
            q.raw("SELECT 1")

    assert db._read_pool.qsize() == pool_size


def test_replace_from_reattaches_auth_db(tmp_path: Path) -> None:
    """replace_from() must re-attach the auth DB so auth tables remain accessible."""
    db = ControllerDB(db_dir=tmp_path)

    # Write to an auth table
    db.execute(
        "INSERT INTO auth.api_keys (key_id, key_hash, key_prefix, name, user_id, created_at_ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("id1", "hash1", "pfx", "test-key", "user1", 1000),
    )

    # Create a copy of the DB to replace from (replace_from expects a directory)
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    db.backup_to(backup_dir / "controller.sqlite3")

    db.replace_from(str(backup_dir))

    # Auth tables should still be accessible after replace_from
    rows = db.fetchall("SELECT name FROM auth.api_keys WHERE key_hash = 'hash1'")
    assert len(rows) == 1
    assert rows[0]["name"] == "test-key"
    db.close()


def test_replace_from_reattaches_profiles_db(tmp_path: Path) -> None:
    """replace_from() must re-attach the profiles DB so profile tables remain accessible."""
    from rigging.timing import Timestamp

    from iris.cluster.controller.db import get_task_profiles, insert_task_profile

    db = ControllerDB(db_dir=tmp_path)
    insert_task_profile(db, "task-1", b"profile-data", Timestamp.now())

    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    db.backup_to(backup_dir / "controller.sqlite3")

    # The profiles DB is a separate WAL-mode file; export a standalone backup
    # so replace_from can restore from a self-contained sqlite file.
    profiles_backup = backup_dir / ControllerDB.PROFILES_DB_FILENAME
    src = sqlite3.connect(str(db.profiles_db_path))
    dest = sqlite3.connect(str(profiles_backup))
    try:
        src.backup(dest)
        dest.execute("PRAGMA journal_mode = DELETE")
        dest.commit()
    finally:
        src.close()
        dest.close()

    db.replace_from(str(backup_dir))

    profiles = get_task_profiles(db, "task-1")
    assert len(profiles) == 1
    db.close()


def test_replace_from_replaces_db_with_live_wal_sidecars_present(tmp_path: Path) -> None:
    """replace_from() must discard stale sidecars from the live DB before reopening."""
    db = ControllerDB(db_dir=tmp_path)

    # Leave main DB WAL/shm sidecars behind on the live path.
    with db.transaction() as cur:
        cur.execute("INSERT INTO meta(key, value) VALUES (?, ?)", ("live-key", 1))

    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    db.backup_to(backup_dir / "controller.sqlite3")

    assert db.db_path.with_name(f"{db.db_path.name}-wal").exists()
    assert db.db_path.with_name(f"{db.db_path.name}-shm").exists()

    db.replace_from(str(backup_dir))

    rows = db.fetchall("SELECT value FROM meta WHERE key = ?", ("live-key",))
    assert len(rows) == 1
    assert rows[0]["value"] == 1
    db.close()


def test_migration_with_dml_does_not_leave_open_transaction(tmp_path: Path) -> None:
    """Migrations that issue DML (e.g. UPDATE) must not leave an implicit
    transaction open, which would cause the subsequent BEGIN IMMEDIATE for
    schema_migrations to fail."""
    # ControllerDB.__init__ already runs apply_migrations which applies all
    # standard migrations. Simulate adding a new migration with DML by
    # directly exercising the commit-after-migrate pattern on the raw conn.
    db = ControllerDB(db_dir=tmp_path)

    # Insert a row so the UPDATE below has something to hit
    with db.transaction() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS dml_test (id INTEGER PRIMARY KEY, val TEXT)")
        cur.execute("INSERT INTO dml_test (id, val) VALUES (?, ?)", (1, "hello"))

    # Simulate what a migration's migrate(conn) does: DML on the raw conn
    # which opens an implicit transaction.
    db._conn.execute("UPDATE dml_test SET val = 'world' WHERE id = 1")

    # Commit the implicit transaction (this is what apply_migrations does).
    db._conn.commit()

    # This would fail with "cannot start a transaction within a transaction"
    # if the commit above were missing.
    with db.transaction() as cur:
        cur.execute("INSERT INTO dml_test (id, val) VALUES (?, ?)", (2, "after_commit"))

    rows = db.fetchall("SELECT id, val FROM dml_test ORDER BY id")
    assert len(rows) == 2
    assert rows[0]["val"] == "world"
    assert rows[1]["val"] == "after_commit"
    db.close()
