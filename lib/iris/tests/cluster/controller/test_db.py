# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ControllerDB transaction, read snapshot, and migration helpers."""

import importlib.util
from pathlib import Path

import pytest
from iris.cluster.controller.db import ControllerDB
from sqlalchemy import text


@pytest.fixture
def db(tmp_path: Path) -> ControllerDB:
    return ControllerDB(db_dir=tmp_path)


# =============================================================================
# Migration 0027_attempt_uid
# =============================================================================

# task_attempts before 0027 added attempt_uid: same columns, PK, FKs, and
# non-unique indexes as schema.py, minus the attempt_uid column and its index.
_PRE_0027_TASK_ATTEMPTS = """
CREATE TABLE task_attempts (
    task_id VARCHAR NOT NULL,
    attempt_id INTEGER NOT NULL,
    worker_id VARCHAR,
    state INTEGER NOT NULL,
    created_at_ms INTEGER NOT NULL,
    started_at_ms INTEGER,
    finished_at_ms INTEGER,
    exit_code INTEGER,
    error VARCHAR,
    PRIMARY KEY (task_id, attempt_id),
    FOREIGN KEY(task_id) REFERENCES tasks (task_id) ON DELETE CASCADE,
    FOREIGN KEY(worker_id) REFERENCES workers (worker_id) ON DELETE SET NULL
)
"""


def _table_info(raw_conn, table: str) -> list[tuple]:
    return raw_conn.execute(f"PRAGMA table_info({table})").fetchall()


def _index_list(raw_conn, table: str) -> list[tuple]:
    return raw_conn.execute(f"PRAGMA index_list({table})").fetchall()


def _revert_to_pre_0027(db: ControllerDB) -> None:
    """Rewind ``task_attempts`` to its pre-0027 shape and unrecord the migration.

    A fresh DB is built from ``schema.py``, which already declares
    ``attempt_uid``. To exercise the real backfill path we rebuild the table
    without that column, then drop the ``0027`` marker so ``apply_migrations``
    treats the migration as pending.
    """
    raw_conn = db._sa_write_engine.raw_connection()
    try:
        raw_conn.execute("PRAGMA foreign_keys=OFF")
        raw_conn.execute("DROP TABLE task_attempts")
        raw_conn.execute(_PRE_0027_TASK_ATTEMPTS)
        raw_conn.execute("PRAGMA foreign_keys=ON")
        raw_conn.execute("DELETE FROM schema_migrations WHERE name = '0027_attempt_uid.py'")
        raw_conn.commit()
    finally:
        raw_conn.close()


def _insert_task_attempt_rows(db: ControllerDB, count: int) -> None:
    """Insert ``count`` task_attempt rows directly (one attempt per synthetic task).

    Migration 0027 only ever reads/writes ``task_attempts``, so the parent
    ``tasks`` / ``jobs`` rows are irrelevant to it. Inserting with foreign keys
    disabled keeps the fixture focused on the migration under test rather than
    on materializing the whole job graph.
    """
    raw_conn = db._sa_write_engine.raw_connection()
    try:
        raw_conn.execute("PRAGMA foreign_keys=OFF")
        for i in range(count):
            raw_conn.execute(
                "INSERT INTO task_attempts (task_id, attempt_id, state, created_at_ms) VALUES (?, ?, ?, ?)",
                (f"/test-user/job-{i}/0", 0, 0, 1000),
            )
        raw_conn.commit()
        raw_conn.execute("PRAGMA foreign_keys=ON")
    finally:
        raw_conn.close()


def _attempt_uids(db: ControllerDB) -> list[str]:
    raw_conn = db._sa_write_engine.raw_connection()
    try:
        return [row[0] for row in raw_conn.execute("SELECT attempt_uid FROM task_attempts").fetchall()]
    finally:
        raw_conn.close()


def test_migration_0027_backfills_unique_uids(tmp_path: Path) -> None:
    """0027 backfills every pre-existing row with a distinct 16-hex attempt_uid."""
    db = ControllerDB(db_dir=tmp_path)
    _revert_to_pre_0027(db)
    _insert_task_attempt_rows(db, count=5)

    db.apply_migrations()

    uids = _attempt_uids(db)
    assert len(uids) == 5
    assert all(uid is not None for uid in uids)
    assert all(len(uid) == 16 for uid in uids)
    assert all(all(c in "0123456789abcdef" for c in uid) for uid in uids)
    assert len(set(uids)) == 5, "every backfilled attempt_uid must be distinct"
    db.close()


def test_migration_0027_backfills_across_chunks(tmp_path: Path) -> None:
    """The chunked backfill loop covers every row when there are more rows than one chunk."""
    db = ControllerDB(db_dir=tmp_path)
    _revert_to_pre_0027(db)
    # BACKFILL_CHUNK is 1000; exceed it so the loop must iterate more than once.
    _insert_task_attempt_rows(db, count=1500)

    db.apply_migrations()

    uids = _attempt_uids(db)
    assert len(uids) == 1500
    assert all(uid is not None and len(uid) == 16 for uid in uids)
    assert len(set(uids)) == 1500, "no row may be left NULL or share a uid across chunks"
    db.close()


def test_migration_0027_promotes_not_null_and_unique_index(tmp_path: Path) -> None:
    """After 0027, attempt_uid is NOT NULL and idx_task_attempts_uid is a unique index."""
    db = ControllerDB(db_dir=tmp_path)
    _revert_to_pre_0027(db)
    _insert_task_attempt_rows(db, count=3)

    db.apply_migrations()

    raw_conn = db._sa_write_engine.raw_connection()
    try:
        # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
        cols = {row[1]: row for row in _table_info(raw_conn, "task_attempts")}
        assert "attempt_uid" in cols
        assert cols["attempt_uid"][3] == 1, "attempt_uid must be NOT NULL after the rebuild"

        # PRAGMA index_list columns: (seq, name, unique, origin, partial)
        indexes = {row[1]: row for row in _index_list(raw_conn, "task_attempts")}
        assert "idx_task_attempts_uid" in indexes
        assert indexes["idx_task_attempts_uid"][2] == 1, "idx_task_attempts_uid must be UNIQUE"

        # The rebuild must preserve the other two indexes from schema.py.
        assert "idx_task_attempts_worker_task" in indexes
        assert "idx_task_attempts_live_workerbound" in indexes

        # The unique index is real: a duplicate attempt_uid is rejected.
        existing = raw_conn.execute("SELECT attempt_uid FROM task_attempts LIMIT 1").fetchone()[0]
        raw_conn.execute("PRAGMA foreign_keys=OFF")
        with pytest.raises(Exception, match="UNIQUE"):
            raw_conn.execute(
                "INSERT INTO task_attempts (task_id, attempt_id, state, created_at_ms, attempt_uid) "
                "VALUES (?, ?, ?, ?, ?)",
                ("/test-user/dup-job/0", 0, 0, 1000, existing),
            )
        raw_conn.rollback()
    finally:
        raw_conn.close()
    db.close()


def test_migration_0027_is_idempotent(tmp_path: Path) -> None:
    """Re-running apply_migrations on an already-migrated DB is a clean no-op."""
    db = ControllerDB(db_dir=tmp_path)
    _revert_to_pre_0027(db)
    _insert_task_attempt_rows(db, count=4)

    db.apply_migrations()
    uids_first = sorted(_attempt_uids(db))

    # Second run: 0027 is already recorded, so the runner skips it entirely;
    # even if it ran again, every step is guarded and would not mutate data.
    db.apply_migrations()
    uids_second = sorted(_attempt_uids(db))

    assert uids_first == uids_second, "a second migration pass must not change attempt_uid values"
    db.close()


def test_migration_0027_runs_unconditionally_idempotent(tmp_path: Path) -> None:
    """0027.migrate() applied directly to an already-migrated DB does not error or mutate.

    The runner skips recorded migrations, but the migration's own guards
    (_has_column / _column_is_notnull / IF NOT EXISTS) must also make a direct
    re-invocation a no-op — that is what makes a crash-and-retry safe.
    """

    db = ControllerDB(db_dir=tmp_path)
    _revert_to_pre_0027(db)
    _insert_task_attempt_rows(db, count=3)
    db.apply_migrations()
    uids_before = sorted(_attempt_uids(db))

    migration_path = Path(__file__).parents[3] / "src/iris/cluster/controller/migrations/0027_attempt_uid.py"
    spec = importlib.util.spec_from_file_location("m0027", migration_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    raw_conn = db._sa_write_engine.raw_connection()
    try:
        module.migrate(raw_conn)
        raw_conn.commit()
    finally:
        raw_conn.close()

    assert sorted(_attempt_uids(db)) == uids_before
    db.close()


def test_fresh_db_schema_matches_0027_end_state(tmp_path: Path) -> None:
    """A DB built fresh from schema.py agrees with 0027's end state.

    schema.py declares attempt_uid NOT NULL with a unique index; the migration
    promotes the same. A fresh DB exercises both (schema create_all, then 0027
    runs as a no-op) and must land on the identical shape.
    """
    db = ControllerDB(db_dir=tmp_path)
    raw_conn = db._sa_write_engine.raw_connection()
    try:
        cols = {row[1]: row for row in _table_info(raw_conn, "task_attempts")}
        assert "attempt_uid" in cols
        assert cols["attempt_uid"][3] == 1, "fresh schema must declare attempt_uid NOT NULL"

        indexes = {row[1]: row for row in _index_list(raw_conn, "task_attempts")}
        assert "idx_task_attempts_uid" in indexes
        assert indexes["idx_task_attempts_uid"][2] == 1, "fresh schema must make idx_task_attempts_uid UNIQUE"

        # 0027 is recorded on a fresh DB — create_all already produced the
        # end state, so the runner ran the (no-op) migration and marked it.
        recorded = {row[0] for row in raw_conn.execute("SELECT name FROM schema_migrations").fetchall()}
        assert "0027_attempt_uid.py" in recorded
    finally:
        raw_conn.close()
    db.close()


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


def test_on_commit_hook_fires(db: ControllerDB) -> None:
    """on_commit alias fires post-commit hooks just like register."""
    _create_simple_table(db)
    calls: list[int] = []

    with db.transaction() as cur:
        cur.execute(text("INSERT INTO kv (key, value) VALUES (:k, :v)"), {"k": "a", "v": "1"})
        cur.on_commit(lambda: calls.append(1))

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


def test_replace_from_reattaches_auth_db(tmp_path: Path) -> None:
    """replace_from() must re-attach the auth DB so auth tables remain accessible."""
    db = ControllerDB(db_dir=tmp_path)

    # Write to an auth table
    with db.transaction() as cur:
        cur.execute(
            text(
                "INSERT INTO auth.api_keys (key_id, key_hash, key_prefix, name, user_id, created_at_ms) "
                "VALUES (:key_id, :key_hash, :key_prefix, :name, :user_id, :created_at_ms)"
            ),
            {
                "key_id": "id1",
                "key_hash": "hash1",
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
        rows = q.execute(text("SELECT name FROM auth.api_keys WHERE key_hash = 'hash1'")).all()
    assert len(rows) == 1
    assert rows[0][0] == "test-key"
    db.close()
