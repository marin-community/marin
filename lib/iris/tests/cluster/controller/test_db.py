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


def test_backup_to_does_not_block_concurrent_writes(tmp_path: Path) -> None:
    """backup_to uses a separate read-only source connection, so writers on
    self._conn must proceed under WAL semantics while the backup runs."""
    db = ControllerDB(db_dir=tmp_path)
    _create_simple_table(db)

    # Seed enough rows that the backup takes at least a few page-copy steps,
    # giving the writer thread a real chance to interleave.
    with db.transaction() as cur:
        for i in range(2000):
            cur.execute("INSERT INTO kv (key, value) VALUES (?, ?)", (f"seed-{i}", "x" * 256))

    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()

    writes_completed = 0
    writer_exc: BaseException | None = None
    stop = threading.Event()

    def writer() -> None:
        nonlocal writes_completed, writer_exc
        try:
            i = 0
            while not stop.is_set():
                with db.transaction() as cur:
                    cur.execute("INSERT INTO kv (key, value) VALUES (?, ?)", (f"live-{i}", "y"))
                writes_completed += 1
                i += 1
        except BaseException as e:
            writer_exc = e

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        db.backup_to(backup_dir / "controller.sqlite3")
    finally:
        stop.set()
        t.join(timeout=5)

    assert writer_exc is None, f"writer crashed: {writer_exc!r}"
    # The writer must have made forward progress during the backup; if the
    # backup path had re-acquired self._lock we'd expect zero writes here.
    assert writes_completed > 0
    db.close()


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
    from iris.cluster.controller.db import get_task_profiles, insert_task_profile
    from rigging.timing import Timestamp

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


def test_auto_vacuum_migrates_legacy_db(tmp_path: Path) -> None:
    """A DB created with the SQLite default (auto_vacuum=0) must migrate to INCREMENTAL."""
    db_dir = tmp_path / "ctrl"
    db_dir.mkdir()
    legacy_path = db_dir / ControllerDB.DB_FILENAME

    # Seed a legacy DB with auto_vacuum disabled and some data to reclaim.
    conn = sqlite3.connect(str(legacy_path))
    conn.execute("PRAGMA auto_vacuum = NONE")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, blob TEXT)")
    conn.executemany("INSERT INTO t (blob) VALUES (?)", [("x" * 4096,) for _ in range(200)])
    conn.commit()
    conn.close()
    assert legacy_path.stat().st_size > 0

    db = ControllerDB(db_dir=db_dir)
    try:
        row = db.fetchone("PRAGMA main.auto_vacuum")
        assert row[0] == 2, f"expected auto_vacuum=INCREMENTAL(2), got {row[0]}"
    finally:
        db.close()


def test_wal_checkpoint_truncate_runs_incremental_vacuum(tmp_path: Path) -> None:
    """After TRUNCATE, freed pages from a large delete should be reclaimed."""
    db = ControllerDB(db_dir=tmp_path)
    try:
        with db.transaction() as cur:
            cur.execute("CREATE TABLE big (id INTEGER PRIMARY KEY, blob TEXT)")
            cur.executemany("INSERT INTO big (blob) VALUES (?)", [("y" * 8192,) for _ in range(500)])

        # Baseline size after population + checkpoint.
        db.wal_checkpoint()
        size_full = db.db_path.stat().st_size

        with db.transaction() as cur:
            cur.execute("DELETE FROM big")

        # TRUNCATE flushes WAL and then reclaims freelist pages; file shrinks.
        db.wal_checkpoint()
        size_after = db.db_path.stat().st_size
        assert size_after < size_full, f"expected shrink: before={size_full} after={size_after}"
    finally:
        db.close()


def test_backfill_attempt_finished_at_migration(tmp_path: Path) -> None:
    """0032 backfills finished_at_ms for orphaned terminal attempts.

    Reproduces the historical bug where a FAILED/WORKER_FAILED attempt whose
    task was retried kept finished_at_ms=NULL. The migration should populate
    it using the next attempt's created_at_ms (and fall back to started_at_ms
    or created_at_ms when no next attempt exists).

    Exercises the migration SQL directly against a minimal schema so it
    doesn't need to negotiate controller triggers / FKs.
    """
    import importlib

    conn = sqlite3.connect(str(tmp_path / "c.sqlite3"))
    conn.execute(
        """
        CREATE TABLE task_attempts (
            task_id TEXT NOT NULL,
            attempt_id INTEGER NOT NULL,
            worker_id TEXT,
            state INTEGER NOT NULL,
            created_at_ms INTEGER NOT NULL,
            started_at_ms INTEGER,
            finished_at_ms INTEGER,
            exit_code INTEGER,
            error TEXT,
            PRIMARY KEY (task_id, attempt_id)
        )
        """
    )
    rows = [
        # task A: FAILED attempt followed by another FAILED attempt.
        # Backfill must use next.created_at_ms (2000), not this row's started_at_ms.
        ("/u/A", 0, 5, 1000, 1100, None),
        ("/u/A", 1, 5, 2000, 2100, 2500),
        # task B: FAILED orphan followed by a still-RUNNING retry.
        # Next exists (created at 4000), so that's the bound — even though the
        # next attempt isn't itself terminal.
        ("/u/B", 0, 5, 3000, 3100, None),
        ("/u/B", 1, 3, 4000, 4100, None),
        # task C: FAILED orphan with no next attempt at all — fall back to
        # started_at_ms (5100).
        ("/u/C", 0, 5, 5000, 5100, None),
        # task D: FAILED orphan, no next attempt, no started_at_ms — fall back
        # to created_at_ms (6000).
        ("/u/D", 0, 5, 6000, None, None),
        # task E: control cases. Already-stamped row must not be rewritten; a
        # non-terminal row must never be touched.
        ("/u/E", 0, 4, 7000, 7100, 7200),
        ("/u/E", 1, 3, 8000, 8100, None),
    ]
    conn.executemany(
        "INSERT INTO task_attempts(task_id, attempt_id, state, created_at_ms, "
        "started_at_ms, finished_at_ms) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    mod = importlib.import_module("iris.cluster.controller.migrations.0032_backfill_attempt_finished_at")
    mod.migrate(conn)
    conn.commit()

    out = {
        (r[0], r[1]): r[2]
        for r in conn.execute("SELECT task_id, attempt_id, finished_at_ms FROM task_attempts").fetchall()
    }
    assert out[("/u/A", 0)] == 2000
    assert out[("/u/A", 1)] == 2500
    assert out[("/u/B", 0)] == 4000
    assert out[("/u/B", 1)] is None
    assert out[("/u/C", 0)] == 5100
    assert out[("/u/D", 0)] == 6000
    assert out[("/u/E", 0)] == 7200
    assert out[("/u/E", 1)] is None
    conn.close()


def test_finalize_orphan_attempts_migration(tmp_path: Path) -> None:
    """0038 finalizes task_attempts orphaned by the cancel_job bug.

    Two orphan classes must be healed: (1) attempt active while task is
    terminal, (2) attempt active but superseded by a newer attempt_id on
    the same task. Healthy active attempts must not be touched.
    """
    import importlib

    conn = sqlite3.connect(str(tmp_path / "c.sqlite3"))
    conn.execute(
        """
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            state INTEGER NOT NULL,
            current_attempt_id INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE task_attempts (
            task_id TEXT NOT NULL,
            attempt_id INTEGER NOT NULL,
            state INTEGER NOT NULL,
            finished_at_ms INTEGER,
            error TEXT,
            PRIMARY KEY (task_id, attempt_id)
        )
        """
    )
    # /u/killed: task was KILLED by cancel_job but attempt is still RUNNING.
    conn.execute("INSERT INTO tasks VALUES ('/u/killed', 6, 0)")  # 6 = KILLED
    conn.execute("INSERT INTO task_attempts VALUES ('/u/killed', 0, 3, NULL, NULL)")  # 3 = RUNNING

    # /u/super: task is RUNNING on attempt 1; attempt 0 was abandoned but never
    # finalized.
    conn.execute("INSERT INTO tasks VALUES ('/u/super', 3, 1)")
    conn.execute("INSERT INTO task_attempts VALUES ('/u/super', 0, 3, NULL, NULL)")  # orphan
    conn.execute("INSERT INTO task_attempts VALUES ('/u/super', 1, 3, NULL, NULL)")  # current

    # /u/healthy: task RUNNING on attempt 0; attempt is current and active.
    # Must not be touched.
    conn.execute("INSERT INTO tasks VALUES ('/u/healthy', 3, 0)")
    conn.execute("INSERT INTO task_attempts VALUES ('/u/healthy', 0, 3, NULL, NULL)")

    # /u/already_done: task succeeded normally — attempt already terminal. The
    # COALESCE(error, ...) clause must not overwrite a NULL error with the
    # reconcile message for rows the migration shouldn't touch at all.
    conn.execute("INSERT INTO tasks VALUES ('/u/already_done', 4, 0)")  # 4 = SUCCEEDED
    conn.execute("INSERT INTO task_attempts VALUES ('/u/already_done', 0, 4, 9999, NULL)")
    conn.commit()

    mod = importlib.import_module("iris.cluster.controller.migrations.0038_finalize_orphan_attempts")
    mod.migrate(conn)
    conn.commit()

    out = {
        (r[0], r[1]): (r[2], r[3], r[4])
        for r in conn.execute("SELECT task_id, attempt_id, state, finished_at_ms, error FROM task_attempts").fetchall()
    }

    # Orphans: PREEMPTED (state 10), finished_at_ms stamped, reason recorded.
    killed_state, killed_finished, killed_error = out[("/u/killed", 0)]
    assert killed_state == 10
    assert killed_finished is not None and killed_finished > 0
    assert "Reconciled" in killed_error

    super_state, super_finished, super_error = out[("/u/super", 0)]
    assert super_state == 10
    assert super_finished is not None and super_finished > 0
    assert "Reconciled" in super_error

    # Live attempt for /u/super and the healthy attempt are untouched.
    assert out[("/u/super", 1)] == (3, None, None)
    assert out[("/u/healthy", 0)] == (3, None, None)

    # Already-terminal row preserved.
    assert out[("/u/already_done", 0)] == (4, 9999, None)

    # Idempotency: rerun is a no-op.
    mod.migrate(conn)
    conn.commit()
    out2 = {
        (r[0], r[1]): (r[2], r[3], r[4])
        for r in conn.execute("SELECT task_id, attempt_id, state, finished_at_ms, error FROM task_attempts").fetchall()
    }
    assert out2 == out
    conn.close()


def test_requeue_split_coscheduled_jobs_migration(tmp_path: Path) -> None:
    """0039 force-requeues coscheduled jobs whose tasks are split across slices.

    Sets up: one coscheduled v5p-64 job J split across two ``md_tpu_name``
    groups (tpu-A workers w0,w1 / tpu-B workers w2,w3) and one healthy
    same-slice coscheduled job K (tpu-C workers w4,w5). Expectation: every
    task of J resets to PENDING with worker cleared, attempts marked
    PREEMPTED, and worker resources are decommitted; K is untouched.
    """
    import importlib

    conn = sqlite3.connect(str(tmp_path / "c.sqlite3"))
    conn.executescript(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            is_reservation_holder INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE job_config (
            job_id TEXT PRIMARY KEY,
            has_coscheduling INTEGER NOT NULL,
            res_cpu_millicores INTEGER NOT NULL,
            res_memory_bytes INTEGER NOT NULL,
            res_device_json TEXT NOT NULL
        );
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            state INTEGER NOT NULL,
            current_attempt_id INTEGER NOT NULL,
            current_worker_id TEXT,
            current_worker_address TEXT,
            error TEXT,
            finished_at_ms INTEGER
        );
        CREATE TABLE task_attempts (
            task_id TEXT NOT NULL,
            attempt_id INTEGER NOT NULL,
            state INTEGER NOT NULL,
            finished_at_ms INTEGER,
            error TEXT,
            PRIMARY KEY (task_id, attempt_id)
        );
        CREATE TABLE workers (
            worker_id TEXT PRIMARY KEY,
            committed_cpu_millicores INTEGER NOT NULL,
            committed_mem_bytes INTEGER NOT NULL,
            committed_gpu INTEGER NOT NULL,
            committed_tpu INTEGER NOT NULL,
            md_tpu_name TEXT NOT NULL DEFAULT ''
        );
        """
    )

    # Coscheduled split job J: 4 tasks across two slices (A: w0,w1 / B: w2,w3).
    conn.execute("INSERT INTO jobs VALUES ('/u/J', 0)")
    conn.execute('INSERT INTO job_config VALUES (\'/u/J\', 1, 1000, 4096, \'{"tpu": {"variant":"v5p-64","count": 4}}\')')
    for i, w in enumerate(("w0", "w1", "w2", "w3")):
        tid = f"/u/J/{i}"
        conn.execute(f"INSERT INTO tasks VALUES ('{tid}', '/u/J', 3, 0, '{w}', '{w}:8080', NULL, NULL)")
        conn.execute(f"INSERT INTO task_attempts VALUES ('{tid}', 0, 3, NULL, NULL)")

    # Coscheduled healthy job K: 2 tasks on the same slice C (w4, w5).
    conn.execute("INSERT INTO jobs VALUES ('/u/K', 0)")
    conn.execute('INSERT INTO job_config VALUES (\'/u/K\', 1, 500, 2048, \'{"tpu": {"variant":"v5p-16","count": 4}}\')')
    for i, w in enumerate(("w4", "w5")):
        tid = f"/u/K/{i}"
        conn.execute(f"INSERT INTO tasks VALUES ('{tid}', '/u/K', 3, 0, '{w}', '{w}:8080', NULL, NULL)")
        conn.execute(f"INSERT INTO task_attempts VALUES ('{tid}', 0, 3, NULL, NULL)")

    # Workers: each w0..w5 has the J / K task's resources committed.
    workers = [
        ("w0", 1000, 4096, 4, "tpu-A"),
        ("w1", 1000, 4096, 4, "tpu-A"),
        ("w2", 1000, 4096, 4, "tpu-B"),
        ("w3", 1000, 4096, 4, "tpu-B"),
        ("w4", 500, 2048, 4, "tpu-C"),
        ("w5", 500, 2048, 4, "tpu-C"),
    ]
    for wid, cpu, mem, tpu, name in workers:
        conn.execute(f"INSERT INTO workers VALUES ('{wid}', {cpu}, {mem}, 0, {tpu}, '{name}')")
    conn.commit()

    mod = importlib.import_module("iris.cluster.controller.migrations.0039_requeue_split_coscheduled_jobs")
    mod.migrate(conn)
    conn.commit()

    # J: every task back to PENDING, worker cleared, attempt PREEMPTED.
    j_tasks = {
        r[0]: (r[1], r[2], r[3], r[4])
        for r in conn.execute(
            "SELECT task_id, state, current_worker_id, current_worker_address, finished_at_ms FROM tasks "
            "WHERE job_id = '/u/J'"
        ).fetchall()
    }
    for tid, (state, worker, addr, finished) in j_tasks.items():
        assert state == 1, f"{tid} state={state}, expected PENDING"
        assert worker is None, f"{tid} worker not cleared: {worker}"
        assert addr is None
        assert finished is None
    j_attempts = {
        (r[0], r[1]): (r[2], r[3], r[4])
        for r in conn.execute(
            "SELECT task_id, attempt_id, state, finished_at_ms, error FROM task_attempts " "WHERE task_id LIKE '/u/J/%'"
        ).fetchall()
    }
    for key, (state, finished, error) in j_attempts.items():
        assert state == 10, f"{key} attempt state={state}, expected PREEMPTED"
        assert finished is not None and finished > 0
        assert "Reconciled" in error and "split-slice" in error

    # J's workers fully decommitted.
    for wid in ("w0", "w1", "w2", "w3"):
        cpu, mem, tpu = conn.execute(
            "SELECT committed_cpu_millicores, committed_mem_bytes, committed_tpu FROM workers WHERE worker_id = ?",
            (wid,),
        ).fetchone()
        assert (cpu, mem, tpu) == (0, 0, 0), f"{wid} not fully decommitted: cpu={cpu} mem={mem} tpu={tpu}"

    # K: completely untouched.
    k_tasks = list(
        conn.execute("SELECT state, current_worker_id, finished_at_ms FROM tasks WHERE job_id = '/u/K'").fetchall()
    )
    for state, worker, finished in k_tasks:
        assert state == 3 and worker is not None and finished is None
    k_attempts = list(
        conn.execute("SELECT state, finished_at_ms FROM task_attempts WHERE task_id LIKE '/u/K/%'").fetchall()
    )
    for state, finished in k_attempts:
        assert state == 3 and finished is None
    for wid, want_cpu, want_mem, want_tpu, _ in workers[4:]:
        cpu, mem, tpu = conn.execute(
            "SELECT committed_cpu_millicores, committed_mem_bytes, committed_tpu FROM workers WHERE worker_id = ?",
            (wid,),
        ).fetchone()
        assert (cpu, mem, tpu) == (want_cpu, want_mem, want_tpu), f"{wid} state mutated"

    # Idempotent rerun: no-op once split jobs are healed.
    snap = list(conn.execute("SELECT task_id, state, current_worker_id FROM tasks").fetchall())
    mod.migrate(conn)
    conn.commit()
    snap2 = list(conn.execute("SELECT task_id, state, current_worker_id FROM tasks").fetchall())
    assert snap == snap2
    conn.close()


def test_drop_resource_history_tables_migration(tmp_path: Path) -> None:
    """0040 drops worker_resource_history, task_resource_history, and the
    cached snapshot_* columns on workers.

    Builds a minimal v0039-shape DB (just the surfaces this migration
    touches), populates each, runs the migration, and asserts the tables
    and columns are gone. A second invocation must be a no-op.
    """
    import importlib

    conn = sqlite3.connect(str(tmp_path / "c.sqlite3"))
    conn.executescript(
        """
        CREATE TABLE workers (
            worker_id TEXT PRIMARY KEY,
            address TEXT NOT NULL,
            snapshot_host_cpu_percent INTEGER,
            snapshot_memory_used_bytes INTEGER,
            snapshot_memory_total_bytes INTEGER,
            snapshot_disk_used_bytes INTEGER,
            snapshot_disk_total_bytes INTEGER,
            snapshot_running_task_count INTEGER,
            snapshot_total_process_count INTEGER,
            snapshot_net_recv_bps INTEGER,
            snapshot_net_sent_bps INTEGER
        );
        CREATE TABLE worker_resource_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT NOT NULL,
            snapshot_host_cpu_percent INTEGER,
            timestamp_ms INTEGER NOT NULL
        );
        CREATE INDEX idx_worker_resource_history_worker
            ON worker_resource_history(worker_id, id DESC);
        CREATE INDEX idx_worker_resource_history_ts
            ON worker_resource_history(worker_id, timestamp_ms DESC);
        CREATE TABLE task_resource_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            attempt_id INTEGER NOT NULL,
            cpu_millicores INTEGER NOT NULL DEFAULT 0,
            timestamp_ms INTEGER NOT NULL
        );
        CREATE INDEX idx_task_resource_history_task_attempt
            ON task_resource_history(task_id, attempt_id, id DESC);
        """
    )
    conn.execute("INSERT INTO workers VALUES ('w1','host:1', 50, 1, 2, 3, 4, 5, 6, 7, 8)")
    conn.execute(
        "INSERT INTO worker_resource_history (worker_id, snapshot_host_cpu_percent, timestamp_ms) "
        "VALUES ('w1', 42, 1000)"
    )
    conn.execute(
        "INSERT INTO task_resource_history (task_id, attempt_id, cpu_millicores, timestamp_ms) "
        "VALUES ('/u/t', 0, 100, 1000)"
    )
    conn.commit()

    mod = importlib.import_module("iris.cluster.controller.migrations.0040_drop_resource_history_tables")
    mod.migrate(conn)
    conn.commit()

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE '%resource_history'"
        ).fetchall()
    }
    assert tables == set(), f"resource_history tables still present: {tables}"

    cols = {row[1] for row in conn.execute("PRAGMA table_info(workers)").fetchall()}
    assert not any(c.startswith("snapshot_") for c in cols), f"snapshot_* still on workers: {sorted(cols)}"

    # Idempotent — re-running on a DB where everything is already gone
    # must be a no-op (no exception).
    mod.migrate(conn)
    conn.commit()
    conn.close()
