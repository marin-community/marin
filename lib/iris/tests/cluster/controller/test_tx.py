# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the v2 ``Tx``, ``write_transaction`` and ``read_snapshot``.

The race tests verify the central atomicity contract: post-commit hooks
registered via ``Tx.register`` fire while the write lock is still held,
so a second thread cannot observe the SQL-committed-but-cache-not-yet-
updated window. This matches today's ``ControllerDB.transaction()``
behavior at ``db.py:476``.
"""

import sqlite3
import threading
import time
from pathlib import Path

import pytest
from iris.cluster.controller.db_v2 import (
    _make_engine,
    read_snapshot,
    write_transaction,
)
from sqlalchemy import text
from sqlalchemy.exc import OperationalError


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Empty controller DB at a temp path. Caller may create tables as needed."""
    return tmp_path / "controller.sqlite3"


@pytest.fixture
def engine(db_path: Path):
    eng = _make_engine(db_path, auth_db_path=None)
    yield eng
    eng.dispose()


@pytest.fixture
def write_lock() -> threading.RLock:
    return threading.RLock()


def _create_kv_table(engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE kv (k TEXT PRIMARY KEY, v TEXT)"))


def test_write_hook_fires_on_commit(engine, write_lock):
    _create_kv_table(engine)
    calls: list[int] = []

    with write_transaction(engine, write_lock) as tx:
        tx.execute(text("INSERT INTO kv VALUES ('a', '1')"))
        tx.register(lambda: calls.append(1))

    assert calls == [1]

    with read_snapshot(engine) as tx:
        rows = tx.execute(text("SELECT k, v FROM kv")).all()
    assert rows == [("a", "1")]


def test_write_hook_skipped_on_rollback(engine, write_lock):
    _create_kv_table(engine)
    calls: list[int] = []

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with write_transaction(engine, write_lock) as tx:
            tx.execute(text("INSERT INTO kv VALUES ('a', '1')"))
            tx.register(lambda: calls.append(1))
            raise Boom

    assert calls == []
    with read_snapshot(engine) as tx:
        rows = tx.execute(text("SELECT k, v FROM kv")).all()
    assert rows == []


def test_hook_runs_under_write_lock(engine, write_lock):
    """The atomicity-critical test: lock is held while hooks run."""
    _create_kv_table(engine)
    hook_entered = threading.Event()
    hook_release = threading.Event()
    b_attempt_done = threading.Event()
    b_acquired_blocking = threading.Event()

    def slow_hook() -> None:
        hook_entered.set()
        # Block until thread B has had a chance to try acquire(blocking=False).
        assert hook_release.wait(timeout=5.0)

    def thread_a() -> None:
        with write_transaction(engine, write_lock) as tx:
            tx.execute(text("INSERT INTO kv VALUES ('a', '1')"))
            tx.register(slow_hook)

    b_result: dict[str, bool] = {}

    def thread_b() -> None:
        # Wait until A is inside its hook, then probe the lock.
        assert hook_entered.wait(timeout=5.0)
        b_result["non_blocking_during_hook"] = write_lock.acquire(blocking=False)
        if b_result["non_blocking_during_hook"]:
            # Should not have happened, but release if it did.
            write_lock.release()
        b_attempt_done.set()
        # Now let A finish; afterwards we should be able to acquire.
        hook_release.set()
        acquired = write_lock.acquire(timeout=5.0)
        b_result["blocking_after_hook"] = acquired
        if acquired:
            write_lock.release()
        b_acquired_blocking.set()

    ta = threading.Thread(target=thread_a)
    tb = threading.Thread(target=thread_b)
    ta.start()
    tb.start()
    ta.join(timeout=10.0)
    tb.join(timeout=10.0)
    assert not ta.is_alive()
    assert not tb.is_alive()

    # While the hook was running, the lock was held: non-blocking acquire fails.
    assert b_result["non_blocking_during_hook"] is False
    # After the hook returned and A released the lock, B succeeds.
    assert b_result["blocking_after_hook"] is True


def test_read_snapshot_does_not_block_writes(engine, write_lock):
    _create_kv_table(engine)
    with write_transaction(engine, write_lock) as tx:
        tx.execute(text("INSERT INTO kv VALUES ('seed', '0')"))

    reader_in_snapshot = threading.Event()
    reader_release = threading.Event()
    writer_done = threading.Event()
    writer_elapsed: dict[str, float] = {}

    def reader() -> None:
        with read_snapshot(engine) as tx:
            tx.execute(text("SELECT 1")).all()
            reader_in_snapshot.set()
            assert reader_release.wait(timeout=5.0)

    def writer() -> None:
        # Wait until the reader is inside its snapshot before writing.
        assert reader_in_snapshot.wait(timeout=5.0)
        t0 = time.monotonic()
        with write_transaction(engine, write_lock) as tx:
            tx.execute(text("INSERT INTO kv VALUES ('w', '1')"))
        writer_elapsed["s"] = time.monotonic() - t0
        writer_done.set()

    tr = threading.Thread(target=reader)
    tw = threading.Thread(target=writer)
    tr.start()
    tw.start()
    # Writer should finish well before we tell the reader to release.
    assert writer_done.wait(timeout=3.0)
    reader_release.set()
    tr.join(timeout=5.0)
    tw.join(timeout=5.0)
    assert not tr.is_alive()
    assert not tw.is_alive()
    # Writer wasn't blocked on the reader.
    assert writer_elapsed["s"] < 1.0


def test_pragma_journal_mode_wal(engine):
    with read_snapshot(engine) as tx:
        mode = tx.execute(text("PRAGMA journal_mode")).scalar()
    assert mode == "wal"


def test_pragma_foreign_keys_on(engine):
    with read_snapshot(engine) as tx:
        fk = tx.execute(text("PRAGMA foreign_keys")).scalar()
    assert fk == 1


def test_pragma_busy_timeout(engine):
    with read_snapshot(engine) as tx:
        bt = tx.execute(text("PRAGMA busy_timeout")).scalar()
    assert bt == 5000


def test_attach_auth_db(tmp_path: Path):
    auth_path = tmp_path / "auth.sqlite3"
    # Pre-create the auth file with a table so the ATTACH points at real data.
    with sqlite3.connect(str(auth_path)) as raw:
        raw.execute("CREATE TABLE secrets (k TEXT PRIMARY KEY)")
        raw.execute("INSERT INTO secrets VALUES ('hi')")
        raw.commit()

    db_path = tmp_path / "controller.sqlite3"
    engine = _make_engine(db_path, auth_db_path=auth_path)
    try:
        with read_snapshot(engine) as tx:
            names = [row[0] for row in tx.execute(text("SELECT name FROM auth.sqlite_master WHERE type='table'")).all()]
            assert "secrets" in names
            row = tx.execute(text("SELECT k FROM auth.secrets")).all()
            assert row == [("hi",)]
    finally:
        engine.dispose()


def test_executemany_inserts_many(engine, write_lock):
    _create_kv_table(engine)
    rows = [{"k": f"k{i}", "v": str(i)} for i in range(10)]
    with write_transaction(engine, write_lock) as tx:
        tx.executemany(text("INSERT INTO kv (k, v) VALUES (:k, :v)"), rows)

    with read_snapshot(engine) as tx:
        count = tx.execute(text("SELECT COUNT(*) FROM kv")).scalar()
    assert count == 10


def test_read_snapshot_rejects_writes(engine, write_lock):
    _create_kv_table(engine)
    with pytest.raises(OperationalError):
        with read_snapshot(engine) as tx:
            tx.execute(text("INSERT INTO kv VALUES ('a', '1')"))


def test_tx_register_accumulates_in_order(engine, write_lock):
    _create_kv_table(engine)
    calls: list[str] = []
    with write_transaction(engine, write_lock) as tx:
        tx.execute(text("INSERT INTO kv VALUES ('a', '1')"))
        tx.register(lambda: calls.append("first"))
        tx.register(lambda: calls.append("second"))
        tx.register(lambda: calls.append("third"))
    assert calls == ["first", "second", "third"]
