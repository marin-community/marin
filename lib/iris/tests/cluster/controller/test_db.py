# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TransactionCursor mutation helpers and read pool in db.py."""

import threading
from pathlib import Path

import pytest
from iris.cluster.controller.db import (
    ControllerDB,
    Row,
    TransactionCursor,
)
from iris.cluster.controller.db import _SqlPredicate


@pytest.fixture
def db(tmp_path: Path) -> ControllerDB:
    return ControllerDB(tmp_path / "test.db")


def _create_simple_table(db: ControllerDB) -> None:
    """Create a simple key/value table for testing mutation helpers."""
    with db.transaction() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)")


def test_transaction_yields_transaction_cursor(db: ControllerDB) -> None:
    _create_simple_table(db)
    with db.transaction() as cur:
        assert isinstance(cur, TransactionCursor)


def test_insert_single_row(db: ControllerDB) -> None:
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.insert("kv", {"key": "hello", "value": "world"})

    rows = db.fetchall("SELECT key, value FROM kv")
    assert len(rows) == 1
    assert rows[0]["key"] == "hello"
    assert rows[0]["value"] == "world"


def test_insert_multiple_rows(db: ControllerDB) -> None:
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.insert("kv", {"key": "a", "value": "1"})
        cur.insert("kv", {"key": "b", "value": "2"})
        cur.insert("kv", {"key": "c", "value": "3"})

    rows = db.fetchall("SELECT key, value FROM kv ORDER BY key")
    assert [r["key"] for r in rows] == ["a", "b", "c"]


def test_update_matching_rows(db: ControllerDB) -> None:
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.insert("kv", {"key": "x", "value": "old"})
        cur.insert("kv", {"key": "y", "value": "old"})

    with db.transaction() as cur:
        rowcount = cur.update("kv", updates={"value": "new"}, where=_SqlPredicate("key = ?", ("x",)))

    assert rowcount == 1
    rows = db.fetchall("SELECT key, value FROM kv ORDER BY key")
    assert rows[0]["value"] == "new"
    assert rows[1]["value"] == "old"


def test_update_returns_affected_count(db: ControllerDB) -> None:
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.insert("kv", {"key": "p", "value": "v"})
        cur.insert("kv", {"key": "q", "value": "v"})

    with db.transaction() as cur:
        rowcount = cur.update("kv", updates={"value": "changed"}, where=_SqlPredicate("value = ?", ("v",)))

    assert rowcount == 2


def test_delete_matching_rows(db: ControllerDB) -> None:
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.insert("kv", {"key": "del1", "value": "x"})
        cur.insert("kv", {"key": "del2", "value": "x"})
        cur.insert("kv", {"key": "keep", "value": "y"})

    with db.transaction() as cur:
        rowcount = cur.delete("kv", where=_SqlPredicate("value = ?", ("x",)))

    assert rowcount == 2
    rows = db.fetchall("SELECT key FROM kv")
    assert len(rows) == 1
    assert rows[0]["key"] == "keep"


def test_delete_returns_affected_count(db: ControllerDB) -> None:
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.insert("kv", {"key": "gone", "value": "v"})

    with db.transaction() as cur:
        rowcount = cur.delete("kv", where=_SqlPredicate("key = ?", ("gone",)))

    assert rowcount == 1


def test_delete_no_match_returns_zero(db: ControllerDB) -> None:
    _create_simple_table(db)
    with db.transaction() as cur:
        rowcount = cur.delete("kv", where=_SqlPredicate("key = ?", ("nonexistent",)))

    assert rowcount == 0


def test_update_no_match_returns_zero(db: ControllerDB) -> None:
    _create_simple_table(db)
    with db.transaction() as cur:
        rowcount = cur.update("kv", updates={"value": "x"}, where=_SqlPredicate("key = ?", ("absent",)))

    assert rowcount == 0


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
            cur.insert("kv", {"key": "should_not_persist", "value": "v"})
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
        cur.insert("kv", {"key": "a", "value": "x"})
        cur.insert("kv", {"key": "b", "value": "x"})
        cur.insert("kv", {"key": "c", "value": "y"})

    with db.snapshot() as snap:
        rows = snap.raw(
            "SELECT value, COUNT(*) AS cnt FROM kv GROUP BY value ORDER BY value",
        )

    assert len(rows) == 2
    assert all(isinstance(r, Row) for r in rows)
    assert rows[0].value == "x"
    assert rows[0].cnt == 2
    assert rows[1].value == "y"
    assert rows[1].cnt == 1


def test_raw_with_decoder(db: ControllerDB) -> None:
    """raw() applies per-column decoders to matching columns."""
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.insert("kv", {"key": "k1", "value": "hello"})

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
        cur.insert("kv", {"key": "k", "value": "v"})

    with db.snapshot() as snap:
        rows = snap.raw("SELECT key FROM kv")

    assert len(rows) == 1
    with pytest.raises(AttributeError, match="no column"):
        _ = rows[0].nonexistent


def test_update_with_composite_predicate(db: ControllerDB) -> None:
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.insert("kv", {"key": "match", "value": "old"})
        cur.insert("kv", {"key": "nomatch", "value": "old"})

    key_pred = _SqlPredicate("key = ?", ("match",))
    val_pred = _SqlPredicate("value = ?", ("old",))
    combined = key_pred & val_pred

    with db.transaction() as cur:
        rowcount = cur.update("kv", updates={"value": "new"}, where=combined)

    assert rowcount == 1
    row = db.fetchone("SELECT value FROM kv WHERE key = 'match'")
    assert row["value"] == "new"
    row2 = db.fetchone("SELECT value FROM kv WHERE key = 'nomatch'")
    assert row2["value"] == "old"


def test_read_snapshot_does_not_block_write(db: ControllerDB) -> None:
    """read_snapshot() uses a separate connection, so a concurrent write transaction proceeds."""
    _create_simple_table(db)
    with db.transaction() as cur:
        cur.insert("kv", {"key": "init", "value": "v"})

    results: dict[str, bool] = {}

    def writer() -> None:
        """Hold the write lock for a short time, recording success."""
        with db.transaction() as cur:
            cur.insert("kv", {"key": "from_writer", "value": "w"})
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
        cur.insert("kv", {"key": "a", "value": "1"})

    with db.read_snapshot() as q:
        rows_start = q.raw("SELECT key FROM kv")
        assert len(rows_start) == 1

        # Commit a new row from outside the snapshot.
        with db.transaction() as cur:
            cur.insert("kv", {"key": "b", "value": "2"})

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
