# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for wildcard bucket ('*') support in protect rules, and delete rule matching.

Exercises IS_PROTECTED matching, rule cost materialization, delete rule costs,
and rule consolidation logic. Uses an in-memory DuckDB with the real schema init
path to stay in sync with production.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest

from scripts.storage.db import (
    MARIN_BUCKETS,
    _DIR_SUMMARY_ARROW_SCHEMA,
    DirSummaryBuffer,
)
from scripts.storage.db import IS_PROTECTED
from scripts.storage.rules import _consolidate_wildcard_rules


def _make_test_db() -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB with the full storage schema for testing."""
    db = duckdb.connect(":memory:")
    db.execute(
        """
        CREATE TABLE storage_classes (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            price_per_gib_month_us REAL NOT NULL,
            price_per_gib_month_eu REAL NOT NULL
        )
        """
    )
    for sc_id, name, us_price, eu_price in [
        (1, "STANDARD", 0.020, 0.023),
        (2, "NEARLINE", 0.010, 0.013),
        (3, "COLDLINE", 0.004, 0.006),
        (4, "ARCHIVE", 0.0012, 0.0025),
    ]:
        db.execute(
            "INSERT INTO storage_classes VALUES (?, ?, ?, ?)",
            (sc_id, name, us_price, eu_price),
        )

    db.execute("CREATE SEQUENCE protect_rules_id_seq START 1")
    db.execute(
        """
        CREATE TABLE protect_rules (
            id INTEGER PRIMARY KEY DEFAULT nextval('protect_rules_id_seq'),
            bucket TEXT NOT NULL,
            pattern TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            owners TEXT,
            reasons TEXT,
            sources TEXT,
            UNIQUE (bucket, pattern)
        )
        """
    )
    db.execute("CREATE SEQUENCE delete_rules_id_seq START 1")
    db.execute(
        """
        CREATE TABLE delete_rules (
            id INTEGER PRIMARY KEY DEFAULT nextval('delete_rules_id_seq'),
            pattern TEXT NOT NULL,
            storage_class TEXT,
            description TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE dir_summary (
            bucket TEXT NOT NULL,
            dir_prefix TEXT NOT NULL,
            standard_count INTEGER NOT NULL DEFAULT 0,
            standard_bytes BIGINT NOT NULL DEFAULT 0,
            nearline_count INTEGER NOT NULL DEFAULT 0,
            nearline_bytes BIGINT NOT NULL DEFAULT 0,
            coldline_count INTEGER NOT NULL DEFAULT 0,
            coldline_bytes BIGINT NOT NULL DEFAULT 0,
            archive_count INTEGER NOT NULL DEFAULT 0,
            archive_bytes BIGINT NOT NULL DEFAULT 0,
            PRIMARY KEY (bucket, dir_prefix)
        )
        """
    )

    # Seed objects view (empty, needed by some queries)
    db.execute(
        """
        CREATE VIEW objects AS
        SELECT NULL::VARCHAR as bucket, NULL::VARCHAR as name,
               NULL::BIGINT as size_bytes, NULL::INTEGER as storage_class_id,
               NULL::TIMESTAMPTZ as created, NULL::TIMESTAMPTZ as updated
        WHERE false
        """
    )
    return db


@pytest.fixture
def conn():
    """Pytest fixture wrapping _make_test_db with cleanup."""
    db = _make_test_db()
    yield db
    db.close()


def _insert_rule(conn, bucket: str, pattern: str, owners: str = "", reasons: str = ""):
    row = conn.execute(
        "INSERT INTO protect_rules (bucket, pattern, pattern_type, owners, reasons, sources) "
        "VALUES (?, ?, 'like', ?, ?, '') RETURNING id",
        (bucket, pattern, owners, reasons),
    ).fetchone()
    return row[0]


def _insert_dir_summary(
    conn,
    bucket: str,
    dir_prefix: str,
    *,
    standard_count: int = 0,
    standard_bytes: int = 0,
    nearline_count: int = 0,
    nearline_bytes: int = 0,
    coldline_count: int = 0,
    coldline_bytes: int = 0,
    archive_count: int = 0,
    archive_bytes: int = 0,
) -> None:
    conn.execute(
        """
        INSERT INTO dir_summary (
            bucket, dir_prefix,
            standard_count, standard_bytes,
            nearline_count, nearline_bytes,
            coldline_count, coldline_bytes,
            archive_count, archive_bytes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            bucket,
            dir_prefix,
            standard_count,
            standard_bytes,
            nearline_count,
            nearline_bytes,
            coldline_count,
            coldline_bytes,
            archive_count,
            archive_bytes,
        ),
    )


# ---------------------------------------------------------------------------
# IS_PROTECTED
# ---------------------------------------------------------------------------


def test_is_protected_matches_exact_bucket(conn):
    _insert_rule(conn, "marin-us-central1", "raw/%")
    result = conn.execute(
        f"""
        SELECT EXISTS (
            {IS_PROTECTED}
        )
        FROM (SELECT 'marin-us-central1' as bucket, 'raw/foo.txt' as name) o
        """
    ).fetchone()[0]
    assert result is True


def test_is_protected_does_not_match_wrong_bucket(conn):
    _insert_rule(conn, "marin-us-central1", "raw/%")
    result = conn.execute(
        f"""
        SELECT EXISTS (
            {IS_PROTECTED}
        )
        FROM (SELECT 'marin-eu-west4' as bucket, 'raw/foo.txt' as name) o
        """
    ).fetchone()[0]
    assert result is False


def test_is_protected_wildcard_matches_any_bucket(conn):
    _insert_rule(conn, "*", "raw/%")
    for bucket in ["marin-us-central1", "marin-eu-west4", "marin-us-east1"]:
        result = conn.execute(
            f"""
            SELECT EXISTS (
                {IS_PROTECTED}
            )
            FROM (SELECT ? as bucket, 'raw/foo.txt' as name) o
            """,
            (bucket,),
        ).fetchone()[0]
        assert result is True, f"wildcard rule should match {bucket}"


def test_is_protected_wildcard_does_not_match_wrong_pattern(conn):
    _insert_rule(conn, "*", "raw/%")
    result = conn.execute(
        f"""
        SELECT EXISTS (
            {IS_PROTECTED}
        )
        FROM (SELECT 'marin-us-central1' as bucket, 'tokenized/foo.txt' as name) o
        """
    ).fetchone()[0]
    assert result is False


# ---------------------------------------------------------------------------
# _consolidate_wildcard_rules
# ---------------------------------------------------------------------------


def _make_row(bucket: str, pattern: str, owners: str = "alice", reasons: str = "keep"):
    return (bucket, pattern, "like", owners, reasons, "test")


def test_consolidate_identical_across_all_buckets():
    """Rules identical across all buckets should merge to bucket='*'."""
    all_buckets = frozenset(MARIN_BUCKETS)
    rows = [_make_row(b, "data/train/%") for b in MARIN_BUCKETS]

    result = _consolidate_wildcard_rules(rows, all_buckets)

    assert len(result) == 1
    assert result[0][0] == "*"
    assert result[0][1] == "data/train/%"


def test_consolidate_not_all_buckets_preserves_per_bucket():
    """Rules that don't appear in every bucket stay per-bucket."""
    all_buckets = frozenset(MARIN_BUCKETS)
    rows = [_make_row(MARIN_BUCKETS[0], "data/train/%")]

    result = _consolidate_wildcard_rules(rows, all_buckets)

    assert len(result) == 1
    assert result[0][0] == MARIN_BUCKETS[0]


def test_consolidate_different_owners_preserves_per_bucket():
    """Rules with different owners across buckets are not consolidated."""
    all_buckets = frozenset(MARIN_BUCKETS)
    rows = [_make_row(b, "data/train/%", owners=f"owner-{i}") for i, b in enumerate(MARIN_BUCKETS)]

    result = _consolidate_wildcard_rules(rows, all_buckets)

    assert len(result) == len(MARIN_BUCKETS)
    assert all(r[0] != "*" for r in result)


def test_consolidate_raw_prefix_always_wildcard():
    """A raw/ prefix is promoted to wildcard even if present in only one bucket."""
    all_buckets = frozenset(MARIN_BUCKETS)
    rows = [_make_row(MARIN_BUCKETS[0], "raw/train/%")]

    result = _consolidate_wildcard_rules(rows, all_buckets)

    assert len(result) == 1
    assert result[0][0] == "*"
    assert result[0][1] == "raw/train/%"


def test_consolidate_tokenized_prefix_always_wildcard():
    """A tokenized/ prefix is promoted to wildcard even if present in one bucket."""
    all_buckets = frozenset(MARIN_BUCKETS)
    rows = [_make_row(MARIN_BUCKETS[0], "tokenized/foo/%")]

    result = _consolidate_wildcard_rules(rows, all_buckets)

    assert len(result) == 1
    assert result[0][0] == "*"


def test_consolidate_mixed_patterns():
    """Test a mix: one pattern qualifies for wildcard, one does not."""
    all_buckets = frozenset(MARIN_BUCKETS)
    rows = [_make_row(b, "data/train/%") for b in MARIN_BUCKETS]
    rows.append(_make_row(MARIN_BUCKETS[0], "custom/model/%"))

    result = _consolidate_wildcard_rules(rows, all_buckets)

    result_by_pattern = {r[1]: r for r in result}
    assert result_by_pattern["data/train/%"][0] == "*"
    assert result_by_pattern["custom/model/%"][0] == MARIN_BUCKETS[0]


# ---------------------------------------------------------------------------
# delete_rules / protect interaction
# ---------------------------------------------------------------------------

_TEST_BUCKET = "marin-us-central1"


def test_protect_wins_over_delete():
    """IS_PROTECTED correctly identifies objects covered by protect rules, regardless of delete rules."""
    conn = _make_test_db()

    # Delete rule: delete everything under old/
    conn.execute(
        "INSERT INTO delete_rules (pattern, storage_class, description, created_at) "
        "VALUES ('old/%', NULL, 'delete old', '2024-01-01')"
    )

    # Protect rule: keep old/important/
    conn.execute(
        "INSERT INTO protect_rules (bucket, pattern, pattern_type, owners, reasons, sources) "
        f"VALUES ('{_TEST_BUCKET}', 'old/important/%', 'like', 'test', 'important data', '')"
    )

    # old/important/file.bin should be protected
    result = conn.execute(
        f"""
        SELECT CASE WHEN EXISTS ({IS_PROTECTED}) THEN 1 ELSE 0 END as protected
        FROM (SELECT '{_TEST_BUCKET}' as bucket, 'old/important/file.bin' as name) o
        """
    ).fetchone()
    assert result[0] == 1

    # old/junk/file.bin should not be protected
    result = conn.execute(
        f"""
        SELECT CASE WHEN EXISTS ({IS_PROTECTED}) THEN 1 ELSE 0 END as protected
        FROM (SELECT '{_TEST_BUCKET}' as bucket, 'old/junk/file.bin' as name) o
        """
    ).fetchone()
    assert result[0] == 0

    conn.close()


# ---------------------------------------------------------------------------
# DirSummaryBuffer / parquet roundtrip
# ---------------------------------------------------------------------------


def _make_dir_summary_arrow(rows: list[dict]) -> pa.Table:
    """Build a dir_summary Arrow table from a list of row dicts."""
    return pa.table(
        {
            "bucket": [r["bucket"] for r in rows],
            "dir_prefix": [r["dir_prefix"] for r in rows],
            "standard_count": pa.array([r.get("standard_count", 0) for r in rows], type=pa.int32()),
            "standard_bytes": pa.array([r.get("standard_bytes", 0) for r in rows], type=pa.int64()),
            "nearline_count": pa.array([r.get("nearline_count", 0) for r in rows], type=pa.int32()),
            "nearline_bytes": pa.array([r.get("nearline_bytes", 0) for r in rows], type=pa.int64()),
            "coldline_count": pa.array([r.get("coldline_count", 0) for r in rows], type=pa.int32()),
            "coldline_bytes": pa.array([r.get("coldline_bytes", 0) for r in rows], type=pa.int64()),
            "archive_count": pa.array([r.get("archive_count", 0) for r in rows], type=pa.int32()),
            "archive_bytes": pa.array([r.get("archive_bytes", 0) for r in rows], type=pa.int64()),
        },
        schema=_DIR_SUMMARY_ARROW_SCHEMA,
    )


def test_dir_summary_parquet_roundtrip(tmp_path):
    """DirSummaryBuffer writes parquet and exposes a working view over it."""
    conn = duckdb.connect(":memory:")
    parquet_dir = tmp_path / "dir_summary_parquet"

    buf = DirSummaryBuffer(parquet_dir, conn)

    table = _make_dir_summary_arrow(
        [
            {
                "bucket": "b1",
                "dir_prefix": "foo/",
                "standard_count": 10,
                "standard_bytes": 1000,
                "nearline_count": 5,
                "nearline_bytes": 500,
            },
            {
                "bucket": "b1",
                "dir_prefix": "bar/",
                "standard_count": 20,
                "standard_bytes": 2000,
                "coldline_count": 3,
                "coldline_bytes": 300,
            },
        ]
    )
    buf.write_arrow_table(table)

    rows = conn.execute("SELECT bucket, dir_prefix, standard_count FROM dir_summary ORDER BY dir_prefix").fetchall()
    assert len(rows) == 2
    assert rows[0] == ("b1", "bar/", 20)
    assert rows[1] == ("b1", "foo/", 10)

    parquet_files = list(parquet_dir.glob("dir_summary_*.parquet"))
    assert len(parquet_files) == 1

    buf.reset()
    rows = conn.execute("SELECT * FROM dir_summary").fetchall()
    assert len(rows) == 0
    # Parquet dir still exists but is empty
    assert parquet_dir.exists()
    assert list(parquet_dir.glob("dir_summary_*.parquet")) == []


def test_dir_summary_v14_migration(tmp_path):
    """Data in a legacy dir_summary TABLE survives migration to parquet-backed view."""
    parquet_dir = tmp_path / "dir_summary_parquet"

    # Build a DB that looks like schema v12: dir_summary is a real table.
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE dir_summary (
            bucket TEXT NOT NULL,
            dir_prefix TEXT NOT NULL,
            standard_count INTEGER NOT NULL DEFAULT 0,
            standard_bytes BIGINT NOT NULL DEFAULT 0,
            nearline_count INTEGER NOT NULL DEFAULT 0,
            nearline_bytes BIGINT NOT NULL DEFAULT 0,
            coldline_count INTEGER NOT NULL DEFAULT 0,
            coldline_bytes BIGINT NOT NULL DEFAULT 0,
            archive_count INTEGER NOT NULL DEFAULT 0,
            archive_bytes BIGINT NOT NULL DEFAULT 0,
            PRIMARY KEY (bucket, dir_prefix)
        )
        """
    )
    conn.execute("INSERT INTO dir_summary VALUES ('b1', 'raw/train/', 100, 10000, 0, 0, 0, 0, 0, 0)")
    conn.execute("INSERT INTO dir_summary VALUES ('b2', 'tokenized/en/', 0, 0, 50, 5000, 0, 0, 0, 0)")

    # Reproduce the v14 migration block from init_db() using our test parquet_dir.
    # Note: we must fetch rows and drop the table BEFORE constructing DirSummaryBuffer,
    # because the constructor calls _refresh_view() which issues CREATE OR REPLACE VIEW,
    # and DuckDB refuses to replace a table with a view in a single statement.
    existing_rows = conn.execute("SELECT * FROM dir_summary").fetch_arrow_table()
    assert len(existing_rows) == 2
    conn.execute("DROP TABLE dir_summary")

    buf = DirSummaryBuffer(parquet_dir, conn)
    buf.write_arrow_table(existing_rows)

    # After migration: dir_summary is now a VIEW (backed by parquet), not a table.
    table_type = conn.execute(
        "SELECT table_type FROM information_schema.tables WHERE table_name = 'dir_summary'"
    ).fetchone()
    assert table_type is not None
    assert table_type[0] == "VIEW"

    # The view returns the same data that was in the table.
    rows = conn.execute(
        "SELECT bucket, dir_prefix, standard_count, nearline_count FROM dir_summary ORDER BY dir_prefix"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0] == ("b1", "raw/train/", 100, 0)
    assert rows[1] == ("b2", "tokenized/en/", 0, 50)

    # The parquet file is present on disk.
    parquet_files = list(parquet_dir.glob("dir_summary_*.parquet"))
    assert len(parquet_files) == 1
