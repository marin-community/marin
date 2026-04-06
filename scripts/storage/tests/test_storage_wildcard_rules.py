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
    materialize_delete_rule_costs,
    materialize_rule_costs,
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
    db.execute(
        """
        CREATE TABLE rule_costs (
            rule_id INTEGER NOT NULL REFERENCES protect_rules(id),
            bucket TEXT NOT NULL,
            storage_class_id INTEGER NOT NULL REFERENCES storage_classes(id),
            object_count INTEGER NOT NULL,
            total_bytes BIGINT NOT NULL,
            monthly_cost_usd REAL NOT NULL,
            PRIMARY KEY (rule_id, storage_class_id, bucket)
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
        CREATE TABLE delete_rule_costs (
            rule_id INTEGER NOT NULL REFERENCES delete_rules(id),
            bucket TEXT NOT NULL,
            storage_class_id INTEGER NOT NULL REFERENCES storage_classes(id),
            object_count INTEGER NOT NULL,
            total_bytes BIGINT NOT NULL,
            monthly_cost_usd REAL NOT NULL,
            PRIMARY KEY (rule_id, storage_class_id, bucket)
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
# materialize_rule_costs
# ---------------------------------------------------------------------------


def test_materialize_rule_costs_bucket_specific(conn):
    """A bucket-specific rule only produces costs for its own bucket."""
    rule_id = _insert_rule(conn, "marin-us-central1", "data/%")
    _insert_dir_summary(conn, "marin-us-central1", "data/train/", nearline_count=100, nearline_bytes=1024 * 1024)
    _insert_dir_summary(conn, "marin-eu-west4", "data/train/", nearline_count=50, nearline_bytes=512 * 1024)

    total = materialize_rule_costs(conn)
    assert total > 0

    rows = conn.execute("SELECT bucket, object_count FROM rule_costs WHERE rule_id = ?", (rule_id,)).fetchall()
    buckets = {r[0] for r in rows}
    assert "marin-us-central1" in buckets
    assert "marin-eu-west4" not in buckets


def test_materialize_rule_costs_wildcard_expands_to_all_matching_buckets(conn):
    """A wildcard rule produces cost rows for every bucket that has matching data."""
    rule_id = _insert_rule(conn, "*", "raw/%")

    # Add data in two buckets
    _insert_dir_summary(conn, "marin-us-central1", "raw/train/", nearline_count=100, nearline_bytes=1024)
    _insert_dir_summary(conn, "marin-eu-west4", "raw/eval/", nearline_count=50, nearline_bytes=512)

    total = materialize_rule_costs(conn)
    assert total > 0

    rows = conn.execute(
        "SELECT bucket, object_count, total_bytes FROM rule_costs WHERE rule_id = ?", (rule_id,)
    ).fetchall()
    cost_by_bucket = {r[0]: (r[1], r[2]) for r in rows}

    assert "marin-us-central1" in cost_by_bucket
    assert "marin-eu-west4" in cost_by_bucket
    assert cost_by_bucket["marin-us-central1"][0] == 100
    assert cost_by_bucket["marin-eu-west4"][0] == 50


def test_materialize_rule_costs_wildcard_and_specific_coexist(conn):
    """Both a wildcard and bucket-specific rule can match the same data."""
    wildcard_id = _insert_rule(conn, "*", "raw/%")
    specific_id = _insert_rule(conn, "marin-us-central1", "raw/special/%")

    _insert_dir_summary(conn, "marin-us-central1", "raw/special/train/", nearline_count=10, nearline_bytes=100)

    materialize_rule_costs(conn)

    wildcard_rows = conn.execute("SELECT bucket FROM rule_costs WHERE rule_id = ?", (wildcard_id,)).fetchall()
    specific_rows = conn.execute("SELECT bucket FROM rule_costs WHERE rule_id = ?", (specific_id,)).fetchall()

    # Wildcard matches marin-us-central1 (where the data is)
    assert any(r[0] == "marin-us-central1" for r in wildcard_rows)
    # Specific rule also matches
    assert any(r[0] == "marin-us-central1" for r in specific_rows)


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
# delete_rules / delete_rule_costs
# ---------------------------------------------------------------------------

# Tests use marin-us-central1 as the canonical bucket since materialize_delete_rule_costs
# iterates plan_rows() which only covers MARIN_BUCKETS. "test-bucket" is not in that list.
_TEST_BUCKET = "marin-us-central1"


def test_delete_rule_matches_pattern():
    """A delete rule with a pattern should match objects via LIKE."""
    conn = _make_test_db()
    conn.execute(
        "INSERT INTO delete_rules (pattern, storage_class, description, created_at) "
        "VALUES ('checkpoints/%', NULL, 'test', '2024-01-01')"
    )
    _insert_dir_summary(conn, _TEST_BUCKET, "checkpoints/exp1/", nearline_count=10, nearline_bytes=1000)
    _insert_dir_summary(conn, _TEST_BUCKET, "data/important/", nearline_count=5, nearline_bytes=500)

    materialize_delete_rule_costs(conn)

    costs = conn.execute("SELECT * FROM delete_rule_costs").fetchall()
    assert len(costs) > 0
    # data/ prefix must not be targeted; only checkpoints/ is matched
    total_bytes = sum(r[4] for r in costs)  # total_bytes is column index 4
    assert total_bytes == 1000

    conn.close()


def test_delete_rule_with_storage_class_filter():
    """A delete rule with storage_class set should only match that class."""
    conn = _make_test_db()
    conn.execute(
        "INSERT INTO delete_rules (pattern, storage_class, description, created_at) "
        "VALUES ('old/%', 'ARCHIVE', 'only archive', '2024-01-01')"
    )
    _insert_dir_summary(
        conn,
        _TEST_BUCKET,
        "old/stuff/",
        nearline_count=5,
        nearline_bytes=500,
        archive_count=10,
        archive_bytes=2000,
    )

    materialize_delete_rule_costs(conn)

    costs = conn.execute(
        """
        SELECT sc.name, drc.total_bytes
        FROM delete_rule_costs drc
        JOIN storage_classes sc ON drc.storage_class_id = sc.id
        """
    ).fetchall()

    class_names = {r[0] for r in costs}
    assert "ARCHIVE" in class_names
    assert "NEARLINE" not in class_names

    conn.close()


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


def test_delete_rule_standard_objects():
    """A delete rule without a storage_class filter targets STANDARD objects too."""
    conn = _make_test_db()
    conn.execute(
        "INSERT INTO delete_rules (pattern, storage_class, description, created_at) "
        "VALUES ('tmp/%', NULL, 'delete all tmp', '2024-01-01')"
    )
    _insert_dir_summary(
        conn,
        _TEST_BUCKET,
        "tmp/scratch/",
        standard_count=20,
        standard_bytes=3000,
        nearline_count=5,
        nearline_bytes=500,
    )

    materialize_delete_rule_costs(conn)

    costs = conn.execute(
        """
        SELECT sc.name, drc.total_bytes
        FROM delete_rule_costs drc
        JOIN storage_classes sc ON drc.storage_class_id = sc.id
        """
    ).fetchall()

    class_names = {r[0] for r in costs}
    assert "STANDARD" in class_names
    assert "NEARLINE" in class_names

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
