# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the registry migrations runner.

The runner backs every catalog schema change after the baseline; pinning
its core invariants (idempotency, transactional apply, partial-failure
rollback, legacy-database adoption) keeps future migration authors honest.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from finelog.store.catalog import Catalog, SegmentLocation, SegmentRow
from finelog.store.migrations import apply_migrations, transactional


def _list_tables(conn: duckdb.DuckDBPyConnection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT table_name FROM duckdb_tables()").fetchall()}


def _applied_migrations(conn: duckdb.DuckDBPyConnection) -> list[str]:
    return [row[0] for row in conn.execute("SELECT name FROM schema_migrations ORDER BY name").fetchall()]


# ---------------------------------------------------------------------------
# Baseline: a fresh registry has all tables and 0001 recorded.
# ---------------------------------------------------------------------------


def test_fresh_registry_runs_baseline_migration(tmp_path):
    db = Catalog(tmp_path)
    try:
        tables = _list_tables(db._conn)
        assert "namespaces" in tables
        assert "segments" in tables
        assert "schema_migrations" in tables
        assert _applied_migrations(db._conn) == [
            "0001_init.py",
            "0002_segment_key_value_bounds.py",
            "0003_segment_level.py",
            "0004_segment_copied_at.py",
            "0005_segments_level_index.py",
            "0006_segment_lifecycle.py",
        ]
    finally:
        db.close()


def test_reopen_is_idempotent(tmp_path):
    """Opening the same data dir twice does not re-apply migrations."""
    Catalog(tmp_path).close()
    db = Catalog(tmp_path)
    try:
        # Still exactly one row, not two.
        assert _applied_migrations(db._conn) == [
            "0001_init.py",
            "0002_segment_key_value_bounds.py",
            "0003_segment_level.py",
            "0004_segment_copied_at.py",
            "0005_segments_level_index.py",
            "0006_segment_lifecycle.py",
        ]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Legacy adoption: a pre-migrations DB inherits the baseline cleanly.
# ---------------------------------------------------------------------------


def test_pre_migrations_database_inherits_baseline(tmp_path):
    """A DB created by an old release (only ``namespaces`` table) opens cleanly."""
    db_path = tmp_path / "_finelog_registry.duckdb"
    legacy = duckdb.connect(str(db_path))
    legacy.execute(
        """
        CREATE TABLE namespaces (
            namespace        TEXT PRIMARY KEY,
            schema_json      TEXT NOT NULL,
            registered_at_ms BIGINT NOT NULL,
            last_modified_ms BIGINT NOT NULL
        )
        """
    )
    legacy.execute('INSERT INTO namespaces VALUES (\'iris.worker\', \'{"columns":[],"key_column":""}\', 0, 0)')
    legacy.close()

    db = Catalog(tmp_path)
    try:
        # Existing row preserved; segments table created; baseline recorded.
        assert db._conn.execute("SELECT namespace FROM namespaces").fetchall() == [("iris.worker",)]
        assert "segments" in _list_tables(db._conn)
        assert _applied_migrations(db._conn) == [
            "0001_init.py",
            "0002_segment_key_value_bounds.py",
            "0003_segment_level.py",
            "0004_segment_copied_at.py",
            "0005_segments_level_index.py",
            "0006_segment_lifecycle.py",
        ]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Transactional apply: a failing migration leaves no row in schema_migrations.
# ---------------------------------------------------------------------------


def test_failing_migration_rolls_back(tmp_path):
    fake_dir = tmp_path / "migs"
    fake_dir.mkdir()
    # 0001 succeeds, 0002 fails. Both should be observable via the runner;
    # only 0001 should be marked applied.
    (fake_dir / "0001_ok.py").write_text(
        "import duckdb\n"
        "def migrate(conn: duckdb.DuckDBPyConnection, *, data_dir) -> None:\n"
        "    conn.execute('CREATE TABLE ok_marker (n INTEGER)')\n"
    )
    (fake_dir / "0002_boom.py").write_text(
        "import duckdb\n"
        "def migrate(conn: duckdb.DuckDBPyConnection, *, data_dir) -> None:\n"
        "    conn.execute('CREATE TABLE will_be_rolled_back (n INTEGER)')\n"
        "    raise RuntimeError('simulated migration failure')\n"
    )

    conn = duckdb.connect(":memory:")
    try:
        with pytest.raises(RuntimeError, match="simulated migration failure"):
            apply_migrations(conn, fake_dir)

        # 0001 is recorded and its table exists.
        assert _applied_migrations(conn) == ["0001_ok.py"]
        assert "ok_marker" in _list_tables(conn)
        # 0002's side-effect was rolled back: neither the table nor a row.
        assert "will_be_rolled_back" not in _list_tables(conn)
    finally:
        conn.close()


def test_failed_migration_retries_on_next_apply(tmp_path):
    """A migration that failed once must run again on the next open."""
    fake_dir = tmp_path / "migs"
    fake_dir.mkdir()
    migration_path = fake_dir / "0001_flaky.py"

    # First version raises after a side-effect, simulating a half-failed apply.
    migration_path.write_text(
        "def migrate(conn, *, data_dir):\n"
        "    conn.execute('CREATE TABLE IF NOT EXISTS marker (n INTEGER)')\n"
        "    raise RuntimeError('not yet')\n"
    )

    conn = duckdb.connect(":memory:")
    try:
        with pytest.raises(RuntimeError, match="not yet"):
            apply_migrations(conn, fake_dir)
        assert _applied_migrations(conn) == []

        # Author fixes the migration; next apply picks it up because no row
        # was inserted on the failed pass.
        migration_path.write_text(
            "def migrate(conn, *, data_dir):\n    conn.execute('CREATE TABLE IF NOT EXISTS marker (n INTEGER)')\n"
        )
        apply_migrations(conn, fake_dir)
        assert _applied_migrations(conn) == ["0001_flaky.py"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# transactional helper exposed for callers (compaction's segment swap, etc.)
# ---------------------------------------------------------------------------


def test_transactional_commits_on_success():
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("CREATE TABLE t (n INTEGER)")
        with transactional(conn):
            conn.execute("INSERT INTO t VALUES (1)")
            conn.execute("INSERT INTO t VALUES (2)")
        assert conn.execute("SELECT count(*) FROM t").fetchone() == (2,)
    finally:
        conn.close()


def test_transactional_rolls_back_on_exception():
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("CREATE TABLE t (n INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        with pytest.raises(ValueError):
            with transactional(conn):
                conn.execute("INSERT INTO t VALUES (2)")
                raise ValueError("rollback please")
        assert conn.execute("SELECT count(*) FROM t").fetchone() == (1,)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Discovery: hidden / dunder files are not picked up.
# ---------------------------------------------------------------------------


def test_underscore_prefixed_files_are_skipped(tmp_path):
    fake_dir = tmp_path / "migs"
    fake_dir.mkdir()
    (fake_dir / "_runner.py").write_text("raise AssertionError('runner module should never be loaded as a migration')")
    (fake_dir / "__init__.py").write_text("raise AssertionError('package init should never be loaded as a migration')")
    (fake_dir / "0001_real.py").write_text(
        "def migrate(conn, *, data_dir): conn.execute('CREATE TABLE m (n INTEGER)')\n"
    )

    conn = duckdb.connect(":memory:")
    try:
        apply_migrations(conn, fake_dir)
        assert _applied_migrations(conn) == ["0001_real.py"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Integrity: a real production-shaped catalog still works after migrations.
# ---------------------------------------------------------------------------


def test_catalog_segments_apis_function_after_migration(tmp_path):
    """End-to-end check: write a segment row through the public Catalog API."""
    db = Catalog(tmp_path)
    try:
        # ``aggregate_namespace_stats`` and friends rely on the baseline schema.
        assert db.aggregate_namespace_stats("ns").row_count == 0
        seg = SegmentRow(
            namespace="ns",
            path=str(Path("/x/seg_L1_0000000000000000001.parquet")),
            level=1,
            min_seq=1,
            max_seq=10,
            row_count=10,
            byte_size=1024,
            created_at_ms=0,
            min_key_value="alpha",
            max_key_value="zeta",
            location=SegmentLocation.BOTH,
        )
        db.upsert_segment(seg)
        stats = db.aggregate_namespace_stats("ns")
        assert stats.row_count == 10
        assert stats.segment_count == 1

        # 0002 + 0003 + 0006 columns survive the round-trip.
        rows = db.list_segments("ns")
        assert len(rows) == 1
        assert rows[0].level == 1
        assert rows[0].min_key_value == "alpha"
        assert rows[0].max_key_value == "zeta"
        assert rows[0].location is SegmentLocation.BOTH

        seg_no_key = SegmentRow(
            namespace="ns",
            path=str(Path("/x/seg_L1_0000000000000000011.parquet")),
            level=1,
            min_seq=11,
            max_seq=20,
            row_count=10,
            byte_size=1024,
            created_at_ms=0,
        )
        db.upsert_segment(seg_no_key)
        rows = db.list_segments("ns")
        no_key_row = next(r for r in rows if r.path.endswith("0000000011.parquet"))
        assert no_key_row.min_key_value is None
        assert no_key_row.max_key_value is None
        assert no_key_row.location is SegmentLocation.LOCAL
    finally:
        db.close()
