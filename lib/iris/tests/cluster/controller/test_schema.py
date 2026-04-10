# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for schema registry DDL generation and projection infrastructure."""

import importlib.util
import re
import sqlite3
from pathlib import Path

import pytest
from iris.cluster.controller.schema import (
    JOBS,
    MAIN_TABLES,
    PROFILES_TABLES,
    generate_full_ddl,
)


def _create_db() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with the full schema registry."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ddl = generate_full_ddl(MAIN_TABLES)
    conn.executescript(ddl)
    conn.execute("ATTACH ':memory:' AS profiles")
    profiles_ddl = generate_full_ddl(PROFILES_TABLES)
    conn.executescript(profiles_ddl)
    return conn


def test_schema_registry_ddl_valid() -> None:
    """DDL from generate_full_ddl(MAIN_TABLES) executes without errors."""
    conn = _create_db()
    # If we got here, the DDL is valid SQL.
    conn.close()


def test_table_completeness() -> None:
    """All tables declared in MAIN_TABLES exist in the created DB."""
    conn = _create_db()
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    actual_tables = {r["name"] for r in rows}

    expected_tables = {t.name for t in MAIN_TABLES}
    # sqlite_sequence is auto-created by AUTOINCREMENT tables
    actual_tables.discard("sqlite_sequence")
    assert expected_tables == actual_tables
    conn.close()


def test_index_completeness() -> None:
    """All indexes declared in MAIN_TABLES exist in the created DB."""
    conn = _create_db()
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'").fetchall()
    actual_indexes = {r["name"] for r in rows}

    expected_indexes: set[str] = set()
    for table in MAIN_TABLES:
        for idx_sql in table.indexes:
            # Extract index name from "CREATE INDEX IF NOT EXISTS <name> ON ..."
            match = re.search(r"CREATE INDEX IF NOT EXISTS (\S+)", idx_sql)
            assert match, f"Cannot parse index name from: {idx_sql}"
            expected_indexes.add(match.group(1))

    assert expected_indexes == actual_indexes
    conn.close()


def test_trigger_completeness() -> None:
    """All triggers declared in MAIN_TABLES exist in the created DB."""
    conn = _create_db()
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
    actual_triggers = {r["name"] for r in rows}

    expected_triggers: set[str] = set()
    for table in MAIN_TABLES:
        for trg_sql in table.triggers:
            match = re.search(r"CREATE TRIGGER IF NOT EXISTS (\S+)", trg_sql)
            assert match, f"Cannot parse trigger name from: {trg_sql}"
            expected_triggers.add(match.group(1))

    assert expected_triggers == actual_triggers
    conn.close()


def test_projection_unknown_column_raises() -> None:
    """Requesting a nonexistent column raises KeyError."""
    with pytest.raises(KeyError, match="nonexistent_column"):
        JOBS.projection("nonexistent_column")


def test_projection_valid_columns() -> None:
    """Requesting valid columns succeeds."""
    proj = JOBS.projection("job_id", "state")
    assert proj is not None
    assert len(proj.columns) == 2


def test_projection_select_clause() -> None:
    """select_clause returns 'alias.col1, alias.col2' format."""
    proj = JOBS.projection("job_id", "state")
    assert proj.select_clause() == "j.job_id, j.state"


def test_projection_decode() -> None:
    """Projection can decode a row inserted via raw SQL."""
    conn = _create_db()

    # Insert minimal data to satisfy FK constraints: users first, then jobs.
    conn.execute("INSERT INTO users (user_id, created_at_ms) VALUES ('u1', 1000)")
    conn.execute(
        "INSERT INTO jobs ("
        "  job_id, user_id, root_job_id, depth, request_proto, state,"
        "  submitted_at_ms, root_submitted_at_ms, num_tasks, is_reservation_holder"
        ") VALUES ("
        "  '/u1/test-job', 'u1', '/u1/test-job', 0, X'', 1, 2000, 2000, 5, 0"
        ")"
    )
    conn.commit()

    proj = JOBS.projection("job_id", "state", "num_tasks")
    rows = conn.execute(f"SELECT {proj.select_clause()} FROM jobs j").fetchall()
    decoded = proj.decode(rows)

    assert len(decoded) == 1
    row = decoded[0]
    assert str(row.job_id) == "/u1/test-job"
    assert row.state == 1
    assert row.num_tasks == 5
    conn.close()


def _normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison: whitespace, quoting, punctuation."""
    # Collapse all whitespace runs (including newlines) to a single space
    s = re.sub(r"\s+", " ", sql).strip()
    # Strip double-quotes around identifiers (ALTER TABLE RENAME adds them)
    s = s.replace('"', "")
    # Normalize space before commas: "foo , bar" -> "foo, bar"
    s = re.sub(r"\s+,", ",", s)
    # Normalize space before closing paren: "foo )" -> "foo)"
    s = re.sub(r"\s+\)", ")", s)
    # Normalize space after opening paren: "( foo" -> "(foo" -- but keep "( " in CREATE TABLE
    # Actually just normalize multiple spaces again after all replacements
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_schema(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """Extract (type, name, normalized_sql) from sqlite_master and profiles.sqlite_master."""
    result = []
    for schema in ("main", "profiles"):
        try:
            rows = conn.execute(f"SELECT type, name, sql FROM {schema}.sqlite_master ORDER BY type, name").fetchall()
        except sqlite3.OperationalError:
            continue
        for row in rows:
            obj_type, name, sql = row[0], row[1], row[2]
            if name == "sqlite_sequence":
                continue
            if name.startswith("sqlite_autoindex_"):
                continue
            normalized = _normalize_sql(sql) if sql else ""
            result.append((obj_type, name, normalized))
    return sorted(result)


def _run_all_migrations() -> sqlite3.Connection:
    """Create an in-memory DB by running all migrations from scratch."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")

    # Attach in-memory DBs for schemas that migrations expect
    conn.execute("ATTACH ':memory:' AS auth")
    conn.execute("ATTACH ':memory:' AS profiles")

    # The migration runner creates schema_migrations first
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name TEXT PRIMARY KEY,
            applied_at_ms INTEGER NOT NULL
        )
        """
    )

    import iris.cluster.controller.db as db_mod

    migrations_dir = Path(db_mod.__file__).resolve().parent / "migrations"
    for path in sorted(migrations_dir.glob("*.py")):
        if path.name.startswith("__"):
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.migrate(conn)
        conn.commit()
        conn.execute(
            "INSERT INTO schema_migrations(name, applied_at_ms) VALUES (?, ?)",
            (path.name, 0),
        )
        conn.commit()

    return conn


def test_schema_registry_matches_migrations() -> None:
    """Schema registry DDL should produce the same schema as running all migrations."""
    # DB from schema registry
    registry_conn = _create_db()
    registry_schema = _extract_schema(registry_conn)
    registry_conn.close()

    # DB from migrations
    migration_conn = _run_all_migrations()
    migration_schema = _extract_schema(migration_conn)
    migration_conn.close()

    # Build lookup dicts for detailed comparison
    registry_dict = {(t, n): sql for t, n, sql in registry_schema}
    migration_dict = {(t, n): sql for t, n, sql in migration_schema}

    registry_keys = set(registry_dict.keys())
    migration_keys = set(migration_dict.keys())

    only_in_registry = registry_keys - migration_keys
    only_in_migrations = migration_keys - registry_keys
    common = registry_keys & migration_keys

    differences: list[str] = []

    for key in sorted(only_in_registry):
        differences.append(f"ONLY IN REGISTRY: {key[0]} {key[1]}")

    for key in sorted(only_in_migrations):
        differences.append(f"ONLY IN MIGRATIONS: {key[0]} {key[1]}")

    for key in sorted(common):
        if registry_dict[key] != migration_dict[key]:
            differences.append(
                f"DIFFERS: {key[0]} {key[1]}\n"
                f"  registry:   {registry_dict[key]}\n"
                f"  migrations: {migration_dict[key]}"
            )

    if differences:
        diff_text = "\n".join(differences)
        # Fail hard: the schema registry must match what migrations produce
        raise AssertionError(f"Schema registry does not match migrations:\n{diff_text}")
