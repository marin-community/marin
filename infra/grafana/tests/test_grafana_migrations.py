# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sqlite3
from contextlib import closing
from pathlib import Path

import grafana_migrations.engine as migration_engine
import pytest
from grafana_migrations import DatabaseBackend, migrate

FIRST_MIGRATION = "m0001_upgrade_iap_viewers"


def _create_org_users(database_path: Path) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE org_user (user_id INTEGER, org_id INTEGER, role TEXT, updated DATETIME)")
        connection.executemany(
            "INSERT INTO org_user VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            [
                (1, 1, "Viewer"),
                (2, 1, "Editor"),
                (3, 1, "Admin"),
                (4, 2, "Viewer"),
                (5, 1, "None"),
            ],
        )


def test_migrate_applies_pending_migration_and_records_it(tmp_path: Path) -> None:
    database_path = tmp_path / "grafana.db"
    _create_org_users(database_path)

    with sqlite3.connect(database_path) as connection:
        assert migrate(connection, DatabaseBackend.SQLITE) == [FIRST_MIGRATION]

    with sqlite3.connect(database_path) as connection:
        memberships = connection.execute("SELECT user_id, org_id, role FROM org_user ORDER BY user_id").fetchall()
        applied = connection.execute("SELECT version, name FROM marin_schema_migrations").fetchall()
    assert memberships == [
        (1, 1, "Editor"),
        (2, 1, "Editor"),
        (3, 1, "Admin"),
        (4, 2, "Editor"),
        (5, 1, "None"),
    ]
    assert applied == [(1, FIRST_MIGRATION)]


def test_migrate_skips_recorded_migration(tmp_path: Path) -> None:
    database_path = tmp_path / "grafana.db"
    _create_org_users(database_path)
    with sqlite3.connect(database_path) as connection:
        migrate(connection, DatabaseBackend.SQLITE)
        connection.execute("INSERT INTO org_user VALUES (6, 1, 'Viewer', CURRENT_TIMESTAMP)")

    with sqlite3.connect(database_path) as connection:
        assert migrate(connection, DatabaseBackend.SQLITE) == []

    with sqlite3.connect(database_path) as connection:
        role = connection.execute("SELECT role FROM org_user WHERE user_id = 6").fetchone()
    assert role == ("Viewer",)


def test_migrate_records_migration_before_grafana_creates_schema(tmp_path: Path) -> None:
    database_path = tmp_path / "grafana.db"
    with sqlite3.connect(database_path) as connection:
        assert migrate(connection, DatabaseBackend.SQLITE) == [FIRST_MIGRATION]

    with sqlite3.connect(database_path) as connection:
        applied = connection.execute("SELECT version, name FROM marin_schema_migrations").fetchall()
        org_user = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'org_user'"
        ).fetchone()
    assert applied == [(1, FIRST_MIGRATION)]
    assert org_user is None


def test_migrate_rolls_back_failed_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "m0001_fails.py").write_text(
        """
def migrate(connection, _backend):
    connection.execute("CREATE TABLE partial_migration (id INTEGER)")
    raise RuntimeError("migration failed")
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(migration_engine, "MIGRATIONS_DIR", migrations_dir)

    database_path = tmp_path / "grafana.db"
    with closing(sqlite3.connect(database_path)) as connection:
        with pytest.raises(RuntimeError):
            migrate(connection, DatabaseBackend.SQLITE)
        applied = connection.execute("SELECT version, name FROM marin_schema_migrations").fetchall()
        partial_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'partial_migration'"
        ).fetchone()
    assert applied == []
    assert partial_table is None
