# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Discover and apply pending numbered Grafana database migrations."""

import importlib.util
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

MIGRATIONS_DIR = Path(__file__).with_name("migrations")
MIGRATION_FILENAME = re.compile(r"^m(?P<version>[0-9]{4})_[a-z0-9_]+\.py$")
CREATE_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS marin_schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
SELECT_APPLIED_MIGRATIONS = "SELECT version, name FROM marin_schema_migrations ORDER BY version"
INSERT_SQLITE_MIGRATION = "INSERT INTO marin_schema_migrations (version, name) VALUES (?, ?)"
INSERT_POSTGRES_MIGRATION = "INSERT INTO marin_schema_migrations (version, name) VALUES (%s, %s)"


class DatabaseBackend(StrEnum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"


class DatabaseCursor(Protocol):
    rowcount: int

    def fetchone(self) -> tuple[object, ...] | None: ...

    def fetchall(self) -> list[tuple[object, ...]]: ...


class DatabaseConnection(Protocol):
    def execute(self, query: str, parameters: tuple[object, ...] = ()) -> DatabaseCursor: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class MigrationModule(Protocol):
    def migrate(self, connection: DatabaseConnection, backend: DatabaseBackend) -> None: ...


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path


def _migration_files() -> list[Migration]:
    """Return numbered migrations in application order."""
    migrations: list[Migration] = []
    versions: set[int] = set()
    for path in sorted(MIGRATIONS_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        match = MIGRATION_FILENAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid Grafana migration filename: {path.name}")
        version = int(match.group("version"))
        if version in versions:
            raise ValueError(f"duplicate Grafana migration version: {version}")
        versions.add(version)
        migrations.append(Migration(version=version, name=path.stem, path=path))
    return migrations


def _load_migration(migration: Migration) -> MigrationModule:
    """Load one migration module from its numbered file."""
    spec = importlib.util.spec_from_file_location(migration.name, migration.path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load Grafana migration: {migration.path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(MigrationModule, module)


def _insert_migration(connection: DatabaseConnection, backend: DatabaseBackend, migration: Migration) -> None:
    """Record one applied migration."""
    query = INSERT_SQLITE_MIGRATION if backend == DatabaseBackend.SQLITE else INSERT_POSTGRES_MIGRATION
    connection.execute(query, (migration.version, migration.name))


def migrate(connection: DatabaseConnection, backend: DatabaseBackend) -> list[str]:
    """Apply pending migrations in separate transactions and return their names."""
    connection.execute(CREATE_MIGRATIONS_TABLE)
    connection.commit()
    applied = {int(version): str(name) for version, name in connection.execute(SELECT_APPLIED_MIGRATIONS).fetchall()}
    connection.commit()
    applied_now: list[str] = []
    for migration in _migration_files():
        applied_name = applied.get(migration.version)
        if applied_name is not None:
            if applied_name != migration.name:
                raise ValueError(
                    f"Grafana migration {migration.version} was applied as {applied_name}, not {migration.name}"
                )
            continue

        try:
            if backend == DatabaseBackend.SQLITE:
                connection.execute("BEGIN IMMEDIATE")
            _load_migration(migration).migrate(connection, backend)
            _insert_migration(connection, backend, migration)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        applied_now.append(migration.name)
    return applied_now
