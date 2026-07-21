# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned Postgres migrations with immutable digest checks."""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

MIGRATION_LOCK_ID = 7_418_905_773
_MIGRATION_PATTERN = re.compile(r"^(?P<version>[0-9]{3,})_[a-z0-9_]+\.sql$")


class Cursor(Protocol):
    def __enter__(self) -> "Cursor": ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...

    def execute(self, query: str, params: Sequence[Any] = ()) -> Any: ...

    def fetchone(self) -> Sequence[Any] | None: ...


class Connection(Protocol):
    def cursor(self) -> Cursor: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class MigrationError(RuntimeError):
    """A migration file is invalid, changed, or failed to apply."""


@dataclass(frozen=True)
class Migration:
    sequence: str
    filename: str
    sha256: str
    sql: str


def migration_plan(directory: Path) -> tuple[Migration, ...]:
    """Load the ordered immutable migration plan from a directory."""

    migrations: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        match = _MIGRATION_PATTERN.fullmatch(path.name)
        if match is None:
            raise MigrationError(f"invalid migration filename: {path.name}")
        raw = path.read_bytes()
        migrations.append(
            Migration(
                sequence=match.group("version"),
                filename=path.name,
                sha256=hashlib.sha256(raw).hexdigest(),
                sql=raw.decode(),
            )
        )
    if len({migration.sequence for migration in migrations}) != len(migrations):
        raise MigrationError("migration versions must be unique")
    return tuple(migrations)


def apply_migrations(connection: Connection, migrations: Sequence[Migration]) -> None:
    """Apply immutable migrations once, serializing concurrent callers."""

    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_ID,))
    try:
        _bootstrap_ledger(connection)
        for migration in migrations:
            _apply_migration(connection, migration)
    finally:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_ID,))
        connection.commit()


def _bootstrap_ledger(connection: Connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version text PRIMARY KEY,
                sha256 text NOT NULL,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
    connection.commit()


def _apply_migration(connection: Connection, migration: Migration) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT sha256 FROM schema_migrations WHERE version = %s", (migration.filename,))
        row = cursor.fetchone()
    if row is not None:
        if row[0] != migration.sha256:
            raise MigrationError(f"applied migration changed: {migration.filename}")
        return

    try:
        with connection.cursor() as cursor:
            cursor.execute(migration.sql)
            cursor.execute(
                "INSERT INTO schema_migrations (version, sha256) VALUES (%s, %s)",
                (migration.filename, migration.sha256),
            )
        connection.commit()
    except Exception as error:
        connection.rollback()
        raise MigrationError(f"failed to apply {migration.filename}") from error
