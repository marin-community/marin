# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply EvalDash database migrations during application startup."""

import importlib.util
from pathlib import Path

import sqlalchemy
from sqlalchemy.engine import Connection, Engine

MIGRATIONS = Path(__file__).with_name("migrations")
MIGRATION_GLOB = "[0-9][0-9][0-9][0-9]_*.py"
_MIGRATION_LOCK = "evaldash-schema-migrations"

schema_migrations = sqlalchemy.Table(
    "schema_migrations",
    sqlalchemy.MetaData(),
    sqlalchemy.Column("name", sqlalchemy.Text, primary_key=True),
    sqlalchemy.Column(
        "applied_at", sqlalchemy.DateTime(timezone=True), nullable=False, server_default=sqlalchemy.func.now()
    ),
)


def _load_migration(path: Path):
    spec = importlib.util.spec_from_file_location(f"evaldash_migration_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lock_migrations(conn: Connection) -> None:
    if conn.dialect.name == "postgresql":
        conn.execute(sqlalchemy.text("SELECT pg_advisory_xact_lock(hashtext(:name))"), {"name": _MIGRATION_LOCK})


def apply_migrations(engine: Engine) -> None:
    """Apply each pending numbered migration exactly once in one transaction."""
    with engine.begin() as conn:
        _lock_migrations(conn)
        schema_migrations.create(conn, checkfirst=True)
        applied = {row[0] for row in conn.execute(sqlalchemy.select(schema_migrations.c.name))}
        paths = sorted(MIGRATIONS.glob(MIGRATION_GLOB))
        known = {path.stem for path in paths}
        if unknown := applied - known:
            raise RuntimeError(f"EvalDash database has migrations unknown to this build: {sorted(unknown)}")
        for path in paths:
            if path.stem in applied:
                continue
            _load_migration(path).upgrade(conn)
            conn.execute(schema_migrations.insert().values(name=path.stem))
