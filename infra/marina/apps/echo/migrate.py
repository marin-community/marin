# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply pending migrations to Echo's schema.

Migrations are ``migrations/mNNNN_*.py`` modules, each exposing ``upgrade(conn)``. Applied
names are recorded in ``schema_migrations``, and each pending migration runs in its own
transaction. The ledger and the tables land in the app's own Postgres schema because every
connection on the kernel's engine searches it first.
"""

import importlib
import logging
from pathlib import Path

import sqlalchemy

MIGRATIONS_PACKAGE = "migrations"
MIGRATIONS_DIR = Path(__file__).parent / MIGRATIONS_PACKAGE
LEDGER_DDL = sqlalchemy.text(
    "CREATE TABLE IF NOT EXISTS schema_migrations ("
    "  name text PRIMARY KEY,"
    "  applied_at timestamptz NOT NULL DEFAULT now())"
)

logger = logging.getLogger(__name__)


def migration_names() -> list[str]:
    """Every migration module's name, in application order."""
    return sorted(path.stem for path in MIGRATIONS_DIR.glob("m[0-9]*.py"))


def apply_migrations(engine: sqlalchemy.Engine) -> None:
    """Run every migration this database has not recorded yet."""
    with engine.begin() as conn:
        conn.execute(LEDGER_DDL)
        applied = {row[0] for row in conn.execute(sqlalchemy.text("SELECT name FROM schema_migrations"))}
    for name in migration_names():
        if name in applied:
            continue
        module = importlib.import_module(f"{__package__}.{MIGRATIONS_PACKAGE}.{name}")
        with engine.begin() as conn:
            module.upgrade(conn)
            conn.execute(sqlalchemy.text("INSERT INTO schema_migrations (name) VALUES (:name)"), {"name": name})
        logger.info("applied migration %s", name)
