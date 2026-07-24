#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cloud-sql-python-connector[pg8000]>=1.9",
#     "google-cloud-secret-manager>=2.20",
#     "sqlalchemy>=2",
#     "pgvector>=0.3",
# ]
# ///
"""Apply pending migrations to the `context` database.

Migrations are `migrations/mNNNN_*.py` modules, each exposing `upgrade(conn)`; applied
names are recorded in `schema_migrations`, and each pending migration runs in its own
transaction. Connects as `pulumi_db_admin` — the pre-existing marin-metadata system role
that owns the tables and can create extensions — through the Cloud SQL connector.

    infra/echo/migrate.py            # apply pending
    infra/echo/migrate.py --list     # show applied/pending and exit
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import sqlalchemy
from google.cloud.secretmanager import SecretManagerServiceClient
from google.cloud.sql.connector import Connector

PROJECT = "hai-gcp-models"
INSTANCE = f"{PROJECT}:us-central1:marin-metadata"
DATABASE = "context"
ADMIN_USER = "pulumi_db_admin"
ADMIN_PASSWORD_SECRET = "cloudsql-pulumi-admin-password"
MIGRATIONS = Path(__file__).parent / "migrations"

sys.path.insert(0, str(Path(__file__).parent))  # migrations import the sibling schema.py


def admin_password() -> str:
    name = f"projects/{PROJECT}/secrets/{ADMIN_PASSWORD_SECRET}/versions/latest"
    return SecretManagerServiceClient().access_secret_version(name=name).payload.data.decode()


def load_migration(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply pending context-database migrations.")
    parser.add_argument("--list", action="store_true", help="show applied/pending and exit")
    args = parser.parse_args()

    connector = Connector(quota_project=PROJECT)
    try:
        engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=lambda: connector.connect(
                INSTANCE, "pg8000", user=ADMIN_USER, password=admin_password(), db=DATABASE
            ),
        )
        with engine.begin() as conn:
            conn.execute(
                sqlalchemy.text(
                    "CREATE TABLE IF NOT EXISTS schema_migrations ("
                    "  name text PRIMARY KEY,"
                    "  applied_at timestamptz NOT NULL DEFAULT now())"
                )
            )
            applied = {r[0] for r in conn.execute(sqlalchemy.text("SELECT name FROM schema_migrations"))}

        for path in sorted(MIGRATIONS.glob("m[0-9]*.py")):
            if path.stem in applied:
                print(f"applied  {path.stem}")
                continue
            if args.list:
                print(f"pending  {path.stem}")
                continue
            with engine.begin() as conn:
                load_migration(path).upgrade(conn)
                conn.execute(sqlalchemy.text("INSERT INTO schema_migrations (name) VALUES (:name)"), {"name": path.stem})
            print(f"applied  {path.stem} (new)")
    finally:
        connector.close()


if __name__ == "__main__":
    main()
