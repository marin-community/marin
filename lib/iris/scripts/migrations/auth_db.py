#!/usr/bin/env python3

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Migrate auth tables from the main controller DB to a separate auth.sqlite3.

One-time script for operators upgrading from single-DB deployments to the
split auth DB layout. Copies api_keys and controller_secrets rows from main
to a new auth.sqlite3, then drops those tables from main.

Idempotent: skips if the tables don't exist in main.

Usage:
  uv run python lib/iris/scripts/migrations/auth_db.py \
      --config lib/iris/examples/marin.yaml

  # Use an already-downloaded sqlite file
  uv run python lib/iris/scripts/migrations/auth_db.py \
      --config lib/iris/examples/marin.yaml --local-db /tmp/controller.sqlite3

  # Dry run
  uv run python lib/iris/scripts/migrations/auth_db.py \
      --config lib/iris/examples/marin.yaml --dry-run
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import tempfile
from pathlib import Path

import click

from iris.cluster.config import load_config
from iris.scripts.migrations.gcloud_helpers import discover_controller_vm, scp_from_controller

logger = logging.getLogger("migrate-auth-db")

CONTROLLER_STATE_DIR = "/var/cache/iris/controller"
AUTH_TABLES = ("api_keys", "controller_secrets")


def _table_exists(conn: sqlite3.Connection, table: str, schema: str = "main") -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def migrate_auth_tables(db_path: Path, dry_run: bool = False) -> int:
    """Move api_keys and controller_secrets from main DB to auth.sqlite3.

    Returns the number of rows migrated.
    """
    auth_path = db_path.with_name("auth.sqlite3")
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("ATTACH DATABASE ? AS auth", (str(auth_path),))

        # Create auth schema tables
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS auth.api_keys (
                key_id TEXT PRIMARY KEY,
                key_hash TEXT NOT NULL UNIQUE,
                key_prefix TEXT NOT NULL,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                last_used_at_ms INTEGER,
                expires_at_ms INTEGER,
                revoked_at_ms INTEGER
            );
            CREATE INDEX IF NOT EXISTS auth.idx_api_keys_hash ON api_keys(key_hash);
            CREATE INDEX IF NOT EXISTS auth.idx_api_keys_user ON api_keys(user_id);
            CREATE TABLE IF NOT EXISTS auth.controller_secrets (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL
            );
        """
        )

        total_rows = 0
        for table in AUTH_TABLES:
            if not _table_exists(conn, table, "main"):
                logger.info("Table %s not found in main DB, skipping", table)
                continue

            count = conn.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
            if dry_run:
                logger.info("[dry-run] Would migrate %d rows from main.%s to auth.%s", count, table, table)
            else:
                conn.execute(f"INSERT OR IGNORE INTO auth.{table} SELECT * FROM main.{table}")
                conn.execute(f"DROP TABLE main.{table}")
                logger.info("Migrated %d rows from main.%s to auth.%s", count, table, table)
            total_rows += count

        if not dry_run:
            conn.commit()

        return total_rows
    finally:
        conn.close()


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option(
    "--local-db",
    "local_db_path",
    type=click.Path(),
    default=None,
    help="Use local sqlite file instead of SCP from controller",
)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--project", default="hai-gcp-models")
@click.option("-v", "--verbose", is_flag=True, default=False)
def main(
    config_path: str,
    local_db_path: str | None,
    dry_run: bool,
    project: str,
    verbose: bool,
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    cluster_config = load_config(Path(config_path))
    label_prefix = cluster_config.platform.label_prefix or "iris"

    if local_db_path:
        db_path = Path(local_db_path)
        if not db_path.exists():
            raise click.ClickException(f"Local DB not found: {db_path}")
    else:
        click.echo("Discovering controller VM...")
        vm_info = discover_controller_vm(project, label_prefix)
        if vm_info is None:
            raise click.ClickException(f"No controller VM found with label iris-{label_prefix}-controller=true")
        vm_name, zone = vm_info
        click.echo(f"  VM: {vm_name} ({zone})")

        remote_sqlite = f"{CONTROLLER_STATE_DIR}/controller.sqlite3"
        tmp_dir = tempfile.mkdtemp(prefix="iris_auth_migration_")
        db_path = Path(tmp_dir) / "controller.sqlite3"

        if dry_run:
            click.echo("[dry-run] Would SCP controller.sqlite3 — cannot proceed")
            return
        click.echo(f"SCP {vm_name}:{remote_sqlite} -> {db_path}")
        scp_from_controller(vm_name, zone, project, remote_sqlite, db_path)

    total = migrate_auth_tables(db_path, dry_run=dry_run)
    auth_path = db_path.with_name("auth.sqlite3")
    click.echo(f"Migrated {total} rows to {auth_path}")


if __name__ == "__main__":
    main()
