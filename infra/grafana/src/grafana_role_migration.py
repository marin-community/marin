# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upgrade persisted Grafana viewers to the IAP-authenticated organization role."""

import logging
import os
import sqlite3
from pathlib import Path

import psycopg

DEFAULT_GRAFANA_DATA_PATH = Path("/var/lib/grafana")
UPDATE_VIEWERS = "UPDATE org_user SET role = 'Editor', updated = CURRENT_TIMESTAMP WHERE role = 'Viewer'"


def upgrade_sqlite(database_path: Path) -> int:
    """Upgrade Viewer memberships in a SQLite Grafana database.

    Returns:
        Number of upgraded memberships, or zero when the schema is absent.
    """
    with sqlite3.connect(database_path) as connection:
        table = connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'org_user'").fetchone()
        if table is None:
            return 0

        cursor = connection.execute(UPDATE_VIEWERS)
        return cursor.rowcount


def upgrade_postgres(database_url: str) -> int:
    """Upgrade Viewer memberships in a PostgreSQL Grafana database.

    Returns:
        Number of upgraded memberships, or zero when the schema is absent.
    """
    with psycopg.connect(database_url) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass('public.org_user')")
        table = cursor.fetchone()
        if table is None or table[0] is None:
            return 0

        cursor.execute(UPDATE_VIEWERS)
        return cursor.rowcount


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    database_url = os.environ.get("GF_DATABASE_URL")
    if database_url:
        upgraded = upgrade_postgres(database_url)
    else:
        data_path = Path(os.environ.get("GF_PATHS_DATA", str(DEFAULT_GRAFANA_DATA_PATH)))
        upgraded = upgrade_sqlite(data_path / "grafana.db")

    logging.info("upgraded %d Grafana Viewer organization memberships to Editor", upgraded)


if __name__ == "__main__":
    main()
