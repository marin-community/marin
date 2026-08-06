# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply pending Marin Grafana database migrations at container startup."""

import logging
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import psycopg

from grafana_migrations import DatabaseBackend, migrate

DEFAULT_GRAFANA_DATA_PATH = Path("/var/lib/grafana")
logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    database_url = os.environ.get("GF_DATABASE_URL")
    if database_url:
        with psycopg.connect(database_url) as connection:
            applied = migrate(connection, DatabaseBackend.POSTGRES)
    else:
        data_path = Path(os.environ.get("GF_PATHS_DATA", str(DEFAULT_GRAFANA_DATA_PATH)))
        with closing(sqlite3.connect(data_path / "grafana.db")) as connection, connection:
            applied = migrate(connection, DatabaseBackend.SQLITE)

    logger.info("applied %d Marin Grafana database migrations: %s", len(applied), applied)


if __name__ == "__main__":
    main()
