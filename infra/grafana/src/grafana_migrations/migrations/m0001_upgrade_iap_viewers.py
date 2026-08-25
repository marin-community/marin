# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upgrade memberships created before IAP users received the Editor role."""

import logging

from grafana_migrations.engine import DatabaseBackend, DatabaseConnection

UPDATE_VIEWERS = "UPDATE org_user SET role = 'Editor', updated = CURRENT_TIMESTAMP WHERE role = 'Viewer'"
logger = logging.getLogger(__name__)


def migrate(connection: DatabaseConnection, backend: DatabaseBackend) -> None:
    """Upgrade existing Viewer memberships when Grafana's schema is present."""
    if backend == DatabaseBackend.SQLITE:
        table = connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'org_user'").fetchone()
    else:
        table = connection.execute("SELECT to_regclass('public.org_user')").fetchone()
        if table is not None and table[0] is None:
            table = None
    if table is None:
        return

    cursor = connection.execute(UPDATE_VIEWERS)
    logger.info("upgraded %d Grafana Viewer organization memberships to Editor", cursor.rowcount)
