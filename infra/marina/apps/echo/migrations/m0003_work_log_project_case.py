# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Store existing work-log project slugs in the trimmed lowercase form the API now writes."""

import sqlalchemy


def upgrade(conn: sqlalchemy.Connection) -> None:
    conn.execute(sqlalchemy.text("SET LOCAL lock_timeout = '5s'"))
    conn.execute(
        sqlalchemy.text("UPDATE work_log SET project = lower(btrim(project)) WHERE project <> lower(btrim(project))")
    )
