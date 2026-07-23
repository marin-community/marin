# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Initial schema: pgvector extension, all tables from schema.metadata, and role grants.

Grants live here beside the DDL they depend on (Pulumi owns the roles themselves):
`agents` reads the corpus and appends to the shared logbook — no UPDATE/DELETE, the
log is append-only; `echo_sync` owns keeping `chunks` and `sync_state` current.
"""

import schema
import sqlalchemy

GRANTS = """
GRANT SELECT ON chunks TO agents;
GRANT SELECT, INSERT ON work_log TO agents;
GRANT SELECT, INSERT, UPDATE, DELETE ON chunks TO echo_sync;
GRANT SELECT, INSERT, UPDATE ON sync_state TO echo_sync;
"""


def upgrade(conn: sqlalchemy.Connection) -> None:
    conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector"))
    schema.metadata.create_all(conn)
    for statement in GRANTS.strip().split(";"):
        if statement.strip():
            conn.execute(sqlalchemy.text(statement))
