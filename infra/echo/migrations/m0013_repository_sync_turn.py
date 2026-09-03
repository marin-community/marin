# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist the next repository turn for the serialized sync job."""

import sqlalchemy

DDL = """
CREATE TABLE repository_sync_turn (
    singleton BOOLEAN PRIMARY KEY DEFAULT true CHECK (singleton),
    next_target INTEGER NOT NULL DEFAULT 0 CHECK (next_target >= 0)
);

GRANT SELECT ON repository_sync_turn
    TO "eng-all@openathena.ai";
GRANT SELECT ON repository_sync_turn
    TO "loom-vm@hai-gcp-models.iam";
GRANT SELECT, INSERT, UPDATE ON repository_sync_turn
    TO "echo-sync@hai-gcp-models.iam";
"""


def upgrade(conn: sqlalchemy.Connection) -> None:
    for statement in DDL.strip().split(";"):
        if statement.strip():
            conn.execute(sqlalchemy.text(statement))
