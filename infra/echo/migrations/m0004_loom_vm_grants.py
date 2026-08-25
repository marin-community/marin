# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grant Loom's VM service account corpus read and logbook append access."""

import sqlalchemy

GRANTS = """
GRANT SELECT ON chunks TO "loom-vm@hai-gcp-models.iam";
GRANT SELECT, INSERT ON work_log TO "loom-vm@hai-gcp-models.iam";
"""


def upgrade(conn: sqlalchemy.Connection) -> None:
    for statement in GRANTS.strip().split(";"):
        if statement.strip():
            conn.execute(sqlalchemy.text(statement))
