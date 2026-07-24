# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grant the echo-api service account corpus read and logbook append.

Added when the HTTP API service (infra/echo/api) was introduced; m0001 predates it, so
its grants live here rather than being edited into an already-applied migration. The
`echo-api@hai-gcp-models.iam` database user is created by Pulumi before migrate.py runs.
"""

import sqlalchemy

GRANTS = """
GRANT SELECT ON chunks TO "echo-api@hai-gcp-models.iam";
GRANT SELECT, INSERT ON work_log TO "echo-api@hai-gcp-models.iam";
"""


def upgrade(conn: sqlalchemy.Connection) -> None:
    for statement in GRANTS.strip().split(";"):
        if statement.strip():
            conn.execute(sqlalchemy.text(statement))
