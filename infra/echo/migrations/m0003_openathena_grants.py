# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Move human table privileges to the organization-wide Cloud Identity group.

Cloud SQL IAM database authentication requires a user or group database principal;
it cannot grant PostgreSQL privileges directly to a Google Workspace domain.
`eng-all@openathena.ai` represents the OpenAthena domain at the database boundary.
Pulumi creates both group users before this migration runs.
"""

import sqlalchemy

GRANTS = """
GRANT SELECT ON chunks TO "eng-all@openathena.ai";
GRANT SELECT, INSERT ON work_log TO "eng-all@openathena.ai";
REVOKE SELECT ON chunks FROM "echo@openathena.ai";
REVOKE SELECT, INSERT ON work_log FROM "echo@openathena.ai";
"""


def upgrade(conn: sqlalchemy.Connection) -> None:
    for statement in GRANTS.strip().split(";"):
        if statement.strip():
            conn.execute(sqlalchemy.text(statement))
