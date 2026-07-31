# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Track resumable repository index builds."""

import sqlalchemy

DDL = """
CREATE TABLE repository_index_builds (
    repository TEXT NOT NULL,
    branch TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    base_sha TEXT,
    mode TEXT NOT NULL CHECK (mode IN ('full', 'incremental')),
    total_files INTEGER NOT NULL,
    completed_files INTEGER NOT NULL DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (repository, branch)
);

GRANT SELECT ON repository_index_builds
    TO "eng-all@openathena.ai";
GRANT SELECT ON repository_index_builds
    TO "loom-vm@hai-gcp-models.iam";
GRANT SELECT ON repository_index_builds
    TO "echo-api@hai-gcp-models.iam";
GRANT SELECT, INSERT, UPDATE, DELETE ON repository_index_builds
    TO "echo-sync@hai-gcp-models.iam";
"""


def upgrade(conn: sqlalchemy.Connection) -> None:
    for statement in DDL.strip().split(";"):
        if statement.strip():
            conn.execute(sqlalchemy.text(statement))
