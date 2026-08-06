# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Initial schema: pgvector extension, tables, indexes, and IAM-user grants.

The DDL is frozen at the time this migration was written (compiled from schema.py as of
then) — it must not track later schema.py changes, or a fresh database would receive the
newest schema here and re-apply it in later migrations. Grants target the Cloud SQL IAM
database users that Pulumi creates before migrate.py runs: the `echo@openathena.ai`
group reads the corpus and appends to the logbook (no UPDATE/DELETE — the log is
append-only), and the sync job's service account keeps `chunks`/`sync_state` current.
"""

import sqlalchemy

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id BIGINT NOT NULL,
    source TEXT NOT NULL,
    kind TEXT NOT NULL,
    ref TEXT,
    parent TEXT,
    title TEXT,
    author TEXT,
    date TIMESTAMP WITH TIME ZONE,
    url TEXT NOT NULL,
    text TEXT,
    hash TEXT,
    embedding VECTOR(384),
    part INTEGER DEFAULT 0 NOT NULL,
    n_parts INTEGER DEFAULT 1 NOT NULL,
    PRIMARY KEY (id)
);
CREATE INDEX idx_chunks_source_kind ON chunks (source, kind);
CREATE INDEX idx_chunks_date ON chunks (date);
CREATE INDEX idx_chunks_url ON chunks (url);
CREATE INDEX idx_chunks_key ON chunks (source, kind, ref);
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE sync_state (
    singleton BOOLEAN DEFAULT true NOT NULL,
    built_at_epoch BIGINT NOT NULL,
    synced_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    PRIMARY KEY (singleton),
    CONSTRAINT sync_state_singleton CHECK (singleton)
);

CREATE TABLE work_log (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    author TEXT NOT NULL,
    project TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    PRIMARY KEY (id)
);
CREATE INDEX idx_work_log_project_at ON work_log (project, at DESC);
CREATE INDEX idx_work_log_at ON work_log (at DESC);

GRANT SELECT ON chunks TO "echo@openathena.ai";
GRANT SELECT, INSERT ON work_log TO "echo@openathena.ai";
GRANT SELECT, INSERT, UPDATE, DELETE ON chunks TO "echo-sync@hai-gcp-models.iam";
GRANT SELECT, INSERT, UPDATE ON sync_state TO "echo-sync@hai-gcp-models.iam";
"""


def upgrade(conn: sqlalchemy.Connection) -> None:
    for statement in DDL.strip().split(";"):
        if statement.strip():
            conn.execute(sqlalchemy.text(statement))
