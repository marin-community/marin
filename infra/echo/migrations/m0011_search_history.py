# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist search executions and their ranked results."""

import sqlalchemy

DDL = """
CREATE TABLE search_executions (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    author TEXT,
    query TEXT NOT NULL,
    normalized_query TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('federated', 'activity', 'grep')),
    domains TEXT[] NOT NULL DEFAULT '{}',
    filters JSONB NOT NULL DEFAULT '{}'::jsonb,
    requested_limit INTEGER NOT NULL CHECK (requested_limit > 0),
    returned_count INTEGER NOT NULL CHECK (returned_count >= 0),
    duration_ms DOUBLE PRECISION NOT NULL CHECK (duration_ms >= 0),
    repository_commit TEXT,
    service_revision TEXT,
    PRIMARY KEY (id)
);
CREATE INDEX idx_search_executions_created_at ON search_executions (created_at DESC);
CREATE INDEX idx_search_executions_normalized_query ON search_executions (normalized_query);
CREATE INDEX idx_search_executions_mode_created_at ON search_executions (mode, created_at DESC);

CREATE TABLE search_execution_results (
    execution_id BIGINT NOT NULL REFERENCES search_executions(id) ON DELETE CASCADE,
    rank SMALLINT NOT NULL CHECK (rank > 0),
    result_id TEXT NOT NULL,
    domain TEXT NOT NULL CHECK (domain IN ('wiki', 'file', 'discord', 'pr', 'issue')),
    title TEXT,
    url TEXT NOT NULL,
    snippet TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    distance DOUBLE PRECISION,
    lexical_score DOUBLE PRECISION,
    PRIMARY KEY (execution_id, rank)
);
CREATE INDEX idx_search_execution_results_result_id ON search_execution_results (result_id);

ALTER TABLE search_feedback
    ADD COLUMN execution_id BIGINT REFERENCES search_executions(id) ON DELETE SET NULL;
CREATE INDEX idx_search_feedback_execution_id ON search_feedback (execution_id);

GRANT SELECT ON search_executions, search_execution_results
    TO "eng-all@openathena.ai";
GRANT SELECT ON search_executions, search_execution_results
    TO "loom-vm@hai-gcp-models.iam";
GRANT SELECT, INSERT ON search_executions, search_execution_results
    TO "echo-api@hai-gcp-models.iam";
GRANT USAGE, SELECT ON SEQUENCE search_executions_id_seq
    TO "echo-api@hai-gcp-models.iam";
"""


def upgrade(conn: sqlalchemy.Connection) -> None:
    for statement in DDL.strip().split(";"):
        if statement.strip():
            conn.execute(sqlalchemy.text(statement))
