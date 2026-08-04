# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add the rolling GitHub repository file index."""

import sqlalchemy

DDL = """
CREATE TABLE repository_file_chunks (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    repository TEXT NOT NULL,
    branch TEXT NOT NULL,
    path TEXT NOT NULL,
    title TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    search_document TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english'::regconfig, path), 'A') ||
        setweight(to_tsvector('english'::regconfig, title), 'A') ||
        setweight(to_tsvector('english'::regconfig, text), 'B')
    ) STORED,
    PRIMARY KEY (id),
    CONSTRAINT repository_file_chunks_path_part
        UNIQUE (repository, branch, path, chunk_index)
);
CREATE INDEX idx_repository_file_chunks_repository_branch
    ON repository_file_chunks (repository, branch);
CREATE INDEX idx_repository_file_chunks_embedding
    ON repository_file_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_repository_file_chunks_search_document
    ON repository_file_chunks USING gin (search_document);
CREATE INDEX idx_repository_file_chunks_path_trgm
    ON repository_file_chunks USING gin (path gin_trgm_ops);
CREATE INDEX idx_repository_file_chunks_text_trgm
    ON repository_file_chunks USING gin (text gin_trgm_ops);

CREATE TABLE repository_index_state (
    repository TEXT NOT NULL,
    branch TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    checked_at TIMESTAMP WITH TIME ZONE NOT NULL,
    indexed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (repository, branch)
);

GRANT SELECT ON repository_file_chunks, repository_index_state
    TO "eng-all@openathena.ai";
GRANT SELECT ON repository_file_chunks, repository_index_state
    TO "loom-vm@hai-gcp-models.iam";
GRANT SELECT ON repository_file_chunks, repository_index_state
    TO "echo-api@hai-gcp-models.iam";
GRANT SELECT, INSERT, UPDATE, DELETE ON repository_file_chunks, repository_index_state
    TO "echo-sync@hai-gcp-models.iam";
GRANT USAGE, SELECT ON SEQUENCE repository_file_chunks_id_seq
    TO "echo-sync@hai-gcp-models.iam";
"""


def upgrade(conn: sqlalchemy.Connection) -> None:
    for statement in DDL.strip().split(";"):
        if statement.strip():
            conn.execute(sqlalchemy.text(statement))
