# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add weighted full-text search to corpus chunks and create the agent wiki."""

import sqlalchemy

DDL = """
ALTER TABLE chunks ADD COLUMN search_document TSVECTOR
GENERATED ALWAYS AS (
    setweight(to_tsvector('english'::regconfig, coalesce(title, '')), 'A') ||
    setweight(to_tsvector('english'::regconfig, coalesce(text, '')), 'B')
) STORED;
CREATE INDEX idx_chunks_search_document ON chunks USING gin (search_document);

CREATE TABLE wiki_entries (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    author TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    reference_count BIGINT DEFAULT 0 NOT NULL,
    embedding VECTOR(384) NOT NULL,
    search_document TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english'::regconfig, title), 'A') ||
        setweight(to_tsvector('english'::regconfig, body), 'B')
    ) STORED,
    PRIMARY KEY (id),
    CONSTRAINT wiki_entries_reference_count_nonnegative CHECK (reference_count >= 0)
);
CREATE INDEX idx_wiki_entries_created_at ON wiki_entries (created_at DESC);
CREATE INDEX idx_wiki_entries_embedding ON wiki_entries USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_wiki_entries_search_document ON wiki_entries USING gin (search_document);

GRANT SELECT ON wiki_entries TO "eng-all@openathena.ai";
GRANT SELECT, INSERT, UPDATE ON wiki_entries TO "echo-api@hai-gcp-models.iam";
GRANT USAGE, SELECT ON SEQUENCE wiki_entries_id_seq TO "echo-api@hai-gcp-models.iam";
"""


def upgrade(conn: sqlalchemy.Connection) -> None:
    for statement in DDL.strip().split(";"):
        if statement.strip():
            conn.execute(sqlalchemy.text(statement))
