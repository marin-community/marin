# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Echo schema: the corpus, the repository file index, the wiki, and search history.

The DDL is frozen as it stood when Echo moved into Marina, compiled from ``schema.py``
at that point. It must not track later ``schema.py`` changes: a fresh database would
then get the newest schema here and receive it again from the migration that follows.
Every table lands in the app's own schema because the engine's ``search_path`` names it;
the extensions are shared and stay in ``public``.
"""

import sqlalchemy

DDL = """
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public;

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
    search_document TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english'::regconfig, coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english'::regconfig, coalesce(text, '')), 'B')
    ) STORED,
    part INTEGER DEFAULT 0 NOT NULL,
    n_parts INTEGER DEFAULT 1 NOT NULL,
    PRIMARY KEY (id)
);
CREATE INDEX idx_chunks_date ON chunks (date);
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_key ON chunks (source, kind, ref);
CREATE INDEX idx_chunks_search_document ON chunks USING gin (search_document);
CREATE INDEX idx_chunks_source_kind ON chunks (source, kind);
CREATE INDEX idx_chunks_text_trgm ON chunks USING gin (text gin_trgm_ops);
CREATE INDEX idx_chunks_url ON chunks (url);

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
    CONSTRAINT repository_file_chunks_path_part UNIQUE (repository, branch, path, chunk_index)
);
CREATE INDEX idx_repository_file_chunks_embedding ON repository_file_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_repository_file_chunks_path_trgm ON repository_file_chunks USING gin (path gin_trgm_ops);
CREATE INDEX idx_repository_file_chunks_repository_branch ON repository_file_chunks (repository, branch);
CREATE INDEX idx_repository_file_chunks_search_document ON repository_file_chunks USING gin (search_document);
CREATE INDEX idx_repository_file_chunks_text_trgm ON repository_file_chunks USING gin (text gin_trgm_ops);

CREATE TABLE repository_index_builds (
    repository TEXT NOT NULL,
    branch TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    base_sha TEXT,
    mode TEXT NOT NULL,
    total_files INTEGER NOT NULL,
    completed_files INTEGER DEFAULT 0 NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (repository, branch)
);

CREATE TABLE repository_index_state (
    repository TEXT NOT NULL,
    branch TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    checked_at TIMESTAMP WITH TIME ZONE NOT NULL,
    indexed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (repository, branch)
);

CREATE TABLE repository_sync_turn (
    singleton BOOLEAN DEFAULT true NOT NULL,
    next_target INTEGER DEFAULT 0 NOT NULL,
    PRIMARY KEY (singleton),
    CONSTRAINT repository_sync_turn_singleton CHECK (singleton),
    CONSTRAINT repository_sync_turn_next_target_nonnegative CHECK (next_target >= 0)
);

CREATE TABLE search_executions (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    author TEXT,
    query TEXT NOT NULL,
    normalized_query TEXT NOT NULL,
    mode TEXT NOT NULL,
    domains TEXT[] DEFAULT '{}'::text[] NOT NULL,
    filters JSONB DEFAULT '{}'::jsonb NOT NULL,
    requested_limit INTEGER NOT NULL,
    returned_count INTEGER NOT NULL,
    duration_ms FLOAT NOT NULL,
    repository_commit TEXT,
    service_revision TEXT,
    PRIMARY KEY (id),
    CONSTRAINT search_executions_mode CHECK (mode IN ('federated', 'activity', 'grep')),
    CONSTRAINT search_executions_requested_limit_positive CHECK (requested_limit > 0),
    CONSTRAINT search_executions_returned_count_nonnegative CHECK (returned_count >= 0),
    CONSTRAINT search_executions_duration_nonnegative CHECK (duration_ms >= 0)
);
CREATE INDEX idx_search_executions_created_at ON search_executions (created_at DESC);
CREATE INDEX idx_search_executions_mode_created_at ON search_executions (mode, created_at DESC);
CREATE INDEX idx_search_executions_normalized_query ON search_executions (normalized_query);

CREATE TABLE sync_state (
    singleton BOOLEAN DEFAULT true NOT NULL,
    built_at_epoch BIGINT NOT NULL,
    synced_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    PRIMARY KEY (singleton),
    CONSTRAINT sync_state_singleton CHECK (singleton)
);

CREATE TABLE wiki_entries (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    author TEXT NOT NULL,
    title TEXT NOT NULL,
    use_when TEXT NOT NULL,
    tags TEXT[] DEFAULT '{}'::text[] NOT NULL,
    body TEXT NOT NULL,
    reference_count BIGINT DEFAULT 0 NOT NULL,
    embedding VECTOR(384) NOT NULL,
    search_document TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english'::regconfig, title), 'A') ||
        setweight(to_tsvector('english'::regconfig, use_when), 'A') ||
        setweight(to_tsvector('english'::regconfig, body), 'B')
    ) STORED,
    PRIMARY KEY (id),
    CONSTRAINT wiki_entries_reference_count_nonnegative CHECK (reference_count >= 0),
    CONSTRAINT wiki_entries_tags_limit CHECK (cardinality(tags) <= 20)
);
CREATE INDEX idx_wiki_entries_created_at ON wiki_entries (created_at DESC);
CREATE INDEX idx_wiki_entries_embedding ON wiki_entries USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_wiki_entries_search_document ON wiki_entries USING gin (search_document);
CREATE INDEX idx_wiki_entries_tags ON wiki_entries USING gin (tags);

CREATE TABLE work_log (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    author TEXT NOT NULL,
    project TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    PRIMARY KEY (id)
);
CREATE INDEX idx_work_log_at ON work_log (at DESC);
CREATE INDEX idx_work_log_project_at ON work_log (project, at DESC);

CREATE TABLE search_execution_results (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    execution_id BIGINT NOT NULL,
    rank SMALLINT NOT NULL,
    result_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    title TEXT,
    url TEXT NOT NULL,
    snippet TEXT NOT NULL,
    score FLOAT NOT NULL,
    distance FLOAT,
    lexical_score FLOAT,
    rerank_score FLOAT,
    PRIMARY KEY (id),
    CONSTRAINT search_execution_results_execution_rank UNIQUE (execution_id, rank),
    CONSTRAINT search_execution_results_rank_positive CHECK (rank > 0),
    CONSTRAINT search_execution_results_domain CHECK (domain IN ('wiki', 'file', 'discord', 'pr', 'issue')),
    FOREIGN KEY(execution_id) REFERENCES search_executions (id) ON DELETE CASCADE
);
CREATE INDEX idx_search_execution_results_result_id ON search_execution_results (result_id);

CREATE TABLE search_feedback (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    author TEXT NOT NULL,
    query TEXT NOT NULL,
    note TEXT NOT NULL,
    execution_id BIGINT,
    PRIMARY KEY (id),
    FOREIGN KEY(execution_id) REFERENCES search_executions (id) ON DELETE SET NULL
);
CREATE INDEX idx_search_feedback_created_at ON search_feedback (created_at DESC);
CREATE INDEX idx_search_feedback_execution_id ON search_feedback (execution_id);

CREATE TABLE search_feedback_grades (
    feedback_id BIGINT NOT NULL,
    result_id TEXT NOT NULL,
    search_result_id BIGINT,
    grade SMALLINT NOT NULL,
    PRIMARY KEY (feedback_id, result_id),
    CONSTRAINT search_feedback_grades_range CHECK (grade BETWEEN 0 AND 10),
    CONSTRAINT search_feedback_grades_search_result UNIQUE (feedback_id, search_result_id),
    FOREIGN KEY(feedback_id) REFERENCES search_feedback (id) ON DELETE CASCADE,
    FOREIGN KEY(search_result_id) REFERENCES search_execution_results (id) ON DELETE SET NULL
);
CREATE INDEX idx_search_feedback_grades_result_id ON search_feedback_grades (result_id);
CREATE INDEX idx_search_feedback_grades_search_result_id ON search_feedback_grades (search_result_id);
"""


def upgrade(conn: sqlalchemy.Connection) -> None:
    for statement in DDL.strip().split(";"):
        if statement.strip():
            conn.execute(sqlalchemy.text(statement))
