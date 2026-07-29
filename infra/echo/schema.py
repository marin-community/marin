# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy definitions for the `context` database tables.

The single source of truth for echo's schema. Migrations (migrations/) create and evolve
these tables; the sync job (sync/main.py) uses them for DML. `chunks` mirrors the
marinmirror corpus schema for the github+discord sources; `repository_file_chunks`
indexes GitHub branch heads; `wiki_entries` holds durable agent-authored notes;
`work_log` is the agents' shared logbook; the state tables hold sync watermarks.
"""

from pgvector.sqlalchemy import Vector
from search_config import TEXT_SEARCH_CONFIG
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Computed,
    DateTime,
    Identity,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects import postgresql

EMBED_DIM = 384  # BAAI/bge-small-en-v1.5, the corpus's prose embedding space
MAX_WIKI_TAGS = 20

metadata = MetaData()

chunks = Table(
    "chunks",
    metadata,
    # Chunk ids come from the corpus, never a sequence.
    Column("id", BigInteger, primary_key=True, autoincrement=False),
    Column("source", Text, nullable=False),
    Column("kind", Text, nullable=False),
    Column("ref", Text),
    Column("parent", Text),
    Column("title", Text),
    Column("author", Text),
    Column("date", DateTime(timezone=True)),
    Column("url", Text, nullable=False),
    Column("text", Text),
    Column("hash", Text),
    Column("embedding", Vector(EMBED_DIM)),
    Column(
        "search_document",
        postgresql.TSVECTOR,
        Computed(
            f"setweight(to_tsvector('{TEXT_SEARCH_CONFIG}'::regconfig, coalesce(title, '')), 'A') || "
            f"setweight(to_tsvector('{TEXT_SEARCH_CONFIG}'::regconfig, coalesce(text, '')), 'B')",
            persisted=True,
        ),
    ),
    Column("part", Integer, nullable=False, server_default=text("0")),
    Column("n_parts", Integer, nullable=False, server_default=text("1")),
    Index("idx_chunks_source_kind", "source", "kind"),
    Index("idx_chunks_date", "date"),
    Index("idx_chunks_url", "url"),
    Index("idx_chunks_key", "source", "kind", "ref"),
    Index(
        "idx_chunks_embedding",
        "embedding",
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    ),
    Index("idx_chunks_search_document", "search_document", postgresql_using="gin"),
    # Trigram index so `grep` (text ILIKE '%pattern%') is an index scan, not a seq scan.
    Index("idx_chunks_text_trgm", "text", postgresql_using="gin", postgresql_ops={"text": "gin_trgm_ops"}),
)

repository_file_chunks = Table(
    "repository_file_chunks",
    metadata,
    Column("id", BigInteger, Identity(always=True), primary_key=True),
    Column("repository", Text, nullable=False),
    Column("branch", Text, nullable=False),
    Column("path", Text, nullable=False),
    Column("title", Text, nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("start_line", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("embedding", Vector(EMBED_DIM), nullable=False),
    Column(
        "search_document",
        postgresql.TSVECTOR,
        Computed(
            f"setweight(to_tsvector('{TEXT_SEARCH_CONFIG}'::regconfig, path), 'A') || "
            f"setweight(to_tsvector('{TEXT_SEARCH_CONFIG}'::regconfig, title), 'A') || "
            f"setweight(to_tsvector('{TEXT_SEARCH_CONFIG}'::regconfig, text), 'B')",
            persisted=True,
        ),
    ),
    UniqueConstraint("repository", "branch", "path", "chunk_index", name="repository_file_chunks_path_part"),
    Index("idx_repository_file_chunks_repository_branch", "repository", "branch"),
    Index(
        "idx_repository_file_chunks_embedding",
        "embedding",
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    ),
    Index("idx_repository_file_chunks_search_document", "search_document", postgresql_using="gin"),
    Index(
        "idx_repository_file_chunks_path_trgm",
        "path",
        postgresql_using="gin",
        postgresql_ops={"path": "gin_trgm_ops"},
    ),
    Index(
        "idx_repository_file_chunks_text_trgm",
        "text",
        postgresql_using="gin",
        postgresql_ops={"text": "gin_trgm_ops"},
    ),
)

repository_index_state = Table(
    "repository_index_state",
    metadata,
    Column("repository", Text, primary_key=True),
    Column("branch", Text, primary_key=True),
    Column("commit_sha", Text, nullable=False),
    Column("checked_at", DateTime(timezone=True), nullable=False),
    Column("indexed_at", DateTime(timezone=True), nullable=False),
)

sync_state = Table(
    "sync_state",
    metadata,
    Column("singleton", Boolean, primary_key=True, server_default=text("true")),
    Column("built_at_epoch", BigInteger, nullable=False),
    Column("synced_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint("singleton", name="sync_state_singleton"),
)

work_log = Table(
    "work_log",
    metadata,
    Column("id", BigInteger, Identity(always=True), primary_key=True),
    Column("at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    # Whose agent wrote it, e.g. 'rav/claude-code'.
    Column("author", Text, nullable=False),
    # Stable slug for the thread of work.
    Column("project", Text, nullable=False),
    # One-line summary; body is short markdown with evidence linked inline.
    Column("title", Text, nullable=False),
    Column("body", Text),
    Index("idx_work_log_project_at", "project", text("at DESC")),
    Index("idx_work_log_at", text("at DESC")),
)

wiki_entries = Table(
    "wiki_entries",
    metadata,
    Column("id", BigInteger, Identity(always=True), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("author", Text, nullable=False),
    Column("title", Text, nullable=False),
    Column("use_when", Text, nullable=False),
    Column("tags", ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")),
    Column("body", Text, nullable=False),
    Column("reference_count", BigInteger, nullable=False, server_default=text("0")),
    Column("embedding", Vector(EMBED_DIM), nullable=False),
    Column(
        "search_document",
        postgresql.TSVECTOR,
        Computed(
            f"setweight(to_tsvector('{TEXT_SEARCH_CONFIG}'::regconfig, title), 'A') || "
            f"setweight(to_tsvector('{TEXT_SEARCH_CONFIG}'::regconfig, use_when), 'A') || "
            f"setweight(to_tsvector('{TEXT_SEARCH_CONFIG}'::regconfig, body), 'B')",
            persisted=True,
        ),
    ),
    CheckConstraint("reference_count >= 0", name="wiki_entries_reference_count_nonnegative"),
    CheckConstraint(f"cardinality(tags) <= {MAX_WIKI_TAGS}", name="wiki_entries_tags_limit"),
    Index("idx_wiki_entries_created_at", text("created_at DESC")),
    Index("idx_wiki_entries_tags", "tags", postgresql_using="gin"),
    Index(
        "idx_wiki_entries_embedding",
        "embedding",
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    ),
    Index("idx_wiki_entries_search_document", "search_document", postgresql_using="gin"),
)
