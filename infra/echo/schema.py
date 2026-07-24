# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLAlchemy definitions for the `context` database tables.

The single source of truth for echo's schema. Migrations (migrations/) create and evolve
these tables; the sync job (sync/main.py) uses them for DML. `chunks` mirrors the
marinmirror corpus schema for the github+discord sources; `work_log` is the agents'
shared logbook; `sync_state` is the corpus-build watermark.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Identity,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    func,
    text,
)

EMBED_DIM = 384  # BAAI/bge-small-en-v1.5, the corpus's prose embedding space

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
