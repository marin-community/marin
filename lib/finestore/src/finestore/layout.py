# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Object-key layout for a finestore archive, and the ``finestore://`` reference scheme.

An archive owns its root directory. Each table is a subdirectory of immutable Parquet shards,
partitioned by the writer that produced them and the compaction generation they belong to::

    {root}/
        _archive.json                           # archive-wide metadata: the on-disk format version
        SEALED                                  # optional marker: the run is complete
        {table}/_schema.json                    # per-table metadata: primary key + logical schema version
        {table}/w={writer}/g={gen}/{seq:016d}-{uid}.parquet

Shard membership is discovered by listing the table directory; a shard's schema and row-group
statistics come from its Parquet footer, so there is no manifest to keep consistent. The generation
and writer are encoded in the key and recovered by :func:`parse_shard_path`. A caller that shares the
root with sibling data (e.g. an eval run's results directory) passes a dedicated subdirectory as the
root; finestore does not impose one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlsplit

from pydantic import AliasChoices, BaseModel, Field
from rigging.filesystem import prefix_join

# The seal marker object; its presence means every writer has finished and the run is immutable.
SEALED_MARKER = "SEALED"

# Per-table metadata object, next to a table's shards: the dedup primary key and the caller's logical
# schema version -- what a Parquet footer cannot record.
SCHEMA_FILE = "_schema.json"

# Archive-wide metadata object at the root: the on-disk format version. The store writes it once when
# it opens the archive.
ARCHIVE_FILE = "_archive.json"

# finestore's own on-disk format version, recorded in the archive metadata so a reader can refuse an
# archive written in a newer format than it understands.
FORMAT_VERSION = 1

# Columns finestore stamps on every row. ``_seq`` is a per-writer monotonic id used to break ties
# during read-time deduplication; ``_writer`` identifies the producing writer.
SEQ_COLUMN = "_seq"
WRITER_COLUMN = "_writer"
# A read-only column the reader synthesises from the shard's generation; never stored.
GEN_COLUMN = "_gen"

# The URI scheme sample rows use to reference payloads held inside the same archive.
URI_SCHEME = "finestore"

# The reserved table backing ``DataStore.write``: an opaque payload store whose primary key is the blob
# name, so writes batch through the background flusher like any table and a compacted shard is sorted
# by name -- a ``name ==`` lookup then prunes to a single row group via the Parquet footer statistics
# rather than scanning. It is also the ``finestore://blobs/<name>`` URI's netloc. Both the writer and
# the reader (``resolve``/``read_blob``) depend on this contract, so it lives in the layout module they
# both import rather than one importing the other.
BLOBS_TABLE = "blobs"
BLOB_NAME_COLUMN = "name"

_GEN_SEGMENT = re.compile(r"^g=(\d+)$")
_WRITER_SEGMENT = re.compile(r"^w=(.+)$")


class ArchiveMetadata(BaseModel):
    """The typed contents of the archive-wide ``_archive.json``.

    ``format_version`` is finestore's on-disk format version; a reader refuses an archive stamped with
    a version newer than it understands. The store writes this once when it opens the archive.
    """

    format_version: int = FORMAT_VERSION


class OnConflict(StrEnum):
    """What a writer does when two rows in one session share a primary key.

    ``ERROR`` raises, because the reader keeps one row per key and would otherwise discard the other
    without reporting it. A repeat whose payload is identical is an at-least-once redelivery and
    collapses. ``SUPERSEDE`` is the upsert contract: the newer row deliberately replaces the older,
    and compaction counts how many it replaced.
    """

    ERROR = "error"
    SUPERSEDE = "supersede"


class TableMetadata(BaseModel):
    """The typed contents of a table's ``_schema.json``.

    Records what a Parquet footer cannot: ``primary_key`` names the columns identifying a row,
    ``schema_version`` is the caller's logical schema version for the table's rows, and
    ``on_conflict`` is the policy the writer applied to repeated keys. ``merge_key`` is accepted as
    the field name written by finestore writers that predate the rename.
    """

    primary_key: tuple[str, ...] = Field(validation_alias=AliasChoices("primary_key", "merge_key"))
    schema_version: int = 1
    on_conflict: OnConflict = OnConflict.ERROR


class SealMarker(BaseModel):
    """The typed contents of the ``SEALED`` marker: which writer sealed the completed archive.

    ``superseded`` records, per table, how many rows compaction dropped because a later write of the
    same primary key replaced them. A non-zero count on an ``ERROR`` table means rows were replaced
    across sessions (a re-run or a resumed migration), which is legal but never silent.
    """

    writer: str
    superseded: dict[str, int] = {}


@dataclass(frozen=True)
class Shard:
    """One discovered Parquet shard: its object path and the writer/generation encoded in the key."""

    path: str
    writer: str
    generation: int


@dataclass(frozen=True)
class FineStoreLayout:
    """Every root-relative object key an archive uses.

    An archive owns its ``root`` directory. Construct one layout per archive and call its methods
    rather than threading ``root`` through free functions.
    """

    root: str

    @property
    def archive_path(self) -> str:
        """The archive-wide metadata object path."""
        return prefix_join(self.root, ARCHIVE_FILE)

    @property
    def sealed_path(self) -> str:
        """The seal-marker object path for the archive."""
        return prefix_join(self.root, SEALED_MARKER)

    def table_dir(self, table: str) -> str:
        """The directory holding one table's shards and metadata."""
        return prefix_join(self.root, table)

    def schema_path(self, table: str) -> str:
        """The ``_schema.json`` object path for a table."""
        return prefix_join(self.table_dir(table), SCHEMA_FILE)

    def shard_path(self, table: str, writer: str, generation: int, seq: int, uid: str) -> str:
        """Build a shard object key. ``seq`` (the batch's minimum ``_seq``) makes keys sort by write order."""
        return prefix_join(self.table_dir(table), f"w={writer}/g={generation}/{seq:016d}-{uid}.parquet")


def parse_shard_path(path: str) -> Shard | None:
    """Recover ``(writer, generation)`` from a shard key, or ``None`` if it is not a shard.

    Tolerates the ``_schema.json`` metadata object and any non-Parquet key by returning ``None``.
    """
    if not path.endswith(".parquet"):
        return None
    segments = path.split("/")
    writer: str | None = None
    generation: int | None = None
    for segment in segments:
        writer_match = _WRITER_SEGMENT.match(segment)
        if writer_match:
            writer = writer_match.group(1)
        gen_match = _GEN_SEGMENT.match(segment)
        if gen_match:
            generation = int(gen_match.group(1))
    if writer is None or generation is None:
        return None
    return Shard(path=path, writer=writer, generation=generation)


@dataclass(frozen=True)
class ArchiveRef:
    """A parsed ``finestore://<table>/<key>`` reference into an archive."""

    table: str
    key: str


def build_uri(table: str, key: str) -> str:
    """Format a ``finestore://<table>/<key>`` reference (e.g. a blob a sample row points at)."""
    return f"{URI_SCHEME}://{table}/{key.lstrip('/')}"


def parse_uri(uri: str) -> ArchiveRef | None:
    """Parse a ``finestore://`` reference into ``(table, key)``, or ``None`` if it is not one."""
    parts = urlsplit(uri)
    if parts.scheme != URI_SCHEME:
        return None
    return ArchiveRef(table=parts.netloc, key=parts.path.lstrip("/"))
