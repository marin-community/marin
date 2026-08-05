# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Object-key layout for a finestore archive, and the ``finestore://`` reference scheme.

An archive owns its root directory. Each table is a subdirectory of immutable Parquet shards,
partitioned by the writer that produced them and the compaction generation they belong to::

    {root}/
        SEALED                                  # optional marker: the run is complete
        {table}/_schema.json                    # merge key + schema/format versions (the only metadata)
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
from urllib.parse import urlsplit

from pydantic import BaseModel

# The seal marker object; its presence means every writer has finished and the run is immutable.
SEALED_MARKER = "SEALED"

# Per-table metadata object. It records only what a Parquet footer cannot: the dedup merge key
# and the schema/format versions. This is the whole "manifest".
SCHEMA_FILE = "_schema.json"

# finestore's own on-disk format version, stamped into every table's metadata so a layout change in a
# future release can be detected when reading an archive an older release wrote.
FORMAT_VERSION = 1

# Columns finestore stamps on every row. ``_seq`` is a per-writer monotonic id used to break ties
# during read-time deduplication; ``_writer`` identifies the producing writer.
SEQ_COLUMN = "_seq"
WRITER_COLUMN = "_writer"
# A read-only column the reader synthesises from the shard's generation; never stored.
GEN_COLUMN = "_gen"

# The URI scheme sample rows use to reference payloads held inside the same archive.
URI_SCHEME = "finestore"

# The reserved table backing ``DataStore.write``: an opaque payload store keyed by blob name. Both the
# writer and the reader (``resolve``/``read_blob``) depend on this contract, so it lives in the layout
# module they both import rather than one importing the other.
BLOBS_TABLE = "blobs"
BLOB_NAME_COLUMN = "name"

_GEN_SEGMENT = re.compile(r"^g=(\d+)$")
_WRITER_SEGMENT = re.compile(r"^w=(.+)$")


class TableMetadata(BaseModel):
    """The typed contents of a table's ``_schema.json`` — the archive's only metadata.

    Records what a Parquet footer cannot: the columns a reader deduplicates on and the versions that
    let the format evolve. ``schema_version`` is the caller's logical schema version; ``format_version``
    is finestore's own on-disk format version.
    """

    merge_key: tuple[str, ...] | None = None
    schema_version: int = 1
    format_version: int = FORMAT_VERSION


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
    def sealed_path(self) -> str:
        """The seal-marker object path for the archive."""
        return f"{self.root.rstrip('/')}/{SEALED_MARKER}"

    def table_dir(self, table: str) -> str:
        """The directory holding one table's shards and metadata."""
        return f"{self.root.rstrip('/')}/{table}"

    def schema_path(self, table: str) -> str:
        """The ``_schema.json`` object path for a table."""
        return f"{self.table_dir(table)}/{SCHEMA_FILE}"

    def shard_path(self, table: str, writer: str, generation: int, seq: int, uid: str) -> str:
        """Build a shard object key. ``seq`` (the batch's minimum ``_seq``) makes keys sort by write order."""
        return f"{self.table_dir(table)}/w={writer}/g={generation}/{seq:016d}-{uid}.parquet"


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
