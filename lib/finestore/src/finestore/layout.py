# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FineStore format-v2 metadata and object-key layout.

Data, schemas, and manifests are immutable. ``HEAD`` is the archive's only mutable
object and points at one complete manifest, making its conditional-write version the
commit token for every table and object in a transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.storage_path import prefix_join

ARCHIVE_FILE = "_archive.json"
HEAD_FILE = "HEAD"
FORMAT_VERSION = 2
DATA_DIR = "data"
MANIFESTS_DIR = "manifests"
SCHEMAS_DIR = "schemas"

SEQ_COLUMN = "_seq"
WRITER_COLUMN = "_writer"
GEN_COLUMN = "_gen"
COMMIT_COLUMN = "_commit"
RESERVED_COLUMNS = frozenset({SEQ_COLUMN, WRITER_COLUMN, GEN_COLUMN, COMMIT_COLUMN})

SUPPORTED_MANIFEST_FEATURES: frozenset[str] = frozenset()

URI_SCHEME = "finestore"
BLOBS_TABLE = "blobs"
BLOB_NAME_COLUMN = "name"
BLOB_DATA_COLUMN = "data"


class FormatVersionError(ValueError):
    """An archive requires migration or a newer FineStore build."""

    def __init__(self, root: str, *, found: int) -> None:
        self.root = root
        self.found = found
        if found < FORMAT_VERSION:
            action = "migrate it explicitly with finestore.migrations.migrate(root) before opening it"
        else:
            action = "upgrade marin-finestore before opening it"
        super().__init__(
            f"archive at {root!r} is FineStore format v{found}; this build supports v{FORMAT_VERSION}; {action}"
        )


class _FormatModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ArchiveMetadata(_FormatModel):
    """The archive-wide format marker advanced by format migrations."""

    format_version: int = FORMAT_VERSION


class OnConflict(StrEnum):
    """How repeated primary keys are handled."""

    ERROR = "error"
    SUPERSEDE = "supersede"


class TableMetadata(_FormatModel):
    """Logical table metadata stored in an immutable schema object."""

    primary_key: tuple[str, ...]
    schema_version: int = 1
    on_conflict: OnConflict = OnConflict.ERROR


class SealMarker(_FormatModel):
    """Seal state embedded in a committed manifest."""

    writer: str
    superseded: dict[str, int] = Field(default_factory=dict)


class Shard(_FormatModel):
    """One immutable Parquet segment referenced by a manifest."""

    path: str
    writer: str
    generation: int
    rows: int
    min_seq: int
    max_seq: int
    size_bytes: int = Field(ge=0)
    content_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    commit_sequence: int = 0
    primary_key_sorted: bool = False


class ManifestTable(_FormatModel):
    """The complete visible state of one table."""

    metadata_path: str
    shards: tuple[Shard, ...] = ()


class Manifest(_FormatModel):
    """One immutable, complete archive snapshot."""

    format_version: int = FORMAT_VERSION
    commit_id: str
    parent_commit_id: str | None = None
    sequence: int
    required_features: tuple[str, ...] = ()
    tables: dict[str, ManifestTable] = Field(default_factory=dict)
    sealed: SealMarker | None = None


class HeadMetadata(_FormatModel):
    """The small mutable pointer published with compare-and-swap."""

    format_version: int = FORMAT_VERSION
    commit_id: str
    sequence: int
    manifest_path: str


@dataclass(frozen=True)
class CommitToken:
    """A durable archive version returned by a successful commit."""

    commit_id: str
    sequence: int
    version: str
    manifest_path: str


@dataclass(frozen=True)
class FineStoreLayout:
    """Every object key owned by one archive root."""

    root: str

    @property
    def archive_path(self) -> str:
        return prefix_join(self.root, ARCHIVE_FILE)

    @property
    def head_path(self) -> str:
        return prefix_join(self.root, HEAD_FILE)

    def manifest_path(self, commit_id: str) -> str:
        return prefix_join(self.root, f"{MANIFESTS_DIR}/{commit_id}.json")

    def schema_path(self, schema_id: str) -> str:
        return prefix_join(self.root, f"{SCHEMAS_DIR}/{schema_id}.json")

    def table_dir(self, table: str) -> str:
        return prefix_join(self.root, f"{DATA_DIR}/{table}")

    def shard_path(self, table: str, writer: str, generation: int, seq: int, uid: str) -> str:
        return prefix_join(self.table_dir(table), f"w={writer}/g={generation}/{seq:016d}-{uid}.parquet")


@dataclass(frozen=True)
class ArchiveRef:
    """A parsed ``finestore://<table>/<key>`` reference."""

    table: str
    key: str


def build_uri(table: str, key: str) -> str:
    """Format a reference to an object held in an archive table."""
    return f"{URI_SCHEME}://{table}/{key.lstrip('/')}"


def parse_uri(uri: str) -> ArchiveRef | None:
    """Parse a FineStore reference, returning ``None`` for another URI scheme."""
    parts = urlsplit(uri)
    if parts.scheme != URI_SCHEME:
        return None
    return ArchiveRef(table=parts.netloc, key=parts.path.lstrip("/"))
