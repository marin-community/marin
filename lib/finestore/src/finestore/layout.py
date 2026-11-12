# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FineStore format-v2 metadata and object-key layout.

Data, schemas, and manifests are immutable. ``HEAD`` is the archive's only mutable
object and points at one complete manifest, making its conditional-write version the
commit token for every table and object in a transaction.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.storage_path import prefix_join

ARCHIVE_FILE = "_archive.json"
HEAD_FILE = "HEAD"
MAINTENANCE_LEASE_FILE = "_maintenance"
FORMAT_VERSION = 2
DATA_DIR = "data"
MANIFESTS_DIR = "manifests"
SCHEMAS_DIR = "schemas"

URI_SCHEME = "finestore"
CHUNKED_BLOBS_FEATURE = "chunked-blobs-v1"
CACHE_HISTORY_FLOOR_FEATURE = "cache-history-floor-v1"


class SystemColumns(StrEnum):
    """Columns FineStore adds to user rows while writing and reading shards."""

    SEQUENCE = "_seq"
    WRITER = "_writer"
    GENERATION = "_gen"
    COMMIT = "_commit"


class BlobTables(StrEnum):
    """Tables that encode named byte objects."""

    DESCRIPTORS = "blobs"
    PARTS = "_finestore_blob_parts"


class BlobColumns(StrEnum):
    """Columns shared by blob descriptors and part rows."""

    NAME = "name"
    DATA = "data"
    SIZE = "size"
    METADATA = "metadata_json"
    PART_COUNT = "part_count"
    PART = "part"


RESERVED_COLUMNS = frozenset(SystemColumns)
SUPPORTED_MANIFEST_FEATURES: frozenset[str] = frozenset({CHUNKED_BLOBS_FEATURE, CACHE_HISTORY_FLOOR_FEATURE})


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


def blob_table_metadata(table: BlobTables) -> TableMetadata:
    """Return the schema contract for a named-object table."""
    primary_key = (BlobColumns.NAME, BlobColumns.PART) if table == BlobTables.PARTS else (BlobColumns.NAME,)
    return TableMetadata(primary_key=primary_key, on_conflict=OnConflict.SUPERSEDE)


class SealMarker(_FormatModel):
    """Seal state embedded in a committed manifest."""

    writer: str
    superseded: dict[str, int] = Field(default_factory=dict)


KeyBound = tuple[str | int, ...]
# Key value types with one total order shared by the writer's bounds and the reader's lookups.
BOUNDED_KEY_TYPES = (str, int)


def _comparable(value: object, bound: object) -> bool:
    return type(value) is type(bound) and type(value) in BOUNDED_KEY_TYPES


@dataclass(frozen=True)
class KeyRange:
    """Inclusive bounds on the primary keys a shard holds.

    Bounds exist only for keys made of strings and integers, so a lookup value compares
    against them without ambiguity. A lookup value of another type than the bound at its
    position is treated as unbounded at that position.
    """

    lowest: KeyBound
    highest: KeyBound

    @classmethod
    def of(cls, keys: Iterable[tuple]) -> KeyRange | None:
        """Return the range spanning ``keys``, or ``None`` for no keys or an unbounded key type."""
        lowest: tuple | None = None
        highest: tuple | None = None
        for key in keys:
            if not all(type(value) in BOUNDED_KEY_TYPES for value in key):
                return None
            if lowest is None or key < lowest:
                lowest = key
            if highest is None or key > highest:
                highest = key
        if lowest is None or highest is None:
            return None
        return cls(lowest, highest)

    def may_contain_prefix(self, prefix: tuple) -> bool:
        """Whether a key starting with ``prefix`` can lie within the range."""
        width = 0
        for value, low, high in zip(prefix, self.lowest, self.highest, strict=False):
            if not _comparable(value, low) or not _comparable(value, high):
                break
            width += 1
        if width == 0:
            return True
        return self.lowest[:width] <= prefix[:width] <= self.highest[:width]

    def may_contain_first(self, values: Iterable[object]) -> bool:
        """Whether any of ``values`` can be the first key column of a key within the range."""
        return any(self.may_contain_prefix((value,)) for value in values)


class Shard(_FormatModel):
    """One immutable Parquet segment referenced by a manifest.

    ``min_key`` and ``max_key`` bound the primary keys stored in the shard when they are
    strings and integers; a reader skips the shard for lookups outside ``key_range``.
    Row-commit bounds retain the original versions through compaction; ``commit_sequence``
    records publication of this file.
    """

    path: str
    writer: str
    generation: int
    rows: int
    min_seq: int
    max_seq: int
    size_bytes: int = Field(ge=0)
    content_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    commit_sequence: int = 0
    min_row_commit: int | None = None
    max_row_commit: int | None = None
    primary_key_sorted: bool = False
    min_key: KeyBound | None = None
    max_key: KeyBound | None = None

    @property
    def row_commit_upper_bound(self) -> int:
        # Older manifests lack row bounds. Publication is always at or after every row's commit.
        return self.commit_sequence if self.max_row_commit is None else self.max_row_commit

    @property
    def key_range(self) -> KeyRange | None:
        if self.min_key is None or self.max_key is None:
            return None
        return KeyRange(self.min_key, self.max_key)


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
    # Advanced by each writer that acquires the maintenance lease. A compaction prepared under
    # an older epoch cannot be published after a takeover.
    maintainer_epoch: int = 0
    # Disposable-cache recovery invalidates these commits across descriptor and chunk tables.
    cache_history_floor: int = 0


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

    @property
    def maintenance_lease_path(self) -> str:
        return prefix_join(self.root, MAINTENANCE_LEASE_FILE)

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
