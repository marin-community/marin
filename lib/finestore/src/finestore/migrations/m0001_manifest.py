# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upgrade listing-based v1 archives to transactional v2 manifests."""

from __future__ import annotations

import dataclasses
import hashlib
import re
import uuid
from collections.abc import Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import AliasChoices, BaseModel, Field
from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object
from rigging.filesystem.storage_path import StoragePath

from finestore.commit import read_snapshot, write_schema
from finestore.layout import (
    BLOB_DATA_COLUMN,
    BLOB_NAME_COLUMN,
    BLOBS_TABLE,
    SEQ_COLUMN,
    ArchiveMetadata,
    CommitToken,
    FineStoreLayout,
    HeadMetadata,
    Manifest,
    ManifestTable,
    OnConflict,
    SealMarker,
    Shard,
    TableMetadata,
    parse_uri,
)
from finestore.reader import _scan_shards

MIGRATION_ID = "0001_manifest"
FROM_VERSION = 1
TO_VERSION = 2
WRITE_OPEN_SAFE = True

_LEGACY_SCHEMA_FILE = "_schema.json"
LEGACY_SEAL_FILE = "SEALED"
_WRITER_SEGMENT = re.compile(r"^w=(.+)$")
_GENERATION_SEGMENT = re.compile(r"^g=(\d+)$")
_HASH_CHUNK_BYTES = 1024 * 1024


class _LegacyArchiveMetadata(BaseModel):
    format_version: int


class _LegacyTableMetadata(BaseModel):
    primary_key: tuple[str, ...] = Field(validation_alias=AliasChoices("primary_key", "merge_key"))
    schema_version: int = 1
    on_conflict: OnConflict = OnConflict.ERROR


@dataclasses.dataclass(frozen=True)
class LegacyTable:
    """One table discovered in a listing-based v1 archive."""

    metadata_path: str
    metadata: TableMetadata
    shards: tuple[Shard, ...]


@dataclasses.dataclass(frozen=True)
class LegacyArchive:
    """One stable v1 state, including its seal when present."""

    seal_path: str | None
    seal: SealMarker | None
    tables: dict[str, LegacyTable]


@dataclasses.dataclass(frozen=True)
class LegacyReadShard:
    """One v1 shard reference captured by a best-effort listing snapshot."""

    path: str
    writer: str
    generation: int
    commit_sequence: int = 0
    primary_key_sorted: bool = False


@dataclasses.dataclass(frozen=True)
class _LegacyReadTable:
    metadata_path: str
    metadata: TableMetadata
    shards: tuple[LegacyReadShard, ...]


class LegacyReadView:
    """A read-only, listing-based view of one format-v1 archive.

    Version 1 has no atomic HEAD, so this view pins the shard names observed while it is
    constructed and has no commit token. It exists for bounded compatibility while old writers
    remain deployed; new readers that require a durable snapshot use :class:`finestore.reader.ReadView`.
    """

    def __init__(self, root: str) -> None:
        self.root = root
        root_path = StoragePath(root)
        archive = _LegacyArchiveMetadata.model_validate_json((root_path / "_archive.json").read_bytes())
        if archive.format_version != FROM_VERSION:
            raise ValueError(
                f"FineStore legacy reader requires format v{FROM_VERSION}; found v{archive.format_version} at {root!r}"
            )
        seal_path = root_path / LEGACY_SEAL_FILE
        self._seal = SealMarker.model_validate_json(seal_path.read_bytes()) if seal_path.exists() else None
        self._tables = {
            name: _LegacyReadTable(
                metadata_path=str(root_path / name / _LEGACY_SCHEMA_FILE),
                metadata=TableMetadata.model_validate(metadata.model_dump()),
                shards=_legacy_read_shards(root_path, name),
            )
            for name, metadata in _legacy_table_metadata(root_path).items()
        }

    @property
    def token(self) -> None:
        """Format v1 has no commit token."""
        return None

    def primary_key(self, table: str) -> tuple[str, ...]:
        return self.table_metadata(table).primary_key

    def table_metadata(self, table: str) -> TableMetadata:
        state = self._tables.get(table)
        if state is None:
            raise KeyError(f"table {table!r} is not present in format-v1 archive {self.root!r}")
        return state.metadata

    def table_metadata_path(self, table: str) -> str | None:
        state = self._tables.get(table)
        return None if state is None else state.metadata_path

    def table_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tables))

    def schema_version(self, table: str) -> int | None:
        state = self._tables.get(table)
        if state is None or not state.shards:
            return None
        return state.metadata.schema_version

    def is_sealed(self) -> bool:
        return self._seal is not None

    def seal_marker(self) -> SealMarker | None:
        return self._seal

    def scan(
        self,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        where: list[tuple[str, str, object]] | None = None,
    ) -> pa.Table | None:
        state = self._tables.get(table)
        if state is None:
            return None
        return _scan_shards(self.root, state.shards, state.metadata.primary_key, columns=columns, where=where)

    def point(self, table: str, **keys) -> dict | None:
        result = self.scan(table, where=[(key, "==", value) for key, value in keys.items()])
        if result is None or result.num_rows == 0:
            return None
        return result.slice(0, 1).to_pylist()[0]

    def keys(self, table: str) -> set[tuple]:
        state = self._tables.get(table)
        if state is None or not state.shards:
            return set()
        result = self.scan(table, columns=list(state.metadata.primary_key))
        if result is None:
            return set()
        values = [result.column(name).to_pylist() for name in state.metadata.primary_key]
        return set(zip(*values, strict=True))

    def read_blob(self, name: str) -> bytes | None:
        row = self.point(BLOBS_TABLE, **{BLOB_NAME_COLUMN: name})
        if row is None:
            return None
        data = row.get(BLOB_DATA_COLUMN)
        return bytes(data) if data is not None else None

    def resolve(self, uri: str) -> bytes | None:
        ref = parse_uri(uri)
        if ref is None:
            raise ValueError(f"not a finestore:// reference: {uri!r}")
        if ref.table != BLOBS_TABLE:
            raise ValueError(f"finestore:// resolution supports the blobs table only, got {ref.table!r}")
        return self.read_blob(ref.key)

    def list_shards(self, table: str) -> list[LegacyReadShard]:
        state = self._tables.get(table)
        return [] if state is None else list(state.shards)


def _legacy_table_metadata(root: StoragePath) -> dict[str, _LegacyTableMetadata]:
    tables = {}
    for directory in root.ls():
        schema_path = directory / _LEGACY_SCHEMA_FILE
        if not directory.isdir() or not schema_path.exists():
            continue
        relative_root = (
            dataclasses.replace(root, scheme=directory.scheme, netloc=directory.netloc, rooted=directory.rooted)
            if root.is_local and directory.is_local
            else root
        )
        name = directory.relative_to(relative_root)
        tables[name] = _LegacyTableMetadata.model_validate_json(schema_path.read_bytes())
    return tables


def _legacy_shard(path: StoragePath) -> tuple[str, int] | None:
    if not path.name.endswith(".parquet"):
        return None
    writer = None
    generation = None
    for segment in path.segments:
        if match := _WRITER_SEGMENT.match(segment):
            writer = match.group(1)
        if match := _GENERATION_SEGMENT.match(segment):
            generation = int(match.group(1))
    if writer is None or generation is None:
        return None
    return writer, generation


def _legacy_read_shards(root: StoragePath, table: str) -> tuple[LegacyReadShard, ...]:
    shards = []
    for directory, _, names in (root / table).walk():
        for name in names:
            path = directory / name
            parsed = _legacy_shard(path)
            if parsed is None:
                continue
            writer, generation = parsed
            shards.append(LegacyReadShard(path=str(path), writer=writer, generation=generation))
    return tuple(sorted(shards, key=lambda shard: shard.path))


def _sequence_bounds(metadata: pq.FileMetaData) -> tuple[int, int]:
    if SEQ_COLUMN not in metadata.schema.names or metadata.num_rows == 0:
        return 0, 0
    column = metadata.schema.names.index(SEQ_COLUMN)
    minima = []
    maxima = []
    for group in range(metadata.num_row_groups):
        statistics = metadata.row_group(group).column(column).statistics
        if statistics is not None and statistics.has_min_max:
            minima.append(int(statistics.min))
            maxima.append(int(statistics.max))
    return (min(minima), max(maxima)) if minima else (0, 0)


def _legacy_shards(root: StoragePath, table: str) -> tuple[Shard, ...]:
    shards = []
    for directory, _, names in (root / table).walk():
        for name in names:
            path = directory / name
            parsed = _legacy_shard(path)
            if parsed is None:
                continue
            writer, generation = parsed
            with path.open("rb") as handle:
                metadata = pq.read_metadata(handle)
                handle.seek(0)
                digest = hashlib.sha256()
                size_bytes = 0
                while chunk := handle.read(_HASH_CHUNK_BYTES):
                    digest.update(chunk)
                    size_bytes += len(chunk)
            min_seq, max_seq = _sequence_bounds(metadata)
            shards.append(
                Shard(
                    path=str(path),
                    writer=writer,
                    generation=generation,
                    rows=metadata.num_rows,
                    min_seq=min_seq,
                    max_seq=max_seq,
                    size_bytes=size_bytes,
                    content_sha256=digest.hexdigest(),
                    commit_sequence=1,
                )
            )
    return tuple(sorted(shards, key=lambda shard: shard.path))


def inspect_legacy_snapshot(root: str) -> LegacyArchive:
    """Read one v1 listing snapshot without changing the archive."""
    root_path = StoragePath(root)
    seal_path = root_path / LEGACY_SEAL_FILE
    has_seal = seal_path.exists()
    seal = SealMarker.model_validate_json(seal_path.read_bytes()) if has_seal else None
    tables = {}
    for name, legacy_metadata in _legacy_table_metadata(root_path).items():
        tables[name] = LegacyTable(
            metadata_path=str(root_path / name / _LEGACY_SCHEMA_FILE),
            metadata=TableMetadata.model_validate(legacy_metadata.model_dump()),
            shards=_legacy_shards(root_path, name),
        )
    return LegacyArchive(seal_path=str(seal_path) if has_seal else None, seal=seal, tables=tables)


def inspect_legacy_archive(root: str) -> LegacyArchive:
    """Read the complete sealed v1 state without changing the archive."""
    archive = inspect_legacy_snapshot(root)
    if archive.seal is None:
        raise ValueError(f"FineStore v1 archive at {root!r} is not sealed; quiesce and seal it before migration")
    return archive


def _v2_token(layout: FineStoreLayout) -> CommitToken | None:
    return read_snapshot(layout).token


def migrate(root: str) -> CommitToken:
    """Publish a sealed, quiescent v1 archive through a v2 manifest and HEAD.

    Existing Parquet payloads remain in place. The format marker advances only after
    the new HEAD is durable, so the migration is safe to retry after interruption.
    """
    root_path = StoragePath(root)
    layout = FineStoreLayout(str(root_path))
    archive_object = conditional_object(layout.archive_path)
    archive = archive_object.read()
    if archive is None:
        raise FileNotFoundError(layout.archive_path)
    archive_metadata = _LegacyArchiveMetadata.model_validate_json(archive.data)
    if archive_metadata.format_version == TO_VERSION:
        token = _v2_token(layout)
        if token is None:
            raise ValueError(f"format-v{TO_VERSION} archive at {root!r} has no committed HEAD")
        return token
    if archive_metadata.format_version != FROM_VERSION:
        raise ValueError(
            f"migration {MIGRATION_ID} requires FineStore format v{FROM_VERSION}; "
            f"found v{archive_metadata.format_version}"
        )

    legacy = inspect_legacy_archive(root)
    assert legacy.seal is not None

    head_object = conditional_object(layout.head_path)
    existing_head = head_object.read()
    if existing_head is None:
        tables = {}
        for name, legacy_table in legacy.tables.items():
            metadata_path = write_schema(layout, legacy_table.metadata)
            tables[name] = ManifestTable(metadata_path=metadata_path, shards=legacy_table.shards)
        commit_id = uuid.uuid4().hex
        manifest = Manifest(
            format_version=TO_VERSION,
            commit_id=commit_id,
            sequence=1,
            tables=tables,
            sealed=legacy.seal,
        )
        manifest_path = layout.manifest_path(commit_id)
        StoragePath(manifest_path).write_text(manifest.model_dump_json(indent=2))
        head = HeadMetadata(format_version=TO_VERSION, commit_id=commit_id, sequence=1, manifest_path=manifest_path)
        try:
            head_object.write(head.model_dump_json().encode(), expected_version=None)
        except ConditionalWriteError:
            # Another migrator won the HEAD race. The shared validation below accepts its
            # complete v2 commit or fails before either migrator advances the format marker.
            pass

    token = _v2_token(layout)
    if token is None:
        raise ValueError(f"format-v{TO_VERSION} archive at {root!r} has no committed HEAD")

    try:
        archive_object.write(
            ArchiveMetadata(format_version=TO_VERSION).model_dump_json().encode(), expected_version=archive.version
        )
    except ConditionalWriteError as exc:
        current = archive_object.read()
        if current is not None and _LegacyArchiveMetadata.model_validate_json(current.data).format_version == TO_VERSION:
            token = _v2_token(layout)
            assert token is not None
            return token
        raise RuntimeError(f"archive at {root!r} changed during format migration") from exc

    return token
