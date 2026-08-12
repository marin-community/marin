# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-shot migration from FineStore's listing-based v1 layout to format v2."""

from __future__ import annotations

import dataclasses
import re
import uuid

import pyarrow.parquet as pq
from pydantic import AliasChoices, BaseModel, Field
from rigging.filesystem import StoragePath
from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object

from finestore.commit import read_snapshot, write_schema
from finestore.layout import (
    FORMAT_VERSION,
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
)

_LEGACY_FORMAT_VERSION = 1
_LEGACY_SCHEMA_FILE = "_schema.json"
_LEGACY_SEALED_FILE = "SEALED"
_WRITER_SEGMENT = re.compile(r"^w=(.+)$")
_GENERATION_SEGMENT = re.compile(r"^g=(\d+)$")


class _LegacyArchiveMetadata(BaseModel):
    format_version: int


class _LegacyTableMetadata(BaseModel):
    primary_key: tuple[str, ...] = Field(validation_alias=AliasChoices("primary_key", "merge_key"))
    schema_version: int = 1
    on_conflict: OnConflict = OnConflict.ERROR


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


def _sequence_bounds(metadata: pq.FileMetaData) -> tuple[int, int]:
    if "_seq" not in metadata.schema.names or metadata.num_rows == 0:
        return 0, 0
    column = metadata.schema.names.index("_seq")
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
            min_seq, max_seq = _sequence_bounds(metadata)
            shards.append(
                Shard(
                    path=str(path),
                    writer=writer,
                    generation=generation,
                    rows=metadata.num_rows,
                    min_seq=min_seq,
                    max_seq=max_seq,
                    commit_sequence=1,
                )
            )
    return tuple(sorted(shards, key=lambda shard: shard.path))


def migrate_v1(root: str) -> CommitToken:
    """Convert one sealed v1 archive to a v2 manifest without rewriting its Parquet shards.

    The archive must be quiescent and sealed. Schema objects, the initial manifest, and ``HEAD``
    are prepared first; the conditional replacement of ``_archive.json`` is the final publication
    step. Legacy shards, schema files, and ``SEALED`` are retained as unreachable source objects.
    Re-running a completed migration returns the current commit token.
    """
    root_path = StoragePath(root.rstrip("/"))
    layout = FineStoreLayout(str(root_path))
    archive_object = conditional_object(layout.archive_path)
    archive = archive_object.read()
    if archive is None:
        raise FileNotFoundError(layout.archive_path)
    archive_metadata = _LegacyArchiveMetadata.model_validate_json(archive.data)
    if archive_metadata.format_version == FORMAT_VERSION:
        token = read_snapshot(layout).token
        if token is None:
            raise ValueError(f"format-v2 archive at {root!r} has no committed HEAD")
        return token
    if archive_metadata.format_version != _LEGACY_FORMAT_VERSION:
        raise ValueError(f"cannot migrate FineStore format v{archive_metadata.format_version}")

    seal_path = root_path / _LEGACY_SEALED_FILE
    if not seal_path.exists():
        raise ValueError(f"FineStore v1 archive at {root!r} is not sealed; quiesce and seal it before migration")
    seal = SealMarker.model_validate_json(seal_path.read_bytes())

    head_object = conditional_object(layout.head_path)
    existing_head = head_object.read()
    if existing_head is None:
        tables = {}
        for name, legacy_metadata in _legacy_table_metadata(root_path).items():
            metadata_path = write_schema(layout, TableMetadata.model_validate(legacy_metadata.model_dump()))
            tables[name] = ManifestTable(metadata_path=metadata_path, shards=_legacy_shards(root_path, name))
        commit_id = uuid.uuid4().hex
        manifest = Manifest(commit_id=commit_id, sequence=1, tables=tables, sealed=seal)
        manifest_path = layout.manifest_path(commit_id)
        StoragePath(manifest_path).write_text(manifest.model_dump_json(indent=2))
        head = HeadMetadata(commit_id=commit_id, sequence=1, manifest_path=manifest_path)
        try:
            head_object.write(head.model_dump_json().encode(), expected_version=None)
        except ConditionalWriteError:
            pass

    existing_head = head_object.read()
    assert existing_head is not None
    head = HeadMetadata.model_validate_json(existing_head.data)
    manifest = Manifest.model_validate_json(StoragePath(head.manifest_path).read_bytes())
    if (manifest.commit_id, manifest.sequence) != (head.commit_id, head.sequence):
        raise ValueError(f"prepared HEAD at {root!r} does not match its manifest")

    try:
        archive_object.write(ArchiveMetadata().model_dump_json().encode(), expected_version=archive.version)
    except ConditionalWriteError as exc:
        current = archive_object.read()
        if current is not None and _LegacyArchiveMetadata.model_validate_json(current.data).format_version == FORMAT_VERSION:
            token = read_snapshot(layout).token
            assert token is not None
            return token
        raise RuntimeError(f"archive at {root!r} changed during format migration") from exc

    token = read_snapshot(layout).token
    assert token is not None
    return token
