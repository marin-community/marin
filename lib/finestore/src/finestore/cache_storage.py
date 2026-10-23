# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Loss-tolerant reads and write-time retention for disposable named-object caches."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

import rigging.filesystem.factory as factory
from rigging.filesystem.storage_path import StoragePath

from finestore.commit import (
    ArchiveSnapshot,
    CommitConflict,
    CommitCoordinator,
    CommitDelta,
    TableReplacement,
    read_snapshot,
    write_schema,
)
from finestore.layout import BlobColumns, BlobTables, FineStoreLayout, OnConflict, TableMetadata
from finestore.reader import BlobCorruptionError, ReadView

logger = logging.getLogger(__name__)

_REPAIR_ATTEMPTS = 3


def _blob_metadata(table: str) -> TableMetadata:
    if table == BlobTables.DESCRIPTORS:
        return TableMetadata(primary_key=(BlobColumns.NAME,), on_conflict=OnConflict.SUPERSEDE)
    if table == BlobTables.PARTS:
        return TableMetadata(primary_key=(BlobColumns.NAME, BlobColumns.PART), on_conflict=OnConflict.SUPERSEDE)
    raise ValueError(f"cache contains unexpected table {table!r}")


class CacheReadView(ReadView):
    """Read disposable blobs, treating expired schemas and incomplete payloads as cache loss."""

    def __init__(self, root: str, snapshot: ArchiveSnapshot | None = None) -> None:
        super().__init__(root, snapshot)
        self.missing_schema = False

    def _meta(self, table: str) -> TableMetadata:
        try:
            return super()._meta(table)
        except FileNotFoundError:
            # The two cache tables have fixed contracts; readers need not recreate expired schemas.
            self.missing_schema = True
            return _blob_metadata(table)

    def read_blobs(self, names: Sequence[str]) -> dict[str, bytes]:
        try:
            return super().read_blobs(names)
        except BlobCorruptionError:
            # One expired chunk must not discard healthy values in the same batch.
            values = {}
            for name in dict.fromkeys(names):
                try:
                    values.update(super().read_blobs((name,)))
                except BlobCorruptionError:
                    logger.warning("FineStore cache blob is incomplete, treating %s as a miss", name)
            return values


def available_cache_snapshot(root: str) -> ArchiveSnapshot:
    """Exclude confirmed missing shards from a read snapshot without changing storage."""
    snapshot = read_snapshot(FineStoreLayout(root))
    tables = {}
    for name, table in snapshot.manifest.tables.items():
        shards = []
        for shard in table.shards:
            try:
                fs, path = factory.url_to_fs(shard.path)
                fs.invalidate_cache(path)
                fs.info(path)
            except FileNotFoundError:
                continue
            shards.append(shard)
        tables[name] = table.model_copy(update={"shards": tuple(shards)})
    return ArchiveSnapshot(manifest=snapshot.manifest.model_copy(update={"tables": tables}), token=snapshot.token)


def _refresh_object(path: str, refresh_before: datetime | None) -> None:
    location = StoragePath(path)
    fs, key = factory.url_to_fs(path)
    fs.invalidate_cache(key)
    modified = fs.created(key) if location.scheme == "gs" else fs.modified(key)
    if refresh_before is None or modified >= refresh_before:
        return
    if location.is_local:
        fs.touch(key, truncate=False)
    elif location.scheme == "s3":
        # S3 requires metadata replacement when copying an object onto itself. FineStore
        # stores its integrity and table metadata in the manifest, not S3 user metadata.
        fs.cp_file(key, key, MetadataDirective="REPLACE")
    elif location.scheme == "gs":
        fs.cp_file(key, key)
    else:
        raise ValueError(f"cache retention does not support {location.scheme!r}")
    fs.invalidate_cache(key)


def maintain_cache(root: str, *, refresh_before: datetime | None) -> None:
    """Repair missing cache references and refresh aging live objects before a write.

    Only confirmed absence discards shards. Permission and transport errors propagate.
    Exact-input replacements preserve additions published by concurrent writers.
    """
    layout = FineStoreLayout(root)
    commits = CommitCoordinator(layout)
    for attempt in range(_REPAIR_ATTEMPTS):
        snapshot = commits.snapshot()
        replacements = {}
        for name, table in snapshot.manifest.tables.items():
            try:
                _refresh_object(table.metadata_path, refresh_before)
            except FileNotFoundError as exc:
                schema_path = write_schema(layout, _blob_metadata(name))
                if schema_path != table.metadata_path:
                    raise ValueError(f"cache table {name!r} has an unexpected schema path") from exc
            missing = set()
            for shard in table.shards:
                try:
                    _refresh_object(shard.path, refresh_before)
                except FileNotFoundError:
                    missing.add(shard.path)
            if missing:
                replacements[name] = TableReplacement(input_paths=frozenset(missing), output_shards=())
        if not replacements:
            return
        try:
            commits.commit(CommitDelta(replacements=replacements), base=snapshot)
        except CommitConflict:
            if attempt == _REPAIR_ATTEMPTS - 1:
                raise
            continue
        logger.warning(
            "FineStore cache repaired %d missing shard references at %s",
            sum(len(replacement.input_paths) for replacement in replacements.values()),
            root,
        )
        return
