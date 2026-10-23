# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Loss-tolerant reads and write-time retention for disposable named-object caches."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from functools import partial

import pyarrow.parquet as pq
import rigging.filesystem.factory as factory
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import retry_with_backoff

from finestore.commit import (
    ArchiveSnapshot,
    CommitConflict,
    CommitCoordinator,
    CommitDelta,
    TableReplacement,
    read_snapshot,
    write_schema,
)
from finestore.layout import BlobTables, FineStoreLayout, Shard, SystemColumns, TableMetadata, blob_table_metadata
from finestore.reader import BlobCorruptionError, ReadView

logger = logging.getLogger(__name__)

_REPAIR_ATTEMPTS = 3
_GCS_SCHEME = "gs"


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
            return blob_table_metadata(BlobTables(table))

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


def _oldest_commit(shard: Shard) -> int | None:
    """Return the oldest row commit, or None when footer statistics cannot prove it."""
    fs, path = factory.url_to_fs(shard.path)
    with fs.open(path, "rb", cache_type="none") as source:
        metadata = pq.read_metadata(source)
    if SystemColumns.COMMIT not in metadata.schema.names:
        return shard.commit_sequence
    column = metadata.schema.names.index(SystemColumns.COMMIT)
    minima = []
    for group in range(metadata.num_row_groups):
        statistics = metadata.row_group(group).column(column).statistics
        if statistics is None or not statistics.has_min_max:
            return None
        minima.append(int(statistics.min))
    return min(minima, default=shard.commit_sequence)


def _discard_cache_history(snapshot: ArchiveSnapshot, missing: set[str]) -> ArchiveSnapshot:
    if not missing:
        return snapshot
    discarded = set(missing)
    by_path = {shard.path: shard for table in snapshot.manifest.tables.values() for shard in table.shards}
    oldest: dict[str, int | None] = {}
    while True:
        cutoff = max(by_path[path].commit_sequence for path in discarded)
        additional = set()
        tables = {}
        for name, table in snapshot.manifest.tables.items():
            survivors = []
            for shard in table.shards:
                if shard.path in discarded:
                    continue
                if shard.commit_sequence <= cutoff:
                    additional.add(shard.path)
                    continue
                try:
                    if shard.path not in oldest:
                        oldest[shard.path] = _oldest_commit(shard)
                except FileNotFoundError:
                    additional.add(shard.path)
                    continue
                minimum = oldest[shard.path]
                if minimum is None or minimum <= cutoff:
                    additional.add(shard.path)
                else:
                    survivors.append(shard)
            tables[name] = table.model_copy(update={"shards": tuple(survivors)})
        if not additional:
            return ArchiveSnapshot(
                manifest=snapshot.manifest.model_copy(update={"tables": tables}), token=snapshot.token
            )
        # Dropping a mixed-age compacted shard can itself expose overwritten rows in
        # another shard. Extend the cutoff until no discarded version can reappear.
        discarded.update(additional)


def available_cache_snapshot(root: str) -> ArchiveSnapshot:
    """Discard missing shards and potentially superseded history without changing storage."""
    snapshot = read_snapshot(FineStoreLayout(root))
    missing = set()
    for table in snapshot.manifest.tables.values():
        for shard in table.shards:
            try:
                fs, path = factory.url_to_fs(shard.path)
                fs.invalidate_cache(path)
                fs.info(path)
            except FileNotFoundError:
                missing.add(shard.path)
    return _discard_cache_history(snapshot, missing)


def _refresh_object(path: str, refresh_before: datetime | None) -> None:
    location = StoragePath(path)
    fs, key = factory.url_to_fs(path)
    fs.invalidate_cache(key)
    modified = fs.created(key) if location.scheme == _GCS_SCHEME else fs.modified(key)
    if refresh_before is None or modified >= refresh_before:
        return
    if location.is_local:
        fs.touch(key, truncate=False)
    elif location.scheme == "s3":
        # S3 requires metadata replacement when copying an object onto itself. FineStore
        # stores its integrity and table metadata in the manifest, not S3 user metadata.
        fs.cp_file(key, key, MetadataDirective="REPLACE")
    elif location.scheme == _GCS_SCHEME:
        fs.cp_file(key, key)
    else:
        raise ValueError(f"cache retention does not support {location.scheme!r}")
    fs.invalidate_cache(key)


def maintain_cache(root: str, *, refresh_before: datetime | None) -> None:
    """Repair missing cache references and refresh aging live objects before a write.

    Confirmed absence also discards history that could expose superseded values.
    Permission and transport errors propagate. Concurrent additions survive repair.
    """
    retry_with_backoff(
        partial(_maintain_cache_once, root, refresh_before=refresh_before),
        retryable=lambda exc: isinstance(exc, CommitConflict),
        max_attempts=_REPAIR_ATTEMPTS,
        operation=f"FineStore cache repair at {root}",
    )


def _maintain_cache_once(root: str, *, refresh_before: datetime | None) -> None:
    layout = FineStoreLayout(root)
    commits = CommitCoordinator(layout)
    snapshot = commits.snapshot()
    missing = set()
    for name, table in snapshot.manifest.tables.items():
        try:
            _refresh_object(table.metadata_path, refresh_before)
        except FileNotFoundError as exc:
            schema_path = write_schema(layout, blob_table_metadata(BlobTables(name)))
            if schema_path != table.metadata_path:
                raise ValueError(f"cache table {name!r} has an unexpected schema path") from exc
        for shard in table.shards:
            try:
                _refresh_object(shard.path, refresh_before)
            except FileNotFoundError:
                missing.add(shard.path)
    available = _discard_cache_history(snapshot, missing)
    replacements = {}
    for name, table in snapshot.manifest.tables.items():
        retained = {shard.path for shard in available.manifest.tables[name].shards}
        discarded = frozenset(shard.path for shard in table.shards if shard.path not in retained)
        if discarded:
            replacements[name] = TableReplacement(input_paths=discarded, output_shards=())
    if not replacements:
        return
    commits.commit(CommitDelta(replacements=replacements), base=snapshot)
    logger.warning(
        "FineStore cache discarded %d shard references after object loss at %s",
        sum(len(replacement.input_paths) for replacement in replacements.values()),
        root,
    )
