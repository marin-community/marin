# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Loss-tolerant reads and write-time retention for disposable named-object caches."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from functools import partial

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
from finestore.layout import BlobTables, FineStoreLayout, TableMetadata, blob_table_metadata
from finestore.reader import BlobCorruptionError, ReadView

logger = logging.getLogger(__name__)

_REPAIR_ATTEMPTS = 3


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


def _discard_cache_history(snapshot: ArchiveSnapshot, missing: set[str]) -> ArchiveSnapshot:
    floor = snapshot.manifest.cache_history_floor
    for table in snapshot.manifest.tables.values():
        for shard in table.shards:
            if shard.path in missing:
                floor = max(floor, shard.row_commit_upper_bound)
    tables = {
        name: table.model_copy(
            update={
                "shards": tuple(
                    shard for shard in table.shards if shard.path not in missing and shard.row_commit_upper_bound > floor
                )
            }
        )
        for name, table in snapshot.manifest.tables.items()
    }
    return ArchiveSnapshot(
        manifest=snapshot.manifest.model_copy(update={"tables": tables, "cache_history_floor": floor}),
        token=snapshot.token,
    )


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
            StoragePath(table.metadata_path).refresh_lifetime(before=refresh_before)
        except FileNotFoundError as exc:
            schema_path = write_schema(layout, blob_table_metadata(BlobTables(name)))
            if schema_path != table.metadata_path:
                raise ValueError(f"cache table {name!r} has an unexpected schema path") from exc
        for shard in table.shards:
            try:
                StoragePath(shard.path).refresh_lifetime(before=refresh_before)
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
    commits.commit(
        CommitDelta(replacements=replacements, cache_history_floor=available.manifest.cache_history_floor), base=snapshot
    )
    logger.warning(
        "FineStore cache discarded %d shard references after object loss at %s",
        sum(len(replacement.input_paths) for replacement in replacements.values()),
        root,
    )
