# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manifest-driven logical compaction for FineStore tables."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Generator, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from dataclasses import dataclass
from functools import partial
from typing import Protocol

import pyarrow as pa
import pyarrow.compute as pc
import rigging.filesystem.factory as factory
from pyarrow.fs import PyFileSystem
from rigging import telemetry
from rigging.filesystem.s3_errors import is_transient_s3_error
from rigging.timing import retry_with_backoff

from finestore import shard_writer
from finestore.commit import ArchiveSnapshot, CommitConflict, CommitCoordinator, CommitDelta, TableReplacement
from finestore.layout import BlobTables, CommitToken, FineStoreLayout, KeyRange, Shard, SystemColumns
from finestore.merge import key_values, merge_sorted_runs, sorted_by_key
from finestore.reader import ReadView, _BoundedReadAheadHandler, _CachedReadHandler, iter_shard_row_groups, shard_schema
from finestore.shard_writer import ShardWriter

logger = logging.getLogger(__name__)

_COMPACTOR = "compactor"
_COMPACTIONS = telemetry.counter("finestore_compactions", unit="{compaction}")
_COMPACTIONS_ABANDONED = telemetry.counter("finestore_compactions_abandoned", unit="{compaction}")
_COMPACTION_INPUT_BYTES = telemetry.counter("finestore_compaction_input_bytes", unit="By")
_CLOSE_COMPACTION_MAX_ATTEMPTS = 3
_COMPACTION_FILE_READ_AHEAD_BYTES = 1024 * 1024
_COMPACTION_PREFETCH_BYTES = 16 * 1024 * 1024
_COMPACTION_PREFETCH_WORKERS = 16
COMPACTION_TARGET_BYTES = 400 * 1024 * 1024


class CompactionCoordinator(Protocol):
    def snapshot(self) -> ArchiveSnapshot: ...

    def last_snapshot(self) -> ArchiveSnapshot: ...

    def commit(self, delta: CommitDelta, *, base: ArchiveSnapshot | None = None) -> CommitToken: ...


@dataclass(frozen=True)
class CompactionResult:
    """Rows written and duplicate inputs superseded by one compaction."""

    written: int
    superseded: int = 0


def _sorted_run(
    shard: Shard, unified: pa.Schema, pa_fs: PyFileSystem, primary_key: tuple[str, ...]
) -> Generator[pa.Table, None, None]:
    """Yield one shard's rows in primary-key order, a row group at a time when the shard is sorted."""
    with closing(iter_shard_row_groups(shard, unified, pa_fs)) as groups:
        yield from groups if shard.primary_key_sorted else sorted_by_key(groups, primary_key)


def _write_output_shards(
    layout: FineStoreLayout,
    table: str,
    generation: int,
    unified: pa.Schema,
    primary_key: tuple[str, ...],
    groups: Iterator[pa.Table],
    target_bytes: int,
) -> list[Shard]:
    """Write key-ordered row groups into shards near ``target_bytes`` decoded bytes.

    A shard always takes at least one row group, so a single oversized group still lands.
    """
    outputs: list[Shard] = []
    group = next(groups, None)
    while group is not None:
        path = layout.shard_path(table, _COMPACTOR, generation, 0, uuid.uuid4().hex[:8])
        first = last = group
        rows = decoded_bytes = 0
        min_seq: int | None = None
        max_seq: int | None = None
        with ShardWriter(path, unified) as writer:
            while group is not None and (rows == 0 or decoded_bytes + group.nbytes <= target_bytes):
                writer.write_table(group)
                sequences = pc.fill_null(group[SystemColumns.SEQUENCE], 0)
                low, high = pc.min(sequences).as_py(), pc.max(sequences).as_py()
                min_seq = low if min_seq is None else min(min_seq, low)
                max_seq = high if max_seq is None else max(max_seq, high)
                rows += group.num_rows
                decoded_bytes += group.nbytes
                last = group
                group = next(groups, None)
        bounds = KeyRange.of([key_values(first, 0, primary_key), key_values(last, last.num_rows - 1, primary_key)])
        outputs.append(
            Shard(
                path=path,
                writer=_COMPACTOR,
                generation=generation,
                rows=rows,
                min_seq=min_seq,
                max_seq=max_seq,
                size_bytes=writer.result.size_bytes,
                content_sha256=writer.result.content_sha256,
                primary_key_sorted=True,
                min_key=None if bounds is None else bounds.lowest,
                max_key=None if bounds is None else bounds.highest,
            )
        )
    return outputs


def compact_table(
    root: str,
    table: str,
    *,
    generation: int | None = None,
    coordinator: CompactionCoordinator | None = None,
    target_bytes: int = COMPACTION_TARGET_BYTES,
) -> CompactionResult:
    """Replace one generation, or all active shards, through one manifest commit.

    Source objects remain immutable and reachable by older read views. Garbage collection is
    deliberately separate from compaction because the store does not yet track reader leases.
    """
    layout = FineStoreLayout(root)
    commits = coordinator or CommitCoordinator(layout)
    snapshot = commits.snapshot()
    view = ReadView(root, snapshot)
    active_shards = view.list_shards(table)
    shards = active_shards if generation is None else tuple(s for s in active_shards if s.generation == generation)
    return _compact_shards(root, table, shards, snapshot=snapshot, coordinator=commits, target_bytes=target_bytes)


def compact_level_zero_tables(
    root: str,
    tables: Sequence[str],
    *,
    max_input_bytes: int,
    max_input_rows: int,
    target_bytes: int,
    coordinator: CompactionCoordinator,
) -> None:
    """Merge each table's level-zero shards, from every writer, within an aggregate input budget."""
    for table in sorted(tables, key=lambda name: (name != BlobTables.DESCRIPTORS, name)):
        snapshot = retry_with_backoff(
            coordinator.snapshot,
            retryable=is_transient_s3_error,
            max_attempts=_CLOSE_COMPACTION_MAX_ATTEMPTS,
            operation=f"FineStore close snapshot at {root}",
        )
        shards = tuple(shard for shard in ReadView(root, snapshot).list_shards(table) if shard.generation == 0)
        if len(shards) < 2:
            continue
        input_bytes = sum(shard.size_bytes for shard in shards)
        input_rows = sum(shard.rows for shard in shards)
        if input_bytes > max_input_bytes or input_rows > max_input_rows:
            continue
        # Filter the merge inputs, keeping the full publication base and unrelated shards.
        retry_with_backoff(
            partial(
                _compact_shards,
                root,
                table,
                shards,
                snapshot=snapshot,
                coordinator=coordinator,
                target_bytes=target_bytes,
            ),
            retryable=is_transient_s3_error,
            max_attempts=_CLOSE_COMPACTION_MAX_ATTEMPTS,
            operation=f"FineStore close compaction at {root} table={table}",
        )
        max_input_bytes -= input_bytes
        max_input_rows -= input_rows


def _compact_shards(
    root: str,
    table: str,
    shards: Sequence[Shard],
    *,
    snapshot: ArchiveSnapshot,
    coordinator: CompactionCoordinator,
    target_bytes: int,
) -> CompactionResult:
    if not shards:
        return CompactionResult(written=0)
    if len(shards) == 1 and shards[0].writer == _COMPACTOR:
        return CompactionResult(written=0)
    layout = FineStoreLayout(root)
    primary_key = ReadView(root, snapshot).primary_key(table)
    next_generation = max(shard.generation for shard in shards) + 1
    input_rows = sum(shard.rows for shard in shards)
    input_bytes = sum(shard.size_bytes for shard in shards)
    _COMPACTIONS.add()
    _COMPACTION_INPUT_BYTES.add(input_bytes)
    logger.info(
        "FineStore compacting %s table=%s input_generations=%s input_shards=%d input_rows=%d "
        "input_bytes=%d output_generation=%d",
        root,
        table,
        sorted({shard.generation for shard in shards}),
        len(shards),
        input_rows,
        input_bytes,
        next_generation,
    )

    fs, _ = factory.url_to_fs(root)
    if input_bytes <= _COMPACTION_PREFETCH_BYTES:
        # Small compactions otherwise reopen each shard for its schema, footer, and rows.
        # Keep every compressed input in one bounded cache and overlap the initial GETs.
        handler = _CachedReadHandler(fs, shards, _COMPACTION_PREFETCH_BYTES)
        with ThreadPoolExecutor(max_workers=min(len(shards), _COMPACTION_PREFETCH_WORKERS)) as pool:
            list(pool.map(handler.prefetch, (shard.path for shard in shards)))
        pa_fs = PyFileSystem(handler)
    else:
        pa_fs = PyFileSystem(_BoundedReadAheadHandler(fs, _COMPACTION_FILE_READ_AHEAD_BYTES))
    unified = pa.unify_schemas([shard_schema(shard.path, pa_fs) for shard in shards], promote_options="permissive")
    if SystemColumns.COMMIT not in unified.names:
        unified = unified.append(pa.field(SystemColumns.COMMIT, pa.int64()))
    if SystemColumns.GENERATION not in unified.names:
        unified = unified.append(pa.field(SystemColumns.GENERATION, pa.int32()))

    with ExitStack() as stack:
        runs = [stack.enter_context(closing(_sorted_run(shard, unified, pa_fs, primary_key))) for shard in shards]
        groups = shard_writer.table_row_groups(table, merge_sorted_runs(runs, primary_key))
        outputs = _write_output_shards(layout, table, next_generation, unified, primary_key, groups, target_bytes)
    written = sum(shard.rows for shard in outputs)
    superseded = input_rows - written

    if not outputs:
        raise ValueError(f"FineStore table {table!r} has {len(shards)} non-empty shard descriptors but no rows")
    try:
        coordinator.commit(
            CommitDelta(
                replacements={
                    table: TableReplacement(
                        input_paths=frozenset(shard.path for shard in shards), output_shards=tuple(outputs)
                    )
                },
                expected_epoch=snapshot.manifest.maintainer_epoch,
            ),
            base=snapshot,
        )
    except CommitConflict:
        _COMPACTIONS_ABANDONED.add()
        logger.info(
            "FineStore abandoned compaction for %s table=%s input_shards=%d because its inputs changed",
            root,
            table,
            len(shards),
        )
        return CompactionResult(written=0)
    logger.info(
        "FineStore compacted %s table=%s input_generations=%s input_shards=%d output_shards=%d "
        "output_generation=%d rows=%d superseded=%d",
        root,
        table,
        sorted({shard.generation for shard in shards}),
        len(shards),
        len(outputs),
        next_generation,
        written,
        superseded,
    )
    return CompactionResult(written=written, superseded=superseded)
