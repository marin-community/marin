# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manifest-driven logical compaction for FineStore tables."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Generator, Iterator, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Protocol

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import rigging.filesystem.factory as factory
from pyarrow.fs import PyFileSystem
from rigging.filesystem.s3_errors import is_transient_s3_error
from rigging.timing import retry_with_backoff

from finestore import shard_writer
from finestore.commit import ArchiveSnapshot, CommitConflict, CommitCoordinator, CommitDelta, TableReplacement
from finestore.layout import BlobTables, CommitToken, FineStoreLayout, Shard, SystemColumns
from finestore.reader import ReadView, _NoReadAheadHandler, _ScanProfile, iter_shard_batches, merge_versioned_rows
from finestore.shard_writer import ShardWriter

logger = logging.getLogger(__name__)

_COMPACTOR = "compactor"
_CLOSE_COMPACTION_MAX_ATTEMPTS = 3
_COMPACTION_INPUT_BYTES = 64 * 1024 * 1024
_COMPACTION_SHARD_BATCH_BYTES = 4 * 1024 * 1024
_COMPACTION_MAX_BATCH_ROWS = 4096


class CompactionCoordinator(Protocol):
    def snapshot(self) -> ArchiveSnapshot: ...

    def commit(self, delta: CommitDelta, *, base: ArchiveSnapshot | None = None) -> CommitToken: ...


@dataclass(frozen=True)
class CompactionResult:
    """Rows written and duplicate inputs superseded by one compaction."""

    written: int
    superseded: int = 0


@dataclass(frozen=True)
class _ArrowRow:
    key: tuple
    commit_sequence: int
    sequence: int
    generation: int
    batch: pa.RecordBatch
    index: int


def _scan_batch_rows(metadata: pq.FileMetaData, shard_count: int) -> int:
    """Estimate a scan batch from the largest Parquet row-group average."""
    bytes_per_row = max(
        (
            (group.total_byte_size + group.num_rows - 1) // group.num_rows
            for group in (metadata.row_group(index) for index in range(metadata.num_row_groups))
            if group.num_rows
        ),
        default=1,
    )
    budget = min(_COMPACTION_SHARD_BATCH_BYTES, _COMPACTION_INPUT_BYTES // shard_count)
    return max(1, min(_COMPACTION_MAX_BATCH_ROWS, budget // max(1, bytes_per_row)))


def _iter_compaction_rows(
    shard: Shard, schema: pa.Schema, primary_key: tuple[str, ...], filesystem: PyFileSystem, batch_rows: int
) -> Generator[_ArrowRow, None, None]:
    """Yield sorted Arrow row references with their key and version coordinates."""
    profile = _ScanProfile(batch_rows=batch_rows, batch_readahead=1, fragment_readahead=1, pre_buffer=False)
    with closing(iter_shard_batches(shard, schema, primary_key, filesystem, scan_profile=profile)) as batches:
        for batch in batches:
            if not batch.num_rows:
                continue
            keys = [batch.column(batch.schema.get_field_index(name)) for name in primary_key]
            commits = batch.column(batch.schema.get_field_index(SystemColumns.COMMIT))
            sequences = batch.column(batch.schema.get_field_index(SystemColumns.SEQUENCE))
            for index in range(batch.num_rows):
                key = tuple((value is None, value) for value in (column[index].as_py() for column in keys))
                yield _ArrowRow(
                    key=key,
                    commit_sequence=commits[index].as_py(),
                    sequence=sequences[index].as_py() or 0,
                    generation=shard.generation,
                    batch=batch,
                    index=index,
                )


def _compaction_groups(
    rows: Iterator[tuple[_ArrowRow, int]], schema: pa.Schema, table: str
) -> Iterator[tuple[pa.Table, int]]:
    """Yield bounded output row groups with each group's superseded-row count."""
    batches: list[pa.RecordBatch] = []
    batch_offsets: dict[int, int] = {}
    indices: list[int] = []
    bytes_used = 0
    source_rows = 0
    superseded = 0
    row_limit = shard_writer.table_row_limit(table)
    for row, dropped in rows:
        row_bytes = row.batch.slice(row.index, 1).nbytes
        if indices and (len(indices) >= row_limit or bytes_used + row_bytes > shard_writer.ROW_GROUP_TARGET_BYTES):
            yield pa.Table.from_batches(batches, schema=schema).take(pa.array(indices, type=pa.int64())), superseded
            batches, batch_offsets, indices = [], {}, []
            bytes_used = source_rows = superseded = 0
        batch_id = id(row.batch)
        if batch_id not in batch_offsets:
            batch_offsets[batch_id] = source_rows
            batches.append(row.batch)
            source_rows += row.batch.num_rows
        indices.append(batch_offsets[batch_id] + row.index)
        bytes_used += row_bytes
        superseded += dropped
    if indices:
        yield pa.Table.from_batches(batches, schema=schema).take(pa.array(indices, type=pa.int64())), superseded


def compact_table(
    root: str,
    table: str,
    *,
    generation: int | None = None,
    coordinator: CompactionCoordinator | None = None,
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
    return _compact_shards(root, table, shards, snapshot=snapshot, coordinator=commits)


def compact_writer_tables(
    root: str,
    tables: Sequence[str],
    *,
    writer_id: str,
    max_input_bytes: int,
    max_input_rows: int,
    coordinator: CompactionCoordinator,
) -> None:
    """Merge a writer's residual level-zero groups within an aggregate input budget."""
    for table in sorted(tables, key=lambda name: (name != BlobTables.DESCRIPTORS, name)):
        snapshot = retry_with_backoff(
            coordinator.snapshot,
            retryable=is_transient_s3_error,
            max_attempts=_CLOSE_COMPACTION_MAX_ATTEMPTS,
            operation=f"FineStore close snapshot at {root}",
        )
        shards = tuple(
            shard
            for shard in ReadView(root, snapshot).list_shards(table)
            if shard.generation == 0 and shard.writer == writer_id
        )
        if len(shards) < 2:
            continue
        input_bytes = sum(shard.size_bytes for shard in shards)
        input_rows = sum(shard.rows for shard in shards)
        if input_bytes > max_input_bytes or input_rows > max_input_rows:
            continue
        # Filter the merge inputs, keeping the full publication base and unrelated shards.
        retry_with_backoff(
            partial(_compact_shards, root, table, shards, snapshot=snapshot, coordinator=coordinator),
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
    pa_fs = PyFileSystem(_NoReadAheadHandler(fs))
    metadata = [pq.read_metadata(shard.path, filesystem=pa_fs) for shard in shards]
    unified = pa.unify_schemas([item.schema.to_arrow_schema() for item in metadata], promote_options="permissive")
    if SystemColumns.COMMIT not in unified.names:
        unified = unified.append(pa.field(SystemColumns.COMMIT, pa.int64()))

    output_path = layout.shard_path(table, _COMPACTOR, next_generation, 0, uuid.uuid4().hex[:8])
    written = 0
    superseded = 0
    min_seq: int | None = None
    max_seq: int | None = None
    with ExitStack() as stack:
        streams = [
            stack.enter_context(
                closing(_iter_compaction_rows(shard, unified, primary_key, pa_fs, _scan_batch_rows(item, len(shards))))
            )
            for shard, item in zip(shards, metadata, strict=True)
        ]
        merged_rows = merge_versioned_rows(streams)
        first = next(merged_rows, None)
        if first is None:
            raise ValueError(f"FineStore table {table!r} has {len(shards)} non-empty shard descriptors but no rows")
        with ShardWriter(output_path, unified) as writer:
            for batch, dropped in _compaction_groups(chain((first,), merged_rows), unified, table):
                writer.write_table(batch)
                sequences = pc.fill_null(batch[SystemColumns.SEQUENCE], 0)
                group_min = pc.min(sequences).as_py()
                group_max = pc.max(sequences).as_py()
                min_seq = group_min if min_seq is None else min(min_seq, group_min)
                max_seq = group_max if max_seq is None else max(max_seq, group_max)
                written += batch.num_rows
                superseded += dropped

    assert min_seq is not None and max_seq is not None
    shard_result = writer.result
    output = Shard(
        path=output_path,
        writer=_COMPACTOR,
        generation=next_generation,
        rows=written,
        min_seq=min_seq,
        max_seq=max_seq,
        size_bytes=shard_result.size_bytes,
        content_sha256=shard_result.content_sha256,
        primary_key_sorted=True,
    )
    try:
        coordinator.commit(
            CommitDelta(
                replacements={
                    table: TableReplacement(
                        input_paths=frozenset(shard.path for shard in shards), output_shards=(output,)
                    )
                }
            ),
            base=snapshot,
        )
    except CommitConflict:
        logger.info(
            "FineStore abandoned compaction for %s table=%s input_shards=%d because its inputs changed",
            root,
            table,
            len(shards),
        )
        return CompactionResult(written=0)
    logger.info(
        "FineStore compacted %s table=%s input_generations=%s input_shards=%d output_shards=1 "
        "output_generation=%d rows=%d superseded=%d",
        root,
        table,
        sorted({shard.generation for shard in shards}),
        len(shards),
        next_generation,
        written,
        superseded,
    )
    return CompactionResult(written=written, superseded=superseded)
