# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manifest-driven logical compaction for FineStore tables."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass, replace
from functools import partial
from itertools import zip_longest
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
from finestore.reader import ReadView, _NoReadAheadHandler, _ScanProfile, iter_shard_batches
from finestore.shard_writer import ShardWriter

logger = logging.getLogger(__name__)

_COMPACTOR = "compactor"
_CLOSE_COMPACTION_MAX_ATTEMPTS = 3
_COMPACTION_INPUT_BYTES = 64 * 1024 * 1024
_COMPACTION_SHARD_BATCH_BYTES = 4 * 1024 * 1024
_COMPACTION_MAX_BATCH_ROWS = 4096
_COMPACTION_BUFFER_ROW_GROUPS = 4


class CompactionCoordinator(Protocol):
    def snapshot(self) -> ArchiveSnapshot: ...

    def commit(self, delta: CommitDelta, *, base: ArchiveSnapshot | None = None) -> CommitToken: ...


@dataclass(frozen=True)
class CompactionResult:
    """Rows written and duplicate inputs superseded by one compaction."""

    written: int
    superseded: int = 0


@dataclass
class _BufferedChunk:
    table: pa.Table
    retained: list[bool]


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


def _sorted_buffer(chunks: list[_BufferedChunk], sort_keys: list[tuple[str, str]]) -> pa.Table:
    tables = [
        chunk.table if all(chunk.retained) else chunk.table.filter(pa.array(chunk.retained))
        for chunk in chunks
        if any(chunk.retained)
    ]
    return pa.concat_tables(tables, promote_options="none").sort_by(sort_keys)


def _compaction_groups(
    batches: Iterator[pa.RecordBatch], schema: pa.Schema, primary_key: tuple[str, ...]
) -> Iterator[tuple[pa.Table, int]]:
    """Keep the newest version seen per key in each bounded Arrow buffer.

    A key may recur in a later buffer; read paths resolve those versions.
    """
    chunks: list[_BufferedChunk] = []
    keys_seen: dict[tuple, tuple[tuple[int, int, int], int, int]] = {}
    bytes_used = rows_used = superseded = 0
    sort_keys = [(key, "ascending") for key in primary_key]
    buffer_target = _COMPACTION_BUFFER_ROW_GROUPS * shard_writer.ROW_GROUP_TARGET_BYTES
    row_target = _COMPACTION_BUFFER_ROW_GROUPS * shard_writer.ROW_GROUP_ROWS
    for batch in batches:
        if chunks and (bytes_used + batch.nbytes > buffer_target or rows_used + batch.num_rows > row_target):
            yield _sorted_buffer(chunks, sort_keys), superseded
            chunks = []
            keys_seen = {}
            bytes_used = rows_used = superseded = 0

        key_columns = [batch.column(batch.schema.get_field_index(name)) for name in primary_key]
        commits = batch.column(batch.schema.get_field_index(SystemColumns.COMMIT))
        sequences = batch.column(batch.schema.get_field_index(SystemColumns.SEQUENCE))
        generations = batch.column(batch.schema.get_field_index(SystemColumns.GENERATION))
        candidates: dict[tuple, tuple[tuple[int, int, int], int]] = {}
        for index in range(batch.num_rows):
            key = tuple(column[index].as_py() for column in key_columns)
            version = (commits[index].as_py(), sequences[index].as_py() or 0, generations[index].as_py())
            candidate = candidates.get(key)
            if candidate is not None:
                if version > candidate[0]:
                    candidates[key] = (version, index)
                continue
            previous = keys_seen.get(key)
            if previous is None or version > previous[0]:
                candidates[key] = (version, index)

        superseded += batch.num_rows - len(candidates)
        if not candidates:
            continue
        for key in candidates:
            previous = keys_seen.get(key)
            if previous is not None:
                chunks[previous[1]].retained[previous[2]] = False
                superseded += 1

        selected = sorted(((index, key, version) for key, (version, index) in candidates.items()))
        source = pa.Table.from_batches([batch], schema=schema)
        chunk = source if len(selected) == batch.num_rows else source.take(pa.array([item[0] for item in selected]))
        chunk_index = len(chunks)
        chunks.append(_BufferedChunk(chunk, [True] * chunk.num_rows))
        for position, (_index, key, version) in enumerate(selected):
            keys_seen[key] = (version, chunk_index, position)
        bytes_used += chunk.nbytes
        rows_used += chunk.num_rows
    if chunks:
        yield _sorted_buffer(chunks, sort_keys), superseded


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
    if SystemColumns.GENERATION not in unified.names:
        unified = unified.append(pa.field(SystemColumns.GENERATION, pa.int32()))

    written = 0
    superseded = 0
    outputs: list[Shard] = []
    profile = _ScanProfile(batch_rows=1, batch_readahead=1, fragment_readahead=1, pre_buffer=False)
    with ExitStack() as stack:
        inputs = sorted(
            zip(shards, metadata, strict=True),
            key=lambda item: (item[0].commit_sequence, item[0].generation),
            reverse=True,
        )
        streams = [
            stack.enter_context(
                closing(
                    iter_shard_batches(
                        shard,
                        unified,
                        primary_key,
                        pa_fs,
                        scan_profile=replace(profile, batch_rows=_scan_batch_rows(item, len(shards))),
                    )
                )
            )
            for shard, item in inputs
        ]
        # Interleave sorted shard scans so versions of nearby keys can meet in one buffer.
        batches = (batch for group in zip_longest(*streams) for batch in group if batch is not None)
        for batch, dropped in _compaction_groups(batches, unified, primary_key):
            output_path = layout.shard_path(table, _COMPACTOR, next_generation, 0, uuid.uuid4().hex[:8])
            with ShardWriter(output_path, unified) as writer:
                rows = (batch.slice(index, 1) for index in range(batch.num_rows))
                for row_group in shard_writer.table_row_groups(table, rows):
                    writer.write_table(row_group)
            sequences = pc.fill_null(batch[SystemColumns.SEQUENCE], 0)
            outputs.append(
                Shard(
                    path=output_path,
                    writer=_COMPACTOR,
                    generation=next_generation,
                    rows=batch.num_rows,
                    min_seq=pc.min(sequences).as_py(),
                    max_seq=pc.max(sequences).as_py(),
                    size_bytes=writer.result.size_bytes,
                    content_sha256=writer.result.content_sha256,
                    primary_key_sorted=True,
                )
            )
            written += batch.num_rows
            superseded += dropped

    if not outputs:
        raise ValueError(f"FineStore table {table!r} has {len(shards)} non-empty shard descriptors but no rows")
    try:
        coordinator.commit(
            CommitDelta(
                replacements={
                    table: TableReplacement(
                        input_paths=frozenset(shard.path for shard in shards), output_shards=tuple(outputs)
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
