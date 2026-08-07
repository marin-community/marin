# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compaction: merge a table's small shards into one sorted, deduplicated next-generation shard.

Compaction is an optimization, not a correctness requirement — a reader deduplicates regardless. It
streams a k-way merge over the table's shards in merge-key order and writes the surviving rows to the
next generation, one row group at a time, then optionally deletes the shards it superseded. Because
the merged rows keep their original ``_seq`` and the new shard has a higher generation, a reader
prefers it; a crash between the write and the delete leaves both, and dedup still returns one row per
key.

The merge is bounded in memory. A compacted shard (generation >= 1) was written in merge-key order,
so it streams a row group at a time; a level-0 shard is one flush (bounded by the writer's row cap),
so it is sorted in memory. Neither path materializes the whole table, so an archive far larger than
memory still compacts.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from pyarrow.fs import FSSpecHandler, PyFileSystem
from rigging.filesystem import factory

from finestore.layout import SEQ_COLUMN, FineStoreLayout, Shard
from finestore.reader import CompositeReader
from finestore.shard_writer import ROW_GROUP_ROWS, ShardWriter

logger = logging.getLogger(__name__)

# The writer identity stamped on compacted shards, distinct from any live append writer.
_COMPACTOR = "compactor"

# Surviving rows the merge accumulates before writing one output batch. Matched to the writer's
# row-group cap so a compacted shard's groups are the same size as a flush's, keeping read pruning
# uniform; it also bounds the merge's working set independently of the archive's total size.
_COMPACT_BATCH_ROWS = ROW_GROUP_ROWS

# One item in the merge heap: (merge-key tuple, generation, seq, row dict). The key sorts the merge;
# seq then generation breaks ties so the latest write of a key wins (same rule the reader applies).
_MergeItem = tuple[tuple, int, int, dict]


def _key_tuple(row: dict, merge_key: tuple[str, ...]) -> tuple:
    """A null-safe, order-stable merge-key tuple: ``None`` sorts before any value without comparing it."""
    return tuple((row.get(name) is None, row.get(name)) for name in merge_key)


def _shard_rows(shard: Shard, unified: pa.Schema, merge_key: tuple[str, ...], pa_fs) -> Iterator[_MergeItem]:
    """Yield a shard's rows in merge-key order, each tagged for the merge's tie-break.

    A level-0 shard is unsorted but bounded by the writer's row cap, so it is sorted in memory; a
    compacted shard was written in merge-key order, so its row groups stream in order. Reading through
    the unified schema promotes columns a shard lacks to null.
    """
    dataset = pds.dataset([shard.path], filesystem=pa_fs, format="parquet", schema=unified)
    if shard.generation == 0:
        table = dataset.to_table()
        sort_columns = [(name, "ascending") for name in merge_key if name in unified.names]
        if sort_columns:
            table = table.sort_by(sort_columns)
        batches = table.to_batches()
    else:
        # A compacted shard is already in merge-key order; scan it single-threaded so its row groups
        # stream in that order (a threaded scan may reorder them, breaking the merge invariant).
        batches = dataset.scanner(use_threads=False).to_batches()
    for batch in batches:
        for row in batch.to_pylist():
            yield _key_tuple(row, merge_key), shard.generation, row.get(SEQ_COLUMN) or 0, row


@dataclass(frozen=True)
class CompactionResult:
    """What one compaction did: rows written, and rows a later write of the same key replaced.

    ``superseded`` counts inputs that lost their merge key to a higher ``(seq, generation)`` row. It
    is zero for a table whose keys are unique, so a non-zero count is the signal that supersession
    actually happened rather than an assumption that it did not.
    """

    written: int
    superseded: int = 0


def _merge_dedup(streams: list[Iterator[_MergeItem]], merge_key: tuple[str, ...], counter: list[int]) -> Iterator[dict]:
    """Merge per-shard sorted streams into one merge-key-ordered stream, one surviving row per key.

    Losing rows are counted into ``counter[0]`` so the caller can report supersession instead of
    discarding it without trace.
    """
    merged = heapq.merge(*streams, key=lambda item: item[0])
    for _key, group in itertools.groupby(merged, key=lambda item: item[0]):
        items = list(group)
        counter[0] += len(items) - 1
        winner = max(items, key=lambda item: (item[2], item[1]))
        yield winner[3]


def compact(root: str, table: str, *, delete_source: bool = True) -> CompactionResult:
    """Compact ``table`` under ``root`` into one sorted shard; report rows written and superseded.

    Writes nothing when the table is empty or has no shards. When ``delete_source`` is set, shards
    from generations below the new one are removed after the merged shard is published.
    """
    reader = CompositeReader(root)
    shards = reader.list_shards(table)
    if not shards:
        return CompactionResult(written=0)
    merge_key = reader.merge_key(table)
    next_generation = max(shard.generation for shard in shards) + 1

    fs, _ = factory.url_to_fs(root)
    pa_fs = PyFileSystem(FSSpecHandler(fs))
    unified = pa.unify_schemas(
        [pq.read_schema(shard.path, filesystem=pa_fs) for shard in shards], promote_options="permissive"
    )

    streams = [_shard_rows(shard, unified, merge_key, pa_fs) for shard in shards]
    superseded = [0]
    survivors = _merge_dedup(streams, merge_key, superseded)
    first = next(survivors, None)
    if first is None:
        return CompactionResult(written=0)

    out_path = FineStoreLayout(root).shard_path(table, _COMPACTOR, next_generation, 0, uuid.uuid4().hex[:8])
    written = 0
    with ShardWriter(out_path, unified) as writer:
        batch = [first]
        for row in survivors:
            batch.append(row)
            if len(batch) >= _COMPACT_BATCH_ROWS:
                writer.write_table(pa.Table.from_pylist(batch, schema=unified))
                written += len(batch)
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=unified))
            written += len(batch)
    logger.info(
        "finestore compacted %s to generation %d (%d rows, %d superseded)",
        table,
        next_generation,
        written,
        superseded[0],
    )

    if delete_source:
        for shard in shards:
            if shard.generation < next_generation:
                fs.rm(shard.path)
    return CompactionResult(written=written, superseded=superseded[0])
