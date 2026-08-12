# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manifest-driven logical compaction for FineStore tables."""

from __future__ import annotations

import heapq
import itertools
import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from pyarrow.fs import FSSpecHandler, PyFileSystem
from rigging.filesystem import factory

from finestore.commit import ArchiveSnapshot, CommitConflict, CommitCoordinator, CommitDelta, TableReplacement
from finestore.layout import COMMIT_COLUMN, SEQ_COLUMN, CommitToken, FineStoreLayout, Shard
from finestore.reader import ReadView
from finestore.shard_writer import ROW_GROUP_ROWS, ShardWriter

logger = logging.getLogger(__name__)

_COMPACTOR = "compactor"
_COMPACT_BATCH_ROWS = ROW_GROUP_ROWS
_MergeItem = tuple[tuple, int, int, int, dict]


class CompactionCoordinator(Protocol):
    def snapshot(self) -> ArchiveSnapshot: ...

    def commit(self, delta: CommitDelta, *, base: ArchiveSnapshot | None = None) -> CommitToken: ...


def _key_tuple(row: dict, primary_key: tuple[str, ...]) -> tuple:
    return tuple((row.get(name) is None, row.get(name)) for name in primary_key)


def _shard_rows(shard: Shard, unified: pa.Schema, primary_key: tuple[str, ...], pa_fs) -> Iterator[_MergeItem]:
    dataset = pds.dataset([shard.path], filesystem=pa_fs, format="parquet", schema=unified)
    if not shard.primary_key_sorted:
        table = dataset.to_table()
        sort_columns = [(name, "ascending") for name in primary_key if name in unified.names]
        if sort_columns:
            table = table.sort_by(sort_columns)
        batches = table.to_batches()
    else:
        batches = dataset.scanner(use_threads=False).to_batches()
    for batch in batches:
        for row in batch.to_pylist():
            commit_sequence = row.get(COMMIT_COLUMN)
            if commit_sequence is None:
                commit_sequence = shard.commit_sequence
            row[COMMIT_COLUMN] = commit_sequence
            yield (
                _key_tuple(row, primary_key),
                commit_sequence,
                row.get(SEQ_COLUMN) or 0,
                shard.generation,
                row,
            )


@dataclass(frozen=True)
class CompactionResult:
    """Rows written and duplicate inputs superseded by one compaction."""

    written: int
    superseded: int = 0


def _merge_dedup(streams: list[Iterator[_MergeItem]], counter: list[int]) -> Iterator[dict]:
    merged = heapq.merge(*streams, key=lambda item: item[0])
    for _key, group in itertools.groupby(merged, key=lambda item: item[0]):
        items = list(group)
        counter[0] += len(items) - 1
        yield max(items, key=lambda item: (item[1], item[2], item[3]))[4]


def compact(root: str, table: str, *, coordinator: CompactionCoordinator | None = None) -> CompactionResult:
    """Replace the currently active shards for ``table`` through one manifest commit.

    Source objects remain immutable and reachable by older read views. Garbage collection is
    deliberately separate from compaction because the store does not yet track reader leases.
    """
    layout = FineStoreLayout(root)
    commits = coordinator or CommitCoordinator(layout)
    snapshot = commits.snapshot()
    view = ReadView(root, snapshot)
    shards = view.list_shards(table)
    if not shards:
        return CompactionResult(written=0)
    primary_key = view.primary_key(table)
    next_generation = max(shard.generation for shard in shards) + 1

    fs, _ = factory.url_to_fs(root)
    pa_fs = PyFileSystem(FSSpecHandler(fs))
    unified = pa.unify_schemas(
        [pq.read_schema(shard.path, filesystem=pa_fs) for shard in shards], promote_options="permissive"
    )
    if COMMIT_COLUMN not in unified.names:
        unified = unified.append(pa.field(COMMIT_COLUMN, pa.int64()))

    superseded = [0]
    survivors = _merge_dedup([_shard_rows(shard, unified, primary_key, pa_fs) for shard in shards], superseded)
    first = next(survivors, None)
    if first is None:
        return CompactionResult(written=0)

    output_path = layout.shard_path(table, _COMPACTOR, next_generation, 0, uuid.uuid4().hex[:8])
    written = 0
    min_seq: int | None = None
    max_seq: int | None = None
    with ShardWriter(output_path, unified) as writer:
        batch = [first]
        for row in survivors:
            batch.append(row)
            if len(batch) >= _COMPACT_BATCH_ROWS:
                writer.write_table(pa.Table.from_pylist(batch, schema=unified))
                sequences = [row.get(SEQ_COLUMN) or 0 for row in batch]
                min_seq = min(sequences) if min_seq is None else min(min_seq, *sequences)
                max_seq = max(sequences) if max_seq is None else max(max_seq, *sequences)
                written += len(batch)
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=unified))
            sequences = [row.get(SEQ_COLUMN) or 0 for row in batch]
            min_seq = min(sequences) if min_seq is None else min(min_seq, *sequences)
            max_seq = max(sequences) if max_seq is None else max(max_seq, *sequences)
            written += len(batch)

    assert min_seq is not None and max_seq is not None
    output = Shard(
        path=output_path,
        writer=_COMPACTOR,
        generation=next_generation,
        rows=written,
        min_seq=min_seq,
        max_seq=max_seq,
        primary_key_sorted=True,
    )
    try:
        commits.commit(
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
        logger.info("FineStore abandoned compaction for %s because its inputs changed", table)
        return CompactionResult(written=0)
    logger.info(
        "FineStore compacted %s to generation %d (%d rows, %d superseded)",
        table,
        next_generation,
        written,
        superseded[0],
    )
    return CompactionResult(written=written, superseded=superseded[0])
