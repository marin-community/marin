# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manifest-driven logical compaction for FineStore tables."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq
import rigging.filesystem.factory as factory
from pyarrow.fs import FSSpecHandler, PyFileSystem

from finestore.commit import ArchiveSnapshot, CommitConflict, CommitCoordinator, CommitDelta, TableReplacement
from finestore.layout import COMMIT_COLUMN, SEQ_COLUMN, CommitToken, FineStoreLayout, Shard
from finestore.reader import ReadView, iter_shard_rows, merge_deduplicated_rows
from finestore.shard_writer import ROW_GROUP_ROWS, ShardWriter

logger = logging.getLogger(__name__)

_COMPACTOR = "compactor"
_COMPACT_BATCH_ROWS = ROW_GROUP_ROWS


class CompactionCoordinator(Protocol):
    def snapshot(self) -> ArchiveSnapshot: ...

    def commit(self, delta: CommitDelta, *, base: ArchiveSnapshot | None = None) -> CommitToken: ...


@dataclass(frozen=True)
class CompactionResult:
    """Rows written and duplicate inputs superseded by one compaction."""

    written: int
    superseded: int = 0


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
    merged_rows = merge_deduplicated_rows([iter_shard_rows(shard, unified, primary_key, pa_fs) for shard in shards])

    def survivor_rows() -> Iterator[dict]:
        for merged in merged_rows:
            superseded[0] += merged.superseded
            yield merged.row

    survivors = survivor_rows()
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
