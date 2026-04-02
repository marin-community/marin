# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-pass external merge sort using Parquet spill files.

Used by the reduce stage when scatter chunks don't fit in memory.

Each scatter chunk is already sorted by (sort_key, sort_secondary).
Pass 1 streams batches of pre-sorted chunk tables through a k-way merge
(no re-sort needed), writing merged runs as Parquet spill files.

Pass 2: streaming k-way merge over the sorted run files, reading one row
group at a time per run.

All buffer sizes are derived from the worker's memory budget, probed from
actual data sizes.
"""

import heapq
import itertools
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import islice

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from iris.env_resources import TaskResources as _TaskResources

from zephyr.spill_writer import SpillWriter, TableAccumulator

logger = logging.getLogger(__name__)

# Fraction of worker memory available for merge (pass 1 and pass 2 are
# sequential, so both use the same budget).
_MERGE_MEMORY_FRACTION = 0.5

# Target spill row group size in bytes. Each run holds one row group in
# memory during merge, so this controls per-run memory footprint.
_SPILL_ROW_GROUP_TARGET_BYTES = 8 * 1024 * 1024  # 8 MB


@dataclass
class _MergeBudget:
    """Derived budget parameters, computed once from probed data sizes."""

    merge_budget_bytes: int
    fan_in: int


def _compute_budget(chunk_bytes: int) -> _MergeBudget:
    """Compute merge budget from probed chunk size and worker memory.

    fan_in = how many chunk tables we can hold in memory simultaneously
    during the k-way merge.
    """
    memory_bytes = _TaskResources.from_environment().memory_bytes
    if memory_bytes <= 0:
        memory_bytes = 16 * 1024**3
    merge_budget = int(memory_bytes * _MERGE_MEMORY_FRACTION)

    fan_in = max(1, min(1000, merge_budget // max(chunk_bytes, 1)))

    budget = _MergeBudget(merge_budget_bytes=merge_budget, fan_in=fan_in)
    logger.info(
        "External sort budget: memory=%dMB, merge_budget=%dMB, fan_in=%d, chunk_bytes=%dMB",
        memory_bytes // (1024 * 1024),
        merge_budget // (1024 * 1024),
        fan_in,
        chunk_bytes // (1024 * 1024),
    )
    return budget


def _write_spill_file(table: pa.Table, path: str) -> None:
    """Write a sorted table as a Parquet file with byte-budgeted row groups."""
    with SpillWriter(path, table.schema, row_group_bytes=_SPILL_ROW_GROUP_TARGET_BYTES) as w:
        w.write_table(table)


def _promote_to_large_string(table: pa.Table) -> pa.Table:
    """Cast string/binary columns to large_string/large_binary to avoid 2GB offset overflow on concat."""
    new_fields = []
    needs_cast = False
    for f in table.schema:
        if f.type == pa.string():
            new_fields.append(f.with_type(pa.large_string()))
            needs_cast = True
        elif f.type == pa.binary():
            new_fields.append(f.with_type(pa.large_binary()))
            needs_cast = True
        else:
            new_fields.append(f)
    if not needs_cast:
        return table
    return table.cast(pa.schema(new_fields))


# ---------------------------------------------------------------------------
# Merge sources
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _MergeEntry:
    """Heap entry keyed by the sort value at the current cursor position."""

    sort_value: tuple
    source_idx: int = field(compare=True)
    source: "_MergeSource" = field(compare=False, repr=False)


class _MergeSource:
    """Abstract source for the k-way merge. Provides a cursor over sorted rows."""

    sort_key_columns: list[str]
    table: pa.Table | None
    cursor: int

    def advance(self) -> bool:
        raise NotImplementedError

    def current_sort_value(self) -> tuple:
        return tuple(self.table.column(c)[self.cursor].as_py() for c in self.sort_key_columns)

    def remaining(self) -> int:
        return len(self.table) - self.cursor

    def take(self, count: int) -> pa.Table:
        sliced = self.table.slice(self.cursor, count)
        self.cursor += count
        if self.cursor >= len(self.table):
            self.advance()
        return sliced

    def rows_le(self, threshold: tuple) -> int:
        """Count rows from cursor whose sort key <= threshold.

        Exploits sorted data — uses vectorized comparison to find the first
        row exceeding the threshold.
        """
        remaining_rows = self.remaining()
        primary_col = self.table.column(self.sort_key_columns[0]).slice(self.cursor, remaining_rows)

        if len(self.sort_key_columns) == 1:
            gt_mask = pc.greater(primary_col, pa.scalar(threshold[0], type=primary_col.type))
            gt_count = pc.sum(gt_mask).as_py() or 0
            return max(1, remaining_rows - gt_count)

        secondary_col = self.table.column(self.sort_key_columns[1]).slice(self.cursor, remaining_rows)
        primary_gt = pc.greater(primary_col, pa.scalar(threshold[0], type=primary_col.type))
        primary_eq = pc.equal(primary_col, pa.scalar(threshold[0], type=primary_col.type))
        secondary_gt = pc.greater(secondary_col, pa.scalar(threshold[1], type=secondary_col.type))
        exceeds = pc.or_(primary_gt, pc.and_(primary_eq, secondary_gt))
        exceed_count = pc.sum(exceeds).as_py() or 0
        return max(1, remaining_rows - exceed_count)

    @property
    def has_data(self) -> bool:
        return self.table is not None


class _RunSource(_MergeSource):
    """Read position within a single sorted Parquet run file."""

    def __init__(self, idx: int, pf: pq.ParquetFile, sort_key_columns: list[str]):
        self.idx = idx
        self.pf = pf
        self.sort_key_columns = sort_key_columns
        self._rg_idx = 0
        self.table = None
        self.cursor = 0

    def advance(self) -> bool:
        while self._rg_idx < self.pf.metadata.num_row_groups:
            self.table = self.pf.read_row_group(self._rg_idx)
            self._rg_idx += 1
            self.cursor = 0
            if len(self.table) > 0:
                return True
        self.table = None
        return False


class _TableSource(_MergeSource):
    """Merge source backed by a single in-memory Arrow table (one scatter chunk)."""

    def __init__(self, idx: int, table: pa.Table, sort_key_columns: list[str]):
        self.idx = idx
        self.sort_key_columns = sort_key_columns
        self._loaded = table
        self.table = None
        self.cursor = 0

    def advance(self) -> bool:
        if self._loaded is not None:
            self.table = self._loaded
            self._loaded = None
            self.cursor = 0
            return len(self.table) > 0
        self.table = None
        return False


# ---------------------------------------------------------------------------
# K-way merge
# ---------------------------------------------------------------------------


def _streaming_k_way_merge(
    sources: list[_MergeSource],
    sort_keys: list[tuple[str, str]],
    output_batch_bytes: int = _SPILL_ROW_GROUP_TARGET_BYTES,
) -> Iterator[pa.Table]:
    """Streaming k-way merge over pre-sorted sources.

    Uses a min-heap to pick the source with the smallest current key,
    and yields batches of sorted rows.
    """
    if not sources:
        return

    heap: list[_MergeEntry] = []
    for i, src in enumerate(sources):
        heapq.heappush(heap, _MergeEntry(src.current_sort_value(), i, src))

    accumulator = TableAccumulator(output_batch_bytes)

    while heap:
        entry = heapq.heappop(heap)
        winner = entry.source

        if heap:
            next_key = heap[0].sort_value
            take_count = winner.rows_le(next_key)
        else:
            take_count = winner.remaining()

        chunk = winner.take(take_count)

        if winner.has_data:
            heapq.heappush(heap, _MergeEntry(winner.current_sort_value(), winner.idx, winner))

        merged = accumulator.add(chunk)
        if merged is not None:
            yield merged

    remaining = accumulator.flush()
    if remaining is not None:
        yield remaining


# ---------------------------------------------------------------------------
# External sort entry point
# ---------------------------------------------------------------------------


def external_sort_merge(
    chunk_tables_gen: Iterator[pa.Table],
    sort_keys: list[tuple[str, str]],
    external_sort_dir: str,
) -> Iterator[pa.Table]:
    """Two-pass external merge yielding sorted Arrow tables.

    Input chunk tables are assumed to be pre-sorted by the scatter writer.

    Pass 1: batch pre-sorted chunk tables into groups of fan_in, stream
    through a k-way merge, and write merged runs as Parquet spill files.
    No re-sort is needed — the merge is O(n) per batch.

    Pass 2: streaming k-way merge over the (much smaller) set of run files.
    """
    from zephyr.writers import ensure_parent_dir

    first = next(chunk_tables_gen, None)
    if first is None:
        return
    budget = _compute_budget(first.nbytes)

    chunk_tables_gen = itertools.chain([first], chunk_tables_gen)

    sort_key_columns = [col for col, _ in sort_keys]
    run_paths: list[str] = []
    batch_idx = 0

    while True:
        batch_tables = list(islice(chunk_tables_gen, budget.fan_in))
        if not batch_tables:
            break

        # Build merge sources from pre-sorted chunk tables
        sources: list[_MergeSource] = []
        for i, t in enumerate(batch_tables):
            t = _promote_to_large_string(t)
            src = _TableSource(idx=i, table=t, sort_key_columns=sort_key_columns)
            if src.advance():
                sources.append(src)

        if not sources:
            continue

        run_path = f"{external_sort_dir}/run-{batch_idx:04d}.parquet"
        ensure_parent_dir(run_path)

        # Stream k-way merge directly to Parquet spill file
        merged_iter = _streaming_k_way_merge(sources, sort_keys)
        first_merged = next(merged_iter, None)
        if first_merged is None:
            continue

        merged_rows = len(first_merged)
        writer = SpillWriter(run_path, first_merged.schema, row_group_bytes=_SPILL_ROW_GROUP_TARGET_BYTES)
        try:
            writer.write_table(first_merged)
            for merged_table in merged_iter:
                writer.write_table(merged_table)
                merged_rows += len(merged_table)
        finally:
            writer.close()

        # Free the batch tables now that they've been merged to disk
        del batch_tables, sources

        run_paths.append(run_path)
        logger.info(
            "External sort: wrote run %d (%d rows) to %s",
            batch_idx + 1,
            merged_rows,
            run_path,
        )
        batch_idx += 1

    if not run_paths:
        return

    # Pass 2: verify merge memory fits in budget using actual Parquet metadata.
    num_runs = len(run_paths)
    max_rg_bytes = 0
    for rp in run_paths:
        meta = pq.read_metadata(rp)
        for i in range(meta.num_row_groups):
            max_rg_bytes = max(max_rg_bytes, meta.row_group(i).total_byte_size)
    merge_estimate = num_runs * max_rg_bytes
    if merge_estimate > budget.merge_budget_bytes:
        logger.warning(
            "External sort merge may exceed budget: %d runs x %.0fMB/rg = %.0fMB > %dMB budget",
            num_runs,
            max_rg_bytes / (1024 * 1024),
            merge_estimate / (1024 * 1024),
            budget.merge_budget_bytes // (1024 * 1024),
        )

    try:
        if len(run_paths) == 1:
            pf = pq.ParquetFile(run_paths[0])
            for i in range(pf.metadata.num_row_groups):
                yield pf.read_row_group(i)
        else:
            run_sources: list[_MergeSource] = []
            for i, path in enumerate(run_paths):
                src = _RunSource(idx=i, pf=pq.ParquetFile(path), sort_key_columns=sort_key_columns)
                if src.advance():
                    run_sources.append(src)
            yield from _streaming_k_way_merge(run_sources, sort_keys)
    finally:
        fs, _ = fsspec.core.url_to_fs(external_sort_dir)
        for path in run_paths:
            try:
                _, fs_path = fsspec.core.url_to_fs(path)
                fs.rm(fs_path)
            except Exception:
                pass
