# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-pass external merge sort using Parquet spill files.

Used by the reduce stage when the number of sorted chunk iterators exceeds
what fits in memory, to avoid opening O(k) scanners simultaneously and
exhausting worker memory.

Pass 1: batch Arrow table iterators into budget-sized groups,
concatenate + sort in Arrow, write as Parquet run files.

Pass 2: streaming k-way merge over the sorted run files, reading one row
group at a time per run. Memory is bounded by O(k * row_group_size + output_batch_size).

All buffer sizes are derived from the worker's memory budget, probed from
actual data sizes — no estimated avg_item_bytes parameter needed.
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

logger = logging.getLogger(__name__)

# Fraction of worker memory available for sort (pass 1 and pass 2 are
# sequential, so both use the same budget).
_SORT_MEMORY_FRACTION = 0.5

# Pass 1 sort amplification factor: input + sort indices + sorted copy.
_SORT_AMPLIFICATION = 3

# Target spill row group size in bytes. Each run holds one row group in
# memory during merge, so this controls per-run memory footprint.
_SPILL_ROW_GROUP_TARGET_BYTES = 8 * 1024 * 1024  # 8 MB

# Output batch size yielded from the merge.
_MERGE_OUTPUT_BATCH_SIZE = 100_000


@dataclass
class _SortBudget:
    """Derived budget parameters, computed once from probed data sizes."""

    sort_budget_bytes: int
    fan_in: int


def _compute_budget(chunk_bytes: int) -> _SortBudget:
    """Compute sort budget from probed chunk size and worker memory."""
    memory_bytes = _TaskResources.from_environment().memory_bytes
    if memory_bytes <= 0:
        # Unknown memory (local dev, tests) — assume 16GB.
        memory_bytes = 16 * 1024**3
    sort_budget = int(memory_bytes * _SORT_MEMORY_FRACTION)

    effective_capacity = sort_budget // _SORT_AMPLIFICATION
    fan_in = max(1, min(1000, effective_capacity // max(chunk_bytes, 1)))

    budget = _SortBudget(sort_budget_bytes=sort_budget, fan_in=fan_in)
    logger.info(
        "External sort budget: memory=%dMB, sort_budget=%dMB, fan_in=%d, chunk_bytes=%dMB",
        memory_bytes // (1024 * 1024),
        sort_budget // (1024 * 1024),
        fan_in,
        chunk_bytes // (1024 * 1024),
    )
    return budget


def _write_spill_file(table: pa.Table, path: str) -> None:
    """Write a sorted table as a Parquet file with byte-budgeted row groups."""
    writer = pq.ParquetWriter(path, table.schema)
    offset = 0
    n = len(table)
    while offset < n:
        # Grow the row group until we hit the byte target.
        # Double the slice size each probe to keep overhead O(log n).
        lo = offset + 1
        hi = n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if table.slice(offset, mid - offset).nbytes <= _SPILL_ROW_GROUP_TARGET_BYTES:
                lo = mid
            else:
                hi = mid - 1
        rg_end = lo
        writer.write_table(table.slice(offset, rg_end - offset))
        offset = rg_end
    writer.close()


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


@dataclass(order=True)
class _MergeEntry:
    """Heap entry keyed by the sort value at the current cursor position."""

    sort_value: tuple
    source_idx: int = field(compare=True)
    source: "_RunSource" = field(compare=False, repr=False)


@dataclass
class _RunSource:
    """Read position within a single sorted run file."""

    idx: int
    pf: pq.ParquetFile
    sort_key_columns: list[str]
    _rg_idx: int = field(init=False, default=0)
    table: pa.Table | None = field(init=False, default=None)
    cursor: int = field(init=False, default=0)

    def advance(self) -> bool:
        """Load next row group. Returns False if exhausted."""
        while self._rg_idx < self.pf.metadata.num_row_groups:
            self.table = self.pf.read_row_group(self._rg_idx)
            self._rg_idx += 1
            self.cursor = 0
            if len(self.table) > 0:
                return True
        self.table = None
        return False

    def current_sort_value(self) -> tuple:
        return tuple(self.table.column(c)[self.cursor].as_py() for c in self.sort_key_columns)

    def remaining(self) -> int:
        return len(self.table) - self.cursor

    def take(self, count: int) -> pa.Table:
        """Slice count rows from cursor, advancing to next row group if needed."""
        sliced = self.table.slice(self.cursor, count)
        self.cursor += count
        if self.cursor >= len(self.table):
            self.advance()
        return sliced

    def rows_le(self, threshold: tuple) -> int:
        """Count rows from cursor whose sort key <= threshold.

        Exploits the fact that each row group is sorted — uses vectorized
        comparison to find the first row exceeding the threshold.
        """
        remaining_rows = self.remaining()
        primary_col = self.table.column(self.sort_key_columns[0]).slice(self.cursor, remaining_rows)

        if len(self.sort_key_columns) == 1:
            gt_mask = pc.greater(primary_col, pa.scalar(threshold[0], type=primary_col.type))
            gt_count = pc.sum(gt_mask).as_py() or 0
            return max(1, remaining_rows - gt_count)

        # Two sort keys: primary, secondary
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


def _streaming_k_way_merge(
    run_paths: list[str],
    sort_keys: list[tuple[str, str]],
) -> Iterator[pa.Table]:
    """Streaming k-way merge over sorted Parquet run files.

    Reads one row group at a time per run, uses a min-heap to pick the
    source with the smallest current key, and yields batches of sorted rows.
    Memory is bounded by O(k * row_group_size + output_batch_size).
    """
    sort_key_columns = [col for col, _ in sort_keys]

    sources: list[_RunSource] = []
    for i, path in enumerate(run_paths):
        src = _RunSource(idx=i, pf=pq.ParquetFile(path), sort_key_columns=sort_key_columns)
        if src.advance():
            sources.append(src)

    if not sources:
        return

    heap: list[_MergeEntry] = []
    for src in sources:
        heapq.heappush(heap, _MergeEntry(src.current_sort_value(), src.idx, src))

    output_chunks: list[pa.Table] = []
    output_rows = 0

    while heap:
        entry = heapq.heappop(heap)
        winner = entry.source

        if heap:
            next_key = heap[0].sort_value
            take_count = winner.rows_le(next_key)
        else:
            take_count = winner.remaining()

        chunk = winner.take(take_count)
        output_chunks.append(chunk)
        output_rows += len(chunk)

        if winner.has_data:
            heapq.heappush(heap, _MergeEntry(winner.current_sort_value(), winner.idx, winner))

        if output_rows >= _MERGE_OUTPUT_BATCH_SIZE:
            yield pa.concat_tables(output_chunks, promote_options="default")
            output_chunks.clear()
            output_rows = 0

    if output_chunks:
        yield pa.concat_tables(output_chunks, promote_options="default")


def external_sort_merge(
    chunk_tables_gen: Iterator[pa.Table],
    sort_keys: list[tuple[str, str]],
    external_sort_dir: str,
) -> Iterator[pa.Table]:
    """Two-pass external sort yielding sorted Arrow tables.

    Pass 1: batch tables into budget-sized groups (fan-in derived from
    worker memory and probed chunk size), concat + sort, write as Parquet runs.
    Pass 2: streaming k-way merge over the sorted run files.

    Passes are strictly sequential — they share the same memory budget.
    """
    from zephyr.writers import ensure_parent_dir

    # Probe the first chunk to derive budget parameters.
    first = next(chunk_tables_gen, None)
    if first is None:
        return
    budget = _compute_budget(first.nbytes)

    # Chain the first chunk back so it isn't lost.
    chunk_tables_gen = itertools.chain([first], chunk_tables_gen)

    run_paths: list[str] = []
    batch_idx = 0

    while True:
        batch_tables = list(islice(chunk_tables_gen, budget.fan_in))
        if not batch_tables:
            break
        combined = pa.concat_tables([_promote_to_large_string(t) for t in batch_tables], promote_options="default")
        indices = pc.sort_indices(combined, sort_keys=sort_keys)
        sorted_table = combined.take(indices)

        run_path = f"{external_sort_dir}/run-{batch_idx:04d}.parquet"
        ensure_parent_dir(run_path)
        _write_spill_file(sorted_table, run_path)
        run_paths.append(run_path)
        logger.info(
            "External sort: wrote run %d (%d rows, %dMB) to %s",
            batch_idx + 1,
            len(sorted_table),
            sorted_table.nbytes // (1024 * 1024),
            run_path,
        )
        batch_idx += 1
        del combined, sorted_table

    if not run_paths:
        return

    # Pass 2: verify merge memory fits in budget using actual Parquet metadata.
    # Use pyarrow-native path resolution (no filesystem arg) so reads go through
    # the same GcsFileSystem that wrote the files, avoiding gcsfs dir-cache staleness.
    num_runs = len(run_paths)
    max_rg_bytes = 0
    for rp in run_paths:
        meta = pq.read_metadata(rp)
        for i in range(meta.num_row_groups):
            max_rg_bytes = max(max_rg_bytes, meta.row_group(i).total_byte_size)
    merge_estimate = num_runs * max_rg_bytes
    if merge_estimate > budget.sort_budget_bytes:
        logger.warning(
            "External sort merge may exceed budget: %d runs x %.0fMB/rg = %.0fMB > %dMB budget",
            num_runs,
            max_rg_bytes / (1024 * 1024),
            merge_estimate / (1024 * 1024),
            budget.sort_budget_bytes // (1024 * 1024),
        )

    try:
        if len(run_paths) == 1:
            pf = pq.ParquetFile(run_paths[0])
            for i in range(pf.metadata.num_row_groups):
                yield pf.read_row_group(i)
        else:
            yield from _streaming_k_way_merge(run_paths, sort_keys)
    finally:
        fs, _ = fsspec.core.url_to_fs(external_sort_dir)
        for path in run_paths:
            try:
                _, fs_path = fsspec.core.url_to_fs(path)
                fs.rm(fs_path)
            except Exception:
                pass
