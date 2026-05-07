# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-pass external merge sort over Arrow RecordBatch streams.

Used by the reduce stage when the number of scatter chunks exceeds
``EXTERNAL_SORT_FAN_IN`` or the estimated merge memory exceeds the worker
budget.

Pass 1: consume the input RecordBatch stream in groups of ``fan_in`` batches.
Merge each group and spill the sorted result to a runfile via :class:`SpillWriter`.

Pass 2: read all run files via :class:`SpillReader`, load their batches into
Polars DataFrames, and merge.  Yield the mergedRecordBatches.  Run files are
deleted after the merge completes.
"""

import heapq
import logging
from collections.abc import Iterator
from itertools import islice

import cloudpickle
import pyarrow as pa
import pyarrow.compute as pc
from rigging.filesystem import url_to_fs

from zephyr.spill import SpillReader, SpillWriter

logger = logging.getLogger(__name__)

# Hard cap on pass-1 batch group size. Limits the number of RecordBatches held
# in memory simultaneously during each merge group.
EXTERNAL_SORT_FAN_IN = 500

_FAN_IN_MEMORY_FRACTION = 0.5
_FAN_IN_FLOOR = 4


def _to_large_type(t: pa.DataType) -> pa.DataType:
    """Convert a PyArrow data type to the larger version of that type."""
    if pa.types.is_string(t):
        return pa.large_string()
    elif pa.types.is_binary(t):
        return pa.large_binary()
    elif pa.types.is_list(t):
        return pa.large_list(_to_large_type(t.value_type))
    elif pa.types.is_struct(t):
        return pa.struct([pa.field(f.name, _to_large_type(f.type), f.nullable, f.metadata) for f in t])
    return t


def in_memory_k_way_merge(batches: Iterator[pa.RecordBatch], key: str) -> Iterator[pa.RecordBatch]:
    """Merge a list of sorted RecordBatches into a single sorted RecordBatch stream.

    concat_tables and sort_indices create a table that's larger than the input tables.
    If that growth results in going from a 32-bit size to a 64-bit size, PyArrow raises
    an overflow error. To avoid this, we cast everything to the larger type.
    """

    tables = [pa.Table.from_batches([b]) for b in batches]
    if not tables:
        return

    table = pa.concat_tables(tables, promote_options="permissive")
    large_schema = pa.schema([pa.field(f.name, _to_large_type(f.type), f.nullable, f.metadata) for f in table.schema])

    if large_schema != table.schema:
        table = table.cast(large_schema)

    sort_indices = pc.sort_indices(table, sort_keys=[(key, "ascending")])
    sorted_table = table.take(sort_indices)

    yield from sorted_table.to_batches()


def compute_fan_in(per_iterator_bytes: int, memory_limit: int) -> int:
    """Pick a pass-1 fan-in that fits within the memory budget.

    ``per_iterator_bytes`` is the caller's estimate of memory held per open
    chunk (typically the compressed chunk byte length). Returns at least
    ``_FAN_IN_FLOOR`` and at most ``EXTERNAL_SORT_FAN_IN``.
    """
    if per_iterator_bytes <= 0 or memory_limit <= 0:
        return EXTERNAL_SORT_FAN_IN
    budget = int(memory_limit * _FAN_IN_MEMORY_FRACTION)
    fan_in = budget // max(1, per_iterator_bytes)
    fan_in = max(_FAN_IN_FLOOR, fan_in)
    return min(fan_in, EXTERNAL_SORT_FAN_IN)


def external_sort_merge(
    batches: Iterator[pa.RecordBatch],
    sort_key: str,
    external_sort_dir: str,
    fan_in: int = EXTERNAL_SORT_FAN_IN,
) -> Iterator[pa.RecordBatch]:
    """Merge a sorted RecordBatch stream via a two-pass external sort.

    Args:
        batches: Flat stream of pa.RecordBatch objects. Each batch must be
            sorted by ``sort_key`` (ascending, binary lexicographic order).
        sort_key: Column name to sort by. Must be a binary column whose
            byte encoding preserves the desired sort order.
        external_sort_dir: Directory path for spill files, e.g.
            ``gs://bucket/.../stage1-external-sort/shard-0042``.
        fan_in: Maximum number of batches to merge per pass-1 group.
            Callers should pass a value from :func:`compute_fan_in` to bound
            peak memory during pass-1.

    Yields:
        pa.RecordBatch objects in merged sort order.
    """
    run_paths: list[str] = []
    batch_idx = 0

    spill_fs, spill_dir = url_to_fs(external_sort_dir)
    spill_fs.makedirs(spill_dir, exist_ok=True)

    logger.info("External sort: pass-1 fan_in=%d", fan_in)

    while True:
        batch_group = list(islice(batches, fan_in))
        if not batch_group:
            break
        merged = in_memory_k_way_merge(batch_group, sort_key)
        run_path = f"{external_sort_dir}/run-{batch_idx:04d}.spill"
        total_rows = 0
        with SpillWriter(run_path) as writer:
            for rb in merged:
                total_rows += rb.num_rows
                writer.write(rb)
        run_paths.append(run_path)
        logger.info("External sort: wrote run %d (%d rows) to %s", batch_idx + 1, total_rows, run_path)
        batch_idx += 1

    if not run_paths:
        return

    try:
        # For the second batch, we fall back to merging in Python because there does not appear to be a streaming k-way
        # merge in PyArrow. It's possible that we can do better using Polars, but that will require more trickery than
        # I or the agents have right now.
        schema = None

        def _read_run(path: str) -> Iterator:
            nonlocal schema

            for batch in SpillReader(path).iter_batches():
                if schema is None:
                    schema = batch.schema
                yield from batch.to_pylist()

        run_iters = [_read_run(p) for p in run_paths]
        fs_run_paths = [f"{spill_dir}/run-{i:04d}.spill" for i in range(len(run_paths))]

        for item in heapq.merge(*run_iters, key=lambda x: x[sort_key]):
            if "_payload" in item and item["_payload"] is not None:
                item = cloudpickle.loads(item["_payload"])
            if "__zephyr_sort_key__" in item:
                del item["__zephyr_sort_key__"]

            yield item

    finally:
        if fs_run_paths:
            try:
                spill_fs.rm(fs_run_paths)
            except Exception:
                # Spill files live under a per-shard temp dir that the worker
                # eventually wipes; log so leaked files are at least traceable.
                logger.warning("Failed to delete external-sort run files under %s", spill_dir, exc_info=True)
