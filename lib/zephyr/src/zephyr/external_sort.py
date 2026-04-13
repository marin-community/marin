# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-pass external merge sort for large k-way merges.

Used by the reduce stage when the number of sorted chunk iterators exceeds
``EXTERNAL_SORT_FAN_IN``, to avoid opening O(k) scanners simultaneously and
exhausting worker memory.

Pass 1: batch the k iterators into groups of EXTERNAL_SORT_FAN_IN, merge each
group with heapq.merge, and spill items to a Parquet run file under
``{external_sort_dir}/run-{i:04d}.parquet``.  Items are pickled into a single
``_zephyr_payload`` binary column and written as byte-budgeted row groups via
:class:`SpillWriter`, overlapping serialization with background I/O.

Pass 2: heapq.merge over the (much smaller) set of run file iterators.  Each
iterator streams row groups from its parquet file, unpickling one batch at a
time; the read batch size is computed from the cgroup memory limit so that all
concurrent batches together stay within ``_READ_MEMORY_FRACTION`` of available
memory.

Run files are deleted after the final merge completes.
"""

import heapq
import logging
import pickle
from collections.abc import Callable, Iterator
from itertools import islice

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from iris.env_resources import TaskResources
from rigging.filesystem import url_to_fs

from zephyr.spill_writer import SpillWriter

logger = logging.getLogger(__name__)

# Maximum simultaneous chunk iterators per pass-1 batch.
EXTERNAL_SORT_FAN_IN = 500

# Items buffered before converting to an Arrow RecordBatch and handing to the
# SpillWriter. Larger values amortize Arrow/parquet overhead.
_WRITE_BATCH_SIZE = 10_000

# Target bytes per parquet row group in pass-1 spills.
_ROW_GROUP_BYTES = 8 * 1024 * 1024

# Fraction of container memory budgeted for pass-2 read buffers.
_READ_MEMORY_FRACTION = 0.25

_SPILL_SCHEMA = pa.schema([pa.field("_zephyr_payload", pa.binary())])


def _items_to_table(items: list) -> pa.Table:
    payloads = [pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL) for item in items]
    return pa.table({"_zephyr_payload": pa.array(payloads, type=pa.binary())})


def _safe_read_batch_size(n_runs: int, sample_run_path: str) -> int:
    """Compute a pass-2 read batch size that fits within the memory budget.

    Uses the parquet file's uncompressed byte size and row count to estimate
    in-memory bytes per item, then divides the memory budget by
    ``n_runs * item_bytes`` so that all concurrent run-file buffers together
    stay within ``_READ_MEMORY_FRACTION`` of available container memory.
    """
    try:
        with fsspec.open(sample_run_path, "rb") as f:
            pf = pq.ParquetFile(f)
            md = pf.metadata
            num_rows = md.num_rows
            total_bytes = sum(md.row_group(i).column(0).total_uncompressed_size for i in range(md.num_row_groups))
    except Exception:
        return _WRITE_BATCH_SIZE

    if num_rows <= 0:
        return _WRITE_BATCH_SIZE

    # Uncompressed payload size x 3 approximates Python object overhead (dicts
    # are ~3x larger in memory than their pickled form).
    item_bytes = max(64, (total_bytes // num_rows) * 3)

    available = TaskResources.from_environment().memory_bytes
    budget = int(available * _READ_MEMORY_FRACTION)
    size = budget // max(1, n_runs * item_bytes)
    result = max(100, min(size, _WRITE_BATCH_SIZE))
    logger.info(
        "External sort pass-2: %d runs x ~%d bytes/item, budget=%.1f GB -> read_batch_size=%d",
        n_runs,
        item_bytes,
        budget / 1e9,
        result,
    )
    return result


def external_sort_merge(
    chunk_iterators_gen: Iterator[Iterator],  # lazy — consumed in batches
    merge_key: Callable,
    external_sort_dir: str,
) -> Iterator:
    """Merge ``chunk_iterators_gen`` via a two-pass external sort.

    Args:
        chunk_iterators_gen: Lazy iterator of sorted iterators (one per scatter chunk).
            Consumed in batches of EXTERNAL_SORT_FAN_IN to avoid opening all file
            handles simultaneously.
        merge_key: Key function passed to heapq.merge.
        external_sort_dir: GCS prefix for spill files, e.g.
            ``gs://bucket/.../stage1-external-sort/shard-0042``.

    Yields:
        Items in merged sort order.
    """
    run_paths: list[str] = []
    batch_idx = 0

    # pq.ParquetWriter does not auto-create parent directories the way
    # fsspec.open("wb") did, so make sure the spill dir exists up front.
    spill_fs, spill_dir = url_to_fs(external_sort_dir)
    spill_fs.makedirs(spill_dir, exist_ok=True)

    while True:
        batch = list(islice(chunk_iterators_gen, EXTERNAL_SORT_FAN_IN))
        if not batch:
            break
        run_path = f"{external_sort_dir}/run-{batch_idx:04d}.parquet"
        item_count = 0
        pending: list = []
        with SpillWriter(run_path, _SPILL_SCHEMA, row_group_bytes=_ROW_GROUP_BYTES) as writer:
            for item in heapq.merge(*batch, key=merge_key):
                pending.append(item)
                if len(pending) >= _WRITE_BATCH_SIZE:
                    writer.write_table(_items_to_table(pending))
                    item_count += len(pending)
                    pending = []
            if pending:
                writer.write_table(_items_to_table(pending))
                item_count += len(pending)
        run_paths.append(run_path)
        logger.info(
            "External sort: wrote run %d (%d items) to %s",
            batch_idx + 1,
            item_count,
            run_path,
        )
        batch_idx += 1

    read_batch_size = _safe_read_batch_size(len(run_paths), run_paths[0]) if run_paths else _WRITE_BATCH_SIZE

    def _read_run(path: str) -> Iterator:
        with fsspec.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            for record_batch in pf.iter_batches(batch_size=read_batch_size, columns=["_zephyr_payload"]):
                payloads = record_batch.column("_zephyr_payload").to_pylist()
                for payload in payloads:
                    yield pickle.loads(payload)

    run_iters = [_read_run(p) for p in run_paths]
    try:
        yield from heapq.merge(*run_iters, key=merge_key)
    finally:
        for path in run_paths:
            try:
                rm_fs, rm_path = url_to_fs(path)
                rm_fs.rm(rm_path)
            except Exception:
                pass
