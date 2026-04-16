# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-pass external merge sort for large k-way merges.

Used by the reduce stage when the number of sorted chunk iterators exceeds
``EXTERNAL_SORT_FAN_IN``, to avoid opening O(k) scanners simultaneously and
exhausting worker memory.

Pass 1: batch the k iterators into groups of EXTERNAL_SORT_FAN_IN, merge each
group with heapq.merge, and spill items to a run file under
``{external_sort_dir}/run-{i:04d}.spill`` via :class:`SpillWriter`.

Pass 2: heapq.merge over the (much smaller) set of run file iterators.  Each
iterator streams chunks from its spill file via :class:`SpillReader`; the read
batch size is computed from the cgroup memory limit so that all concurrent
batches together stay within ``_READ_MEMORY_FRACTION`` of available memory.

Run files are deleted after the final merge completes.
"""

import heapq
import logging
from collections.abc import Callable, Iterator
from itertools import islice

from iris.env_resources import TaskResources
from rigging.filesystem import url_to_fs

from zephyr.spill import SpillReader, SpillWriter

logger = logging.getLogger(__name__)

# Maximum simultaneous chunk iterators per pass-1 batch.
EXTERNAL_SORT_FAN_IN = 500

# Items buffered before handing to the SpillWriter. Larger values amortize
# per-chunk overhead in the spill format.
_WRITE_BATCH_SIZE = 10_000

# Target bytes per spill chunk in pass-1 runs.
_ROW_GROUP_BYTES = 8 * 1024 * 1024

# Fraction of container memory budgeted for pass-2 read buffers.
_READ_MEMORY_FRACTION = 0.25


def _safe_read_batch_size(n_runs: int, sample_run_path: str) -> int:
    """Compute a pass-2 read batch size that fits within the memory budget.

    Uses the spill's per-item byte estimate to divide the memory budget across
    concurrent run-file buffers so they together stay within
    ``_READ_MEMORY_FRACTION`` of available container memory.
    """
    try:
        item_bytes_raw = SpillReader(sample_run_path).approx_item_bytes
    except Exception:
        logger.warning(
            "Failed to read spill metadata from %s; falling back to default batch size",
            sample_run_path,
            exc_info=True,
        )
        return _WRITE_BATCH_SIZE

    if item_bytes_raw <= 0:
        return _WRITE_BATCH_SIZE

    # Payload size x 3 approximates Python object overhead (dicts are ~3x
    # larger in memory than their pickled form).
    item_bytes = max(64, item_bytes_raw * 3)

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

    # SpillWriter does not auto-create parent directories, so ensure the spill
    # dir exists up front.
    spill_fs, spill_dir = url_to_fs(external_sort_dir)
    spill_fs.makedirs(spill_dir, exist_ok=True)

    while True:
        batch = list(islice(chunk_iterators_gen, EXTERNAL_SORT_FAN_IN))
        if not batch:
            break
        run_path = f"{external_sort_dir}/run-{batch_idx:04d}.spill"
        item_count = 0
        pending: list = []
        with SpillWriter(run_path, row_group_bytes=_ROW_GROUP_BYTES) as writer:
            for item in heapq.merge(*batch, key=merge_key):
                pending.append(item)
                if len(pending) >= _WRITE_BATCH_SIZE:
                    writer.write(pending)
                    item_count += len(pending)
                    pending = []
            if pending:
                writer.write(pending)
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
        reader = SpillReader(path, batch_size=read_batch_size)
        for chunk in reader.iter_chunks():
            yield from chunk

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
