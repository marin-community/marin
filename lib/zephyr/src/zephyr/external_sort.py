# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-pass external merge sort for large k-way merges.

Used by the reduce stage when the number of sorted chunk iterators exceeds
``EXTERNAL_SORT_FAN_IN``, to avoid opening O(k) scanners simultaneously and
exhausting worker memory.

Pass 1: batch the k iterators into groups of ``fan_in`` (defaulting to
``EXTERNAL_SORT_FAN_IN`` but typically lowered via :func:`compute_fan_in` to
fit the worker's memory budget), merge each group with ``heapq.merge``, and
spill items to a run file under ``{external_sort_dir}/run-{i:04d}.spill`` via
:class:`SpillWriter`.

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

# Hard cap on simultaneous chunk iterators per pass-1 batch. Used as the
# default when the caller cannot estimate per-iterator memory; otherwise
# ``compute_fan_in`` lowers it to fit within the worker's memory budget.
EXTERNAL_SORT_FAN_IN = 500

# Fraction of worker memory budgeted for the open chunk iterators during a
# pass-1 merge batch.
_FAN_IN_MEMORY_FRACTION = 0.5

# Floor on fan-in. Below 2, pass-1 just rewrites each chunk to its own run
# file with no merging — pass-2 still produces correct output but the extra
# round-trip is wasteful, so we keep at least a small merge fan-in.
_FAN_IN_FLOOR = 4

# Default item count per write into the SpillWriter in pass-1. Large enough
# for good compression + low per-call overhead. For large items (e.g. 1 MB
# each) the caller should pass a smaller ``write_batch_size`` via
# :func:`compute_write_batch_size` so the in-memory ``pending`` buffer stays
# bounded by bytes rather than count.
_WRITE_BATCH_SIZE = 10_000

# Target bytes for the in-memory pass-1 spill buffer.
_WRITE_BATCH_TARGET_BYTES = 64 * 1024 * 1024

# Target bytes per spill chunk in pass-1 runs.
_ROW_GROUP_BYTES = 8 * 1024 * 1024

# Fraction of container memory budgeted for pass-2 read buffers.
_READ_MEMORY_FRACTION = 0.25


def compute_fan_in(per_iterator_bytes: int, memory_limit: int) -> int:
    """Pick a pass-1 fan-in that fits within the memory budget.

    ``per_iterator_bytes`` is the caller's estimate of memory held per open
    chunk iterator (typically compressed chunk bytes plus a small decoded
    buffer). Returns at least ``_FAN_IN_FLOOR`` and at most
    ``EXTERNAL_SORT_FAN_IN``.
    """
    if per_iterator_bytes <= 0 or memory_limit <= 0:
        return EXTERNAL_SORT_FAN_IN
    budget = int(memory_limit * _FAN_IN_MEMORY_FRACTION)
    fan_in = budget // per_iterator_bytes
    fan_in = max(_FAN_IN_FLOOR, fan_in)
    return min(fan_in, EXTERNAL_SORT_FAN_IN)


def compute_write_batch_size(avg_item_bytes: float) -> int:
    """Pick a pass-1 pending-buffer size sized to a byte budget.

    Caps at the ``_WRITE_BATCH_SIZE`` default when items are small.
    """
    if avg_item_bytes <= 0:
        return _WRITE_BATCH_SIZE
    by_bytes = int(_WRITE_BATCH_TARGET_BYTES // avg_item_bytes)
    return max(1, min(by_bytes, _WRITE_BATCH_SIZE))


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
    fan_in: int = EXTERNAL_SORT_FAN_IN,
    write_batch_size: int = _WRITE_BATCH_SIZE,
) -> Iterator:
    """Merge ``chunk_iterators_gen`` via a two-pass external sort.

    Args:
        chunk_iterators_gen: Lazy iterator of sorted iterators (one per scatter chunk).
            Consumed in batches of ``fan_in`` to avoid opening all file
            handles simultaneously.
        merge_key: Key function passed to heapq.merge.
        external_sort_dir: GCS prefix for spill files, e.g.
            ``gs://bucket/.../stage1-external-sort/shard-0042``.
        fan_in: Maximum number of chunk iterators to merge in one pass-1
            batch. Defaults to ``EXTERNAL_SORT_FAN_IN``; callers should pass
            a value computed by :func:`compute_fan_in` to bound memory.
        write_batch_size: Item count threshold for the pass-1 ``pending``
            buffer. Callers should pass a value from
            :func:`compute_write_batch_size` to keep the buffer bounded by
            bytes rather than item count.

    Yields:
        Items in merged sort order.
    """
    run_paths: list[str] = []
    batch_idx = 0

    # SpillWriter does not auto-create parent directories, so ensure the spill
    # dir exists up front.
    spill_fs, spill_dir = url_to_fs(external_sort_dir)
    spill_fs.makedirs(spill_dir, exist_ok=True)

    logger.info("External sort: pass-1 fan_in=%d, write_batch_size=%d", fan_in, write_batch_size)

    while True:
        batch = list(islice(chunk_iterators_gen, fan_in))
        if not batch:
            break
        run_path = f"{external_sort_dir}/run-{batch_idx:04d}.spill"
        item_count = 0
        pending: list = []
        with SpillWriter(run_path, row_group_bytes=_ROW_GROUP_BYTES) as writer:
            for item in heapq.merge(*batch, key=merge_key):
                pending.append(item)
                if len(pending) >= write_batch_size:
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
