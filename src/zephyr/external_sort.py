# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-pass external merge sort for large k-way merges.

Used by the reduce stage when the number of sorted chunk iterators exceeds
``EXTERNAL_SORT_FAN_IN``, to avoid opening O(k) scanners simultaneously and
exhausting worker memory.

Pass 1: batch the k iterators into groups of EXTERNAL_SORT_FAN_IN, merge each
group with heapq.merge, and spill items in batches of ``_WRITE_BATCH_SIZE`` to
a zstd-compressed pickle run file under
``{external_sort_dir}/run-{i:04d}.pkl.zst``.  Items are streamed to disk
rather than accumulated in a list, so peak memory per batch is bounded by the
number of open iterators rather than their total item count.

Pass 2: heapq.merge over the (much smaller) set of run file iterators.  Each
iterator reads one batch at a time and yields items one-by-one; the read batch
size is computed from the cgroup memory limit so that all concurrent batches
together stay within ``_READ_MEMORY_FRACTION`` of available memory.

Run files are deleted after the final merge completes.
"""

import heapq
import logging
import pickle
from collections.abc import Callable, Iterator
from itertools import islice

import fsspec
import zstandard as zstd
from iris.env_resources import TaskResources
from rigging.filesystem import url_to_fs

logger = logging.getLogger(__name__)

# Maximum simultaneous chunk iterators per pass-1 batch.
EXTERNAL_SORT_FAN_IN = 500

# Items per pickle.dump in pass-1.  Larger batches compress better (zstd
# dictionary spans the whole batch) and reduce per-call overhead.
_WRITE_BATCH_SIZE = 10_000

# Fraction of container memory budgeted for pass-2 read buffers.
_READ_MEMORY_FRACTION = 0.25


def _safe_read_batch_size(n_runs: int, sample_run_path: str) -> int:
    """Compute a pass-2 read batch size that fits within the memory budget.

    Probes the first batch from ``sample_run_path`` to estimate in-memory
    bytes per item, then divides the memory budget by ``n_runs * item_bytes``
    so that all concurrent run-file buffers together stay within
    ``_READ_MEMORY_FRACTION`` of available container memory.
    """
    dctx = zstd.ZstdDecompressor()
    try:
        with fsspec.open(sample_run_path, "rb") as raw_f:
            with dctx.stream_reader(raw_f) as f:
                sample_batch: list = pickle.load(f)
    except Exception:
        return _WRITE_BATCH_SIZE

    sample = sample_batch[:100]
    if not sample:
        return _WRITE_BATCH_SIZE
    # pickle size x 3 approximates Python object overhead (dicts are ~3x larger
    # in memory than their serialised form).
    item_bytes = max(64, len(pickle.dumps(sample)) // len(sample) * 3)

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
    cctx = zstd.ZstdCompressor(level=3)
    run_paths: list[str] = []
    batch_idx = 0

    while True:
        batch = list(islice(chunk_iterators_gen, EXTERNAL_SORT_FAN_IN))
        if not batch:
            break
        run_path = f"{external_sort_dir}/run-{batch_idx:04d}.pkl.zst"
        item_count = 0
        pending: list = []
        with fsspec.open(run_path, "wb") as raw_f:
            with cctx.stream_writer(raw_f, closefd=False) as f:
                for item in heapq.merge(*batch, key=merge_key):
                    pending.append(item)
                    if len(pending) >= _WRITE_BATCH_SIZE:
                        pickle.dump(pending, f, protocol=pickle.HIGHEST_PROTOCOL)
                        item_count += len(pending)
                        pending = []
                if pending:
                    pickle.dump(pending, f, protocol=pickle.HIGHEST_PROTOCOL)
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
        with fsspec.open(path, "rb") as raw_f:
            with zstd.ZstdDecompressor().stream_reader(raw_f) as f:
                while True:
                    try:
                        items: list = pickle.load(f)
                        # Yield in read_batch_size chunks and delete consumed
                        # items in-place so memory is released progressively
                        # even while the generator is suspended in heapq.merge.
                        while items:
                            chunk = items[:read_batch_size]
                            del items[:read_batch_size]
                            yield from chunk
                    except EOFError:
                        break

    run_iters = [_read_run(p) for p in run_paths]
    try:
        yield from heapq.merge(*run_iters, key=merge_key)
    finally:
        fs, _ = fsspec.core.url_to_fs(external_sort_dir)
        for path in run_paths:
            try:
                _, fs_path = url_to_fs(path)
                fs.rm(fs_path)
            except Exception:
                pass
