# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-pass external merge sort for large k-way merges.

Used by the reduce stage when the number of sorted chunk iterators exceeds
``EXTERNAL_SORT_FAN_IN``, to avoid opening O(k) scanners simultaneously and
exhausting worker memory.

Pass 1: batch the k iterators into groups of EXTERNAL_SORT_FAN_IN, merge each
group with heapq.merge, and spill the sorted result to a GCS pickle run file
under ``{external_sort_dir}/run-{i:04d}.pkl``.

Pass 2: heapq.merge over the (much smaller) set of run file iterators.

Run files are deleted after the final merge completes.
"""

import heapq
import logging
import pickle
from collections.abc import Callable, Iterator
from itertools import islice

import fsspec
from iris.marin_fs import url_to_fs

logger = logging.getLogger(__name__)

# Maximum simultaneous chunk iterators before spilling to GCS run files.
# At 500 x ~5 MB/row-group (100 K rows x 50 B) = 2.5 GB peak during pass 1.
EXTERNAL_SORT_FAN_IN = 500


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
    while True:
        batch = list(islice(chunk_iterators_gen, EXTERNAL_SORT_FAN_IN))
        if not batch:
            break
        merged = list(heapq.merge(*batch, key=merge_key))
        run_path = f"{external_sort_dir}/run-{batch_idx:04d}.pkl"
        with fsspec.open(run_path, "wb") as f:
            pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
        run_paths.append(run_path)
        logger.info(
            "External sort: wrote run %d (%d items) to %s",
            batch_idx + 1,
            len(merged),
            run_path,
        )
        del merged
        batch_idx += 1

    def _read_run(path: str) -> Iterator:
        with fsspec.open(path, "rb") as f:
            yield from pickle.load(f)

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
