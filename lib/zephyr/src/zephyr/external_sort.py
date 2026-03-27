# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-pass external merge sort using Parquet spill files.

Used by the reduce stage when the number of sorted chunk iterators exceeds
``EXTERNAL_SORT_FAN_IN``, to avoid opening O(k) scanners simultaneously and
exhausting worker memory.

Pass 1: batch Arrow table iterators into groups of EXTERNAL_SORT_FAN_IN,
concatenate + sort in Arrow, write as Parquet run files.

Pass 2: read run files, yield items via heapq.merge over Python iterators.
Run files are deleted after the final merge completes.
"""

import heapq
import logging
from collections.abc import Callable, Iterator
from itertools import islice
from typing import Any

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from iris.marin_fs import url_to_fs

logger = logging.getLogger(__name__)

# Maximum simultaneous chunk iterators per pass-1 batch.
EXTERNAL_SORT_FAN_IN = 500


def _read_parquet_run(path: str) -> Iterator:
    """Yield rows from a Parquet run file as Python dicts, one batch at a time."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches():
        yield from batch.to_pylist()


def external_sort_merge(
    chunk_tables_gen: Iterator[pa.Table],
    sort_keys: list[tuple[str, str]],
    merge_key: Callable[[Any], Any],
    external_sort_dir: str,
) -> Iterator:
    """Two-pass external sort using Parquet spill files.

    Pass 1: batch Arrow table iterators into groups of EXTERNAL_SORT_FAN_IN,
    concatenate + sort in Arrow, write as Parquet run files.
    Pass 2: read run files, yield items via heapq.merge over Python iterators.
    """
    from zephyr.writers import ensure_parent_dir

    run_paths: list[str] = []
    batch_idx = 0

    while True:
        batch_tables = list(islice(chunk_tables_gen, EXTERNAL_SORT_FAN_IN))
        if not batch_tables:
            break
        combined = pa.concat_tables(batch_tables, promote_options="default")
        indices = pc.sort_indices(combined, sort_keys=sort_keys)
        sorted_table = combined.take(indices)

        run_path = f"{external_sort_dir}/run-{batch_idx:04d}.parquet"
        ensure_parent_dir(run_path)
        pq.write_table(sorted_table, run_path)
        run_paths.append(run_path)
        logger.info(
            "External sort: wrote run %d (%d rows) to %s",
            batch_idx + 1,
            len(sorted_table),
            run_path,
        )
        batch_idx += 1
        del combined, sorted_table

    if not run_paths:
        return

    run_iters = [_read_parquet_run(p) for p in run_paths]
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
