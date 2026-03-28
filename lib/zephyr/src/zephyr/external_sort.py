# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-pass external merge sort using Parquet spill files.

Used by the reduce stage when the number of sorted chunk iterators exceeds
``EXTERNAL_SORT_FAN_IN``, to avoid opening O(k) scanners simultaneously and
exhausting worker memory.

Pass 1: batch Arrow table iterators into groups of EXTERNAL_SORT_FAN_IN,
concatenate + sort in Arrow, write as Parquet run files.

Pass 2: read run files back as Arrow tables, concat + sort, yield batches.
Run files are deleted after the merge completes.
"""

import logging
from collections.abc import Iterator
from itertools import islice

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import zstandard as zstd
from iris.env_resources import TaskResources
from rigging.filesystem import url_to_fs

logger = logging.getLogger(__name__)

# Maximum simultaneous chunk iterators per pass-1 batch.
EXTERNAL_SORT_FAN_IN = 500


def external_sort_merge(
    chunk_tables_gen: Iterator[pa.Table],
    sort_keys: list[tuple[str, str]],
    external_sort_dir: str,
) -> Iterator[pa.Table]:
    """Two-pass external sort yielding sorted Arrow tables.

    Pass 1: batch tables into groups of EXTERNAL_SORT_FAN_IN,
    concat + sort in Arrow, write as Parquet run files.
    Pass 2: read run files back as Arrow tables, concat + sort, yield batches.
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

    try:
        if len(run_paths) == 1:
            table = pq.read_table(run_paths[0])
            for batch in table.to_batches(max_chunksize=100_000):
                yield pa.Table.from_batches([batch], schema=table.schema)
        else:
            all_tables = [pq.read_table(p) for p in run_paths]
            combined = pa.concat_tables(all_tables, promote_options="default")
            indices = pc.sort_indices(combined, sort_keys=sort_keys)
            sorted_table = combined.take(indices)
            for batch in sorted_table.to_batches(max_chunksize=100_000):
                yield pa.Table.from_batches([batch], schema=sorted_table.schema)
    finally:
        fs, _ = fsspec.core.url_to_fs(external_sort_dir)
        for path in run_paths:
            try:
                _, fs_path = url_to_fs(path)
                fs.rm(fs_path)
            except Exception:
                pass
