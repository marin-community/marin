# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-pass external merge sort over Polars LazyFrame streams.

Used by the reduce stage when the number of scatter chunks exceeds
``EXTERNAL_SORT_FAN_IN`` or the estimated merge memory exceeds the worker
budget.

Pass 1: consume the input LazyFrame stream in groups of ``fan_in`` frames.
Merge each group with :func:`polars.merge_sorted` and spill the sorted result
to a zstd-compressed parquet run file.

Pass 2: read all run files via :func:`polars.scan_parquet`, merge with
:func:`polars.merge_sorted`, and yield DataFrame batches. Run files are
deleted after the merge completes (or on error).
"""

import io
import logging
from collections.abc import Iterator

import polars as pl
from rigging.filesystem import open_url, url_to_fs

from zephyr.polars_io import scan_parquet_chunk

logger = logging.getLogger(__name__)


def external_sort_merge(
    input_frames: list[pl.LazyFrame],
    sort_key: str,
    external_sort_dir: str,
    fan_in: int,
    shard: int,
) -> Iterator[pl.DataFrame]:
    """Merge pre-sorted LazyFrames via a two-pass external sort with spill files.

    Pass 1 groups ``input_frames`` into batches of at most ``fan_in`` frames,
    merges each batch with :func:`polars.merge_sorted`, and writes the result to
    a zstd-compressed parquet run file under ``external_sort_dir``. Pass 2 scans
    all run files, merges them again with :func:`polars.merge_sorted`, and
    streams the result as DataFrame batches. Run files are deleted when pass 2
    finishes (or on error).

    Args:
        input_frames: LazyFrames already sorted ascending on ``sort_key``. Order
            within each frame must match the global sort order for that key.
        sort_key: Column used for k-way merge (shuffle uses the binary sort-key
            column whose byte encoding preserves total order).
        external_sort_dir: Writable directory for intermediate run files (local
            temp dir or ``gs://.../stage1-external-sort/shard-NNNN``).
        fan_in: Maximum frames merged in one pass-1 group; bounds peak memory
            during pass 1. Callers typically use ``ceil(sqrt(n_chunks))``.
        shard: Target shard id for log messages only.

    Yields:
        :class:`polars.DataFrame` batches in global merged sort order.
    """
    if len(input_frames) == 0:
        return

    spill_fs, spill_dir = url_to_fs(external_sort_dir)
    spill_fs.makedirs(spill_dir, exist_ok=True)

    logger.info("[shard %d] External sort: pass-1 fan_in=%d", shard, fan_in)

    spill_files: list[str] = []

    try:
        # TODO: When we upgrade to Python 3.12, use itertools.batched
        batches = [input_frames[i : i + fan_in] for i in range(0, len(input_frames), fan_in)]
        for idx, batch in enumerate(batches):
            merged = pl.merge_sorted(batch, key=sort_key).collect()

            spill_file = f"{external_sort_dir}/run-{idx:04d}.spill"
            buf = io.BytesIO()
            merged.write_parquet(buf, compression="zstd")
            with open_url(spill_file, "wb") as f:
                f.write(buf.getvalue())

            spill_files.append(spill_file)
            logger.info("[shard %d] External sort: wrote run %d to %s", shard, idx, spill_file)

        logger.info("[shard %d] External sort: pass-2 merging %d run files", shard, len(spill_files))

        merged = pl.merge_sorted([scan_parquet_chunk(p) for p in spill_files], key=sort_key)
        yield from merged.collect_batches()
    finally:
        if spill_files:
            try:
                spill_fs.rm([f"{spill_dir}/run-{i:04d}.spill" for i in range(len(spill_files))])
            except Exception:
                # Spill files live under a per-shard temp dir that the worker
                # eventually wipes; log so leaked files are at least traceable.
                logger.warning("Failed to delete external-sort run files under %s", spill_dir, exc_info=True)
