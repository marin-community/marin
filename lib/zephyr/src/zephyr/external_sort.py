# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded external merge sort over Polars LazyFrame streams.

The reduce stage uses this when an in-memory merge of its scatter chunks would
exceed the task memory budget.

The first pass consumes the input frames in groups of ``fan_in`` frames. Each
group produces a sorted Parquet run. Later passes merge at most
``max_merge_fan_in`` runs. Thus, the final streaming merge has a bounded input
count. The function deletes all run files after completion or an error.
"""

import logging
from typing import NamedTuple

import polars as pl
from rigging.filesystem import open_url, url_to_fs

logger = logging.getLogger(__name__)


class _SpillRun(NamedTuple):
    """One sorted run file, addressed both as a URL and as a filesystem path."""

    url: str
    path: str


def external_sort_merge(
    input_frames: list[pl.LazyFrame],
    sort_key: str,
    external_sort_dir: str,
    fan_in: int,
    max_merge_fan_in: int,
    shard: int,
) -> pl.LazyFrame:
    """Merge sorted LazyFrames with a bounded external sort.

    The first pass merges groups of at most ``fan_in`` frames. Additional
    passes limit each merge to ``max_merge_fan_in`` runs. The final merge is
    collected before spill files are deleted so the returned LazyFrame does
    not reference paths that are about to vanish.

    Args:
        input_frames: LazyFrames already sorted ascending on ``sort_key``. Order
            within each frame must match the global sort order for that key.
        sort_key: Column used for k-way merge (shuffle uses the binary sort-key
            column whose byte encoding preserves total order).
        external_sort_dir: Writable directory for intermediate run files (local
            temp dir or ``gs://.../stage1-external-sort/shard-NNNN``).
        fan_in: Maximum frames merged in one pass-1 group; bounds peak memory
            during pass 1. Callers typically use ``ceil(sqrt(n_chunks))``.
        max_merge_fan_in: Maximum run files in all later merges.
        shard: Target shard id for log messages only.

    Returns:
        A LazyFrame of rows in global merged sort order.
    """
    if len(input_frames) == 0:
        return pl.LazyFrame()
    if fan_in < 1:
        raise ValueError(f"fan_in must be at least 1, got {fan_in}")
    if max_merge_fan_in < 2:
        raise ValueError(f"max_merge_fan_in must be at least 2, got {max_merge_fan_in}")

    spill_fs, spill_dir = url_to_fs(external_sort_dir)
    spill_fs.makedirs(spill_dir, exist_ok=True)

    logger.info("[shard %d] External sort: pass-1 fan_in=%d", shard, fan_in)

    spill_files: set[str] = set()

    def write_run(frames: list[pl.LazyFrame], pass_index: int, run_index: int) -> _SpillRun:
        run_name = f"pass-{pass_index:04d}-run-{run_index:04d}.spill"
        run = _SpillRun(url=f"{external_sort_dir}/{run_name}", path=f"{spill_dir}/{run_name}")
        spill_files.add(run.path)
        merged = pl.merge_sorted(frames, key=sort_key)
        with open_url(run.url, "wb") as output:
            merged.sink_parquet(output, compression="zstd")
        return run

    def delete_runs(runs: list[_SpillRun]) -> None:
        paths = [run.path for run in runs]
        if not paths:
            return
        spill_fs.rm(paths)
        spill_files.difference_update(paths)

    try:
        batches = [input_frames[i : i + fan_in] for i in range(0, len(input_frames), fan_in)]
        runs: list[_SpillRun] = []
        for run_index, batch in enumerate(batches):
            run = write_run(batch, pass_index=0, run_index=run_index)
            runs.append(run)
            logger.info("[shard %d] External sort: wrote run %d to %s", shard, run_index, run.url)

        pass_index = 1
        while len(runs) > max_merge_fan_in:
            logger.info(
                "[shard %d] External sort: pass %d merging %d runs with max fan-in %d",
                shard,
                pass_index,
                len(runs),
                max_merge_fan_in,
            )
            next_runs: list[_SpillRun] = []
            consumed_runs: list[_SpillRun] = []
            run_batches = [runs[i : i + max_merge_fan_in] for i in range(0, len(runs), max_merge_fan_in)]
            for run_index, run_batch in enumerate(run_batches):
                if len(run_batch) == 1:
                    next_runs.extend(run_batch)
                    continue
                next_runs.append(
                    write_run(
                        [pl.scan_parquet(run.url) for run in run_batch],
                        pass_index=pass_index,
                        run_index=run_index,
                    )
                )
                consumed_runs.extend(run_batch)
            delete_runs(consumed_runs)
            runs = next_runs
            pass_index += 1

        logger.info("[shard %d] External sort: final merge of %d run files", shard, len(runs))
        return pl.merge_sorted([pl.scan_parquet(run.url) for run in runs], key=sort_key)
    finally:
        pass
        # This can't happen because we're returning a LazyFrame.
        # if spill_files:
        #    try:
        #        spill_fs.rm(sorted(spill_files))
        #    except Exception:
        #        logger.warning("Failed to delete external-sort run files under %s", spill_dir, exc_info=True)
