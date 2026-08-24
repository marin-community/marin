# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded external merge sort over DataFusion streams.

The reduce stage uses this when an in-memory merge of its scatter chunks would
exceed the task memory budget. Intermediate Parquet runs use Zephyr's stage
filesystem, including GCS for production stage paths.
"""

import logging
from collections.abc import Iterator
from typing import NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
from datafusion import DataFrame, SessionContext, col
from rigging.filesystem.factory import open_url, url_to_fs
from rigging.filesystem.storage_path import prefix_join

from zephyr.parquet_scan import register_object_stores, scan_parquet

logger = logging.getLogger(__name__)


class _SpillRun(NamedTuple):
    """One sorted run file, addressed both as a URL and as a filesystem path."""

    url: str
    path: str


def merged_frame(input_frames: list[DataFrame], sort_key: str) -> DataFrame:
    """Return a streaming k-way merge over sorted DataFusion inputs.

    Each input declares its ordering through ``scan_parquet``. DataFusion
    therefore plans the sort over their union as ``SortPreservingMergeExec``
    rather than materializing and sorting the combined rows.
    """
    if not input_frames:
        raise ValueError("input_frames must not be empty")
    merged = input_frames[0]
    for frame in input_frames[1:]:
        merged = merged.union(frame)
    return merged.sort(col(sort_key).sort())


def external_sort_merge(
    context: SessionContext,
    input_frames: list[DataFrame],
    sort_key: str,
    external_sort_dir: str,
    fan_in: int,
    max_merge_fan_in: int,
    shard: int,
) -> Iterator[pa.RecordBatch]:
    """Merge sorted DataFusion frames through bounded Parquet runs.

    The first pass merges groups of at most ``fan_in`` frames. Additional
    passes limit each merge to ``max_merge_fan_in`` runs. The final merge
    streams Arrow batches. All intermediate files are removed after completion
    or an error.
    """
    if not input_frames:
        return
    if fan_in < 1:
        raise ValueError(f"fan_in must be at least 1, got {fan_in}")
    if max_merge_fan_in < 2:
        raise ValueError(f"max_merge_fan_in must be at least 2, got {max_merge_fan_in}")

    spill_fs, spill_dir = url_to_fs(external_sort_dir)
    spill_fs.makedirs(spill_dir, exist_ok=True)
    register_object_stores(context, [external_sort_dir])

    logger.info("[shard %d] External sort: pass-1 fan_in=%d", shard, fan_in)
    spill_files: set[str] = set()

    def write_run(frames: list[DataFrame], pass_index: int, run_index: int) -> _SpillRun:
        run_name = f"pass-{pass_index:04d}-run-{run_index:04d}.parquet"
        run = _SpillRun(
            url=prefix_join(external_sort_dir, run_name),
            path=prefix_join(spill_dir, run_name),
        )
        spill_files.add(run.path)
        merged = merged_frame(frames, sort_key)
        with open_url(run.url, "wb") as output:
            with pq.ParquetWriter(output, merged.schema(), compression="zstd") as writer:
                for batch in merged.execute_stream():
                    writer.write_batch(batch.to_pyarrow())
        return run

    def run_frame(run: _SpillRun, schema: pa.Schema) -> DataFrame:
        return scan_parquet(context, run.url, schema=schema, sorted_by=(sort_key,))

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

        schema = input_frames[0].schema()
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
                        [run_frame(run, schema) for run in run_batch],
                        pass_index=pass_index,
                        run_index=run_index,
                    )
                )
                consumed_runs.extend(run_batch)
            delete_runs(consumed_runs)
            runs = next_runs
            pass_index += 1

        logger.info("[shard %d] External sort: final merge of %d run files", shard, len(runs))
        final = merged_frame([run_frame(run, schema) for run in runs], sort_key)
        for batch in final.execute_stream():
            yield batch.to_pyarrow()
    finally:
        if spill_files:
            try:
                spill_fs.rm(sorted(spill_files))
            except Exception:
                logger.warning("Failed to delete external-sort run files under %s", spill_dir, exc_info=True)
