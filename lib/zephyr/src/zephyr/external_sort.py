# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-pass external merge sort over Polars LazyFrame streams.

Used by the reduce stage when the number of scatter chunks exceeds
``EXTERNAL_SORT_FAN_IN`` or the estimated merge memory exceeds the worker
budget.

Pass 1: consume the input LazyFrame stream in groups of ``fan_in`` frames.
Merge each group with ``in_memory_k_way_merge`` and spill the sorted result
to a runfile via :class:`SpillWriter`.

Pass 2: read all run files via :class:`SpillReader`, merge with
``polars.merge_sorted``, and yield items. Run files are deleted after the
merge completes.
"""

import heapq
import logging
from collections.abc import Iterator
from itertools import islice

import cloudpickle
import polars as pl
from rigging.filesystem import url_to_fs

logger = logging.getLogger(__name__)

# Sort-key column written by shuffle write path; stripped when reconstructing items.
_SORT_KEY_COL = "__zephyr_sort_key__"
# Written when item values are not Polars-representable; deserialized via cloudpickle.
_PAYLOAD_COL = "_payload"

# Hard cap on pass-1 batch group size. Limits the number of LazyFrames held
# in memory simultaneously during each merge group.
EXTERNAL_SORT_FAN_IN = 500

_FAN_IN_MEMORY_FRACTION = 0.5
_FAN_IN_FLOOR = 4


def _dataframe_to_items(df: pl.DataFrame) -> Iterator:
    """Yield Python items from a DataFrame, stripping internal columns and deserializing payloads.

    Strips ``_SORT_KEY_COL`` and deserializes the cloudpickle payload column
    when present. Handles mixed schemas where some rows are Arrow-typed and
    some are cloudpickled.
    """
    if _SORT_KEY_COL in df.columns:
        df = df.drop(_SORT_KEY_COL)

    if _PAYLOAD_COL not in df.columns:
        yield from df.iter_rows(named=True)
        return

    payload_series = df[_PAYLOAD_COL]
    if payload_series.null_count() > 0:
        # Mixed batch: some rows are Arrow-typed, some are cloudpickled.
        # payload_series has None for Arrow-typed rows.
        data_df = df.drop(_PAYLOAD_COL)
        for payload, row in zip(payload_series.to_list(), data_df.iter_rows(named=True), strict=True):
            if payload is not None:
                yield cloudpickle.loads(payload)
            else:
                yield row
    else:
        # Pure payload batch: all rows are cloudpickled.
        for p in payload_series.to_list():
            yield cloudpickle.loads(p)


def compute_fan_in(per_iterator_bytes: int, memory_limit: int) -> int:
    """Pick a pass-1 fan-in that fits within the memory budget.

    ``per_iterator_bytes`` is the caller's estimate of memory held per open
    chunk (typically the compressed chunk byte length). Returns at least
    ``_FAN_IN_FLOOR`` and at most ``EXTERNAL_SORT_FAN_IN``.
    """
    if per_iterator_bytes <= 0 or memory_limit <= 0:
        return EXTERNAL_SORT_FAN_IN
    budget = int(memory_limit * _FAN_IN_MEMORY_FRACTION)
    fan_in = budget // max(1, per_iterator_bytes)
    fan_in = max(_FAN_IN_FLOOR, fan_in)
    return min(fan_in, EXTERNAL_SORT_FAN_IN)


# TODO: Can compute the schemas when writing the scatter shards, so we don't need to collect them
def polars_unify_schemas(frames: list[pl.LazyFrame]) -> pl.Schema:

    schemas = [frame.collect_schema() for frame in frames]
    empty_frames = [schema.to_frame() for schema in schemas]
    unified_schema = pl.concat(empty_frames, how="vertical_relaxed").schema

    requires_cast = any(schema != unified_schema for schema in schemas)
    return requires_cast, unified_schema


def external_sort_merge(
    batches: Iterator[pl.LazyFrame],
    sort_key: str,
    external_sort_dir: str,
    fan_in: int = EXTERNAL_SORT_FAN_IN,
) -> Iterator[dict]:
    """Merge a sorted LazyFrame stream via a two-pass external sort.

    Args:
        batches: Flat stream of pl.LazyFrame objects. Each frame must be
            sorted by ``sort_key`` (ascending, binary lexicographic order).
        sort_key: Column name to sort by. Must be a binary column whose
            byte encoding preserves the desired sort order.
        external_sort_dir: Directory path for spill files, e.g.
            ``gs://bucket/.../stage1-external-sort/shard-0042``.
        fan_in: Maximum number of frames to merge per pass-1 group.
            Callers should pass a value from :func:`compute_fan_in` to bound
            peak memory during pass-1.

    Yields:
        Python dicts in merged sort order.
    """
    run_paths: list[str] = []
    batch_idx = 0

    spill_fs, spill_dir = url_to_fs(external_sort_dir)
    spill_fs.makedirs(spill_dir, exist_ok=True)

    logger.info("External sort: pass-1 fan_in=%d", fan_in)

    while True:
        frame_group = list(islice(batches, fan_in))
        if not frame_group:
            break
        run_path = f"{external_sort_dir}/run-{batch_idx:04d}.spill"

        requires_cast, unified_schema = polars_unify_schemas(frame_group)

        if requires_cast:
            logger.info(
                "External sort requires cast to unified schema, original schemas: %s",
                [frame.collect_schema() for frame in frame_group],
            )
            frame_group = [frame.cast(unified_schema) for frame in frame_group]

        logger.info("Merging frames with schema %s", unified_schema)
        merged = pl.merge_sorted(frame_group, key=sort_key)
        merged.sink_parquet(run_path, compression="zstd")

        run_paths.append(run_path)
        logger.info("External sort: wrote run %d to %s", batch_idx + 1, run_path)
        batch_idx += 1

    if not run_paths:
        return

    try:
        fs_run_paths = [f"{spill_dir}/run-{i:04d}.spill" for i in range(len(run_paths))]
        frames = [pl.scan_parquet(p) for p in run_paths]
        merged = pl.merge_sorted(frames, key=sort_key)
        for batch in merged.collect_batches():
            yield from _dataframe_to_items(batch)
    finally:
        if fs_run_paths:
            try:
                spill_fs.rm(fs_run_paths)
            except Exception:
                # Spill files live under a per-shard temp dir that the worker
                # eventually wipes; log so leaked files are at least traceable.
                logger.warning("Failed to delete external-sort run files under %s", spill_dir, exc_info=True)
