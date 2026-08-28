# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Iterator, Sequence
from typing import Any, TypedDict

import dupekit
import pyarrow as pa
import pyarrow.compute as pc
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from zephyr import counters
from zephyr.batches import ArrowBatch, iter_record_batches
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.expr import col
from zephyr.sql import SqlScalarFunction, sql
from zephyr.writers import write_parquet_file

logger = logging.getLogger(__name__)


def _find_last_complete_iteration(
    output_dir: str, max_iterations: int, expected_parquets: int
) -> tuple[int, list[str]] | None:
    """Return (last_iteration, parquet_paths) from prior run outputs, or None if nothing reusable.

    A CC iteration ``it_N/`` is considered complete iff its parquet file count equals
    ``expected_parquets`` (= ``ctx.max_workers`` at write time). Iteration 0 uses the
    ``part-{shard:05d}.parquet`` naming; iterations 1+ use ``part-{shard:05d}-of-{total:05d}.parquet``.
    Both are detected by globbing ``it_N/*.parquet``.
    """
    last_complete = -1
    last_paths: list[str] = []
    for i in range(max_iterations + 1):
        paths = [str(m) for m in StoragePath(f"{output_dir}/it_{i}/*.parquet").glob()]
        if len(paths) != expected_parquets:
            break
        last_complete = i
        last_paths = paths
    if last_complete < 0:
        return None
    return last_complete, last_paths


class CCInput(TypedDict):
    bucket: str
    id: Any
    file_idx: int


def _hash_string_ids(values: pa.Array) -> pa.Array:
    return pa.array(
        [None if value is None else str(dupekit.hash_xxh3_128(value.encode())) for value in values.to_pylist()],
        type=pa.string(),
    )


def _cc_input_batches(
    items: Iterator[CCInput | ArrowBatch],
    _shard: ShardInfo,
) -> Iterator[pa.RecordBatch]:
    """Return connected-component inputs as validated Arrow batches."""

    def validate(batch: pa.RecordBatch) -> pa.RecordBatch:
        id_values = batch.column("id")
        if not (pa.types.is_integer(id_values.type) or pa.types.is_string(id_values.type)):
            raise ValueError(f"Unsupported id type: {id_values.type}")
        if id_values.null_count:
            raise ValueError("Connected-components ids must not be null")
        return batch

    rows: list[CCInput] = []
    for item in items:
        if isinstance(item, ArrowBatch):
            if rows:
                yield validate(pa.RecordBatch.from_pylist(rows))
                rows = []
            yield from (validate(batch) for batch in iter_record_batches([item]))
            continue
        rows.append(item)
        if len(rows) == 8192:
            yield validate(pa.RecordBatch.from_pylist(rows))
            rows = []
    if rows:
        yield validate(pa.RecordBatch.from_pylist(rows))


_PREPARE_BUCKETS_SQL = """
    WITH normalized AS (
        SELECT bucket, id, file_idx,
               CASE
                   WHEN arrow_typeof(id) LIKE 'Int%' OR arrow_typeof(id) LIKE 'UInt%'
                   THEN CAST(id AS VARCHAR)
                   ELSE cc_hash_string_id(CAST(id AS VARCHAR))
               END AS id_norm
        FROM input
    ), ranked AS (
        SELECT *, row_number() OVER (PARTITION BY bucket, id_norm ORDER BY id_norm) AS occurrence
        FROM normalized
    )
    SELECT bucket, id, file_idx, id_norm
    FROM ranked
    WHERE occurrence = 1
"""


def _bucket_links_sql(preserve_singletons: bool) -> str:
    singleton_filter = "bucket_size = 1" if preserve_singletons else "false"
    return f"""
        WITH unique_nodes AS (
            SELECT *, row_number() OVER (PARTITION BY bucket, id_norm ORDER BY id_norm) AS occurrence
            FROM input
        ), nodes AS (
            SELECT bucket, id, file_idx, id_norm
            FROM unique_nodes
            WHERE occurrence = 1
        ), tagged AS (
            SELECT *,
                   first_value(id) OVER (PARTITION BY bucket ORDER BY id_norm) AS hub_record_id,
                   first_value(id_norm) OVER (PARTITION BY bucket ORDER BY id_norm) AS hub_id_norm,
                   first_value(file_idx) OVER (PARTITION BY bucket ORDER BY id_norm) AS hub_file_idx,
                   count(*) OVER (PARTITION BY bucket) AS bucket_size
            FROM nodes
        )
        SELECT hub_record_id AS source_record_id,
               hub_id_norm AS source_id_norm,
               hub_file_idx AS source_file_idx,
               id_norm AS dest_id_norm
        FROM tagged
        WHERE id_norm != hub_id_norm
        UNION ALL
        SELECT id AS source_record_id,
               id_norm AS source_id_norm,
               file_idx AS source_file_idx,
               hub_id_norm AS dest_id_norm
        FROM tagged
        WHERE id_norm != hub_id_norm
        UNION ALL
        SELECT id AS source_record_id,
               id_norm AS source_id_norm,
               file_idx AS source_file_idx,
               id_norm AS dest_id_norm
        FROM tagged
        WHERE {singleton_filter}
    """


_BUILD_ADJACENCY_SQL = """
    SELECT min(source_record_id) AS record_id,
           source_id_norm AS id_norm,
           array_agg(DISTINCT dest_id_norm ORDER BY dest_id_norm) AS adjacency_list,
           source_id_norm AS component_id,
           true AS changed,
           min(source_file_idx) AS file_idx
    FROM input
    GROUP BY source_id_norm
"""


_EMIT_MESSAGES_SQL = """
    SELECT id_norm AS key,
           true AS is_self,
           record_id,
           id_norm,
           adjacency_list,
           component_id,
           changed,
           file_idx
    FROM input
    UNION ALL
    SELECT unnest(adjacency_list) AS key,
           false AS is_self,
           record_id,
           '' AS id_norm,
           make_array('') AS adjacency_list,
           component_id,
           false AS changed,
           CAST(0 AS BIGINT) AS file_idx
    FROM input
"""


_REDUCE_NODE_SQL = """
    WITH propagated AS (
        SELECT key, least(key, min(component_id)) AS minimum_component
        FROM input
        GROUP BY key
    ), self_messages AS (
        SELECT *, row_number() OVER (PARTITION BY key ORDER BY key) AS occurrence
        FROM input
        WHERE is_self
    )
    SELECT self_messages.record_id,
           self_messages.id_norm,
           self_messages.adjacency_list,
           propagated.minimum_component AS component_id,
           propagated.minimum_component < self_messages.component_id AS changed,
           self_messages.file_idx
    FROM self_messages
    JOIN propagated USING (key)
    WHERE self_messages.occurrence = 1
"""


def connected_components(
    ds: Dataset[CCInput | ArrowBatch],
    ctx: ZephyrContext,
    *,
    output_dir: str,
    max_iterations: int = 10,
    preserve_singletons: bool = True,
    resume: bool = False,
    num_reduce_shards: int | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
) -> tuple[bool, Sequence[str]]:
    """
    Connected Components implementation using Zephyr Dataset API and Hash-to-Min algorithm (https://arxiv.org/abs/1203.5387)

    Args:
        ds: Input dataset of row records or Arrow batches containing ``bucket``,
            ``id``, and ``file_idx`` fields, typically from MinHash LSH output.
        ctx: ZephyrContext to use for execution.
        output_dir: Directory to write intermediate and final output files
        max_iterations: Maximum number of iterations to run the connected components algorithm
        preserve_singletons: Whether to preserve single-node buckets in the output
        resume: If True, skip complete prior iterations and start at the first
            incomplete iteration. If no complete state exists, run from scratch.
        num_reduce_shards: Shuffle shard count. Defaults to the context worker cap.
    """

    # Determine reduce shard count. Default to ctx max_workers to avoid
    # I/O amplification.
    num_reduce_shards = num_reduce_shards or ctx.max_workers

    start_iteration = 1
    curr_it: Sequence[str]
    resumed = _find_last_complete_iteration(output_dir, max_iterations, num_reduce_shards) if resume else None
    if resumed is not None:
        last_it, last_paths = resumed
        curr_it = last_paths
        start_iteration = last_it + 1
        if start_iteration > max_iterations and last_it >= 1:
            # The prior run stopped exactly at the iteration cap, so the loop
            # below would not execute and convergence could not be observed --
            # a resumed-at-cap run would always report converged=False even if
            # it had actually converged. Replay the final iteration (a pure,
            # idempotent function of it_{last_it-1}) so its num_changes is read
            # and convergence reported accurately (see marin#6798).
            curr_it = sorted(str(m) for m in StoragePath(f"{output_dir}/it_{last_it - 1}/*.parquet").glob())
            start_iteration = last_it
        logger.info("CC resume: through it_%d, starting at it_%d", last_it, start_iteration)
    else:
        prepared = ds.map_shard(_cc_input_batches).sql(
            _PREPARE_BUCKETS_SQL,
            scalar_functions=(
                SqlScalarFunction(
                    "cc_hash_string_id",
                    _hash_string_ids,
                    (pa.string(),),
                    pa.string(),
                ),
            ),
        )
        curr_it = ctx.execute(
            prepared
            # Group nodes in buckets, deduplicate, and emit pairwise links
            .group_by(
                col("bucket"),
                reducer=sql(_bucket_links_sql(preserve_singletons)),
                # Sort each bucket's nodes by id_norm so the reducer always anchors
                # the star on the true minimum, independent of shuffle/arrival order.
                # Without this the star-vs-chain topology depends on how many reduce
                # shards (= executors) the run used, making a capped (unconverged) run
                # produce different component labels on different machine counts
                # (marin#6798). The converged result is unchanged; this only pins the
                # intermediate topology (and speeds convergence).
                sort_by=col("id_norm"),
                num_output_shards=num_reduce_shards,
            )
            # Construct Node state, init with:
            #  * each node is its own component
            #  * adjacency list from links
            .group_by(
                col("source_id_norm"),
                reducer=sql(_BUILD_ADJACENCY_SQL),
                num_output_shards=num_reduce_shards,
            ).write_parquet(f"{output_dir}/it_0/part-{{shard:05d}}.parquet"),
            verbose=True,
            map_task_resources=map_task_resources,
            reduce_task_resources=reduce_task_resources,
        ).results

    def _get_write_shard_and_count_fn(iteration: int):
        # NOTE: this function exists to make the iteration number closure capture explicit
        def _write_shard_and_count(nodes: Iterator[ArrowBatch], shard_info: ShardInfo) -> Iterator[dict]:
            num_changes = 0

            def counting_batches() -> Iterator[pa.RecordBatch]:
                nonlocal num_changes
                for batch in iter_record_batches(nodes):
                    changes = int(pc.sum(batch.column("changed")).as_py() or 0)
                    num_changes += changes
                    counters.pipeline.update_counter("cc/iteration_nodes", len(batch))
                    counters.pipeline.update_counter("cc/changes", changes)
                    yield batch

            path = (
                f"{output_dir}/it_{iteration}/part-{shard_info.shard_idx:05d}-of-{shard_info.total_shards:05d}.parquet"
            )
            result = write_parquet_file(counting_batches(), path)
            yield {**result, "num_changes": num_changes}

        return _write_shard_and_count

    converged = False
    for i in range(start_iteration, max_iterations + 1):  # type: ignore[bad-assignment]
        logger.info(f"Connected components iteration {i}...")

        shard_results = ctx.execute(
            Dataset.from_list(curr_it)
            .load_parquet(batch_mode=True)
            .sql(_EMIT_MESSAGES_SQL)
            .group_by(
                key=col("key"),
                reducer=sql(_REDUCE_NODE_SQL),
                num_output_shards=num_reduce_shards,
            )
            .map_shard(_get_write_shard_and_count_fn(i)),
            verbose=True,
            map_task_resources=map_task_resources,
            reduce_task_resources=reduce_task_resources,
        ).results

        curr_it = [r["path"] for r in shard_results]
        num_changes = sum(r["num_changes"] for r in shard_results)

        if num_changes == 0:
            converged = True
            logger.info(f"Connected components converged after {i} iterations.")
            break
        else:
            logger.info(f"Connected components iteration {i} found {num_changes:,} changes.")

    return converged, curr_it
