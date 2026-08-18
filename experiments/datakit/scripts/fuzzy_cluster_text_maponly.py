# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize fuzzy-duplicate candidate text grouped by cluster, without a shuffle.

``fuzzy_cluster_text.py`` regroups 45 TB of candidate text with a zephyr
``group_by``. Every member of a cluster lives in a different normalized shard
(distinct shards per cluster: p50=2, mean=3.5, shards/members=1.00), so the
regroup itself is unavoidable -- but the reduce stage is not. A reducer merges
every mapper's every chunk, and at 30,678 chunks per reducer that merge is a
multi-pass external sort spilling to ``/tmp``. It died twice.

This job does the same regroup as two map-only phases with durable, skippable
outputs:

* ``scatter`` joins a group of shards, sorts one buffered flush by
  ``(partition, cluster_key, id)``, and writes each partition's rows as a
  self-contained Parquet slice concatenated into one chunk object. A per-task
  index records ``(partition, offset, length)`` and is written last, so its
  presence commits the task.
* ``gather`` reads one partition's slices with one ranged GET per chunk, sorts
  the partition in memory, and writes the grouped text file that
  ``fuzzy_cluster_verify.py`` already consumes.

Between them the driver pivots the per-task indexes into one plan file per
partition. That pivot moves ~850 MB of integers and never touches text.

The trade is fan-in, not bytes: 45 TB still crosses the network once, but a
consumer merges 3,387 slices instead of 30,678 chunks, and its 5.6 GB partition
sorts in RAM with no spill. Resume granularity drops from the whole stage to one
13.5 GB map task or one 5.6 GB partition.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
        --no-wait --priority batch --cpu 16 --memory 96GB \
        -- python experiments/datakit/scripts/fuzzy_cluster_text_maponly.py \
            --prefix s3://.../marin --candidates datakit/dedup_709f5997 \
            --verified datakit/verify_fuzzy_dups_c757e4f0 \
            --large-clusters s3://.../user/rav/dedup/large_clusters/v1/large_clusters.parquet \
            --out s3://.../user/rav/dedup/cluster_text/v7 \
            --partitions 8192 --shards-per-task 48 --max-workers 96
"""

import argparse
import dataclasses
import json
import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.writers import write_parquet_file

from experiments.datakit.scripts.fuzzy_cluster_text import (
    _SHARED_OVERSIZED_KEY,
    _TEXT_SCHEMA,
    SPLIT_NGRAM_SIZE,
    SPLIT_SUBDIVISIONS,
    TextShard,
    _join_shard_group,
    build_shards,
    group_key_of,
    load_oversized,
)

logger = logging.getLogger(__name__)

COUNTER_PREFIX = "fuzzy/cluster_text_maponly"
PARTITION_COLUMN = "partition"

# Rows are staged in Python lists and sealed into an Arrow batch every time the
# staged text passes this size. Python strings cost roughly 1.5x their payload,
# so a small seal interval keeps that overhead off the flush buffer.
BATCH_TEXT_BYTES = 64 << 20
# Ranged GETs a gather task issues at once. The slices are ~0.5 MB compressed,
# so this is latency-bound and wants concurrency, not bandwidth.
SLICE_READ_CONCURRENCY = 32
# Rows materialized per ``take`` when writing a sorted partition out. The sorted
# copy is the gather task's only allocation on top of the partition itself.
TAKE_ROWS = 65_536

_SCATTER_SCHEMA = pa.schema([*_TEXT_SCHEMA, pa.field(PARTITION_COLUMN, pa.int32(), nullable=False)])

_SLICE_SCHEMA = pa.schema(
    [
        pa.field("task", pa.int32(), nullable=False),
        pa.field("chunk", pa.int32(), nullable=False),
        pa.field(PARTITION_COLUMN, pa.int32(), nullable=False),
        pa.field("offset", pa.int64(), nullable=False),
        pa.field("length", pa.int64(), nullable=False),
        pa.field("rows", pa.int64(), nullable=False),
    ]
)


def chunk_name(task: int, chunk: int) -> str:
    """Name of one flush's combined slice file."""
    return f"task-{task:05d}-c{chunk:03d}.slices"


def chunk_path(out: str, task: int, chunk: int) -> str:
    return prefix_join(prefix_join(out, "scatter/chunks"), chunk_name(task, chunk))


def index_path(out: str, task: int) -> str:
    return prefix_join(prefix_join(out, "scatter/index"), f"task-{task:05d}.parquet")


def plan_path(out: str, partition: int) -> str:
    return prefix_join(prefix_join(out, "scatter/plan"), f"part-{partition:06d}.parquet")


def text_path(out: str, partition: int) -> str:
    return prefix_join(prefix_join(out, "text"), f"part-{partition:06d}.parquet")


def _partition_bounds(values: np.ndarray, partitions: int) -> np.ndarray:
    """Start offset of every partition's run in a partition-sorted array.

    Returns ``partitions + 1`` offsets, so partition ``p`` owns
    ``[bounds[p], bounds[p + 1])``.
    """
    return np.searchsorted(values, np.arange(partitions + 1), side="left")


def _encode_slice(table: pa.Table, indices: pa.Array) -> bytes:
    """Serialize one partition's rows as a self-contained Parquet file."""
    rows = table.take(indices).drop_columns([PARTITION_COLUMN])
    sink = pa.BufferOutputStream()
    pq.write_table(rows, sink, compression="zstd", compression_level=3)
    return sink.getvalue().to_pybytes()


class SliceWriter:
    """Buffers joined rows and writes one combined slice file per flush.

    A mapper cannot hold one open writer per partition: at 8,192 partitions even
    a 2 MB buffer each is 16 GB, and one file per partition per flush would be
    tens of millions of objects. Instead the whole flush is sorted by
    ``(partition, cluster_key, id)`` once and written as a run of Parquet slices
    inside a single object, with the byte ranges recorded in the index.
    """

    def __init__(self, out: str, task: int, partitions: int, flush_bytes: int) -> None:
        self._out = out
        self._task = task
        self._partitions = partitions
        self._flush_bytes = flush_bytes
        self._staged: dict[str, list[Any]] = {field.name: [] for field in _SCATTER_SCHEMA}
        self._staged_bytes = 0
        self._batches: list[pa.RecordBatch] = []
        self._buffered_bytes = 0
        self._chunk = 0
        self._slices: list[dict[str, Any]] = []

    def add(self, record: dict[str, Any]) -> None:
        for field in _TEXT_SCHEMA:
            self._staged[field.name].append(record[field.name])
        self._staged[PARTITION_COLUMN].append(group_key_of(record["cluster_key"], self._partitions))
        self._staged_bytes += len(record["text"])
        if self._staged_bytes >= BATCH_TEXT_BYTES:
            self._seal()

    def close(self) -> list[dict[str, Any]]:
        """Flush what is left and return the slice index rows."""
        self._seal()
        self._flush()
        return self._slices

    def _seal(self) -> None:
        if not self._staged_bytes:
            return
        batch = pa.RecordBatch.from_pydict(self._staged, schema=_SCATTER_SCHEMA)
        self._staged = {field.name: [] for field in _SCATTER_SCHEMA}
        self._staged_bytes = 0
        self._batches.append(batch)
        self._buffered_bytes += batch.nbytes
        if self._buffered_bytes >= self._flush_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._batches:
            return
        table = pa.Table.from_batches(self._batches, schema=_SCATTER_SCHEMA)
        self._batches = []
        self._buffered_bytes = 0

        order = pc.sort_indices(
            table,
            sort_keys=[(PARTITION_COLUMN, "ascending"), ("cluster_key", "ascending"), ("id", "ascending")],
        )
        bounds = _partition_bounds(np.asarray(pc.take(table.column(PARTITION_COLUMN), order)), self._partitions)

        path = chunk_path(self._out, self._task, self._chunk)
        offset = 0
        with StoragePath(path).open("wb") as handle:
            for partition in range(self._partitions):
                start, stop = int(bounds[partition]), int(bounds[partition + 1])
                if start == stop:
                    continue
                payload = _encode_slice(table, order.slice(start, stop - start))
                handle.write(payload)
                self._slices.append(
                    {
                        "task": self._task,
                        "chunk": self._chunk,
                        PARTITION_COLUMN: partition,
                        "offset": offset,
                        "length": len(payload),
                        "rows": stop - start,
                    }
                )
                offset += len(payload)
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/chunk_bytes", offset)
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/chunks", 1)
        self._chunk += 1


def scatter_shard_group(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Join one group of shards and write its slices, unless the task is committed.

    The index is written last and atomically, so a task whose index exists has
    every chunk it references complete. A retry rewrites the same chunk paths
    from the same shards in the same order, so a partial chunk left by a dead
    pod is overwritten rather than merged.
    """
    task = spec["task"]
    out = spec["out"]
    if StoragePath(index_path(out, task)).exists():
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/tasks_skipped", 1)
        return

    writer = SliceWriter(out, task, spec["partitions"], spec["flush_bytes"])
    members = 0
    for record in _join_shard_group(spec["shards"]):
        writer.add(record)
        members += 1
    slices = writer.close()

    write_parquet_file(iter(slices), index_path(out, task), schema=_SLICE_SCHEMA)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/tasks_scattered", 1)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/members", members)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/slices", len(slices))
    yield {"task": task, "members": members, "slices": len(slices)}


def _read_index(path: str) -> pa.Table:
    with StoragePath(path).open("rb") as handle:
        return pq.ParquetFile(handle).read(columns=_SLICE_SCHEMA.names)


def build_plan(out: str, partitions: int, tasks: int, threads: int) -> None:
    """Pivot the per-task slice indexes into one plan file per partition.

    A gather task must not read all ``tasks`` indexes itself: that would be
    ``tasks * partitions`` small GETs across the job. The driver reads them once
    -- a few hundred megabytes of integers -- and writes each partition the
    ranges it owns.
    """
    marker = prefix_join(out, "scatter/plan/plan.json")
    expected = {"tasks": tasks, "partitions": partitions}
    if StoragePath(marker).exists() and json.loads(StoragePath(marker).read_bytes()) == expected:
        logger.info("Plan already built for %d tasks and %d partitions", tasks, partitions)
        return

    paths = [index_path(out, task) for task in range(tasks)]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        table = pa.concat_tables(list(pool.map(_read_index, paths)))
    logger.info("Pivoting %d slices from %d task indexes", table.num_rows, tasks)

    order = pc.sort_indices(table, sort_keys=[(PARTITION_COLUMN, "ascending"), ("task", "ascending")])
    bounds = _partition_bounds(np.asarray(pc.take(table.column(PARTITION_COLUMN), order)), partitions)

    def write_one(partition: int) -> int:
        start, stop = int(bounds[partition]), int(bounds[partition + 1])
        rows = table.take(order.slice(start, stop - start))
        write_parquet_file(rows.to_batches(), plan_path(out, partition), schema=_SLICE_SCHEMA)
        return stop - start

    with ThreadPoolExecutor(max_workers=threads) as pool:
        written = list(pool.map(write_one, range(partitions)))
    logger.info("Wrote %d plan files, %d..%d slices each", partitions, min(written), max(written))
    StoragePath(marker).write_bytes(json.dumps(expected).encode())


def _ordered_batches(table: pa.Table, order: pa.Array) -> Iterator[pa.RecordBatch]:
    """Yield the table in sorted order without copying it whole."""
    for start in range(0, len(order), TAKE_ROWS):
        yield from table.take(order.slice(start, TAKE_ROWS)).to_batches()


def gather_partition(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Read one partition's slices, sort it by cluster, and write the text file.

    The output is the same schema and the same cluster-key ordering
    ``fuzzy_cluster_text.py`` produces, so ``fuzzy_cluster_verify.py`` and
    ``fuzzy_rule_sweep.py`` read it unchanged.
    """
    partition = spec[PARTITION_COLUMN]
    out = spec["out"]
    path = text_path(out, partition)
    if StoragePath(path).exists():
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/partitions_skipped", 1)
        return

    plan = _read_index(plan_path(out, partition))
    # fsspec caches filesystems per thread, so building the handle once here
    # keeps the read pool from constructing one S3 client per worker thread.
    fs, chunks_root = url_to_fs(prefix_join(out, "scatter/chunks"))

    def read_slice(row: dict[str, Any]) -> pa.Table:
        name = chunk_name(row["task"], row["chunk"])
        payload = fs.cat_file(f"{chunks_root}/{name}", start=row["offset"], end=row["offset"] + row["length"])
        return pq.read_table(pa.BufferReader(pa.py_buffer(payload)))

    rows = plan.to_pylist()
    if not rows:
        write_parquet_file(iter(()), path, schema=_TEXT_SCHEMA)
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/partitions_empty", 1)
        yield {PARTITION_COLUMN: partition, "slices": 0, "members": 0}
        return

    with ThreadPoolExecutor(max_workers=SLICE_READ_CONCURRENCY) as pool:
        slices = list(pool.map(read_slice, rows))
    table = pa.concat_tables(slices)
    del slices

    order = pc.sort_indices(table, sort_keys=[("cluster_key", "ascending"), ("id", "ascending")])
    write_parquet_file(_ordered_batches(table, order), path, schema=_TEXT_SCHEMA)

    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/partitions_written", 1)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/rows_written", table.num_rows)
    yield {PARTITION_COLUMN: partition, "slices": len(rows), "members": table.num_rows}


def write_manifest(out: str, shards: list[TextShard], oversized: dict[str, int], args: argparse.Namespace) -> None:
    """Write the manifest ``fuzzy_cluster_verify.py`` reads to name output shards."""
    manifest = {
        "version": "v1-maponly",
        "candidates": args.candidates,
        "verified": args.verified,
        "max_cluster_size": args.max_cluster_size,
        "partitions": args.partitions,
        "shards_per_task": args.shards_per_task,
        "flush_bytes": args.flush_bytes,
        "duplicate_rule": None,
        "split_ngram_size": SPLIT_NGRAM_SIZE,
        "split_subdivisions": SPLIT_SUBDIVISIONS,
        "oversized_clusters": oversized,
        "shards": [dataclasses.asdict(shard) for shard in shards],
    }
    StoragePath(prefix_join(out, "manifest.json")).write_bytes(json.dumps(manifest).encode())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--verified", required=True, help="Verified artifact that pins the source order")
    parser.add_argument("--large-clusters", required=True, help="large_clusters.parquet from fuzzy_large_clusters.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--phase", choices=("scatter", "plan", "gather", "all"), default="all")
    parser.add_argument("--max-cluster-size", type=int, default=100_000)
    parser.add_argument("--partitions", type=int, default=8192, help="Output files, and gather tasks")
    parser.add_argument("--shards-per-task", type=int, default=48, help="Input shards joined by one scatter task")
    parser.add_argument("--flush-bytes", type=int, default=14 << 30, help="Buffered Arrow bytes per combined chunk")
    parser.add_argument("--plan-threads", type=int, default=32, help="Driver threads for the index pivot")
    parser.add_argument("--limit-shards", type=int, default=0, help="Scatter only the first N shards; for a smoke run")
    parser.add_argument("--max-workers", type=int, default=96)
    parser.add_argument("--worker-cpu", type=float, default=32)
    parser.add_argument("--worker-ram", default="200g")
    parser.add_argument("--worker-disk", default="512g")
    parser.add_argument("--scatter-task-cpu", type=float, default=4)
    parser.add_argument("--scatter-task-ram", default="24g", help="Holds one flush buffer plus its sorted slice")
    parser.add_argument("--gather-task-cpu", type=float, default=2)
    parser.add_argument("--gather-task-ram", default="16g", help="Holds one partition plus its largest cluster")
    parser.add_argument("--task-disk", default="48g")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    shards = build_shards(args.prefix, args.candidates, args.verified)
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    oversized = load_oversized(args.large_clusters, args.max_cluster_size)
    shard_groups = [
        shards[start : start + args.shards_per_task] for start in range(0, len(shards), args.shards_per_task)
    ]
    logger.info(
        "%d shards in %d scatter tasks; %d clusters exceed %d members and will be split",
        len(shards),
        len(shard_groups),
        len(oversized),
        args.max_cluster_size,
    )
    write_manifest(args.out, shards, oversized, args)

    worker = ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk=args.worker_disk)
    scatter_task = ResourceConfig(cpu=args.scatter_task_cpu, ram=args.scatter_task_ram, disk=args.task_disk)
    gather_task = ResourceConfig(cpu=args.gather_task_cpu, ram=args.gather_task_ram, disk=args.task_disk)
    context = ZephyrContext(name="fuzzy-cluster-text-maponly", resources=worker, max_workers=args.max_workers)
    context.put(_SHARED_OVERSIZED_KEY, oversized)

    tally: dict[str, Any] = {}
    if args.phase in ("scatter", "all"):
        specs = [
            {
                "task": task,
                "shards": group,
                "out": args.out,
                "partitions": args.partitions,
                "flush_bytes": args.flush_bytes,
            }
            for task, group in enumerate(shard_groups)
        ]
        outcome = context.execute(
            Dataset.from_list(specs).flat_map(scatter_shard_group), verbose=True, map_task_resources=scatter_task
        )
        tally["scatter"] = dict(sorted(outcome.counters.items()))

    if args.phase in ("plan", "all"):
        build_plan(args.out, args.partitions, len(shard_groups), args.plan_threads)

    if args.phase in ("gather", "all"):
        specs = [{PARTITION_COLUMN: partition, "out": args.out} for partition in range(args.partitions)]
        outcome = context.execute(
            Dataset.from_list(specs).flat_map(gather_partition), verbose=True, map_task_resources=gather_task
        )
        tally["gather"] = dict(sorted(outcome.counters.items()))

    payload = {
        "manifest": prefix_join(args.out, "manifest.json"),
        "phase": args.phase,
        "shards": len(shards),
        "scatter_tasks": len(shard_groups),
        "partitions": args.partitions,
        "oversized_clusters": len(oversized),
        "counters": tally,
    }
    StoragePath(prefix_join(args.out, "summary.json")).write_bytes(json.dumps(payload, indent=1).encode())
    logger.info("Wrote %s", prefix_join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
