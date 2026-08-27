# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find the fuzzy-duplicate clusters that are too large to group in one piece.

The stage that groups candidate text by cluster has to know which clusters are
oversized before it shuffles, because the split key is assigned on the map
side. Only clusters of at least ``--minimum-size`` members matter, and a
cluster that large keeps hundreds of rows under a coarse row sample, so the
count does not have to be exact.

Sampling turns this into a map-only job: each task counts a stride of its own
shards and writes the counts, and the driver adds them up. A shuffle over the
5.95 billion cluster members would move two orders of magnitude more data and
put every map output in every reducer's merge.

    uv run iris --cluster=cw-us-east-02a job run --no-wait --priority batch \
        --cpu 16 --memory 64GB -- python experiments/datakit/scripts/fuzzy_large_clusters.py \
            --prefix s3://.../marin --candidates datakit/dedup_709f5997 \
            --out s3://.../user/rav/dedup/large_clusters/v1
"""

import argparse
import json
import logging
import time
from collections.abc import Iterator
from typing import Any

import dupekit
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import read_record
from marin.processing.classification.deduplication.cluster_text import resolve_data_path
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.writers import write_parquet_file

logger = logging.getLogger(__name__)

COUNTER_PREFIX = "fuzzy/large_clusters"
_COUNT_SCHEMA = pa.schema(
    [
        pa.field("dup_cluster_id", pa.string(), nullable=False),
        pa.field("n", pa.int64(), nullable=False),
    ]
)


def candidate_shard_paths(prefix: str, candidates: str) -> list[str]:
    """List the candidate attribute shards in source and filename order."""
    candidate_path = resolve_data_path(prefix, candidates)
    record = read_record(candidate_path)
    if record is None or record.result is None:
        raise FileNotFoundError(f"No candidate artifact payload at {candidate_path}")
    artifact = FuzzyDupsAttrData.model_validate(record.result)
    paths: list[str] = []
    for entry in sorted(artifact.sources.values(), key=lambda item: item.attr_dir):
        directory = resolve_data_path(prefix, entry.attr_dir)
        fs, root = url_to_fs(directory)
        if not fs.exists(root):
            continue
        names = sorted(
            str(path).rsplit("/", 1)[-1] for path in fs.ls(root, detail=False) if str(path).endswith(".parquet")
        )
        paths.extend(prefix_join(directory, name) for name in names)
    return paths


def _sample_indices(ids: list[str], stride: int) -> list[int]:
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    hashes = dupekit.hash_xxh3_64_batch([record_id.encode("utf-8", "surrogatepass") for record_id in ids])
    return [index for index, value in enumerate(hashes) if value % stride == 0]


def _count_group(task: dict[str, Any]) -> dict[str, Any]:
    """Count sampled cluster members across one group of candidate shards."""
    tallies = []
    rows_seen = 0
    for path in task["paths"]:
        with StoragePath(path).open("rb") as handle:
            parquet = pq.ParquetFile(handle)
            if parquet.metadata.num_rows == 0:
                continue
            table = parquet.read(columns=["id", "dup_cluster_id"])
            column = table.column("dup_cluster_id").combine_chunks()
        rows_seen += len(column)
        selected = _sample_indices(table.column("id").to_pylist(), task["stride"])
        sampled = column.take(pa.array(selected, type=pa.int64()))
        if len(sampled):
            tallies.append(pa.table({"dup_cluster_id": sampled}))
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/rows_seen", rows_seen)

    def rows() -> Iterator[dict[str, Any]]:
        if not tallies:
            return
        merged = pa.concat_tables(tallies).column("dup_cluster_id").combine_chunks()
        counted = pc.value_counts(merged)
        for cluster_id, count in zip(
            counted.field("values").to_pylist(), counted.field("counts").to_pylist(), strict=True
        ):
            yield {"dup_cluster_id": cluster_id, "n": count}

    path = prefix_join(task["output_dir"], f"part-{task['index']:05d}.parquet")
    result = write_parquet_file(rows(), path, schema=_COUNT_SCHEMA)
    return {"index": task["index"], "path": path, "count": result["count"]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--minimum-size", type=int, default=100_000)
    parser.add_argument("--shards-per-task", type=int, default=64)
    parser.add_argument("--max-workers", type=int, default=48)
    parser.add_argument("--worker-cpu", type=float, default=16)
    parser.add_argument("--worker-ram", default="128g")
    parser.add_argument("--worker-disk", default="64g")
    parser.add_argument("--task-cpu", type=float, default=1)
    parser.add_argument("--task-ram", default="7g")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.monotonic()

    paths = candidate_shard_paths(args.prefix, args.candidates)
    counts_dir = prefix_join(args.out, "counts")
    tasks = [
        {
            "index": index,
            "paths": paths[start : start + args.shards_per_task],
            "stride": args.stride,
            "output_dir": counts_dir,
        }
        for index, start in enumerate(range(0, len(paths), args.shards_per_task))
    ]
    logger.info("Counting %d shards in %d map tasks at stride %d", len(paths), len(tasks), args.stride)

    worker = ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk=args.worker_disk)
    task_resources = ResourceConfig(cpu=args.task_cpu, ram=args.task_ram, disk="32g")
    context = ZephyrContext(name="fuzzy-large-clusters", resources=worker, max_workers=args.max_workers)
    outcome = context.execute(
        Dataset.from_list(tasks).map(_count_group),
        verbose=True,
        map_task_resources=task_resources,
    )
    logger.info("Map stage wrote %d count files in %.0fs", len(outcome.results), time.monotonic() - started)

    fs, root = url_to_fs(counts_dir)
    names = sorted(str(path).rsplit("/", 1)[-1] for path in fs.ls(root, detail=False) if str(path).endswith(".parquet"))
    tables = []
    for name in names:
        with StoragePath(prefix_join(counts_dir, name)).open("rb") as handle:
            tables.append(pq.ParquetFile(handle).read(columns=["dup_cluster_id", "n"]))
    merged = pa.concat_tables(tables)
    logger.info("Aggregating %d sampled count rows", merged.num_rows)
    grouped = merged.group_by("dup_cluster_id").aggregate([("n", "sum")])
    sizes = pc.multiply(grouped.column("n_sum"), pa.scalar(args.stride, type=pa.int64()))
    keep = pc.greater_equal(sizes, pa.scalar(args.minimum_size, type=pa.int64()))
    large = pa.table(
        {"dup_cluster_id": grouped.column("dup_cluster_id").filter(keep), "size": sizes.filter(keep)}
    ).sort_by([("size", "descending")])

    with StoragePath(prefix_join(args.out, "large_clusters.parquet")).open("wb") as handle:
        pq.write_table(large, handle)
    payload = {
        "candidates": args.candidates,
        "stride": args.stride,
        "minimum_size": args.minimum_size,
        "sampled_rows": merged.num_rows,
        "distinct_sampled_clusters": grouped.num_rows,
        "large_clusters": large.num_rows,
        "large_cluster_members": int(pc.sum(large.column("size")).as_py() or 0),
        "largest": large.column("size").to_pylist()[:20],
        "elapsed_seconds": time.monotonic() - started,
        "counters": dict(sorted(outcome.counters.items())),
    }
    StoragePath(prefix_join(args.out, "summary.json")).write_bytes(json.dumps(payload, indent=2).encode())
    logger.info("Large clusters: %s", json.dumps({k: v for k, v in payload.items() if k != "counters"}, indent=1))


if __name__ == "__main__":
    main()
