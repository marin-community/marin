# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count the members of every fuzzy-duplicate candidate cluster.

Connected components produce a heavily skewed cluster size distribution: most
clusters hold a handful of documents and a few hold millions. Any stage that
groups candidate text by cluster has to know which clusters are oversized
before it shuffles, because the split key has to be assigned on the map side.

This reads the ``dup_cluster_id`` column of every candidate shard, counts the
members of each cluster, writes the clusters at or above ``--min-size``, and
records the whole distribution as log2 counters.

    uv run iris --cluster=cw-us-east-02a job run --no-wait --priority batch \
        --cpu 4 --memory 16GB -- python experiments/datakit/scripts/fuzzy_cluster_sizes.py \
            --candidates datakit/dedup_709f5997 --prefix s3://.../marin \
            --out s3://.../user/rav/dedup/cluster_sizes/v1
"""

import argparse
import json
import logging
from collections.abc import Iterator
from typing import Any

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

logger = logging.getLogger(__name__)

COUNTER_PREFIX = "fuzzy/cluster_sizes"
_SIZE_SCHEMA = pa.schema(
    [
        pa.field("dup_cluster_id", pa.string(), nullable=False),
        pa.field("size", pa.int64(), nullable=False),
    ]
)


def _read_shard_group(paths: list[str], sample_stride: int = 1) -> Iterator[dict[str, Any]]:
    """Count a group of shards in one task.

    The shuffle merges every map output in each reducer, so 166,775 map tasks
    put 166,775 open readers in one process and exhaust its memory. Grouping
    the inputs bounds that fan-in without changing the counts.
    """
    for path in paths:
        yield from _read_shard_counts(path, sample_stride)


def _read_shard_counts(path: str, sample_stride: int = 1) -> Iterator[dict[str, Any]]:
    """Pre-aggregate one candidate shard into per-cluster counts.

    ``value_counts`` keeps the aggregation in Arrow. A shard holds tens of
    thousands of rows and almost as many distinct clusters, so a Python
    dictionary over 166,775 shards would dominate the stage.
    """
    with StoragePath(path).open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        if parquet.metadata.num_rows == 0:
            counters.pipeline.update_counter(f"{COUNTER_PREFIX}/shards_empty", 1)
            return
        column = parquet.read(columns=["dup_cluster_id"]).column("dup_cluster_id").combine_chunks()
    if sample_stride > 1:
        # Every cluster this stage has to act on holds at least 100,000
        # members, so a fixed stride over the rows finds it with certainty
        # while shrinking the shuffle by the stride.
        column = column.take(pa.array(range(0, len(column), sample_stride), type=pa.int64()))
    tally = pc.value_counts(column)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/shards_read", 1)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/members_read", len(column))
    values = tally.field("values").to_pylist()
    occurrences = tally.field("counts").to_pylist()
    for cluster_id, count in zip(values, occurrences, strict=True):
        yield {"dup_cluster_id": cluster_id, "n": count}


def _combine(key: str, records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    yield {"dup_cluster_id": key, "n": sum(record["n"] for record in records)}


def _size_bin(size: int) -> str:
    """Log2 bin label, so the distribution fits in a bounded counter set."""
    return f"{1 << (size.bit_length() - 1):09d}"


def _emit_size(
    key: str, records: Iterator[dict[str, Any]], min_size: int, sample_stride: int = 1
) -> Iterator[dict[str, Any]]:
    size = sum(record["n"] for record in records)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/histogram/{_size_bin(size)}", 1)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/clusters", 1)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/members", size)
    if size * sample_stride >= min_size:
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/large_clusters", 1)
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/large_cluster_members", size)
        yield {"dup_cluster_id": key, "size": size * sample_stride}


def _write_sizes(shard: int, records: Iterator[dict[str, Any]], output_dir: str) -> dict[str, Any]:
    path = prefix_join(output_dir, f"part-{shard:05d}.parquet")
    return {**write_parquet_file(records, path, schema=_SIZE_SCHEMA), "shard": shard}


def output_shard_of(cluster_id: str, shards: int) -> int:
    """Stable output shard for a cluster.

    A cluster ID is a decimal 128-bit integer, so its own low bits are a
    uniform, process-independent partition key. Python's ``hash`` is salted per
    process and would place the same cluster differently on different workers.
    """
    return int(cluster_id) % shards


def candidate_shard_paths(prefix: str, candidates: str) -> list[str]:
    """Every candidate attribute shard of a fuzzy-dedup tree."""
    artifact_path = prefix_join(prefix_join(prefix, candidates), ".artifact.json")
    artifact = json.loads(StoragePath(artifact_path).read_bytes())["result"]
    fs, _ = url_to_fs(prefix)
    paths: list[str] = []
    for entry in sorted(artifact["sources"].values(), key=lambda item: str(item["attr_dir"])):
        directory = prefix_join(prefix, str(entry["attr_dir"]))
        _, root = url_to_fs(directory)
        # ``ls`` answers with store-relative keys, so rebuild the full URL: a
        # scheme-less path would be read as a local file by the next stage.
        names = sorted(str(path).rsplit("/", 1)[-1] for path in fs.ls(root, detail=False) if str(path).endswith(".parquet"))
        paths.extend(prefix_join(directory, name) for name in names)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="Storage root the artifact paths resolve against")
    parser.add_argument("--candidates", required=True, help="Candidate artifact path relative to --prefix")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-stride", type=int, default=1, help="Count every Nth row; scales sizes by N")
    parser.add_argument("--min-size", type=int, default=1024, help="Smallest cluster written to the output")
    parser.add_argument("--max-workers", type=int, default=48)
    parser.add_argument("--worker-cpu", type=float, default=48)
    parser.add_argument("--worker-ram", default="256g")
    parser.add_argument("--worker-disk", default="1024g")
    parser.add_argument("--task-cpu", type=float, default=1)
    parser.add_argument("--task-ram", default="5g")
    parser.add_argument("--shards-per-task", type=int, default=64, help="Input shards counted by one map task")
    parser.add_argument("--shuffle-shards", type=int, default=4096)
    parser.add_argument("--output-shards", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    paths = candidate_shard_paths(args.prefix, args.candidates)
    groups = [paths[start : start + args.shards_per_task] for start in range(0, len(paths), args.shards_per_task)]
    logger.info("Counting %d candidate shards in %d map tasks", len(paths), len(groups))

    worker = ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk=args.worker_disk)
    task = ResourceConfig(cpu=args.task_cpu, ram=args.task_ram, disk="64g")
    context = ZephyrContext(name="fuzzy-cluster-sizes", resources=worker, max_workers=args.max_workers)

    pipeline = (
        Dataset.from_list(groups)
        .flat_map(lambda group: _read_shard_group(group, args.sample_stride))
        .group_by(
            key=lambda record: record["dup_cluster_id"],
            reducer=lambda key, records: _emit_size(key, records, args.min_size, args.sample_stride),
            combiner=_combine,
            num_output_shards=args.shuffle_shards,
        )
        .group_by(
            key=lambda record: output_shard_of(record["dup_cluster_id"], args.output_shards),
            reducer=lambda shard, records: _write_sizes(shard, records, prefix_join(args.out, "sizes")),
            sort_by=lambda record: -record["size"],
            num_output_shards=args.output_shards,
        )
    )
    outcome = context.execute(pipeline, verbose=True, map_task_resources=task, reduce_task_resources=task)

    payload = {
        "candidates": args.candidates,
        "min_size": args.min_size,
        "shards": len(paths),
        "counters": dict(sorted(outcome.counters.items())),
    }
    StoragePath(prefix_join(args.out, "summary.json")).write_bytes(json.dumps(payload, indent=1).encode())
    logger.info("Wrote %s", prefix_join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
