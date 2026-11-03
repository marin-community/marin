# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count frozen cluster assignments before selection, without reading document text."""

import argparse
import hashlib
import json
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice

import pyarrow.compute as pc
import pyarrow.parquet as pq
from iris.cluster.client.job_info import get_job_info
from marin.datakit import partition_filename
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusterShard:
    source: str
    path: str


def inventory_shards(inventory: dict, max_shards_per_source: int | None) -> list[ClusterShard]:
    """Resolve complete cluster shard lists from a recovered frozen-store inventory."""
    prefix = str(StoragePath(inventory["store_uri"]).parent.parent)
    shards = []
    for source in sorted(inventory["sources"], key=lambda item: item["marin_source"]):
        root = source["artifact_roots"]["cluster"]
        result = inventory["records"][root]["record"]["result"]
        output = result["output_dir"]
        if "://" not in output and not output.startswith("/"):
            output = prefix_join(prefix, output)
        if "cluster_shards" in source:
            paths = source["cluster_shards"]
            if not paths:
                raise ValueError(f"Empty cluster shard manifest for {source['marin_source']}")
        else:
            count = source["normalized_shard_count_hint"]
            if not isinstance(count, int) or count <= 0:
                raise ValueError(f"Unknown normalized shard count for {source['marin_source']}")
            paths = [prefix_join(output, partition_filename(index, count)) for index in range(count)]
        if max_shards_per_source is not None:
            paths = paths[:max_shards_per_source]
        shards.extend(ClusterShard(source["marin_source"], path) for path in paths)
    return shards


def count_cluster_shard(shard: ClusterShard, cluster_view: int, batch_size: int) -> dict:
    """Count all assignments in one Parquet shard using bounded column batches."""
    started = time.monotonic()
    column = f"cluster_{cluster_view}"
    fs, path = filesystem_for(shard.path)
    counts: Counter[int] = Counter()
    with fs.open(path, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        projected_bytes = sum(
            group.column(index).total_compressed_size
            for group in (parquet.metadata.row_group(i) for i in range(parquet.metadata.num_row_groups))
            for index in range(group.num_columns)
            if group.column(index).path_in_schema == column
        )
        for batch in parquet.iter_batches(columns=[column], batch_size=batch_size):
            values = batch.column(0)
            if values.null_count:
                raise ValueError(f"Null cluster assignments in {shard.path}")
            counts.update({item["values"]: item["counts"] for item in pc.value_counts(values).to_pylist()})
        if sum(counts.values()) != parquet.metadata.num_rows:
            raise ValueError(f"Incomplete cluster scan for {shard.path}")
    return {
        "source": shard.source,
        "path": shard.path,
        "rows": sum(counts.values()),
        "cluster_counts": {str(key): value for key, value in sorted(counts.items())},
        "projected_compressed_bytes": projected_bytes,
        "elapsed_seconds": time.monotonic() - started,
    }


def _read_json(path: str) -> dict:
    fs, resolved = filesystem_for(path)
    with fs.open(resolved, "r") as stream:
        return json.load(stream)


def _write_json(path: str, value: dict) -> None:
    fs, resolved = filesystem_for(path)
    parent = str(StoragePath(path).parent)
    fs.makedirs(filesystem_for(parent)[1], exist_ok=True)
    with fs.open(resolved, "w") as stream:
        json.dump(value, stream, sort_keys=True)


def run_census(
    shards: list[ClusterShard],
    output: str,
    *,
    inventory_digest: str,
    cluster_view: int,
    task_index: int,
    task_count: int,
    workers: int,
    batch_size: int,
    checkpoint_interval: float,
    sampled: bool,
) -> dict:
    """Resume one deterministic census task and checkpoint completed shards.

    Counts describe cluster assignments before deduplication, contamination
    filtering, or quality selection. They are not release document counts.
    """
    if not 0 <= task_index < task_count or workers <= 0 or batch_size <= 0 or checkpoint_interval < 0:
        raise ValueError("Invalid task partition, worker count, batch size, or checkpoint interval")
    assigned = shards[task_index::task_count]
    config = {
        "inventory_sha256": inventory_digest,
        "cluster_view": cluster_view,
        "task_index": task_index,
        "task_count": task_count,
        "sampled": sampled,
        "shards_sha256": hashlib.sha256(json.dumps([(s.source, s.path) for s in assigned]).encode()).hexdigest(),
    }
    fs, resolved = filesystem_for(output)
    report: dict = (
        _read_json(output)
        if fs.exists(resolved)
        else {
            "config": config,
            "count_scope": "preselection_cluster_assignments",
            "complete": False,
            "assigned_shards": len(assigned),
            "completed": [],
        }
    )
    if report["config"] != config:
        raise ValueError("Census checkpoint does not match the requested frozen inputs and task partition")
    completed = {item["path"] for item in report["completed"]}
    remaining = iter(shard for shard in assigned if shard.path not in completed)
    last_checkpoint = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while batch := list(islice(remaining, workers)):
            futures = [executor.submit(count_cluster_shard, shard, cluster_view, batch_size) for shard in batch]
            try:
                for future in futures:
                    report["completed"].append(future.result())
            except Exception:
                _write_json(output, report)
                raise
            logger.info("Census task %d: %d/%d shards", task_index, len(report["completed"]), len(assigned))
            if time.monotonic() - last_checkpoint >= checkpoint_interval:
                _write_json(output, report)
                last_checkpoint = time.monotonic()
    source_counts: dict[str, Counter[str]] = {}
    for item in report["completed"]:
        source_counts.setdefault(item["source"], Counter()).update(item["cluster_counts"])
    report["source_cluster_counts"] = {source: dict(counts) for source, counts in sorted(source_counts.items())}
    report["total_rows"] = sum(item["rows"] for item in report["completed"])
    report["projected_compressed_bytes"] = sum(item["projected_compressed_bytes"] for item in report["completed"])
    report["summed_shard_seconds"] = sum(item["elapsed_seconds"] for item in report["completed"])
    report["complete"] = len(report["completed"]) == len(assigned)
    _write_json(output, report)
    return report


def aggregate_census_reports(
    inventory: dict, reports: list[dict], *, sampled_shards_per_source: int | None = None
) -> dict:
    """Aggregate only a complete, disjoint set of census tasks pinned to this inventory."""
    if not reports:
        raise ValueError("No census task reports supplied")
    shards = inventory_shards(inventory, sampled_shards_per_source)
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    task_count = reports[0]["config"]["task_count"]
    indexes = [report["config"]["task_index"] for report in reports]
    if sorted(indexes) != list(range(task_count)):
        raise ValueError("Census reports do not cover every task exactly once")
    sources: dict[str, Counter[str]] = {}
    total_rows = 0
    projected_bytes = 0
    for report in reports:
        config = report["config"]
        assigned = shards[config["task_index"] :: task_count]
        expected_config = {
            "inventory_sha256": digest,
            "cluster_view": inventory["store_config"]["cluster_view"],
            "task_index": config["task_index"],
            "task_count": task_count,
            "sampled": sampled_shards_per_source is not None,
            "shards_sha256": hashlib.sha256(json.dumps([(s.source, s.path) for s in assigned]).encode()).hexdigest(),
        }
        if config != expected_config or not report["complete"]:
            raise ValueError("Incomplete census task or mismatched frozen inputs")
        expected_sources = {shard.path: shard.source for shard in assigned}
        completed = report["completed"]
        if len(completed) != len(expected_sources) or {item["path"] for item in completed} != set(expected_sources):
            raise ValueError("Census task has missing, repeated, or unexpected shards")
        for item in completed:
            if item["source"] != expected_sources[item["path"]] or sum(item["cluster_counts"].values()) != item["rows"]:
                raise ValueError("Census shard counts or source identity do not match")
            sources.setdefault(item["source"], Counter()).update(item["cluster_counts"])
            total_rows += item["rows"]
            projected_bytes += item["projected_compressed_bytes"]
    return {
        "inventory_sha256": digest,
        "cluster_view": inventory["store_config"]["cluster_view"],
        "count_scope": "preselection_cluster_assignments",
        "sampled": sampled_shards_per_source is not None,
        "complete": True,
        "task_count": task_count,
        "completed_shards": len(shards),
        "total_rows": total_rows,
        "projected_compressed_bytes": projected_bytes,
        "source_cluster_counts": {source: dict(counts) for source, counts in sorted(sources.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--checkpoint-interval", type=float, default=60)
    parser.add_argument("--max-shards-per-source", type=int)
    args = parser.parse_args()
    if args.max_shards_per_source is not None and args.max_shards_per_source <= 0:
        parser.error("--max-shards-per-source must be positive")
    info = get_job_info()
    task_index = args.task_index if args.task_index is not None else info.task_index if info is not None else None
    task_count = args.task_count if args.task_count is not None else info.num_tasks if info is not None else None
    if task_index is None or task_count is None:
        parser.error("Supply --task-index and --task-count outside an Iris task")
    inventory = _read_json(args.inventory)
    shards = inventory_shards(inventory, args.max_shards_per_source)
    output = prefix_join(args.output_prefix, f"task-{task_index:05d}-of-{task_count:05d}.json")
    result = run_census(
        shards,
        output,
        inventory_digest=hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
        cluster_view=inventory["store_config"]["cluster_view"],
        task_index=task_index,
        task_count=task_count,
        workers=args.workers,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        sampled=args.max_shards_per_source is not None,
    )
    logger.info("Completed census: %d rows, output %s", result["total_rows"], output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
