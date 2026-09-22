# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Estimate oversized cluster sizes before the text shuffle.

The production sample takes every 256th candidate row within each shard.
Counts are estimates. A cluster near the split threshold can be missed.
"""

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.writers import write_parquet_file

from marin.datakit.source_key import DatakitArtifactPath
from marin.execution.artifact import read_record
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.cluster_text import resolve_data_path
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData

logger = logging.getLogger(__name__)

COUNTER_PREFIX = "fuzzy/large_clusters"
_COUNT_SCHEMA = pa.schema(
    [
        pa.field("dup_cluster_id", pa.string(), nullable=False),
        pa.field("n", pa.int64(), nullable=False),
    ]
)


class LargeClusterParams(BaseModel):
    """Production sample stride and minimum reported cluster size."""

    model_config = ConfigDict(frozen=True)

    stride: int = Field(default=256, ge=1)
    minimum_size: int = Field(default=100_000, ge=1)


class LargeClusterPlan(BaseModel):
    """Sampled cluster counts and the candidate artifact they describe."""

    candidates: DatakitArtifactPath
    counts_path: DatakitArtifactPath
    params: LargeClusterParams
    counters: dict[str, int | float]


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


def _count_group(task: dict[str, Any]) -> dict[str, Any]:
    """Count sampled cluster members across one group of candidate shards."""
    tallies = []
    rows_seen = 0
    for path in task["paths"]:
        with StoragePath(path).open("rb") as handle:
            parquet = pq.ParquetFile(handle)
            if parquet.metadata.num_rows == 0:
                continue
            column = parquet.read(columns=["dup_cluster_id"]).column("dup_cluster_id").combine_chunks()
        rows_seen += len(column)
        selected = range(0, len(column), task["stride"])
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


def plan_large_clusters(
    *,
    prefix: str,
    candidates: str,
    output_path: str,
    params: LargeClusterParams = LargeClusterParams(),
    shards_per_task: int = 64,
    max_workers: int = 48,
    worker_resources: ResourceConfig | None = None,
    task_resources: ResourceConfig | None = None,
) -> LargeClusterPlan:
    """Estimate cluster sizes from the production row-stride sample."""
    if shards_per_task < 1:
        raise ValueError("shards_per_task must be positive")
    StoragePath(output_path).mkdirs()
    started = time.monotonic()

    paths = candidate_shard_paths(prefix, candidates)
    counts_dir = prefix_join(output_path, "counts")
    tasks = [
        {
            "index": index,
            "paths": paths[start : start + shards_per_task],
            "stride": params.stride,
            "output_dir": counts_dir,
        }
        for index, start in enumerate(range(0, len(paths), shards_per_task))
    ]
    logger.info("Counting %d shards in %d map tasks at stride %d", len(paths), len(tasks), params.stride)

    context = ZephyrContext(name="fuzzy-large-clusters", resources=worker_resources, max_workers=max_workers)
    outcome = context.execute(
        Dataset.from_list(tasks).map(_count_group),
        verbose=True,
        map_task_resources=task_resources,
    )
    logger.info("Map stage wrote %d count files in %.0fs", len(outcome.results), time.monotonic() - started)

    tables = []
    for result in outcome.results:
        with StoragePath(result["path"]).open("rb") as handle:
            tables.append(pq.ParquetFile(handle).read(columns=["dup_cluster_id", "n"]))
    merged = pa.concat_tables(tables) if tables else pa.Table.from_pylist([], schema=_COUNT_SCHEMA)
    logger.info("Aggregating %d sampled count rows", merged.num_rows)
    grouped = merged.group_by("dup_cluster_id").aggregate([("n", "sum")])
    sizes = pc.multiply(grouped.column("n_sum"), pa.scalar(params.stride, type=pa.int64()))
    keep = pc.greater_equal(sizes, pa.scalar(params.minimum_size, type=pa.int64()))
    large = pa.table(
        {"dup_cluster_id": grouped.column("dup_cluster_id").filter(keep), "size": sizes.filter(keep)}
    ).sort_by([("size", "descending")])

    with StoragePath(prefix_join(output_path, "large_clusters.parquet")).open("wb") as handle:
        pq.write_table(large, handle)
    payload = {
        "candidates": resolve_data_path(prefix, candidates),
        "stride": params.stride,
        "minimum_size": params.minimum_size,
        "sampled_rows": merged.num_rows,
        "distinct_sampled_clusters": grouped.num_rows,
        "large_clusters": large.num_rows,
        "large_cluster_members": int(pc.sum(large.column("size")).as_py() or 0),
        "largest": large.column("size").to_pylist()[:20],
        "elapsed_seconds": time.monotonic() - started,
        "counters": dict(sorted(outcome.counters.items())),
    }
    StoragePath(prefix_join(output_path, "summary.json")).write_bytes(json.dumps(payload, indent=2).encode())
    logger.info("Large clusters: %s", json.dumps({k: v for k, v in payload.items() if k != "counters"}, indent=1))
    return LargeClusterPlan(
        candidates=resolve_data_path(prefix, candidates),
        counts_path=prefix_join(output_path, "large_clusters.parquet"),
        params=params,
        counters=outcome.counters,
    )


def large_clusters_step(
    *,
    name: str,
    candidates: StepSpec,
    params: LargeClusterParams = LargeClusterParams(),
    shards_per_task: int = 64,
    max_workers: int = 48,
    worker_resources: ResourceConfig | None = None,
    task_resources: ResourceConfig | None = None,
) -> StepSpec:
    """Create a cluster-size plan with candidate lineage and sample identity."""
    return StepSpec(
        name=name,
        deps=[candidates],
        hash_attrs={"version": 1, "params": params.model_dump(mode="json")},
        fn=lambda output_path: plan_large_clusters(
            prefix=marin_prefix(),
            candidates=candidates.output_path,
            output_path=output_path,
            params=params,
            shards_per_task=shards_per_task,
            max_workers=max_workers,
            worker_resources=worker_resources,
            task_resources=task_resources,
        ),
    )
