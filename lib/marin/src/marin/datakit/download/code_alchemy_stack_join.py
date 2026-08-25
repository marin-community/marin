# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prefix-aligned Code Alchemy to Stack v3 source join."""

import json
import logging
from dataclasses import asdict, dataclass, field

import draccus
import polars as pl
from fray.types import ResourceConfig
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.filesystem.s3_errors import is_transient_s3_error
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.parquet_scan import scan_parquet, storage_options_for_path
from rigging.timing import retry_with_backoff

from marin.utilities.validation_utils import write_provenance_json

BLOB_PREFIX_COUNT = 256
SUBSETS = ("code-dev", "code-dialogue")
DEFAULT_CODE_ALCHEMY_PATH = "s3://marin-us-east-02a/tmp/ttl=30d/code-alchemy/data"
DEFAULT_STACK_V3_PATH = "s3://marin-us-east-02a/tmp/ttl=30d/stack-v3-content-ids/data"


@dataclass(frozen=True)
class CodeAlchemyStackJoinConfig:
    output_path: str = field(default_factory=lambda: marin_temp_bucket(ttl_days=30, prefix="code-alchemy-hydration"))
    code_alchemy_path: str = DEFAULT_CODE_ALCHEMY_PATH
    stack_v3_path: str = DEFAULT_STACK_V3_PATH
    subsets: tuple[str, ...] = SUBSETS
    max_workers: int = 128
    worker_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=16, ram="96g", disk="20g"))
    task_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=8, ram="64g", disk="10g"))
    coordinator_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=2, ram="8g", preemptible=False)
    )


@dataclass(frozen=True)
class CodeAlchemyStackJoinTask:
    prefix: str
    requested_paths: tuple[str, ...]
    stack_path: str
    matches_path: str
    missing_path: str
    requested_stage_path: str
    stack_groups_stage_path: str


@dataclass(frozen=True)
class CodeAlchemyStackJoinResult:
    prefix: str
    requested_unique_ids: int
    matched_ids: int
    missing_ids: int
    stack_candidate_rows: int
    stack_duplicate_rows_for_requested_ids: int
    stack_conflicting_ids_for_requested_ids: int
    requested_input_paths: tuple[str, ...]
    stack_input_path: str
    matches_output_path: str
    missing_output_path: str


def _glob_parquet(path: str) -> tuple[str, ...]:
    return retry_with_backoff(
        lambda: tuple(sorted(str(item) for item in StoragePath(prefix_join(path, "*.parquet")).glob())),
        retryable=is_transient_s3_error,
        max_attempts=10,
        operation=f"list Parquet files under {path}",
    )


def list_code_alchemy_stack_join_tasks(cfg: CodeAlchemyStackJoinConfig) -> list[CodeAlchemyStackJoinTask]:
    """Create exactly 256 tasks whose reads cannot cross prefix boundaries."""
    tasks = []
    for index in range(BLOB_PREFIX_COUNT):
        prefix = f"{index:02x}"
        requested_paths = tuple(
            path
            for subset in cfg.subsets
            for path in _glob_parquet(
                prefix_join(prefix_join(cfg.code_alchemy_path, f"subset={subset}"), f"blob_prefix={prefix}")
            )
        )
        if not requested_paths:
            raise FileNotFoundError(f"No Code Alchemy parquet files for prefix {prefix}")
        tasks.append(
            CodeAlchemyStackJoinTask(
                prefix=prefix,
                requested_paths=requested_paths,
                stack_path=prefix_join(prefix_join(cfg.stack_v3_path, f"content_prefix={prefix}"), "part-00000.parquet"),
                matches_path=prefix_join(
                    prefix_join(prefix_join(cfg.output_path, "stack-v3-matches/data"), f"blob_prefix={prefix}"),
                    "part-00000.parquet",
                ),
                missing_path=prefix_join(
                    prefix_join(prefix_join(cfg.output_path, "missing-ids/data"), f"blob_prefix={prefix}"),
                    "part-00000.parquet",
                ),
                requested_stage_path=prefix_join(
                    prefix_join(prefix_join(cfg.output_path, "_intermediate/requested"), f"blob_prefix={prefix}"),
                    "part-00000.parquet",
                ),
                stack_groups_stage_path=prefix_join(
                    prefix_join(prefix_join(cfg.output_path, "_intermediate/stack-groups"), f"blob_prefix={prefix}"),
                    "part-00000.parquet",
                ),
            )
        )
    return tasks


def _storage_options(path: str) -> dict[str, object]:
    options: dict[str, object] = dict(storage_options_for_path(path) or {})
    options["max_retries"] = 10
    return options


def _requested(task: CodeAlchemyStackJoinTask) -> pl.LazyFrame:
    return (
        pl.concat([scan_parquet(path).select(pl.col("blob_id").cast(pl.String)) for path in task.requested_paths])
        .filter(pl.col("blob_id").is_not_null())
        .unique(subset="blob_id")
        .sort("blob_id")
    )


def _stack_groups(task: CodeAlchemyStackJoinTask, requested: pl.LazyFrame) -> pl.LazyFrame:
    """Semi-join before grouping to avoid expanding irrelevant Stack duplicates."""
    return (
        scan_parquet(task.stack_path)
        .select(
            pl.col("content_id").cast(pl.String).str.to_lowercase().alias("blob_id"),
            pl.col("content").cast(pl.String).alias("source"),
        )
        .filter(pl.col("blob_id").is_not_null() & pl.col("source").is_not_null())
        .join(requested, on="blob_id", how="semi")
        .group_by("blob_id")
        .agg(
            pl.len().alias("stack_row_count"),
            pl.col("source").n_unique().alias("distinct_source_count"),
            pl.col("source").min().alias("source"),
        )
    )


def join_code_alchemy_stack_prefix(task: CodeAlchemyStackJoinTask) -> CodeAlchemyStackJoinResult:
    """Materialize each expensive input once, then write canonical matches and misses."""
    _requested(task).sink_parquet(
        task.requested_stage_path,
        compression="zstd",
        compression_level=3,
        statistics=True,
        mkdir=True,
        engine="streaming",
        storage_options=_storage_options(task.requested_stage_path),
    )
    requested = scan_parquet(task.requested_stage_path)
    _stack_groups(task, requested).sink_parquet(
        task.stack_groups_stage_path,
        compression="zstd",
        compression_level=3,
        statistics=True,
        mkdir=True,
        engine="streaming",
        storage_options=_storage_options(task.stack_groups_stage_path),
    )
    groups = scan_parquet(task.stack_groups_stage_path)
    candidate_rows, duplicate_rows, conflicting_ids = (
        int(value or 0)
        for value in groups.select(
            pl.col("stack_row_count").sum().alias("candidate_rows"),
            (pl.col("stack_row_count") - 1).sum().alias("duplicate_rows"),
            (pl.col("distinct_source_count") > 1).sum().alias("conflicting_ids"),
        ).collect(engine="streaming").row(0)
    )
    # Stack v3 can retain one content_id across several redacted source variants.
    # Those IDs are not safe hydration inputs; send them through the authoritative
    # Software Heritage download path with ordinary misses.
    sources = groups.filter(pl.col("distinct_source_count") == 1).select("blob_id", "source")
    sources.sort("blob_id").sink_parquet(
        task.matches_path, compression="zstd", compression_level=3, statistics=True, mkdir=True,
        engine="streaming", storage_options=_storage_options(task.matches_path),
    )
    requested.join(sources.select("blob_id"), on="blob_id", how="anti").select("blob_id").sort("blob_id").sink_parquet(
        task.missing_path, compression="zstd", compression_level=3, statistics=True, mkdir=True,
        engine="streaming", storage_options=_storage_options(task.missing_path),
    )
    requested_count = int(requested.select(pl.len()).collect(engine="streaming").item())
    matched_count = int(scan_parquet(task.matches_path).select(pl.len()).collect(engine="streaming").item())
    missing_count = int(scan_parquet(task.missing_path).select(pl.len()).collect(engine="streaming").item())
    if requested_count != matched_count + missing_count:
        raise RuntimeError(f"Prefix {task.prefix} coverage mismatch: {requested_count} != {matched_count} + {missing_count}")
    return CodeAlchemyStackJoinResult(
        prefix=task.prefix, requested_unique_ids=requested_count, matched_ids=matched_count, missing_ids=missing_count,
        stack_candidate_rows=candidate_rows, stack_duplicate_rows_for_requested_ids=duplicate_rows,
        stack_conflicting_ids_for_requested_ids=conflicting_ids, requested_input_paths=task.requested_paths,
        stack_input_path=task.stack_path, matches_output_path=task.matches_path, missing_output_path=task.missing_path,
    )


def build_code_alchemy_stack_join_pipeline(tasks: list[CodeAlchemyStackJoinTask], metrics_path: str) -> Dataset[str]:
    """Metrics shards are completion markers, so finished prefixes are skipped on rerun."""
    return Dataset.from_list(tasks).map(join_code_alchemy_stack_prefix).write_jsonl(
        prefix_join(metrics_path, "prefix-{shard:05d}-of-{total:05d}.jsonl"), skip_existing=True
    )


def _write_aggregate(cfg: CodeAlchemyStackJoinConfig, metrics_path: str) -> dict[str, object]:
    metric_paths = sorted(StoragePath(prefix_join(metrics_path, "prefix-*.jsonl")).glob(), key=str)
    records = [
        json.loads(line)
        for path in metric_paths
        for line in path.read_text().splitlines()
        if line
    ]
    if len(records) != BLOB_PREFIX_COUNT:
        raise RuntimeError(f"Expected 256 prefix metrics, found {len(records)}")
    metrics = pl.DataFrame(records)
    columns = ("requested_unique_ids", "matched_ids", "missing_ids", "stack_candidate_rows",
               "stack_duplicate_rows_for_requested_ids", "stack_conflicting_ids_for_requested_ids")
    totals = {column: int(metrics[column].sum()) for column in columns}
    if totals["requested_unique_ids"] != totals["matched_ids"] + totals["missing_ids"]:
        raise RuntimeError(f"Global coverage mismatch: {totals}")
    aggregate: dict[str, object] = {
        **totals, "matched_fraction": totals["matched_ids"] / totals["requested_unique_ids"],
        "prefix_count": BLOB_PREFIX_COUNT, "subsets": list(cfg.subsets),
        "code_alchemy_path": cfg.code_alchemy_path, "stack_v3_path": cfg.stack_v3_path,
        "matches_path": prefix_join(cfg.output_path, "stack-v3-matches/data"),
        "missing_path": prefix_join(cfg.output_path, "missing-ids/data"),
    }
    with open_url(prefix_join(metrics_path, "aggregate-coverage.json"), "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return aggregate


def join_code_alchemy_to_stack_v3(cfg: CodeAlchemyStackJoinConfig) -> None:
    tasks = list_code_alchemy_stack_join_tasks(cfg)
    metrics_path = prefix_join(cfg.output_path, "stack-v3-matches/.metrics")
    context = ZephyrContext(
        name="code-alchemy-stack-v3-prefix-join", resources=cfg.worker_resources,
        coordinator_resources=cfg.coordinator_resources, max_workers=min(cfg.max_workers, len(tasks)),
        max_shard_failures=5, max_execution_retries=10,
    )
    context.execute(build_code_alchemy_stack_join_pipeline(tasks, metrics_path), map_task_resources=cfg.task_resources)
    aggregate = _write_aggregate(cfg, metrics_path)
    write_provenance_json(prefix_join(cfg.output_path, "stack-v3-matches"), metadata={
        **aggregate, "partition_key": "lower(blob_id[:2])",
        "requested_deduplication": "unique blob_id across code-dev and code-dialogue",
        "stack_deduplication": "semi-join requested IDs and accept only IDs with one distinct source",
        "conflict_policy": "treat IDs with distinct Stack source values as misses for Software Heritage recovery",
        "output_columns": ["blob_id", "source"], "missing_output_columns": ["blob_id"],
        "worker_resources": asdict(cfg.worker_resources), "task_resources": asdict(cfg.task_resources),
    })


@draccus.wrap()
def main(cfg: CodeAlchemyStackJoinConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    join_code_alchemy_to_stack_v3(cfg)


if __name__ == "__main__":
    main()
