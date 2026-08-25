# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge recovered sources and hydrate Code Alchemy placeholder records.

Both stages are co-partitioned by the lowercase first two hexadecimal characters
of ``blob_id``. The merge gives Stack v3 deterministic precedence over a
Software Heritage download when the two sources disagree, while writing every
conflicting value to a side table. Hydration is a left join, so it never drops
or multiplies Code Alchemy rows.
"""

import dataclasses
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, TypeVar

import draccus
import polars as pl
from fray.types import ResourceConfig
from rigging.filesystem.s3_errors import is_transient_s3_error
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.parquet_scan import scan_parquet, storage_options_for_path
from rigging.timing import retry_with_backoff

from marin.utilities.validation_utils import write_provenance_json

logger = logging.getLogger(__name__)

BLOB_PREFIX_HEX_WIDTH = 2
BLOB_PREFIX_PARTITION_COUNT = 16**BLOB_PREFIX_HEX_WIDTH
BLOB_NULL_PARTITION = "null"
PLACEHOLDER = "{{{REPLACE_WITH_BLOB_ID_SOURCE}}}"
DEFAULT_ROOT_PATH = "s3://marin-us-east-02a/tmp/ttl=30d/code-alchemy-hydration"
DEFAULT_CODE_ALCHEMY_PATH = "s3://marin-us-east-02a/tmp/ttl=30d/code-alchemy"
DEFAULT_STACK_SOURCES_PATH = prefix_join(DEFAULT_ROOT_PATH, "stack-v3-matches")
DEFAULT_DOWNLOADED_SOURCES_PATH = prefix_join(DEFAULT_ROOT_PATH, "downloaded-sources")
DEFAULT_FALLBACK_SOURCES_PATH = prefix_join(DEFAULT_ROOT_PATH, "fallback-sources")
DEFAULT_SUBSETS = ("code-dev", "code-dialogue")
OBJECT_STORE_MAX_RETRIES = 10
DEFAULT_MAX_WORKERS = 128
DEFAULT_MAX_SHARD_FAILURES = 5

_HEX_PREFIX = re.compile(r"^[0-9a-f]{2}$")
_SOURCE_SCHEMA = {"blob_id": pl.String, "source": pl.String}
_CONFLICT_SCHEMA = {
    "blob_id": pl.String,
    "source": pl.String,
    "origin": pl.String,
    "selected": pl.Boolean,
}
_SOURCE_UNRESOLVED_SCHEMA = {
    "blob_id": pl.String,
    "source": pl.String,
    "origin": pl.String,
    "reason": pl.String,
}
_ROW_ORDER_COLUMN = "__code_alchemy_hydrate_row_order"
_JOINED_SOURCE_COLUMN = "__code_alchemy_hydrate_source"


@dataclass(frozen=True)
class CodeAlchemyHydrateConfig:
    """Paths, resources, and completion policy for both hydration stages."""

    output_path: str = DEFAULT_ROOT_PATH
    code_alchemy_path: str = DEFAULT_CODE_ALCHEMY_PATH
    stack_sources_path: str = DEFAULT_STACK_SOURCES_PATH
    downloaded_sources_path: str = DEFAULT_DOWNLOADED_SOURCES_PATH
    fallback_sources_path: str = DEFAULT_FALLBACK_SOURCES_PATH
    subsets: tuple[str, ...] = DEFAULT_SUBSETS
    diagnostic_only: bool = False
    max_workers: int = DEFAULT_MAX_WORKERS
    worker_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=16, ram="80g", disk="16g"))
    merge_task_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=4, ram="24g", disk="4g")
    )
    hydrate_task_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=4, ram="32g", disk="4g")
    )
    coordinator_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=1, ram="8g", preemptible=False)
    )


@dataclass(frozen=True)
class MergeSourceTask:
    prefix: str
    stack_paths: tuple[str, ...]
    downloaded_paths: tuple[str, ...]
    fallback_paths: tuple[str, ...]
    output_path: str
    conflicts_path: str
    unresolved_path: str


@dataclass(frozen=True)
class MergeSourceResult:
    prefix: str
    stack_rows: int
    downloaded_rows: int
    fallback_rows: int
    valid_rows: int
    output_rows: int
    duplicate_identical_rows: int
    conflicting_blob_ids: int
    conflicting_source_values: int
    null_blob_id_rows: int
    null_source_rows: int
    prefix_mismatch_rows: int


@dataclass(frozen=True)
class HydrateTask:
    subset: str
    prefix: str
    input_paths: tuple[str, ...]
    source_paths: tuple[str, ...]
    output_path: str
    unresolved_rows_path: str
    unresolved_ids_path: str
    diagnostic_only: bool


@dataclass(frozen=True)
class HydrateResult:
    subset: str
    prefix: str
    input_rows: int
    output_rows: int
    rows_with_placeholder: int
    rows_with_available_source: int
    missing_source_rows: int
    null_blob_id_rows: int
    prefix_mismatch_rows: int
    unresolved_placeholder_rows: int
    unresolved_blob_ids: int


def _empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _collect_parquet(paths: tuple[str, ...], schema: dict[str, pl.DataType] | None = None) -> pl.DataFrame:
    if not paths:
        if schema is None:
            raise ValueError("An explicit schema is required when collecting no Parquet files")
        return _empty_frame(schema)
    return pl.concat([scan_parquet(path) for path in paths]).collect(engine="streaming")


def _sink_parquet(frame: pl.DataFrame, path: str) -> None:
    options: dict[str, object] = dict(storage_options_for_path(path) or {})
    options["max_retries"] = OBJECT_STORE_MAX_RETRIES
    frame.lazy().sink_parquet(
        path,
        compression="zstd",
        compression_level=3,
        statistics=True,
        mkdir=True,
        engine="streaming",
        storage_options=options,
    )


def _source_with_origin(frame: pl.DataFrame, origin: str, rank: int) -> pl.DataFrame:
    missing = set(_SOURCE_SCHEMA) - set(frame.columns)
    if missing:
        raise ValueError(f"{origin} source table is missing canonical columns: {sorted(missing)}")
    return frame.select(
        pl.col("blob_id").cast(pl.String),
        pl.col("source").cast(pl.String),
    ).with_columns(
        pl.lit(origin).alias("origin"),
        pl.lit(rank, dtype=pl.UInt8).alias("__origin_rank"),
    )


def merge_source_frames(
    stack: pl.DataFrame,
    downloaded: pl.DataFrame,
    fallback: pl.DataFrame | None = None,
    *,
    prefix: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, MergeSourceResult]:
    """Merge one source prefix with deterministic conflict selection.

    Stack v3 wins a cross-origin conflict. Ties within an origin use lexical
    source order, making the selected source independent of input file order.
    """

    if _HEX_PREFIX.fullmatch(prefix) is None:
        raise ValueError(f"Invalid blob prefix {prefix!r}")

    if fallback is None:
        fallback = _empty_frame(_SOURCE_SCHEMA)
    stack_rows = _source_with_origin(stack, "stack_v3", 0)
    downloaded_rows = _source_with_origin(downloaded, "software_heritage", 1)
    fallback_rows = _source_with_origin(fallback, "common_pile_stack_edu", 2)
    combined = pl.concat([stack_rows, downloaded_rows, fallback_rows])
    prefix_mismatch = (
        pl.col("blob_id").is_not_null()
        & (pl.col("blob_id").str.slice(0, BLOB_PREFIX_HEX_WIDTH).str.to_lowercase() != pl.lit(prefix))
    )
    null_blob_id_rows = combined.filter(pl.col("blob_id").is_null()).height
    null_source_rows = combined.filter(pl.col("source").is_null()).height
    prefix_mismatch_rows = combined.filter(prefix_mismatch).height

    unresolved = (
        combined.filter(pl.col("blob_id").is_null() | pl.col("source").is_null() | prefix_mismatch)
        .with_columns(
            pl.when(pl.col("blob_id").is_null())
            .then(pl.lit("null_blob_id"))
            .when(pl.col("source").is_null())
            .then(pl.lit("null_source"))
            .otherwise(pl.lit("prefix_mismatch"))
            .alias("reason")
        )
        .select(*_SOURCE_UNRESOLVED_SCHEMA)
    )
    valid = combined.filter(pl.col("blob_id").is_not_null() & pl.col("source").is_not_null() & ~prefix_mismatch)
    distinct_values = (
        valid.sort(["blob_id", "source", "__origin_rank", "origin"])
        .unique(subset=["blob_id", "source"], keep="first", maintain_order=True)
        .sort(["blob_id", "__origin_rank", "source", "origin"])
    )
    selected = distinct_values.unique(subset=["blob_id"], keep="first", maintain_order=True)
    merged = selected.select("blob_id", "source").sort("blob_id")

    conflict_ids = (
        distinct_values.group_by("blob_id")
        .agg(pl.len().alias("__source_count"))
        .filter(pl.col("__source_count") > 1)
    )
    selected_values = selected.select(
        "blob_id",
        pl.col("source").alias("__selected_source"),
    )
    conflicts = (
        distinct_values.join(conflict_ids, on="blob_id", how="inner")
        .join(selected_values, on="blob_id", how="left", validate="m:1")
        .with_columns((pl.col("source") == pl.col("__selected_source")).alias("selected"))
        .select(*_CONFLICT_SCHEMA)
        .sort(["blob_id", "selected", "origin", "source"], descending=[False, True, False, False])
    )

    result = MergeSourceResult(
        prefix=prefix,
        stack_rows=stack.height,
        downloaded_rows=downloaded.height,
        fallback_rows=fallback.height,
        valid_rows=valid.height,
        output_rows=merged.height,
        duplicate_identical_rows=valid.height - distinct_values.height,
        conflicting_blob_ids=conflict_ids.height,
        conflicting_source_values=conflicts.height,
        null_blob_id_rows=null_blob_id_rows,
        null_source_rows=null_source_rows,
        prefix_mismatch_rows=prefix_mismatch_rows,
    )
    return merged, conflicts, unresolved, result


def merge_source_partition(task: MergeSourceTask) -> MergeSourceResult:
    """Merge and materialize one co-partitioned source task."""

    stack = _collect_parquet(task.stack_paths, _SOURCE_SCHEMA)
    downloaded = _collect_parquet(task.downloaded_paths, _SOURCE_SCHEMA)
    fallback = _collect_parquet(task.fallback_paths, _SOURCE_SCHEMA)
    merged, conflicts, unresolved, result = merge_source_frames(
        stack,
        downloaded,
        fallback,
        prefix=task.prefix,
    )
    _sink_parquet(conflicts, task.conflicts_path)
    _sink_parquet(unresolved, task.unresolved_path)
    if result.conflicting_blob_ids or unresolved.height:
        raise RuntimeError(
            f"Source prefix {task.prefix} is not canonical: "
            f"conflicting_blob_ids={result.conflicting_blob_ids}, invalid_rows={unresolved.height}"
        )
    _sink_parquet(merged, task.output_path)
    logger.info("Merged source prefix %s: %d output IDs", task.prefix, result.output_rows)
    return result


def hydrate_frame(
    rows: pl.DataFrame,
    sources: pl.DataFrame,
    *,
    subset: str,
    prefix: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, HydrateResult]:
    """Hydrate one Code Alchemy frame without changing its schema or row order."""

    if prefix != BLOB_NULL_PARTITION and _HEX_PREFIX.fullmatch(prefix) is None:
        raise ValueError(f"Invalid blob prefix {prefix!r}")
    required_columns = {"blob_id", "text_with_placeholders"}
    missing = required_columns - set(rows.columns)
    if missing:
        raise ValueError(f"Code Alchemy {subset} table is missing required columns: {sorted(missing)}")
    if _ROW_ORDER_COLUMN in rows.columns or _JOINED_SOURCE_COLUMN in rows.columns:
        raise ValueError("Code Alchemy input collides with reserved hydration columns")
    missing_source_columns = set(_SOURCE_SCHEMA) - set(sources.columns)
    if missing_source_columns:
        raise ValueError(f"Source table is missing canonical columns: {sorted(missing_source_columns)}")

    canonical_sources = sources.select(
        pl.col("blob_id").cast(pl.String),
        pl.col("source").cast(pl.String).str.strip_chars().alias(_JOINED_SOURCE_COLUMN),
    )
    if canonical_sources.get_column("blob_id").n_unique() != canonical_sources.height:
        raise ValueError(f"Source prefix {prefix} contains duplicate blob_id rows")

    original_columns = rows.columns
    joined = (
        rows.with_row_index(_ROW_ORDER_COLUMN)
        .join(canonical_sources, on="blob_id", how="left", validate="m:1")
        .sort(_ROW_ORDER_COLUMN)
    )
    original_has_marker = pl.col("text_with_placeholders").str.contains(PLACEHOLDER, literal=True)
    hydrated_with_helpers = joined.with_columns(
        pl.when(original_has_marker & pl.col(_JOINED_SOURCE_COLUMN).is_not_null())
        .then(
            pl.col("text_with_placeholders")
            .str.split(PLACEHOLDER)
            .list.join(pl.col(_JOINED_SOURCE_COLUMN))
        )
        .otherwise(pl.col("text_with_placeholders"))
        .alias("text_with_placeholders")
    )
    remaining_marker = pl.col("text_with_placeholders").str.contains(PLACEHOLDER, literal=True)
    unresolved_rows = (
        hydrated_with_helpers.filter(remaining_marker)
        .with_columns(
            pl.when(pl.col("blob_id").is_null())
            .then(pl.lit("null_blob_id"))
            .when(pl.col(_JOINED_SOURCE_COLUMN).is_null())
            .then(pl.lit("source_not_found"))
            .otherwise(pl.lit("replacement_source_contains_marker"))
            .alias("unresolved_reason")
        )
        .select(*original_columns, "unresolved_reason")
    )
    unresolved_ids = (
        unresolved_rows.group_by(["blob_id", "unresolved_reason"], maintain_order=True)
        .agg(pl.len().alias("row_count"))
        .sort(["blob_id", "unresolved_reason"], nulls_last=False)
    )
    hydrated = hydrated_with_helpers.select(original_columns)

    if prefix == BLOB_NULL_PARTITION:
        prefix_mismatch_rows = rows.filter(pl.col("blob_id").is_not_null()).height
    else:
        prefix_mismatch_rows = rows.filter(
            pl.col("blob_id").is_not_null()
            & (pl.col("blob_id").str.slice(0, BLOB_PREFIX_HEX_WIDTH).str.to_lowercase() != pl.lit(prefix))
        ).height
    rows_with_placeholder = rows.filter(original_has_marker).height
    rows_with_available_source = joined.filter(original_has_marker & pl.col(_JOINED_SOURCE_COLUMN).is_not_null()).height
    result = HydrateResult(
        subset=subset,
        prefix=prefix,
        input_rows=rows.height,
        output_rows=hydrated.height,
        rows_with_placeholder=rows_with_placeholder,
        rows_with_available_source=rows_with_available_source,
        missing_source_rows=joined.filter(pl.col(_JOINED_SOURCE_COLUMN).is_null()).height,
        null_blob_id_rows=rows.filter(pl.col("blob_id").is_null()).height,
        prefix_mismatch_rows=prefix_mismatch_rows,
        unresolved_placeholder_rows=unresolved_rows.height,
        unresolved_blob_ids=unresolved_ids.height,
    )
    return hydrated, unresolved_rows, unresolved_ids, result


def hydrate_partition(task: HydrateTask) -> HydrateResult:
    """Hydrate and materialize one Code Alchemy subset/prefix task."""

    rows = _collect_parquet(task.input_paths)
    sources = _collect_parquet(task.source_paths, _SOURCE_SCHEMA)
    hydrated, unresolved_rows, unresolved_ids, result = hydrate_frame(
        rows,
        sources,
        subset=task.subset,
        prefix=task.prefix,
    )
    _sink_parquet(hydrated, task.output_path)
    _sink_parquet(unresolved_rows, task.unresolved_rows_path)
    _sink_parquet(unresolved_ids, task.unresolved_ids_path)
    logger.info(
        "Hydrated %s/%s: %d rows, %d unresolved placeholders",
        task.subset,
        task.prefix,
        result.output_rows,
        result.unresolved_placeholder_rows,
    )
    if result.unresolved_placeholder_rows and not task.diagnostic_only:
        raise RuntimeError(
            f"Hydration left {result.unresolved_placeholder_rows} unresolved placeholders "
            f"in {task.subset}/{task.prefix}"
        )
    return result


def _parquet_paths(stage_path: str, prefix: str) -> tuple[str, ...]:
    pattern = StoragePath(prefix_join(prefix_join(stage_path, "data"), f"blob_prefix={prefix}")) / "*.parquet"
    return retry_with_backoff(
        lambda: tuple(sorted(str(path) for path in pattern.glob())),
        retryable=is_transient_s3_error,
        max_attempts=10,
        operation=f"list source Parquet files for prefix {prefix}",
    )


def list_merge_tasks(cfg: CodeAlchemyHydrateConfig) -> list[MergeSourceTask]:
    """Create exactly one merge task for each hexadecimal prefix."""

    tasks = []
    for prefix_index in range(BLOB_PREFIX_PARTITION_COUNT):
        prefix = f"{prefix_index:02x}"
        stack_paths = _parquet_paths(cfg.stack_sources_path, prefix)
        downloaded_paths = _parquet_paths(cfg.downloaded_sources_path, prefix)
        fallback_paths = _parquet_paths(cfg.fallback_sources_path, prefix)
        if not stack_paths or not downloaded_paths:
            raise FileNotFoundError(
                f"Source stages are incomplete for prefix {prefix}: "
                f"stack_files={len(stack_paths)}, downloaded_files={len(downloaded_paths)}"
            )
        tasks.append(
            MergeSourceTask(
                prefix=prefix,
                stack_paths=stack_paths,
                downloaded_paths=downloaded_paths,
                fallback_paths=fallback_paths,
                output_path=prefix_join(
                    prefix_join(prefix_join(cfg.output_path, "sources/data"), f"blob_prefix={prefix}"),
                    "part-00000.parquet",
                ),
                conflicts_path=prefix_join(
                    prefix_join(prefix_join(cfg.output_path, "sources-conflicts/data"), f"blob_prefix={prefix}"),
                    "part-00000.parquet",
                ),
                unresolved_path=prefix_join(
                    prefix_join(prefix_join(cfg.output_path, "sources-unresolved/data"), f"blob_prefix={prefix}"),
                    "part-00000.parquet",
                ),
            )
        )
    return tasks


def _partition_from_path(path: str) -> str:
    for part in path.split("/"):
        if part.startswith("blob_prefix="):
            prefix = part.removeprefix("blob_prefix=")
            if prefix == BLOB_NULL_PARTITION or _HEX_PREFIX.fullmatch(prefix):
                return prefix
            raise ValueError(f"Invalid Code Alchemy blob partition in {path}: {prefix!r}")
    raise ValueError(f"Code Alchemy path has no blob_prefix partition: {path}")


def list_hydrate_tasks(cfg: CodeAlchemyHydrateConfig) -> list[HydrateTask]:
    """Discover original partitions without inventing or dropping partitions."""

    tasks = []
    source_stage = prefix_join(cfg.output_path, "sources")
    for subset in cfg.subsets:
        subset_path = prefix_join(prefix_join(cfg.code_alchemy_path, "data"), f"subset={subset}")
        pattern = StoragePath(subset_path) / "blob_prefix=*" / "*.parquet"
        grouped: dict[str, list[str]] = {}
        paths = retry_with_backoff(
            lambda: tuple(sorted(str(path) for path in pattern.glob())),
            retryable=is_transient_s3_error,
            max_attempts=10,
            operation=f"list Code Alchemy Parquet files for subset {subset}",
        )
        for path in paths:
            grouped.setdefault(_partition_from_path(path), []).append(path)
        if not grouped:
            raise FileNotFoundError(f"No partitioned Code Alchemy Parquet files found for subset {subset} under {subset_path}")
        for prefix, paths in sorted(grouped.items(), key=lambda item: (item[0] == BLOB_NULL_PARTITION, item[0])):
            source_paths = () if prefix == BLOB_NULL_PARTITION else _parquet_paths(source_stage, prefix)
            if prefix != BLOB_NULL_PARTITION and not source_paths:
                raise FileNotFoundError(f"No merged source Parquet found for {subset}/{prefix}")
            tasks.append(
                HydrateTask(
                    subset=subset,
                    prefix=prefix,
                    input_paths=tuple(paths),
                    source_paths=source_paths,
                    output_path=prefix_join(
                        prefix_join(
                            prefix_join(cfg.output_path, f"hydrated/subset={subset}"),
                            f"blob_prefix={prefix}",
                        ),
                        "part-00000.parquet",
                    ),
                    unresolved_rows_path=prefix_join(
                        prefix_join(
                            prefix_join(cfg.output_path, f"hydrated-unresolved/rows/subset={subset}"),
                            f"blob_prefix={prefix}",
                        ),
                        "part-00000.parquet",
                    ),
                    unresolved_ids_path=prefix_join(
                        prefix_join(
                            prefix_join(cfg.output_path, f"hydrated-unresolved/ids/subset={subset}"),
                            f"blob_prefix={prefix}",
                        ),
                        "part-00000.parquet",
                    ),
                    diagnostic_only=cfg.diagnostic_only,
                )
            )
    return tasks


def _write_metrics(stage_path: str, stage: str, results: Iterable[object]) -> dict[str, int]:
    records = [dataclasses.asdict(result) for result in results]
    records.sort(key=lambda record: (str(record.get("subset", "")), str(record.get("prefix", ""))))
    metrics_path = StoragePath(prefix_join(stage_path, ".metrics"))
    metrics_path.mkdirs()
    (metrics_path / f"{stage}-tasks.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    totals: dict[str, int] = {"task_count": len(records)}
    for record in records:
        for key, value in record.items():
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    (metrics_path / f"{stage}-summary.json").write_text(
        json.dumps(totals, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return totals

_ResultT = TypeVar("_ResultT")


def _read_task_results(metrics_pattern: str, result_type: type[_ResultT]) -> list[_ResultT]:
    paths = sorted(StoragePath(metrics_pattern).glob(), key=str)
    return [
        result_type(**json.loads(line))
        for path in paths
        for line in path.read_text().splitlines()
        if line
    ]

def merge_sources(cfg: CodeAlchemyHydrateConfig) -> list[MergeSourceResult]:
    """Run the 256-way canonical source merge and record its provenance."""

    tasks = list_merge_tasks(cfg)
    stage_path = prefix_join(cfg.output_path, "sources")
    execution_metrics = prefix_join(stage_path, ".metrics/merge-execution-{shard:05d}-of-{total:05d}.jsonl")
    pipeline = Dataset.from_list(tasks).map(merge_source_partition).write_jsonl(
        execution_metrics,
        skip_existing=True,
    )
    ZephyrContext(
        name="code-alchemy-merge-sources",
        resources=cfg.worker_resources,
        coordinator_resources=cfg.coordinator_resources,
        max_workers=min(cfg.max_workers, len(tasks)),
        max_shard_failures=DEFAULT_MAX_SHARD_FAILURES,
        max_execution_retries=10,
    ).execute(pipeline, map_task_resources=cfg.merge_task_resources)
    results = _read_task_results(
        prefix_join(stage_path, ".metrics/merge-execution-*.jsonl"),
        MergeSourceResult,
    )
    if len(results) != len(tasks):
        raise RuntimeError(f"Expected {len(tasks)} merge results, found {len(results)}")
    totals = _write_metrics(stage_path, "merge", results)
    write_provenance_json(
        stage_path,
        metadata={
            "stack_sources_path": cfg.stack_sources_path,
            "downloaded_sources_path": cfg.downloaded_sources_path,
            "fallback_sources_path": cfg.fallback_sources_path,
            "output_columns": ["blob_id", "source"],
            "partition_key": "lower(blob_id[:2])",
            "partition_count": BLOB_PREFIX_PARTITION_COUNT,
            "conflicts_path": prefix_join(cfg.output_path, "sources-conflicts"),
            "unresolved_input_rows_path": prefix_join(cfg.output_path, "sources-unresolved"),
            "conflict_policy": "write diagnostics and fail; differing sources for one blob_id are corruption",
            "metrics": totals,
        },
    )
    return results


def ensure_no_unresolved_placeholders(results: Iterable[HydrateResult], *, diagnostic_only: bool) -> None:
    """Enforce successful completion only when every marker was resolved."""

    unresolved = sum(result.unresolved_placeholder_rows for result in results)
    if unresolved and not diagnostic_only:
        raise RuntimeError(
            f"Hydration left {unresolved} rows containing the exact placeholder marker; "
            "inspect hydrated-unresolved or rerun with diagnostic_only=true"
        )


def hydrate_code_alchemy(cfg: CodeAlchemyHydrateConfig) -> list[HydrateResult]:
    """Run co-partitioned hydration, record diagnostics, and enforce completion."""

    tasks = list_hydrate_tasks(cfg)
    stage_path = prefix_join(cfg.output_path, "hydrated")
    execution_metrics = prefix_join(stage_path, ".metrics/hydrate-execution-{shard:05d}-of-{total:05d}.jsonl")
    pipeline = Dataset.from_list(tasks).map(hydrate_partition).write_jsonl(
        execution_metrics,
        skip_existing=True,
    )
    ZephyrContext(
        name="code-alchemy-hydrate",
        resources=cfg.worker_resources,
        coordinator_resources=cfg.coordinator_resources,
        max_workers=min(cfg.max_workers, len(tasks)),
        max_shard_failures=DEFAULT_MAX_SHARD_FAILURES,
        max_execution_retries=10,
    ).execute(pipeline, map_task_resources=cfg.hydrate_task_resources)
    results = _read_task_results(
        prefix_join(stage_path, ".metrics/hydrate-execution-*.jsonl"),
        HydrateResult,
    )
    if len(results) != len(tasks):
        raise RuntimeError(f"Expected {len(tasks)} hydration results, found {len(results)}")
    totals = _write_metrics(stage_path, "hydrate", results)
    unresolved = totals.get("unresolved_placeholder_rows", 0)
    completion_status = (
        "complete"
        if unresolved == 0
        else "diagnostic_only_with_unresolved_placeholders"
        if cfg.diagnostic_only
        else "failed_unresolved_placeholders"
    )
    write_provenance_json(
        stage_path,
        metadata={
            "code_alchemy_path": cfg.code_alchemy_path,
            "source_path": prefix_join(cfg.output_path, "sources"),
            "subsets": list(cfg.subsets),
            "placeholder": PLACEHOLDER,
            "source_normalization": "strip leading and trailing whitespace per the Code Alchemy dataset card",
            "partition_key": "lower(blob_id[:2]) with null preserved as blob_prefix=null",
            "schema_policy": "all original columns, names, and dtypes preserved",
            "row_policy": "left join; lexical input-file order and original within-file row order preserved",
            "unresolved_path": prefix_join(cfg.output_path, "hydrated-unresolved"),
            "diagnostic_only": cfg.diagnostic_only,
            "completion_status": completion_status,
            "metrics": totals,
        },
    )
    ensure_no_unresolved_placeholders(results, diagnostic_only=cfg.diagnostic_only)
    return results


def run(cfg: CodeAlchemyHydrateConfig) -> None:
    merge_sources(cfg)
    hydrate_code_alchemy(cfg)


@draccus.wrap()
def main(cfg: CodeAlchemyHydrateConfig) -> None:
    configure_logging(level=logging.INFO)
    run(cfg)


if __name__ == "__main__":
    main()
