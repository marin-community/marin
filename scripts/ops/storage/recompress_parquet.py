#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite Parquet objects in place with zstd compression through Zephyr.

Each task validates a temporary sibling before replacing the source. Apply only
to quiescent prefixes; lifecycle-managed paths are rejected.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

import click
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.buckets import filesystem_for
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner
from zephyr.shuffle import ListShard
from zephyr.stage_io import ShardTask, TaskResult
from zephyr.worker_context import CounterEntry
from zephyr.writers import (
    DEFAULT_PARQUET_COMPRESSION,
    DEFAULT_PARQUET_COMPRESSION_LEVEL,
    DEFAULT_PARQUET_MAX_ROWS_PER_PAGE,
    DEFAULT_PARQUET_WRITE_PAGE_INDEX,
    accumulate_record_batch_tables,
)

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = 16
DEFAULT_BATCH_ROWS = 1_024
DEFAULT_ROW_GROUP_BYTES = 128 * 1024 * 1024
MIN_INPUT_BYTES = 10 * 1024 * 1024
DEFAULT_WORKER_CPU = 1
DEFAULT_WORKER_RAM = "2g"
DEFAULT_WORKER_DISK = "8g"
DEFAULT_COORDINATOR_CPU = 0.5
DEFAULT_COORDINATOR_RAM = "1g"
DEFAULT_COORDINATOR_DISK = "8g"
REWRITE_SUBPROCESS_MEMORY_LIMIT = 5 * 1024**3
REWRITE_CONTEXT_NAME = "recompress-parquet"
COUNTER_PREFIX = "parquet_recompression"
SKIPPED_PATHS = frozenset(
    {
        "s3://marin-us-east-02a/marin/datakit/tokenize/nemotron_cc_v2_1/"
        "medium_high_quality_synthetic_bad94a5f/train/part-00000-of-00001.parquet",
    }
)


class RewriteDisposition(StrEnum):
    """Outcome of inspecting or rewriting one Parquet object."""

    REWRITTEN = "rewritten"
    ALREADY_TARGET = "already_target"
    DRY_RUN = "dry_run"
    NOT_SMALLER = "not_smaller"
    TOO_SMALL = "too_small"


class RewriteMode(StrEnum):
    """Whether a migration inventories candidates or replaces sources."""

    DRY_RUN = "dry_run"
    APPLY = "apply"


@dataclass(frozen=True)
class RewriteOptions:
    """Controls one bounded Parquet rewrite."""

    mode: RewriteMode = RewriteMode.DRY_RUN
    batch_rows: int = DEFAULT_BATCH_ROWS
    row_group_bytes: int = DEFAULT_ROW_GROUP_BYTES
    compression_level: int = DEFAULT_PARQUET_COMPRESSION_LEVEL


@dataclass(frozen=True)
class RewriteResult:
    """Observable outcome for one Parquet object."""

    path: str
    disposition: RewriteDisposition
    input_bytes: int
    output_bytes: int | None
    rows: int | None


@dataclass(frozen=True)
class ObjectFingerprint:
    """Fields that change when an object is replaced during a rewrite."""

    size: int
    etag: str | None
    version: str | None
    modified: str | None


class _OutputNotSmaller(Exception):
    pass


class _OOMSkippingSubprocessRunner:
    """Treat an OOM-killed rewrite subprocess as an empty output shard."""

    def __init__(self) -> None:
        self._runner = SubprocessRunner(memory_limit_bytes=REWRITE_SUBPROCESS_MEMORY_LIMIT)

    def execute(
        self,
        task: ShardTask,
        chunk_prefix: str,
        execution_id: str,
    ) -> tuple[TaskResult, dict[str, CounterEntry]]:
        try:
            return self._runner.execute(task, chunk_prefix, execution_id)
        except MemoryError as error:
            logger.error("Skipping OOM-killed Parquet shard %d: %s", task.shard_idx, error)
            task_counters = self._runner.live_counters()
            task_counters[f"{COUNTER_PREFIX}/files_skipped_oom"] = CounterEntry(1, stage=task.stage_name)
            return TaskResult(shard=ListShard(refs=[])), task_counters

    def live_counters(self) -> dict[str, CounterEntry]:
        return self._runner.live_counters()


def _info_value(info: dict, *names: str) -> str | None:
    for name in names:
        value = info.get(name)
        if value is not None:
            return str(value)
    return None


def _fingerprint(info: dict) -> ObjectFingerprint:
    return ObjectFingerprint(
        size=int(info["size"]),
        etag=_info_value(info, "etag", "ETag"),
        version=_info_value(info, "version_id", "VersionId", "generation"),
        modified=_info_value(info, "mtime", "updated", "LastModified"),
    )


def _column_compressions(metadata: pq.FileMetaData) -> set[str]:
    return {
        metadata.row_group(row_group).column(column).compression.upper()
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    }


def _has_page_indexes(metadata: pq.FileMetaData) -> bool:
    return all(
        metadata.row_group(row_group).column(column).num_values == 0
        or (
            metadata.row_group(row_group).column(column).has_column_index
            and metadata.row_group(row_group).column(column).has_offset_index
        )
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    )


def _validate_rewrite(source: pq.ParquetFile, rewritten: pq.ParquetFile) -> None:
    if rewritten.schema_arrow != source.schema_arrow:
        raise ValueError("rewritten Parquet schema differs from the source")
    if rewritten.metadata.num_rows != source.metadata.num_rows:
        raise ValueError(
            f"rewritten Parquet row count differs: {rewritten.metadata.num_rows} != {source.metadata.num_rows}"
        )
    codecs = _column_compressions(rewritten.metadata)
    if codecs and codecs != {DEFAULT_PARQUET_COMPRESSION.upper()}:
        raise ValueError(f"rewritten Parquet has unexpected compression codecs: {sorted(codecs)}")


def recompress_parquet(path: str, options: RewriteOptions = RewriteOptions()) -> RewriteResult:
    """Inspect or rewrite one Parquet object to zstd with page indexes.

    The source is replaced only after the temporary output has the same schema
    and row count, uses zstd for every column chunk, and the source fingerprint
    still matches the value observed before reading it. Non-zstd sources are
    preserved when the rewritten object would not be smaller. Object-store
    callers must stop producers for the source prefix before applying rewrites.
    """
    if "/tmp/ttl=" in path:
        raise ValueError(f"refusing to reset lifecycle retention by replacing {path}")
    if options.batch_rows <= 0:
        raise ValueError(f"batch_rows must be positive, got {options.batch_rows}")
    if options.row_group_bytes <= 0:
        raise ValueError(f"row_group_bytes must be positive, got {options.row_group_bytes}")

    fs, fs_path = filesystem_for(path)
    source_fingerprint = _fingerprint(fs.info(fs_path))
    if source_fingerprint.size < MIN_INPUT_BYTES:
        return RewriteResult(path, RewriteDisposition.TOO_SMALL, source_fingerprint.size, None, None)

    with fs.open(fs_path, "rb") as source_handle:
        source = pq.ParquetFile(source_handle)
        rows = source.metadata.num_rows
        source_is_zstd = _column_compressions(source.metadata) == {DEFAULT_PARQUET_COMPRESSION.upper()}
        if source_is_zstd and _has_page_indexes(source.metadata):
            return RewriteResult(path, RewriteDisposition.ALREADY_TARGET, source_fingerprint.size, None, rows)
        if options.mode is RewriteMode.DRY_RUN:
            return RewriteResult(path, RewriteDisposition.DRY_RUN, source_fingerprint.size, None, rows)

        try:
            for abandoned_path in fs.glob(f"{fs_path}.tmp.*"):
                fs.rm(abandoned_path)
            with atomic_rename(fs_path, filesystem=fs) as temp_path:
                with fs.open(temp_path, "wb") as output_handle:
                    with pq.ParquetWriter(
                        output_handle,
                        source.schema_arrow,
                        compression=DEFAULT_PARQUET_COMPRESSION,
                        compression_level=options.compression_level,
                        write_page_index=DEFAULT_PARQUET_WRITE_PAGE_INDEX,
                        max_rows_per_page=DEFAULT_PARQUET_MAX_ROWS_PER_PAGE,
                    ) as writer:
                        for table in accumulate_record_batch_tables(
                            source.iter_batches(batch_size=options.batch_rows),
                            schema=source.schema_arrow,
                            target_bytes=options.row_group_bytes,
                        ):
                            writer.write_table(table, row_group_size=table.num_rows)

                output_bytes = int(fs.info(temp_path)["size"])
                if output_bytes >= source_fingerprint.size and not source_is_zstd:
                    raise _OutputNotSmaller

                with fs.open(temp_path, "rb") as rewritten_handle:
                    _validate_rewrite(source, pq.ParquetFile(rewritten_handle))

                current_fingerprint = _fingerprint(fs.info(fs_path))
                if current_fingerprint != source_fingerprint:
                    raise ValueError(f"source changed while it was being recompressed: {path}")
        except _OutputNotSmaller:
            return RewriteResult(
                path,
                RewriteDisposition.NOT_SMALLER,
                source_fingerprint.size,
                output_bytes,
                rows,
            )

    fs.invalidate_cache(fs_path)
    return RewriteResult(path, RewriteDisposition.REWRITTEN, source_fingerprint.size, output_bytes, rows)


def _rewrite_for_zephyr(path: str, options: RewriteOptions) -> list[dict]:
    """Record rewrite counters without emitting downstream rows."""
    if path in SKIPPED_PATHS:
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/files_skipped", 1)
        logger.warning("Skipping known pathological Parquet: %s", path)
        return []
    logger.info("Inspecting Parquet: %s", path)
    try:
        result = recompress_parquet(path, options)
    except Exception:
        # One bad object must not abort its entire inventory rollup.
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/files_failed", 1)
        logger.exception("Skipping failed Parquet rewrite: %s", path)
        return []

    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/files_{result.disposition}", 1)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/input_bytes_{result.disposition}", result.input_bytes)
    if result.output_bytes is not None:
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/output_bytes_{result.disposition}", result.output_bytes)
    if result.rows is not None:
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/rows_{result.disposition}", result.rows)
    logger.info(
        "%s: %s (%d -> %s bytes)",
        result.disposition,
        path,
        result.input_bytes,
        result.output_bytes,
    )
    return []


def run_migration(
    source_globs: Sequence[str],
    *,
    context: ZephyrContext,
    options: RewriteOptions,
) -> dict[str, int | float]:
    """Run the bounded recompression pipeline and return aggregate counters."""
    if not source_globs:
        raise ValueError("source_globs must contain at least one pattern")

    pipeline = Dataset.from_file_patterns(
        source_globs,
        empty_glob_ok=True,
        minimum_file_size=MIN_INPUT_BYTES,
    ).flat_map(partial(_rewrite_for_zephyr, options=options))
    outcome = context.execute(pipeline, verbose=True)
    return dict(outcome.counters)


def create_rewrite_context(
    *,
    workers: int,
    worker_cpu: int,
    worker_ram: str,
    worker_disk: str,
    coordinator_cpu: float,
) -> ZephyrContext:
    """Create the shared worker pool used by Parquet rewrites."""
    return ZephyrContext(
        name=REWRITE_CONTEXT_NAME,
        resources=ResourceConfig(cpu=worker_cpu, ram=worker_ram, disk=worker_disk, preemptible=False),
        coordinator_resources=ResourceConfig(
            cpu=coordinator_cpu,
            ram=DEFAULT_COORDINATOR_RAM,
            disk=DEFAULT_COORDINATOR_DISK,
            preemptible=False,
        ),
        max_workers=workers,
        stage_runner_factory=_OOMSkippingSubprocessRunner,
    )


@click.command()
@click.argument("source_glob")
@click.option("--workers", default=DEFAULT_WORKERS, show_default=True, type=click.IntRange(min=1))
@click.option("--worker-cpu", default=DEFAULT_WORKER_CPU, show_default=True, type=click.IntRange(min=1))
@click.option("--worker-ram", default=DEFAULT_WORKER_RAM, show_default=True)
@click.option("--worker-disk", default=DEFAULT_WORKER_DISK, show_default=True)
@click.option(
    "--coordinator-cpu",
    default=DEFAULT_COORDINATOR_CPU,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
)
@click.option("--batch-rows", default=DEFAULT_BATCH_ROWS, show_default=True, type=click.IntRange(min=1))
@click.option(
    "--row-group-bytes",
    default=DEFAULT_ROW_GROUP_BYTES,
    show_default=True,
    type=click.IntRange(min=1),
)
@click.option(
    "--compression-level",
    default=DEFAULT_PARQUET_COMPRESSION_LEVEL,
    show_default=True,
    type=click.IntRange(min=1, max=22),
)
@click.option(
    "--apply-to-quiescent-prefix",
    "apply",
    is_flag=True,
    help="Replace objects in a prefix whose producers are stopped. Without this flag the job only reports candidates.",
)
def main(
    source_glob: str,
    workers: int,
    worker_cpu: int,
    worker_ram: str,
    worker_disk: str,
    coordinator_cpu: float,
    batch_rows: int,
    row_group_bytes: int,
    compression_level: int,
    apply: bool,
) -> None:
    """Recompress the Parquet objects matching SOURCE_GLOB with zstd."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    context = create_rewrite_context(
        workers=workers,
        worker_cpu=worker_cpu,
        worker_ram=worker_ram,
        worker_disk=worker_disk,
        coordinator_cpu=coordinator_cpu,
    )
    counters_result = run_migration(
        (source_glob,),
        context=context,
        options=RewriteOptions(
            mode=RewriteMode.APPLY if apply else RewriteMode.DRY_RUN,
            batch_rows=batch_rows,
            row_group_bytes=row_group_bytes,
            compression_level=compression_level,
        ),
    )
    for name, value in sorted(counters_result.items()):
        click.echo(f"{name}: {value}")


if __name__ == "__main__":
    main()
