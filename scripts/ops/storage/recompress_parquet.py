#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite Parquet objects in place with zstd compression through Zephyr.

Each Zephyr task handles one object. It writes and validates a temporary sibling
before replacing the source, so a failed or interrupted task leaves the source
intact. Reruns skip objects whose columns use zstd and have page indexes.

Run the coordinator on a cluster local to the bucket. Start with a narrow glob
and omit ``--apply`` to inventory candidates before rewriting them::

    uv run iris --cluster=marin job run \
        --cpu 1 --memory 2GB --target-cluster cw-us-east-02a -- \
        python scripts/ops/storage/recompress_parquet.py \
        's3://marin-us-east-02a/marin/normalized/example/**/*.parquet' \
        --workers 16

Add ``--apply`` after reviewing the dry-run counters. Do not target lifecycle
prefixes: the command rejects ``tmp/ttl=`` paths because replacing an object
would restart its retention clock.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

import click
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import StoreType, data_buckets
from rigging.filesystem.s3_compat import configure_fsspec_s3, credentials_hint, s3_credentials, s3_endpoint
from rigging.filesystem.storage_path import StoragePath
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.writers import (
    DEFAULT_PARQUET_COMPRESSION,
    DEFAULT_PARQUET_COMPRESSION_LEVEL,
    DEFAULT_PARQUET_WRITE_PAGE_INDEX,
)

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = 16
DEFAULT_BATCH_ROWS = 8_192
DEFAULT_WORKER_CPU = 2
DEFAULT_WORKER_RAM = "8g"
COUNTER_PREFIX = "parquet_recompression"


class RewriteDisposition(StrEnum):
    """Outcome of inspecting or rewriting one Parquet object."""

    REWRITTEN = "rewritten"
    ALREADY_ZSTD = "already_zstd"
    DRY_RUN = "dry_run"
    NOT_SMALLER = "not_smaller"


@dataclass(frozen=True)
class RewriteOptions:
    """Controls one bounded Parquet rewrite."""

    apply: bool = False
    batch_rows: int = DEFAULT_BATCH_ROWS
    compression_level: int = DEFAULT_PARQUET_COMPRESSION_LEVEL


@dataclass(frozen=True)
class RewriteResult:
    """Observable outcome for one Parquet object."""

    path: str
    disposition: RewriteDisposition
    input_bytes: int
    output_bytes: int | None
    rows: int


@dataclass(frozen=True)
class ObjectFingerprint:
    """Fields that change when an object is replaced during a rewrite."""

    size: int
    etag: str | None
    version: str | None
    modified: str | None


class _OutputNotSmaller(Exception):
    pass


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
        metadata.row_group(row_group).column(column).has_column_index
        and metadata.row_group(row_group).column(column).has_offset_index
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
    if not _has_page_indexes(rewritten.metadata):
        raise ValueError("rewritten Parquet is missing page indexes")


def recompress_parquet(path: str, options: RewriteOptions = RewriteOptions()) -> RewriteResult:
    """Rewrite one Parquet object to zstd with page indexes.

    The source is replaced only after the temporary output has the same schema
    and row count, uses zstd for every column chunk, and the source fingerprint
    still matches the value observed before reading it. Non-zstd sources are
    preserved when the rewritten object would not be smaller.
    """
    if "/tmp/ttl=" in path:
        raise ValueError(f"refusing to reset lifecycle retention by replacing {path}")
    if options.batch_rows <= 0:
        raise ValueError(f"batch_rows must be positive, got {options.batch_rows}")

    fs, fs_path = filesystem_for(path)
    source_fingerprint = _fingerprint(fs.info(fs_path))

    with fs.open(fs_path, "rb") as source_handle:
        source = pq.ParquetFile(source_handle)
        rows = source.metadata.num_rows
        source_is_zstd = _column_compressions(source.metadata) == {DEFAULT_PARQUET_COMPRESSION.upper()}
        if source_is_zstd and _has_page_indexes(source.metadata):
            return RewriteResult(path, RewriteDisposition.ALREADY_ZSTD, source_fingerprint.size, None, rows)
        if not options.apply:
            return RewriteResult(path, RewriteDisposition.DRY_RUN, source_fingerprint.size, None, rows)

        try:
            with atomic_rename(fs_path, filesystem=fs) as temp_path:
                with fs.open(temp_path, "wb") as output_handle:
                    with pq.ParquetWriter(
                        output_handle,
                        source.schema_arrow,
                        compression=DEFAULT_PARQUET_COMPRESSION,
                        compression_level=options.compression_level,
                        write_page_index=DEFAULT_PARQUET_WRITE_PAGE_INDEX,
                    ) as writer:
                        for batch in source.iter_batches(batch_size=options.batch_rows):
                            writer.write_batch(batch)

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
    result = recompress_parquet(path, options)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/files_{result.disposition}", 1)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/input_bytes_{result.disposition}", result.input_bytes)
    if result.output_bytes is not None:
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/output_bytes_{result.disposition}", result.output_bytes)
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/rows_{result.disposition}", result.rows)
    logger.info(
        "%s: %s (%d -> %s bytes)",
        result.disposition,
        path,
        result.input_bytes,
        result.output_bytes,
    )
    return []


def _configure_source_filesystem(source_glob: str) -> None:
    parsed = StoragePath(source_glob)
    if parsed.scheme != "s3":
        return
    bucket = data_buckets().get(parsed.bucket)
    if bucket is None or bucket.store not in (StoreType.COREWEAVE, StoreType.R2):
        return
    credentials = s3_credentials(bucket.store)
    if credentials is None:
        raise ValueError(f"cannot list {source_glob}: {credentials_hint(bucket.store)}")
    key, secret = credentials
    configure_fsspec_s3(s3_endpoint(bucket.store), key=key, secret=secret)


def run_migration(
    source_glob: str,
    *,
    workers: int,
    worker_cpu: int,
    worker_ram: str,
    options: RewriteOptions,
) -> dict[str, int | float]:
    """Run the bounded recompression pipeline and return aggregate counters."""
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")

    _configure_source_filesystem(source_glob)
    pipeline = Dataset.from_files(source_glob).flat_map(partial(_rewrite_for_zephyr, options=options))
    context = ZephyrContext(
        name="recompress-parquet",
        resources=ResourceConfig(cpu=worker_cpu, ram=worker_ram),
        max_workers=workers,
    )
    outcome = context.execute(pipeline, verbose=True)
    return dict(outcome.counters)


@click.command()
@click.argument("source_glob")
@click.option("--workers", default=DEFAULT_WORKERS, show_default=True, type=click.IntRange(min=1))
@click.option("--worker-cpu", default=DEFAULT_WORKER_CPU, show_default=True, type=click.IntRange(min=1))
@click.option("--worker-ram", default=DEFAULT_WORKER_RAM, show_default=True)
@click.option("--batch-rows", default=DEFAULT_BATCH_ROWS, show_default=True, type=click.IntRange(min=1))
@click.option(
    "--compression-level",
    default=DEFAULT_PARQUET_COMPRESSION_LEVEL,
    show_default=True,
    type=click.IntRange(min=1, max=22),
)
@click.option(
    "--apply",
    is_flag=True,
    help="Replace source objects after validation. Without this flag the job only reports candidates.",
)
def main(
    source_glob: str,
    workers: int,
    worker_cpu: int,
    worker_ram: str,
    batch_rows: int,
    compression_level: int,
    apply: bool,
) -> None:
    """Recompress the Parquet objects matching SOURCE_GLOB with zstd."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    counters_result = run_migration(
        source_glob,
        workers=workers,
        worker_cpu=worker_cpu,
        worker_ram=worker_ram,
        options=RewriteOptions(apply=apply, batch_rows=batch_rows, compression_level=compression_level),
    )
    for name, value in sorted(counters_result.items()):
        click.echo(f"{name}: {value}")


if __name__ == "__main__":
    main()
