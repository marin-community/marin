# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decompress partitioned Code Alchemy source payloads.

Each Zephyr task owns one ``blob_prefix`` partition and reserves an entire
CoreWeave Genoa CPU node. Inside the task, a bounded thread pool decompresses
contiguous batches. CPython's gzip/zlib implementation releases the GIL while
inflating, so the native DEFLATE work can use every allocated core without
multiprocessing or one future per row.

Malformed gzip members are omitted and counted. Source bytes that are not
strict UTF-8 are decoded with ``errors="ignore"`` to match Code Alchemy's
official hydration recipe, with every lossy row counted.
"""

import gzip
import json
import logging
import os
import re
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

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

from marin.datakit.download.huggingface import DOWNLOAD_SUCCESS_METRICS_TEMPLATE
from marin.utilities.validation_utils import write_provenance_json

logger = logging.getLogger(__name__)

HYDRATION_ROOT = "s3://marin-us-east-02a/tmp/ttl=30d/code-alchemy-hydration"
DEFAULT_INPUT_PATH = prefix_join(HYDRATION_ROOT, "downloaded-gzip")
DEFAULT_OUTPUT_PATH = prefix_join(HYDRATION_ROOT, "downloaded-sources")
BLOB_PREFIX_HEX_WIDTH = 2
BLOB_PREFIX_PARTITION_COUNT = 16**BLOB_PREFIX_HEX_WIDTH
DEFAULT_THREAD_COUNT = 192
# One 180-vCPU task saturates the usable cores of a Genoa node while leaving
# scheduler and Kubernetes headroom. Prefixes run sequentially on that worker.
DEFAULT_MAX_WORKERS = 1
DEFAULT_MAX_SHARD_FAILURES = 5
OBJECT_STORE_MAX_RETRIES = 10

_HEX_PREFIX = re.compile(rf"^[0-9a-f]{{{BLOB_PREFIX_HEX_WIDTH}}}$")


@dataclass(frozen=True)
class CodeAlchemyDecompressConfig:
    """Configuration for full-node, prefix-partitioned source decompression."""

    input_path: str = DEFAULT_INPUT_PATH
    output_path: str = DEFAULT_OUTPUT_PATH
    thread_count: int = DEFAULT_THREAD_COUNT
    max_workers: int = DEFAULT_MAX_WORKERS
    worker_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=180, ram="1200g", disk="1000g")
    )
    task_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=180, ram="1200g", disk="1000g")
    )
    coordinator_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=2, ram="8g", preemptible=False)
    )


@dataclass(frozen=True)
class DecompressPrefixTask:
    """All compressed Parquets and the canonical output for one prefix."""

    prefix: str
    input_paths: tuple[str, ...]
    output_path: str
    thread_count: int


@dataclass(frozen=True)
class DecompressionMetrics:
    """Exhaustive outcome counts for one decompression batch or prefix."""

    input_rows: int
    decoded_rows: int
    corrupt_gzip_rows: int
    lossy_utf8_rows: int


@dataclass(frozen=True)
class DecompressPrefixResult:
    prefix: str
    input_file_count: int
    output_path: str
    input_rows: int
    decoded_rows: int
    corrupt_gzip_rows: int
    lossy_utf8_rows: int


@dataclass(frozen=True)
class _ChunkResult:
    rows: list[tuple[str, str]]
    corrupt_gzip_rows: int
    lossy_utf8_rows: int


def _balanced_ranges(row_count: int, worker_count: int) -> list[tuple[int, int]]:
    """Return deterministic contiguous ranges with sizes differing by at most one."""

    active_workers = min(row_count, worker_count)
    if active_workers == 0:
        return []
    quotient, remainder = divmod(row_count, active_workers)
    ranges = []
    start = 0
    for worker_index in range(active_workers):
        end = start + quotient + (worker_index < remainder)
        ranges.append((start, end))
        start = end
    return ranges


def _decode_chunk(item: tuple[tuple[int, int], list[str], list[bytes]]) -> _ChunkResult:
    (start, end), blob_ids, payloads = item
    rows: list[tuple[str, str]] = []
    corrupt_gzip_rows = 0
    lossy_utf8_rows = 0
    for row_index in range(start, end):
        if payloads[row_index][:2] != b"\x1f\x8b":
            corrupt_gzip_rows += 1
            continue
        try:
            source_bytes = gzip.decompress(payloads[row_index])
        except (OSError, EOFError, zlib.error):
            corrupt_gzip_rows += 1
            continue
        try:
            source = source_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            lossy_utf8_rows += 1
            source = source_bytes.decode("utf-8", errors="ignore")
        rows.append((blob_ids[row_index], source))
    return _ChunkResult(
        rows=rows,
        corrupt_gzip_rows=corrupt_gzip_rows,
        lossy_utf8_rows=lossy_utf8_rows,
    )


def _validate_input_frame(frame: pl.DataFrame) -> None:
    required_schema = {"blob_id": pl.String, "source_gzip": pl.Binary}
    missing = required_schema.keys() - frame.schema.keys()
    if missing:
        raise ValueError(f"Compressed source input is missing columns: {sorted(missing)}")
    wrong_types = {
        name: (frame.schema[name], dtype)
        for name, dtype in required_schema.items()
        if frame.schema[name] != dtype
    }
    if wrong_types:
        details = ", ".join(
            f"{name}={actual!r} (expected {expected!r})" for name, (actual, expected) in wrong_types.items()
        )
        raise TypeError(f"Compressed source input has non-canonical types: {details}")
    null_counts = frame.select(pl.col("blob_id", "source_gzip").null_count()).row(0)
    if any(null_counts):
        raise ValueError(
            "Compressed source input contains nulls: "
            f"blob_id={null_counts[0]}, source_gzip={null_counts[1]}"
        )
    if frame.get_column("blob_id").n_unique() != frame.height:
        raise ValueError("Compressed source input contains duplicate blob_id values")


def decompress_gzip_batch(
    frame: pl.DataFrame,
    *,
    thread_count: int | None = None,
) -> tuple[pl.DataFrame, DecompressionMetrics]:
    """Deterministically decode one in-memory batch with bounded native threads.

    Results preserve input order among successfully decoded rows. Work is
    submitted once per contiguous core-sized chunk, not once per payload.
    """

    if thread_count is None:
        thread_count = os.cpu_count() or 1
    if thread_count <= 0:
        raise ValueError(f"thread_count must be positive, got {thread_count}")

    _validate_input_frame(frame)
    selected = frame.select("blob_id", "source_gzip")
    blob_ids = selected.get_column("blob_id").to_list()
    payloads = selected.get_column("source_gzip").to_list()
    ranges = _balanced_ranges(frame.height, thread_count)

    if ranges:
        items = [(row_range, blob_ids, payloads) for row_range in ranges]
        with ThreadPoolExecutor(max_workers=len(ranges), thread_name_prefix="gzip-decode") as pool:
            chunk_results = list(pool.map(_decode_chunk, items))
    else:
        chunk_results = []

    decoded_rows = [row for chunk in chunk_results for row in chunk.rows]
    output = pl.DataFrame(
        {
            "blob_id": pl.Series((row[0] for row in decoded_rows), dtype=pl.String),
            "source": pl.Series((row[1] for row in decoded_rows), dtype=pl.String),
        }
    )
    metrics = DecompressionMetrics(
        input_rows=frame.height,
        decoded_rows=output.height,
        corrupt_gzip_rows=sum(chunk.corrupt_gzip_rows for chunk in chunk_results),
        lossy_utf8_rows=sum(chunk.lossy_utf8_rows for chunk in chunk_results),
    )
    if metrics.input_rows != metrics.decoded_rows + metrics.corrupt_gzip_rows:
        raise AssertionError("Decompression outcome counts do not cover every input row")
    return output, metrics


def list_decompress_prefix_tasks(cfg: CodeAlchemyDecompressConfig) -> list[DecompressPrefixTask]:
    """Discover exactly one complete input and output task per hexadecimal prefix."""

    if cfg.thread_count <= 0:
        raise ValueError(f"thread_count must be positive, got {cfg.thread_count}")
    if cfg.max_workers <= 0:
        raise ValueError(f"max_workers must be positive, got {cfg.max_workers}")

    tasks = []
    missing_prefixes = []
    for prefix_index in range(BLOB_PREFIX_PARTITION_COUNT):
        prefix = f"{prefix_index:0{BLOB_PREFIX_HEX_WIDTH}x}"
        input_directory = prefix_join(prefix_join(cfg.input_path, "data"), f"blob_prefix={prefix}")
        pattern = StoragePath(input_directory) / "*.parquet"
        input_paths = retry_with_backoff(
            lambda: tuple(sorted(str(path) for path in pattern.glob())),
            retryable=is_transient_s3_error,
            max_attempts=10,
            operation=f"list compressed source Parquet files for prefix {prefix}",
        )
        if not input_paths:
            missing_prefixes.append(prefix)
            continue
        tasks.append(
            DecompressPrefixTask(
                prefix=prefix,
                input_paths=input_paths,
                output_path=prefix_join(
                    prefix_join(prefix_join(cfg.output_path, "data"), f"blob_prefix={prefix}"),
                    "part-00000.parquet",
                ),
                thread_count=cfg.thread_count,
            )
        )

    if missing_prefixes:
        shown = ", ".join(missing_prefixes[:16])
        suffix = "..." if len(missing_prefixes) > 16 else ""
        raise FileNotFoundError(
            f"Missing compressed Parquet input for {len(missing_prefixes)} blob prefixes: {shown}{suffix}"
        )
    return tasks


def decompress_prefix_task(task: DecompressPrefixTask) -> DecompressPrefixResult:
    """Read, validate, decode, and write one canonical prefix partition."""

    if _HEX_PREFIX.fullmatch(task.prefix) is None:
        raise ValueError(f"Invalid lowercase hexadecimal blob prefix: {task.prefix!r}")
    if not task.input_paths:
        raise ValueError(f"Prefix {task.prefix} has no compressed input Parquets")

    compressed = pl.concat([scan_parquet(path).select("blob_id", "source_gzip") for path in task.input_paths]).collect(
        engine="streaming"
    )
    _validate_input_frame(compressed)
    mismatched_prefix_rows = compressed.select(
        (pl.col("blob_id").str.slice(0, BLOB_PREFIX_HEX_WIDTH).str.to_lowercase() != task.prefix).sum()
    ).item()
    if mismatched_prefix_rows:
        raise ValueError(
            f"Prefix {task.prefix} contains {mismatched_prefix_rows} blob_id values from another partition"
        )

    sources, metrics = decompress_gzip_batch(compressed, thread_count=task.thread_count)
    storage_options: dict[str, object] = dict(storage_options_for_path(task.output_path) or {})
    storage_options["max_retries"] = OBJECT_STORE_MAX_RETRIES
    sources.lazy().sink_parquet(
        task.output_path,
        compression="zstd",
        compression_level=3,
        statistics=True,
        mkdir=True,
        engine="streaming",
        storage_options=storage_options,
    )
    logger.info(
        "Decompressed prefix %s: input=%d decoded=%d corrupt_gzip=%d lossy_utf8=%d",
        task.prefix,
        metrics.input_rows,
        metrics.decoded_rows,
        metrics.corrupt_gzip_rows,
        metrics.lossy_utf8_rows,
    )
    result = DecompressPrefixResult(
        prefix=task.prefix,
        input_file_count=len(task.input_paths),
        output_path=task.output_path,
        input_rows=metrics.input_rows,
        decoded_rows=metrics.decoded_rows,
        corrupt_gzip_rows=metrics.corrupt_gzip_rows,
        lossy_utf8_rows=metrics.lossy_utf8_rows,
    )
    if metrics.corrupt_gzip_rows:
        raise RuntimeError(
            f"Strict gzip decompression failed for prefix {task.prefix}: "
            f"corrupt_gzip={metrics.corrupt_gzip_rows}"
        )
    return result


def build_decompression_pipeline(tasks: list[DecompressPrefixTask], output_path: str) -> Dataset[str]:
    """Build a resumable pipeline whose success files are per-prefix completion markers."""

    return (
        Dataset.from_list(tasks)
        .map(decompress_prefix_task)
        .write_jsonl(
            prefix_join(output_path, DOWNLOAD_SUCCESS_METRICS_TEMPLATE),
            skip_existing=True,
        )
    )


def _aggregate_decompression_results(output_path: str) -> dict[str, int]:
    pattern = prefix_join(output_path, ".metrics/success-part-*.jsonl")
    paths = sorted(StoragePath(pattern).glob(), key=str)
    records = [
        json.loads(line)
        for path in paths
        for line in path.read_text().splitlines()
        if line
    ]
    if len(records) != BLOB_PREFIX_PARTITION_COUNT:
        raise RuntimeError(
            f"Expected {BLOB_PREFIX_PARTITION_COUNT} decompression results, found {len(records)}"
        )
    fields = ("input_rows", "decoded_rows", "corrupt_gzip_rows", "lossy_utf8_rows")
    totals = {field: sum(int(record[field]) for record in records) for field in fields}
    if totals["input_rows"] != totals["decoded_rows"] + totals["corrupt_gzip_rows"]:
        raise RuntimeError(f"Aggregate decompression accounting mismatch: {totals}")
    return totals


def decompress_code_alchemy_sources(cfg: CodeAlchemyDecompressConfig) -> None:
    """Run one full-node task per prefix and record stage provenance."""

    tasks = list_decompress_prefix_tasks(cfg)
    logger.info(
        "Decompressing %d blob-prefix partitions with up to %d full-node workers, %d threads per task",
        len(tasks),
        min(cfg.max_workers, len(tasks)),
        cfg.thread_count,
    )
    context = ZephyrContext(
        name="code-alchemy-decompress",
        resources=cfg.worker_resources,
        coordinator_resources=cfg.coordinator_resources,
        max_workers=cfg.max_workers,
        max_shard_failures=DEFAULT_MAX_SHARD_FAILURES,
        max_execution_retries=10,
    )
    with context:
        context.execute(
            build_decompression_pipeline(tasks, cfg.output_path),
            map_task_resources=cfg.task_resources,
        )

    totals = _aggregate_decompression_results(cfg.output_path)

    write_provenance_json(
        cfg.output_path,
        metadata={
            "input_path": cfg.input_path,
            "input_columns": ["blob_id", "source_gzip"],
            "output_columns": ["blob_id", "source"],
            "partition_key": f"lower(blob_id[:{BLOB_PREFIX_HEX_WIDTH}])",
            "partition_count": BLOB_PREFIX_PARTITION_COUNT,
            "thread_count_per_task": cfg.thread_count,
            "max_full_node_workers": cfg.max_workers,
            "task_cpu": cfg.task_resources.cpu,
            "task_memory": cfg.task_resources.ram,
            "task_disk": cfg.task_resources.disk,
            "parallelism": "one bounded gzip/zlib thread per allocated CPU; contiguous chunks; no multiprocessing",
            "corrupt_gzip_behavior": "omit row and increment corrupt_gzip_rows",
            "utf8_behavior": 'decode with errors="ignore" per the Code Alchemy dataset card; count lossy_utf8_rows',
            "duplicate_blob_id_behavior": "fail prefix task",
            "metrics_path": prefix_join(cfg.output_path, ".metrics"),
            "decompression_totals": totals,
        },
    )


@draccus.wrap()
def main(cfg: CodeAlchemyDecompressConfig) -> None:
    configure_logging(level=logging.INFO)
    decompress_code_alchemy_sources(cfg)


if __name__ == "__main__":
    main()
