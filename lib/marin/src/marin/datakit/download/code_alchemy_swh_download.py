# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download Code Alchemy blobs from the public Software Heritage S3 bucket.

A single Zephyr task compiles the checked-in Rust downloader on x86_64 Linux and
publishes a content-addressed executable. Download tasks then read balanced
slices of the missing-ID prefix Parquets, reuse that executable within their
process, and publish canonical ``blob_id``/``source_gzip`` Parquet shards.
"""

import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import draccus
import polars as pl
from fray.types import ResourceConfig
from rigging.filesystem.s3_errors import is_transient_s3_error
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.parquet_scan import scan_parquet
from rigging.timing import retry_with_backoff

from marin.utilities.validation_utils import write_provenance_json

logger = logging.getLogger(__name__)

DATASET_ROOT = "s3://marin-us-east-02a/tmp/ttl=30d/code-alchemy-hydration"
DEFAULT_INPUT_PATH = prefix_join(DATASET_ROOT, "missing-ids")
DEFAULT_OUTPUT_PATH = prefix_join(DATASET_ROOT, "downloaded-gzip")
BLOB_ID_COLUMN = "blob_id"
RUST_CONTENT_COLUMN = "content"
SOURCE_GZIP_COLUMN = "source_gzip"
PREFIX_WIDTH = 2
PREFIX_COUNT = 16**PREFIX_WIDTH
DEFAULT_IDS_PER_TASK = 500_000
DEFAULT_METADATA_WORKERS = 32
DEFAULT_MAX_WORKERS = 16
DEFAULT_FLEET = 512
DEFAULT_MAX_SHARD_FAILURES = 5
RUST_BINARY_NAME = "swh_downloader"
RUST_CRATE_DIR = Path(__file__).with_name("code_alchemy_rust_down")
_HEX_PREFIX = re.compile(r"^[0-9a-f]{2}$")
_LOCAL_BINARY_CACHE: dict[str, Path] = {}


@dataclass(frozen=True)
class CodeAlchemySwhDownloadConfig:
    input_path: str = DEFAULT_INPUT_PATH
    output_path: str = DEFAULT_OUTPUT_PATH
    ids_per_task: int = DEFAULT_IDS_PER_TASK
    metadata_workers: int = DEFAULT_METADATA_WORKERS
    max_workers: int = DEFAULT_MAX_WORKERS
    fleet: int = DEFAULT_FLEET
    pipeline: int = 16
    max_attempts: int = 6
    attempt_timeout_seconds: int = 15
    connect_timeout_seconds: int = 5
    tokio_workers: int = 1
    row_group_mb: int = 96
    worker_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=32, ram="160g", disk="60g")
    )
    build_task_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=16, ram="32g", disk="40g")
    )
    download_task_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=16, ram="96g", disk="20g")
    )
    coordinator_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=2, ram="8g", preemptible=False)
    )


@dataclass(frozen=True)
class RustBuildTask:
    crate_fingerprint: str
    binary_path: str
    metadata_path: str


@dataclass(frozen=True)
class RustBuildResult:
    crate_fingerprint: str
    binary_path: str
    version: str
    reused: bool


@dataclass(frozen=True)
class DownloadShardTask:
    prefix: str
    shard_index: int
    input_paths: tuple[str, ...]
    row_start: int
    row_count: int
    output_path: str
    failure_path: str
    metrics_path: str
    binary_path: str
    crate_fingerprint: str
    fleet: int
    pipeline: int
    max_attempts: int
    attempt_timeout_seconds: int
    connect_timeout_seconds: int
    tokio_workers: int
    row_group_mb: int


@dataclass(frozen=True)
class DownloadShardResult:
    prefix: str
    shard_index: int
    input_rows: int
    success: int
    not_found: int
    retryable_failure: int
    permanent_failure: int
    retry_events: int
    bytes: int
    output_path: str
    failure_path: str
    metrics_path: str
    reused: bool = False


def rust_crate_fingerprint(crate_dir: Path = RUST_CRATE_DIR) -> str:
    """Return a stable digest of all build inputs checked into the crate."""

    digest = hashlib.sha256()
    for relative in ("Cargo.toml", "Cargo.lock", "src/main.rs"):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((crate_dir / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def balanced_slices(row_count: int, target_rows: int) -> list[tuple[int, int]]:
    """Split rows into deterministic, near-equal slices no larger than target."""

    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    if target_rows <= 0:
        raise ValueError("target_rows must be positive")
    if row_count == 0:
        return []
    shard_count = (row_count + target_rows - 1) // target_rows
    base, remainder = divmod(row_count, shard_count)
    sizes = [base + (index < remainder) for index in range(shard_count)]
    starts: list[tuple[int, int]] = []
    offset = 0
    for size in sizes:
        starts.append((offset, size))
        offset += size
    return starts


def _prefix_inputs(input_path: str) -> dict[str, tuple[str, ...]]:
    inputs: dict[str, tuple[str, ...]] = {}
    missing_prefixes: list[str] = []
    for prefix_index in range(PREFIX_COUNT):
        prefix = f"{prefix_index:02x}"
        pattern = prefix_join(input_path, f"data/blob_prefix={prefix}/*.parquet")
        matches = retry_with_backoff(
            lambda: tuple(sorted(str(path) for path in StoragePath(pattern).glob())),
            retryable=is_transient_s3_error,
            max_attempts=10,
            operation=f"list missing-ID Parquet files for prefix {prefix}",
        )
        if matches:
            inputs[prefix] = matches
        else:
            missing_prefixes.append(prefix)
    if missing_prefixes:
        raise FileNotFoundError(
            f"Missing-ID stage is incomplete under {input_path!r}: expected all {PREFIX_COUNT} "
            f"prefixes before launch, missing {len(missing_prefixes)} (first: {missing_prefixes[:8]})"
        )
    return inputs


def _scan_paths(paths: tuple[str, ...]) -> pl.LazyFrame:
    frames = [scan_parquet(path) for path in paths]
    return frames[0] if len(frames) == 1 else pl.concat(frames)


def _count_prefix_rows(prefix: str, paths: tuple[str, ...]) -> tuple[str, int]:
    lazy = _scan_paths(paths)
    schema = lazy.collect_schema()
    if BLOB_ID_COLUMN not in schema:
        raise ValueError(f"Missing {BLOB_ID_COLUMN!r} in prefix {prefix}: {paths}")
    if schema[BLOB_ID_COLUMN] != pl.String:
        raise TypeError(f"Expected {BLOB_ID_COLUMN}: Utf8 for prefix {prefix}, got {schema[BLOB_ID_COLUMN]}")
    count = lazy.select(pl.len()).collect(engine="streaming").item()
    return prefix, int(count)


def _validate_config(cfg: CodeAlchemySwhDownloadConfig) -> None:
    positive = {
        "ids_per_task": cfg.ids_per_task,
        "metadata_workers": cfg.metadata_workers,
        "max_workers": cfg.max_workers,
        "fleet": cfg.fleet,
        "pipeline": cfg.pipeline,
        "attempt_timeout_seconds": cfg.attempt_timeout_seconds,
        "connect_timeout_seconds": cfg.connect_timeout_seconds,
        "row_group_mb": cfg.row_group_mb,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if not 1 <= cfg.max_attempts <= 255:
        raise ValueError(f"max_attempts must be in [1, 255], got {cfg.max_attempts}")
    if cfg.tokio_workers < 0:
        raise ValueError(f"tokio_workers must be non-negative, got {cfg.tokio_workers}")


def list_download_shards(cfg: CodeAlchemySwhDownloadConfig) -> list[DownloadShardTask]:
    """Plan balanced, deterministic slices within each canonical prefix."""

    _validate_config(cfg)
    prefix_inputs = _prefix_inputs(cfg.input_path)
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=min(cfg.metadata_workers, len(prefix_inputs))) as executor:
        futures = {
            executor.submit(_count_prefix_rows, prefix, paths): prefix for prefix, paths in prefix_inputs.items()
        }
        for future in as_completed(futures):
            prefix, count = future.result()
            counts[prefix] = count

    fingerprint = rust_crate_fingerprint()
    binary_path = prefix_join(cfg.output_path, f"binary/{fingerprint}/{RUST_BINARY_NAME}")
    tasks: list[DownloadShardTask] = []
    for prefix in sorted(prefix_inputs):
        if _HEX_PREFIX.fullmatch(prefix) is None:
            raise ValueError(f"Invalid blob prefix {prefix!r}")
        slices = balanced_slices(counts[prefix], cfg.ids_per_task) or [(0, 0)]
        for shard_index, (row_start, row_count) in enumerate(slices):
            name = f"part-{shard_index:05d}"
            tasks.append(
                DownloadShardTask(
                    prefix=prefix,
                    shard_index=shard_index,
                    input_paths=prefix_inputs[prefix],
                    row_start=row_start,
                    row_count=row_count,
                    output_path=prefix_join(
                        cfg.output_path, f"data/blob_prefix={prefix}/{name}.parquet"
                    ),
                    failure_path=prefix_join(
                        cfg.output_path, f"failures/blob_prefix={prefix}/{name}.tsv"
                    ),
                    metrics_path=prefix_join(
                        cfg.output_path, f".metrics/blob_prefix={prefix}/{name}.json"
                    ),
                    binary_path=binary_path,
                    crate_fingerprint=fingerprint,
                    fleet=cfg.fleet,
                    pipeline=cfg.pipeline,
                    max_attempts=cfg.max_attempts,
                    attempt_timeout_seconds=cfg.attempt_timeout_seconds,
                    connect_timeout_seconds=cfg.connect_timeout_seconds,
                    tokio_workers=cfg.tokio_workers,
                    row_group_mb=cfg.row_group_mb,
                )
            )
    existing_pattern = prefix_join(cfg.output_path, "data/blob_prefix=*/*.parquet")
    existing_outputs = retry_with_backoff(
        lambda: {str(path) for path in StoragePath(existing_pattern).glob()},
        retryable=is_transient_s3_error,
        max_attempts=10,
        operation=f"list existing Software Heritage outputs under {cfg.output_path}",
    )
    planned_outputs = {task.output_path for task in tasks}
    stale_outputs = sorted(existing_outputs - planned_outputs)
    if stale_outputs:
        raise RuntimeError(
            f"Output root contains {len(stale_outputs)} stale download shards not present in the current plan; "
            f"first stale paths: {stale_outputs[:8]}"
        )
    return tasks


def _read_json(path: str) -> dict[str, object]:
    return json.loads(StoragePath(path).read_text())


def _write_json(path: str, payload: dict[str, object]) -> None:
    StoragePath(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_rust_downloader(task: RustBuildTask) -> RustBuildResult:
    """Compile once on the x86 Zephyr worker and publish a versioned binary."""

    binary = StoragePath(task.binary_path)
    metadata = StoragePath(task.metadata_path)
    if binary.exists() and metadata.exists():
        prior = _read_json(task.metadata_path)
        if prior.get("crate_fingerprint") == task.crate_fingerprint:
            return RustBuildResult(
                crate_fingerprint=task.crate_fingerprint,
                binary_path=task.binary_path,
                version=str(prior["version"]),
                reused=True,
            )

    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64"}:
        raise RuntimeError(f"Rust downloader must be built on x86_64 Linux, got {platform.platform()}")
    if platform.system() != "Linux":
        raise RuntimeError(f"Rust downloader must be built for Linux, got {platform.platform()}")

    manifest = RUST_CRATE_DIR / "Cargo.toml"
    command = ["cargo", "build", "--locked", "--release", "--manifest-path", str(manifest)]
    logger.info("Building Rust downloader once: %s", " ".join(command))
    subprocess.run(command, check=True)
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", RUST_CRATE_DIR / "target"))
    local_binary = target_dir / "release" / RUST_BINARY_NAME
    if not local_binary.is_file():
        raise FileNotFoundError(f"cargo succeeded but did not produce {local_binary}")
    version = subprocess.run(
        [str(local_binary), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    binary.upload_from(str(local_binary))
    payload: dict[str, object] = {
        "status": "complete",
        "crate_fingerprint": task.crate_fingerprint,
        "binary_path": task.binary_path,
        "version": version,
        "target": "x86_64-unknown-linux-gnu",
        "cargo_command": command,
    }
    _write_json(task.metadata_path, payload)
    return RustBuildResult(
        crate_fingerprint=task.crate_fingerprint,
        binary_path=task.binary_path,
        version=version,
        reused=False,
    )


def _local_rust_binary(binary_path: str, fingerprint: str) -> Path:
    # A Zephyr worker executes one shard at a time, so process-local cache access
    # is serial. Keeping a threading.Lock here makes the map function unpicklable.
    cached = _LOCAL_BINARY_CACHE.get(fingerprint)
    if cached is not None and cached.is_file():
        return cached
    cache_dir = Path(tempfile.gettempdir()) / "code-alchemy-swh-downloader" / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / RUST_BINARY_NAME
    if not destination.is_file():
        partial = destination.with_suffix(f".partial-{os.getpid()}")
        StoragePath(binary_path).download_to(str(partial))
        partial.chmod(0o755)
        subprocess.run([str(partial), "--version"], check=True, capture_output=True, text=True)
        os.replace(partial, destination)
    destination.chmod(0o755)
    _LOCAL_BINARY_CACHE[fingerprint] = destination
    return destination


def _task_identity(task: DownloadShardTask) -> dict[str, object]:
    return {
        "prefix": task.prefix,
        "shard_index": task.shard_index,
        "input_paths": list(task.input_paths),
        "row_start": task.row_start,
        "input_rows": task.row_count,
        "output_path": task.output_path,
        "failure_path": task.failure_path,
        "crate_fingerprint": task.crate_fingerprint,
        "binary_path": task.binary_path,
        "fleet": task.fleet,
        "pipeline": task.pipeline,
        "max_attempts": task.max_attempts,
        "attempt_timeout_seconds": task.attempt_timeout_seconds,
        "connect_timeout_seconds": task.connect_timeout_seconds,
        "tokio_workers": task.tokio_workers,
        "row_group_mb": task.row_group_mb,
    }


def _completed_result(
    task: DownloadShardTask,
    *,
    accept_retryable: bool = False,
) -> DownloadShardResult | None:
    metrics = StoragePath(task.metrics_path)
    if not metrics.exists() or not StoragePath(task.output_path).exists() or not StoragePath(task.failure_path).exists():
        return None
    payload = _read_json(task.metrics_path)
    if payload.get("status") != "complete":
        return None
    if any(payload.get(key) != value for key, value in _task_identity(task).items()):
        return None
    if int(payload.get("retryable_failure", 0)) and not accept_retryable:
        return None
    return DownloadShardResult(
        prefix=task.prefix,
        shard_index=task.shard_index,
        input_rows=int(payload["input_rows"]),
        success=int(payload["success"]),
        not_found=int(payload["not_found"]),
        retryable_failure=int(payload["retryable_failure"]),
        permanent_failure=int(payload["permanent_failure"]),
        retry_events=int(payload["retry_events"]),
        bytes=int(payload["bytes"]),
        output_path=task.output_path,
        failure_path=task.failure_path,
        metrics_path=task.metrics_path,
        reused=True,
    )


def download_swh_shard(task: DownloadShardTask) -> DownloadShardResult:
    """Run one bounded Rust process and publish its canonical shard atomically."""

    completed = _completed_result(task)
    if completed is not None:
        return completed
    if _HEX_PREFIX.fullmatch(task.prefix) is None:
        raise ValueError(f"Invalid blob prefix {task.prefix!r}")

    started = time.monotonic()
    binary = _local_rust_binary(task.binary_path, task.crate_fingerprint)
    with tempfile.TemporaryDirectory(prefix=f"swh-{task.prefix}-{task.shard_index:05d}-") as temp:
        temp_path = Path(temp)
        input_dir = temp_path / "input"
        rust_output_dir = temp_path / "rust-output"
        input_dir.mkdir()
        rust_output_dir.mkdir()
        input_file = input_dir / "ids.txt"
        rust_metrics_path = temp_path / "rust-metrics.json"

        ids = (
            _scan_paths(task.input_paths)
            .slice(task.row_start, task.row_count)
            .select(pl.col(BLOB_ID_COLUMN))
            .collect(engine="streaming")
        )
        if ids.height != task.row_count:
            raise ValueError(
                f"Planned {task.row_count} IDs for {task.prefix}/{task.shard_index}, read {ids.height}"
            )
        id_series = ids.get_column(BLOB_ID_COLUMN)
        invalid = id_series.str.contains(r"^[0-9a-f]+$").not_().sum()
        wrong_prefix = (id_series.str.slice(0, PREFIX_WIDTH) != task.prefix).sum()
        if invalid or wrong_prefix:
            raise ValueError(
                f"Invalid IDs in {task.prefix}/{task.shard_index}: invalid={invalid}, wrong_prefix={wrong_prefix}"
            )
        if id_series.n_unique() != task.row_count:
            raise ValueError(f"Duplicate blob_id values in {task.prefix}/{task.shard_index}")
        ids.write_csv(input_file, include_header=False, quote_style="never")

        env = os.environ.copy()
        env.update(
            {
                "INPUT_DIR": str(input_dir),
                "OUTPUT_DIR": str(rust_output_dir),
                "METRICS_PATH": str(rust_metrics_path),
                "ACTIVE_FILES": "1",
                "FLEET": str(task.fleet),
                "PIPELINE": str(task.pipeline),
                "MAX_ATTEMPTS": str(task.max_attempts),
                "ATTEMPT_TIMEOUT_S": str(task.attempt_timeout_seconds),
                "CONNECT_TIMEOUT_S": str(task.connect_timeout_seconds),
                "TOKIO_WORKERS": str(task.tokio_workers),
                "ROW_GROUP_MB": str(task.row_group_mb),
                "SCHEME": "http",
                "BUCKET": "softwareheritage",
            }
        )
        subprocess.run([str(binary)], check=True, env=env)

        rust_metrics = json.loads(rust_metrics_path.read_text())
        classified = sum(
            int(rust_metrics[key])
            for key in ("success", "not_found", "retryable_failure", "permanent_failure")
        )
        if classified != task.row_count or int(rust_metrics["classified"]) != task.row_count:
            raise RuntimeError(
                f"Rust accounting mismatch for {task.prefix}/{task.shard_index}: "
                f"expected={task.row_count}, classified={classified}, metrics={rust_metrics}"
            )

        rust_parquet = rust_output_dir / "ids.parquet"
        normalized_parquet = temp_path / "source-gzip.parquet"
        source = scan_parquet(str(rust_parquet))
        rust_schema = source.collect_schema()
        expected_schema = {BLOB_ID_COLUMN: pl.String, RUST_CONTENT_COLUMN: pl.Binary}
        for column, dtype in expected_schema.items():
            if column not in rust_schema or rust_schema[column] != dtype:
                raise TypeError(f"Unexpected Rust output schema {rust_schema}; expected {expected_schema}")
        source.select(
            pl.col(BLOB_ID_COLUMN).cast(pl.String),
            pl.col(RUST_CONTENT_COLUMN).cast(pl.Binary).alias(SOURCE_GZIP_COLUMN),
        ).sink_parquet(
            normalized_parquet,
            compression="zstd",
            compression_level=3,
            statistics=True,
            mkdir=True,
            engine="streaming",
        )
        output_rows = pl.scan_parquet(normalized_parquet).select(pl.len()).collect(engine="streaming").item()
        if int(output_rows) != int(rust_metrics["success"]):
            raise RuntimeError(
                f"Output row mismatch for {task.prefix}/{task.shard_index}: "
                f"success={rust_metrics['success']}, parquet={output_rows}"
            )

        local_failures = rust_output_dir / "ids.failures.tsv"
        if not local_failures.exists():
            local_failures.touch()
        StoragePath(task.output_path).upload_from(str(normalized_parquet))
        StoragePath(task.failure_path).upload_from(str(local_failures))

    payload = {
        "status": "complete",
        **_task_identity(task),
        "elapsed_seconds": time.monotonic() - started,
        **rust_metrics,
    }
    _write_json(task.metrics_path, payload)
    return DownloadShardResult(
        prefix=task.prefix,
        shard_index=task.shard_index,
        input_rows=task.row_count,
        success=int(rust_metrics["success"]),
        not_found=int(rust_metrics["not_found"]),
        retryable_failure=int(rust_metrics["retryable_failure"]),
        permanent_failure=int(rust_metrics["permanent_failure"]),
        retry_events=int(rust_metrics["retry_events"]),
        bytes=int(rust_metrics["bytes"]),
        output_path=task.output_path,
        failure_path=task.failure_path,
        metrics_path=task.metrics_path,
    )


def _build_pipeline(task: RustBuildTask, output_path: str) -> Dataset[str]:
    return (
        Dataset.from_list([task])
        .map(build_rust_downloader)
        .write_jsonl(prefix_join(output_path, ".metrics/build-execution-{shard:05d}.jsonl"), skip_existing=False)
    )


def _download_pipeline(tasks: list[DownloadShardTask], output_path: str) -> Dataset[str]:
    return (
        Dataset.from_list(tasks)
        .map(download_swh_shard)
        .write_jsonl(
            prefix_join(output_path, ".metrics/download-execution-{shard:05d}-of-{total:05d}.jsonl"),
            skip_existing=False,
        )
    )


def _aggregate_download_results(tasks: list[DownloadShardTask], output_path: str) -> dict[str, int]:
    fields = (
        "input_rows",
        "success",
        "not_found",
        "retryable_failure",
        "permanent_failure",
        "retry_events",
        "bytes",
    )
    totals = {field: 0 for field in fields}
    for task in tasks:
        result = _completed_result(task, accept_retryable=True)
        if result is None:
            raise RuntimeError(f"Download task has no complete durable result: {task.metrics_path}")
        for field in fields:
            totals[field] += int(getattr(result, field))
    classified = (
        totals["success"]
        + totals["not_found"]
        + totals["retryable_failure"]
        + totals["permanent_failure"]
    )
    if classified != totals["input_rows"]:
        raise RuntimeError(f"Aggregate Software Heritage accounting mismatch: {totals}")
    _write_json(prefix_join(output_path, ".metrics/aggregate-download.json"), totals)
    return totals


def download_code_alchemy_swh(cfg: CodeAlchemySwhDownloadConfig) -> None:
    tasks = list_download_shards(cfg)
    fingerprint = tasks[0].crate_fingerprint
    binary_path = tasks[0].binary_path
    build_task = RustBuildTask(
        crate_fingerprint=fingerprint,
        binary_path=binary_path,
        metadata_path=prefix_join(cfg.output_path, f"binary/{fingerprint}/build.json"),
    )
    logger.info(
        "Downloading %d missing IDs in %d balanced shards across %d prefixes with up to %d workers",
        sum(task.row_count for task in tasks),
        len(tasks),
        len({task.prefix for task in tasks}),
        min(cfg.max_workers, len(tasks)),
    )

    context = ZephyrContext(
        name="code-alchemy-swh-download",
        resources=cfg.worker_resources,
        coordinator_resources=cfg.coordinator_resources,
        max_workers=cfg.max_workers,
        max_shard_failures=DEFAULT_MAX_SHARD_FAILURES,
        max_execution_retries=10,
    )
    with context:
        context.execute(_build_pipeline(build_task, cfg.output_path), map_task_resources=cfg.build_task_resources)
        context.execute(
            _download_pipeline(tasks, cfg.output_path),
            map_task_resources=cfg.download_task_resources,
        )
    totals = _aggregate_download_results(tasks, cfg.output_path)
    write_provenance_json(
        cfg.output_path,
        metadata={
            "input_path": cfg.input_path,
            "input_rows": sum(task.row_count for task in tasks),
            "task_count": len(tasks),
            "prefix_count": len({task.prefix for task in tasks}),
            "ids_per_task_target": cfg.ids_per_task,
            "binary_path": binary_path,
            "crate_fingerprint": fingerprint,
            "output_columns": [BLOB_ID_COLUMN, SOURCE_GZIP_COLUMN],
            "failure_classes": ["not_found", "retryable_failure", "permanent_failure"],
            "download_totals": totals,
            "download_task_config": {
                key: value
                for key, value in asdict(cfg).items()
                if key
                not in {
                    "worker_resources",
                    "build_task_resources",
                    "download_task_resources",
                    "coordinator_resources",
                }
            },
        },
    )
    unresolved = totals["not_found"] + totals["retryable_failure"] + totals["permanent_failure"]
    if unresolved:
        raise RuntimeError(
            f"Software Heritage recovery left {unresolved} blob IDs unresolved; "
            f"not_found={totals['not_found']}, retryable_failure={totals['retryable_failure']}, "
            f"permanent_failure={totals['permanent_failure']}"
        )


@draccus.wrap()
def main(cfg: CodeAlchemySwhDownloadConfig) -> None:
    configure_logging(level=logging.INFO)
    download_code_alchemy_swh(cfg)


if __name__ == "__main__":
    main()
