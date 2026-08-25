# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream Hugging Face parquet shards into partitions keyed by ``blob_id``."""

import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import polars as pl
from fray.types import ResourceConfig
from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url
from polars.io.partition import FileProviderArgs
from rigging.filesystem.storage_path import prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.parquet_scan import storage_options_for_path

from marin.datakit.download.huggingface import (
    DOWNLOAD_SUCCESS_METRICS_TEMPLATE,
    HF_PROTOCOL_PREFIX,
    _relative_path_in_source,
    list_hf_repo_files,
)
from marin.utilities.validation_utils import write_provenance_json

logger = logging.getLogger(__name__)

BLOB_PREFIX_COLUMN = "__blob_prefix"
BLOB_PREFIX_HEX_WIDTH = 2
BLOB_PREFIX_PARTITION_COUNT = 16**BLOB_PREFIX_HEX_WIDTH
BLOB_NULL_PARTITION = "null"
OBJECT_STORE_MAX_RETRIES = 10
DEFAULT_MAX_WORKERS = 128
DEFAULT_MAX_SHARD_FAILURES = 5

_HEX_PREFIX = re.compile(rf"^[0-9a-f]{{{BLOB_PREFIX_HEX_WIDTH}}}$")


@dataclass(frozen=True)
class BlobIdDownloadConfig:
    """Shared resource and execution configuration for blob-partition downloads."""

    output_path: str
    revision: str
    max_workers: int = DEFAULT_MAX_WORKERS
    max_files: int | None = None
    worker_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=16, ram="80g", disk="16g"))
    task_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=4, ram="16g", disk="4g"))
    coordinator_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=1, ram="8g", preemptible=False)
    )


@dataclass(frozen=True)
class BlobIdParquetTask:
    """One Hugging Face parquet file and its partitioned destination."""

    source_url: str
    relative_source_path: str
    output_path: str
    dataset_id: str
    revision: str
    require_token: bool = False

    @property
    def source_id(self) -> str:
        return hashlib.sha256(self.relative_source_path.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class BlobIdDownloadResult:
    """Completion record emitted after one source parquet is fully partitioned."""

    source_url: str
    relative_source_path: str


@dataclass(frozen=True)
class _BlobPartitionPathProvider:
    """Create collision-free output names for one source parquet file."""

    source_id: str

    def __call__(self, args: FileProviderArgs) -> str:
        prefix = args.partition_keys.item(0, 0)
        if prefix is None:
            prefix = BLOB_NULL_PARTITION
        elif not isinstance(prefix, str) or _HEX_PREFIX.fullmatch(prefix) is None:
            raise ValueError(f"blob_id must start with {BLOB_PREFIX_HEX_WIDTH} hexadecimal characters, got {prefix!r}")
        return f"blob_prefix={prefix}/source-{self.source_id}-{args.index_in_partition:05d}.parquet"


def hf_token(dataset_id: str, *, required: bool) -> str | None:
    """Return the configured Hugging Face token, enforcing gated access when needed."""

    token = get_token()
    if required and not token:
        raise RuntimeError(
            f"{dataset_id} is gated; set HF_TOKEN in the driver and Iris worker environment "
            "after accepting the dataset terms on Hugging Face"
        )
    return token


def resolve_source_url(task: BlobIdParquetTask) -> str:
    """Resolve one ``hf://`` path to a signed CDN URL."""

    if not task.source_url.startswith(HF_PROTOCOL_PREFIX):
        return task.source_url
    resolve_url = hf_hub_url(
        repo_id=task.dataset_id,
        filename=task.relative_source_path,
        repo_type="dataset",
        revision=task.revision,
    )
    metadata = get_hf_file_metadata(
        resolve_url,
        token=hf_token(task.dataset_id, required=task.require_token),
        retry_on_errors=True,
    )
    return metadata.location


def _input_storage_options(source_url: str) -> dict[str, object] | None:
    if source_url.startswith(("http://", "https://")):
        return {"max_retries": OBJECT_STORE_MAX_RETRIES}
    return None


def list_blob_id_parquet_tasks(
    cfg: BlobIdDownloadConfig,
    *,
    dataset_id: str,
    parquet_glob: str,
    require_token: bool,
    output_path_for_relative: Callable[[str], str],
) -> list[BlobIdParquetTask]:
    """List pinned parquet shards and validate content access before worker startup."""

    listing = list_hf_repo_files(
        hf_dataset_id=dataset_id,
        revision=cfg.revision,
        hf_urls_glob=[parquet_glob],
        token=hf_token(dataset_id, required=require_token),
    )
    files = sorted(listing.files)
    if cfg.max_files is not None:
        files = files[: cfg.max_files]
    if not files:
        raise ValueError(f"No parquet files matched {dataset_id}@{cfg.revision}:{parquet_glob}")

    tasks = []
    for file in files:
        relative_source_path = _relative_path_in_source(file, listing.source_root)
        tasks.append(
            BlobIdParquetTask(
                source_url=f"{HF_PROTOCOL_PREFIX}{file}",
                relative_source_path=relative_source_path,
                output_path=output_path_for_relative(relative_source_path),
                dataset_id=dataset_id,
                revision=cfg.revision,
                require_token=require_token,
            )
        )
    resolve_source_url(tasks[0])
    return tasks


def partition_parquet_by_blob_id(task: BlobIdParquetTask) -> BlobIdDownloadResult:
    """Stream one remote parquet file into partitions keyed by ``blob_id[:2]``."""

    resolved_source_url = resolve_source_url(task)
    source = pl.scan_parquet(
        resolved_source_url,
        storage_options=_input_storage_options(resolved_source_url),
        cache=False,
        low_memory=True,
        parallel="row_groups",
    ).with_columns(pl.col("blob_id").str.slice(0, BLOB_PREFIX_HEX_WIDTH).str.to_lowercase().alias(BLOB_PREFIX_COLUMN))
    output_storage_options: dict[str, object] = dict(storage_options_for_path(task.output_path) or {})
    output_storage_options["max_retries"] = OBJECT_STORE_MAX_RETRIES

    source.sink_parquet(
        pl.PartitionBy(
            task.output_path,
            key=BLOB_PREFIX_COLUMN,
            include_key=False,
            file_path_provider=_BlobPartitionPathProvider(task.source_id),
        ),
        compression="zstd",
        statistics=True,
        mkdir=True,
        engine="streaming",
        storage_options=output_storage_options,
    )
    logger.info("Partitioned %s into %s", task.source_url, task.output_path)
    return BlobIdDownloadResult(
        source_url=task.source_url,
        relative_source_path=task.relative_source_path,
    )


def build_blob_id_partition_pipeline(tasks: list[BlobIdParquetTask], output_path: str) -> Dataset[str]:
    """Build a one-source-file-per-shard Zephyr pipeline."""

    return (
        Dataset.from_list(tasks)
        .map(partition_parquet_by_blob_id)
        .write_jsonl(
            prefix_join(output_path, DOWNLOAD_SUCCESS_METRICS_TEMPLATE),
            skip_existing=True,
        )
    )


def execute_blob_id_partition_download(
    tasks: list[BlobIdParquetTask],
    cfg: BlobIdDownloadConfig,
    *,
    job_name: str,
    provenance_metadata: dict[str, object],
) -> None:
    """Execute a partition pipeline and record its pinned source provenance."""

    pipeline = build_blob_id_partition_pipeline(tasks, cfg.output_path)
    ctx = ZephyrContext(
        name=job_name,
        resources=cfg.worker_resources,
        coordinator_resources=cfg.coordinator_resources,
        max_workers=min(cfg.max_workers, len(tasks)),
        max_shard_failures=DEFAULT_MAX_SHARD_FAILURES,
        max_execution_retries=10,
    )
    ctx.execute(pipeline, map_task_resources=cfg.task_resources)
    write_provenance_json(cfg.output_path, metadata=provenance_metadata)
