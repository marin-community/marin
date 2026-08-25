# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream The Stack v2 parquet shards from Hugging Face into blob-prefix partitions.

The pipeline has one Zephyr source shard per Hugging Face parquet file. Each worker
uses Polars' native ``hf://`` range reader and streaming parquet sink, so the source
parquet is never materialized on worker disk. This deliberately avoids ``hf_xet``'s
file downloader, which reconstructs a complete local file before a transform can run.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field

import draccus
import polars as pl
from fray.types import ResourceConfig
from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url
from polars.io.partition import FileProviderArgs
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging
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

HF_DATASET_ID = "bigcode/the-stack-v2"
HF_REVISION = "e565caa3a78c2423bd374333a472b049eb090e47"
HF_PARQUET_GLOB = "data/*/*.parquet"
BLOB_PREFIX_COLUMN = "__blob_prefix"
BLOB_PREFIX_HEX_WIDTH = 2
BLOB_PREFIX_PARTITION_COUNT = 16**BLOB_PREFIX_HEX_WIDTH
DEFAULT_MAX_WORKERS = 128
DEFAULT_MAX_SHARD_FAILURES = 5
OBJECT_STORE_MAX_RETRIES = 10

_HEX_PREFIX = re.compile(rf"^[0-9a-f]{{{BLOB_PREFIX_HEX_WIDTH}}}$")


@dataclass(frozen=True)
class StackV2DownloadConfig:
    """Configuration for the direct Hugging Face-to-object-store transfer."""

    output_path: str = field(default_factory=lambda: marin_temp_bucket(ttl_days=30, prefix="stack-v2"))
    revision: str = HF_REVISION
    max_workers: int = DEFAULT_MAX_WORKERS
    max_files: int | None = None
    worker_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=16, ram="80g", disk="16g"))
    task_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=4, ram="16g", disk="4g"))
    coordinator_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=1, ram="8g", preemptible=False)
    )


@dataclass(frozen=True)
class StackV2ParquetTask:
    """One Hugging Face parquet file and its partitioned destination."""

    source_url: str
    relative_source_path: str
    output_path: str
    revision: str = HF_REVISION

    @property
    def source_id(self) -> str:
        return hashlib.sha256(self.relative_source_path.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class StackV2DownloadResult:
    """Completion record emitted after one source parquet is fully partitioned."""

    source_url: str
    relative_source_path: str


@dataclass(frozen=True)
class _BlobPartitionPathProvider:
    """Create collision-free output names for one source parquet file."""

    source_id: str

    def __call__(self, args: FileProviderArgs) -> str:
        prefix = args.partition_keys.item(0, 0)
        if not isinstance(prefix, str) or _HEX_PREFIX.fullmatch(prefix) is None:
            raise ValueError(f"blob_id must start with {BLOB_PREFIX_HEX_WIDTH} hexadecimal characters, got {prefix!r}")
        return f"blob_prefix={prefix}/source-{self.source_id}-{args.index_in_partition:05d}.parquet"


def _hf_token() -> str:
    token = get_token()
    if not token:
        raise RuntimeError(
            "bigcode/the-stack-v2 is gated; set HF_TOKEN in the driver and Iris worker environment "
            "after accepting the dataset terms on Hugging Face"
        )
    return token


def list_stack_v2_tasks(cfg: StackV2DownloadConfig) -> list[StackV2ParquetTask]:
    """List the pinned parquet files and make one Zephyr task per file."""

    listing = list_hf_repo_files(
        hf_dataset_id=HF_DATASET_ID,
        revision=cfg.revision,
        hf_urls_glob=[HF_PARQUET_GLOB],
        token=_hf_token(),
    )
    source_root = listing.source_root
    file_info = listing.files
    files = sorted(file_info)
    if cfg.max_files is not None:
        files = files[: cfg.max_files]
    if not files:
        raise ValueError(f"No parquet files matched {HF_DATASET_ID}@{cfg.revision}:{HF_PARQUET_GLOB}")

    data_output_path = prefix_join(cfg.output_path, "data")
    tasks = [
        StackV2ParquetTask(
            source_url=f"{HF_PROTOCOL_PREFIX}{file}",
            relative_source_path=_relative_path_in_source(file, source_root),
            output_path=data_output_path,
            revision=cfg.revision,
        )
        for file in files
    ]
    # Repository trees are public even when file content is gated. Validate one file
    # before provisioning the worker fleet so a missing grant fails at the driver.
    _assert_hf_access(tasks[0])
    return tasks


def _assert_hf_access(task: StackV2ParquetTask) -> None:
    if not task.source_url.startswith(HF_PROTOCOL_PREFIX):
        return
    resolve_url = hf_hub_url(
        repo_id=HF_DATASET_ID,
        filename=task.relative_source_path,
        repo_type="dataset",
        revision=task.revision,
    )
    get_hf_file_metadata(resolve_url, token=_hf_token(), retry_on_errors=True)


def _input_storage_options(source_url: str) -> dict[str, object] | None:
    if not source_url.startswith(HF_PROTOCOL_PREFIX):
        return None
    return {"token": _hf_token(), "max_retries": OBJECT_STORE_MAX_RETRIES}


def partition_stack_v2_parquet(task: StackV2ParquetTask) -> StackV2DownloadResult:
    """Stream one remote parquet file into partitions keyed by ``blob_id[:2]``."""

    source = pl.scan_parquet(
        task.source_url,
        storage_options=_input_storage_options(task.source_url),
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
    return StackV2DownloadResult(
        source_url=task.source_url,
        relative_source_path=task.relative_source_path,
    )


def build_stack_v2_pipeline(tasks: list[StackV2ParquetTask], output_path: str) -> Dataset[str]:
    """Build the one-source-file-per-shard Zephyr pipeline."""

    return (
        Dataset.from_list(tasks)
        .map(partition_stack_v2_parquet)
        .write_jsonl(
            prefix_join(output_path, DOWNLOAD_SUCCESS_METRICS_TEMPLATE),
            skip_existing=True,
        )
    )


def download_stack_v2(cfg: StackV2DownloadConfig) -> None:
    """List the repository and execute the direct streaming transfer."""

    tasks = list_stack_v2_tasks(cfg)
    logger.info(
        "Downloading %d parquet files with up to %d workers into %d blob-prefix partitions",
        len(tasks),
        min(cfg.max_workers, len(tasks)),
        BLOB_PREFIX_PARTITION_COUNT,
    )
    pipeline = build_stack_v2_pipeline(tasks, cfg.output_path)
    ctx = ZephyrContext(
        name="download-stack-v2",
        resources=cfg.worker_resources,
        coordinator_resources=cfg.coordinator_resources,
        max_workers=min(cfg.max_workers, len(tasks)),
        max_shard_failures=DEFAULT_MAX_SHARD_FAILURES,
        max_execution_retries=10,
    )
    ctx.execute(pipeline, map_task_resources=cfg.task_resources)
    write_provenance_json(
        cfg.output_path,
        metadata={
            "dataset": HF_DATASET_ID,
            "revision": cfg.revision,
            "source_glob": HF_PARQUET_GLOB,
            "source_file_count": len(tasks),
            "partition_key": f"lower(blob_id[:{BLOB_PREFIX_HEX_WIDTH}])",
            "partition_count": BLOB_PREFIX_PARTITION_COUNT,
        },
    )


@draccus.wrap()
def main(cfg: StackV2DownloadConfig) -> None:
    """CLI entrypoint for the Stack v2 streaming downloader."""

    configure_logging(level=logging.INFO)
    download_stack_v2(cfg)


if __name__ == "__main__":
    main()
