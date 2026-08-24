# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream The Stack v2 parquet shards from Hugging Face into blob-prefix partitions.

The pipeline has one Zephyr source shard per Hugging Face parquet file. Each worker
uses Polars' native ``hf://`` object-store reader and streaming parquet sink, so the
source parquet is never materialized on worker disk. This deliberately uses Polars'
native range reader instead of ``hf_xet``'s file downloader, which reconstructs a
complete local file before a transform can run.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field

import draccus
import polars as pl
from fray.types import ResourceConfig
from huggingface_hub import HfFileSystem, get_token
from polars.io.partition import FileProviderArgs
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.download.huggingface import _relative_path_in_source
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

    output_path: str
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
    expected_size: int | None = None

    @property
    def source_id(self) -> str:
        return hashlib.sha256(self.relative_source_path.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class StackV2DownloadResult:
    """Metrics emitted after one source parquet is fully partitioned."""

    source_url: str
    relative_source_path: str
    expected_size: int | None


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

    source_root = f"datasets/{HF_DATASET_ID}"
    source_fs = HfFileSystem(token=_hf_token())
    listing = source_fs.glob(
        f"{source_root}/{HF_PARQUET_GLOB}",
        detail=True,
        revision=cfg.revision,
    )
    if not isinstance(listing, dict):
        raise TypeError("HfFileSystem.glob(detail=True) returned paths without file metadata")
    file_info = listing
    files = sorted(file_info)
    if cfg.max_files is not None:
        files = files[: cfg.max_files]
    if not files:
        raise ValueError(f"No parquet files matched {HF_DATASET_ID}@{cfg.revision}:{HF_PARQUET_GLOB}")

    data_output_path = prefix_join(cfg.output_path, "data")
    return [
        StackV2ParquetTask(
            source_url=f"hf://{file}",
            relative_source_path=_relative_path_in_source(file, source_root),
            output_path=data_output_path,
            expected_size=file_info[file].get("size"),
        )
        for file in files
    ]


def _input_storage_options(source_url: str) -> dict[str, object] | None:
    if not source_url.startswith("hf://"):
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
        storage_options={"max_retries": OBJECT_STORE_MAX_RETRIES},
    )
    logger.info("Partitioned %s into %s", task.source_url, task.output_path)
    return StackV2DownloadResult(
        source_url=task.source_url,
        relative_source_path=task.relative_source_path,
        expected_size=task.expected_size,
    )


def build_stack_v2_pipeline(tasks: list[StackV2ParquetTask], output_path: str) -> Dataset[str]:
    """Build the one-source-file-per-shard Zephyr pipeline."""

    return (
        Dataset.from_list(tasks)
        .map(partition_stack_v2_parquet)
        .write_jsonl(
            prefix_join(output_path, ".metrics/success-part-{shard:05d}-of-{total:05d}.jsonl"),
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
