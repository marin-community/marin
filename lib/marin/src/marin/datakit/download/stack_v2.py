# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream The Stack v2 parquet shards into ``blob_id`` prefix partitions."""

import logging
from dataclasses import dataclass, field

import draccus
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging

from marin.datakit.download.blob_id import (
    BLOB_PREFIX_HEX_WIDTH,
    BLOB_PREFIX_PARTITION_COUNT,
    BlobIdDownloadConfig,
    BlobIdParquetTask,
    execute_blob_id_partition_download,
    list_blob_id_parquet_tasks,
)

logger = logging.getLogger(__name__)

HF_DATASET_ID = "bigcode/the-stack-v2"
HF_REVISION = "e565caa3a78c2423bd374333a472b049eb090e47"
HF_PARQUET_GLOB = "data/*/*.parquet"


@dataclass(frozen=True)
class StackV2DownloadConfig(BlobIdDownloadConfig):
    """Configuration for the direct Hugging Face-to-object-store transfer."""

    output_path: str = field(default_factory=lambda: marin_temp_bucket(ttl_days=30, prefix="stack-v2"))
    revision: str = HF_REVISION


def list_stack_v2_tasks(cfg: StackV2DownloadConfig) -> list[BlobIdParquetTask]:
    """List the pinned parquet files and make one Zephyr task per file."""

    data_output_path = prefix_join(cfg.output_path, "data")
    return list_blob_id_parquet_tasks(
        cfg,
        dataset_id=HF_DATASET_ID,
        parquet_glob=HF_PARQUET_GLOB,
        require_token=True,
        output_path_for_relative=lambda _: data_output_path,
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
    execute_blob_id_partition_download(
        tasks,
        cfg,
        job_name="download-stack-v2",
        provenance_metadata={
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
    configure_logging(level=logging.INFO)
    download_stack_v2(cfg)


if __name__ == "__main__":
    main()
